//! VT100 / ANSI escape sequence parser feeding [`bb_terminal_model`].
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! [`Parser`] is a stateful byte-stream parser in the spirit of the
//! classic VT500-series state diagram. It consumes raw bytes emitted by a
//! PTY-bound shell or remote process, applies the resulting actions to a
//! [`bb_terminal_model::Screen`], and returns the semantic [`VtEvent`]s
//! (title changes, bells, hyperlinks, OSC 133 prompt markers) that higher
//! layers such as the block list consume. It handles:
//!
//! - C0 control codes (BEL, BS, HT, LF/VT/FF, CR; DEL is discarded)
//! - `ESC` sequences (DECSC/DECRC, IND/NEL/RI, HTS, RIS, keypad and
//!   charset designations)
//! - CSI escape sequences (cursor positioning, editing, scrolling,
//!   ANSI/DEC modes, `DECSTR` soft reset, SGR colors including
//!   256-color and truecolor)
//! - OSC sequences (titles, working directory, hyperlinks, the OSC 133
//!   prompt protocol), terminated by `BEL` or `ST`
//! - DCS / SOS / PM / APC passthrough (contents discarded safely; as in
//!   the DEC state diagram, a bare `ESC` cancels the string rather than
//!   being swallowed by it, so an unterminated one cannot wedge the
//!   stream)
//! - Incremental UTF-8 decoding with chunk-boundary safety (invalid
//!   bytes decode to U+FFFD)
//!
//! # Positioned events
//!
//! Every event is returned as a [`PositionedEvent`]: the [`VtEvent`]
//! itself plus the cursor position *at the instant the parser emitted
//! it*, captured while the parser still holds the screen. Consumers must
//! never re-read the cursor after `feed` returns to locate an event —
//! that reads the position at the *end* of the chunk, which collapses
//! every marker in the chunk onto one row and makes the result depend on
//! how the host happened to split its PTY reads.
//!
//! # Chunk safety
//!
//! [`Parser::feed`] may be called with the stream split at *any* byte
//! boundary — mid-UTF-8 codepoint, mid-CSI, mid-OSC. All parse state
//! persists across calls, so feeding a stream byte-at-a-time produces the
//! exact same screen state and the exact same [`PositionedEvent`]s —
//! positions included — as feeding it in one call.
//!
//! # Example
//!
//! ```
//! use bb_terminal_model::Screen;
//! use bb_vt_parser::{Parser, PositionedEvent, VtEvent};
//!
//! let mut screen = Screen::new(24, 80);
//! let mut parser = Parser::new();
//! let events = parser.feed(&mut screen, b"\x1b[1mhello\x07");
//! assert_eq!(screen.row_text(screen.absolute_row(0)).as_deref(), Some("hello"));
//! assert!(screen.pen().bold);
//! // The bell fired with the cursor just past "hello" on the first row.
//! assert_eq!(events, vec![PositionedEvent::new(VtEvent::Bell, 0, 5)]);
//! ```
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod osc;
mod sgr;

use bb_terminal_model::{CursorShape, EraseMode, Screen};

/// A [`VtEvent`] together with where the terminal cursor was when it
/// fired.
///
/// The position is sampled by the parser at the exact byte that completed
/// the sequence, before any later byte in the same chunk moves the
/// cursor. That makes it independent of how the PTY stream was chunked:
/// the same transcript fed one byte at a time and in one 64 KiB call
/// yields the identical `Vec<PositionedEvent>`.
///
/// This is what makes command-block boundaries correct. A consumer that
/// instead reads `Screen::cursor_absolute_row()` after `feed` returns
/// gets the position at the *end* of the chunk for every event in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionedEvent {
    /// The semantic event.
    pub event: VtEvent,
    /// Absolute row (see [`Screen::absolute_row`]) of the cursor when the
    /// event was emitted. Stable across scrollback eviction.
    pub cursor_row: usize,
    /// Cursor column when the event was emitted.
    pub cursor_col: usize,
    /// Whether the **alternate screen** was active when the event fired.
    ///
    /// Absolute rows address the primary screen's timeline, so a
    /// row-addressed consumer (a block list) must not act on rows carried
    /// by alt-screen events — a TUI's cursor position is meaningless on
    /// that timeline. Carrying the flag here lets consumers feed whole
    /// PTY chunks without splitting them just to sample this state
    /// between escape sequences.
    pub alt_screen: bool,
}

impl PositionedEvent {
    /// Pair `event` with the cursor position it was emitted at (on the
    /// primary screen; see [`PositionedEvent::new_alt`]).
    ///
    /// Mostly useful for building expected values in tests; the parser
    /// fills these in itself.
    pub fn new(event: VtEvent, cursor_row: usize, cursor_col: usize) -> Self {
        Self {
            event,
            cursor_row,
            cursor_col,
            alt_screen: false,
        }
    }

    /// [`PositionedEvent::new`], but for an event fired while the
    /// alternate screen was active.
    pub fn new_alt(event: VtEvent, cursor_row: usize, cursor_col: usize) -> Self {
        Self {
            event,
            cursor_row,
            cursor_col,
            alt_screen: true,
        }
    }

    /// Discard the position, keeping only the semantic event.
    ///
    /// Handy for consumers that only care about *what* happened:
    /// `events.into_iter().map(PositionedEvent::into_event)`.
    pub fn into_event(self) -> VtEvent {
        self.event
    }

    /// The `(row, col)` the event was emitted at, as a pair.
    pub fn cursor(&self) -> (usize, usize) {
        (self.cursor_row, self.cursor_col)
    }

    /// Stamp `event` with `screen`'s current cursor position.
    fn at(screen: &Screen, event: VtEvent) -> Self {
        Self {
            event,
            cursor_row: screen.cursor_absolute_row(),
            cursor_col: screen.cursor().col,
            alt_screen: screen.is_alt_screen(),
        }
    }
}

impl From<PositionedEvent> for VtEvent {
    fn from(positioned: PositionedEvent) -> Self {
        positioned.event
    }
}

/// Semantic events surfaced by the parser alongside its screen effects.
///
/// These are the signals that higher layers (e.g. the block list) consume
/// to segment output into blocks, update window chrome, and render
/// hyperlinks. They are returned by [`Parser::feed`] in stream order,
/// each wrapped in a [`PositionedEvent`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VtEvent {
    /// OSC 133;A — shell is printing the prompt.
    PromptStart,
    /// OSC 133;B — prompt done, user is typing the command.
    CommandStart,
    /// OSC 133;C — command execution began (output follows).
    CommandExecuted,
    /// OSC 133;D\[;exit_code\] — command finished.
    CommandFinished {
        /// The command's exit code, when the shell reported one.
        exit_code: Option<i32>,
    },
    /// OSC 0 / OSC 2 — window title change.
    Title(String),
    /// OSC 7 — working directory report (`file://` URL as sent).
    WorkingDirectory(String),
    /// OSC 8 — hyperlink open (`Some(uri)`) or close (`None`).
    Hyperlink(Option<String>),
    /// BEL used as an alert (i.e. a bare `0x07` in ground state).
    Bell,
}

/// Maximum buffered OSC payload; overlong OSC strings are discarded.
const OSC_MAX: usize = 4096;
/// Maximum number of CSI parameters retained; extras are dropped.
const MAX_PARAMS: usize = 16;
/// Maximum number of colon-separated sub-parameters per CSI parameter.
const MAX_SUBPARAMS: usize = 8;
/// Maximum number of intermediate bytes before a CSI is ignored.
const MAX_INTERMEDIATES: usize = 2;
/// Emitted for invalid UTF-8 input.
const REPLACEMENT: char = '\u{FFFD}';

/// States of the escape-sequence state machine.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Printing text and executing C0 controls.
    #[default]
    Ground,
    /// Seen `ESC`, waiting for the sequence selector.
    Escape,
    /// Seen `ESC` + intermediate byte(s) (e.g. charset designations).
    EscapeIntermediate,
    /// Inside `CSI`, collecting parameters.
    CsiParam,
    /// Inside `CSI`, collecting intermediate bytes after the parameters.
    CsiIntermediate,
    /// Inside a malformed `CSI`; consuming until the final byte.
    CsiIgnore,
    /// Inside an `OSC` string, buffering the payload.
    OscString,
    /// Seen `ESC` inside an `OSC` string (potential `ST` terminator).
    OscEscape,
    /// Inside a DCS / SOS / PM / APC string; contents discarded.
    StringBody,
    /// Seen `ESC` inside a discarded string (potential `ST` terminator).
    StringEscape,
}

/// A stateful VT100/ANSI parser driving a [`Screen`].
///
/// Create one per PTY stream with [`Parser::new`] and call
/// [`Parser::feed`] with each chunk of output. The parser owns no screen
/// state itself; it holds only in-flight parse state (partial UTF-8
/// codepoints, partial escape sequences, buffered OSC payloads), which is
/// why `feed` is safe to call with arbitrarily split chunks.
#[derive(Debug, Default)]
pub struct Parser {
    state: State,
    // Incremental UTF-8 decoding (ground state printing).
    utf8_buf: [u8; 4],
    utf8_len: u8,
    utf8_need: u8,
    // CSI collection. Each parameter is a list of colon-separated
    // sub-parameters; a plain parameter has exactly one entry.
    params: Vec<Vec<u16>>,
    cur_parts: Vec<u16>,
    cur_val: u16,
    csi_has_any: bool,
    private: Option<u8>,
    intermediates: Vec<u8>,
    // OSC collection.
    osc_buf: Vec<u8>,
    osc_overflow: bool,
}

impl Parser {
    /// Create a parser in the ground state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw PTY bytes; applies effects to `screen` and returns the
    /// semantic events in stream order, each carrying the cursor position
    /// at the instant it fired.
    ///
    /// The byte stream may be split anywhere across calls (mid-UTF-8,
    /// mid-escape-sequence); all parse state persists in `self`, and the
    /// returned events — positions included — are identical however the
    /// stream was chunked.
    pub fn feed(&mut self, screen: &mut Screen, bytes: &[u8]) -> Vec<PositionedEvent> {
        let mut events = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            // Fast path: in ground state with no pending UTF-8, a run of
            // printable ASCII needs no state machine at all — hand the
            // whole run to the screen in one call. `Screen::print_str` is
            // semantically identical to printing char-by-char (the
            // chunk-equivalence tests feed the same input byte-at-a-time,
            // which takes the per-byte path below, and assert identical
            // screens), so this changes throughput, not behavior.
            if self.state == State::Ground && self.utf8_need == 0 {
                let run = bytes[i..]
                    .iter()
                    .take_while(|&&b| (0x20..=0x7E).contains(&b))
                    .count();
                if run >= 2 {
                    if let Ok(text) = std::str::from_utf8(&bytes[i..i + run]) {
                        screen.print_str(text);
                        i += run;
                        continue;
                    }
                }
            }
            self.step(screen, &mut events, bytes[i]);
            i += 1;
        }
        events
    }

    // -----------------------------------------------------------------
    // State dispatch
    // -----------------------------------------------------------------

    fn step(&mut self, screen: &mut Screen, events: &mut Vec<PositionedEvent>, b: u8) {
        match self.state {
            State::Ground => self.on_ground(screen, events, b),
            State::Escape => self.on_escape(screen, b),
            State::EscapeIntermediate => self.on_escape_intermediate(screen, b),
            State::CsiParam => self.on_csi_param(screen, b),
            State::CsiIntermediate => self.on_csi_intermediate(screen, b),
            State::CsiIgnore => self.on_csi_ignore(screen, b),
            State::OscString => self.on_osc_string(screen, events, b),
            State::OscEscape => self.on_osc_escape(screen, events, b),
            State::StringBody => self.on_string_body(b),
            State::StringEscape => self.on_string_escape(screen, b),
        }
    }

    fn on_ground(&mut self, screen: &mut Screen, events: &mut Vec<PositionedEvent>, b: u8) {
        if self.utf8_need > 0 {
            self.utf8_continue(screen, events, b);
            return;
        }
        match b {
            0x07 => events.push(PositionedEvent::at(screen, VtEvent::Bell)),
            0x1B => self.enter_escape(),
            0x00..=0x06 | 0x08..=0x1A | 0x1C..=0x1F => Self::execute_ctrl(screen, b),
            0x20..=0x7E => screen.print(char::from(b)),
            0x7F => {} // DEL: discarded entirely.
            0x80..=0xFF => self.utf8_start(screen, b),
        }
    }

    fn on_escape(&mut self, screen: &mut Screen, b: u8) {
        match b {
            0x18 | 0x1A => self.state = State::Ground,
            0x1B => self.intermediates.clear(), // restart ESC
            0x00..=0x17 | 0x19 | 0x1C..=0x1F => Self::execute_ctrl(screen, b),
            0x20..=0x2F => {
                self.intermediates.push(b);
                self.state = State::EscapeIntermediate;
            }
            0x30..=0x7E => self.esc_dispatch(screen, b),
            0x7F => {}
            0x80..=0xFF => self.state = State::Ground, // invalid in a sequence
        }
    }

    fn on_escape_intermediate(&mut self, screen: &mut Screen, b: u8) {
        match b {
            0x18 | 0x1A => self.state = State::Ground,
            0x1B => self.enter_escape(),
            0x00..=0x17 | 0x19 | 0x1C..=0x1F => Self::execute_ctrl(screen, b),
            0x20..=0x2F => {
                // Bounded exactly like the CSI intermediate path. Without
                // a cap, `ESC` followed by a run of `0x20..=0x2F` (SPACE
                // included, so ordinary padded output can do it by
                // accident) grows this Vec byte-for-byte with the input
                // and holds it until the Parser is dropped. Excess bytes
                // are dropped; the sequence is still consumed through its
                // final byte, which `esc_dispatch` then ignores because
                // `intermediates` is non-empty.
                if self.intermediates.len() < MAX_INTERMEDIATES {
                    self.intermediates.push(b);
                }
            }
            0x30..=0x7E => self.esc_dispatch(screen, b),
            0x7F => {}
            0x80..=0xFF => self.state = State::Ground,
        }
    }

    fn on_csi_param(&mut self, screen: &mut Screen, b: u8) {
        match b {
            0x18 | 0x1A => self.state = State::Ground,
            0x1B => self.enter_escape(),
            0x00..=0x17 | 0x19 | 0x1C..=0x1F => Self::execute_ctrl(screen, b),
            b'0'..=b'9' => {
                self.cur_val = self
                    .cur_val
                    .saturating_mul(10)
                    .saturating_add(u16::from(b - b'0'));
                self.csi_has_any = true;
            }
            b';' => {
                self.end_param();
                self.csi_has_any = true;
            }
            b':' => {
                self.end_subparam();
                self.csi_has_any = true;
            }
            0x3C..=0x3F => {
                // Private markers are only valid before any parameter.
                if self.csi_has_any || self.private.is_some() {
                    self.state = State::CsiIgnore;
                } else {
                    self.private = Some(b);
                }
            }
            0x20..=0x2F => {
                self.intermediates.push(b);
                self.state = State::CsiIntermediate;
            }
            0x40..=0x7E => self.csi_dispatch(screen, b),
            0x7F => {}
            0x80..=0xFF => self.state = State::CsiIgnore,
        }
    }

    fn on_csi_intermediate(&mut self, screen: &mut Screen, b: u8) {
        match b {
            0x18 | 0x1A => self.state = State::Ground,
            0x1B => self.enter_escape(),
            0x00..=0x17 | 0x19 | 0x1C..=0x1F => Self::execute_ctrl(screen, b),
            0x20..=0x2F => {
                if self.intermediates.len() >= MAX_INTERMEDIATES {
                    self.state = State::CsiIgnore;
                } else {
                    self.intermediates.push(b);
                }
            }
            0x30..=0x3F => self.state = State::CsiIgnore, // params after intermediates
            0x40..=0x7E => self.csi_dispatch(screen, b),
            0x7F => {}
            0x80..=0xFF => self.state = State::CsiIgnore,
        }
    }

    fn on_csi_ignore(&mut self, screen: &mut Screen, b: u8) {
        match b {
            0x18 | 0x1A => self.state = State::Ground,
            0x1B => self.enter_escape(),
            0x00..=0x17 | 0x19 | 0x1C..=0x1F => Self::execute_ctrl(screen, b),
            0x40..=0x7E => self.state = State::Ground,
            _ => {}
        }
    }

    fn on_osc_string(&mut self, screen: &Screen, events: &mut Vec<PositionedEvent>, b: u8) {
        match b {
            0x07 => {
                self.osc_dispatch(screen, events);
                self.state = State::Ground;
            }
            0x1B => self.state = State::OscEscape,
            0x18 | 0x1A => self.state = State::Ground, // abort, discard
            0x00..=0x1F => {}                          // other C0: ignored inside OSC
            _ => {
                if self.osc_buf.len() < OSC_MAX {
                    self.osc_buf.push(b);
                } else {
                    self.osc_overflow = true;
                }
            }
        }
    }

    fn on_osc_escape(&mut self, screen: &mut Screen, events: &mut Vec<PositionedEvent>, b: u8) {
        if b == b'\\' {
            // ESC \ = ST, the standard OSC terminator.
            self.osc_dispatch(screen, events);
            self.state = State::Ground;
        } else {
            // ESC followed by anything else aborts the OSC; the ESC then
            // starts a fresh escape sequence that consumes this byte.
            self.osc_buf.clear();
            self.osc_overflow = false;
            self.enter_escape();
            self.on_escape(screen, b);
        }
    }

    fn on_string_body(&mut self, b: u8) {
        match b {
            0x1B => self.state = State::StringEscape,
            0x18 | 0x1A => self.state = State::Ground,
            _ => {} // DCS/SOS/PM/APC contents: discarded
        }
    }

    fn on_string_escape(&mut self, screen: &mut Screen, b: u8) {
        // `ESC` is an "anywhere" transition in the DEC ANSI parser: it
        // cancels whatever sequence or control string is in progress, and
        // then begins a fresh escape sequence that consumes this byte.
        // Falling back into the string body instead (the old behavior)
        // meant one stray `ESC P` from a buggy program wedged the terminal
        // permanently, silently swallowing every later byte — prompts,
        // command output, OSC 133 markers and all.
        //
        // The `\` of an `ST` needs no special case: `esc_dispatch` treats
        // `ESC \` with nothing open as a no-op back to ground, which is
        // exactly "the string ended here".
        self.enter_escape();
        self.on_escape(screen, b);
    }

    // -----------------------------------------------------------------
    // C0 execution
    // -----------------------------------------------------------------

    /// Execute a C0 control's screen effect. BEL is intentionally *not*
    /// handled here: as an alert it only fires in ground state (handled
    /// by [`Parser::on_ground`]); mid-sequence it is ignored.
    fn execute_ctrl(screen: &mut Screen, b: u8) {
        match b {
            0x08 => screen.backspace(),
            0x09 => screen.tab(),
            0x0A..=0x0C => screen.linefeed(), // LF, VT, FF
            0x0D => screen.carriage_return(),
            _ => {} // NUL, ENQ, SO, SI, CAN/SUB (in ground), ...: ignored
        }
    }

    // -----------------------------------------------------------------
    // ESC dispatch
    // -----------------------------------------------------------------

    fn enter_escape(&mut self) {
        self.state = State::Escape;
        self.intermediates.clear();
    }

    fn csi_reset(&mut self) {
        self.params.clear();
        self.cur_parts.clear();
        self.cur_val = 0;
        self.csi_has_any = false;
        self.private = None;
        self.intermediates.clear();
    }

    fn esc_dispatch(&mut self, screen: &mut Screen, b: u8) {
        self.state = State::Ground;
        if !self.intermediates.is_empty() {
            // Charset designations (ESC ( X, ESC ) X, ...), DECALN, and
            // friends: parsed and ignored.
            return;
        }
        match b {
            b'7' => screen.save_cursor(),    // DECSC
            b'8' => screen.restore_cursor(), // DECRC
            b'D' => screen.linefeed(),       // IND: down, scrolling at bottom margin
            b'E' => {
                // NEL: next line
                screen.linefeed();
                screen.carriage_return();
            }
            b'M' => {
                // RI: reverse index, scrolling down at the top margin.
                if screen.cursor().row == screen.scroll_region().0 {
                    screen.scroll_down(1);
                } else {
                    screen.cursor_up(1);
                }
            }
            b'H' => screen.set_tab_stop(), // HTS
            b'c' => screen.reset(),        // RIS
            b'[' => {
                self.csi_reset();
                self.state = State::CsiParam;
            }
            b']' => {
                self.osc_buf.clear();
                self.osc_overflow = false;
                self.state = State::OscString;
            }
            b'P' | b'X' | b'^' | b'_' => self.state = State::StringBody, // DCS/SOS/PM/APC
            b'\\' => {}                                                  // ST with nothing open
            b'=' | b'>' => {} // DECKPAM / DECKPNM: parsed and ignored
            _ => {}
        }
    }

    // -----------------------------------------------------------------
    // CSI parameters
    // -----------------------------------------------------------------

    fn end_subparam(&mut self) {
        if self.cur_parts.len() < MAX_SUBPARAMS {
            self.cur_parts.push(self.cur_val);
        }
        self.cur_val = 0;
    }

    fn end_param(&mut self) {
        self.end_subparam();
        let parts = std::mem::take(&mut self.cur_parts);
        if self.params.len() < MAX_PARAMS {
            self.params.push(parts);
        }
    }

    /// Parameter `i` (first sub-parameter), defaulting to 0.
    fn p(&self, i: usize) -> u16 {
        self.params
            .get(i)
            .and_then(|parts| parts.first())
            .copied()
            .unwrap_or(0)
    }

    /// Parameter `i` as a count, defaulting to 1 (0 also means 1).
    fn p1(&self, i: usize) -> usize {
        usize::from(self.p(i)).max(1)
    }

    // -----------------------------------------------------------------
    // CSI dispatch
    // -----------------------------------------------------------------

    fn csi_dispatch(&mut self, screen: &mut Screen, f: u8) {
        self.state = State::Ground;
        if self.csi_has_any {
            self.end_param();
        }
        if !self.intermediates.is_empty() {
            // Two intermediate forms carry meaning for us; the rest
            // (DECRQM's `$p`, DECSCL's `"p`, ...) are parsed and ignored.
            if self.private.is_none() {
                match (self.intermediates.as_slice(), f) {
                    // DECSCUSR — CSI Ps SP q
                    ([b' '], b'q') => self.set_cursor_style(screen),
                    // DECSTR — CSI ! p. Shells send this to recover a
                    // wedged terminal, so a no-op here means the recovery
                    // path silently does nothing.
                    ([b'!'], b'p') => screen.soft_reset(),
                    _ => {}
                }
            }
            return;
        }
        match self.private {
            Some(b'?') => match f {
                b'h' => self.dec_private_mode(screen, true),
                b'l' => self.dec_private_mode(screen, false),
                _ => {} // DECRQM and friends: parsed and ignored
            },
            Some(_) => {} // CSI > / CSI = sequences: parsed and ignored
            None => self.csi_standard(screen, f),
        }
    }

    fn csi_standard(&mut self, screen: &mut Screen, f: u8) {
        match f {
            b'A' => screen.cursor_up(self.p1(0)),    // CUU
            b'B' => screen.cursor_down(self.p1(0)),  // CUD
            b'C' => screen.cursor_right(self.p1(0)), // CUF
            b'D' => screen.cursor_left(self.p1(0)),  // CUB
            b'E' => {
                // CNL
                screen.cursor_down(self.p1(0));
                screen.carriage_return();
            }
            b'F' => {
                // CPL
                screen.cursor_up(self.p1(0));
                screen.carriage_return();
            }
            b'G' => screen.cursor_to_col(self.p1(0) - 1), // CHA
            b'H' | b'f' => screen.cursor_goto(self.p1(0) - 1, self.p1(1) - 1), // CUP / HVP
            b'd' => screen.cursor_to_row(self.p1(0) - 1), // VPA
            b'I' => {
                // CHT
                for _ in 0..self.p1(0) {
                    screen.tab();
                }
            }
            b'Z' => {
                // CBT: back-tab `n` times against the screen's own stops.
                // Column 0 is a fixed point, so stop early once we hit it.
                for _ in 0..self.p1(0) {
                    if screen.cursor().col == 0 {
                        break;
                    }
                    screen.back_tab();
                }
            }
            b'J' => {
                // ED. `CSI 3 J` is xterm's "Erase Saved Lines" (XTERM_ED
                // 3): it DROPS the retained scrollback and leaves the grid
                // alone — it is not a stronger `CSI 2 J`. `/usr/bin/clear`
                // under `xterm-256color` emits `ESC[3J ESC[H ESC[2J`, so
                // getting this backwards turns every `clear` into another
                // screenful of blank scrollback that evicts real history.
                if self.p(0) == 3 {
                    screen.erase_scrollback();
                } else if let Some(mode) = Self::erase_mode(self.p(0)) {
                    screen.erase_in_display(mode);
                }
            }
            b'K' => {
                // EL
                if let Some(mode) = Self::erase_mode(self.p(0)) {
                    screen.erase_in_line(mode);
                }
            }
            b'@' => screen.insert_chars(self.p1(0)), // ICH
            b'P' => screen.delete_chars(self.p1(0)), // DCH
            b'X' => screen.erase_chars(self.p1(0)),  // ECH
            b'L' => screen.insert_lines(self.p1(0)), // IL
            b'M' => screen.delete_lines(self.p1(0)), // DL
            b'S' => screen.scroll_up(self.p1(0)),    // SU
            b'T' => screen.scroll_down(self.p1(0)),  // SD
            b'r' => {
                // DECSTBM (1-based args, 0 = default)
                screen.set_scroll_region(usize::from(self.p(0)), usize::from(self.p(1)));
            }
            b'g' => self.tab_clear(screen), // TBC
            b'h' => self.ansi_mode(screen, true),
            b'l' => self.ansi_mode(screen, false),
            b'm' => sgr::apply(screen.pen_mut(), &self.params), // SGR
            _ => {}
        }
    }

    /// The erase extent shared by ED and EL. Only 0/1/2 are defined; `3`
    /// is *not* an extent — for ED it selects "Erase Saved Lines"
    /// (handled by the caller), and for EL it is undefined and ignored.
    fn erase_mode(p: u16) -> Option<EraseMode> {
        match p {
            0 => Some(EraseMode::ToEnd),
            1 => Some(EraseMode::ToStart),
            2 => Some(EraseMode::All),
            _ => None,
        }
    }

    fn set_cursor_style(&self, screen: &mut Screen) {
        let shape = match self.p(0) {
            0..=2 => CursorShape::Block,
            3 | 4 => CursorShape::Underline,
            5 | 6 => CursorShape::Bar,
            _ => return,
        };
        screen.set_cursor_shape(shape);
    }

    fn mode_codes(&self) -> Vec<u16> {
        self.params
            .iter()
            .map(|parts| parts.first().copied().unwrap_or(0))
            .collect()
    }

    fn ansi_mode(&self, screen: &mut Screen, set: bool) {
        for code in self.mode_codes() {
            if code == 4 {
                screen.set_insert_mode(set); // IRM
            }
            // Other ANSI modes: parsed and ignored.
        }
    }

    fn dec_private_mode(&self, screen: &mut Screen, set: bool) {
        for code in self.mode_codes() {
            match code {
                6 => screen.set_origin_mode(set),     // DECOM
                7 => screen.set_autowrap(set),        // DECAWM
                25 => screen.set_cursor_visible(set), // DECTCEM
                47 | 1047 => {
                    if set {
                        screen.enter_alt_screen();
                    } else {
                        screen.exit_alt_screen();
                    }
                }
                1049 => {
                    if set {
                        screen.save_cursor();
                        screen.enter_alt_screen(); // clears the alt grid
                    } else {
                        screen.exit_alt_screen();
                        screen.restore_cursor();
                    }
                }
                _ => {} // 2004 (bracketed paste) and others: parsed and ignored
            }
        }
    }

    // -----------------------------------------------------------------
    // Tab stops
    //
    // The screen owns the tab stops outright — the parser keeps no mirror.
    // It used to, because the screen exposed no back-tab, and the two
    // copies silently desynchronized: `Screen::resize` rebuilds its stops
    // unconditionally, while the mirror was rebuilt only when the *column*
    // count changed, so a rows-only resize left HT and CBT disagreeing
    // about where the stops were.
    // -----------------------------------------------------------------

    /// TBC: clear the stop under the cursor (0) or every stop (3).
    fn tab_clear(&self, screen: &mut Screen) {
        match self.p(0) {
            0 => screen.clear_tab_stop(),
            3 => screen.clear_all_tab_stops(),
            _ => {}
        }
    }

    // -----------------------------------------------------------------
    // OSC dispatch
    // -----------------------------------------------------------------

    /// Parse the buffered OSC payload. The cursor position is sampled
    /// here — at the terminator byte, after everything printed before the
    /// sequence has been applied — which is precisely where an OSC 133
    /// marker semantically belongs.
    fn osc_dispatch(&mut self, screen: &Screen, events: &mut Vec<PositionedEvent>) {
        let buf = std::mem::take(&mut self.osc_buf);
        if self.osc_overflow {
            self.osc_overflow = false;
            return; // overlong OSC: discarded entirely
        }
        if let Some(event) = osc::dispatch(&buf) {
            events.push(PositionedEvent::at(screen, event));
        }
    }

    // -----------------------------------------------------------------
    // UTF-8 decoding (ground state)
    // -----------------------------------------------------------------

    fn utf8_start(&mut self, screen: &mut Screen, b: u8) {
        let need = match b {
            0xC2..=0xDF => 1,
            0xE0..=0xEF => 2,
            0xF0..=0xF4 => 3,
            _ => {
                // Invalid lead (lone continuation, overlong C0/C1, > F4).
                screen.print(REPLACEMENT);
                return;
            }
        };
        self.utf8_buf[0] = b;
        self.utf8_len = 1;
        self.utf8_need = need;
    }

    fn utf8_continue(&mut self, screen: &mut Screen, events: &mut Vec<PositionedEvent>, b: u8) {
        // The first continuation byte carries lead-specific range limits
        // (rejects overlong encodings, surrogates, and > U+10FFFF).
        let ok = if self.utf8_len == 1 {
            match self.utf8_buf[0] {
                0xE0 => (0xA0..=0xBF).contains(&b),
                0xED => (0x80..=0x9F).contains(&b),
                0xF0 => (0x90..=0xBF).contains(&b),
                0xF4 => (0x80..=0x8F).contains(&b),
                _ => (0x80..=0xBF).contains(&b),
            }
        } else {
            (0x80..=0xBF).contains(&b)
        };
        if !ok {
            // Emit U+FFFD for the failed sequence and reprocess this byte
            // fresh (it may be ASCII, a control, or a new lead byte).
            self.utf8_reset();
            screen.print(REPLACEMENT);
            self.on_ground(screen, events, b);
            return;
        }
        if let Some(slot) = self.utf8_buf.get_mut(usize::from(self.utf8_len)) {
            *slot = b;
        }
        self.utf8_len += 1;
        if self.utf8_len == self.utf8_need + 1 {
            let len = usize::from(self.utf8_len);
            let ch = self
                .utf8_buf
                .get(..len)
                .and_then(|bytes| std::str::from_utf8(bytes).ok())
                .and_then(|s| s.chars().next())
                .unwrap_or(REPLACEMENT);
            self.utf8_reset();
            screen.print(ch);
        }
    }

    fn utf8_reset(&mut self) {
        self.utf8_len = 0;
        self.utf8_need = 0;
    }
}
