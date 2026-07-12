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
//!   ANSI/DEC modes, SGR colors including 256-color and truecolor)
//! - OSC sequences (titles, working directory, hyperlinks, the OSC 133
//!   prompt protocol), terminated by `BEL` or `ST`
//! - DCS / SOS / PM / APC passthrough (contents discarded safely)
//! - Incremental UTF-8 decoding with chunk-boundary safety (invalid
//!   bytes decode to U+FFFD)
//!
//! # Chunk safety
//!
//! [`Parser::feed`] may be called with the stream split at *any* byte
//! boundary — mid-UTF-8 codepoint, mid-CSI, mid-OSC. All parse state
//! persists across calls, so feeding a stream byte-at-a-time produces the
//! exact same screen state and events as feeding it in one call.
//!
//! # Example
//!
//! ```
//! use bb_terminal_model::Screen;
//! use bb_vt_parser::{Parser, VtEvent};
//!
//! let mut screen = Screen::new(24, 80);
//! let mut parser = Parser::new();
//! let events = parser.feed(&mut screen, b"\x1b[1mhello\x07");
//! assert_eq!(screen.row_text(screen.absolute_row(0)).as_deref(), Some("hello"));
//! assert!(screen.pen().bold);
//! assert_eq!(events, vec![VtEvent::Bell]);
//! ```
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod osc;
mod sgr;

use bb_terminal_model::{CursorShape, EraseMode, Screen};

/// Semantic events surfaced by the parser alongside its screen effects.
///
/// These are the signals that higher layers (e.g. the block list) consume
/// to segment output into blocks, update window chrome, and render
/// hyperlinks. They are returned by [`Parser::feed`] in stream order.
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
    // Tab-stop mirror used for CBT (the screen exposes no back-tab).
    // Rebuilt from the every-8-columns default whenever the column count
    // changes; kept in sync with the HTS / TBC sequences we forward.
    tabs: Option<Vec<bool>>,
}

impl Parser {
    /// Create a parser in the ground state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw PTY bytes; applies effects to `screen`, returns semantic
    /// events in stream order.
    ///
    /// The byte stream may be split anywhere across calls (mid-UTF-8,
    /// mid-escape-sequence); all parse state persists in `self`.
    pub fn feed(&mut self, screen: &mut Screen, bytes: &[u8]) -> Vec<VtEvent> {
        let mut events = Vec::new();
        for &b in bytes {
            self.step(screen, &mut events, b);
        }
        events
    }

    // -----------------------------------------------------------------
    // State dispatch
    // -----------------------------------------------------------------

    fn step(&mut self, screen: &mut Screen, events: &mut Vec<VtEvent>, b: u8) {
        match self.state {
            State::Ground => self.on_ground(screen, events, b),
            State::Escape => self.on_escape(screen, b),
            State::EscapeIntermediate => self.on_escape_intermediate(screen, b),
            State::CsiParam => self.on_csi_param(screen, b),
            State::CsiIntermediate => self.on_csi_intermediate(screen, b),
            State::CsiIgnore => self.on_csi_ignore(screen, b),
            State::OscString => self.on_osc_string(events, b),
            State::OscEscape => self.on_osc_escape(screen, events, b),
            State::StringBody => self.on_string_body(b),
            State::StringEscape => self.on_string_escape(b),
        }
    }

    fn on_ground(&mut self, screen: &mut Screen, events: &mut Vec<VtEvent>, b: u8) {
        if self.utf8_need > 0 {
            self.utf8_continue(screen, events, b);
            return;
        }
        match b {
            0x07 => events.push(VtEvent::Bell),
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
            0x20..=0x2F => self.intermediates.push(b),
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

    fn on_osc_string(&mut self, events: &mut Vec<VtEvent>, b: u8) {
        match b {
            0x07 => {
                self.osc_dispatch(events);
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

    fn on_osc_escape(&mut self, screen: &mut Screen, events: &mut Vec<VtEvent>, b: u8) {
        if b == b'\\' {
            // ESC \ = ST, the standard OSC terminator.
            self.osc_dispatch(events);
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

    fn on_string_escape(&mut self, b: u8) {
        match b {
            b'\\' => self.state = State::Ground, // ST terminates the string
            0x1B => {}                           // stay: another potential ST
            _ => self.state = State::StringBody,
        }
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
            b'H' => {
                // HTS
                screen.set_tab_stop();
                self.mirror_set_tab(screen);
            }
            b'c' => {
                // RIS
                screen.reset();
                self.tabs = None;
            }
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
            // DECSCUSR is the only intermediate form we act on.
            if self.intermediates == [b' '] && f == b'q' && self.private.is_none() {
                self.set_cursor_style(screen);
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
            b'Z' => self.back_tab(screen, self.p1(0)), // CBT
            b'J' => {
                // ED (3 = "also clear scrollback" elsewhere; treated as 2)
                if let Some(mode) = Self::erase_mode(self.p(0)) {
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

    fn erase_mode(p: u16) -> Option<EraseMode> {
        match p {
            0 => Some(EraseMode::ToEnd),
            1 => Some(EraseMode::ToStart),
            2 | 3 => Some(EraseMode::All),
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
    // Tab-stop mirror (CBT support)
    // -----------------------------------------------------------------

    /// (Re)build the tab-stop mirror at the every-8-columns default when
    /// missing or when the column count changed (the screen resets its
    /// own stops on resize, so a width change invalidates the mirror).
    fn ensure_tabs(&mut self, cols: usize) {
        let stale = match &self.tabs {
            Some(t) => t.len() != cols,
            None => true,
        };
        if stale {
            self.tabs = Some((0..cols).map(|c| c != 0 && c % 8 == 0).collect());
        }
    }

    fn mirror_set_tab(&mut self, screen: &Screen) {
        self.ensure_tabs(screen.cols());
        let col = screen.cursor().col;
        if let Some(tabs) = self.tabs.as_mut() {
            if let Some(stop) = tabs.get_mut(col) {
                *stop = true;
            }
        }
    }

    fn tab_clear(&mut self, screen: &mut Screen) {
        match self.p(0) {
            0 => {
                screen.clear_tab_stop();
                self.ensure_tabs(screen.cols());
                let col = screen.cursor().col;
                if let Some(tabs) = self.tabs.as_mut() {
                    if let Some(stop) = tabs.get_mut(col) {
                        *stop = false;
                    }
                }
            }
            3 => {
                screen.clear_all_tab_stops();
                self.ensure_tabs(screen.cols());
                if let Some(tabs) = self.tabs.as_mut() {
                    for stop in tabs.iter_mut() {
                        *stop = false;
                    }
                }
            }
            _ => {}
        }
    }

    /// CBT: move the cursor back `n` tab stops (column 0 if none remain).
    fn back_tab(&mut self, screen: &mut Screen, n: usize) {
        self.ensure_tabs(screen.cols());
        let Some(tabs) = self.tabs.as_ref() else {
            return;
        };
        let mut col = screen.cursor().col;
        for _ in 0..n {
            col = (0..col)
                .rev()
                .find(|&c| tabs.get(c) == Some(&true))
                .unwrap_or(0);
        }
        screen.cursor_to_col(col);
    }

    // -----------------------------------------------------------------
    // OSC dispatch
    // -----------------------------------------------------------------

    fn osc_dispatch(&mut self, events: &mut Vec<VtEvent>) {
        let buf = std::mem::take(&mut self.osc_buf);
        if self.osc_overflow {
            self.osc_overflow = false;
            return; // overlong OSC: discarded entirely
        }
        osc::dispatch(&buf, events);
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

    fn utf8_continue(&mut self, screen: &mut Screen, events: &mut Vec<VtEvent>, b: u8) {
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
