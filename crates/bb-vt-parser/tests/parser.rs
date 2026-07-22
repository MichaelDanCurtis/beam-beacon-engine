//! Integration tests for `bb-vt-parser`.
//!
//! Every input goes through [`run_split`], which feeds the byte stream
//! three ways — one-shot, byte-at-a-time, and in 3-byte chunks — and
//! asserts that all three produce identical screen state and identical
//! [`PositionedEvent`]s, *cursor positions included*. That makes
//! chunk-boundary safety a property checked by *every* test, not just the
//! dedicated split tests. Tests that need the screen mutated mid-stream
//! (a resize, say) call [`run_split`] directly, which keeps the guarantee
//! across the mutation too.
//!
//! Tests that only care about *what* happened use [`run`], which maps the
//! positioned events down to bare [`VtEvent`]s; tests about *where* an
//! event fired use [`run_positioned`].

#![allow(clippy::unwrap_used, clippy::expect_used)]

use bb_terminal_model::{Color, CursorShape, Screen};
use bb_vt_parser::{Parser, PositionedEvent, VtEvent};

/// Feed `before`, apply `between` to the screen, then feed `after` —
/// asserting that one-shot, byte-at-a-time, and 3-byte-chunk feeding are
/// equivalent on both sides of the mutation: same screen, same events,
/// same event positions.
///
/// Every other runner delegates here, so chunk-boundary safety is a
/// property checked by *every* test rather than by dedicated split tests.
fn run_split(
    make_screen: impl Fn() -> Screen,
    before: &[u8],
    between: impl Fn(&mut Screen),
    after: &[u8],
) -> (Screen, Vec<PositionedEvent>) {
    let once = |chunk: usize| {
        let mut parser = Parser::new();
        let mut screen = make_screen();
        let mut events = Vec::new();
        let mut feed = |screen: &mut Screen, events: &mut Vec<PositionedEvent>, part: &[u8]| {
            if chunk == 0 {
                events.extend(parser.feed(screen, part));
            } else {
                for piece in part.chunks(chunk) {
                    events.extend(parser.feed(screen, piece));
                }
            }
        };
        feed(&mut screen, &mut events, before);
        between(&mut screen);
        feed(&mut screen, &mut events, after);
        (screen, events)
    };

    let (s1, ev1) = once(0);
    for chunk in [1, 3] {
        let (other, ev) = once(chunk);
        assert_eq!(s1, other, "screen mismatch at chunk size {chunk}");
        assert_eq!(ev1, ev, "event mismatch at chunk size {chunk}");
    }
    (s1, ev1)
}

/// Feed `bytes` into a fresh screen from `make_screen`, asserting that
/// one-shot, byte-at-a-time, and 3-byte-chunk feeding are equivalent.
fn run_with(make_screen: impl Fn() -> Screen, bytes: &[u8]) -> (Screen, Vec<PositionedEvent>) {
    run_split(make_screen, bytes, |_| {}, b"")
}

/// [`run_with`] on a fresh `rows x cols` screen.
fn run_positioned_dims(bytes: &[u8], rows: usize, cols: usize) -> (Screen, Vec<PositionedEvent>) {
    run_with(|| Screen::new(rows, cols), bytes)
}

/// [`run_positioned_dims`] with an explicit scrollback capacity.
fn run_scrollback_dims(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    max_scrollback: usize,
) -> (Screen, Vec<PositionedEvent>) {
    run_with(
        || Screen::with_scrollback(rows, cols, max_scrollback),
        bytes,
    )
}

/// [`run_positioned_dims`] on a default 24x80 screen.
fn run_positioned(bytes: &[u8]) -> (Screen, Vec<PositionedEvent>) {
    run_positioned_dims(bytes, 24, 80)
}

/// [`run_positioned_dims`], keeping only the semantic events.
fn run_dims(bytes: &[u8], rows: usize, cols: usize) -> (Screen, Vec<VtEvent>) {
    let (screen, events) = run_positioned_dims(bytes, rows, cols);
    (
        screen,
        events
            .into_iter()
            .map(PositionedEvent::into_event)
            .collect(),
    )
}

/// [`run_dims`] on a default 24x80 screen.
fn run(bytes: &[u8]) -> (Screen, Vec<VtEvent>) {
    run_dims(bytes, 24, 80)
}

/// Text of grid row `row` (trailing blanks trimmed).
fn text(s: &Screen, row: usize) -> String {
    s.row_text(s.absolute_row(row)).unwrap_or_default()
}

/// Text of the *active* grid's row `row` (trailing blanks trimmed).
///
/// Absolute rows address the primary timeline even while the alternate
/// screen is up, so assertions about what the alternate screen displays
/// must read the active grid rather than [`text`].
fn active_text(s: &Screen, row: usize) -> String {
    let cells = s.grid().row(row).unwrap_or_default();
    let rendered: String = cells
        .iter()
        .filter(|c| c.width != bb_terminal_model::CellWidth::WideTrail)
        .map(|c| c.ch)
        .collect();
    rendered.trim_end().to_owned()
}

fn pos(s: &Screen) -> (usize, usize) {
    (s.cursor().row, s.cursor().col)
}

// ---------------------------------------------------------------------
// Printing, UTF-8, C0
// ---------------------------------------------------------------------

#[test]
fn prints_plain_text() {
    let (s, ev) = run(b"hello");
    assert_eq!(text(&s, 0), "hello");
    assert_eq!(pos(&s), (0, 5));
    assert!(ev.is_empty());
}

#[test]
fn prints_utf8_multibyte_and_emoji() {
    let (s, _) = run("héllo 世界 🚀".as_bytes());
    assert_eq!(text(&s, 0), "héllo 世界 🚀");
    // h é l l o SP (6) + 世界 (4) + SP (1) + 🚀 (2) = 13 columns.
    assert_eq!(pos(&s), (0, 13));
}

#[test]
fn utf8_split_across_feed_calls() {
    let mut p = Parser::new();
    let mut s = Screen::new(24, 80);
    p.feed(&mut s, b"\xF0\x9F\x92");
    assert_eq!(text(&s, 0), ""); // nothing printed yet
    p.feed(&mut s, b"\x96");
    assert_eq!(text(&s, 0), "💖");
}

#[test]
fn invalid_utf8_emits_replacement() {
    let (s, _) = run(b"a\xFFb");
    assert_eq!(text(&s, 0), "a\u{FFFD}b");
    // Lone continuation and overlong lead each become one U+FFFD.
    let (s, _) = run(b"\x80\xC0");
    assert_eq!(text(&s, 0), "\u{FFFD}\u{FFFD}");
    // Truncated sequence interrupted by ASCII.
    let (s, _) = run(b"\xE4\xB8x");
    assert_eq!(text(&s, 0), "\u{FFFD}x");
}

#[test]
fn esc_interrupts_utf8_sequence() {
    let (s, _) = run(b"\xE4\xB8\x1b[CX");
    // U+FFFD printed, then cursor-forward, then X.
    assert_eq!(text(&s, 0), "\u{FFFD} X");
    assert_eq!(pos(&s), (0, 3));
}

#[test]
fn cr_lf_handling() {
    let (s, _) = run(b"one\r\ntwo\nthree");
    assert_eq!(text(&s, 0), "one");
    assert_eq!(text(&s, 1), "two");
    // Bare LF keeps the column.
    assert_eq!(text(&s, 2), "   three");
    assert_eq!(pos(&s), (2, 8));
}

#[test]
fn del_is_discarded_not_backspace() {
    let (s, _) = run(b"ab\x7f");
    assert_eq!(text(&s, 0), "ab");
    assert_eq!(pos(&s), (0, 2)); // cursor still after 'b'
    let (s, _) = run(b"ab\x7fc");
    assert_eq!(text(&s, 0), "abc");
    assert_eq!(pos(&s), (0, 3));
}

#[test]
fn backspace_and_tab() {
    let (s, _) = run(b"abc\x08x");
    assert_eq!(text(&s, 0), "abx");
    let (s, _) = run(b"\tx");
    assert_eq!(pos(&s), (0, 9)); // tab to col 8, then x
    assert_eq!(text(&s, 0), "        x");
}

#[test]
fn bell_event_in_ground() {
    let (s, ev) = run(b"a\x07b");
    assert_eq!(text(&s, 0), "ab");
    assert_eq!(ev, vec![VtEvent::Bell]);
}

#[test]
fn bell_mid_csi_is_not_an_alert() {
    let (s, ev) = run(b"\x1b[3\x071C x");
    // BEL executed silently mid-sequence; CSI 31 C moves right 31.
    assert!(ev.is_empty());
    assert_eq!(pos(&s), (0, 33)); // 31 right + " x"
    assert_eq!(text(&s, 0).trim_start(), "x");
}

#[test]
fn autowrap_wraps_long_lines() {
    let (s, _) = run_dims(b"abcdefg", 2, 5);
    assert_eq!(text(&s, 0), "abcde");
    assert_eq!(text(&s, 1), "fg");
}

// ---------------------------------------------------------------------
// ESC sequences
// ---------------------------------------------------------------------

#[test]
fn decsc_decrc_save_restore() {
    let (s, _) = run(b"\x1b[5;10H\x1b[31m\x1b7\x1b[0m\x1b[1;1Hx\x1b8");
    assert_eq!(pos(&s), (4, 9));
    assert_eq!(s.pen().fg, Color::Indexed(1)); // pen restored too
}

#[test]
fn ris_resets_everything() {
    let (s, ev) = run(b"hi\x1b[31m\x1b[?1049h\x1bc");
    assert!(ev.is_empty());
    assert!(!s.is_alt_screen());
    assert_eq!(s.pen().fg, Color::Default);
    assert_eq!(pos(&s), (0, 0));
    assert_eq!(text(&s, 0), "");
}

#[test]
fn nel_moves_to_next_line_col0() {
    let (s, _) = run(b"abc\x1bEx");
    assert_eq!(text(&s, 1), "x");
    assert_eq!(pos(&s), (1, 1));
}

#[test]
fn ind_scrolls_at_bottom_margin() {
    let (s, _) = run_dims(b"r0\r\nr1\r\nr2\x1bD\x1bDx", 3, 10);
    // Cursor was on the last row; two INDs scroll twice.
    assert_eq!(text(&s, 0), "r2");
    assert_eq!(text(&s, 2), "  x");
    // Scrolled-out rows land in scrollback.
    assert_eq!(s.scrollback().len(), 2);
    assert_eq!(s.row_text(0).as_deref(), Some("r0"));
    assert_eq!(s.row_text(1).as_deref(), Some("r1"));
}

#[test]
fn ri_scrolls_down_at_top_margin() {
    let (s, _) = run_dims(b"r0\r\nr1\x1b[1;1H\x1bMx", 3, 10);
    assert_eq!(text(&s, 0), "x");
    assert_eq!(text(&s, 1), "r0");
    assert_eq!(text(&s, 2), "r1");
}

#[test]
fn ri_moves_up_when_not_at_margin() {
    let (s, _) = run(b"\x1b[5;3H\x1bM");
    assert_eq!(pos(&s), (3, 2));
}

#[test]
fn charset_and_keypad_sequences_ignored() {
    let (s, _) = run(b"\x1b(B\x1b)0hi\x1b=x\x1b>y");
    assert_eq!(text(&s, 0), "hixy");
}

#[test]
fn st_alone_is_noop() {
    let (s, _) = run(b"a\x1b\\b");
    assert_eq!(text(&s, 0), "ab");
}

// ---------------------------------------------------------------------
// CSI cursor movement
// ---------------------------------------------------------------------

#[test]
fn cup_and_hvp_position_cursor() {
    let (s, _) = run(b"\x1b[5;10H");
    assert_eq!(pos(&s), (4, 9));
    let (s, _) = run(b"\x1b[3;4f");
    assert_eq!(pos(&s), (2, 3));
    // Defaults and empty params.
    let (s, _) = run(b"\x1b[5;10H\x1b[H");
    assert_eq!(pos(&s), (0, 0));
    let (s, _) = run(b"\x1b[;5H");
    assert_eq!(pos(&s), (0, 4));
    // Clamped to the grid.
    let (s, _) = run(b"\x1b[999;999H");
    assert_eq!(pos(&s), (23, 79));
}

#[test]
fn cuu_cud_cuf_cub() {
    let (s, _) = run(b"\x1b[10;20H\x1b[3A");
    assert_eq!(pos(&s), (6, 19));
    let (s, _) = run(b"\x1b[10;20H\x1b[2B");
    assert_eq!(pos(&s), (11, 19));
    let (s, _) = run(b"\x1b[10;20H\x1b[5C");
    assert_eq!(pos(&s), (9, 24));
    let (s, _) = run(b"\x1b[10;20H\x1b[4D");
    assert_eq!(pos(&s), (9, 15));
    // Default parameter is 1; clamping at edges.
    let (s, _) = run(b"\x1b[A\x1b[D");
    assert_eq!(pos(&s), (0, 0));
    let (s, _) = run(b"\x1b[999B\x1b[999C");
    assert_eq!(pos(&s), (23, 79));
}

#[test]
fn cnl_cpl_move_and_return() {
    let (s, _) = run(b"\x1b[10;20H\x1b[2E");
    assert_eq!(pos(&s), (11, 0));
    let (s, _) = run(b"\x1b[10;20H\x1b[3F");
    assert_eq!(pos(&s), (6, 0));
}

#[test]
fn cha_and_vpa() {
    let (s, _) = run(b"\x1b[10;20H\x1b[5G");
    assert_eq!(pos(&s), (9, 4));
    let (s, _) = run(b"\x1b[10;20H\x1b[7d");
    assert_eq!(pos(&s), (6, 19));
    let (s, _) = run(b"\x1b[10;20H\x1b[G");
    assert_eq!(pos(&s), (9, 0));
}

#[test]
fn cht_and_cbt_tab_navigation() {
    let (s, _) = run(b"\x1b[I");
    assert_eq!(pos(&s), (0, 8));
    let (s, _) = run(b"\x1b[2I");
    assert_eq!(pos(&s), (0, 16));
    let (s, _) = run(b"\x1b[3I\x1b[Z");
    assert_eq!(pos(&s), (0, 16));
    let (s, _) = run(b"\x1b[3I\x1b[2Z");
    assert_eq!(pos(&s), (0, 8));
    let (s, _) = run(b"\x1b[3I\x1b[99Z");
    assert_eq!(pos(&s), (0, 0));
}

#[test]
fn hts_sets_custom_stop_for_cbt() {
    // Set a custom stop at column 5, then back-tab from column 20.
    let (s, _) = run(b"\x1b[6G\x1bH\x1b[21G\x1b[Z\x1b[Z\x1b[Z");
    // 20 -> 16 (default) -> 8 (default) -> 5 (custom).
    assert_eq!(pos(&s), (0, 5));
    // Forward tab honors it too (via the screen's own stops).
    let (s, _) = run(b"\x1b[6G\x1bH\x1b[1G\t");
    assert_eq!(pos(&s), (0, 5));
}

#[test]
fn tbc_clears_stops() {
    // Clear the stop at column 8, tab jumps to 16.
    let (s, _) = run(b"\x1b[9G\x1b[g\x1b[1G\t");
    assert_eq!(pos(&s), (0, 16));
    // TBC 3 clears everything: tab goes to the last column, CBT to 0.
    let (s, _) = run(b"\x1b[3g\t");
    assert_eq!(pos(&s), (0, 79));
    let (s, _) = run(b"\x1b[3g\x1b[21G\x1b[Z");
    assert_eq!(pos(&s), (0, 0));
}

#[test]
fn ht_and_cbt_agree_after_a_rows_only_resize() {
    // The parser used to keep a private tab-stop mirror for CBT, rebuilt
    // only when the COLUMN count changed. `Screen::resize` rebuilds its
    // own stops unconditionally, so a rows-only resize desynchronized the
    // two: HT walked the screen's rebuilt every-8 defaults while CBT still
    // saw the custom stop the mirror had recorded (col 20 here). The
    // mirror is gone; both now read the screen.
    let make = || Screen::new(24, 40);
    let before: &[u8] = b"\x1b[3g\x1b[1;21H\x1bH"; // clear all stops, HTS at col 20
    let resize = |s: &mut Screen| s.resize(20, 40); // ROWS change only

    let (s, _) = run_split(make, before, resize, b"\x1b[1;1H\t");
    assert_eq!(pos(&s), (0, 8), "HT should use the rebuilt defaults");
    let (s, _) = run_split(make, before, resize, b"\x1b[1;40H\x1b[Z");
    assert_eq!(pos(&s), (0, 32), "CBT should use the rebuilt defaults");

    // Without a resize the custom stop is of course still honored by both.
    let (s, _) = run_split(make, before, |_| {}, b"\x1b[1;1H\t");
    assert_eq!(pos(&s), (0, 20));
    let (s, _) = run_split(make, before, |_| {}, b"\x1b[1;40H\x1b[Z");
    assert_eq!(pos(&s), (0, 20));
}

// ---------------------------------------------------------------------
// CSI editing
// ---------------------------------------------------------------------

#[test]
fn el_variants() {
    let (s, _) = run(b"abcdef\x1b[4G\x1b[K");
    assert_eq!(text(&s, 0), "abc");
    let (s, _) = run(b"abcdef\x1b[4G\x1b[1K");
    assert_eq!(text(&s, 0), "    ef");
    let (s, _) = run(b"abcdef\x1b[2K");
    assert_eq!(text(&s, 0), "");
}

#[test]
fn ed_variants() {
    let (s, _) = run(b"line1\r\nline2\r\nline3\x1b[2;3H\x1b[J");
    assert_eq!(text(&s, 0), "line1");
    assert_eq!(text(&s, 1), "li");
    assert_eq!(text(&s, 2), "");
    let (s, _) = run(b"line1\r\nline2\r\nline3\x1b[2;3H\x1b[1J");
    assert_eq!(text(&s, 0), "");
    assert_eq!(text(&s, 1), "   e2");
    assert_eq!(text(&s, 2), "line3");
}

#[test]
fn ed_2_clears_and_preserves_history() {
    let (s, _) = run(b"a\r\nb\x1b[2J");
    for r in 0..24 {
        assert_eq!(text(&s, r), "");
    }
    // The cleared screen was pushed to scrollback, so its rows are still
    // addressable. Only the rows that held something become history —
    // `Screen::erase_in_display` stops at the last non-blank row rather
    // than pushing the full height of the window.
    assert_eq!(s.scrollback().len(), 2);
    assert_eq!(s.row_text(0).as_deref(), Some("a"));
    assert_eq!(s.row_text(1).as_deref(), Some("b"));
}

#[test]
fn ed_3_erases_scrollback_and_leaves_the_grid_alone() {
    // `CSI 3 J` is xterm's "Erase Saved Lines": it drops the retained
    // history and does NOT touch the visible screen. This test previously
    // asserted the exact opposite (that 3J cleared the grid like 2J),
    // which is what let `clear` shred scrollback — see
    // `clear_sequence_does_not_bury_history_in_blank_scrollback`.
    let (s, _) = run_dims(b"a\r\nb\r\nc\r\nd\x1b[3J", 2, 10);
    // History is gone...
    assert_eq!(s.scrollback().len(), 0);
    // ...but the two on-screen rows are untouched.
    assert_eq!(text(&s, 0), "c");
    assert_eq!(text(&s, 1), "d");

    // Absolute ids stay monotonic across the erase: the ids the two
    // dropped rows held are never handed to the rows still on screen.
    assert_eq!(s.scrollback().evicted(), 2);
    assert_eq!(s.absolute_row(0), 2);
    assert_eq!(s.row_text(0), None);
    assert_eq!(s.row_text(1), None);
    assert_eq!(s.row_text(2).as_deref(), Some("c"));

    // With no scrollback to erase it is simply a no-op.
    let (s, _) = run(b"hello\x1b[3J");
    assert_eq!(text(&s, 0), "hello");
    assert_eq!(s.scrollback().len(), 0);
}

#[test]
fn clear_sequence_does_not_bury_history_in_blank_scrollback() {
    // `/usr/bin/clear` under TERM=xterm-256color emits exactly
    // `ESC[3J ESC[H ESC[2J`. Mapping the 3J onto EraseMode::All made every
    // `clear` push a *second*, blank screenful into scrollback: measured
    // over 300 clears on a 24-row screen, all 10,000 retained rows were
    // blank and 4,400 real rows had been evicted, unrecoverably.
    let clear: &[u8] = b"\x1b[3J\x1b[H\x1b[2J";
    let mut bytes = Vec::new();
    for i in 0..300 {
        bytes.extend_from_slice(format!("line{i}\r\n").as_bytes());
        bytes.extend_from_slice(clear);
    }
    let (s, _) = run_scrollback_dims(&bytes, 24, 80, 10_000);

    // Erase-Saved-Lines leads the sequence, so only the screenful the
    // trailing 2J pushed is ever retained — however many clears run.
    let retained = s.scrollback().len();
    assert!(
        retained <= 24,
        "scrollback grew to {retained} rows; blank screenfuls are accumulating"
    );
    // And the last real line of output is the oldest thing in it, rather
    // than being buried under blanks or pushed out of the ring entirely.
    assert_eq!(
        s.row_text(s.scrollback().evicted()).as_deref(),
        Some("line299")
    );
}

#[test]
fn el_3_is_undefined_and_ignored() {
    // Only EL 0/1/2 are defined; `3` is an ED-only selector. It used to
    // share ED's mapping and erase the whole line.
    let (s, _) = run(b"abcdef\x1b[3K");
    assert_eq!(text(&s, 0), "abcdef");
}

#[test]
fn ich_dch_ech() {
    let (s, _) = run(b"abcdef\r\x1b[2C\x1b[3@");
    assert_eq!(text(&s, 0), "ab   cdef");
    let (s, _) = run(b"abcdef\r\x1b[2P");
    assert_eq!(text(&s, 0), "cdef");
    let (s, _) = run(b"abcdef\r\x1b[3X");
    assert_eq!(text(&s, 0), "   def");
}

#[test]
fn il_dl() {
    let (s, _) = run_dims(b"r0\r\nr1\r\nr2\x1b[1;1H\x1b[L", 4, 10);
    assert_eq!(text(&s, 0), "");
    assert_eq!(text(&s, 1), "r0");
    assert_eq!(text(&s, 2), "r1");
    assert_eq!(text(&s, 3), "r2");
    let (s, _) = run_dims(b"r0\r\nr1\r\nr2\x1b[1;1H\x1b[2M", 4, 10);
    assert_eq!(text(&s, 0), "r2");
    assert_eq!(text(&s, 1), "");
}

#[test]
fn su_sd() {
    let (s, _) = run_dims(b"r0\r\nr1\r\nr2\x1b[S", 3, 10);
    assert_eq!(text(&s, 0), "r1");
    assert_eq!(text(&s, 2), "");
    assert_eq!(s.row_text(0).as_deref(), Some("r0")); // pushed to scrollback
    let (s, _) = run_dims(b"r0\r\nr1\r\nr2\x1b[T", 3, 10);
    assert_eq!(text(&s, 0), "");
    assert_eq!(text(&s, 1), "r0");
    assert_eq!(text(&s, 2), "r1");
}

// ---------------------------------------------------------------------
// Scroll region, DECOM
// ---------------------------------------------------------------------

#[test]
fn decstbm_homes_cursor_and_scopes_scrolling() {
    let (s, _) = run_dims(b"r0\r\nr1\r\nr2\r\nr3\r\nr4\x1b[2;4r", 5, 10);
    assert_eq!(pos(&s), (0, 0));
    // LF at the region's bottom margin scrolls only the region.
    let (s, _) = run_dims(b"r0\r\nr1\r\nr2\r\nr3\r\nr4\x1b[2;4r\x1b[4;1H\n", 5, 10);
    assert_eq!(text(&s, 0), "r0");
    assert_eq!(text(&s, 1), "r2");
    assert_eq!(text(&s, 2), "r3");
    assert_eq!(text(&s, 3), "");
    assert_eq!(text(&s, 4), "r4");
    assert_eq!(s.scrollback().len(), 0); // partial region: no scrollback
}

#[test]
fn ri_at_region_top_scrolls_region_down() {
    let (s, _) = run_dims(b"r0\r\nr1\r\nr2\r\nr3\r\nr4\x1b[2;4r\x1b[2;1H\x1bM", 5, 10);
    assert_eq!(text(&s, 0), "r0");
    assert_eq!(text(&s, 1), "");
    assert_eq!(text(&s, 2), "r1");
    assert_eq!(text(&s, 3), "r2");
    assert_eq!(text(&s, 4), "r4");
}

#[test]
fn decstbm_default_resets_region() {
    let (s, _) = run_dims(b"\x1b[2;4r\x1b[r", 5, 10);
    assert_eq!(s.scroll_region(), (0, 4));
}

#[test]
fn decom_addresses_relative_to_region() {
    let (s, _) = run(b"\x1b[3;6r\x1b[?6h");
    assert_eq!(pos(&s), (2, 0)); // homed to the region origin
    let (s, _) = run(b"\x1b[3;6r\x1b[?6h\x1b[2;1H");
    assert_eq!(pos(&s), (3, 0)); // row 2 = region top + 1
    let (s, _) = run(b"\x1b[3;6r\x1b[?6h\x1b[99;1H");
    assert_eq!(pos(&s), (5, 0)); // clamped to the region bottom
    let (s, _) = run(b"\x1b[3;6r\x1b[?6h\x1b[?6l");
    assert_eq!(pos(&s), (0, 0));
    assert!(!s.origin_mode());
}

// ---------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------

#[test]
fn irm_insert_mode() {
    let (s, _) = run(b"abc\r\x1b[4hXY");
    assert_eq!(text(&s, 0), "XYabc");
    let (s, _) = run(b"abc\r\x1b[4h\x1b[4lXY");
    assert_eq!(text(&s, 0), "XYc"); // replace mode again
}

#[test]
fn decawm_autowrap_toggle() {
    let (s, _) = run_dims(b"\x1b[?7labcdefgh", 2, 5);
    assert_eq!(text(&s, 0), "abcdh"); // overwrites the last column
    assert_eq!(text(&s, 1), "");
    let (s, _) = run_dims(b"\x1b[?7l\x1b[?7habcdefgh", 2, 5);
    assert_eq!(text(&s, 1), "fgh");
}

#[test]
fn dectcem_cursor_visibility() {
    let (s, _) = run(b"\x1b[?25l");
    assert!(!s.cursor().visible);
    let (s, _) = run(b"\x1b[?25l\x1b[?25h");
    assert!(s.cursor().visible);
}

#[test]
fn multiple_modes_in_one_sequence() {
    let (s, _) = run(b"\x1b[?25;7l");
    assert!(!s.cursor().visible);
    assert!(!s.autowrap());
}

#[test]
fn unknown_modes_parsed_and_ignored() {
    let (s, _) = run(b"\x1b[?2004h\x1b[?1000h\x1b[20hx");
    assert_eq!(text(&s, 0), "x");
}

// ---------------------------------------------------------------------
// Alternate screen
// ---------------------------------------------------------------------

#[test]
fn alt_screen_1049_round_trip() {
    let (s, _) = run(b"hello\x1b[?1049h");
    assert!(s.is_alt_screen());
    assert_eq!(active_text(&s, 0), ""); // alt starts cleared
    assert_eq!(text(&s, 0), "hello"); // primary timeline untouched
    let (s, _) = run(b"hello\x1b[?1049h\x1b[Hworld\x1b[?1049l");
    assert!(!s.is_alt_screen());
    assert_eq!(text(&s, 0), "hello"); // primary content intact
    assert_eq!(pos(&s), (0, 5)); // cursor restored
}

#[test]
fn alt_screen_47_and_1047() {
    let (s, _) = run(b"hi\x1b[?47h");
    assert!(s.is_alt_screen());
    let (s, _) = run(b"hi\x1b[?47h\x1b[?47l");
    assert!(!s.is_alt_screen());
    assert_eq!(text(&s, 0), "hi");
    let (s, _) = run(b"\x1b[?1047h");
    assert!(s.is_alt_screen());
}

// ---------------------------------------------------------------------
// DECSCUSR
// ---------------------------------------------------------------------

#[test]
fn decscusr_cursor_shapes() {
    let (s, _) = run(b"\x1b[4 q");
    assert_eq!(s.cursor().shape, CursorShape::Underline);
    let (s, _) = run(b"\x1b[6 q");
    assert_eq!(s.cursor().shape, CursorShape::Bar);
    let (s, _) = run(b"\x1b[6 q\x1b[ q");
    assert_eq!(s.cursor().shape, CursorShape::Block);
    let (s, _) = run(b"\x1b[6 q\x1b[2 q");
    assert_eq!(s.cursor().shape, CursorShape::Block);
    let (s, _) = run(b"\x1b[6 q\x1b[99 q"); // out of range: ignored
    assert_eq!(s.cursor().shape, CursorShape::Bar);
}

// ---------------------------------------------------------------------
// DECSTR
// ---------------------------------------------------------------------

#[test]
fn decstr_soft_reset() {
    // `CSI ! p` used to be parsed and dropped: `csi_dispatch` bailed out
    // on any intermediate other than SP (DECSCUSR). Shells send DECSTR to
    // recover a wedged terminal, so the recovery path did nothing at all.
    let (s, _) = run(b"\x1b[?6h\x1b[4h\x1b[3;6r\x1b[1;31m\x1b[?25l\x1b[!p");
    assert!(!s.origin_mode());
    assert!(!s.insert_mode());
    assert_eq!(s.scroll_region(), (0, 23));
    assert_eq!(*s.pen(), bb_terminal_model::CellAttrs::default());
    assert!(s.cursor().visible);
    assert!(!s.pending_wrap());

    // It clears the pending-wrap flag: printing after a soft reset at the
    // right margin overwrites the last column instead of wrapping.
    let (s, _) = run_dims(b"abcde\x1b[!pX", 2, 5);
    assert_eq!(text(&s, 0), "abcdX");
    assert_eq!(text(&s, 1), "");

    // It does NOT clear the screen or the scrollback.
    let (s, _) = run_dims(b"gone\r\nkept\r\nalso\x1b[!p", 2, 10);
    assert_eq!(text(&s, 0), "kept");
    assert_eq!(text(&s, 1), "also");
    assert_eq!(s.scrollback().len(), 1);
    assert_eq!(s.row_text(0).as_deref(), Some("gone"));
}

#[test]
fn decstr_requires_no_private_marker() {
    // `CSI ? ! p` is not DECSTR; it must be parsed and ignored.
    let (s, _) = run(b"\x1b[4h\x1b[?!p");
    assert!(s.insert_mode());
}

// ---------------------------------------------------------------------
// SGR
// ---------------------------------------------------------------------

#[test]
fn sgr_basic_attributes() {
    let (s, _) = run(b"\x1b[1;2;3;4;5;7;8;9m");
    let pen = s.pen();
    assert!(pen.bold && pen.dim && pen.italic && pen.underline);
    assert!(pen.blink && pen.reverse && pen.hidden && pen.strikethrough);
}

#[test]
fn sgr_resets() {
    let (s, _) = run(b"\x1b[1;2;3;4;5;7;8;9m\x1b[22;23;24;25;27;28;29m");
    assert_eq!(*s.pen(), bb_terminal_model::CellAttrs::default());
    // SGR 0 resets everything, including colors.
    let (s, _) = run(b"\x1b[1;31;42m\x1b[0m");
    assert_eq!(*s.pen(), bb_terminal_model::CellAttrs::default());
    // Empty SGR = SGR 0.
    let (s, _) = run(b"\x1b[1;31m\x1b[m");
    assert_eq!(*s.pen(), bb_terminal_model::CellAttrs::default());
    // 21 clears bold. KNOWN DIVERGENCE: ECMA-48 and xterm both define 21
    // as doubly-underlined, but `CellAttrs::underline` has no style, so
    // we follow the rival bold-off convention instead. See `sgr.rs`.
    let (s, _) = run(b"\x1b[1m\x1b[21m");
    assert!(!s.pen().bold);
    // 22 is "normal intensity" and clears both, which the model can
    // represent exactly — `bold` and `dim` are independent fields.
    let (s, _) = run(b"\x1b[1;2m\x1b[22m");
    assert!(!s.pen().bold && !s.pen().dim);
}

#[test]
fn sgr_indexed_colors() {
    let (s, _) = run(b"\x1b[31m");
    assert_eq!(s.pen().fg, Color::Indexed(1));
    let (s, _) = run(b"\x1b[44m");
    assert_eq!(s.pen().bg, Color::Indexed(4));
    let (s, _) = run(b"\x1b[31;39m");
    assert_eq!(s.pen().fg, Color::Default);
    let (s, _) = run(b"\x1b[44;49m");
    assert_eq!(s.pen().bg, Color::Default);
}

#[test]
fn sgr_bright_colors() {
    let (s, _) = run(b"\x1b[91m");
    assert_eq!(s.pen().fg, Color::Indexed(9));
    let (s, _) = run(b"\x1b[105m");
    assert_eq!(s.pen().bg, Color::Indexed(13));
}

#[test]
fn sgr_256_color() {
    let (s, _) = run(b"\x1b[38;5;123m");
    assert_eq!(s.pen().fg, Color::Indexed(123));
    let (s, _) = run(b"\x1b[48;5;200m");
    assert_eq!(s.pen().bg, Color::Indexed(200));
    // Out-of-range index saturates.
    let (s, _) = run(b"\x1b[38;5;999m");
    assert_eq!(s.pen().fg, Color::Indexed(255));
}

#[test]
fn sgr_truecolor_semicolon_form() {
    let (s, _) = run(b"\x1b[38;2;10;20;30m");
    assert_eq!(s.pen().fg, Color::Rgb(10, 20, 30));
    let (s, _) = run(b"\x1b[48;2;255;0;128m");
    assert_eq!(s.pen().bg, Color::Rgb(255, 0, 128));
    // Attributes after the color are still applied.
    let (s, _) = run(b"\x1b[38;2;1;2;3;1m");
    assert_eq!(s.pen().fg, Color::Rgb(1, 2, 3));
    assert!(s.pen().bold);
}

#[test]
fn sgr_colon_forms() {
    let (s, _) = run(b"\x1b[38:5:123m");
    assert_eq!(s.pen().fg, Color::Indexed(123));
    let (s, _) = run(b"\x1b[38:2:10:20:30m");
    assert_eq!(s.pen().fg, Color::Rgb(10, 20, 30));
    // ITU T.416 form with a colorspace id.
    let (s, _) = run(b"\x1b[38:2::10:20:30m");
    assert_eq!(s.pen().fg, Color::Rgb(10, 20, 30));
    let (s, _) = run(b"\x1b[48:2:0:40:50:60m");
    assert_eq!(s.pen().bg, Color::Rgb(40, 50, 60));
    // Colon form does not swallow following parameters.
    let (s, _) = run(b"\x1b[38:5:9;1m");
    assert_eq!(s.pen().fg, Color::Indexed(9));
    assert!(s.pen().bold);
}

#[test]
fn sgr_applies_to_printed_cells() {
    let (s, _) = run(b"\x1b[1;31mX");
    let cell = s.grid().cell(0, 0).unwrap();
    assert_eq!(cell.ch, 'X');
    assert!(cell.attrs.bold);
    assert_eq!(cell.attrs.fg, Color::Indexed(1));
}

// ---------------------------------------------------------------------
// Parameter robustness
// ---------------------------------------------------------------------

#[test]
fn params_saturate_and_cap() {
    // Huge values saturate instead of overflowing.
    let (s, _) = run(b"\x1b[4294967295C");
    assert_eq!(pos(&s), (0, 79));
    // More than 16 parameters: extras are dropped (the 17th, 31, would
    // set the foreground red if it were retained).
    let (s, _) = run(b"\x1b[0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;31m");
    assert_eq!(s.pen().fg, Color::Default);
    // Exactly 16 params: the last one still lands.
    let (s, _) = run(b"\x1b[0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;31m");
    assert_eq!(s.pen().fg, Color::Indexed(1));
}

#[test]
fn can_aborts_csi() {
    let (s, _) = run(b"\x1b[3\x18A");
    assert_eq!(text(&s, 0), "A"); // sequence aborted, 'A' prints
    let (s, _) = run(b"\x1b[3\x1aA");
    assert_eq!(text(&s, 0), "A");
}

#[test]
fn unknown_csi_with_intermediates_ignored() {
    let (s, _) = run(b"\x1b[1$p\x1b[?1$px");
    assert_eq!(text(&s, 0), "x");
}

#[test]
fn private_marker_mid_params_ignored() {
    let (s, _) = run(b"\x1b[1?25lx"); // malformed: '?' after digits
    assert!(s.cursor().visible);
    assert_eq!(text(&s, 0), "x");
}

#[test]
fn escape_intermediates_are_bounded() {
    // `ESC` followed by a run of 0x20..=0x2F used to be pushed into the
    // parser's intermediate buffer with no cap at all, so the live heap
    // grew byte-for-byte with the input and stayed there until the Parser
    // was dropped: 8 MiB of `/` cost 8,388,608 retained bytes. SPACE is in
    // that range too, so ordinary padded output could trigger it.
    //
    // `Parser`'s derived Debug embeds the buffer, so an unbounded one is
    // observable from outside the crate without an allocator hook.
    for filler in [b'/', b' '] {
        // Two intermediates is the cap, so this is the widest state the
        // parser is ever allowed to be in mid-sequence.
        let baseline = {
            let mut screen = Screen::new(24, 80);
            let mut parser = Parser::new();
            parser.feed(&mut screen, &[0x1B, filler, filler]);
            format!("{parser:?}")
        };

        let mut screen = Screen::new(24, 80);
        let mut parser = Parser::new();
        parser.feed(&mut screen, b"\x1b");
        parser.feed(&mut screen, &vec![filler; 1 << 20]); // 1 MiB
        assert_eq!(
            format!("{parser:?}"),
            baseline,
            "{:?} intermediates were retained instead of capped",
            char::from(filler),
        );

        // The sequence is still consumed correctly: its final byte ends
        // it (and is ignored, since intermediates are present), and
        // ordinary printing resumes immediately after.
        parser.feed(&mut screen, b"0after");
        assert_eq!(text(&screen, 0), "after");
    }

    // The screen-visible half of the same fix, run through the standard
    // chunking-invariant harness: over-long intermediate runs are dropped
    // without disturbing where the sequence ends.
    for bytes in [&b"\x1b//////0after"[..], b"\x1b      0after"] {
        let (s, _) = run(bytes);
        assert_eq!(text(&s, 0), "after");
    }
}

// ---------------------------------------------------------------------
// OSC
// ---------------------------------------------------------------------

#[test]
fn osc_title_bel_terminated() {
    let (_, ev) = run(b"\x1b]0;My Title\x07");
    assert_eq!(ev, vec![VtEvent::Title("My Title".into())]);
}

#[test]
fn osc_title_st_terminated() {
    let (_, ev) = run(b"\x1b]2;Other\x1b\\");
    assert_eq!(ev, vec![VtEvent::Title("Other".into())]);
    let (_, ev) = run(b"\x1b]2;\x07"); // empty title
    assert_eq!(ev, vec![VtEvent::Title(String::new())]);
}

#[test]
fn osc_title_utf8() {
    let (_, ev) = run("\x1b]0;héllo 🚀\x07".as_bytes());
    assert_eq!(ev, vec![VtEvent::Title("héllo 🚀".into())]);
}

#[test]
fn osc7_working_directory() {
    let (_, ev) = run(b"\x1b]7;file://host/Users/me/src\x07");
    assert_eq!(
        ev,
        vec![VtEvent::WorkingDirectory("file://host/Users/me/src".into())]
    );
}

#[test]
fn osc8_hyperlink_open_and_close() {
    let (_, ev) = run(b"\x1b]8;;https://example.com\x07link\x1b]8;;\x07");
    assert_eq!(
        ev,
        vec![
            VtEvent::Hyperlink(Some("https://example.com".into())),
            VtEvent::Hyperlink(None),
        ]
    );
    // With id params.
    let (_, ev) = run(b"\x1b]8;id=1;https://x.dev\x1b\\");
    assert_eq!(ev, vec![VtEvent::Hyperlink(Some("https://x.dev".into()))]);
    // Malformed (no second separator): ignored.
    let (_, ev) = run(b"\x1b]8;nourl\x07");
    assert!(ev.is_empty());
}

#[test]
fn osc133_prompt_lifecycle() {
    let (_, ev) = run(b"\x1b]133;A\x07$ \x1b]133;B\x07ls\r\n\x1b]133;C\x07out\r\n\x1b]133;D;0\x07");
    assert_eq!(
        ev,
        vec![
            VtEvent::PromptStart,
            VtEvent::CommandStart,
            VtEvent::CommandExecuted,
            VtEvent::CommandFinished { exit_code: Some(0) },
        ]
    );
}

#[test]
fn osc133_finished_variants() {
    let (_, ev) = run(b"\x1b]133;D\x07");
    assert_eq!(ev, vec![VtEvent::CommandFinished { exit_code: None }]);
    let (_, ev) = run(b"\x1b]133;D;127\x1b\\");
    assert_eq!(
        ev,
        vec![VtEvent::CommandFinished {
            exit_code: Some(127)
        }]
    );
    // Extra key=value fields after the code (some shells send these).
    let (_, ev) = run(b"\x1b]133;D;1;aid=42\x07");
    assert_eq!(ev, vec![VtEvent::CommandFinished { exit_code: Some(1) }]);
    // Non-numeric code: reported as None.
    let (_, ev) = run(b"\x1b]133;D;abc\x07");
    assert_eq!(ev, vec![VtEvent::CommandFinished { exit_code: None }]);
    // Unknown subcommand: ignored.
    let (_, ev) = run(b"\x1b]133;P;k=v\x07");
    assert!(ev.is_empty());
}

#[test]
fn osc_unknown_codes_ignored() {
    let (s, ev) = run(b"\x1b]52;c;aGVsbG8=\x07\x1b]666;x\x1b\\ok");
    assert!(ev.is_empty());
    assert_eq!(text(&s, 0), "ok");
}

#[test]
fn osc_split_across_feed_calls() {
    let mut p = Parser::new();
    let mut s = Screen::new(24, 80);
    // Two printed lines first, so the reported position is non-trivial.
    assert!(p.feed(&mut s, b"out\r\nmore").is_empty());
    assert!(p.feed(&mut s, b"\x1b]133;D;").is_empty());
    assert!(p.feed(&mut s, b"1").is_empty());
    let ev = p.feed(&mut s, b"2\x07");
    assert_eq!(
        ev,
        vec![PositionedEvent::new(
            VtEvent::CommandFinished {
                exit_code: Some(12)
            },
            1,
            4,
        )]
    );
    // ST split between the ESC and the backslash.
    let mut p = Parser::new();
    assert!(p.feed(&mut s, b"\x1b]0;T\x1b").is_empty());
    assert_eq!(
        p.feed(&mut s, b"\\"),
        vec![PositionedEvent::new(VtEvent::Title("T".into()), 1, 4)]
    );
}

#[test]
fn overlong_osc_discarded() {
    let mut bytes = b"\x1b]0;".to_vec();
    bytes.extend(std::iter::repeat(b'x').take(5000));
    bytes.push(0x07);
    bytes.extend_from_slice(b"ok");
    let (s, ev) = run(&bytes);
    assert!(ev.is_empty()); // no truncated title event
    assert_eq!(text(&s, 0), "ok"); // parsing continues cleanly
}

#[test]
fn osc_aborted_by_can() {
    let (s, ev) = run(b"\x1b]0;title\x1aX");
    assert!(ev.is_empty());
    assert_eq!(text(&s, 0), "X");
}

#[test]
fn osc_aborted_by_esc_non_st() {
    // ESC not followed by '\' aborts the OSC and starts a new sequence.
    let (s, ev) = run(b"\x1b]0;title\x1b[31mx");
    assert!(ev.is_empty());
    assert_eq!(s.pen().fg, Color::Indexed(1));
    assert_eq!(text(&s, 0), "x");
}

// ---------------------------------------------------------------------
// Positioned events
//
// Events carry the cursor position captured at the byte that completed
// the sequence. Without that, a consumer has to read the cursor after
// `feed` returns, which yields the end-of-chunk position for *every*
// event in the chunk — collapsing all of a 64 KiB read's command blocks
// onto one row.
// ---------------------------------------------------------------------

/// A three-command session with OSC 133 markers, as a shell with
/// integration enabled emits it. Each command occupies its own rows.
const THREE_COMMANDS: &[u8] = b"\x1b]133;A\x07$ \x1b]133;B\x07ls\r\n\
     \x1b]133;C\x07a.txt\r\n\x1b]133;D;0\x07\
     \x1b]133;A\x07$ \x1b]133;B\x07build\r\n\
     \x1b]133;C\x07done\r\n\x1b]133;D;0\x07\
     \x1b]133;A\x07$ \x1b]133;B\x07false\r\n\
     \x1b]133;C\x07\x1b]133;D;1\x07";

/// Where each marker in [`THREE_COMMANDS`] actually occurs.
fn three_commands_expected() -> Vec<PositionedEvent> {
    vec![
        // "$ ls" on row 0, output "a.txt" on row 1.
        PositionedEvent::new(VtEvent::PromptStart, 0, 0),
        PositionedEvent::new(VtEvent::CommandStart, 0, 2),
        PositionedEvent::new(VtEvent::CommandExecuted, 1, 0),
        PositionedEvent::new(VtEvent::CommandFinished { exit_code: Some(0) }, 2, 0),
        // "$ build" on row 2, output "done" on row 3.
        PositionedEvent::new(VtEvent::PromptStart, 2, 0),
        PositionedEvent::new(VtEvent::CommandStart, 2, 2),
        PositionedEvent::new(VtEvent::CommandExecuted, 3, 0),
        PositionedEvent::new(VtEvent::CommandFinished { exit_code: Some(0) }, 4, 0),
        // "$ false" on row 4, no output.
        PositionedEvent::new(VtEvent::PromptStart, 4, 0),
        PositionedEvent::new(VtEvent::CommandStart, 4, 2),
        PositionedEvent::new(VtEvent::CommandExecuted, 5, 0),
        PositionedEvent::new(VtEvent::CommandFinished { exit_code: Some(1) }, 5, 0),
    ]
}

/// Feed `bytes` into a screen from `make_screen`, in `chunk`-sized pieces
/// (`chunk == 0` means one single call with the whole slice).
fn feed_chunked(
    make_screen: impl Fn() -> Screen,
    bytes: &[u8],
    chunk: usize,
) -> (Screen, Vec<PositionedEvent>) {
    let mut parser = Parser::new();
    let mut screen = make_screen();
    let mut events = Vec::new();
    if chunk == 0 {
        events.extend(parser.feed(&mut screen, bytes));
    } else {
        for piece in bytes.chunks(chunk) {
            events.extend(parser.feed(&mut screen, piece));
        }
    }
    (screen, events)
}

/// Assert that every chunking of `bytes` — one byte at a time up to one
/// 64 KiB read — yields byte-identical screens and positioned events.
fn assert_chunking_invariant(
    make_screen: impl Fn() -> Screen + Copy,
    bytes: &[u8],
) -> Vec<PositionedEvent> {
    let (screen, events) = feed_chunked(make_screen, bytes, 0);
    for chunk in [1, 2, 3, 5, 7, 13, 64, 256, 65_536] {
        let (other_screen, other_events) = feed_chunked(make_screen, bytes, chunk);
        assert_eq!(screen, other_screen, "screen differs at chunk size {chunk}");
        assert_eq!(
            events, other_events,
            "positioned events differ at chunk size {chunk}"
        );
    }
    events
}

#[test]
fn osc133_markers_carry_their_own_row_in_one_shot_feed() {
    // The whole transcript in ONE call — the common case, since a real
    // terminal reads the PTY in chunks of up to 64 KiB.
    let mut parser = Parser::new();
    let mut screen = Screen::new(24, 80);
    let events = parser.feed(&mut screen, THREE_COMMANDS);
    assert_eq!(events, three_commands_expected());

    // Every prompt marker lands on a distinct, increasing row, even
    // though the cursor ends up on row 5 for all of them if you read it
    // after the call.
    let prompt_rows: Vec<usize> = events
        .iter()
        .filter(|e| e.event == VtEvent::PromptStart)
        .map(|e| e.cursor_row)
        .collect();
    assert_eq!(prompt_rows, vec![0, 2, 4]);
    assert!(prompt_rows.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(screen.cursor_absolute_row(), 5);

    // The rows are real: they hold the command lines they mark.
    assert_eq!(screen.row_text(0).as_deref(), Some("$ ls"));
    assert_eq!(screen.row_text(1).as_deref(), Some("a.txt"));
    assert_eq!(screen.row_text(2).as_deref(), Some("$ build"));
    assert_eq!(screen.row_text(3).as_deref(), Some("done"));
    assert_eq!(screen.row_text(4).as_deref(), Some("$ false"));
}

#[test]
fn osc133_positions_identical_across_every_chunking() {
    // The regression test for the chunk-dependent-block-boundaries bug:
    // one byte at a time, 3 bytes at a time, and one 64 KiB call must all
    // produce the exact same Vec<PositionedEvent>.
    let events = assert_chunking_invariant(|| Screen::new(24, 80), THREE_COMMANDS);
    assert_eq!(events, three_commands_expected());
}

#[test]
fn marker_row_accounts_for_scrollback_eviction() {
    // A 3-row grid retaining only 2 scrollback lines: printing 6 lines
    // pushes 3 out of the grid and evicts the oldest from scrollback.
    // Absolute rows keep counting past the evicted row, so a marker
    // recorded before the eviction still names the row it fired on.
    let make = || Screen::with_scrollback(3, 20, 2);
    let bytes = b"\x1b]133;A\x07l0\r\nl1\r\nl2\r\nl3\r\nl4\r\n\x1b]133;A\x07";
    let events = assert_chunking_invariant(make, bytes);
    let (screen, _) = feed_chunked(make, bytes, 0);

    assert_eq!(screen.scrollback().evicted(), 1, "expected one eviction");
    assert_eq!(
        events,
        vec![
            PositionedEvent::new(VtEvent::PromptStart, 0, 0),
            PositionedEvent::new(VtEvent::PromptStart, 5, 0),
        ]
    );
    // Row 0 has been evicted, but its id was never reused, so the first
    // marker is still unambiguous.
    assert!(events[0].cursor_row < screen.scrollback().evicted());
    assert_eq!(screen.row_text(events[0].cursor_row), None);
    // The second marker is at the live cursor; the first is not. Reading
    // the cursor after the feed would have reported 5 for both.
    assert_eq!(events[1].cursor_row, screen.cursor_absolute_row());
    assert_ne!(events[0].cursor_row, screen.cursor_absolute_row());
}

#[test]
fn title_cwd_hyperlink_and_bell_carry_positions() {
    let (_, ev) = run_positioned(
        b"boot\x1b]0;term\x07\r\n\x1b]7;file:///tmp\x07see \
          \x1b]8;;https://x.dev\x07link\x1b]8;;\x07!\x07",
    );
    assert_eq!(
        ev,
        vec![
            // After "boot" on row 0.
            PositionedEvent::new(VtEvent::Title("term".into()), 0, 4),
            // Start of row 1.
            PositionedEvent::new(VtEvent::WorkingDirectory("file:///tmp".into()), 1, 0),
            // After "see ".
            PositionedEvent::new(VtEvent::Hyperlink(Some("https://x.dev".into())), 1, 4),
            // After "link".
            PositionedEvent::new(VtEvent::Hyperlink(None), 1, 8),
            // After "!".
            PositionedEvent::new(VtEvent::Bell, 1, 9),
        ]
    );
}

#[test]
fn positions_follow_cursor_addressing_and_scrolling() {
    // Absolute addressing: the marker reports where CUP put the cursor.
    let (_, ev) = run_positioned(b"\x1b[5;10H\x1b]133;A\x07");
    assert_eq!(ev, vec![PositionedEvent::new(VtEvent::PromptStart, 4, 9)]);

    // A marker after the grid has scrolled reports the absolute row, not
    // the grid row, so it stays valid as the row moves into scrollback.
    let (screen, ev) = run_positioned_dims(b"a\r\nb\r\nc\r\nd\x1b]133;C\x07", 3, 10);
    assert_eq!(
        ev,
        vec![PositionedEvent::new(VtEvent::CommandExecuted, 3, 1)]
    );
    assert_eq!(screen.cursor().row, 2); // grid row differs from row 3
    assert_eq!(screen.row_text(3).as_deref(), Some("d"));
}

#[test]
fn positioned_event_conversions() {
    let event = PositionedEvent::new(VtEvent::Bell, 7, 3);
    assert_eq!(event.cursor(), (7, 3));
    assert_eq!(event.clone().into_event(), VtEvent::Bell);
    assert_eq!(VtEvent::from(event), VtEvent::Bell);

    // The ergonomic "I only care what happened" path.
    let (_, ev) = run_positioned(b"\x1b]133;A\x07x\x07");
    let plain: Vec<VtEvent> = ev.into_iter().map(PositionedEvent::into_event).collect();
    assert_eq!(plain, vec![VtEvent::PromptStart, VtEvent::Bell]);
}

// ---------------------------------------------------------------------
// DCS / SOS / PM / APC
// ---------------------------------------------------------------------

#[test]
fn dcs_content_discarded() {
    let (s, ev) = run(b"\x1bPq#0;2;0;0;0#0~~\x1b\\after");
    assert!(ev.is_empty());
    assert_eq!(text(&s, 0), "after");
}

#[test]
fn dcs_body_discards_c0_but_esc_cancels() {
    // This test used to assert that a lone ESC inside a DCS did *not*
    // terminate it — the divergence that let one stray `ESC P` wedge the
    // parser forever. Per the DEC state diagram ESC is an "anywhere"
    // transition: C0 bytes are part of the discarded payload, ESC is not.
    let (s, ev) = run(b"\x1bPdata\x07more\x1b\\x");
    assert!(ev.is_empty()); // the BEL was payload, not an alert
    assert_eq!(text(&s, 0), "x");

    // ESC ESC \ still lands in ground: the first ESC cancels the string,
    // the second restarts the escape, and `\` is a lone ST (a no-op).
    let (s, ev) = run(b"\x1bPdata\x1b\x1b\\x");
    assert!(ev.is_empty());
    assert_eq!(text(&s, 0), "x");

    // The cancelling ESC hands its byte straight to the new sequence.
    let (s, _) = run(b"\x1bPsixel\x1b[31mred");
    assert_eq!(text(&s, 0), "red");
    assert_eq!(s.pen().fg, Color::Indexed(1));
}

#[test]
fn esc_cancels_every_control_string_type() {
    // `\x1bPq\x1b[0mHELLO` used to leave row 0 empty; xterm prints HELLO.
    for opener in [&b"\x1bPq"[..], b"\x1bX", b"\x1b^", b"\x1b_"] {
        let mut bytes = opener.to_vec();
        bytes.extend_from_slice(b"\x1b[0mHELLO");
        let (s, _) = run(&bytes);
        assert_eq!(text(&s, 0), "HELLO", "{opener:?} still swallowed output");
    }
}

#[test]
fn unterminated_control_string_recovers_at_the_next_escape() {
    // A stray `ESC P` used to wedge the parser permanently: every later
    // byte — kilobytes of output, the OSC 133 markers, the prompt — was
    // discarded until a CAN/SUB/ST that a real shell never sends. The
    // next ESC in the stream now cancels the string and recovers.
    let mut bytes = b"\x1bPtruncated-sixel-payload".to_vec();
    bytes.extend_from_slice(b"\x1b[0m"); // first ESC: the recovery point
    bytes.extend(std::iter::repeat(b'o').take(8192));
    bytes.extend_from_slice(b"\r\n\x1b]133;A\x07$ ");
    let (s, ev) = run(&bytes);

    // The prompt marker survived, ...
    assert_eq!(ev, vec![VtEvent::PromptStart]);
    // ... the 8 KiB of output is on screen (wrapped across 103 rows), ...
    assert_eq!(s.row_text(0).as_deref(), Some("o".repeat(80).as_str()));
    // ... and the prompt is on the last row (`row_text` trims blanks).
    assert_eq!(text(&s, 23), "$");
    assert_eq!(pos(&s), (23, 2));
}

#[test]
fn sos_pm_apc_discarded() {
    let (s, _) = run(b"\x1b_apc payload\x1b\\x\x1b^pm\x1b\\y\x1bXsos\x1b\\z");
    assert_eq!(text(&s, 0), "xyz");
}

// ---------------------------------------------------------------------
// Fuzz: malformed input never panics
// ---------------------------------------------------------------------

/// Simple deterministic LCG so we need no `rand` dependency.
struct Lcg(u64);

impl Lcg {
    fn next_byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

#[test]
fn fuzz_random_bytes_never_panic() {
    let mut rng = Lcg(0x5EED_CAFE_F00D_1234);
    let mut parser = Parser::new();
    let mut screen = Screen::new(24, 80);
    let bytes: Vec<u8> = (0..50_000).map(|_| rng.next_byte()).collect();
    for chunk in bytes.chunks(7) {
        parser.feed(&mut screen, chunk);
    }
    // Cursor invariants survive the abuse.
    assert!(screen.cursor().row < 24);
    assert!(screen.cursor().col < 80);
}

#[test]
fn fuzz_random_bytes_interleaved_with_valid_sequences() {
    let mut rng = Lcg(0xDEAD_BEEF_0BAD_F00D);
    let mut parser = Parser::new();
    let mut screen = Screen::new(10, 40);
    let valid: &[&[u8]] = &[
        b"\x1b[1;31mhi\x1b[0m",
        b"\x1b]133;A\x07",
        b"\x1b[2J\x1b[H",
        "🚀é".as_bytes(),
        b"\x1b[?1049h",
        b"\x1b[?1049l",
        b"\x1bPjunk\x1b\\",
    ];
    for i in 0..2_000 {
        let noise: Vec<u8> = (0..(rng.next_byte() % 9))
            .map(|_| rng.next_byte())
            .collect();
        parser.feed(&mut screen, &noise);
        parser.feed(&mut screen, valid[i % valid.len()]);
    }
    assert!(screen.cursor().row < 10);
    assert!(screen.cursor().col < 40);
}

#[test]
fn fuzz_byte_at_a_time_matches_chunked() {
    // Deterministic noise must ALSO be chunk-safe, not just panic-free.
    let mut rng = Lcg(42);
    let bytes: Vec<u8> = (0..4_096).map(|_| rng.next_byte()).collect();
    run_dims(&bytes, 12, 40);
}
