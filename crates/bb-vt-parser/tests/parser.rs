//! Integration tests for `bb-vt-parser`.
//!
//! Every input goes through [`run_dims`], which feeds the byte stream
//! three ways — one-shot, byte-at-a-time, and in 3-byte chunks — and
//! asserts that all three produce identical screen state and events.
//! That makes chunk-boundary safety a property checked by *every* test,
//! not just the dedicated split tests.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use bb_terminal_model::{Color, CursorShape, Screen};
use bb_vt_parser::{Parser, VtEvent};

/// Feed `bytes` into a fresh `rows x cols` screen, asserting that
/// one-shot, byte-at-a-time, and 3-byte-chunk feeding are equivalent.
fn run_dims(bytes: &[u8], rows: usize, cols: usize) -> (Screen, Vec<VtEvent>) {
    // One-shot.
    let mut p1 = Parser::new();
    let mut s1 = Screen::new(rows, cols);
    let ev1 = p1.feed(&mut s1, bytes);

    // Byte at a time.
    let mut p2 = Parser::new();
    let mut s2 = Screen::new(rows, cols);
    let mut ev2 = Vec::new();
    for &b in bytes {
        ev2.extend(p2.feed(&mut s2, &[b]));
    }
    assert_eq!(s1, s2, "byte-at-a-time screen mismatch for {bytes:?}");
    assert_eq!(ev1, ev2, "byte-at-a-time event mismatch for {bytes:?}");

    // 3-byte chunks.
    let mut p3 = Parser::new();
    let mut s3 = Screen::new(rows, cols);
    let mut ev3 = Vec::new();
    for chunk in bytes.chunks(3) {
        ev3.extend(p3.feed(&mut s3, chunk));
    }
    assert_eq!(s1, s3, "chunked screen mismatch for {bytes:?}");
    assert_eq!(ev1, ev3, "chunked event mismatch for {bytes:?}");

    (s1, ev1)
}

/// [`run_dims`] on a default 24x80 screen.
fn run(bytes: &[u8]) -> (Screen, Vec<VtEvent>) {
    run_dims(bytes, 24, 80)
}

/// Text of grid row `row` (trailing blanks trimmed).
fn text(s: &Screen, row: usize) -> String {
    s.row_text(s.absolute_row(row)).unwrap_or_default()
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
    // The cleared screen was pushed to scrollback (rows still addressable).
    assert_eq!(s.scrollback().len(), 24);
    assert_eq!(s.row_text(0).as_deref(), Some("a"));
    assert_eq!(s.row_text(1).as_deref(), Some("b"));
}

#[test]
fn ed_3_treated_as_full_clear() {
    let (s, _) = run(b"hello\x1b[3J");
    assert_eq!(text(&s, 0), "");
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
    assert_eq!(text(&s, 0), ""); // alt starts cleared
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
    // 21 clears bold.
    let (s, _) = run(b"\x1b[1m\x1b[21m");
    assert!(!s.pen().bold);
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
    assert!(p.feed(&mut s, b"\x1b]133;D;").is_empty());
    assert!(p.feed(&mut s, b"1").is_empty());
    let ev = p.feed(&mut s, b"2\x07");
    assert_eq!(
        ev,
        vec![VtEvent::CommandFinished {
            exit_code: Some(12)
        }]
    );
    // ST split between the ESC and the backslash.
    let mut p = Parser::new();
    assert!(p.feed(&mut s, b"\x1b]0;T\x1b").is_empty());
    assert_eq!(p.feed(&mut s, b"\\"), vec![VtEvent::Title("T".into())]);
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
// DCS / SOS / PM / APC
// ---------------------------------------------------------------------

#[test]
fn dcs_content_discarded() {
    let (s, ev) = run(b"\x1bPq#0;2;0;0;0#0~~\x1b\\after");
    assert!(ev.is_empty());
    assert_eq!(text(&s, 0), "after");
}

#[test]
fn dcs_ignores_embedded_controls_and_esc() {
    // BEL and a lone ESC inside DCS do not terminate it; ESC ESC \ does.
    let (s, ev) = run(b"\x1bPdata\x07more\x1b\x1b\\x");
    assert!(ev.is_empty());
    assert_eq!(text(&s, 0), "x");
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
