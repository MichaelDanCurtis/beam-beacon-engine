//! Behavioral tests for [`bb_terminal_model::Screen`] via the public API.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use bb_terminal_model::{
    Cell, CellAttrs, CellWidth, Color, CursorShape, EraseMode, Screen, Viewport,
};

/// Write `text` starting at `(row, 0)`.
fn put(screen: &mut Screen, row: usize, text: &str) {
    screen.cursor_goto(row, 0);
    for ch in text.chars() {
        screen.print(ch);
    }
}

/// The text of grid row `row` (trailing blanks trimmed).
fn grid_text(screen: &Screen, row: usize) -> String {
    screen.row_text(screen.absolute_row(row)).unwrap()
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn new_screen_has_expected_defaults() {
    let s = Screen::new(5, 10);
    assert_eq!(s.rows(), 5);
    assert_eq!(s.cols(), 10);
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
    assert!(s.cursor().visible);
    assert_eq!(*s.pen(), CellAttrs::default());
    assert!(!s.is_alt_screen());
    assert!(!s.pending_wrap());
    assert!(s.autowrap());
    assert!(!s.origin_mode());
    assert!(!s.insert_mode());
    assert_eq!(s.scroll_region(), (0, 4));
    assert_eq!(s.scrollback().max_lines(), Screen::DEFAULT_SCROLLBACK_LINES);
    assert_eq!(s.total_rows(), 5);
}

#[test]
fn zero_dimensions_are_clamped_to_one() {
    let mut s = Screen::new(0, 0);
    assert_eq!((s.rows(), s.cols()), (1, 1));
    s.print('a'); // must not panic
    assert_eq!(s.grid().cell(0, 0).unwrap().ch, 'a');
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

#[test]
fn print_writes_char_with_pen_and_advances() {
    let mut s = Screen::new(2, 10);
    s.pen_mut().bold = true;
    s.pen_mut().fg = Color::Indexed(2);
    s.print('A');
    let cell = s.grid().cell(0, 0).unwrap();
    assert_eq!(cell.ch, 'A');
    assert!(cell.attrs.bold);
    assert_eq!(cell.attrs.fg, Color::Indexed(2));
    assert_eq!(cell.width, CellWidth::Single);
    assert_eq!(s.cursor().col, 1);
}

#[test]
fn print_ignores_control_and_zero_width_chars() {
    let mut s = Screen::new(2, 10);
    s.print('\u{1b}'); // ESC: width None
    s.print('\u{0301}'); // combining acute: width 0
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
    assert_eq!(s.grid().cell(0, 0).unwrap().ch, ' ');
}

#[test]
fn print_in_last_column_sets_pending_wrap_and_holds_cursor() {
    let mut s = Screen::new(2, 4);
    for ch in "abcd".chars() {
        s.print(ch);
    }
    assert_eq!((s.cursor().row, s.cursor().col), (0, 3));
    assert!(s.pending_wrap());
    assert_eq!(grid_text(&s, 0), "abcd");
}

#[test]
fn deferred_wrap_happens_on_next_printable() {
    let mut s = Screen::new(2, 4);
    for ch in "abcde".chars() {
        s.print(ch);
    }
    assert_eq!(grid_text(&s, 0), "abcd");
    assert_eq!(grid_text(&s, 1), "e");
    assert_eq!((s.cursor().row, s.cursor().col), (1, 1));
}

#[test]
fn carriage_return_cancels_pending_wrap() {
    let mut s = Screen::new(2, 4);
    for ch in "abcd".chars() {
        s.print(ch);
    }
    s.carriage_return();
    assert!(!s.pending_wrap());
    s.print('X');
    assert_eq!(grid_text(&s, 0), "Xbcd");
    assert_eq!(s.cursor().row, 0);
}

#[test]
fn cursor_motion_cancels_pending_wrap() {
    let mut s = Screen::new(2, 4);
    for ch in "abcd".chars() {
        s.print(ch);
    }
    s.cursor_left(1);
    assert!(!s.pending_wrap());
    s.print('X');
    assert_eq!(grid_text(&s, 0), "abXd");
}

#[test]
fn autowrap_off_overprints_last_column() {
    let mut s = Screen::new(2, 4);
    s.set_autowrap(false);
    for ch in "abcdef".chars() {
        s.print(ch);
    }
    assert_eq!(grid_text(&s, 0), "abcf");
    assert_eq!((s.cursor().row, s.cursor().col), (0, 3));
}

#[test]
fn wrap_at_bottom_scrolls_and_pushes_scrollback() {
    let mut s = Screen::with_scrollback(2, 2, 10);
    for ch in "abcde".chars() {
        s.print(ch);
    }
    // "ab" scrolled into scrollback; grid is now ["cd", "e "].
    assert_eq!(s.scrollback().len(), 1);
    assert_eq!(s.row_text(0).unwrap(), "ab");
    assert_eq!(grid_text(&s, 0), "cd");
    assert_eq!(grid_text(&s, 1), "e");
    assert_eq!(s.cursor_absolute_row(), 2);
}

#[test]
fn insert_mode_shifts_row_right() {
    let mut s = Screen::new(1, 5);
    for ch in "abc".chars() {
        s.print(ch);
    }
    s.carriage_return();
    s.set_insert_mode(true);
    s.print('X');
    assert_eq!(grid_text(&s, 0), "Xabc");
    assert_eq!(s.cursor().col, 1);
}

#[test]
fn insert_mode_drops_cells_pushed_past_the_end() {
    let mut s = Screen::new(1, 3);
    for ch in "abc".chars() {
        s.print(ch);
    }
    s.carriage_return();
    s.set_insert_mode(true);
    s.print('X');
    assert_eq!(grid_text(&s, 0), "Xab");
}

// ---------------------------------------------------------------------------
// Wide characters
// ---------------------------------------------------------------------------

#[test]
fn wide_char_occupies_lead_and_trail_cells() {
    let mut s = Screen::new(2, 4);
    s.print('你');
    let lead = s.grid().cell(0, 0).unwrap();
    let trail = s.grid().cell(0, 1).unwrap();
    assert_eq!(lead.ch, '你');
    assert_eq!(lead.width, CellWidth::WideLead);
    assert_eq!(trail.ch, ' ');
    assert_eq!(trail.width, CellWidth::WideTrail);
    assert_eq!(s.cursor().col, 2);
}

#[test]
fn wide_char_filling_the_line_sets_pending_wrap() {
    let mut s = Screen::new(2, 4);
    s.print('你');
    s.print('好');
    assert_eq!((s.cursor().row, s.cursor().col), (0, 3));
    assert!(s.pending_wrap());
    assert_eq!(grid_text(&s, 0), "你好");
}

#[test]
fn wide_char_at_last_column_wraps_and_leaves_spacer() {
    let mut s = Screen::new(2, 3);
    s.print('a');
    s.print('b'); // cursor now at col 2 (last)
    s.print('你'); // doesn't fit: spacer at (0,2), wraps
    let spacer = s.grid().cell(0, 2).unwrap();
    assert_eq!(spacer.ch, ' ');
    assert_eq!(spacer.width, CellWidth::Single);
    assert_eq!(s.grid().cell(1, 0).unwrap().ch, '你');
    assert_eq!(s.grid().cell(1, 0).unwrap().width, CellWidth::WideLead);
    assert_eq!(s.grid().cell(1, 1).unwrap().width, CellWidth::WideTrail);
    assert_eq!((s.cursor().row, s.cursor().col), (1, 2));
}

#[test]
fn wide_char_at_last_column_without_autowrap_prints_in_last_two_columns() {
    let mut s = Screen::new(2, 4);
    s.set_autowrap(false);
    s.cursor_goto(0, 3);
    s.print('你');
    assert_eq!(s.grid().cell(0, 2).unwrap().ch, '你');
    assert_eq!(s.grid().cell(0, 2).unwrap().width, CellWidth::WideLead);
    assert_eq!(s.grid().cell(0, 3).unwrap().width, CellWidth::WideTrail);
    assert_eq!(s.cursor().row, 0);
}

#[test]
fn overwriting_wide_trail_blanks_the_lead() {
    let mut s = Screen::new(1, 4);
    s.print('你');
    s.cursor_goto(0, 1);
    s.print('x');
    assert_eq!(s.grid().cell(0, 0).unwrap().ch, ' ');
    assert_eq!(s.grid().cell(0, 0).unwrap().width, CellWidth::Single);
    assert_eq!(s.grid().cell(0, 1).unwrap().ch, 'x');
}

#[test]
fn overwriting_wide_lead_blanks_the_trail() {
    let mut s = Screen::new(1, 4);
    s.print('你');
    s.cursor_goto(0, 0);
    s.print('x');
    assert_eq!(s.grid().cell(0, 0).unwrap().ch, 'x');
    assert_eq!(s.grid().cell(0, 1).unwrap().ch, ' ');
    assert_eq!(s.grid().cell(0, 1).unwrap().width, CellWidth::Single);
}

#[test]
fn wide_char_on_one_column_screen_is_ignored() {
    let mut s = Screen::new(1, 1);
    s.print('你');
    assert_eq!(s.grid().cell(0, 0).unwrap().ch, ' ');
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
}

#[test]
fn delete_chars_on_wide_trail_cleans_the_pair() {
    let mut s = Screen::new(1, 3);
    s.print('你');
    s.print('x');
    s.cursor_goto(0, 1);
    s.delete_chars(1);
    assert_eq!(s.grid().cell(0, 0).unwrap().ch, ' ');
    assert_eq!(s.grid().cell(0, 0).unwrap().width, CellWidth::Single);
    assert_eq!(s.grid().cell(0, 1).unwrap().ch, 'x');
}

// ---------------------------------------------------------------------------
// Control operations
// ---------------------------------------------------------------------------

#[test]
fn linefeed_moves_down_within_screen() {
    let mut s = Screen::new(3, 4);
    s.cursor_goto(0, 2);
    s.linefeed();
    assert_eq!((s.cursor().row, s.cursor().col), (1, 2));
}

#[test]
fn linefeed_at_bottom_scrolls_and_pushes_scrollback() {
    let mut s = Screen::with_scrollback(2, 3, 10);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bb");
    s.linefeed();
    assert_eq!(s.scrollback().len(), 1);
    assert_eq!(s.row_text(0).unwrap(), "aaa");
    assert_eq!(grid_text(&s, 0), "bb");
    assert_eq!(grid_text(&s, 1), "");
    assert_eq!(s.cursor().row, 1);
}

#[test]
fn linefeed_at_region_bottom_scrolls_region_only_without_scrollback() {
    let mut s = Screen::new(4, 3);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb");
    put(&mut s, 2, "ccc");
    put(&mut s, 3, "ddd");
    s.set_scroll_region(2, 3); // rows 1..=2
    s.cursor_goto(2, 0);
    s.linefeed();
    assert!(s.scrollback().is_empty());
    assert_eq!(grid_text(&s, 0), "aaa"); // outside region
    assert_eq!(grid_text(&s, 1), "ccc");
    assert_eq!(grid_text(&s, 2), "");
    assert_eq!(grid_text(&s, 3), "ddd"); // outside region
    assert_eq!(s.cursor().row, 2);
}

#[test]
fn linefeed_below_region_stops_at_last_row() {
    let mut s = Screen::new(4, 3);
    s.set_scroll_region(1, 2);
    s.cursor_goto(3, 0);
    s.linefeed();
    assert_eq!(s.cursor().row, 3);
}

#[test]
fn backspace_stops_at_column_zero() {
    let mut s = Screen::new(1, 4);
    s.print('a');
    s.backspace();
    assert_eq!(s.cursor().col, 0);
    s.backspace();
    assert_eq!(s.cursor().col, 0);
}

#[test]
fn tab_advances_to_default_stops_every_eight() {
    let mut s = Screen::new(1, 20);
    s.tab();
    assert_eq!(s.cursor().col, 8);
    s.tab();
    assert_eq!(s.cursor().col, 16);
    s.tab();
    assert_eq!(s.cursor().col, 19); // no more stops: last column
    s.tab();
    assert_eq!(s.cursor().col, 19);
}

#[test]
fn set_and_clear_tab_stops() {
    let mut s = Screen::new(1, 20);
    s.cursor_to_col(4);
    s.set_tab_stop();
    s.cursor_to_col(8);
    s.clear_tab_stop();
    s.cursor_to_col(0);
    s.tab();
    assert_eq!(s.cursor().col, 4);
    s.tab();
    assert_eq!(s.cursor().col, 16); // 8 was cleared
}

#[test]
fn clear_all_tab_stops_leaves_only_last_column() {
    let mut s = Screen::new(1, 20);
    s.clear_all_tab_stops();
    s.tab();
    assert_eq!(s.cursor().col, 19);
}

// ---------------------------------------------------------------------------
// Cursor movement
// ---------------------------------------------------------------------------

#[test]
fn cursor_moves_clamp_to_screen_bounds() {
    let mut s = Screen::new(3, 5);
    s.cursor_goto(1, 2);
    s.cursor_up(99);
    assert_eq!(s.cursor().row, 0);
    s.cursor_down(99);
    assert_eq!(s.cursor().row, 2);
    s.cursor_left(99);
    assert_eq!(s.cursor().col, 0);
    s.cursor_right(99);
    assert_eq!(s.cursor().col, 4);
    s.cursor_goto(99, 99);
    assert_eq!((s.cursor().row, s.cursor().col), (2, 4));
}

#[test]
fn cursor_up_down_respect_scroll_region_margins() {
    let mut s = Screen::new(5, 5);
    s.set_scroll_region(2, 4); // rows 1..=3
    s.cursor_goto(2, 0);
    s.cursor_up(99);
    assert_eq!(s.cursor().row, 1); // stops at top margin
    s.cursor_down(99);
    assert_eq!(s.cursor().row, 3); // stops at bottom margin
    s.cursor_goto(0, 0);
    s.cursor_up(1);
    assert_eq!(s.cursor().row, 0); // above region: clamps to row 0
    s.cursor_goto(4, 0);
    s.cursor_down(1);
    assert_eq!(s.cursor().row, 4); // below region: clamps to last row
}

#[test]
fn cursor_to_row_and_col_clamp() {
    let mut s = Screen::new(3, 5);
    s.cursor_to_row(99);
    assert_eq!(s.cursor().row, 2);
    s.cursor_to_col(99);
    assert_eq!(s.cursor().col, 4);
    s.cursor_to_col(1);
    assert_eq!(s.cursor().col, 1);
    s.cursor_to_row(0);
    assert_eq!(s.cursor().row, 0);
    assert_eq!(s.cursor().col, 1); // column preserved
}

#[test]
fn origin_mode_makes_rows_region_relative() {
    let mut s = Screen::new(4, 4);
    s.set_scroll_region(2, 3); // rows 1..=2
    s.set_origin_mode(true);
    assert_eq!((s.cursor().row, s.cursor().col), (1, 0)); // homed to region origin
    s.cursor_goto(0, 0);
    assert_eq!(s.cursor().row, 1);
    s.cursor_goto(9, 2);
    assert_eq!((s.cursor().row, s.cursor().col), (2, 2)); // clamped to region bottom
    s.cursor_to_row(1);
    assert_eq!(s.cursor().row, 2);
    s.set_origin_mode(false);
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0)); // homed to screen origin
    s.cursor_goto(3, 0);
    assert_eq!(s.cursor().row, 3); // absolute again
}

#[test]
fn save_and_restore_cursor_round_trips_position_pen_and_origin() {
    let mut s = Screen::new(4, 4);
    s.pen_mut().bold = true;
    s.cursor_goto(2, 3);
    s.save_cursor();
    s.pen_mut().bold = false;
    s.pen_mut().italic = true;
    s.cursor_goto(0, 0);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (2, 3));
    assert!(s.pen().bold);
    assert!(!s.pen().italic);
}

#[test]
fn restore_without_save_resets_to_defaults() {
    let mut s = Screen::new(4, 4);
    s.pen_mut().bold = true;
    s.cursor_goto(2, 2);
    s.set_scroll_region(2, 3);
    s.set_origin_mode(true);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
    assert_eq!(*s.pen(), CellAttrs::default());
    assert!(!s.origin_mode());
}

#[test]
fn primary_and_alt_screens_have_independent_save_slots() {
    let mut s = Screen::new(4, 4);
    s.cursor_goto(1, 1);
    s.save_cursor();
    s.enter_alt_screen();
    s.cursor_goto(2, 2);
    s.save_cursor();
    s.cursor_goto(0, 0);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (2, 2));
    s.exit_alt_screen();
    s.cursor_goto(0, 0);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (1, 1));
}

// ---------------------------------------------------------------------------
// Erase operations
// ---------------------------------------------------------------------------

#[test]
fn erase_in_line_to_end_to_start_and_all() {
    let mut s = Screen::new(1, 5);
    put(&mut s, 0, "abcde");
    s.cursor_goto(0, 2);
    s.erase_in_line(EraseMode::ToEnd);
    assert_eq!(grid_text(&s, 0), "ab");

    put(&mut s, 0, "abcde");
    s.cursor_goto(0, 2);
    s.erase_in_line(EraseMode::ToStart);
    assert_eq!(s.grid().cell(0, 2).unwrap().ch, ' '); // cursor cell included
    assert_eq!(grid_text(&s, 0), "   de");

    put(&mut s, 0, "abcde");
    s.erase_in_line(EraseMode::All);
    assert_eq!(grid_text(&s, 0), "");
}

#[test]
fn erase_in_display_to_end_and_to_start() {
    let mut s = Screen::new(3, 3);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb");
    put(&mut s, 2, "ccc");
    s.cursor_goto(1, 1);
    s.erase_in_display(EraseMode::ToEnd);
    assert_eq!(grid_text(&s, 0), "aaa");
    assert_eq!(grid_text(&s, 1), "b");
    assert_eq!(grid_text(&s, 2), "");

    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb");
    put(&mut s, 2, "ccc");
    s.cursor_goto(1, 1);
    s.erase_in_display(EraseMode::ToStart);
    assert_eq!(grid_text(&s, 0), "");
    assert_eq!(grid_text(&s, 1), "  b");
    assert_eq!(grid_text(&s, 2), "ccc");
    assert_eq!((s.cursor().row, s.cursor().col), (1, 1)); // cursor unmoved
}

#[test]
fn erase_all_on_primary_pushes_screen_to_scrollback() {
    let mut s = Screen::with_scrollback(2, 3, 10);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb");
    let abs_row0 = s.absolute_row(0);
    s.erase_in_display(EraseMode::All);
    assert_eq!(s.scrollback().len(), 2);
    assert_eq!(grid_text(&s, 0), "");
    assert_eq!(grid_text(&s, 1), "");
    // The old rows keep their absolute ids.
    assert_eq!(s.row_text(abs_row0).unwrap(), "aaa");
    assert_eq!(s.row_text(abs_row0 + 1).unwrap(), "bbb");
}

#[test]
fn erase_all_on_alt_screen_does_not_touch_scrollback() {
    let mut s = Screen::new(2, 3);
    s.enter_alt_screen();
    put(&mut s, 0, "xxx");
    s.erase_in_display(EraseMode::All);
    assert!(s.scrollback().is_empty());
    assert_eq!(grid_text(&s, 0), "");
}

#[test]
fn erase_uses_pen_background_color() {
    let mut s = Screen::new(1, 3);
    put(&mut s, 0, "abc");
    s.pen_mut().bg = Color::Indexed(3);
    s.pen_mut().bold = true; // must NOT carry into blanks
    s.cursor_goto(0, 0);
    s.erase_in_line(EraseMode::All);
    let cell = s.grid().cell(0, 1).unwrap();
    assert_eq!(cell.ch, ' ');
    assert_eq!(cell.attrs.bg, Color::Indexed(3));
    assert!(!cell.attrs.bold);
}

#[test]
fn erase_chars_blanks_in_place_without_shifting() {
    let mut s = Screen::new(1, 5);
    put(&mut s, 0, "abcde");
    s.cursor_goto(0, 1);
    s.erase_chars(2);
    assert_eq!(grid_text(&s, 0), "a  de");
    s.erase_chars(99); // clamps to end of row
    assert_eq!(grid_text(&s, 0), "a");
}

#[test]
fn insert_chars_shifts_right_and_blanks() {
    let mut s = Screen::new(1, 5);
    put(&mut s, 0, "abcde");
    s.cursor_goto(0, 1);
    s.insert_chars(2);
    assert_eq!(grid_text(&s, 0), "a  bc");
}

#[test]
fn delete_chars_shifts_left_and_blanks_the_tail() {
    let mut s = Screen::new(1, 5);
    put(&mut s, 0, "abcde");
    s.cursor_goto(0, 1);
    s.delete_chars(2);
    assert_eq!(grid_text(&s, 0), "ade");
    s.delete_chars(99); // clamps
    assert_eq!(grid_text(&s, 0), "a");
}

// ---------------------------------------------------------------------------
// Line insert/delete and region scrolling
// ---------------------------------------------------------------------------

fn four_row_screen() -> Screen {
    let mut s = Screen::new(4, 3);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb");
    put(&mut s, 2, "ccc");
    put(&mut s, 3, "ddd");
    s
}

#[test]
fn insert_lines_shifts_region_down() {
    let mut s = four_row_screen();
    s.cursor_goto(1, 2);
    s.insert_lines(1);
    assert_eq!(grid_text(&s, 0), "aaa");
    assert_eq!(grid_text(&s, 1), "");
    assert_eq!(grid_text(&s, 2), "bbb");
    assert_eq!(grid_text(&s, 3), "ccc"); // "ddd" pushed out
    assert_eq!(s.cursor().col, 2); // column unchanged
}

#[test]
fn delete_lines_shifts_region_up() {
    let mut s = four_row_screen();
    s.cursor_goto(1, 0);
    s.delete_lines(1);
    assert_eq!(grid_text(&s, 0), "aaa");
    assert_eq!(grid_text(&s, 1), "ccc");
    assert_eq!(grid_text(&s, 2), "ddd");
    assert_eq!(grid_text(&s, 3), "");
    assert!(s.scrollback().is_empty()); // DL never feeds scrollback
}

#[test]
fn insert_delete_lines_are_noops_outside_the_region() {
    let mut s = four_row_screen();
    s.set_scroll_region(2, 3); // rows 1..=2
    s.cursor_goto(3, 0); // below region
    s.insert_lines(1);
    s.delete_lines(1);
    assert_eq!(grid_text(&s, 0), "aaa");
    assert_eq!(grid_text(&s, 1), "bbb");
    assert_eq!(grid_text(&s, 2), "ccc");
    assert_eq!(grid_text(&s, 3), "ddd");
}

#[test]
fn insert_delete_lines_respect_region_bottom() {
    let mut s = four_row_screen();
    s.set_scroll_region(1, 3); // rows 0..=2
    s.cursor_goto(1, 0);
    s.insert_lines(9); // clamped to region
    assert_eq!(grid_text(&s, 0), "aaa");
    assert_eq!(grid_text(&s, 1), "");
    assert_eq!(grid_text(&s, 2), "");
    assert_eq!(grid_text(&s, 3), "ddd"); // outside region untouched
}

#[test]
fn scroll_up_full_screen_pushes_to_scrollback() {
    let mut s = four_row_screen();
    s.scroll_up(2);
    assert_eq!(s.scrollback().len(), 2);
    assert_eq!(s.row_text(0).unwrap(), "aaa");
    assert_eq!(s.row_text(1).unwrap(), "bbb");
    assert_eq!(grid_text(&s, 0), "ccc");
    assert_eq!(grid_text(&s, 1), "ddd");
    assert_eq!(grid_text(&s, 2), "");
    assert_eq!(grid_text(&s, 3), "");
}

#[test]
fn scroll_up_with_partial_region_does_not_push_scrollback() {
    let mut s = four_row_screen();
    s.set_scroll_region(2, 3); // rows 1..=2
    s.scroll_up(1);
    assert!(s.scrollback().is_empty());
    assert_eq!(grid_text(&s, 0), "aaa");
    assert_eq!(grid_text(&s, 1), "ccc");
    assert_eq!(grid_text(&s, 2), "");
    assert_eq!(grid_text(&s, 3), "ddd");
}

#[test]
fn scroll_down_shifts_region_down_without_scrollback() {
    let mut s = four_row_screen();
    s.scroll_down(1);
    assert!(s.scrollback().is_empty());
    assert_eq!(grid_text(&s, 0), "");
    assert_eq!(grid_text(&s, 1), "aaa");
    assert_eq!(grid_text(&s, 2), "bbb");
    assert_eq!(grid_text(&s, 3), "ccc");
}

// ---------------------------------------------------------------------------
// Scroll region setup
// ---------------------------------------------------------------------------

#[test]
fn set_scroll_region_clamps_and_homes_cursor() {
    let mut s = Screen::new(4, 4);
    s.cursor_goto(3, 3);
    s.set_scroll_region(2, 99);
    assert_eq!(s.scroll_region(), (1, 3));
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
}

#[test]
fn set_scroll_region_zero_means_default() {
    let mut s = Screen::new(4, 4);
    s.set_scroll_region(0, 0);
    assert_eq!(s.scroll_region(), (0, 3));
}

#[test]
fn set_scroll_region_ignores_degenerate_ranges() {
    let mut s = Screen::new(4, 4);
    s.set_scroll_region(2, 3);
    s.set_scroll_region(3, 3); // top == bottom: ignored
    s.set_scroll_region(3, 2); // inverted: ignored
    assert_eq!(s.scroll_region(), (1, 2));
}

#[test]
fn set_scroll_region_homes_to_region_origin_in_origin_mode() {
    let mut s = Screen::new(4, 4);
    s.set_origin_mode(true);
    s.set_scroll_region(2, 4);
    assert_eq!((s.cursor().row, s.cursor().col), (1, 0));
}

#[test]
fn reset_scroll_region_restores_full_screen() {
    let mut s = Screen::new(4, 4);
    s.set_scroll_region(2, 3);
    s.reset_scroll_region();
    assert_eq!(s.scroll_region(), (0, 3));
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
}

// ---------------------------------------------------------------------------
// Alternate screen
// ---------------------------------------------------------------------------

#[test]
fn alt_screen_is_isolated_from_primary() {
    let mut s = Screen::new(2, 3);
    put(&mut s, 0, "abc");
    s.enter_alt_screen();
    assert!(s.is_alt_screen());
    assert_eq!(grid_text(&s, 0), ""); // alt starts blank
    put(&mut s, 0, "xyz");
    s.exit_alt_screen();
    assert!(!s.is_alt_screen());
    assert_eq!(grid_text(&s, 0), "abc"); // primary intact
}

#[test]
fn alt_screen_scrolling_never_touches_scrollback() {
    let mut s = Screen::with_scrollback(2, 3, 10);
    s.enter_alt_screen();
    put(&mut s, 1, "xxx");
    s.cursor_goto(1, 0);
    s.linefeed(); // scrolls alt grid
    s.scroll_up(1);
    assert!(s.scrollback().is_empty());
    assert_eq!(s.scrollback().evicted(), 0);
}

#[test]
fn enter_alt_screen_twice_does_not_reclear() {
    let mut s = Screen::new(2, 3);
    s.enter_alt_screen();
    put(&mut s, 0, "abc");
    s.enter_alt_screen(); // no-op
    assert_eq!(grid_text(&s, 0), "abc");
    s.exit_alt_screen();
    s.exit_alt_screen(); // no-op
    assert!(!s.is_alt_screen());
}

#[test]
fn reentering_alt_screen_clears_previous_alt_contents() {
    let mut s = Screen::new(2, 3);
    s.enter_alt_screen();
    put(&mut s, 0, "abc");
    s.exit_alt_screen();
    s.enter_alt_screen();
    assert_eq!(grid_text(&s, 0), "");
}

// ---------------------------------------------------------------------------
// Scrollback eviction and absolute row ids
// ---------------------------------------------------------------------------

#[test]
fn absolute_row_ids_stay_stable_across_eviction() {
    let mut s = Screen::with_scrollback(1, 1, 2);
    for ch in "abcd".chars() {
        s.print(ch); // each print past the first scrolls one row out
    }
    // Rows "a".."c" scrolled out; capacity 2 keeps "b","c"; "a" evicted.
    assert_eq!(s.scrollback().evicted(), 1);
    assert_eq!(s.scrollback().len(), 2);
    assert!(s.line(0).is_none()); // "a" is gone forever
    assert_eq!(s.row_text(1).unwrap(), "b");
    assert_eq!(s.row_text(2).unwrap(), "c");
    assert_eq!(s.absolute_row(0), 3);
    assert_eq!(s.row_text(3).unwrap(), "d");
    assert_eq!(s.cursor_absolute_row(), 3);
    assert_eq!(s.total_rows(), 3); // 2 retained + 1 grid row
    assert!(s.line(4).is_none()); // past the end
}

#[test]
fn line_returns_scrollback_and_grid_rows() {
    let mut s = Screen::with_scrollback(2, 3, 10);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb");
    s.linefeed(); // pushes "aaa"
    let sb = s.line(0).unwrap();
    assert_eq!(sb[0].ch, 'a');
    let grid_row = s.line(s.absolute_row(0)).unwrap();
    assert_eq!(grid_row[0].ch, 'b');
    assert_eq!(sb.len(), 3);
}

#[test]
fn row_text_trims_trailing_blanks_and_skips_wide_trails() {
    let mut s = Screen::new(2, 6);
    put(&mut s, 0, "a你b");
    assert_eq!(s.row_text(s.absolute_row(0)).unwrap(), "a你b");
    assert_eq!(s.row_text(s.absolute_row(1)).unwrap(), "");
}

#[test]
fn row_text_keeps_interior_blanks() {
    let mut s = Screen::new(1, 8);
    put(&mut s, 0, "a b");
    assert_eq!(grid_text(&s, 0), "a b");
}

#[test]
fn viewport_rows_projects_across_scrollback_and_grid() {
    let mut s = Screen::with_scrollback(2, 2, 10);
    for ch in "abcdef".chars() {
        s.print(ch); // "ab" scrolls into scrollback; grid holds "cd", "ef"
    }
    assert_eq!(s.scrollback().len(), 1);
    let all = s.viewport_rows(Viewport { top: 0, height: 4 });
    assert_eq!(all.len(), 3); // only 3 rows exist in total
    assert_eq!(all[0][0].ch, 'a');
    assert_eq!(all[1][0].ch, 'c');
    assert_eq!(all[2][0].ch, 'e');
    let tail = s.viewport_rows(Viewport { top: 2, height: 10 });
    assert_eq!(tail.len(), 1); // only the last grid row exists
    assert_eq!(tail[0][0].ch, 'e');
}

#[test]
fn viewport_rows_omits_evicted_rows() {
    let mut s = Screen::with_scrollback(1, 1, 1);
    for ch in "abc".chars() {
        s.print(ch);
    }
    // "a" evicted, "b" retained, "c" on the grid.
    assert_eq!(s.scrollback().evicted(), 1);
    let rows = s.viewport_rows(Viewport { top: 0, height: 3 });
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0].ch, 'b');
    assert_eq!(rows[1][0].ch, 'c');
}

// ---------------------------------------------------------------------------
// Resize and reset
// ---------------------------------------------------------------------------

#[test]
fn resize_is_destructive_but_keeps_scrollback() {
    let mut s = Screen::with_scrollback(2, 3, 10);
    put(&mut s, 0, "aaa");
    s.erase_in_display(EraseMode::All); // pushes 2 rows to scrollback
    s.cursor_goto(1, 2);
    s.set_scroll_region(1, 2);
    s.resize(4, 8);
    assert_eq!((s.rows(), s.cols()), (4, 8));
    assert_eq!(s.scrollback().len(), 2);
    assert_eq!(s.row_text(0).unwrap(), "aaa");
    assert_eq!(s.scroll_region(), (0, 3));
    assert_eq!(grid_text(&s, 0), ""); // destructive
    assert!(!s.pending_wrap());
}

#[test]
fn resize_clamps_cursor_into_new_bounds() {
    let mut s = Screen::new(4, 8);
    s.cursor_goto(3, 7);
    s.resize(2, 3);
    assert_eq!((s.cursor().row, s.cursor().col), (1, 2));
}

#[test]
fn resize_resets_tab_stops_for_new_width() {
    let mut s = Screen::new(2, 4);
    s.resize(2, 20);
    s.tab();
    assert_eq!(s.cursor().col, 8);
}

#[test]
fn reset_restores_defaults_but_keeps_scrollback() {
    let mut s = Screen::with_scrollback(3, 4, 10);
    put(&mut s, 0, "aaaa");
    s.erase_in_display(EraseMode::All); // seed scrollback
    s.enter_alt_screen();
    put(&mut s, 1, "alt");
    s.pen_mut().bold = true;
    s.set_origin_mode(true);
    s.set_insert_mode(true);
    s.set_autowrap(false);
    s.set_scroll_region(2, 3);
    s.set_cursor_visible(false);
    s.set_cursor_shape(CursorShape::Bar);
    s.save_cursor();
    s.reset();
    assert!(!s.is_alt_screen());
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
    assert!(s.cursor().visible);
    assert_eq!(s.cursor().shape, CursorShape::Block);
    assert_eq!(*s.pen(), CellAttrs::default());
    assert!(s.autowrap());
    assert!(!s.origin_mode());
    assert!(!s.insert_mode());
    assert_eq!(s.scroll_region(), (0, 2));
    assert_eq!(grid_text(&s, 0), ""); // grids cleared
    assert_eq!(s.scrollback().len(), 3); // history survives RIS
    assert_eq!(s.row_text(0).unwrap(), "aaaa");
    // Saved cursor was discarded: restore falls back to defaults.
    s.cursor_goto(1, 1);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
    // Tab stops are back to defaults.
    s.tab();
    assert_eq!(s.cursor().col, 3); // 4 cols: no stop before last column
}

// ---------------------------------------------------------------------------
// Cursor appearance
// ---------------------------------------------------------------------------

#[test]
fn cursor_visibility_and_shape_are_settable() {
    let mut s = Screen::new(2, 2);
    s.set_cursor_visible(false);
    assert!(!s.cursor().visible);
    s.set_cursor_visible(true);
    assert!(s.cursor().visible);
    s.set_cursor_shape(CursorShape::Underline);
    assert_eq!(s.cursor().shape, CursorShape::Underline);
}

// ---------------------------------------------------------------------------
// Serde feature
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
#[test]
fn pure_data_types_implement_serde_traits() {
    fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}
    assert_serde::<Cell>();
    assert_serde::<CellAttrs>();
    assert_serde::<CellWidth>();
    assert_serde::<Color>();
    assert_serde::<bb_terminal_model::Cursor>();
    assert_serde::<CursorShape>();
    assert_serde::<bb_terminal_model::Grid>();
    assert_serde::<Viewport>();
    assert_serde::<bb_terminal_model::Scrollback>();
    assert_serde::<EraseMode>();
}

#[cfg(not(feature = "serde"))]
#[test]
fn cell_type_is_exported() {
    // Keep the `Cell` import used in both feature configurations.
    let c = Cell::default();
    assert_eq!(c.ch, ' ');
}
