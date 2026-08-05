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

/// The text of primary grid row `row` by absolute id (trailing blanks
/// trimmed). This is the *scrollback timeline* view.
fn grid_text(screen: &Screen, row: usize) -> String {
    screen.row_text(screen.absolute_row(row)).unwrap()
}

/// The text of row `row` of the *active* grid — what a renderer would
/// draw. Differs from [`grid_text`] while the alternate screen is up.
fn active_text(screen: &Screen, row: usize) -> String {
    let cells = screen.grid().row(row).unwrap();
    let end = cells
        .iter()
        .rposition(|c| !(c.width == CellWidth::Single && c.ch == ' '))
        .map_or(0, |i| i + 1);
    cells[..end]
        .iter()
        .filter(|c| c.width != CellWidth::WideTrail)
        .map(|c| c.ch)
        .collect()
}

/// A row's width tags as a compact string, e.g. `"S S L T S"`.
fn widths(cells: &[Cell]) -> String {
    cells
        .iter()
        .map(|c| match c.width {
            CellWidth::Single => "S",
            CellWidth::WideLead => "L",
            CellWidth::WideTrail => "T",
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The width tags of active-grid row `row`.
fn row_widths(screen: &Screen, row: usize) -> String {
    widths(screen.grid().row(row).unwrap())
}

/// Panic if `cells` violates the wide-pair invariant: every `WideLead` is
/// immediately followed by a `WideTrail` and every `WideTrail` is
/// immediately preceded by a `WideLead` — row edges included, so a lead in
/// the last column and a trail in the first are both orphans.
fn assert_row_pairs_intact(cells: &[Cell], what: &str) {
    for (i, cell) in cells.iter().enumerate() {
        match cell.width {
            CellWidth::WideLead => assert_eq!(
                cells.get(i + 1).map(|c| c.width),
                Some(CellWidth::WideTrail),
                "{what}: orphaned WideLead at col {i} in [{}]",
                widths(cells)
            ),
            CellWidth::WideTrail => assert_eq!(
                i.checked_sub(1).and_then(|j| cells.get(j)).map(|c| c.width),
                Some(CellWidth::WideLead),
                "{what}: orphaned WideTrail at col {i} in [{}]",
                widths(cells)
            ),
            CellWidth::Single => {}
        }
    }
}

/// Assert the wide-pair invariant over everything a `Screen` exposes: both
/// grids and every retained scrollback row. An orphan that reaches
/// scrollback is frozen there forever, so history is checked too.
fn assert_pairs_intact(screen: &Screen, what: &str) {
    for r in 0..screen.rows() {
        assert_row_pairs_intact(
            screen.grid().row(r).unwrap(),
            &format!("{what}: active row {r}"),
        );
        assert_row_pairs_intact(
            screen.primary_grid().row(r).unwrap(),
            &format!("{what}: primary row {r}"),
        );
    }
    for i in 0..screen.scrollback().len() {
        assert_row_pairs_intact(
            screen.scrollback().row(i).unwrap(),
            &format!("{what}: scrollback row {i}"),
        );
    }
}

/// A 1×8 screen holding `ab漢cd`: `a b [漢 lead] [trail] c d _ _`. The
/// wide pair sits at columns 2–3, so every editing operation can be aimed
/// at its left half, its right half, or straight through it.
fn wide_pair_row() -> Screen {
    let mut s = Screen::new(1, 8);
    put(&mut s, 0, "ab漢cd");
    assert_eq!(row_widths(&s, 0), "S S L T S S S S");
    s
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
    s.pen_mut().set_bold(true);
    s.pen_mut().fg = Color::Indexed(2);
    s.print('A');
    let cell = s.grid().cell(0, 0).unwrap();
    assert_eq!(cell.ch, 'A');
    assert!(cell.attrs.bold());
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

// ---------------------------------------------------------------------------
// Wide characters: no editing path may strand half a pair
//
// Each case below is a real escape sequence applied to `ab漢cd`. Before the
// fix every one of them left a `WideLead` with no `WideTrail` to its right
// or a `WideTrail` with no `WideLead` to its left — and once such a row
// scrolls into scrollback the corruption is frozen there forever.
// ---------------------------------------------------------------------------

#[test]
fn dch_through_a_wide_pair_leaves_no_orphan_trail() {
    // `\x1b[H\x1b[3P`: the trail cell gets shifted into column 0, where it
    // has no lead to its left.
    let mut s = wide_pair_row();
    s.cursor_goto(0, 0);
    s.delete_chars(3);
    assert_eq!(row_widths(&s, 0), "S S S S S S S S");
    assert_eq!(grid_text(&s, 0), " cd");
    assert_pairs_intact(&s, "DCH 3");
}

#[test]
fn ich_pushing_a_wide_lead_off_the_row_end_leaves_no_orphan() {
    // `\x1b[H\x1b[2@` on `ab漢`: the shift pushes the trail past the last
    // column, stranding the lead there.
    let mut s = Screen::new(1, 5);
    put(&mut s, 0, "ab漢");
    assert_eq!(row_widths(&s, 0), "S S L T S");
    s.cursor_goto(0, 0);
    s.insert_chars(2);
    assert_eq!(row_widths(&s, 0), "S S S S S");
    assert_eq!(grid_text(&s, 0), "  ab");
    assert_pairs_intact(&s, "ICH 2");
}

#[test]
fn ech_over_a_wide_lead_leaves_no_orphan_trail() {
    // `\x1b[H\x1b[3X`: the span ends between the lead and its trail.
    let mut s = wide_pair_row();
    s.cursor_goto(0, 0);
    s.erase_chars(3);
    assert_eq!(row_widths(&s, 0), "S S S S S S S S");
    assert_eq!(grid_text(&s, 0), "    cd");
    assert_pairs_intact(&s, "ECH 3");
}

#[test]
fn ech_from_a_wide_trail_leaves_no_orphan_lead() {
    // `\x1b[4G\x1b[2X`: the span starts on the trail, stranding the lead.
    let mut s = wide_pair_row();
    s.cursor_goto(0, 3);
    s.erase_chars(2);
    assert_eq!(row_widths(&s, 0), "S S S S S S S S");
    assert_eq!(grid_text(&s, 0), "ab   d");
    assert_pairs_intact(&s, "ECH 2 from trail");
}

#[test]
fn el_to_end_from_a_wide_trail_leaves_no_orphan_lead() {
    // `\x1b[4G\x1b[0K`.
    let mut s = wide_pair_row();
    s.cursor_goto(0, 3);
    s.erase_in_line(EraseMode::ToEnd);
    assert_eq!(row_widths(&s, 0), "S S S S S S S S");
    assert_eq!(grid_text(&s, 0), "ab");
    assert_pairs_intact(&s, "EL ToEnd from trail");
}

#[test]
fn el_to_start_over_a_wide_lead_leaves_no_orphan_trail() {
    // `\x1b[3G\x1b[1K`.
    let mut s = wide_pair_row();
    s.cursor_goto(0, 2);
    s.erase_in_line(EraseMode::ToStart);
    assert_eq!(row_widths(&s, 0), "S S S S S S S S");
    assert_eq!(grid_text(&s, 0), "    cd");
    assert_pairs_intact(&s, "EL ToStart over lead");
}

#[test]
fn ed_to_end_from_a_wide_trail_leaves_no_orphan_lead() {
    // ED delegates the cursor row to EL, so it inherits the same cut.
    let mut s = Screen::new(2, 8);
    put(&mut s, 0, "ab漢cd");
    put(&mut s, 1, "zz");
    s.cursor_goto(0, 3);
    s.erase_in_display(EraseMode::ToEnd);
    assert_eq!(row_widths(&s, 0), "S S S S S S S S");
    assert_eq!(grid_text(&s, 0), "ab");
    assert_eq!(grid_text(&s, 1), "");
    assert_pairs_intact(&s, "ED ToEnd from trail");
}

#[test]
fn ed_to_start_over_a_wide_lead_leaves_no_orphan_trail() {
    let mut s = Screen::new(2, 8);
    put(&mut s, 0, "zz");
    put(&mut s, 1, "ab漢cd");
    s.cursor_goto(1, 2);
    s.erase_in_display(EraseMode::ToStart);
    assert_eq!(row_widths(&s, 1), "S S S S S S S S");
    assert_eq!(grid_text(&s, 0), "");
    assert_eq!(grid_text(&s, 1), "    cd");
    assert_pairs_intact(&s, "ED ToStart over lead");
}

#[test]
fn wrap_spacer_cleans_the_pair_it_overwrites() {
    // A wide char that does not fit drops a spacer in the last column
    // before wrapping. If that column was a trail, its lead is stranded.
    let mut s = Screen::new(2, 3);
    s.print('a');
    s.print('你'); // occupies columns 1-2
    assert_eq!(row_widths(&s, 0), "S L T");
    s.cursor_goto(0, 2); // park on the trail, clearing pending wrap
    s.print('好'); // needs two columns, only one left: spacer + wrap
    assert_eq!(row_widths(&s, 0), "S S S");
    assert_eq!(grid_text(&s, 0), "a");
    assert_eq!(grid_text(&s, 1), "好");
    assert_pairs_intact(&s, "wrap spacer");
}

#[test]
fn insert_mode_print_over_a_wide_pair_leaves_no_orphan() {
    let mut s = Screen::new(1, 6);
    put(&mut s, 0, "a漢b");
    s.cursor_goto(0, 1); // on the lead
    s.set_insert_mode(true);
    s.print('x');
    assert_pairs_intact(&s, "IRM print over lead");
    s.cursor_goto(0, 2);
    s.print('你'); // wide insert, pushes the row end off
    assert_pairs_intact(&s, "IRM wide print");
}

#[test]
fn wide_pair_invariant_survives_a_randomized_op_storm() {
    // Deterministic xorshift: no dev-dependency, same sequence every run.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut s = Screen::with_scrollback(6, 9, 40);
    // A mix of narrow, wide, and combining characters.
    let alphabet = ['a', 'Z', '漢', '你', '好', ' ', '\u{0301}', 'e'];

    for step in 0..20_000 {
        let rows = s.rows();
        let cols = s.cols();
        let row = rng.below(rows);
        let col = rng.below(cols);
        let n = rng.below(cols + 2);
        let mode = match rng.below(3) {
            0 => EraseMode::ToEnd,
            1 => EraseMode::ToStart,
            _ => EraseMode::All,
        };
        let op = rng.below(22);
        match op {
            0 => {
                s.cursor_goto(row, col);
                s.print(alphabet[rng.below(alphabet.len())]);
            }
            1 => s.print(alphabet[rng.below(alphabet.len())]),
            2 => {
                s.cursor_goto(row, col);
                s.erase_chars(n);
            }
            3 => {
                s.cursor_goto(row, col);
                s.insert_chars(n);
            }
            4 => {
                s.cursor_goto(row, col);
                s.delete_chars(n);
            }
            5 => {
                s.cursor_goto(row, col);
                s.erase_in_line(mode);
            }
            6 => {
                s.cursor_goto(row, col);
                s.erase_in_display(mode);
            }
            7 => {
                s.cursor_goto(row, col);
                s.insert_lines(n);
            }
            8 => {
                s.cursor_goto(row, col);
                s.delete_lines(n);
            }
            9 => s.scroll_up(n),
            10 => s.scroll_down(n),
            11 => s.linefeed(),
            12 => s.carriage_return(),
            13 => s.tab(),
            14 => s.back_tab(),
            15 => s.set_insert_mode(rng.below(2) == 0),
            16 => s.set_autowrap(rng.below(2) == 0),
            17 => {
                if rng.below(2) == 0 {
                    s.enter_alt_screen();
                } else {
                    s.exit_alt_screen();
                }
            }
            18 => s.resize(1 + rng.below(7), 1 + rng.below(10)),
            19 => s.soft_reset(),
            20 => s.erase_scrollback(),
            _ => {
                if rng.below(2) == 0 {
                    s.save_cursor();
                } else {
                    s.restore_cursor();
                }
            }
        }
        assert_pairs_intact(&s, &format!("step {step} (op {op})"));
        // Cursors must stay addressable no matter what the storm did.
        assert!(s.cursor().row < s.rows());
        assert!(s.cursor().col < s.cols());
    }
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
    s.pen_mut().set_bold(true);
    s.cursor_goto(2, 3);
    s.save_cursor();
    s.pen_mut().set_bold(false);
    s.pen_mut().set_italic(true);
    s.cursor_goto(0, 0);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (2, 3));
    assert!(s.pen().bold());
    assert!(!s.pen().italic());
}

#[test]
fn restore_without_save_resets_to_defaults() {
    let mut s = Screen::new(4, 4);
    s.pen_mut().set_bold(true);
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
fn erase_all_pushes_only_the_rows_that_hold_content() {
    // A 24-row screen with two lines of output costs two scrollback rows,
    // not twenty-four.
    let mut s = Screen::with_scrollback(24, 20, 10_000);
    put(&mut s, 0, "$ echo hi");
    put(&mut s, 1, "hi");
    s.cursor_goto(0, 0);
    s.erase_in_display(EraseMode::All);
    assert_eq!(s.scrollback().len(), 2);
    assert_eq!(s.row_text(0).as_deref(), Some("$ echo hi"));
    assert_eq!(s.row_text(1).as_deref(), Some("hi"));
}

#[test]
fn erase_all_on_a_blank_screen_pushes_nothing() {
    let mut s = Screen::with_scrollback(24, 20, 10_000);
    s.erase_in_display(EraseMode::All);
    s.erase_in_display(EraseMode::All);
    assert!(s.scrollback().is_empty());
    assert_eq!(s.scrollback().evicted(), 0);
}

#[test]
fn erase_all_keeps_a_row_painted_with_a_background_color() {
    // "Blank" means every cell is exactly the default. A screen painted
    // with a background is content and must survive as history.
    let mut s = Screen::with_scrollback(3, 4, 100);
    s.pen_mut().bg = Color::Indexed(4);
    s.cursor_goto(1, 0);
    s.erase_in_line(EraseMode::All); // paints row 1, leaves 0 and 2 default
    s.erase_in_display(EraseMode::All);
    assert_eq!(s.scrollback().len(), 2); // rows 0..=1; row 2 is still blank
    assert_eq!(
        s.scrollback().row(1).unwrap()[0].attrs.bg,
        Color::Indexed(4)
    );
}

#[test]
fn repeated_clear_costs_content_not_screenfuls() {
    // The regression that made this matter: `clear` on a 24-row screen
    // used to spend 24 scrollback rows every single time, so a few hundred
    // invocations evicted every real line of history behind a wall of
    // blanks. Cost must be O(content), not O(clears × rows).
    let rows = 24;
    let clears = 300;
    let mut s = Screen::with_scrollback(rows, 40, 10_000);
    for i in 0..clears {
        put(&mut s, 0, &format!("real output {i}"));
        s.cursor_goto(0, 0);
        s.erase_in_display(EraseMode::All); // `ESC[2J` as `clear` sends it
    }

    // Exactly one row per line of real output, and nothing evicted.
    assert_eq!(s.scrollback().len(), clears);
    assert_eq!(s.scrollback().evicted(), 0);
    for i in 0..clears {
        assert_eq!(
            s.row_text(i).as_deref(),
            Some(format!("real output {i}").as_str()),
            "output from clear #{i} is unrecoverable"
        );
    }
    // Not one blank row made it into history.
    for i in 0..s.scrollback().len() {
        let row = s.scrollback().row(i).unwrap();
        assert!(
            row.iter().any(|c| c.ch != ' '),
            "scrollback row {i} is a blank screen row"
        );
    }
}

#[test]
fn clear_with_e3_drops_history_and_keeps_ids_monotonic() {
    // `clear` with the E3 capability: `ESC[3J` then `ESC[H ESC[2J`.
    let mut s = Screen::with_scrollback(24, 40, 10_000);
    for i in 0..50 {
        put(&mut s, 0, &format!("line {i}"));
        s.erase_scrollback();
        s.cursor_goto(0, 0);
        s.erase_in_display(EraseMode::All);
    }
    // Each round wiped the previous round's history and filed one row of
    // its own, so history never grows without bound...
    assert_eq!(s.scrollback().len(), 1);
    assert_eq!(s.row_text(49).as_deref(), Some("line 49"));
    // ...and every id the wiped rows held is spent forever, never reused.
    assert_eq!(s.scrollback().evicted(), 49);
    assert!(s.line(0).is_none());
    assert!(s.line(48).is_none());

    // Wiping the last retained row moves no id: the grid keeps its own.
    let grid_zero = s.absolute_row(0);
    s.erase_scrollback();
    assert!(s.scrollback().is_empty());
    assert_eq!(s.scrollback().evicted(), 50);
    assert_eq!(s.absolute_row(0), grid_zero);
    assert!(s.line(49).is_none());
}

#[test]
fn erase_all_on_alt_screen_does_not_touch_scrollback() {
    let mut s = Screen::new(2, 3);
    s.enter_alt_screen();
    put(&mut s, 0, "xxx");
    s.erase_in_display(EraseMode::All);
    assert!(s.scrollback().is_empty());
    assert_eq!(active_text(&s, 0), "");
}

#[test]
fn erase_uses_pen_background_color() {
    let mut s = Screen::new(1, 3);
    put(&mut s, 0, "abc");
    s.pen_mut().bg = Color::Indexed(3);
    s.pen_mut().set_bold(true); // must NOT carry into blanks
    s.cursor_goto(0, 0);
    s.erase_in_line(EraseMode::All);
    let cell = s.grid().cell(0, 1).unwrap();
    assert_eq!(cell.ch, ' ');
    assert_eq!(cell.attrs.bg, Color::Indexed(3));
    assert!(!cell.attrs.bold());
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
    assert_eq!(active_text(&s, 0), ""); // alt starts blank
    put(&mut s, 0, "xyz");
    assert_eq!(active_text(&s, 0), "xyz");
    s.exit_alt_screen();
    assert!(!s.is_alt_screen());
    assert_eq!(active_text(&s, 0), "abc"); // primary intact
    assert_eq!(grid_text(&s, 0), "abc");
}

#[test]
fn absolute_rows_address_the_primary_grid_while_alt_is_active() {
    let mut s = Screen::with_scrollback(3, 12, 10);
    put(&mut s, 0, "$ vim");
    put(&mut s, 1, "opening");
    put(&mut s, 2, "reading");
    let abs = [s.absolute_row(0), s.absolute_row(1), s.absolute_row(2)];

    s.enter_alt_screen();
    for (row, text) in ["ALTUI one", "ALTUI two", "ALTUI three"].iter().enumerate() {
        put(&mut s, row, text);
    }

    // `grid()` is what to draw: the app's frame buffer.
    assert_eq!(active_text(&s, 0), "ALTUI one");
    assert_eq!(active_text(&s, 2), "ALTUI three");

    // Absolute rows are the scrollback timeline: still the primary screen.
    assert_eq!(s.row_text(abs[0]).as_deref(), Some("$ vim"));
    assert_eq!(s.row_text(abs[1]).as_deref(), Some("opening"));
    assert_eq!(s.row_text(abs[2]).as_deref(), Some("reading"));
    assert_eq!(s.line(abs[1]).unwrap()[0].ch, 'o');
    assert!(s
        .viewport_rows(Viewport {
            top: abs[0],
            height: 3,
        })
        .iter()
        .all(|row| row.iter().all(|c| c.ch != 'A')));

    // `primary_grid()` reaches the same content directly.
    assert_eq!(s.primary_grid().cell(0, 0).unwrap().ch, '$');
    assert_eq!(s.primary_grid().cell(1, 0).unwrap().ch, 'o');

    s.exit_alt_screen();
    assert_eq!(active_text(&s, 0), "$ vim"); // primary restored
    assert_eq!(s.row_text(abs[2]).as_deref(), Some("reading"));
}

#[test]
fn alt_screen_episode_leaves_the_timeline_untouched() {
    let mut s = Screen::with_scrollback(3, 8, 10);
    for text in ["out one", "out two", "out three", "out four"] {
        put(&mut s, 2, text);
        s.linefeed(); // scrolls the primary screen, feeding scrollback
    }
    let len_before = s.scrollback().len();
    let evicted_before = s.scrollback().evicted();
    let total_before = s.total_rows();
    let cursor_abs_before = s.cursor_absolute_row();
    let timeline_before: Vec<String> = (evicted_before..evicted_before + total_before)
        .filter_map(|abs| s.row_text(abs))
        .collect();

    s.enter_alt_screen();
    put(&mut s, 0, "ALTUI");
    s.cursor_goto(2, 0);
    for i in 0..20 {
        put(&mut s, 2, &format!("ALT {i}"));
        s.cursor_goto(2, 0);
        s.linefeed(); // scrolls the alt grid hard
    }
    s.scroll_up(2);

    // Nothing about the primary timeline moved.
    assert_eq!(s.scrollback().len(), len_before);
    assert_eq!(s.scrollback().evicted(), evicted_before);
    assert_eq!(s.total_rows(), total_before);
    assert_eq!(s.cursor_absolute_row(), cursor_abs_before);
    let timeline_during: Vec<String> = (evicted_before..evicted_before + total_before)
        .filter_map(|abs| s.row_text(abs))
        .collect();
    assert_eq!(timeline_during, timeline_before);
    for row in &timeline_during {
        assert!(!row.contains("ALT"), "alt content leaked into {row:?}");
    }

    s.exit_alt_screen();
    let timeline_after: Vec<String> = (evicted_before..evicted_before + total_before)
        .filter_map(|abs| s.row_text(abs))
        .collect();
    assert_eq!(timeline_after, timeline_before);
    assert_eq!(s.scrollback().evicted(), evicted_before);
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
    assert_eq!(active_text(&s, 0), "abc");
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
    assert_eq!(active_text(&s, 0), "");
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
fn resize_preserves_content_and_keeps_scrollback() {
    let mut s = Screen::with_scrollback(2, 3, 10);
    put(&mut s, 0, "aaa");
    s.erase_in_display(EraseMode::All); // pushes the one row that held text
    put(&mut s, 0, "new");
    s.cursor_goto(1, 2);
    s.set_scroll_region(1, 2);
    s.resize(4, 8);
    assert_eq!((s.rows(), s.cols()), (4, 8));
    assert_eq!(s.scrollback().len(), 1);
    assert_eq!(s.row_text(0).unwrap(), "aaa");
    assert_eq!(s.scroll_region(), (0, 3));
    assert_eq!(grid_text(&s, 0), "new"); // on-screen content survives
    assert!(!s.pending_wrap());
}

#[test]
fn resize_growing_keeps_cells_and_attrs_and_blanks_the_new_area() {
    let mut s = Screen::new(3, 6);
    s.pen_mut().set_bold(true);
    s.pen_mut().fg = Color::Indexed(4);
    put(&mut s, 0, "row0ab");
    *s.pen_mut() = CellAttrs::default();
    put(&mut s, 1, "row1cd");
    put(&mut s, 2, "row2ef");
    s.cursor_goto(2, 5);

    s.resize(5, 8);
    assert_eq!((s.rows(), s.cols()), (5, 8));
    assert_eq!(grid_text(&s, 0), "row0ab");
    assert_eq!(grid_text(&s, 1), "row1cd");
    assert_eq!(grid_text(&s, 2), "row2ef");
    assert_eq!(grid_text(&s, 3), ""); // new rows are blank
    assert_eq!(grid_text(&s, 4), "");
    let kept = s.grid().cell(0, 0).unwrap();
    assert!(kept.attrs.bold()); // attributes survive the resize
    assert_eq!(kept.attrs.fg, Color::Indexed(4));
    assert!(!s.grid().cell(1, 0).unwrap().attrs.bold());
    assert_eq!(*s.grid().cell(0, 6).unwrap(), Cell::default()); // new column
    assert_eq!((s.cursor().row, s.cursor().col), (2, 5)); // cursor unmoved
}

#[test]
fn resize_narrower_truncates_columns_without_reflow() {
    let mut s = Screen::new(2, 6);
    put(&mut s, 0, "abcdef");
    put(&mut s, 1, "ghijkl");
    s.cursor_goto(0, 0);
    s.resize(2, 3);
    assert_eq!(s.cols(), 3);
    assert_eq!(grid_text(&s, 0), "abc"); // truncated, never rewrapped
    assert_eq!(grid_text(&s, 1), "ghi");
}

#[test]
fn resize_shrinking_rows_sheds_the_top_into_scrollback() {
    let mut s = Screen::with_scrollback(4, 4, 10);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb");
    put(&mut s, 2, "ccc");
    put(&mut s, 3, "ddd");
    let abs: Vec<usize> = (0..4).map(|r| s.absolute_row(r)).collect();
    s.cursor_goto(3, 1); // cursor on the last row: the bottom must be kept

    s.resize(2, 4);
    assert_eq!(s.rows(), 2);
    assert_eq!(s.scrollback().len(), 2); // two top rows shed, not discarded
    assert_eq!(s.scrollback().evicted(), 0);
    assert_eq!(s.total_rows(), 4);
    // Every row is still addressable at the id it always had.
    assert_eq!(s.row_text(abs[0]).as_deref(), Some("aaa"));
    assert_eq!(s.row_text(abs[1]).as_deref(), Some("bbb"));
    assert_eq!(s.row_text(abs[2]).as_deref(), Some("ccc"));
    assert_eq!(s.row_text(abs[3]).as_deref(), Some("ddd"));
    // The bottom of the screen is what stayed on the grid.
    assert_eq!(active_text(&s, 0), "ccc");
    assert_eq!(active_text(&s, 1), "ddd");
    // The cursor followed its own row, so its absolute id is unchanged.
    assert_eq!((s.cursor().row, s.cursor().col), (1, 1));
    assert_eq!(s.cursor_absolute_row(), abs[3]);
}

#[test]
fn resize_shrinking_rows_never_drops_content_off_the_bottom() {
    // The cursor fitting in the new height is not enough: rows *below* it
    // still hold output, and a shrink must not silently delete them.
    let mut s = Screen::with_scrollback(4, 4, 10);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb");
    put(&mut s, 2, "ccc");
    put(&mut s, 3, "ddd");
    let abs: Vec<usize> = (0..4).map(|r| s.absolute_row(r)).collect();
    s.cursor_goto(1, 2);

    s.resize(2, 4);
    // Two rows had to leave the top to keep "ccc"/"ddd" on a 2-row screen.
    assert_eq!(s.scrollback().len(), 2);
    assert_eq!(s.scrollback().evicted(), 0);
    // Every row is still addressable at the id it always had.
    for (r, &a) in abs.iter().enumerate() {
        assert_eq!(
            s.row_text(a).as_deref(),
            Some(["aaa", "bbb", "ccc", "ddd"][r]),
            "absolute row {a}"
        );
    }
    assert_eq!(active_text(&s, 0), "ccc");
    assert_eq!(active_text(&s, 1), "ddd");
    // The cursor's own row went to scrollback, and a cursor cannot point
    // into history, so it lands on the topmost row that survived.
    assert_eq!((s.cursor().row, s.cursor().col), (0, 2));
    assert_eq!(s.cursor_absolute_row(), s.absolute_row(0));
}

#[test]
fn resize_shrinking_rows_discards_only_blank_trailing_rows() {
    // Mirror image: the rows that no longer fit are genuinely blank, so
    // they can just go — nothing reaches scrollback and no id moves.
    let mut s = Screen::with_scrollback(4, 4, 10);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb"); // rows 2 and 3 stay blank
    s.cursor_goto(1, 2);
    let cursor_abs = s.cursor_absolute_row();

    s.resize(2, 4);
    assert!(s.scrollback().is_empty()); // nothing had to leave the top
    assert_eq!(grid_text(&s, 0), "aaa");
    assert_eq!(grid_text(&s, 1), "bbb");
    assert_eq!((s.cursor().row, s.cursor().col), (1, 2));
    assert_eq!(s.cursor_absolute_row(), cursor_abs);
}

#[test]
fn resize_on_the_alt_screen_sheds_only_the_primarys_own_rows() {
    let mut s = Screen::with_scrollback(4, 8, 10);
    put(&mut s, 0, "p-zero");
    put(&mut s, 1, "p-one");
    put(&mut s, 2, "p-two");
    put(&mut s, 3, "p-three");
    s.enter_alt_screen();
    put(&mut s, 0, "a-zero");
    put(&mut s, 1, "a-one");
    put(&mut s, 2, "a-two");
    put(&mut s, 3, "a-three");
    s.cursor_goto(3, 0);

    s.resize(2, 8);
    // The alt grid kept its own bottom, and lost its top without a trace.
    assert_eq!(active_text(&s, 0), "a-two");
    assert_eq!(active_text(&s, 1), "a-three");
    // Scrollback holds the primary's own rows and nothing else: the two
    // that no longer fit, still under the ids they were born with.
    assert_eq!(s.scrollback().len(), 2);
    assert_eq!(s.scrollback().evicted(), 0);
    assert_eq!(s.row_text(0).as_deref(), Some("p-zero"));
    assert_eq!(s.row_text(1).as_deref(), Some("p-one"));
    assert_eq!(s.row_text(2).as_deref(), Some("p-two"));
    assert_eq!(s.row_text(3).as_deref(), Some("p-three"));
    for i in 0..s.scrollback().len() {
        let text: String = s
            .scrollback()
            .row(i)
            .unwrap()
            .iter()
            .map(|c| c.ch)
            .collect();
        assert!(!text.contains("a-"), "alt content leaked into scrollback");
    }
    // Each grid preserved its own content.
    s.exit_alt_screen();
    assert_eq!(active_text(&s, 0), "p-two");
    assert_eq!(active_text(&s, 1), "p-three");
}

#[test]
fn resize_on_the_alt_screen_never_feeds_scrollback() {
    // The name means what it says: when the primary's own content still
    // fits, an alt-screen resize touches neither scrollback nor absolute
    // ids — no matter where the running application parked its cursor.
    let mut s = Screen::with_scrollback(10, 10, 1000);
    put(&mut s, 0, "$ vim");
    s.cursor_goto(1, 0);
    let abs_zero = s.absolute_row(0);

    s.enter_alt_screen();
    s.cursor_goto(9, 0); // the app parks its cursor on the bottom row
    put(&mut s, 9, "~");
    s.resize(4, 10);

    assert!(s.scrollback().is_empty());
    assert_eq!(s.scrollback().evicted(), 0);
    assert_eq!(s.absolute_row(0), abs_zero); // the timeline did not move
    assert_eq!(s.row_text(abs_zero).as_deref(), Some("$ vim"));
    assert_eq!(s.total_rows(), 4);
    // The alt grid still sheds its own top to keep its own cursor on
    // screen; it simply has nowhere to file the rows it drops.
    assert_eq!(active_text(&s, 3), "~");
    s.exit_alt_screen();
    assert_eq!(active_text(&s, 0), "$ vim");
}

#[test]
fn resize_while_alt_is_up_keeps_the_shell_output_underneath() {
    // Shrinking the window while a full-screen app is running used to
    // delete the shell output behind it: the alt cursor sat at the top, so
    // nothing was shed, and the primary rows that no longer fit fell off
    // the bottom without ever reaching scrollback.
    let mut s = Screen::with_scrollback(10, 10, 1000);
    for r in 0..10 {
        put(&mut s, r, &format!("out {r}"));
    }
    let abs: Vec<usize> = (0..10).map(|r| s.absolute_row(r)).collect();

    s.enter_alt_screen(); // `\x1b[?1049h`
    s.cursor_goto(0, 0); // `\x1b[1;1H`
    s.resize(4, 10);

    // Every line of shell output is still addressable at its own id.
    for (r, &a) in abs.iter().enumerate() {
        assert_eq!(
            s.row_text(a).as_deref(),
            Some(format!("out {r}").as_str()),
            "absolute row {a}"
        );
    }
    assert_eq!(s.scrollback().len(), 6);
    assert_eq!(s.scrollback().evicted(), 0);

    s.exit_alt_screen();
    assert_eq!(active_text(&s, 0), "out 6");
    assert_eq!(active_text(&s, 3), "out 9");
}

#[test]
fn resize_clamps_cursor_into_new_bounds() {
    let mut s = Screen::new(4, 8);
    s.cursor_goto(3, 7);
    s.resize(2, 3);
    assert_eq!((s.cursor().row, s.cursor().col), (1, 2));
}

#[test]
fn resize_keeps_the_saved_cursor_sane() {
    let mut s = Screen::new(4, 8);
    s.cursor_goto(3, 7);
    s.save_cursor();
    s.resize(2, 3); // sheds 2 rows off the top
    assert_eq!((s.cursor().row, s.cursor().col), (1, 2));
    s.cursor_goto(0, 0);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (1, 2)); // saved cursor followed
}

#[test]
fn resize_heals_a_wide_pair_split_by_the_new_right_edge() {
    let mut s = Screen::new(1, 4);
    put(&mut s, 0, "a你b");
    s.cursor_goto(0, 0);
    s.resize(1, 3); // the pair still fits
    assert_eq!(grid_text(&s, 0), "a你");
    assert_eq!(s.grid().cell(0, 1).unwrap().width, CellWidth::WideLead);

    let mut s = Screen::new(1, 4);
    put(&mut s, 0, "a你b");
    s.cursor_goto(0, 0);
    s.resize(1, 2); // the trail falls off the edge
    assert_eq!(grid_text(&s, 0), "a");
    let orphan = s.grid().cell(0, 1).unwrap();
    assert_eq!(orphan.ch, ' ');
    assert_eq!(orphan.width, CellWidth::Single);
}

#[test]
fn resize_never_renumbers_a_row_and_keeps_the_cursor_anchored() {
    // Downstream block lists key off absolute row ids, so a resize must
    // never renumber one: rows shed off the top move into scrollback,
    // which grows by exactly the same amount.
    //
    // `cursor_absolute_row` follows the same rule, with one honest
    // exception. When the cursor sits *above* content that has to be
    // preserved, its own row is shed into scrollback — and a cursor cannot
    // point into history, so it lands on the topmost surviving row. That
    // is exactly `absolute_row(0)`, so the cursor's id is always
    // `max(before, absolute_row(0))`: unchanged whenever its row survived.
    for &(rows, cols) in &[
        (1_usize, 1_usize),
        (2, 4),
        (3, 40),
        (6, 10),
        (12, 80),
        (40, 200),
    ] {
        for &cursor_row in &[0_usize, 2, 5] {
            let mut s = Screen::with_scrollback(6, 20, 100);
            for r in 0..6 {
                put(&mut s, r, &format!("line {r}"));
            }
            s.cursor_goto(cursor_row, 3);
            let before = s.cursor_absolute_row();
            let abs: Vec<usize> = (0..6).map(|r| s.absolute_row(r)).collect();

            s.resize(rows, cols);

            let what = format!("resize({rows}, {cols}) with the cursor on row {cursor_row}");
            // No row was dropped: every id still resolves.
            for &a in &abs {
                assert!(
                    s.row_text(a).is_some(),
                    "{what}: absolute row {a} stopped resolving"
                );
            }
            assert_eq!(
                s.cursor_absolute_row(),
                before.max(s.absolute_row(0)),
                "{what}"
            );
            assert!(s.cursor().row < s.rows());
            assert!(s.cursor().col < s.cols());
        }
    }
}

#[test]
fn resize_keeps_block_style_absolute_row_lookups_working() {
    let mut s = Screen::with_scrollback(5, 16, 100);
    put(&mut s, 0, "$ tail log");
    put(&mut s, 1, "before A");
    put(&mut s, 2, "before B");
    s.cursor_goto(3, 0);
    // A block-list-style range of absolute row ids, captured before resize.
    let block: Vec<usize> = (0..3).map(|r| s.absolute_row(r)).collect();
    let evicted_before = s.scrollback().evicted();

    s.resize(3, 12); // shrink in both dimensions
    assert_eq!((s.rows(), s.cols()), (3, 12));
    assert_eq!(s.scrollback().evicted(), evicted_before);

    let text: Vec<String> = block.iter().filter_map(|&a| s.row_text(a)).collect();
    assert_eq!(text, ["$ tail log", "before A", "before B"]);
    // The viewport projection agrees with the per-row lookups.
    let rows = s.viewport_rows(Viewport {
        top: block[0],
        height: 3,
    });
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][0].ch, '$');
    assert_eq!(rows[2][0].ch, 'b');

    // New output lands at the next absolute id, so the block just grows.
    s.cursor_goto(2, 0);
    for ch in "after C".chars() {
        s.print(ch);
    }
    assert_eq!(s.absolute_row(2), block[2] + 1);
    assert_eq!(s.row_text(block[2] + 1).as_deref(), Some("after C"));
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
    s.pen_mut().set_bold(true);
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
    assert_eq!(s.scrollback().len(), 1); // history survives RIS
    assert_eq!(s.row_text(0).unwrap(), "aaaa");
    // Saved cursor was discarded: restore falls back to defaults.
    s.cursor_goto(1, 1);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
    // Tab stops are back to defaults.
    s.tab();
    assert_eq!(s.cursor().col, 3); // 4 cols: no stop before last column
}

#[test]
fn resize_growth_fills_the_new_area_with_the_pen_background() {
    // Background-color-erase: xterm paints the area exposed by a resize
    // with the current background, exactly like every other erase path.
    let mut s = Screen::new(2, 3);
    put(&mut s, 0, "ab");
    s.pen_mut().bg = Color::Indexed(5);
    s.pen_mut().set_bold(true); // must NOT carry into the new cells
    s.resize(4, 5);
    let new_col = s.grid().cell(0, 4).unwrap();
    assert_eq!(new_col.ch, ' ');
    assert_eq!(new_col.attrs.bg, Color::Indexed(5));
    assert!(!new_col.attrs.bold());
    let new_row = s.grid().cell(3, 0).unwrap();
    assert_eq!(new_row.attrs.bg, Color::Indexed(5));
    assert_eq!(grid_text(&s, 0), "ab"); // old content untouched
}

#[test]
fn entering_the_alt_screen_clears_to_the_pen_background() {
    let mut s = Screen::new(2, 3);
    s.pen_mut().bg = Color::Indexed(2);
    s.pen_mut().set_underline(true);
    s.enter_alt_screen();
    let cell = s.grid().cell(1, 1).unwrap();
    assert_eq!(cell.ch, ' ');
    assert_eq!(cell.attrs.bg, Color::Indexed(2));
    assert!(!cell.attrs.underline());
}

// ---------------------------------------------------------------------------
// Host-supplied geometry
// ---------------------------------------------------------------------------

#[test]
fn extreme_dimensions_are_clamped_rather_than_panicking() {
    // Dimensions arrive from the host (a window resize, TIOCSWINSZ). They
    // must cap, never overflow.
    let s = Screen::new(usize::MAX, 2);
    assert_eq!((s.rows(), s.cols()), (bb_terminal_model::Grid::MAX_ROWS, 2));
    let s = Screen::new(2, usize::MAX);
    assert_eq!((s.rows(), s.cols()), (2, bb_terminal_model::Grid::MAX_COLS));

    let mut s = Screen::new(4, 4);
    put(&mut s, 0, "keep");
    s.resize(usize::MAX, 2);
    assert_eq!((s.rows(), s.cols()), (bb_terminal_model::Grid::MAX_ROWS, 2));
    s.resize(3, usize::MAX);
    assert_eq!((s.rows(), s.cols()), (3, bb_terminal_model::Grid::MAX_COLS));
    // Still a usable screen afterwards.
    s.print('x');
    assert!(s.cursor().col < s.cols());
}

#[test]
fn an_absurd_viewport_neither_overflows_nor_spins() {
    // The height must bound the *request*, not the work: a `usize::MAX`
    // height once overflowed the allocator, and clamping the reservation
    // alone would have left an unbounded scan behind it.
    let mut s = Screen::with_scrollback(2, 2, 10);
    for ch in "abcdef".chars() {
        s.print(ch);
    }
    let all = s.viewport_rows(Viewport {
        top: 0,
        height: usize::MAX,
    });
    assert_eq!(all.len(), s.total_rows());
    assert_eq!(all[0][0].ch, 'a');

    // A `top` past the end, and a `top + height` that would overflow.
    assert!(s
        .viewport_rows(Viewport {
            top: usize::MAX,
            height: 4,
        })
        .is_empty());
    assert_eq!(
        s.viewport_rows(Viewport {
            top: 1,
            height: usize::MAX,
        })
        .len(),
        s.total_rows() - 1
    );
}

// ---------------------------------------------------------------------------
// Erase saved lines (CSI 3 J), CBT, and DECSTR
// ---------------------------------------------------------------------------

#[test]
fn erase_scrollback_drops_history_without_reusing_ids() {
    let mut s = Screen::with_scrollback(2, 3, 10);
    put(&mut s, 0, "aaa");
    put(&mut s, 1, "bbb");
    s.linefeed();
    s.linefeed(); // "aaa" and "bbb" are now in scrollback
    assert_eq!(s.scrollback().len(), 2);
    let grid_zero = s.absolute_row(0);
    let cursor_abs = s.cursor_absolute_row();

    s.erase_scrollback();

    assert!(s.scrollback().is_empty());
    assert_eq!(s.scrollback().evicted(), 2); // ids 0 and 1 are spent
    assert!(s.line(0).is_none());
    assert!(s.line(1).is_none());
    // The grid did not move: its rows keep the ids they already had.
    assert_eq!(s.absolute_row(0), grid_zero);
    assert_eq!(s.cursor_absolute_row(), cursor_abs);
    assert_eq!(s.total_rows(), 2);
    // New output lands after the spent ids, never on top of them.
    put(&mut s, 0, "ccc");
    assert_eq!(s.row_text(grid_zero).as_deref(), Some("ccc"));
}

#[test]
fn erase_scrollback_leaves_the_screen_and_modes_alone() {
    let mut s = Screen::with_scrollback(2, 4, 10);
    put(&mut s, 0, "keep");
    s.erase_in_display(EraseMode::All);
    put(&mut s, 0, "live");
    s.cursor_goto(1, 2);
    s.set_insert_mode(true);
    s.erase_scrollback();
    assert_eq!(active_text(&s, 0), "live");
    assert_eq!((s.cursor().row, s.cursor().col), (1, 2));
    assert!(s.insert_mode());
}

#[test]
fn back_tab_walks_back_through_tab_stops() {
    let mut s = Screen::new(1, 20);
    s.cursor_to_col(19);
    s.back_tab();
    assert_eq!(s.cursor().col, 16);
    s.back_tab();
    assert_eq!(s.cursor().col, 8);
    s.back_tab();
    assert_eq!(s.cursor().col, 0);
    s.back_tab(); // already home: stays put
    assert_eq!(s.cursor().col, 0);
}

#[test]
fn back_tab_honors_the_screens_own_tab_stops() {
    let mut s = Screen::new(1, 20);
    s.clear_all_tab_stops();
    s.cursor_to_col(5);
    s.set_tab_stop();
    s.cursor_to_col(12);
    s.set_tab_stop();
    s.cursor_to_col(19);
    s.back_tab();
    assert_eq!(s.cursor().col, 12);
    s.back_tab();
    assert_eq!(s.cursor().col, 5);
    s.back_tab();
    assert_eq!(s.cursor().col, 0); // no stop below 5
}

#[test]
fn back_tab_is_the_inverse_of_tab_and_clears_pending_wrap() {
    let mut s = Screen::new(1, 4);
    for ch in "abcd".chars() {
        s.print(ch);
    }
    assert!(s.pending_wrap());
    s.back_tab();
    assert!(!s.pending_wrap());
    assert_eq!(s.cursor().col, 0);
    s.tab();
    let after_tab = s.cursor().col;
    s.back_tab();
    assert!(s.cursor().col < after_tab);
}

#[test]
fn soft_reset_restores_modes_without_clearing_the_screen() {
    let mut s = Screen::with_scrollback(4, 10, 10);
    put(&mut s, 0, "keep me");
    s.erase_in_display(EraseMode::All); // seed scrollback
    put(&mut s, 0, "on screen");
    s.pen_mut().set_bold(true);
    s.pen_mut().bg = Color::Indexed(6);
    s.set_origin_mode(true);
    s.set_insert_mode(true);
    s.set_autowrap(false);
    s.set_scroll_region(2, 3);
    s.set_cursor_visible(false);
    s.cursor_goto(2, 4);
    s.save_cursor();
    s.cursor_goto(1, 1);
    let scrollback_before = s.scrollback().len();
    let cursor_before = (s.cursor().row, s.cursor().col);

    s.soft_reset();

    assert!(!s.origin_mode());
    assert!(!s.insert_mode());
    assert!(s.autowrap());
    assert_eq!(s.scroll_region(), (0, 3));
    assert_eq!(*s.pen(), CellAttrs::default());
    assert!(s.cursor().visible);
    assert!(!s.pending_wrap());
    // Screen, scrollback, and cursor position are untouched.
    assert_eq!(active_text(&s, 0), "on screen");
    assert_eq!(s.scrollback().len(), scrollback_before);
    assert_eq!(s.row_text(0).as_deref(), Some("keep me"));
    assert_eq!((s.cursor().row, s.cursor().col), cursor_before);
    // The saved cursor was dropped: DECRC now homes with a default pen.
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0));
    assert_eq!(*s.pen(), CellAttrs::default());
}

#[test]
fn soft_reset_keeps_tab_stops_and_the_alt_screen() {
    let mut s = Screen::new(3, 20);
    s.clear_all_tab_stops();
    s.cursor_to_col(5);
    s.set_tab_stop();
    s.enter_alt_screen();
    put(&mut s, 0, "alt");

    s.soft_reset();

    assert!(s.is_alt_screen()); // DECSTR does not leave the alt screen
    assert_eq!(active_text(&s, 0), "alt");
    s.cursor_to_col(0);
    s.tab();
    assert_eq!(s.cursor().col, 5); // custom tab stops survive
}

#[test]
fn soft_reset_only_drops_the_active_screens_saved_cursor() {
    let mut s = Screen::new(4, 4);
    s.cursor_goto(1, 1);
    s.save_cursor(); // primary slot
    s.enter_alt_screen();
    s.cursor_goto(2, 2);
    s.save_cursor(); // alt slot
    s.soft_reset(); // clears the alt slot only
    s.cursor_goto(3, 3);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (0, 0)); // alt slot gone
    s.exit_alt_screen();
    s.cursor_goto(3, 3);
    s.restore_cursor();
    assert_eq!((s.cursor().row, s.cursor().col), (1, 1)); // primary intact
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
