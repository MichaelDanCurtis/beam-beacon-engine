//! The `Screen` aggregate: grids, cursor, pen, modes, and the semantic
//! operations a VT parser drives.

use std::borrow::Cow;

use unicode_width::UnicodeWidthChar;

use crate::cell::{Cell, CellAttrs, CellWidth};
use crate::cursor::{Cursor, CursorShape};
use crate::grid::{Grid, Viewport};
use crate::scrollback::Scrollback;

/// Which portion of the display or line an erase operation targets.
///
/// Maps to the parameter of `ED` (erase in display) / `EL` (erase in line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EraseMode {
    /// From the cursor (inclusive) to the end.
    ToEnd,
    /// From the start to the cursor (inclusive).
    ToStart,
    /// Everything.
    All,
}

/// Cursor state captured by `save_cursor` (DECSC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SavedCursor {
    cursor: Cursor,
    pen: CellAttrs,
    origin_mode: bool,
}

/// The central terminal-state aggregate that a VT parser drives.
///
/// Owns the primary and alternate grids, cursor, saved cursor, current pen
/// attributes, scroll region, tab stops, pending-wrap flag, scrollback, and
/// the DEC/ANSI modes this crate models (autowrap, origin, insert, cursor
/// visibility).
///
/// # Coordinates
///
/// All rows/columns in this API are **0-indexed** except
/// [`Screen::set_scroll_region`], which takes the 1-based values that appear
/// in the `DECSTBM` escape sequence. When origin mode (DECOM) is enabled,
/// the row arguments of [`Screen::cursor_goto`] and [`Screen::cursor_to_row`]
/// are relative to the top of the scroll region and clamped inside it.
///
/// # Two grids, two coordinate systems
///
/// A `Screen` holds two grids and they answer two different questions:
///
/// - [`Screen::grid`] is **what to draw**: the *active* grid, which is the
///   alternate grid while a full-screen application (vim, htop, less) owns
///   the screen and the primary grid otherwise.
/// - Absolute rows are **the scrollback timeline**: they always address the
///   *primary* screen's history (scrollback plus [`Screen::primary_grid`]).
///   The alternate screen has no scrollback and never advances that
///   timeline, so an alt-screen episode leaves absolute rows completely
///   untouched — [`Screen::line`], [`Screen::row_text`], and
///   [`Screen::viewport_rows`] keep returning the shell output underneath
///   the running application, not the application's frame buffer.
///
/// # Absolute rows
///
/// Every row ever produced has a stable *absolute row id*:
/// `absolute = scrollback.evicted() + scrollback_index`, and primary grid
/// row `r` is `absolute_row(r) = evicted + scrollback.len() + r`. Ids of
/// evicted rows are never reused, so downstream consumers (e.g. a block
/// list) can hold them forever.
///
/// # Erase/scroll fill
///
/// Cells vacated by *any* operation — erase, scroll, insert, delete, the
/// area exposed by growing the screen, and the alternate screen's clear on
/// entry — are filled with a blank cell carrying the current pen's
/// *background* color (background-color-erase), with all other attributes
/// reset. Only [`Screen::reset`] (RIS) fills with the terminal default,
/// because RIS resets the pen first.
///
/// # Wide-character invariant
///
/// Every public operation leaves the grids *paired*: a
/// [`CellWidth::WideLead`] always has a [`CellWidth::WideTrail`]
/// immediately to its right, and a `WideTrail` always has a `WideLead`
/// immediately to its left — row edges included. Any operation that
/// writes or shifts part of a row heals both edges of the span it
/// touched, so no half of a wide character can be stranded (and then
/// frozen into scrollback) by an edit that cut a pair down the middle.
///
/// # Geometry limits
///
/// Dimensions are host-supplied and therefore untrusted. [`Screen::new`],
/// [`Screen::with_scrollback`], and [`Screen::resize`] clamp them to at
/// least `1 × 1` and at most [`Grid::MAX_ROWS`] × [`Grid::MAX_COLS`];
/// absurd geometry yields a capped screen rather than a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Screen {
    primary: Grid,
    alt: Grid,
    alt_active: bool,
    scrollback: Scrollback,
    cursor: Cursor,
    /// Live cursor of the screen that is **not** currently active.
    ///
    /// Cursor state is per-screen. The alternate screen's cursor must
    /// never decide what the *primary* grid sheds on a resize (a
    /// full-screen app parks its cursor wherever it likes, which has
    /// nothing to do with where the shell's output ended), and vice
    /// versa. [`Screen::enter_alt_screen`] / [`Screen::exit_alt_screen`]
    /// park the outgoing screen's cursor here; the incoming screen
    /// inherits the live position, which is the documented behavior.
    inactive_cursor: Cursor,
    saved_primary: Option<SavedCursor>,
    saved_alt: Option<SavedCursor>,
    pen: CellAttrs,
    scroll_top: usize,
    scroll_bottom: usize,
    tabs: Vec<bool>,
    pending_wrap: bool,
    autowrap: bool,
    origin_mode: bool,
    insert_mode: bool,
}

impl Screen {
    /// Default scrollback capacity used by [`Screen::new`], in lines.
    pub const DEFAULT_SCROLLBACK_LINES: usize = 10_000;

    /// Create a screen of `rows × cols` with the default scrollback
    /// capacity ([`Screen::DEFAULT_SCROLLBACK_LINES`]).
    ///
    /// Dimensions are clamped to at least `1 × 1` and at most
    /// [`Grid::MAX_ROWS`] × [`Grid::MAX_COLS`].
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_scrollback(rows, cols, Self::DEFAULT_SCROLLBACK_LINES)
    }

    /// Create a screen of `rows × cols` retaining at most `max_scrollback`
    /// lines of history. Dimensions are clamped to at least `1 × 1` and at
    /// most [`Grid::MAX_ROWS`] × [`Grid::MAX_COLS`].
    pub fn with_scrollback(rows: usize, cols: usize, max_scrollback: usize) -> Self {
        let (rows, cols) = Self::clamp_geometry(rows, cols);
        Self {
            primary: Grid::new(rows, cols),
            alt: Grid::new(rows, cols),
            alt_active: false,
            scrollback: Scrollback::new(max_scrollback),
            cursor: Cursor::default(),
            inactive_cursor: Cursor::default(),
            saved_primary: None,
            saved_alt: None,
            pen: CellAttrs::default(),
            scroll_top: 0,
            scroll_bottom: rows - 1,
            tabs: Self::default_tabs(cols),
            pending_wrap: false,
            autowrap: true,
            origin_mode: false,
            insert_mode: false,
        }
    }

    // -------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------

    /// Number of rows in the (primary and alternate) grid.
    pub fn rows(&self) -> usize {
        self.primary.rows()
    }

    /// Number of columns in the (primary and alternate) grid.
    pub fn cols(&self) -> usize {
        self.primary.cols()
    }

    /// Borrow the currently **active** grid: the alternate grid while the
    /// alternate screen is up, the primary grid otherwise.
    ///
    /// This is *what to draw* — renderers should always paint this grid,
    /// because it is what the user is looking at.
    ///
    /// It is deliberately **not** what absolute row ids address. Absolute
    /// rows are *the scrollback timeline* and always resolve against the
    /// primary screen (see [`Screen::line`], [`Screen::row_text`],
    /// [`Screen::viewport_rows`], and [`Screen::primary_grid`]). While a
    /// full-screen application owns the alternate screen, `grid()` returns
    /// the application's frame buffer and `row_text(absolute)` returns the
    /// shell output underneath it; both are correct answers to different
    /// questions.
    pub fn grid(&self) -> &Grid {
        if self.alt_active {
            &self.alt
        } else {
            &self.primary
        }
    }

    /// Borrow the **primary** grid, whether or not the alternate screen is
    /// currently active.
    ///
    /// This is the grid that absolute row ids address: primary grid row `r`
    /// is absolute row `evicted + scrollback.len() + r`. Use
    /// [`Screen::grid`] to draw the screen and this to reason about the
    /// session's output timeline (blocks, search, transcripts), which an
    /// alt-screen episode must not disturb.
    pub fn primary_grid(&self) -> &Grid {
        &self.primary
    }

    /// Borrow the cursor.
    pub fn cursor(&self) -> &Cursor {
        &self.cursor
    }

    /// Borrow the current pen (the attributes applied to printed cells).
    pub fn pen(&self) -> &CellAttrs {
        &self.pen
    }

    /// Mutably borrow the current pen. SGR handling in the parser mutates
    /// this directly.
    pub fn pen_mut(&mut self) -> &mut CellAttrs {
        &mut self.pen
    }

    /// Whether the alternate screen is active.
    pub fn is_alt_screen(&self) -> bool {
        self.alt_active
    }

    /// Whether a wrap is pending (a printable was written in the last
    /// column; the next printable wraps first if autowrap is enabled).
    pub fn pending_wrap(&self) -> bool {
        self.pending_wrap
    }

    /// The scroll region as inclusive 0-based `(top, bottom)` rows.
    pub fn scroll_region(&self) -> (usize, usize) {
        (self.scroll_top, self.scroll_bottom)
    }

    /// Borrow the scrollback history (always the primary screen's; the
    /// alternate screen has no scrollback).
    pub fn scrollback(&self) -> &Scrollback {
        &self.scrollback
    }

    // -------------------------------------------------------------------
    // Printing
    // -------------------------------------------------------------------

    /// Write `ch` at the cursor with the current pen, advancing the cursor.
    ///
    /// - Control characters (width `None`) and zero-width characters
    ///   (e.g. combining marks) are ignored.
    /// - Wide (two-column) characters occupy a [`CellWidth::WideLead`] +
    ///   [`CellWidth::WideTrail`] cell pair; overwriting either half of an
    ///   existing pair blanks the orphaned half.
    /// - Xterm-style deferred autowrap: printing in the last column leaves
    ///   the cursor there and sets the pending-wrap flag; the wrap happens
    ///   just before the *next* printable (if autowrap is enabled).
    /// - A wide character that does not fit in the remaining columns wraps
    ///   first (autowrap on, leaving a blank spacer in the last column) or
    ///   is printed at the last two columns (autowrap off).
    /// - In insert mode (IRM), existing cells shift right before writing.
    pub fn print(&mut self, ch: char) {
        let Some(width) = UnicodeWidthChar::width(ch) else {
            return; // control character
        };
        if width == 0 {
            return; // combining/zero-width: not modeled per-cell
        }
        let cols = self.cols();
        if width == 2 && cols < 2 {
            return; // cannot fit a wide char at all
        }

        // Deferred wrap from a previous print that filled the last column.
        if self.pending_wrap && self.autowrap {
            self.cursor.col = 0;
            self.scroll_or_step_down();
        }
        self.pending_wrap = false;

        let mut col = self.cursor.col.min(cols - 1);
        let row = self.cursor.row.min(self.rows() - 1);

        // A wide char that doesn't fit in the remaining columns.
        if width == 2 && col + 2 > cols {
            if self.autowrap {
                // The spacer overwrites the last column, which may be half
                // of an existing pair; clear the whole pair first.
                self.clean_wide_at(row, col);
                let spacer = self.blank_cell();
                if let Some(c) = self.grid_active_mut().cell_mut(row, col) {
                    *c = spacer;
                }
                self.cursor.col = 0;
                self.scroll_or_step_down();
                col = 0;
            } else {
                col = cols - 2;
            }
        }
        let row = self.cursor.row.min(self.rows() - 1);

        if self.insert_mode {
            self.clean_wide_at(row, col);
            let blank = self.blank_cell();
            let shift = width.min(cols - col);
            let grid = self.grid_active_mut();
            if let Some(cells) = grid.row_mut(row) {
                if let Some(span) = cells.get_mut(col..) {
                    span.rotate_right(shift);
                    if let Some(head) = span.get_mut(..shift) {
                        for c in head {
                            *c = blank;
                        }
                    }
                }
            }
            // The shift moved everything from `col` to the row end; a pair
            // pushed across that end leaves an orphaned lead behind.
            Self::heal_wide_span(grid, row, col, cols, blank);
        }

        self.clean_wide_at(row, col);
        if width == 2 {
            self.clean_wide_at(row, col + 1);
        }

        let attrs = self.pen;
        let grid = self.grid_active_mut();
        if let Some(c) = grid.cell_mut(row, col) {
            *c = Cell {
                ch,
                attrs,
                width: if width == 2 {
                    CellWidth::WideLead
                } else {
                    CellWidth::Single
                },
            };
        }
        if width == 2 {
            if let Some(c) = grid.cell_mut(row, col + 1) {
                *c = Cell {
                    ch: ' ',
                    attrs,
                    width: CellWidth::WideTrail,
                };
            }
        }

        let end = col + width;
        if end >= cols {
            self.cursor.col = cols - 1;
            self.pending_wrap = true;
        } else {
            self.cursor.col = end;
        }
    }

    /// Print every character of `s` as if by repeated [`Screen::print`].
    ///
    /// Semantically identical to `for ch in s.chars() { screen.print(ch) }`
    /// — the crate's chunk-equivalence tests hold it to that — but runs of
    /// single-width characters are written in one pass instead of paying
    /// the full per-character path (width lookup, wide-pair healing, cell
    /// addressing) for every cell. Wide pairs straddling a run's edges are
    /// healed exactly as `print` would; pairs wholly inside the run are
    /// simply overwritten.
    ///
    /// This is the parser's hot path for plain text; hosts echoing strings
    /// can use it directly.
    pub fn print_str(&mut self, s: &str) {
        let mut chars = s.chars();
        while let Some(ch) = chars.next() {
            // Anything that needs the full state machine goes through
            // `print`: wide and zero-width chars, insert mode, and a
            // pending wrap (which may scroll).
            let single = UnicodeWidthChar::width(ch) == Some(1);
            if !single || self.insert_mode || self.pending_wrap {
                self.print(ch);
                continue;
            }

            let cols = self.cols();
            let row = self.cursor.row.min(self.rows() - 1);
            let col = self.cursor.col.min(cols - 1);
            let free = cols - col;

            // Collect the burst: `ch` plus as many further single-width
            // chars as fit in the rest of this row. `as_str` lets us peek
            // without consuming past the first char that breaks the run.
            let rest = chars.as_str();
            let mut burst = 1usize;
            let mut consumed = 0usize;
            for next in rest.chars() {
                if burst == free || UnicodeWidthChar::width(next) != Some(1) {
                    break;
                }
                burst += 1;
                consumed += next.len_utf8();
            }
            chars = rest
                .get(consumed..)
                .map(str::chars)
                .unwrap_or_else(|| "".chars());

            // Heal wide pairs cut by the span's edges; interior pairs are
            // fully overwritten below. (Matches `print`, which heals at
            // every written cell — interior heals are no-ops here because
            // both halves get new content.)
            self.clean_wide_at(row, col);
            if burst > 1 {
                self.clean_wide_at(row, col + burst - 1);
            }

            let attrs = self.pen;
            let grid = self.grid_active_mut();
            if let Some(cells) = grid.row_mut(row) {
                if let Some(span) = cells.get_mut(col..col + burst) {
                    let mut it = std::iter::once(ch).chain(rest.chars());
                    for cell in span {
                        if let Some(c) = it.next() {
                            *cell = Cell {
                                ch: c,
                                attrs,
                                width: CellWidth::Single,
                            };
                        }
                    }
                }
            }

            let end = col + burst;
            if end >= cols {
                self.cursor.col = cols - 1;
                self.pending_wrap = true;
            } else {
                self.cursor.col = end;
            }
        }
    }

    // -------------------------------------------------------------------
    // Control operations
    // -------------------------------------------------------------------

    /// Line feed: move the cursor down one row; if it is on the bottom
    /// margin of the scroll region, scroll the region up instead, pushing
    /// the scrolled-out row to scrollback iff the primary screen is active
    /// and the scroll region covers the full screen.
    pub fn linefeed(&mut self) {
        self.pending_wrap = false;
        self.scroll_or_step_down();
    }

    /// Carriage return: move the cursor to column 0.
    pub fn carriage_return(&mut self) {
        self.pending_wrap = false;
        self.cursor.col = 0;
    }

    /// Backspace: move the cursor one column left, stopping at column 0.
    pub fn backspace(&mut self) {
        self.pending_wrap = false;
        self.cursor.col = self.cursor.col.saturating_sub(1);
    }

    /// Horizontal tab: move the cursor to the next tab stop, or to the last
    /// column if there is none.
    pub fn tab(&mut self) {
        self.pending_wrap = false;
        let cols = self.cols();
        let next = (self.cursor.col + 1..cols).find(|&c| self.tabs.get(c) == Some(&true));
        self.cursor.col = next.unwrap_or(cols - 1);
    }

    /// Backward tab (CBT): move the cursor to the previous tab stop on the
    /// current row, or to column 0 if there is none.
    ///
    /// Honors the screen's own tab stops, so it is the exact inverse of
    /// [`Screen::tab`] over whatever stops HTS/TBC have left in place.
    /// Already at column 0, this is a no-op.
    pub fn back_tab(&mut self) {
        self.pending_wrap = false;
        let prev = (0..self.cursor.col)
            .rev()
            .find(|&c| self.tabs.get(c) == Some(&true));
        self.cursor.col = prev.unwrap_or(0);
    }

    /// Set a tab stop at the cursor column (HTS).
    pub fn set_tab_stop(&mut self) {
        if let Some(t) = self.tabs.get_mut(self.cursor.col) {
            *t = true;
        }
    }

    /// Clear the tab stop at the cursor column (TBC 0).
    pub fn clear_tab_stop(&mut self) {
        if let Some(t) = self.tabs.get_mut(self.cursor.col) {
            *t = false;
        }
    }

    /// Clear all tab stops (TBC 3).
    pub fn clear_all_tab_stops(&mut self) {
        for t in &mut self.tabs {
            *t = false;
        }
    }

    // -------------------------------------------------------------------
    // Cursor movement
    // -------------------------------------------------------------------

    /// Move the cursor up `n` rows, stopping at the top margin (or row 0 if
    /// the cursor is above the scroll region).
    pub fn cursor_up(&mut self, n: usize) {
        self.pending_wrap = false;
        let floor = if self.cursor.row >= self.scroll_top {
            self.scroll_top
        } else {
            0
        };
        self.cursor.row = self.cursor.row.saturating_sub(n).max(floor);
    }

    /// Move the cursor down `n` rows, stopping at the bottom margin (or the
    /// last row if the cursor is below the scroll region).
    pub fn cursor_down(&mut self, n: usize) {
        self.pending_wrap = false;
        let ceil = if self.cursor.row <= self.scroll_bottom {
            self.scroll_bottom
        } else {
            self.rows() - 1
        };
        self.cursor.row = self.cursor.row.saturating_add(n).min(ceil);
    }

    /// Move the cursor left `n` columns, stopping at column 0.
    pub fn cursor_left(&mut self, n: usize) {
        self.pending_wrap = false;
        self.cursor.col = self.cursor.col.saturating_sub(n);
    }

    /// Move the cursor right `n` columns, stopping at the last column.
    pub fn cursor_right(&mut self, n: usize) {
        self.pending_wrap = false;
        self.cursor.col = self.cursor.col.saturating_add(n).min(self.cols() - 1);
    }

    /// Move the cursor to 0-based `(row, col)`, clamped to the grid. With
    /// origin mode enabled, `row` is relative to the scroll region's top
    /// and clamped inside the region.
    pub fn cursor_goto(&mut self, row: usize, col: usize) {
        self.pending_wrap = false;
        self.cursor.row = self.resolve_row(row);
        self.cursor.col = col.min(self.cols() - 1);
    }

    /// Move the cursor to 0-based column `col` (clamped), keeping the row.
    pub fn cursor_to_col(&mut self, col: usize) {
        self.pending_wrap = false;
        self.cursor.col = col.min(self.cols() - 1);
    }

    /// Move the cursor to 0-based row `row` (clamped, origin-mode aware),
    /// keeping the column.
    pub fn cursor_to_row(&mut self, row: usize) {
        self.pending_wrap = false;
        self.cursor.row = self.resolve_row(row);
    }

    /// Save the cursor position, pen, and origin mode (DECSC). The primary
    /// and alternate screens have independent save slots.
    pub fn save_cursor(&mut self) {
        let saved = SavedCursor {
            cursor: self.cursor,
            pen: self.pen,
            origin_mode: self.origin_mode,
        };
        *self.saved_slot_mut() = Some(saved);
    }

    /// Restore the state captured by [`Screen::save_cursor`] (DECRC),
    /// clamping the position to the current grid. If nothing was saved,
    /// resets the cursor to the origin with default pen and origin mode
    /// off (xterm behavior).
    pub fn restore_cursor(&mut self) {
        self.pending_wrap = false;
        match *self.saved_slot_mut() {
            Some(saved) => {
                let shape = self.cursor.shape;
                let visible = self.cursor.visible;
                self.cursor = Cursor {
                    row: saved.cursor.row.min(self.rows() - 1),
                    col: saved.cursor.col.min(self.cols() - 1),
                    shape,
                    visible,
                };
                self.pen = saved.pen;
                self.origin_mode = saved.origin_mode;
            }
            None => {
                self.cursor.row = 0;
                self.cursor.col = 0;
                self.pen = CellAttrs::default();
                self.origin_mode = false;
            }
        }
    }

    // -------------------------------------------------------------------
    // Editing
    // -------------------------------------------------------------------

    /// Erase in display (ED). With [`EraseMode::All`] on the primary screen,
    /// the screen's contents are first pushed to scrollback (like modern
    /// terminals), so their absolute row ids remain addressable. The cursor
    /// does not move.
    ///
    /// Only rows up to the **last non-blank row** are pushed; trailing
    /// blank rows are dropped. This matters more than it sounds: `clear(1)`
    /// on a mostly-empty screen would otherwise spend a full screen of
    /// scrollback per invocation, and a few hundred `clear`s would evict
    /// every real line of history behind a wall of blanks. A row counts as
    /// blank only when every cell is exactly [`Cell::default`] — a screen
    /// painted with a non-default background is content, not emptiness. A
    /// wholly blank screen pushes nothing at all.
    pub fn erase_in_display(&mut self, mode: EraseMode) {
        self.pending_wrap = false;
        let blank = self.blank_cell();
        let rows = self.rows();
        match mode {
            EraseMode::ToEnd => {
                self.erase_in_line(EraseMode::ToEnd);
                let row = self.cursor.row;
                let grid = self.grid_active_mut();
                for r in row + 1..rows {
                    grid.fill_row(r, blank);
                }
            }
            EraseMode::ToStart => {
                let row = self.cursor.row;
                {
                    let grid = self.grid_active_mut();
                    for r in 0..row {
                        grid.fill_row(r, blank);
                    }
                }
                self.erase_in_line(EraseMode::ToStart);
            }
            EraseMode::All => {
                if !self.alt_active {
                    // History is the rows that held something, not the
                    // height of the window they were shown in.
                    if let Some(last) = Self::last_non_blank_row(&self.primary) {
                        for r in 0..=last {
                            if let Some(cells) = self.primary.row(r) {
                                self.scrollback.push_row_copy(cells);
                            }
                        }
                    }
                }
                self.grid_active_mut().fill(blank);
            }
        }
    }

    /// Erase in line (EL) on the cursor's row. The cursor does not move.
    ///
    /// A wide-character pair straddling either end of the erased span is
    /// cleared whole, so no orphaned half survives.
    pub fn erase_in_line(&mut self, mode: EraseMode) {
        self.pending_wrap = false;
        let blank = self.blank_cell();
        let cols = self.cols();
        let row = self.cursor.row;
        let col = self.cursor.col.min(cols - 1);
        let (start, end) = match mode {
            EraseMode::ToEnd => (col, cols),
            EraseMode::ToStart => (0, col + 1),
            EraseMode::All => (0, cols),
        };
        let grid = self.grid_active_mut();
        if let Some(cells) = grid.row_mut(row) {
            if let Some(span) = cells.get_mut(start..end) {
                for c in span {
                    *c = blank;
                }
            }
        }
        Self::heal_wide_span(grid, row, start, end, blank);
    }

    /// Erase `n` characters at and after the cursor on the current row
    /// (ECH). Cells are blanked in place; nothing shifts.
    ///
    /// A wide-character pair straddling either end of the erased span is
    /// cleared whole, so no orphaned half survives.
    pub fn erase_chars(&mut self, n: usize) {
        self.pending_wrap = false;
        let blank = self.blank_cell();
        let cols = self.cols();
        let row = self.cursor.row;
        let col = self.cursor.col.min(cols - 1);
        let end = col.saturating_add(n).min(cols);
        let grid = self.grid_active_mut();
        if let Some(cells) = grid.row_mut(row) {
            if let Some(span) = cells.get_mut(col..end) {
                for c in span {
                    *c = blank;
                }
            }
        }
        Self::heal_wide_span(grid, row, col, end, blank);
    }

    /// Insert `n` blank characters at the cursor (ICH), shifting the rest
    /// of the row right; cells pushed past the last column are lost.
    ///
    /// A wide-character pair split at the insertion point, or pushed
    /// across the row end, is cleared whole rather than left orphaned.
    pub fn insert_chars(&mut self, n: usize) {
        self.pending_wrap = false;
        let blank = self.blank_cell();
        let cols = self.cols();
        let row = self.cursor.row;
        let col = self.cursor.col.min(cols - 1);
        let n = n.min(cols - col);
        if n == 0 {
            return;
        }
        self.clean_wide_at(row, col);
        let grid = self.grid_active_mut();
        if let Some(cells) = grid.row_mut(row) {
            if let Some(span) = cells.get_mut(col..) {
                span.rotate_right(n);
                if let Some(head) = span.get_mut(..n) {
                    for c in head {
                        *c = blank;
                    }
                }
            }
        }
        Self::heal_wide_span(grid, row, col, cols, blank);
    }

    /// Delete `n` characters at the cursor (DCH), shifting the rest of the
    /// row left and blanking the vacated cells at the end.
    ///
    /// A wide-character pair split at the cursor, or cut in half by the
    /// text shifted in from `col + n`, is cleared whole rather than left
    /// orphaned.
    pub fn delete_chars(&mut self, n: usize) {
        self.pending_wrap = false;
        let blank = self.blank_cell();
        let cols = self.cols();
        let row = self.cursor.row;
        let col = self.cursor.col.min(cols - 1);
        let n = n.min(cols - col);
        if n == 0 {
            return;
        }
        self.clean_wide_at(row, col);
        let grid = self.grid_active_mut();
        if let Some(cells) = grid.row_mut(row) {
            if let Some(span) = cells.get_mut(col..) {
                span.rotate_left(n);
                let blank_from = span.len() - n;
                if let Some(tail) = span.get_mut(blank_from..) {
                    for c in tail {
                        *c = blank;
                    }
                }
            }
        }
        Self::heal_wide_span(grid, row, col, cols, blank);
    }

    /// Insert `n` blank lines at the cursor row (IL), shifting lines inside
    /// the scroll region down. No-op if the cursor is outside the region.
    /// The cursor column is unchanged.
    pub fn insert_lines(&mut self, n: usize) {
        self.pending_wrap = false;
        let row = self.cursor.row;
        if row < self.scroll_top || row > self.scroll_bottom {
            return;
        }
        let blank = self.blank_cell();
        let bottom = self.scroll_bottom;
        self.grid_active_mut()
            .scroll_down_within(row, bottom, n, blank);
    }

    /// Delete `n` lines at the cursor row (DL), shifting lines inside the
    /// scroll region up. No-op if the cursor is outside the region. Deleted
    /// lines are *not* pushed to scrollback.
    pub fn delete_lines(&mut self, n: usize) {
        self.pending_wrap = false;
        let row = self.cursor.row;
        if row < self.scroll_top || row > self.scroll_bottom {
            return;
        }
        let blank = self.blank_cell();
        let bottom = self.scroll_bottom;
        self.grid_active_mut()
            .scroll_up_within(row, bottom, n, blank);
    }

    /// Scroll the scroll region up `n` lines (SU). Scrolled-out rows are
    /// pushed to scrollback iff the primary screen is active and the scroll
    /// region covers the full screen (same rule as [`Screen::linefeed`]).
    pub fn scroll_up(&mut self, n: usize) {
        self.pending_wrap = false;
        self.scroll_region_up(n);
    }

    /// Scroll the scroll region down `n` lines (SD). Nothing is pushed to
    /// scrollback.
    pub fn scroll_down(&mut self, n: usize) {
        self.pending_wrap = false;
        let blank = self.blank_cell();
        let (top, bottom) = (self.scroll_top, self.scroll_bottom);
        self.grid_active_mut()
            .scroll_down_within(top, bottom, n, blank);
    }

    // -------------------------------------------------------------------
    // Scroll region and modes
    // -------------------------------------------------------------------

    /// Set the scroll region from **1-based** top/bottom rows (DECSTBM
    /// argument convention; `0` means "default", i.e. the first/last row).
    /// Values are clamped to the screen; a region smaller than two rows is
    /// ignored. On success the cursor moves to the origin (of the region if
    /// origin mode is on, of the screen otherwise).
    pub fn set_scroll_region(&mut self, top: usize, bottom: usize) {
        let rows = self.rows();
        let top = if top == 0 { 1 } else { top };
        let bottom = if bottom == 0 { rows } else { bottom.min(rows) };
        if top >= bottom {
            return;
        }
        self.scroll_top = top - 1;
        self.scroll_bottom = bottom - 1;
        self.pending_wrap = false;
        self.cursor.row = if self.origin_mode { self.scroll_top } else { 0 };
        self.cursor.col = 0;
    }

    /// Reset the scroll region to the full screen and move the cursor to
    /// the origin.
    pub fn reset_scroll_region(&mut self) {
        self.scroll_top = 0;
        self.scroll_bottom = self.rows() - 1;
        self.pending_wrap = false;
        self.cursor.row = 0;
        self.cursor.col = 0;
    }

    /// Enable/disable autowrap (DECAWM). Default: enabled.
    pub fn set_autowrap(&mut self, on: bool) {
        self.autowrap = on;
    }

    /// Whether autowrap (DECAWM) is enabled.
    pub fn autowrap(&self) -> bool {
        self.autowrap
    }

    /// Enable/disable origin mode (DECOM). Toggling it homes the cursor
    /// (to the scroll region's origin when enabling). Default: disabled.
    pub fn set_origin_mode(&mut self, on: bool) {
        self.origin_mode = on;
        self.pending_wrap = false;
        self.cursor.row = if on { self.scroll_top } else { 0 };
        self.cursor.col = 0;
    }

    /// Whether origin mode (DECOM) is enabled.
    pub fn origin_mode(&self) -> bool {
        self.origin_mode
    }

    /// Enable/disable insert mode (IRM). Default: disabled.
    pub fn set_insert_mode(&mut self, on: bool) {
        self.insert_mode = on;
    }

    /// Whether insert mode (IRM) is enabled.
    pub fn insert_mode(&self) -> bool {
        self.insert_mode
    }

    /// Show or hide the cursor (DECTCEM).
    pub fn set_cursor_visible(&mut self, visible: bool) {
        self.cursor.visible = visible;
    }

    /// Set the cursor's visual shape.
    pub fn set_cursor_shape(&mut self, shape: CursorShape) {
        self.cursor.shape = shape;
    }

    // -------------------------------------------------------------------
    // Alternate screen
    // -------------------------------------------------------------------

    /// Switch to the alternate screen, clearing it first (to the current
    /// pen's background). No-op if already active. The alternate screen
    /// has no scrollback. Cursor position is preserved (callers
    /// implementing mode 1049 should bracket this with
    /// [`Screen::save_cursor`] / [`Screen::restore_cursor`]).
    ///
    /// The primary screen's cursor is parked, so a [`Screen::resize`]
    /// during the alt-screen episode still knows where the *shell's*
    /// output ended and does not let the running application's cursor
    /// decide what the primary grid sheds.
    pub fn enter_alt_screen(&mut self) {
        if self.alt_active {
            return;
        }
        self.inactive_cursor = self.cursor;
        self.alt_active = true;
        let blank = self.blank_cell();
        self.alt.fill(blank);
        self.pending_wrap = false;
    }

    /// Switch back to the primary screen. No-op if already primary. The
    /// alternate screen's cursor is parked; the primary screen inherits
    /// the live cursor position.
    pub fn exit_alt_screen(&mut self) {
        if !self.alt_active {
            return;
        }
        self.inactive_cursor = self.cursor;
        self.alt_active = false;
        self.pending_wrap = false;
    }

    // -------------------------------------------------------------------
    // Resize and reset
    // -------------------------------------------------------------------

    /// Resize both grids to `rows × cols`, **preserving on-screen
    /// content**. Dimensions are clamped to at least `1 × 1` and at most
    /// [`Grid::MAX_ROWS`] × [`Grid::MAX_COLS`].
    ///
    /// # Policy
    ///
    /// - **Overlap is kept.** A cell at `(row, col)` that exists in both the
    ///   old and the new dimensions keeps its character *and* its
    ///   attributes. Cells exposed by growth are blank, carrying the
    ///   current pen's background (background-color-erase).
    /// - **No reflow.** Lines are neither rewrapped when the screen narrows
    ///   nor unwrapped when it widens; columns past the new width are
    ///   dropped. (Reflow is a much larger change and is out of scope.) A
    ///   wide-character pair split by the new right edge has its orphaned
    ///   lead blanked so no half character survives.
    /// - **Shrinking height never loses content.** Rows that no longer fit
    ///   are taken off the *top* and **pushed into scrollback**, exactly as
    ///   if the shell had scrolled them off, so they stay addressable at
    ///   *the same absolute row ids*. Enough rows are shed to keep both the
    ///   cursor row *and* the last row holding content on screen. Only
    ///   genuinely blank trailing rows — every cell exactly
    ///   [`Cell::default`] — are discarded off the bottom.
    /// - **Each screen answers for itself.** The primary grid's shed count
    ///   is decided by the *primary* cursor and the *primary* content; the
    ///   alternate grid's by the alternate cursor. A full-screen
    ///   application parking its cursor at the top or bottom has no say
    ///   over the shell output underneath it. The alternate screen has no
    ///   scrollback, so its shed rows are simply discarded and it never
    ///   feeds — or renumbers — the primary timeline.
    /// - **Absolute rows stay coherent.** Every preserved row keeps its
    ///   absolute id: rows shed off the top move into scrollback, which
    ///   grows by exactly the same amount. Scrollback itself is untouched —
    ///   its rows keep the width they had when they were pushed.
    ///   [`Screen::cursor_absolute_row`] is likewise unchanged whenever the
    ///   cursor's row survives on the grid. It can only advance when the
    ///   cursor sat *above* content that had to be preserved and its own
    ///   row was shed into scrollback; the cursor then lands on the topmost
    ///   surviving row, because a cursor cannot point into history.
    /// - The cursor, the parked cursor of the inactive screen, and both
    ///   saved cursors follow their own screen's content and are clamped
    ///   into the new bounds; the scroll region resets to the full screen,
    ///   tab stops are rebuilt every 8 columns, and the pending-wrap flag
    ///   is cleared (all standard on resize).
    pub fn resize(&mut self, rows: usize, cols: usize) {
        let (rows, cols) = Self::clamp_geometry(rows, cols);
        let old_rows = self.rows();
        let blank = self.blank_cell();

        // Each grid decides its own shed from its own cursor and its own
        // content. Sharing one count is how the alt screen used to delete
        // the shell output underneath it.
        let primary_shed = self.primary_shed(rows);
        let alt_shed = self.alt_shed(rows);

        if primary_shed > 0 {
            // Primary rows leaving the top keep their absolute ids by
            // moving into scrollback.
            for r in 0..primary_shed {
                if let Some(cells) = self.primary.row(r) {
                    self.scrollback.push_row_copy(cells);
                }
            }
            self.primary
                .scroll_up_within(0, old_rows - 1, primary_shed, blank);
        }
        if alt_shed > 0 {
            // The alternate screen has no history: its shed rows are gone.
            self.alt.scroll_up_within(0, old_rows - 1, alt_shed, blank);
        }

        self.primary.resize_preserving(rows, cols, blank);
        self.alt.resize_preserving(rows, cols, blank);
        Self::heal_wide_edges(&mut self.primary, blank);
        Self::heal_wide_edges(&mut self.alt, blank);

        self.scroll_top = 0;
        self.scroll_bottom = rows - 1;
        self.tabs = Self::default_tabs(cols);

        let (live_shed, parked_shed) = if self.alt_active {
            (alt_shed, primary_shed)
        } else {
            (primary_shed, alt_shed)
        };
        Self::follow_resize(&mut self.cursor, live_shed, rows, cols);
        Self::follow_resize(&mut self.inactive_cursor, parked_shed, rows, cols);
        if let Some(saved) = self.saved_primary.as_mut() {
            Self::follow_resize(&mut saved.cursor, primary_shed, rows, cols);
        }
        if let Some(saved) = self.saved_alt.as_mut() {
            Self::follow_resize(&mut saved.cursor, alt_shed, rows, cols);
        }
        self.pending_wrap = false;
    }

    /// Full reset (RIS): clears both grids, returns to the primary screen,
    /// and restores every mode, the pen, tab stops, scroll region, cursor,
    /// and saved cursors to their defaults. Scrollback is kept intact so
    /// absolute row ids stay stable.
    pub fn reset(&mut self) {
        let rows = self.rows();
        let cols = self.cols();
        self.primary.clear();
        self.alt.clear();
        self.alt_active = false;
        self.cursor = Cursor::default();
        self.inactive_cursor = Cursor::default();
        self.saved_primary = None;
        self.saved_alt = None;
        self.pen = CellAttrs::default();
        self.scroll_top = 0;
        self.scroll_bottom = rows - 1;
        self.tabs = Self::default_tabs(cols);
        self.pending_wrap = false;
        self.autowrap = true;
        self.origin_mode = false;
        self.insert_mode = false;
    }

    /// Soft terminal reset (DECSTR, `CSI ! p`).
    ///
    /// Restores origin mode, insert mode, and autowrap to their defaults,
    /// resets the scroll region to the full screen, resets the pen, shows
    /// the cursor, clears the pending-wrap flag, and drops the active
    /// screen's saved cursor (so a later [`Screen::restore_cursor`] homes
    /// the cursor with a default pen, as xterm does).
    ///
    /// Deliberately **does not** clear the screen, the scrollback, or the
    /// tab stops, and does not move the cursor — that is [`Screen::reset`]
    /// (RIS).
    pub fn soft_reset(&mut self) {
        self.origin_mode = false;
        self.insert_mode = false;
        self.autowrap = true;
        self.scroll_top = 0;
        self.scroll_bottom = self.rows() - 1;
        self.pen = CellAttrs::default();
        self.cursor.visible = true;
        self.pending_wrap = false;
        *self.saved_slot_mut() = None;
    }

    /// Drop every retained scrollback row (xterm "Erase Saved Lines",
    /// `CSI 3 J`).
    ///
    /// The grids, the cursor, and every mode are untouched — only history
    /// goes. Absolute-id monotonicity is preserved: the dropped rows are
    /// accounted for as *evicted*, so their ids are never reused and every
    /// surviving row (including every grid row) keeps the id it already
    /// had. [`Screen::line`] and [`Screen::row_text`] simply return `None`
    /// for the dropped ids from now on, exactly as they do for rows that
    /// aged out of a full buffer.
    pub fn erase_scrollback(&mut self) {
        self.scrollback.clear();
    }

    // -------------------------------------------------------------------
    // Absolute-row addressing
    //
    // Absolute rows are a coordinate on the PRIMARY screen's scrollback
    // timeline. Every method in this section resolves against scrollback
    // plus the primary grid, never the alternate grid — see the type-level
    // "Two grids, two coordinate systems" docs.
    // -------------------------------------------------------------------

    /// Number of currently addressable rows on the primary timeline:
    /// retained scrollback rows plus primary grid rows. (Valid absolute ids
    /// span `scrollback().evicted() .. scrollback().evicted() +
    /// total_rows()`.)
    ///
    /// Unaffected by the alternate screen: it has no scrollback and the
    /// grids share their dimensions.
    pub fn total_rows(&self) -> usize {
        self.scrollback.len() + self.rows()
    }

    /// The stable absolute row id of **primary** grid row `grid_row`:
    /// `evicted + scrollback.len() + grid_row`.
    ///
    /// This is a coordinate on the scrollback timeline, so it keeps its
    /// meaning while the alternate screen is active (the alternate screen
    /// does not advance the timeline).
    pub fn absolute_row(&self, grid_row: usize) -> usize {
        self.scrollback.evicted() + self.scrollback.len() + grid_row
    }

    /// The absolute row id of the cursor's row, on the primary timeline.
    pub fn cursor_absolute_row(&self) -> usize {
        self.absolute_row(self.cursor.row)
    }

    /// Fetch a row by absolute id, unified over scrollback and the
    /// **primary** grid. Returns `None` if the row was evicted or the id is
    /// out of range.
    ///
    /// Never reads the alternate grid: while a full-screen application is
    /// running this still returns the primary-screen content that owns
    /// those absolute ids. Use [`Screen::grid`] to see what is on screen.
    pub fn line(&self, absolute_row: usize) -> Option<Cow<'_, [Cell]>> {
        self.row_slice(absolute_row).map(Cow::Borrowed)
    }

    /// The text of the row at `absolute_row`: trailing blank cells are
    /// trimmed and [`CellWidth::WideTrail`] spacers are skipped. Returns
    /// `None` if the row was evicted or out of range.
    ///
    /// Reads scrollback plus the **primary** grid, like [`Screen::line`].
    pub fn row_text(&self, absolute_row: usize) -> Option<String> {
        let cells = self.row_slice(absolute_row)?;
        let mut end = cells.len();
        while end > 0 {
            match cells.get(end - 1) {
                Some(c) if c.width == CellWidth::Single && c.ch == ' ' => end -= 1,
                _ => break,
            }
        }
        Some(
            cells
                .get(..end)
                .unwrap_or(&[])
                .iter()
                .filter(|c| c.width != CellWidth::WideTrail)
                .map(|c| c.ch)
                .collect(),
        )
    }

    /// Project a [`Viewport`] onto the buffer: returns up to `vp.height`
    /// row slices starting at absolute row `vp.top`. Rows that were evicted
    /// or fall past the end of the grid are omitted.
    ///
    /// [`Viewport::top`] is an absolute row id, so this addresses the
    /// **primary** timeline (scrollback plus the primary grid) exactly like
    /// [`Screen::line`] — including while the alternate screen is active.
    /// A viewport is a window onto history; to render the live screen, draw
    /// [`Screen::grid`] instead.
    ///
    /// `vp` is host-supplied and only bounds the *request*: both the
    /// reservation and the scan are clipped to the ids that can actually
    /// resolve, so an absurd `top` or `height` costs nothing — it can
    /// neither overflow the allocator nor spin through a range of ids that
    /// were never produced.
    pub fn viewport_rows(&self, vp: Viewport) -> Vec<&[Cell]> {
        // Ids outside `evicted .. evicted + total_rows` never resolve, so
        // clip to that window rather than walking the requested range.
        let first = self.scrollback.evicted();
        let past_end = first.saturating_add(self.total_rows());
        let start = vp.top.max(first);
        let end = vp.top.saturating_add(vp.height).min(past_end);
        let mut out = Vec::with_capacity(end.saturating_sub(start));
        for abs in start..end {
            if let Some(row) = self.row_slice(abs) {
                out.push(row);
            }
        }
        out
    }

    // -------------------------------------------------------------------
    // Internals
    // -------------------------------------------------------------------

    fn default_tabs(cols: usize) -> Vec<bool> {
        (0..cols).map(|c| c != 0 && c % 8 == 0).collect()
    }

    /// Clamp a host-supplied geometry into the range this crate will
    /// allocate: at least `1 × 1`, at most [`Grid::MAX_ROWS`] ×
    /// [`Grid::MAX_COLS`].
    fn clamp_geometry(rows: usize, cols: usize) -> (usize, usize) {
        (rows.clamp(1, Grid::MAX_ROWS), cols.clamp(1, Grid::MAX_COLS))
    }

    /// Restore the wide-pair invariant across the boundary immediately
    /// *before* column `col` of `row`.
    ///
    /// Exactly two things can be wrong at a boundary, and this fixes both:
    /// a [`CellWidth::WideLead`] at `col - 1` whose trail is no longer to
    /// its right, and a [`CellWidth::WideTrail`] at `col` whose lead is no
    /// longer to its left. Passing `col == 0` checks the row's left edge
    /// and `col == cols` its right edge, since a lead in the last column
    /// and a trail in the first are orphans by definition.
    fn heal_wide_boundary(grid: &mut Grid, row: usize, col: usize, blank: Cell) {
        let left_col = col.checked_sub(1);
        let left = left_col.and_then(|c| grid.cell(row, c)).map(|c| c.width);
        let right = grid.cell(row, col).map(|c| c.width);
        let paired = left == Some(CellWidth::WideLead) && right == Some(CellWidth::WideTrail);
        if !paired {
            if left == Some(CellWidth::WideLead) {
                if let Some(cell) = left_col.and_then(|c| grid.cell_mut(row, c)) {
                    *cell = blank;
                }
            }
            if right == Some(CellWidth::WideTrail) {
                if let Some(cell) = grid.cell_mut(row, col) {
                    *cell = blank;
                }
            }
        }
    }

    /// Restore the wide-pair invariant at **both** edges of the span
    /// `[start, end)` that was just written to, shifted, or shifted into.
    ///
    /// Every span edit cuts the row in exactly two places; a wide pair
    /// straddling either cut loses one half and must lose the other too.
    fn heal_wide_span(grid: &mut Grid, row: usize, start: usize, end: usize, blank: Cell) {
        Self::heal_wide_boundary(grid, row, start, blank);
        Self::heal_wide_boundary(grid, row, end, blank);
    }

    /// Restore the wide-pair invariant at both edges of every row.
    ///
    /// Narrowing a grid without reflow can cut a wide pair in half: the
    /// trail cell falls off the edge and the lead would otherwise claim a
    /// column it no longer has.
    fn heal_wide_edges(grid: &mut Grid, blank: Cell) {
        let cols = grid.cols();
        for r in 0..grid.rows() {
            Self::heal_wide_span(grid, r, 0, cols, blank);
        }
    }

    /// Whether `cells` holds nothing worth remembering: every cell is
    /// exactly [`Cell::default`] (a space, default colors, no attributes,
    /// single width).
    fn row_is_blank(cells: &[Cell]) -> bool {
        cells.iter().all(|c| *c == Cell::default())
    }

    /// Index of the last row of `grid` that holds content, or `None` if
    /// the whole grid is blank.
    fn last_non_blank_row(grid: &Grid) -> Option<usize> {
        (0..grid.rows())
            .rev()
            .find(|&r| grid.row(r).is_some_and(|cells| !Self::row_is_blank(cells)))
    }

    /// How many rows the **primary** grid must shed off its top to shrink
    /// to `rows`.
    ///
    /// Enough to keep the primary cursor on screen *and* enough that no
    /// row still holding content falls off the bottom — whichever is
    /// larger. Both bounds are at most the height actually lost, so the
    /// result never exceeds it.
    fn primary_shed(&self, rows: usize) -> usize {
        let lost = self.rows().saturating_sub(rows);
        if lost == 0 {
            return 0;
        }
        let cursor_row = if self.alt_active {
            self.inactive_cursor.row
        } else {
            self.cursor.row
        };
        let for_cursor = (cursor_row + 1).saturating_sub(rows);
        let for_content = Self::last_non_blank_row(&self.primary)
            .map_or(0, |last| (last + 1).saturating_sub(rows));
        for_cursor.max(for_content).min(lost)
    }

    /// How many rows the **alternate** grid must shed off its top to
    /// shrink to `rows`.
    ///
    /// Driven by the alternate cursor alone. There is nowhere to preserve
    /// alt content — the alternate screen has no scrollback — so shed rows
    /// are discarded and only the cursor's position matters.
    fn alt_shed(&self, rows: usize) -> usize {
        let lost = self.rows().saturating_sub(rows);
        if lost == 0 {
            return 0;
        }
        let cursor_row = if self.alt_active {
            self.cursor.row
        } else {
            self.inactive_cursor.row
        };
        (cursor_row + 1).saturating_sub(rows).min(lost)
    }

    /// Move a cursor along with the `shed` rows its grid lost off the top,
    /// then clamp it into the new bounds.
    fn follow_resize(cursor: &mut Cursor, shed: usize, rows: usize, cols: usize) {
        cursor.row = cursor.row.saturating_sub(shed).min(rows - 1);
        cursor.col = cursor.col.min(cols - 1);
    }

    fn grid_active_mut(&mut self) -> &mut Grid {
        if self.alt_active {
            &mut self.alt
        } else {
            &mut self.primary
        }
    }

    fn saved_slot_mut(&mut self) -> &mut Option<SavedCursor> {
        if self.alt_active {
            &mut self.saved_alt
        } else {
            &mut self.saved_primary
        }
    }

    /// Look up a row by absolute id across scrollback and the **primary**
    /// grid.
    ///
    /// Deliberately never consults the alternate grid: absolute ids belong
    /// to the primary screen's timeline, which the alternate screen does
    /// not advance. Resolving them against the active grid would make a
    /// full-screen application's frame buffer masquerade as the primary
    /// content that owns those ids.
    fn row_slice(&self, absolute_row: usize) -> Option<&[Cell]> {
        let evicted = self.scrollback.evicted();
        if absolute_row < evicted {
            return None; // evicted forever
        }
        let idx = absolute_row - evicted;
        if idx < self.scrollback.len() {
            return self.scrollback.row(idx);
        }
        self.primary.row(idx - self.scrollback.len())
    }

    /// Blank cell used to fill vacated cells: current pen background,
    /// everything else default (background-color-erase).
    fn blank_cell(&self) -> Cell {
        Cell::blank(CellAttrs {
            bg: self.pen.bg,
            ..CellAttrs::default()
        })
    }

    /// Resolve a 0-based row argument, honoring origin mode.
    fn resolve_row(&self, row: usize) -> usize {
        if self.origin_mode {
            (self.scroll_top + row).min(self.scroll_bottom)
        } else {
            row.min(self.rows() - 1)
        }
    }

    /// Move the cursor down one row, scrolling the region if it sits on the
    /// bottom margin (shared by LF and autowrap).
    fn scroll_or_step_down(&mut self) {
        if self.cursor.row == self.scroll_bottom {
            self.scroll_region_up(1);
        } else if self.cursor.row + 1 < self.rows() {
            self.cursor.row += 1;
        }
    }

    /// Scroll the region up `n` rows, pushing scrolled-out rows to
    /// scrollback iff on the primary screen with a full-screen region.
    fn scroll_region_up(&mut self, n: usize) {
        let (top, bottom) = (self.scroll_top, self.scroll_bottom);
        let height = bottom - top + 1;
        let n = n.min(height);
        if n == 0 {
            return;
        }
        if !self.alt_active && top == 0 && bottom == self.rows() - 1 {
            for r in top..top + n {
                if let Some(cells) = self.primary.row(r) {
                    self.scrollback.push_row_copy(cells);
                }
            }
        }
        let blank = self.blank_cell();
        self.grid_active_mut()
            .scroll_up_within(top, bottom, n, blank);
    }

    /// If the cell at `(row, col)` is half of a wide pair, blank both
    /// halves so no orphaned lead/trail survives an overwrite.
    fn clean_wide_at(&mut self, row: usize, col: usize) {
        let blank = self.blank_cell();
        let grid = self.grid_active_mut();
        match grid.cell(row, col).map(|c| c.width) {
            Some(CellWidth::WideLead) => {
                if let Some(c) = grid.cell_mut(row, col) {
                    *c = blank;
                }
                if let Some(c) = grid.cell_mut(row, col + 1) {
                    if c.width == CellWidth::WideTrail {
                        *c = blank;
                    }
                }
            }
            Some(CellWidth::WideTrail) => {
                if let Some(c) = grid.cell_mut(row, col) {
                    *c = blank;
                }
                if col > 0 {
                    if let Some(c) = grid.cell_mut(row, col - 1) {
                        if c.width == CellWidth::WideLead {
                            *c = blank;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
