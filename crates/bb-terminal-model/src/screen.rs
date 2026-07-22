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
/// Cells vacated by erase, scroll, insert, and delete operations are filled
/// with a blank cell carrying the current pen's *background* color
/// (background-color-erase), with all other attributes reset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Screen {
    primary: Grid,
    alt: Grid,
    alt_active: bool,
    scrollback: Scrollback,
    cursor: Cursor,
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
    /// Dimensions are clamped to at least `1 × 1`.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_scrollback(rows, cols, Self::DEFAULT_SCROLLBACK_LINES)
    }

    /// Create a screen of `rows × cols` retaining at most `max_scrollback`
    /// lines of history. Dimensions are clamped to at least `1 × 1`.
    pub fn with_scrollback(rows: usize, cols: usize, max_scrollback: usize) -> Self {
        let rows = rows.max(1);
        let cols = cols.max(1);
        Self {
            primary: Grid::new(rows, cols),
            alt: Grid::new(rows, cols),
            alt_active: false,
            scrollback: Scrollback::new(max_scrollback),
            cursor: Cursor::default(),
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
            if let Some(cells) = self.grid_active_mut().row_mut(row) {
                let shift = width.min(cols - col);
                if let Some(span) = cells.get_mut(col..) {
                    span.rotate_right(shift);
                    if let Some(head) = span.get_mut(..shift) {
                        for c in head {
                            *c = blank;
                        }
                    }
                }
                // A wide pair split at the row end leaves an orphan lead.
                if let Some(last) = cells.last_mut() {
                    if last.width == CellWidth::WideLead {
                        *last = blank;
                    }
                }
            }
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
    /// the current screen contents are first pushed to scrollback (like
    /// modern terminals), so their absolute row ids remain addressable.
    /// The cursor does not move.
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
                    for r in 0..rows {
                        if let Some(cells) = self.primary.row(r) {
                            let owned = cells.to_vec();
                            self.scrollback.push_row(owned);
                        }
                    }
                }
                self.grid_active_mut().fill(blank);
            }
        }
    }

    /// Erase in line (EL) on the cursor's row. The cursor does not move.
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
        if let Some(cells) = self.grid_active_mut().row_mut(row) {
            if let Some(span) = cells.get_mut(start..end) {
                for c in span {
                    *c = blank;
                }
            }
        }
    }

    /// Erase `n` characters at and after the cursor on the current row
    /// (ECH). Cells are blanked in place; nothing shifts.
    pub fn erase_chars(&mut self, n: usize) {
        self.pending_wrap = false;
        let blank = self.blank_cell();
        let cols = self.cols();
        let row = self.cursor.row;
        let col = self.cursor.col.min(cols - 1);
        let end = col.saturating_add(n).min(cols);
        if let Some(cells) = self.grid_active_mut().row_mut(row) {
            if let Some(span) = cells.get_mut(col..end) {
                for c in span {
                    *c = blank;
                }
            }
        }
    }

    /// Insert `n` blank characters at the cursor (ICH), shifting the rest
    /// of the row right; cells pushed past the last column are lost.
    pub fn insert_chars(&mut self, n: usize) {
        self.pending_wrap = false;
        let blank = self.blank_cell();
        let cols = self.cols();
        let row = self.cursor.row;
        let col = self.cursor.col.min(cols - 1);
        self.clean_wide_at(row, col);
        let n = n.min(cols - col);
        if n == 0 {
            return;
        }
        if let Some(cells) = self.grid_active_mut().row_mut(row) {
            if let Some(span) = cells.get_mut(col..) {
                span.rotate_right(n);
                if let Some(head) = span.get_mut(..n) {
                    for c in head {
                        *c = blank;
                    }
                }
            }
        }
    }

    /// Delete `n` characters at the cursor (DCH), shifting the rest of the
    /// row left and blanking the vacated cells at the end.
    pub fn delete_chars(&mut self, n: usize) {
        self.pending_wrap = false;
        let blank = self.blank_cell();
        let cols = self.cols();
        let row = self.cursor.row;
        let col = self.cursor.col.min(cols - 1);
        self.clean_wide_at(row, col);
        let n = n.min(cols - col);
        if n == 0 {
            return;
        }
        if let Some(cells) = self.grid_active_mut().row_mut(row) {
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

    /// Switch to the alternate screen, clearing it first. No-op if already
    /// active. The alternate screen has no scrollback. Cursor position is
    /// preserved (callers implementing mode 1049 should bracket this with
    /// [`Screen::save_cursor`] / [`Screen::restore_cursor`]).
    pub fn enter_alt_screen(&mut self) {
        if self.alt_active {
            return;
        }
        self.alt_active = true;
        self.alt.clear();
        self.pending_wrap = false;
    }

    /// Switch back to the primary screen. No-op if already primary.
    pub fn exit_alt_screen(&mut self) {
        if !self.alt_active {
            return;
        }
        self.alt_active = false;
        self.pending_wrap = false;
    }

    // -------------------------------------------------------------------
    // Resize and reset
    // -------------------------------------------------------------------

    /// Resize both grids to `rows × cols` (clamped to at least `1 × 1`),
    /// **preserving on-screen content**.
    ///
    /// # Policy
    ///
    /// - **Overlap is kept.** A cell at `(row, col)` that exists in both the
    ///   old and the new dimensions keeps its character *and* its
    ///   attributes. Cells exposed by growth are blank with default attrs.
    /// - **No reflow.** Lines are neither rewrapped when the screen narrows
    ///   nor unwrapped when it widens; columns past the new width are
    ///   dropped. (Reflow is a much larger change and is out of scope.) A
    ///   wide-character pair split by the new right edge has its orphaned
    ///   lead blanked so no half character survives.
    /// - **Shrinking height keeps the bottom.** The cursor and the most
    ///   recent output live near the bottom, so the rows that go are taken
    ///   off the *top*: exactly as many as are needed to keep the cursor
    ///   row on screen (`cursor_row + 1 - rows`, never more than the height
    ///   lost). Those rows leave the primary grid the same way they would
    ///   if the shell had scrolled them off — **pushed into scrollback** —
    ///   so they are still addressable at *the same absolute row ids*. The
    ///   alternate screen has no scrollback, so its shed rows are simply
    ///   discarded. Any remaining height reduction comes off the bottom,
    ///   which is where a terminal's blank/stale rows are.
    /// - **Absolute rows stay coherent.** Every preserved row keeps its
    ///   absolute id (rows shed off the top move into scrollback, which
    ///   grows by exactly the same amount), and so does
    ///   [`Screen::cursor_absolute_row`]. Scrollback itself is untouched —
    ///   its rows keep the width they had when they were pushed.
    /// - The cursor and both saved cursors follow their content and are
    ///   clamped into the new bounds, the scroll region resets to the full
    ///   screen, tab stops are rebuilt every 8 columns, and the
    ///   pending-wrap flag is cleared (all standard on resize).
    ///
    /// Both grids are resized, each preserving its own contents. The single
    /// shared cursor row decides the shed count for both.
    pub fn resize(&mut self, rows: usize, cols: usize) {
        let rows = rows.max(1);
        let cols = cols.max(1);
        let old_rows = self.rows();

        // Rows that must leave the top so the cursor stays on screen. When
        // the cursor already fits, this is 0 and the height loss is taken
        // off the bottom instead.
        let lost = old_rows.saturating_sub(rows);
        let shed = (self.cursor.row + 1).saturating_sub(rows).min(lost);
        if shed > 0 {
            // Primary rows leaving the top keep their absolute ids by
            // moving into scrollback; the alternate screen has none.
            for r in 0..shed {
                if let Some(cells) = self.primary.row(r) {
                    let owned = cells.to_vec();
                    self.scrollback.push_row(owned);
                }
            }
            let blank = Cell::default();
            self.primary.scroll_up_within(0, old_rows - 1, shed, blank);
            self.alt.scroll_up_within(0, old_rows - 1, shed, blank);
        }

        let blank = Cell::default();
        self.primary.resize_preserving(rows, cols, blank);
        self.alt.resize_preserving(rows, cols, blank);
        Self::heal_wide_edge(&mut self.primary, blank);
        Self::heal_wide_edge(&mut self.alt, blank);

        self.scroll_top = 0;
        self.scroll_bottom = rows - 1;
        self.tabs = Self::default_tabs(cols);
        self.cursor.row = self.cursor.row.saturating_sub(shed).min(rows - 1);
        self.cursor.col = self.cursor.col.min(cols - 1);
        for saved in [&mut self.saved_primary, &mut self.saved_alt]
            .into_iter()
            .flatten()
        {
            saved.cursor.row = saved.cursor.row.saturating_sub(shed).min(rows - 1);
            saved.cursor.col = saved.cursor.col.min(cols - 1);
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
    pub fn viewport_rows(&self, vp: Viewport) -> Vec<&[Cell]> {
        let mut out = Vec::with_capacity(vp.height);
        for abs in vp.top..vp.top.saturating_add(vp.height) {
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

    /// Blank any [`CellWidth::WideLead`] left stranded in the last column.
    ///
    /// Narrowing a grid without reflow can cut a wide pair in half; the
    /// trail cell falls off the edge and the lead would otherwise claim a
    /// column it no longer has.
    fn heal_wide_edge(grid: &mut Grid, blank: Cell) {
        let Some(last) = grid.cols().checked_sub(1) else {
            return;
        };
        for r in 0..grid.rows() {
            if let Some(cell) = grid.cell_mut(r, last) {
                if cell.width == CellWidth::WideLead {
                    *cell = blank;
                }
            }
        }
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
                    let owned = cells.to_vec();
                    self.scrollback.push_row(owned);
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
