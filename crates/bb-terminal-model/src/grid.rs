//! The rectangular cell grid and viewport window.

use crate::cell::Cell;

/// A rectangular grid of [`Cell`]s.
///
/// The grid stores cells in row-major order. Out-of-bounds access returns
/// `None` rather than panicking — this is a deliberate choice so callers
/// can defensively check shape mismatches without unwrapping.
///
/// # Geometry limits
///
/// Dimensions come from the host (a window resize, a `TIOCSWINSZ`), so
/// they are *untrusted input*. Every constructor and resize **clamps**
/// them to [`Grid::MAX_ROWS`] × [`Grid::MAX_COLS`] instead of trusting
/// them: an absurd request yields a capped grid rather than an integer
/// overflow or an allocation the machine cannot serve. There is no lower
/// clamp — a zero-sized grid is legal here (but not for
/// [`crate::Screen`], which clamps up to `1 × 1`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Grid {
    rows: usize,
    cols: usize,
    cells: Vec<Cell>,
}

impl Grid {
    /// Largest number of rows a grid will allocate. Larger requests are
    /// clamped to this. Chosen far above any real terminal geometry while
    /// keeping `MAX_ROWS * MAX_COLS` a comfortably allocatable cell count.
    pub const MAX_ROWS: usize = 4_096;

    /// Largest number of columns a grid will allocate. Larger requests are
    /// clamped to this. See [`Grid::MAX_ROWS`].
    pub const MAX_COLS: usize = 4_096;

    /// Clamp a requested geometry to what this crate will allocate,
    /// returning `(rows, cols, cell_count)`.
    ///
    /// The per-dimension cap keeps the product far inside `usize`, so the
    /// `checked_mul` can never actually fail; it is spelled out anyway so
    /// that raising the constants can only ever degrade to an empty grid,
    /// never to an overflow panic.
    fn clamped(rows: usize, cols: usize) -> (usize, usize, usize) {
        let rows = rows.min(Self::MAX_ROWS);
        let cols = cols.min(Self::MAX_COLS);
        match rows.checked_mul(cols) {
            Some(cells) => (rows, cols, cells),
            None => (0, 0, 0),
        }
    }

    /// Create a fresh blank grid of `rows × cols` cells.
    ///
    /// Dimensions are clamped to [`Grid::MAX_ROWS`] × [`Grid::MAX_COLS`].
    pub fn new(rows: usize, cols: usize) -> Self {
        let (rows, cols, cells) = Self::clamped(rows, cols);
        Self {
            rows,
            cols,
            cells: vec![Cell::default(); cells],
        }
    }

    /// Number of rows in the grid.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns in the grid.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Convenience accessor for `(rows, cols)`.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Total number of cells (`rows * cols`).
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the grid has zero cells.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Read the cell at `(row, col)`. Returns `None` if out of bounds.
    pub fn cell(&self, row: usize, col: usize) -> Option<&Cell> {
        if row < self.rows && col < self.cols {
            self.cells.get(row * self.cols + col)
        } else {
            None
        }
    }

    /// Mutate the cell at `(row, col)`. Returns `None` if out of bounds.
    pub fn cell_mut(&mut self, row: usize, col: usize) -> Option<&mut Cell> {
        if row < self.rows && col < self.cols {
            let idx = row * self.cols + col;
            self.cells.get_mut(idx)
        } else {
            None
        }
    }

    /// Borrow a row's cells in column order. Returns `None` if out of bounds.
    pub fn row(&self, row: usize) -> Option<&[Cell]> {
        if row < self.rows {
            let start = row * self.cols;
            self.cells.get(start..start + self.cols)
        } else {
            None
        }
    }

    /// Mutably borrow a row's cells in column order. Returns `None` if out
    /// of bounds.
    pub fn row_mut(&mut self, row: usize) -> Option<&mut [Cell]> {
        if row < self.rows {
            let start = row * self.cols;
            self.cells.get_mut(start..start + self.cols)
        } else {
            None
        }
    }

    /// Reset every cell to the default (blank space, default attrs).
    pub fn clear(&mut self) {
        self.fill(Cell::default());
    }

    /// Set every cell in the grid to `cell`.
    pub fn fill(&mut self, cell: Cell) {
        for c in &mut self.cells {
            *c = cell;
        }
    }

    /// Set every cell of row `row` to `cell`. Out-of-bounds rows are ignored.
    pub fn fill_row(&mut self, row: usize, cell: Cell) {
        if let Some(cells) = self.row_mut(row) {
            for c in cells {
                *c = cell;
            }
        }
    }

    /// Scroll the inclusive row range `[top, bottom]` up by `n` rows.
    ///
    /// Rows scrolled past `top` are discarded (callers that want them —
    /// e.g. for scrollback — must copy them out first). The `n` rows
    /// vacated at the bottom of the range are filled with `blank`.
    /// Invalid ranges (`top > bottom` or `bottom >= rows`) are ignored.
    pub fn scroll_up_within(&mut self, top: usize, bottom: usize, n: usize, blank: Cell) {
        if top > bottom || bottom >= self.rows || n == 0 || self.cols == 0 {
            return;
        }
        let height = bottom - top + 1;
        let n = n.min(height);
        let start = top * self.cols;
        let end = (bottom + 1) * self.cols;
        if let Some(span) = self.cells.get_mut(start..end) {
            span.rotate_left(n * self.cols);
            let blank_from = span.len() - n * self.cols;
            if let Some(tail) = span.get_mut(blank_from..) {
                for c in tail {
                    *c = blank;
                }
            }
        }
    }

    /// Scroll the inclusive row range `[top, bottom]` down by `n` rows.
    ///
    /// Rows scrolled past `bottom` are discarded. The `n` rows vacated at
    /// the top of the range are filled with `blank`. Invalid ranges are
    /// ignored.
    pub fn scroll_down_within(&mut self, top: usize, bottom: usize, n: usize, blank: Cell) {
        if top > bottom || bottom >= self.rows || n == 0 || self.cols == 0 {
            return;
        }
        let height = bottom - top + 1;
        let n = n.min(height);
        let start = top * self.cols;
        let end = (bottom + 1) * self.cols;
        if let Some(span) = self.cells.get_mut(start..end) {
            span.rotate_right(n * self.cols);
            if let Some(head) = span.get_mut(..n * self.cols) {
                for c in head {
                    *c = blank;
                }
            }
        }
    }

    /// Resize the grid, **discarding all contents**. Every cell in the new
    /// grid is blank.
    ///
    /// This is the low-level destructive primitive, kept for callers that
    /// genuinely want a fresh buffer of a new size. Callers that want a
    /// window resize to behave like a real terminal — on-screen text
    /// survives — want [`Grid::resize_preserving`] instead;
    /// [`crate::Screen::resize`] uses that one.
    ///
    /// Dimensions are clamped to [`Grid::MAX_ROWS`] × [`Grid::MAX_COLS`].
    pub fn resize(&mut self, rows: usize, cols: usize) {
        let (rows, cols, cells) = Self::clamped(rows, cols);
        self.rows = rows;
        self.cols = cols;
        self.cells.clear();
        self.cells.resize(cells, Cell::default());
    }

    /// Resize the grid, **preserving the overlapping region**.
    ///
    /// The grid is anchored at its top-left corner: a cell at `(row, col)`
    /// that exists in both the old and the new dimensions keeps its
    /// character, attributes, and [`crate::CellWidth`] tag. Rows past the
    /// new height and columns past the new width are dropped, and cells
    /// exposed by growth are set to `blank`.
    ///
    /// There is **no reflow**: lines are neither rewrapped when the grid
    /// narrows nor unwrapped when it widens. Narrowing can therefore split
    /// a wide-character pair across the new right edge; callers that model
    /// wide characters should blank an orphaned [`crate::CellWidth::WideLead`]
    /// in the last column afterwards (this is what [`crate::Screen::resize`]
    /// does).
    ///
    /// Because the anchor is the *top* row, shrinking the height keeps the
    /// top of the grid. Callers that want to keep the *bottom* (where a
    /// terminal's cursor and most recent output live) should first scroll
    /// the rows they want to shed off the top with
    /// [`Grid::scroll_up_within`] — that also gives them a chance to copy
    /// those rows into scrollback.
    ///
    /// Dimensions are clamped to [`Grid::MAX_ROWS`] × [`Grid::MAX_COLS`].
    pub fn resize_preserving(&mut self, rows: usize, cols: usize, blank: Cell) {
        let (rows, cols, len) = Self::clamped(rows, cols);
        if self.rows == rows && self.cols == cols {
            return;
        }
        let mut cells = vec![blank; len];
        let copy_rows = self.rows.min(rows);
        let copy_cols = self.cols.min(cols);
        for r in 0..copy_rows {
            let src = self.cells.get(r * self.cols..r * self.cols + copy_cols);
            let dst = cells.get_mut(r * cols..r * cols + copy_cols);
            if let (Some(src), Some(dst)) = (src, dst) {
                dst.copy_from_slice(src);
            }
        }
        self.rows = rows;
        self.cols = cols;
        self.cells = cells;
    }
}

/// A windowed view into the terminal buffer (scrollback plus grid).
///
/// Useful for UI layers that show only a portion of the buffer (e.g. when
/// the user has scrolled back through history). `top` is expressed in
/// *absolute row* coordinates — see [`crate::Screen::absolute_row`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Viewport {
    /// First visible row, as an absolute row id (0-indexed from the first
    /// row ever produced, including rows since evicted).
    pub top: usize,
    /// Number of visible rows.
    pub height: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::cell::{CellAttrs, Color};

    fn grid_with_letters(rows: usize, cols: usize) -> Grid {
        // Row r is filled with the letter 'A' + r.
        let mut g = Grid::new(rows, cols);
        for r in 0..rows {
            let ch = char::from(b'A' + u8::try_from(r).unwrap());
            for c in 0..cols {
                g.cell_mut(r, c).unwrap().ch = ch;
            }
        }
        g
    }

    #[test]
    fn grid_dimensions_are_preserved() {
        let g = Grid::new(2, 3);
        assert_eq!(g.dimensions(), (2, 3));
        assert_eq!(g.len(), 6);
        assert!(!g.is_empty());
    }

    #[test]
    fn grid_zero_sized_is_empty() {
        let g = Grid::new(0, 5);
        assert!(g.is_empty());
        assert!(g.row(0).is_none());
    }

    #[test]
    fn grid_default_cells_are_blank() {
        let g = Grid::new(2, 3);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(*g.cell(r, c).unwrap(), Cell::default());
            }
        }
    }

    #[test]
    fn grid_cell_round_trip_via_mut() {
        let mut g = Grid::new(4, 4);
        let c = g.cell_mut(1, 2).unwrap();
        c.ch = 'X';
        c.attrs.bold = true;
        c.attrs.fg = Color::Rgb(255, 0, 0);
        let read = g.cell(1, 2).unwrap();
        assert_eq!(read.ch, 'X');
        assert!(read.attrs.bold);
        assert_eq!(read.attrs.fg, Color::Rgb(255, 0, 0));
    }

    #[test]
    fn grid_out_of_bounds_returns_none() {
        let mut g = Grid::new(2, 2);
        assert!(g.cell(2, 0).is_none());
        assert!(g.cell(0, 2).is_none());
        assert!(g.cell(99, 99).is_none());
        assert!(g.cell_mut(2, 0).is_none());
        assert!(g.row_mut(2).is_none());
    }

    #[test]
    fn grid_row_slice_has_correct_length() {
        let g = Grid::new(3, 5);
        assert_eq!(g.row(0).unwrap().len(), 5);
        assert!(g.row(3).is_none());
    }

    #[test]
    fn grid_clear_resets_all_cells() {
        let mut g = Grid::new(2, 2);
        g.cell_mut(0, 0).unwrap().ch = 'A';
        g.cell_mut(1, 1).unwrap().ch = 'B';
        g.clear();
        assert_eq!(g.cell(0, 0).unwrap().ch, ' ');
        assert_eq!(g.cell(1, 1).unwrap().ch, ' ');
    }

    #[test]
    fn grid_fill_row_sets_one_row_only() {
        let mut g = Grid::new(2, 2);
        let blank = Cell::blank(CellAttrs {
            bg: Color::Indexed(1),
            ..CellAttrs::default()
        });
        g.fill_row(0, blank);
        assert_eq!(g.cell(0, 1).unwrap().attrs.bg, Color::Indexed(1));
        assert_eq!(g.cell(1, 1).unwrap().attrs.bg, Color::Default);
        g.fill_row(99, blank); // out of bounds: no panic
    }

    #[test]
    fn grid_resize_is_destructive() {
        let mut g = Grid::new(2, 2);
        g.cell_mut(0, 0).unwrap().ch = 'A';
        g.resize(3, 3);
        assert_eq!(g.dimensions(), (3, 3));
        assert_eq!(g.cell(0, 0).unwrap().ch, ' ');
    }

    #[test]
    fn grid_resize_preserving_keeps_the_overlapping_region() {
        let mut g = grid_with_letters(3, 3);
        g.cell_mut(1, 1).unwrap().attrs.bold = true;
        g.resize_preserving(2, 2, Cell::default());
        assert_eq!(g.dimensions(), (2, 2));
        assert_eq!(g.cell(0, 0).unwrap().ch, 'A');
        assert_eq!(g.cell(0, 1).unwrap().ch, 'A');
        assert_eq!(g.cell(1, 0).unwrap().ch, 'B');
        assert!(g.cell(1, 1).unwrap().attrs.bold); // attributes survive
        assert!(g.cell(2, 0).is_none()); // row 'C' dropped
    }

    #[test]
    fn grid_resize_preserving_pads_growth_with_the_blank_cell() {
        let mut g = grid_with_letters(1, 2);
        let blank = Cell::blank(CellAttrs {
            bg: Color::Indexed(5),
            ..CellAttrs::default()
        });
        g.resize_preserving(2, 3, blank);
        assert_eq!(g.dimensions(), (2, 3));
        assert_eq!(g.cell(0, 1).unwrap().ch, 'A'); // old content kept
        assert_eq!(*g.cell(0, 2).unwrap(), blank); // new column
        assert_eq!(*g.cell(1, 0).unwrap(), blank); // new row
    }

    #[test]
    fn grid_resize_preserving_is_a_noop_at_the_same_dimensions() {
        let mut g = grid_with_letters(2, 2);
        let before = g.clone();
        g.resize_preserving(2, 2, Cell::default());
        assert_eq!(g, before);
    }

    #[test]
    fn grid_clamps_extreme_dimensions_instead_of_overflowing() {
        // Host-supplied geometry is untrusted: it must cap, never panic.
        let g = Grid::new(usize::MAX, 2);
        assert_eq!(g.dimensions(), (Grid::MAX_ROWS, 2));
        assert_eq!(g.len(), Grid::MAX_ROWS * 2);
        let g = Grid::new(2, usize::MAX);
        assert_eq!(g.dimensions(), (2, Grid::MAX_COLS));
        let g = Grid::new(usize::MAX, usize::MAX);
        assert_eq!(g.dimensions(), (Grid::MAX_ROWS, Grid::MAX_COLS));
        assert_eq!(g.len(), Grid::MAX_ROWS * Grid::MAX_COLS);
    }

    #[test]
    fn grid_resize_clamps_extreme_dimensions() {
        let mut g = Grid::new(2, 2);
        g.resize(usize::MAX, 2);
        assert_eq!(g.dimensions(), (Grid::MAX_ROWS, 2));

        let mut g = grid_with_letters(2, 2);
        g.resize_preserving(usize::MAX, 3, Cell::default());
        assert_eq!(g.dimensions(), (Grid::MAX_ROWS, 3));
        assert_eq!(g.cell(0, 0).unwrap().ch, 'A'); // content still preserved
        assert_eq!(g.cell(Grid::MAX_ROWS - 1, 0).unwrap().ch, ' ');
    }

    #[test]
    fn scroll_up_within_shifts_and_blanks() {
        let mut g = grid_with_letters(4, 2);
        g.scroll_up_within(1, 3, 1, Cell::default());
        assert_eq!(g.cell(0, 0).unwrap().ch, 'A'); // outside region, untouched
        assert_eq!(g.cell(1, 0).unwrap().ch, 'C');
        assert_eq!(g.cell(2, 0).unwrap().ch, 'D');
        assert_eq!(g.cell(3, 0).unwrap().ch, ' ');
    }

    #[test]
    fn scroll_down_within_shifts_and_blanks() {
        let mut g = grid_with_letters(4, 2);
        g.scroll_down_within(0, 2, 1, Cell::default());
        assert_eq!(g.cell(0, 0).unwrap().ch, ' ');
        assert_eq!(g.cell(1, 0).unwrap().ch, 'A');
        assert_eq!(g.cell(2, 0).unwrap().ch, 'B');
        assert_eq!(g.cell(3, 0).unwrap().ch, 'D'); // outside region, untouched
    }

    #[test]
    fn scroll_within_clamps_n_to_region_height() {
        let mut g = grid_with_letters(3, 1);
        g.scroll_up_within(0, 2, 99, Cell::default());
        for r in 0..3 {
            assert_eq!(g.cell(r, 0).unwrap().ch, ' ');
        }
    }

    #[test]
    fn scroll_within_ignores_invalid_ranges() {
        let mut g = grid_with_letters(3, 1);
        let before = g.clone();
        g.scroll_up_within(2, 1, 1, Cell::default()); // top > bottom
        g.scroll_up_within(0, 3, 1, Cell::default()); // bottom out of range
        g.scroll_down_within(2, 1, 1, Cell::default());
        g.scroll_up_within(0, 2, 0, Cell::default()); // n == 0
        assert_eq!(g, before);
    }
}
