//! Terminal state: grid, cells, attributes, cursor, viewport.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-terminal-model` owns the canonical in-memory representation of a
//! terminal screen: row × col grid of cells, per-cell graphics
//! attributes, cursor position and shape, and a windowed [`Viewport`]
//! projection.
//!
//! It is a *pure data* crate. Parsing escape sequences belongs in
//! [`bb-vt-parser`]; surfacing visible blocks of output belongs in
//! [`bb-block-list`]. Both consume this crate's types.
//!
//! # Status
//!
//! **Strawman type surface — design discussion welcome.** Names and
//! shapes here mirror what you'd find in `vt100`, `alacritty`, and
//! similar libraries. The API will evolve as Beam and Beacon start
//! consuming it. See [`docs/MIGRATION-CHECKLIST.md`] Phase 1 for the
//! plan.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-vt-parser`]: https://docs.rs/bb-vt-parser
//! [`bb-block-list`]: https://docs.rs/bb-block-list
//! [`docs/MIGRATION-CHECKLIST.md`]: https://github.com/MichaelDanCurtis/beam-beacon-engine/blob/main/docs/MIGRATION-CHECKLIST.md

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// -----------------------------------------------------------------------------
// Cell
// -----------------------------------------------------------------------------

/// A single character cell on the terminal grid.
///
/// A cell holds the printable character at that position plus the graphics
/// attributes ([`CellAttrs`]) that apply to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    /// The character displayed at this cell. `' '` for blank.
    pub ch: char,
    /// Foreground / background colors and style flags.
    pub attrs: CellAttrs,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            attrs: CellAttrs::default(),
        }
    }
}

// -----------------------------------------------------------------------------
// CellAttrs
// -----------------------------------------------------------------------------

/// Per-cell graphics attributes.
///
/// Flags are simple booleans for readability. If memory pressure becomes
/// a concern (terminals can hold millions of cells in scrollback), this
/// can be bit-packed without changing the public API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CellAttrs {
    /// Foreground color.
    pub fg: Color,
    /// Background color.
    pub bg: Color,
    /// Bold (SGR 1).
    pub bold: bool,
    /// Italic (SGR 3).
    pub italic: bool,
    /// Underline (SGR 4).
    pub underline: bool,
    /// Reverse video (SGR 7).
    pub reverse: bool,
    /// Strikethrough (SGR 9).
    pub strikethrough: bool,
    /// Dim / faint (SGR 2).
    pub dim: bool,
    /// Blink (SGR 5).
    pub blink: bool,
    /// Hidden (SGR 8).
    pub hidden: bool,
}

// -----------------------------------------------------------------------------
// Color
// -----------------------------------------------------------------------------

/// Cell foreground or background color.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Color {
    /// The terminal-default color (i.e. no SGR override applied).
    #[default]
    Default,
    /// One of the 256 indexed ANSI colors.
    Indexed(u8),
    /// True-color (24-bit) RGB.
    Rgb(u8, u8, u8),
}

// -----------------------------------------------------------------------------
// Cursor
// -----------------------------------------------------------------------------

/// Position and visual style of the terminal cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cursor {
    /// Row (0-indexed from the top of the [`Grid`]).
    pub row: usize,
    /// Column (0-indexed from the left).
    pub col: usize,
    /// Visual shape of the cursor.
    pub shape: CursorShape,
    /// Whether the cursor should be drawn.
    pub visible: bool,
}

impl Default for Cursor {
    fn default() -> Self {
        Self {
            row: 0,
            col: 0,
            shape: CursorShape::Block,
            visible: true,
        }
    }
}

/// Visual style of the cursor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CursorShape {
    /// Solid block covering the cell.
    #[default]
    Block,
    /// Thin underline along the bottom of the cell.
    Underline,
    /// Thin vertical bar at the left of the cell.
    Bar,
}

// -----------------------------------------------------------------------------
// Grid
// -----------------------------------------------------------------------------

/// A rectangular grid of [`Cell`]s.
///
/// The grid stores cells in row-major order. Out-of-bounds access returns
/// `None` rather than panicking — this is a deliberate choice so callers
/// can defensively check shape mismatches without unwrapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    rows: usize,
    cols: usize,
    cells: Vec<Cell>,
}

impl Grid {
    /// Create a fresh blank grid of `rows × cols` cells.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            cells: vec![Cell::default(); rows * cols],
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
            Some(&self.cells[row * self.cols + col])
        } else {
            None
        }
    }

    /// Mutate the cell at `(row, col)`. Returns `None` if out of bounds.
    pub fn cell_mut(&mut self, row: usize, col: usize) -> Option<&mut Cell> {
        if row < self.rows && col < self.cols {
            let idx = row * self.cols + col;
            Some(&mut self.cells[idx])
        } else {
            None
        }
    }

    /// Borrow a row's cells in column order. Returns `None` if out of bounds.
    pub fn row(&self, row: usize) -> Option<&[Cell]> {
        if row < self.rows {
            let start = row * self.cols;
            Some(&self.cells[start..start + self.cols])
        } else {
            None
        }
    }

    /// Reset every cell to the default (blank space, default attrs).
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            *cell = Cell::default();
        }
    }

    /// Resize the grid. New cells are blank; truncated cells are discarded.
    ///
    /// This is a destructive operation — callers that need to preserve
    /// content across a resize should snapshot first, then re-render. A
    /// reflow-preserving resize is a future addition.
    pub fn resize(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        self.cells.clear();
        self.cells.resize(rows * cols, Cell::default());
    }
}

// -----------------------------------------------------------------------------
// Viewport
// -----------------------------------------------------------------------------

/// A windowed view into a [`Grid`] (or its scrollback).
///
/// Useful for UI layers that show only a portion of the buffer (e.g. when
/// the user has scrolled back through history).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Viewport {
    /// First visible row (0-indexed from the top of the buffer).
    pub top: usize,
    /// Number of visible rows.
    pub height: usize,
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cell_default_is_blank_with_default_attrs() {
        let cell = Cell::default();
        assert_eq!(cell.ch, ' ');
        assert_eq!(cell.attrs, CellAttrs::default());
    }

    #[test]
    fn grid_dimensions_are_preserved() {
        let g = Grid::new(2, 3);
        assert_eq!(g.dimensions(), (2, 3));
        assert_eq!(g.len(), 6);
        assert!(!g.is_empty());
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
        let g = Grid::new(2, 2);
        assert!(g.cell(2, 0).is_none());
        assert!(g.cell(0, 2).is_none());
        assert!(g.cell(99, 99).is_none());
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
    fn grid_resize_is_destructive() {
        let mut g = Grid::new(2, 2);
        g.cell_mut(0, 0).unwrap().ch = 'A';
        g.resize(3, 3);
        assert_eq!(g.dimensions(), (3, 3));
        assert_eq!(g.cell(0, 0).unwrap().ch, ' ');
    }

    #[test]
    fn cursor_defaults_to_block_visible_origin() {
        let c = Cursor::default();
        assert_eq!(c.row, 0);
        assert_eq!(c.col, 0);
        assert_eq!(c.shape, CursorShape::Block);
        assert!(c.visible);
    }

    #[test]
    fn color_default_is_terminal_default() {
        assert_eq!(Color::default(), Color::Default);
    }
}
