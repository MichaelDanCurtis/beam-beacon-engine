//! Cursor position and shape.

/// Position and visual style of the terminal cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cursor {
    /// Row (0-indexed from the top of the grid).
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CursorShape {
    /// Solid block covering the cell.
    #[default]
    Block,
    /// Thin underline along the bottom of the cell.
    Underline,
    /// Thin vertical bar at the left of the cell.
    Bar,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cursor_defaults_to_block_visible_origin() {
        let c = Cursor::default();
        assert_eq!(c.row, 0);
        assert_eq!(c.col, 0);
        assert_eq!(c.shape, CursorShape::Block);
        assert!(c.visible);
    }
}
