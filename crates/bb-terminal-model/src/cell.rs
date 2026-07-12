//! Character cells and their graphics attributes.

/// A single character cell on the terminal grid.
///
/// A cell holds the printable character at that position plus the graphics
/// attributes ([`CellAttrs`]) that apply to it, and a [`CellWidth`] tag
/// used to represent East Asian wide characters that occupy two columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cell {
    /// The character displayed at this cell. `' '` for blank.
    pub ch: char,
    /// Foreground / background colors and style flags.
    pub attrs: CellAttrs,
    /// How many columns this cell participates in (single, or half of a
    /// wide-character pair).
    pub width: CellWidth,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            attrs: CellAttrs::default(),
            width: CellWidth::Single,
        }
    }
}

impl Cell {
    /// A blank (space) cell carrying the given attributes.
    ///
    /// This is what erase and scroll operations fill vacated cells with.
    pub fn blank(attrs: CellAttrs) -> Self {
        Self {
            ch: ' ',
            attrs,
            width: CellWidth::Single,
        }
    }
}

/// Column-width classification of a [`Cell`].
///
/// Wide (two-column) characters occupy two cells on the grid: the first
/// carries the character and is tagged [`CellWidth::WideLead`]; the second
/// is a blank spacer tagged [`CellWidth::WideTrail`]. Renderers should draw
/// the lead cell across both columns and skip trail cells.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CellWidth {
    /// An ordinary single-column cell.
    #[default]
    Single,
    /// First column of a wide (two-column) character; holds the character.
    WideLead,
    /// Second column of a wide character; a blank spacer.
    WideTrail,
}

/// Per-cell graphics attributes.
///
/// Flags are simple booleans for readability. If memory pressure becomes
/// a concern (terminals can hold millions of cells in scrollback), this
/// can be bit-packed without changing the public API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Cell foreground or background color.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Color {
    /// The terminal-default color (i.e. no SGR override applied).
    #[default]
    Default,
    /// One of the 256 indexed ANSI colors.
    Indexed(u8),
    /// True-color (24-bit) RGB.
    Rgb(u8, u8, u8),
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cell_default_is_blank_single_width_default_attrs() {
        let cell = Cell::default();
        assert_eq!(cell.ch, ' ');
        assert_eq!(cell.attrs, CellAttrs::default());
        assert_eq!(cell.width, CellWidth::Single);
    }

    #[test]
    fn cell_blank_carries_attrs() {
        let attrs = CellAttrs {
            bg: Color::Indexed(4),
            ..CellAttrs::default()
        };
        let cell = Cell::blank(attrs);
        assert_eq!(cell.ch, ' ');
        assert_eq!(cell.attrs.bg, Color::Indexed(4));
        assert_eq!(cell.width, CellWidth::Single);
    }

    #[test]
    fn color_default_is_terminal_default() {
        assert_eq!(Color::default(), Color::Default);
    }

    #[test]
    fn cell_width_default_is_single() {
        assert_eq!(CellWidth::default(), CellWidth::Single);
    }
}
