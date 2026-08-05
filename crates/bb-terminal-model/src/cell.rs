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
    /// Style flags, bit-packed.
    ///
    /// Packing the eight style booleans into one `u16` (instead of eight
    /// `bool` fields) shrinks [`Cell`] from 24 to 16 bytes. Scrollback is
    /// the engine's dominant allocation — cells × retained rows — so this
    /// is a third of every session's history memory. Read and write the
    /// flags through the accessor pairs ([`CellAttrs::bold`] /
    /// [`CellAttrs::set_bold`], …).
    pub(crate) flags: u16,
}

impl CellAttrs {
    pub(crate) const BOLD: u16 = 1 << 0;
    pub(crate) const ITALIC: u16 = 1 << 1;
    pub(crate) const UNDERLINE: u16 = 1 << 2;
    pub(crate) const REVERSE: u16 = 1 << 3;
    pub(crate) const STRIKETHROUGH: u16 = 1 << 4;
    pub(crate) const DIM: u16 = 1 << 5;
    pub(crate) const BLINK: u16 = 1 << 6;
    pub(crate) const HIDDEN: u16 = 1 << 7;

    #[inline]
    fn get(&self, mask: u16) -> bool {
        self.flags & mask != 0
    }

    #[inline]
    fn set(&mut self, mask: u16, on: bool) {
        if on {
            self.flags |= mask;
        } else {
            self.flags &= !mask;
        }
    }

    /// Bold (SGR 1).
    pub fn bold(&self) -> bool {
        self.get(Self::BOLD)
    }

    /// Set bold (SGR 1 / part of SGR 22).
    pub fn set_bold(&mut self, on: bool) {
        self.set(Self::BOLD, on);
    }

    /// Italic (SGR 3).
    pub fn italic(&self) -> bool {
        self.get(Self::ITALIC)
    }

    /// Set italic (SGR 3 / SGR 23).
    pub fn set_italic(&mut self, on: bool) {
        self.set(Self::ITALIC, on);
    }

    /// Underline (SGR 4).
    pub fn underline(&self) -> bool {
        self.get(Self::UNDERLINE)
    }

    /// Set underline (SGR 4 / SGR 24).
    pub fn set_underline(&mut self, on: bool) {
        self.set(Self::UNDERLINE, on);
    }

    /// Reverse video (SGR 7).
    pub fn reverse(&self) -> bool {
        self.get(Self::REVERSE)
    }

    /// Set reverse video (SGR 7 / SGR 27).
    pub fn set_reverse(&mut self, on: bool) {
        self.set(Self::REVERSE, on);
    }

    /// Strikethrough (SGR 9).
    pub fn strikethrough(&self) -> bool {
        self.get(Self::STRIKETHROUGH)
    }

    /// Set strikethrough (SGR 9 / SGR 29).
    pub fn set_strikethrough(&mut self, on: bool) {
        self.set(Self::STRIKETHROUGH, on);
    }

    /// Dim / faint (SGR 2).
    pub fn dim(&self) -> bool {
        self.get(Self::DIM)
    }

    /// Set dim / faint (SGR 2 / part of SGR 22).
    pub fn set_dim(&mut self, on: bool) {
        self.set(Self::DIM, on);
    }

    /// Blink (SGR 5).
    pub fn blink(&self) -> bool {
        self.get(Self::BLINK)
    }

    /// Set blink (SGR 5/6 / SGR 25).
    pub fn set_blink(&mut self, on: bool) {
        self.set(Self::BLINK, on);
    }

    /// Hidden (SGR 8).
    pub fn hidden(&self) -> bool {
        self.get(Self::HIDDEN)
    }

    /// Set hidden (SGR 8 / SGR 28).
    pub fn set_hidden(&mut self, on: bool) {
        self.set(Self::HIDDEN, on);
    }
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
    fn cell_stays_16_bytes() {
        // Scrollback memory is cells x retained rows, so Cell's size is a
        // budget, not an accident: 10k lines x 80 cols is 12.3 MB at 16
        // bytes and 18.4 MB at the 24 bytes the unpacked layout cost.
        // Growing Cell is allowed, but must be a deliberate decision.
        assert_eq!(std::mem::size_of::<Cell>(), 16);
        assert_eq!(std::mem::size_of::<CellAttrs>(), 10);
    }

    #[test]
    fn cell_width_default_is_single() {
        assert_eq!(CellWidth::default(), CellWidth::Single);
    }
}
