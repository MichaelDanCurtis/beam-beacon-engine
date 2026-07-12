//! Terminal state: grid, cells, attributes, cursor, scrollback, viewport.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-terminal-model` owns the canonical in-memory representation of a
//! terminal: a row × col [`Grid`] of [`Cell`]s with per-cell graphics
//! attributes ([`CellAttrs`]), the [`Cursor`], bounded [`Scrollback`]
//! history, a windowed [`Viewport`] projection, and the [`Screen`]
//! aggregate that ties them together with the semantic operations a VT
//! parser drives (printing, cursor movement, erase/insert/delete, scroll
//! regions, tab stops, the alternate screen, and DEC/ANSI modes).
//!
//! It is a *pure data* crate. Parsing escape sequences belongs in
//! [`bb-vt-parser`]; surfacing visible blocks of output belongs in
//! [`bb-block-list`]. Both consume this crate's types.
//!
//! # Absolute row ids
//!
//! Rows never change identity: grid row `r` has absolute id
//! `scrollback.evicted() + scrollback.len() + r`, and once a row scrolls
//! into (and eventually out of) scrollback its id is never reused. See
//! [`Screen::absolute_row`], [`Screen::line`], and [`Screen::row_text`].
//!
//! # Feature flags
//!
//! - `serde` (off by default): derives `Serialize`/`Deserialize` on the
//!   pure-data types ([`Cell`], [`CellAttrs`], [`CellWidth`], [`Color`],
//!   [`Cursor`], [`CursorShape`], [`Grid`], [`Viewport`], [`Scrollback`],
//!   [`EraseMode`]). Deserialization does not re-validate internal
//!   invariants (e.g. that a [`Grid`]'s cell count matches its
//!   dimensions); only feed it data serialized from these same types.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-vt-parser`]: https://docs.rs/bb-vt-parser
//! [`bb-block-list`]: https://docs.rs/bb-block-list

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod cell;
mod cursor;
mod grid;
mod screen;
mod scrollback;

pub use cell::{Cell, CellAttrs, CellWidth, Color};
pub use cursor::{Cursor, CursorShape};
pub use grid::{Grid, Viewport};
pub use screen::{EraseMode, Screen};
pub use scrollback::Scrollback;
