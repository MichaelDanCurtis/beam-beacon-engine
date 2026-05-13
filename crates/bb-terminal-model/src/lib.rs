//! Terminal state: grid, cells, attributes, cursor, viewport.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-terminal-model` owns the canonical in-memory representation of a
//! terminal screen: row × col grid of cells, per-cell graphics
//! attributes, cursor position and shape, scrollback ring, and the
//! viewport projection into the visible region.
//!
//! It is a *pure data* crate. Parsing escape sequences belongs in
//! [`bb-vt-parser`]; surfacing visible blocks of output belongs in
//! [`bb-block-list`]. Both consume this crate's types.
//!
//! # Status
//!
//! Scaffold. Real implementation lands in subsequent PRs as code is
//! extracted from Beam / Beacon.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-vt-parser`]: https://docs.rs/bb-vt-parser
//! [`bb-block-list`]: https://docs.rs/bb-block-list

#![forbid(unsafe_code)]
