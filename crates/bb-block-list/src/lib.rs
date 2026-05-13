//! Warp-style command-output blocks: lifecycle, identity, search, navigation.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-block-list` is the structural layer on top of [`bb-terminal-model`]:
//! it groups raw terminal output into command-bounded "blocks" (the
//! Warp-style UX where each shell command and its output is selectable as
//! a unit). It owns:
//!
//! - Block identity & lifecycle (started → running → finished)
//! - Detection of command boundaries (OSC 133 prompt protocol +
//!   heuristic fallbacks)
//! - Block-level metadata: command line, exit code, duration, working
//!   directory
//! - Search across the block list
//! - Cursor navigation by block (next / previous / jump-to-tagged)
//!
//! # Status
//!
//! Scaffold. Real implementation lands in subsequent PRs.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-terminal-model`]: https://docs.rs/bb-terminal-model

#![forbid(unsafe_code)]
