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
//! - Block identity ([`BlockId`]) & lifecycle ([`BlockLifecycle`]:
//!   prompt → composing → running → finished)
//! - Command boundaries driven by the OSC 133 prompt protocol
//!   ([`PromptMarker`]), with a heuristic single-block fallback when no
//!   shell integration is present
//! - Block-level metadata: command line, working directory, exit code
//! - Search across the block list, over metadata and over output text
//! - Navigation by block (first / last / next / previous / at-row)
//!
//! # Layering
//!
//! This crate depends **only** on [`bb-terminal-model`] — deliberately not
//! on the VT parser. It defines its own input marker enum
//! ([`PromptMarker`]); the host's session glue observes OSC 133 (and OSC 7
//! for the working directory) in the parser and translates them into calls
//! on [`BlockList`]. Blocks address rows by the screen's *absolute row
//! ids*, which stay stable even after rows scroll into (and out of)
//! scrollback.
//!
//! # Driving a [`BlockList`]
//!
//! ```
//! use bb_block_list::{BlockLifecycle, BlockList, PromptMarker};
//!
//! let mut blocks = BlockList::new();
//! // OSC 133;A — shell draws a prompt on absolute row 0.
//! blocks.on_marker(PromptMarker::PromptStart, 0);
//! // OSC 133;B — user types a command.
//! blocks.on_marker(PromptMarker::CommandStart, 0);
//! blocks.set_command("cargo test");
//! blocks.set_cwd("/repo");
//! // OSC 133;C — command starts executing.
//! blocks.on_marker(PromptMarker::CommandExecuted, 1);
//! // Output grows; the host reports row progress after each parser feed.
//! blocks.observe_row(7);
//! // OSC 133;D — command exits.
//! blocks.on_marker(PromptMarker::CommandFinished { exit_code: Some(0) }, 8);
//! // The block stays open (it visually includes the output) until the
//! // next prompt begins the next block.
//! blocks.on_marker(PromptMarker::PromptStart, 9);
//!
//! let done = blocks.first().map(|b| (b.lifecycle(), b.succeeded(), b.end_row()));
//! assert_eq!(done, Some((BlockLifecycle::Finished, Some(true), Some(9))));
//! ```
//!
//! # Heuristic fallback
//!
//! Without shell integration no markers ever arrive. Hosts can still call
//! [`BlockList::observe_row`] after each feed: the first call on a list
//! that has never seen a marker lazily creates one implicit
//! [`BlockLifecycle::Running`] block at row 0, so the whole session stays
//! navigable as a single block.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-terminal-model`]: https://docs.rs/bb-terminal-model

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod block;
mod list;

pub use block::{Block, BlockId, BlockLifecycle, PromptMarker};
pub use list::BlockList;
