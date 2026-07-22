//! Session state shared across surfaces.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-shared-session` is the top-level glue crate. It owns the
//! [`Session`] aggregate that holds:
//!
//! - The current [`bb_terminal_model`] grid + scrollback, driven by a
//!   [`bb_vt_parser::Parser`] via [`Session::feed_pty`]
//! - The [`bb_block_list`] view over that grid, kept in sync with the
//!   OSC 133 / OSC 7 events the parser surfaces
//! - The [`bb_agent_loop`] conversation for any in-flight agent turn
//!   ([`Session::run_agent_turn`])
//! - Persistent history ([`SessionHistory`]) backed by a host-provided
//!   KV store ([`KvStore`] — sqlite in both Beam and Beacon today)
//! - Identity / auth context surfaced from the host
//!   ([`IdentityContext`]: anonymous, signed in, BYOK, ARC-attached)
//!
//! This is the crate that Beam and Beacon both depend on at the top
//! level; the rest of the engine crates are reachable through it.
//!
//! # What persists, what doesn't
//!
//! A [`SessionSnapshot`] stores the *durable* state: conversation
//! transcript, title, cwd, and identity. The live screen and block list
//! are views over the PTY byte stream — they are rebuilt from PTY output
//! on reattach and never serialized.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod history;
mod identity;
mod kv;
mod session;

pub use history::{
    HistoryError, SessionHistory, SessionSnapshot, SessionSummary, SNAPSHOT_VERSION,
};
pub use identity::IdentityContext;
pub use kv::{KvError, KvStore, MemoryKvStore};
pub use session::{Session, SessionId};
