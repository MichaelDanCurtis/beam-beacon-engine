//! Session state shared across surfaces.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-shared-session` is the top-level glue crate. It owns the
//! `Session` aggregate that holds:
//!
//! - The current [`bb-terminal-model`] grid + scrollback
//! - The [`bb-block-list`] view over that grid
//! - The [`bb-agent-loop`] state for any in-flight agent turn
//! - Persistent history (recent sessions, named workspaces) backed by a
//!   host-provided KV store (sqlite in both Beam and Beacon today)
//! - Identity / auth context surfaced from the host (anonymous, signed
//!   in, BYOK, ARC-attached)
//!
//! This is the crate that Beam and Beacon will both depend on at the
//! top level; the rest of the engine crates are reachable through it.
//!
//! # Status
//!
//! Scaffold. Real implementation lands in subsequent PRs.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-terminal-model`]: https://docs.rs/bb-terminal-model
//! [`bb-block-list`]: https://docs.rs/bb-block-list
//! [`bb-agent-loop`]: https://docs.rs/bb-agent-loop

#![forbid(unsafe_code)]
