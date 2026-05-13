//! VT100 / ANSI escape sequence parser feeding [`bb-terminal-model`].
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-vt-parser` is a stateful parser that consumes a stream of bytes
//! emitted by a PTY-bound shell or remote process and applies the
//! resulting actions to a [`bb-terminal-model`] instance. It handles:
//!
//! - C0 / C1 control codes
//! - CSI escape sequences (cursor positioning, SGR colors, scrolling)
//! - OSC sequences (titles, hyperlinks, the OSC 133 prompt protocol)
//! - DCS / APC / SOS / PM passthrough
//! - UTF-8 decoding with chunk-boundary safety
//!
//! # Status
//!
//! Scaffold. Real implementation lands in subsequent PRs.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-terminal-model`]: https://docs.rs/bb-terminal-model

#![forbid(unsafe_code)]
