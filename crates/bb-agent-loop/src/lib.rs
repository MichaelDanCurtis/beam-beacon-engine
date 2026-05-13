//! The core AI agent loop: tool-use, streaming, retry policy, conversation state.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-agent-loop` orchestrates an AI agent turn-by-turn over the
//! abstractions in [`bb-providers`] and [`bb-tool-bridge`]. It owns:
//!
//! - Turn structure: user input → provider call → tool calls → tool
//!   results → next provider call → terminal output
//! - Streaming token delivery to the UI
//! - Cancellation (Esc / Ctrl-C / explicit stop) at every await point
//! - Tool-call dispatch via [`bb-tool-bridge`]
//! - Retry / fallback policy for transient provider errors
//! - Conversation state: history, summarization triggers, context
//!   window management
//!
//! # Status
//!
//! Scaffold. Real implementation lands in subsequent PRs.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-providers`]: https://docs.rs/bb-providers
//! [`bb-tool-bridge`]: https://docs.rs/bb-tool-bridge

#![forbid(unsafe_code)]
