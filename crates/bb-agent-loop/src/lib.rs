//! The core AI agent loop: tool-use, streaming, retry policy, conversation
//! state.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-agent-loop` orchestrates an AI agent turn-by-turn over the
//! abstractions in [`bb-providers`] (message/event model) and
//! [`bb-tool-bridge`] (tool dispatch). The host implements the actual
//! transport (HTTP + SSE) behind the [`Provider`] trait defined here. This
//! crate owns:
//!
//! - **Turn structure** — user input → provider call → tool calls → tool
//!   results → next provider call → final output
//!   ([`AgentLoop::run_turn`]). Every assistant message carrying tool
//!   calls is always paired with tool results, on every exit path, so a
//!   turn can never leave history the provider will reject.
//! - **Streaming** — [`AgentEvent`]s delivered to the UI as they happen.
//!   A stream that ends cleanly without an explicit stop reason still
//!   keeps its content ([`IMPLICIT_STOP_REASON`]).
//! - **Cancellation** — a sticky, clonable [`CancellationToken`] honored
//!   at every await point, never at the cost of conversation history.
//! - **Retry policy** — exponential backoff for transient provider errors
//!   ([`RetryPolicy`]).
//! - **Conversation state** — [`Conversation`] history with
//!   context-window trimming ([`ContextPolicy`]) and an optional
//!   [`Summarizer`] hook feeding a single rolling summary slot.
//! - **Telemetry** — structured [`MetricEvent`]s handed to an optional
//!   host-provided [`MetricsSink`] ([`AgentLoop::with_metrics`]), so each
//!   consumer plugs its own backend. A loop without a sink records
//!   nothing.
//!
//! # Example wiring
//!
//! ```no_run
//! use std::sync::Arc;
//! use bb_agent_loop::{AgentConfig, AgentLoop, CancellationToken, Conversation, UserInput};
//! use bb_tool_bridge::{ToolBridge, ToolContext};
//!
//! # async fn example(provider: Arc<dyn bb_agent_loop::Provider>) -> Result<(), bb_agent_loop::AgentError> {
//! let tools = Arc::new(ToolBridge::new());
//! let config = AgentConfig { model: "claude-sonnet-4-5".to_owned(), ..AgentConfig::default() };
//! let agent = AgentLoop::new(provider, tools, config);
//!
//! let mut conversation = Conversation::with_system("You are a terminal assistant.");
//! let cancel = CancellationToken::new();
//! let (events_tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();
//!
//! let outcome = agent
//!     .run_turn(
//!         &mut conversation,
//!         UserInput::Text("list the files here".to_owned()),
//!         &ToolContext::default(),
//!         &cancel,
//!         events_tx,
//!     )
//!     .await?;
//! # let _ = (outcome, events_rx.try_recv());
//! # Ok(())
//! # }
//! ```
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-providers`]: https://docs.rs/bb-providers
//! [`bb-tool-bridge`]: https://docs.rs/bb-tool-bridge

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod agent;
mod cancel;
mod config;
mod conversation;
mod events;
mod metrics;
mod provider;
pub mod testing;

pub use agent::{AgentLoop, CANCELLATION_MARKER, IMPLICIT_STOP_REASON};
pub use cancel::CancellationToken;
pub use config::{AgentConfig, ContextPolicy, RetryPolicy};
pub use conversation::Conversation;
pub use events::{AgentError, AgentEvent, TurnOutcome, UserInput};
pub use metrics::{MetricEvent, MetricsSink, NoopMetricsSink};
pub use provider::{Provider, ProviderCallError, Summarizer};
