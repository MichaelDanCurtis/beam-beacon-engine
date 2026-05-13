//! Bridge between the agent's tool calls and the host runtime.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! When [`bb-agent-loop`] receives a tool-use intent from a model,
//! `bb-tool-bridge` is what actually runs it. The host (Beam or Beacon)
//! injects implementations for each tool kind:
//!
//! - `Shell` — run a command in a PTY, capture stdout/stderr, surface
//!   exit code; can be the host's own block-aware shell or a sandbox.
//! - `Read` / `Write` / `Edit` — filesystem actions with the host's
//!   permission model.
//! - `ComputerUse` — screenshot + click + keystroke for Anthropic's
//!   computer-use tool; results return as `ToolDispatchResult.images`
//!   for the vision pipeline.
//! - `Subagent` — recursive agent dispatch.
//! - `WebFetch` / `WebSearch` — host-provided network tools.
//!
//! The bridge produces a uniform `ToolDispatchResult` (text + optional
//! images + structured fields) that [`bb-agent-loop`] surfaces back to
//! the provider as a `ToolResult` provider message.
//!
//! # Status
//!
//! Scaffold. Real implementation lands in subsequent PRs.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-agent-loop`]: https://docs.rs/bb-agent-loop

#![forbid(unsafe_code)]
