//! Turn-level types: UI events, user input, outcomes and errors.

use bb_providers::{StopReason, TokenUsage, UserContent};

/// Events surfaced to the UI while a turn runs.
///
/// Delivered through the `events_tx` channel passed to
/// [`AgentLoop::run_turn`](crate::AgentLoop::run_turn), in order.
///
/// # Retries and partial deltas
///
/// Deltas are forwarded live, so when a provider call fails mid-stream with
/// a transient error, deltas already emitted for that attempt cannot be
/// unsent. UIs should treat [`AgentEvent::Retrying`] as "discard the
/// current partial round text": the retried attempt re-streams the round
/// from the start, and the conversation only ever records text from the
/// attempt that succeeded.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AgentEvent {
    /// The turn began (always the first event of a non-pre-cancelled turn).
    TurnStarted,
    /// A fragment of visible assistant text.
    TextDelta(String),
    /// A fragment of thinking / reasoning text.
    ThinkingDelta(String),
    /// A tool call is about to be dispatched.
    ToolCallStarted {
        /// Provider-assigned call id.
        id: String,
        /// Tool name.
        name: String,
    },
    /// A dispatched tool call produced its result.
    ToolCallFinished {
        /// Provider-assigned call id.
        id: String,
        /// Tool name.
        name: String,
        /// True when the tool failed and its result describes the error.
        is_error: bool,
    },
    /// A provider round completed; more rounds follow if tools ran.
    RoundCompleted {
        /// Why the provider stopped this round.
        stop_reason: StopReason,
    },
    /// Token accounting reported by the provider, as it arrives.
    Usage(TokenUsage),
    /// A transient provider error occurred; the call will be retried after
    /// a backoff sleep.
    Retrying {
        /// 1-based number of the attempt about to be made (the first retry
        /// reports `2`).
        attempt: u32,
        /// Description of the transient error that triggered the retry.
        error: String,
    },
    /// The turn finished successfully (always the final event).
    TurnCompleted {
        /// Stop reason of the final round.
        stop_reason: StopReason,
    },
    /// The turn failed (always the final event); `run_turn` returns the
    /// matching [`AgentError`].
    TurnFailed {
        /// Human-readable description of the failure.
        error: String,
    },
    /// The turn was cancelled (always the final event, emitted exactly
    /// once); `run_turn` returns [`TurnOutcome::Cancelled`].
    TurnCancelled,
}

/// One user turn of input.
#[derive(Debug, Clone, PartialEq)]
pub enum UserInput {
    /// Plain text.
    Text(String),
    /// Mixed text and images.
    Multimodal(Vec<UserContent>),
}

/// How a turn ended, when it did not error.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnOutcome {
    /// The turn ran to completion.
    Completed {
        /// Stop reason of the final round.
        stop_reason: StopReason,
        /// Token usage summed across every successful round of the turn.
        usage: TokenUsage,
    },
    /// The turn was cancelled via its
    /// [`CancellationToken`](crate::CancellationToken).
    Cancelled,
}

/// Errors a turn can fail with.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AgentError {
    /// Permanent provider failure, or a transient failure that exhausted
    /// the retry budget.
    #[error("provider failure: {0}")]
    Provider(String),
    /// The turn was still requesting tools after the configured maximum
    /// number of provider rounds.
    #[error("round limit exceeded: turn still requesting tools after {0} rounds")]
    RoundLimitExceeded(u32),
}
