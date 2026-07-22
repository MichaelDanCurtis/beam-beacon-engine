//! Host-provided telemetry: the [`MetricsSink`] hook and the
//! [`MetricEvent`]s the agent loop records.
//!
//! # Why a host trait
//!
//! The engine is shared by consumers with incompatible observability
//! stacks (Beam ships an OSS console; Beacon ships enterprise telemetry),
//! so the loop does not pick a metrics backend. It records structured
//! moments and the host decides where they go. Attaching a sink is
//! additive — see [`AgentLoop::with_metrics`](crate::AgentLoop::with_metrics)
//! — and a loop without one records nothing.
//!
//! # Example: a console sink
//!
//! ```
//! use std::sync::Arc;
//! use bb_agent_loop::{MetricEvent, MetricsSink};
//!
//! struct ConsoleSink;
//!
//! impl MetricsSink for ConsoleSink {
//!     fn record(&self, event: &MetricEvent<'_>) {
//!         println!("[metric] {} {}", event.kind(), event.detail());
//!     }
//! }
//!
//! let sink: Arc<dyn MetricsSink> = Arc::new(ConsoleSink);
//! sink.record(&MetricEvent::TurnStarted { model: "claude-sonnet-4-5" });
//! ```

use std::fmt::Write as _;
use std::time::Duration;

use bb_providers::{StopReason, TokenUsage};

/// Host-provided telemetry sink.
///
/// Implementations **must be cheap and non-blocking**: [`record`] is called
/// from inside the agent turn's hot path (between streaming chunks, around
/// provider calls and tool dispatch), on the task driving
/// [`AgentLoop::run_turn`](crate::AgentLoop::run_turn). Do the minimum
/// possible work here — increment a counter, push onto an unbounded
/// channel, hand off to a background exporter. Never block, never await,
/// never perform I/O that can stall.
///
/// Panicking inside `record` panics the turn; implementations should be
/// infallible.
///
/// [`record`]: MetricsSink::record
pub trait MetricsSink: Send + Sync {
    /// Record one observable moment of an agent turn.
    ///
    /// The event borrows from the loop's internal state and is only valid
    /// for the duration of this call; copy out whatever the sink needs
    /// (see [`RecordedEvent`](crate::testing::RecordedEvent) for an owned
    /// mirror).
    fn record(&self, event: &MetricEvent<'_>);
}

/// A [`MetricsSink`] that discards every event.
///
/// Useful for hosts that plumb a sink unconditionally and want to disable
/// telemetry at the edge. Note that an [`AgentLoop`](crate::AgentLoop)
/// with no sink installed is *cheaper still*: it skips constructing the
/// events entirely.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoopMetricsSink;

impl MetricsSink for NoopMetricsSink {
    #[inline]
    fn record(&self, _event: &MetricEvent<'_>) {}
}

/// A single observable moment in an agent turn.
///
/// Events borrow from the loop's state (`&'a str`, `&'a StopReason`) so
/// recording allocates nothing. Sinks that outlive the call must copy —
/// [`MetricEvent::kind`] / [`MetricEvent::detail`] /
/// [`MetricEvent::duration`] give a ready-made owned projection, and
/// [`RecordedEvent`](crate::testing::RecordedEvent) bundles all three.
///
/// # Ordering guarantees
///
/// Within one turn:
///
/// - [`TurnStarted`] is recorded first, and only for turns that actually
///   start (a turn cancelled *before* it starts records only
///   [`TurnCancelled`]).
/// - Exactly **one** terminal event — [`TurnCompleted`], [`TurnCancelled`]
///   or [`TurnFailed`] — is recorded, always last.
///
/// This enum is `#[non_exhaustive]`: new observable moments can be added
/// without a breaking change, so downstream `match`es need a `_` arm.
///
/// [`TurnStarted`]: MetricEvent::TurnStarted
/// [`TurnCompleted`]: MetricEvent::TurnCompleted
/// [`TurnCancelled`]: MetricEvent::TurnCancelled
/// [`TurnFailed`]: MetricEvent::TurnFailed
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MetricEvent<'a> {
    /// A turn began. Recorded once, before the first provider call.
    TurnStarted {
        /// Model id the turn runs against.
        model: &'a str,
    },
    /// One provider round finished successfully.
    ProviderCallCompleted {
        /// Model id the call was made against.
        model: &'a str,
        /// 1-based attempt number that succeeded (`1` when no retry was
        /// needed).
        attempt: u32,
        /// Wall time of this attempt, from request to assembled round.
        duration: Duration,
        /// Token accounting reported by this round.
        usage: TokenUsage,
        /// Why the provider stopped this round.
        stop_reason: &'a StopReason,
    },
    /// One provider attempt failed. `transient` distinguishes retryable
    /// from fatal.
    ProviderCallFailed {
        /// Model id the call was made against.
        model: &'a str,
        /// 1-based number of the attempt that failed.
        attempt: u32,
        /// `true` for
        /// [`ProviderCallError::Transient`](crate::ProviderCallError::Transient)
        /// (a retry may follow), `false` for
        /// [`ProviderCallError::Permanent`](crate::ProviderCallError::Permanent)
        /// (the turn fails immediately).
        transient: bool,
        /// Provider-supplied description of the failure.
        error: &'a str,
    },
    /// A retry is about to be slept for.
    RetryScheduled {
        /// Model id the retry will be made against.
        model: &'a str,
        /// 1-based number of the attempt about to be made — the first
        /// retry reports `2`, matching
        /// [`AgentEvent::Retrying`](crate::AgentEvent::Retrying).
        attempt: u32,
        /// Backoff slept before that attempt.
        backoff: Duration,
    },
    /// A tool call was dispatched and produced a result.
    ///
    /// The loop dispatches a round's tool calls as one batch
    /// ([`ToolBridge::dispatch_all`](bb_tool_bridge::ToolBridge::dispatch_all)),
    /// which does not expose per-call timing. One event is recorded per
    /// result, and **every event of a batch carries the same `duration`:
    /// the wall time of the whole batch**. For a single-tool round (the
    /// common case) that is exactly the tool's own duration; for a
    /// multi-tool round, sum-per-name aggregations will over-count and
    /// should be read as "batch latency observed with this tool in it".
    ToolDispatched {
        /// Tool name, as dispatched.
        name: &'a str,
        /// Wall time of the dispatch batch this result came from.
        duration: Duration,
        /// `true` when the tool reported failure.
        is_error: bool,
    },
    /// Context-window trimming ran and evicted history.
    ///
    /// Only recorded when trimming actually evicted something; a
    /// within-budget conversation records nothing.
    ContextTrimmed {
        /// Number of messages evicted from the conversation.
        evicted_messages: usize,
        /// `true` when the [`Summarizer`](crate::Summarizer) hook produced
        /// a summary that was spliced back in. `false` when no summarizer
        /// is installed, it declined, or it was cancelled.
        summarized: bool,
    },
    /// The turn completed successfully. Terminal.
    TurnCompleted {
        /// Wall time of the whole turn.
        duration: Duration,
        /// Provider rounds completed during the turn.
        rounds: u32,
        /// Token usage summed across every successful round.
        usage: TokenUsage,
        /// Stop reason of the final round.
        stop_reason: &'a StopReason,
    },
    /// The turn was cancelled. Terminal.
    TurnCancelled {
        /// Wall time until cancellation was observed (near zero for a turn
        /// cancelled before it started).
        duration: Duration,
        /// Provider rounds completed before cancellation.
        rounds: u32,
    },
    /// The turn failed. Terminal.
    TurnFailed {
        /// Wall time until the failure.
        duration: Duration,
        /// Provider rounds completed before the failure.
        rounds: u32,
        /// Description of the failure (the
        /// [`AgentError`](crate::AgentError) `run_turn` returns).
        error: &'a str,
    },
}

impl MetricEvent<'_> {
    /// Stable, low-cardinality name for this event — the variant name.
    ///
    /// Safe to use as a metric name or label: it is a fixed set of
    /// `&'static str`s that never embeds runtime data.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::TurnStarted { .. } => "TurnStarted",
            Self::ProviderCallCompleted { .. } => "ProviderCallCompleted",
            Self::ProviderCallFailed { .. } => "ProviderCallFailed",
            Self::RetryScheduled { .. } => "RetryScheduled",
            Self::ToolDispatched { .. } => "ToolDispatched",
            Self::ContextTrimmed { .. } => "ContextTrimmed",
            Self::TurnCompleted { .. } => "TurnCompleted",
            Self::TurnCancelled { .. } => "TurnCancelled",
            Self::TurnFailed { .. } => "TurnFailed",
        }
    }

    /// The event's [`Duration`] field, when it has one.
    ///
    /// `None` for [`TurnStarted`](Self::TurnStarted) and
    /// [`ContextTrimmed`](Self::ContextTrimmed); for
    /// [`RetryScheduled`](Self::RetryScheduled) this is the pending
    /// `backoff` rather than an elapsed measurement.
    pub fn duration(&self) -> Option<Duration> {
        match *self {
            Self::TurnStarted { .. } | Self::ContextTrimmed { .. } => None,
            Self::ProviderCallCompleted { duration, .. }
            | Self::ToolDispatched { duration, .. }
            | Self::TurnCompleted { duration, .. }
            | Self::TurnCancelled { duration, .. }
            | Self::TurnFailed { duration, .. } => Some(duration),
            Self::RetryScheduled { backoff, .. } => Some(backoff),
            Self::ProviderCallFailed { .. } => None,
        }
    }

    /// Human-readable rendering of the event's payload as space-separated
    /// `key=value` pairs, e.g. `model=claude-sonnet-4-5 attempt=1`.
    ///
    /// **Durations are deliberately excluded** — every `Duration` field is
    /// available through [`MetricEvent::duration`] instead. That keeps
    /// `detail` free of wall-clock measurements, so log lines and test
    /// assertions built on it stay deterministic. Values are rendered
    /// verbatim and may themselves contain spaces (error strings), so
    /// prefer `contains` over exact equality when asserting.
    pub fn detail(&self) -> String {
        let mut out = String::new();
        match *self {
            Self::TurnStarted { model } => {
                let _ = write!(out, "model={model}");
            }
            Self::ProviderCallCompleted {
                model,
                attempt,
                usage,
                stop_reason,
                duration: _,
            } => {
                let _ = write!(
                    out,
                    "model={model} attempt={attempt} stop_reason={stop_reason:?}"
                );
                write_usage(&mut out, usage);
            }
            Self::ProviderCallFailed {
                model,
                attempt,
                transient,
                error,
            } => {
                let _ = write!(
                    out,
                    "model={model} attempt={attempt} transient={transient} error={error}"
                );
            }
            Self::RetryScheduled {
                model,
                attempt,
                backoff: _,
            } => {
                let _ = write!(out, "model={model} attempt={attempt}");
            }
            Self::ToolDispatched {
                name,
                is_error,
                duration: _,
            } => {
                let _ = write!(out, "name={name} is_error={is_error}");
            }
            Self::ContextTrimmed {
                evicted_messages,
                summarized,
            } => {
                let _ = write!(
                    out,
                    "evicted_messages={evicted_messages} summarized={summarized}"
                );
            }
            Self::TurnCompleted {
                rounds,
                usage,
                stop_reason,
                duration: _,
            } => {
                let _ = write!(out, "rounds={rounds} stop_reason={stop_reason:?}");
                write_usage(&mut out, usage);
            }
            Self::TurnCancelled { rounds, .. } => {
                let _ = write!(out, "rounds={rounds}");
            }
            Self::TurnFailed { rounds, error, .. } => {
                let _ = write!(out, "rounds={rounds} error={error}");
            }
        }
        out
    }
}

/// Append the `key=value` rendering of `usage` to `out`.
fn write_usage(out: &mut String, usage: TokenUsage) {
    let _ = write!(
        out,
        " input_tokens={} output_tokens={} cache_read_tokens={}",
        usage.input_tokens, usage.output_tokens, usage.cache_read_tokens
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const USAGE: TokenUsage = TokenUsage {
        input_tokens: 10,
        output_tokens: 20,
        cache_read_tokens: 30,
    };

    fn all_events<'a>(stop: &'a StopReason, error: &'a str) -> Vec<MetricEvent<'a>> {
        vec![
            MetricEvent::TurnStarted { model: "m" },
            MetricEvent::ProviderCallCompleted {
                model: "m",
                attempt: 1,
                duration: Duration::from_millis(5),
                usage: USAGE,
                stop_reason: stop,
            },
            MetricEvent::ProviderCallFailed {
                model: "m",
                attempt: 1,
                transient: true,
                error,
            },
            MetricEvent::RetryScheduled {
                model: "m",
                attempt: 2,
                backoff: Duration::from_millis(500),
            },
            MetricEvent::ToolDispatched {
                name: "echo",
                duration: Duration::from_millis(7),
                is_error: false,
            },
            MetricEvent::ContextTrimmed {
                evicted_messages: 2,
                summarized: true,
            },
            MetricEvent::TurnCompleted {
                duration: Duration::from_millis(9),
                rounds: 2,
                usage: USAGE,
                stop_reason: stop,
            },
            MetricEvent::TurnCancelled {
                duration: Duration::from_millis(3),
                rounds: 1,
            },
            MetricEvent::TurnFailed {
                duration: Duration::from_millis(4),
                rounds: 0,
                error,
            },
        ]
    }

    #[test]
    fn kind_is_the_variant_name_for_every_event() {
        let stop = StopReason::EndTurn;
        let kinds: Vec<&'static str> = all_events(&stop, "boom").iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "TurnStarted",
                "ProviderCallCompleted",
                "ProviderCallFailed",
                "RetryScheduled",
                "ToolDispatched",
                "ContextTrimmed",
                "TurnCompleted",
                "TurnCancelled",
                "TurnFailed",
            ]
        );
    }

    #[test]
    fn duration_is_exposed_only_where_the_variant_has_one() {
        let stop = StopReason::EndTurn;
        let durations: Vec<Option<Duration>> = all_events(&stop, "boom")
            .iter()
            .map(MetricEvent::duration)
            .collect();
        assert_eq!(
            durations,
            [
                None,
                Some(Duration::from_millis(5)),
                None,
                // `RetryScheduled` reports the pending backoff.
                Some(Duration::from_millis(500)),
                Some(Duration::from_millis(7)),
                None,
                Some(Duration::from_millis(9)),
                Some(Duration::from_millis(3)),
                Some(Duration::from_millis(4)),
            ]
        );
    }

    #[test]
    fn detail_renders_payload_without_durations() {
        let stop = StopReason::ToolUse;
        let details: Vec<String> = all_events(&stop, "rate limited")
            .iter()
            .map(MetricEvent::detail)
            .collect();
        assert_eq!(details[0], "model=m");
        assert_eq!(
            details[1],
            "model=m attempt=1 stop_reason=ToolUse \
             input_tokens=10 output_tokens=20 cache_read_tokens=30"
        );
        assert_eq!(
            details[2],
            "model=m attempt=1 transient=true error=rate limited"
        );
        assert_eq!(details[3], "model=m attempt=2");
        assert_eq!(details[4], "name=echo is_error=false");
        assert_eq!(details[5], "evicted_messages=2 summarized=true");
        assert_eq!(
            details[6],
            "rounds=2 stop_reason=ToolUse input_tokens=10 output_tokens=20 cache_read_tokens=30"
        );
        assert_eq!(details[7], "rounds=1");
        assert_eq!(details[8], "rounds=0 error=rate limited");
        // No rendering leaks a wall-clock measurement.
        assert!(details.iter().all(|d| !d.contains("duration")));
    }

    #[test]
    fn noop_sink_accepts_every_event_and_does_nothing() {
        let stop = StopReason::EndTurn;
        let sink = NoopMetricsSink;
        for event in all_events(&stop, "boom") {
            sink.record(&event);
        }
        // Also usable behind the dyn boundary hosts plumb it through.
        let dynamic: &dyn MetricsSink = &sink;
        dynamic.record(&MetricEvent::TurnStarted { model: "m" });
        let boxed: std::sync::Arc<dyn MetricsSink> = std::sync::Arc::new(NoopMetricsSink);
        boxed.record(&MetricEvent::TurnStarted { model: "m" });
        assert_eq!(sink, NoopMetricsSink);
    }
}
