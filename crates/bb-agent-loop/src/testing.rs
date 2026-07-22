//! Test helpers for asserting on agent-loop telemetry.
//!
//! Available unconditionally (not behind a feature flag or `cfg(test)`) so
//! host crates — Beam, Beacon and their integration suites — can assert on
//! what the loop recorded without dev-dependency gymnastics, mirroring
//! [`bb_tool_bridge::testing`].

use std::sync::Mutex;
use std::time::Duration;

use crate::metrics::{MetricEvent, MetricsSink};

/// An owned snapshot of one [`MetricEvent`].
///
/// [`MetricEvent`] borrows from the loop's state (`&'a str`,
/// `&'a StopReason`), so a sink that wants to keep events must copy them
/// out. `RecordedEvent` is that copy: the event's stable
/// [`kind`](MetricEvent::kind), its payload rendered by
/// [`detail`](MetricEvent::detail), and its
/// [`duration`](MetricEvent::duration) kept as a real `Duration` rather
/// than folded into the string.
///
/// Splitting `duration` out is what makes assertions deterministic:
/// `detail` never contains a wall-clock measurement, so it can be compared
/// exactly, while timing assertions go through the typed field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedEvent {
    /// Variant name of the recorded event, e.g. `"TurnStarted"`.
    pub kind: &'static str,
    /// Space-separated `key=value` rendering of the payload, excluding
    /// durations. See [`MetricEvent::detail`].
    pub detail: String,
    /// The event's duration, when its variant carries one.
    pub duration: Option<Duration>,
}

impl From<&MetricEvent<'_>> for RecordedEvent {
    fn from(event: &MetricEvent<'_>) -> Self {
        Self {
            kind: event.kind(),
            detail: event.detail(),
            duration: event.duration(),
        }
    }
}

/// A [`MetricsSink`] that stores every event it is handed, for tests.
///
/// Recording takes a lock and allocates, so this is a **test** sink, not a
/// production one.
///
/// ```
/// use std::sync::Arc;
/// use bb_agent_loop::testing::RecordingMetricsSink;
/// use bb_agent_loop::{MetricEvent, MetricsSink};
///
/// let sink = Arc::new(RecordingMetricsSink::new());
/// sink.record(&MetricEvent::TurnStarted { model: "claude-sonnet-4-5" });
///
/// assert_eq!(sink.kinds(), ["TurnStarted"]);
/// assert_eq!(sink.details("TurnStarted"), ["model=claude-sonnet-4-5"]);
/// ```
#[derive(Debug, Default)]
pub struct RecordingMetricsSink {
    events: Mutex<Vec<RecordedEvent>>,
}

impl RecordingMetricsSink {
    /// Create an empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every event recorded so far, in order.
    pub fn events(&self) -> Vec<RecordedEvent> {
        self.lock().clone()
    }

    /// The [`kind`](RecordedEvent::kind) of every event recorded so far, in
    /// order — the usual way to assert on an event sequence.
    pub fn kinds(&self) -> Vec<&'static str> {
        self.lock().iter().map(|e| e.kind).collect()
    }

    /// Every event of `kind`, in order.
    pub fn of_kind(&self, kind: &str) -> Vec<RecordedEvent> {
        self.lock()
            .iter()
            .filter(|e| e.kind == kind)
            .cloned()
            .collect()
    }

    /// The [`detail`](RecordedEvent::detail) of every event of `kind`, in
    /// order.
    pub fn details(&self, kind: &str) -> Vec<String> {
        self.lock()
            .iter()
            .filter(|e| e.kind == kind)
            .map(|e| e.detail.clone())
            .collect()
    }

    /// How many events of `kind` were recorded.
    pub fn count(&self, kind: &str) -> usize {
        self.lock().iter().filter(|e| e.kind == kind).count()
    }

    /// The terminal events recorded — `TurnCompleted`, `TurnCancelled` and
    /// `TurnFailed`.
    ///
    /// A turn records **exactly one** of these, always last, so
    /// `assert_eq!(sink.terminal().len(), 1)` is the invariant check.
    pub fn terminal(&self) -> Vec<RecordedEvent> {
        self.lock()
            .iter()
            .filter(|e| matches!(e.kind, "TurnCompleted" | "TurnCancelled" | "TurnFailed"))
            .cloned()
            .collect()
    }

    /// Drop every recorded event.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// Lock the buffer, recovering from a poisoned mutex so a failing
    /// assertion in one test never cascades into unrelated panics.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<RecordedEvent>> {
        self.events.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl MetricsSink for RecordingMetricsSink {
    fn record(&self, event: &MetricEvent<'_>) {
        self.lock().push(RecordedEvent::from(event));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use bb_providers::{StopReason, TokenUsage};

    use super::*;

    #[test]
    fn records_kinds_details_and_durations_in_order() {
        let stop = StopReason::EndTurn;
        let sink = RecordingMetricsSink::new();
        sink.record(&MetricEvent::TurnStarted { model: "m" });
        sink.record(&MetricEvent::ToolDispatched {
            name: "echo",
            duration: Duration::from_millis(7),
            is_error: true,
        });
        sink.record(&MetricEvent::TurnCompleted {
            duration: Duration::from_millis(11),
            rounds: 1,
            usage: TokenUsage::default(),
            stop_reason: &stop,
        });

        assert_eq!(
            sink.kinds(),
            ["TurnStarted", "ToolDispatched", "TurnCompleted"]
        );
        assert_eq!(sink.details("ToolDispatched"), ["name=echo is_error=true"]);
        assert_eq!(sink.count("TurnStarted"), 1);
        assert_eq!(sink.count("RetryScheduled"), 0);

        let events = sink.events();
        assert_eq!(events[0].duration, None);
        assert_eq!(events[1].duration, Some(Duration::from_millis(7)));

        let terminal = sink.terminal();
        assert_eq!(terminal.len(), 1);
        assert_eq!(terminal[0].kind, "TurnCompleted");
    }

    #[test]
    fn of_kind_and_clear() {
        let sink = RecordingMetricsSink::default();
        for attempt in 1..=2 {
            sink.record(&MetricEvent::RetryScheduled {
                model: "m",
                attempt,
                backoff: Duration::from_millis(u64::from(attempt) * 100),
            });
        }
        let retries = sink.of_kind("RetryScheduled");
        assert_eq!(retries.len(), 2);
        assert_eq!(retries[1].detail, "model=m attempt=2");
        assert_eq!(retries[1].duration, Some(Duration::from_millis(200)));

        sink.clear();
        assert!(sink.events().is_empty());
        assert!(sink.terminal().is_empty());
    }
}
