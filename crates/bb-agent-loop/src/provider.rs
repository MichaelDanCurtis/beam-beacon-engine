//! Host-implemented traits: the [`Provider`] transport and the optional
//! [`Summarizer`] hook.

use bb_providers::{ProviderEvent, ProviderMessage, ProviderRequest};
use futures::future::BoxFuture;
use tokio::sync::mpsc::UnboundedSender;

/// Error classification for one provider call; drives the retry policy.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProviderCallError {
    /// Transient failure (rate limit, 5xx, network) — retried per
    /// [`RetryPolicy`](crate::RetryPolicy).
    #[error("transient provider error: {0}")]
    Transient(String),
    /// Permanent failure (auth, bad request, content policy) — fails the
    /// turn immediately, no retry.
    #[error("permanent provider error: {0}")]
    Permanent(String),
}

/// Host-implemented transport: performs one (streaming) model call.
///
/// The engine is transport-agnostic; the host (Beam or Beacon) implements
/// this over its own HTTP + SSE stack, typically encoding `request` with a
/// [`bb_providers`] adapter and decoding the stream back into
/// [`ProviderEvent`]s.
///
/// Contract:
/// - Deliver events through `events_tx` **as they arrive** (the agent loop
///   forwards text/thinking deltas to the UI live).
/// - Send [`ProviderEvent::Finished`] before resolving on success.
/// - Resolve when the stream completes; classify all failures as
///   [`ProviderCallError::Transient`] or [`ProviderCallError::Permanent`].
/// - Drop `events_tx` (and every clone) when done, so the loop can observe
///   end-of-stream.
pub trait Provider: Send + Sync {
    /// Model/provider identifier for diagnostics.
    fn name(&self) -> &'static str;

    /// Execute the request, delivering [`ProviderEvent`]s as they arrive
    /// via `events_tx`. Resolves when the stream completes. Errors must be
    /// classified.
    fn call<'a>(
        &'a self,
        request: &'a ProviderRequest,
        events_tx: UnboundedSender<ProviderEvent>,
    ) -> BoxFuture<'a, Result<(), ProviderCallError>>;
}

/// Optional host hook that produces a summary of evicted history.
///
/// When context trimming evicts messages, the agent loop hands them to the
/// summarizer (typically a cheap model call host-side); a `Some` result is
/// spliced back into the conversation as a `System` message reading
/// `"Summary of earlier conversation: {summary}"`. Returning `None` skips
/// the splice and the evicted history is simply dropped.
pub trait Summarizer: Send + Sync {
    /// Summarize `evicted` (oldest first). Return `None` to skip.
    fn summarize<'a>(&'a self, evicted: &'a [ProviderMessage]) -> BoxFuture<'a, Option<String>>;
}
