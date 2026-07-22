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
///
/// # `Ok(())` means the stream ended cleanly
///
/// This is the load-bearing half of the contract. Returning `Ok(())` is an
/// assertion that the transport saw a **normal** end of stream. A truncated
/// HTTP body, a dropped socket, a read error, a timeout — every one of
/// those **must** surface as `Err(ProviderCallError::…)`, never as `Ok(())`
/// with a short event stream.
///
/// Given that, `Ok(())` without a [`ProviderEvent::Finished`] can only mean
/// the server closed a complete stream without sending its terminator
/// sentinel — an `OpenAI`-compatible server omitting the trailing
/// `data: [DONE]` (llama.cpp's server and some vLLM / LM Studio / proxy
/// configurations do this). The response is all there; only the full stop
/// is missing. So the agent loop **keeps** whatever was assembled and
/// completes the round with
/// [`IMPLICIT_STOP_REASON`](crate::IMPLICIT_STOP_REASON); see
/// [`AgentLoop::run_turn`](crate::AgentLoop::run_turn). That leniency is
/// sound *only* because hosts honor the rule above: it never applies to the
/// `Err` paths, so a genuine truncation still fails (and retries) as it
/// should rather than being silently passed off as a finished answer.
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
/// summarizer (typically a cheap model call host-side) and stores a `Some`
/// result in the conversation's single rolling summary slot, rendered into
/// the provider request as a `System` message reading `"Summary of earlier
/// conversation: {summary}"`.
///
/// # Rolling, not accumulating
///
/// There is exactly one summary slot per
/// [`Conversation`](crate::Conversation), and a new summary **replaces**
/// the old one. `previous_summary` is that old summary: fold it into the
/// result so nothing is lost as history keeps rolling off. Because the slot
/// is fixed, the summarizer owns the length budget — a summarizer that lets
/// each round's output grow will grow the request with it.
///
/// Returning `None` leaves the previous summary in place and drops the
/// evicted history; it does not clear the slot.
///
/// # Cancellation
///
/// The hook is awaited under the turn's
/// [`CancellationToken`](crate::CancellationToken), and the loop only
/// commits the eviction once the hook resolves. A summarizer cancelled
/// mid-flight therefore costs nothing: the conversation is left exactly as
/// it was, history included.
pub trait Summarizer: Send + Sync {
    /// Summarize `evicted` (oldest first), folding in `previous_summary`
    /// when the conversation already has one. Return `None` to skip.
    fn summarize<'a>(
        &'a self,
        previous_summary: Option<&'a str>,
        evicted: &'a [ProviderMessage],
    ) -> BoxFuture<'a, Option<String>>;
}
