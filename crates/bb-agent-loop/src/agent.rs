//! The [`AgentLoop`] itself: turn orchestration, streaming assembly,
//! retries, cancellation and context trimming.

use std::sync::Arc;

use bb_providers::{
    AssistantMessage, ProviderEvent, ProviderMessage, ProviderRequest, StopReason, ThinkingBlock,
    TokenUsage, ToolCall, ToolResult,
};
use bb_tool_bridge::{ToolBridge, ToolContext};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::time::Instant;

use crate::cancel::CancellationToken;
use crate::config::AgentConfig;
use crate::conversation::Conversation;
use crate::events::{AgentError, AgentEvent, TurnOutcome, UserInput};
use crate::metrics::{MetricEvent, MetricsSink};
use crate::provider::{Provider, ProviderCallError, Summarizer};

/// Marker appended to partial assistant text pushed into the conversation
/// when a turn is cancelled mid-stream, so later rounds (and humans reading
/// the transcript) can tell the response was cut short.
pub const CANCELLATION_MARKER: &str = "[response interrupted by cancellation]";

/// [`StopReason::Other`] payload the loop synthesizes when a provider
/// stream ended cleanly but never delivered [`ProviderEvent::Finished`].
///
/// Namespaced so it is unambiguously engine-generated: a host or
/// [`MetricsSink`] seeing `StopReason::Other(s)` can compare `s` against
/// this constant to tell "no terminator arrived" apart from a real
/// provider-specific stop reason, which is passed through verbatim.
///
/// See [`AgentLoop::run_turn`] and the [`Provider`] contract for why the
/// round is completed rather than failed.
pub const IMPLICIT_STOP_REASON: &str =
    "bb-agent-loop: stream ended without an explicit stop reason";

/// Orchestrates AI agent turns over a [`Provider`] transport and a
/// [`ToolBridge`].
///
/// One `AgentLoop` is cheap and stateless between turns: all conversation
/// state lives in the host-owned [`Conversation`] passed to
/// [`AgentLoop::run_turn`].
pub struct AgentLoop {
    provider: Arc<dyn Provider>,
    tools: Arc<ToolBridge>,
    config: AgentConfig,
    summarizer: Option<Arc<dyn Summarizer>>,
    metrics: Option<Arc<dyn MetricsSink>>,
}

impl std::fmt::Debug for AgentLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentLoop")
            .field("provider", &self.provider.name())
            .field("tools", &self.tools)
            .field("config", &self.config)
            .field("summarizer", &self.summarizer.is_some())
            .field("metrics", &self.metrics.is_some())
            .finish()
    }
}

impl AgentLoop {
    /// Build an agent loop over a provider transport and a tool registry.
    pub fn new(provider: Arc<dyn Provider>, tools: Arc<ToolBridge>, config: AgentConfig) -> Self {
        Self {
            provider,
            tools,
            config,
            summarizer: None,
            metrics: None,
        }
    }

    /// Attach a [`Summarizer`] invoked when context trimming evicts
    /// history.
    #[must_use]
    pub fn with_summarizer(mut self, summarizer: Arc<dyn Summarizer>) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    /// Attach a [`MetricsSink`] that observes every [`MetricEvent`] of
    /// every turn this loop runs.
    ///
    /// Purely additive: without it the loop behaves exactly as before and
    /// records nothing — it does not even construct the events, so the
    /// un-instrumented path costs one `Option` branch per instrumentation
    /// site. See [`MetricEvent`] for what is recorded and when.
    ///
    /// The sink is called from the turn's hot path and must be cheap and
    /// non-blocking; see [`MetricsSink`].
    #[must_use]
    pub fn with_metrics(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics = Some(sink);
        self
    }

    /// The configuration this loop runs with.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Hand the configured sink, if any, to `f`.
    ///
    /// Taking a closure (rather than a built event) keeps event
    /// construction — and any allocation it needs — out of the default
    /// no-sink path entirely.
    #[inline]
    fn with_sink(&self, f: impl FnOnce(&dyn MetricsSink)) {
        if let Some(sink) = &self.metrics {
            f(sink.as_ref());
        }
    }

    /// Run one user turn to completion.
    ///
    /// Pushes the user message, then loops: provider call (streamed) → if
    /// the round requested tools, dispatch them via the bridge, push the
    /// results and go again — until a round stops with something other
    /// than [`StopReason::ToolUse`] or
    /// [`AgentConfig::max_rounds`] is hit. [`AgentEvent`]s are emitted via
    /// `events_tx` throughout (send failures are ignored, so a dropped
    /// receiver never aborts a turn).
    ///
    /// # The tool-pairing invariant
    ///
    /// **An `Assistant` message carrying tool calls is always immediately
    /// followed by a `ToolResults` message covering every one of its call
    /// ids** — on every exit path, whatever the stop reason, however the
    /// turn ended. Nothing else is a valid conversation: Anthropic rejects
    /// the next request with *"tool_use ids were found without
    /// tool_result blocks"* and `OpenAI` with *"an assistant message with
    /// 'tool_calls' must be followed by tool messages"*, so a single
    /// unpaired assistant message poisons the session's history for good.
    ///
    /// The interesting case is a round that streamed tool calls but
    /// finished with a stop reason other than [`StopReason::ToolUse`] —
    /// exactly what a provider reports when the token limit truncates a
    /// `tool_use` block (Anthropic `stop_reason: max_tokens`, `OpenAI`
    /// `finish_reason: length` mid-`tool_calls`). The loop **does not**
    /// dispatch those calls: the model never finished asking, so their
    /// arguments may be half-written, and running a truncated request is
    /// worse than not running it. Instead it pairs them with synthesized
    /// `is_error` results naming the stop reason, the same machinery
    /// cancellation uses. The turn then completes normally and reports the
    /// provider's real stop reason, so the caller still sees *why* the
    /// round ended and the model can re-request the tool next turn.
    ///
    /// # Cancellation
    ///
    /// `cancel` is honored at every await point: the provider stream, tool
    /// dispatch, retry backoff sleeps, and the summarizer hook. On cancel
    /// the conversation is left coherent:
    ///
    /// - mid-stream: any partial assistant text/thinking is pushed with
    ///   [`CANCELLATION_MARKER`] appended (incomplete tool calls are
    ///   dropped);
    /// - mid-dispatch: the already-pushed assistant message is paired with
    ///   synthesized `is_error` tool results noting the cancellation, so
    ///   no tool call is left unanswered.
    ///
    /// [`AgentEvent::TurnCancelled`] is emitted exactly once and the call
    /// returns `Ok(TurnOutcome::Cancelled)`. A token cancelled before
    /// `run_turn` starts returns `Cancelled` without pushing the user
    /// message or calling the provider.
    ///
    /// # Telemetry
    ///
    /// When a [`MetricsSink`] is attached
    /// ([`AgentLoop::with_metrics`]), the turn records a
    /// [`MetricEvent::TurnStarted`], per-attempt provider events, one
    /// [`MetricEvent::ToolDispatched`] per tool result, and exactly one
    /// terminal event. All durations are measured with
    /// [`tokio::time::Instant`], so they follow a paused test clock.
    pub async fn run_turn(
        &self,
        conversation: &mut Conversation,
        user_input: UserInput,
        tool_ctx: &ToolContext,
        cancel: &CancellationToken,
        events_tx: UnboundedSender<AgentEvent>,
    ) -> Result<TurnOutcome, AgentError> {
        let started = Instant::now();
        let mut rounds: u32 = 0;

        if cancel.is_cancelled() {
            emit(&events_tx, AgentEvent::TurnCancelled);
            self.with_sink(|sink| {
                sink.record(&MetricEvent::TurnCancelled {
                    duration: started.elapsed(),
                    rounds,
                });
            });
            return Ok(TurnOutcome::Cancelled);
        }

        emit(&events_tx, AgentEvent::TurnStarted);
        self.with_sink(|sink| {
            sink.record(&MetricEvent::TurnStarted {
                model: &self.config.model,
            });
        });
        conversation.push(match user_input {
            UserInput::Text(text) => ProviderMessage::User(text),
            UserInput::Multimodal(content) => ProviderMessage::UserMultimodal(content),
        });

        let tools = self.tools.specs();
        let mut total_usage = TokenUsage::default();

        for _round in 0..self.config.max_rounds {
            if let TrimOutcome::Cancelled = self.trim_context(conversation, cancel).await {
                emit(&events_tx, AgentEvent::TurnCancelled);
                self.with_sink(|sink| {
                    sink.record(&MetricEvent::TurnCancelled {
                        duration: started.elapsed(),
                        rounds,
                    });
                });
                return Ok(TurnOutcome::Cancelled);
            }

            let request = ProviderRequest {
                model: self.config.model.clone(),
                system: None,
                messages: conversation.provider_messages(),
                tools: tools.clone(),
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                stream: true,
            };

            let round = match self.call_with_retry(&request, cancel, &events_tx).await {
                Err(err) => {
                    self.with_sink(|sink| {
                        sink.record(&MetricEvent::TurnFailed {
                            duration: started.elapsed(),
                            rounds,
                            error: &err.to_string(),
                        });
                    });
                    return Err(err);
                }
                Ok(CallOutcome::Cancelled { partial }) => {
                    if let Some(assistant) = partial {
                        conversation.push(ProviderMessage::Assistant(assistant));
                    }
                    emit(&events_tx, AgentEvent::TurnCancelled);
                    self.with_sink(|sink| {
                        sink.record(&MetricEvent::TurnCancelled {
                            duration: started.elapsed(),
                            rounds,
                        });
                    });
                    return Ok(TurnOutcome::Cancelled);
                }
                Ok(CallOutcome::Round(round)) => round,
            };

            rounds += 1;
            add_usage(&mut total_usage, round.usage);
            emit(
                &events_tx,
                AgentEvent::RoundCompleted {
                    stop_reason: round.stop_reason.clone(),
                },
            );

            let tool_calls = round.message.tool_calls.clone();
            let is_tool_round = round.stop_reason == StopReason::ToolUse && !tool_calls.is_empty();
            conversation.push(ProviderMessage::Assistant(round.message));

            if !is_tool_round {
                // A `ToolUse` stop with zero assembled tool calls is
                // degenerate; the turn completes rather than looping.
                //
                // Tool calls under a non-`ToolUse` stop reason mean the
                // round was cut short mid-request. They are never
                // dispatched, but they must still be answered or the
                // history becomes unencodable — see the tool-pairing
                // invariant in this method's docs.
                if !tool_calls.is_empty() {
                    conversation.push(ProviderMessage::ToolResults(unfinished_results(
                        &tool_calls,
                        &round.stop_reason,
                    )));
                }
                emit(
                    &events_tx,
                    AgentEvent::TurnCompleted {
                        stop_reason: round.stop_reason.clone(),
                    },
                );
                self.with_sink(|sink| {
                    sink.record(&MetricEvent::TurnCompleted {
                        duration: started.elapsed(),
                        rounds,
                        usage: total_usage,
                        stop_reason: &round.stop_reason,
                    });
                });
                return Ok(TurnOutcome::Completed {
                    stop_reason: round.stop_reason,
                    usage: total_usage,
                });
            }

            for call in &tool_calls {
                emit(
                    &events_tx,
                    AgentEvent::ToolCallStarted {
                        id: call.id.clone(),
                        name: call.name.clone(),
                    },
                );
            }

            let dispatch_started = Instant::now();
            let results = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    conversation.push(ProviderMessage::ToolResults(cancelled_results(&tool_calls)));
                    emit(&events_tx, AgentEvent::TurnCancelled);
                    self.with_sink(|sink| {
                        sink.record(&MetricEvent::TurnCancelled {
                            duration: started.elapsed(),
                            rounds,
                        });
                    });
                    return Ok(TurnOutcome::Cancelled);
                }
                results = self.tools.dispatch_all(&tool_calls, tool_ctx) => results,
            };
            let dispatch_duration = dispatch_started.elapsed();

            for result in &results {
                emit(
                    &events_tx,
                    AgentEvent::ToolCallFinished {
                        id: result.call_id.clone(),
                        name: result.name.clone(),
                        is_error: result.is_error,
                    },
                );
                self.with_sink(|sink| {
                    sink.record(&MetricEvent::ToolDispatched {
                        name: &result.name,
                        duration: dispatch_duration,
                        is_error: result.is_error,
                    });
                });
            }
            conversation.push(ProviderMessage::ToolResults(results));
        }

        let err = AgentError::RoundLimitExceeded(self.config.max_rounds);
        let error = err.to_string();
        self.with_sink(|sink| {
            sink.record(&MetricEvent::TurnFailed {
                duration: started.elapsed(),
                rounds,
                error: &error,
            });
        });
        emit(&events_tx, AgentEvent::TurnFailed { error });
        Err(err)
    }

    /// Trim the conversation to the context budget, invoking the
    /// summarizer (cancellably) when messages would be evicted.
    ///
    /// # Atomic with respect to cancellation
    ///
    /// Eviction is planned first and committed last: nothing is removed
    /// from the conversation until the summarizer has resolved. A turn
    /// cancelled inside the summarizer — a user pressing Esc while a cheap
    /// model call is in flight — therefore leaves the conversation byte
    /// for byte as it found it, rather than returning `Cancelled` over a
    /// history that has already been destroyed and never summarized.
    /// Dropping the whole `run_turn` future mid-summarizer is equally
    /// safe, for the same reason.
    ///
    /// The summary lands in the conversation's single rolling slot,
    /// replacing whatever was there, with the old summary handed to the
    /// summarizer as prior context. See [`Conversation`] for why that slot
    /// is bounded by design.
    ///
    /// Records [`MetricEvent::ContextTrimmed`] whenever eviction was
    /// planned — including when the summarizer hook was cancelled, in
    /// which case `summarized` is `false` and nothing was evicted after
    /// all.
    async fn trim_context(
        &self,
        conversation: &mut Conversation,
        cancel: &CancellationToken,
    ) -> TrimOutcome {
        let plan = conversation.plan_eviction(&self.config.context);
        if plan.is_empty() {
            return TrimOutcome::Done;
        }
        let evicted_messages = plan.len();

        let mut new_summary = None;
        if let Some(summarizer) = &self.summarizer {
            let previous = conversation.summary().map(str::to_owned);
            let preview = conversation.evicted_preview(&plan);
            new_summary = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    self.with_sink(|sink| {
                        sink.record(&MetricEvent::ContextTrimmed {
                            evicted_messages,
                            summarized: false,
                        });
                    });
                    return TrimOutcome::Cancelled;
                }
                summary = summarizer.summarize(previous.as_deref(), &preview) => summary,
            };
        }

        // Commit: past this point the turn is no longer cancellable
        // without losing the evicted history, so it is the last thing done.
        conversation.commit_eviction(&plan);
        let summarized = new_summary.is_some();
        if let Some(text) = new_summary {
            conversation.set_summary(text);
        }
        self.with_sink(|sink| {
            sink.record(&MetricEvent::ContextTrimmed {
                evicted_messages,
                summarized,
            });
        });
        TrimOutcome::Done
    }

    /// One provider round with retry/backoff on transient errors.
    ///
    /// Emits `Retrying` before each backoff sleep and `TurnFailed` before
    /// returning `Err`. Accumulated streaming state is reset per attempt;
    /// only the successful attempt's content becomes the round result.
    ///
    /// Records [`MetricEvent::ProviderCallCompleted`] /
    /// [`MetricEvent::ProviderCallFailed`] per attempt, and
    /// [`MetricEvent::RetryScheduled`] before each backoff sleep. The
    /// turn-terminal [`MetricEvent::TurnFailed`] is recorded by the caller,
    /// which knows the turn's duration and round count.
    async fn call_with_retry(
        &self,
        request: &ProviderRequest,
        cancel: &CancellationToken,
        events_tx: &UnboundedSender<AgentEvent>,
    ) -> Result<CallOutcome, AgentError> {
        let max_attempts = self.config.retry.max_attempts.max(1);
        let mut attempt: u32 = 1;
        loop {
            let attempt_started = Instant::now();
            match self.call_once(request, cancel, events_tx).await {
                AttemptOutcome::Cancelled { partial } => {
                    return Ok(CallOutcome::Cancelled { partial });
                }
                AttemptOutcome::Complete(round) => {
                    self.with_sink(|sink| {
                        sink.record(&MetricEvent::ProviderCallCompleted {
                            model: &request.model,
                            attempt,
                            duration: attempt_started.elapsed(),
                            usage: round.usage,
                            stop_reason: &round.stop_reason,
                        });
                    });
                    return Ok(CallOutcome::Round(round));
                }
                AttemptOutcome::Failed(ProviderCallError::Permanent(detail)) => {
                    self.with_sink(|sink| {
                        sink.record(&MetricEvent::ProviderCallFailed {
                            model: &request.model,
                            attempt,
                            transient: false,
                            error: &detail,
                        });
                    });
                    let err = AgentError::Provider(detail);
                    emit(
                        events_tx,
                        AgentEvent::TurnFailed {
                            error: err.to_string(),
                        },
                    );
                    return Err(err);
                }
                AttemptOutcome::Failed(ProviderCallError::Transient(detail)) => {
                    self.with_sink(|sink| {
                        sink.record(&MetricEvent::ProviderCallFailed {
                            model: &request.model,
                            attempt,
                            transient: true,
                            error: &detail,
                        });
                    });
                    if attempt >= max_attempts {
                        let err = AgentError::Provider(format!(
                            "transient error persisted after {attempt} attempts: {detail}"
                        ));
                        emit(
                            events_tx,
                            AgentEvent::TurnFailed {
                                error: err.to_string(),
                            },
                        );
                        return Err(err);
                    }
                    let backoff = self.config.retry.backoff_after(attempt);
                    attempt += 1;
                    emit(
                        events_tx,
                        AgentEvent::Retrying {
                            attempt,
                            error: detail,
                        },
                    );
                    self.with_sink(|sink| {
                        sink.record(&MetricEvent::RetryScheduled {
                            model: &request.model,
                            attempt,
                            backoff,
                        });
                    });
                    tokio::select! {
                        biased;
                        () = cancel.cancelled() => {
                            return Ok(CallOutcome::Cancelled { partial: None });
                        }
                        () = tokio::time::sleep(backoff) => {}
                    }
                }
            }
        }
    }

    /// One provider call attempt: stream events, forward deltas, assemble
    /// the assistant message.
    async fn call_once(
        &self,
        request: &ProviderRequest,
        cancel: &CancellationToken,
        events_tx: &UnboundedSender<AgentEvent>,
    ) -> AttemptOutcome {
        let (tx, mut rx) = mpsc::unbounded_channel::<ProviderEvent>();
        let mut call_fut = self.provider.call(request, tx);
        let mut call_result: Option<Result<(), ProviderCallError>> = None;
        let mut acc = RoundAccumulator::default();

        // Drain the event stream while concurrently driving the call
        // future; cancellation wins over both.
        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    return AttemptOutcome::Cancelled { partial: acc.into_partial() };
                }
                result = &mut call_fut, if call_result.is_none() => {
                    call_result = Some(result);
                }
                event = rx.recv() => match event {
                    Some(event) => acc.ingest(event, events_tx),
                    // Channel closed: every sender is dropped and all
                    // buffered events have been consumed.
                    None => break,
                },
            }
        }

        let result = match call_result {
            Some(result) => result,
            // The provider dropped its sender before resolving; await the
            // call future itself (still cancellable).
            None => tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    return AttemptOutcome::Cancelled { partial: acc.into_partial() };
                }
                result = &mut call_fut => result,
            },
        };

        match result {
            Err(error) => AttemptOutcome::Failed(error),
            Ok(()) => match acc.finish() {
                Some(round) => AttemptOutcome::Complete(round),
                None => AttemptOutcome::Failed(ProviderCallError::Permanent(
                    "provider stream ended without a Finished event".to_owned(),
                )),
            },
        }
    }
}

/// How one call-with-retry ended (errors are returned separately).
enum CallOutcome {
    /// Cancelled; `partial` carries any partial assistant content.
    Cancelled { partial: Option<AssistantMessage> },
    /// A round completed.
    Round(RoundResult),
}

/// How a single call attempt ended.
enum AttemptOutcome {
    Cancelled { partial: Option<AssistantMessage> },
    Complete(RoundResult),
    Failed(ProviderCallError),
}

/// The assembled output of one successful provider round.
struct RoundResult {
    message: AssistantMessage,
    stop_reason: StopReason,
    usage: TokenUsage,
}

/// Whether context trimming ran to completion or was cancelled.
enum TrimOutcome {
    Done,
    Cancelled,
}

/// Streaming accumulator for one call attempt.
///
/// Text/thinking deltas are appended and simultaneously forwarded as
/// [`AgentEvent`]s; tool-call arguments are assembled by concatenating
/// `json_fragment`s and parsed at `ToolCallEnd`.
#[derive(Default)]
struct RoundAccumulator {
    text: String,
    thinking: String,
    /// Provider-opaque signature authenticating the thinking block, from
    /// [`ProviderEvent::ThinkingSignature`]. Last one wins.
    thinking_signature: Option<String>,
    calls: Vec<PendingToolCall>,
    usage: TokenUsage,
    stop_reason: Option<StopReason>,
}

/// A tool call still being streamed.
struct PendingToolCall {
    index: usize,
    id: String,
    name: String,
    arguments_json: String,
    complete: bool,
}

impl RoundAccumulator {
    fn ingest(&mut self, event: ProviderEvent, events_tx: &UnboundedSender<AgentEvent>) {
        match event {
            ProviderEvent::MessageStart => {}
            ProviderEvent::TextDelta(delta) => {
                self.text.push_str(&delta);
                emit(events_tx, AgentEvent::TextDelta(delta));
            }
            ProviderEvent::ThinkingDelta(delta) => {
                self.thinking.push_str(&delta);
                emit(events_tx, AgentEvent::ThinkingDelta(delta));
            }
            // Opaque authentication for the thinking block, not content:
            // stored for verbatim replay, never surfaced to the UI.
            ProviderEvent::ThinkingSignature(signature) => {
                self.thinking_signature = Some(signature);
            }
            ProviderEvent::ToolCallStart { index, id, name } => {
                self.calls.push(PendingToolCall {
                    index,
                    id,
                    name,
                    arguments_json: String::new(),
                    complete: false,
                });
            }
            ProviderEvent::ToolCallArgumentsDelta {
                index,
                json_fragment,
            } => {
                if let Some(call) = self.open_call_mut(index) {
                    call.arguments_json.push_str(&json_fragment);
                }
            }
            ProviderEvent::ToolCallEnd { index } => {
                if let Some(call) = self.open_call_mut(index) {
                    call.complete = true;
                }
            }
            ProviderEvent::Usage(usage) => {
                add_usage(&mut self.usage, usage);
                emit(events_tx, AgentEvent::Usage(usage));
            }
            ProviderEvent::Finished(stop_reason) => self.stop_reason = Some(stop_reason),
        }
    }

    /// The most recent not-yet-complete call at `index`.
    fn open_call_mut(&mut self, index: usize) -> Option<&mut PendingToolCall> {
        self.calls
            .iter_mut()
            .rev()
            .find(|c| c.index == index && !c.complete)
    }

    /// Finish the round.
    ///
    /// `None` — which the caller turns into a failed attempt — only when
    /// the stream delivered **neither** a `Finished` event **nor** any
    /// content at all.
    ///
    /// # Why a missing `Finished` is not fatal on its own
    ///
    /// This runs after the transport resolved `Ok(())`, which per the
    /// [`Provider`] contract asserts the stream ended *normally*; every
    /// truncation, socket drop and read error is required to surface as
    /// `Err` instead, and those paths are untouched by this leniency. So
    /// content assembled here is a complete response from a server that
    /// simply never sent its terminator sentinel — an `OpenAI`-compatible
    /// server omitting `data: [DONE]` is the common case, and it is the
    /// audience local-model adapters serve. The text has already been
    /// streamed to the user's screen; discarding it and failing the turn
    /// would be a user-visible data loss over a missing full stop.
    ///
    /// The round therefore completes with
    /// [`StopReason::Other(IMPLICIT_STOP_REASON)`](crate::IMPLICIT_STOP_REASON),
    /// which is distinguishable from any provider-sent stop reason, so
    /// hosts and metrics sinks can still tell that no terminator arrived.
    /// An empty stream stays an error: there is nothing to salvage, and
    /// "the provider said nothing" is a real failure worth reporting.
    fn finish(self) -> Option<RoundResult> {
        let has_content =
            !self.text.is_empty() || !self.thinking.is_empty() || !self.calls.is_empty();
        let stop_reason = match self.stop_reason {
            Some(stop_reason) => stop_reason,
            None if has_content => StopReason::Other(IMPLICIT_STOP_REASON.to_owned()),
            None => return None,
        };
        let tool_calls = self
            .calls
            .into_iter()
            .map(|call| ToolCall {
                id: call.id,
                name: call.name,
                arguments: parse_tool_arguments(&call.arguments_json),
            })
            .collect();
        Some(RoundResult {
            message: AssistantMessage {
                text: self.text,
                thinking: thinking_block(self.thinking, self.thinking_signature),
                tool_calls,
            },
            stop_reason,
            usage: self.usage,
        })
    }

    /// Partial assistant content to persist on cancellation, with
    /// [`CANCELLATION_MARKER`] appended. `None` when nothing was streamed
    /// yet. Incomplete tool calls are intentionally dropped so the
    /// conversation never contains an assistant tool request without a
    /// matching result.
    fn into_partial(self) -> Option<AssistantMessage> {
        if self.text.is_empty() && self.thinking.is_empty() {
            return None;
        }
        let mut text = self.text;
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(CANCELLATION_MARKER);
        Some(AssistantMessage {
            text,
            thinking: thinking_block(self.thinking, self.thinking_signature),
            tool_calls: Vec::new(),
        })
    }
}

/// Parse assembled tool-call arguments.
///
/// - Empty/whitespace input becomes `{}` (providers commonly stream no
///   fragments for zero-argument tools).
/// - Unparseable input becomes `{"_raw": "<original text>"}` so the raw
///   payload is preserved for the tool handler / logs instead of being
///   silently discarded.
fn parse_tool_arguments(raw: &str) -> serde_json::Value {
    if raw.trim().is_empty() {
        return serde_json::Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::json!({ "_raw": raw }))
}

/// Synthesized `is_error` results for tool calls aborted by cancellation,
/// keeping every assistant tool request paired with a result.
fn cancelled_results(calls: &[ToolCall]) -> Vec<ToolResult> {
    calls
        .iter()
        .map(|call| ToolResult {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: "Tool execution cancelled by user.".to_owned(),
            images: Vec::new(),
            is_error: true,
        })
        .collect()
}

/// Synthesized `is_error` results for tool calls the model streamed but
/// never finished requesting, because the round stopped for some reason
/// other than [`StopReason::ToolUse`].
///
/// These calls are deliberately **not** dispatched — see the tool-pairing
/// invariant on [`AgentLoop::run_turn`]. The content names the stop reason
/// so the model can see why its request was dropped and decide whether to
/// re-issue it.
fn unfinished_results(calls: &[ToolCall], stop_reason: &StopReason) -> Vec<ToolResult> {
    let content = format!(
        "Tool call not executed: the response stopped with {stop_reason:?} \
         before the tool-use request was complete, so it was discarded. \
         Request the tool again if it is still needed."
    );
    calls
        .iter()
        .map(|call| ToolResult {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: content.clone(),
            images: Vec::new(),
            is_error: true,
        })
        .collect()
}

/// The assembled thinking block, `None` when the round streamed no
/// thinking at all.
///
/// `signature` is the provider's opaque attestation of the block
/// ([`ProviderEvent::ThinkingSignature`]); it is carried through untouched
/// because Anthropic rejects a replayed thinking block whose signature is
/// missing or altered.
fn thinking_block(text: String, signature: Option<String>) -> Option<ThinkingBlock> {
    if text.is_empty() && signature.is_none() {
        return None;
    }
    Some(ThinkingBlock { text, signature })
}

fn add_usage(total: &mut TokenUsage, delta: TokenUsage) {
    total.input_tokens = total.input_tokens.saturating_add(delta.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(delta.output_tokens);
    total.cache_read_tokens = total
        .cache_read_tokens
        .saturating_add(delta.cache_read_tokens);
}

/// Best-effort event emission: a dropped receiver never aborts a turn.
fn emit(events_tx: &UnboundedSender<AgentEvent>, event: AgentEvent) {
    let _ = events_tx.send(event);
}
