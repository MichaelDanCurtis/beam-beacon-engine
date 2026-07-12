//! The [`AgentLoop`] itself: turn orchestration, streaming assembly,
//! retries, cancellation and context trimming.

use std::sync::Arc;

use bb_providers::{
    AssistantMessage, ProviderEvent, ProviderMessage, ProviderRequest, StopReason, TokenUsage,
    ToolCall, ToolResult,
};
use bb_tool_bridge::{ToolBridge, ToolContext};
use tokio::sync::mpsc::{self, UnboundedSender};

use crate::cancel::CancellationToken;
use crate::config::AgentConfig;
use crate::conversation::Conversation;
use crate::events::{AgentError, AgentEvent, TurnOutcome, UserInput};
use crate::provider::{Provider, ProviderCallError, Summarizer};

/// Marker appended to partial assistant text pushed into the conversation
/// when a turn is cancelled mid-stream, so later rounds (and humans reading
/// the transcript) can tell the response was cut short.
pub const CANCELLATION_MARKER: &str = "[response interrupted by cancellation]";

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
}

impl std::fmt::Debug for AgentLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentLoop")
            .field("provider", &self.provider.name())
            .field("tools", &self.tools)
            .field("config", &self.config)
            .field("summarizer", &self.summarizer.is_some())
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
        }
    }

    /// Attach a [`Summarizer`] invoked when context trimming evicts
    /// history.
    #[must_use]
    pub fn with_summarizer(mut self, summarizer: Arc<dyn Summarizer>) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    /// The configuration this loop runs with.
    pub fn config(&self) -> &AgentConfig {
        &self.config
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
    pub async fn run_turn(
        &self,
        conversation: &mut Conversation,
        user_input: UserInput,
        tool_ctx: &ToolContext,
        cancel: &CancellationToken,
        events_tx: UnboundedSender<AgentEvent>,
    ) -> Result<TurnOutcome, AgentError> {
        if cancel.is_cancelled() {
            emit(&events_tx, AgentEvent::TurnCancelled);
            return Ok(TurnOutcome::Cancelled);
        }

        emit(&events_tx, AgentEvent::TurnStarted);
        conversation.push(match user_input {
            UserInput::Text(text) => ProviderMessage::User(text),
            UserInput::Multimodal(content) => ProviderMessage::UserMultimodal(content),
        });

        let tools = self.tools.specs();
        let mut total_usage = TokenUsage::default();

        for _round in 0..self.config.max_rounds {
            if let TrimOutcome::Cancelled = self.trim_context(conversation, cancel).await {
                emit(&events_tx, AgentEvent::TurnCancelled);
                return Ok(TurnOutcome::Cancelled);
            }

            let request = ProviderRequest {
                model: self.config.model.clone(),
                system: None,
                messages: conversation.messages().to_vec(),
                tools: tools.clone(),
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                stream: true,
            };

            let round = match self.call_with_retry(&request, cancel, &events_tx).await? {
                CallOutcome::Cancelled { partial } => {
                    if let Some(assistant) = partial {
                        conversation.push(ProviderMessage::Assistant(assistant));
                    }
                    emit(&events_tx, AgentEvent::TurnCancelled);
                    return Ok(TurnOutcome::Cancelled);
                }
                CallOutcome::Round(round) => round,
            };

            add_usage(&mut total_usage, round.usage);
            emit(
                &events_tx,
                AgentEvent::RoundCompleted {
                    stop_reason: round.stop_reason.clone(),
                },
            );

            let is_tool_round =
                round.stop_reason == StopReason::ToolUse && !round.message.tool_calls.is_empty();
            let tool_calls = round.message.tool_calls.clone();
            conversation.push(ProviderMessage::Assistant(round.message));

            if !is_tool_round {
                // A `ToolUse` stop with zero assembled tool calls is
                // degenerate; the turn completes rather than looping.
                emit(
                    &events_tx,
                    AgentEvent::TurnCompleted {
                        stop_reason: round.stop_reason.clone(),
                    },
                );
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

            let results = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    conversation.push(ProviderMessage::ToolResults(cancelled_results(&tool_calls)));
                    emit(&events_tx, AgentEvent::TurnCancelled);
                    return Ok(TurnOutcome::Cancelled);
                }
                results = self.tools.dispatch_all(&tool_calls, tool_ctx) => results,
            };

            for result in &results {
                emit(
                    &events_tx,
                    AgentEvent::ToolCallFinished {
                        id: result.call_id.clone(),
                        name: result.name.clone(),
                        is_error: result.is_error,
                    },
                );
            }
            conversation.push(ProviderMessage::ToolResults(results));
        }

        let err = AgentError::RoundLimitExceeded(self.config.max_rounds);
        emit(
            &events_tx,
            AgentEvent::TurnFailed {
                error: err.to_string(),
            },
        );
        Err(err)
    }

    /// Trim the conversation to the context budget, invoking the
    /// summarizer (cancellably) when messages were evicted.
    async fn trim_context(
        &self,
        conversation: &mut Conversation,
        cancel: &CancellationToken,
    ) -> TrimOutcome {
        let evicted = conversation.evict_over_budget(&self.config.context);
        if evicted.is_empty() {
            return TrimOutcome::Done;
        }
        if let Some(summarizer) = &self.summarizer {
            let summary = tokio::select! {
                biased;
                () = cancel.cancelled() => return TrimOutcome::Cancelled,
                summary = summarizer.summarize(&evicted) => summary,
            };
            if let Some(text) = summary {
                conversation.insert_summary(&text);
            }
        }
        TrimOutcome::Done
    }

    /// One provider round with retry/backoff on transient errors.
    ///
    /// Emits `Retrying` before each backoff sleep and `TurnFailed` before
    /// returning `Err`. Accumulated streaming state is reset per attempt;
    /// only the successful attempt's content becomes the round result.
    async fn call_with_retry(
        &self,
        request: &ProviderRequest,
        cancel: &CancellationToken,
        events_tx: &UnboundedSender<AgentEvent>,
    ) -> Result<CallOutcome, AgentError> {
        let max_attempts = self.config.retry.max_attempts.max(1);
        let mut attempt: u32 = 1;
        loop {
            match self.call_once(request, cancel, events_tx).await {
                AttemptOutcome::Cancelled { partial } => {
                    return Ok(CallOutcome::Cancelled { partial });
                }
                AttemptOutcome::Complete(round) => return Ok(CallOutcome::Round(round)),
                AttemptOutcome::Failed(ProviderCallError::Permanent(detail)) => {
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

    /// Finish the round; `None` when the stream never delivered `Finished`.
    fn finish(self) -> Option<RoundResult> {
        let stop_reason = self.stop_reason?;
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
                thinking: non_empty(self.thinking),
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
            thinking: non_empty(self.thinking),
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

/// `None` for an empty string, `Some` otherwise.
fn non_empty(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
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
