//! Integration tests for `bb-agent-loop`, driven by a scripted
//! `MockProvider` and `bb_tool_bridge::testing::FnTool`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bb_agent_loop::testing::RecordingMetricsSink;
use bb_agent_loop::{
    AgentConfig, AgentError, AgentEvent, AgentLoop, CancellationToken, ContextPolicy, Conversation,
    NoopMetricsSink, Provider, ProviderCallError, Summarizer, TurnOutcome, UserInput,
    CANCELLATION_MARKER,
};
use bb_providers::{
    AssistantMessage, ProviderEvent, ProviderMessage, ProviderRequest, StopReason, TokenUsage,
    ToolCall, ToolResult,
};
use bb_tool_bridge::testing::FnTool;
use bb_tool_bridge::{ToolBridge, ToolContext, ToolDispatchResult, ToolHandler};
use futures::future::BoxFuture;
use futures::FutureExt;
use serde_json::json;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/// One scripted provider response.
enum Script {
    /// Send the events, then resolve `Ok(())`.
    Events(Vec<ProviderEvent>),
    /// Sleep, then send the events and resolve `Ok(())` (for duration
    /// assertions under a paused clock).
    Slow(Duration, Vec<ProviderEvent>),
    /// Send the events, then resolve with the error.
    Fail(Vec<ProviderEvent>, ProviderCallError),
    /// Send the events, then never resolve (for cancellation tests).
    Hang(Vec<ProviderEvent>),
}

/// Scripted [`Provider`] replaying canned event sequences, one script per
/// call, recording every request it receives.
struct MockProvider {
    scripts: Mutex<VecDeque<Script>>,
    requests: Mutex<Vec<ProviderRequest>>,
}

impl MockProvider {
    fn new(scripts: Vec<Script>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(scripts.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    fn request(&self, i: usize) -> ProviderRequest {
        self.requests.lock().unwrap()[i].clone()
    }
}

impl Provider for MockProvider {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn call<'a>(
        &'a self,
        request: &'a ProviderRequest,
        events_tx: UnboundedSender<ProviderEvent>,
    ) -> BoxFuture<'a, Result<(), ProviderCallError>> {
        async move {
            self.requests.lock().unwrap().push(request.clone());
            let script = self
                .scripts
                .lock()
                .unwrap()
                .pop_front()
                .expect("MockProvider called more times than scripted");
            match script {
                Script::Events(events) => {
                    for event in events {
                        let _ = events_tx.send(event);
                    }
                    Ok(())
                }
                Script::Slow(delay, events) => {
                    tokio::time::sleep(delay).await;
                    for event in events {
                        let _ = events_tx.send(event);
                    }
                    Ok(())
                }
                Script::Fail(events, error) => {
                    for event in events {
                        let _ = events_tx.send(event);
                    }
                    Err(error)
                }
                Script::Hang(events) => {
                    for event in events {
                        let _ = events_tx.send(event);
                    }
                    std::future::pending::<()>().await;
                    unreachable!("pending future resolved")
                }
            }
        }
        .boxed()
    }
}

/// Summarizer that records what it was asked to summarize, and the prior
/// summary it was handed.
struct RecordingSummarizer {
    seen: Mutex<Vec<Vec<ProviderMessage>>>,
    previous: Mutex<Vec<Option<String>>>,
    reply: Option<String>,
}

impl RecordingSummarizer {
    fn new(reply: Option<&str>) -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
            previous: Mutex::new(Vec::new()),
            reply: reply.map(str::to_owned),
        }
    }
}

impl Summarizer for RecordingSummarizer {
    fn summarize<'a>(
        &'a self,
        previous_summary: Option<&'a str>,
        evicted: &'a [ProviderMessage],
    ) -> BoxFuture<'a, Option<String>> {
        async move {
            self.seen.lock().unwrap().push(evicted.to_vec());
            self.previous
                .lock()
                .unwrap()
                .push(previous_summary.map(str::to_owned));
            self.reply.clone()
        }
        .boxed()
    }
}

/// Summarizer that signals it started, then never finishes — for asserting
/// that cancelling inside the hook costs no history.
struct HangingSummarizer {
    started: Mutex<Option<oneshot::Sender<()>>>,
}

impl Summarizer for HangingSummarizer {
    fn summarize<'a>(
        &'a self,
        _previous_summary: Option<&'a str>,
        _evicted: &'a [ProviderMessage],
    ) -> BoxFuture<'a, Option<String>> {
        async move {
            if let Some(tx) = self.started.lock().unwrap().take() {
                let _ = tx.send(());
            }
            std::future::pending::<()>().await;
            unreachable!("pending future resolved")
        }
        .boxed()
    }
}

/// Summarizer returning a fixed-length reply, counting its invocations.
struct CountingSummarizer {
    calls: Mutex<Vec<Option<String>>>,
}

impl Summarizer for CountingSummarizer {
    fn summarize<'a>(
        &'a self,
        previous_summary: Option<&'a str>,
        _evicted: &'a [ProviderMessage],
    ) -> BoxFuture<'a, Option<String>> {
        async move {
            let mut calls = self.calls.lock().unwrap();
            calls.push(previous_summary.map(str::to_owned));
            // A realistic summarizer folds the prior summary into a reply
            // of roughly constant length. Fixed width here so the token
            // plateau below can be asserted exactly.
            Some(format!("rolling summary #{:04}", calls.len()))
        }
        .boxed()
    }
}

/// Tool whose dispatch signals it started, then never finishes.
struct SlowTool {
    started: Mutex<Option<oneshot::Sender<()>>>,
}

impl ToolHandler for SlowTool {
    fn spec(&self) -> bb_providers::ToolSpec {
        bb_providers::ToolSpec {
            name: "slow".to_owned(),
            description: "never finishes".to_owned(),
            input_schema: json!({"type": "object", "properties": {}}),
        }
    }

    fn dispatch<'a>(
        &'a self,
        _call: &'a ToolCall,
        _ctx: &'a ToolContext,
    ) -> BoxFuture<'a, ToolDispatchResult> {
        async move {
            if let Some(tx) = self.started.lock().unwrap().take() {
                let _ = tx.send(());
            }
            std::future::pending::<()>().await;
            unreachable!("pending future resolved")
        }
        .boxed()
    }
}

/// Tool that sleeps before succeeding (for duration assertions under a
/// paused clock).
struct SleepyTool {
    delay: Duration,
}

impl ToolHandler for SleepyTool {
    fn spec(&self) -> bb_providers::ToolSpec {
        bb_providers::ToolSpec {
            name: "sleepy".to_owned(),
            description: "sleeps, then succeeds".to_owned(),
            input_schema: json!({"type": "object", "properties": {}}),
        }
    }

    fn dispatch<'a>(
        &'a self,
        _call: &'a ToolCall,
        _ctx: &'a ToolContext,
    ) -> BoxFuture<'a, ToolDispatchResult> {
        async move {
            tokio::time::sleep(self.delay).await;
            ToolDispatchResult::ok("slept")
        }
        .boxed()
    }
}

fn echo_bridge() -> Arc<ToolBridge> {
    let mut bridge = ToolBridge::new();
    bridge
        .register(Arc::new(FnTool::new(
            "echo",
            "echoes its arguments",
            |call, _ctx| ToolDispatchResult::ok(format!("echo:{}", call.arguments)),
        )))
        .unwrap();
    Arc::new(bridge)
}

/// `echo` (always succeeds) plus `boom` (always fails), so one dispatch
/// batch can produce both an ok and an `is_error` result.
fn echo_and_boom_bridge() -> Arc<ToolBridge> {
    let mut bridge = ToolBridge::new();
    bridge
        .register(Arc::new(FnTool::new(
            "echo",
            "echoes its arguments",
            |call, _ctx| ToolDispatchResult::ok(format!("echo:{}", call.arguments)),
        )))
        .unwrap();
    bridge
        .register(Arc::new(FnTool::new(
            "boom",
            "always fails",
            |_call, _ctx| ToolDispatchResult::error("kaboom"),
        )))
        .unwrap();
    Arc::new(bridge)
}

fn test_config() -> AgentConfig {
    AgentConfig {
        model: "test-model".to_owned(),
        ..AgentConfig::default()
    }
}

fn agent(provider: Arc<MockProvider>, tools: Arc<ToolBridge>, config: AgentConfig) -> AgentLoop {
    AgentLoop::new(provider, tools, config)
}

/// An agent loop with a [`RecordingMetricsSink`] attached, plus the sink.
fn recording_agent(
    provider: Arc<MockProvider>,
    tools: Arc<ToolBridge>,
    config: AgentConfig,
) -> (AgentLoop, Arc<RecordingMetricsSink>) {
    let sink = Arc::new(RecordingMetricsSink::new());
    let agent = AgentLoop::new(provider, tools, config).with_metrics(sink.clone());
    (agent, sink)
}

/// Run one text turn to completion, returning the sink it recorded into.
async fn run_simple_turn(agent: &AgentLoop, cancel: &CancellationToken) -> TurnOutcome {
    let mut conversation = Conversation::new();
    let (tx, _rx) = unbounded_channel();
    agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            cancel,
            tx,
        )
        .await
        .unwrap()
}

fn drain(rx: &mut UnboundedReceiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    events
}

fn text_round(text: &[&str], stop: StopReason) -> Script {
    let mut events = vec![ProviderEvent::MessageStart];
    events.extend(
        text.iter()
            .map(|t| ProviderEvent::TextDelta((*t).to_owned())),
    );
    events.push(ProviderEvent::Finished(stop));
    Script::Events(events)
}

/// A round requesting several tool calls at once (each with `{}` args).
fn multi_tool_round(calls: &[(&str, &str)]) -> Script {
    let mut events = vec![ProviderEvent::MessageStart];
    for (index, (id, name)) in calls.iter().enumerate() {
        events.push(ProviderEvent::ToolCallStart {
            index,
            id: (*id).to_owned(),
            name: (*name).to_owned(),
        });
        events.push(ProviderEvent::ToolCallArgumentsDelta {
            index,
            json_fragment: "{}".to_owned(),
        });
        events.push(ProviderEvent::ToolCallEnd { index });
    }
    events.push(ProviderEvent::Finished(StopReason::ToolUse));
    Script::Events(events)
}

fn tool_round(id: &str, name: &str, arg_fragments: &[&str]) -> Script {
    let mut events = vec![
        ProviderEvent::MessageStart,
        ProviderEvent::ToolCallStart {
            index: 0,
            id: id.to_owned(),
            name: name.to_owned(),
        },
    ];
    events.extend(
        arg_fragments
            .iter()
            .map(|f| ProviderEvent::ToolCallArgumentsDelta {
                index: 0,
                json_fragment: (*f).to_owned(),
            }),
    );
    events.push(ProviderEvent::ToolCallEnd { index: 0 });
    events.push(ProviderEvent::Finished(StopReason::ToolUse));
    Script::Events(events)
}

fn assistant_tool_calls(msg: &ProviderMessage) -> &[ToolCall] {
    match msg {
        ProviderMessage::Assistant(a) => &a.tool_calls,
        other => panic!("expected Assistant message, got {other:?}"),
    }
}

fn count_cancelled(events: &[AgentEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, AgentEvent::TurnCancelled))
        .count()
}

// ---------------------------------------------------------------------------
// Happy paths
// ---------------------------------------------------------------------------

#[tokio::test]
async fn single_round_text_turn() {
    let provider = MockProvider::new(vec![text_round(&["Hel", "lo"], StopReason::EndTurn)]);
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let outcome = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("hi".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    assert_eq!(
        outcome,
        TurnOutcome::Completed {
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
        }
    );
    assert_eq!(
        drain(&mut rx),
        vec![
            AgentEvent::TurnStarted,
            AgentEvent::TextDelta("Hel".to_owned()),
            AgentEvent::TextDelta("lo".to_owned()),
            AgentEvent::RoundCompleted {
                stop_reason: StopReason::EndTurn
            },
            AgentEvent::TurnCompleted {
                stop_reason: StopReason::EndTurn
            },
        ]
    );
    assert_eq!(
        conversation.messages(),
        &[
            ProviderMessage::User("hi".to_owned()),
            ProviderMessage::Assistant(AssistantMessage {
                text: "Hello".to_owned(),
                ..AssistantMessage::default()
            }),
        ]
    );
    // The request advertised the registered tools and asked to stream.
    let request = provider.request(0);
    assert!(request.stream);
    assert_eq!(request.model, "test-model");
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "echo");
}

#[tokio::test]
async fn two_round_tool_turn_has_exact_conversation_shape() {
    let provider = MockProvider::new(vec![
        tool_round("call_1", "echo", &[r#"{"q":"x"}"#]),
        text_round(&["done"], StopReason::EndTurn),
    ]);
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let outcome = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("run echo".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    assert!(matches!(
        outcome,
        TurnOutcome::Completed {
            stop_reason: StopReason::EndTurn,
            ..
        }
    ));

    let expected_call = ToolCall {
        id: "call_1".to_owned(),
        name: "echo".to_owned(),
        arguments: json!({"q": "x"}),
    };
    let expected_result = ToolResult {
        call_id: "call_1".to_owned(),
        name: "echo".to_owned(),
        content: r#"echo:{"q":"x"}"#.to_owned(),
        images: Vec::new(),
        is_error: false,
    };
    assert_eq!(
        conversation.messages(),
        &[
            ProviderMessage::User("run echo".to_owned()),
            ProviderMessage::Assistant(AssistantMessage {
                text: String::new(),
                thinking: None,
                tool_calls: vec![expected_call],
            }),
            ProviderMessage::ToolResults(vec![expected_result.clone()]),
            ProviderMessage::Assistant(AssistantMessage {
                text: "done".to_owned(),
                ..AssistantMessage::default()
            }),
        ]
    );

    // The second provider call saw the tool results.
    assert_eq!(provider.request_count(), 2);
    let second = provider.request(1);
    assert_eq!(second.messages.len(), 3);
    assert_eq!(
        second.messages[2],
        ProviderMessage::ToolResults(vec![expected_result])
    );

    let events = drain(&mut rx);
    assert_eq!(
        events,
        vec![
            AgentEvent::TurnStarted,
            AgentEvent::RoundCompleted {
                stop_reason: StopReason::ToolUse
            },
            AgentEvent::ToolCallStarted {
                id: "call_1".to_owned(),
                name: "echo".to_owned()
            },
            AgentEvent::ToolCallFinished {
                id: "call_1".to_owned(),
                name: "echo".to_owned(),
                is_error: false
            },
            AgentEvent::TextDelta("done".to_owned()),
            AgentEvent::RoundCompleted {
                stop_reason: StopReason::EndTurn
            },
            AgentEvent::TurnCompleted {
                stop_reason: StopReason::EndTurn
            },
        ]
    );
}

#[tokio::test]
async fn tool_arguments_assemble_across_multibyte_fragments() {
    // Fragment boundary falls inside the JSON string, between multibyte
    // characters.
    let provider = MockProvider::new(vec![
        tool_round("call_1", "echo", &[r#"{"msg":"日本"#, r#"語 — テキスト"}"#]),
        text_round(&["ok"], StopReason::EndTurn),
    ]);
    let agent = agent(provider, echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, _rx) = unbounded_channel();

    agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    let calls = assistant_tool_calls(&conversation.messages()[1]);
    assert_eq!(calls[0].arguments, json!({"msg": "日本語 — テキスト"}));
}

#[tokio::test]
async fn tool_arguments_fallbacks_for_unparseable_and_empty() {
    let script = Script::Events(vec![
        ProviderEvent::MessageStart,
        ProviderEvent::ToolCallStart {
            index: 0,
            id: "call_1".to_owned(),
            name: "echo".to_owned(),
        },
        ProviderEvent::ToolCallArgumentsDelta {
            index: 0,
            json_fragment: "not json{{".to_owned(),
        },
        ProviderEvent::ToolCallEnd { index: 0 },
        ProviderEvent::ToolCallStart {
            index: 1,
            id: "call_2".to_owned(),
            name: "echo".to_owned(),
        },
        ProviderEvent::ToolCallEnd { index: 1 },
        ProviderEvent::Finished(StopReason::ToolUse),
    ]);
    let provider = MockProvider::new(vec![script, text_round(&["ok"], StopReason::EndTurn)]);
    let agent = agent(provider, echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, _rx) = unbounded_channel();

    agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    let calls = assistant_tool_calls(&conversation.messages()[1]);
    // Unparseable arguments keep the raw text under `_raw`.
    assert_eq!(calls[0].arguments, json!({"_raw": "not json{{"}));
    // No fragments at all become an empty object.
    assert_eq!(calls[1].arguments, json!({}));
}

#[tokio::test]
async fn usage_is_summed_across_rounds() {
    let round1 = Script::Events(vec![
        ProviderEvent::MessageStart,
        ProviderEvent::Usage(TokenUsage {
            input_tokens: 100,
            output_tokens: 10,
            cache_read_tokens: 5,
        }),
        ProviderEvent::ToolCallStart {
            index: 0,
            id: "call_1".to_owned(),
            name: "echo".to_owned(),
        },
        ProviderEvent::ToolCallEnd { index: 0 },
        ProviderEvent::Finished(StopReason::ToolUse),
    ]);
    let round2 = Script::Events(vec![
        ProviderEvent::MessageStart,
        ProviderEvent::TextDelta("ok".to_owned()),
        ProviderEvent::Usage(TokenUsage {
            input_tokens: 120,
            output_tokens: 30,
            cache_read_tokens: 50,
        }),
        ProviderEvent::Finished(StopReason::EndTurn),
    ]);
    let provider = MockProvider::new(vec![round1, round2]);
    let agent = agent(provider, echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let outcome = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    assert_eq!(
        outcome,
        TurnOutcome::Completed {
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 220,
                output_tokens: 40,
                cache_read_tokens: 55,
            },
        }
    );
    let usage_events = drain(&mut rx)
        .into_iter()
        .filter(|e| matches!(e, AgentEvent::Usage(_)))
        .count();
    assert_eq!(usage_events, 2);
}

// ---------------------------------------------------------------------------
// Retry / errors
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn transient_error_then_success_retries_with_backoff() {
    // The failed attempt streams a partial delta which is forwarded but
    // must not leak into the recorded assistant message.
    let provider = MockProvider::new(vec![
        Script::Fail(
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("gar".to_owned()),
            ],
            ProviderCallError::Transient("rate limited".to_owned()),
        ),
        text_round(&["ok"], StopReason::EndTurn),
    ]);
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let started = tokio::time::Instant::now();
    let outcome = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();
    let elapsed = started.elapsed();

    assert!(matches!(outcome, TurnOutcome::Completed { .. }));
    assert!(
        elapsed >= Duration::from_millis(500),
        "expected >= 500ms backoff, slept {elapsed:?}"
    );
    assert_eq!(provider.request_count(), 2);

    let events = drain(&mut rx);
    assert!(events.contains(&AgentEvent::Retrying {
        attempt: 2,
        error: "rate limited".to_owned(),
    }));
    // Partial delta from the failed attempt was forwarded live...
    assert!(events.contains(&AgentEvent::TextDelta("gar".to_owned())));
    // ...but the conversation only records the successful attempt.
    assert_eq!(
        conversation.messages()[1],
        ProviderMessage::Assistant(AssistantMessage {
            text: "ok".to_owned(),
            ..AssistantMessage::default()
        })
    );
}

#[tokio::test(start_paused = true)]
async fn transient_errors_exhaust_retries() {
    let transient = || Script::Fail(Vec::new(), ProviderCallError::Transient("boom".to_owned()));
    let provider = MockProvider::new(vec![transient(), transient(), transient()]);
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let err = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap_err();

    assert!(matches!(err, AgentError::Provider(_)));
    assert_eq!(provider.request_count(), 3);

    let events = drain(&mut rx);
    let retrying: Vec<u32> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::Retrying { attempt, .. } => Some(*attempt),
            _ => None,
        })
        .collect();
    assert_eq!(retrying, vec![2, 3]);
    assert!(matches!(events.last(), Some(AgentEvent::TurnFailed { .. })));
}

#[tokio::test]
async fn permanent_error_fails_immediately_without_retry() {
    let provider = MockProvider::new(vec![Script::Fail(
        Vec::new(),
        ProviderCallError::Permanent("invalid api key".to_owned()),
    )]);
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let err = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap_err();

    assert_eq!(err, AgentError::Provider("invalid api key".to_owned()));
    assert_eq!(provider.request_count(), 1);

    let events = drain(&mut rx);
    assert!(!events
        .iter()
        .any(|e| matches!(e, AgentEvent::Retrying { .. })));
    assert!(matches!(events.last(), Some(AgentEvent::TurnFailed { .. })));
}

#[tokio::test]
async fn round_limit_exceeded() {
    let scripts = (0..3)
        .map(|i| tool_round(&format!("call_{i}"), "echo", &[r#"{}"#]))
        .collect();
    let provider = MockProvider::new(scripts);
    let config = AgentConfig {
        max_rounds: 3,
        ..test_config()
    };
    let agent = agent(provider.clone(), echo_bridge(), config);
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let err = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap_err();

    assert_eq!(err, AgentError::RoundLimitExceeded(3));
    assert_eq!(provider.request_count(), 3);
    // User + 3 x (Assistant + ToolResults): every round is recorded.
    assert_eq!(conversation.messages().len(), 7);
    let events = drain(&mut rx);
    assert!(matches!(events.last(), Some(AgentEvent::TurnFailed { .. })));
}

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pre_cancelled_token_returns_without_calling_provider() {
    let provider = MockProvider::new(Vec::new());
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let cancel = CancellationToken::new();
    cancel.cancel();

    let outcome = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &cancel,
            tx,
        )
        .await
        .unwrap();

    assert_eq!(outcome, TurnOutcome::Cancelled);
    assert_eq!(provider.request_count(), 0);
    assert!(conversation.messages().is_empty());
    assert_eq!(drain(&mut rx), vec![AgentEvent::TurnCancelled]);
}

#[tokio::test]
async fn cancel_during_streaming_pushes_partial_with_marker() {
    let provider = MockProvider::new(vec![Script::Hang(vec![
        ProviderEvent::MessageStart,
        ProviderEvent::TextDelta("Hello".to_owned()),
        ProviderEvent::TextDelta(", wor".to_owned()),
    ])]);
    let agent = agent(provider, echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let cancel = CancellationToken::new();
    let canceller_token = cancel.clone();
    let watcher = async move {
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            let is_second_delta = matches!(event, AgentEvent::TextDelta(ref t) if t == ", wor");
            events.push(event);
            if is_second_delta {
                canceller_token.cancel();
            }
        }
        events
    };

    let tool_ctx = ToolContext::default();
    let (outcome, events) = tokio::join!(
        agent.run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &tool_ctx,
            &cancel,
            tx,
        ),
        watcher,
    );

    assert_eq!(outcome.unwrap(), TurnOutcome::Cancelled);
    assert_eq!(count_cancelled(&events), 1);
    assert!(matches!(events.last(), Some(AgentEvent::TurnCancelled)));
    assert_eq!(
        conversation.messages(),
        &[
            ProviderMessage::User("go".to_owned()),
            ProviderMessage::Assistant(AssistantMessage {
                text: format!("Hello, wor\n{CANCELLATION_MARKER}"),
                ..AssistantMessage::default()
            }),
        ]
    );
}

#[tokio::test]
async fn cancel_during_tool_dispatch_synthesizes_error_results() {
    let (started_tx, started_rx) = oneshot::channel();
    let mut bridge = ToolBridge::new();
    bridge
        .register(Arc::new(SlowTool {
            started: Mutex::new(Some(started_tx)),
        }))
        .unwrap();

    let provider = MockProvider::new(vec![tool_round("call_1", "slow", &[r#"{}"#])]);
    let agent = agent(provider, Arc::new(bridge), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let cancel = CancellationToken::new();
    let canceller_token = cancel.clone();
    let canceller = async move {
        started_rx.await.unwrap();
        canceller_token.cancel();
    };

    let tool_ctx = ToolContext::default();
    let (outcome, ()) = tokio::join!(
        agent.run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &tool_ctx,
            &cancel,
            tx,
        ),
        canceller,
    );

    assert_eq!(outcome.unwrap(), TurnOutcome::Cancelled);
    let events = drain(&mut rx);
    assert_eq!(count_cancelled(&events), 1);
    assert!(matches!(events.last(), Some(AgentEvent::TurnCancelled)));

    // Assistant tool request stays paired with (synthesized, error) results.
    assert_eq!(conversation.messages().len(), 3);
    match &conversation.messages()[2] {
        ProviderMessage::ToolResults(results) => {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].call_id, "call_1");
            assert!(results[0].is_error);
            assert!(results[0].content.to_lowercase().contains("cancelled"));
        }
        other => panic!("expected ToolResults, got {other:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn cancel_during_backoff_sleep() {
    let provider = MockProvider::new(vec![Script::Fail(
        Vec::new(),
        ProviderCallError::Transient("rate limited".to_owned()),
    )]);
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let cancel = CancellationToken::new();
    let canceller_token = cancel.clone();
    let watcher = async move {
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            let is_retrying = matches!(event, AgentEvent::Retrying { .. });
            events.push(event);
            if is_retrying {
                canceller_token.cancel();
            }
        }
        events
    };

    let tool_ctx = ToolContext::default();
    let (outcome, events) = tokio::join!(
        agent.run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &tool_ctx,
            &cancel,
            tx,
        ),
        watcher,
    );

    assert_eq!(outcome.unwrap(), TurnOutcome::Cancelled);
    // Only one provider call was made: the retry never ran.
    assert_eq!(provider.request_count(), 1);
    assert_eq!(count_cancelled(&events), 1);
    assert!(matches!(events.last(), Some(AgentEvent::TurnCancelled)));
}

// ---------------------------------------------------------------------------
// Context trimming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn context_trimming_evicts_summarizes_and_keeps_tool_pairs() {
    let provider = MockProvider::new(vec![text_round(&["reply"], StopReason::EndTurn)]);
    let summarizer = Arc::new(RecordingSummarizer::new(Some("stuff happened")));
    let config = AgentConfig {
        context: ContextPolicy {
            max_context_tokens: 150,
            keep_recent_messages: 3,
        },
        ..test_config()
    };
    let agent = agent(provider.clone(), echo_bridge(), config).with_summarizer(summarizer.clone());

    // Prefill: System + two big evictable messages + a tool pair that
    // straddles the protection boundary + a recent assistant message.
    let old_user = ProviderMessage::User("u".repeat(400));
    let old_assistant = ProviderMessage::Assistant(AssistantMessage {
        text: "a".repeat(400),
        ..AssistantMessage::default()
    });
    let tool_assistant = ProviderMessage::Assistant(AssistantMessage {
        text: String::new(),
        thinking: None,
        tool_calls: vec![ToolCall {
            id: "call_1".to_owned(),
            name: "echo".to_owned(),
            arguments: json!({"q": "x"}),
        }],
    });
    let tool_results = ProviderMessage::ToolResults(vec![ToolResult {
        call_id: "call_1".to_owned(),
        name: "echo".to_owned(),
        content: "r".repeat(400),
        images: Vec::new(),
        is_error: false,
    }]);
    let recent_assistant = ProviderMessage::Assistant(AssistantMessage {
        text: "recent".to_owned(),
        ..AssistantMessage::default()
    });

    let mut conversation = Conversation::with_system("sys");
    conversation.push(old_user.clone());
    conversation.push(old_assistant.clone());
    conversation.push(tool_assistant.clone());
    conversation.push(tool_results.clone());
    conversation.push(recent_assistant.clone());

    let (tx, _rx) = unbounded_channel();
    agent
        .run_turn(
            &mut conversation,
            UserInput::Text("new question".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    // The summarizer saw exactly the two evicted old messages, in order,
    // and had no prior summary to fold in on the first trim.
    let seen = summarizer.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0], vec![old_user, old_assistant]);
    drop(seen);
    assert_eq!(summarizer.previous.lock().unwrap().as_slice(), &[None]);

    // The summary lives in its own slot, not in the host-owned history.
    assert_eq!(conversation.summary(), Some("stuff happened"));

    // keep_recent_messages = 3 protects [tool_results, recent_assistant,
    // User("new question")]; the tool pair must survive intact because its
    // ToolResults half is protected.
    assert_eq!(
        conversation.messages(),
        &[
            ProviderMessage::System("sys".to_owned()),
            tool_assistant.clone(),
            tool_results.clone(),
            recent_assistant.clone(),
            ProviderMessage::User("new question".to_owned()),
            ProviderMessage::Assistant(AssistantMessage {
                text: "reply".to_owned(),
                ..AssistantMessage::default()
            }),
        ]
    );
    // ...and is spliced right after the host's system prompt on the way to
    // the provider, without displacing it.
    assert_eq!(
        conversation.provider_messages(),
        vec![
            ProviderMessage::System("sys".to_owned()),
            ProviderMessage::System("Summary of earlier conversation: stuff happened".to_owned()),
            tool_assistant,
            tool_results,
            recent_assistant,
            ProviderMessage::User("new question".to_owned()),
            ProviderMessage::Assistant(AssistantMessage {
                text: "reply".to_owned(),
                ..AssistantMessage::default()
            }),
        ]
    );

    // The provider request reflected the trimmed history.
    let request = provider.request(0);
    assert_eq!(request.messages.len(), 6);
    assert!(matches!(
        request.messages[1],
        ProviderMessage::System(ref s) if s.starts_with("Summary of earlier conversation:")
    ));
}

#[tokio::test]
async fn repeated_trimming_keeps_exactly_one_summary_and_bounded_tokens() {
    // Regression: every trim used to append a fresh, permanently
    // non-evictable `System` summary. 40 turns produced 39 of them, an
    // approx_tokens far above the budget, and no way back under it.
    const TURNS: usize = 40;
    let scripts = (0..TURNS)
        .map(|_| text_round(&["r".repeat(400).as_str()], StopReason::EndTurn))
        .collect();
    let provider = MockProvider::new(scripts);
    let summarizer = Arc::new(CountingSummarizer {
        calls: Mutex::new(Vec::new()),
    });
    let config = AgentConfig {
        context: ContextPolicy {
            max_context_tokens: 300,
            keep_recent_messages: 2,
        },
        ..test_config()
    };
    let agent = agent(provider, echo_bridge(), config).with_summarizer(summarizer.clone());

    let mut conversation = Conversation::new();
    let mut tokens_over_time = Vec::new();
    for turn in 0..TURNS {
        let (tx, _rx) = unbounded_channel();
        agent
            .run_turn(
                &mut conversation,
                UserInput::Text(format!("q{turn} {}", "u".repeat(400))),
                &ToolContext::default(),
                &CancellationToken::new(),
                tx,
            )
            .await
            .unwrap();
        tokens_over_time.push(conversation.approx_tokens());
    }

    // Exactly one engine summary, in its own slot.
    assert!(conversation.summary().is_some());
    let systems = conversation
        .provider_messages()
        .into_iter()
        .filter(|m| matches!(m, ProviderMessage::System(_)))
        .count();
    assert_eq!(
        systems, 1,
        "expected a single rolling summary System message"
    );
    assert_eq!(
        conversation.provider_messages().len(),
        conversation.messages().len() + 1
    );

    // The history is bounded: keep_recent plus the tail of the final turn,
    // never a function of the turn count.
    assert!(
        conversation.messages().len() <= 8,
        "history grew to {} messages",
        conversation.messages().len()
    );

    // ...and so is the token count. It plateaus rather than ratcheting.
    let (early, late) = (tokens_over_time[TURNS / 4], tokens_over_time[TURNS - 1]);
    assert!(
        late <= early,
        "approx_tokens ratcheted from {early} to {late} over {TURNS} turns"
    );
    assert!(
        late < 400,
        "approx_tokens {late} unbounded against a 300 budget"
    );

    // The summarizer was handed the previous summary from the second trim
    // on, so it can fold rather than restart.
    let calls = summarizer.calls.lock().unwrap();
    assert!(
        calls.len() >= 2,
        "expected repeated trimming, got {calls:?}"
    );
    assert_eq!(calls[0], None);
    assert!(
        calls[1..].iter().all(Option::is_some),
        "summarizer lost the prior summary: {calls:?}"
    );
}

#[tokio::test]
async fn cancel_during_summarizer_preserves_evicted_history() {
    // Regression: eviction used to be committed before the summarizer ran,
    // so cancelling inside the hook returned `Cancelled` over a
    // conversation whose history had already been thrown away unsummarized.
    let (started_tx, started_rx) = oneshot::channel();
    let provider = MockProvider::new(vec![text_round(&["never"], StopReason::EndTurn)]);
    let config = AgentConfig {
        context: ContextPolicy {
            max_context_tokens: 150,
            keep_recent_messages: 2,
        },
        ..test_config()
    };
    let agent = agent(provider.clone(), echo_bridge(), config).with_summarizer(Arc::new(
        HangingSummarizer {
            started: Mutex::new(Some(started_tx)),
        },
    ));

    let mut conversation = Conversation::new();
    conversation.push(ProviderMessage::User("a".repeat(400)));
    conversation.push(ProviderMessage::Assistant(AssistantMessage {
        text: "b".repeat(400),
        ..AssistantMessage::default()
    }));
    conversation.push(ProviderMessage::User("c".repeat(400)));
    let before = conversation.clone();

    let cancel = CancellationToken::new();
    let canceller_token = cancel.clone();
    let canceller = async move {
        started_rx.await.unwrap();
        canceller_token.cancel();
    };

    let (tx, mut rx) = unbounded_channel();
    let tool_ctx = ToolContext::default();
    let (outcome, ()) = tokio::join!(
        agent.run_turn(
            &mut conversation,
            UserInput::Text("new".to_owned()),
            &tool_ctx,
            &cancel,
            tx,
        ),
        canceller,
    );

    assert_eq!(outcome.unwrap(), TurnOutcome::Cancelled);
    assert_eq!(provider.request_count(), 0);
    assert_eq!(count_cancelled(&drain(&mut rx)), 1);

    // Nothing was evicted and no summary was invented: the only difference
    // from the starting state is this turn's own user message.
    let mut expected = before.messages().to_vec();
    expected.push(ProviderMessage::User("new".to_owned()));
    assert_eq!(conversation.messages(), expected.as_slice());
    assert_eq!(conversation.summary(), None);
}

// ---------------------------------------------------------------------------
// Malformed / truncated provider streams
// ---------------------------------------------------------------------------

/// A round that streams one complete tool call, then stops for `stop`.
fn tool_call_round_stopping_with(stop: StopReason) -> Script {
    Script::Events(vec![
        ProviderEvent::MessageStart,
        ProviderEvent::ToolCallStart {
            index: 0,
            id: "call_1".to_owned(),
            name: "echo".to_owned(),
        },
        ProviderEvent::ToolCallArgumentsDelta {
            index: 0,
            json_fragment: r#"{"q":"x"}"#.to_owned(),
        },
        ProviderEvent::ToolCallEnd { index: 0 },
        ProviderEvent::Finished(stop),
    ])
}

/// A bridge whose `echo` tool counts how many times it was dispatched.
fn counting_bridge() -> (Arc<ToolBridge>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let mut bridge = ToolBridge::new();
    bridge
        .register(Arc::new(FnTool::new(
            "echo",
            "echoes its arguments",
            move |call, _ctx| {
                counter.fetch_add(1, Ordering::SeqCst);
                ToolDispatchResult::ok(format!("echo:{}", call.arguments))
            },
        )))
        .unwrap();
    (Arc::new(bridge), calls)
}

#[tokio::test]
async fn tool_calls_under_a_non_tool_use_stop_are_paired_and_not_dispatched() {
    // Regression: a round that streamed tool calls but stopped for any
    // reason other than `ToolUse` used to push a lone `Assistant` message
    // carrying `tool_use` ids with no `ToolResults` after it, and report
    // success. Every subsequent request in that session was then rejected
    // by the provider — a permanently bricked conversation.
    //
    // `MaxTokens` is the reachable one: it is what Anthropic returns when
    // the token limit truncates a `tool_use` block, and what OpenAI's
    // `finish_reason: length` mid-`tool_calls` maps to.
    for stop in [
        StopReason::MaxTokens,
        StopReason::EndTurn,
        StopReason::StopSequence,
        StopReason::ContentFilter,
        StopReason::Other("length".to_owned()),
    ] {
        let provider = MockProvider::new(vec![tool_call_round_stopping_with(stop.clone())]);
        let (bridge, dispatched) = counting_bridge();
        let agent = agent(provider.clone(), bridge, test_config());
        let mut conversation = Conversation::new();
        let (tx, mut rx) = unbounded_channel();

        let outcome = agent
            .run_turn(
                &mut conversation,
                UserInput::Text("go".to_owned()),
                &ToolContext::default(),
                &CancellationToken::new(),
                tx,
            )
            .await
            .unwrap();

        // The real stop reason is still reported...
        assert_eq!(
            outcome,
            TurnOutcome::Completed {
                stop_reason: stop.clone(),
                usage: TokenUsage::default(),
            },
            "stop reason not preserved for {stop:?}"
        );
        // ...the turn did not loop for another round...
        assert_eq!(provider.request_count(), 1, "looped for {stop:?}");
        // ...the truncated request was never executed...
        assert_eq!(
            dispatched.load(Ordering::SeqCst),
            0,
            "dispatched a tool the model never finished requesting ({stop:?})"
        );

        // ...and the assistant tool request is paired with an is_error
        // result for every call id, so the history stays encodable.
        assert_eq!(conversation.messages().len(), 3, "shape wrong for {stop:?}");
        let calls = assistant_tool_calls(&conversation.messages()[1]);
        assert_eq!(calls.len(), 1);
        match &conversation.messages()[2] {
            ProviderMessage::ToolResults(results) => {
                assert_eq!(results.len(), calls.len());
                assert_eq!(results[0].call_id, calls[0].id);
                assert_eq!(results[0].name, calls[0].name);
                assert!(results[0].is_error, "result not an error for {stop:?}");
                assert!(
                    results[0].content.contains("not executed"),
                    "unhelpful content for {stop:?}: {}",
                    results[0].content
                );
            }
            other => panic!("expected ToolResults for {stop:?}, got {other:?}"),
        }

        let events = drain(&mut rx);
        assert!(
            events.contains(&AgentEvent::TurnCompleted {
                stop_reason: stop.clone()
            }),
            "missing TurnCompleted for {stop:?}"
        );
        // Nothing ran, so nothing is reported as having run.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolCallStarted { .. })),
            "announced a dispatch that never happened ({stop:?})"
        );
    }
}

#[tokio::test]
async fn tool_use_stop_with_zero_calls_completes_without_tool_results() {
    // The degenerate converse: `ToolUse` but nothing assembled. There is
    // no call to pair, so no `ToolResults` message is invented.
    let provider = MockProvider::new(vec![text_round(&["hm"], StopReason::ToolUse)]);
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, _rx) = unbounded_channel();

    let outcome = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    assert!(matches!(
        outcome,
        TurnOutcome::Completed {
            stop_reason: StopReason::ToolUse,
            ..
        }
    ));
    assert_eq!(conversation.messages().len(), 2);
    assert_eq!(provider.request_count(), 1);
}

#[tokio::test]
async fn stream_ending_without_finished_keeps_the_assembled_answer() {
    // Regression: an OpenAI-compatible server that omits the trailing
    // `data: [DONE]` (llama.cpp's server, some vLLM / LM Studio / proxy
    // setups) streamed the whole answer to the UI and then had it thrown
    // away, with the turn failing `Permanent` — never retried, nothing
    // recorded. `Ok(())` from the transport asserts a clean end of stream,
    // so the content is complete and is kept.
    let provider = MockProvider::new(vec![Script::Events(vec![
        ProviderEvent::MessageStart,
        ProviderEvent::TextDelta("The answer ".to_owned()),
        ProviderEvent::TextDelta("is 42.".to_owned()),
        // No `Finished`.
    ])]);
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let outcome = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    let implicit = StopReason::Other(bb_agent_loop::IMPLICIT_STOP_REASON.to_owned());
    assert_eq!(
        outcome,
        TurnOutcome::Completed {
            stop_reason: implicit.clone(),
            usage: TokenUsage::default(),
        }
    );
    assert_eq!(
        conversation.messages(),
        &[
            ProviderMessage::User("go".to_owned()),
            ProviderMessage::Assistant(AssistantMessage {
                text: "The answer is 42.".to_owned(),
                ..AssistantMessage::default()
            }),
        ]
    );
    // The synthesized reason is distinguishable from a provider-sent one.
    let events = drain(&mut rx);
    assert!(events.contains(&AgentEvent::TurnCompleted {
        stop_reason: implicit
    }));
    assert!(!events
        .iter()
        .any(|e| matches!(e, AgentEvent::TurnFailed { .. })));
    assert_eq!(provider.request_count(), 1);
}

#[tokio::test]
async fn stream_ending_without_finished_or_content_still_fails() {
    // Nothing to salvage: "the provider said nothing at all" stays an
    // error rather than a silently empty success.
    let provider = MockProvider::new(vec![Script::Events(vec![ProviderEvent::MessageStart])]);
    let agent = agent(provider.clone(), echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let err = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap_err();

    assert!(matches!(err, AgentError::Provider(_)), "got {err:?}");
    // Only the user message: no empty assistant turn was recorded.
    assert_eq!(conversation.messages().len(), 1);
    assert!(matches!(
        drain(&mut rx).last(),
        Some(AgentEvent::TurnFailed { .. })
    ));
}

#[tokio::test]
async fn transport_error_is_never_salvaged_into_a_completed_turn() {
    // The leniency above is scoped to `Ok(())`. A transport that reports
    // failure still fails, even with content already streamed — otherwise
    // a genuine truncation would masquerade as a finished answer.
    let provider = MockProvider::new(vec![Script::Fail(
        vec![
            ProviderEvent::MessageStart,
            ProviderEvent::TextDelta("The answer is 4".to_owned()),
        ],
        ProviderCallError::Permanent("connection reset mid-body".to_owned()),
    )]);
    let agent = agent(provider, echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, _rx) = unbounded_channel();

    let err = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap_err();

    assert_eq!(
        err,
        AgentError::Provider("connection reset mid-body".to_owned())
    );
    assert_eq!(conversation.messages().len(), 1);
}

#[tokio::test]
async fn stream_without_finished_pairs_its_tool_calls() {
    // The two leniencies compose: no terminator means we cannot know the
    // tool request was complete, so it is answered rather than dispatched
    // and the history stays encodable.
    let provider = MockProvider::new(vec![Script::Events(vec![
        ProviderEvent::MessageStart,
        ProviderEvent::ToolCallStart {
            index: 0,
            id: "call_1".to_owned(),
            name: "echo".to_owned(),
        },
        ProviderEvent::ToolCallArgumentsDelta {
            index: 0,
            json_fragment: r#"{"q":"x"}"#.to_owned(),
        },
        ProviderEvent::ToolCallEnd { index: 0 },
        // No `Finished`.
    ])]);
    let (bridge, dispatched) = counting_bridge();
    let agent = agent(provider, bridge, test_config());
    let mut conversation = Conversation::new();
    let (tx, _rx) = unbounded_channel();

    let outcome = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    assert!(matches!(outcome, TurnOutcome::Completed { .. }));
    assert_eq!(dispatched.load(Ordering::SeqCst), 0);
    assert_eq!(conversation.messages().len(), 3);
    assert!(matches!(
        &conversation.messages()[2],
        ProviderMessage::ToolResults(r) if r.len() == 1 && r[0].is_error
    ));
}

// ---------------------------------------------------------------------------
// Telemetry / metrics
// ---------------------------------------------------------------------------

/// Script for a single-round text turn that also reports token usage.
fn text_round_with_usage(text: &str, usage: TokenUsage) -> Script {
    Script::Events(vec![
        ProviderEvent::MessageStart,
        ProviderEvent::TextDelta(text.to_owned()),
        ProviderEvent::Usage(usage),
        ProviderEvent::Finished(StopReason::EndTurn),
    ])
}

/// The single terminal event of a turn, asserting the invariant that there
/// is exactly one.
fn sole_terminal(sink: &RecordingMetricsSink) -> bb_agent_loop::testing::RecordedEvent {
    let terminal = sink.terminal();
    assert_eq!(
        terminal.len(),
        1,
        "expected exactly one terminal metric event, got {terminal:?} (all: {:?})",
        sink.kinds()
    );
    terminal.into_iter().next().unwrap()
}

/// A conversation whose oldest two messages get evicted under
/// [`trim_policy`].
fn trim_prefill() -> Conversation {
    let mut conversation = Conversation::with_system("sys");
    conversation.push(ProviderMessage::User("u".repeat(400)));
    conversation.push(ProviderMessage::Assistant(AssistantMessage {
        text: "a".repeat(400),
        ..AssistantMessage::default()
    }));
    conversation.push(ProviderMessage::User("z".repeat(400)));
    conversation
}

fn trim_policy() -> ContextPolicy {
    ContextPolicy {
        max_context_tokens: 150,
        keep_recent_messages: 2,
    }
}

#[tokio::test]
async fn metrics_sink_does_not_change_turn_behaviour() {
    // Same script, three loops: no sink, a no-op sink, a recording sink.
    // The AgentEvent stream and the outcome must be byte-identical.
    let script = || {
        vec![
            tool_round("call_1", "echo", &[r#"{"q":"x"}"#]),
            text_round(&["done"], StopReason::EndTurn),
        ]
    };

    let mut streams = Vec::new();
    let mut outcomes = Vec::new();
    let sinks: Vec<Option<Arc<dyn bb_agent_loop::MetricsSink>>> = vec![
        None,
        Some(Arc::new(NoopMetricsSink)),
        Some(Arc::new(RecordingMetricsSink::new())),
    ];
    for sink in sinks {
        let provider = MockProvider::new(script());
        let mut agent = AgentLoop::new(provider, echo_bridge(), test_config());
        if let Some(sink) = sink {
            agent = agent.with_metrics(sink);
        }
        let mut conversation = Conversation::new();
        let (tx, mut rx) = unbounded_channel();
        let outcome = agent
            .run_turn(
                &mut conversation,
                UserInput::Text("run echo".to_owned()),
                &ToolContext::default(),
                &CancellationToken::new(),
                tx,
            )
            .await
            .unwrap();
        streams.push(drain(&mut rx));
        outcomes.push((outcome, conversation));
    }

    assert_eq!(streams[0], streams[1]);
    assert_eq!(streams[0], streams[2]);
    assert_eq!(outcomes[0], outcomes[1]);
    assert_eq!(outcomes[0], outcomes[2]);
}

#[tokio::test]
async fn metrics_happy_path_records_start_call_and_completion() {
    let usage = TokenUsage {
        input_tokens: 100,
        output_tokens: 10,
        cache_read_tokens: 5,
    };
    let provider = MockProvider::new(vec![text_round_with_usage("hello", usage)]);
    let (agent, sink) = recording_agent(provider, echo_bridge(), test_config());

    let outcome = run_simple_turn(&agent, &CancellationToken::new()).await;
    assert!(matches!(outcome, TurnOutcome::Completed { .. }));

    assert_eq!(
        sink.kinds(),
        ["TurnStarted", "ProviderCallCompleted", "TurnCompleted"]
    );
    assert_eq!(sink.details("TurnStarted"), ["model=test-model"]);
    assert_eq!(
        sink.details("ProviderCallCompleted"),
        ["model=test-model attempt=1 stop_reason=EndTurn \
          input_tokens=100 output_tokens=10 cache_read_tokens=5"]
    );

    let terminal = sole_terminal(&sink);
    assert_eq!(terminal.kind, "TurnCompleted");
    assert_eq!(
        terminal.detail,
        "rounds=1 stop_reason=EndTurn input_tokens=100 output_tokens=10 cache_read_tokens=5"
    );
    assert!(terminal.duration.is_some());
}

#[tokio::test]
async fn metrics_tool_turn_records_one_dispatch_per_result() {
    let provider = MockProvider::new(vec![
        multi_tool_round(&[("call_1", "echo"), ("call_2", "boom")]),
        text_round(&["done"], StopReason::EndTurn),
    ]);
    let (agent, sink) = recording_agent(provider, echo_and_boom_bridge(), test_config());

    run_simple_turn(&agent, &CancellationToken::new()).await;

    assert_eq!(
        sink.kinds(),
        [
            "TurnStarted",
            "ProviderCallCompleted",
            "ToolDispatched",
            "ToolDispatched",
            "ProviderCallCompleted",
            "TurnCompleted",
        ]
    );
    // One event per ToolResult, in result order, carrying the tool's own
    // name and the result's `is_error`.
    assert_eq!(
        sink.details("ToolDispatched"),
        ["name=echo is_error=false", "name=boom is_error=true"]
    );
    // Both events of a batch share the batch's wall duration.
    let dispatched = sink.of_kind("ToolDispatched");
    assert_eq!(dispatched[0].duration, dispatched[1].duration);
    assert!(dispatched[0].duration.is_some());

    let terminal = sole_terminal(&sink);
    assert_eq!(terminal.kind, "TurnCompleted");
    assert!(
        terminal.detail.starts_with("rounds=2 "),
        "expected two rounds, got {}",
        terminal.detail
    );
}

#[tokio::test(start_paused = true)]
async fn metrics_transient_failure_records_failure_retry_then_completion() {
    let provider = MockProvider::new(vec![
        Script::Fail(
            Vec::new(),
            ProviderCallError::Transient("rate limited".to_owned()),
        ),
        text_round(&["ok"], StopReason::EndTurn),
    ]);
    let (agent, sink) = recording_agent(provider, echo_bridge(), test_config());

    run_simple_turn(&agent, &CancellationToken::new()).await;

    assert_eq!(
        sink.kinds(),
        [
            "TurnStarted",
            "ProviderCallFailed",
            "RetryScheduled",
            "ProviderCallCompleted",
            "TurnCompleted",
        ]
    );
    assert_eq!(
        sink.details("ProviderCallFailed"),
        ["model=test-model attempt=1 transient=true error=rate limited"]
    );
    // `RetryScheduled` names the attempt about to be made and carries the
    // policy's backoff.
    let retry = sink.of_kind("RetryScheduled").remove(0);
    assert_eq!(retry.detail, "model=test-model attempt=2");
    assert_eq!(retry.duration, Some(Duration::from_millis(500)));
    // The successful attempt reports its own 1-based number.
    assert!(sink.details("ProviderCallCompleted")[0].contains("attempt=2"));
    assert_eq!(sole_terminal(&sink).kind, "TurnCompleted");
}

#[tokio::test]
async fn metrics_permanent_failure_records_failed_without_retry() {
    let provider = MockProvider::new(vec![Script::Fail(
        Vec::new(),
        ProviderCallError::Permanent("invalid api key".to_owned()),
    )]);
    let (agent, sink) = recording_agent(provider, echo_bridge(), test_config());

    let mut conversation = Conversation::new();
    let (tx, _rx) = unbounded_channel();
    let err = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, AgentError::Provider(_)));

    assert_eq!(
        sink.kinds(),
        ["TurnStarted", "ProviderCallFailed", "TurnFailed"]
    );
    assert_eq!(
        sink.details("ProviderCallFailed"),
        ["model=test-model attempt=1 transient=false error=invalid api key"]
    );
    assert_eq!(sink.count("RetryScheduled"), 0);

    let terminal = sole_terminal(&sink);
    assert_eq!(terminal.kind, "TurnFailed");
    assert_eq!(
        terminal.detail,
        "rounds=0 error=provider failure: invalid api key"
    );
}

#[tokio::test]
async fn metrics_round_limit_records_a_single_terminal_failure() {
    let scripts = (0..2)
        .map(|i| tool_round(&format!("call_{i}"), "echo", &[r#"{}"#]))
        .collect();
    let provider = MockProvider::new(scripts);
    let config = AgentConfig {
        max_rounds: 2,
        ..test_config()
    };
    let (agent, sink) = recording_agent(provider, echo_bridge(), config);

    let mut conversation = Conversation::new();
    let (tx, _rx) = unbounded_channel();
    let err = agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap_err();
    assert_eq!(err, AgentError::RoundLimitExceeded(2));

    let terminal = sole_terminal(&sink);
    assert_eq!(terminal.kind, "TurnFailed");
    assert!(
        terminal
            .detail
            .starts_with("rounds=2 error=round limit exceeded"),
        "unexpected detail: {}",
        terminal.detail
    );
}

#[tokio::test]
async fn metrics_cancelled_turn_records_exactly_one_cancellation() {
    let provider = MockProvider::new(vec![Script::Hang(vec![
        ProviderEvent::MessageStart,
        ProviderEvent::TextDelta("Hello".to_owned()),
    ])]);
    let (agent, sink) = recording_agent(provider, echo_bridge(), test_config());
    let mut conversation = Conversation::new();
    let (tx, mut rx) = unbounded_channel();

    let cancel = CancellationToken::new();
    let canceller_token = cancel.clone();
    let watcher = async move {
        while let Some(event) = rx.recv().await {
            if matches!(event, AgentEvent::TextDelta(_)) {
                canceller_token.cancel();
            }
        }
    };

    let tool_ctx = ToolContext::default();
    let (outcome, ()) = tokio::join!(
        agent.run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &tool_ctx,
            &cancel,
            tx,
        ),
        watcher,
    );

    assert_eq!(outcome.unwrap(), TurnOutcome::Cancelled);
    // The turn started, so `TurnStarted` was recorded...
    assert_eq!(sink.count("TurnStarted"), 1);
    // ...and cancellation is the one and only terminal event.
    assert_eq!(sink.count("TurnCompleted"), 0);
    let terminal = sole_terminal(&sink);
    assert_eq!(terminal.kind, "TurnCancelled");
    assert_eq!(terminal.detail, "rounds=0");
}

#[tokio::test]
async fn metrics_pre_cancelled_turn_records_only_cancellation() {
    let provider = MockProvider::new(Vec::new());
    let (agent, sink) = recording_agent(provider.clone(), echo_bridge(), test_config());

    let cancel = CancellationToken::new();
    cancel.cancel();
    let outcome = run_simple_turn(&agent, &cancel).await;

    assert_eq!(outcome, TurnOutcome::Cancelled);
    assert_eq!(provider.request_count(), 0);
    // No `TurnStarted`: the turn never started.
    assert_eq!(sink.kinds(), ["TurnCancelled"]);
    assert_eq!(sole_terminal(&sink).detail, "rounds=0");
}

#[tokio::test]
async fn metrics_context_trimming_records_eviction_count_and_summary_flag() {
    // With a summarizer that answers, the splice is reported.
    let provider = MockProvider::new(vec![text_round(&["reply"], StopReason::EndTurn)]);
    let config = AgentConfig {
        context: trim_policy(),
        ..test_config()
    };
    let (agent, sink) = recording_agent(provider, echo_bridge(), config);
    let agent = agent.with_summarizer(Arc::new(RecordingSummarizer::new(Some("stuff happened"))));
    let mut conversation = trim_prefill();
    let (tx, _rx) = unbounded_channel();
    agent
        .run_turn(
            &mut conversation,
            UserInput::Text("new question".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    assert_eq!(
        sink.details("ContextTrimmed"),
        ["evicted_messages=2 summarized=true"]
    );
    assert_eq!(sink.of_kind("ContextTrimmed")[0].duration, None);
    assert_eq!(sole_terminal(&sink).kind, "TurnCompleted");

    // Without a summarizer the same eviction reports `summarized=false`.
    let provider = MockProvider::new(vec![text_round(&["reply"], StopReason::EndTurn)]);
    let config = AgentConfig {
        context: trim_policy(),
        ..test_config()
    };
    let (bare_agent, bare_sink) = recording_agent(provider, echo_bridge(), config);
    let mut conversation = trim_prefill();
    let (tx, _rx) = unbounded_channel();
    bare_agent
        .run_turn(
            &mut conversation,
            UserInput::Text("new question".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap();

    assert_eq!(
        bare_sink.details("ContextTrimmed"),
        ["evicted_messages=2 summarized=false"]
    );
}

#[tokio::test]
async fn metrics_within_budget_turn_records_no_trimming() {
    let provider = MockProvider::new(vec![text_round(&["ok"], StopReason::EndTurn)]);
    let (agent, sink) = recording_agent(provider, echo_bridge(), test_config());
    run_simple_turn(&agent, &CancellationToken::new()).await;
    assert_eq!(sink.count("ContextTrimmed"), 0);
}

#[tokio::test(start_paused = true)]
async fn metrics_durations_follow_the_paused_clock() {
    // Provider round 1 takes 2s, the tool takes 3s, provider round 2 takes
    // 1s: 6s of turn, all of it virtual.
    let provider = MockProvider::new(vec![
        Script::Slow(
            Duration::from_secs(2),
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::ToolCallStart {
                    index: 0,
                    id: "call_1".to_owned(),
                    name: "sleepy".to_owned(),
                },
                ProviderEvent::ToolCallEnd { index: 0 },
                ProviderEvent::Finished(StopReason::ToolUse),
            ],
        ),
        Script::Slow(
            Duration::from_secs(1),
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("done".to_owned()),
                ProviderEvent::Finished(StopReason::EndTurn),
            ],
        ),
    ]);
    let mut bridge = ToolBridge::new();
    bridge
        .register(Arc::new(SleepyTool {
            delay: Duration::from_secs(3),
        }))
        .unwrap();
    let (agent, sink) = recording_agent(provider, Arc::new(bridge), test_config());

    run_simple_turn(&agent, &CancellationToken::new()).await;

    let calls = sink.of_kind("ProviderCallCompleted");
    assert_eq!(calls[0].duration, Some(Duration::from_secs(2)));
    assert_eq!(calls[1].duration, Some(Duration::from_secs(1)));
    assert_eq!(
        sink.of_kind("ToolDispatched")[0].duration,
        Some(Duration::from_secs(3))
    );

    let terminal = sole_terminal(&sink);
    assert_eq!(terminal.kind, "TurnCompleted");
    assert_eq!(terminal.duration, Some(Duration::from_secs(6)));
}

#[tokio::test(start_paused = true)]
async fn metrics_exhausted_retries_record_every_attempt_then_one_failure() {
    let transient = || Script::Fail(Vec::new(), ProviderCallError::Transient("boom".to_owned()));
    let provider = MockProvider::new(vec![transient(), transient(), transient()]);
    let (agent, sink) = recording_agent(provider, echo_bridge(), test_config());

    let mut conversation = Conversation::new();
    let (tx, _rx) = unbounded_channel();
    agent
        .run_turn(
            &mut conversation,
            UserInput::Text("go".to_owned()),
            &ToolContext::default(),
            &CancellationToken::new(),
            tx,
        )
        .await
        .unwrap_err();

    // Every attempt is reported, each retry is announced with its backoff,
    // and the turn still ends with a single terminal event.
    assert_eq!(
        sink.kinds(),
        [
            "TurnStarted",
            "ProviderCallFailed",
            "RetryScheduled",
            "ProviderCallFailed",
            "RetryScheduled",
            "ProviderCallFailed",
            "TurnFailed",
        ]
    );
    let backoffs: Vec<Option<Duration>> = sink
        .of_kind("RetryScheduled")
        .iter()
        .map(|e| e.duration)
        .collect();
    assert_eq!(
        backoffs,
        [
            Some(Duration::from_millis(500)),
            Some(Duration::from_millis(1000))
        ]
    );
    assert!(sink
        .details("ProviderCallFailed")
        .iter()
        .all(|d| d.contains("transient=true")));
    assert_eq!(sole_terminal(&sink).kind, "TurnFailed");
}
