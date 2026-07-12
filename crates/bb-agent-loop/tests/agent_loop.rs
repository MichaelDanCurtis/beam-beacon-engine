//! Integration tests for `bb-agent-loop`, driven by a scripted
//! `MockProvider` and `bb_tool_bridge::testing::FnTool`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bb_agent_loop::{
    AgentConfig, AgentError, AgentEvent, AgentLoop, CancellationToken, ContextPolicy, Conversation,
    Provider, ProviderCallError, Summarizer, TurnOutcome, UserInput, CANCELLATION_MARKER,
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

/// Summarizer that records what it was asked to summarize.
struct RecordingSummarizer {
    seen: Mutex<Vec<Vec<ProviderMessage>>>,
    reply: Option<String>,
}

impl Summarizer for RecordingSummarizer {
    fn summarize<'a>(&'a self, evicted: &'a [ProviderMessage]) -> BoxFuture<'a, Option<String>> {
        async move {
            self.seen.lock().unwrap().push(evicted.to_vec());
            self.reply.clone()
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

fn test_config() -> AgentConfig {
    AgentConfig {
        model: "test-model".to_owned(),
        ..AgentConfig::default()
    }
}

fn agent(provider: Arc<MockProvider>, tools: Arc<ToolBridge>, config: AgentConfig) -> AgentLoop {
    AgentLoop::new(provider, tools, config)
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
    let summarizer = Arc::new(RecordingSummarizer {
        seen: Mutex::new(Vec::new()),
        reply: Some("stuff happened".to_owned()),
    });
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

    // The summarizer saw exactly the two evicted old messages, in order.
    let seen = summarizer.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0], vec![old_user, old_assistant]);
    drop(seen);

    // keep_recent_messages = 3 protects [tool_results, recent_assistant,
    // User("new question")]; the tool pair must survive intact because its
    // ToolResults half is protected.
    assert_eq!(
        conversation.messages(),
        &[
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
