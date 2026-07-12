//! OpenAI-compatible Chat Completions adapter.
//!
//! Wire format: `POST {base}/chat/completions` with `Authorization: Bearer`
//! (omitted when no API key is configured, which is what local servers
//! want). The base URL conventionally includes the version segment, e.g.
//! `https://api.openai.com/v1`.
//!
//! Covers OpenAI, OpenRouter, Together, Groq, Moonshot/Kimi, and custom
//! proxies. Deviations forced by the wire format:
//!
//! - Assistant `thinking` has no standard Chat Completions field and is
//!   **dropped on encode**. On decode, a `reasoning_content` (or
//!   `reasoning`) field — the de-facto extension used by DeepSeek-style
//!   servers — is surfaced as thinking.
//! - `role:"tool"` messages cannot carry images, so images attached to a
//!   [`ToolResult`] are re-sent in a follow-up `role:"user"` message with
//!   an explanatory note.

use std::collections::BTreeSet;

use serde_json::{json, Value};

use crate::adapters::{data_url, join_url, CompletedResponse, ProviderAdapter, StreamDecoder};
use crate::sse::SseFramer;
use crate::types::{
    AssistantMessage, HttpRequestSpec, ProviderConfig, ProviderError, ProviderEvent,
    ProviderMessage, ProviderRequest, StopReason, TokenUsage, ToolCall, ToolResult, UserContent,
};

const PROVIDER: &str = "openai-compat";

/// Text prepended to the follow-up user message that carries tool-result
/// images (the `tool` role cannot hold images on this wire format).
const TOOL_IMAGE_NOTE: &str = "The preceding tool result(s) produced the following image(s):";

/// Adapter for OpenAI-compatible Chat Completions endpoints.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiCompatAdapter;

fn map_finish_reason(raw: Option<&str>) -> StopReason {
    match raw {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::ContentFilter,
        Some(other) => StopReason::Other(other.to_owned()),
        None => StopReason::Other("unknown".to_owned()),
    }
}

/// Parse a tool-call `arguments` value: servers send a JSON-encoded string,
/// but some compatible servers inline the object directly.
fn parse_arguments(raw: Option<&Value>) -> Result<Value, ProviderError> {
    match raw {
        None => Ok(Value::Null),
        Some(Value::String(text)) => {
            if text.trim().is_empty() {
                return Ok(Value::Null);
            }
            serde_json::from_str(text).map_err(|err| {
                ProviderError::decode(PROVIDER, format!("invalid tool-call arguments JSON: {err}"))
            })
        }
        Some(other) => Ok(other.clone()),
    }
}

fn encode_tool_results(results: &[ToolResult], out: &mut Vec<Value>) {
    for result in results {
        out.push(json!({
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": result.content,
        }));
    }
    let images: Vec<&crate::types::MessageImage> =
        results.iter().flat_map(|result| &result.images).collect();
    if !images.is_empty() {
        let mut parts = vec![json!({"type": "text", "text": TOOL_IMAGE_NOTE})];
        parts.extend(
            images
                .iter()
                .map(|image| json!({"type": "image_url", "image_url": {"url": data_url(image)}})),
        );
        out.push(json!({"role": "user", "content": parts}));
    }
}

fn encode_message(message: &ProviderMessage, out: &mut Vec<Value>) -> Result<(), ProviderError> {
    match message {
        ProviderMessage::System(text) => {
            out.push(json!({"role": "system", "content": text}));
        }
        ProviderMessage::User(text) => {
            out.push(json!({"role": "user", "content": text}));
        }
        ProviderMessage::UserMultimodal(parts) => {
            let content: Vec<Value> = parts
                .iter()
                .map(|part| match part {
                    UserContent::Text(text) => json!({"type": "text", "text": text}),
                    UserContent::Image(image) => {
                        json!({"type": "image_url", "image_url": {"url": data_url(image)}})
                    }
                })
                .collect();
            out.push(json!({"role": "user", "content": content}));
        }
        ProviderMessage::Assistant(assistant) => {
            let mut message = json!({"role": "assistant"});
            message["content"] = if assistant.text.is_empty() && !assistant.tool_calls.is_empty() {
                Value::Null
            } else {
                Value::String(assistant.text.clone())
            };
            if !assistant.tool_calls.is_empty() {
                let calls = assistant
                    .tool_calls
                    .iter()
                    .map(|call| {
                        let arguments = serde_json::to_string(&call.arguments).map_err(|err| {
                            ProviderError::encode(
                                PROVIDER,
                                format!("tool-call arguments not serializable: {err}"),
                            )
                        })?;
                        Ok(json!({
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": arguments},
                        }))
                    })
                    .collect::<Result<Vec<Value>, ProviderError>>()?;
                message["tool_calls"] = Value::Array(calls);
            }
            out.push(message);
        }
        ProviderMessage::ToolResults(results) => encode_tool_results(results, out),
    }
    Ok(())
}

impl ProviderAdapter for OpenAiCompatAdapter {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn encode_request(
        &self,
        config: &ProviderConfig,
        request: &ProviderRequest,
    ) -> Result<HttpRequestSpec, ProviderError> {
        let mut messages = Vec::new();
        if let Some(system) = &request.system {
            messages.push(json!({"role": "system", "content": system}));
        }
        for message in &request.messages {
            encode_message(message, &mut messages)?;
        }

        let mut body = json!({
            "model": request.model,
            "messages": messages,
        });
        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools);
        }
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if request.stream {
            body["stream"] = json!(true);
        }

        let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
        if let Some(key) = &config.api_key {
            headers.push(("authorization".to_owned(), format!("Bearer {key}")));
        }
        headers.extend(config.extra_headers.iter().cloned());

        Ok(HttpRequestSpec {
            method: "POST".to_owned(),
            url: join_url(&config.base_url, "chat/completions"),
            headers,
            body,
        })
    }

    fn decode_response(&self, body: &Value) -> Result<CompletedResponse, ProviderError> {
        let choice = body
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .ok_or_else(|| ProviderError::decode(PROVIDER, "missing `choices[0]`"))?;
        let message = choice
            .get("message")
            .ok_or_else(|| ProviderError::decode(PROVIDER, "missing `choices[0].message`"))?;

        let text = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let thinking = message
            .get("reasoning_content")
            .or_else(|| message.get("reasoning"))
            .and_then(Value::as_str)
            .map(str::to_owned);

        let mut tool_calls = Vec::new();
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for (position, call) in calls.iter().enumerate() {
                tool_calls.push(ToolCall {
                    id: call
                        .get("id")
                        .and_then(Value::as_str)
                        .map_or_else(|| format!("call_{position}"), str::to_owned),
                    name: call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                    arguments: parse_arguments(call.pointer("/function/arguments"))?,
                });
            }
        }

        let stop_reason = map_finish_reason(choice.get("finish_reason").and_then(Value::as_str));
        let usage = TokenUsage {
            input_tokens: body
                .pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            output_tokens: body
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            cache_read_tokens: body
                .pointer("/usage/prompt_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        };

        Ok(CompletedResponse {
            message: AssistantMessage {
                text,
                thinking,
                tool_calls,
            },
            stop_reason,
            usage,
        })
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(OpenAiStreamDecoder::new())
    }
}

/// Streaming decoder for Chat Completions SSE (`data:` JSON chunks with a
/// terminating `data: [DONE]`).
#[derive(Debug)]
struct OpenAiStreamDecoder {
    framer: SseFramer,
    started: bool,
    open_tool_indices: BTreeSet<usize>,
    stop_reason: Option<StopReason>,
}

impl OpenAiStreamDecoder {
    fn new() -> Self {
        Self {
            framer: SseFramer::new(PROVIDER),
            started: false,
            open_tool_indices: BTreeSet::new(),
            stop_reason: None,
        }
    }

    fn handle_chunk(&mut self, data: &Value, out: &mut Vec<ProviderEvent>) {
        if !self.started {
            self.started = true;
            out.push(ProviderEvent::MessageStart);
        }

        if let Some(usage) = data.get("usage").filter(|usage| usage.is_object()) {
            out.push(ProviderEvent::Usage(TokenUsage {
                input_tokens: usage
                    .get("prompt_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                output_tokens: usage
                    .get("completion_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                cache_read_tokens: usage
                    .pointer("/prompt_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            }));
        }

        let Some(choice) = data
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return;
        };

        if let Some(delta) = choice.get("delta") {
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    out.push(ProviderEvent::TextDelta(text.to_owned()));
                }
            }
            if let Some(thinking) = delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .and_then(Value::as_str)
            {
                if !thinking.is_empty() {
                    out.push(ProviderEvent::ThinkingDelta(thinking.to_owned()));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    self.handle_tool_delta(call, out);
                }
            }
        }

        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            // All tool calls are complete once a finish reason arrives.
            while let Some(index) = self.open_tool_indices.pop_first() {
                out.push(ProviderEvent::ToolCallEnd { index });
            }
            self.stop_reason = Some(map_finish_reason(Some(reason)));
        }
    }

    fn handle_tool_delta(&mut self, call: &Value, out: &mut Vec<ProviderEvent>) {
        let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        if !self.open_tool_indices.contains(&index) {
            self.open_tool_indices.insert(index);
            out.push(ProviderEvent::ToolCallStart {
                index,
                id: call
                    .get("id")
                    .and_then(Value::as_str)
                    .map_or_else(|| format!("call_{index}"), str::to_owned),
                name: call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
            });
        }
        if let Some(fragment) = call.pointer("/function/arguments").and_then(Value::as_str) {
            if !fragment.is_empty() {
                out.push(ProviderEvent::ToolCallArgumentsDelta {
                    index,
                    json_fragment: fragment.to_owned(),
                });
            }
        }
    }
}

impl StreamDecoder for OpenAiStreamDecoder {
    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut out = Vec::new();
        for frame in self.framer.feed(chunk)? {
            let payload = frame.data.trim();
            if payload.is_empty() {
                continue;
            }
            if payload == "[DONE]" {
                while let Some(index) = self.open_tool_indices.pop_first() {
                    out.push(ProviderEvent::ToolCallEnd { index });
                }
                let reason = self
                    .stop_reason
                    .take()
                    .unwrap_or(StopReason::Other("unknown".to_owned()));
                out.push(ProviderEvent::Finished(reason));
                continue;
            }
            let data: Value = serde_json::from_str(payload).map_err(|err| {
                ProviderError::decode(PROVIDER, format!("invalid JSON in SSE data: {err}"))
            })?;
            self.handle_chunk(&data, &mut out);
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::test_support::{assert_split_stable, config, full_conversation, image};

    #[test]
    fn encode_simple_text_conversation() {
        let request = ProviderRequest {
            model: "gpt-4o".to_owned(),
            system: Some("Be terse.".to_owned()),
            messages: vec![
                ProviderMessage::User("Hi".to_owned()),
                ProviderMessage::Assistant(AssistantMessage {
                    text: "Hello!".to_owned(),
                    thinking: None,
                    tool_calls: vec![],
                }),
            ],
            tools: vec![],
            max_tokens: Some(128),
            temperature: Some(0.5),
            stream: false,
        };
        let spec = OpenAiCompatAdapter
            .encode_request(&config("https://api.openai.com/v1", Some("sk-x")), &request)
            .unwrap();
        assert_eq!(spec.url, "https://api.openai.com/v1/chat/completions");
        assert!(spec
            .headers
            .contains(&("authorization".to_owned(), "Bearer sk-x".to_owned())));
        assert_eq!(
            spec.body,
            json!({
                "model": "gpt-4o",
                "max_tokens": 128,
                "temperature": 0.5,
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ],
            })
        );
    }

    #[test]
    fn encode_multimodal_uses_data_urls() {
        let request = ProviderRequest {
            model: "gpt-4o".to_owned(),
            system: None,
            messages: vec![ProviderMessage::UserMultimodal(vec![
                UserContent::Text("Describe:".to_owned()),
                UserContent::Image(image()),
            ])],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
        };
        let spec = OpenAiCompatAdapter
            .encode_request(&config("https://api.openai.com/v1", None), &request)
            .unwrap();
        assert_eq!(
            spec.body,
            json!({
                "model": "gpt-4o",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe:"},
                        {"type": "image_url",
                         "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                    ],
                }],
            })
        );
    }

    #[test]
    fn encode_tool_call_and_result_round_trip() {
        let request = ProviderRequest {
            model: "gpt-4o".to_owned(),
            system: None,
            messages: vec![
                ProviderMessage::Assistant(AssistantMessage {
                    text: String::new(),
                    thinking: None,
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_owned(),
                        name: "get_weather".to_owned(),
                        arguments: json!({"city": "Paris"}),
                    }],
                }),
                ProviderMessage::ToolResults(vec![ToolResult {
                    call_id: "call_1".to_owned(),
                    name: "get_weather".to_owned(),
                    content: "Sunny".to_owned(),
                    images: vec![image()],
                    is_error: false,
                }]),
            ],
            tools: vec![crate::types::ToolSpec {
                name: "get_weather".to_owned(),
                description: "Get weather".to_owned(),
                input_schema: json!({"type": "object"}),
            }],
            max_tokens: None,
            temperature: None,
            stream: true,
        };
        let spec = OpenAiCompatAdapter
            .encode_request(&config("http://localhost:8080/v1", None), &request)
            .unwrap();
        assert_eq!(
            spec.body,
            json!({
                "model": "gpt-4o",
                "stream": true,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }],
                "messages": [
                    {"role": "assistant", "content": null, "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather",
                                     "arguments": "{\"city\":\"Paris\"}"},
                    }]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
                    // Tool messages can't carry images -> follow-up user message.
                    {"role": "user", "content": [
                        {"type": "text", "text": TOOL_IMAGE_NOTE},
                        {"type": "image_url",
                         "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                    ]},
                ],
            })
        );
    }

    #[test]
    fn encode_every_message_variant_succeeds() {
        OpenAiCompatAdapter
            .encode_request(
                &config("https://api.openai.com/v1", Some("k")),
                &full_conversation("gpt-4o"),
            )
            .unwrap();
    }

    #[test]
    fn no_authorization_header_without_api_key() {
        let spec = OpenAiCompatAdapter
            .encode_request(
                &config("http://localhost:11434/v1", None),
                &full_conversation("llama3"),
            )
            .unwrap();
        assert!(spec.headers.iter().all(|(name, _)| name != "authorization"));
    }

    #[test]
    fn decode_response_with_tool_calls_and_usage() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "thinking...",
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "get_weather",
                                     "arguments": "{\"city\":\"Oslo\"}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 9,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        });
        let response = OpenAiCompatAdapter.decode_response(&body).unwrap();
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.message.thinking.as_deref(), Some("thinking..."));
        assert_eq!(
            response.message.tool_calls,
            vec![ToolCall {
                id: "call_9".to_owned(),
                name: "get_weather".to_owned(),
                arguments: json!({"city": "Oslo"}),
            }]
        );
        assert_eq!(
            response.usage,
            TokenUsage {
                input_tokens: 30,
                output_tokens: 9,
                cache_read_tokens: 4,
            }
        );
    }

    #[test]
    fn decode_response_plain_text() {
        let body = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        });
        let response = OpenAiCompatAdapter.decode_response(&body).unwrap();
        assert_eq!(response.message.text, "Hi there");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
    }

    fn tool_stream() -> Vec<u8> {
        let mut s = String::new();
        s.push_str("data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n");
        s.push_str("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"caf\u{e9} \u{1f389}\"}}]}\n\n");
        s.push_str("data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"\"}}]}}]}\n\n");
        s.push_str("data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"q\\\":\\\"caf\u{e9}\\\"}\"}}]}}]}\n\n");
        s.push_str(
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        );
        s.push_str("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":6,\"prompt_tokens_details\":{\"cached_tokens\":2}}}\n\n");
        s.push_str("data: [DONE]\n\n");
        s.into_bytes()
    }

    #[test]
    fn stream_decodes_full_event_sequence() {
        let mut decoder = OpenAiCompatAdapter.stream_decoder();
        let events = decoder.feed(&tool_stream()).unwrap();
        assert_eq!(
            events,
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("caf\u{e9} \u{1f389}".to_owned()),
                ProviderEvent::ToolCallStart {
                    index: 0,
                    id: "call_a".to_owned(),
                    name: "lookup".to_owned(),
                },
                ProviderEvent::ToolCallArgumentsDelta {
                    index: 0,
                    json_fragment: "{\"q\":\"caf\u{e9}\"}".to_owned(),
                },
                ProviderEvent::ToolCallEnd { index: 0 },
                ProviderEvent::Usage(TokenUsage {
                    input_tokens: 15,
                    output_tokens: 6,
                    cache_read_tokens: 2,
                }),
                ProviderEvent::Finished(StopReason::ToolUse),
            ]
        );
    }

    #[test]
    fn stream_is_stable_under_pathological_chunking() {
        assert_split_stable(&OpenAiCompatAdapter, &tool_stream());
    }

    #[test]
    fn stream_maps_finish_reasons() {
        for (raw, mapped) in [
            ("stop", StopReason::EndTurn),
            ("length", StopReason::MaxTokens),
            ("content_filter", StopReason::ContentFilter),
        ] {
            let bytes = format!(
                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"x\"}},\"finish_reason\":\"{raw}\"}}]}}\n\ndata: [DONE]\n\n"
            );
            let mut decoder = OpenAiCompatAdapter.stream_decoder();
            let events = decoder.feed(bytes.as_bytes()).unwrap();
            assert_eq!(events.last(), Some(&ProviderEvent::Finished(mapped)));
        }
    }
}
