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
//!   **dropped on encode** (signature included). On decode, a
//!   `reasoning_content` (or `reasoning`) field — the de-facto extension
//!   used by DeepSeek-style servers — is surfaced as thinking, without a
//!   signature.
//! - `role:"tool"` messages cannot carry images, so images attached to a
//!   [`ToolResult`] are re-sent in a follow-up `role:"user"` message with
//!   an explanatory note.
//! - `role:"tool"` messages have **no error flag**, so a failed
//!   [`ToolResult`] (`is_error`) has its content prefixed with
//!   [`TOOL_ERROR_PREFIX`](crate::TOOL_ERROR_PREFIX). Without it every failed tool would reach the
//!   model as a structurally successful result.
//! - Messages carrying no content are omitted, so the neutral model
//!   behaves identically here and on Anthropic (which rejects them); a
//!   request whose messages are all empty is an encode error.
//! - `data: [DONE]` is optional in practice (llama.cpp's server and some
//!   vLLM/LM Studio/proxy configs omit it), so the decoder synthesizes
//!   [`ProviderEvent::Finished`] from
//!   [`StreamDecoder::finish`](crate::StreamDecoder::finish) when the
//!   terminator never arrives.

use std::collections::BTreeSet;

use serde_json::{json, Value};

use crate::adapters::{
    data_url, empty_request_error, is_empty_message, join_url, tool_arguments_or_empty,
    tool_result_content, unknown_stop_reason, CompletedResponse, ProviderAdapter, StreamDecoder,
};
use crate::sse::SseFramer;
use crate::types::{
    AssistantMessage, HttpRequestSpec, ProviderConfig, ProviderError, ProviderEvent,
    ProviderMessage, ProviderRequest, StopReason, ThinkingBlock, TokenUsage, ToolCall, ToolResult,
    UserContent,
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
///
/// An absent payload, and the `""` OpenAI commonly sends for a
/// zero-argument tool, both normalize to an empty JSON object — see
/// [`tool_arguments_or_empty`].
fn parse_arguments(raw: Option<&Value>) -> Result<Value, ProviderError> {
    match raw {
        Some(Value::String(text)) => {
            if text.trim().is_empty() {
                return Ok(tool_arguments_or_empty(None));
            }
            serde_json::from_str(text).map_err(|err| {
                ProviderError::decode(PROVIDER, format!("invalid tool-call arguments JSON: {err}"))
            })
        }
        other => Ok(tool_arguments_or_empty(other)),
    }
}

fn encode_tool_results(results: &[ToolResult], out: &mut Vec<Value>) {
    for result in results {
        out.push(json!({
            "role": "tool",
            "tool_call_id": result.call_id,
            // No error flag on this wire format: mark failures in-band.
            "content": tool_result_content(result),
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
                        let arguments =
                            serde_json::to_string(&tool_arguments_or_empty(Some(&call.arguments)))
                                .map_err(|err| {
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
            if !system.is_empty() {
                messages.push(json!({"role": "system", "content": system}));
            }
        }
        for message in &request.messages {
            if is_empty_message(message) {
                continue;
            }
            encode_message(message, &mut messages)?;
        }
        if messages.is_empty() {
            return Err(empty_request_error(PROVIDER));
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
        // No Chat Completions server issues a thinking signature.
        let thinking = message
            .get("reasoning_content")
            .or_else(|| message.get("reasoning"))
            .and_then(Value::as_str)
            .map(ThinkingBlock::new);

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

/// Streaming decoder for Chat Completions SSE (`data:` JSON chunks,
/// usually — but not always — with a terminating `data: [DONE]`).
#[derive(Debug)]
struct OpenAiStreamDecoder {
    framer: SseFramer,
    started: bool,
    open_tool_indices: BTreeSet<usize>,
    stop_reason: Option<StopReason>,
    /// Set once a `finish_reason` arrived. Every tool call is closed at
    /// that point, so later tool deltas are ignored rather than reopening
    /// an index as a brand-new, name-less call.
    finish_seen: bool,
    /// Set once `Finished` was emitted, so `[DONE]` followed by
    /// [`StreamDecoder::finish`] cannot produce two.
    finished: bool,
}

impl OpenAiStreamDecoder {
    fn new() -> Self {
        Self {
            framer: SseFramer::new(PROVIDER),
            started: false,
            open_tool_indices: BTreeSet::new(),
            stop_reason: None,
            finish_seen: false,
            finished: false,
        }
    }

    /// Close any still-open tool calls and emit `Finished` once, carrying
    /// the `finish_reason` seen so far.
    fn emit_finished(&mut self, out: &mut Vec<ProviderEvent>) {
        if self.finished {
            return;
        }
        self.finished = true;
        while let Some(index) = self.open_tool_indices.pop_first() {
            out.push(ProviderEvent::ToolCallEnd { index });
        }
        let reason = self.stop_reason.take().unwrap_or_else(unknown_stop_reason);
        out.push(ProviderEvent::Finished(reason));
    }

    fn decode_frames(
        &mut self,
        frames: Vec<crate::sse::SseEvent>,
        out: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        for frame in frames {
            let payload = frame.data.trim();
            if payload.is_empty() {
                continue;
            }
            if payload == "[DONE]" {
                self.emit_finished(out);
                continue;
            }
            let data: Value = serde_json::from_str(payload).map_err(|err| {
                ProviderError::decode(PROVIDER, format!("invalid JSON in SSE data: {err}"))
            })?;
            self.handle_chunk(&data, out);
        }
        Ok(())
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
            self.finish_seen = true;
            self.stop_reason = Some(map_finish_reason(Some(reason)));
        }
    }

    fn handle_tool_delta(&mut self, call: &Value, out: &mut Vec<ProviderEvent>) {
        if self.finish_seen {
            // The call at this index was closed by `finish_reason`.
            // Re-opening it would fabricate a second, name-less tool call
            // and leave the real one with truncated arguments.
            return;
        }
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
        let frames = self.framer.feed(chunk)?;
        self.decode_frames(frames, &mut out)?;
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut out = Vec::new();
        // Flush a final frame that never got its blank line, then
        // synthesize `Finished` for servers that omit `data: [DONE]`.
        let frames = self.framer.finish()?;
        self.decode_frames(frames, &mut out)?;
        self.emit_finished(&mut out);
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::test_support::{
        assert_split_stable, assert_terminator_optional, config, decode_stream, full_conversation,
        image,
    };

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

    /// The same stream as [`tool_stream`] from a server that never sends
    /// `data: [DONE]` — llama.cpp's server, and some vLLM / LM Studio /
    /// proxy configurations.
    fn tool_stream_without_done() -> Vec<u8> {
        let full = String::from_utf8(tool_stream()).unwrap();
        full.replace("data: [DONE]\n\n", "").into_bytes()
    }

    #[test]
    fn stream_without_done_marker_decodes_identically() {
        assert_terminator_optional(
            &OpenAiCompatAdapter,
            &tool_stream(),
            &tool_stream_without_done(),
        );
    }

    #[test]
    fn stream_without_done_marker_still_finishes_the_turn() {
        // The regression: the full answer streamed, then the whole
        // assistant message was discarded because `Finished` never came.
        let bytes = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The answer is 42.\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        );
        assert_eq!(
            decode_stream(&OpenAiCompatAdapter, bytes.as_bytes()),
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("The answer is 42.".to_owned()),
                ProviderEvent::Finished(StopReason::EndTurn),
            ]
        );
    }

    #[test]
    fn stream_cut_before_any_finish_reason_finishes_as_unknown() {
        let bytes = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n";
        assert_eq!(
            decode_stream(&OpenAiCompatAdapter, bytes.as_bytes()).last(),
            Some(&ProviderEvent::Finished(StopReason::Other(
                "unknown".to_owned()
            )))
        );
    }

    #[test]
    fn stream_flushes_a_frame_with_no_trailing_blank_line() {
        let mut decoder = OpenAiCompatAdapter.stream_decoder();
        assert!(decoder
            .feed(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"tail\"}}]}\n")
            .unwrap()
            .is_empty());
        assert_eq!(
            decoder.finish().unwrap(),
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("tail".to_owned()),
                ProviderEvent::Finished(StopReason::Other("unknown".to_owned())),
            ]
        );
    }

    #[test]
    fn finish_reports_a_stream_cut_mid_json() {
        // Half a frame is not a completed turn; the host needs to know the
        // response was truncated.
        let mut decoder = OpenAiCompatAdapter.stream_decoder();
        decoder
            .feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"par")
            .unwrap();
        assert!(matches!(
            decoder.finish().unwrap_err(),
            ProviderError::Decode {
                provider: "openai-compat",
                ..
            }
        ));
    }

    #[test]
    fn finish_after_done_marker_emits_nothing_more() {
        let mut decoder = OpenAiCompatAdapter.stream_decoder();
        let events = decoder.feed(&tool_stream()).unwrap();
        assert_eq!(
            events.last(),
            Some(&ProviderEvent::Finished(StopReason::ToolUse))
        );
        assert!(decoder.finish().unwrap().is_empty());
        assert!(decoder.finish().unwrap().is_empty());
    }

    #[test]
    fn stream_ignores_tool_deltas_after_finish_reason() {
        // A trailing argument delta at an index `finish_reason` already
        // closed must not reopen it as a second, name-less tool call.
        let bytes = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_real\",\"function\":{\"name\":\"shell\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"stray\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let events = decode_stream(&OpenAiCompatAdapter, bytes.as_bytes());
        assert_eq!(
            events,
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::ToolCallStart {
                    index: 0,
                    id: "call_real".to_owned(),
                    name: "shell".to_owned(),
                },
                ProviderEvent::ToolCallArgumentsDelta {
                    index: 0,
                    json_fragment: "{}".to_owned(),
                },
                ProviderEvent::ToolCallEnd { index: 0 },
                ProviderEvent::Finished(StopReason::ToolUse),
            ]
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ProviderEvent::ToolCallStart { name, .. } if name.is_empty())),
            "a name-less tool call was fabricated: {events:?}"
        );
    }

    #[test]
    fn encode_marks_failed_tool_results_in_content() {
        // This wire format has no `is_error`, so the failure has to be
        // legible in the content or the model reads it as a success.
        let request = ProviderRequest {
            model: "gpt-4o".to_owned(),
            system: None,
            messages: vec![ProviderMessage::ToolResults(vec![
                ToolResult {
                    call_id: "call_1".to_owned(),
                    name: "shell".to_owned(),
                    content: "exit status 1".to_owned(),
                    images: vec![],
                    is_error: true,
                },
                ToolResult {
                    call_id: "call_2".to_owned(),
                    name: "shell".to_owned(),
                    content: "ok".to_owned(),
                    images: vec![],
                    is_error: false,
                },
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
            spec.body["messages"],
            json!([
                {"role": "tool", "tool_call_id": "call_1",
                 "content": "Error: exit status 1"},
                {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
            ])
        );
        assert!(spec.body["messages"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with(crate::TOOL_ERROR_PREFIX));
    }

    #[test]
    fn decode_absent_or_empty_arguments_becomes_empty_object() {
        for raw in [json!(""), json!("   "), Value::Null] {
            let mut function = json!({"name": "now"});
            if !raw.is_null() {
                function["arguments"] = raw.clone();
            }
            let body = json!({"choices": [{
                "message": {"tool_calls": [{"id": "c1", "type": "function",
                                            "function": function}]},
                "finish_reason": "tool_calls",
            }]});
            let response = OpenAiCompatAdapter.decode_response(&body).unwrap();
            assert_eq!(
                response.message.tool_calls[0].arguments,
                json!({}),
                "raw arguments {raw:?} should normalize to an empty object"
            );

            // ...and re-encoding must not put `"arguments": "null"` back on
            // the wire.
            let spec = OpenAiCompatAdapter
                .encode_request(
                    &config("https://api.openai.com/v1", None),
                    &ProviderRequest {
                        model: "gpt-4o".to_owned(),
                        system: None,
                        messages: vec![ProviderMessage::Assistant(response.message)],
                        tools: vec![],
                        max_tokens: None,
                        temperature: None,
                        stream: false,
                    },
                )
                .unwrap();
            assert_eq!(
                spec.body["messages"][0]["tool_calls"][0]["function"]["arguments"],
                json!("{}")
            );
        }
    }

    #[test]
    fn encode_omits_empty_messages() {
        let request = ProviderRequest {
            model: "gpt-4o".to_owned(),
            system: Some(String::new()),
            messages: vec![
                ProviderMessage::User(String::new()),
                ProviderMessage::UserMultimodal(vec![]),
                ProviderMessage::ToolResults(vec![]),
                ProviderMessage::Assistant(AssistantMessage::default()),
                ProviderMessage::User("real".to_owned()),
            ],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
        };
        let spec = OpenAiCompatAdapter
            .encode_request(&config("https://api.openai.com/v1", None), &request)
            .unwrap();
        assert_eq!(
            spec.body["messages"],
            json!([{"role": "user", "content": "real"}])
        );
    }

    #[test]
    fn encode_rejects_a_request_with_nothing_to_send() {
        let request = ProviderRequest {
            model: "gpt-4o".to_owned(),
            system: None,
            messages: vec![ProviderMessage::User(String::new())],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
        };
        assert!(matches!(
            OpenAiCompatAdapter
                .encode_request(&config("https://api.openai.com/v1", None), &request)
                .unwrap_err(),
            ProviderError::Encode {
                provider: "openai-compat",
                ..
            }
        ));
    }
}
