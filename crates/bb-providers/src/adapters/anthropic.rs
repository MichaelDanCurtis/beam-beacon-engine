//! Anthropic Messages API adapter.
//!
//! Wire format: `POST {base}/v1/messages` with `x-api-key` and
//! `anthropic-version: 2023-06-01` headers. Streaming uses SSE with
//! `message_start` / `content_block_start` / `content_block_delta` /
//! `content_block_stop` / `message_delta` / `message_stop` events.
//!
//! Wire-format notes:
//!
//! - Thinking blocks round-trip their `signature`
//!   ([`ThinkingBlock::signature`]) — decoded from the non-streaming
//!   `signature` field and the streaming `signature_delta`, and re-encoded
//!   verbatim. The API **rejects** a replayed thinking block whose
//!   signature is missing or altered, so dropping it would break the
//!   second round of every extended-thinking tool turn.
//! - Failed tool results set the native `is_error` flag; the content is
//!   sent unmodified.
//! - Messages carrying no content are omitted rather than encoded as an
//!   empty text block or empty content array, both of which the API
//!   rejects (see [`is_empty_message`]).

use std::collections::BTreeSet;

use serde_json::{json, Value};

use crate::adapters::{
    empty_request_error, is_empty_message, join_url, tool_arguments_or_empty, unknown_stop_reason,
    CompletedResponse, ProviderAdapter, StreamDecoder,
};
use crate::sse::SseFramer;
use crate::types::{
    AssistantMessage, HttpRequestSpec, MessageImage, ProviderConfig, ProviderError, ProviderEvent,
    ProviderMessage, ProviderRequest, StopReason, ThinkingBlock, TokenUsage, ToolCall, UserContent,
};

const PROVIDER: &str = "anthropic";

/// API version sent in the `anthropic-version` header.
const API_VERSION: &str = "2023-06-01";

/// `max_tokens` is mandatory on the Messages API; this default applies when
/// [`ProviderRequest::max_tokens`] is `None`.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Adapter for the Anthropic Messages API.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicAdapter;

fn image_block(image: &MessageImage) -> Value {
    json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image.media_type,
            "data": image.data_base64,
        },
    })
}

fn map_stop_reason(raw: Option<&str>) -> StopReason {
    match raw {
        Some("end_turn") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some(other) => StopReason::Other(other.to_owned()),
        None => StopReason::Other("unknown".to_owned()),
    }
}

fn encode_message(message: &ProviderMessage, out: &mut Vec<Value>) {
    match message {
        ProviderMessage::System(_) => {} // hoisted to top-level `system`
        ProviderMessage::User(text) => {
            out.push(json!({"role": "user", "content": text}));
        }
        ProviderMessage::UserMultimodal(parts) => {
            let blocks: Vec<Value> = parts
                .iter()
                .map(|part| match part {
                    UserContent::Text(text) => json!({"type": "text", "text": text}),
                    UserContent::Image(image) => image_block(image),
                })
                .collect();
            out.push(json!({"role": "user", "content": blocks}));
        }
        ProviderMessage::Assistant(assistant) => {
            let mut blocks = Vec::new();
            if let Some(thinking) = &assistant.thinking {
                let mut block = json!({"type": "thinking", "thinking": thinking.text});
                // Anthropic requires the original signature on any thinking
                // block replayed in a later turn.
                if let Some(signature) = &thinking.signature {
                    block["signature"] = json!(signature);
                }
                blocks.push(block);
            }
            if !assistant.text.is_empty() {
                blocks.push(json!({"type": "text", "text": assistant.text}));
            }
            for call in &assistant.tool_calls {
                blocks.push(json!({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": tool_arguments_or_empty(Some(&call.arguments)),
                }));
            }
            out.push(json!({"role": "assistant", "content": blocks}));
        }
        ProviderMessage::ToolResults(results) => {
            let blocks: Vec<Value> = results
                .iter()
                .map(|result| {
                    let content: Value = if result.images.is_empty() {
                        Value::String(result.content.clone())
                    } else {
                        let mut inner = Vec::new();
                        if !result.content.is_empty() {
                            inner.push(json!({"type": "text", "text": result.content}));
                        }
                        inner.extend(result.images.iter().map(image_block));
                        Value::Array(inner)
                    };
                    let mut block = json!({
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": content,
                    });
                    if result.is_error {
                        block["is_error"] = json!(true);
                    }
                    block
                })
                .collect();
            out.push(json!({"role": "user", "content": blocks}));
        }
    }
}

impl ProviderAdapter for AnthropicAdapter {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn encode_request(
        &self,
        config: &ProviderConfig,
        request: &ProviderRequest,
    ) -> Result<HttpRequestSpec, ProviderError> {
        let mut system_parts: Vec<&str> = Vec::new();
        if let Some(system) = &request.system {
            if !system.is_empty() {
                system_parts.push(system);
            }
        }
        for message in &request.messages {
            if let ProviderMessage::System(text) = message {
                if !text.is_empty() {
                    system_parts.push(text);
                }
            }
        }

        let mut messages = Vec::new();
        for message in &request.messages {
            if is_empty_message(message) {
                continue;
            }
            encode_message(message, &mut messages);
        }
        if messages.is_empty() {
            return Err(empty_request_error(PROVIDER));
        }

        let mut body = json!({
            "model": request.model,
            "max_tokens": request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            "messages": messages,
        });
        if !system_parts.is_empty() {
            body["system"] = json!(system_parts.join("\n\n"));
        }
        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if request.stream {
            body["stream"] = json!(true);
        }

        let mut headers = vec![
            ("content-type".to_owned(), "application/json".to_owned()),
            ("anthropic-version".to_owned(), API_VERSION.to_owned()),
        ];
        if let Some(key) = &config.api_key {
            headers.push(("x-api-key".to_owned(), key.clone()));
        }
        headers.extend(config.extra_headers.iter().cloned());

        Ok(HttpRequestSpec {
            method: "POST".to_owned(),
            url: join_url(&config.base_url, "v1/messages"),
            headers,
            body,
        })
    }

    fn decode_response(&self, body: &Value) -> Result<CompletedResponse, ProviderError> {
        let content = body
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| ProviderError::decode(PROVIDER, "missing `content` array"))?;

        let mut text = String::new();
        let mut thinking = String::new();
        let mut signature: Option<String> = None;
        let mut tool_calls = Vec::new();
        for block in content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    text.push_str(block.get("text").and_then(Value::as_str).unwrap_or(""));
                }
                Some("thinking") => {
                    thinking.push_str(block.get("thinking").and_then(Value::as_str).unwrap_or(""));
                    // The signature authenticates the whole thinking block;
                    // the last one wins if a response somehow carries more.
                    if let Some(raw) = block.get("signature").and_then(Value::as_str) {
                        signature = Some(raw.to_owned());
                    }
                }
                Some("tool_use") => {
                    let id = block
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            ProviderError::decode(PROVIDER, "tool_use block missing `id`")
                        })?
                        .to_owned();
                    let name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            ProviderError::decode(PROVIDER, "tool_use block missing `name`")
                        })?
                        .to_owned();
                    let arguments = tool_arguments_or_empty(block.get("input"));
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                _ => {}
            }
        }

        let stop_reason = map_stop_reason(body.get("stop_reason").and_then(Value::as_str));
        let usage = TokenUsage {
            input_tokens: body
                .pointer("/usage/input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            output_tokens: body
                .pointer("/usage/output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            cache_read_tokens: body
                .pointer("/usage/cache_read_input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        };

        Ok(CompletedResponse {
            message: AssistantMessage {
                text,
                thinking: if thinking.is_empty() && signature.is_none() {
                    None
                } else {
                    Some(ThinkingBlock {
                        text: thinking,
                        signature,
                    })
                },
                tool_calls,
            },
            stop_reason,
            usage,
        })
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(AnthropicStreamDecoder::new())
    }
}

/// Streaming decoder for the Anthropic Messages SSE protocol.
#[derive(Debug)]
struct AnthropicStreamDecoder {
    framer: SseFramer,
    /// Content-block indices that hold `tool_use` blocks, so
    /// `content_block_stop` can emit [`ProviderEvent::ToolCallEnd`] only
    /// for tool blocks.
    tool_block_indices: BTreeSet<usize>,
    input_tokens: u64,
    cache_read_tokens: u64,
    stop_reason: Option<StopReason>,
    /// Set once `message_stop` (or the EOF flush) emitted `Finished`, so a
    /// second one is never produced.
    finished: bool,
}

impl AnthropicStreamDecoder {
    fn new() -> Self {
        Self {
            framer: SseFramer::new(PROVIDER),
            tool_block_indices: BTreeSet::new(),
            input_tokens: 0,
            cache_read_tokens: 0,
            stop_reason: None,
            finished: false,
        }
    }

    fn handle_event(
        &mut self,
        data: &Value,
        out: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        let index = data.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        match data.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                self.input_tokens = data
                    .pointer("/message/usage/input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                self.cache_read_tokens = data
                    .pointer("/message/usage/cache_read_input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                out.push(ProviderEvent::MessageStart);
            }
            Some("content_block_start") => {
                if data.pointer("/content_block/type").and_then(Value::as_str) == Some("tool_use") {
                    self.tool_block_indices.insert(index);
                    out.push(ProviderEvent::ToolCallStart {
                        index,
                        id: data
                            .pointer("/content_block/id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                        name: data
                            .pointer("/content_block/name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                    });
                }
            }
            Some("content_block_delta") => {
                match data.pointer("/delta/type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = data.pointer("/delta/text").and_then(Value::as_str) {
                            out.push(ProviderEvent::TextDelta(text.to_owned()));
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(thinking) =
                            data.pointer("/delta/thinking").and_then(Value::as_str)
                        {
                            out.push(ProviderEvent::ThinkingDelta(thinking.to_owned()));
                        }
                    }
                    Some("signature_delta") => {
                        if let Some(signature) =
                            data.pointer("/delta/signature").and_then(Value::as_str)
                        {
                            out.push(ProviderEvent::ThinkingSignature(signature.to_owned()));
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(fragment) =
                            data.pointer("/delta/partial_json").and_then(Value::as_str)
                        {
                            out.push(ProviderEvent::ToolCallArgumentsDelta {
                                index,
                                json_fragment: fragment.to_owned(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                if self.tool_block_indices.remove(&index) {
                    out.push(ProviderEvent::ToolCallEnd { index });
                }
            }
            Some("message_delta") => {
                if let Some(reason) = data.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(map_stop_reason(Some(reason)));
                }
                if let Some(output) = data.pointer("/usage/output_tokens").and_then(Value::as_u64) {
                    out.push(ProviderEvent::Usage(TokenUsage {
                        input_tokens: self.input_tokens,
                        output_tokens: output,
                        cache_read_tokens: self.cache_read_tokens,
                    }));
                }
            }
            Some("message_stop") => self.emit_finished(out),
            Some("error") => {
                return Err(ProviderError::decode(
                    PROVIDER,
                    format!(
                        "stream error event: {}",
                        data.pointer("/error/message")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown")
                    ),
                ));
            }
            _ => {} // ping and forward-compatible unknown events
        }
        Ok(())
    }

    /// Emit `Finished` once, carrying the stop reason seen so far.
    fn emit_finished(&mut self, out: &mut Vec<ProviderEvent>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let reason = self.stop_reason.take().unwrap_or_else(unknown_stop_reason);
        out.push(ProviderEvent::Finished(reason));
    }

    fn decode_frames(
        &mut self,
        frames: Vec<crate::sse::SseEvent>,
        out: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        for frame in frames {
            if frame.data.is_empty() {
                continue;
            }
            let data: Value = serde_json::from_str(&frame.data).map_err(|err| {
                ProviderError::decode(PROVIDER, format!("invalid JSON in SSE data: {err}"))
            })?;
            self.handle_event(&data, out)?;
        }
        Ok(())
    }
}

impl StreamDecoder for AnthropicStreamDecoder {
    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut out = Vec::new();
        let frames = self.framer.feed(chunk)?;
        self.decode_frames(frames, &mut out)?;
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut out = Vec::new();
        // A `message_stop` frame not followed by a blank line is still a
        // complete frame; flush it before deciding whether to synthesize.
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

    fn simple_request() -> ProviderRequest {
        ProviderRequest {
            model: "claude-sonnet-4-5".to_owned(),
            system: Some("Be terse.".to_owned()),
            messages: vec![
                ProviderMessage::User("Hi".to_owned()),
                ProviderMessage::Assistant(AssistantMessage {
                    text: "Hello!".to_owned(),
                    thinking: None,
                    tool_calls: vec![],
                }),
                ProviderMessage::User("Bye".to_owned()),
            ],
            tools: vec![],
            max_tokens: Some(1024),
            temperature: Some(0.5),
            stream: false,
        }
    }

    #[test]
    fn encode_simple_text_conversation() {
        let spec = AnthropicAdapter
            .encode_request(
                &config("https://api.anthropic.com/", Some("sk-ant")),
                &simple_request(),
            )
            .unwrap();
        assert_eq!(spec.method, "POST");
        assert_eq!(spec.url, "https://api.anthropic.com/v1/messages");
        assert!(spec
            .headers
            .contains(&("x-api-key".to_owned(), "sk-ant".to_owned())));
        assert!(spec
            .headers
            .contains(&("anthropic-version".to_owned(), "2023-06-01".to_owned())));
        assert_eq!(
            spec.body,
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "system": "Be terse.",
                "temperature": 0.5,
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
                    {"role": "user", "content": "Bye"},
                ],
            })
        );
    }

    #[test]
    fn encode_multimodal_user_message() {
        let request = ProviderRequest {
            model: "claude-sonnet-4-5".to_owned(),
            system: None,
            messages: vec![ProviderMessage::UserMultimodal(vec![
                UserContent::Text("What is this?".to_owned()),
                UserContent::Image(image()),
            ])],
            tools: vec![],
            max_tokens: Some(64),
            temperature: None,
            stream: false,
        };
        let spec = AnthropicAdapter
            .encode_request(&config("https://api.anthropic.com", None), &request)
            .unwrap();
        assert_eq!(
            spec.body,
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        }},
                    ],
                }],
            })
        );
    }

    #[test]
    fn encode_tool_call_and_result_round_trip() {
        let request = ProviderRequest {
            model: "claude-sonnet-4-5".to_owned(),
            system: None,
            messages: vec![
                ProviderMessage::User("Read a.txt".to_owned()),
                ProviderMessage::Assistant(AssistantMessage {
                    text: String::new(),
                    thinking: Some(ThinkingBlock::new("Need the file.")),
                    tool_calls: vec![ToolCall {
                        id: "toolu_1".to_owned(),
                        name: "read_file".to_owned(),
                        arguments: json!({"path": "a.txt"}),
                    }],
                }),
                ProviderMessage::ToolResults(vec![crate::types::ToolResult {
                    call_id: "toolu_1".to_owned(),
                    name: "read_file".to_owned(),
                    content: "contents".to_owned(),
                    images: vec![image()],
                    is_error: true,
                }]),
            ],
            tools: vec![crate::types::ToolSpec {
                name: "read_file".to_owned(),
                description: "Read a file".to_owned(),
                input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            }],
            max_tokens: Some(256),
            temperature: None,
            stream: true,
        };
        let spec = AnthropicAdapter
            .encode_request(&config("https://api.anthropic.com", None), &request)
            .unwrap();
        assert_eq!(
            spec.body,
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 256,
                "stream": true,
                "tools": [{
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                }],
                "messages": [
                    {"role": "user", "content": "Read a.txt"},
                    {"role": "assistant", "content": [
                        {"type": "thinking", "thinking": "Need the file."},
                        {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                         "input": {"path": "a.txt"}},
                    ]},
                    {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "contents"},
                            {"type": "image", "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGVsbG8=",
                            }},
                        ],
                        "is_error": true,
                    }]},
                ],
            })
        );
    }

    #[test]
    fn encode_every_message_variant_succeeds() {
        let spec = AnthropicAdapter
            .encode_request(
                &config("https://api.anthropic.com", Some("k")),
                &full_conversation("claude-sonnet-4-5"),
            )
            .unwrap();
        // System variants hoist into the top-level `system` string.
        assert_eq!(spec.body["system"], json!("Root system.\n\nInline system."));
    }

    #[test]
    fn decode_response_text_and_thinking() {
        let body = json!({
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "The answer is 4."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 7, "cache_read_input_tokens": 3},
        });
        let response = AnthropicAdapter.decode_response(&body).unwrap();
        assert_eq!(response.message.text, "The answer is 4.");
        assert_eq!(response.message.thinking.as_deref(), Some("hmm"));
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(
            response.usage,
            TokenUsage {
                input_tokens: 12,
                output_tokens: 7,
                cache_read_tokens: 3,
            }
        );
    }

    #[test]
    fn decode_response_tool_use() {
        let body = json!({
            "content": [
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "toolu_9", "name": "read_file",
                 "input": {"path": "b.txt"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        });
        let response = AnthropicAdapter.decode_response(&body).unwrap();
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(
            response.message.tool_calls,
            vec![ToolCall {
                id: "toolu_9".to_owned(),
                name: "read_file".to_owned(),
                arguments: json!({"path": "b.txt"}),
            }]
        );
    }

    fn tool_stream() -> Vec<u8> {
        let mut s = String::new();
        s.push_str("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25,\"cache_read_input_tokens\":5}}}\n\n");
        s.push_str("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
        s.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Caf\u{e9} \u{1f389}\"}}\n\n");
        s.push_str(
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        );
        s.push_str("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"read_file\",\"input\":{}}}\n\n");
        s.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\"}}\n\n");
        s.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"caf\u{e9}.txt\\\"}\"}}\n\n");
        s.push_str(
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        );
        s.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":12}}\n\n");
        s.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
        s.into_bytes()
    }

    #[test]
    fn stream_decodes_full_event_sequence() {
        let mut decoder = AnthropicAdapter.stream_decoder();
        let events = decoder.feed(&tool_stream()).unwrap();
        assert_eq!(
            events,
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("Caf\u{e9} \u{1f389}".to_owned()),
                ProviderEvent::ToolCallStart {
                    index: 1,
                    id: "toolu_1".to_owned(),
                    name: "read_file".to_owned(),
                },
                ProviderEvent::ToolCallArgumentsDelta {
                    index: 1,
                    json_fragment: "{\"path\":".to_owned(),
                },
                ProviderEvent::ToolCallArgumentsDelta {
                    index: 1,
                    json_fragment: "\"caf\u{e9}.txt\"}".to_owned(),
                },
                ProviderEvent::ToolCallEnd { index: 1 },
                ProviderEvent::Usage(TokenUsage {
                    input_tokens: 25,
                    output_tokens: 12,
                    cache_read_tokens: 5,
                }),
                ProviderEvent::Finished(StopReason::ToolUse),
            ]
        );
    }

    #[test]
    fn stream_is_stable_under_pathological_chunking() {
        assert_split_stable(&AnthropicAdapter, &tool_stream());
    }

    #[test]
    fn stream_thinking_and_ping_events() {
        let bytes = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
            "event: ping\ndata: {\"type\":\"ping\"}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"let me see\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        )
        .as_bytes();
        let mut decoder = AnthropicAdapter.stream_decoder();
        let events = decoder.feed(bytes).unwrap();
        assert_eq!(
            events,
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::ThinkingDelta("let me see".to_owned()),
                ProviderEvent::Usage(TokenUsage {
                    input_tokens: 1,
                    output_tokens: 2,
                    cache_read_tokens: 0,
                }),
                ProviderEvent::Finished(StopReason::EndTurn),
            ]
        );
    }

    #[test]
    fn stream_without_trailing_blank_line_decodes_identically() {
        // A `message_stop` frame the server didn't follow with a blank
        // line is still a complete frame.
        let full = tool_stream();
        let truncated = full.strip_suffix(b"\n").unwrap().to_vec();
        assert_terminator_optional(&AnthropicAdapter, &full, &truncated);
    }

    #[test]
    fn stream_without_message_stop_still_finishes_with_the_stated_reason() {
        let full = String::from_utf8(tool_stream()).unwrap();
        let truncated = full
            .replace(
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                "",
            )
            .into_bytes();
        assert_terminator_optional(&AnthropicAdapter, &tool_stream(), &truncated);
    }

    #[test]
    fn stream_cut_before_any_stop_reason_finishes_as_unknown() {
        let bytes = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
        );
        assert_eq!(
            decode_stream(&AnthropicAdapter, bytes.as_bytes()).last(),
            Some(&ProviderEvent::Finished(StopReason::Other(
                "unknown".to_owned()
            )))
        );
    }

    #[test]
    fn finish_after_message_stop_emits_nothing_more() {
        let mut decoder = AnthropicAdapter.stream_decoder();
        decoder.feed(&tool_stream()).unwrap();
        assert!(decoder.finish().unwrap().is_empty());
        assert!(decoder.finish().unwrap().is_empty());
    }

    #[test]
    fn stream_surfaces_thinking_signature() {
        let bytes = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"tick\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"SIG\"}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        )
        .as_bytes();
        assert_eq!(
            decode_stream(&AnthropicAdapter, bytes),
            vec![
                ProviderEvent::ThinkingDelta("tick".to_owned()),
                ProviderEvent::ThinkingSignature("SIG".to_owned()),
                ProviderEvent::Finished(StopReason::Other("unknown".to_owned())),
            ]
        );
        assert_split_stable(&AnthropicAdapter, bytes);
    }

    #[test]
    fn thinking_signature_survives_decode_and_re_encode() {
        // The API rejects a replayed thinking block whose signature is
        // missing, so this round trip is what makes round 2 of an
        // extended-thinking tool turn work at all.
        let body = json!({
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "SIGNATURE_ABC"},
                {"type": "text", "text": "ok"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        });
        let decoded = AnthropicAdapter.decode_response(&body).unwrap();
        assert_eq!(
            decoded.message.thinking,
            Some(ThinkingBlock::with_signature("hmm", "SIGNATURE_ABC"))
        );

        let spec = AnthropicAdapter
            .encode_request(
                &config("https://api.anthropic.com", None),
                &ProviderRequest {
                    model: "claude-sonnet-4-5".to_owned(),
                    system: None,
                    messages: vec![ProviderMessage::Assistant(decoded.message)],
                    tools: vec![],
                    max_tokens: None,
                    temperature: None,
                    stream: false,
                },
            )
            .unwrap();
        assert_eq!(
            spec.body["messages"][0]["content"][0],
            json!({"type": "thinking", "thinking": "hmm", "signature": "SIGNATURE_ABC"})
        );
    }

    #[test]
    fn thinking_without_signature_encodes_without_the_field() {
        let spec = AnthropicAdapter
            .encode_request(
                &config("https://api.anthropic.com", None),
                &ProviderRequest {
                    model: "claude-sonnet-4-5".to_owned(),
                    system: None,
                    messages: vec![ProviderMessage::Assistant(AssistantMessage {
                        text: "hi".to_owned(),
                        thinking: Some(ThinkingBlock::new("plain")),
                        tool_calls: vec![],
                    })],
                    tools: vec![],
                    max_tokens: None,
                    temperature: None,
                    stream: false,
                },
            )
            .unwrap();
        assert_eq!(
            spec.body["messages"][0]["content"][0],
            json!({"type": "thinking", "thinking": "plain"})
        );
    }

    #[test]
    fn decode_tool_use_without_input_becomes_empty_object() {
        let body = json!({
            "content": [{"type": "tool_use", "id": "t1", "name": "now"}],
            "stop_reason": "tool_use",
        });
        let decoded = AnthropicAdapter.decode_response(&body).unwrap();
        assert_eq!(decoded.message.tool_calls[0].arguments, json!({}));

        // Re-encoding must not send `"input": null`, which the API rejects.
        let spec = AnthropicAdapter
            .encode_request(
                &config("https://api.anthropic.com", None),
                &ProviderRequest {
                    model: "claude-sonnet-4-5".to_owned(),
                    system: None,
                    messages: vec![ProviderMessage::Assistant(decoded.message)],
                    tools: vec![],
                    max_tokens: None,
                    temperature: None,
                    stream: false,
                },
            )
            .unwrap();
        assert_eq!(spec.body["messages"][0]["content"][0]["input"], json!({}));
    }

    #[test]
    fn encode_omits_empty_messages_instead_of_empty_blocks() {
        // Anthropic rejects empty text blocks and empty content arrays.
        let request = ProviderRequest {
            model: "claude-sonnet-4-5".to_owned(),
            system: Some(String::new()),
            messages: vec![
                ProviderMessage::User(String::new()),
                ProviderMessage::UserMultimodal(vec![]),
                ProviderMessage::ToolResults(vec![]),
                ProviderMessage::Assistant(AssistantMessage::default()),
                ProviderMessage::System(String::new()),
                ProviderMessage::User("real".to_owned()),
            ],
            tools: vec![],
            max_tokens: Some(16),
            temperature: None,
            stream: false,
        };
        let spec = AnthropicAdapter
            .encode_request(&config("https://api.anthropic.com", None), &request)
            .unwrap();
        assert_eq!(
            spec.body,
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "real"}],
            })
        );
    }

    #[test]
    fn encode_rejects_a_request_with_nothing_to_send() {
        let request = ProviderRequest {
            model: "claude-sonnet-4-5".to_owned(),
            system: None,
            messages: vec![ProviderMessage::User(String::new())],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
        };
        assert!(matches!(
            AnthropicAdapter
                .encode_request(&config("https://api.anthropic.com", None), &request)
                .unwrap_err(),
            ProviderError::Encode {
                provider: "anthropic",
                ..
            }
        ));
    }
}
