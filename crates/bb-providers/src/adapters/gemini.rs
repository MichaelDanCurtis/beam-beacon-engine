//! Google Gemini `generateContent` adapter.
//!
//! Wire format: `POST {base}/v1beta/models/{model}:generateContent`
//! (`:streamGenerateContent?alt=sse` when streaming) with an
//! `x-goog-api-key` header. Streaming frames are SSE `data:` lines each
//! carrying a complete `GenerateContentResponse`.
//!
//! Wire-format notes:
//!
//! - Gemini `functionCall` parts carry **no id**, so stable ids are
//!   synthesized as `call_{index}` in call order (per response). Hosts must
//!   echo the tool *name* back via [`ToolResult::name`]; the synthesized id
//!   is only for neutral-model bookkeeping.
//! - `functionResponse.response` must be a JSON object: successful results
//!   encode as `{"content": <text>}`, failed results as
//!   `{"error": <text>}`.
//! - Tool-result images ride along as sibling `inline_data` parts in the
//!   same user turn.
//! - Assistant `thinking` has no input slot and is dropped on encode
//!   (signature included); on decode, parts flagged `"thought": true` are
//!   surfaced as thinking, without a signature.
//! - Messages carrying no content are omitted, matching the neutral-model
//!   rule the other adapters follow; a request whose messages are all
//!   empty is an encode error.

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

const PROVIDER: &str = "gemini";

/// Adapter for the Google Gemini `generateContent` API.
#[derive(Debug, Clone, Copy, Default)]
pub struct GeminiAdapter;

fn inline_data_part(image: &MessageImage) -> Value {
    json!({
        "inline_data": {
            "mime_type": image.media_type,
            "data": image.data_base64,
        },
    })
}

fn map_finish_reason(raw: &str, saw_tool_call: bool) -> StopReason {
    match raw {
        "STOP" => {
            if saw_tool_call {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }
        }
        "MAX_TOKENS" => StopReason::MaxTokens,
        "SAFETY" => StopReason::ContentFilter,
        other => StopReason::Other(other.to_owned()),
    }
}

fn usage_from_metadata(metadata: &Value) -> TokenUsage {
    TokenUsage {
        input_tokens: metadata
            .get("promptTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: metadata
            .get("candidatesTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cache_read_tokens: metadata
            .get("cachedContentTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    }
}

fn encode_content(message: &ProviderMessage, out: &mut Vec<Value>) {
    match message {
        ProviderMessage::System(_) => {} // hoisted to system_instruction
        ProviderMessage::User(text) => {
            out.push(json!({"role": "user", "parts": [{"text": text}]}));
        }
        ProviderMessage::UserMultimodal(parts) => {
            let parts: Vec<Value> = parts
                .iter()
                .map(|part| match part {
                    UserContent::Text(text) => json!({"text": text}),
                    UserContent::Image(image) => inline_data_part(image),
                })
                .collect();
            out.push(json!({"role": "user", "parts": parts}));
        }
        ProviderMessage::Assistant(assistant) => {
            let mut parts = Vec::new();
            if !assistant.text.is_empty() {
                parts.push(json!({"text": assistant.text}));
            }
            for call in &assistant.tool_calls {
                parts.push(json!({
                    "functionCall": {
                        "name": call.name,
                        "args": tool_arguments_or_empty(Some(&call.arguments)),
                    },
                }));
            }
            out.push(json!({"role": "model", "parts": parts}));
        }
        ProviderMessage::ToolResults(results) => {
            let mut parts = Vec::new();
            for result in results {
                let response = if result.is_error {
                    json!({"error": result.content})
                } else {
                    json!({"content": result.content})
                };
                parts.push(json!({
                    "functionResponse": {"name": result.name, "response": response},
                }));
                parts.extend(result.images.iter().map(inline_data_part));
            }
            out.push(json!({"role": "user", "parts": parts}));
        }
    }
}

/// Decode one `candidates[0].content.parts` array into neutral pieces.
/// `tool_index` carries the running functionCall counter across frames.
fn decode_parts(
    parts: &[Value],
    tool_index: &mut usize,
) -> Result<(String, String, Vec<ToolCall>), ProviderError> {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut tool_calls = Vec::new();
    for part in parts {
        if let Some(call) = part.get("functionCall") {
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| ProviderError::decode(PROVIDER, "functionCall missing `name`"))?
                .to_owned();
            let arguments = tool_arguments_or_empty(call.get("args"));
            tool_calls.push(ToolCall {
                id: format!("call_{tool_index}", tool_index = *tool_index),
                name,
                arguments,
            });
            *tool_index += 1;
        } else if let Some(part_text) = part.get("text").and_then(Value::as_str) {
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                thinking.push_str(part_text);
            } else {
                text.push_str(part_text);
            }
        }
    }
    Ok((text, thinking, tool_calls))
}

impl ProviderAdapter for GeminiAdapter {
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

        let mut contents = Vec::new();
        for message in &request.messages {
            if is_empty_message(message) {
                continue;
            }
            encode_content(message, &mut contents);
        }
        if contents.is_empty() {
            return Err(empty_request_error(PROVIDER));
        }

        let mut body = json!({"contents": contents});
        if !system_parts.is_empty() {
            body["system_instruction"] = json!({"parts": [{"text": system_parts.join("\n\n")}]});
        }
        if !request.tools.is_empty() {
            let declarations: Vec<Value> = request
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    })
                })
                .collect();
            body["tools"] = json!([{"function_declarations": declarations}]);
        }
        let mut generation_config = serde_json::Map::new();
        if let Some(max_tokens) = request.max_tokens {
            generation_config.insert("maxOutputTokens".to_owned(), json!(max_tokens));
        }
        if let Some(temperature) = request.temperature {
            generation_config.insert("temperature".to_owned(), json!(temperature));
        }
        if !generation_config.is_empty() {
            body["generationConfig"] = Value::Object(generation_config);
        }

        let path = if request.stream {
            format!(
                "v1beta/models/{model}:streamGenerateContent?alt=sse",
                model = request.model
            )
        } else {
            format!(
                "v1beta/models/{model}:generateContent",
                model = request.model
            )
        };

        let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
        if let Some(key) = &config.api_key {
            headers.push(("x-goog-api-key".to_owned(), key.clone()));
        }
        headers.extend(config.extra_headers.iter().cloned());

        Ok(HttpRequestSpec {
            method: "POST".to_owned(),
            url: join_url(&config.base_url, &path),
            headers,
            body,
        })
    }

    fn decode_response(&self, body: &Value) -> Result<CompletedResponse, ProviderError> {
        let candidate = body
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
            .ok_or_else(|| ProviderError::decode(PROVIDER, "missing `candidates[0]`"))?;

        let empty = Vec::new();
        let parts = candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
            .unwrap_or(&empty);
        let mut tool_index = 0usize;
        let (text, thinking, tool_calls) = decode_parts(parts, &mut tool_index)?;

        let stop_reason = match candidate.get("finishReason").and_then(Value::as_str) {
            Some(reason) => map_finish_reason(reason, !tool_calls.is_empty()),
            None => StopReason::Other("unknown".to_owned()),
        };
        let usage = body
            .get("usageMetadata")
            .map(usage_from_metadata)
            .unwrap_or_default();

        Ok(CompletedResponse {
            message: AssistantMessage {
                text,
                // Gemini issues no thinking signature.
                thinking: if thinking.is_empty() {
                    None
                } else {
                    Some(ThinkingBlock::new(thinking))
                },
                tool_calls,
            },
            stop_reason,
            usage,
        })
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(GeminiStreamDecoder::new())
    }
}

/// Streaming decoder for `:streamGenerateContent?alt=sse`. Each SSE frame
/// carries a complete `GenerateContentResponse`; `functionCall` parts
/// arrive whole, so each one emits start / arguments / end back-to-back
/// with a synthesized `call_{index}` id.
#[derive(Debug)]
struct GeminiStreamDecoder {
    framer: SseFramer,
    started: bool,
    next_tool_index: usize,
    saw_tool_call: bool,
    latest_usage: Option<TokenUsage>,
    /// Set once a `finishReason` (or the EOF flush) emitted `Finished`.
    finished: bool,
}

impl GeminiStreamDecoder {
    fn new() -> Self {
        Self {
            framer: SseFramer::new(PROVIDER),
            started: false,
            next_tool_index: 0,
            saw_tool_call: false,
            latest_usage: None,
            finished: false,
        }
    }

    /// Emit any pending usage plus `Finished`, at most once.
    fn emit_finished(&mut self, reason: StopReason, out: &mut Vec<ProviderEvent>) {
        if self.finished {
            return;
        }
        self.finished = true;
        if let Some(usage) = self.latest_usage.take() {
            out.push(ProviderEvent::Usage(usage));
        }
        out.push(ProviderEvent::Finished(reason));
    }

    fn decode_frames(
        &mut self,
        frames: Vec<crate::sse::SseEvent>,
        out: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        for frame in frames {
            if frame.data.trim().is_empty() {
                continue;
            }
            let data: Value = serde_json::from_str(&frame.data).map_err(|err| {
                ProviderError::decode(PROVIDER, format!("invalid JSON in SSE data: {err}"))
            })?;
            self.handle_frame(&data, out)?;
        }
        Ok(())
    }

    fn handle_frame(
        &mut self,
        data: &Value,
        out: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        if !self.started {
            self.started = true;
            out.push(ProviderEvent::MessageStart);
        }

        if let Some(metadata) = data.get("usageMetadata") {
            self.latest_usage = Some(usage_from_metadata(metadata));
        }

        let Some(candidate) = data
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
        else {
            return Ok(());
        };

        if let Some(parts) = candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(call) = part.get("functionCall") {
                    let index = self.next_tool_index;
                    self.next_tool_index += 1;
                    self.saw_tool_call = true;
                    let name = call
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    let arguments = call.get("args").cloned().unwrap_or(Value::Null);
                    let fragment = serde_json::to_string(&arguments).map_err(|err| {
                        ProviderError::decode(
                            PROVIDER,
                            format!("functionCall args not serializable: {err}"),
                        )
                    })?;
                    out.push(ProviderEvent::ToolCallStart {
                        index,
                        id: format!("call_{index}"),
                        name,
                    });
                    out.push(ProviderEvent::ToolCallArgumentsDelta {
                        index,
                        json_fragment: fragment,
                    });
                    out.push(ProviderEvent::ToolCallEnd { index });
                } else if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if text.is_empty() {
                        continue;
                    }
                    if part.get("thought").and_then(Value::as_bool) == Some(true) {
                        out.push(ProviderEvent::ThinkingDelta(text.to_owned()));
                    } else {
                        out.push(ProviderEvent::TextDelta(text.to_owned()));
                    }
                }
            }
        }

        if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
            let reason = map_finish_reason(reason, self.saw_tool_call);
            self.emit_finished(reason, out);
        }
        Ok(())
    }
}

impl StreamDecoder for GeminiStreamDecoder {
    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut out = Vec::new();
        let frames = self.framer.feed(chunk)?;
        self.decode_frames(frames, &mut out)?;
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut out = Vec::new();
        // The last `data:` frame often arrives without its blank line;
        // flush it before deciding whether to synthesize a terminator.
        let frames = self.framer.finish()?;
        self.decode_frames(frames, &mut out)?;
        self.emit_finished(unknown_stop_reason(), &mut out);
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
    use crate::types::{ToolResult, ToolSpec};

    #[test]
    fn encode_full_request_shape() {
        let request = ProviderRequest {
            model: "gemini-2.0-flash".to_owned(),
            system: Some("Be terse.".to_owned()),
            messages: vec![
                ProviderMessage::UserMultimodal(vec![
                    UserContent::Text("What is this?".to_owned()),
                    UserContent::Image(image()),
                ]),
                ProviderMessage::Assistant(AssistantMessage {
                    text: "Checking.".to_owned(),
                    thinking: None,
                    tool_calls: vec![ToolCall {
                        id: "call_0".to_owned(),
                        name: "lookup".to_owned(),
                        arguments: json!({"q": "cat"}),
                    }],
                }),
                ProviderMessage::ToolResults(vec![ToolResult {
                    call_id: "call_0".to_owned(),
                    name: "lookup".to_owned(),
                    content: "A cat.".to_owned(),
                    images: vec![image()],
                    is_error: false,
                }]),
            ],
            tools: vec![ToolSpec {
                name: "lookup".to_owned(),
                description: "Look things up".to_owned(),
                input_schema: json!({"type": "object"}),
            }],
            max_tokens: Some(200),
            temperature: Some(0.5),
            stream: false,
        };
        let spec = GeminiAdapter
            .encode_request(
                &config("https://generativelanguage.googleapis.com", Some("g-key")),
                &request,
            )
            .unwrap();
        assert_eq!(
            spec.url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        );
        assert!(spec
            .headers
            .contains(&("x-goog-api-key".to_owned(), "g-key".to_owned())));
        assert_eq!(
            spec.body,
            json!({
                "system_instruction": {"parts": [{"text": "Be terse."}]},
                "contents": [
                    {"role": "user", "parts": [
                        {"text": "What is this?"},
                        {"inline_data": {"mime_type": "image/png", "data": "aGVsbG8="}},
                    ]},
                    {"role": "model", "parts": [
                        {"text": "Checking."},
                        {"functionCall": {"name": "lookup", "args": {"q": "cat"}}},
                    ]},
                    {"role": "user", "parts": [
                        {"functionResponse": {"name": "lookup",
                                              "response": {"content": "A cat."}}},
                        {"inline_data": {"mime_type": "image/png", "data": "aGVsbG8="}},
                    ]},
                ],
                "tools": [{"function_declarations": [{
                    "name": "lookup",
                    "description": "Look things up",
                    "parameters": {"type": "object"},
                }]}],
                "generationConfig": {"maxOutputTokens": 200, "temperature": 0.5},
            })
        );
    }

    #[test]
    fn encode_stream_url_uses_sse_alt() {
        let mut request = full_conversation("gemini-2.0-flash");
        request.stream = true;
        let spec = GeminiAdapter
            .encode_request(
                &config("https://generativelanguage.googleapis.com/", None),
                &request,
            )
            .unwrap();
        assert_eq!(
            spec.url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn encode_error_tool_result_uses_error_key() {
        let request = ProviderRequest {
            model: "gemini-2.0-flash".to_owned(),
            system: None,
            messages: vec![ProviderMessage::ToolResults(vec![ToolResult {
                call_id: "call_0".to_owned(),
                name: "lookup".to_owned(),
                content: "boom".to_owned(),
                images: vec![],
                is_error: true,
            }])],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
        };
        let spec = GeminiAdapter
            .encode_request(&config("https://g", None), &request)
            .unwrap();
        assert_eq!(
            spec.body["contents"][0]["parts"][0],
            json!({"functionResponse": {"name": "lookup", "response": {"error": "boom"}}})
        );
    }

    #[test]
    fn encode_every_message_variant_succeeds() {
        GeminiAdapter
            .encode_request(
                &config("https://generativelanguage.googleapis.com", Some("k")),
                &full_conversation("gemini-2.0-flash"),
            )
            .unwrap();
    }

    #[test]
    fn decode_response_synthesizes_tool_call_ids() {
        let body = json!({
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "thinking hard", "thought": true},
                    {"text": "Using tools."},
                    {"functionCall": {"name": "lookup", "args": {"q": "a"}}},
                    {"functionCall": {"name": "fetch", "args": {"url": "b"}}},
                ]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 5,
                "cachedContentTokenCount": 1,
            },
        });
        let response = GeminiAdapter.decode_response(&body).unwrap();
        assert_eq!(response.message.text, "Using tools.");
        assert_eq!(response.message.thinking.as_deref(), Some("thinking hard"));
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(
            response.message.tool_calls,
            vec![
                ToolCall {
                    id: "call_0".to_owned(),
                    name: "lookup".to_owned(),
                    arguments: json!({"q": "a"}),
                },
                ToolCall {
                    id: "call_1".to_owned(),
                    name: "fetch".to_owned(),
                    arguments: json!({"url": "b"}),
                },
            ]
        );
        assert_eq!(
            response.usage,
            TokenUsage {
                input_tokens: 8,
                output_tokens: 5,
                cache_read_tokens: 1,
            }
        );
    }

    #[test]
    fn decode_response_maps_safety_finish() {
        let body = json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "no"}]},
                "finishReason": "SAFETY",
            }],
        });
        let response = GeminiAdapter.decode_response(&body).unwrap();
        assert_eq!(response.stop_reason, StopReason::ContentFilter);
    }

    fn tool_stream() -> Vec<u8> {
        let mut s = String::new();
        s.push_str("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Bonjour caf\u{e9} \u{1f389}\"}]}}]}\n\n");
        s.push_str("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"q\":\"caf\u{e9}\"}}}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":7,\"candidatesTokenCount\":3}}\n\n");
        s.into_bytes()
    }

    #[test]
    fn stream_decodes_full_event_sequence() {
        let mut decoder = GeminiAdapter.stream_decoder();
        let events = decoder.feed(&tool_stream()).unwrap();
        assert_eq!(
            events,
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("Bonjour caf\u{e9} \u{1f389}".to_owned()),
                ProviderEvent::ToolCallStart {
                    index: 0,
                    id: "call_0".to_owned(),
                    name: "lookup".to_owned(),
                },
                ProviderEvent::ToolCallArgumentsDelta {
                    index: 0,
                    json_fragment: "{\"q\":\"caf\u{e9}\"}".to_owned(),
                },
                ProviderEvent::ToolCallEnd { index: 0 },
                ProviderEvent::Usage(TokenUsage {
                    input_tokens: 7,
                    output_tokens: 3,
                    cache_read_tokens: 0,
                }),
                ProviderEvent::Finished(StopReason::ToolUse),
            ]
        );
    }

    #[test]
    fn stream_is_stable_under_pathological_chunking() {
        assert_split_stable(&GeminiAdapter, &tool_stream());
    }

    #[test]
    fn stream_without_trailing_blank_line_decodes_identically() {
        let full = tool_stream();
        let truncated = full.strip_suffix(b"\n").unwrap().to_vec();
        assert_terminator_optional(&GeminiAdapter, &full, &truncated);
    }

    #[test]
    fn stream_without_finish_reason_still_finishes() {
        let bytes = concat!(
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",",
            "\"parts\":[{\"text\":\"partial\"}]}}],",
            "\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":1}}\n\n",
        )
        .as_bytes();
        assert_eq!(
            decode_stream(&GeminiAdapter, bytes),
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("partial".to_owned()),
                ProviderEvent::Usage(TokenUsage {
                    input_tokens: 4,
                    output_tokens: 1,
                    cache_read_tokens: 0,
                }),
                ProviderEvent::Finished(StopReason::Other("unknown".to_owned())),
            ]
        );
    }

    #[test]
    fn finish_after_finish_reason_emits_nothing_more() {
        let mut decoder = GeminiAdapter.stream_decoder();
        decoder.feed(&tool_stream()).unwrap();
        assert!(decoder.finish().unwrap().is_empty());
        assert!(decoder.finish().unwrap().is_empty());
    }

    #[test]
    fn decode_function_call_without_args_becomes_empty_object() {
        let body = json!({
            "candidates": [{
                "content": {"parts": [{"functionCall": {"name": "now"}}]},
                "finishReason": "STOP",
            }],
        });
        let decoded = GeminiAdapter.decode_response(&body).unwrap();
        assert_eq!(decoded.message.tool_calls[0].arguments, json!({}));

        let spec = GeminiAdapter
            .encode_request(
                &config("https://g", None),
                &ProviderRequest {
                    model: "gemini-2.0-flash".to_owned(),
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
            spec.body["contents"][0]["parts"][0]["functionCall"]["args"],
            json!({})
        );
    }

    #[test]
    fn encode_omits_empty_messages_and_rejects_an_empty_request() {
        let request = ProviderRequest {
            model: "gemini-2.0-flash".to_owned(),
            system: Some(String::new()),
            messages: vec![
                ProviderMessage::User(String::new()),
                ProviderMessage::Assistant(AssistantMessage::default()),
                ProviderMessage::User("real".to_owned()),
            ],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
        };
        let spec = GeminiAdapter
            .encode_request(&config("https://g", None), &request)
            .unwrap();
        assert_eq!(
            spec.body,
            json!({"contents": [{"role": "user", "parts": [{"text": "real"}]}]})
        );

        let empty = ProviderRequest {
            messages: vec![ProviderMessage::ToolResults(vec![])],
            ..request
        };
        assert!(matches!(
            GeminiAdapter
                .encode_request(&config("https://g", None), &empty)
                .unwrap_err(),
            ProviderError::Encode {
                provider: "gemini",
                ..
            }
        ));
    }
}
