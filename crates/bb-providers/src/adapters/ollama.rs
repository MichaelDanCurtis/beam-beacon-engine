//! Ollama native chat adapter.
//!
//! Wire format: `POST {base}/api/chat`, no auth. User images travel as a
//! base64 `images` array on the message; tools use the OpenAI-ish
//! `{"type":"function","function":{...}}` shape; `max_tokens` /
//! `temperature` map to `options.num_predict` / `options.temperature`.
//! Streaming is NDJSON: one JSON object per line, terminated by a
//! `"done": true` object carrying `prompt_eval_count` / `eval_count`.
//!
//! Wire-format notes:
//!
//! - Ollama tool calls carry no id; stable ids are synthesized as
//!   `call_{index}` in call order.
//! - Tool results encode as `role:"tool"` messages with a `tool_name`
//!   field; result images ride on the same message's `images` array.
//!   There is **no error flag** on that message, so a failed
//!   [`ToolResult`](crate::types::ToolResult) (`is_error`) has its content
//!   prefixed with [`TOOL_ERROR_PREFIX`](crate::TOOL_ERROR_PREFIX); otherwise every failed tool would
//!   reach the model as a structurally successful result.
//! - Assistant/stream `thinking` maps to Ollama's native `thinking` field.
//!   Ollama issues no thinking signature, so
//!   [`ThinkingBlock::signature`] is dropped on encode and never set on
//!   decode.
//! - Messages carrying no content are omitted, matching the neutral-model
//!   rule the other adapters follow; a request whose messages are all
//!   empty is an encode error.
//! - The final NDJSON line may legitimately arrive without a trailing
//!   newline, so hosts must call
//!   [`StreamDecoder::finish`](crate::StreamDecoder::finish) or the
//!   `"done": true` object — usage and stop reason — is lost.

use serde_json::{json, Value};

use crate::adapters::{
    empty_request_error, is_empty_message, join_url, tool_arguments_or_empty, tool_result_content,
    unknown_stop_reason, CompletedResponse, ProviderAdapter, StreamDecoder,
};
use crate::sse::NdjsonFramer;
use crate::types::{
    AssistantMessage, HttpRequestSpec, ProviderConfig, ProviderError, ProviderEvent,
    ProviderMessage, ProviderRequest, StopReason, ThinkingBlock, TokenUsage, ToolCall, UserContent,
};

const PROVIDER: &str = "ollama";

/// Adapter for the Ollama native chat API.
#[derive(Debug, Clone, Copy, Default)]
pub struct OllamaAdapter;

fn map_done_reason(raw: Option<&str>, saw_tool_call: bool) -> StopReason {
    match raw {
        Some("stop") | None => {
            if saw_tool_call {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }
        }
        Some("length") => StopReason::MaxTokens,
        Some(other) => StopReason::Other(other.to_owned()),
    }
}

fn decode_tool_calls(message: &Value, first_index: usize) -> Vec<ToolCall> {
    let Some(calls) = message.get("tool_calls").and_then(Value::as_array) else {
        return Vec::new();
    };
    calls
        .iter()
        .enumerate()
        .map(|(offset, call)| {
            let index = first_index + offset;
            ToolCall {
                id: format!("call_{index}"),
                name: call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                arguments: tool_arguments_or_empty(call.pointer("/function/arguments")),
            }
        })
        .collect()
}

fn encode_message(message: &ProviderMessage, out: &mut Vec<Value>) {
    match message {
        ProviderMessage::System(text) => {
            out.push(json!({"role": "system", "content": text}));
        }
        ProviderMessage::User(text) => {
            out.push(json!({"role": "user", "content": text}));
        }
        ProviderMessage::UserMultimodal(parts) => {
            let mut text_parts = Vec::new();
            let mut images = Vec::new();
            for part in parts {
                match part {
                    UserContent::Text(text) => text_parts.push(text.as_str()),
                    UserContent::Image(image) => images.push(image.data_base64.clone()),
                }
            }
            let mut message = json!({"role": "user", "content": text_parts.join("\n")});
            if !images.is_empty() {
                message["images"] = json!(images);
            }
            out.push(message);
        }
        ProviderMessage::Assistant(assistant) => {
            let mut message = json!({"role": "assistant", "content": assistant.text});
            if let Some(thinking) = &assistant.thinking {
                // Ollama has no slot for a signature; only the text maps.
                message["thinking"] = json!(thinking.text);
            }
            if !assistant.tool_calls.is_empty() {
                let calls: Vec<Value> = assistant
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({"function": {
                            "name": call.name,
                            "arguments": tool_arguments_or_empty(Some(&call.arguments)),
                        }})
                    })
                    .collect();
                message["tool_calls"] = Value::Array(calls);
            }
            out.push(message);
        }
        ProviderMessage::ToolResults(results) => {
            for result in results {
                let mut message = json!({
                    "role": "tool",
                    "tool_name": result.name,
                    // No error flag on this wire format: mark failures in-band.
                    "content": tool_result_content(result),
                });
                if !result.images.is_empty() {
                    let images: Vec<&str> = result
                        .images
                        .iter()
                        .map(|image| image.data_base64.as_str())
                        .collect();
                    message["images"] = json!(images);
                }
                out.push(message);
            }
        }
    }
}

impl ProviderAdapter for OllamaAdapter {
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
            encode_message(message, &mut messages);
        }
        if messages.is_empty() {
            return Err(empty_request_error(PROVIDER));
        }

        // Ollama defaults to streaming, so `stream` is always explicit.
        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
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
        let mut options = serde_json::Map::new();
        if let Some(max_tokens) = request.max_tokens {
            options.insert("num_predict".to_owned(), json!(max_tokens));
        }
        if let Some(temperature) = request.temperature {
            options.insert("temperature".to_owned(), json!(temperature));
        }
        if !options.is_empty() {
            body["options"] = Value::Object(options);
        }

        let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
        headers.extend(config.extra_headers.iter().cloned());

        Ok(HttpRequestSpec {
            method: "POST".to_owned(),
            url: join_url(&config.base_url, "api/chat"),
            headers,
            body,
        })
    }

    fn decode_response(&self, body: &Value) -> Result<CompletedResponse, ProviderError> {
        let message = body
            .get("message")
            .ok_or_else(|| ProviderError::decode(PROVIDER, "missing `message` object"))?;

        let text = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let thinking = message
            .get("thinking")
            .and_then(Value::as_str)
            .filter(|thinking| !thinking.is_empty())
            .map(ThinkingBlock::new);
        let tool_calls = decode_tool_calls(message, 0);

        let stop_reason = map_done_reason(
            body.get("done_reason").and_then(Value::as_str),
            !tool_calls.is_empty(),
        );
        let usage = TokenUsage {
            input_tokens: body
                .get("prompt_eval_count")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            output_tokens: body.get("eval_count").and_then(Value::as_u64).unwrap_or(0),
            cache_read_tokens: 0,
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
        Box::new(OllamaStreamDecoder::new())
    }
}

/// Streaming decoder for Ollama's NDJSON chat stream. Tool calls arrive
/// whole inside a message object, so each emits start / arguments / end
/// back-to-back with a synthesized `call_{index}` id.
#[derive(Debug)]
struct OllamaStreamDecoder {
    framer: NdjsonFramer,
    started: bool,
    next_tool_index: usize,
    saw_tool_call: bool,
    /// Set once a `"done": true` line (or the EOF flush) emitted
    /// `Finished`.
    finished: bool,
}

impl OllamaStreamDecoder {
    fn new() -> Self {
        Self {
            framer: NdjsonFramer::new(PROVIDER),
            started: false,
            next_tool_index: 0,
            saw_tool_call: false,
            finished: false,
        }
    }

    fn decode_lines(
        &mut self,
        lines: Vec<String>,
        out: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        for line in lines {
            let data: Value = serde_json::from_str(&line).map_err(|err| {
                ProviderError::decode(PROVIDER, format!("invalid JSON in NDJSON line: {err}"))
            })?;
            self.handle_line(&data, out)?;
        }
        Ok(())
    }

    fn handle_line(
        &mut self,
        data: &Value,
        out: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        if !self.started {
            self.started = true;
            out.push(ProviderEvent::MessageStart);
        }

        if let Some(message) = data.get("message") {
            if let Some(thinking) = message.get("thinking").and_then(Value::as_str) {
                if !thinking.is_empty() {
                    out.push(ProviderEvent::ThinkingDelta(thinking.to_owned()));
                }
            }
            if let Some(text) = message.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    out.push(ProviderEvent::TextDelta(text.to_owned()));
                }
            }
            for call in decode_tool_calls(message, self.next_tool_index) {
                let index = self.next_tool_index;
                self.next_tool_index += 1;
                self.saw_tool_call = true;
                let fragment = serde_json::to_string(&call.arguments).map_err(|err| {
                    ProviderError::decode(
                        PROVIDER,
                        format!("tool-call arguments not serializable: {err}"),
                    )
                })?;
                out.push(ProviderEvent::ToolCallStart {
                    index,
                    id: call.id,
                    name: call.name,
                });
                out.push(ProviderEvent::ToolCallArgumentsDelta {
                    index,
                    json_fragment: fragment,
                });
                out.push(ProviderEvent::ToolCallEnd { index });
            }
        }

        if data.get("done").and_then(Value::as_bool) == Some(true) && !self.finished {
            self.finished = true;
            out.push(ProviderEvent::Usage(TokenUsage {
                input_tokens: data
                    .get("prompt_eval_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                output_tokens: data.get("eval_count").and_then(Value::as_u64).unwrap_or(0),
                cache_read_tokens: 0,
            }));
            out.push(ProviderEvent::Finished(map_done_reason(
                data.get("done_reason").and_then(Value::as_str),
                self.saw_tool_call,
            )));
        }
        Ok(())
    }
}

impl StreamDecoder for OllamaStreamDecoder {
    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut out = Vec::new();
        let lines = self.framer.feed(chunk)?;
        self.decode_lines(lines, &mut out)?;
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut out = Vec::new();
        // NDJSON's final line needs no trailing newline, and it is the one
        // carrying `"done": true` — flush it before synthesizing.
        let lines = self.framer.finish()?;
        self.decode_lines(lines, &mut out)?;
        if !self.finished {
            self.finished = true;
            // No `"done": true` object arrived, so there is no usage to
            // report and no stated reason for stopping.
            out.push(ProviderEvent::Finished(unknown_stop_reason()));
        }
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
            model: "llama3.2".to_owned(),
            system: Some("Be terse.".to_owned()),
            messages: vec![
                ProviderMessage::UserMultimodal(vec![
                    UserContent::Text("What is this?".to_owned()),
                    UserContent::Image(image()),
                ]),
                ProviderMessage::Assistant(AssistantMessage {
                    text: String::new(),
                    thinking: Some(ThinkingBlock::new("hmm")),
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
            max_tokens: Some(50),
            temperature: Some(0.5),
            stream: true,
        };
        let spec = OllamaAdapter
            .encode_request(&config("http://localhost:11434", None), &request)
            .unwrap();
        assert_eq!(spec.url, "http://localhost:11434/api/chat");
        assert_eq!(
            spec.headers,
            vec![("content-type".to_owned(), "application/json".to_owned())]
        );
        assert_eq!(
            spec.body,
            json!({
                "model": "llama3.2",
                "stream": true,
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "What is this?", "images": ["aGVsbG8="]},
                    {"role": "assistant", "content": "", "thinking": "hmm",
                     "tool_calls": [{"function": {"name": "lookup",
                                                  "arguments": {"q": "cat"}}}]},
                    {"role": "tool", "tool_name": "lookup", "content": "A cat.",
                     "images": ["aGVsbG8="]},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look things up",
                        "parameters": {"type": "object"},
                    },
                }],
                "options": {"num_predict": 50, "temperature": 0.5},
            })
        );
    }

    #[test]
    fn encode_every_message_variant_succeeds() {
        OllamaAdapter
            .encode_request(
                &config("http://localhost:11434", None),
                &full_conversation("llama3.2"),
            )
            .unwrap();
    }

    #[test]
    fn decode_response_with_tool_calls() {
        let body = json!({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "checking",
                "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}],
            },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 6,
        });
        let response = OllamaAdapter.decode_response(&body).unwrap();
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.message.thinking.as_deref(), Some("checking"));
        assert_eq!(
            response.message.tool_calls,
            vec![ToolCall {
                id: "call_0".to_owned(),
                name: "lookup".to_owned(),
                arguments: json!({"q": "x"}),
            }]
        );
        assert_eq!(
            response.usage,
            TokenUsage {
                input_tokens: 20,
                output_tokens: 6,
                cache_read_tokens: 0,
            }
        );
    }

    #[test]
    fn decode_response_length_stop() {
        let body = json!({
            "message": {"role": "assistant", "content": "truncat"},
            "done": true,
            "done_reason": "length",
        });
        let response = OllamaAdapter.decode_response(&body).unwrap();
        assert_eq!(response.stop_reason, StopReason::MaxTokens);
    }

    fn text_stream() -> Vec<u8> {
        let mut s = String::new();
        s.push_str("{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"H\u{e9}\"},\"done\":false}\n");
        s.push_str("{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"llo \u{1f389}\"},\"done\":false}\n");
        s.push_str("{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"lookup\",\"arguments\":{\"q\":\"caf\u{e9}\"}}}]},\"done\":false}\n");
        s.push_str("{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":11,\"eval_count\":4}\n");
        s.into_bytes()
    }

    #[test]
    fn stream_decodes_full_event_sequence() {
        let mut decoder = OllamaAdapter.stream_decoder();
        let events = decoder.feed(&text_stream()).unwrap();
        assert_eq!(
            events,
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("H\u{e9}".to_owned()),
                ProviderEvent::TextDelta("llo \u{1f389}".to_owned()),
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
                    input_tokens: 11,
                    output_tokens: 4,
                    cache_read_tokens: 0,
                }),
                // The stream carried a tool call, so a plain "stop" means
                // the model is waiting on tool results.
                ProviderEvent::Finished(StopReason::ToolUse),
            ]
        );
    }

    #[test]
    fn stream_is_stable_under_pathological_chunking() {
        assert_split_stable(&OllamaAdapter, &text_stream());
    }

    #[test]
    fn stream_without_trailing_newline_decodes_identically() {
        // NDJSON's last line needs no trailing newline, and it is the one
        // carrying `"done": true` — usage and stop reason.
        let full = text_stream();
        let truncated = full.strip_suffix(b"\n").unwrap().to_vec();
        assert_terminator_optional(&OllamaAdapter, &full, &truncated);
    }

    #[test]
    fn stream_without_done_object_still_finishes() {
        let bytes = b"{\"message\":{\"content\":\"partial\"},\"done\":false}\n";
        assert_eq!(
            decode_stream(&OllamaAdapter, bytes),
            vec![
                ProviderEvent::MessageStart,
                ProviderEvent::TextDelta("partial".to_owned()),
                ProviderEvent::Finished(StopReason::Other("unknown".to_owned())),
            ]
        );
    }

    #[test]
    fn finish_after_done_object_emits_nothing_more() {
        let mut decoder = OllamaAdapter.stream_decoder();
        decoder.feed(&text_stream()).unwrap();
        assert!(decoder.finish().unwrap().is_empty());
        assert!(decoder.finish().unwrap().is_empty());
    }

    #[test]
    fn encode_marks_failed_tool_results_in_content() {
        // Ollama's `role:"tool"` message has no error flag either.
        let request = ProviderRequest {
            model: "llama3.2".to_owned(),
            system: None,
            messages: vec![ProviderMessage::ToolResults(vec![
                ToolResult {
                    call_id: "call_0".to_owned(),
                    name: "shell".to_owned(),
                    content: "exit status 1".to_owned(),
                    images: vec![],
                    is_error: true,
                },
                ToolResult {
                    call_id: "call_1".to_owned(),
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
        let spec = OllamaAdapter
            .encode_request(&config("http://localhost:11434", None), &request)
            .unwrap();
        assert_eq!(
            spec.body["messages"],
            json!([
                {"role": "tool", "tool_name": "shell",
                 "content": "Error: exit status 1"},
                {"role": "tool", "tool_name": "shell", "content": "ok"},
            ])
        );
        assert!(spec.body["messages"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with(crate::TOOL_ERROR_PREFIX));
    }

    #[test]
    fn decode_tool_call_without_arguments_becomes_empty_object() {
        let body = json!({
            "message": {"content": "", "tool_calls": [{"function": {"name": "now"}}]},
            "done": true,
        });
        let decoded = OllamaAdapter.decode_response(&body).unwrap();
        assert_eq!(decoded.message.tool_calls[0].arguments, json!({}));

        let spec = OllamaAdapter
            .encode_request(
                &config("http://localhost:11434", None),
                &ProviderRequest {
                    model: "llama3.2".to_owned(),
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
            spec.body["messages"][0]["tool_calls"][0]["function"]["arguments"],
            json!({})
        );
    }

    #[test]
    fn encode_omits_empty_messages_and_rejects_an_empty_request() {
        let request = ProviderRequest {
            model: "llama3.2".to_owned(),
            system: Some(String::new()),
            messages: vec![
                ProviderMessage::User(String::new()),
                ProviderMessage::ToolResults(vec![]),
                ProviderMessage::User("real".to_owned()),
            ],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
        };
        let spec = OllamaAdapter
            .encode_request(&config("http://localhost:11434", None), &request)
            .unwrap();
        assert_eq!(
            spec.body["messages"],
            json!([{"role": "user", "content": "real"}])
        );

        let empty = ProviderRequest {
            messages: vec![ProviderMessage::UserMultimodal(vec![])],
            ..request
        };
        assert!(matches!(
            OllamaAdapter
                .encode_request(&config("http://localhost:11434", None), &empty)
                .unwrap_err(),
            ProviderError::Encode {
                provider: "ollama",
                ..
            }
        ));
    }
}
