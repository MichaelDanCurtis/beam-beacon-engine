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
//! - Assistant/stream `thinking` maps to Ollama's native `thinking` field.

use serde_json::{json, Value};

use crate::adapters::{join_url, CompletedResponse, ProviderAdapter, StreamDecoder};
use crate::sse::NdjsonFramer;
use crate::types::{
    AssistantMessage, HttpRequestSpec, ProviderConfig, ProviderError, ProviderEvent,
    ProviderMessage, ProviderRequest, StopReason, TokenUsage, ToolCall, UserContent,
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
                arguments: call
                    .pointer("/function/arguments")
                    .cloned()
                    .unwrap_or(Value::Null),
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
                message["thinking"] = json!(thinking);
            }
            if !assistant.tool_calls.is_empty() {
                let calls: Vec<Value> = assistant
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({"function": {"name": call.name, "arguments": call.arguments}})
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
                    "content": result.content,
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
            messages.push(json!({"role": "system", "content": system}));
        }
        for message in &request.messages {
            encode_message(message, &mut messages);
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
            .map(str::to_owned);
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
}

impl OllamaStreamDecoder {
    fn new() -> Self {
        Self {
            framer: NdjsonFramer::new(PROVIDER),
            started: false,
            next_tool_index: 0,
            saw_tool_call: false,
        }
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

        if data.get("done").and_then(Value::as_bool) == Some(true) {
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
        for line in self.framer.feed(chunk)? {
            let data: Value = serde_json::from_str(&line).map_err(|err| {
                ProviderError::decode(PROVIDER, format!("invalid JSON in NDJSON line: {err}"))
            })?;
            self.handle_line(&data, &mut out)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::test_support::{assert_split_stable, config, full_conversation, image};
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
                    thinking: Some("hmm".to_owned()),
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
}
