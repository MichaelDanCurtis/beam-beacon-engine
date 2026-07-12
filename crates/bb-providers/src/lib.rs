//! AI provider abstraction.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-providers` defines the provider-neutral message/event model for AI
//! chat ([`ProviderMessage`] in, [`ProviderEvent`] out) and the
//! per-provider **wire-format** encoders/decoders behind the
//! [`ProviderAdapter`] trait:
//!
//! - [`AnthropicAdapter`] — native Messages API: image blocks, tool use,
//!   thinking, SSE streaming.
//! - [`OpenAiCompatAdapter`] — Chat Completions; `image_url` data URLs;
//!   covers OpenAI, OpenRouter, Together, Groq, Moonshot/Kimi, custom
//!   proxies.
//! - [`GeminiAdapter`] — inline image data, function calling, SSE
//!   streaming.
//! - [`OllamaAdapter`] — native chat with `images` arrays, NDJSON
//!   streaming.
//! - [`LocalOpenAiAdapter`] — Ollama-on-`/v1`, LM Studio, vLLM,
//!   llama.cpp (delegates to [`OpenAiCompatAdapter`]).
//!
//! The same neutral conversation — text, images, tool calls, tool
//! results — round-trips through any adapter without lossy conversion.
//!
//! # Transport-agnostic by design
//!
//! This crate performs **no I/O**: no HTTP client, no network, no async.
//! [`ProviderAdapter::encode_request`] turns a neutral request into an
//! [`HttpRequestSpec`] (method, URL, headers, JSON body) for the host's
//! own HTTP stack; [`ProviderAdapter::decode_response`] and
//! [`StreamDecoder::feed`] turn the raw response back into neutral values.
//! Stream decoders buffer raw bytes internally, so network chunks may
//! split anywhere — mid-UTF-8-codepoint, mid-line, mid-JSON — without
//! corrupting output (see [`sse`]).
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod adapters;
pub mod sse;
pub mod types;

pub use adapters::{
    adapter_for, AnthropicAdapter, CompletedResponse, GeminiAdapter, LocalOpenAiAdapter,
    OllamaAdapter, OpenAiCompatAdapter, ProviderAdapter, ProviderKind, StreamDecoder,
};
pub use types::{
    AssistantMessage, HttpRequestSpec, MessageImage, ProviderConfig, ProviderError, ProviderEvent,
    ProviderMessage, ProviderRequest, StopReason, TokenUsage, ToolCall, ToolResult, ToolSpec,
    UserContent,
};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(crate) mod test_support {
    //! Shared fixtures and helpers for adapter tests.

    use serde_json::json;

    use crate::{
        AssistantMessage, MessageImage, ProviderAdapter, ProviderConfig, ProviderEvent,
        ProviderMessage, ProviderRequest, ToolCall, ToolResult, ToolSpec, UserContent,
    };

    /// A small valid base64 image fixture.
    pub(crate) fn image() -> MessageImage {
        MessageImage {
            media_type: "image/png".to_owned(),
            data_base64: "aGVsbG8=".to_owned(),
        }
    }

    /// Build a [`ProviderConfig`] for tests.
    pub(crate) fn config(base_url: &str, api_key: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            base_url: base_url.to_owned(),
            api_key: api_key.map(str::to_owned),
            extra_headers: vec![],
        }
    }

    /// A conversation exercising every [`ProviderMessage`] variant, for the
    /// encode round-trip property tests.
    pub(crate) fn full_conversation(model: &str) -> ProviderRequest {
        ProviderRequest {
            model: model.to_owned(),
            system: Some("Root system.".to_owned()),
            messages: vec![
                ProviderMessage::System("Inline system.".to_owned()),
                ProviderMessage::User("Plain user text.".to_owned()),
                ProviderMessage::UserMultimodal(vec![
                    UserContent::Text("Look at this:".to_owned()),
                    UserContent::Image(image()),
                ]),
                ProviderMessage::Assistant(AssistantMessage {
                    text: "Calling a tool.".to_owned(),
                    thinking: Some("Let me think.".to_owned()),
                    tool_calls: vec![ToolCall {
                        id: "call_0".to_owned(),
                        name: "lookup".to_owned(),
                        arguments: json!({"q": "caf\u{e9}"}),
                    }],
                }),
                ProviderMessage::ToolResults(vec![ToolResult {
                    call_id: "call_0".to_owned(),
                    name: "lookup".to_owned(),
                    content: "Result \u{1f389}".to_owned(),
                    images: vec![image()],
                    is_error: false,
                }]),
                ProviderMessage::Assistant(AssistantMessage {
                    text: "Done.".to_owned(),
                    thinking: None,
                    tool_calls: vec![],
                }),
            ],
            tools: vec![ToolSpec {
                name: "lookup".to_owned(),
                description: "Look things up.".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                }),
            }],
            max_tokens: Some(512),
            temperature: Some(0.5),
            stream: false,
        }
    }

    /// Decode `bytes` in `size`-byte chunks, collecting every event.
    pub(crate) fn events_with_chunk_size(
        adapter: &dyn ProviderAdapter,
        bytes: &[u8],
        size: usize,
    ) -> Vec<ProviderEvent> {
        let mut decoder = adapter.stream_decoder();
        let mut out = Vec::new();
        for chunk in bytes.chunks(size.max(1)) {
            out.extend(decoder.feed(chunk).unwrap());
        }
        out
    }

    /// Assert that decoding `bytes` yields identical events no matter how
    /// pathologically the stream is chunked (including one byte at a time,
    /// which splits multi-byte UTF-8 codepoints).
    pub(crate) fn assert_split_stable(adapter: &dyn ProviderAdapter, bytes: &[u8]) {
        let expected = events_with_chunk_size(adapter, bytes, bytes.len().max(1));
        assert!(
            !expected.is_empty(),
            "fixture stream produced no events; test is vacuous"
        );
        for size in [1, 2, 3, 5, 7, 13] {
            assert_eq!(
                events_with_chunk_size(adapter, bytes, size),
                expected,
                "event mismatch at chunk size {size}"
            );
        }
    }
}
