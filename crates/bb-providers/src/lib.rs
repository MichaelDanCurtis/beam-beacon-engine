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
//! # Streaming: always `feed` then `finish`
//!
//! A stream decoder can only emit what a frame delimiter has completed,
//! and real endpoints routinely omit the last one. [`StreamDecoder::finish`]
//! flushes the residue and guarantees a terminating
//! [`ProviderEvent::Finished`]; skipping it silently drops the end of the
//! turn.
//!
//! ```
//! use bb_providers::{adapter_for, ProviderConfig, ProviderKind, ProviderMessage,
//!                    ProviderRequest, ProviderEvent};
//!
//! # fn main() -> Result<(), bb_providers::ProviderError> {
//! let adapter = adapter_for(ProviderKind::OpenAiCompat);
//! let config = ProviderConfig {
//!     base_url: "https://api.openai.com/v1".to_owned(),
//!     api_key: Some("sk-...".to_owned()),
//!     extra_headers: vec![],
//! };
//! let request = ProviderRequest {
//!     model: "gpt-4o".to_owned(),
//!     system: None,
//!     messages: vec![ProviderMessage::User("Hi".to_owned())],
//!     tools: vec![],
//!     max_tokens: None,
//!     temperature: None,
//!     stream: true,
//! };
//!
//! // 1. Encode, and send `spec` with the host's own HTTP stack.
//! let spec = adapter.encode_request(&config, &request)?;
//! assert_eq!(spec.url, "https://api.openai.com/v1/chat/completions");
//!
//! // 2. Feed every response chunk, in whatever sizes they arrive.
//! let mut decoder = adapter.stream_decoder();
//! let mut events = Vec::new();
//! for chunk in [
//!     &b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi!\"},"[..],
//!     &b"\"finish_reason\":\"stop\"}]}\n\n"[..],
//! ] {
//!     events.extend(decoder.feed(chunk)?);
//! }
//!
//! // 3. Flush at end-of-stream. REQUIRED: this server sent no
//! //    `data: [DONE]`, so without it the turn never finishes.
//! events.extend(decoder.finish()?);
//!
//! assert!(matches!(events.last(), Some(ProviderEvent::Finished(_))));
//! # Ok(())
//! # }
//! ```
//!
//! # Neutral-model rules every adapter enforces
//!
//! - **Empty messages are omitted.** A [`ProviderMessage`] with no content
//!   (blank text, no multimodal parts, no tool results, an assistant turn
//!   with nothing in it) has no wire representation — Anthropic rejects
//!   empty text blocks and empty content arrays outright. If that leaves
//!   nothing to send, `encode_request` returns a
//!   [`ProviderError::Encode`].
//! - **Absent tool arguments are an empty object,** never `null`, on both
//!   decode and encode.
//! - **Failed tool results stay legible.** Anthropic sets `is_error` and
//!   Gemini uses an `error` response key; the OpenAI-compatible and Ollama
//!   formats have no such flag, so those adapters prefix the content with
//!   [`TOOL_ERROR_PREFIX`].
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
    TOOL_ERROR_PREFIX,
};
pub use types::{
    AssistantMessage, HttpRequestSpec, MessageImage, ProviderConfig, ProviderError, ProviderEvent,
    ProviderMessage, ProviderRequest, StopReason, ThinkingBlock, TokenUsage, ToolCall, ToolResult,
    ToolSpec, UserContent,
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
                    thinking: Some(crate::ThinkingBlock::with_signature(
                        "Let me think.",
                        "sig-abc",
                    )),
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
    /// `feed`-only: does **not** call [`StreamDecoder::finish`].
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

    /// Decode `bytes` in `size`-byte chunks and then flush at EOF — the
    /// full contract hosts are required to follow.
    pub(crate) fn events_with_finish(
        adapter: &dyn ProviderAdapter,
        bytes: &[u8],
        size: usize,
    ) -> Vec<ProviderEvent> {
        let mut decoder = adapter.stream_decoder();
        let mut out = Vec::new();
        for chunk in bytes.chunks(size.max(1)) {
            out.extend(decoder.feed(chunk).unwrap());
        }
        out.extend(decoder.finish().unwrap());
        out
    }

    /// Decode the whole of `bytes` in one feed, then flush at EOF.
    pub(crate) fn decode_stream(adapter: &dyn ProviderAdapter, bytes: &[u8]) -> Vec<ProviderEvent> {
        events_with_finish(adapter, bytes, bytes.len().max(1))
    }

    /// Assert that decoding `bytes` yields identical events no matter how
    /// pathologically the stream is chunked (including one byte at a time,
    /// which splits multi-byte UTF-8 codepoints).
    ///
    /// Checked twice: `feed`-only, and `feed` + [`StreamDecoder::finish`],
    /// since the EOF flush must be just as chunking-insensitive.
    pub(crate) fn assert_split_stable(adapter: &dyn ProviderAdapter, bytes: &[u8]) {
        for with_finish in [false, true] {
            let decode = |size: usize| {
                if with_finish {
                    events_with_finish(adapter, bytes, size)
                } else {
                    events_with_chunk_size(adapter, bytes, size)
                }
            };
            let expected = decode(bytes.len().max(1));
            assert!(
                !expected.is_empty(),
                "fixture stream produced no events; test is vacuous \
                 (with_finish={with_finish})"
            );
            for size in [1, 2, 3, 5, 7, 13] {
                assert_eq!(
                    decode(size),
                    expected,
                    "event mismatch at chunk size {size} (with_finish={with_finish})"
                );
            }
        }
    }

    /// Assert that a stream and the same stream *without its terminator*
    /// decode to the same events once `feed` + `finish` has run — the
    /// property that makes a missing `[DONE]` / blank line / trailing
    /// newline harmless.
    ///
    /// Both forms are also checked under pathological chunking.
    pub(crate) fn assert_terminator_optional(
        adapter: &dyn ProviderAdapter,
        terminated: &[u8],
        unterminated: &[u8],
    ) {
        assert_ne!(
            terminated, unterminated,
            "fixture pair is identical; test is vacuous"
        );
        let expected = decode_stream(adapter, terminated);
        assert!(
            matches!(expected.last(), Some(ProviderEvent::Finished(_))),
            "terminated fixture must end in Finished, got {expected:?}"
        );
        assert_eq!(
            decode_stream(adapter, unterminated),
            expected,
            "dropping the stream terminator changed the decoded events"
        );
        assert_split_stable(adapter, unterminated);
    }
}
