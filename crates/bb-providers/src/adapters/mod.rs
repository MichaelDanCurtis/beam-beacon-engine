//! The [`ProviderAdapter`] trait and the five built-in adapters.

pub mod anthropic;
pub mod gemini;
pub mod local;
pub mod ollama;
pub mod openai_compat;

use serde::{Deserialize, Serialize};

use crate::types::{
    AssistantMessage, HttpRequestSpec, MessageImage, ProviderConfig, ProviderError, ProviderEvent,
    ProviderRequest, StopReason, TokenUsage,
};

pub use anthropic::AnthropicAdapter;
pub use gemini::GeminiAdapter;
pub use local::LocalOpenAiAdapter;
pub use ollama::OllamaAdapter;
pub use openai_compat::OpenAiCompatAdapter;

/// A provider's wire-format encoder/decoder.
///
/// Implementations are pure translators: they never perform I/O. Hosts
/// send the [`HttpRequestSpec`] with their own HTTP stack, then hand the
/// response back — a parsed JSON body to [`decode_response`] for
/// non-streaming calls, or raw bytes to the [`StreamDecoder`] from
/// [`stream_decoder`] for streaming calls.
///
/// The trait is object-safe; use [`adapter_for`] to obtain a boxed adapter
/// by [`ProviderKind`].
///
/// [`decode_response`]: ProviderAdapter::decode_response
/// [`stream_decoder`]: ProviderAdapter::stream_decoder
pub trait ProviderAdapter: Send + Sync {
    /// Stable adapter name, e.g. `"anthropic"`.
    fn name(&self) -> &'static str;

    /// Encode a neutral request into a ready-to-send HTTP request spec.
    fn encode_request(
        &self,
        config: &ProviderConfig,
        request: &ProviderRequest,
    ) -> Result<HttpRequestSpec, ProviderError>;

    /// Decode a complete (non-streaming) response body into a neutral
    /// [`CompletedResponse`].
    fn decode_response(&self, body: &serde_json::Value)
        -> Result<CompletedResponse, ProviderError>;

    /// Create a fresh stateful decoder for one streaming response.
    fn stream_decoder(&self) -> Box<dyn StreamDecoder>;
}

/// A fully-decoded non-streaming provider response.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletedResponse {
    /// The assistant turn (text, thinking, tool calls).
    pub message: AssistantMessage,
    /// Why generation stopped.
    pub stop_reason: StopReason,
    /// Token accounting reported by the provider.
    pub usage: TokenUsage,
}

/// Stateful decoder for one streaming response.
///
/// Chunks may split anywhere — mid-UTF-8-codepoint, mid-line, mid-JSON —
/// so implementations buffer raw bytes internally and only decode complete
/// frames. Feeding the same byte stream in different chunkings always
/// yields the same event sequence.
///
/// # Hosts MUST call [`finish`] at end-of-stream
///
/// [`feed`] can only emit what a delimiter has completed. Real endpoints
/// routinely omit the final delimiter: llama.cpp's server and some
/// vLLM/LM Studio/proxy configurations never send `data: [DONE]`, an
/// Anthropic `message_stop` frame may not be followed by a blank line, and
/// a final Ollama NDJSON line legitimately has no trailing newline. Skip
/// [`finish`] and the decoder emits no [`ProviderEvent::Finished`] — the
/// agent loop then treats the turn as never having completed and discards
/// an assistant message it already streamed to the UI in full.
///
/// So: feed every chunk, then call [`finish`] exactly once when the
/// transport reports EOF, and treat its events as part of the same stream.
///
/// ```
/// # use bb_providers::{adapter_for, ProviderKind, ProviderError};
/// # fn run(chunks: &[&[u8]]) -> Result<(), ProviderError> {
/// let mut decoder = adapter_for(ProviderKind::OpenAiCompat).stream_decoder();
/// let mut events = Vec::new();
/// for chunk in chunks {
///     events.extend(decoder.feed(chunk)?);
/// }
/// events.extend(decoder.finish()?); // <- required, even on a clean stream
/// # Ok(())
/// # }
/// ```
///
/// [`feed`]: StreamDecoder::feed
/// [`finish`]: StreamDecoder::finish
pub trait StreamDecoder: Send {
    /// Feed raw bytes as they arrive off the wire; returns fully-decoded
    /// events. An empty vec just means no frame completed yet.
    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<ProviderEvent>, ProviderError>;

    /// Called once when the transport reports end-of-stream. Flushes any
    /// buffered frame and emits a synthesized [`ProviderEvent::Finished`]
    /// if the provider never sent an explicit terminator. Hosts MUST call
    /// this.
    ///
    /// The synthesized `Finished` carries the stop reason observed so far
    /// (an OpenAI-compatible `finish_reason` seen before the stream ended,
    /// say), or [`StopReason::Other("unknown")`] when the provider never
    /// stated one. Implementations never emit a second `Finished` if the
    /// stream already terminated properly, and calling `finish` more than
    /// once yields no further events — so after `feed` + `finish` the event
    /// sequence always ends with exactly one `Finished`, even if the
    /// provider sent nothing at all.
    ///
    /// Only call it when the response body ended *cleanly*. After a
    /// transport error, or after `feed` returned an error, the decoder is
    /// dead; call `finish` there only if you deliberately want to salvage
    /// partial output as a completed turn.
    ///
    /// # Errors
    ///
    /// Returns a [`ProviderError::Decode`] when the buffered residue is a
    /// *partial* frame — a stream cut mid-JSON or mid-UTF-8-codepoint. The
    /// response really was truncated, and reporting that beats fabricating
    /// a completed turn out of half a frame.
    ///
    /// [`StopReason::Other("unknown")`]: StopReason::Other
    fn finish(&mut self) -> Result<Vec<ProviderEvent>, ProviderError>;
}

/// Stop reason used when a stream ends without the provider stating one.
pub(crate) fn unknown_stop_reason() -> StopReason {
    StopReason::Other("unknown".to_owned())
}

/// Selector for the built-in adapters, for use with [`adapter_for`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderKind {
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI-compatible Chat Completions (OpenAI, OpenRouter, Together,
    /// Groq, Moonshot/Kimi, custom proxies).
    OpenAiCompat,
    /// Google Gemini `generateContent` API.
    Gemini,
    /// Ollama native chat API.
    Ollama,
    /// Local OpenAI-compatible servers (Ollama-on-`/v1`, LM Studio, vLLM,
    /// llama.cpp).
    LocalOpenAi,
}

/// Construct the built-in adapter for `kind`.
pub fn adapter_for(kind: ProviderKind) -> Box<dyn ProviderAdapter> {
    match kind {
        ProviderKind::Anthropic => Box::new(AnthropicAdapter),
        ProviderKind::OpenAiCompat => Box::new(OpenAiCompatAdapter),
        ProviderKind::Gemini => Box::new(GeminiAdapter),
        ProviderKind::Ollama => Box::new(OllamaAdapter),
        ProviderKind::LocalOpenAi => Box::new(LocalOpenAiAdapter),
    }
}

/// Marker prepended to a failed tool result's content on wire formats with
/// no structured error flag (OpenAI-compatible `role:"tool"` messages and
/// Ollama's equivalent).
///
/// Anthropic has `is_error` and Gemini has an `error` response key, so
/// those adapters signal failure structurally and never add this prefix.
/// The other two would otherwise present a failed tool — a non-zero shell
/// exit, an unknown tool, the agent loop's own cancellation marker — to the
/// model as a structurally *successful* result, so the failure is made
/// legible in the content instead. Hosts that surface tool output verbatim
/// can strip this prefix.
pub const TOOL_ERROR_PREFIX: &str = "Error: ";

/// Render a [`ToolResult`](crate::types::ToolResult)'s content for a wire
/// format with no error flag, marking failures with [`TOOL_ERROR_PREFIX`].
pub(crate) fn tool_result_content(result: &crate::types::ToolResult) -> String {
    if result.is_error {
        format!("{TOOL_ERROR_PREFIX}{}", result.content)
    } else {
        result.content.clone()
    }
}

/// Normalize a decoded tool-call `arguments` value.
///
/// An absent or `null` payload becomes an empty JSON **object**: every
/// provider models "this tool takes no arguments" that way (OpenAI sends
/// `"arguments": ""`, Anthropic omits `input`, Gemini omits `args`,
/// Ollama omits `function.arguments`), and re-encoding a `null` would put
/// `"input": null` / `"arguments": "null"` / `"args": null` back on the
/// wire. This matches what the streaming path already produces for a tool
/// call that streamed no argument fragments.
pub(crate) fn tool_arguments_or_empty(raw: Option<&serde_json::Value>) -> serde_json::Value {
    match raw {
        None | Some(serde_json::Value::Null) => serde_json::Value::Object(serde_json::Map::new()),
        Some(other) => other.clone(),
    }
}

/// True when a message carries nothing the provider could represent.
///
/// Every adapter omits such messages: Anthropic rejects empty text blocks
/// and empty content arrays outright, and the alternative — encoding
/// `{"content": ""}` or `{"content": []}` — turns a harmless empty turn
/// into an opaque HTTP 400. [`encode_request`](ProviderAdapter::encode_request)
/// errors only if omission would leave the request with no messages at
/// all, which is a genuine "nothing to send" bug in the caller.
///
/// A [`ProviderMessage::Assistant`] holding thinking or tool calls is not
/// empty even when its text is, so tool-call/tool-result pairing is never
/// broken by this rule.
pub(crate) fn is_empty_message(message: &crate::types::ProviderMessage) -> bool {
    use crate::types::ProviderMessage as M;
    match message {
        M::System(text) | M::User(text) => text.is_empty(),
        M::UserMultimodal(parts) => parts.is_empty(),
        M::Assistant(assistant) => {
            let thinking_empty = match &assistant.thinking {
                Some(thinking) => thinking.is_empty(),
                None => true,
            };
            assistant.text.is_empty() && assistant.tool_calls.is_empty() && thinking_empty
        }
        M::ToolResults(results) => results.is_empty(),
    }
}

/// Error for a request whose messages were all empty.
pub(crate) fn empty_request_error(provider: &'static str) -> ProviderError {
    ProviderError::encode(
        provider,
        "request contains no non-empty messages; every message was empty \
         (blank text, no multimodal parts, or no tool results)",
    )
}

/// Join a base URL and a path with exactly one `/` between them.
pub(crate) fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

/// Render an image as an RFC 2397 data URL (OpenAI-style `image_url`).
pub(crate) fn data_url(image: &MessageImage) -> String {
    format!("data:{};base64,{}", image.media_type, image.data_base64)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn adapter_for_returns_each_kind() {
        let cases = [
            (ProviderKind::Anthropic, "anthropic"),
            (ProviderKind::OpenAiCompat, "openai-compat"),
            (ProviderKind::Gemini, "gemini"),
            (ProviderKind::Ollama, "ollama"),
            (ProviderKind::LocalOpenAi, "local-openai"),
        ];
        for (kind, name) in cases {
            assert_eq!(adapter_for(kind).name(), name);
        }
    }

    #[test]
    fn join_url_normalizes_slashes() {
        assert_eq!(join_url("http://x", "v1/messages"), "http://x/v1/messages");
        assert_eq!(
            join_url("http://x/", "/v1/messages"),
            "http://x/v1/messages"
        );
    }
}
