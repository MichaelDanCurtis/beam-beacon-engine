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
pub trait StreamDecoder: Send {
    /// Feed raw bytes as they arrive off the wire; returns fully-decoded
    /// events. An empty vec just means no frame completed yet.
    fn feed(&mut self, chunk: &[u8]) -> Result<Vec<ProviderEvent>, ProviderError>;
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
