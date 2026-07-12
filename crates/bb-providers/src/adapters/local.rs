//! Local OpenAI-compatible server adapter.
//!
//! A thin delegating wrapper over [`OpenAiCompatAdapter`] for local
//! endpoints: Ollama-on-`/v1`, LM Studio, vLLM, llama.cpp. The wire format
//! is identical; the only differences are the adapter name
//! (`"local-openai"`) and the expectation that
//! [`ProviderConfig::api_key`] is usually `None` — the underlying encoder
//! already omits the `Authorization` header in that case.
//!
//! [`ProviderConfig::api_key`]: crate::types::ProviderConfig::api_key

use crate::adapters::{CompletedResponse, OpenAiCompatAdapter, ProviderAdapter, StreamDecoder};
use crate::types::{HttpRequestSpec, ProviderConfig, ProviderError, ProviderRequest};

/// Adapter for local OpenAI-compatible servers. Delegates every operation
/// to [`OpenAiCompatAdapter`].
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalOpenAiAdapter;

impl ProviderAdapter for LocalOpenAiAdapter {
    fn name(&self) -> &'static str {
        "local-openai"
    }

    fn encode_request(
        &self,
        config: &ProviderConfig,
        request: &ProviderRequest,
    ) -> Result<HttpRequestSpec, ProviderError> {
        OpenAiCompatAdapter.encode_request(config, request)
    }

    fn decode_response(
        &self,
        body: &serde_json::Value,
    ) -> Result<CompletedResponse, ProviderError> {
        OpenAiCompatAdapter.decode_response(body)
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        OpenAiCompatAdapter.stream_decoder()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::test_support::{config, full_conversation};

    #[test]
    fn name_is_local_openai() {
        assert_eq!(LocalOpenAiAdapter.name(), "local-openai");
    }

    #[test]
    fn encodes_without_authorization_when_no_key() {
        let spec = LocalOpenAiAdapter
            .encode_request(
                &config("http://localhost:1234/v1", None),
                &full_conversation("qwen2.5-coder"),
            )
            .unwrap();
        assert_eq!(spec.url, "http://localhost:1234/v1/chat/completions");
        assert!(spec.headers.iter().all(|(name, _)| name != "authorization"));
    }

    #[test]
    fn forwards_bearer_when_key_present() {
        let spec = LocalOpenAiAdapter
            .encode_request(
                &config("http://localhost:8000/v1", Some("vllm-key")),
                &full_conversation("qwen2.5-coder"),
            )
            .unwrap();
        assert!(spec
            .headers
            .contains(&("authorization".to_owned(), "Bearer vllm-key".to_owned())));
    }
}
