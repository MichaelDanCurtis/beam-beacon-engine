//! Provider-neutral message, request, and event model.
//!
//! Everything in this module is wire-format agnostic: hosts and the agent
//! loop build conversations out of these types, and the per-provider
//! adapters in [`crate::adapters`] translate them to and from each
//! provider's JSON shape.

use serde::{Deserialize, Serialize};

/// One message in a provider-neutral conversation transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProviderMessage {
    /// A system instruction. Adapters hoist this into whatever slot the
    /// provider uses for system prompts (top-level `system`, a
    /// `role:"system"` message, or `system_instruction`).
    System(String),
    /// A plain-text user message.
    User(String),
    /// A user message mixing text and images.
    UserMultimodal(Vec<UserContent>),
    /// An assistant turn: text, optional thinking, and any tool calls the
    /// model requested.
    Assistant(AssistantMessage),
    /// Results for previously requested tool calls, sent back to the model.
    ToolResults(Vec<ToolResult>),
}

/// One piece of a multimodal user message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UserContent {
    /// A text fragment.
    Text(String),
    /// An inline image.
    Image(MessageImage),
}

/// The content of an assistant turn.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// Visible response text (may be empty when the turn is pure tool use).
    pub text: String,
    /// Extended-thinking / reasoning content, when the provider surfaced it.
    pub thinking: Option<ThinkingBlock>,
    /// Tool invocations the model requested this turn, in order.
    pub tool_calls: Vec<ToolCall>,
}

/// Extended-thinking / reasoning content from one assistant turn.
///
/// The visible reasoning text plus an optional provider-opaque
/// **signature**. Anthropic requires the signature to be echoed verbatim
/// when a thinking block is replayed in a later assistant turn (extended
/// thinking + tool use), so it must survive the round trip through the
/// neutral model. Providers that expose no signature simply leave it
/// `None`; providers that ignore thinking altogether ignore both fields.
///
/// [`String`] converts in (`ThinkingBlock::from`, [`ThinkingBlock::new`]),
/// and the block derefs to `&str`, so signature-unaware code can treat it
/// like the text it wraps.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct ThinkingBlock {
    /// The reasoning text.
    pub text: String,
    /// Provider-opaque signature that authenticates `text`, when the
    /// provider issued one. Never synthesize or mutate it — replay it
    /// exactly as received or not at all.
    pub signature: Option<String>,
}

impl ThinkingBlock {
    /// A block with no signature.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            signature: None,
        }
    }

    /// A block carrying the provider's signature.
    pub fn with_signature(text: impl Into<String>, signature: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            signature: Some(signature.into()),
        }
    }

    /// True when there is neither text nor a signature to send.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.signature.is_none()
    }
}

impl From<String> for ThinkingBlock {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<&str> for ThinkingBlock {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

impl std::ops::Deref for ThinkingBlock {
    type Target = str;

    fn deref(&self) -> &str {
        &self.text
    }
}

impl<'de> Deserialize<'de> for ThinkingBlock {
    /// Accepts both the current struct form and a bare JSON string, so
    /// conversations persisted before the signature field existed still
    /// load.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Text(String),
            Block {
                text: String,
                #[serde(default)]
                signature: Option<String>,
            },
        }
        Ok(match Repr::deserialize(deserializer)? {
            Repr::Text(text) => Self::new(text),
            Repr::Block { text, signature } => Self { text, signature },
        })
    }
}

/// A single tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call id (synthesized as `call_{index}` for
    /// providers that don't issue ids, e.g. Gemini and Ollama).
    pub id: String,
    /// Name of the tool being invoked.
    pub name: String,
    /// Arguments as a decoded JSON value.
    pub arguments: serde_json::Value,
}

/// The host's answer to one [`ToolCall`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// The [`ToolCall::id`] this result answers.
    pub call_id: String,
    /// Name of the tool that ran (some providers require it on the result).
    pub name: String,
    /// Textual result content.
    pub content: String,
    /// Images produced by the tool (screenshots, renders, ...).
    pub images: Vec<MessageImage>,
    /// True when the tool failed and `content` describes the error.
    pub is_error: bool,
}

/// An inline base64 image attachment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageImage {
    /// MIME type, e.g. `"image/png"`.
    pub media_type: String,
    /// Raw image bytes, base64-encoded (no data-URL prefix).
    pub data_base64: String,
}

/// Declaration of a tool the model is allowed to call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// Human/model-readable description of what the tool does.
    pub description: String,
    /// JSON Schema for the tool's arguments.
    pub input_schema: serde_json::Value,
}

/// A provider-neutral chat-completion request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderRequest {
    /// Model identifier, in the provider's own naming scheme.
    pub model: String,
    /// System prompt (adapters merge this with any
    /// [`ProviderMessage::System`] entries in `messages`).
    pub system: Option<String>,
    /// Conversation transcript, oldest first.
    pub messages: Vec<ProviderMessage>,
    /// Tools the model may call. Empty means no tools are declared.
    pub tools: Vec<ToolSpec>,
    /// Maximum tokens to generate, when the host wants to cap it.
    pub max_tokens: Option<u32>,
    /// Sampling temperature, when the host wants to override the default.
    pub temperature: Option<f32>,
    /// True to request a streaming response.
    pub stream: bool,
}

/// One event decoded from a provider's streaming response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProviderEvent {
    /// The provider started producing a message.
    MessageStart,
    /// A fragment of visible response text.
    TextDelta(String),
    /// A fragment of thinking / reasoning text.
    ThinkingDelta(String),
    /// The provider-opaque signature authenticating this turn's thinking
    /// content (Anthropic `signature_delta`).
    ///
    /// Carries the complete signature for the block; if more than one
    /// arrives, the last one wins. Hosts must store it in
    /// [`ThinkingBlock::signature`] — Anthropic rejects a replayed thinking
    /// block whose signature is missing or altered. Providers that issue no
    /// signature never emit this event.
    ThinkingSignature(String),
    /// The model opened tool call `index` (0-based per message).
    ToolCallStart {
        /// Position of this tool call within the message.
        index: usize,
        /// Provider-assigned (or synthesized) call id.
        id: String,
        /// Name of the tool being invoked.
        name: String,
    },
    /// A fragment of the JSON-encoded arguments for tool call `index`.
    ToolCallArgumentsDelta {
        /// Position of the tool call this fragment belongs to.
        index: usize,
        /// Raw partial-JSON text; concatenate fragments to reconstruct the
        /// full arguments document.
        json_fragment: String,
    },
    /// Tool call `index` is complete; its arguments JSON is now whole.
    ToolCallEnd {
        /// Position of the tool call that finished.
        index: usize,
    },
    /// Token accounting reported by the provider.
    Usage(TokenUsage),
    /// The message finished; no further events will follow for this turn.
    Finished(StopReason),
}

/// Why the provider stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// The model finished its turn normally.
    EndTurn,
    /// The model stopped to call one or more tools.
    ToolUse,
    /// Generation hit the token limit.
    MaxTokens,
    /// A stop sequence was matched.
    StopSequence,
    /// The provider's content filter ended generation.
    ContentFilter,
    /// A provider-specific reason not covered above (raw value preserved).
    Other(String),
}

/// Token accounting for one request/response exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Tokens consumed by the prompt.
    pub input_tokens: u64,
    /// Tokens generated in the response.
    pub output_tokens: u64,
    /// Prompt tokens served from the provider's cache.
    pub cache_read_tokens: u64,
}

/// Host-supplied connection settings for one provider endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Endpoint base URL, e.g. `"https://api.anthropic.com"` or
    /// `"https://api.openai.com/v1"`. Trailing slashes are tolerated.
    pub base_url: String,
    /// API key, when the endpoint requires one.
    pub api_key: Option<String>,
    /// Additional headers appended verbatim to every request.
    pub extra_headers: Vec<(String, String)>,
}

/// A fully-encoded HTTP request, ready for the host's HTTP stack.
///
/// This crate performs no I/O: hosts take this spec, send it with their
/// own client, and feed the response back through
/// [`crate::adapters::ProviderAdapter::decode_response`] or a
/// [`crate::adapters::StreamDecoder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HttpRequestSpec {
    /// HTTP method (always `"POST"` for the built-in adapters).
    pub method: String,
    /// Absolute URL (base URL joined with the provider path).
    pub url: String,
    /// Request headers, in order.
    pub headers: Vec<(String, String)>,
    /// JSON request body.
    pub body: serde_json::Value,
}

/// Errors produced while encoding requests or decoding responses.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// The neutral request could not be encoded into the provider's wire
    /// format.
    #[error("{provider}: failed to encode request: {detail}")]
    Encode {
        /// Adapter that produced the error.
        provider: &'static str,
        /// Human-readable description of what went wrong.
        detail: String,
    },
    /// Response or stream bytes could not be decoded.
    #[error("{provider}: failed to decode response: {detail}")]
    Decode {
        /// Adapter (or framing helper) that produced the error.
        provider: &'static str,
        /// Human-readable description of what went wrong.
        detail: String,
    },
    /// The request uses a feature this provider cannot express.
    #[error("{provider}: unsupported feature: {feature}")]
    Unsupported {
        /// Adapter that produced the error.
        provider: &'static str,
        /// Description of the unsupported feature.
        feature: String,
    },
}

impl ProviderError {
    /// Shorthand constructor for [`ProviderError::Encode`].
    pub(crate) fn encode(provider: &'static str, detail: impl Into<String>) -> Self {
        Self::Encode {
            provider,
            detail: detail.into(),
        }
    }

    /// Shorthand constructor for [`ProviderError::Decode`].
    pub(crate) fn decode(provider: &'static str, detail: impl Into<String>) -> Self {
        Self::Decode {
            provider,
            detail: detail.into(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn thinking_block_converts_from_strings() {
        assert_eq!(
            ThinkingBlock::from("hi".to_owned()),
            ThinkingBlock::new("hi")
        );
        assert_eq!(ThinkingBlock::from("hi"), ThinkingBlock::new("hi"));
        assert_eq!(ThinkingBlock::new("hi").signature, None);
    }

    #[test]
    fn thinking_block_derefs_to_its_text() {
        // Keeps `Option<ThinkingBlock>::as_deref()` and the usual `&str`
        // methods working for signature-unaware call sites.
        let block = Some(ThinkingBlock::with_signature("reasoning", "sig"));
        assert_eq!(block.as_deref(), Some("reasoning"));
        assert_eq!(block.as_deref().map(str::len), Some(9));
    }

    #[test]
    fn thinking_block_is_empty_accounts_for_the_signature() {
        assert!(ThinkingBlock::default().is_empty());
        assert!(ThinkingBlock::new("").is_empty());
        assert!(!ThinkingBlock::new("x").is_empty());
        assert!(!ThinkingBlock::with_signature("", "sig").is_empty());
    }

    #[test]
    fn thinking_block_json_round_trips() {
        let block = ThinkingBlock::with_signature("reasoning", "sig");
        let json = serde_json::to_string(&block).unwrap();
        assert_eq!(json, r#"{"text":"reasoning","signature":"sig"}"#);
        assert_eq!(serde_json::from_str::<ThinkingBlock>(&json).unwrap(), block);
    }

    #[test]
    fn thinking_block_reads_legacy_bare_string() {
        // Sessions persisted before the signature field existed stored
        // `thinking` as a plain JSON string.
        let legacy = r#"{"text":"hi","thinking":"older reasoning","tool_calls":[]}"#;
        let message: AssistantMessage = serde_json::from_str(legacy).unwrap();
        assert_eq!(
            message.thinking,
            Some(ThinkingBlock::new("older reasoning"))
        );
    }

    #[test]
    fn assistant_message_round_trips_through_json() {
        let message = AssistantMessage {
            text: "hi".to_owned(),
            thinking: Some(ThinkingBlock::with_signature("why", "sig")),
            tool_calls: vec![],
        };
        let json = serde_json::to_string(&message).unwrap();
        assert_eq!(
            serde_json::from_str::<AssistantMessage>(&json).unwrap(),
            message
        );
    }
}
