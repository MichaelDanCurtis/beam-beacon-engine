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
    /// Extended-thinking / reasoning text, when the provider surfaced it.
    pub thinking: Option<String>,
    /// Tool invocations the model requested this turn, in order.
    pub tool_calls: Vec<ToolCall>,
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
