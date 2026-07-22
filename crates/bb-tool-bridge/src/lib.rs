//! Bridge between the agent's tool calls and the host runtime.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! When [`bb-agent-loop`] receives a tool-use intent from a model, this
//! crate dispatches it. The host (Beam or Beacon) injects a
//! [`ToolHandler`] implementation for each tool it supports — shell,
//! filesystem, computer-use, subagents, web — and registers them in a
//! [`ToolBridge`]. The bridge owns:
//!
//! - the dyn-safe [`ToolHandler`] trait surface,
//! - the registry ([`ToolBridge`]) the agent loop talks to,
//! - the uniform result type ([`ToolDispatchResult`]),
//! - the conversion back into provider messages
//!   ([`ToolDispatchResult::into_tool_result`] →
//!   [`bb_providers::ToolResult`]).
//!
//! # Well-known tool specs
//!
//! The [`specs`] module ships constructor functions for the tool
//! declarations both products share (`shell`, `read_file`, ...). They are
//! specs only — implementations are always host-side. Hosts may use them
//! verbatim or bring their own.
//!
//! # Test helpers
//!
//! The [`testing`] module provides [`testing::FnTool`], a one-line way to
//! stub a tool from a sync closure in downstream tests.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-agent-loop`]: https://docs.rs/bb-agent-loop

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod specs;
pub mod testing;

use std::collections::BTreeMap;
use std::sync::Arc;

use bb_providers::{MessageImage, ToolCall, ToolResult, ToolSpec};
use futures::future::BoxFuture;

/// Uniform result of running one tool call.
///
/// Every [`ToolHandler`] produces one of these regardless of what the tool
/// did; the agent loop converts it into a provider-facing
/// [`ToolResult`] via [`ToolDispatchResult::into_tool_result`].
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDispatchResult {
    /// Human/model-readable result text (or error description when
    /// `is_error` is set).
    pub text: String,
    /// Images produced by the tool (screenshots, renders, ...).
    pub images: Vec<MessageImage>,
    /// True when the tool failed and `text` describes the error.
    pub is_error: bool,
    /// Optional machine-readable payload (e.g. exit code, structured file
    /// info). Kept host-side: it is **not** serialized into the provider
    /// [`ToolResult`] — only `text` and `images` reach the model.
    pub structured: Option<serde_json::Value>,
}

impl ToolDispatchResult {
    /// Successful result carrying only text.
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
            is_error: false,
            structured: None,
        }
    }

    /// Failed result; `text` should describe what went wrong.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
            is_error: true,
            structured: None,
        }
    }

    /// Attach images (replacing any already present).
    #[must_use]
    pub fn with_images(mut self, images: Vec<MessageImage>) -> Self {
        self.images = images;
        self
    }

    /// Attach a machine-readable payload (replacing any already present).
    ///
    /// The payload never reaches the model; see the [`Self::structured`]
    /// field docs.
    #[must_use]
    pub fn with_structured(mut self, value: serde_json::Value) -> Self {
        self.structured = Some(value);
        self
    }

    /// Convert into the provider-facing [`ToolResult`] for a given call.
    ///
    /// `call_id` and `name` are taken from `call`; `text`, `images` and
    /// `is_error` map across directly. The `structured` payload is
    /// intentionally dropped here — it exists for host-side consumers
    /// (UI, logging), not for the model.
    pub fn into_tool_result(self, call: &ToolCall) -> ToolResult {
        ToolResult {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: self.text,
            images: self.images,
            is_error: self.is_error,
        }
    }
}

/// Context handed to every tool invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolContext {
    /// Working directory the tool should operate in, when the host has one.
    pub cwd: Option<String>,
    /// Host-defined opaque session id (for permission scoping / logging).
    pub session_id: Option<String>,
}

/// A host-injected tool implementation.
///
/// Dyn-safe; async via [`BoxFuture`]. The host (Beam or Beacon) implements
/// this for each tool kind it supports and registers the handlers in a
/// [`ToolBridge`].
pub trait ToolHandler: Send + Sync {
    /// The spec advertised to the model (name must be unique in a registry).
    fn spec(&self) -> ToolSpec;

    /// Execute the call. Implementations should catch their own panicky
    /// paths and return `is_error` results rather than panicking.
    fn dispatch<'a>(
        &'a self,
        call: &'a ToolCall,
        ctx: &'a ToolContext,
    ) -> BoxFuture<'a, ToolDispatchResult>;
}

/// Error type for [`ToolBridge`] registry operations.
///
/// Note that *dispatching* never returns this — an unknown tool name comes
/// back as an `is_error` [`ToolDispatchResult`] so the model always
/// receives a `tool_result` and the conversation can continue.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolBridgeError {
    /// A handler with this tool name is already registered.
    #[error("a tool named `{0}` is already registered")]
    DuplicateTool(String),
}

/// Registry of tool handlers; what the agent loop talks to.
///
/// Handlers are keyed by their [`ToolSpec::name`] and stored in sorted
/// order, so [`ToolBridge::specs`] is stable across runs (deterministic
/// prompts and tests).
#[derive(Default)]
pub struct ToolBridge {
    handlers: BTreeMap<String, Arc<dyn ToolHandler>>,
}

impl std::fmt::Debug for ToolBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolBridge")
            .field("tools", &self.handlers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ToolBridge {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a handler under its [`ToolSpec::name`].
    ///
    /// # Errors
    ///
    /// Returns [`ToolBridgeError::DuplicateTool`] when a handler with the
    /// same name is already registered; the existing handler is kept.
    pub fn register(&mut self, handler: Arc<dyn ToolHandler>) -> Result<(), ToolBridgeError> {
        let name = handler.spec().name;
        if self.handlers.contains_key(&name) {
            return Err(ToolBridgeError::DuplicateTool(name));
        }
        self.handlers.insert(name, handler);
        Ok(())
    }

    /// True when no handlers are registered.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Specs for every registered tool, sorted by name (stable ordering
    /// for prompts/tests).
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.handlers.values().map(|h| h.spec()).collect()
    }

    /// True when a handler named `name` is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// Dispatch one call to its registered handler.
    ///
    /// An unknown tool name is **not** an `Err` — the model needs a
    /// `tool_result` back to continue the conversation, so it comes back
    /// as an `is_error` [`ToolDispatchResult`] naming the missing tool and
    /// listing what is available.
    pub async fn dispatch(&self, call: &ToolCall, ctx: &ToolContext) -> ToolDispatchResult {
        match self.handlers.get(&call.name) {
            Some(handler) => handler.dispatch(call, ctx).await,
            None => {
                let known = self
                    .handlers
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                ToolDispatchResult::error(format!(
                    "Unknown tool `{}`. Available tools: [{known}]. \
                     Retry with one of the available tool names.",
                    call.name
                ))
            }
        }
    }

    /// Dispatch a batch sequentially, preserving order, returning
    /// provider-ready results (one [`ToolResult`] per call, in the same
    /// order as `calls`).
    pub async fn dispatch_all(&self, calls: &[ToolCall], ctx: &ToolContext) -> Vec<ToolResult> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            let dispatched = self.dispatch(call, ctx).await;
            results.push(dispatched.into_tool_result(call));
        }
        results
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::testing::FnTool;
    use super::*;
    use serde_json::json;

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments: json!({}),
        }
    }

    fn echo_tool(name: &str) -> Arc<dyn ToolHandler> {
        Arc::new(FnTool::new(name, "echoes its arguments", |call, _ctx| {
            ToolDispatchResult::ok(format!("echo:{}", call.arguments))
        }))
    }

    #[test]
    fn tool_context_defaults_are_empty() {
        let ctx = ToolContext::default();
        assert_eq!(ctx.cwd, None);
        assert_eq!(ctx.session_id, None);
    }

    #[test]
    fn register_contains_len_and_duplicate() {
        let mut bridge = ToolBridge::new();
        assert!(bridge.is_empty());
        assert_eq!(bridge.len(), 0);

        bridge.register(echo_tool("alpha")).unwrap();
        assert!(bridge.contains("alpha"));
        assert!(!bridge.contains("beta"));
        assert!(!bridge.is_empty());
        assert_eq!(bridge.len(), 1);

        let err = bridge.register(echo_tool("alpha")).unwrap_err();
        assert_eq!(err, ToolBridgeError::DuplicateTool("alpha".to_owned()));
        assert_eq!(bridge.len(), 1);
    }

    #[test]
    fn specs_are_sorted_by_name() {
        let mut bridge = ToolBridge::new();
        for name in ["zulu", "alpha", "mike"] {
            bridge.register(echo_tool(name)).unwrap();
        }
        let names: Vec<String> = bridge.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["alpha", "mike", "zulu"]);
    }

    #[tokio::test]
    async fn dispatch_happy_path() {
        let mut bridge = ToolBridge::new();
        bridge.register(echo_tool("echo")).unwrap();

        let mut call = call("call_1", "echo");
        call.arguments = json!({"msg": "hi"});
        let result = bridge.dispatch(&call, &ToolContext::default()).await;

        assert!(!result.is_error);
        assert_eq!(result.text, r#"echo:{"msg":"hi"}"#);
        assert!(result.images.is_empty());
        assert_eq!(result.structured, None);
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_is_error_result_not_err() {
        let mut bridge = ToolBridge::new();
        bridge.register(echo_tool("echo")).unwrap();

        let result = bridge
            .dispatch(&call("call_1", "nonexistent"), &ToolContext::default())
            .await;

        assert!(result.is_error);
        assert!(result.text.contains("nonexistent"));
        assert!(result.text.contains("echo"));
    }

    #[tokio::test]
    async fn dispatch_all_preserves_order_and_maps_ids() {
        let mut bridge = ToolBridge::new();
        bridge.register(echo_tool("echo")).unwrap();

        let calls = vec![
            call("call_a", "echo"),
            call("call_b", "missing"),
            call("call_c", "echo"),
        ];
        let results = bridge.dispatch_all(&calls, &ToolContext::default()).await;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].call_id, "call_a");
        assert_eq!(results[0].name, "echo");
        assert!(!results[0].is_error);
        assert_eq!(results[1].call_id, "call_b");
        assert_eq!(results[1].name, "missing");
        assert!(results[1].is_error);
        assert_eq!(results[2].call_id, "call_c");
        assert_eq!(results[2].name, "echo");
        assert!(!results[2].is_error);
    }

    #[test]
    fn into_tool_result_propagates_images_and_drops_structured() {
        let image = MessageImage {
            media_type: "image/png".to_owned(),
            data_base64: "aGVsbG8=".to_owned(),
        };
        let result = ToolDispatchResult::ok("took screenshot")
            .with_images(vec![image.clone()])
            .with_structured(json!({"exit_code": 0}));

        assert_eq!(result.structured, Some(json!({"exit_code": 0})));

        let tool_result = result.into_tool_result(&call("call_9", "computer_use"));
        assert_eq!(tool_result.call_id, "call_9");
        assert_eq!(tool_result.name, "computer_use");
        assert_eq!(tool_result.content, "took screenshot");
        assert_eq!(tool_result.images, vec![image]);
        assert!(!tool_result.is_error);
    }

    #[test]
    fn error_flag_propagates_into_tool_result() {
        let tool_result =
            ToolDispatchResult::error("command failed").into_tool_result(&call("call_2", "shell"));
        assert!(tool_result.is_error);
        assert_eq!(tool_result.content, "command failed");
    }

    #[tokio::test]
    async fn fn_tool_sees_context() {
        let mut bridge = ToolBridge::new();
        bridge
            .register(Arc::new(FnTool::new(
                "pwd",
                "reports the cwd",
                |_call, ctx| ToolDispatchResult::ok(ctx.cwd.clone().unwrap_or_default()),
            )))
            .unwrap();

        let ctx = ToolContext {
            cwd: Some("/tmp/project".to_owned()),
            session_id: Some("session-1".to_owned()),
        };
        let result = bridge.dispatch(&call("call_1", "pwd"), &ctx).await;
        assert_eq!(result.text, "/tmp/project");
    }

    #[test]
    fn well_known_specs_have_object_schemas() {
        let all = [
            specs::shell(),
            specs::read_file(),
            specs::write_file(),
            specs::edit_file(),
            specs::computer_use(),
            specs::web_fetch(),
            specs::web_search(),
            specs::subagent(),
        ];
        for spec in &all {
            assert!(!spec.name.is_empty());
            assert!(!spec.description.is_empty());
            assert_eq!(spec.input_schema["type"], "object", "{}", spec.name);
            assert!(spec.input_schema["properties"].is_object(), "{}", spec.name);
        }
        let subagent = specs::subagent();
        let kinds = subagent.input_schema["properties"]["kind"]["enum"]
            .as_array()
            .unwrap();
        assert!(kinds.contains(&json!("cli")));
        assert!(kinds.contains(&json!("documentation_search")));
    }
}
