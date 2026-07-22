//! Test helpers for stubbing tools.
//!
//! Available unconditionally (not behind a feature flag) so downstream
//! crates — notably `bb-agent-loop` — can stub tools in their own tests
//! without dev-dependency gymnastics.

use bb_providers::{ToolCall, ToolSpec};
use futures::future::BoxFuture;
use futures::FutureExt;

use crate::{ToolContext, ToolDispatchResult, ToolHandler};

/// The boxed sync closure an [`FnTool`] runs on dispatch.
type FnToolHandler = Box<dyn Fn(&ToolCall, &ToolContext) -> ToolDispatchResult + Send + Sync>;

/// A [`ToolHandler`] built from a name, a description and a sync closure.
///
/// The closure runs synchronously and is wrapped into the async trait
/// surface, so a test can stub a tool in one line:
///
/// ```
/// use bb_tool_bridge::testing::FnTool;
/// use bb_tool_bridge::ToolDispatchResult;
///
/// let tool = FnTool::new("echo", "echoes back", |call, _ctx| {
///     ToolDispatchResult::ok(call.arguments.to_string())
/// });
/// ```
pub struct FnTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    handler: FnToolHandler,
}

impl std::fmt::Debug for FnTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("input_schema", &self.input_schema)
            .finish_non_exhaustive()
    }
}

impl FnTool {
    /// Build a stub tool from `name`, `description` and a sync closure.
    ///
    /// The advertised `input_schema` defaults to a permissive empty object
    /// schema; use [`FnTool::with_schema`] to override it.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        handler: impl Fn(&ToolCall, &ToolContext) -> ToolDispatchResult + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            handler: Box::new(handler),
        }
    }

    /// Replace the advertised `input_schema`.
    #[must_use]
    pub fn with_schema(mut self, input_schema: serde_json::Value) -> Self {
        self.input_schema = input_schema;
        self
    }
}

impl ToolHandler for FnTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }

    fn dispatch<'a>(
        &'a self,
        call: &'a ToolCall,
        ctx: &'a ToolContext,
    ) -> BoxFuture<'a, ToolDispatchResult> {
        let result = (self.handler)(call, ctx);
        std::future::ready(result).boxed()
    }
}
