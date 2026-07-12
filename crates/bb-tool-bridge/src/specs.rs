//! Constructor functions for the well-known tool specs Beam and Beacon
//! share.
//!
//! These are declarations only — the model-facing name, description and
//! JSON-schema `input_schema` for each tool. Implementations are always
//! host-side [`crate::ToolHandler`]s. Hosts may advertise these specs
//! verbatim (recommended, so prompts stay consistent across products) or
//! bring their own specs with different names/schemas.

use bb_providers::ToolSpec;
use serde_json::json;

/// Spec for the `shell` tool: run a command in the host shell.
pub fn shell() -> ToolSpec {
    ToolSpec {
        name: "shell".to_owned(),
        description: "Run a shell command in the user's terminal and return its output. \
                      Output includes stdout and stderr; the exit code is reported when \
                      the command fails."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command line to execute."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory to run in. Defaults to the session's current directory."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds; the command is terminated when exceeded."
                }
            },
            "required": ["command"]
        }),
    }
}

/// Spec for the `read_file` tool: read a file from the host filesystem.
pub fn read_file() -> ToolSpec {
    ToolSpec {
        name: "read_file".to_owned(),
        description: "Read a file from the filesystem and return its contents as text. \
                      Supports reading a slice of the file via offset/limit for large files."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file to read."
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from. Defaults to the beginning."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return."
                }
            },
            "required": ["path"]
        }),
    }
}

/// Spec for the `write_file` tool: create or overwrite a file.
pub fn write_file() -> ToolSpec {
    ToolSpec {
        name: "write_file".to_owned(),
        description: "Write content to a file, creating it if it does not exist and \
                      overwriting it if it does. Parent directories are created as needed."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file."
                }
            },
            "required": ["path", "content"]
        }),
    }
}

/// Spec for the `edit_file` tool: exact string replacement in a file.
pub fn edit_file() -> ToolSpec {
    ToolSpec {
        name: "edit_file".to_owned(),
        description: "Edit a file by replacing an exact string with a new string. \
                      The old string must match the file contents exactly (including \
                      whitespace) and must be unique unless replace_all is set."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file to edit."
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace."
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring a unique match. Defaults to false."
                }
            },
            "required": ["path", "old_string", "new_string"]
        }),
    }
}

/// Spec for the `computer_use` tool: screenshot / mouse / keyboard control.
pub fn computer_use() -> ToolSpec {
    ToolSpec {
        name: "computer_use".to_owned(),
        description: "Interact with the user's screen: take screenshots, move and click \
                      the mouse, type text, and press keys. Screenshot results are \
                      returned as images."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "screenshot",
                        "left_click",
                        "right_click",
                        "double_click",
                        "mouse_move",
                        "type",
                        "key",
                        "scroll",
                        "wait"
                    ],
                    "description": "The action to perform."
                },
                "coordinate": {
                    "type": "array",
                    "items": { "type": "integer" },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[x, y] pixel coordinate for mouse actions."
                },
                "text": {
                    "type": "string",
                    "description": "Text to type (for `type`) or key chord to press (for `key`)."
                },
                "duration_ms": {
                    "type": "integer",
                    "description": "How long to wait, in milliseconds (for `wait`)."
                }
            },
            "required": ["action"]
        }),
    }
}

/// Spec for the `web_fetch` tool: fetch one URL and return its content.
pub fn web_fetch() -> ToolSpec {
    ToolSpec {
        name: "web_fetch".to_owned(),
        description: "Fetch a URL over HTTP(S) and return the page content as text \
                      (HTML converted to readable form where possible)."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch."
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional question to answer using the fetched content."
                }
            },
            "required": ["url"]
        }),
    }
}

/// Spec for the `web_search` tool: search the web.
pub fn web_search() -> ToolSpec {
    ToolSpec {
        name: "web_search".to_owned(),
        description: "Search the web and return a list of results (title, URL, snippet)."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return."
                }
            },
            "required": ["query"]
        }),
    }
}

/// Spec for the `subagent` tool: delegate a task to a specialized subagent.
pub fn subagent() -> ToolSpec {
    ToolSpec {
        name: "subagent".to_owned(),
        description: "Delegate a self-contained task to a specialized subagent and \
                      return its final report. Choose the `kind` that matches the task."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "cli",
                        "research",
                        "advice",
                        "summarization",
                        "computer_use",
                        "conversation_search",
                        "documentation_search"
                    ],
                    "description": "Which specialized subagent to run."
                },
                "prompt": {
                    "type": "string",
                    "description": "Self-contained task description for the subagent."
                }
            },
            "required": ["kind", "prompt"]
        }),
    }
}
