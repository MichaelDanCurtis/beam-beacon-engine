//! Conversation state: an ordered history of [`ProviderMessage`]s with
//! approximate token accounting and context-window trimming.

use bb_providers::{ProviderMessage, UserContent};

use crate::config::ContextPolicy;

/// Flat char cost assigned to each inline image when approximating tokens
/// (~1500 tokens at the chars/4 heuristic).
const IMAGE_APPROX_CHARS: usize = 6_000;

/// Conversation state: the message history one agent session accumulates.
///
/// The [`AgentLoop`](crate::AgentLoop) reads and appends to a
/// `Conversation` during [`run_turn`](crate::AgentLoop::run_turn); hosts
/// own the value and can persist/inspect it between turns.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Conversation {
    messages: Vec<ProviderMessage>,
}

impl Conversation {
    /// Create an empty conversation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a conversation seeded with one `System` message.
    pub fn with_system(system: impl Into<String>) -> Self {
        Self {
            messages: vec![ProviderMessage::System(system.into())],
        }
    }

    /// The full message history, oldest first.
    pub fn messages(&self) -> &[ProviderMessage] {
        &self.messages
    }

    /// Append a message to the history.
    pub fn push(&mut self, msg: ProviderMessage) {
        self.messages.push(msg);
    }

    /// Remove every message.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Approximate token count for the whole history.
    ///
    /// Heuristic: total characters of textual content divided by 4, summed
    /// per message. Assistant tool-call arguments and tool-result content
    /// count as text; each inline image counts as a flat ~1500 tokens.
    pub fn approx_tokens(&self) -> usize {
        self.messages.iter().map(approx_message_tokens).sum()
    }

    /// Evict oldest messages until the history fits `policy`, returning the
    /// evicted messages in their original order (empty when no trimming was
    /// needed).
    ///
    /// Rules:
    /// - `System` messages are never evicted.
    /// - The most recent `policy.keep_recent_messages` messages are never
    ///   evicted.
    /// - An `Assistant` message with tool calls and the `ToolResults`
    ///   message immediately following it form one eviction unit: they are
    ///   kept or evicted together, so no provider encoder ever sees orphan
    ///   tool results.
    ///
    /// Best effort: when everything evictable is gone the history may still
    /// exceed the budget.
    pub(crate) fn evict_over_budget(&mut self, policy: &ContextPolicy) -> Vec<ProviderMessage> {
        let mut current = self.approx_tokens();
        if current <= policy.max_context_tokens {
            return Vec::new();
        }

        let len = self.messages.len();
        let protected_from = len.saturating_sub(policy.keep_recent_messages);
        let mut evict = vec![false; len];

        let mut start = 0usize;
        while start < len && current > policy.max_context_tokens {
            // Unit = [start] or [start, start+1] for an assistant-with-tool-
            // calls immediately followed by its tool results.
            let end = if is_tool_call_assistant(&self.messages[start])
                && start + 1 < len
                && matches!(self.messages[start + 1], ProviderMessage::ToolResults(_))
            {
                start + 1
            } else {
                start
            };

            let protected = (start..=end).any(|i| {
                i >= protected_from || matches!(self.messages[i], ProviderMessage::System(_))
            });
            if !protected {
                for (i, flag) in evict.iter_mut().enumerate().take(end + 1).skip(start) {
                    *flag = true;
                    current = current.saturating_sub(approx_message_tokens(&self.messages[i]));
                }
            }
            start = end + 1;
        }

        if !evict.contains(&true) {
            return Vec::new();
        }

        let mut kept = Vec::with_capacity(len);
        let mut evicted = Vec::new();
        for (i, msg) in std::mem::take(&mut self.messages).into_iter().enumerate() {
            if evict[i] {
                evicted.push(msg);
            } else {
                kept.push(msg);
            }
        }
        self.messages = kept;
        evicted
    }

    /// Insert a summary of evicted history as a `System` message positioned
    /// right after the first `System` message (or at the front when there
    /// is none).
    pub(crate) fn insert_summary(&mut self, summary: &str) {
        let msg = ProviderMessage::System(format!("Summary of earlier conversation: {summary}"));
        let pos = self
            .messages
            .iter()
            .position(|m| matches!(m, ProviderMessage::System(_)))
            .map_or(0, |p| p + 1);
        self.messages.insert(pos, msg);
    }
}

/// True for an assistant message that requested at least one tool call.
fn is_tool_call_assistant(msg: &ProviderMessage) -> bool {
    matches!(msg, ProviderMessage::Assistant(a) if !a.tool_calls.is_empty())
}

/// chars/4 heuristic for one message; see [`Conversation::approx_tokens`].
fn approx_message_tokens(msg: &ProviderMessage) -> usize {
    let chars = match msg {
        ProviderMessage::System(s) | ProviderMessage::User(s) => s.chars().count(),
        ProviderMessage::UserMultimodal(parts) => parts
            .iter()
            .map(|part| match part {
                UserContent::Text(t) => t.chars().count(),
                UserContent::Image(_) => IMAGE_APPROX_CHARS,
            })
            .sum(),
        ProviderMessage::Assistant(a) => {
            a.text.chars().count()
                + a.thinking.as_deref().map_or(0, |t| t.chars().count())
                + a.tool_calls
                    .iter()
                    .map(|c| c.name.chars().count() + c.arguments.to_string().chars().count())
                    .sum::<usize>()
        }
        ProviderMessage::ToolResults(results) => results
            .iter()
            .map(|r| {
                r.name.chars().count()
                    + r.content.chars().count()
                    + r.images.len() * IMAGE_APPROX_CHARS
            })
            .sum(),
    };
    chars / 4
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use bb_providers::{AssistantMessage, ToolCall, ToolResult};
    use serde_json::json;

    use super::*;

    fn user(text: &str) -> ProviderMessage {
        ProviderMessage::User(text.to_owned())
    }

    fn assistant(text: &str) -> ProviderMessage {
        ProviderMessage::Assistant(AssistantMessage {
            text: text.to_owned(),
            ..AssistantMessage::default()
        })
    }

    fn tool_pair(text_len: usize) -> (ProviderMessage, ProviderMessage) {
        let call = ToolCall {
            id: "call_1".to_owned(),
            name: "shell".to_owned(),
            arguments: json!({"cmd": "ls"}),
        };
        let assistant = ProviderMessage::Assistant(AssistantMessage {
            text: String::new(),
            thinking: None,
            tool_calls: vec![call],
        });
        let results = ProviderMessage::ToolResults(vec![ToolResult {
            call_id: "call_1".to_owned(),
            name: "shell".to_owned(),
            content: "x".repeat(text_len),
            images: Vec::new(),
            is_error: false,
        }]);
        (assistant, results)
    }

    #[test]
    fn with_system_and_push_and_clear() {
        let mut convo = Conversation::with_system("be terse");
        assert_eq!(
            convo.messages(),
            &[ProviderMessage::System("be terse".to_owned())]
        );
        convo.push(user("hi"));
        assert_eq!(convo.messages().len(), 2);
        convo.clear();
        assert!(convo.messages().is_empty());
    }

    #[test]
    fn approx_tokens_is_chars_over_four() {
        let mut convo = Conversation::new();
        convo.push(user(&"a".repeat(400)));
        assert_eq!(convo.approx_tokens(), 100);
    }

    #[test]
    fn no_eviction_when_under_budget() {
        let mut convo = Conversation::new();
        convo.push(user("short"));
        let evicted = convo.evict_over_budget(&ContextPolicy::default());
        assert!(evicted.is_empty());
        assert_eq!(convo.messages().len(), 1);
    }

    #[test]
    fn eviction_stops_once_under_budget_and_keeps_system() {
        let mut convo = Conversation::with_system("sys");
        convo.push(user(&"a".repeat(400))); // 100 tokens
        convo.push(assistant(&"b".repeat(400))); // 100 tokens
        convo.push(user(&"c".repeat(400))); // 100 tokens
        convo.push(assistant("done")); // ~1 token

        let policy = ContextPolicy {
            max_context_tokens: 220,
            keep_recent_messages: 2,
        };
        let evicted = convo.evict_over_budget(&policy);
        // Evicting the first user message (100 tokens) brings the total to
        // ~201 <= 220, so eviction stops there.
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], user(&"a".repeat(400)));
        assert!(matches!(
            convo.messages()[0],
            ProviderMessage::System(ref s) if s == "sys"
        ));
        assert_eq!(convo.messages().len(), 4);
    }

    #[test]
    fn tool_pair_is_one_eviction_unit() {
        let mut convo = Conversation::new();
        let (a1, r1) = tool_pair(400);
        convo.push(a1.clone());
        convo.push(r1.clone());
        convo.push(user(&"z".repeat(400)));
        convo.push(assistant("tail"));

        let policy = ContextPolicy {
            max_context_tokens: 10,
            keep_recent_messages: 2,
        };
        let evicted = convo.evict_over_budget(&policy);
        // The pair is evicted together, never split.
        assert_eq!(evicted, vec![a1, r1]);
        assert_eq!(convo.messages().len(), 2);
    }

    #[test]
    fn tool_pair_partially_protected_is_fully_kept() {
        let mut convo = Conversation::new();
        convo.push(user(&"a".repeat(400)));
        let (a1, r1) = tool_pair(400);
        convo.push(a1.clone());
        convo.push(r1.clone());
        convo.push(user("tail"));

        // keep_recent_messages = 2 protects [r1, tail]; a1 must survive too
        // so r1 is never orphaned.
        let policy = ContextPolicy {
            max_context_tokens: 10,
            keep_recent_messages: 2,
        };
        let evicted = convo.evict_over_budget(&policy);
        assert_eq!(evicted, vec![user(&"a".repeat(400))]);
        assert_eq!(convo.messages()[0], a1);
        assert_eq!(convo.messages()[1], r1);
    }

    #[test]
    fn insert_summary_goes_after_first_system_or_front() {
        let mut convo = Conversation::with_system("sys");
        convo.push(user("hello"));
        convo.insert_summary("old stuff");
        assert_eq!(
            convo.messages()[1],
            ProviderMessage::System("Summary of earlier conversation: old stuff".to_owned())
        );

        let mut no_system = Conversation::new();
        no_system.push(user("hello"));
        no_system.insert_summary("old stuff");
        assert!(matches!(
            no_system.messages()[0],
            ProviderMessage::System(_)
        ));
    }
}
