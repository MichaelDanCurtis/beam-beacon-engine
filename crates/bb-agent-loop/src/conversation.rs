//! Conversation state: an ordered history of [`ProviderMessage`]s with
//! approximate token accounting and context-window trimming.

use bb_providers::{ProviderMessage, UserContent};

use crate::config::ContextPolicy;

/// Flat char cost assigned to each inline image when approximating tokens
/// (~1500 tokens at the chars/4 heuristic).
const IMAGE_APPROX_CHARS: usize = 6_000;

/// Prefix the rolling summary is rendered with when it is spliced into the
/// provider request; see [`Conversation::provider_messages`].
const SUMMARY_PREFIX: &str = "Summary of earlier conversation: ";

/// Conversation state: the message history one agent session accumulates.
///
/// The [`AgentLoop`](crate::AgentLoop) reads and appends to a
/// `Conversation` during [`run_turn`](crate::AgentLoop::run_turn); hosts
/// own the value and can persist/inspect it between turns.
///
/// # The rolling summary slot
///
/// A conversation holds **at most one** engine-generated summary of
/// history that context trimming has evicted, in a dedicated field rather
/// than as a message in [`messages`](Self::messages). Two reasons:
///
/// - **Boundedness.** Appending a fresh `System` message per trim is an
///   unbounded ratchet: `System` messages are never evictable, so every
///   turn past saturation added one more permanently-resident message and
///   one more summarizer call, growing the request forever with no way
///   back under budget. One slot that is *replaced* (the previous summary
///   is handed to the [`Summarizer`](crate::Summarizer) as prior context
///   so it can be folded into the new one) is bounded by construction.
/// - **Ownership.** The engine's summary and a host-supplied system prompt
///   are structurally distinct, so trimming can never rewrite, displace or
///   accidentally fold away the host's prompt, and neither side has to
///   sniff message text to tell them apart.
///
/// [`messages`](Self::messages) therefore returns only host-owned history.
/// The sequence actually sent to the provider — summary included — is
/// [`provider_messages`](Self::provider_messages). Hosts that persist a
/// conversation should persist [`summary`](Self::summary) alongside
/// [`messages`](Self::messages) and restore it with
/// [`set_summary`](Self::set_summary).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Conversation {
    messages: Vec<ProviderMessage>,
    summary: Option<String>,
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
            summary: None,
        }
    }

    /// The host-owned message history, oldest first.
    ///
    /// This does **not** include the engine's rolling summary of evicted
    /// history — that lives in [`summary`](Self::summary). For the exact
    /// sequence sent to the provider use
    /// [`provider_messages`](Self::provider_messages).
    pub fn messages(&self) -> &[ProviderMessage] {
        &self.messages
    }

    /// The engine's rolling summary of evicted history, when trimming has
    /// produced one.
    ///
    /// There is at most one, and a new summary replaces it rather than
    /// accumulating; see the type-level docs.
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Replace the rolling summary slot.
    ///
    /// The engine calls this after context trimming. Hosts call it to
    /// restore a persisted conversation, so a reloaded session keeps the
    /// summary of history it evicted in an earlier process.
    pub fn set_summary(&mut self, summary: impl Into<String>) {
        self.summary = Some(summary.into());
    }

    /// Drop the rolling summary, if any.
    pub fn clear_summary(&mut self) {
        self.summary = None;
    }

    /// The exact message sequence a provider request carries: the host
    /// history with the rolling summary spliced in as a `System` message,
    /// immediately after the first `System` message (or at the front when
    /// there is none).
    ///
    /// The summary is rendered as `"Summary of earlier conversation:
    /// {summary}"`. Without a summary this is just a copy of
    /// [`messages`](Self::messages).
    pub fn provider_messages(&self) -> Vec<ProviderMessage> {
        match &self.summary {
            None => self.messages.clone(),
            Some(summary) => {
                let pos = self.summary_position();
                let mut out = Vec::with_capacity(self.messages.len() + 1);
                out.extend_from_slice(&self.messages[..pos]);
                out.push(ProviderMessage::System(format!(
                    "{SUMMARY_PREFIX}{summary}"
                )));
                out.extend_from_slice(&self.messages[pos..]);
                out
            }
        }
    }

    /// Append a message to the history.
    pub fn push(&mut self, msg: ProviderMessage) {
        self.messages.push(msg);
    }

    /// Remove every message **and** the rolling summary.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.summary = None;
    }

    /// Approximate token count for what a provider request would carry —
    /// the history plus the rolling summary.
    ///
    /// Heuristic: total characters of textual content divided by 4, summed
    /// per message. Assistant tool-call arguments and tool-result content
    /// count as text; each inline image counts as a flat ~1500 tokens.
    ///
    /// The summary counts because it occupies the same budget; that is
    /// what keeps the trimming loop honest about non-evictable overhead.
    pub fn approx_tokens(&self) -> usize {
        let messages: usize = self.messages.iter().map(approx_message_tokens).sum();
        messages + self.summary_approx_tokens()
    }

    /// Approximate tokens the rolling summary contributes, as rendered.
    fn summary_approx_tokens(&self) -> usize {
        self.summary.as_ref().map_or(0, |summary| {
            (SUMMARY_PREFIX.chars().count() + summary.chars().count()) / 4
        })
    }

    /// Index in [`messages`](Self::messages) the summary is spliced at.
    fn summary_position(&self) -> usize {
        self.messages
            .iter()
            .position(|m| matches!(m, ProviderMessage::System(_)))
            .map_or(0, |p| p + 1)
    }

    /// Which message indices trimming would evict to fit `policy`, in
    /// ascending order (empty when no trimming is needed).
    ///
    /// **Pure**: nothing is mutated. Planning is split from committing so
    /// the agent loop can run the (cancellable, possibly slow)
    /// [`Summarizer`](crate::Summarizer) hook *before* any history is
    /// destroyed — a turn cancelled inside the summarizer must leave the
    /// conversation exactly as it found it. See
    /// [`commit_eviction`](Self::commit_eviction).
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
    pub(crate) fn plan_eviction(&self, policy: &ContextPolicy) -> Vec<usize> {
        let mut current = self.approx_tokens();
        if current <= policy.max_context_tokens {
            return Vec::new();
        }

        let len = self.messages.len();
        let protected_from = len.saturating_sub(policy.keep_recent_messages);
        let mut plan = Vec::new();

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
                for i in start..=end {
                    plan.push(i);
                    current = current.saturating_sub(approx_message_tokens(&self.messages[i]));
                }
            }
            start = end + 1;
        }

        plan
    }

    /// Copies of the messages `plan` would evict, in history order.
    ///
    /// Used to show the [`Summarizer`](crate::Summarizer) what is about to
    /// go while the conversation is still intact. Copying is what buys
    /// atomicity: it costs one clone of content that is about to be
    /// dropped anyway, and only on the trimming path.
    pub(crate) fn evicted_preview(&self, plan: &[usize]) -> Vec<ProviderMessage> {
        plan.iter()
            .filter_map(|&i| self.messages.get(i).cloned())
            .collect()
    }

    /// Apply an eviction `plan` from [`plan_eviction`](Self::plan_eviction),
    /// returning the evicted messages in their original order.
    ///
    /// `plan` must be ascending and in range; indices that are not are
    /// ignored.
    pub(crate) fn commit_eviction(&mut self, plan: &[usize]) -> Vec<ProviderMessage> {
        if plan.is_empty() {
            return Vec::new();
        }
        let mut evicted = Vec::with_capacity(plan.len());
        let mut kept = Vec::with_capacity(self.messages.len());
        let mut next = 0usize;
        for (i, msg) in std::mem::take(&mut self.messages).into_iter().enumerate() {
            if plan.get(next) == Some(&i) {
                next += 1;
                evicted.push(msg);
            } else {
                kept.push(msg);
            }
        }
        self.messages = kept;
        evicted
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

    /// Plan + commit in one step, the way the loop's trimming does.
    fn evict(convo: &mut Conversation, policy: &ContextPolicy) -> Vec<ProviderMessage> {
        let plan = convo.plan_eviction(policy);
        convo.commit_eviction(&plan)
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
        let evicted = evict(&mut convo, &ContextPolicy::default());
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
        let evicted = evict(&mut convo, &policy);
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
        let evicted = evict(&mut convo, &policy);
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
        let evicted = evict(&mut convo, &policy);
        assert_eq!(evicted, vec![user(&"a".repeat(400))]);
        assert_eq!(convo.messages()[0], a1);
        assert_eq!(convo.messages()[1], r1);
    }

    #[test]
    fn summary_is_spliced_after_first_system_or_at_front() {
        let mut convo = Conversation::with_system("sys");
        convo.push(user("hello"));
        convo.set_summary("old stuff");
        assert_eq!(
            convo.provider_messages()[1],
            ProviderMessage::System("Summary of earlier conversation: old stuff".to_owned())
        );
        // The host history itself is untouched.
        assert_eq!(
            convo.messages(),
            &[ProviderMessage::System("sys".to_owned()), user("hello")]
        );

        let mut no_system = Conversation::new();
        no_system.push(user("hello"));
        no_system.set_summary("old stuff");
        assert!(matches!(
            no_system.provider_messages()[0],
            ProviderMessage::System(_)
        ));
    }

    #[test]
    fn summary_slot_replaces_and_never_accumulates() {
        let mut convo = Conversation::with_system("sys");
        convo.push(user("hello"));
        for i in 0..50 {
            convo.set_summary(format!("summary {i}"));
        }
        assert_eq!(convo.summary(), Some("summary 49"));
        // Exactly one summary System message, plus the host's own.
        let systems = convo
            .provider_messages()
            .into_iter()
            .filter(|m| matches!(m, ProviderMessage::System(_)))
            .count();
        assert_eq!(systems, 2);
        assert_eq!(convo.provider_messages().len(), convo.messages().len() + 1);

        convo.clear_summary();
        assert_eq!(convo.summary(), None);
        assert_eq!(convo.provider_messages().len(), convo.messages().len());
    }

    #[test]
    fn summary_counts_against_the_token_budget() {
        let mut convo = Conversation::new();
        convo.push(user(&"a".repeat(400)));
        assert_eq!(convo.approx_tokens(), 100);
        convo.set_summary("s".repeat(400));
        // 400 chars of summary + the rendered prefix, over 4.
        let expected = 100 + (400 + "Summary of earlier conversation: ".len()) / 4;
        assert_eq!(convo.approx_tokens(), expected);
    }

    #[test]
    fn clear_drops_history_and_summary() {
        let mut convo = Conversation::with_system("sys");
        convo.push(user("hi"));
        convo.set_summary("old");
        convo.clear();
        assert!(convo.messages().is_empty());
        assert_eq!(convo.summary(), None);
        assert!(convo.provider_messages().is_empty());
    }

    #[test]
    fn plan_eviction_is_pure_until_committed() {
        let mut convo = Conversation::new();
        convo.push(user(&"a".repeat(400)));
        convo.push(user(&"b".repeat(400)));
        convo.push(user("tail"));
        let before = convo.clone();

        let policy = ContextPolicy {
            max_context_tokens: 10,
            keep_recent_messages: 1,
        };
        let plan = convo.plan_eviction(&policy);
        assert_eq!(plan, vec![0, 1]);
        // Planning alone changed nothing — this is what makes trimming
        // atomic with respect to cancellation.
        assert_eq!(convo, before);

        let evicted = convo.commit_eviction(&plan);
        assert_eq!(evicted.len(), 2);
        assert_eq!(convo.messages(), &[user("tail")]);
    }

    #[test]
    fn commit_eviction_handles_non_contiguous_plans() {
        // System messages interleaved mean the plan is not a prefix.
        let mut convo = Conversation::new();
        convo.push(user(&"a".repeat(400)));
        convo.push(ProviderMessage::System("mid".to_owned()));
        convo.push(user(&"b".repeat(400)));
        convo.push(user("tail"));

        let policy = ContextPolicy {
            max_context_tokens: 10,
            keep_recent_messages: 1,
        };
        let plan = convo.plan_eviction(&policy);
        assert_eq!(plan, vec![0, 2]);
        let evicted = convo.commit_eviction(&plan);
        assert_eq!(
            evicted,
            vec![user(&"a".repeat(400)), user(&"b".repeat(400))]
        );
        assert_eq!(
            convo.messages(),
            &[ProviderMessage::System("mid".to_owned()), user("tail")]
        );
    }

    #[test]
    fn evicted_preview_matches_what_commit_removes() {
        let mut convo = Conversation::new();
        convo.push(user(&"a".repeat(400)));
        convo.push(user(&"b".repeat(400)));
        convo.push(user("tail"));
        let policy = ContextPolicy {
            max_context_tokens: 10,
            keep_recent_messages: 1,
        };
        let plan = convo.plan_eviction(&policy);
        let preview = convo.evicted_preview(&plan);
        let evicted = convo.commit_eviction(&plan);
        assert_eq!(preview, evicted);
    }
}
