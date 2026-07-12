//! The [`Session`] aggregate and its [`SessionId`].

use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bb_agent_loop::{
    AgentError, AgentEvent, AgentLoop, CancellationToken, Conversation, TurnOutcome, UserInput,
};
use bb_block_list::{BlockList, PromptMarker};
use bb_providers::ProviderMessage;
use bb_terminal_model::Screen;
use bb_tool_bridge::ToolContext;
use bb_vt_parser::{Parser, VtEvent};
use serde::{Deserialize, Serialize};

use crate::IdentityContext;

/// Process-wide counter mixed into [`SessionId::generate`] so ids created
/// in the same nanosecond still differ.
static SESSION_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Stable identity for a [`Session`], unique per host installation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    /// Generate a fresh id without any external randomness dependency.
    ///
    /// The id is 32 lowercase hex chars derived from two independently
    /// seeded [`RandomState`] hashers (the std `HashMap` SipHash seeds,
    /// which are randomly initialized per process) fed with the current
    /// wall-clock nanoseconds and a process-wide atomic counter. The
    /// counter alone guarantees in-process uniqueness; the random seeds
    /// plus timestamp make cross-process collisions vanishingly unlikely.
    /// This is collision-resistant enough for local session ids — it is
    /// **not** a cryptographic token.
    pub fn generate() -> Self {
        let count = SESSION_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        let mut high = RandomState::new().build_hasher();
        high.write_u128(nanos);
        high.write_u64(count);
        let high = high.finish();

        let mut low = RandomState::new().build_hasher();
        low.write_u64(count);
        low.write_u64(high);
        low.write_u128(nanos);
        let low = low.finish();

        Self(format!("{high:016x}{low:016x}"))
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Everything the UI needs about one terminal session.
///
/// A `Session` aggregates the whole engine stack for a single PTY:
///
/// - the [`Screen`] (grid + scrollback) driven by a [`Parser`],
/// - the [`BlockList`] view over that screen,
/// - the agent [`Conversation`] for this session,
/// - metadata (title, cwd) and the host-surfaced [`IdentityContext`].
///
/// The host owns the PTY and the provider/tool wiring; the session owns
/// the state in between. Feed PTY output through
/// [`Session::feed_pty`], run agent turns through
/// [`Session::run_agent_turn`], and persist/restore durable state via
/// [`SessionHistory`](crate::SessionHistory).
#[derive(Debug)]
pub struct Session {
    id: SessionId,
    title: Option<String>,
    cwd: Option<String>,
    identity: IdentityContext,
    screen: Screen,
    parser: Parser,
    blocks: BlockList,
    conversation: Conversation,
}

impl Session {
    /// Create a fresh session; `rows`/`cols` are the initial PTY size.
    pub fn new(id: SessionId, rows: usize, cols: usize) -> Self {
        Self {
            id,
            title: None,
            cwd: None,
            identity: IdentityContext::default(),
            screen: Screen::new(rows, cols),
            parser: Parser::new(),
            blocks: BlockList::new(),
            conversation: Conversation::new(),
        }
    }

    /// Rebuild a session from persisted parts (see
    /// [`SessionHistory::restore`](crate::SessionHistory::restore)). The
    /// screen and block list start fresh: they are live views over the PTY
    /// and are rebuilt from PTY output on reattach, never persisted.
    pub(crate) fn from_parts(
        id: SessionId,
        title: Option<String>,
        cwd: Option<String>,
        identity: IdentityContext,
        messages: Vec<ProviderMessage>,
        rows: usize,
        cols: usize,
    ) -> Self {
        let mut conversation = Conversation::new();
        for message in messages {
            conversation.push(message);
        }
        Self {
            id,
            title,
            cwd,
            identity,
            screen: Screen::new(rows, cols),
            parser: Parser::new(),
            blocks: BlockList::new(),
            conversation,
        }
    }

    /// This session's stable identity.
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    /// The window title, when the shell has reported one (OSC 0/2).
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// The working directory, when the shell has reported one (OSC 7).
    pub fn cwd(&self) -> Option<&str> {
        self.cwd.as_deref()
    }

    /// The identity context this session runs under.
    pub fn identity(&self) -> &IdentityContext {
        &self.identity
    }

    /// Replace the identity context (host calls this on sign-in/out).
    pub fn set_identity(&mut self, id: IdentityContext) {
        self.identity = id;
    }

    // ------------------------------------------------------------------
    // Terminal side
    // ------------------------------------------------------------------

    /// Feed raw PTY bytes into the session.
    ///
    /// Drives the [`Parser`] into the [`Screen`], then translates the
    /// semantic [`VtEvent`]s into engine state:
    ///
    /// - OSC 133 prompt markers → [`BlockList::on_marker`], stamped with
    ///   the cursor's absolute row *after* this feed (feed in small chunks
    ///   for finer-grained marker rows);
    /// - [`VtEvent::WorkingDirectory`] → [`BlockList::set_cwd`] and the
    ///   session cwd. The `file://` URL is parsed leniently: the scheme
    ///   and host are stripped and the remaining path is used verbatim —
    ///   **no percent-decoding is performed**. A bare absolute path is
    ///   accepted as-is; anything unparseable leaves the cwd unchanged;
    /// - [`VtEvent::Title`] → the session title.
    ///
    /// After the feed, [`BlockList::observe_row`] is called with the
    /// current cursor row so open blocks track output growth. The full
    /// event list is returned for the host UI (bells, hyperlinks, ...).
    ///
    /// Feeding an empty byte slice is a no-op and returns no events.
    pub fn feed_pty(&mut self, bytes: &[u8]) -> Vec<VtEvent> {
        if bytes.is_empty() {
            return Vec::new();
        }

        let events = self.parser.feed(&mut self.screen, bytes);
        let cursor_row = self.screen.cursor_absolute_row();
        for event in &events {
            match event {
                VtEvent::PromptStart => {
                    self.blocks.on_marker(PromptMarker::PromptStart, cursor_row);
                }
                VtEvent::CommandStart => {
                    self.blocks
                        .on_marker(PromptMarker::CommandStart, cursor_row);
                }
                VtEvent::CommandExecuted => {
                    self.blocks
                        .on_marker(PromptMarker::CommandExecuted, cursor_row);
                }
                VtEvent::CommandFinished { exit_code } => {
                    self.blocks.on_marker(
                        PromptMarker::CommandFinished {
                            exit_code: *exit_code,
                        },
                        cursor_row,
                    );
                }
                VtEvent::Title(title) => {
                    self.title = Some(title.clone());
                }
                VtEvent::WorkingDirectory(url) => {
                    if let Some(path) = parse_file_url_path(url) {
                        self.blocks.set_cwd(&path);
                        self.cwd = Some(path);
                    }
                }
                // Bells, hyperlinks, and any future event kinds are the
                // host UI's business; they pass through in the return
                // value untouched.
                _ => {}
            }
        }
        self.blocks.observe_row(self.screen.cursor_absolute_row());
        events
    }

    /// The terminal screen (grid, scrollback, cursor).
    pub fn screen(&self) -> &Screen {
        &self.screen
    }

    /// The block list view over the screen.
    pub fn blocks(&self) -> &BlockList {
        &self.blocks
    }

    /// Resize the PTY view. Blocks (absolute-row addressed) and the
    /// conversation are unaffected.
    pub fn resize(&mut self, rows: usize, cols: usize) {
        self.screen.resize(rows, cols);
    }

    /// Record the command line the user submitted (shell integration).
    ///
    /// Attaches to the latest block via [`BlockList::set_command`]; a
    /// no-op when no block exists yet.
    pub fn record_command(&mut self, command: &str) {
        self.blocks.set_command(command);
    }

    // ------------------------------------------------------------------
    // Agent side
    // ------------------------------------------------------------------

    /// The agent conversation attached to this session.
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// Mutable access to the conversation (seed a system prompt, clear
    /// history, ...).
    pub fn conversation_mut(&mut self) -> &mut Conversation {
        &mut self.conversation
    }

    /// Run one agent turn against this session's conversation.
    ///
    /// Pure orchestration: the host owns the provider, tools and config
    /// inside `agent`; this method supplies the session's conversation and
    /// a [`ToolContext`] whose `cwd` defaults to the session cwd (i.e. the
    /// last OSC 7 report — `None` when the shell never reported one) and
    /// whose `session_id` is this session's id.
    pub async fn run_agent_turn(
        &mut self,
        agent: &AgentLoop,
        input: UserInput,
        cancel: &CancellationToken,
        events_tx: tokio::sync::mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<TurnOutcome, AgentError> {
        let tool_ctx = ToolContext {
            cwd: self.cwd.clone(),
            session_id: Some(self.id.0.clone()),
        };
        agent
            .run_turn(&mut self.conversation, input, &tool_ctx, cancel, events_tx)
            .await
    }
}

/// Extract a filesystem path from an OSC 7 `file://` URL, leniently.
///
/// `file://host/path` → `/path` (the host component is ignored),
/// `file:///path` → `/path`, and a bare absolute path is accepted
/// verbatim. Percent-escapes are **not** decoded — the path is taken
/// byte-for-byte as the shell sent it. Returns `None` (leaving the cwd
/// unchanged) for anything else: relative junk, a `file://` URL with no
/// path component, or an empty payload.
fn parse_file_url_path(url: &str) -> Option<String> {
    if let Some(rest) = url.strip_prefix("file://") {
        // `rest` is `host/path`, `/path` (empty host), or junk without a
        // path separator.
        let slash = rest.find('/')?;
        return Some(rest[slash..].to_owned());
    }
    if url.starts_with('/') {
        return Some(url.to_owned());
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use bb_agent_loop::{AgentConfig, Provider, ProviderCallError};
    use bb_block_list::BlockLifecycle;
    use bb_providers::{ProviderEvent, ProviderRequest, StopReason};
    use bb_tool_bridge::testing::FnTool;
    use bb_tool_bridge::{ToolBridge, ToolDispatchResult};
    use futures::future::BoxFuture;
    use futures::FutureExt;
    use tokio::sync::mpsc::UnboundedSender;

    use super::*;

    fn session() -> Session {
        Session::new(SessionId("test".to_owned()), 24, 80)
    }

    #[test]
    fn session_id_generate_is_unique_and_hex() {
        let mut seen = HashSet::new();
        for _ in 0..10_000 {
            let id = SessionId::generate();
            assert_eq!(id.0.len(), 32);
            assert!(id.0.chars().all(|c| c.is_ascii_hexdigit()));
            assert!(seen.insert(id.0), "duplicate session id generated");
        }
    }

    #[test]
    fn session_id_display_is_inner_string() {
        assert_eq!(SessionId("abc".to_owned()).to_string(), "abc");
    }

    #[test]
    fn feed_pty_end_to_end_blocks_title_cwd_events() {
        let mut s = session();

        // Prompt on row 0.
        let ev = s.feed_pty(b"\x1b]133;A\x07$ ");
        assert_eq!(ev, vec![VtEvent::PromptStart]);
        s.feed_pty(b"\x1b]133;B\x07");
        s.record_command("echo hi");
        // Command echoed, then execution begins on row 1.
        s.feed_pty(b"echo hi\r\n\x1b]133;C\x07");
        // Title, cwd, output, a bell, and a hyperlink arrive mid-command.
        let ev = s.feed_pty(
            b"\x1b]0;my session\x07\x1b]7;file://myhost/Users/me/proj\x07\
              hi\r\n\x07\x1b]8;;https://example.com\x07link\x1b]8;;\x07more\r\n",
        );
        assert!(ev.contains(&VtEvent::Bell));
        assert!(ev.contains(&VtEvent::Hyperlink(Some("https://example.com".to_owned()))));
        assert!(ev.contains(&VtEvent::Hyperlink(None)));
        assert!(ev.contains(&VtEvent::Title("my session".to_owned())));
        // Finished on row 3, next prompt on row 3 closes the block.
        s.feed_pty(b"\x1b]133;D;0\x07");
        s.feed_pty(b"\x1b]133;A\x07$ ");

        assert_eq!(s.title(), Some("my session"));
        assert_eq!(s.cwd(), Some("/Users/me/proj"));

        assert_eq!(s.blocks().len(), 2);
        let first = s.blocks().first().unwrap();
        assert_eq!(first.lifecycle(), BlockLifecycle::Finished);
        assert_eq!(first.start_row(), 0);
        assert_eq!(first.end_row(), Some(3));
        assert_eq!(first.command(), Some("echo hi"));
        assert_eq!(first.cwd(), Some("/Users/me/proj"));
        assert_eq!(first.exit_code(), Some(0));
        assert_eq!(first.succeeded(), Some(true));
        let second = s.blocks().last().unwrap();
        assert_eq!(second.start_row(), 3);
        assert_eq!(second.end_row(), None);

        // The screen actually holds the printed output.
        assert_eq!(s.screen().row_text(0).as_deref(), Some("$ echo hi"));
        assert_eq!(s.screen().row_text(1).as_deref(), Some("hi"));
        assert_eq!(s.screen().row_text(2).as_deref(), Some("linkmore"));
    }

    #[test]
    fn feed_pty_empty_bytes_is_noop() {
        let mut s = session();
        assert!(s.feed_pty(b"").is_empty());
        // No implicit block was created by the trailing observe_row.
        assert!(s.blocks().is_empty());
        assert_eq!(s.title(), None);
        assert_eq!(s.cwd(), None);
    }

    #[test]
    fn osc7_junk_url_leaves_cwd_unchanged() {
        let mut s = session();
        s.feed_pty(b"\x1b]7;file://myhost/first\x07");
        assert_eq!(s.cwd(), Some("/first"));
        // Junk variants: relative garbage, no path component, empty.
        for junk in ["not-a-url", "file://hostonly", "http://x/y", ""] {
            let seq = format!("\x1b]7;{junk}\x07");
            s.feed_pty(seq.as_bytes());
            assert_eq!(s.cwd(), Some("/first"), "cwd changed by junk {junk:?}");
        }
        // Lenient acceptance: empty host and bare absolute path.
        s.feed_pty(b"\x1b]7;file:///second\x07");
        assert_eq!(s.cwd(), Some("/second"));
        s.feed_pty(b"\x1b]7;/third\x07");
        assert_eq!(s.cwd(), Some("/third"));
    }

    #[test]
    fn resize_preserves_blocks_and_conversation() {
        let mut s = session();
        s.feed_pty(b"\x1b]133;A\x07$ \x1b]133;B\x07");
        s.record_command("ls");
        s.conversation_mut()
            .push(ProviderMessage::User("hello".to_owned()));

        s.resize(30, 100);

        assert_eq!(s.screen().rows(), 30);
        assert_eq!(s.screen().cols(), 100);
        assert_eq!(s.blocks().len(), 1);
        assert_eq!(s.blocks().latest().unwrap().command(), Some("ls"));
        assert_eq!(
            s.conversation().messages(),
            &[ProviderMessage::User("hello".to_owned())]
        );
    }

    #[test]
    fn record_command_flows_into_latest_block() {
        let mut s = session();
        s.feed_pty(b"\x1b]133;A\x07");
        s.feed_pty(b"\x1b]133;B\x07");
        s.record_command("cargo test");
        assert_eq!(s.blocks().latest().unwrap().command(), Some("cargo test"));
    }

    #[test]
    fn set_identity_replaces_default() {
        let mut s = session();
        assert_eq!(s.identity(), &IdentityContext::Anonymous);
        s.set_identity(IdentityContext::Byok);
        assert_eq!(s.identity(), &IdentityContext::Byok);
    }

    #[test]
    fn parse_file_url_path_cases() {
        assert_eq!(
            parse_file_url_path("file://host/a/b").as_deref(),
            Some("/a/b")
        );
        assert_eq!(parse_file_url_path("file:///a").as_deref(), Some("/a"));
        // Percent-escapes are not decoded.
        assert_eq!(
            parse_file_url_path("file://h/a%20b").as_deref(),
            Some("/a%20b")
        );
        assert_eq!(parse_file_url_path("/bare").as_deref(), Some("/bare"));
        assert_eq!(parse_file_url_path("file://nohost-nopath"), None);
        assert_eq!(parse_file_url_path("junk"), None);
        assert_eq!(parse_file_url_path(""), None);
    }

    /// Scripted provider: first call requests the `pwd` tool, second call
    /// answers with plain text.
    struct MockProvider {
        calls: AtomicUsize,
    }

    impl Provider for MockProvider {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn call<'a>(
            &'a self,
            _request: &'a ProviderRequest,
            events_tx: UnboundedSender<ProviderEvent>,
        ) -> BoxFuture<'a, Result<(), ProviderCallError>> {
            let call_index = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if call_index == 0 {
                    let _ = events_tx.send(ProviderEvent::ToolCallStart {
                        index: 0,
                        id: "call_1".to_owned(),
                        name: "pwd".to_owned(),
                    });
                    let _ = events_tx.send(ProviderEvent::ToolCallEnd { index: 0 });
                    let _ = events_tx.send(ProviderEvent::Finished(StopReason::ToolUse));
                } else {
                    let _ = events_tx.send(ProviderEvent::TextDelta("done".to_owned()));
                    let _ = events_tx.send(ProviderEvent::Finished(StopReason::EndTurn));
                }
                Ok(())
            }
            .boxed()
        }
    }

    #[tokio::test]
    async fn run_agent_turn_defaults_tool_ctx_cwd_to_session_cwd() {
        let mut s = session();
        s.feed_pty(b"\x1b]7;file://host/work/dir\x07");
        assert_eq!(s.cwd(), Some("/work/dir"));

        let mut bridge = ToolBridge::new();
        bridge
            .register(Arc::new(FnTool::new(
                "pwd",
                "echoes the tool context cwd and session id",
                |_call, ctx| {
                    ToolDispatchResult::ok(format!(
                        "cwd={} session={}",
                        ctx.cwd.clone().unwrap_or_else(|| "<none>".to_owned()),
                        ctx.session_id.clone().unwrap_or_default(),
                    ))
                },
            )))
            .unwrap();

        let agent = AgentLoop::new(
            Arc::new(MockProvider {
                calls: AtomicUsize::new(0),
            }),
            Arc::new(bridge),
            AgentConfig {
                model: "mock-model".to_owned(),
                ..AgentConfig::default()
            },
        );

        let cancel = CancellationToken::new();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();
        let outcome = s
            .run_agent_turn(
                &agent,
                UserInput::Text("where am I?".to_owned()),
                &cancel,
                events_tx,
            )
            .await
            .unwrap();

        assert_eq!(
            outcome,
            TurnOutcome::Completed {
                stop_reason: StopReason::EndTurn,
                usage: bb_providers::TokenUsage::default(),
            }
        );

        // Conversation gained: user, assistant(tool call), tool results,
        // assistant("done").
        let messages = s.conversation().messages();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0], ProviderMessage::User("where am I?".to_owned()));
        match &messages[2] {
            ProviderMessage::ToolResults(results) => {
                assert_eq!(results.len(), 1);
                assert!(!results[0].is_error);
                assert_eq!(results[0].content, "cwd=/work/dir session=test");
            }
            other => panic!("expected tool results, got {other:?}"),
        }
        match &messages[3] {
            ProviderMessage::Assistant(a) => assert_eq!(a.text, "done"),
            other => panic!("expected assistant, got {other:?}"),
        }

        // Events were streamed.
        let mut saw_completed = false;
        while let Ok(event) = events_rx.try_recv() {
            if matches!(event, AgentEvent::TurnCompleted { .. }) {
                saw_completed = true;
            }
        }
        assert!(saw_completed);
    }
}
