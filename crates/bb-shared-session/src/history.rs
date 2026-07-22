//! Persistent session history over a host-provided [`KvStore`].

use bb_providers::ProviderMessage;
use serde::{Deserialize, Serialize};

use crate::kv::{KvError, KvStore};
use crate::session::{Session, SessionId};
use crate::IdentityContext;

/// Current on-disk snapshot format version. Bumped on breaking changes to
/// [`SessionSnapshot`]; readers refuse anything newer (see
/// [`HistoryError::UnsupportedVersion`]).
pub const SNAPSHOT_VERSION: u32 = 1;

/// Errors from [`SessionHistory`] operations.
#[derive(Debug, thiserror::Error)]
pub enum HistoryError {
    /// The underlying [`KvStore`] failed.
    #[error(transparent)]
    Store(#[from] KvError),
    /// A snapshot could not be (de)serialized.
    #[error("snapshot serialization error: {0}")]
    Serde(String),
    /// The stored snapshot was written by a newer engine version.
    #[error("unsupported snapshot version {0} (this engine supports up to {SNAPSHOT_VERSION})")]
    UnsupportedVersion(u32),
}

/// Summary row for session pickers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSummary {
    /// The session's id.
    pub id: SessionId,
    /// Its title, when the shell reported one.
    pub title: Option<String>,
    /// Its working directory, when the shell reported one.
    pub cwd: Option<String>,
    /// Number of messages in the persisted conversation.
    pub message_count: usize,
}

/// Persisted snapshot of a session's durable state: conversation +
/// metadata + identity.
///
/// The live [`Screen`](bb_terminal_model::Screen) and
/// [`BlockList`](bb_block_list::BlockList) are deliberately **not**
/// persisted — they are views over the PTY byte stream and are rebuilt
/// from PTY output when the host reattaches the session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// Snapshot format version; always [`SNAPSHOT_VERSION`] when written
    /// by this engine.
    pub version: u32,
    /// The session's id.
    pub id: SessionId,
    /// Its title, when the shell reported one.
    pub title: Option<String>,
    /// Its working directory, when the shell reported one.
    pub cwd: Option<String>,
    /// The identity context the session ran under.
    pub identity: IdentityContext,
    /// The full conversation transcript, oldest first.
    pub messages: Vec<ProviderMessage>,
    /// The rolling context summary, when the agent loop has trimmed history
    /// at least once.
    ///
    /// This is stored separately because it is not part of `messages` — the
    /// agent loop keeps exactly one summary slot beside the transcript. It
    /// carries everything that was evicted, so dropping it would make a
    /// restored session forget its own past.
    ///
    /// `#[serde(default)]` keeps version-1 blobs written before this field
    /// existed loadable; they simply restore with no summary.
    #[serde(default)]
    pub summary: Option<String>,
}

/// Persistent session history behind a host-provided [`KvStore`].
///
/// # Key scheme
///
/// Each session is stored as JSON under the key `session/{id}` (the id's
/// inner string, verbatim). [`SessionHistory::list`] enumerates the
/// `session/` prefix.
pub struct SessionHistory<S: KvStore> {
    store: S,
}

impl<S: KvStore> std::fmt::Debug for SessionHistory<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionHistory").finish_non_exhaustive()
    }
}

impl<S: KvStore> SessionHistory<S> {
    /// Wrap a host-provided store.
    pub fn new(store: S) -> Self {
        Self { store }
    }

    /// The storage key for `id` (see the type-level key scheme docs).
    fn key(id: &SessionId) -> String {
        format!("session/{}", id.0)
    }

    /// Persist `session`'s durable state (conversation, metadata,
    /// identity) under `session/{id}`, replacing any previous snapshot.
    pub async fn save(&self, session: &Session) -> Result<(), HistoryError> {
        let snapshot = SessionSnapshot {
            version: SNAPSHOT_VERSION,
            id: session.id().clone(),
            title: session.title().map(str::to_owned),
            cwd: session.cwd().map(str::to_owned),
            identity: session.identity().clone(),
            messages: session.conversation().messages().to_vec(),
            summary: session.conversation().summary().map(str::to_owned),
        };
        let bytes =
            serde_json::to_vec(&snapshot).map_err(|e| HistoryError::Serde(e.to_string()))?;
        self.store.put(&Self::key(&snapshot.id), bytes).await?;
        Ok(())
    }

    /// Load the snapshot stored for `id`, or `None` when there is none.
    ///
    /// # Errors
    ///
    /// [`HistoryError::UnsupportedVersion`] when the stored snapshot's
    /// `version` is greater than [`SNAPSHOT_VERSION`];
    /// [`HistoryError::Serde`] when the blob is not a valid snapshot.
    pub async fn load(&self, id: &SessionId) -> Result<Option<SessionSnapshot>, HistoryError> {
        match self.store.get(&Self::key(id)).await? {
            None => Ok(None),
            Some(bytes) => parse_snapshot(&bytes).map(Some),
        }
    }

    /// Rebuild a [`Session`] from a snapshot. `rows`/`cols` are supplied
    /// by the host (the current PTY size); the screen and block list start
    /// empty and are rebuilt from PTY output on reattach.
    pub fn restore(snapshot: SessionSnapshot, rows: usize, cols: usize) -> Session {
        Session::from_parts(
            snapshot.id,
            snapshot.title,
            snapshot.cwd,
            snapshot.identity,
            snapshot.messages,
            snapshot.summary,
            rows,
            cols,
        )
    }

    /// Delete the snapshot stored for `id`; a no-op when there is none.
    pub async fn delete(&self, id: &SessionId) -> Result<(), HistoryError> {
        self.store.delete(&Self::key(id)).await?;
        Ok(())
    }

    /// Summaries for every stored session, in key (id) order.
    ///
    /// Lenient by design: entries that fail to parse or were written by a
    /// newer engine are skipped rather than failing the whole listing, so
    /// one bad row never breaks the session picker.
    pub async fn list(&self) -> Result<Vec<SessionSummary>, HistoryError> {
        let keys = self.store.list_keys("session/").await?;
        let mut summaries = Vec::with_capacity(keys.len());
        for key in keys {
            let Some(bytes) = self.store.get(&key).await? else {
                continue; // Deleted between list and get.
            };
            let Ok(snapshot) = parse_snapshot(&bytes) else {
                continue; // Unparseable or newer-version row: skip.
            };
            summaries.push(SessionSummary {
                id: snapshot.id,
                title: snapshot.title,
                cwd: snapshot.cwd,
                message_count: snapshot.messages.len(),
            });
        }
        Ok(summaries)
    }
}

/// Decode one stored snapshot, gating on the version field *before*
/// attempting the full deserialization (a newer snapshot may not even
/// have the same field layout).
fn parse_snapshot(bytes: &[u8]) -> Result<SessionSnapshot, HistoryError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| HistoryError::Serde(e.to_string()))?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| HistoryError::Serde("snapshot has no numeric `version` field".to_owned()))?;
    if version > u64::from(SNAPSHOT_VERSION) {
        return Err(HistoryError::UnsupportedVersion(
            u32::try_from(version).unwrap_or(u32::MAX),
        ));
    }
    serde_json::from_value(value).map_err(|e| HistoryError::Serde(e.to_string()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use crate::MemoryKvStore;

    use super::*;

    fn populated_session(id: &str) -> Session {
        let mut session = Session::new(SessionId(id.to_owned()), 24, 80);
        session.feed_pty(b"\x1b]0;my title\x07\x1b]7;file://h/work\x07some output\r\n");
        session.set_identity(IdentityContext::SignedIn {
            user_id: "u_9".to_owned(),
            email: None,
        });
        session
            .conversation_mut()
            .push(ProviderMessage::System("be terse".to_owned()));
        session
            .conversation_mut()
            .push(ProviderMessage::User("hi".to_owned()));
        session
    }

    #[tokio::test]
    async fn round_trip_preserves_the_rolling_context_summary() {
        // The agent loop keeps exactly one summary slot BESIDE the message
        // list, so it is not covered by snapshotting `messages`. If it were
        // dropped here, reopening a long session would silently lose
        // everything that had ever been trimmed out of the transcript.
        let history = SessionHistory::new(MemoryKvStore::new());
        let mut session = populated_session("s_summary");
        session
            .conversation_mut()
            .set_summary("earlier: user set up a Rust project");
        history.save(&session).await.unwrap();

        let snapshot = history
            .load(&SessionId("s_summary".to_owned()))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            snapshot.summary.as_deref(),
            Some("earlier: user set up a Rust project")
        );

        let restored = SessionHistory::<MemoryKvStore>::restore(snapshot, 24, 80);
        assert_eq!(
            restored.conversation().summary(),
            Some("earlier: user set up a Rust project")
        );
    }

    #[tokio::test]
    async fn snapshot_without_a_summary_field_still_loads() {
        // Version-1 blobs were written before `summary` existed; they must
        // keep loading rather than failing to deserialize.
        let store = MemoryKvStore::new();
        let blob = serde_json::json!({
            "version": 1,
            "id": "s_old",
            "title": null,
            "cwd": null,
            "identity": "Anonymous",
            "messages": []
        });
        store
            .put("session/s_old", serde_json::to_vec(&blob).unwrap())
            .await
            .unwrap();

        let history = SessionHistory::new(store);
        let snapshot = history
            .load(&SessionId("s_old".to_owned()))
            .await
            .unwrap()
            .expect("v1 snapshot should load");
        assert_eq!(snapshot.summary, None);
    }

    #[tokio::test]
    async fn save_load_restore_round_trip() {
        let history = SessionHistory::new(MemoryKvStore::new());
        let session = populated_session("s1");
        history.save(&session).await.unwrap();

        let snapshot = history
            .load(&SessionId("s1".to_owned()))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.version, SNAPSHOT_VERSION);
        assert_eq!(snapshot.id, SessionId("s1".to_owned()));
        assert_eq!(snapshot.title.as_deref(), Some("my title"));
        assert_eq!(snapshot.cwd.as_deref(), Some("/work"));
        assert_eq!(snapshot.messages.len(), 2);

        let restored = SessionHistory::<MemoryKvStore>::restore(snapshot, 30, 100);
        assert_eq!(restored.id(), &SessionId("s1".to_owned()));
        assert_eq!(restored.title(), Some("my title"));
        assert_eq!(restored.cwd(), Some("/work"));
        assert_eq!(
            restored.identity(),
            &IdentityContext::SignedIn {
                user_id: "u_9".to_owned(),
                email: None,
            }
        );
        assert_eq!(
            restored.conversation().messages(),
            session.conversation().messages()
        );
        // The screen and block list are rebuilt fresh at the host's size,
        // not restored: the original's PTY output is gone.
        assert_eq!(restored.screen().rows(), 30);
        assert_eq!(restored.screen().cols(), 100);
        assert_eq!(restored.screen().row_text(0).as_deref(), Some(""));
        assert!(restored.blocks().is_empty());
    }

    #[tokio::test]
    async fn snapshot_json_excludes_screen_state() {
        let store = MemoryKvStore::new();
        let history = SessionHistory::new(store);
        history.save(&populated_session("s1")).await.unwrap();

        let bytes = history
            .store
            .get("session/s1")
            .await
            .unwrap()
            .expect("snapshot stored under the documented key scheme");
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        // Pins the persisted schema: adding a key here must be a deliberate
        // decision, and screen state must never become one.
        let mut expected = vec![
            "cwd", "id", "identity", "messages", "summary", "title", "version",
        ];
        expected.sort_unstable();
        let mut actual = keys;
        actual.sort_unstable();
        assert_eq!(actual, expected);
        // The screen contents (the printed PTY output) are not persisted.
        assert!(!String::from_utf8(bytes).unwrap().contains("some output"));
    }

    #[tokio::test]
    async fn load_missing_returns_none() {
        let history = SessionHistory::new(MemoryKvStore::new());
        assert!(history
            .load(&SessionId("nope".to_owned()))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn list_and_delete() {
        let history = SessionHistory::new(MemoryKvStore::new());
        history.save(&populated_session("a")).await.unwrap();
        history.save(&populated_session("b")).await.unwrap();

        let summaries = history.list().await.unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].id, SessionId("a".to_owned()));
        assert_eq!(summaries[0].title.as_deref(), Some("my title"));
        assert_eq!(summaries[0].cwd.as_deref(), Some("/work"));
        assert_eq!(summaries[0].message_count, 2);
        assert_eq!(summaries[1].id, SessionId("b".to_owned()));

        history.delete(&SessionId("a".to_owned())).await.unwrap();
        assert!(history
            .load(&SessionId("a".to_owned()))
            .await
            .unwrap()
            .is_none());
        assert_eq!(history.list().await.unwrap().len(), 1);

        // Deleting a missing session is a no-op.
        history.delete(&SessionId("a".to_owned())).await.unwrap();
    }

    #[tokio::test]
    async fn newer_version_snapshot_is_rejected_on_load_and_skipped_on_list() {
        let store = MemoryKvStore::new();
        // Hand-crafted v2 blob with a field layout v1 doesn't know.
        let v2 = br#"{"version":2,"id":"future","payload":{"totally":"different"}}"#;
        {
            use crate::kv::KvStore as _;
            store.put("session/future", v2.to_vec()).await.unwrap();
        }
        let history = SessionHistory::new(store);

        let err = history
            .load(&SessionId("future".to_owned()))
            .await
            .unwrap_err();
        assert!(matches!(err, HistoryError::UnsupportedVersion(2)));

        // A good row next to it still lists; the v2 row is skipped.
        history.save(&populated_session("ok")).await.unwrap();
        let summaries = history.list().await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, SessionId("ok".to_owned()));
    }

    #[tokio::test]
    async fn garbage_blob_is_serde_error() {
        let store = MemoryKvStore::new();
        {
            use crate::kv::KvStore as _;
            store
                .put("session/bad", b"not json".to_vec())
                .await
                .unwrap();
        }
        let history = SessionHistory::new(store);
        let err = history
            .load(&SessionId("bad".to_owned()))
            .await
            .unwrap_err();
        assert!(matches!(err, HistoryError::Serde(_)));
    }
}
