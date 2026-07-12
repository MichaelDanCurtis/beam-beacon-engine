//! The host-provided persistent key/value store abstraction and the
//! in-memory implementation used for tests and ephemeral sessions.

use std::collections::BTreeMap;
use std::sync::Mutex;

use futures::future::BoxFuture;
use futures::FutureExt;

/// Error produced by a [`KvStore`] operation.
///
/// Deliberately opaque: hosts back the store with whatever they like
/// (sqlite in both Beam and Beacon today) and surface failures as a
/// human-readable string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("kv store error: {0}")]
pub struct KvError(pub String);

impl KvError {
    /// Build a [`KvError`] from anything displayable.
    pub fn new(detail: impl Into<String>) -> Self {
        Self(detail.into())
    }
}

/// Host-provided persistent KV store (sqlite in both consumers today).
///
/// Dyn-safe; async via [`BoxFuture`]. Keys are flat strings; namespacing
/// is by prefix convention (see
/// [`SessionHistory`](crate::SessionHistory) for the `session/{id}`
/// scheme). Values are opaque byte blobs.
pub trait KvStore: Send + Sync {
    /// Fetch the value stored under `key`, or `None` when absent.
    fn get<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<Vec<u8>>, KvError>>;

    /// Store `value` under `key`, replacing any existing value.
    fn put<'a>(&'a self, key: &'a str, value: Vec<u8>) -> BoxFuture<'a, Result<(), KvError>>;

    /// Remove the value stored under `key`. Deleting a missing key is not
    /// an error.
    fn delete<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<(), KvError>>;

    /// Every key starting with `prefix`, in ascending lexicographic order.
    fn list_keys<'a>(&'a self, prefix: &'a str) -> BoxFuture<'a, Result<Vec<String>, KvError>>;
}

/// In-memory [`KvStore`] for tests and ephemeral (never-persisted)
/// sessions.
///
/// Backed by a [`std::sync::Mutex`]`<BTreeMap>`; the lock is never held
/// across an await point (every operation completes synchronously inside
/// its future). A poisoned lock is recovered from rather than propagated —
/// the map itself cannot be left in a torn state by any operation here.
#[derive(Debug, Default)]
pub struct MemoryKvStore {
    entries: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl MemoryKvStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Lock the map, recovering from poisoning (see type-level docs).
    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, Vec<u8>>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl KvStore for MemoryKvStore {
    fn get<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<Vec<u8>>, KvError>> {
        async move { Ok(self.lock().get(key).cloned()) }.boxed()
    }

    fn put<'a>(&'a self, key: &'a str, value: Vec<u8>) -> BoxFuture<'a, Result<(), KvError>> {
        async move {
            self.lock().insert(key.to_owned(), value);
            Ok(())
        }
        .boxed()
    }

    fn delete<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<(), KvError>> {
        async move {
            self.lock().remove(key);
            Ok(())
        }
        .boxed()
    }

    fn list_keys<'a>(&'a self, prefix: &'a str) -> BoxFuture<'a, Result<Vec<String>, KvError>> {
        async move {
            Ok(self
                .lock()
                .range(prefix.to_owned()..)
                .take_while(|(k, _)| k.starts_with(prefix))
                .map(|(k, _)| k.clone())
                .collect())
        }
        .boxed()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn memory_kv_round_trip() {
        let store = MemoryKvStore::new();
        assert_eq!(store.get("a").await.unwrap(), None);

        store.put("a", b"one".to_vec()).await.unwrap();
        assert_eq!(store.get("a").await.unwrap(), Some(b"one".to_vec()));

        // Overwrite replaces.
        store.put("a", b"two".to_vec()).await.unwrap();
        assert_eq!(store.get("a").await.unwrap(), Some(b"two".to_vec()));

        store.delete("a").await.unwrap();
        assert_eq!(store.get("a").await.unwrap(), None);

        // Deleting a missing key is fine.
        store.delete("a").await.unwrap();
    }

    #[tokio::test]
    async fn list_keys_filters_by_prefix_and_sorts() {
        let store = MemoryKvStore::new();
        for key in ["session/b", "session/a", "workspace/x", "sessio"] {
            store.put(key, vec![1]).await.unwrap();
        }
        assert_eq!(
            store.list_keys("session/").await.unwrap(),
            vec!["session/a".to_owned(), "session/b".to_owned()]
        );
        assert_eq!(store.list_keys("").await.unwrap().len(), 4);
        assert!(store.list_keys("nope/").await.unwrap().is_empty());
    }
}
