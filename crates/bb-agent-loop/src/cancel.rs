//! Cheap-to-clone cancellation token.
//!
//! Implemented over [`tokio::sync::watch`] so we don't need a `tokio-util`
//! dependency: cancellation is a sticky `false → true` transition and
//! `send_replace` wakes every waiter.

use std::sync::Arc;

use tokio::sync::watch;

/// A cheap-to-clone cancellation token.
///
/// [`CancellationToken::cancel`] is *sticky*: once cancelled, a token stays
/// cancelled forever and every current and future waiter on
/// [`CancellationToken::cancelled`] resolves immediately. All clones share
/// the same state.
#[derive(Clone)]
pub struct CancellationToken {
    /// Shared flag. Holding the sender in every clone guarantees the
    /// channel never closes while a token is alive.
    tx: Arc<watch::Sender<bool>>,
}

impl CancellationToken {
    /// Create a new, un-cancelled token.
    pub fn new() -> Self {
        let (tx, _rx) = watch::channel(false);
        Self { tx: Arc::new(tx) }
    }

    /// Cancel the token, waking all waiters. Idempotent and sticky.
    pub fn cancel(&self) {
        let _previous = self.tx.send_replace(true);
    }

    /// True once [`CancellationToken::cancel`] has been called on any clone.
    pub fn is_cancelled(&self) -> bool {
        *self.tx.borrow()
    }

    /// Resolves when the token is cancelled (immediately if it already is).
    ///
    /// Usable directly inside `tokio::select!`.
    pub async fn cancelled(&self) {
        let mut rx = self.tx.subscribe();
        loop {
            let cancelled = *rx.borrow_and_update();
            if cancelled {
                return;
            }
            // `changed()` can only fail when every sender is dropped; we
            // hold one via `self.tx`, so this branch is unreachable while
            // `self` is borrowed. Pend forever rather than resolving as a
            // spurious cancellation.
            if rx.changed().await.is_err() {
                std::future::pending::<()>().await;
            }
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn starts_uncancelled_and_cancel_is_sticky() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        token.cancel(); // idempotent
        assert!(token.is_cancelled());
    }

    #[test]
    fn clones_share_state() {
        let token = CancellationToken::default();
        let clone = token.clone();
        clone.cancel();
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn cancelled_resolves_immediately_when_already_cancelled() {
        let token = CancellationToken::new();
        token.cancel();
        token.cancelled().await; // must not hang
    }

    #[tokio::test]
    async fn cancelled_wakes_pending_waiter() {
        let token = CancellationToken::new();
        let waiter = token.clone();
        let wait = async move { waiter.cancelled().await };
        let fire = async {
            tokio::task::yield_now().await;
            token.cancel();
        };
        tokio::join!(wait, fire);
    }
}
