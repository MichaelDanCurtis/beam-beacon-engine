//! Configuration types: retry/backoff, context-window policy, and the
//! top-level [`AgentConfig`].

use std::time::Duration;

/// Retry/backoff policy applied to
/// [`ProviderCallError::Transient`](crate::ProviderCallError::Transient)
/// provider errors.
///
/// [`ProviderCallError::Permanent`](crate::ProviderCallError::Permanent)
/// errors are never retried.
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    /// Total number of call attempts per provider round (initial call
    /// included), so `3` means one call plus up to two retries. Values
    /// below `1` are treated as `1`.
    pub max_attempts: u32,
    /// Backoff slept before the first retry.
    pub initial_backoff: Duration,
    /// Multiplier applied to the backoff after each failed attempt.
    ///
    /// Clamped to `>= 0.0` when applied: a negative multiplier would make
    /// the backoff sequence alternate sign (and, after the `0.0` floor,
    /// non-monotonic — `[500ms, 0s, 2s]`), which is never what a caller
    /// means. `NaN` is treated as `0.0`.
    pub backoff_multiplier: f64,
    /// Upper bound on any single backoff sleep.
    ///
    /// [`Duration::MAX`] is accepted as the spelling of "no cap".
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    /// 3 attempts, 500ms initial backoff, 2.0 multiplier, 10s cap.
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(500),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(10),
        }
    }
}

impl RetryPolicy {
    /// Backoff to sleep after `failed_attempts` attempts have failed
    /// (1-based: after the first failure this is `initial_backoff`).
    ///
    /// # Total on every input
    ///
    /// This runs inside the turn task of a crate that denies
    /// `unwrap`/`expect`, so it must not panic for *any* field values a
    /// host can construct — `RetryPolicy` is a plain public struct with
    /// public fields and no validating constructor. The pathological cases
    /// and what they yield:
    ///
    /// - **Negative or `NaN` `backoff_multiplier`** — clamped to `0.0`, so
    ///   the sequence stays monotonic instead of alternating sign.
    /// - **Overflow to infinity** (huge `initial_backoff`, huge
    ///   multiplier, large `failed_attempts`) — the `min` against
    ///   `max_backoff` catches it, because `f64::min` returns the finite
    ///   operand.
    /// - **`max_backoff: Duration::MAX`** — the natural spelling of "no
    ///   cap". Its `as_secs_f64()` rounds *up* to exactly `2^64`, one ULP
    ///   above `u64::MAX`, which [`Duration::from_secs_f64`] panics on
    ///   while still being `is_finite()`. Hence
    ///   [`Duration::try_from_secs_f64`] plus a fallback to `max_backoff`
    ///   itself: out of range can only mean "at or above the cap", and the
    ///   cap is by definition representable.
    pub(crate) fn backoff_after(&self, failed_attempts: u32) -> Duration {
        let exponent = failed_attempts.saturating_sub(1).min(i32::MAX as u32) as i32;
        // `f64::max` returns the non-NaN operand, so this folds NaN to 0.0
        // as well as clamping negatives.
        let multiplier = self.backoff_multiplier.max(0.0);
        let secs = self.initial_backoff.as_secs_f64() * multiplier.powi(exponent);
        let capped = secs.min(self.max_backoff.as_secs_f64()).max(0.0);
        Duration::try_from_secs_f64(capped).unwrap_or(self.max_backoff)
    }
}

/// Context-window management config.
///
/// Token counts are approximate: the loop uses a `chars / 4` heuristic over
/// all textual message content (see
/// [`Conversation::approx_tokens`](crate::Conversation::approx_tokens)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPolicy {
    /// Approximate token budget for the conversation. When exceeded, the
    /// oldest non-`System` messages are evicted before the next provider
    /// call.
    pub max_context_tokens: usize,
    /// When trimming, at least this many most-recent messages are kept
    /// regardless of budget.
    pub keep_recent_messages: usize,
}

impl Default for ContextPolicy {
    /// 150 000 approximate tokens, keeping the 8 most recent messages.
    fn default() -> Self {
        Self {
            max_context_tokens: 150_000,
            keep_recent_messages: 8,
        }
    }
}

/// Configuration for an [`AgentLoop`](crate::AgentLoop).
#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfig {
    /// Model identifier passed through to the provider. The `Default`
    /// value is an **empty string**: hosts must set a real model id before
    /// running turns against a live provider.
    pub model: String,
    /// Maximum tokens to generate per provider call, when capped.
    pub max_tokens: Option<u32>,
    /// Sampling temperature override, when set.
    pub temperature: Option<f32>,
    /// Retry/backoff policy for transient provider errors.
    pub retry: RetryPolicy,
    /// Context-window trimming policy.
    pub context: ContextPolicy,
    /// Safety valve: maximum provider rounds per turn. When a turn is still
    /// requesting tools after this many rounds it fails with
    /// [`AgentError::RoundLimitExceeded`](crate::AgentError::RoundLimitExceeded).
    pub max_rounds: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_tokens: None,
            temperature: None,
            retry: RetryPolicy::default(),
            context: ContextPolicy::default(),
            max_rounds: 25,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_values() {
        let retry = RetryPolicy::default();
        assert_eq!(retry.max_attempts, 3);
        assert_eq!(retry.initial_backoff, Duration::from_millis(500));
        assert!((retry.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(retry.max_backoff, Duration::from_secs(10));

        let context = ContextPolicy::default();
        assert_eq!(context.max_context_tokens, 150_000);
        assert_eq!(context.keep_recent_messages, 8);

        let config = AgentConfig::default();
        assert_eq!(config.model, "");
        assert_eq!(config.max_rounds, 25);
        assert_eq!(config.max_tokens, None);
        assert_eq!(config.temperature, None);
    }

    #[test]
    fn backoff_grows_exponentially_and_caps() {
        let retry = RetryPolicy::default();
        assert_eq!(retry.backoff_after(1), Duration::from_millis(500));
        assert_eq!(retry.backoff_after(2), Duration::from_millis(1000));
        assert_eq!(retry.backoff_after(3), Duration::from_millis(2000));
        // 500ms * 2^9 = 256s, capped at 10s.
        assert_eq!(retry.backoff_after(10), Duration::from_secs(10));
    }

    /// Every backoff of a policy, for the attempt numbers a turn can reach
    /// plus the extremes.
    fn backoffs(retry: &RetryPolicy) -> Vec<Duration> {
        [0, 1, 2, 3, 10, 64, 1_000, u32::MAX]
            .into_iter()
            .map(|n| retry.backoff_after(n))
            .collect()
    }

    #[test]
    fn uncapped_max_backoff_does_not_panic() {
        // `Duration::MAX.as_secs_f64()` is exactly 2^64 — finite, but one
        // ULP above what a `Duration` can hold.
        let retry = RetryPolicy {
            max_backoff: Duration::MAX,
            ..RetryPolicy::default()
        };
        assert_eq!(backoffs(&retry)[0], Duration::from_millis(500));
        assert_eq!(retry.backoff_after(1), Duration::from_millis(500));
        // Anything that overshoots the (unrepresentable) cap lands on it.
        assert_eq!(retry.backoff_after(u32::MAX), Duration::MAX);
    }

    #[test]
    fn uncapped_max_backoff_with_large_initial_does_not_panic() {
        let retry = RetryPolicy {
            initial_backoff: Duration::from_secs(3_600),
            max_backoff: Duration::MAX,
            ..RetryPolicy::default()
        };
        assert_eq!(retry.backoff_after(1), Duration::from_secs(3_600));
        assert_eq!(retry.backoff_after(u32::MAX), Duration::MAX);
    }

    #[test]
    fn max_initial_backoff_does_not_panic() {
        let retry = RetryPolicy {
            initial_backoff: Duration::MAX,
            max_backoff: Duration::MAX,
            ..RetryPolicy::default()
        };
        // `Duration::MAX.as_secs_f64()` is unrepresentable on the way back
        // in, so the cap answers for it.
        assert_eq!(retry.backoff_after(1), Duration::MAX);
        assert_eq!(retry.backoff_after(u32::MAX), Duration::MAX);
    }

    #[test]
    fn absurd_multiplier_overflows_to_the_cap_not_a_panic() {
        let retry = RetryPolicy {
            backoff_multiplier: 1e300,
            ..RetryPolicy::default()
        };
        assert_eq!(retry.backoff_after(1), Duration::from_millis(500));
        // 500ms * 1e300 overflows f64 on the second step; the cap holds.
        assert_eq!(retry.backoff_after(2), Duration::from_secs(10));
        assert_eq!(retry.backoff_after(u32::MAX), Duration::from_secs(10));

        let uncapped = RetryPolicy {
            backoff_multiplier: 1e300,
            max_backoff: Duration::MAX,
            ..RetryPolicy::default()
        };
        assert_eq!(uncapped.backoff_after(3), Duration::MAX);
    }

    #[test]
    fn negative_multiplier_is_clamped_and_monotonic() {
        let retry = RetryPolicy {
            backoff_multiplier: -2.0,
            ..RetryPolicy::default()
        };
        // Clamped to 0.0: [initial, 0, 0, ...] — never the old
        // [500ms, 0s, 2s] zig-zag.
        assert_eq!(retry.backoff_after(1), Duration::from_millis(500));
        assert_eq!(retry.backoff_after(2), Duration::ZERO);
        assert_eq!(retry.backoff_after(3), Duration::ZERO);
        let all = backoffs(&retry);
        assert!(
            all.windows(2)
                .all(|w| w[0] <= w[1] || w[1] == Duration::ZERO),
            "backoffs must not zig-zag: {all:?}"
        );
    }

    #[test]
    fn nan_and_infinite_fields_do_not_panic() {
        for multiplier in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, 0.0] {
            let retry = RetryPolicy {
                backoff_multiplier: multiplier,
                ..RetryPolicy::default()
            };
            // The assertion under test is simply "returns without
            // panicking", for every attempt number.
            let all = backoffs(&retry);
            assert!(
                all.iter().all(|d| *d <= retry.max_backoff),
                "multiplier {multiplier} produced {all:?} above the cap"
            );
        }
    }

    #[test]
    fn zero_and_max_attempt_numbers_are_in_range() {
        let retry = RetryPolicy::default();
        // `failed_attempts = 0` saturates to exponent 0.
        assert_eq!(retry.backoff_after(0), Duration::from_millis(500));
        assert_eq!(retry.backoff_after(u32::MAX), Duration::from_secs(10));
    }
}
