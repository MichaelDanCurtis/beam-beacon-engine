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
    pub backoff_multiplier: f64,
    /// Upper bound on any single backoff sleep.
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
    pub(crate) fn backoff_after(&self, failed_attempts: u32) -> Duration {
        let exponent = failed_attempts.saturating_sub(1).min(i32::MAX as u32) as i32;
        let secs = self.initial_backoff.as_secs_f64() * self.backoff_multiplier.powi(exponent);
        let capped = secs.min(self.max_backoff.as_secs_f64()).max(0.0);
        if capped.is_finite() {
            Duration::from_secs_f64(capped)
        } else {
            self.max_backoff
        }
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
}
