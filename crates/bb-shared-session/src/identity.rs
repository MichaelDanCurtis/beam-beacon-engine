//! Identity / auth context surfaced from the host.

use serde::{Deserialize, Serialize};

/// Who (if anyone) this session is running as.
///
/// The engine never authenticates anybody itself — the host (Beam or
/// Beacon) resolves identity and hands the result to the
/// [`Session`](crate::Session), where it rides along in persisted
/// snapshots and is available to host-side features (telemetry scoping,
/// entitlement checks, provider key selection).
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityContext {
    /// No identity: not signed in, no user-supplied keys.
    #[default]
    Anonymous,
    /// Signed in to a hosted account.
    SignedIn {
        /// Host-defined stable user id.
        user_id: String,
        /// The account email, when the host shares it.
        email: Option<String>,
    },
    /// Bring-your-own-key: provider keys supplied by the user; no account.
    Byok,
    /// Attached to the Ailuxis ARC service tier.
    ArcAttached {
        /// Host-defined stable user id.
        user_id: String,
        /// ARC tenant identifier, for multi-tenant deployments.
        tenant: Option<String>,
    },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_is_anonymous() {
        assert_eq!(IdentityContext::default(), IdentityContext::Anonymous);
    }

    #[test]
    fn serde_round_trip() {
        let identities = [
            IdentityContext::Anonymous,
            IdentityContext::SignedIn {
                user_id: "u_1".to_owned(),
                email: Some("a@b.c".to_owned()),
            },
            IdentityContext::Byok,
            IdentityContext::ArcAttached {
                user_id: "u_2".to_owned(),
                tenant: None,
            },
        ];
        for identity in identities {
            let json = serde_json::to_string(&identity).unwrap();
            let back: IdentityContext = serde_json::from_str(&json).unwrap();
            assert_eq!(back, identity);
        }
    }
}
