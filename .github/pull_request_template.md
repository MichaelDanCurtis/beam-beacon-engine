<!--
Thanks for contributing to beam-beacon-engine. Please fill the sections
below so review can be fast and the merge commit has the context the
consumers (Beam, Beacon) will need to track this change.
-->

## Summary

<!-- 1-3 sentences. What does this PR do and why. -->

## Affected crates

<!-- Check every crate this PR touches. Useful for downstream consumers. -->

- [ ] `bb-terminal-model`
- [ ] `bb-vt-parser`
- [ ] `bb-block-list`
- [ ] `bb-agent-loop`
- [ ] `bb-providers`
- [ ] `bb-tool-bridge`
- [ ] `bb-shared-session`
- [ ] Workspace / tooling only

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would break consumer code)
- [ ] Refactor / cleanup (no behavior change)
- [ ] Docs / metadata only
- [ ] **Extraction from Beam or Beacon** — see `docs/INTEGRATION.md` for
      the extraction recipe; tick the boxes there too.

## Test plan

<!--
Bulleted markdown checklist of how reviewers should verify this change.
Include cargo commands, hand-verification steps, and any consumer-side
verification needed.
-->

- [ ] `cargo check --workspace`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo fmt --all -- --check`
- [ ] `cargo test --workspace --all-targets`
- [ ] *(if extraction)* Beam smoke-tested against this branch via path-override
- [ ] *(if extraction)* Beacon smoke-tested against this branch via path-override

## Consumer-side impact

<!--
If this PR changes a public API or extracts behavior from Beam / Beacon,
note here:
  - which consumer needs the matching change
  - whether the change is breaking or back-compat
  - link to the consumer PR (if open)
-->

- Beam: <!-- N/A or link -->
- Beacon: <!-- N/A or link -->

## Notes for the reviewer

<!-- Anything else? Tradeoffs you considered. Things you're unsure about. -->
