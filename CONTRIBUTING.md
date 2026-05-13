# Contributing to beam-beacon-engine

This repository is currently **internal Ailuxis infrastructure**. We're not
soliciting outside contributions yet — the engine needs to stabilize and
both consumers (Beam, Beacon) need to be wired up before we can offer a
clean contribution path. If you're an external developer and have ideas
for the engine, please raise them on the upstream Beam project first.

For internal contributors:

## Principles

1. **No upstream-Warp code.** This repo is hand-coded by us. If you're
   tempted to copy-paste from `MichaelDanCurtis/WarpMod` (Beam),
   `MichaelDanCurtis/Beacon`, or `warpdotdev/warp`, **stop**. Rewrite the
   feature here cleanly. This keeps the dual MIT-OR-Apache-2.0 grant
   unambiguous and avoids any tension with upstream Warp's dual MIT/AGPL
   offer.
2. **Crate boundaries are deliberate.** The seven-crate split (see
   `README.md`) was chosen to let either consumer opt in to whatever
   subset they need. Don't merge crates without raising it in a PR
   discussion first.
3. **Public API stability.** Crates start at `0.0.0`. Bump to `0.1.x`
   when first consumed in earnest. Treat any `pub` item as a stability
   commitment from `0.1.0` onward — internal types should be `pub(crate)`.

## Development flow

```bash
# Quick sanity check across the workspace
cargo check --workspace

# Type-check + build
cargo build --workspace

# Run every test
cargo test --workspace

# Format + lint before committing
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

We use Rust edition 2021 and target `rust-version = "1.78"`. Don't bump
the MSRV without flagging it in the PR description.

## Commit style

- Subject line under 70 chars.
- Imperative mood ("add", "fix", "rewrite"), not past tense.
- Reference the affected crate prefix when relevant, e.g.
  `bb-agent-loop: handle Esc cancellation across all await points`.
- Body explains the *why* — the diff already shows the *what*.

## PR flow

- One PR per logical change. If you're extracting code from Beam or
  Beacon, do the extraction in a dedicated PR with a clear before/after
  in the description.
- Always include a `## Test plan` section in the PR body. CI runs
  `cargo check + clippy + test` on every push.
- Squash-merge on land. Keep the squash commit message tidy.

## When extracting from Beam or Beacon

Both consumers may have features that should ultimately live here. The
extraction pattern:

1. **Rewrite, don't copy.** Read the Beam / Beacon implementation as
   reference, design the engine version in this repo, write fresh code.
2. **Land the engine crate first.** Get it green and tested in
   isolation before any consumer depends on it.
3. **Wire the consumer next.** Beam or Beacon adds a git dependency on
   the engine crate, deletes the old in-tree implementation in a separate
   PR, and lands the cutover.
4. **Mirror the test coverage.** Tests live with the engine; the
   consumer keeps integration-level tests only.

## License of contributions

Unless you explicitly state otherwise, any contribution you intentionally
submit for inclusion in this work shall be dual-licensed as
`MIT OR Apache-2.0`, without any additional terms or conditions. See
[`NOTICE.md`](NOTICE.md).
