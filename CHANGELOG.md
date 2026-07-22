# Changelog

All notable changes to `beam-beacon-engine` will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
starting at `0.1.0`. The current `0.0.0` is a pre-release scaffold; breaking
changes can land any time without a version bump.

## [Unreleased]

### Added

- **Full implementations of all seven engine crates** (freshly written,
  no code vendored from any consumer or upstream project):
  - `bb-terminal-model` — `Screen` aggregate (pen, scroll regions, tab
    stops, origin/insert/autowrap modes, xterm-style pending wrap,
    DECSC/DECRC, alternate screen, RIS), `Scrollback` ring with eviction
    tracking and forever-stable absolute row ids, wide-character cells
    (lead/trail pairs) via `unicode-width`, optional `serde` feature.
  - `bb-vt-parser` — VT500-style state machine (ground/escape/CSI/OSC/DCS)
    with incremental strict UTF-8 decoding that survives arbitrary chunk
    splits; C0 controls (DEL discarded), full CSI/SGR support including
    256-color and truecolor (semicolon and colon forms), DEC private
    modes including alt screen (?1049), OSC 0/2/7/8/133 surfaced as
    semantic `VtEvent`s.
  - `bb-block-list` — Warp-style command blocks driven by OSC 133
    `PromptMarker`s: lifecycle (Prompt → Composing → Running → Finished),
    metadata + full-text search, next/prev navigation, `block_at_row`,
    eviction-aware pruning, implicit-block heuristic when no shell
    integration is present.
  - `bb-providers` — provider-neutral `ProviderMessage` / `ProviderEvent`
    model and transport-agnostic wire adapters for Anthropic Messages,
    OpenAI-compatible Chat Completions, Gemini, Ollama (NDJSON), and
    local-OpenAI; chunk-boundary-safe SSE/NDJSON framers.
  - `bb-tool-bridge` — dyn-safe `ToolHandler` trait + `ToolBridge`
    registry, uniform `ToolDispatchResult`, well-known tool spec
    constructors, `FnTool` test helper.
  - `bb-agent-loop` — `AgentLoop::run_turn` orchestration (streamed
    provider rounds → tool dispatch → next round), sticky
    `CancellationToken` honored at every await point, exponential-backoff
    retry for transient errors, context-window trimming with a
    `Summarizer` hook that never orphans tool results.
  - `bb-shared-session` — `Session` aggregate wiring parser → screen →
    blocks and hosting agent turns; `KvStore` trait + `MemoryKvStore`;
    versioned `SessionSnapshot` persistence via `SessionHistory`;
    `IdentityContext` (Anonymous / SignedIn / Byok / ArcAttached).
- Host-provided telemetry: `MetricsSink` / `MetricEvent` in
  `bb-agent-loop`, wired via the additive `AgentLoop::with_metrics`
  (resolves open decision #4 in `docs/INTEGRATION.md` before consumers
  pin the API).
- End-to-end golden transcript tests driving realistic shell sessions
  through the whole seam (PTY bytes → parser → screen → blocks →
  queryable output).
- Workspace dependency table (serde, serde_json, thiserror, futures,
  tokio, unicode-width) and intra-workspace path entries.

### Fixed

Found by the end-to-end seam tests, which is precisely what they exist
for — every one of these passed the per-crate unit suites:

- **Command blocks depended on PTY read size.** `Parser::feed` returned
  events only after consuming a chunk, so markers were all stamped with
  a single post-feed cursor row. Fed in small chunks blocks were correct;
  fed in one read — what a real PTY does — every block collapsed to a
  zero-height range with empty text. `Parser::feed` now returns
  `PositionedEvent`s stamped at emission, and block layout is identical
  for every chunking from 1 byte to 64 KiB.
- **`bb-agent-loop` did not build as a library.** `tokio::select!`
  needed the `macros` feature, present only under `[dev-dependencies]`;
  `--all-targets` builds unified it in, so all of CI was green while the
  crate was unbuildable for a consumer.
- **The alternate screen shadowed the scrollback timeline**, so
  `block_text` returned a TUI's frame buffer while vim/htop was running.
- **`resize` destroyed on-screen content**, silently blanking the
  in-flight command's output on any window drag. It now preserves
  content (no reflow) and keeps `cursor_absolute_row` invariant.
- **A phantom empty block** was created by any first read that didn't
  complete an OSC 133 sequence — including bash's own bracketed-paste
  enable.
- **Block pruning was unreachable** through `Session`, so the block list
  grew unboundedly. `Session::prune_blocks` and
  `Session::with_scrollback` added.
- Initial scaffold: workspace, dual-license (MIT OR Apache-2.0), NOTICE
  with the no-upstream-Warp-code principle.
- Architecture and integration docs under `docs/`.
- GitHub Actions CI (build + test + clippy + fmt) for `main` and PRs.

### Notes

The engine-side implementation work (design + implementation for every
crate) is complete: 365 tests, 96% workspace line coverage, and eight CI
gates — libs-only build, `--all-targets` tests, doctests, coverage floor,
clippy `-D warnings`, fmt, MSRV 1.78, and `cargo deny`. The remaining
migration steps — Beam and Beacon adoption PRs, in-tree cleanup, and the
`0.1.0` tag — happen in the consumer repos; see
[`docs/INTEGRATION.md`](docs/INTEGRATION.md) and
[`docs/MIGRATION-CHECKLIST.md`](docs/MIGRATION-CHECKLIST.md).
