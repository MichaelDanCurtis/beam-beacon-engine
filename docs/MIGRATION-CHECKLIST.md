# Migration Checklist

A concrete, ordered set of tasks for filling in the empty engine crates.
This is `docs/INTEGRATION.md`'s plan made actionable. Tick boxes as you
go; each box is roughly one PR-sized chunk.

## Phase 0 — Repo bootstrap (DONE)

- [x] Workspace + dual license (MIT OR Apache-2.0)
- [x] 7 stub crates compile, clippy-clean, fmt-clean, test-clean
- [x] CONTRIBUTING / CHANGELOG / SECURITY / ARCHITECTURE / INTEGRATION
- [x] `script/check` runs every gate locally (no GitHub Actions)
- [x] PR template + issue templates + CODEOWNERS

## Phase 1 — `bb-terminal-model` (first real crate)

The lowest-level crate, no engine-internal deps. The natural starting
point because both Beam and Beacon have well-tested implementations and
the data model is uncontroversial.

- [x] **Design PR.** Open an `[extraction]` issue listing the source
      paths in Beam (`crates/vt100/...`, `app/src/blocks/grid.rs`) and
      Beacon's equivalent. Sketch the public type surface:
      - `Cell { ch: char, attrs: CellAttrs }`
      - `CellAttrs { fg, bg, bold, italic, underline, hyperlink, … }`
      - `Grid { rows, cols, cells: Vec<Cell> }`
      - `Cursor { row, col, shape, visible }`
      - `Viewport { top: usize, height: usize }`
      - `Scrollback` (ring buffer)
- [x] **Implementation PR.** Land the types + tests. Aim for 80%+ line
      coverage on `bb-terminal-model` since it's the foundation.
- [ ] **Beam adoption PR.** Replace Beam's in-tree types with engine
      types via a thin shim that preserves Beam's call sites.
- [ ] **Beacon adoption PR.** Same shape against Beacon's call sites.
- [ ] **Cleanup PR.** Delete the now-dead in-tree implementations.

## Phase 2 — `bb-vt-parser`

Depends on `bb-terminal-model`. Mostly mechanical because the VT100 /
ANSI handling is well-known.

- [x] Design PR — public API: `Parser { fn feed(&mut self, screen: &mut Screen, bytes: &[u8]) -> Vec<PositionedEvent> }`.
      (The original sketch took a `&mut Grid` and returned nothing. It became
      `&mut Screen` because the parser needs scroll regions and scrollback, and
      it returns *positioned* events because a marker's cursor row must be
      captured when the marker fires — reading it after the chunk makes block
      boundaries depend on PTY read size.)
- [x] Implementation PR — full CSI / SGR / OSC / DCS support. Property
      tests against the existing in-tree parsers (run the same input
      through engine + consumer-in-tree and assert identical grid state)
- [ ] Beam adoption PR
- [ ] Beacon adoption PR
- [ ] Cleanup PR

## Phase 3 — `bb-block-list`

Depends on `bb-terminal-model`. The Warp-style block lifecycle.

- [x] Design PR — public API: `BlockList { … }`, `Block { id, lifecycle, … }`
- [x] Implementation PR — OSC 133 + heuristics, search, navigation
- [ ] Beam adoption PR
- [ ] Beacon adoption PR
- [ ] Cleanup PR

## Phase 4 — `bb-providers` (biggest API surface)

No engine-internal deps. Defines the `ProviderMessage` /
`ProviderEvent` types that the agent loop will use.

- [x] Design PR — `ProviderMessage` enum, `MessageImage`, `ToolCall`,
      `ToolResult`, `ProviderEvent` (streaming), `StopReason`.
      Per-provider encoder traits.
- [x] Implementation PR — Anthropic encoder
- [x] Implementation PR — OpenAI-compat encoder (covers OpenAI,
      OpenRouter, Together, Groq, Moonshot, custom proxies)
- [x] Implementation PR — Gemini encoder
- [x] Implementation PR — Ollama encoder
- [x] Implementation PR — Local-OpenAI-compat encoder
- [ ] Beam adoption PR — Beam's request-time runner consumes
      `bb-providers` instead of its own `agent_loop::provider`.
- [ ] Beacon adoption PR — same shape.
- [ ] Cleanup PR

## Phase 5 — `bb-tool-bridge`

Depends on `bb-providers` (for `ToolResult` shape).

- [x] Design PR — `Tool` trait, `ToolDispatchResult`,
      `ComputerUseDispatcher` analogue.
- [x] Implementation PR — Shell, Read, Write, Edit
- [x] Implementation PR — ComputerUse (Anthropic computer-use tool;
      images flow through `ToolDispatchResult.images`)
- [x] Implementation PR — Subagent dispatcher (Cli, Research, Advice,
      Summarization, ComputerUse, ConversationSearch, WarpDocumentationSearch)
- [x] Implementation PR — WebFetch, WebSearch
- [ ] Beam adoption PR
- [ ] Beacon adoption PR
- [ ] Cleanup PR

## Phase 6 — `bb-agent-loop`

Largest surface area. Depends on `bb-providers` and `bb-tool-bridge`.

- [x] Design PR — `AgentLoop`, `Turn`, `Conversation`, `CancellationToken`
- [x] Implementation PR — turn structure, streaming, retry policy
- [x] Implementation PR — cancellation at every await point (see prior
      `tokio-broadcast-forwarder-stops-before-publisher-drops` and
      `ui-cancel-must-update-async-driven-state` notes)
- [x] Implementation PR — context window / summarization
- [ ] Beam adoption PR
- [ ] Beacon adoption PR
- [ ] Cleanup PR

## Phase 7 — `bb-shared-session` (top-level glue)

Depends on every other crate.

- [x] Design PR — `Session` aggregate
- [x] Implementation PR — wire together terminal-model + block-list +
      agent-loop + history (KV store trait)
- [ ] Beam adoption PR — Beam's top-level session module becomes a thin
      wrapper around `bb-shared-session`.
- [ ] Beacon adoption PR — same shape.
- [ ] Cleanup PR

## Phase 8 — 0.1.0 release

After all seven crates have implementations and both consumers depend
on the engine:

- [ ] Tag `v0.1.0` and bump every crate's version
- [ ] Both consumers pin to `v0.1.0` instead of git SHAs
- [ ] Decide whether to flip the engine repo to public (currently
      private — public is fine once the migration is settled)
- [ ] Public release notes summarizing what the engine is and how
      Beam users / Beacon customers benefit
