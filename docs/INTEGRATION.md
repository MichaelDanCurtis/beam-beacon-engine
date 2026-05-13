# Integration Plan: Beam ↔ engine ↔ Beacon

This doc captures the staged migration from "engine code lives in Beam
and Beacon directly" to "engine code lives in `beam-beacon-engine`,
both consumers pull it." Read this BEFORE landing any extraction PR
so we don't end up with three copies of the same logic.

## Current state (snapshot at scaffold time)

- **Beam** (`MichaelDanCurtis/WarpMod`) has an in-tree implementation of
  every engine concern. Most of it sits in `app/src/ai/...` (agent
  loop, providers, tool bridge), `crates/vt100/...` (vt parser),
  `app/src/blocks/...` (block list), etc.
- **Beacon** (`MichaelDanCurtis/Beacon` — `WarpClone/beacon-core` +
  `beacon-mac`) has hand-coded analogues with substantial overlap. The
  agent loop and the provider abstraction are the most duplicated.
- **`beam-beacon-engine`** (this repo) is empty. Crates exist as
  stubs. Nothing depends on them yet.

## Migration order

The order matters because the crates depend on each other (see
`docs/ARCHITECTURE.md` for the graph). We extract bottom-up:

1. **`bb-terminal-model`** — pure data, no dependencies. The natural
   first extraction.
2. **`bb-vt-parser`** — depends on `bb-terminal-model`. Already a
   well-known pattern; mostly mechanical.
3. **`bb-block-list`** — depends on both above. The Warp-style block
   identity / lifecycle is unusually well-understood from years of
   Beam UX iteration.
4. **`bb-providers`** — depends on nothing in this workspace. Big
   surface area (5 encoders) but the unifying `ProviderMessage` type
   already exists in Beam's `app/src/ai/agent_loop/loop_runner.rs`
   (`MessageImage`, `ProviderMessage::UserMultimodal`, `ToolResult.images`).
5. **`bb-tool-bridge`** — depends on `bb-providers`. The `ComputerUseDispatcher`
   pattern (Beam's `app/src/ai/agent_loop/computer_use_dispatcher.rs`)
   is the cleanest reference.
6. **`bb-agent-loop`** — depends on providers + tool bridge. Largest
   crate by surface area; extract last among the "core" five.
7. **`bb-shared-session`** — depends on all of the above. Becomes
   the consumer-facing entry point.

## Extraction recipe (per crate)

For each crate above, follow this recipe:

### 1. Design the engine API

- Read both Beam's and Beacon's current implementations as reference.
- Sketch the public API you want **here, in the engine**, not just a
  copy-paste of either consumer's shape. The engine should be the
  prettier abstraction, not the union of two messy concrete ones.
- Open a *design* PR against this repo that adds types + trait
  signatures + doc comments. Zero implementation yet. Merge when both
  consumers' implementers have signed off that the shape works.

### 2. Implement in the engine

- Write fresh code. No copy-paste from Beam or Beacon. Use them as
  reference but rewrite cleanly here (legal cleanliness + an
  opportunity to fix design debt accumulated in the consumers).
- Land tests with the implementation. Engine crates should have
  high test coverage because the consumers will rely on the engine
  staying stable.
- Run `cargo check --workspace + cargo clippy --workspace -- -D warnings`
  + `cargo test --workspace` before opening the PR.

### 3. Wire Beam

- Add a git dependency on this repo in Beam's `Cargo.toml`. Pin to the
  commit SHA that landed the engine crate.
- Migrate Beam's in-tree implementation to call into the engine. Often
  this can be a thin shim that delegates to the engine type while
  preserving Beam's call sites.
- Run Beam's full test suite (`./script/presubmit`). Don't merge until
  it's green.
- During this migration phase, Beam still has the old implementation
  in tree alongside the engine call. Don't delete the old code yet.

### 4. Wire Beacon

- Same shape as Beam. Beacon is a private repo so the engine git URL
  needs to work with Beacon's CI auth (typically a deploy key or PAT).
- Run Beacon's test suite. Don't merge until green.

### 5. Delete the old in-tree code

- Once both consumers ship with the engine version and we've burned
  in for a release cycle, open a *cleanup* PR on each consumer that
  removes the old in-tree implementation.
- This is the only PR that actually deletes the duplicated code. Until
  this step, every change still has to happen in three places.

### 6. Update versions

- Bump the engine crate's version (e.g. `0.0.0` → `0.1.0` when first
  consumed; `0.1.0` → `0.2.0` if the API breaks during the migration).
- Update consumers to pin to the new tag rather than a raw SHA.

## Local development with patch.crates-io

When iterating on the engine and a consumer simultaneously, avoid the
push-pull-push-pull dance by using a path-override patch in the
consumer's `Cargo.toml`:

```toml
[patch."https://github.com/MichaelDanCurtis/beam-beacon-engine"]
bb-shared-session = { path = "../beam-beacon-engine/crates/bb-shared-session" }
bb-agent-loop     = { path = "../beam-beacon-engine/crates/bb-agent-loop" }
# ...etc
```

This works for both Beam and Beacon. Remove the patch block before
opening a PR; commit only the git-dep entry.

## Risks and mitigations

| Risk                                                          | Mitigation                                                              |
| ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Engine API churn during migration breaks Beam OR Beacon       | Pin consumers to specific commits, not branches. Bump only after both pass. |
| Engine code accidentally vendored from upstream Warp          | Code review checks every new file's provenance; `CONTRIBUTING.md` is explicit. |
| Beacon-only feature leaks into the engine                     | Keep Beacon-specific features in Beacon. The engine is the **intersection**, not the union. |
| ARC client logic ends up here                                 | ARC is a service; its client lives in either Beam, Beacon, or its own client crate — not here. |
| Public API set in stone too early                             | Crates stay at `0.0.x` until both consumers depend on them in earnest. |

## Open decisions

These don't block the scaffold but need answers before we hit `0.1.0`:

1. **Crate naming.** Current prefix is `bb-` (Beam Beacon). Do we want
   to rename to a brand-neutral prefix once `ailuxis.com` solidifies as
   the parent brand? Mechanical rename — defer until first consumer
   integration.
2. **`bb-providers` granularity.** Should each provider live in its
   own sub-crate (`bb-providers-anthropic`, `bb-providers-openai`)?
   Probably yes for the bigger ones. Wait until we see the real
   compile-time cost of bundling all encoders in one crate.
3. **WebAssembly target.** Does Beacon ever need a wasm build of any
   engine crate? If yes, gate `bb-tool-bridge` features behind a `wasm`
   cfg flag.
4. **Telemetry / metrics.** Where does the agent loop emit metrics from?
   Probably a host-provided trait so each consumer can plug its own
   sink (Beam → console; Beacon → enterprise telemetry; tests → in-memory).
