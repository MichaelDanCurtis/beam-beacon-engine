# beam-beacon-engine

Shared engine for the two Ailuxis terminal products:

- **[Beam](https://www.beam.Ailuxis.com)** — the open-source agentic terminal (a Warp fork, MIT OR Apache-2.0).
- **Beacon** — the proprietary enterprise terminal (closed source, sold).

Both products consume this repository as a workspace of small focused crates. Because the engine is licensed permissively (MIT OR Apache-2.0) and contains no upstream-Warp code, it cleanly bridges the OSS and commercial sides of the product line without forcing a copyleft cascade.

```
                         ARC (proprietary service)
                                  ▲ HTTPS / WS
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
                Beam (OSS,                    Beacon (proprietary,
                MIT/Apache-2.0,               enterprise, sold)
                free)                             │
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                       beam-beacon-engine
                       (this repo, MIT/Apache-2.0,
                       100% hand-coded, internal)
```

## Status

> **Implemented, pre-adoption.** All seven crates below have full, freshly-written implementations backed by 487 tests at 97% line coverage, including end-to-end tests that drive realistic shell transcripts through the whole seam and adversarial probes for wide-character integrity, parser state-machine recovery, and agent cancellation. CI enforces eight gates: libs-only build, tests, doctests, a coverage floor, `clippy -D warnings`, `fmt --check`, an MSRV build, and `cargo deny`. What remains is consumer-side: wiring Beam and Beacon to depend on the engine and deleting their in-tree duplicates — see [`docs/INTEGRATION.md`](docs/INTEGRATION.md) and [`docs/MIGRATION-CHECKLIST.md`](docs/MIGRATION-CHECKLIST.md).

## Crates

Each lives under [`crates/`](crates/). The split mirrors the natural seams in a Warp-style agentic terminal so both consumers can opt in to whatever subset they need.

| Crate                | Responsibility                                                                 |
| -------------------- | ------------------------------------------------------------------------------ |
| `bb-terminal-model`  | Terminal state: grid, cells, attributes, cursor, viewport.                     |
| `bb-vt-parser`       | VT100 / ANSI escape sequence parser feeding `bb-terminal-model`.               |
| `bb-block-list`      | Warp-style command-output blocks: lifecycle, identity, search, navigation.     |
| `bb-agent-loop`      | The core AI agent loop: tool-use, streaming, retry policy, conversation state. |
| `bb-providers`       | AI provider abstraction (Anthropic, OpenAI-compat, Gemini, Ollama, local).     |
| `bb-tool-bridge`     | Bridge between the agent's tool calls and the host runtime (shell, FS, etc.).  |
| `bb-shared-session`  | Session state shared across surfaces: blocks, agent, terminal model, history.  |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the rationale behind each split and the rough dependency graph.

## Consuming this engine

### From Beam (OSS)

In `Cargo.toml`:

```toml
[workspace.dependencies]
bb-shared-session = { git = "https://github.com/MichaelDanCurtis/beam-beacon-engine", package = "bb-shared-session", branch = "main" }
# ...etc
```

Pin to a specific commit or tag for release builds.

### From Beacon (proprietary)

Same shape; Beacon pulls the engine via private git auth (the engine repo is private until both consumers are wired up).

### Local dev (when iterating on the engine itself)

In both consumer workspaces, switch the dep to a path override during development:

```toml
[patch."https://github.com/MichaelDanCurtis/beam-beacon-engine"]
bb-shared-session = { path = "../beam-beacon-engine/crates/bb-shared-session" }
# ...etc
```

This lets you test engine changes in Beam or Beacon without pushing intermediate engine commits.

## License

Dual-licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT license](LICENSE-MIT)

at your option. Workspace `Cargo.toml` declares `license = "MIT OR Apache-2.0"`. See [`NOTICE.md`](NOTICE.md) for provenance.

## Contributing

This is currently internal Ailuxis infrastructure. We'll open contribution paths once the engine stabilizes. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the internal flow.

## Development

```bash
cargo check --workspace          # fast type-check across all crates
cargo build --workspace --lib    # build the LIBRARIES, as a consumer does
cargo test  --workspace          # run every test, including doctests
cargo fmt   --all                # format everything
cargo clippy --workspace -- -D warnings
cargo deny check                 # license / advisory / source gate
cargo llvm-cov --workspace       # line coverage (CI floor: 85%)
```

Two `--all-targets` traps this repo has already been bitten by, both of which
CI now guards separately:

- `cargo test --workspace --all-targets` **skips doctests**. CI runs
  `--all-targets` and `--doc` as separate steps.
- `--all-targets` also builds test targets, which unifies
  `[dev-dependencies]` features into the build. That can leave a library
  that only compiles because a dev-dependency enabled a feature it uses —
  green CI, unbuildable crate. CI therefore also runs
  `cargo build --workspace --lib`.

Minimum Rust version: **1.78** (set in `[workspace.package].rust-version`)
and verified by a dedicated CI job, since both consumers pin to it.
