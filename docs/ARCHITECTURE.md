# Architecture

This document explains why `beam-beacon-engine` exists, what each crate
does, and how the pieces talk to each other.

## Why a shared engine

Ailuxis ships two terminal products that overlap substantially:

- **Beam** — open-source, MIT/Apache-2.0, free. A Warp-derived agentic
  terminal that anyone can download and use.
- **Beacon** — proprietary, closed-source, sold to enterprises. Built
  from scratch by Ailuxis with no Warp lineage. Targets paid users with
  features Beam can't ship (admin policy, fleet management, custom
  integrations).

A growing share of the codebase is duplicated between the two:

- A VT100 / ANSI parser
- A terminal model (grid, cells, attributes, cursor)
- The Warp-style command-output block list
- The AI agent loop (provider calls, tool use, streaming)
- The provider abstraction (Anthropic, OpenAI-compat, Gemini, Ollama,
  local)
- The tool-call bridge to the host (shell, FS, computer-use)
- Session state and history

`beam-beacon-engine` is where this shared substrate lives. Both consumers
pull from this repo. The engine is permissively licensed
(MIT OR Apache-2.0) so Beam can stay free / open-source while Beacon can
ship as a closed-source commercial product.

## License-cascade safety

The engine is licensed `MIT OR Apache-2.0`. **No upstream-Warp code lives
in this repository.** That rule is enforced socially (in
`CONTRIBUTING.md`) and reinforced in `NOTICE.md`:

- The Beam fork still descends from upstream Warp. Even though Beam
  itself flipped to MIT OR Apache-2.0 by exercising Warp's MIT offer,
  having Warp-derived code in a *separate* engine repo would re-introduce
  the dual-license question every time we vendor a snippet across.
- Beacon is 100% hand-coded. The engine is the same.
- The engine doesn't need to extract anything from either consumer
  verbatim — when a feature lives in both, we rewrite it in the engine
  and the consumers cut over.

## Product topology

```
                         ARC (proprietary service)
                                  ▲ HTTPS / WS
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
                Beam (OSS,                    Beacon (proprietary,
                MIT/Apache-2.0,               enterprise, sold)
                free, AGPL-fork-               │
                relicensed)                    │
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                       beam-beacon-engine
                       (this repo, MIT/Apache-2.0,
                       100% hand-coded, internal)
```

- **ARC** is the Ailuxis service tier. Both Beam and Beacon talk to it
  over HTTPS / WebSockets. ARC code does NOT live in this engine — ARC
  is a server product, the engine is the client substrate.
- **Beam** is the OSS client. It consumes the engine plus its own
  product-specific UX layer.
- **Beacon** is the proprietary client. Same shape: engine plus its
  own UX layer, with extra enterprise features.

## Crate split

The engine is intentionally a workspace of small focused crates, not a
single mega-crate. This lets either consumer opt in to whatever subset
it needs and keeps the dependency graph inspectable.

```
                ┌─────────────────────────────┐
                │     bb-shared-session       │  <- top-level glue
                └──┬──────────┬──────────┬────┘
                   │          │          │
            ┌──────▼──┐  ┌────▼─────┐  ┌─▼─────────────┐
            │ blocks  │  │ agent    │  │ session state │
            │ ┌─────┐ │  │ loop     │  │ (history,     │
            │ │bl-l │ │  │          │  │  identity,    │
            │ └─────┘ │  │          │  │  auth ctx)    │
            └────┬────┘  └─┬────┬───┘  └───────────────┘
                 │         │    │
                 │   ┌─────▼─┐  │
                 │   │ tool  │  │
                 │   │bridge │  │
                 │   └───┬───┘  │
                 │       │      │
            ┌────▼───────▼──┐   │
            │ bb-terminal-  │   │
            │   model       │   │
            └────┬──────────┘   │
                 │              │
            ┌────▼─────────┐ ┌──▼──────────────┐
            │ bb-vt-parser │ │ bb-providers    │
            └──────────────┘ │ (anthropic,     │
                             │  openai-compat, │
                             │  gemini,        │
                             │  ollama, local) │
                             └─────────────────┘
```

### `bb-terminal-model`

Pure data: the in-memory representation of a terminal screen. Row × col
grid of cells, per-cell graphics attributes (foreground, background,
bold, italic, underline, hyperlink), cursor position and shape,
scrollback ring, viewport projection.

No parsing. No rendering. No I/O. Just types and the operations that
mutate them.

### `bb-vt-parser`

Stateful parser. Consumes bytes from a PTY stream and applies actions to
a `bb-terminal-model` instance. Handles C0/C1 control codes, CSI/SGR/OSC
sequences, DCS passthrough, UTF-8 decoding with chunk-boundary safety
(see the `vt100-contents-formatted-returns-ansi-bytes` and
`rust-sse-from-utf8-lossy-chunk-boundary-corruption` notes from prior
work).

### `bb-block-list`

The Warp-style command-output block structure. Sits on top of
`bb-terminal-model`. Groups output rows into command-bounded blocks
using the OSC 133 prompt protocol where available, falling back to
heuristics. Owns block identity, lifecycle (started → running →
finished), metadata (command, exit code, duration, cwd), and search.

### `bb-providers`

The trait surface that the agent loop calls and the per-provider
implementations behind it. The core abstraction is a `ProviderMessage`
/ `ProviderEvent` pair whose multimodal content (text + images + tool
calls + tool results) is encoded into each provider's wire format by a
per-provider encoder. The same `ProviderMessage` round-trips through any
provider without lossy conversion. Encoders covered:

- Anthropic Messages API (image blocks, thinking, prompt caching)
- OpenAI-compatible Chat Completions and Responses APIs (`image_url`
  data URLs; covers OpenAI, OpenRouter, Together, Groq, Moonshot/Kimi,
  custom proxies)
- Gemini (inline image data, tool calling, streaming)
- Ollama native chat (`images` array)
- Local OpenAI-compatible (Ollama-on-`/v1`, LM Studio, vLLM, llama.cpp)

### `bb-tool-bridge`

When the agent loop receives a tool-use intent from a model, the tool
bridge is what actually runs it. The host (Beam or Beacon) injects
implementations for each tool kind:

- `Shell` — run a command in a PTY, capture stdout/stderr, surface exit
  code. Often delegates back to the host's own block-aware shell.
- `Read` / `Write` / `Edit` — filesystem actions with the host's
  permission model.
- `ComputerUse` — screenshot + click + keystroke for Anthropic's
  computer-use tool; results return as `ToolDispatchResult.images`
  for the vision pipeline.
- `Subagent` — recursive agent dispatch (Cli, Research, Advice,
  Summarization, ComputerUse, ConversationSearch, WarpDocumentationSearch).
- `WebFetch` / `WebSearch` — host-provided network tools.

The bridge produces a uniform `ToolDispatchResult` (text + optional
images + structured fields) that `bb-agent-loop` surfaces back to the
provider as a `ToolResult` provider message.

### `bb-agent-loop`

Orchestrates an AI agent turn-by-turn over the abstractions in
`bb-providers` and `bb-tool-bridge`. Owns:

- Turn structure: user input → provider call → tool calls → tool
  results → next provider call → terminal output.
- Streaming token delivery to the UI.
- Cancellation (Esc / Ctrl-C / explicit stop) at every await point —
  see the `tokio-broadcast-forwarder-stops-before-publisher-drops` and
  `ui-cancel-must-update-async-driven-state` notes.
- Tool-call dispatch via `bb-tool-bridge`.
- Retry / fallback policy for transient provider errors.
- Conversation state: history, summarization triggers, context window
  management.

### `bb-shared-session`

Top-level glue. Owns the `Session` aggregate:

- The current `bb-terminal-model` grid + scrollback
- The `bb-block-list` view over that grid
- The `bb-agent-loop` state for any in-flight agent turn
- Persistent history (recent sessions, named workspaces) backed by a
  host-provided KV store (sqlite in both consumers today)
- Identity / auth context surfaced from the host (anonymous, signed in,
  BYOK, ARC-attached)

This is the crate that both consumers depend on at the top level. The
other engine crates are reachable through it.

## Versioning

Crates start at `0.0.0`. We'll bump to `0.1.x` when the first consumer
actually depends on a crate in production. After that:

- Patch (`0.1.x` → `0.1.x+1`): bug fixes, no API changes.
- Minor (`0.1.x` → `0.2.0`): API additions, possible breaking changes
  for as long as we're still in `0.y`.
- Major (`1.0.0`): once the API surface is settled and we're comfortable
  committing to semver.

All crates ship together. A workspace-level version bump touches every
crate.
