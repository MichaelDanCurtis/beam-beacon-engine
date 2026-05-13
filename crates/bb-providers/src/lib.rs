//! AI provider abstraction.
//!
//! Part of the [`beam-beacon-engine`] workspace — the shared engine for
//! Beam (OSS) and Beacon (proprietary).
//!
//! # Responsibility
//!
//! `bb-providers` defines the trait surface that [`bb-agent-loop`] calls
//! and the per-provider implementations behind it:
//!
//! - **Anthropic** — native Messages API, image blocks, tool use,
//!   thinking, prompt caching.
//! - **OpenAI-compatible** — Chat Completions and Responses APIs;
//!   `image_url` data URLs; covers OpenAI, OpenRouter, Together, Groq,
//!   Moonshot/Kimi, custom proxies.
//! - **Gemini** — inline image data, tool calling, streaming.
//! - **Ollama** — native chat with `images` arrays.
//! - **Local OpenAI-compatible** — Ollama-on-`/v1`, LM Studio, vLLM,
//!   llama.cpp.
//!
//! The core abstraction is a `ProviderMessage` / `ProviderEvent` pair
//! whose multimodal content (text + images + tool calls + tool results)
//! is encoded into each provider's wire format by a per-provider
//! encoder. The same `ProviderMessage` round-trips through any
//! provider without lossy conversion.
//!
//! # Status
//!
//! Scaffold. Real implementation lands in subsequent PRs.
//!
//! [`beam-beacon-engine`]: https://github.com/MichaelDanCurtis/beam-beacon-engine
//! [`bb-agent-loop`]: https://docs.rs/bb-agent-loop

#![forbid(unsafe_code)]
