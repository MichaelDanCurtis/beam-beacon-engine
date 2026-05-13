# Changelog

All notable changes to `beam-beacon-engine` will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
starting at `0.1.0`. The current `0.0.0` is a pre-release scaffold; breaking
changes can land any time without a version bump.

## [Unreleased]

### Added

- Initial scaffold: workspace, dual-license (MIT OR Apache-2.0), NOTICE
  with the no-upstream-Warp-code principle.
- Seven stub crates: `bb-terminal-model`, `bb-vt-parser`, `bb-block-list`,
  `bb-agent-loop`, `bb-providers`, `bb-tool-bridge`, `bb-shared-session`.
- Architecture and integration docs under `docs/`.
- GitHub Actions CI (build + test + clippy + fmt) for `main` and PRs.

### Notes

This is the genesis commit. The crates are empty — real implementations
land in subsequent PRs. See [`docs/INTEGRATION.md`](docs/INTEGRATION.md)
for the migration plan from in-tree Beam / Beacon implementations to
shared engine crates.
