# Security Policy

This repository is currently private Ailuxis infrastructure consumed by Beam
(OSS) and Beacon (proprietary). If you discover a security vulnerability in
the engine itself or in either consumer:

## Reporting

Email <security@ailuxis.com> with:

- A clear description of the issue
- Reproduction steps (or a minimal proof-of-concept)
- Your assessment of impact (data exposure, privilege escalation, RCE, etc.)
- Whether you'd like credit on the fix

**Do not file a public GitHub issue** for a vulnerability. We'll respond
within 72 hours with an initial triage.

## Disclosure timeline

- We aim to ship a fix within 30 days of triage for serious issues
  (anything we'd classify as high or critical).
- Once the fix is released to both Beam and Beacon, we'll publish a
  short advisory in this repo's release notes (if the issue is in the
  engine) or in Beam's repo (if the issue is consumer-side).
- Coordinated disclosure with reporters is welcome; please negotiate the
  timeline with us before publishing.

## Scope

In-scope:

- Bugs in any crate under `crates/` that could lead to RCE, data
  corruption, privilege escalation, or sensitive-data leakage.
- API-shape issues that force consumers into insecure patterns.

Out-of-scope (file a regular bug on the affected repo instead):

- Vulnerabilities in upstream dependencies (`cargo audit` will surface
  these; file them with the dep maintainer).
- Issues only reproducible in Beam or Beacon and not traceable to this
  engine — file those on `MichaelDanCurtis/WarpMod` or the Beacon repo.
- Tooling configuration (CI, build scripts).
