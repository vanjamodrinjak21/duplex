# duplex: agent setup

Rust workspace: `proto` (tonic), `crates/gateway` (mic, VAD, playback, barge-in), `crates/worker` (STT, LLM, turns). macOS Apple Silicon only.

- Spec of record: `docs/PRD.md` (gitignored, read it from the main checkout: `$(git rev-parse --git-common-dir)/../docs/PRD.md`). Review notes in `docs/REVIEW.md`.
- Fresh checkout setup: nothing to install. Use a shared target dir to save disk: `export CARGO_TARGET_DIR=$HOME/.cache/duplex-target`.
- Models live in the main checkout under `models/` (gitignored): `ggml-small.bin`; Ollama serves `qwen2.5:3b` at http://localhost:11434.
- Validate: `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`.
- Conventions: conventional commits, no AI attribution, never commit `docs/`, `.claude/`, models. Bounded channels everywhere; lossy only for audio-in. Cancellation via `CancellationToken` + `select!` (biased, cancel first). No `unwrap` on I/O paths. No heap allocation in audio callbacks.
- No UI exists; UX/UI reviewers should judge CLI output and error messages only.
- Mic/speaker cannot be tested headless: provide a WAV-fixture fake-mic mode and unit/integration tests; live latency is measured by the user.
- Record decisions with `madelog note "Chose X over Y: reason."` (one sentence, <140 chars).
