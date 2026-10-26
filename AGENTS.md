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

## Running

Env vars (defaults in brackets):
- Worker: `WORKER_ADDR` [`127.0.0.1:50051`], `WHISPER_MODEL` (required, path to `ggml-small.bin`), `OLLAMA_URL` [`http://localhost:11434`], `OLLAMA_MODEL` [`qwen2.5:3b`].
- Gateway: `WORKER_ADDR` [`http://127.0.0.1:50051`], `SAY_VOICE` [system voice], `VAD_THRESHOLD` [500, RMS of i16 samples], `FAKE_MIC` [unset; path to a 16 kHz mono s16 WAV replaces the mic].

Commands (from the repo root, with `CARGO_TARGET_DIR` exported):
- Worker: `WHISPER_MODEL=$(git rev-parse --git-common-dir)/../models/ggml-small.bin cargo run --release -p worker`
- Gateway (mic, use headphones): `cargo run --release -p gateway`
- Gateway with the fake mic: `FAKE_MIC=fixtures/en.wav cargo run --release -p gateway`
- Real-model tests (ignored by default): `WHISPER_MODEL=/abs/path/ggml-small.bin cargo test -p worker -- --ignored`

The gateway prints one line per turn to stdout (`turn= ttfa_ms= stt_ms= ttft_ms=`, plus `barge_in_ms=` for barged-in turns); logs go to stderr (`RUST_LOG` filters both binaries).
