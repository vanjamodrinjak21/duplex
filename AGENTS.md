# duplex: agent setup

Rust workspace: `proto` (tonic), `crates/gateway` (mic, VAD, playback, barge-in), `crates/worker` (STT, LLM, turns). macOS Apple Silicon only.

- Spec of record: `docs/PRD.md` (gitignored, read it from the main checkout: `$(git rev-parse --git-common-dir)/../docs/PRD.md`). Review notes in `docs/REVIEW.md`.
- Fresh checkout setup: nothing to install. Use a shared target dir to save disk: `export CARGO_TARGET_DIR=$HOME/.cache/duplex-target`.
- Models live in the main checkout under `models/` (gitignored): `ggml-small.bin`, `kokoro-v1.0.int8.onnx`, `voices/` (see Running); Ollama serves `qwen2.5:3b` at http://localhost:11434.
- Validate: `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`.
- Conventions: conventional commits, no AI attribution, never commit `docs/`, `.claude/`, models. Bounded channels everywhere; lossy only for audio-in. Cancellation via `CancellationToken` + `select!` (biased, cancel first). No `unwrap` on I/O paths. No heap allocation in audio callbacks.
- No UI exists; UX/UI reviewers should judge CLI output and error messages only.
- Mic/speaker cannot be tested headless: provide a WAV-fixture fake-mic mode and unit/integration tests; live latency is measured by the user.
- Record decisions with `madelog note "Chose X over Y: reason."` (one sentence, <140 chars).

## Running

Env vars (defaults in brackets):
- Worker: `WORKER_ADDR` [`127.0.0.1:50051`], `WHISPER_MODEL` (required, path to `ggml-small.bin`), `OLLAMA_URL` [`http://localhost:11434`], `OLLAMA_MODEL` [`qwen2.5:3b`].
- Gateway: `WORKER_ADDR` [`http://127.0.0.1:50051`], `SAY_VOICE` [system voice], `VAD_THRESHOLD` [500, RMS of i16 samples], `FAKE_MIC` [unset; path to a 16 kHz mono s16 WAV replaces the mic].
- Gateway TTS: `TTS` [`kokoro` if the model files exist, else a warning and `say`; `kokoro` | `say`, anything else is a startup error], `KOKORO_MODEL` [`models/kokoro-v1.0.int8.onnx`, relative to cwd], `KOKORO_VOICES` [`models/voices`, a directory of per-voice `.bin` files], `KOKORO_VOICE` [`af_heart`], `KOKORO_SPEED` [1.0]. TTS is Kokoro in the gateway (`crates/gateway/src/tts.rs`, crate `kokoro-en`); `say` is the fallback.
- Kokoro files (from the repo root, about 120 MB, the layout `kokoro-en` expects):
  `mkdir -p models/voices && cd models && curl -L -o kokoro-v1.0.int8.onnx https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx` then one `curl -L -O` per voice from `https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/<name>.bin` into `models/voices/` (`af_heart` is the default; `af_bella`, `am_adam`, `bf_emma` also work). The fp32 `onnx/model.onnx` (~325 MB) also loads.
- German: Kokoro v1.0 has no German voice; use `TTS=say SAY_VOICE=Anna`. There is no language detection.

Commands (from the repo root, with `CARGO_TARGET_DIR` exported):
- Worker: `WHISPER_MODEL=$(git rev-parse --git-common-dir)/../models/ggml-small.bin cargo run --release -p worker`
- Gateway (mic, use headphones): `cargo run --release -p gateway`
- Gateway with the fake mic: `FAKE_MIC=fixtures/en.wav cargo run --release -p gateway`
- Real-model tests (ignored by default): `WHISPER_MODEL=/abs/path/ggml-small.bin cargo test -p worker -- --ignored`; `KOKORO_MODEL=/abs/kokoro-v1.0.int8.onnx KOKORO_VOICES=/abs/voices cargo test --release -p gateway -- --ignored --nocapture` (writes `$TMPDIR/kokoro.wav`, prints synth time)

The gateway prints one line per turn to stdout (`turn= ttfa_ms= stt_ms= ttft_ms=`, plus `barge_in_ms=` for barged-in turns); logs go to stderr (`RUST_LOG` filters both binaries).
