# duplex: agent setup

Rust workspace: `proto` (tonic), `crates/gateway` (mic, VAD, playback, barge-in), `crates/worker` (STT, LLM, turns). macOS Apple Silicon only.

- Spec of record: `docs/PRD.md` (gitignored, read it from the main checkout: `$(git rev-parse --git-common-dir)/../docs/PRD.md`). Review notes in `docs/REVIEW.md`.
- Fresh checkout setup: nothing to install. Use a shared target dir to save disk: `export CARGO_TARGET_DIR=$HOME/.cache/duplex-target`.
- Models live in the main checkout under `models/` (gitignored): `ggml-small.bin`, `kokoro-v1.0.int8.onnx`, `voices/` (see Running); Ollama serves `qwen2.5:3b` at http://localhost:11434.
- Validate: `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`; with the optional backend also `DOCS_RS=1 cargo clippy -p gateway --all-targets --features anytts -- -D warnings` (`DOCS_RS=1` only for check/clippy, see lindera note).
- Conventions: conventional commits, no AI attribution, never commit `docs/`, `.claude/`, models. Bounded channels everywhere; lossy only for audio-in. Cancellation via `CancellationToken` + `select!` (biased, cancel first). No `unwrap` on I/O paths. No heap allocation in audio callbacks.
- No UI exists; UX/UI reviewers should judge CLI output and error messages only.
- Mic/speaker cannot be tested headless: provide a WAV-fixture fake-mic mode and unit/integration tests; live latency is measured by the user.
- Record decisions with `madelog note "Chose X over Y: reason."` (one sentence, <140 chars).

## Running

Env vars (defaults in brackets):
- Worker: `WORKER_ADDR` [`127.0.0.1:50051`], `WHISPER_MODEL` (required, path to `ggml-small.bin`), `OLLAMA_URL` [`http://localhost:11434`], `OLLAMA_MODEL` [`qwen2.5:3b`].
- Gateway: `WORKER_ADDR` [`http://127.0.0.1:50051`], `SAY_VOICE` [system voice], `VAD_THRESHOLD` [500, RMS of i16 samples], `FAKE_MIC` [unset; path to a 16 kHz mono s16 WAV replaces the mic].
- Gateway TTS: `TTS` [`kokoro` if the model files exist, else a warning and `say`; `kokoro` | `anytts` | `say`, anything else is a startup error; `anytts` needs `--features anytts`], `KOKORO_MODEL` [`models/kokoro-v1.0.int8.onnx`, relative to cwd], `KOKORO_VOICES` [`models/voices`, a directory of per-voice `.bin` files], `KOKORO_VOICE` [`af_heart`], `KOKORO_SPEED` [1.0]. TTS is in `crates/gateway/src/tts.rs`: Kokoro via `kokoro-en` (ort, default), optional Kokoro via `any-tts` (candle, Metal or CPU, `TTS=anytts`), and `say` as the fallback. Startup warmup synthesizes two sentences (short, medium) and logs each stage; the gateway does not capture or speak before it ends.
- Kokoro files (about 120 MB with all voices, the layout `kokoro-en` expects; run from the repo root):
  `mkdir -p models/voices`
  `curl -L -o models/kokoro-v1.0.int8.onnx https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx`
  `curl -L -o models/voices/af_heart.bin https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af_heart.bin`
  `af_heart.bin` alone is enough for the default; fetch any other `voices/<name>.bin` the same way and select it with `KOKORO_VOICE`. The fp32 `onnx/model.onnx` (~325 MB) also loads. `ort` is pinned to `=2.0.0-rc.13` in `crates/gateway/Cargo.toml` (kokoro-en breaks on newer rc).
- Optional `anytts` backend (`any-tts =0.2.0`, candle Kokoro; cargo feature `anytts`, off by default): `ANYTTS_DEVICE` [`metal` | `cpu` (Accelerate)], `ANYTTS_MODEL` [`models/Kokoro-82M`, relative to cwd], `ANYTTS_GAIN` [1.44, a number above 0], voice from `KOKORO_VOICE`, speed from `KOKORO_SPEED`. Weights, ~330 MB, from the repo root (verified against what any-tts loads: `config.json`, one `*.pth` or `*.safetensors`, `voices/<name>.pt`):
  `mkdir -p models/Kokoro-82M/voices`
  `curl -L -o models/Kokoro-82M/kokoro-v1_0.pth https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth`
  `curl -L -o models/Kokoro-82M/config.json https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json`
  `curl -L -o models/Kokoro-82M/voices/af_heart.pt https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt`
  Run: `TTS=anytts cargo run --release -p gateway --features anytts`. Test (ignored): `ANYTTS_MODEL=$PWD/models/Kokoro-82M [ANYTTS_DEVICE=cpu] cargo test --release -p gateway --features anytts -- --ignored --nocapture anytts` (writes `$TMPDIR/anytts.wav`, prints RTF); `anytts_first_sentence_latency` takes `ANYTTS_WARMUP_STAGES` (1 or 2). Run from the repo root; the `models/` paths are relative to the cwd (in a worktree, point `ANYTTS_MODEL` at the main checkout's `models/Kokoro-82M`).
  Gain: any-tts output is quieter than the ort path; the spike measured output RMS ort/any-tts on three sentences, ratios 1.43, 1.46, 1.42, mean 1.44, which is the default `ANYTTS_GAIN` (applied in `synth()`, clamped to [-1, 1]). A measured starting point; tune with `ANYTTS_GAIN`.
  LINDERA, BUILD-TIME NETWORK: any-tts's `kokoro` feature hardcodes `lindera` with `embed-ipadic`, and lindera-ipadic's build.rs downloads `https://Lindera.dev/mecab-ipadic-2.7.0-20250920.tar.gz` (Japanese dictionary) when building with `--features anytts`. This cannot be turned off (no feature; `DOCS_RS=1` skips it but also makes ort-sys skip linking, so it works for check/clippy only). Set `LINDERA_DICTIONARIES_PATH=$HOME/.cache/lindera-dicts` (LINDERA_CACHE is deprecated) so it downloads once. The default build never builds any-tts, candle or lindera. There is no runtime network access (no `download` feature).
  Cost: RSS about 1.4 GB (Metal) to 2.5 GB (CPU). Startup cost (load + warmup, before the gateway is ready): a freshly built binary on Metal takes ~4.3 s load + ~3.6 s "Hello." + ~0.4 s medium sentence (~8 s total); later runs ~0.25 s + ~0.7 s (~1 s); CPU ~0.3 s + ~0.55 s + ~0.85 s (~1.7 s). The prewarm pays the cold cost, so the first spoken sentence is not slow. A third, long warmup sentence was dropped: it added ~1.7 s (Metal) / ~3.4 s (CPU) with no measured gain. Steady RTF measured: Metal 0.16, CPU 0.35 (ort 0.33-0.5).
- German: Kokoro v1.0 has no German voice; use `TTS=say SAY_VOICE=Anna`. There is no language detection.

Commands (from the repo root, with `CARGO_TARGET_DIR` exported):
- Worker: `WHISPER_MODEL=$(git rev-parse --git-common-dir)/../models/ggml-small.bin cargo run --release -p worker`
- Gateway (mic, use headphones): `cargo run --release -p gateway`
- Gateway with the fake mic: `FAKE_MIC=fixtures/en.wav cargo run --release -p gateway`
- Real-model tests (ignored by default): `WHISPER_MODEL=/abs/path/ggml-small.bin cargo test -p worker -- --ignored`; `KOKORO_MODEL=/abs/kokoro-v1.0.int8.onnx KOKORO_VOICES=/abs/voices cargo test --release -p gateway -- --ignored --nocapture` (writes `$TMPDIR/kokoro.wav`, prints synth time)

The gateway prints one line per turn to stdout (`turn= ttfa_ms= stt_ms= ttft_ms=`, plus `barge_in_ms=` for barged-in turns); logs go to stderr (`RUST_LOG` filters both binaries).
