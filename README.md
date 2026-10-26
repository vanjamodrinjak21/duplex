# duplex

A local full-duplex voice loop for macOS: speak into the mic, a Rust worker transcribes with whisper.cpp (Metal) and streams a reply from Ollama, and the gateway speaks it sentence by sentence, stopping the moment you interrupt.

## Run it

```sh
export CARGO_TARGET_DIR=$HOME/.cache/duplex-target
ollama pull qwen2.5:3b
WHISPER_MODEL=models/ggml-small.bin cargo run --release -p worker
cargo run --release -p gateway                                # live mic, headphones on
FAKE_MIC=fixtures/en.wav cargo run --release -p gateway       # headless, WAV as the mic
```

See `AGENTS.md` for all environment variables.

## Results

| Metric | Value |
|---|---|
| TTFA (end of speech to `say` started) | to be measured |
| Barge-in (speech detected to `say` killed) | to be measured |

## Limitations

- Headphones required: there is no echo cancellation, so speaker output would trigger barge-in.
- `say` starts per sentence as a new process, which adds its own startup latency to TTFA.
- Ollama serialises requests by default, so concurrent sessions queue behind each other.
