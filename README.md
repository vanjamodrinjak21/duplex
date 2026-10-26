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

See `AGENTS.md` for all environment variables. An optional candle-based Kokoro backend (`TTS=anytts`, `--features anytts`) is described there too.

## Results

| Metric | Value |
|---|---|
| TTFA (end of speech to first audio (speaker started)) | to be measured |
| Barge-in (speech detected to playback stopped) | to be measured |

The audio callbacks (mic input and speaker output) make 0 heap allocations over 2000 calls each, including underrun, ring overflow and kill; this is checked by counting-allocator unit tests, not by live measurement.

## Measuring

Run the worker and the gateway in two terminals, headphones on, and save the gateway's stdout:

```sh
export CARGO_TARGET_DIR=$HOME/.cache/duplex-target
WHISPER_MODEL=models/ggml-small.bin cargo run --release -p worker
cargo run --release -p gateway | tee run.log
```

Talk for a few turns, interrupting some replies, then stop with Ctrl-C. Each turn prints one stdout line: `ttfa_ms=` is end of speech to first audio (speaker started), and barged-in turns add `barge_in_ms=` (speech detected to playback stopped). Medians of both fields:

```sh
for f in ttfa_ms barge_in_ms; do grep -o "$f=[0-9]*" run.log | cut -d= -f2 | sort -n | awk -v f=$f '{a[NR]=$1} END {if (NR) print f, (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2), "n=" NR}'; done
```

Paste the medians into the Results table above.

## Limitations

- Headphones required: there is no echo cancellation, so speaker output would trigger barge-in.
- With `TTS=say`, `say` starts per sentence as a new process, which adds its own startup latency to TTFA.
- If the TTS engine fails on one sentence, that sentence is skipped with a warning on stderr (a silent gap).
- Ollama serialises requests by default, so concurrent sessions queue behind each other.
