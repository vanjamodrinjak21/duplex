# Security

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue for a security problem.

## Security model

duplex is a local development tool, not a hardened network service. Know these limits before running it:

- **No authentication.** The worker's gRPC service accepts any client. It binds to `127.0.0.1` by default. Do not bind it to `0.0.0.0` or expose it beyond your machine without putting authentication and TLS in front of it.
- **Microphone.** The gateway captures your microphone as soon as it is ready. Use headphones; there is no echo cancellation.
- **Local files and processes.** Model, voice and fixture paths come from environment variables you set and are trusted. The gateway can spawn `say` (macOS) and, through `kokoro-en`, `espeak-ng`; text that reaches them is stripped of leading dashes, but treat the LLM output as untrusted input.
- **Input limits.** The worker caps one utterance at 60 s of audio, keeps the last 6 turns of history, caps `spoken_text` at 4 KiB and fails an Ollama response line longer than 1 MiB.
- **Build-time network access.** Building with `--features anytts` downloads a dictionary from lindera.dev once (MD5-checked). The default build does not. `ort` downloads ONNX Runtime binaries at build time.
- **Models are not checksummed.** The download commands in `AGENTS.md` do not pin SHA-256 hashes yet. Verify the files yourself.

There is no supported-version policy yet; fixes land on `main`.
