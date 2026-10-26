use std::sync::{atomic::AtomicU64, Arc};
use std::time::Duration;

use anyhow::Context;
use gateway::audio::{start_mic, FakeMic};
use gateway::session::{run, SaySpeaker};
use gateway::tts::{parse_tts, AnySpeaker, KokoroSpeaker, KokoroSynth, Tts};
use gateway::{DropOldest, DEFAULT_VAD_THRESHOLD};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logs go to stderr so stdout carries only the per-turn latency lines.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let addr = std::env::var("WORKER_ADDR").unwrap_or_else(|_| "http://127.0.0.1:50051".into());
    let threshold = match std::env::var("VAD_THRESHOLD") {
        Ok(v) => v
            .parse()
            .with_context(|| format!("VAD_THRESHOLD={v} is not a number"))?,
        Err(_) => DEFAULT_VAD_THRESHOLD,
    };
    let (speaker, _speaker_stream) = pick_speaker().await?;
    let speaker = Arc::new(speaker);
    let shutdown = CancellationToken::new();
    tokio::spawn({
        let s = shutdown.clone();
        async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                tracing::info!("SIGINT: stopping");
            }
            s.cancel();
        }
    });
    // Hard stop 2 s after SIGINT even if the worker does not answer.
    let deadline = async {
        shutdown.cancelled().await;
        tokio::time::sleep(Duration::from_secs(2)).await;
    };

    if let Ok(path) = std::env::var("FAKE_MIC") {
        let mut src = FakeMic::open(&path)?;
        tracing::info!(%path, "fake mic");
        tokio::select! {
            _ = run(&addr, &mut src, speaker, threshold, shutdown.clone()) => {}
            _ = deadline => tracing::warn!("forced exit after 2 s"),
        }
        return Ok(());
    }
    let q = DropOldest::new(50);
    let ring_dropped = Arc::new(AtomicU64::new(0));
    let (_stream, _capture) = start_mic(q.clone(), ring_dropped.clone())?;
    let mut src = q.clone();
    tokio::select! {
        _ = run(&addr, &mut src, speaker, threshold, shutdown.clone()) => {}
        _ = deadline => tracing::warn!("forced exit after 2 s"),
    }
    tracing::info!(
        frames_dropped = q.dropped(),
        ring_samples_dropped = ring_dropped.load(std::sync::atomic::Ordering::Relaxed),
        "shutdown"
    );
    Ok(())
}

/// Chooses the speaker from `TTS`: Kokoro when its files exist (the default), else `say`.
async fn pick_speaker() -> anyhow::Result<(AnySpeaker, Option<cpal::Stream>)> {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.into());
    let say = || AnySpeaker::Say(SaySpeaker::new(std::env::var("SAY_VOICE").ok()));
    let want = parse_tts(std::env::var("TTS").ok().as_deref())?;
    if want == Some(Tts::Say) {
        return Ok((say(), None));
    }
    let model = env("KOKORO_MODEL", "models/kokoro-v1.0.int8.onnx");
    let voices = env("KOKORO_VOICES", "models/voices");
    if let Some(missing) = [&model, &voices]
        .into_iter()
        .find(|p| !std::path::Path::new(p.as_str()).exists())
    {
        anyhow::ensure!(
            want.is_none(),
            "TTS=kokoro but {missing} is missing (see AGENTS.md for the download commands)"
        );
        tracing::warn!("{missing} not found; using say instead of Kokoro");
        return Ok((say(), None));
    }
    let speed = match std::env::var("KOKORO_SPEED") {
        Ok(v) => v
            .parse()
            .with_context(|| format!("KOKORO_SPEED={v} is not a number"))?,
        Err(_) => 1.0,
    };
    let synth = KokoroSynth::new(&model, &voices, &env("KOKORO_VOICE", "af_heart"), speed).await?;
    let (speaker, stream) = KokoroSpeaker::start(synth)?;
    speaker.prewarm().await?;
    Ok((AnySpeaker::Kokoro(speaker), Some(stream)))
}
