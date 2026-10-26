use std::sync::{atomic::AtomicU64, Arc};
use std::time::Duration;

use anyhow::Context;
use gateway::audio::{start_mic, FakeMic};
use gateway::session::{run, SaySpeaker};
#[cfg(feature = "anytts")]
use gateway::tts::{parse_anytts_device, parse_anytts_gain, parse_voice, AnyTtsSynth};
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
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,ort=warn,any_tts=error")),
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
        tracing::info!("tts=say");
        return Ok((say(), None));
    }
    if want == Some(Tts::AnyTts) {
        return anytts_speaker(&env("KOKORO_VOICE", "af_heart"), kokoro_speed()?).await;
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
        tracing::warn!("{missing} not found; using say instead of Kokoro (see AGENTS.md; paths are relative to the cwd)");
        tracing::info!("tts=say");
        return Ok((say(), None));
    }
    let speed = kokoro_speed()?;
    let voice = env("KOKORO_VOICE", "af_heart");
    let synth = KokoroSynth::new(&model, &voices, &voice, speed).await?;
    let (speaker, stream) = KokoroSpeaker::start(synth)?;
    speaker.prewarm().await?;
    tracing::info!(%voice, speed, "tts=kokoro");
    Ok((AnySpeaker::Kokoro(speaker), Some(stream)))
}

/// `KOKORO_SPEED` [1.0], shared by both Kokoro engines.
fn kokoro_speed() -> anyhow::Result<f32> {
    match std::env::var("KOKORO_SPEED") {
        Ok(v) => match v.parse::<f32>() {
            Ok(x) if x.is_finite() && x > 0.0 => Ok(x),
            _ => anyhow::bail!("KOKORO_SPEED={v} must be a number above 0"),
        },
        Err(_) => Ok(1.0),
    }
}

#[cfg(not(feature = "anytts"))]
async fn anytts_speaker(_: &str, _: f32) -> anyhow::Result<(AnySpeaker, Option<cpal::Stream>)> {
    anyhow::bail!(
        "TTS=anytts needs the optional backend: rebuild with --features anytts (see AGENTS.md)"
    )
}

/// `TTS=anytts`: candle Kokoro from `ANYTTS_MODEL`; logs load and prewarm time.
#[cfg(feature = "anytts")]
async fn anytts_speaker(
    voice: &str,
    speed: f32,
) -> anyhow::Result<(AnySpeaker, Option<cpal::Stream>)> {
    let model = std::env::var("ANYTTS_MODEL").unwrap_or_else(|_| "models/Kokoro-82M".into());
    let device = parse_anytts_device(std::env::var("ANYTTS_DEVICE").ok().as_deref())?;
    let gain = parse_anytts_gain(std::env::var("ANYTTS_GAIN").ok().as_deref())?;
    let voice = parse_voice(voice)?;
    anyhow::ensure!(
        std::path::Path::new(&model).is_dir(),
        "TTS=anytts but ANYTTS_MODEL={model} is not a directory (see AGENTS.md for the download commands)"
    );
    tracing::info!(
        model,
        device = device.as_str(),
        "loading Kokoro-82M (anytts)"
    );
    let t = std::time::Instant::now();
    let v = voice.to_string();
    let synth =
        tokio::task::spawn_blocking(move || AnyTtsSynth::new(&model, device, &v, speed, gain))
            .await??;
    tracing::info!(ms = t.elapsed().as_millis() as u64, "anytts load");
    let (speaker, stream) = KokoroSpeaker::start(synth)?;
    tracing::info!("warming up the model; the gateway is not ready yet");
    speaker.prewarm().await?;
    tracing::info!(
        device = device.as_str(),
        voice,
        speed,
        gain = format_args!("{gain:.2}"),
        "tts=anytts"
    );
    Ok((AnySpeaker::AnyTts(speaker), Some(stream)))
}
