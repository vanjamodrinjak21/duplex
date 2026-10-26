use std::{sync::Arc, time::Duration};

use anyhow::Context;
use duplex_proto::voice_server::VoiceServer;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};
use worker::{ollama::OllamaLlm, whisper::WhisperStt, Service, Stt};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,whisper_rs=warn")),
        )
        .with_span_events(FmtSpan::CLOSE)
        .init();
    let addr = env_or("WORKER_ADDR", "127.0.0.1:50051")
        .parse()
        .context("WORKER_ADDR is not a host:port")?;
    let model = std::env::var("WHISPER_MODEL")
        .map_err(|_| anyhow::anyhow!("WHISPER_MODEL is not set (path to ggml-small.bin)"))?;
    // whisper.cpp/ggml would otherwise print straight to stderr.
    whisper_rs::install_logging_hooks();
    let stt = tokio::task::spawn_blocking(move || WhisperStt::load(&model)).await??;
    let llm = OllamaLlm::new(
        env_or("OLLAMA_URL", "http://localhost:11434"),
        env_or("OLLAMA_MODEL", "qwen2.5:3b"),
    );

    // Prewarm both models so the first turn does not pay load time.
    let whisper_ok = match stt
        .transcribe(vec![0; 32_000], CancellationToken::new())
        .await
    {
        Ok(_) => true,
        Err(e) => {
            tracing::error!("whisper prewarm failed: {e:#}");
            false
        }
    };
    let ollama_ok = match llm.prewarm().await {
        Ok(()) => true,
        Err(e) => {
            tracing::error!(url = %llm.url, "ollama prewarm failed, reporting unhealthy: {e:#}");
            false
        }
    };

    let root = CancellationToken::new();
    let stop = CancellationToken::new();
    let svc = Service::new(
        Arc::new(stt),
        Arc::new(llm),
        whisper_ok && ollama_ok,
        root.clone(),
    );
    let mut server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(VoiceServer::new(svc))
            .serve_with_shutdown(addr, stop.clone().cancelled_owned()),
    );
    tracing::info!(%addr, model_loaded = whisper_ok && ollama_ok, "worker listening");
    tokio::select! {
        r = tokio::signal::ctrl_c() => r.context("SIGINT handler")?,
        r = &mut server => return Ok(r??),
    }
    tracing::info!("SIGINT: stopping");
    stop.cancel();
    root.cancel();
    if tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .is_err()
    {
        tracing::warn!("sessions did not close within 2 s");
    }
    Ok(())
}
