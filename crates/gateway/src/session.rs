//! Session loop: VAD-gated audio streaming, sentence TTS, barge-in, health-gated reconnect.
use std::collections::VecDeque;
use std::future::Future;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use duplex_proto::{
    client_msg, server_msg, voice_client::VoiceClient, AudioChunk, Cancel, CancelReason, ClientMsg,
    EndOfUtterance, HealthReq,
};
use tokio::sync::{mpsc, watch};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;
use tracing::Span;

use crate::audio::FrameSource;
use crate::{SentenceSplitter, Vad, VadEvent, HANGOVER_FRAMES, MIN_SPEECH_FRAMES};

/// Text-to-speech output, one sentence at a time.
pub trait Speaker: Send + Sync + 'static {
    /// Speaks `text`; returns when it finished or was killed.
    fn speak(&self, text: String) -> impl Future<Output = ()> + Send;
    /// Stops the sentence being spoken now.
    fn kill(&self);
}

/// macOS `say`, one child process per sentence.
pub struct SaySpeaker {
    voice: Option<String>,
    kills: watch::Sender<u64>,
}

impl SaySpeaker {
    pub fn new(voice: Option<String>) -> Self {
        Self {
            voice,
            kills: watch::Sender::new(0),
        }
    }
}

impl Speaker for SaySpeaker {
    async fn speak(&self, text: String) {
        // shortcut: a kill() that lands before this subscribe is missed; the player already skips stale turns, so at most one sentence leaks.
        let mut killed = self.kills.subscribe();
        let mut cmd = tokio::process::Command::new("say");
        if let Some(v) = &self.voice {
            cmd.args(["-v", v]);
        }
        // Leading space so text starting with '-' is not parsed as an option.
        let mut child = match cmd.arg(format!(" {text}")).kill_on_drop(true).spawn() {
            Ok(c) => c,
            Err(e) => return tracing::warn!("say: {e}"),
        };
        tokio::select! {
            biased;
            _ = killed.changed() => { let _ = child.kill().await; }
            r = child.wait() => if let Err(e) = r { tracing::warn!("say: {e}") },
        }
    }

    fn kill(&self) {
        self.kills.send_modify(|n| *n += 1);
    }
}

/// Exponential backoff for attempt `n` (0-based): 100 ms doubling, capped at 2 s.
pub fn backoff(n: u32) -> Duration {
    Duration::from_millis(100 << n.min(5)).min(Duration::from_secs(2))
}

async fn healthy_client(addr: &str) -> anyhow::Result<VoiceClient<Channel>> {
    let mut c = VoiceClient::connect(addr.to_string())
        .await
        .context("connect")?;
    let h = c.health(HealthReq {}).await.context("health")?.into_inner();
    anyhow::ensure!(h.model_loaded, "worker reports model_loaded=false");
    Ok(c)
}

/// Retries `Health` with backoff until the worker is reachable and loaded; `None` on shutdown.
pub async fn connect_healthy(
    addr: &str,
    shutdown: &CancellationToken,
) -> Option<VoiceClient<Channel>> {
    let mut attempt = 0;
    loop {
        let r = tokio::select! {
            biased;
            _ = shutdown.cancelled() => return None,
            r = healthy_client(addr) => r,
        };
        match r {
            Ok(c) => return Some(c),
            Err(e) => {
                let wait = backoff(attempt);
                tracing::warn!(
                    attempt,
                    addr,
                    retry_in_ms = wait.as_millis() as u64,
                    "worker not ready: {e:#}"
                );
                attempt += 1;
                tokio::select! {
                    biased;
                    _ = shutdown.cancelled() => return None,
                    _ = tokio::time::sleep(wait) => {}
                }
            }
        }
    }
}

/// Runs sessions against the worker at `addr`, reconnecting when the stream drops, until shutdown or the source ends.
pub async fn run<F: FrameSource, S: Speaker>(
    addr: &str,
    src: &mut F,
    speaker: Arc<S>,
    threshold: f32,
    shutdown: CancellationToken,
) {
    while let Some(client) = connect_healthy(addr, &shutdown).await {
        match run_session(client, src, speaker.clone(), threshold, &shutdown).await {
            Ok(()) => return,
            Err(e) => tracing::warn!("worker stream lost: {e:#}; reconnecting"),
        }
    }
}

enum Played {
    Started {
        turn: u64,
        text: String,
        at: Instant,
    },
    Finished {
        turn: u64,
    },
}

/// Speaks queued sentences one at a time, skipping any whose turn was barged in.
async fn player<S: Speaker>(
    speaker: Arc<S>,
    mut rx: mpsc::Receiver<(u64, String)>,
    current: Arc<AtomicU64>,
    ev: mpsc::Sender<Played>,
) {
    while let Some((turn, text)) = rx.recv().await {
        if turn != current.load(Ordering::Acquire) {
            continue;
        }
        if ev
            .send(Played::Started {
                turn,
                text: text.clone(),
                at: Instant::now(),
            })
            .await
            .is_err()
        {
            return;
        }
        speaker.speak(text).await;
        let _ = ev.send(Played::Finished { turn }).await;
    }
}

fn client(k: client_msg::Kind) -> ClientMsg {
    ClientMsg { kind: Some(k) }
}

fn audio(turn_id: u64, seq: u64, frame: &[i16]) -> ClientMsg {
    client(client_msg::Kind::Audio(AudioChunk {
        turn_id,
        seq,
        pcm: frame.iter().flat_map(|s| s.to_le_bytes()).collect(),
    }))
}

/// Logs `ms` inside a span opened at `t`.
fn close(span: Option<(Span, Instant)>) {
    if let Some((s, t)) = span {
        s.in_scope(|| tracing::info!(ms = t.elapsed().as_millis() as u64, "span closed"));
    }
}

/// Per-turn numbers for the latency line.
#[derive(Default)]
struct Stats {
    eou_at: Option<Instant>,
    ttfa_ms: Option<u128>,
    stt_ms: u32,
    ttft_ms: u32,
    done: bool,
    printed: bool,
}

impl Stats {
    fn line(&self, turn: u64) -> String {
        let ttfa = self.ttfa_ms.map_or("-".into(), |v| v.to_string());
        format!(
            "turn={turn} ttfa_ms={ttfa} stt_ms={} ttft_ms={}",
            self.stt_ms, self.ttft_ms
        )
    }
    fn maybe_print(&mut self, turn: u64) {
        if self.done && self.ttfa_ms.is_some() && !self.printed {
            self.printed = true;
            println!("{}", self.line(turn));
        }
    }
}

/// A barged-in turn waiting for its `TurnDone{cancelled}` ack.
struct Barged {
    turn: u64,
    span: (Span, Instant),
    barge_in_ms: u128,
    stats: Stats,
}

async fn run_session<F: FrameSource, S: Speaker>(
    mut grpc: VoiceClient<Channel>,
    src: &mut F,
    speaker: Arc<S>,
    threshold: f32,
    shutdown: &CancellationToken,
) -> anyhow::Result<()> {
    let (out, out_rx) = mpsc::channel::<ClientMsg>(64);
    let mut inbound = grpc
        .converse(ReceiverStream::new(out_rx))
        .await
        .context("open Converse")?
        .into_inner();
    let (say_tx, say_rx) = mpsc::channel::<(u64, String)>(3);
    let (ev_tx, mut ev_rx) = mpsc::channel::<Played>(16);
    let play_turn = Arc::new(AtomicU64::new(0));
    tokio::spawn(player(speaker.clone(), say_rx, play_turn.clone(), ev_tx));

    let mut vad = Vad::new(threshold, MIN_SPEECH_FRAMES, HANGOVER_FRAMES);
    let (mut turn, mut seq) = (0u64, 0u64);
    // EOU sent and TurnDone not yet received: tokens for `turn` may still arrive.
    let mut awaiting = false;
    let mut playing = 0usize;
    let mut spoken = String::new();
    let mut splitter = SentenceSplitter::default();
    let mut backlog: VecDeque<String> = VecDeque::new();
    let mut stats = Stats::default();
    let mut stale = 0u64;
    let mut vad_tail: Option<(Span, Instant)> = None;
    let mut first_sentence: Option<(Span, Instant)> = None;
    let mut got_token = false;
    let mut barged: Option<Barged> = None;

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                speaker.kill();
                if awaiting || playing > 0 {
                    let c = Cancel { turn_id: turn, reason: CancelReason::ClientClosed as i32, spoken_text: spoken };
                    let _ = out.send(client(client_msg::Kind::Cancel(c))).await;
                }
                // Close our side and let the worker drain so the Cancel is delivered before exit.
                drop(out);
                let _ = tokio::time::timeout(Duration::from_secs(1), async { while inbound.next().await.is_some() {} }).await;
                return Ok(());
            }
            m = inbound.next() => {
                let m = match m {
                    Some(Ok(m)) => m,
                    Some(Err(e)) => bail!("worker stream error: {e}"),
                    None => bail!("worker closed the stream"),
                };
                match m.kind {
                    Some(server_msg::Kind::Token(t)) => {
                        if t.turn_id != turn || !awaiting {
                            stale += 1;
                            tracing::debug!(turn_id = t.turn_id, current = turn, stale, "dropped stale token");
                            continue;
                        }
                        if !got_token {
                            got_token = true;
                            first_sentence = Some((tracing::info_span!("first_sentence", turn_id = turn), Instant::now()));
                        }
                        backlog.extend(splitter.push(&t.text));
                    }
                    Some(server_msg::Kind::Done(d)) => {
                        if d.cancelled && barged.as_ref().is_some_and(|b| b.turn == d.turn_id) {
                            if let Some(mut b) = barged.take() {
                                close(Some(b.span));
                                if b.stats.printed {
                                    println!("turn={} barge_in_ms={}", b.turn, b.barge_in_ms);
                                } else {
                                    (b.stats.stt_ms, b.stats.ttft_ms) = (d.stt_ms, d.ttft_ms);
                                    println!("{} barge_in_ms={}", b.stats.line(b.turn), b.barge_in_ms);
                                }
                            }
                        } else if d.turn_id == turn && awaiting {
                            awaiting = false;
                            if !d.cancelled {
                                backlog.extend(splitter.flush());
                                (stats.stt_ms, stats.ttft_ms, stats.done) = (d.stt_ms, d.ttft_ms, true);
                                if backlog.is_empty() && playing == 0 && stats.ttfa_ms.is_none() {
                                    // Nothing to say: print without TTFA.
                                    stats.printed = true;
                                    println!("{}", stats.line(turn));
                                }
                                stats.maybe_print(turn);
                            }
                        }
                    }
                    Some(server_msg::Kind::Transcript(t)) => tracing::info!(turn_id = t.turn_id, lang = %t.lang, text = %t.text, "transcript"),
                    Some(server_msg::Kind::Error(e)) => tracing::warn!(turn_id = e.turn_id, code = %e.code, "worker error: {}", e.message),
                    None => {}
                }
            }
            Some(ev) = ev_rx.recv() => match ev {
                Played::Started { turn: t, text, at } if t == turn => {
                    playing += 1;
                    if !spoken.is_empty() { spoken.push(' '); }
                    spoken.push_str(&text);
                    if stats.ttfa_ms.is_none() {
                        stats.ttfa_ms = stats.eou_at.map(|e| at.saturating_duration_since(e).as_millis());
                        stats.maybe_print(turn);
                    }
                }
                Played::Finished { turn: t } if t == turn => playing = playing.saturating_sub(1),
                _ => {}
            },
            Ok(permit) = say_tx.reserve(), if !backlog.is_empty() => {
                if let Some(s) = backlog.pop_front() {
                    close(first_sentence.take());
                    permit.send((turn, s));
                }
            }
            f = src.next() => {
                let Some(frame) = f else { return Ok(()) };
                let loud = vad.is_loud(&frame);
                match vad.push(&frame) {
                    Some(VadEvent::SpeechStart) => {
                        let t0 = Instant::now();
                        if awaiting || playing > 0 {
                            // Barge-in: silence playback, then cancel the old turn with what was actually handed to `say`.
                            play_turn.store(turn + 1, Ordering::Release);
                            speaker.kill();
                            let barge_in_ms = t0.elapsed().as_millis();
                            backlog.clear();
                            while let Ok(ev) = ev_rx.try_recv() {
                                if let Played::Started { turn: t, text, .. } = ev {
                                    if t == turn {
                                        if !spoken.is_empty() { spoken.push(' '); }
                                        spoken.push_str(&text);
                                    }
                                }
                            }
                            let span = tracing::info_span!("cancel", turn_id = turn);
                            let c = Cancel { turn_id: turn, reason: CancelReason::BargeIn as i32, spoken_text: std::mem::take(&mut spoken) };
                            out.send(client(client_msg::Kind::Cancel(c))).await.context("send Cancel")?;
                            barged = Some(Barged { turn, span: (span, t0), barge_in_ms, stats: std::mem::take(&mut stats) });
                        }
                        turn += 1;
                        seq = 0;
                        play_turn.store(turn, Ordering::Release);
                        (awaiting, playing, got_token) = (false, 0, false);
                        spoken.clear();
                        splitter = SentenceSplitter::default();
                        stats = Stats::default();
                        first_sentence = None;
                        for f in vad.take_pending() {
                            out.send(audio(turn, seq, &f)).await.context("send audio")?;
                            seq += 1;
                        }
                    }
                    Some(VadEvent::EndOfUtterance) => {
                        close(vad_tail.take());
                        out.send(client(client_msg::Kind::Eou(EndOfUtterance { turn_id: turn }))).await.context("send EOU")?;
                        awaiting = true;
                        stats.eou_at = Some(Instant::now());
                    }
                    None if vad.speaking() => {
                        if loud {
                            vad_tail = None;
                        } else if vad_tail.is_none() {
                            vad_tail = Some((tracing::info_span!("vad_tail", turn_id = turn), Instant::now()));
                        }
                        out.send(audio(turn, seq, &frame)).await.context("send audio")?;
                        seq += 1;
                    }
                    None => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_and_caps() {
        let ms: Vec<u128> = (0..8).map(|n| backoff(n).as_millis()).collect();
        assert_eq!(ms, [100, 200, 400, 800, 1600, 2000, 2000, 2000]);
    }
}
