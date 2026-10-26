//! Worker: turns audio into transcript + streamed tokens, with per-turn cancellation.
use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Instant,
};

use async_trait::async_trait;
use duplex_proto::{
    client_msg, server_msg, voice_server::Voice, ClientMsg, Error, HealthReq, HealthResp,
    ServerMsg, Token, Transcript, TurnDone,
};
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status, Streaming};
use tracing::Instrument;

pub mod ollama;
pub mod whisper;

/// `Ok(None)` means the transcription was aborted by `cancel`.
#[async_trait]
pub trait Stt: Send + Sync + 'static {
    async fn transcribe(
        &self,
        pcm: Vec<u8>,
        cancel: CancellationToken,
    ) -> anyhow::Result<Option<(String, String)>>;
}

/// Streams tokens into `tx`; must return promptly once `cancel` fires.
/// `prompt` alternates user/assistant, oldest first, ending with the current user message.
#[async_trait]
pub trait Llm: Send + Sync + 'static {
    async fn stream(
        &self,
        prompt: Vec<String>,
        tx: mpsc::Sender<String>,
        cancel: CancellationToken,
    ) -> anyhow::Result<()>;
}

pub struct Service {
    pub stt: Arc<dyn Stt>,
    pub llm: Arc<dyn Llm>,
    pub model_loaded: bool,
    /// Cancelling it ends every session (graceful shutdown).
    pub root: CancellationToken,
    sessions: Arc<AtomicU32>,
}

impl Service {
    pub fn new(
        stt: Arc<dyn Stt>,
        llm: Arc<dyn Llm>,
        model_loaded: bool,
        root: CancellationToken,
    ) -> Self {
        Self {
            stt,
            llm,
            model_loaded,
            root,
            sessions: Arc::default(),
        }
    }
}

/// ~60 s of 16 kHz mono i16; a longer utterance closes the stream.
const MAX_PCM_BYTES: usize = 16_000 * 2 * 60;
/// Last 6 user+assistant pairs.
const MAX_HISTORY: usize = 12;

type Out = Pin<Box<dyn Stream<Item = Result<ServerMsg, Status>> + Send>>;
/// (user transcript, assistant reply) of a turn that got past STT.
type TurnResult = Option<(String, String)>;

struct Turn {
    id: u64,
    token: CancellationToken,
    handle: JoinHandle<TurnResult>,
    /// `Cancel.spoken_text`, replaces the reply in history.
    spoken: Option<String>,
}

#[allow(clippy::result_large_err)] // tonic::Status is what the stream item carries
fn msg(k: server_msg::Kind) -> Result<ServerMsg, Status> {
    Ok(ServerMsg { kind: Some(k) })
}

/// Waits for a (cancelled or finished) turn and appends it to history.
async fn harvest(turn: Turn, history: &mut Vec<String>) {
    if let Ok(Some((user, reply))) = turn.handle.await {
        history.push(user);
        history.push(turn.spoken.unwrap_or(reply));
        let excess = history.len().saturating_sub(MAX_HISTORY);
        history.drain(..excess);
    }
}

struct SessionGuard(Arc<AtomicU32>);
impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

#[tonic::async_trait]
impl Voice for Service {
    type ConverseStream = Out;

    async fn health(&self, _: Request<HealthReq>) -> Result<Response<HealthResp>, Status> {
        Ok(Response::new(HealthResp {
            model_loaded: self.model_loaded,
            queue_depth: 0,
            active_sessions: self.sessions.load(Ordering::Relaxed),
        }))
    }

    async fn converse(&self, req: Request<Streaming<ClientMsg>>) -> Result<Response<Out>, Status> {
        let mut inbound = req.into_inner();
        let (tx, rx) = mpsc::channel::<Result<ServerMsg, Status>>(64);
        let (stt, llm) = (self.stt.clone(), self.llm.clone());
        let session = self.root.child_token();
        self.sessions.fetch_add(1, Ordering::Relaxed);
        let guard = SessionGuard(self.sessions.clone());
        tokio::spawn(async move {
            let _guard = guard;
            let mut pcm: Vec<u8> = Vec::new();
            let mut current: Option<Turn> = None;
            let mut history: Vec<String> = Vec::new();
            loop {
                let m = tokio::select! {
                    biased;
                    _ = session.cancelled() => break,
                    m = inbound.next() => match m { Some(Ok(m)) => m, _ => break },
                };
                match m.kind {
                    Some(client_msg::Kind::Audio(a)) => {
                        // A higher turn_id implies cancel of the old turn, even if Cancel was lost.
                        if let Some(t) = &current {
                            if a.turn_id > t.id && !t.token.is_cancelled() {
                                t.token.cancel();
                                pcm.clear();
                            }
                        }
                        if pcm.len() + a.pcm.len() > MAX_PCM_BYTES {
                            let _ = tx
                                .send(Err(Status::resource_exhausted("utterance too long")))
                                .await;
                            break;
                        }
                        pcm.extend_from_slice(&a.pcm);
                    }
                    Some(client_msg::Kind::Cancel(c)) => {
                        pcm.clear();
                        match &mut current {
                            Some(t) if t.id == c.turn_id => {
                                t.spoken = Some(c.spoken_text);
                                t.token.cancel();
                                // The turn already sent its TurnDone; ack so the gateway can close its cancel span.
                                if t.handle.is_finished() {
                                    let _ = tx
                                        .send(msg(server_msg::Kind::Done(TurnDone {
                                            turn_id: c.turn_id,
                                            cancelled: true,
                                            ..Default::default()
                                        })))
                                        .await;
                                }
                            }
                            // Ack comes from the turn task; ack here if no such turn is running.
                            _ => {
                                let _ = tx
                                    .send(msg(server_msg::Kind::Done(TurnDone {
                                        turn_id: c.turn_id,
                                        cancelled: true,
                                        ..Default::default()
                                    })))
                                    .await;
                            }
                        }
                    }
                    Some(client_msg::Kind::Eou(e)) => {
                        if let Some(old) = current.take() {
                            old.token.cancel();
                            harvest(old, &mut history).await;
                        }
                        let token = session.child_token();
                        let audio = std::mem::take(&mut pcm);
                        let handle = tokio::spawn(run_turn(
                            e.turn_id,
                            audio,
                            history.clone(),
                            stt.clone(),
                            llm.clone(),
                            tx.clone(),
                            token.clone(),
                        ));
                        current = Some(Turn {
                            id: e.turn_id,
                            token,
                            handle,
                            spoken: None,
                        });
                    }
                    _ => {}
                }
            }
            session.cancel();
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

async fn run_turn(
    turn_id: u64,
    audio: Vec<u8>,
    mut history: Vec<String>,
    stt: Arc<dyn Stt>,
    llm: Arc<dyn Llm>,
    tx: mpsc::Sender<Result<ServerMsg, Status>>,
    cancel: CancellationToken,
) -> TurnResult {
    let t0 = Instant::now();
    let done = |cancelled, stt_ms, ttft_ms| {
        msg(server_msg::Kind::Done(TurnDone {
            turn_id,
            stt_ms,
            ttft_ms,
            cancelled,
        }))
    };
    let error = |code: &str, e: anyhow::Error| {
        tracing::warn!(turn_id, code, "{e:#}");
        msg(server_msg::Kind::Error(Error {
            turn_id,
            code: code.into(),
            message: format!("{e:#}"),
        }))
    };
    let stt_span = tracing::info_span!("stt", turn_id);
    let (text, lang) = match stt
        .transcribe(audio, cancel.clone())
        .instrument(stt_span.clone())
        .await
    {
        Ok(Some(r)) => r,
        Ok(None) => {
            let _ = tx.send(done(true, 0, 0)).await;
            return None;
        }
        Err(e) => {
            let _ = tx.send(error("stt", e)).await;
            let _ = tx.send(done(false, 0, 0)).await;
            return None;
        }
    };
    let stt_ms = t0.elapsed().as_millis() as u32;
    if cancel.is_cancelled() {
        let _ = tx.send(done(true, stt_ms, 0)).await;
        return None;
    }
    let _ = tx
        .send(msg(server_msg::Kind::Transcript(Transcript {
            turn_id,
            text: text.clone(),
            lang,
        })))
        .instrument(stt_span)
        .await;
    history.push(text.clone());
    let (ttx, mut trx) = mpsc::channel::<String>(256);
    let t1 = Instant::now();
    let mut ttft_span = Some(tracing::info_span!("llm_ttft", turn_id));
    let gen = tokio::spawn({
        let (llm, cancel) = (llm.clone(), cancel.clone());
        async move { llm.stream(history, ttx, cancel).await }
    });
    let mut ttft_ms = 0;
    let mut reply = String::new();
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            t = trx.recv() => match t {
                Some(text) => {
                    if ttft_span.take().is_some() { ttft_ms = t1.elapsed().as_millis() as u32; }
                    reply.push_str(&text);
                    if tx.send(msg(server_msg::Kind::Token(Token { turn_id, text }))).await.is_err() { break; }
                }
                None => break,
            }
        }
    }
    let failed = match gen.await {
        Ok(Ok(())) => false,
        Ok(Err(e)) => {
            let _ = tx.send(error("llm", e)).await;
            true
        }
        Err(e) => {
            let _ = tx.send(error("llm", e.into())).await;
            true
        }
    };
    let _ = tx.send(done(cancel.is_cancelled(), stt_ms, ttft_ms)).await;
    (!failed).then_some((text, reply))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duplex_proto::{
        voice_client::VoiceClient, voice_server::VoiceServer, AudioChunk, Cancel, EndOfUtterance,
    };
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio_stream::wrappers::TcpListenerStream;

    struct MockStt;
    #[async_trait]
    impl Stt for MockStt {
        async fn transcribe(
            &self,
            _: Vec<u8>,
            _: CancellationToken,
        ) -> anyhow::Result<Option<(String, String)>> {
            Ok(Some(("hello".into(), "en".into())))
        }
    }
    /// Emits `n` tokens `t0 t1 ...`, one per `every`, until cancelled; records each prompt.
    /// Fails its first `fail` calls.
    struct MockLlm {
        n: usize,
        every: Duration,
        fail: AtomicU32,
        prompts: Mutex<Vec<Vec<String>>>,
    }
    impl MockLlm {
        fn new(n: usize, every_ms: u64) -> Arc<Self> {
            Arc::new(Self {
                n,
                every: Duration::from_millis(every_ms),
                fail: 0.into(),
                prompts: Mutex::default(),
            })
        }
    }
    #[async_trait]
    impl Llm for MockLlm {
        async fn stream(
            &self,
            prompt: Vec<String>,
            tx: mpsc::Sender<String>,
            cancel: CancellationToken,
        ) -> anyhow::Result<()> {
            self.prompts.lock().unwrap().push(prompt);
            if self.fail.load(Ordering::Relaxed) > 0 {
                self.fail.fetch_sub(1, Ordering::Relaxed);
                anyhow::bail!("ollama down");
            }
            for i in 0..self.n {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return Ok(()),
                    _ = tokio::time::sleep(self.every) => {}
                }
                if tx.send(format!("t{i}")).await.is_err() {
                    return Ok(());
                }
            }
            Ok(())
        }
    }

    async fn start_with(llm: Arc<dyn Llm>) -> VoiceClient<tonic::transport::Channel> {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let svc = Service::new(Arc::new(MockStt), llm, true, CancellationToken::new());
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(VoiceServer::new(svc))
                .serve_with_incoming(TcpListenerStream::new(l)),
        );
        VoiceClient::connect(format!("http://{addr}"))
            .await
            .unwrap()
    }

    async fn start() -> VoiceClient<tonic::transport::Channel> {
        start_with(MockLlm::new(100, 20)).await
    }

    type Conv = (mpsc::Sender<ClientMsg>, Streaming<ServerMsg>);

    async fn open(c: &mut VoiceClient<tonic::transport::Channel>) -> Conv {
        let (tx, rx) = mpsc::channel(8);
        let out = c
            .converse(ReceiverStream::new(rx))
            .await
            .unwrap()
            .into_inner();
        (tx, out)
    }

    fn send(k: client_msg::Kind) -> ClientMsg {
        ClientMsg { kind: Some(k) }
    }

    async fn utter(tx: &mpsc::Sender<ClientMsg>, turn_id: u64) {
        tx.send(send(client_msg::Kind::Audio(AudioChunk {
            turn_id,
            seq: 0,
            pcm: vec![0; 640],
        })))
        .await
        .unwrap();
        tx.send(send(client_msg::Kind::Eou(EndOfUtterance { turn_id })))
            .await
            .unwrap();
    }

    /// Reads until TurnDone for `turn_id`, returning every message kind seen.
    async fn until_done(out: &mut Streaming<ServerMsg>, turn_id: u64) -> Vec<server_msg::Kind> {
        let mut seen = Vec::new();
        loop {
            let k = out.next().await.unwrap().unwrap().kind.unwrap();
            let end = matches!(&k, server_msg::Kind::Done(d) if d.turn_id == turn_id);
            seen.push(k);
            if end {
                return seen;
            }
        }
    }

    #[tokio::test]
    async fn health_roundtrip() {
        let mut c = start().await;
        let h = c.health(HealthReq {}).await.unwrap().into_inner();
        assert!(h.model_loaded);
        assert_eq!(h.active_sessions, 0);
        let _conv = open(&mut c).await;
        assert_eq!(
            c.health(HealthReq {})
                .await
                .unwrap()
                .into_inner()
                .active_sessions,
            1
        );
    }

    #[tokio::test]
    async fn cancel_mid_stream_stops_tokens_and_acks() {
        let mut c = start().await;
        let (tx, mut out) = open(&mut c).await;
        utter(&tx, 1).await;
        // wait for the first token, then cancel
        loop {
            if let Some(server_msg::Kind::Token(_)) = out.next().await.unwrap().unwrap().kind {
                break;
            }
        }
        tx.send(send(client_msg::Kind::Cancel(Cancel {
            turn_id: 1,
            reason: 0,
            spoken_text: "t0".into(),
        })))
        .await
        .unwrap();
        let mut tokens_after_ack = 0;
        let mut acked = false;
        while let Ok(Some(Ok(m))) =
            tokio::time::timeout(Duration::from_millis(300), out.next()).await
        {
            match m.kind {
                Some(server_msg::Kind::Done(d)) => {
                    assert!(d.cancelled);
                    acked = true;
                }
                Some(server_msg::Kind::Token(_)) if acked => tokens_after_ack += 1,
                _ => {}
            }
        }
        assert!(acked, "no TurnDone{{cancelled}} ack");
        assert_eq!(tokens_after_ack, 0);
    }

    #[tokio::test]
    async fn cancelled_turn_stored_as_spoken_text() {
        let llm = MockLlm::new(3, 50);
        let mut c = start_with(llm.clone()).await;
        let (tx, mut out) = open(&mut c).await;
        utter(&tx, 1).await;
        loop {
            if let Some(server_msg::Kind::Token(_)) = out.next().await.unwrap().unwrap().kind {
                break;
            }
        }
        tx.send(send(client_msg::Kind::Cancel(Cancel {
            turn_id: 1,
            reason: 0,
            spoken_text: "t0".into(),
        })))
        .await
        .unwrap();
        until_done(&mut out, 1).await;
        utter(&tx, 2).await;
        until_done(&mut out, 2).await;
        assert_eq!(llm.prompts.lock().unwrap()[1], ["hello", "t0", "hello"]);
    }

    #[tokio::test]
    async fn history_is_capped_to_six_pairs() {
        let llm = MockLlm::new(1, 1);
        let mut c = start_with(llm.clone()).await;
        let (tx, mut out) = open(&mut c).await;
        for t in 1..=8 {
            utter(&tx, t).await;
            until_done(&mut out, t).await;
        }
        let prompts = llm.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 8);
        assert_eq!(prompts[1], ["hello", "t0", "hello"]);
        assert!(prompts[7].len() <= 13, "{}", prompts[7].len());
    }

    #[tokio::test]
    async fn llm_error_sends_error_then_done_and_session_survives() {
        let llm = MockLlm::new(2, 1);
        llm.fail.store(1, Ordering::Relaxed);
        let mut c = start_with(llm).await;
        let (tx, mut out) = open(&mut c).await;
        utter(&tx, 1).await;
        let seen = until_done(&mut out, 1).await;
        let n = seen.len();
        assert!(
            matches!(&seen[n - 2], server_msg::Kind::Error(e) if e.turn_id == 1 && e.code == "llm"),
            "{seen:?}"
        );
        assert!(matches!(&seen[n - 1], server_msg::Kind::Done(d) if !d.cancelled));
        utter(&tx, 2).await;
        let seen = until_done(&mut out, 2).await;
        assert!(
            seen.iter().any(|k| matches!(k, server_msg::Kind::Token(_))),
            "{seen:?}"
        );
        assert!(!seen.iter().any(|k| matches!(k, server_msg::Kind::Error(_))));
    }
}
