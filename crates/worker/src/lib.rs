//! Worker: turns audio into transcript + streamed tokens, with per-turn cancellation.
use std::{pin::Pin, sync::Arc, time::Instant};

use async_trait::async_trait;
use duplex_proto::{
    client_msg, server_msg, voice_server::Voice, ClientMsg, HealthReq, HealthResp, ServerMsg,
    Token, TurnDone, Transcript,
};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status, Streaming};

#[async_trait]
pub trait Stt: Send + Sync + 'static {
    async fn transcribe(&self, pcm: Vec<u8>, cancel: CancellationToken) -> Option<(String, String)>;
}

/// Streams tokens into `tx`; must return promptly once `cancel` fires.
#[async_trait]
pub trait Llm: Send + Sync + 'static {
    async fn stream(&self, prompt: Vec<String>, tx: mpsc::Sender<String>, cancel: CancellationToken);
}

pub struct Service {
    pub stt: Arc<dyn Stt>,
    pub llm: Arc<dyn Llm>,
}

type Out = Pin<Box<dyn Stream<Item = Result<ServerMsg, Status>> + Send>>;

fn msg(k: server_msg::Kind) -> Result<ServerMsg, Status> {
    Ok(ServerMsg { kind: Some(k) })
}

#[tonic::async_trait]
impl Voice for Service {
    type ConverseStream = Out;

    async fn health(&self, _: Request<HealthReq>) -> Result<Response<HealthResp>, Status> {
        Ok(Response::new(HealthResp { model_loaded: true, queue_depth: 0, active_sessions: 0 }))
    }

    async fn converse(&self, req: Request<Streaming<ClientMsg>>) -> Result<Response<Out>, Status> {
        let mut inbound = req.into_inner();
        let (tx, rx) = mpsc::channel::<Result<ServerMsg, Status>>(64);
        let (stt, llm) = (self.stt.clone(), self.llm.clone());
        let session = CancellationToken::new();
        tokio::spawn(async move {
            let mut pcm: Vec<u8> = Vec::new();
            let mut current: Option<(u64, CancellationToken)> = None;
            let mut history: Vec<String> = Vec::new();
            while let Some(Ok(m)) = inbound.next().await {
                match m.kind {
                    Some(client_msg::Kind::Audio(a)) => {
                        // A higher turn_id implies cancel of the old turn, even if Cancel was lost.
                        if let Some((id, t)) = &current {
                            if a.turn_id > *id {
                                t.cancel();
                                current = None;
                                pcm.clear();
                            }
                        }
                        pcm.extend_from_slice(&a.pcm);
                    }
                    Some(client_msg::Kind::Cancel(c)) => {
                        if let Some((id, t)) = &current {
                            if *id == c.turn_id {
                                t.cancel();
                            }
                        }
                        pcm.clear();
                        // Ack comes from the turn task; ack here too if no turn is running.
                        if current.as_ref().map(|(id, _)| *id) != Some(c.turn_id) {
                            let _ = tx.send(msg(server_msg::Kind::Done(TurnDone {
                                turn_id: c.turn_id, cancelled: true, ..Default::default()
                            }))).await;
                        }
                    }
                    Some(client_msg::Kind::Eou(e)) => {
                        let token = session.child_token();
                        current = Some((e.turn_id, token.clone()));
                        let (stt, llm, tx) = (stt.clone(), llm.clone(), tx.clone());
                        let audio = std::mem::take(&mut pcm);
                        let hist = history.clone();
                        history.push(String::new());
                        tokio::spawn(run_turn(e.turn_id, audio, hist, stt, llm, tx, token));
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
) {
    let t0 = Instant::now();
    let done = |cancelled, stt_ms, ttft_ms| {
        msg(server_msg::Kind::Done(TurnDone { turn_id, stt_ms, ttft_ms, cancelled }))
    };
    let Some((text, lang)) = stt.transcribe(audio, cancel.clone()).await else {
        let _ = tx.send(done(true, 0, 0)).await;
        return;
    };
    let stt_ms = t0.elapsed().as_millis() as u32;
    if cancel.is_cancelled() {
        let _ = tx.send(done(true, stt_ms, 0)).await;
        return;
    }
    let _ = tx.send(msg(server_msg::Kind::Transcript(Transcript { turn_id, text: text.clone(), lang }))).await;
    history.push(text);
    let (ttx, mut trx) = mpsc::channel::<String>(256);
    let t1 = Instant::now();
    let gen = tokio::spawn({
        let (llm, cancel) = (llm.clone(), cancel.clone());
        async move { llm.stream(history, ttx, cancel).await }
    });
    let mut ttft_ms = 0;
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            t = trx.recv() => match t {
                Some(text) => {
                    if ttft_ms == 0 { ttft_ms = t1.elapsed().as_millis() as u32; }
                    if tx.send(msg(server_msg::Kind::Token(Token { turn_id, text }))).await.is_err() { break; }
                }
                None => break,
            }
        }
    }
    let _ = gen.await;
    let _ = tx.send(done(cancel.is_cancelled(), stt_ms, ttft_ms)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use duplex_proto::{voice_client::VoiceClient, voice_server::VoiceServer, AudioChunk, Cancel, EndOfUtterance};
    use std::time::Duration;
    use tokio_stream::wrappers::TcpListenerStream;

    struct MockStt;
    #[async_trait]
    impl Stt for MockStt {
        async fn transcribe(&self, _: Vec<u8>, _: CancellationToken) -> Option<(String, String)> {
            Some(("hello".into(), "en".into()))
        }
    }
    /// Emits one token every 20 ms until cancelled.
    struct SlowLlm;
    #[async_trait]
    impl Llm for SlowLlm {
        async fn stream(&self, _: Vec<String>, tx: mpsc::Sender<String>, cancel: CancellationToken) {
            for i in 0..100 {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(Duration::from_millis(20)) => {}
                }
                if tx.send(format!("t{i} ")).await.is_err() { return; }
            }
        }
    }

    async fn start() -> VoiceClient<tonic::transport::Channel> {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let svc = Service { stt: Arc::new(MockStt), llm: Arc::new(SlowLlm) };
        tokio::spawn(tonic::transport::Server::builder()
            .add_service(VoiceServer::new(svc))
            .serve_with_incoming(TcpListenerStream::new(l)));
        VoiceClient::connect(format!("http://{addr}")).await.unwrap()
    }

    #[tokio::test]
    async fn health_roundtrip() {
        let mut c = start().await;
        assert!(c.health(HealthReq {}).await.unwrap().into_inner().model_loaded);
    }

    #[tokio::test]
    async fn cancel_mid_stream_stops_tokens_and_acks() {
        let mut c = start().await;
        let (tx, rx) = mpsc::channel(8);
        let mut out = c.converse(ReceiverStream::new(rx)).await.unwrap().into_inner();
        let send = |k| ClientMsg { kind: Some(k) };
        tx.send(send(client_msg::Kind::Audio(AudioChunk { turn_id: 1, seq: 0, pcm: vec![0; 640] }))).await.unwrap();
        tx.send(send(client_msg::Kind::Eou(EndOfUtterance { turn_id: 1 }))).await.unwrap();
        // wait for the first token, then cancel
        loop {
            if let Some(server_msg::Kind::Token(_)) = out.next().await.unwrap().unwrap().kind { break; }
        }
        tx.send(send(client_msg::Kind::Cancel(Cancel { turn_id: 1, reason: 0, spoken_text: "t0".into() }))).await.unwrap();
        let mut tokens_after_ack = 0;
        let mut acked = false;
        while let Ok(Some(Ok(m))) = tokio::time::timeout(Duration::from_millis(300), out.next()).await {
            match m.kind {
                Some(server_msg::Kind::Done(d)) => { assert!(d.cancelled); acked = true; }
                Some(server_msg::Kind::Token(_)) if acked => tokens_after_ack += 1,
                _ => {}
            }
        }
        assert!(acked, "no TurnDone{{cancelled}} ack");
        assert_eq!(tokens_after_ack, 0);
    }
}
