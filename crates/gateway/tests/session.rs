//! Session loop against an in-process mock worker and a mock speaker.
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use duplex_proto::{
    client_msg, server_msg,
    voice_server::{Voice, VoiceServer},
    ClientMsg, HealthReq, HealthResp, ServerMsg, Token, TurnDone,
};
use gateway::audio::{FakeMic, Frame, FRAME};
use gateway::session::{run, Speaker};
use gateway::DEFAULT_VAD_THRESHOLD;
use tokio::sync::{mpsc, Notify};
use tokio_stream::{wrappers::ReceiverStream, wrappers::TcpListenerStream, Stream, StreamExt};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status, Streaming};

/// Everything both mocks observed, in one totally ordered channel.
#[derive(Debug)]
enum Ev {
    Client(ClientMsg),
    LlmEnd,
    Spoke(String),
    Killed,
}

#[derive(Clone, Copy)]
enum Script {
    /// "Hello there. " now, "Bye." 300 ms later, then TurnDone.
    Quick,
    /// "One. " then nothing until cancelled.
    Slow,
}

type ServerTx = mpsc::Sender<Result<ServerMsg, Status>>;

struct Svc(Arc<MockWorker>);

impl std::ops::Deref for Svc {
    type Target = MockWorker;
    fn deref(&self) -> &MockWorker {
        &self.0
    }
}

struct MockWorker {
    healthy: bool,
    script: Script,
    ev: mpsc::Sender<Ev>,
    health_calls: Mutex<Vec<Instant>>,
    converse_calls: AtomicU32,
    server_tx: Arc<Mutex<Option<ServerTx>>>,
}

#[allow(clippy::result_large_err)] // tonic::Status is what the stream item carries
fn smsg(k: server_msg::Kind) -> Result<ServerMsg, Status> {
    Ok(ServerMsg { kind: Some(k) })
}

#[tonic::async_trait]
impl Voice for Svc {
    type ConverseStream = Pin<Box<dyn Stream<Item = Result<ServerMsg, Status>> + Send>>;

    async fn health(&self, _: Request<HealthReq>) -> Result<Response<HealthResp>, Status> {
        self.health_calls.lock().unwrap().push(Instant::now());
        Ok(Response::new(HealthResp {
            model_loaded: self.healthy,
            ..Default::default()
        }))
    }

    async fn converse(
        &self,
        req: Request<Streaming<ClientMsg>>,
    ) -> Result<Response<Self::ConverseStream>, Status> {
        self.converse_calls.fetch_add(1, Ordering::Relaxed);
        let mut inbound = req.into_inner();
        let (tx, rx) = mpsc::channel(64);
        *self.server_tx.lock().unwrap() = Some(tx.clone());
        let (ev, script) = (self.ev.clone(), self.script);
        tokio::spawn(async move {
            while let Some(Ok(m)) = inbound.next().await {
                let kind = m.kind.clone();
                let _ = ev.send(Ev::Client(m)).await;
                match kind {
                    Some(client_msg::Kind::Eou(e)) => {
                        let (tx, ev) = (tx.clone(), ev.clone());
                        let turn_id = e.turn_id;
                        tokio::spawn(async move {
                            let tok = |t: &str| {
                                smsg(server_msg::Kind::Token(Token {
                                    turn_id,
                                    text: t.into(),
                                }))
                            };
                            let _ = tx
                                .send(tok(if matches!(script, Script::Quick) {
                                    "Hello there. "
                                } else {
                                    "One. "
                                }))
                                .await;
                            if matches!(script, Script::Quick) {
                                tokio::time::sleep(Duration::from_millis(300)).await;
                                let _ = ev.send(Ev::LlmEnd).await;
                                let _ = tx.send(tok("Bye.")).await;
                                let _ = tx
                                    .send(smsg(server_msg::Kind::Done(TurnDone {
                                        turn_id,
                                        ..Default::default()
                                    })))
                                    .await;
                            }
                        });
                    }
                    Some(client_msg::Kind::Cancel(c)) => {
                        let done = TurnDone {
                            turn_id: c.turn_id,
                            cancelled: true,
                            ..Default::default()
                        };
                        let _ = tx.send(smsg(server_msg::Kind::Done(done))).await;
                    }
                    _ => {}
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

async fn worker(healthy: bool, script: Script, ev: mpsc::Sender<Ev>) -> (String, Arc<MockWorker>) {
    let w = Arc::new(MockWorker {
        healthy,
        script,
        ev,
        health_calls: Mutex::default(),
        converse_calls: 0.into(),
        server_tx: Arc::default(),
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = format!("http://{}", l.local_addr().unwrap());
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(VoiceServer::new(Svc(w.clone())))
            .serve_with_incoming(TcpListenerStream::new(l)),
    );
    (addr, w)
}

/// Records speak/kill; with `block`, speak waits until killed.
struct MockSpeaker {
    block: bool,
    ev: mpsc::Sender<Ev>,
    killed: Notify,
}

impl Speaker for MockSpeaker {
    async fn speak(&self, text: String) {
        let killed = self.killed.notified();
        tokio::pin!(killed);
        killed.as_mut().enable();
        let _ = self.ev.send(Ev::Spoke(text)).await;
        if self.block {
            killed.await;
        }
    }
    fn kill(&self) {
        let _ = self.ev.try_send(Ev::Killed);
        self.killed.notify_waiters();
    }
}

async fn next_ev(rx: &mut mpsc::Receiver<Ev>) -> Ev {
    tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("timed out waiting for event")
        .unwrap()
}

#[tokio::test]
async fn fake_mic_streams_while_speaking_and_speaks_first_sentence_early() {
    let (ev_tx, mut ev) = mpsc::channel(4096);
    let (addr, _w) = worker(true, Script::Quick, ev_tx.clone()).await;
    let speaker = Arc::new(MockSpeaker {
        block: false,
        ev: ev_tx,
        killed: Notify::new(),
    });
    let mut mic = FakeMic::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/en.wav"
    ))
    .unwrap();
    let shutdown = CancellationToken::new();
    let s = shutdown.clone();
    let task =
        tokio::spawn(async move { run(&addr, &mut mic, speaker, DEFAULT_VAD_THRESHOLD, s).await });

    let mut log = Vec::new();
    loop {
        let e = next_ev(&mut ev).await;
        let end = matches!(e, Ev::Spoke(ref t) if t == "Bye.");
        log.push(e);
        if end {
            break;
        }
    }
    shutdown.cancel();
    task.await.unwrap();

    let pos = |f: &dyn Fn(&Ev) -> bool| log.iter().position(f).unwrap();
    let first_audio = pos(&|e| {
        matches!(
            e,
            Ev::Client(ClientMsg {
                kind: Some(client_msg::Kind::Audio(_))
            })
        )
    });
    let eou = pos(&|e| {
        matches!(
            e,
            Ev::Client(ClientMsg {
                kind: Some(client_msg::Kind::Eou(_))
            })
        )
    });
    let first_spoken = pos(&|e| matches!(e, Ev::Spoke(t) if t == "Hello there."));
    let llm_end = pos(&|e| matches!(e, Ev::LlmEnd));
    assert!(first_audio < eou, "audio must stream before EOU");
    assert!(
        first_spoken < llm_end,
        "first sentence must be spoken before the LLM stream ends"
    );
    let seqs: Vec<u64> = log
        .iter()
        .filter_map(|e| match e {
            Ev::Client(ClientMsg {
                kind: Some(client_msg::Kind::Audio(a)),
            }) => Some(a.seq),
            _ => None,
        })
        .collect();
    assert!(seqs.len() > 50, "{} chunks", seqs.len());
    assert!(
        seqs.windows(2).all(|w| w[1] == w[0] + 1),
        "seq not monotonic: {seqs:?}"
    );
}

#[tokio::test]
async fn barge_in_kills_speaker_cancels_and_drops_stale_tokens() {
    let (ev_tx, mut ev) = mpsc::channel(4096);
    let (addr, w) = worker(true, Script::Slow, ev_tx.clone()).await;
    let speaker = Arc::new(MockSpeaker {
        block: true,
        ev: ev_tx,
        killed: Notify::new(),
    });
    let (frames, mut src) = mpsc::channel::<Frame>(256);
    let shutdown = CancellationToken::new();
    let s = shutdown.clone();
    tokio::spawn(async move { run(&addr, &mut src, speaker, DEFAULT_VAD_THRESHOLD, s).await });
    let (loud, quiet) = (vec![3000i16; FRAME], vec![0i16; FRAME]);

    for _ in 0..10 {
        frames.send(loud.clone()).await.unwrap();
    }
    for _ in 0..25 {
        frames.send(quiet.clone()).await.unwrap();
    }
    loop {
        if let Ev::Spoke(t) = next_ev(&mut ev).await {
            assert_eq!(t, "One.");
            break;
        }
    }
    for _ in 0..10 {
        frames.send(loud.clone()).await.unwrap();
    }

    let mut killed = false;
    let mut cancel = None;
    loop {
        match next_ev(&mut ev).await {
            Ev::Killed => killed = true,
            Ev::Client(ClientMsg {
                kind: Some(client_msg::Kind::Cancel(c)),
            }) => cancel = Some(c),
            Ev::Client(ClientMsg {
                kind: Some(client_msg::Kind::Audio(a)),
            }) if cancel.is_some() => {
                assert_eq!(a.turn_id, 2);
                break;
            }
            Ev::Client(ClientMsg {
                kind: Some(client_msg::Kind::Audio(a)),
            }) => panic!("audio for turn {} before Cancel", a.turn_id),
            _ => {}
        }
    }
    assert!(killed, "speaker was not killed");
    let c = cancel.unwrap();
    assert_eq!((c.turn_id, c.spoken_text.as_str()), (1, "One."));

    // A late token for the barged-in turn must never reach the speaker.
    let tx = w.server_tx.lock().unwrap().clone().unwrap();
    tx.send(smsg(server_msg::Kind::Token(Token {
        turn_id: 1,
        text: "Stale. ".into(),
    })))
    .await
    .unwrap();
    let late = tokio::time::timeout(Duration::from_millis(300), async {
        loop {
            if let Ev::Spoke(t) = next_ev(&mut ev).await {
                return t;
            }
        }
    })
    .await;
    assert!(late.is_err(), "stale token was spoken: {late:?}");
    shutdown.cancel();
}

#[tokio::test]
async fn unhealthy_worker_is_never_conversed_and_retries_back_off() {
    let (ev_tx, _ev) = mpsc::channel(64);
    let (addr, w) = worker(false, Script::Quick, ev_tx).await;
    let shutdown = CancellationToken::new();
    let (s, speaker) = (
        shutdown.clone(),
        Arc::new(MockSpeaker {
            block: false,
            ev: mpsc::channel(1).0,
            killed: Notify::new(),
        }),
    );
    let (_frames, mut src) = mpsc::channel::<Frame>(1);
    let task =
        tokio::spawn(async move { run(&addr, &mut src, speaker, DEFAULT_VAD_THRESHOLD, s).await });
    tokio::time::sleep(Duration::from_millis(1700)).await;
    assert!(!task.is_finished());
    shutdown.cancel();
    task.await.unwrap();

    assert_eq!(w.converse_calls.load(Ordering::Relaxed), 0);
    let calls = w.health_calls.lock().unwrap().clone();
    assert!(calls.len() >= 4, "{} health calls", calls.len());
    let gaps: Vec<Duration> = calls.windows(2).map(|p| p[1] - p[0]).collect();
    assert!(
        gaps.windows(2).all(|g| g[1] > g[0]),
        "gaps not increasing: {gaps:?}"
    );
}

#[tokio::test]
async fn missing_worker_keeps_retrying_until_shutdown() {
    let port = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let shutdown = CancellationToken::new();
    let (s, speaker) = (
        shutdown.clone(),
        Arc::new(MockSpeaker {
            block: false,
            ev: mpsc::channel(1).0,
            killed: Notify::new(),
        }),
    );
    let (_frames, mut src) = mpsc::channel::<Frame>(1);
    let task = tokio::spawn(async move {
        run(
            &format!("http://127.0.0.1:{port}"),
            &mut src,
            speaker,
            DEFAULT_VAD_THRESHOLD,
            s,
        )
        .await
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(!task.is_finished());
    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap();
}
