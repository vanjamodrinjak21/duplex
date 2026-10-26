//! Kokoro neural TTS: a blocking `Synth`, and a `Speaker` that plays it through a cpal output ring.
use std::future::Future;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtrb::{Consumer, Producer, RingBuffer};
use rubato::{FftFixedIn, Resampler};

use crate::session::{close, SaySpeaker, Speaker};

/// Seconds of device-rate audio the output ring holds.
const RING_SECONDS: usize = 10;
/// How often waiting tasks re-check the generation / drain counters.
const POLL: Duration = Duration::from_millis(2);
/// A few voices named in the invalid-voice error.
const VOICE_HINT: &str = "af_heart, af_bella, am_adam, bf_emma";

/// Blocking, CPU-bound text-to-speech engine.
pub trait Synth: Send + Sync + 'static {
    const SAMPLE_RATE: u32;
    /// Mono f32 samples at `SAMPLE_RATE`.
    fn synth(&self, text: &str) -> anyhow::Result<Vec<f32>>;
}

/// Kokoro v1.0 through the `kokoro-en` crate.
pub struct KokoroSynth {
    tts: kokoro_en::KokoroTts,
    voice: kokoro_en::Voice,
    rt: tokio::runtime::Handle,
}

impl KokoroSynth {
    pub async fn new(model: &str, voices: &str, voice: &str, speed: f32) -> anyhow::Result<Self> {
        let tts = kokoro_en::KokoroTts::new(model, voices)
            .await
            .with_context(|| format!("load Kokoro model {model} / voices {voices}"))?;
        Ok(Self {
            tts,
            voice: kokoro_en::Voice::new(voice).with_speed(speed),
            rt: tokio::runtime::Handle::current(),
        })
    }
}

impl Synth for KokoroSynth {
    const SAMPLE_RATE: u32 = 24_000;

    /// Must run off the async runtime threads (it blocks on the engine's future).
    fn synth(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        match self.rt.block_on(self.tts.synth(text, self.voice.clone())) {
            Ok((samples, _)) => Ok(samples),
            Err(kokoro_en::KokoroError::VoiceNotFound(v)) => {
                bail!("KOKORO_VOICE={v} is not in the voices file (try {VOICE_HINT})")
            }
            Err(e) => Err(e.into()),
        }
    }
}

/// Resamples mono `x` from `from` to `to` Hz; the output has exactly `x.len() * to / from` samples.
fn resample(x: &[f32], from: u32, to: u32) -> anyhow::Result<Vec<f32>> {
    if from == to {
        return Ok(x.to_vec());
    }
    let want = (x.len() as u64 * to as u64 / from as u64) as usize;
    let mut rs = FftFixedIn::<f32>::new(from as usize, to as usize, 1024, 2, 1)
        .context("output resampler")?;
    let delay = rs.output_delay();
    let mut out = Vec::with_capacity(want + delay + 2048);
    let mut i = 0;
    while x.len() - i >= rs.input_frames_next() {
        let n = rs.input_frames_next();
        out.extend_from_slice(&rs.process(&[&x[i..i + n]], None)?[0]);
        i += n;
    }
    out.extend_from_slice(&rs.process_partial(Some(&[&x[i..]]), None)?[0]);
    while out.len() < want + delay {
        out.extend_from_slice(&rs.process_partial::<&[f32]>(None, None)?[0]);
    }
    Ok(out[delay..delay + want].to_vec())
}

/// State shared with the output callback.
struct Shared {
    /// Bumped by `kill()`; the callback drops everything queued when it changes.
    gen: AtomicU64,
    /// Samples popped or discarded by the callback.
    consumed: AtomicU64,
}

/// Output callback body. REAL-TIME: no allocation, locks or logging; only ring pops and atomics.
fn fill(cons: &mut Consumer<f32>, shared: &Shared, seen_gen: &mut u64, out: &mut [f32]) {
    let g = shared.gen.load(Ordering::Acquire);
    if g != *seen_gen {
        *seen_gen = g;
        let n = cons.slots();
        if let Ok(chunk) = cons.read_chunk(n) {
            chunk.commit_all();
        }
        shared.consumed.fetch_add(n as u64, Ordering::Relaxed);
        out.fill(0.0);
        return;
    }
    let n = out.len().min(cons.slots());
    if let Ok(chunk) = cons.read_chunk(n) {
        let (a, b) = chunk.as_slices();
        out[..a.len()].copy_from_slice(a);
        out[a.len()..n].copy_from_slice(b);
        chunk.commit_all();
    }
    out[n..].fill(0.0);
    shared.consumed.fetch_add(n as u64, Ordering::Relaxed);
}

/// Advances `serving` past `ticket` on every exit path, so a killed sentence never blocks later ones.
struct Turn<'a> {
    serving: &'a AtomicU64,
    ticket: u64,
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.serving.fetch_max(self.ticket + 1, Ordering::AcqRel);
    }
}

/// Speaks through a cpal output stream fed by a preallocated ring.
pub struct KokoroSpeaker<Y: Synth> {
    synth: Arc<Y>,
    rate: u32,
    channels: usize,
    prod: Mutex<Producer<f32>>,
    shared: Arc<Shared>,
    /// Samples pushed so far (one pusher at a time, ordered by ticket).
    pushed: AtomicU64,
    /// Next ticket to hand out; tickets are taken in `speak()` call order.
    next: AtomicU64,
    /// Ticket allowed to push now.
    serving: AtomicU64,
}

impl<Y: Synth> KokoroSpeaker<Y> {
    /// A speaker plus the ring consumer the output callback must drain; opens no audio device.
    fn with_ring(synth: Y, rate: u32, channels: usize) -> (Self, Consumer<f32>) {
        let (prod, cons) = RingBuffer::new(rate as usize * channels * RING_SECONDS);
        let s = Self {
            synth: Arc::new(synth),
            rate,
            channels,
            prod: Mutex::new(prod),
            shared: Arc::new(Shared {
                gen: 0.into(),
                consumed: 0.into(),
            }),
            pushed: 0.into(),
            next: 0.into(),
            serving: 0.into(),
        };
        (s, cons)
    }

    /// Opens the default output device. Keep the returned stream alive.
    pub fn start(synth: Y) -> anyhow::Result<(Self, cpal::Stream)> {
        let device = cpal::default_host()
            .default_output_device()
            .context("no default output device")?;
        let supported = device.default_output_config().context("output config")?;
        if supported.sample_format() != cpal::SampleFormat::F32 {
            bail!(
                "output sample format {:?} unsupported (need f32)",
                supported.sample_format()
            );
        }
        let config = supported.config();
        let (speaker, mut cons) =
            Self::with_ring(synth, config.sample_rate, config.channels as usize);
        let shared = speaker.shared.clone();
        let mut seen = 0;
        let stream = device
            .build_output_stream::<f32, _, _>(
                config,
                move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    fill(&mut cons, &shared, &mut seen, out)
                },
                |e| tracing::warn!("speaker: {e}"),
                None,
            )
            .context("open output stream")?;
        stream.play().context("start output")?;
        Ok((speaker, stream))
    }

    /// Synthesizes once so the first real sentence is not slow; logs the duration.
    pub async fn prewarm(&self) -> anyhow::Result<()> {
        let t = Instant::now();
        let synth = self.synth.clone();
        tokio::task::spawn_blocking(move || synth.synth("Hello."))
            .await
            .context("prewarm task")??;
        tracing::info!(ms = t.elapsed().as_millis() as u64, "kokoro prewarm");
        Ok(())
    }

    /// Waits until `cond` holds (true) or `kill()` ran since `gen` (false).
    async fn until(&self, gen: u64, cond: impl Fn() -> bool) -> bool {
        loop {
            if self.shared.gen.load(Ordering::Acquire) != gen {
                return false;
            }
            if cond() {
                return true;
            }
            tokio::time::sleep(POLL).await;
        }
    }

    /// Pushes interleaved `samples` into the ring, waiting for room; false if killed meanwhile.
    async fn push(
        &self,
        gen: u64,
        mut samples: &[f32],
        span: &mut Option<(tracing::Span, Instant)>,
    ) -> bool {
        while !samples.is_empty() {
            if self.shared.gen.load(Ordering::Acquire) != gen {
                return false;
            }
            let n = {
                let mut prod = self.prod.lock().unwrap();
                let n = samples.len().min(prod.slots()) / self.channels * self.channels;
                if let Ok(chunk) = prod.write_chunk_uninit(n) {
                    chunk.fill_from_iter(samples[..n].iter().copied());
                }
                n
            };
            self.pushed.fetch_add(n as u64, Ordering::AcqRel);
            if n > 0 {
                close(span.take());
            }
            samples = &samples[n..];
            if !samples.is_empty() {
                tokio::time::sleep(POLL).await;
            }
        }
        true
    }
}

impl<Y: Synth> Speaker for KokoroSpeaker<Y> {
    fn lookahead(&self) -> bool {
        true
    }

    fn speak(&self, text: String) -> impl Future<Output = ()> + Send {
        // Taken at call time so sentences push in the order they were handed over.
        let mut span = Some((tracing::info_span!("tts_first_audio"), Instant::now()));
        let gen = self.shared.gen.load(Ordering::Acquire);
        let ticket = self.next.fetch_add(1, Ordering::AcqRel);
        let turn = Turn {
            serving: &self.serving,
            ticket,
        };
        let (synth, rate, channels) = (self.synth.clone(), self.rate, self.channels);
        async move {
            let job = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
                let mono = resample(&synth.synth(&text)?, Y::SAMPLE_RATE, rate)?;
                Ok(mono
                    .iter()
                    .flat_map(|&s| std::iter::repeat_n(s, channels))
                    .collect())
            });
            let samples = tokio::select! {
                biased;
                false = self.until(gen, || false) => return,
                r = job => r,
            };
            let samples = match samples {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => return tracing::warn!("tts: {e:#}"),
                Err(e) => return tracing::warn!("tts task: {e}"),
            };
            if !self
                .until(gen, || self.serving.load(Ordering::Acquire) >= ticket)
                .await
                || !self.push(gen, &samples, &mut span).await
            {
                return;
            }
            drop(turn);
            let target = self.pushed.load(Ordering::Acquire);
            self.until(gen, || {
                self.shared.consumed.load(Ordering::Relaxed) >= target
            })
            .await;
        }
    }

    fn kill(&self) {
        self.shared.gen.fetch_add(1, Ordering::AcqRel);
        // Every ticket handed out so far is void.
        self.serving
            .fetch_max(self.next.load(Ordering::Acquire), Ordering::AcqRel);
    }
}

/// Which engine `TTS` selected.
#[derive(Debug, PartialEq)]
pub enum Tts {
    Kokoro,
    Say,
}

/// Parses the `TTS` env value; `None` when unset.
pub fn parse_tts(v: Option<&str>) -> anyhow::Result<Option<Tts>> {
    match v {
        None => Ok(None),
        Some("kokoro") => Ok(Some(Tts::Kokoro)),
        Some("say") => Ok(Some(Tts::Say)),
        Some(o) => bail!("TTS={o} is not valid (use kokoro or say)"),
    }
}

/// The speaker `main` picked, delegating to the concrete one.
pub enum AnySpeaker {
    Say(SaySpeaker),
    Kokoro(KokoroSpeaker<KokoroSynth>),
}

impl Speaker for AnySpeaker {
    fn lookahead(&self) -> bool {
        matches!(self, Self::Kokoro(_))
    }

    fn speak(&self, text: String) -> impl Future<Output = ()> + Send {
        // Boxed so both arms share one type; one allocation per sentence.
        let f: std::pin::Pin<Box<dyn Future<Output = ()> + Send + '_>> = match self {
            Self::Say(s) => Box::pin(s.speak(text)),
            Self::Kokoro(k) => Box::pin(k.speak(text)),
        };
        f
    }

    fn kill(&self) {
        match self {
            Self::Say(s) => s.kill(),
            Self::Kokoro(k) => k.kill(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock engine: `len` samples of a constant per call, after `delay`; records when each call started.
    struct Mock {
        len: usize,
        delay: Duration,
        calls: Mutex<Vec<(String, Instant)>>,
    }

    impl Mock {
        fn new(len: usize, delay_ms: u64) -> Self {
            Self {
                len,
                delay: Duration::from_millis(delay_ms),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl Synth for Arc<Mock> {
        const SAMPLE_RATE: u32 = 24_000;
        fn synth(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            self.calls
                .lock()
                .unwrap()
                .push((text.into(), Instant::now()));
            std::thread::sleep(self.delay);
            // First character as the value, so order is visible in the ring.
            Ok(vec![text.bytes().next().unwrap() as f32 / 255.0; self.len])
        }
    }

    fn drain(cons: &mut Consumer<f32>) -> Vec<f32> {
        let n = cons.slots();
        let chunk = cons.read_chunk(n).unwrap();
        let (a, b) = chunk.as_slices();
        let v = [a, b].concat();
        chunk.commit_all();
        v
    }

    #[tokio::test]
    async fn speak_pushes_exact_resampled_count_in_order() {
        let (sp, mut cons) = KokoroSpeaker::with_ring(Arc::new(Mock::new(2400, 0)), 48_000, 2);
        let check = async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            // 2400 samples at 24 kHz -> 4800 at 48 kHz -> 9600 interleaved stereo, per sentence.
            let got = drain(&mut cons);
            assert_eq!(got.len(), 2 * 9600);
            assert!((got[4000] - 97.0 / 255.0).abs() < 0.05, "first is a");
            assert!(
                (got[9600 + 4000] - 98.0 / 255.0).abs() < 0.05,
                "second is b"
            );
            assert_eq!(got[4000], got[4001], "channels duplicated");
            sp.kill();
        };
        tokio::join!(sp.speak("a".into()), sp.speak("b".into()), check);
    }

    #[tokio::test]
    async fn next_sentence_synthesizes_while_previous_is_unconsumed() {
        let mock = Arc::new(Mock::new(2400, 200));
        let (sp, cons) = KokoroSpeaker::with_ring(mock.clone(), 24_000, 1);
        let check = async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            // `a` is pushed but nobody consumes it; `b` must already be synthesizing.
            assert!(cons.slots() >= 2400);
            let calls = mock.calls.lock().unwrap();
            assert_eq!(calls.len(), 2);
            assert!(calls[1].1.duration_since(calls[0].1) < Duration::from_millis(100));
            drop(calls);
            sp.kill();
        };
        tokio::join!(sp.speak("a".into()), sp.speak("b".into()), check);
    }

    #[test]
    fn kill_silences_within_one_callback() {
        let (sp, mut cons) = KokoroSpeaker::with_ring(Arc::new(Mock::new(0, 0)), 24_000, 1);
        let mut prod = sp.prod.lock().unwrap();
        let chunk = prod.write_chunk_uninit(24_000 * 5).unwrap();
        chunk.fill_from_iter(std::iter::repeat(0.5));
        drop(prod);
        let t = Instant::now();
        sp.kill();
        let mut out = [1.0; 480];
        fill(&mut cons, &sp.shared, &mut 0, &mut out);
        assert!(t.elapsed() < Duration::from_millis(20));
        assert!(out.iter().all(|&s| s == 0.0));
        assert_eq!(cons.slots(), 0);
    }

    #[tokio::test]
    async fn kill_during_synth_pushes_nothing_and_speak_returns() {
        let (sp, cons) = KokoroSpeaker::with_ring(Arc::new(Mock::new(2400, 300)), 24_000, 1);
        let killed_at = Mutex::new(None);
        let speak = async {
            sp.speak("a".into()).await;
            // Returned after kill, long before the 300 ms synth finished.
            let k: Instant = killed_at.lock().unwrap().unwrap();
            assert!(k.elapsed() < Duration::from_millis(50));
        };
        let kill = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            *killed_at.lock().unwrap() = Some(Instant::now());
            sp.kill();
        };
        tokio::join!(speak, kill);
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(cons.slots(), 0);
    }

    #[test]
    fn underrun_writes_zeros() {
        let (sp, mut cons) = KokoroSpeaker::with_ring(Arc::new(Mock::new(0, 0)), 24_000, 1);
        let mut out = [1.0; 256];
        fill(&mut cons, &sp.shared, &mut 0, &mut out);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn callback_is_allocation_and_lock_free() {
        let src = include_str!("tts.rs");
        let start = src.find("fn fill(").unwrap();
        let body = &src[start..start + src[start..].find("\n}\n").unwrap()];
        for bad in ["Vec", "Box", "String", "format!", ".lock()", "tracing::"] {
            assert!(!body.contains(bad), "callback uses {bad}");
        }
    }

    #[test]
    fn tts_env_parsing() {
        assert_eq!(parse_tts(Some("kokoro")).unwrap(), Some(Tts::Kokoro));
        assert_eq!(parse_tts(Some("say")).unwrap(), Some(Tts::Say));
        assert_eq!(parse_tts(None).unwrap(), None);
        assert!(parse_tts(Some("piper"))
            .unwrap_err()
            .to_string()
            .contains("TTS"));
    }

    /// Needs the model files: `KOKORO_MODEL=... KOKORO_VOICES=... cargo test -p gateway -- --ignored --nocapture`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn kokoro_synthesizes_a_sentence() {
        let model = std::env::var("KOKORO_MODEL").expect("KOKORO_MODEL");
        let voices = std::env::var("KOKORO_VOICES").expect("KOKORO_VOICES");
        let voice = std::env::var("KOKORO_VOICE").unwrap_or_else(|_| "af_heart".into());
        let synth = Arc::new(
            KokoroSynth::new(&model, &voices, &voice, 1.0)
                .await
                .unwrap(),
        );
        let text = "Hello there, how are you doing today?";
        let run = |s: Arc<KokoroSynth>| {
            tokio::task::spawn_blocking(move || {
                let t = Instant::now();
                (s.synth(text).unwrap(), t.elapsed())
            })
        };
        let _ = run(synth.clone()).await.unwrap(); // warm-up
        let (samples, took) = run(synth).await.unwrap();
        let secs = samples.len() as f32 / KokoroSynth::SAMPLE_RATE as f32;
        println!(
            "synth {took:?} for {secs:.2}s audio, RTF {:.3}",
            took.as_secs_f32() / secs
        );
        let path = std::env::temp_dir().join("kokoro.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: KokoroSynth::SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for s in &samples {
            w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                .unwrap();
        }
        w.finalize().unwrap();
        println!("wrote {}", path.display());
        let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(secs >= 1.0 && peak > 0.05, "secs={secs} peak={peak}");
    }
}
