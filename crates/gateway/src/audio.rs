//! Frame sources: cpal mic (lock-free ring, resampled to 16 kHz) and a WAV fake mic.
use std::future::Future;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Duration;

use anyhow::{bail, Context};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{FftFixedInOut, Resampler};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::DropOldest;

pub const RATE: u32 = 16_000;
/// 20 ms at 16 kHz.
pub const FRAME: usize = 320;
pub type Frame = Vec<i16>;

/// Yields 20 ms 16 kHz mono frames; `None` ends the session. `next` must be cancel-safe.
pub trait FrameSource: Send {
    fn next(&mut self) -> impl Future<Output = Option<Frame>> + Send;
}

impl FrameSource for Arc<DropOldest<Frame>> {
    async fn next(&mut self) -> Option<Frame> {
        Some(self.pop().await)
    }
}

impl FrameSource for mpsc::Receiver<Frame> {
    async fn next(&mut self) -> Option<Frame> {
        self.recv().await
    }
}

/// Plays a 16 kHz mono i16 WAV in real time, then silence forever.
pub struct FakeMic {
    frames: Vec<Frame>,
    i: usize,
    tick: tokio::time::Interval,
}

impl FakeMic {
    pub fn open(path: &str) -> anyhow::Result<Self> {
        let r = hound::WavReader::open(path).with_context(|| format!("FAKE_MIC={path}"))?;
        let s = r.spec();
        if s.channels != 1
            || s.sample_rate != RATE
            || s.bits_per_sample != 16
            || s.sample_format != hound::SampleFormat::Int
        {
            bail!("FAKE_MIC={path}: need 16 kHz mono 16-bit PCM, got {s:?}");
        }
        let samples: Vec<i16> = r
            .into_samples::<i16>()
            .collect::<Result<_, _>>()
            .with_context(|| format!("FAKE_MIC={path}"))?;
        let frames = samples
            .chunks(FRAME)
            .map(|c| {
                let mut f = c.to_vec();
                f.resize(FRAME, 0);
                f
            })
            .collect();
        Ok(Self {
            frames,
            i: 0,
            tick: tokio::time::interval(Duration::from_millis(20)),
        })
    }
}

impl FrameSource for FakeMic {
    async fn next(&mut self) -> Option<Frame> {
        self.tick.tick().await;
        let f = self
            .frames
            .get(self.i)
            .cloned()
            .unwrap_or_else(|| vec![0; FRAME]);
        self.i += 1;
        Some(f)
    }
}

/// Downmixes interleaved f32 to mono, resamples to 16 kHz and cuts 320-sample i16 frames.
pub struct Framer {
    channels: usize,
    rs: Option<FftFixedInOut<f32>>,
    pending: Vec<f32>,
    out: Vec<i16>,
}

impl Framer {
    pub fn new(in_rate: u32, channels: u16) -> anyhow::Result<Self> {
        let rs = (in_rate != RATE)
            .then(|| FftFixedInOut::new(in_rate as usize, RATE as usize, in_rate as usize / 50, 1))
            .transpose()
            .context("resampler")?;
        Ok(Self {
            channels: channels.max(1) as usize,
            rs,
            pending: Vec::new(),
            out: Vec::new(),
        })
    }

    pub fn push(&mut self, interleaved: &[f32]) -> Vec<Frame> {
        let ch = self.channels;
        self.pending.extend(
            interleaved
                .chunks_exact(ch)
                .map(|f| f.iter().sum::<f32>() / ch as f32),
        );
        let to_i16 = |s: &f32| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        match &mut self.rs {
            None => {
                self.out.extend(self.pending.iter().map(to_i16));
                self.pending.clear();
            }
            Some(rs) => {
                while self.pending.len() >= rs.input_frames_next() {
                    let n = rs.input_frames_next();
                    match rs.process(&[&self.pending[..n]], None) {
                        Ok(o) => self.out.extend(o[0].iter().map(to_i16)),
                        Err(e) => tracing::warn!("resample: {e}"),
                    }
                    self.pending.drain(..n);
                }
            }
        }
        let whole = self.out.len() - self.out.len() % FRAME;
        let frames = self.out[..whole]
            .chunks(FRAME)
            .map(<[i16]>::to_vec)
            .collect();
        self.out.drain(..whole);
        frames
    }
}

/// Input callback body. REAL-TIME: copies into the preallocated ring and bumps a counter, nothing else.
fn mic_callback(prod: &mut rtrb::Producer<f32>, dropped: &AtomicU64, data: &[f32]) {
    let n = data.len().min(prod.slots());
    if let Ok(chunk) = prod.write_chunk_uninit(n) {
        chunk.fill_from_iter(data[..n].iter().copied());
    }
    if n < data.len() {
        dropped.fetch_add((data.len() - n) as u64, Ordering::Relaxed);
    }
}

/// Starts the default input device. Keep the returned stream alive; `ring_dropped` counts samples lost to a full ring.
pub fn start_mic(
    q: Arc<DropOldest<Frame>>,
    ring_dropped: Arc<AtomicU64>,
) -> anyhow::Result<(cpal::Stream, JoinHandle<()>)> {
    let device = cpal::default_host()
        .default_input_device()
        .context("no default input device")?;
    let supported = device.default_input_config().context("mic config")?;
    if supported.sample_format() != cpal::SampleFormat::F32 {
        bail!(
            "mic sample format {:?} unsupported (need f32)",
            supported.sample_format()
        );
    }
    let config = supported.config();
    let (rate, channels) = (config.sample_rate, config.channels);
    let mut framer = Framer::new(rate, channels)?;
    // 1 s of headroom, allocated once here.
    let (mut prod, mut cons) = rtrb::RingBuffer::<f32>::new(rate as usize * channels as usize);
    let dropped = ring_dropped.clone();
    let stream = device
        .build_input_stream::<f32, _, _>(
            config,
            // Real-time thread: only copy into the preallocated ring and bump a counter.
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                mic_callback(&mut prod, &dropped, data)
            },
            |e| tracing::warn!("mic: {e}"),
            None,
        )
        .context("open mic stream")?;
    stream.play().context("start mic")?;
    let task = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(10));
        let mut buf = Vec::with_capacity(rate as usize * channels as usize);
        loop {
            tick.tick().await;
            let n = cons.slots() - cons.slots() % channels as usize;
            let Ok(chunk) = cons.read_chunk(n) else {
                continue;
            };
            let (a, b) = chunk.as_slices();
            buf.clear();
            buf.extend_from_slice(a);
            buf.extend_from_slice(b);
            chunk.commit_all();
            for f in framer.push(&buf) {
                q.push(f);
            }
        }
    });
    Ok((stream, task))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resamples_48k_stereo_to_16k_mono_frames() {
        let mut f = Framer::new(48_000, 2).unwrap();
        // 1 s of a 440 Hz tone, fed in 10 ms callbacks.
        let input: Vec<f32> = (0..48_000)
            .flat_map(|i| {
                let s = (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5;
                [s, s]
            })
            .collect();
        let frames: Vec<Frame> = input.chunks(960).flat_map(|c| f.push(c)).collect();
        assert!(frames.iter().all(|f| f.len() == FRAME));
        // ~16 000 samples out = 50 frames, minus resampler latency.
        assert!((47..=50).contains(&frames.len()), "{}", frames.len());
        assert!(frames[25].iter().any(|&s| s > 10_000), "tone lost");
    }

    #[tokio::test]
    async fn fake_mic_emits_fixture_then_silence() {
        let mut m = FakeMic::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/en.wav"
        ))
        .unwrap();
        let n = m.frames.len();
        assert!(n > 50);
        m.i = n;
        assert_eq!(m.next().await, Some(vec![0; FRAME]));
    }

    #[test]
    fn mic_callback_makes_zero_heap_allocations() {
        let (mut prod, mut cons) = rtrb::RingBuffer::<f32>::new(4_800);
        let dropped = AtomicU64::new(0);
        let data = [0.2f32; 960]; // 10 ms of 48 kHz stereo
        let n = crate::alloc_count::count(|| {
            for i in 0..2_000 {
                mic_callback(&mut prod, &dropped, &data);
                if i % 7 == 0 {
                    // Consumer lags: the ring fills and the drop path runs.
                    let k = cons.slots().min(960);
                    if let Ok(c) = cons.read_chunk(k) {
                        c.commit_all();
                    }
                }
            }
        });
        assert_eq!(n, 0, "mic callback allocated");
        assert!(dropped.load(Ordering::Relaxed) > 0);
    }

    /// Informational: allocations per 20 ms frame on the capture-to-sender path (Framer + queue push).
    #[test]
    fn capture_path_allocations_per_frame() {
        let mut framer = Framer::new(48_000, 2).unwrap();
        let q = crate::DropOldest::new(50);
        let chunk = vec![0.1f32; 960];
        for _ in 0..200 {
            for f in framer.push(&chunk) {
                q.push(f);
            }
        }
        let mut frames = 0u64;
        let n = crate::alloc_count::count(|| {
            for _ in 0..1_000 {
                for f in framer.push(&chunk) {
                    q.push(f);
                    frames += 1;
                }
            }
        });
        println!(
            "capture path: {n} allocs over {frames} frames = {:.2} per frame",
            n as f64 / frames as f64
        );
        assert!(frames > 0);
    }
}
