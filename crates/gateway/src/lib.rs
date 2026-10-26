//! Gateway building blocks: lossy audio queue, energy VAD, sentence splitter.
pub mod audio;
pub mod session;
pub mod tts;

/// Test-only counting allocator: per-thread allocation counts, so parallel tests do not interfere.
#[cfg(test)]
pub(crate) mod alloc_count {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local!(static COUNT: Cell<u64> = const { Cell::new(0) });

    struct Counting;

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
            System.alloc(l)
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            System.dealloc(p, l)
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
            System.realloc(p, l, n)
        }
    }

    #[global_allocator]
    static A: Counting = Counting;

    /// Heap allocations (alloc + realloc) made by `f` on this thread.
    pub(crate) fn count(f: impl FnOnce()) -> u64 {
        let before = COUNT.with(Cell::get);
        f();
        COUNT.with(Cell::get) - before
    }
}

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

/// Bounded queue that evicts the oldest item when full; tokio mpsc cannot do this.
pub struct DropOldest<T> {
    q: Mutex<VecDeque<T>>,
    cap: usize,
    notify: Notify,
    dropped: std::sync::atomic::AtomicU64,
}

impl<T> DropOldest<T> {
    pub fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self {
            q: Mutex::new(VecDeque::new()),
            cap,
            notify: Notify::new(),
            dropped: 0.into(),
        })
    }
    pub fn push(&self, v: T) {
        let mut q = self.q.lock().unwrap();
        if q.len() == self.cap {
            q.pop_front();
            self.dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        q.push_back(v);
        drop(q);
        self.notify.notify_one();
    }
    pub async fn pop(&self) -> T {
        loop {
            if let Some(v) = self.q.lock().unwrap().pop_front() {
                return v;
            }
            self.notify.notified().await;
        }
    }
    pub fn dropped(&self) -> u64 {
        self.dropped.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum VadEvent {
    SpeechStart,
    EndOfUtterance,
}

/// RMS over i16 samples; `say` speech at normal volume sits well above it.
pub const DEFAULT_VAD_THRESHOLD: f32 = 500.0;
/// 8 x 20 ms = 160 ms of loud frames before `SpeechStart`.
pub const MIN_SPEECH_FRAMES: u32 = 8;
/// 20 x 20 ms = 400 ms of quiet frames before `EndOfUtterance`.
pub const HANGOVER_FRAMES: u32 = 20;

/// Energy VAD over 20 ms frames: `min_speech` consecutive loud frames open an utterance,
/// `hangover` quiet frames close it.
pub struct Vad {
    threshold: f32,
    min_speech: u32,
    hangover: u32,
    loud: u32,
    quiet: u32,
    speaking: bool,
    pending: Vec<Vec<i16>>,
}

impl Vad {
    pub fn new(threshold: f32, min_speech_frames: u32, hangover_frames: u32) -> Self {
        Self {
            threshold,
            min_speech: min_speech_frames.max(1),
            hangover: hangover_frames,
            loud: 0,
            quiet: 0,
            speaking: false,
            pending: Vec::new(),
        }
    }
    pub fn is_loud(&self, frame: &[i16]) -> bool {
        let rms = (frame.iter().map(|&s| (s as f32).powi(2)).sum::<f32>()
            / frame.len().max(1) as f32)
            .sqrt();
        rms >= self.threshold
    }
    pub fn speaking(&self) -> bool {
        self.speaking
    }
    pub fn push(&mut self, frame: &[i16]) -> Option<VadEvent> {
        if self.is_loud(frame) {
            self.quiet = 0;
            if !self.speaking {
                self.pending.push(frame.to_vec());
                self.loud += 1;
                if self.loud >= self.min_speech {
                    self.speaking = true;
                    self.loud = 0;
                    return Some(VadEvent::SpeechStart);
                }
            }
        } else if self.speaking {
            self.quiet += 1;
            if self.quiet >= self.hangover {
                self.speaking = false;
                self.quiet = 0;
                return Some(VadEvent::EndOfUtterance);
            }
        } else {
            self.loud = 0;
            self.pending.clear();
        }
        None
    }
    /// Frames buffered while speech was unconfirmed, including the one that fired `SpeechStart`.
    pub fn take_pending(&mut self) -> Vec<Vec<i16>> {
        std::mem::take(&mut self.pending)
    }
}

/// Buffers token deltas and yields complete sentences.
#[derive(Default)]
pub struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        while let Some(i) = self.buf.char_indices().find_map(|(i, c)| {
            let end = i + c.len_utf8();
            (matches!(c, '.' | '!' | '?' | '\n')
                && self.buf[end..].starts_with(|n: char| n.is_whitespace()))
            .then_some(end)
        }) {
            let s = self.buf[..i].trim().to_string();
            self.buf.drain(..i);
            if !s.is_empty() {
                out.push(s);
            }
        }
        out
    }
    pub fn flush(&mut self) -> Option<String> {
        let s = std::mem::take(&mut self.buf).trim().to_string();
        (!s.is_empty()).then_some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn drop_oldest_evicts_and_counts() {
        let q = DropOldest::new(2);
        for i in 0..4 {
            q.push(i);
        }
        assert_eq!(q.dropped(), 2);
        assert_eq!((q.pop().await, q.pop().await), (2, 3));
    }

    #[test]
    fn vad_hangover() {
        let mut v = Vad::new(500.0, 1, 3);
        let (loud, quiet) = ([2000i16; 320], [0i16; 320]);
        assert_eq!(v.push(&quiet), None);
        assert_eq!(v.push(&loud), Some(VadEvent::SpeechStart));
        assert_eq!(v.push(&quiet), None);
        assert_eq!(v.push(&loud), None); // resets hangover
        assert_eq!((v.push(&quiet), v.push(&quiet)), (None, None));
        assert_eq!(v.push(&quiet), Some(VadEvent::EndOfUtterance));
    }

    #[test]
    fn vad_min_speech_keeps_pending_frames() {
        let (loud, quiet) = ([2000i16; 320], [0i16; 320]);
        let mut v = Vad::new(500.0, MIN_SPEECH_FRAMES, HANGOVER_FRAMES);
        for _ in 0..7 {
            assert_eq!(v.push(&loud), None);
        }
        for _ in 0..30 {
            assert_eq!(v.push(&quiet), None);
        }
        assert!(v.take_pending().is_empty());
        for _ in 0..7 {
            assert_eq!(v.push(&loud), None);
        }
        assert_eq!(v.push(&loud), Some(VadEvent::SpeechStart));
        assert_eq!(v.take_pending().len(), 8);
        for _ in 0..19 {
            assert_eq!(v.push(&quiet), None);
        }
        assert_eq!(v.push(&quiet), Some(VadEvent::EndOfUtterance));
    }

    #[test]
    fn splits_sentences_across_deltas() {
        let mut s = SentenceSplitter::default();
        assert!(s.push("Hello the").is_empty());
        assert_eq!(s.push("re. How are"), vec!["Hello there."]);
        assert_eq!(s.push(" you? Fine"), vec!["How are you?"]);
        assert_eq!(s.flush().as_deref(), Some("Fine"));
        let mut d = SentenceSplitter::default();
        assert!(d.push("Pi is 3.14 ok").is_empty()); // no split inside a number
    }
}
