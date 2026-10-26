//! Gateway building blocks: lossy audio queue, energy VAD, sentence splitter.
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
        Arc::new(Self { q: Mutex::new(VecDeque::new()), cap, notify: Notify::new(), dropped: 0.into() })
    }
    pub fn push(&self, v: T) {
        let mut q = self.q.lock().unwrap();
        if q.len() == self.cap {
            q.pop_front();
            self.dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
pub enum VadEvent { SpeechStart, EndOfUtterance }

/// Energy VAD over 20 ms frames: `hangover` quiet frames close the utterance.
pub struct Vad { threshold: f32, hangover: u32, quiet: u32, speaking: bool }

impl Vad {
    pub fn new(threshold: f32, hangover_frames: u32) -> Self {
        Self { threshold, hangover: hangover_frames, quiet: 0, speaking: false }
    }
    pub fn push(&mut self, frame: &[i16]) -> Option<VadEvent> {
        let rms = (frame.iter().map(|&s| (s as f32).powi(2)).sum::<f32>() / frame.len().max(1) as f32).sqrt();
        if rms >= self.threshold {
            self.quiet = 0;
            if !self.speaking { self.speaking = true; return Some(VadEvent::SpeechStart); }
        } else if self.speaking {
            self.quiet += 1;
            if self.quiet >= self.hangover { self.speaking = false; self.quiet = 0; return Some(VadEvent::EndOfUtterance); }
        }
        None
    }
}

/// Buffers token deltas and yields complete sentences.
#[derive(Default)]
pub struct SentenceSplitter { buf: String }

impl SentenceSplitter {
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        while let Some(i) = self.buf.char_indices().find_map(|(i, c)| {
            let end = i + c.len_utf8();
            (matches!(c, '.' | '!' | '?' | '\n') && self.buf[end..].starts_with(|n: char| n.is_whitespace())).then_some(end)
        }) {
            let s = self.buf[..i].trim().to_string();
            self.buf.drain(..i);
            if !s.is_empty() { out.push(s); }
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
        for i in 0..4 { q.push(i); }
        assert_eq!(q.dropped(), 2);
        assert_eq!((q.pop().await, q.pop().await), (2, 3));
    }

    #[test]
    fn vad_hangover() {
        let mut v = Vad::new(500.0, 3);
        let (loud, quiet) = ([2000i16; 320], [0i16; 320]);
        assert_eq!(v.push(&quiet), None);
        assert_eq!(v.push(&loud), Some(VadEvent::SpeechStart));
        assert_eq!(v.push(&quiet), None);
        assert_eq!(v.push(&loud), None); // resets hangover
        assert_eq!((v.push(&quiet), v.push(&quiet)), (None, None));
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
