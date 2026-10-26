//! whisper.cpp STT on Metal; cancel aborts inference through the abort callback.
use std::ffi::c_void;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::Stt;

/// One model and one state, loaded once; inferences are serialised on the state.
pub struct WhisperStt {
    state: Arc<Mutex<WhisperState>>,
}

impl WhisperStt {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let ctx = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .map_err(|e| anyhow!("WHISPER_MODEL={path}: cannot load model: {e}"))?;
        let state = ctx.create_state().context("whisper state")?;
        Ok(Self {
            state: Arc::new(Mutex::new(state)),
        })
    }
}

unsafe extern "C" fn abort_cb(data: *mut c_void) -> bool {
    (*(data as *const AtomicBool)).load(Ordering::Relaxed)
}

#[async_trait]
impl Stt for WhisperStt {
    async fn transcribe(
        &self,
        pcm: Vec<u8>,
        cancel: CancellationToken,
    ) -> anyhow::Result<Option<(String, String)>> {
        let abort = Arc::new(AtomicBool::new(false));
        let watcher = tokio::spawn({
            let (abort, cancel) = (abort.clone(), cancel.clone());
            async move {
                cancel.cancelled().await;
                abort.store(true, Ordering::Relaxed);
            }
        });
        let state = self.state.clone();
        let flag = abort.clone();
        let res = tokio::task::spawn_blocking(move || {
            let audio: Vec<f32> = pcm
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_language(Some("auto"));
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            // whisper-rs 0.16 `set_abort_callback_safe` casts its user data to the wrong type, so register the raw callback.
            // SAFETY: `flag` outlives `full()` below, and `abort_cb` only reads the AtomicBool it points to.
            unsafe {
                params.set_abort_callback(Some(abort_cb));
                params.set_abort_callback_user_data(Arc::as_ptr(&flag) as *mut c_void);
            }
            let mut st = state
                .lock()
                .map_err(|_| anyhow!("whisper state poisoned"))?;
            st.full(params, &audio).context("whisper inference")?;
            let text: String = st
                .as_iter()
                .filter_map(|s| s.to_str_lossy().ok().map(|t| t.into_owned()))
                .collect();
            let lang = whisper_rs::get_lang_str(st.full_lang_id_from_state())
                .unwrap_or("")
                .to_string();
            anyhow::Ok((text.trim().to_string(), lang))
        })
        .await;
        watcher.abort();
        if abort.load(Ordering::Relaxed) {
            return Ok(None);
        }
        Ok(Some(res.context("whisper task")??))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn fixture(name: &str) -> Vec<u8> {
        let path = format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        let r = hound::WavReader::open(path).unwrap();
        r.into_samples::<i16>()
            .flat_map(|s| s.unwrap().to_le_bytes())
            .collect()
    }

    fn stt() -> WhisperStt {
        WhisperStt::load(&std::env::var("WHISPER_MODEL").expect("set WHISPER_MODEL")).unwrap()
    }

    #[tokio::test]
    #[ignore = "needs WHISPER_MODEL"]
    async fn detects_en_and_de() {
        let s = stt();
        let (text, lang) = s
            .transcribe(fixture("en.wav"), CancellationToken::new())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(lang, "en");
        assert!(!text.is_empty());
        let (_, lang) = s
            .transcribe(fixture("de.wav"), CancellationToken::new())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(lang, "de");
    }

    #[tokio::test]
    #[ignore = "needs WHISPER_MODEL"]
    async fn cancel_aborts_inference() {
        let s = stt();
        let pcm = fixture("en.wav");
        let t = Instant::now();
        s.transcribe(pcm.clone(), CancellationToken::new())
            .await
            .unwrap()
            .unwrap();
        let full = t.elapsed();
        let cancel = CancellationToken::new();
        tokio::spawn({
            let c = cancel.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                c.cancel();
            }
        });
        let t = Instant::now();
        assert!(s.transcribe(pcm, cancel).await.unwrap().is_none());
        assert!(
            t.elapsed() < full,
            "cancelled {:?} vs full {full:?}",
            t.elapsed()
        );
    }
}
