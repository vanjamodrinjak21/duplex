//! Ollama /api/chat streaming LLM.
use anyhow::{anyhow, Context};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::Llm;

/// Keeps the model resident between turns so TTFT stays low.
pub const KEEP_ALIVE: &str = "30m";

pub struct OllamaLlm {
    pub url: String,
    pub model: String,
    client: reqwest::Client,
}

impl OllamaLlm {
    pub fn new(url: String, model: String) -> Self {
        Self { url, model, client: reqwest::Client::new() }
    }

    /// Loads the model into memory before the first turn.
    pub async fn prewarm(&self) -> anyhow::Result<()> {
        let body = json!({ "model": self.model, "keep_alive": KEEP_ALIVE });
        let r = self.client.post(format!("{}/api/generate", self.url)).body(body.to_string()).send().await?;
        anyhow::ensure!(r.status().is_success(), "ollama prewarm: HTTP {}", r.status());
        Ok(())
    }
}

/// Oldest first, alternating user/assistant; the last entry is the current user message.
pub fn messages(prompt: &[String]) -> Vec<Value> {
    prompt
        .iter()
        .enumerate()
        .map(|(i, c)| json!({ "role": if i % 2 == 0 { "user" } else { "assistant" }, "content": c }))
        .collect()
}

#[async_trait]
impl Llm for OllamaLlm {
    async fn stream(&self, prompt: Vec<String>, tx: mpsc::Sender<String>, cancel: CancellationToken) -> anyhow::Result<()> {
        let body = json!({ "model": self.model, "stream": true, "keep_alive": KEEP_ALIVE, "messages": messages(&prompt) });
        let req = self.client.post(format!("{}/api/chat", self.url)).header("content-type", "application/json").body(body.to_string());
        let resp = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            r = req.send() => r.context("ollama request")?,
        };
        anyhow::ensure!(resp.status().is_success(), "ollama: HTTP {}", resp.status());
        let mut body = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        loop {
            let chunk = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Ok(()),
                c = body.next() => match c {
                    Some(c) => c.context("ollama stream")?,
                    None => return Ok(()),
                },
            };
            buf.extend_from_slice(&chunk);
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                let v: Value = serde_json::from_slice(&line).context("ollama ndjson")?;
                if let Some(e) = v.get("error") {
                    return Err(anyhow!("ollama: {e}"));
                }
                if let Some(t) = v["message"]["content"].as_str().filter(|t| !t.is_empty()) {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return Ok(()),
                        r = tx.send(t.to_string()) => if r.is_err() { return Ok(()) },
                    }
                }
                if v["done"].as_bool() == Some(true) {
                    return Ok(());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Serves one request: returns its body via the handle and writes `lines`, then (if `forever`) more lines until the client hangs up.
    async fn fake(lines: Vec<&'static str>, forever: bool) -> (String, tokio::task::JoinHandle<String>) {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", l.local_addr().unwrap());
        let h = tokio::spawn(async move {
            let (mut s, _) = l.accept().await.unwrap();
            let mut req = Vec::new();
            let mut b = [0u8; 4096];
            loop {
                let n = s.read(&mut b).await.unwrap();
                req.extend_from_slice(&b[..n]);
                let text = String::from_utf8_lossy(&req).to_string();
                if let Some(i) = text.find("\r\n\r\n") {
                    let len: usize = text.to_lowercase().split("content-length: ").nth(1).and_then(|r| r.split("\r\n").next()?.parse().ok()).unwrap_or(0);
                    if req.len() >= i + 4 + len {
                        break;
                    }
                }
            }
            let text = String::from_utf8_lossy(&req).to_string();
            let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
            let _ = s.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\nconnection: close\r\n\r\n").await;
            for l in lines {
                let _ = s.write_all(format!("{l}\n").as_bytes()).await;
            }
            while forever {
                if s.write_all(b"{\"message\":{\"content\":\"x\"},\"done\":false}\n").await.is_err() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            body
        });
        (url, h)
    }

    #[tokio::test]
    async fn streams_deltas_in_order() {
        let (url, h) = fake(
            vec![
                r#"{"message":{"role":"assistant","content":"Hi"},"done":false}"#,
                r#"{"message":{"role":"assistant","content":" there"},"done":false}"#,
                r#"{"message":{"role":"assistant","content":""},"done":true}"#,
            ],
            false,
        )
        .await;
        let llm = OllamaLlm::new(url, "qwen2.5:3b".into());
        let (tx, mut rx) = mpsc::channel(8);
        llm.stream(vec!["a".into(), "b".into(), "c".into()], tx, CancellationToken::new()).await.unwrap();
        assert_eq!((rx.recv().await.as_deref(), rx.recv().await.as_deref(), rx.recv().await), (Some("Hi"), Some(" there"), None));
        let body: Value = serde_json::from_str(&h.await.unwrap()).unwrap();
        assert_eq!(body["stream"], true);
        assert_eq!(body["keep_alive"], KEEP_ALIVE);
        let roles: Vec<_> = body["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "assistant", "user"]);
    }

    #[tokio::test]
    async fn cancel_stops_endless_stream() {
        let (url, _h) = fake(vec![], true).await;
        let llm = OllamaLlm::new(url, "m".into());
        let (tx, mut rx) = mpsc::channel(1024);
        let cancel = CancellationToken::new();
        let task = tokio::spawn({
            let c = cancel.clone();
            async move { llm.stream(vec!["hi".into()], tx, c).await }
        });
        rx.recv().await.unwrap();
        let t = Instant::now();
        cancel.cancel();
        tokio::time::timeout(Duration::from_millis(100), task).await.expect("stream() did not return within 100 ms").unwrap().unwrap();
        assert!(t.elapsed() < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn http_500_is_an_error() {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", l.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut s, _) = l.accept().await.unwrap();
            let mut b = [0u8; 4096];
            let _ = s.read(&mut b).await;
            let _ = s.write_all(b"HTTP/1.1 500 Internal Server Error\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").await;
        });
        let (tx, _rx) = mpsc::channel(8);
        assert!(OllamaLlm::new(url, "m".into()).stream(vec!["hi".into()], tx, CancellationToken::new()).await.is_err());
    }
}
