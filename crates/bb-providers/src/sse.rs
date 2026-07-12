//! Chunk-boundary-safe framing for streaming responses.
//!
//! Network chunks can split anywhere: mid-UTF-8-codepoint, mid-line,
//! mid-JSON-document. Both framers here therefore buffer raw **bytes**
//! and only decode text once a complete line is available — never
//! lossy-decoding a partial chunk.
//!
//! - [`SseFramer`] frames `text/event-stream` bytes into [`SseEvent`]s
//!   (used by the Anthropic, OpenAI-compatible, and Gemini adapters).
//! - [`NdjsonFramer`] frames newline-delimited JSON into complete lines
//!   (used by the Ollama adapter).

use crate::types::ProviderError;

/// A single decoded Server-Sent Events frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Value of the frame's `event:` field, if present.
    pub event: Option<String>,
    /// All `data:` lines of the frame, joined with `\n`.
    pub data: String,
}

/// Incremental SSE parser that tolerates arbitrary chunk boundaries.
///
/// Feed raw bytes as they arrive; complete frames (terminated by a blank
/// line) are returned. Bytes belonging to an unfinished line or frame stay
/// buffered until later feeds complete them. `\n` and `\r\n` line endings
/// are accepted; comment lines (leading `:`) and unknown fields (`id:`,
/// `retry:`, ...) are ignored per the SSE spec.
#[derive(Debug)]
pub struct SseFramer {
    provider: &'static str,
    buf: Vec<u8>,
    event_name: Option<String>,
    data_lines: Vec<String>,
}

impl SseFramer {
    /// Create a framer. `provider` tags any decode errors with the adapter
    /// name for context.
    pub fn new(provider: &'static str) -> Self {
        Self {
            provider,
            buf: Vec::new(),
            event_name: None,
            data_lines: Vec::new(),
        }
    }

    /// Feed raw bytes off the wire; returns every frame completed by this
    /// chunk. Incomplete trailing bytes remain buffered.
    ///
    /// Errors if a *complete* line is not valid UTF-8; the framer should be
    /// considered dead after an error.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, ProviderError> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        let mut consumed = 0usize;
        loop {
            let rest = self.buf.get(consumed..).unwrap_or(&[]);
            let Some(rel) = rest.iter().position(|&b| b == b'\n') else {
                break;
            };
            let raw = rest.get(..rel).unwrap_or(&[]);
            let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
            let line = match std::str::from_utf8(raw) {
                Ok(line) => line.to_owned(),
                Err(err) => {
                    self.buf.drain(..consumed);
                    return Err(ProviderError::decode(
                        self.provider,
                        format!("invalid UTF-8 in SSE line: {err}"),
                    ));
                }
            };
            consumed += rel + 1;
            self.handle_line(&line, &mut out);
        }
        self.buf.drain(..consumed);
        Ok(out)
    }

    fn handle_line(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            // Blank line dispatches the pending frame, if any.
            if self.event_name.is_some() || !self.data_lines.is_empty() {
                out.push(SseEvent {
                    event: self.event_name.take(),
                    data: self.data_lines.join("\n"),
                });
                self.data_lines.clear();
            }
            return;
        }
        if line.starts_with(':') {
            return; // comment
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => self.event_name = Some(value.to_owned()),
            "data" => self.data_lines.push(value.to_owned()),
            _ => {} // id, retry, unknown fields
        }
    }
}

/// Incremental newline-delimited-JSON framer (one JSON document per line).
///
/// Feed raw bytes; complete non-blank lines are returned as `String`s for
/// the caller to parse. Like [`SseFramer`], it buffers bytes and never
/// decodes a partial line, so chunks may split mid-codepoint.
#[derive(Debug)]
pub struct NdjsonFramer {
    provider: &'static str,
    buf: Vec<u8>,
}

impl NdjsonFramer {
    /// Create a framer. `provider` tags any decode errors with the adapter
    /// name for context.
    pub fn new(provider: &'static str) -> Self {
        Self {
            provider,
            buf: Vec::new(),
        }
    }

    /// Feed raw bytes off the wire; returns every complete non-blank line.
    /// Incomplete trailing bytes remain buffered.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<String>, ProviderError> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        let mut consumed = 0usize;
        loop {
            let rest = self.buf.get(consumed..).unwrap_or(&[]);
            let Some(rel) = rest.iter().position(|&b| b == b'\n') else {
                break;
            };
            let raw = rest.get(..rel).unwrap_or(&[]);
            let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
            match std::str::from_utf8(raw) {
                Ok(line) => {
                    if !line.trim().is_empty() {
                        out.push(line.to_owned());
                    }
                }
                Err(err) => {
                    self.buf.drain(..consumed);
                    return Err(ProviderError::decode(
                        self.provider,
                        format!("invalid UTF-8 in NDJSON line: {err}"),
                    ));
                }
            }
            consumed += rel + 1;
        }
        self.buf.drain(..consumed);
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn feed_all(framer: &mut SseFramer, bytes: &[u8], size: usize) -> Vec<SseEvent> {
        let mut out = Vec::new();
        for chunk in bytes.chunks(size.max(1)) {
            out.extend(framer.feed(chunk).unwrap());
        }
        out
    }

    #[test]
    fn frames_basic_event() {
        let mut framer = SseFramer::new("test");
        let events = framer
            .feed(b"event: message_start\ndata: {\"a\":1}\n\n")
            .unwrap();
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("message_start".to_owned()),
                data: "{\"a\":1}".to_owned(),
            }]
        );
    }

    #[test]
    fn accepts_crlf_line_endings() {
        let mut framer = SseFramer::new("test");
        let events = framer.feed(b"data: hello\r\n\r\n").unwrap();
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "hello".to_owned(),
            }]
        );
    }

    #[test]
    fn joins_multi_line_data() {
        let mut framer = SseFramer::new("test");
        let events = framer.feed(b"data: line1\ndata: line2\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn ignores_comments_and_unknown_fields() {
        let mut framer = SseFramer::new("test");
        let events = framer
            .feed(b": keep-alive\nid: 42\nretry: 100\ndata: x\n\n")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
        assert_eq!(events[0].event, None);
    }

    #[test]
    fn incomplete_frame_stays_buffered() {
        let mut framer = SseFramer::new("test");
        assert!(framer.feed(b"data: partial").unwrap().is_empty());
        assert!(framer.feed(b" line\n").unwrap().is_empty());
        let events = framer.feed(b"\n").unwrap();
        assert_eq!(events[0].data, "partial line");
    }

    #[test]
    fn split_mid_codepoint_is_lossless() {
        let bytes = "data: caf\u{e9} \u{1f389} done\n\n".as_bytes();
        let mut whole = SseFramer::new("test");
        let expected = whole.feed(bytes).unwrap();
        assert_eq!(expected[0].data, "caf\u{e9} \u{1f389} done");
        for size in [1, 2, 3, 5] {
            let mut framer = SseFramer::new("test");
            assert_eq!(feed_all(&mut framer, bytes, size), expected, "size {size}");
        }
    }

    #[test]
    fn rejects_invalid_utf8_in_complete_line() {
        let mut framer = SseFramer::new("test");
        // 0xFF can never appear in UTF-8.
        let err = framer.feed(b"data: \xFF\n\n").unwrap_err();
        assert!(matches!(
            err,
            ProviderError::Decode {
                provider: "test",
                ..
            }
        ));
    }

    #[test]
    fn ndjson_frames_lines_and_skips_blanks() {
        let mut framer = NdjsonFramer::new("test");
        let lines = framer.feed(b"{\"a\":1}\n\n{\"b\":2}\r\n{\"part").unwrap();
        assert_eq!(lines, vec!["{\"a\":1}".to_owned(), "{\"b\":2}".to_owned()]);
        let lines = framer.feed(b"ial\":3}\n").unwrap();
        assert_eq!(lines, vec!["{\"partial\":3}".to_owned()]);
    }

    #[test]
    fn ndjson_split_mid_codepoint_is_lossless() {
        let bytes = "{\"text\":\"\u{e9}\u{1f389}\"}\n{\"done\":true}\n".as_bytes();
        let mut whole = NdjsonFramer::new("test");
        let expected = whole.feed(bytes).unwrap();
        assert_eq!(expected.len(), 2);
        for size in [1, 2, 3, 7] {
            let mut framer = NdjsonFramer::new("test");
            let mut out = Vec::new();
            for chunk in bytes.chunks(size) {
                out.extend(framer.feed(chunk).unwrap());
            }
            assert_eq!(out, expected, "size {size}");
        }
    }
}
