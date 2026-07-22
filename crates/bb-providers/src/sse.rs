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
//!
//! Both framers expose `finish()`, called once when the transport reports
//! end-of-stream: it flushes whatever is still buffered as a final
//! line/frame. This matters because a trailing delimiter is not guaranteed
//! — NDJSON's last line legitimately has no `\n`, and servers routinely
//! close an SSE stream right after the last `data:` line without the
//! terminating blank line. Without the flush the final frame — often the
//! one carrying the stop reason — would be silently dropped.
//!
//! Both framers are also **bounded**: an unterminated line or frame may
//! not exceed [`MAX_FRAME_BYTES`], so an endpoint that never emits a
//! newline produces a [`ProviderError`] instead of exhausting memory.

use crate::types::ProviderError;

/// Maximum bytes buffered for a single unterminated line or frame.
///
/// Chosen to sit far above any legitimate frame — a base64 image payload
/// on one `data:` line is well under 16 MiB — while still bounding what a
/// misbehaving or hostile endpoint can make the client allocate. Crossing
/// it is a [`ProviderError::Decode`]: never a panic, and never a silent
/// truncation that would corrupt the frame.
///
/// The bound applies to the residue left after each feed, so peak usage is
/// this limit plus the single chunk being processed.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

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
/// buffered until later feeds complete them, or until [`finish`] flushes
/// them at end-of-stream. `\n` and `\r\n` line endings are accepted;
/// comment lines (leading `:`) and unknown fields (`id:`, `retry:`, ...)
/// are ignored per the SSE spec.
///
/// A single unterminated line, and the accumulated payload of a single
/// unterminated frame, are each capped at [`MAX_FRAME_BYTES`].
///
/// [`finish`]: SseFramer::finish
#[derive(Debug)]
pub struct SseFramer {
    provider: &'static str,
    buf: Vec<u8>,
    event_name: Option<String>,
    data_lines: Vec<String>,
    /// Bytes accumulated in `data_lines`, tracked so an endpoint that
    /// streams endless `data:` lines without a blank line is bounded too.
    data_bytes: usize,
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
            data_bytes: 0,
        }
    }

    /// Feed raw bytes off the wire; returns every frame completed by this
    /// chunk. Incomplete trailing bytes remain buffered.
    ///
    /// Errors if a *complete* line is not valid UTF-8, or if the buffered
    /// line or frame exceeds [`MAX_FRAME_BYTES`]; the framer should be
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
            self.check_frame_limit()?;
        }
        self.buf.drain(..consumed);
        self.check_line_limit()?;
        Ok(out)
    }

    /// Flush at end-of-stream: any buffered trailing bytes are treated as
    /// a final line, then a pending frame is dispatched as if the blank
    /// line had arrived.
    ///
    /// Servers commonly close the connection right after the last `data:`
    /// line, so without this the final frame — typically the one carrying
    /// the stop reason — would be lost. Calling it twice is harmless: the
    /// second call has nothing buffered and returns no frames.
    ///
    /// Errors if the buffered trailing bytes are not valid UTF-8.
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, ProviderError> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let raw = std::mem::take(&mut self.buf);
            let raw = raw.strip_suffix(b"\r").unwrap_or(&raw);
            let line = std::str::from_utf8(raw)
                .map_err(|err| {
                    ProviderError::decode(
                        self.provider,
                        format!("invalid UTF-8 in trailing SSE line at end of stream: {err}"),
                    )
                })?
                .to_owned();
            self.handle_line(&line, &mut out);
            self.check_frame_limit()?;
        }
        // Dispatch whatever the (never-arriving) blank line would have.
        self.handle_line("", &mut out);
        Ok(out)
    }

    fn check_line_limit(&self) -> Result<(), ProviderError> {
        if self.buf.len() > MAX_FRAME_BYTES {
            return Err(ProviderError::decode(
                self.provider,
                format!(
                    "SSE line exceeded {MAX_FRAME_BYTES} bytes without a newline \
                     ({} buffered)",
                    self.buf.len()
                ),
            ));
        }
        Ok(())
    }

    fn check_frame_limit(&self) -> Result<(), ProviderError> {
        if self.data_bytes > MAX_FRAME_BYTES {
            return Err(ProviderError::decode(
                self.provider,
                format!(
                    "SSE frame exceeded {MAX_FRAME_BYTES} bytes without a blank line \
                     ({} buffered)",
                    self.data_bytes
                ),
            ));
        }
        Ok(())
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
                self.data_bytes = 0;
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
            "data" => {
                self.data_bytes = self
                    .data_bytes
                    .saturating_add(value.len().saturating_add(1));
                self.data_lines.push(value.to_owned());
            }
            _ => {} // id, retry, unknown fields
        }
    }
}

/// Incremental newline-delimited-JSON framer (one JSON document per line).
///
/// Feed raw bytes; complete non-blank lines are returned as `String`s for
/// the caller to parse. Like [`SseFramer`], it buffers bytes and never
/// decodes a partial line, so chunks may split mid-codepoint, and a single
/// unterminated line is capped at [`MAX_FRAME_BYTES`].
///
/// NDJSON's final line is *spec-legal without a trailing newline*, so
/// [`finish`](NdjsonFramer::finish) must be called at end-of-stream or that
/// line — for Ollama, the `"done": true` object — is lost.
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
    ///
    /// Errors if a *complete* line is not valid UTF-8, or if the buffered
    /// line exceeds [`MAX_FRAME_BYTES`]; the framer should be considered
    /// dead after an error.
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
        if self.buf.len() > MAX_FRAME_BYTES {
            return Err(ProviderError::decode(
                self.provider,
                format!(
                    "NDJSON line exceeded {MAX_FRAME_BYTES} bytes without a newline \
                     ({} buffered)",
                    self.buf.len()
                ),
            ));
        }
        Ok(out)
    }

    /// Flush at end-of-stream: buffered trailing bytes become one final
    /// line (a trailing newline is optional in NDJSON). Calling it twice is
    /// harmless — the second call has nothing buffered.
    ///
    /// Errors if the buffered trailing bytes are not valid UTF-8.
    pub fn finish(&mut self) -> Result<Vec<String>, ProviderError> {
        if self.buf.is_empty() {
            return Ok(Vec::new());
        }
        let raw = std::mem::take(&mut self.buf);
        let raw = raw.strip_suffix(b"\r").unwrap_or(&raw);
        let line = std::str::from_utf8(raw).map_err(|err| {
            ProviderError::decode(
                self.provider,
                format!("invalid UTF-8 in trailing NDJSON line at end of stream: {err}"),
            )
        })?;
        if line.trim().is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![line.to_owned()])
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
    fn finish_flushes_frame_without_trailing_blank_line() {
        let mut framer = SseFramer::new("test");
        assert!(framer
            .feed(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n")
            .unwrap()
            .is_empty());
        assert_eq!(
            framer.finish().unwrap(),
            vec![SseEvent {
                event: Some("message_stop".to_owned()),
                data: "{\"type\":\"message_stop\"}".to_owned(),
            }]
        );
    }

    #[test]
    fn finish_flushes_line_without_trailing_newline() {
        let mut framer = SseFramer::new("test");
        assert!(framer.feed(b"data: tail").unwrap().is_empty());
        let events = framer.finish().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "tail");
    }

    #[test]
    fn finish_is_idempotent_and_empty_when_nothing_buffered() {
        let mut framer = SseFramer::new("test");
        assert_eq!(framer.feed(b"data: x\n\n").unwrap().len(), 1);
        assert!(framer.finish().unwrap().is_empty());
        assert!(framer.finish().unwrap().is_empty());
    }

    #[test]
    fn finish_rejects_invalid_utf8_tail() {
        let mut framer = SseFramer::new("test");
        framer.feed(b"data: \xFF").unwrap();
        assert!(matches!(
            framer.finish().unwrap_err(),
            ProviderError::Decode { .. }
        ));
    }

    #[test]
    fn line_under_cap_is_accepted() {
        let mut framer = SseFramer::new("test");
        // A frame just under the cap (e.g. a big base64 image payload)
        // must still frame cleanly.
        let mut bytes = b"data: ".to_vec();
        bytes.resize(MAX_FRAME_BYTES - 2, b'a');
        bytes.extend_from_slice(b"\n\n");
        let events = framer.feed(&bytes).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data.len(), MAX_FRAME_BYTES - 8);
    }

    #[test]
    fn line_exactly_at_cap_is_accepted() {
        let mut framer = SseFramer::new("test");
        assert!(framer
            .feed(&vec![b'a'; MAX_FRAME_BYTES])
            .unwrap()
            .is_empty());
    }

    #[test]
    fn line_over_cap_is_an_error_not_an_oom() {
        let mut framer = SseFramer::new("test");
        let err = framer
            .feed(&vec![b'a'; MAX_FRAME_BYTES + 1])
            .expect_err("unterminated line past the cap must be rejected");
        assert!(matches!(err, ProviderError::Decode { .. }));
    }

    #[test]
    fn line_over_cap_across_many_chunks_is_an_error() {
        // The pathology that actually exhausts memory: a stream that never
        // emits a newline, one modest chunk at a time.
        let mut framer = SseFramer::new("test");
        let chunk = vec![b'a'; 1024 * 1024];
        let err = loop {
            match framer.feed(&chunk) {
                Ok(events) => assert!(events.is_empty()),
                Err(err) => break err,
            }
        };
        assert!(matches!(err, ProviderError::Decode { .. }));
    }

    #[test]
    fn frame_over_cap_across_many_data_lines_is_an_error() {
        // Endless `data:` lines with no blank line grow the pending frame
        // even though no single line is oversized.
        let mut framer = SseFramer::new("test");
        let mut line = b"data: ".to_vec();
        line.resize(1024 * 1024, b'a');
        line.push(b'\n');
        let err = loop {
            match framer.feed(&line) {
                Ok(events) => assert!(events.is_empty()),
                Err(err) => break err,
            }
        };
        assert!(matches!(err, ProviderError::Decode { .. }));
    }

    #[test]
    fn ndjson_finish_flushes_line_without_trailing_newline() {
        let mut framer = NdjsonFramer::new("test");
        assert!(framer.feed(b"{\"done\":true}").unwrap().is_empty());
        assert_eq!(framer.finish().unwrap(), vec!["{\"done\":true}".to_owned()]);
        assert!(framer.finish().unwrap().is_empty());
    }

    #[test]
    fn ndjson_finish_ignores_blank_tail() {
        let mut framer = NdjsonFramer::new("test");
        framer.feed(b"{\"a\":1}\n  ").unwrap();
        assert!(framer.finish().unwrap().is_empty());
    }

    #[test]
    fn ndjson_finish_rejects_invalid_utf8_tail() {
        let mut framer = NdjsonFramer::new("test");
        framer.feed(b"\xFF").unwrap();
        assert!(matches!(
            framer.finish().unwrap_err(),
            ProviderError::Decode { .. }
        ));
    }

    #[test]
    fn ndjson_line_cap_boundaries() {
        let mut framer = NdjsonFramer::new("test");
        assert!(framer
            .feed(&vec![b'a'; MAX_FRAME_BYTES - 1])
            .unwrap()
            .is_empty());
        let mut framer = NdjsonFramer::new("test");
        assert!(framer
            .feed(&vec![b'a'; MAX_FRAME_BYTES])
            .unwrap()
            .is_empty());
        let mut framer = NdjsonFramer::new("test");
        assert!(matches!(
            framer
                .feed(&vec![b'a'; MAX_FRAME_BYTES + 1])
                .expect_err("unterminated line past the cap must be rejected"),
            ProviderError::Decode { .. }
        ));
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
