#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Golden-transcript **end-to-end** tests for the whole engine seam.
//!
//! Every other test in this workspace exercises one crate in isolation.
//! These drive the full stack the way a host actually does:
//!
//! ```text
//! raw PTY bytes
//!   -> bb_vt_parser::Parser        (escape/OSC state machine)
//!   -> bb_terminal_model::Screen   (grid + scrollback + absolute rows)
//!   -> bb_block_list::BlockList    (OSC 133 block boundaries)
//!   -> queryable output            (block_text / search / navigation)
//! ```
//!
//! The transcripts are realistic shell interactions (bash with OSC 133
//! shell integration, a vim-style alt-screen app, a shell with no
//! integration at all, and a stream of hostile garbage), and the
//! assertions are on semantically meaningful output — command text, exit
//! codes, block text partitioning — rather than on exact row numbers,
//! except where a row number *is* the thing under test.
//!
//! The load-bearing property here is **chunking independence**: the
//! engine's state must not depend on where the host's `read(2)` calls
//! happened to land in the byte stream. [`interleaved_chunk_boundaries`]
//! is the regression test for that, and every other transcript is written
//! so it could be fed either way.

use bb_block_list::{BlockId, BlockLifecycle, BlockList};
use bb_shared_session::{Session, SessionId};
use bb_terminal_model::{CellWidth, Screen};
use bb_vt_parser::{PositionedEvent, VtEvent};

// ---------------------------------------------------------------------
// Transcript building blocks
// ---------------------------------------------------------------------

/// OSC 133;A — the shell is about to draw a prompt.
const PROMPT_START: &str = "\x1b]133;A\x07";
/// OSC 133;B — the prompt is drawn, the user is composing.
const COMMAND_START: &str = "\x1b]133;B\x07";
/// OSC 133;C — the command was submitted and is now executing.
const COMMAND_EXECUTED: &str = "\x1b]133;C\x07";

/// OSC 133;D\[;code\] — the command finished.
fn command_finished(exit_code: i32) -> String {
    format!("\x1b]133;D;{exit_code}\x07")
}

/// OSC 0 — set the window title.
fn set_title(title: &str) -> String {
    format!("\x1b]0;{title}\x07")
}

/// OSC 7 — report the working directory as a `file://` URL.
fn report_cwd(host: &str, path: &str) -> String {
    format!("\x1b]7;file://{host}{path}\x07")
}

/// OSC 8 — a hyperlink with `label` pointing at `uri`.
fn hyperlink(uri: &str, label: &str) -> String {
    format!("\x1b]8;;{uri}\x07{label}\x1b]8;;\x07")
}

/// One step of a scripted host/PTY interaction.
///
/// A transcript interleaves bytes arriving from the PTY with the
/// out-of-band host call that records the command line, because
/// [`Session::record_command`] is not part of the byte stream — the host
/// captures it from the input editor.
#[derive(Debug, Clone)]
enum Step {
    /// Bytes the shell wrote to the PTY, as one `write(2)`.
    Pty(Vec<u8>),
    /// The host telling the session what the user submitted.
    Command(&'static str),
}

/// A PTY write, from a `&str` for readability.
fn pty(text: &str) -> Step {
    Step::Pty(text.as_bytes().to_vec())
}

/// The shell's prompt line: marker, prompt text, marker.
fn prompt_line() -> Step {
    pty(&format!("{PROMPT_START}user@host:~/proj$ {COMMAND_START}"))
}

// ---------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------

/// Feed a scripted transcript one PTY write at a time — the fine-grained
/// behavior a host gets when the shell's writes are not coalesced.
fn drive(session: &mut Session, steps: &[Step]) -> Vec<PositionedEvent> {
    let mut events = Vec::new();
    for step in steps {
        match step {
            Step::Pty(bytes) => events.extend(session.feed_pty(bytes)),
            Step::Command(cmd) => session.record_command(cmd),
        }
    }
    events
}

/// The semantic events of a feed, with their positions dropped.
fn kinds(events: &[PositionedEvent]) -> Vec<VtEvent> {
    events.iter().map(|e| e.event.clone()).collect()
}

/// The raw byte stream of a transcript, with the host-side
/// [`Step::Command`] steps stripped out.
fn pty_bytes(steps: &[Step]) -> Vec<u8> {
    let mut out = Vec::new();
    for step in steps {
        if let Step::Pty(bytes) = step {
            out.extend_from_slice(bytes);
        }
    }
    out
}

/// The command lines a transcript submits, in order.
fn commands(steps: &[Step]) -> Vec<&'static str> {
    steps
        .iter()
        .filter_map(|s| match s {
            Step::Command(c) => Some(*c),
            Step::Pty(_) => None,
        })
        .collect()
}

/// Feed pre-split segments, recording the next queued command whenever a
/// [`VtEvent::CommandStart`] surfaces — i.e. exactly what a host that
/// learns about command submission from shell integration would do.
///
/// The command is attributed with [`Session::record_command_at_row`] using
/// the marker's *own* row, so a host that received the whole transcript in
/// one `read(2)` still files each command under its own block.
fn feed_segments<'a>(
    session: &mut Session,
    segments: impl Iterator<Item = &'a [u8]>,
    queued: &[&str],
) -> Vec<PositionedEvent> {
    let mut pending = queued.iter();
    let mut all = Vec::new();
    for segment in segments {
        let events = session.feed_pty(segment);
        for event in &events {
            if event.event == VtEvent::CommandStart {
                if let Some(cmd) = pending.next() {
                    assert!(
                        session.record_command_at_row(event.cursor_row, cmd),
                        "no block owned row {} when recording {cmd:?}",
                        event.cursor_row
                    );
                }
            }
        }
        all.extend(events);
    }
    all
}

/// Feed `bytes` in fixed-size chunks (a chunk size of `bytes.len()` is a
/// single one-shot feed).
fn feed_in_chunks(
    session: &mut Session,
    bytes: &[u8],
    chunk_size: usize,
    queued: &[&str],
) -> Vec<PositionedEvent> {
    feed_segments(session, bytes.chunks(chunk_size.max(1)), queued)
}

/// Feed `bytes` split at the given ascending byte offsets.
fn feed_at_offsets(
    session: &mut Session,
    bytes: &[u8],
    offsets: &[usize],
    queued: &[&str],
) -> Vec<PositionedEvent> {
    let mut segments = Vec::new();
    let mut prev = 0usize;
    for &cut in offsets {
        let cut = cut.min(bytes.len());
        if cut > prev {
            segments.push(&bytes[prev..cut]);
            prev = cut;
        }
    }
    if prev < bytes.len() {
        segments.push(&bytes[prev..]);
    }
    feed_segments(session, segments.into_iter(), queued)
}

/// Byte offsets that cut every OSC 133 sequence and every multi-byte
/// UTF-8 scalar in `bytes` in half.
fn adversarial_offsets(bytes: &[u8]) -> Vec<usize> {
    const OSC133: &[u8] = b"\x1b]133;";
    let mut cuts = Vec::new();
    for i in 0..bytes.len() {
        // Split the introducer ("\x1b]1" | "33;...") and again just
        // before the BEL terminator.
        if bytes[i..].starts_with(OSC133) {
            cuts.push(i + 3);
            cuts.push(i + OSC133.len() + 1);
        }
        // Split every multi-byte scalar right after its lead byte.
        if bytes[i] >= 0xC0 {
            cuts.push(i + 1);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

// ---------------------------------------------------------------------
// Observation helpers
// ---------------------------------------------------------------------

/// Every addressable row of the **primary timeline** (scrollback plus the
/// primary grid), oldest first. This is the session's output history — it
/// is what blocks address, and it is unaffected by the alternate screen.
fn screen_rows(screen: &Screen) -> Vec<String> {
    let base = screen.scrollback().evicted();
    (base..base + screen.total_rows())
        .filter_map(|abs| screen.row_text(abs))
        .collect()
}

/// Every row of the grid a renderer would actually paint — the alternate
/// grid while a full-screen app owns the screen, the primary grid
/// otherwise. The counterpart to [`screen_rows`].
fn drawn_rows(screen: &Screen) -> Vec<String> {
    let grid = screen.grid();
    (0..grid.rows())
        .map(|row| {
            grid.row(row)
                .unwrap_or(&[])
                .iter()
                .filter(|cell| cell.width != CellWidth::WideTrail)
                .map(|cell| cell.ch)
                .collect::<String>()
                .trim_end()
                .to_owned()
        })
        .collect()
}

/// The stable, semantically meaningful projection of a block: everything
/// except the absolute row numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BlockFacts {
    id: u64,
    lifecycle: BlockLifecycle,
    command: Option<String>,
    cwd: Option<String>,
    exit_code: Option<i32>,
    succeeded: Option<bool>,
    /// `block_text` with trailing blank rows trimmed (an open block runs
    /// to the end of the grid, so it always has a blank tail).
    text: String,
}

fn block_facts(session: &Session) -> Vec<BlockFacts> {
    session
        .blocks()
        .blocks()
        .map(|b| BlockFacts {
            id: b.id().0,
            lifecycle: b.lifecycle(),
            command: b.command().map(str::to_owned),
            cwd: b.cwd().map(str::to_owned),
            exit_code: b.exit_code(),
            succeeded: b.succeeded(),
            text: session
                .blocks()
                .block_text(session.screen(), b.id())
                .unwrap()
                .trim_end()
                .to_owned(),
        })
        .collect()
}

/// `(start_row, end_row)` for every block, in order.
fn block_ranges(blocks: &BlockList) -> Vec<(usize, Option<usize>)> {
    blocks
        .blocks()
        .map(|b| (b.start_row(), b.end_row()))
        .collect()
}

/// Assert block extents partition the row space: strictly increasing
/// starts, and each closed block ending exactly where the next begins.
fn assert_contiguous_and_disjoint(blocks: &BlockList) {
    let ranges = block_ranges(blocks);
    for window in ranges.windows(2) {
        let (start, end) = window[0];
        let (next_start, _) = window[1];
        assert!(
            end.is_some(),
            "only the newest block may stay open, saw {ranges:?}"
        );
        assert_eq!(
            end,
            Some(next_start),
            "block extents must be contiguous, saw {ranges:?}"
        );
        assert!(
            start <= next_start,
            "block start rows must not go backwards, saw {ranges:?}"
        );
    }
}

/// The blocks whose extent covers `abs_row` (should always be exactly
/// one once any block exists).
fn blocks_covering(blocks: &BlockList, abs_row: usize) -> Vec<BlockId> {
    blocks
        .blocks()
        .filter(|b| match b.end_row() {
            Some(end) => abs_row >= b.start_row() && abs_row < end,
            None => abs_row >= b.start_row(),
        })
        .map(|b| b.id())
        .collect()
}

// =====================================================================
// 1. A realistic multi-command bash session
// =====================================================================

/// prompt -> `ls` -> output -> exit 0
/// prompt -> `cargo build` -> colored output -> exit 0
/// prompt -> `false` -> exit 1
fn bash_transcript() -> Vec<Step> {
    vec![
        prompt_line(),
        Step::Command("ls"),
        pty(&format!("ls\r\n{COMMAND_EXECUTED}")),
        pty("Cargo.toml\r\nsrc\r\nREADME.md\r\n"),
        pty(&command_finished(0)),
        prompt_line(),
        Step::Command("cargo build"),
        pty(&format!("cargo build\r\n{COMMAND_EXECUTED}")),
        // A real cargo line: SGR green, bold-green, and non-ASCII text so
        // the UTF-8 decoder is on the hook too.
        pty("   \x1b[32mCompiling\x1b[0m beam v0.1.0\r\n    \x1b[1;32mFinished\x1b[0m dev — café ✓\r\n"),
        pty(&command_finished(0)),
        prompt_line(),
        Step::Command("false"),
        pty(&format!("false\r\n{COMMAND_EXECUTED}")),
        pty(&command_finished(1)),
    ]
}

/// The event stream `bash_transcript` produces, in order.
fn bash_expected_events() -> Vec<VtEvent> {
    vec![
        VtEvent::PromptStart,
        VtEvent::CommandStart,
        VtEvent::CommandExecuted,
        VtEvent::CommandFinished { exit_code: Some(0) },
        VtEvent::PromptStart,
        VtEvent::CommandStart,
        VtEvent::CommandExecuted,
        VtEvent::CommandFinished { exit_code: Some(0) },
        VtEvent::PromptStart,
        VtEvent::CommandStart,
        VtEvent::CommandExecuted,
        VtEvent::CommandFinished { exit_code: Some(1) },
    ]
}

#[test]
fn realistic_bash_session() {
    let steps = bash_transcript();
    let mut session = Session::new(SessionId("bash".to_owned()), 24, 80);
    let events = drive(&mut session, &steps);

    // The parser surfaced the prompt protocol in stream order, and the
    // SGR / UTF-8 payload produced no spurious semantic events.
    assert_eq!(kinds(&events), bash_expected_events());
    assert_bash_session_facts(&session, "one write per shell write");

    // The very same bytes delivered as a SINGLE read must land in exactly
    // the same engine state. Marker rows come from the events themselves,
    // so nothing here depends on where the host's reads fell.
    let bytes = pty_bytes(&steps);
    let mut one_shot = Session::new(SessionId("bash-one-shot".to_owned()), 24, 80);
    let one_shot_events = feed_in_chunks(&mut one_shot, &bytes, bytes.len(), &commands(&steps));
    assert_eq!(kinds(&one_shot_events), bash_expected_events());
    assert_bash_session_facts(&one_shot, "one coalesced read");
    assert_eq!(
        block_facts(&one_shot),
        block_facts(&session),
        "a coalesced read produced different blocks than per-write feeds"
    );
    assert_eq!(
        block_ranges(one_shot.blocks()),
        block_ranges(session.blocks())
    );
}

/// Every semantic claim `realistic_bash_session` makes, so both the
/// per-write and the one-shot feed can be held to it.
fn assert_bash_session_facts(session: &Session, how: &str) {
    // Three commands ran, so three blocks exist — no phantom leading
    // block from the feeds that carried no marker.
    assert_eq!(session.blocks().len(), 3, "block count ({how})");
    let facts = block_facts(session);

    assert_eq!(
        facts
            .iter()
            .map(|f| f.command.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("ls"), Some("cargo build"), Some("false")],
        "commands ({how})"
    );
    assert_eq!(
        facts.iter().map(|f| f.exit_code).collect::<Vec<_>>(),
        vec![Some(0), Some(0), Some(1)],
        "exit codes ({how})"
    );
    assert_eq!(
        facts.iter().map(|f| f.succeeded).collect::<Vec<_>>(),
        vec![Some(true), Some(true), Some(false)],
        "success flags ({how})"
    );
    // The last block is still open (no following prompt) but the shell
    // reported D, so it is already Finished.
    assert_eq!(
        facts.iter().map(|f| f.lifecycle).collect::<Vec<_>>(),
        vec![
            BlockLifecycle::Finished,
            BlockLifecycle::Finished,
            BlockLifecycle::Finished
        ],
        "lifecycles ({how})"
    );

    // --- the key seam assertion: block_text partitions the output ---
    let ls = &facts[0].text;
    let build = &facts[1].text;
    let fail = &facts[2].text;

    assert_eq!(
        ls, "user@host:~/proj$ ls\nCargo.toml\nsrc\nREADME.md",
        "block 0 text ({how})"
    );
    assert_eq!(
        build, "user@host:~/proj$ cargo build\n   Compiling beam v0.1.0\n    Finished dev — café ✓",
        "block 1 text ({how})"
    );
    assert_eq!(fail, "user@host:~/proj$ false", "block 2 text ({how})");

    // No block leaks its neighbours' output.
    for (needle, owner) in [
        ("Cargo.toml", 0usize),
        ("README.md", 0),
        ("Compiling beam", 1),
        ("café", 1),
        ("false", 2),
    ] {
        for (idx, f) in facts.iter().enumerate() {
            assert_eq!(
                f.text.contains(needle),
                idx == owner,
                "block {idx} text {:?} vs needle {needle:?} ({how})",
                f.text
            );
        }
    }

    // SGR is applied to the screen, not leaked into the text.
    assert!(!build.contains('\x1b'));
    assert!(!build.contains("[32m"));

    // Extents are contiguous, non-overlapping, and every row belongs to
    // exactly one block.
    assert_contiguous_and_disjoint(session.blocks());
    assert_eq!(
        block_ranges(session.blocks()),
        vec![(0, Some(4)), (4, Some(7)), (7, None)],
        "block ranges ({how})"
    );
    for abs_row in 0..=session.screen().cursor_absolute_row() {
        assert_eq!(
            blocks_covering(session.blocks(), abs_row).len(),
            1,
            "row {abs_row} must belong to exactly one block ({how})"
        );
    }

    // Navigation is consistent with that ordering.
    let ids: Vec<BlockId> = session.blocks().blocks().map(|b| b.id()).collect();
    assert_eq!(session.blocks().next(ids[0]).map(|b| b.id()), Some(ids[1]));
    assert_eq!(session.blocks().prev(ids[2]).map(|b| b.id()), Some(ids[1]));
    assert!(session.blocks().next(ids[2]).is_none());
    assert!(session.blocks().prev(ids[0]).is_none());
    assert_eq!(session.blocks().latest().map(|b| b.id()), Some(ids[2]));
    assert_eq!(
        session.blocks().block_at_row(2).map(|b| b.id()),
        Some(ids[0])
    );
    assert_eq!(
        session.blocks().block_at_row(5).map(|b| b.id()),
        Some(ids[1])
    );
}

// =====================================================================
// 2. The same transcript, chopped up every which way
// =====================================================================

/// **The** regression test for chunking independence.
///
/// The same transcript is delivered five ways — as one giant read, one
/// byte at a time, in 3- and 7-byte chunks, and cut in half through every
/// OSC 133 sequence and every multi-byte UTF-8 scalar. Every observable
/// must be identical: positioned events, the rendered screen, block
/// ranges, commands, exit codes, lifecycles, cwds and block *text*.
///
/// Marker rows come from the events themselves, which the parser stamps
/// while it still holds the screen, so no observable can depend on where
/// the host's `read(2)` boundaries fell.
#[test]
fn interleaved_chunk_boundaries() {
    let steps = bash_transcript();
    let bytes = pty_bytes(&steps);
    let queued = commands(&steps);
    let offsets = adversarial_offsets(&bytes);
    assert!(
        offsets.len() > 20,
        "expected many adversarial split points, got {}",
        offsets.len()
    );

    // (label, session, events) for every delivery strategy. The one-shot
    // read comes first and is the baseline everything else is compared
    // against — it is the case the old code got wrong.
    let mut variants: Vec<(String, Session, Vec<PositionedEvent>)> = Vec::new();
    for chunk in [bytes.len(), 1, 3, 7] {
        let label = if chunk == bytes.len() {
            "one-shot".to_owned()
        } else {
            format!("{chunk}-byte chunks")
        };
        let mut session = Session::new(SessionId("chunk".to_owned()), 24, 80);
        let events = feed_in_chunks(&mut session, &bytes, chunk, &queued);
        variants.push((label, session, events));
    }
    {
        let mut session = Session::new(SessionId("chunk".to_owned()), 24, 80);
        let events = feed_at_offsets(&mut session, &bytes, &offsets, &queued);
        variants.push(("adversarial split".to_owned(), session, events));
    }

    let (base_label, base_session, base_events) = &variants[0];
    let base_rows = screen_rows(base_session.screen());
    let base_facts = block_facts(base_session);
    let base_ranges = block_ranges(base_session.blocks());

    for (label, session, events) in &variants[1..] {
        assert_eq!(
            events, base_events,
            "positioned event stream differs: {label} vs {base_label}"
        );
        assert_eq!(
            screen_rows(session.screen()),
            base_rows,
            "screen differs: {label} vs {base_label}"
        );
        assert_eq!(
            block_ranges(session.blocks()),
            base_ranges,
            "block ranges differ: {label} vs {base_label} — \
             marker rows must come from the events, not from the cursor \
             position after the feed"
        );
        assert_eq!(
            block_facts(session),
            base_facts,
            "block contents differ (commands / cwds / exit codes / text): \
             {label} vs {base_label}"
        );
        assert_contiguous_and_disjoint(session.blocks());
    }

    // The multi-byte scalars survived being cut in half.
    assert!(base_rows.iter().any(|r| r.contains("dev — café ✓")));

    // ---- and the shared answer is the RIGHT one ----
    //
    // Invariance alone would also be satisfied by every chunking being
    // uniformly wrong, so pin the absolute values too: exactly three
    // blocks (no phantom leading block from the feeds that carried only a
    // fragment of `ESC ] 1 3 3 ; A`), at their real rows, each owning its
    // own command's output.
    assert_eq!(
        base_facts.len(),
        3,
        "expected exactly three blocks, got {base_ranges:?}"
    );
    assert_eq!(base_ranges, vec![(0, Some(4)), (4, Some(7)), (7, None)]);
    assert_eq!(
        base_facts
            .iter()
            .map(|f| f.command.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("ls"), Some("cargo build"), Some("false")]
    );
    assert_eq!(
        base_facts.iter().map(|f| f.exit_code).collect::<Vec<_>>(),
        vec![Some(0), Some(0), Some(1)]
    );
    assert_eq!(
        base_facts
            .iter()
            .map(|f| f.text.as_str())
            .collect::<Vec<_>>(),
        vec![
            "user@host:~/proj$ ls\nCargo.toml\nsrc\nREADME.md",
            "user@host:~/proj$ cargo build\n   Compiling beam v0.1.0\n    Finished dev — café ✓",
            "user@host:~/proj$ false",
        ]
    );

    // And it is the same state `realistic_bash_session` asserts for the
    // per-shell-write feed, so all six deliveries agree.
    for (label, session, _) in &variants {
        assert_bash_session_facts(session, label);
    }
}

// =====================================================================
// 3. Scrollback eviction
// =====================================================================

#[test]
fn scrollback_eviction_keeps_block_ids_stable() {
    // `Session::with_scrollback` bounds the history, so eviction is
    // reachable in a few hundred rows instead of the ~12_000 it would
    // take to out-produce `Screen::DEFAULT_SCROLLBACK_LINES`.
    const COMMANDS: u32 = 12;
    const LINES: u32 = 20;
    const SCROLLBACK: usize = 64;

    let mut session = Session::with_scrollback(SessionId("evict".to_owned()), 24, 80, SCROLLBACK);
    let mut highest_row = 0usize;
    for cmd in 0..COMMANDS {
        session.feed_pty(prompt_line_bytes().as_slice());
        session.record_command(&format!("gen {cmd}"));
        session.feed_pty(format!("gen {cmd}\r\n{COMMAND_EXECUTED}").as_bytes());
        let mut output = String::new();
        for line in 0..LINES {
            output.push_str(&format!("cmd{cmd} line{line}\r\n"));
        }
        session.feed_pty(output.as_bytes());
        session.feed_pty(command_finished(0).as_bytes());

        // Absolute rows only ever move forward.
        let cursor = session.screen().cursor_absolute_row();
        assert!(
            cursor >= highest_row,
            "absolute row went backwards: {cursor} < {highest_row}"
        );
        highest_row = cursor;
    }
    session.feed_pty(prompt_line_bytes().as_slice());

    let screen_evicted = session.screen().scrollback().evicted();
    assert!(screen_evicted > 0, "transcript did not force any eviction");
    assert_eq!(session.blocks().len() as u32, COMMANDS + 1);

    // Ids are dense and monotonic — allocation never restarted.
    let ids: Vec<u64> = session.blocks().blocks().map(|b| b.id().0).collect();
    assert_eq!(ids, (0..=u64::from(COMMANDS)).collect::<Vec<_>>());

    // Absolute row ids never repeat: extents are contiguous, disjoint,
    // and strictly increasing even across thousands of evictions.
    assert_contiguous_and_disjoint(session.blocks());
    let ranges = block_ranges(session.blocks());
    for window in ranges.windows(2) {
        assert!(
            window[0].0 < window[1].0,
            "start rows must strictly increase"
        );
    }
    for abs_row in (0..highest_row).step_by(97) {
        assert_eq!(
            blocks_covering(session.blocks(), abs_row).len(),
            1,
            "row {abs_row} must belong to exactly one block"
        );
    }

    // Survivors still resolve their text from the retained scrollback.
    let last_finished = session
        .blocks()
        .blocks()
        .nth(COMMANDS as usize - 1)
        .unwrap();
    let text = session
        .blocks()
        .block_text(session.screen(), last_finished.id())
        .unwrap();
    assert!(text.contains(&format!("cmd{} line0", COMMANDS - 1)));
    assert!(text.contains(&format!("cmd{} line{}", COMMANDS - 1, LINES - 1)));

    // The oldest block's rows are all gone, so its text is empty — but
    // the block, and its metadata, are still there.
    let oldest = session.blocks().first().unwrap();
    assert!(oldest.end_row().unwrap() <= screen_evicted);
    assert_eq!(
        session
            .blocks()
            .block_text(session.screen(), oldest.id())
            .unwrap(),
        ""
    );
    assert_eq!(oldest.command(), Some("gen 0"));

    // A host driving the engine through `Session` reclaims the metadata
    // of blocks whose rows are long gone with `Session::prune_blocks`.
    let ids_before: Vec<u64> = session.blocks().blocks().map(|b| b.id().0).collect();
    let pruned = session.prune_blocks();
    assert!(pruned > 0, "fully-evicted blocks should be pruned");
    let ids_after: Vec<u64> = session.blocks().blocks().map(|b| b.id().0).collect();
    // Survivor ids are unchanged — pruning never renumbers, so any id a
    // host is still holding stays valid (or resolves to `None`).
    assert_eq!(ids_after, ids_before[pruned..]);
    assert_eq!(session.blocks().len(), ids_before.len() - pruned);
    // Every survivor still resolves at least some text (or is the open
    // block at the live cursor).
    for block in session.blocks().blocks() {
        assert!(
            block.end_row().is_none() || block.end_row().unwrap() > screen_evicted,
            "a fully-evicted block survived pruning: {block:?}"
        );
        assert!(session
            .blocks()
            .block_text(session.screen(), block.id())
            .is_some());
    }
    // Navigation still walks the survivors, and the pruned ids resolve to
    // nothing rather than to somebody else's block.
    assert_contiguous_and_disjoint(session.blocks());
    for gone in &ids_before[..pruned] {
        assert!(session.blocks().get(BlockId(*gone)).is_none());
        assert!(session.blocks().next(BlockId(*gone)).is_none());
    }
    // Pruning is idempotent.
    assert_eq!(session.prune_blocks(), 0);

    // The surviving blocks keep flowing: the session is still usable.
    session.feed_pty(prompt_line_bytes().as_slice());
    assert_eq!(session.blocks().len(), ids_before.len() - pruned + 1);
    assert_contiguous_and_disjoint(session.blocks());
}

/// `prompt_line()` as raw bytes (the eviction test feeds in a loop).
fn prompt_line_bytes() -> Vec<u8> {
    match prompt_line() {
        Step::Pty(bytes) => bytes,
        Step::Command(_) => unreachable!("prompt_line is a PTY write"),
    }
}

// =====================================================================
// 4. A vim-style full-screen app on the alternate screen
// =====================================================================

#[test]
fn vim_style_alt_screen_roundtrip() {
    // A small screen so the primary side actually spills into scrollback
    // before the alt screen is entered.
    let mut session = Session::new(SessionId("vim".to_owned()), 6, 40);

    drive(
        &mut session,
        &[
            prompt_line(),
            Step::Command("vim notes.txt"),
            pty(&format!("vim notes.txt\r\n{COMMAND_EXECUTED}")),
            pty("opening notes.txt\r\nreading swap file\r\n"),
        ],
    );
    let scrollback_before = session.screen().scrollback().len();
    let evicted_before = session.screen().scrollback().evicted();
    let screen_rows_before_alt = screen_rows(session.screen());
    let block_id = session.blocks().latest().unwrap().id();

    // --- enter the alternate screen (xterm mode 1049) ---
    session.feed_pty(b"\x1b[?1049h");
    assert!(session.screen().is_alt_screen());

    // Draw a full-screen UI, then make it scroll a lot.
    for row in 1..=6 {
        session.feed_pty(format!("\x1b[{row};1HALTUI row {row} ~~~~~~~~").as_bytes());
    }
    session.feed_pty(b"\x1b[6;1H");
    for i in 0..20 {
        session.feed_pty(format!("ALTSCROLL {i}\r\n").as_bytes());
    }

    // The alternate screen has no scrollback of its own, and scrolling
    // it must not push anything into the primary history.
    assert_eq!(session.screen().scrollback().len(), scrollback_before);
    assert_eq!(session.screen().scrollback().evicted(), evicted_before);

    // --- the two coordinate systems, live at the same time ---
    //
    // Absolute rows are the *scrollback timeline*, which the alternate
    // screen does not advance, so `block_text` keeps answering with the
    // command's real primary-screen output even while vim owns the
    // display. A host snapshotting or searching blocks mid-command gets
    // the shell's output, not the app's frame buffer.
    let during_alt = session
        .blocks()
        .block_text(session.screen(), block_id)
        .unwrap();
    assert!(
        during_alt.contains("opening notes.txt") && during_alt.contains("reading swap file"),
        "block text lost the primary output during the alt screen: {during_alt:?}"
    );
    assert!(
        !during_alt.contains("ALTSCROLL") && !during_alt.contains("ALTUI"),
        "the alt screen leaked into block text: {during_alt:?}"
    );
    assert_eq!(
        screen_rows(session.screen()),
        screen_rows_before_alt,
        "the primary timeline moved while the alternate screen was up"
    );
    // Searching the block list agrees with `block_text`.
    assert_eq!(
        session
            .blocks()
            .search_output(session.screen(), "opening notes.txt")
            .len(),
        1
    );
    assert!(session
        .blocks()
        .search_output(session.screen(), "ALTSCROLL")
        .is_empty());

    // ...and the TUI *is* still what a renderer would draw: `grid()`
    // returns the active (alternate) grid, which holds vim's frame.
    let drawn = drawn_rows(session.screen());
    assert!(
        drawn.iter().any(|r| r.contains("ALTSCROLL")),
        "the alt screen is not being drawn: {drawn:?}"
    );
    assert!(
        !drawn.iter().any(|r| r.contains("opening notes.txt")),
        "primary content leaked onto the drawn alt screen: {drawn:?}"
    );

    // --- leave the alternate screen ---
    session.feed_pty(b"\x1b[?1049l");
    assert!(!session.screen().is_alt_screen());

    drive(
        &mut session,
        &[
            pty("exited vim\r\n"),
            pty(&command_finished(0)),
            prompt_line(),
        ],
    );

    // The alt-screen episode left no trace: not in scrollback, not in
    // the restored primary grid, not in any block's text.
    assert_eq!(session.screen().scrollback().evicted(), evicted_before);
    let rows = screen_rows(session.screen());
    for row in &rows {
        assert!(!row.contains("ALTUI"), "alt UI leaked into row {row:?}");
        assert!(
            !row.contains("ALTSCROLL"),
            "alt output leaked into row {row:?}"
        );
    }

    // The primary content the command produced before and after the alt
    // episode is intact and back under the right block.
    let text = session
        .blocks()
        .block_text(session.screen(), block_id)
        .unwrap();
    assert!(text.contains("vim notes.txt"));
    assert!(text.contains("opening notes.txt"));
    assert!(text.contains("reading swap file"));
    assert!(text.contains("exited vim"));
    assert!(!text.contains("ALTUI"));
    assert!(!text.contains("ALTSCROLL"));

    let block = session.blocks().get(block_id).unwrap();
    assert_eq!(block.command(), Some("vim notes.txt"));
    assert_eq!(block.exit_code(), Some(0));
    assert_eq!(block.succeeded(), Some(true));
    assert_eq!(block.lifecycle(), BlockLifecycle::Finished);
    // The next prompt opened a second block; extents stay sane.
    assert_eq!(session.blocks().len(), 2);
    assert_contiguous_and_disjoint(session.blocks());
}

// =====================================================================
// 5. A shell with no OSC 133 shell integration at all
// =====================================================================

#[test]
fn no_shell_integration_fallback() {
    let mut session = Session::new(SessionId("plain".to_owned()), 24, 80);

    // Plain `$ ` prompts, echoed commands, output — not one escape
    // sequence in sight.
    session.feed_pty(b"$ echo hello\r\n");
    session.feed_pty(b"hello\r\n");
    session.feed_pty(b"$ uname -s\r\n");
    session.feed_pty(b"Darwin\r\n");
    session.feed_pty(b"$ ");

    // The heuristic yields exactly one navigable block covering
    // everything, still open because no command boundary was ever
    // observed.
    assert_eq!(session.blocks().len(), 1);
    let block = session.blocks().latest().unwrap();
    assert_eq!(block.id(), BlockId(0));
    assert_eq!(block.lifecycle(), BlockLifecycle::Running);
    assert_eq!(block.start_row(), 0);
    assert_eq!(block.end_row(), None);
    assert_eq!(block.command(), None);
    assert_eq!(block.exit_code(), None);
    assert_eq!(block.succeeded(), None);

    let text = session
        .blocks()
        .block_text(session.screen(), block.id())
        .unwrap();
    for needle in ["$ echo hello", "hello", "$ uname -s", "Darwin"] {
        assert!(text.contains(needle), "fallback block lost {needle:?}");
    }

    // It is navigable: every produced row resolves to it, and the
    // navigation API degrades gracefully with a single block.
    for abs_row in 0..=session.screen().cursor_absolute_row() {
        assert_eq!(
            session.blocks().block_at_row(abs_row).map(|b| b.id()),
            Some(block.id())
        );
    }
    assert!(session.blocks().next(block.id()).is_none());
    assert!(session.blocks().prev(block.id()).is_none());
    assert_eq!(session.blocks().first().map(|b| b.id()), Some(block.id()));
    assert_eq!(session.blocks().last().map(|b| b.id()), Some(block.id()));

    // Search still works over the fallback block.
    assert_eq!(
        session
            .blocks()
            .search_output(session.screen(), "Darwin")
            .len(),
        1
    );
    assert!(session
        .blocks()
        .search_output(session.screen(), "not-in-this-session")
        .is_empty());
    // No command/cwd was ever recorded, so metadata search finds nothing.
    assert!(session.blocks().search_metadata("echo").is_empty());

    // Unknown ids and out-of-range rows are `None`, not panics.
    assert!(session.blocks().get(BlockId(999)).is_none());
    assert_eq!(
        session.blocks().block_text(session.screen(), BlockId(999)),
        None
    );
    assert_eq!(
        session.blocks().block_at_row(usize::MAX).map(|b| b.id()),
        Some(block.id())
    );

    // A still-open block is never pruned, however much is evicted.
    assert_eq!(session.prune_blocks(), 0);
    assert_eq!(session.blocks().len(), 1);
}

/// A shell with integration must never grow the fallback block, however
/// the host's very first reads split the stream. The fallback exists for
/// shells that emit *no* markers at all; a feed that prints nothing is not
/// evidence of that.
#[test]
fn partial_first_read_does_not_create_a_phantom_block() {
    /// (label, reads that must create nothing, reads that finish the
    /// prompt marker and legitimately open the first block)
    type Case = (
        &'static str,
        &'static [&'static [u8]],
        &'static [&'static [u8]],
    );

    let cases: [Case; 4] = [
        // An OSC 133 introducer torn in half by the read boundary.
        ("split escape", &[b"\x1b"], &[b"]133;A\x07"]),
        // Bash turning on bracketed paste before its first prompt.
        (
            "bracketed paste first",
            &[b"\x1b[?2004h"],
            &[b"\x1b]133;A\x07$ "],
        ),
        // The pathological case: one byte per read.
        (
            "one byte per read",
            &[b"\x1b", b"]", b"1", b"3", b"3", b";", b"A"],
            &[b"\x07", b"$", b" "],
        ),
        // Several no-op control sequences before anything is printed.
        (
            "modes and cursor moves first",
            &[b"\x1b[?1h", b"\x1b[?25l", b"\x1b[H"],
            &[b"\x1b]133;A\x07$ "],
        ),
    ];

    for (label, silent, rest) in cases {
        let mut session = Session::new(SessionId(label.to_owned()), 24, 80);
        for (index, read) in silent.iter().enumerate() {
            session.feed_pty(read);
            assert!(
                session.blocks().is_empty(),
                "{label}: read {index} ({read:?}) printed nothing yet manufactured a block"
            );
        }
        for read in rest {
            session.feed_pty(read);
        }
        assert_eq!(
            session.blocks().len(),
            1,
            "{label}: expected exactly the prompt's own block, got {:?}",
            block_ranges(session.blocks())
        );
        let block = session.blocks().latest().unwrap();
        assert_eq!(block.start_row(), 0, "{label}");
        assert_eq!(block.end_row(), None, "{label}");
        assert_eq!(block.lifecycle(), BlockLifecycle::Prompt, "{label}");
    }
}

/// ...and the fallback still appears the moment a shell without any
/// integration actually prints something, whatever the read boundaries.
#[test]
fn fallback_block_appears_on_the_first_real_output() {
    let mut session = Session::new(SessionId("fallback".to_owned()), 24, 80);
    // A mode set and a cursor move print nothing.
    session.feed_pty(b"\x1b[?2004h");
    session.feed_pty(b"\x1b[1;1H");
    assert!(session.blocks().is_empty());

    // The first printed character creates the fallback block at row 0.
    session.feed_pty(b"$");
    assert_eq!(session.blocks().len(), 1);
    let block = session.blocks().latest().unwrap();
    assert_eq!(block.id(), BlockId(0));
    assert_eq!(block.start_row(), 0);
    assert_eq!(block.lifecycle(), BlockLifecycle::Running);

    // ...and it stays the only one, however much more arrives.
    session.feed_pty(b" echo hi\r\nhi\r\n$ ");
    assert_eq!(session.blocks().len(), 1);
    let text = session
        .blocks()
        .block_text(session.screen(), BlockId(0))
        .unwrap();
    assert!(text.contains("$ echo hi") && text.contains("hi"));
}

// =====================================================================
// 10. Bounded scrollback and block pruning through `Session`
// =====================================================================

#[test]
fn bounded_scrollback_makes_pruning_reachable() {
    // 4 retained lines on an 8-row screen: eviction starts almost at once.
    const COMMANDS: usize = 12;
    let mut session = Session::with_scrollback(SessionId("bounded".to_owned()), 8, 40, 4);
    assert_eq!(session.screen().scrollback().max_lines(), 4);

    for i in 0..COMMANDS {
        drive(
            &mut session,
            &[
                prompt_line(),
                Step::Command("echo"),
                pty(&format!("echo {i}\r\n{COMMAND_EXECUTED}")),
                pty(&format!("out {i}\r\n")),
                pty(&command_finished(0)),
            ],
        );
    }
    session.feed_pty(prompt_line_bytes().as_slice());
    assert_eq!(session.blocks().len(), COMMANDS + 1);
    let evicted = session.screen().scrollback().evicted();
    assert!(evicted > 0, "bounded scrollback should have evicted rows");

    // The oldest blocks' rows are gone; their metadata is still costing
    // memory until the host prunes.
    let doomed: Vec<BlockId> = session
        .blocks()
        .blocks()
        .filter(|b| b.end_row().is_some_and(|end| end <= evicted))
        .map(|b| b.id())
        .collect();
    assert!(!doomed.is_empty(), "expected fully-evicted blocks");
    for id in &doomed {
        assert_eq!(
            session
                .blocks()
                .block_text(session.screen(), *id)
                .as_deref(),
            Some(""),
            "block {id} still resolves text, so it is not fully evicted"
        );
    }

    let pruned = session.prune_blocks();
    assert_eq!(pruned, doomed.len());
    for id in &doomed {
        assert!(session.blocks().get(*id).is_none());
    }
    assert_eq!(session.blocks().len(), COMMANDS + 1 - pruned);
    assert_eq!(session.prune_blocks(), 0, "pruning must be idempotent");
    assert_contiguous_and_disjoint(session.blocks());

    // `Session::new` is unchanged: the default capacity is large enough
    // that this transcript evicts nothing at all.
    let mut default = Session::new(SessionId("default".to_owned()), 8, 40);
    for i in 0..6 {
        default.feed_pty(format!("out {i}\r\n").as_bytes());
    }
    assert_eq!(
        default.screen().scrollback().max_lines(),
        Screen::DEFAULT_SCROLLBACK_LINES
    );
    assert_eq!(default.screen().scrollback().evicted(), 0);
    assert_eq!(default.prune_blocks(), 0);
}

// =====================================================================
// 6. Title / cwd / hyperlink / bell pass-through
// =====================================================================

#[test]
fn title_cwd_and_hyperlinks_flow_through() {
    let mut session = Session::new(SessionId("meta".to_owned()), 24, 80);
    assert_eq!(session.title(), None);
    assert_eq!(session.cwd(), None);

    // A prompt that also reports title and cwd, as a real integrated
    // shell's PS1 does.
    let events = session.feed_pty(
        format!(
            "{PROMPT_START}{}{}user@host:~/proj$ {COMMAND_START}",
            set_title("bash — ~/proj"),
            report_cwd("mac.local", "/Users/me/proj"),
        )
        .as_bytes(),
    );
    assert_eq!(
        kinds(&events),
        vec![
            VtEvent::PromptStart,
            VtEvent::Title("bash — ~/proj".to_owned()),
            VtEvent::WorkingDirectory("file://mac.local/Users/me/proj".to_owned()),
            VtEvent::CommandStart,
        ]
    );
    // Every one of them fired on row 0 — the prompt line — and the
    // CommandStart after the 18-column prompt text.
    assert_eq!(
        events
            .iter()
            .map(PositionedEvent::cursor)
            .collect::<Vec<_>>(),
        vec![(0, 0), (0, 0), (0, 0), (0, 18)]
    );
    assert_eq!(session.title(), Some("bash — ~/proj"));
    assert_eq!(session.cwd(), Some("/Users/me/proj"));
    // The cwd propagated into the block metadata too.
    assert_eq!(
        session.blocks().latest().unwrap().cwd(),
        Some("/Users/me/proj")
    );

    session.record_command("curl -s https://example.com");
    let events = session.feed_pty(
        format!(
            "curl -s https://example.com\r\n{COMMAND_EXECUTED}see {}\r\n\x07",
            hyperlink("https://example.com/docs", "the docs")
        )
        .as_bytes(),
    );
    // Hyperlink open/close and the bell are handed back to the caller,
    // not swallowed: the session itself has no opinion on them.
    assert_eq!(
        kinds(&events),
        vec![
            VtEvent::CommandExecuted,
            VtEvent::Hyperlink(Some("https://example.com/docs".to_owned())),
            VtEvent::Hyperlink(None),
            VtEvent::Bell,
        ]
    );
    // The positions locate each one precisely for the host UI: the
    // hyperlink opens 4 columns into row 1 and closes 12 columns in,
    // which is exactly the extent of the label "the docs".
    assert_eq!(
        events
            .iter()
            .map(PositionedEvent::cursor)
            .collect::<Vec<_>>(),
        vec![(1, 0), (1, 4), (1, 12), (2, 0)]
    );
    // The hyperlink's *label* is ordinary printed text; the URI is not.
    assert_eq!(
        session.screen().row_text(1).as_deref(),
        Some("see the docs")
    );
    assert!(!screen_rows(session.screen())
        .iter()
        .any(|r| r.contains("example.com/docs")));

    session.feed_pty(command_finished(0).as_bytes());

    // `cd` mid-session: a new title (OSC 2 this time) and a new cwd, and
    // the *next* block picks up the new cwd.
    let events = session.feed_pty(
        format!(
            "{PROMPT_START}\x1b]2;bash — ~/proj/src\x07{}user@host:~/proj/src$ {COMMAND_START}",
            report_cwd("mac.local", "/Users/me/proj/src")
        )
        .as_bytes(),
    );
    assert!(kinds(&events).contains(&VtEvent::Title("bash — ~/proj/src".to_owned())));
    assert_eq!(session.title(), Some("bash — ~/proj/src"));
    assert_eq!(session.cwd(), Some("/Users/me/proj/src"));

    let blocks: Vec<_> = session.blocks().blocks().collect();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].cwd(), Some("/Users/me/proj"));
    assert_eq!(blocks[1].cwd(), Some("/Users/me/proj/src"));

    // A malformed OSC 7 leaves the cwd alone but is still reported.
    let events = session.feed_pty(b"\x1b]7;not-a-url\x07");
    assert_eq!(
        kinds(&events),
        vec![VtEvent::WorkingDirectory("not-a-url".to_owned())]
    );
    assert_eq!(session.cwd(), Some("/Users/me/proj/src"));
}

// =====================================================================
// 7. Search across blocks
// =====================================================================

#[test]
fn search_across_blocks() {
    let mut session = Session::new(SessionId("search".to_owned()), 24, 80);

    let runs: [(&'static str, &str); 3] = [
        (
            "grep alpha src/",
            "alpha-only-hit here\nshared-token appears\n",
        ),
        (
            "grep beta src/",
            "beta-only-hit here\nshared-token appears\n",
        ),
        (
            "grep gamma src/",
            "gamma-only-hit here\nshared-token appears\n",
        ),
    ];
    let mut steps = Vec::new();
    for (cmd, output) in runs {
        steps.push(prompt_line());
        steps.push(Step::Command(cmd));
        steps.push(pty(&format!("{cmd}\r\n{COMMAND_EXECUTED}")));
        steps.push(pty(&output.replace('\n', "\r\n")));
        steps.push(pty(&command_finished(0)));
    }
    steps.push(prompt_line());
    drive(&mut session, &steps);
    assert_eq!(session.blocks().len(), 4); // 3 commands + the open prompt

    // Search is a query over block extents, so it inherits chunking
    // independence: the same transcript read in one gulp indexes the same
    // way, block for block.
    let bytes = pty_bytes(&steps);
    let mut one_shot = Session::new(SessionId("search-one-shot".to_owned()), 24, 80);
    feed_in_chunks(&mut one_shot, &bytes, bytes.len(), &commands(&steps));
    assert_eq!(block_facts(&one_shot), block_facts(&session));

    let commands_of = |hits: Vec<&bb_block_list::Block>| -> Vec<Option<String>> {
        hits.into_iter()
            .map(|b| b.command().map(str::to_owned))
            .collect()
    };

    // The shared word hits every block that has it, newest first. The
    // trailing prompt-only block has no output, so it does not match.
    assert_eq!(
        commands_of(
            session
                .blocks()
                .search_output(session.screen(), "shared-token")
        ),
        vec![
            Some("grep gamma src/".to_owned()),
            Some("grep beta src/".to_owned()),
            Some("grep alpha src/".to_owned()),
        ]
    );

    // Unique words select exactly one block each.
    for (needle, expected) in [
        ("alpha-only-hit", "grep alpha src/"),
        ("beta-only-hit", "grep beta src/"),
        ("gamma-only-hit", "grep gamma src/"),
    ] {
        assert_eq!(
            commands_of(session.blocks().search_output(session.screen(), needle)),
            vec![Some(expected.to_owned())],
            "search_output({needle:?})"
        );
    }

    // Output search is case-insensitive and misses cleanly.
    assert_eq!(
        session
            .blocks()
            .search_output(session.screen(), "SHARED-TOKEN")
            .len(),
        3
    );
    assert!(session
        .blocks()
        .search_output(session.screen(), "delta-only-hit")
        .is_empty());

    // Metadata search matches the recorded command text (case-
    // insensitively) and the cwd, not the output.
    assert_eq!(
        commands_of(session.blocks().search_metadata("BETA")),
        vec![Some("grep beta src/".to_owned())]
    );
    assert_eq!(session.blocks().search_metadata("grep").len(), 3);
    // "shared-token" only ever appears in output, never in metadata.
    assert!(session.blocks().search_metadata("shared-token").is_empty());
    // ...and conversely the command line appears in the block *text*
    // too, because the tty echoed it above the output.
    assert_eq!(
        commands_of(
            session
                .blocks()
                .search_output(session.screen(), "grep beta")
        ),
        vec![Some("grep beta src/".to_owned())]
    );

    // Every one of those queries answers identically on the coalesced
    // feed — the block layer, not just the screen, is chunk-invariant.
    for needle in [
        "shared-token",
        "alpha-only-hit",
        "beta-only-hit",
        "gamma-only-hit",
        "grep beta",
        "delta-only-hit",
    ] {
        assert_eq!(
            commands_of(session.blocks().search_output(session.screen(), needle)),
            commands_of(one_shot.blocks().search_output(one_shot.screen(), needle)),
            "search_output({needle:?}) differs between per-write and one-shot feeds"
        );
    }
}

// =====================================================================
// 8. Resize in the middle of a command
// =====================================================================

#[test]
fn resize_midstream_preserves_blocks() {
    let mut session = Session::new(SessionId("resize".to_owned()), 24, 80);

    drive(
        &mut session,
        &[
            prompt_line(),
            Step::Command("tail -f log"),
            pty(&format!("tail -f log\r\n{COMMAND_EXECUTED}")),
            pty(&report_cwd("mac.local", "/var/log")),
            pty("before resize A\r\nbefore resize B\r\n"),
        ],
    );
    let before_ranges = block_ranges(session.blocks());
    let block_id = session.blocks().latest().unwrap().id();
    let cursor_before = session.screen().cursor_absolute_row();

    // The user drags the window; the host forwards the new PTY size.
    session.resize(10, 40);
    assert_eq!(session.screen().rows(), 10);
    assert_eq!(session.screen().cols(), 40);

    // Blocks are absolute-row addressed, so nothing about them moved.
    assert_eq!(block_ranges(session.blocks()), before_ranges);
    assert_eq!(session.screen().cursor_absolute_row(), cursor_before);

    // The rest of the transcript keeps flowing into the same block.
    drive(
        &mut session,
        &[
            pty("after resize C\r\n"),
            pty(&command_finished(0)),
            prompt_line(),
            Step::Command("echo done"),
            pty(&format!("echo done\r\n{COMMAND_EXECUTED}done\r\n")),
            pty(&command_finished(0)),
        ],
    );

    // Metadata survived the resize untouched.
    let block = session.blocks().get(block_id).unwrap();
    assert_eq!(block.command(), Some("tail -f log"));
    assert_eq!(block.cwd(), Some("/var/log"));
    assert_eq!(block.exit_code(), Some(0));
    assert_eq!(block.succeeded(), Some(true));
    assert_eq!(block.lifecycle(), BlockLifecycle::Finished);
    assert_eq!(session.blocks().len(), 2);
    assert_contiguous_and_disjoint(session.blocks());
    assert_eq!(
        session.blocks().last().unwrap().command(),
        Some("echo done")
    );
    assert_eq!(session.cwd(), Some("/var/log"));

    // Post-resize output is queryable through the block seam.
    let text = session
        .blocks()
        .block_text(session.screen(), block_id)
        .unwrap();
    assert!(text.contains("after resize C"));
    assert_eq!(
        session
            .blocks()
            .search_output(session.screen(), "after resize C")
            .len(),
        1
    );
    assert_eq!(session.blocks().search_metadata("tail -f").len(), 1);

    // Pre-resize output SURVIVES: `Screen::resize` preserves on-screen
    // content, so the rows the block points at still hold their text.
    assert!(
        text.contains("before resize A") && text.contains("before resize B"),
        "resize destroyed pre-resize output: {text:?}"
    );
    assert_eq!(
        session
            .blocks()
            .search_output(session.screen(), "before resize")
            .len(),
        1
    );
    // The whole block reads back in order, across the resize seam.
    assert_eq!(
        text,
        "user@host:~/proj$ tail -f log\nbefore resize A\nbefore resize B\nafter resize C"
    );
}

/// Shrinking the screen under the cursor sheds rows off the *top* into
/// scrollback, so absolute rows — and therefore blocks — keep addressing
/// the same content.
#[test]
fn resize_shrinking_under_the_cursor_sheds_into_scrollback() {
    let mut session = Session::new(SessionId("shrink".to_owned()), 8, 40);
    drive(
        &mut session,
        &[
            prompt_line(),
            Step::Command("seq 6"),
            pty(&format!("seq 6\r\n{COMMAND_EXECUTED}")),
            pty("one\r\ntwo\r\nthree\r\nfour\r\nfive\r\nsix\r\n"),
        ],
    );
    let block_id = session.blocks().latest().unwrap().id();
    let cursor_before = session.screen().cursor_absolute_row();
    let text_before = session
        .blocks()
        .block_text(session.screen(), block_id)
        .unwrap();
    assert!(session.screen().scrollback().is_empty());

    // The cursor is below the new bottom, so rows must leave the top.
    session.resize(3, 40);
    assert_eq!(session.screen().rows(), 3);
    assert!(
        !session.screen().scrollback().is_empty(),
        "shed rows should have moved into scrollback"
    );
    // The cursor's absolute row is exactly invariant across the resize.
    assert_eq!(session.screen().cursor_absolute_row(), cursor_before);
    // ...so the block still resolves the same text, from scrollback now.
    assert_eq!(
        session
            .blocks()
            .block_text(session.screen(), block_id)
            .unwrap(),
        text_before
    );
    assert!(text_before.contains("one") && text_before.contains("six"));
}

// =====================================================================
// 9. Malformed and hostile input
// =====================================================================

#[test]
fn malformed_and_hostile_input() {
    let mut session = Session::new(SessionId("hostile".to_owned()), 24, 80);

    // A well-formed command first, so we can prove the garbage does not
    // retroactively corrupt it.
    drive(
        &mut session,
        &[
            prompt_line(),
            Step::Command("cat /dev/urandom"),
            pty(&format!("cat /dev/urandom\r\n{COMMAND_EXECUTED}")),
        ],
    );

    let mut hostile: Vec<u8> = Vec::new();
    hostile.extend_from_slice(b"\x1b"); // a lone ESC, never completed
    hostile.extend_from_slice(b"\x1b["); // truncated CSI
    hostile.extend_from_slice(b"\x1b[99999999999999999999;;;;m"); // param overflow
    hostile.extend_from_slice(b"\x1b[?\x1b[1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19H");
    hostile.extend_from_slice(b"\x1b]"); // truncated OSC introducer
    hostile.extend_from_slice(&vec![b'Z'; 9_000]); // over-long OSC payload
    hostile.extend_from_slice(b"\x07"); // ...terminated at last
    hostile.extend_from_slice(b"\x1b]133"); // truncated OSC 133, no terminator
    hostile.extend_from_slice(b"\x1b\\"); // ST closes it
    hostile.extend_from_slice(b"\x7f\x7f"); // DEL bytes (must be discarded)
    hostile.extend_from_slice(b"\x00\x00\x00"); // NULs
    hostile.extend_from_slice(&[0xff, 0xfe, 0xc3]); // invalid UTF-8
    hostile.extend_from_slice(b"\x1bPsome dcs payload\x1b\\"); // DCS passthrough
    hostile.extend_from_slice(b"survivor\r\n");

    // Feed it whole, then again split at every byte, to make sure no
    // chunk boundary inside the garbage panics either.
    let events_whole = session.feed_pty(&hostile);
    // The over-long OSC payload is discarded *including* its BEL
    // terminator, so no `Bell` is reported for it; nothing else in this
    // stream is a semantic event, so the parser stays silent.
    assert_eq!(kinds(&events_whole), Vec::<VtEvent>::new());

    let mut byte_at_a_time = Session::new(SessionId("hostile2".to_owned()), 24, 80);
    for &b in &hostile {
        byte_at_a_time.feed_pty(&[b]);
    }
    assert!(screen_rows(byte_at_a_time.screen())
        .iter()
        .any(|r| r.contains("survivor")));

    // Garbage is chunk-safe too: fed whole or one byte at a time into
    // fresh sessions, the screen AND the block state come out identical.
    let mut whole = Session::new(SessionId("hostile3".to_owned()), 24, 80);
    whole.feed_pty(&hostile);
    assert_eq!(
        screen_rows(byte_at_a_time.screen()),
        screen_rows(whole.screen()),
        "hostile input rendered differently when chunked"
    );
    assert_eq!(
        block_ranges(byte_at_a_time.blocks()),
        block_ranges(whole.blocks()),
        "hostile input produced different blocks when chunked"
    );
    // No marker ever completed, so both got exactly the one fallback
    // block the printable tail earned.
    assert_eq!(whole.blocks().len(), 1);
    assert_eq!(whole.blocks().latest().unwrap().start_row(), 0);

    // Cursor positioning inside the garbage moved the cursor, so the
    // block's own output landed wherever the CUP said. What matters is
    // that the stream is still parseable and the printable tail made it
    // onto the screen exactly once, with invalid bytes replaced.
    let rows = screen_rows(session.screen());
    let survivors: Vec<&String> = rows.iter().filter(|r| r.contains("survivor")).collect();
    assert_eq!(
        survivors.len(),
        1,
        "expected exactly one survivor row in {rows:?}"
    );
    assert!(
        survivors[0].contains('\u{FFFD}'),
        "invalid UTF-8 should decode to U+FFFD, got {:?}",
        survivors[0]
    );
    // DEL, NUL, and the DCS payload are discarded outright.
    assert!(!survivors[0].contains('\u{7f}'));
    assert!(!survivors[0].contains('\0'));
    for row in &rows {
        assert!(!row.contains("some dcs payload"), "DCS leaked into {row:?}");
        assert!(!row.contains("ZZZZ"), "over-long OSC leaked into {row:?}");
    }

    // The session survives and keeps parsing valid shell integration on
    // the far side of the garbage.
    drive(
        &mut session,
        &[
            pty(&command_finished(130)),
            prompt_line(),
            Step::Command("echo recovered"),
            pty(&format!(
                "echo recovered\r\n{COMMAND_EXECUTED}recovered\r\n"
            )),
            pty(&command_finished(0)),
        ],
    );

    assert_eq!(session.blocks().len(), 2);
    let facts = block_facts(&session);
    assert_eq!(facts[0].command.as_deref(), Some("cat /dev/urandom"));
    assert_eq!(facts[0].exit_code, Some(130));
    assert_eq!(facts[0].succeeded, Some(false));
    assert_eq!(facts[1].command.as_deref(), Some("echo recovered"));
    assert_eq!(facts[1].exit_code, Some(0));
    assert_eq!(facts[1].succeeded, Some(true));
    assert!(facts[1].text.contains("echo recovered"));
    assert!(facts[1].text.contains("recovered"));
    assert_contiguous_and_disjoint(session.blocks());

    // Queries over a garbage-laden screen do not panic.
    assert_eq!(
        session
            .blocks()
            .search_output(session.screen(), "recovered")
            .len(),
        1
    );
    assert_eq!(session.blocks().search_metadata("urandom").len(), 1);
    // Nothing has been evicted, so nothing prunes away.
    assert_eq!(session.prune_blocks(), 0);
    assert_eq!(session.blocks().len(), 2);
}
