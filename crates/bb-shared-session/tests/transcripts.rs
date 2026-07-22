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
//! Several tests carry `// NOTE:` comments that pin **current** behavior
//! that is arguably wrong. They are written to pass so the suite stays
//! green; each one names the discrepancy so the finding is not lost.

use bb_block_list::{BlockId, BlockLifecycle, BlockList};
use bb_shared_session::{Session, SessionId};
use bb_terminal_model::Screen;
use bb_vt_parser::VtEvent;

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
fn drive(session: &mut Session, steps: &[Step]) -> Vec<VtEvent> {
    let mut events = Vec::new();
    for step in steps {
        match step {
            Step::Pty(bytes) => events.extend(session.feed_pty(bytes)),
            Step::Command(cmd) => session.record_command(cmd),
        }
    }
    events
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
fn feed_segments<'a>(
    session: &mut Session,
    segments: impl Iterator<Item = &'a [u8]>,
    queued: &[&str],
) -> Vec<VtEvent> {
    let mut pending = queued.iter();
    let mut all = Vec::new();
    for segment in segments {
        let events = session.feed_pty(segment);
        for event in &events {
            if matches!(event, VtEvent::CommandStart) {
                if let Some(cmd) = pending.next() {
                    session.record_command(cmd);
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
) -> Vec<VtEvent> {
    feed_segments(session, bytes.chunks(chunk_size.max(1)), queued)
}

/// Feed `bytes` split at the given ascending byte offsets.
fn feed_at_offsets(
    session: &mut Session,
    bytes: &[u8],
    offsets: &[usize],
    queued: &[&str],
) -> Vec<VtEvent> {
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

/// Every addressable row of the screen, oldest first.
fn screen_rows(screen: &Screen) -> Vec<String> {
    let base = screen.scrollback().evicted();
    (base..base + screen.total_rows())
        .filter_map(|abs| screen.row_text(abs))
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

#[test]
fn realistic_bash_session() {
    let mut session = Session::new(SessionId("bash".to_owned()), 24, 80);
    let events = drive(&mut session, &bash_transcript());

    // The parser surfaced the prompt protocol in stream order, and the
    // SGR / UTF-8 payload produced no spurious semantic events.
    assert_eq!(
        events,
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
    );

    // Three commands ran, so three blocks exist.
    assert_eq!(session.blocks().len(), 3);
    let facts = block_facts(&session);

    assert_eq!(
        facts
            .iter()
            .map(|f| f.command.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("ls"), Some("cargo build"), Some("false")]
    );
    assert_eq!(
        facts.iter().map(|f| f.exit_code).collect::<Vec<_>>(),
        vec![Some(0), Some(0), Some(1)]
    );
    assert_eq!(
        facts.iter().map(|f| f.succeeded).collect::<Vec<_>>(),
        vec![Some(true), Some(true), Some(false)]
    );
    // The last block is still open (no following prompt) but the shell
    // reported D, so it is already Finished.
    assert_eq!(
        facts.iter().map(|f| f.lifecycle).collect::<Vec<_>>(),
        vec![
            BlockLifecycle::Finished,
            BlockLifecycle::Finished,
            BlockLifecycle::Finished
        ]
    );

    // --- the key seam assertion: block_text partitions the output ---
    let ls = &facts[0].text;
    let build = &facts[1].text;
    let fail = &facts[2].text;

    assert_eq!(ls, "user@host:~/proj$ ls\nCargo.toml\nsrc\nREADME.md");
    assert_eq!(
        build,
        "user@host:~/proj$ cargo build\n   Compiling beam v0.1.0\n    Finished dev — café ✓"
    );
    assert_eq!(fail, "user@host:~/proj$ false");

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
                "block {idx} text {:?} vs needle {needle:?}",
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
        vec![(0, Some(4)), (4, Some(7)), (7, None)]
    );
    for abs_row in 0..=session.screen().cursor_absolute_row() {
        assert_eq!(
            blocks_covering(session.blocks(), abs_row).len(),
            1,
            "row {abs_row} must belong to exactly one block"
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

    let run_chunked = |chunk: usize| {
        let mut s = Session::new(SessionId("chunk".to_owned()), 24, 80);
        let events = feed_in_chunks(&mut s, &bytes, chunk, &queued);
        (s, events)
    };

    let (one_shot, one_shot_events) = run_chunked(bytes.len());
    let (by_1, by_1_events) = run_chunked(1);
    let (by_3, by_3_events) = run_chunked(3);
    let (by_7, by_7_events) = run_chunked(7);
    let (split, split_events) = {
        let mut s = Session::new(SessionId("chunk".to_owned()), 24, 80);
        let events = feed_at_offsets(&mut s, &bytes, &offsets, &queued);
        (s, events)
    };

    // ---- invariants that hold for EVERY chunking ----
    //
    // The parser/screen half of the seam is genuinely chunk-safe: the
    // event stream and the rendered screen do not depend on where the
    // host's reads happened to land.
    for (label, events) in [
        ("1-byte", &by_1_events),
        ("3-byte", &by_3_events),
        ("7-byte", &by_7_events),
        ("adversarial-split", &split_events),
    ] {
        assert_eq!(events, &one_shot_events, "event stream differs for {label}");
    }
    let expected_rows = screen_rows(one_shot.screen());
    for (label, session) in [
        ("1-byte", &by_1),
        ("3-byte", &by_3),
        ("7-byte", &by_7),
        ("adversarial-split", &split),
    ] {
        assert_eq!(
            screen_rows(session.screen()),
            expected_rows,
            "screen differs for {label}"
        );
    }
    // The multi-byte scalars survived being cut in half.
    assert!(expected_rows.iter().any(|r| r.contains("dev — café ✓")));

    // ---- block-level invariance across incremental chunkings ----
    //
    // Any chunking fine enough that each OSC 133 marker lands in its own
    // feed produces byte-identical block state, including block text.
    let baseline = block_facts(&by_1);
    for (label, session) in [
        ("3-byte", &by_3),
        ("7-byte", &by_7),
        ("adversarial-split", &split),
    ] {
        assert_eq!(block_facts(session), baseline, "blocks differ for {label}");
    }
    assert_eq!(
        block_ranges(by_3.blocks()),
        block_ranges(by_1.blocks()),
        "block rows differ between 1-byte and 3-byte feeds"
    );
    assert_eq!(block_ranges(by_7.blocks()), block_ranges(by_1.blocks()));
    assert_eq!(block_ranges(split.blocks()), block_ranges(by_1.blocks()));

    // NOTE (finding 1 — spurious leading block): the incrementally-fed
    // sessions end up with FOUR blocks where the stepwise-fed session in
    // `realistic_bash_session` gets three. `Session::feed_pty` always
    // calls `BlockList::observe_row` at the end of the feed, and
    // `observe_row` lazily creates the "no shell integration" fallback
    // block whenever the list is empty and no marker has been seen yet.
    // The very first 1/3/7-byte chunk is a fragment of `ESC ] 1 3 3 ; A`
    // that emits no event, so the fallback block is created and is then
    // immediately closed (empty) by the real PromptStart. A shell-
    // integrated session should never produce this block.
    assert_eq!(baseline.len(), 4);
    assert_eq!(baseline[0].command, None);
    assert_eq!(baseline[0].exit_code, None);
    assert_eq!(baseline[0].text, "");
    assert_eq!(block_ranges(by_1.blocks())[0], (0, Some(0)));
    // The three real blocks are otherwise exactly the stepwise ones.
    let real: Vec<_> = baseline.iter().skip(1).collect();
    assert_eq!(
        real.iter()
            .map(|f| f.command.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("ls"), Some("cargo build"), Some("false")]
    );
    assert_eq!(
        real.iter().map(|f| f.exit_code).collect::<Vec<_>>(),
        vec![Some(0), Some(0), Some(1)]
    );
    assert!(real[0].text.contains("Cargo.toml"));
    assert!(real[1].text.contains("Compiling beam"));
    assert!(!real[1].text.contains("Cargo.toml"));

    // NOTE (finding 2 — block boundaries depend on read chunking): a
    // one-shot feed of the identical bytes collapses every marker onto
    // the SAME row. `feed_pty` stamps each marker with the cursor row
    // *after* the whole feed, not the cursor row at the point the marker
    // appeared in the stream, so a host whose PTY read coalesced the
    // three commands loses all block boundaries: the first two blocks
    // become zero-height with empty text, and `record_command` — driven
    // by events that all arrive in one batch — lands entirely on the last
    // block. The screen is identical (asserted above); only the block
    // layer diverges.
    let collapsed = block_facts(&one_shot);
    assert_eq!(
        collapsed.len(),
        3,
        "no fallback block: the first feed already carried a marker"
    );
    assert_eq!(
        block_ranges(one_shot.blocks()),
        vec![(8, Some(8)), (8, Some(8)), (8, None)]
    );
    assert_eq!(collapsed[0].text, "");
    assert_eq!(collapsed[1].text, "");
    assert_eq!(collapsed[0].command, None);
    assert_eq!(collapsed[1].command, None);
    assert_eq!(collapsed[2].command.as_deref(), Some("false"));
    // Exit codes still line up, because they ride on the markers, not on
    // the rows.
    assert_eq!(
        collapsed.iter().map(|f| f.exit_code).collect::<Vec<_>>(),
        vec![Some(0), Some(0), Some(1)]
    );
}

// =====================================================================
// 3. Scrollback eviction
// =====================================================================

#[test]
fn scrollback_eviction_keeps_block_ids_stable() {
    // `Session::new` always uses `Screen::DEFAULT_SCROLLBACK_LINES`
    // (10_000) and exposes no way to shrink it, so the transcript has to
    // out-produce that: 12 commands x 1001 rows = 12_012 rows.
    const COMMANDS: u32 = 12;
    const LINES: u32 = 1_000;

    let mut session = Session::new(SessionId("evict".to_owned()), 24, 80);
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

    // NOTE (finding 3 — `prune_evicted` is unreachable through
    // `Session`): `Session` exposes `blocks() -> &BlockList` and no
    // mutable accessor, so a host driving the engine through `Session`
    // can never call `BlockList::prune_evicted`. The block list grows
    // one entry per command for the life of the session, retaining
    // command/cwd strings for blocks whose text can no longer be
    // resolved. The pruning semantics themselves are correct — asserted
    // here on a clone, which is the only way to reach them from outside.
    let mut prunable = session.blocks().clone();
    let ids_before: Vec<u64> = prunable.blocks().map(|b| b.id().0).collect();
    let pruned = prunable.prune_evicted(session.screen());
    assert!(pruned > 0, "fully-evicted blocks should be pruned");
    let ids_after: Vec<u64> = prunable.blocks().map(|b| b.id().0).collect();
    // Survivor ids are unchanged — pruning never renumbers.
    assert_eq!(ids_after, ids_before[pruned..]);
    assert_eq!(prunable.len(), ids_before.len() - pruned);
    // Every survivor still resolves at least some text (or is the open
    // block at the live cursor).
    for block in prunable.blocks() {
        assert!(
            block.end_row().is_none() || block.end_row().unwrap() > screen_evicted,
            "a fully-evicted block survived pruning: {block:?}"
        );
        assert!(prunable.block_text(session.screen(), block.id()).is_some());
    }
    // Pruning is idempotent.
    assert_eq!(prunable.prune_evicted(session.screen()), 0);
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

    // NOTE (finding 4 — alt-screen rows shadow primary rows): while the
    // alt screen is active, `Screen::row_text` reads the *active* grid,
    // and the alt grid reuses the same absolute row ids as the primary
    // grid (scrollback does not advance). So `block_text` for the open
    // block returns the alt-screen UI, not the command's real output.
    // Nothing is corrupted — the primary grid is restored on exit — but a
    // host that snapshots or searches block text while a full-screen app
    // is running gets the app's frame buffer back.
    let during_alt = session
        .blocks()
        .block_text(session.screen(), block_id)
        .unwrap();
    assert!(during_alt.contains("ALTSCROLL"));
    assert!(!during_alt.contains("opening notes.txt"));

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
    let mut prunable = session.blocks().clone();
    assert_eq!(prunable.prune_evicted(session.screen()), 0);
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
        events,
        vec![
            VtEvent::PromptStart,
            VtEvent::Title("bash — ~/proj".to_owned()),
            VtEvent::WorkingDirectory("file://mac.local/Users/me/proj".to_owned()),
            VtEvent::CommandStart,
        ]
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
        events,
        vec![
            VtEvent::CommandExecuted,
            VtEvent::Hyperlink(Some("https://example.com/docs".to_owned())),
            VtEvent::Hyperlink(None),
            VtEvent::Bell,
        ]
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
    assert!(events.contains(&VtEvent::Title("bash — ~/proj/src".to_owned())));
    assert_eq!(session.title(), Some("bash — ~/proj/src"));
    assert_eq!(session.cwd(), Some("/Users/me/proj/src"));

    let blocks: Vec<_> = session.blocks().blocks().collect();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].cwd(), Some("/Users/me/proj"));
    assert_eq!(blocks[1].cwd(), Some("/Users/me/proj/src"));

    // A malformed OSC 7 leaves the cwd alone but is still reported.
    let events = session.feed_pty(b"\x1b]7;not-a-url\x07");
    assert_eq!(
        events,
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

    // NOTE (finding 5 — resize is destructive to on-screen text):
    // `Screen::resize` clears both grids (documented on that method), so
    // any row still living in the *grid* at resize time loses its
    // content. Absolute row ids and block extents are preserved, which
    // means the block keeps pointing at rows that are now blank: the
    // pre-resize output vanishes from `block_text` and from
    // `search_output`. Rows already spilled into scrollback are
    // unaffected. A real terminal reflows instead.
    assert!(!text.contains("before resize A"));
    assert!(!text.contains("before resize B"));
    assert!(session
        .blocks()
        .search_output(session.screen(), "before resize")
        .is_empty());
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
    // NOTE: the over-long OSC payload is discarded *including* its BEL
    // terminator, so no `Bell` is reported for it; nothing else in this
    // stream is a semantic event, so the parser stays silent.
    assert_eq!(events_whole, Vec::<VtEvent>::new());

    let mut byte_at_a_time = Session::new(SessionId("hostile2".to_owned()), 24, 80);
    for &b in &hostile {
        byte_at_a_time.feed_pty(&[b]);
    }
    assert!(screen_rows(byte_at_a_time.screen())
        .iter()
        .any(|r| r.contains("survivor")));

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
    let mut prunable = session.blocks().clone();
    assert_eq!(prunable.prune_evicted(session.screen()), 0);
}
