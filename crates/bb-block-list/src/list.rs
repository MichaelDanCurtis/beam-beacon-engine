//! The ordered [`BlockList`] and its lifecycle / query operations.

use bb_terminal_model::Screen;

use crate::block::{Block, BlockId, BlockLifecycle, PromptMarker};

/// An ordered list of command-output [`Block`]s for one terminal session.
///
/// Blocks are created and closed by feeding prompt-protocol markers to
/// [`BlockList::on_marker`]; at most one block (always the newest) is open
/// at a time. Blocks are ordered oldest → newest, which is also
/// non-decreasing [`Block::start_row`] and strictly ascending [`BlockId`]
/// order.
///
/// # Invariant
///
/// Terminal rows are a **finite, reusable** resource: when a prompt starts
/// at absolute row `R`, every row from `R` onward is about to be
/// overwritten on screen, so any older block's claim on those rows is
/// void. The list therefore maintains, at all times:
///
/// 1. **No overlaps.** Every absolute row is owned by at most one block.
/// 2. **Non-decreasing starts.** `blocks[i].start_row <=
///    blocks[i + 1].start_row`, so list order *is* row order — which is
///    what makes [`BlockList::next`] / [`BlockList::prev`] "jump to the
///    next/previous command on screen".
/// 3. **At most one open block**, and it is always the newest.
/// 4. **Strictly ascending ids**, never reused (so [`BlockList::get`] can
///    binary-search, and a pruned block's id simply stops resolving).
///
/// A block that is left owning **zero rows** and carrying no recorded
/// command is not a block at all — nothing can be selected, copied or
/// searched through it — and is dropped rather than retained forever.
///
/// Enforcement lives in exactly one place: every block is created through
/// the private `start_block_at`, which reclaims the rows first. Both
/// [`BlockList::on_marker`] and [`BlockList::observe_row`] go through it.
/// [`BlockList::assert_invariants`] checks all four properties and is
/// asserted after every mutation in debug builds.
///
/// # Heuristic fallback (no shell integration)
///
/// If [`BlockList::on_marker`] is never called — the shell emits no OSC 133
/// markers — [`BlockList::observe_row`] lazily creates a single implicit
/// [`BlockLifecycle::Running`] block starting at absolute row `0`, so all
/// output is still navigable as one block.
///
/// That block materializes on the first `observe_row` whose screen actually
/// holds output, **not** merely because a feed happened: a host's first PTY
/// reads routinely carry nothing but a partial escape sequence, and a block
/// created for those would be an empty phantom that the shell's real first
/// prompt immediately closes. See [`BlockList::observe_row`] for the exact
/// rule.
#[derive(Debug, Clone, Default)]
pub struct BlockList {
    blocks: Vec<Block>,
    next_id: u64,
    saw_marker: bool,
}

impl BlockList {
    /// Create an empty block list.
    pub fn new() -> Self {
        Self::default()
    }

    // -------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------

    /// Drive lifecycle from a prompt marker observed at the given absolute
    /// cursor row.
    ///
    /// Semantics (OSC 133 prompt protocol):
    ///
    /// - [`PromptMarker::PromptStart`] at row `R` reclaims every row from
    ///   `R` onward (see the type-level invariant) and starts a new
    ///   [`Prompt`](BlockLifecycle::Prompt) block at `start_row = R`. A
    ///   still-open block is closed at `end_row = R`; if it was still
    ///   [`Running`](BlockLifecycle::Running) — a command that never
    ///   reported `D` — it becomes [`Finished`](BlockLifecycle::Finished)
    ///   with `exit_code` left `None`. A repeated `PromptStart` at the same
    ///   row as the current prompt block (a shell redrawing its prompt) is
    ///   ignored.
    /// - [`PromptMarker::CommandStart`] moves the current block to
    ///   [`Composing`](BlockLifecycle::Composing).
    /// - [`PromptMarker::CommandExecuted`] moves the current block to
    ///   [`Running`](BlockLifecycle::Running).
    /// - [`PromptMarker::CommandFinished`] moves the current block to
    ///   [`Finished`](BlockLifecycle::Finished) and records the exit code.
    ///   The block's end stays open until the *next* `PromptStart` (Warp
    ///   semantics: the block visually includes the command's output, and
    ///   the next prompt starts the next block). With **no** open block it
    ///   creates nothing — see the comment on that arm.
    ///
    /// # Rows may go backwards
    ///
    /// `cursor_abs_row` is wherever the cursor happened to be, and nothing
    /// stops it from being *before* the open block's start: a `CUP` before
    /// the prompt, an alt-screen application emitting `OSC 133;A`, or a
    /// resize that shifts the cursor all do it. Such a marker is handled
    /// like any other — rows at or after `R` are reclaimed — so the list's
    /// invariant holds for every possible marker sequence, and blocks never
    /// come to overlap.
    ///
    /// Degenerate marker *orders* likewise never panic: a
    /// `PromptStart` / `CommandStart` / `CommandExecuted` with no open
    /// block creates one at `cursor_abs_row` already in the implied state,
    /// `CommandExecuted` without `CommandStart` simply jumps the block to
    /// `Running`, and a `CommandStart`/`CommandExecuted` arriving after
    /// `CommandFinished` closes the finished block and starts a new one.
    pub fn on_marker(&mut self, marker: PromptMarker, cursor_abs_row: usize) {
        self.retract_fallback_guess();
        self.saw_marker = true;
        match marker {
            PromptMarker::PromptStart => {
                if let Some(open) = self.open_block_mut() {
                    if open.lifecycle == BlockLifecycle::Prompt && open.start_row == cursor_abs_row
                    {
                        return; // prompt redraw at the same row
                    }
                }
                self.start_block_at(cursor_abs_row, BlockLifecycle::Prompt);
            }
            PromptMarker::CommandStart => match self.open_block_mut() {
                Some(open)
                    if matches!(
                        open.lifecycle,
                        BlockLifecycle::Prompt | BlockLifecycle::Composing
                    ) =>
                {
                    open.lifecycle = BlockLifecycle::Composing;
                }
                // B after C/D without a fresh prompt (or with nothing open
                // at all): start a new block.
                _ => self.start_block_at(cursor_abs_row, BlockLifecycle::Composing),
            },
            PromptMarker::CommandExecuted => match self.open_block_mut() {
                Some(open) if open.lifecycle != BlockLifecycle::Finished => {
                    open.lifecycle = BlockLifecycle::Running;
                }
                // C after D without a fresh prompt (or with nothing open at
                // all): start a new block.
                _ => self.start_block_at(cursor_abs_row, BlockLifecycle::Running),
            },
            PromptMarker::CommandFinished { exit_code } => match self.open_block_mut() {
                Some(open) => {
                    open.lifecycle = BlockLifecycle::Finished;
                    open.exit_code = exit_code;
                }
                None => {
                    // `D` with nothing open finishes no command, so it
                    // creates no block. This is not a corner case: the
                    // standard integration for zsh (`precmd`) and bash
                    // (`PROMPT_COMMAND`) emits `D` immediately before `A`,
                    // and those hooks also run before the FIRST prompt — so
                    // a real session's very first bytes are `OSC 133;D;0`
                    // with no command behind them. Recording it produced a
                    // phantom zero-width block reporting success, which a
                    // UI renders as an empty completed command at the top
                    // of every session.
                }
            },
        }
        debug_assert!(self.tail_is_sound(), "{:?}", self.blocks);
    }

    /// Retract the no-shell-integration fallback block when the first
    /// marker disproves the guess it was based on.
    ///
    /// The fallback is a guess made while no prompt protocol had been seen.
    /// The first marker proves the shell *does* speak OSC 133, so the guess
    /// must go rather than leave a phantom block for the real markers to
    /// act on. Without this, whether the phantom exists depends purely on
    /// how the host chunked its PTY reads: a read carrying the shell's
    /// startup output but no *complete* marker creates it, a read carrying
    /// both does not.
    ///
    /// # The exception: a fallback block the host gave a command
    ///
    /// Retracting unconditionally also threw away real, unrecoverable
    /// state: a shell that prints an MOTD or ssh banner before its first
    /// prompt had that output orphaned (`block_at_row(0)` went `None`, so
    /// the banner could not be selected, copied, or found by
    /// [`BlockList::search_output`]) along with any command the host had
    /// recorded on it. So the retraction is now conditional: a fallback
    /// block the host recorded a command for is *kept* and simply closed at
    /// the marker row, like any other block.
    ///
    /// The trade-off is deliberate and narrow. The tempting wider rule —
    /// "keep the fallback if it holds any non-blank row" — regresses the
    /// bug this retraction was added for: real zsh paints startup escape
    /// output including a visible `%` before its first marker, so a
    /// non-blank row is exactly what the phantom looks like. A
    /// host-recorded command is the one signal that cannot come from the
    /// shell's own preamble. The cost is that a banner *without* a recorded
    /// command is still dropped.
    fn retract_fallback_guess(&mut self) {
        if self.saw_marker {
            return;
        }
        let retract = match self.blocks.first() {
            Some(block) => self.blocks.len() == 1 && block.implicit && block.command.is_none(),
            None => false,
        };
        if retract {
            self.blocks.clear(); // `next_id` is untouched: ids are never reused.
        }
    }

    /// Record the command text for the composing/just-started block (the
    /// host captures it from the input line or shell integration).
    ///
    /// Applies to the newest block; no-op if the list is empty. This is the
    /// right call for a *live* host, which records the command the moment
    /// the user submits it — the newest block is then necessarily the one
    /// being composed. A host replaying or reattaching to a transcript can
    /// see several commands' markers in a single feed; it should use
    /// [`BlockList::set_command_at_row`] instead, which is independent of
    /// how the byte stream was chunked.
    pub fn set_command(&mut self, command: &str) {
        if let Some(block) = self.blocks.last_mut() {
            block.command = Some(command.to_owned());
        }
    }

    /// Record the command text for the block that owns absolute row
    /// `abs_row` (the row an `OSC 133;B` marker was observed at).
    ///
    /// Returns whether a block owned that row; `false` (a no-op) when no
    /// block covers it. Row ownership is inclusive of [`Block::start_row`]
    /// and exclusive of [`Block::end_row`], exactly like
    /// [`BlockList::block_at_row`].
    pub fn set_command_at_row(&mut self, abs_row: usize, command: &str) -> bool {
        match self.index_at_row(abs_row) {
            Some(index) => {
                if let Some(block) = self.blocks.get_mut(index) {
                    block.command = Some(command.to_owned());
                }
                true
            }
            None => false,
        }
    }

    /// Record the working directory for the current block (the host
    /// translates OSC 7).
    ///
    /// Applies to the newest block; no-op if the list is empty. See
    /// [`BlockList::set_cwd_at_row`] for the chunking-independent variant.
    pub fn set_cwd(&mut self, cwd: &str) {
        if let Some(block) = self.blocks.last_mut() {
            block.cwd = Some(cwd.to_owned());
        }
    }

    /// Record the working directory for the block that owns absolute row
    /// `abs_row` (the row an `OSC 7` report was observed at).
    ///
    /// Returns whether a block owned that row; `false` (a no-op) when no
    /// block covers it.
    pub fn set_cwd_at_row(&mut self, abs_row: usize, cwd: &str) -> bool {
        match self.index_at_row(abs_row) {
            Some(index) => {
                if let Some(block) = self.blocks.get_mut(index) {
                    block.cwd = Some(cwd.to_owned());
                }
                true
            }
            None => false,
        }
    }

    /// Inform the list that the terminal has advanced, so the open block's
    /// provisional extent tracks output growth. Hosts call this once after
    /// every parser feed, handing over the live screen.
    ///
    /// # When the implicit fallback block is created
    ///
    /// This call creates the "no shell integration" fallback block (see the
    /// type-level docs) when, and only when, **all three** hold:
    ///
    /// 1. no [`PromptMarker`] has ever been seen — the shell is not
    ///    speaking the OSC 133 prompt protocol;
    /// 2. the list is still empty; and
    /// 3. `screen` has produced output — some row of the primary grid is
    ///    non-blank, or rows have already moved into (or out of)
    ///    scrollback.
    ///
    /// Condition 3 is what stops a shell that *does* have integration from
    /// growing a phantom leading block. A host's first PTY read very often
    /// carries nothing but a partial escape sequence — a lone `ESC`, a
    /// bracketed-paste enable (`ESC [ ? 2 0 0 4 h`), or the first bytes of
    /// `ESC ] 1 3 3 ; A` — which prints nothing and completes no marker.
    /// Creating a block for it would manufacture an empty, zero-height
    /// block that the shell's real first prompt then immediately closes.
    /// Nothing about the *rule* depends on how the stream was chunked: the
    /// fallback appears on the first feed that puts text on the screen, and
    /// never appears at all once a marker has been seen.
    pub fn observe_row(&mut self, screen: &Screen) {
        if self.blocks.is_empty() && !self.saw_marker && Self::has_output(screen) {
            self.start_block_at(0, BlockLifecycle::Running);
            if let Some(block) = self.blocks.last_mut() {
                block.implicit = true;
            }
            debug_assert!(self.tail_is_sound(), "{:?}", self.blocks);
        }
    }

    // -------------------------------------------------------------------
    // Queries and navigation
    // -------------------------------------------------------------------

    /// Number of blocks currently held.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the list holds no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Iterate over all blocks, oldest first.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter()
    }

    /// Look up a block by id.
    pub fn get(&self, id: BlockId) -> Option<&Block> {
        self.index_of(id).and_then(|i| self.blocks.get(i))
    }

    /// The newest block (the open one, if the list is non-empty).
    pub fn latest(&self) -> Option<&Block> {
        self.blocks.last()
    }

    /// The block containing the given absolute row, if any: inclusive of
    /// [`Block::start_row`], exclusive of [`Block::end_row`]. A still-open
    /// block owns every row from its start onward.
    pub fn block_at_row(&self, abs_row: usize) -> Option<&Block> {
        self.index_at_row(abs_row).and_then(|i| self.blocks.get(i))
    }

    /// The block after the one with `id` (Warp-style jump to the next
    /// command). `None` if `id` is unknown or already the last block.
    ///
    /// List order is row order — the invariant keeps [`Block::start_row`]
    /// non-decreasing — so "next" always moves *down* the screen and
    /// "previous" always moves up, whatever the shell did to the cursor.
    pub fn next(&self, id: BlockId) -> Option<&Block> {
        self.index_of(id).and_then(|i| self.blocks.get(i + 1))
    }

    /// The block before the one with `id` (Warp-style jump to the previous
    /// command). `None` if `id` is unknown or already the first block. See
    /// [`BlockList::next`] for the ordering guarantee.
    pub fn prev(&self, id: BlockId) -> Option<&Block> {
        self.index_of(id)
            .and_then(|i| i.checked_sub(1))
            .and_then(|i| self.blocks.get(i))
    }

    /// The oldest block.
    pub fn first(&self) -> Option<&Block> {
        self.blocks.first()
    }

    /// The newest block.
    pub fn last(&self) -> Option<&Block> {
        self.blocks.last()
    }

    // -------------------------------------------------------------------
    // Search and text extraction
    // -------------------------------------------------------------------

    /// Blocks whose stored metadata (command / cwd) contains `query`,
    /// newest first.
    ///
    /// Matching is a case-insensitive substring test — see
    /// [`BlockList::search_output`] for the exact folding rule and for the
    /// shared empty-query semantics (an empty query matches **every**
    /// block, including blocks with no metadata at all).
    pub fn search_metadata(&self, query: &str) -> Vec<&Block> {
        let needle = fold_case(query);
        self.blocks
            .iter()
            .rev()
            .filter(|b| {
                needle.is_empty()
                    || b.command
                        .as_deref()
                        .is_some_and(|c| fold_case(c).contains(&needle))
                    || b.cwd
                        .as_deref()
                        .is_some_and(|c| fold_case(c).contains(&needle))
            })
            .collect()
    }

    /// Full-text search over block *output* via the screen
    /// (case-insensitive substring), newest first. Rows evicted from
    /// scrollback are silently skipped.
    ///
    /// # Empty query
    ///
    /// `""` is a substring of every string, so an empty query matches
    /// **every** block — here and in [`BlockList::search_metadata`], which
    /// used to disagree by also requiring a command or cwd to be present.
    /// Callers that want "blocks that have metadata" should filter on
    /// [`Block::command`] / [`Block::cwd`] themselves; a search function
    /// must not smuggle in a second predicate.
    ///
    /// # Case insensitivity
    ///
    /// Both sides are folded with [`str::to_uppercase`] followed by
    /// [`str::to_lowercase`], which is a closer approximation of Unicode
    /// case folding than lowercasing alone: it makes `ß`/`SS`, the `ﬁ`
    /// ligature/`FI`, and final vs. medial sigma compare equal. It is
    /// **not** true case folding, and two limitations are worth naming:
    ///
    /// - Characters whose lowercase form gains a combining mark still do
    ///   not match their unmarked form. `İ` (U+0130) lowercases to
    ///   `i` + U+0307, so `İstanbul` matches `İSTANBUL` but not
    ///   `istanbul`. Fixing that needs Unicode normalization, which needs
    ///   a dependency this crate does not take.
    /// - The round trip is slightly *more* permissive than case folding in
    ///   Turkic scripts: dotless `ı` uppercases to `I`, so it matches `i`.
    ///   For a search box, over-matching is the safer error.
    pub fn search_output<'a>(&'a self, screen: &Screen, query: &str) -> Vec<&'a Block> {
        let needle = fold_case(query);
        self.blocks
            .iter()
            .rev()
            .filter(|b| fold_case(&Self::collect_text(b, screen)).contains(&needle))
            .collect()
    }

    /// Extract a block's visible text from the screen: rows joined with
    /// `\n`, rows evicted from scrollback silently skipped. A still-open
    /// block extends to the screen's last addressable row. Returns `None`
    /// only when `id` is unknown.
    pub fn block_text(&self, screen: &Screen, id: BlockId) -> Option<String> {
        self.get(id).map(|b| Self::collect_text(b, screen))
    }

    /// Drop blocks whose rows have been entirely evicted from scrollback.
    /// Returns the number of blocks pruned. Still-open blocks are never
    /// pruned.
    ///
    /// Note that this can only reclaim blocks whose rows have *aged out*.
    /// It is not the answer to a block list growing without bound while
    /// the rows are still on screen — the list keeps itself free of
    /// row-less junk itself (see the invariant).
    pub fn prune_evicted(&mut self, screen: &Screen) -> usize {
        let evicted = screen.scrollback().evicted();
        let before = self.blocks.len();
        self.blocks.retain(|b| match b.end {
            // Keep unless every row (start..end, exclusive end) is evicted.
            // A zero-row block sitting exactly on the eviction frontier
            // (`start == end == evicted`) is deliberately kept: it marks a
            // boundary that has not aged out yet.
            Some(end) => end > evicted || b.start_row >= evicted,
            None => true,
        });
        debug_assert!(self.tail_is_sound(), "{:?}", self.blocks);
        before - self.blocks.len()
    }

    // -------------------------------------------------------------------
    // Internals
    // -------------------------------------------------------------------

    /// Start a new block at absolute row `start_row` — **the only way a
    /// block is ever created**, and therefore the single place the
    /// type-level invariant is enforced.
    ///
    /// Reclaims rows `start_row..` from the blocks that currently claim
    /// them (see [`BlockList::reclaim_rows_from`]) before pushing, so the
    /// new block's `[start_row, ∞)` claim can never overlap an older one
    /// and starts stay non-decreasing.
    fn start_block_at(&mut self, start_row: usize, lifecycle: BlockLifecycle) {
        self.reclaim_rows_from(start_row);
        let id = BlockId(self.next_id);
        self.next_id += 1;
        self.blocks.push(Block::new(id, start_row, lifecycle));
    }

    /// Void every existing claim on absolute rows `row..`.
    ///
    /// Those rows are about to be written over on screen, so no earlier
    /// block can still own them:
    ///
    /// - a block straddling `row` is closed at `end_row = row`;
    /// - a block lying entirely at or after `row` loses everything it ever
    ///   owned and collapses onto `row` as a zero-row husk (which keeps
    ///   starts non-decreasing);
    /// - a block left owning zero rows and carrying no recorded command is
    ///   dropped — it can never again be found by row, holds no text, and
    ///   would otherwise be scanned forever. Retaining it *with* a command
    ///   is what preserves a real command that happened to print nothing.
    ///
    /// A block that was still [`Running`](BlockLifecycle::Running) becomes
    /// [`Finished`](BlockLifecycle::Finished) with no exit code — its
    /// command never reported `D`.
    ///
    /// Costs `O(log n)` in the overwhelmingly common case where the new row
    /// is at or past every existing end: block ends are non-decreasing
    /// (a consequence of the invariant), so the affected blocks are a
    /// suffix found by binary search, and only that suffix is touched.
    fn reclaim_rows_from(&mut self, row: usize) {
        // First block whose claim reaches past `row`. `None` (still open)
        // sorts last, exactly as the ∞ it stands for.
        let first = self
            .blocks
            .partition_point(|b| b.end.is_some_and(|end| end <= row));
        for block in self.blocks.iter_mut().skip(first) {
            block.start_row = block.start_row.min(row);
            block.end = Some(row);
            if block.lifecycle == BlockLifecycle::Running {
                block.lifecycle = BlockLifecycle::Finished;
            }
        }
        // Compact the touched suffix in place, dropping the husks. Only the
        // suffix moves, so this stays proportional to what was reclaimed.
        let len = self.blocks.len();
        let mut keep = first;
        for index in first..len {
            let worth_keeping = match self.blocks.get(index) {
                Some(block) => block.end != Some(block.start_row) || block.command.is_some(),
                None => false,
            };
            if worth_keeping {
                self.blocks.swap(keep, index);
                keep += 1;
            }
        }
        self.blocks.truncate(keep);
    }

    /// Whether `screen` has produced any output at all: some row of the
    /// primary grid is non-blank, or rows have already moved into (or out
    /// of) scrollback. Bounded by the grid's size — scrollback is answered
    /// by its length alone, never scanned.
    fn has_output(screen: &Screen) -> bool {
        let scrollback = screen.scrollback();
        if !scrollback.is_empty() || scrollback.evicted() > 0 {
            return true;
        }
        // Equivalent to "some row's `row_text` is non-empty" — a row's
        // text is non-empty exactly when it holds a non-space character —
        // but scans cells directly. This runs on *every* feed until the
        // first output appears, so it must not allocate a String per row
        // per feed the way `row_text` does.
        let grid = screen.primary_grid();
        (0..grid.rows()).any(|row| {
            grid.row(row)
                .is_some_and(|cells| cells.iter().any(|cell| cell.ch != ' '))
        })
    }

    /// Index of the block owning `abs_row`, if any.
    ///
    /// `O(log n)`: starts are non-decreasing, so the owner — if there is
    /// one — is the newest block starting at or before `abs_row`, which a
    /// binary search finds directly. The old linear scan was worst-case on
    /// every call, because the row being looked up is almost always in the
    /// *newest* block while the scan started from the oldest; a host
    /// resolving one `OSC 7` per prompt therefore paid `O(blocks²)` over a
    /// session.
    ///
    /// Resolving from the newest backwards is also defence in depth: if the
    /// list ever did hold overlapping claims, the row belongs to whatever
    /// most recently wrote it, never to a stale block.
    fn index_at_row(&self, abs_row: usize) -> Option<usize> {
        let past = self.blocks.partition_point(|b| b.start_row <= abs_row);
        for index in (0..past).rev() {
            let block = self.blocks.get(index)?;
            if block.contains_row(abs_row) {
                return Some(index);
            }
            // `block` starts at or before `abs_row` and does not own it, so
            // it is closed with `end <= abs_row`. Unless it owns no rows at
            // all, every older block ends no later than its start, so none
            // of them can own `abs_row` either — stop. Only a run of
            // zero-row husks is worth stepping over.
            if block.end != Some(block.start_row) {
                return None;
            }
        }
        None
    }

    /// The open block, if any. Invariant: only the newest block can be
    /// open (every earlier block was closed when its successor started).
    fn open_block_mut(&mut self) -> Option<&mut Block> {
        self.blocks.last_mut().filter(|b| b.end.is_none())
    }

    /// The `O(1)` slice of the invariant that a mutation can actually
    /// break: every mutation here only appends to, or truncates, the tail,
    /// so checking the newest pair catches a broken enforcement path
    /// without making debug builds quadratic. The full check is
    /// [`BlockList::assert_invariants`].
    fn tail_is_sound(&self) -> bool {
        let len = self.blocks.len();
        let (previous, last) = match (
            len.checked_sub(2).and_then(|i| self.blocks.get(i)),
            self.blocks.last(),
        ) {
            (_, None) => return true,
            (None, Some(last)) => return last.end.map_or(true, |end| end >= last.start_row),
            (Some(previous), Some(last)) => (previous, last),
        };
        previous.id < last.id
            && previous.start_row <= last.start_row
            && previous.end.is_some_and(|end| end <= last.start_row)
            && last.end.map_or(true, |end| end >= last.start_row)
    }

    /// Check the type-level invariant, naming the first violation.
    fn check_invariants(&self) -> Result<(), String> {
        let mut previous: Option<&Block> = None;
        for (index, block) in self.blocks.iter().enumerate() {
            if block.end.is_none() && index + 1 != self.blocks.len() {
                return Err(format!("block {} is open but is not the newest", block.id));
            }
            if let Some(end) = block.end {
                if end < block.start_row {
                    return Err(format!("block {} ends before it starts", block.id));
                }
            }
            if let Some(prev) = previous {
                if prev.id >= block.id {
                    return Err(format!("ids not ascending at {} / {}", prev.id, block.id));
                }
                if prev.start_row > block.start_row {
                    return Err(format!(
                        "start rows go backwards at {} / {}",
                        prev.id, block.id
                    ));
                }
                // `prev` is closed (checked above); it must not reach into
                // `block`.
                if prev.end.map_or(true, |end| end > block.start_row) {
                    return Err(format!("blocks {} and {} overlap", prev.id, block.id));
                }
            }
            previous = Some(block);
        }
        Ok(())
    }

    /// Panic unless the type-level invariant holds. Test/debug helper —
    /// hosts never need it, and it is asserted internally after every
    /// mutation in debug builds.
    #[doc(hidden)]
    pub fn assert_invariants(&self) {
        if let Err(why) = self.check_invariants() {
            panic!(
                "BlockList invariant violated: {why}; blocks: {:?}",
                self.blocks
            );
        }
    }

    /// The half-open absolute-row range a block's text spans: its closed
    /// extent, or — while still growing — up to the screen's last
    /// addressable row.
    fn text_range(block: &Block, screen: &Screen) -> (usize, usize) {
        let end = block
            .end
            .unwrap_or_else(|| screen.scrollback().evicted() + screen.total_rows());
        (block.start_row, end.max(block.start_row))
    }

    fn collect_text(block: &Block, screen: &Screen) -> String {
        let (start, end) = Self::text_range(block, screen);
        let mut lines: Vec<String> = Vec::new();
        for abs_row in start..end {
            if let Some(text) = screen.row_text(abs_row) {
                lines.push(text);
            }
        }
        // An OPEN block extends to the screen's last addressable row, so it
        // would otherwise pick up one empty line per not-yet-written row —
        // roughly a full screen of blank lines for a freshly started
        // command. Those rows hold no output, so drop the trailing blank
        // run. Interior blank lines are real output and are preserved.
        while lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }
        lines.join("\n")
    }

    fn index_of(&self, id: BlockId) -> Option<usize> {
        // Ids are allocated monotonically as blocks are pushed, so the vec
        // is sorted by id even after pruning.
        self.blocks.binary_search_by_key(&id, |b| b.id).ok()
    }
}

/// Fold `text` for case-insensitive comparison.
///
/// `str::to_lowercase` alone is *not* case folding: it leaves `ß` and `SS`
/// different, and the same for the `ﬁ` ligature and `FI`. Uppercasing
/// first expands those multi-character mappings, and lowercasing the
/// result normalises what the round trip produced. The residual
/// limitations are documented on [`BlockList::search_output`].
fn fold_case(text: &str) -> String {
    text.to_uppercase().to_lowercase()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // -- Marker-only helpers (no screen needed) --------------------------

    /// Run a full A/B/C/D command: prompt at `prompt_row`, finish at
    /// `finish_row`, leaving the block open (awaiting the next prompt).
    fn run_command(
        bl: &mut BlockList,
        prompt_row: usize,
        command: &str,
        exit: i32,
        finish_row: usize,
    ) {
        bl.on_marker(PromptMarker::PromptStart, prompt_row);
        bl.on_marker(PromptMarker::CommandStart, prompt_row);
        bl.set_command(command);
        bl.on_marker(PromptMarker::CommandExecuted, prompt_row + 1);
        bl.on_marker(
            PromptMarker::CommandFinished {
                exit_code: Some(exit),
            },
            finish_row,
        );
    }

    // -- Screen helpers ---------------------------------------------------

    /// Print `text` on the current row, then CR + LF.
    fn feed_line(screen: &mut Screen, text: &str) {
        for ch in text.chars() {
            screen.print(ch);
        }
        screen.carriage_return();
        screen.linefeed();
    }

    /// A screen whose cursor sits on absolute row `row`, with content on
    /// every row above it — the shape a real terminal has once output has
    /// scrolled that far.
    fn screen_at_row(row: usize) -> Screen {
        let mut screen = Screen::new(row + 1, 20);
        for r in 0..row {
            feed_line(&mut screen, &format!("row{r}"));
        }
        assert_eq!(screen.cursor_absolute_row(), row);
        screen
    }

    /// A screen that has never been written to.
    fn blank_screen() -> Screen {
        Screen::new(4, 20)
    }

    // -- Lifecycle ---------------------------------------------------------

    #[test]
    fn full_lifecycle_happy_path() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        let block = bl.latest().unwrap();
        assert_eq!(block.lifecycle(), BlockLifecycle::Prompt);
        assert_eq!(block.start_row(), 0);
        assert_eq!(block.end_row(), None);

        bl.on_marker(PromptMarker::CommandStart, 0);
        bl.set_command("cargo build");
        bl.set_cwd("/home/me/project");
        assert_eq!(bl.latest().unwrap().lifecycle(), BlockLifecycle::Composing);

        bl.on_marker(PromptMarker::CommandExecuted, 1);
        assert_eq!(bl.latest().unwrap().lifecycle(), BlockLifecycle::Running);

        bl.observe_row(&screen_at_row(4));
        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(0) }, 5);
        let block = bl.latest().unwrap();
        assert_eq!(block.lifecycle(), BlockLifecycle::Finished);
        assert_eq!(block.exit_code(), Some(0));
        assert_eq!(block.succeeded(), Some(true));
        // End stays open until the next prompt (Warp semantics).
        assert_eq!(block.end_row(), None);

        bl.on_marker(PromptMarker::PromptStart, 6);
        assert_eq!(bl.len(), 2);
        let first = bl.first().unwrap();
        assert_eq!(first.end_row(), Some(6));
        assert_eq!(first.command(), Some("cargo build"));
        assert_eq!(first.cwd(), Some("/home/me/project"));
        let latest = bl.latest().unwrap();
        assert_eq!(latest.lifecycle(), BlockLifecycle::Prompt);
        assert_eq!(latest.start_row(), 6);
    }

    #[test]
    fn multiple_sequential_commands_chain_contiguously() {
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "first", 0, 2);
        run_command(&mut bl, 3, "second", 1, 7);
        run_command(&mut bl, 8, "third", 0, 9);
        assert_eq!(bl.len(), 3);

        let blocks: Vec<&Block> = bl.blocks().collect();
        assert_eq!(blocks[0].command(), Some("first"));
        assert_eq!(blocks[1].command(), Some("second"));
        assert_eq!(blocks[2].command(), Some("third"));
        // Each block ends exactly where its successor starts.
        assert_eq!(blocks[0].end_row(), Some(blocks[1].start_row()));
        assert_eq!(blocks[1].end_row(), Some(blocks[2].start_row()));
        assert_eq!(blocks[2].end_row(), None);
    }

    #[test]
    fn command_that_never_finishes_is_closed_by_next_prompt() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::CommandStart, 0);
        bl.set_command("sleep 999");
        bl.on_marker(PromptMarker::CommandExecuted, 1);
        // No CommandFinished; the shell draws a new prompt (e.g. Ctrl-C).
        bl.on_marker(PromptMarker::PromptStart, 4);

        let first = bl.first().unwrap();
        assert_eq!(first.lifecycle(), BlockLifecycle::Finished);
        assert_eq!(first.end_row(), Some(4));
        assert_eq!(first.exit_code(), None);
        assert_eq!(first.succeeded(), None);
    }

    #[test]
    fn prompt_only_block_closed_without_becoming_finished() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::PromptStart, 1); // user just hit enter
        let first = bl.first().unwrap();
        assert_eq!(first.lifecycle(), BlockLifecycle::Prompt);
        assert_eq!(first.end_row(), Some(1));
    }

    #[test]
    fn exit_codes_map_to_success_and_failure() {
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "true", 0, 1);
        run_command(&mut bl, 2, "false", 1, 3);
        bl.on_marker(PromptMarker::PromptStart, 4);
        bl.on_marker(PromptMarker::CommandExecuted, 4);
        bl.on_marker(PromptMarker::CommandFinished { exit_code: None }, 5);

        let blocks: Vec<&Block> = bl.blocks().collect();
        assert_eq!(blocks[0].succeeded(), Some(true));
        assert_eq!(blocks[1].succeeded(), Some(false));
        assert_eq!(blocks[1].exit_code(), Some(1));
        assert_eq!(blocks[2].succeeded(), None);
    }

    // -- Degenerate marker orders -------------------------------------------

    #[test]
    fn finished_without_prompt_creates_no_block() {
        // `D` with nothing open finishes no command. Previously this
        // manufactured a zero-width Finished block, which real shells
        // trigger on their very first bytes (see the regression test
        // below), putting a phantom empty block at the top of every
        // session.
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(3) }, 7);
        assert!(bl.is_empty());
    }

    #[test]
    fn first_marker_retracts_a_fallback_block_created_before_it() {
        // A host reading the PTY in small chunks sees the shell's startup
        // output (zsh paints an inverse-video marker and a line of spaces)
        // BEFORE any complete OSC 133 sequence arrives, which materializes
        // the no-integration fallback block. The first real marker proves
        // the shell does have integration, so that guess must be retracted
        // — otherwise the block list's contents would depend on how the
        // host happened to chunk its reads.
        let mut screen = Screen::new(24, 80);
        let mut bl = BlockList::new();
        for ch in "startup noise".chars() {
            screen.print(ch);
        }
        bl.observe_row(&screen);
        assert_eq!(bl.len(), 1, "fallback block should exist before markers");

        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(0) }, 0);
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::CommandExecuted, 1);

        assert_eq!(bl.len(), 1, "the retracted guess must not linger");
        let block = bl.latest().unwrap();
        assert_eq!(block.lifecycle(), BlockLifecycle::Running);
        assert_eq!(
            block.exit_code(),
            None,
            "the preamble's exit code belongs to no command"
        );
    }

    #[test]
    fn real_shell_integration_preamble_creates_no_phantom_block() {
        // Captured from a real zsh session: `precmd` runs before the FIRST
        // prompt as well as after each command, so the session opens with
        // `D` (from no command at all) immediately followed by `A`. Bash's
        // PROMPT_COMMAND integration has the same shape.
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(0) }, 0);
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::CommandStart, 0);
        bl.on_marker(PromptMarker::CommandExecuted, 1);
        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(0) }, 2);

        // Exactly one block: the command that actually ran.
        assert_eq!(bl.len(), 1);
        let block = bl.latest().unwrap();
        assert_eq!(block.start_row(), 0);
        assert_eq!(block.exit_code(), Some(0));
        assert_eq!(block.lifecycle(), BlockLifecycle::Finished);
    }

    #[test]
    fn executed_without_command_start_goes_straight_to_running() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::CommandExecuted, 1); // C without B
        assert_eq!(bl.len(), 1);
        assert_eq!(bl.latest().unwrap().lifecycle(), BlockLifecycle::Running);
    }

    #[test]
    fn command_start_without_prompt_creates_composing_block() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::CommandStart, 2);
        assert_eq!(bl.len(), 1);
        let block = bl.latest().unwrap();
        assert_eq!(block.lifecycle(), BlockLifecycle::Composing);
        assert_eq!(block.start_row(), 2);
    }

    #[test]
    fn executed_without_any_block_creates_running_block() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::CommandExecuted, 5);
        assert_eq!(bl.len(), 1);
        assert_eq!(bl.latest().unwrap().lifecycle(), BlockLifecycle::Running);
    }

    #[test]
    fn repeated_prompt_start_at_same_row_is_deduplicated() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::PromptStart, 0);
        assert_eq!(bl.len(), 1);
        assert_eq!(bl.latest().unwrap().start_row(), 0);
        assert_eq!(bl.latest().unwrap().end_row(), None);
    }

    #[test]
    fn command_start_after_finished_starts_new_block() {
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "ls", 0, 1);
        bl.on_marker(PromptMarker::CommandStart, 2); // B with no fresh A
        assert_eq!(bl.len(), 2);
        assert_eq!(bl.first().unwrap().end_row(), Some(2));
        let latest = bl.latest().unwrap();
        assert_eq!(latest.lifecycle(), BlockLifecycle::Composing);
        assert_eq!(latest.start_row(), 2);
    }

    #[test]
    fn executed_after_finished_starts_new_block() {
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "ls", 0, 1);
        bl.on_marker(PromptMarker::CommandExecuted, 2); // C with no fresh A
        assert_eq!(bl.len(), 2);
        assert_eq!(bl.latest().unwrap().lifecycle(), BlockLifecycle::Running);
    }

    #[test]
    fn repeated_finished_updates_exit_code() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(1) }, 1);
        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(0) }, 2);
        assert_eq!(bl.len(), 1);
        assert_eq!(bl.latest().unwrap().exit_code(), Some(0));
    }

    #[test]
    fn set_command_and_cwd_on_empty_list_are_noops() {
        let mut bl = BlockList::new();
        bl.set_command("ls");
        bl.set_cwd("/tmp");
        assert!(bl.is_empty());
    }

    // -- Implicit-block heuristic ------------------------------------------

    #[test]
    fn observe_row_without_markers_creates_single_implicit_block() {
        let mut bl = BlockList::new();
        assert!(bl.is_empty());
        bl.observe_row(&screen_at_row(3));
        bl.observe_row(&screen_at_row(9));
        assert_eq!(bl.len(), 1);
        let block = bl.latest().unwrap();
        assert_eq!(block.start_row(), 0);
        assert_eq!(block.lifecycle(), BlockLifecycle::Running);
        assert_eq!(block.end_row(), None);
        assert!(bl.block_at_row(0).is_some());
        assert!(bl.block_at_row(9).is_some());
    }

    #[test]
    fn observe_row_on_a_blank_screen_creates_nothing() {
        // The host fed bytes that printed nothing (a partial escape
        // sequence). No block may be manufactured for that.
        let mut bl = BlockList::new();
        bl.observe_row(&blank_screen());
        bl.observe_row(&blank_screen());
        assert!(bl.is_empty(), "a blank screen must not create a block");

        // The very first printed character does create it.
        let mut screen = blank_screen();
        screen.print('$');
        bl.observe_row(&screen);
        assert_eq!(bl.len(), 1);
        assert_eq!(bl.latest().unwrap().start_row(), 0);
    }

    #[test]
    fn observe_row_treats_scrolled_content_as_output() {
        // Content that has already scrolled off the grid still counts,
        // even though every visible row is now blank.
        let mut screen = Screen::with_scrollback(2, 20, 8);
        feed_line(&mut screen, "hello");
        feed_line(&mut screen, "");
        feed_line(&mut screen, "");
        assert!(!screen.scrollback().is_empty());

        let mut bl = BlockList::new();
        bl.observe_row(&screen);
        assert_eq!(bl.len(), 1);
    }

    #[test]
    fn observe_row_after_markers_does_not_create_extra_blocks() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 2);
        bl.observe_row(&screen_at_row(5));
        assert_eq!(bl.len(), 1);
        assert_eq!(bl.latest().unwrap().start_row(), 2);
    }

    #[test]
    fn observe_row_never_creates_the_fallback_once_a_marker_was_seen() {
        // A marker arrived and then its block was pruned away: the list is
        // empty again, but the shell is known to speak OSC 133, so the
        // fallback must stay retired.
        let mut screen = Screen::with_scrollback(2, 20, 1);
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::PromptStart, 1);
        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(0) }, 1);
        while screen.scrollback().evicted() < 4 {
            feed_line(&mut screen, "x");
        }
        bl.blocks.clear(); // simulate every block being pruned away
        bl.observe_row(&screen);
        assert!(bl.is_empty());
    }

    // -- Row-addressed metadata ---------------------------------------------

    #[test]
    fn set_command_and_cwd_at_row_target_the_owning_block() {
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "", 0, 1);
        bl.on_marker(PromptMarker::PromptStart, 4); // closes block 0 at [0,4)
        run_command(&mut bl, 4, "", 0, 5);
        bl.on_marker(PromptMarker::PromptStart, 9); // closes block 1 at [4,9)

        assert!(bl.set_command_at_row(0, "first"));
        assert!(bl.set_cwd_at_row(3, "/a"));
        assert!(bl.set_command_at_row(4, "second"));
        assert!(bl.set_cwd_at_row(8, "/b"));
        assert!(bl.set_command_at_row(9, "third")); // the open block
                                                    // An open block owns every row from its start onward.
        assert!(bl.set_command_at_row(usize::MAX, "third"));

        let blocks: Vec<&Block> = bl.blocks().collect();
        assert_eq!(blocks[0].command(), Some("first"));
        assert_eq!(blocks[0].cwd(), Some("/a"));
        assert_eq!(blocks[1].command(), Some("second"));
        assert_eq!(blocks[1].cwd(), Some("/b"));
        assert_eq!(blocks[2].command(), Some("third"));
        assert_eq!(blocks[2].cwd(), None);
    }

    #[test]
    fn set_command_and_cwd_at_unowned_row_are_noops() {
        let mut bl = BlockList::new();
        assert!(!bl.set_command_at_row(0, "ls"));
        assert!(!bl.set_cwd_at_row(0, "/tmp"));
        assert!(bl.is_empty());

        bl.on_marker(PromptMarker::PromptStart, 5);
        // Rows before the first block belong to nobody.
        assert!(!bl.set_command_at_row(4, "ls"));
        assert!(!bl.set_cwd_at_row(0, "/tmp"));
        assert_eq!(bl.latest().unwrap().command(), None);
        assert_eq!(bl.latest().unwrap().cwd(), None);
    }

    // -- block_at_row boundaries --------------------------------------------

    #[test]
    fn block_at_row_is_inclusive_start_exclusive_end() {
        let mut bl = BlockList::new();
        run_command(&mut bl, 2, "a", 0, 4);
        bl.on_marker(PromptMarker::PromptStart, 5); // closes [2, 5)
        let first_id = bl.first().unwrap().id();
        let second_id = bl.latest().unwrap().id();

        assert!(bl.block_at_row(0).is_none());
        assert!(bl.block_at_row(1).is_none());
        assert_eq!(bl.block_at_row(2).unwrap().id(), first_id); // inclusive
        assert_eq!(bl.block_at_row(4).unwrap().id(), first_id);
        assert_eq!(bl.block_at_row(5).unwrap().id(), second_id); // exclusive
    }

    #[test]
    fn open_block_owns_all_rows_from_its_start() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 3);
        let id = bl.latest().unwrap().id();
        assert!(bl.block_at_row(2).is_none());
        assert_eq!(bl.block_at_row(3).unwrap().id(), id);
        assert_eq!(bl.block_at_row(1_000).unwrap().id(), id);
    }

    // -- Navigation ----------------------------------------------------------

    #[test]
    fn navigation_next_prev_first_last() {
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "a", 0, 1);
        run_command(&mut bl, 2, "b", 0, 3);
        run_command(&mut bl, 4, "c", 0, 5);

        let first = bl.first().unwrap().id();
        let last = bl.last().unwrap().id();
        assert_eq!(bl.first().unwrap().command(), Some("a"));
        assert_eq!(bl.last().unwrap().command(), Some("c"));

        let mid = bl.next(first).unwrap();
        assert_eq!(mid.command(), Some("b"));
        assert_eq!(bl.next(mid.id()).unwrap().id(), last);
        assert!(bl.next(last).is_none());

        assert_eq!(bl.prev(last).unwrap().command(), Some("b"));
        assert_eq!(bl.prev(mid.id()).unwrap().id(), first);
        assert!(bl.prev(first).is_none());

        assert!(bl.get(BlockId(999)).is_none());
        assert!(bl.next(BlockId(999)).is_none());
        assert!(bl.prev(BlockId(999)).is_none());
        assert_eq!(bl.get(first).unwrap().id(), first);
        assert_eq!(bl.latest().unwrap().id(), last);
    }

    #[test]
    fn empty_list_queries() {
        let bl = BlockList::default();
        assert_eq!(bl.len(), 0);
        assert!(bl.is_empty());
        assert!(bl.first().is_none());
        assert!(bl.last().is_none());
        assert!(bl.latest().is_none());
        assert!(bl.block_at_row(0).is_none());
        assert_eq!(bl.blocks().count(), 0);
        assert!(bl.search_metadata("x").is_empty());
    }

    // -- Metadata search -------------------------------------------------------

    #[test]
    fn search_metadata_matches_command_and_cwd_newest_first() {
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "cargo build", 0, 1);
        bl.set_cwd("/repo/engine");
        run_command(&mut bl, 2, "ls -la", 0, 3);
        run_command(&mut bl, 4, "cargo test", 0, 5);

        let hits = bl.search_metadata("cargo");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].command(), Some("cargo test")); // newest first
        assert_eq!(hits[1].command(), Some("cargo build"));

        // Case-insensitive.
        assert_eq!(bl.search_metadata("CARGO").len(), 2);
        // Matches cwd too.
        let cwd_hits = bl.search_metadata("engine");
        assert_eq!(cwd_hits.len(), 1);
        assert_eq!(cwd_hits[0].cwd(), Some("/repo/engine"));
        // No match.
        assert!(bl.search_metadata("docker").is_empty());
    }

    // -- Output search / text extraction against a real Screen ------------------

    /// Drive a real Screen through two commands with output, mirroring how
    /// session glue would interleave markers, prints, and observe_row.
    fn screen_with_two_commands() -> (Screen, BlockList) {
        let mut screen = Screen::new(6, 30);
        let mut bl = BlockList::new();

        bl.on_marker(PromptMarker::PromptStart, screen.cursor_absolute_row());
        bl.on_marker(PromptMarker::CommandStart, screen.cursor_absolute_row());
        bl.set_command("echo Hello");
        feed_line(&mut screen, "$ echo Hello");
        bl.on_marker(PromptMarker::CommandExecuted, screen.cursor_absolute_row());
        feed_line(&mut screen, "Hello world");
        bl.observe_row(&screen);
        bl.on_marker(
            PromptMarker::CommandFinished { exit_code: Some(0) },
            screen.cursor_absolute_row(),
        );

        bl.on_marker(PromptMarker::PromptStart, screen.cursor_absolute_row());
        bl.on_marker(PromptMarker::CommandStart, screen.cursor_absolute_row());
        bl.set_command("rg needle");
        feed_line(&mut screen, "$ rg needle");
        bl.on_marker(PromptMarker::CommandExecuted, screen.cursor_absolute_row());
        feed_line(&mut screen, "src/lib.rs: needle found");
        bl.observe_row(&screen);
        bl.on_marker(
            PromptMarker::CommandFinished { exit_code: Some(1) },
            screen.cursor_absolute_row(),
        );

        (screen, bl)
    }

    #[test]
    fn search_output_finds_text_case_insensitively_newest_first() {
        let (screen, bl) = screen_with_two_commands();
        assert_eq!(bl.len(), 2);

        let hits = bl.search_output(&screen, "needle");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].command(), Some("rg needle"));

        // Case-insensitive: output says "Hello world".
        let hits = bl.search_output(&screen, "hello WORLD");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].command(), Some("echo Hello"));

        // "$" appears in both command echo lines; newest first.
        let hits = bl.search_output(&screen, "$");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].command(), Some("rg needle"));
        assert_eq!(hits[1].command(), Some("echo Hello"));

        assert!(bl.search_output(&screen, "no such output").is_empty());
    }

    #[test]
    fn block_text_extracts_joined_rows() {
        let (screen, bl) = screen_with_two_commands();
        let first_id = bl.first().unwrap().id();
        let text = bl.block_text(&screen, first_id).unwrap();
        assert_eq!(text, "$ echo Hello\nHello world");

        // An open block spans to the screen's last addressable row, but the
        // not-yet-written rows must not show up as trailing empty lines.
        let last_id = bl.last().unwrap().id();
        let text = bl.block_text(&screen, last_id).unwrap();
        assert_eq!(text, "$ rg needle\nsrc/lib.rs: needle found");

        assert!(bl.block_text(&screen, BlockId(42)).is_none());
    }

    #[test]
    fn open_block_text_has_no_trailing_blank_lines() {
        // A freshly started command on a tall screen: the open block reaches
        // the last row, so every unwritten row below the output would
        // otherwise be joined in as an empty line (a full screen of
        // whitespace for a host's "copy output" action).
        let mut screen = Screen::new(24, 80);
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, screen.cursor_absolute_row());
        bl.on_marker(PromptMarker::CommandExecuted, screen.cursor_absolute_row());
        for ch in "building".chars() {
            screen.print(ch);
        }
        screen.linefeed();
        screen.carriage_return();
        bl.observe_row(&screen);

        let id = bl.latest().unwrap().id();
        let text = bl.block_text(&screen, id).unwrap();
        assert_eq!(text, "building");
        assert!(!text.ends_with('\n'));
    }

    #[test]
    fn interior_blank_lines_are_preserved() {
        // Only the TRAILING blank run is dropped: a blank line between two
        // lines of real output is real output.
        let mut screen = Screen::new(24, 80);
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, screen.cursor_absolute_row());
        bl.on_marker(PromptMarker::CommandExecuted, screen.cursor_absolute_row());
        for line in ["first", "", "third"] {
            for ch in line.chars() {
                screen.print(ch);
            }
            screen.linefeed();
            screen.carriage_return();
        }
        bl.observe_row(&screen);

        let id = bl.latest().unwrap().id();
        assert_eq!(bl.block_text(&screen, id).unwrap(), "first\n\nthird");
    }

    #[test]
    fn block_text_skips_evicted_rows() {
        // 2 grid rows, only 1 scrollback row retained: old rows evict fast.
        let mut screen = Screen::with_scrollback(2, 20, 1);
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, screen.cursor_absolute_row());
        feed_line(&mut screen, "line0");
        feed_line(&mut screen, "line1");
        feed_line(&mut screen, "line2");
        feed_line(&mut screen, "line3");
        bl.observe_row(&screen);
        assert!(screen.scrollback().evicted() > 0);

        let id = bl.latest().unwrap().id();
        let text = bl.block_text(&screen, id).unwrap();
        // Earliest lines are gone (evicted) but later ones survive.
        assert!(!text.contains("line0"));
        assert!(text.contains("line3"));
    }

    // -- Pruning ------------------------------------------------------------------

    #[test]
    fn prune_evicted_drops_fully_evicted_blocks_only() {
        // 2 grid rows, 2 scrollback rows: eviction kicks in quickly.
        let mut screen = Screen::with_scrollback(2, 20, 2);
        let mut bl = BlockList::new();

        // Three blocks: [0,2), [2,4), and open at 4.
        run_command(&mut bl, 0, "first", 0, 1);
        run_command(&mut bl, 2, "second", 0, 3);
        bl.on_marker(PromptMarker::PromptStart, 4);
        let first_id = bl.first().unwrap().id();
        let open_id = bl.latest().unwrap().id();

        // Nothing evicted yet: nothing pruned.
        assert_eq!(bl.prune_evicted(&screen), 0);
        assert_eq!(bl.len(), 3);

        // Scroll until rows 0..3 are evicted (evicted() == 3): block one
        // ([0,2)) is fully gone, block two ([2,4)) still has row 3.
        while screen.scrollback().evicted() < 3 {
            feed_line(&mut screen, "x");
        }
        assert_eq!(bl.prune_evicted(&screen), 1);
        assert_eq!(bl.len(), 2);
        assert!(bl.get(first_id).is_none());
        assert_eq!(bl.first().unwrap().command(), Some("second"));

        // Evict everything: the closed block goes too, but the open block
        // is never pruned.
        while screen.scrollback().evicted() < 10 {
            feed_line(&mut screen, "x");
        }
        assert_eq!(bl.prune_evicted(&screen), 1);
        assert_eq!(bl.len(), 1);
        assert_eq!(bl.latest().unwrap().id(), open_id);
        assert!(bl.latest().unwrap().end_row().is_none());
    }

    #[test]
    fn navigation_still_works_after_pruning() {
        let mut screen = Screen::with_scrollback(2, 20, 2);
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "a", 0, 1);
        run_command(&mut bl, 2, "b", 0, 3);
        run_command(&mut bl, 4, "c", 0, 5);
        while screen.scrollback().evicted() < 2 {
            feed_line(&mut screen, "x");
        }
        assert_eq!(bl.prune_evicted(&screen), 1); // "a" ([0,2)) pruned
        let b_id = bl.first().unwrap().id();
        assert_eq!(bl.first().unwrap().command(), Some("b"));
        assert_eq!(bl.next(b_id).unwrap().command(), Some("c"));
        assert!(bl.prev(b_id).is_none());
        assert_eq!(bl.get(b_id).unwrap().command(), Some("b"));
    }

    // -- Identity -----------------------------------------------------------------

    // -- The invariant: rows are a finite, reusable resource ----------------

    /// Every marker kind at every row in `rows`, for the exhaustive fuzz.
    fn all_markers(rows: &[usize]) -> Vec<(PromptMarker, usize)> {
        let kinds = [
            PromptMarker::PromptStart,
            PromptMarker::CommandStart,
            PromptMarker::CommandExecuted,
            PromptMarker::CommandFinished { exit_code: Some(0) },
        ];
        rows.iter()
            .flat_map(|row| kinds.iter().map(move |kind| (*kind, *row)))
            .collect()
    }

    #[test]
    fn a_marker_before_the_open_block_reclaims_its_rows() {
        // The critical regression: a prompt marker at a row *earlier* than
        // the open block's start. Clamping the end to the start used to
        // leave a zero-row block behind while the new block re-claimed
        // rows two older blocks already owned, so `block_at_row` answered
        // with a stale block and output was orphaned.
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "one", 0, 2);
        bl.on_marker(PromptMarker::PromptStart, 3);
        bl.on_marker(PromptMarker::CommandStart, 3);
        bl.set_command("two");
        bl.on_marker(PromptMarker::CommandExecuted, 4);

        // The cursor jumps back to row 2 (a CUP, a resize, a TUI) and the
        // shell draws a prompt there.
        bl.on_marker(PromptMarker::PromptStart, 2);
        bl.assert_invariants();

        let ranges: Vec<(usize, Option<usize>)> =
            bl.blocks().map(|b| (b.start_row(), b.end_row())).collect();
        // "one" gives up rows 2.. ; "two" owned only rows >= 3, all of
        // which are reclaimed, but it keeps its command as a zero-row
        // record; the new prompt owns row 2 onward.
        assert_eq!(ranges, vec![(0, Some(2)), (2, Some(2)), (2, None)]);
        assert_eq!(bl.blocks().nth(1).unwrap().command(), Some("two"));
        // Row 2 resolves to the newest owner, never the stale block.
        assert_eq!(bl.block_at_row(2).unwrap().id(), bl.last().unwrap().id());
        assert_eq!(bl.block_at_row(1).unwrap().command(), Some("one"));
        // Metadata written by row lands on the live block, not an older one.
        assert!(bl.set_command_at_row(2, "three"));
        assert_eq!(bl.first().unwrap().command(), Some("one"));
        assert_eq!(bl.last().unwrap().command(), Some("three"));
    }

    #[test]
    fn every_four_marker_sequence_preserves_the_invariant() {
        // 4 marker kinds x 3 rows = 12 steps, exhaustively to depth 4:
        // 20_736 sequences, including every degenerate order and every
        // backwards row jump. Before the row-reclaiming invariant, 1_040
        // of these produced overlapping blocks and 7_131 produced
        // non-ascending starts.
        let steps = all_markers(&[0, 1, 2]);
        let mut checked = 0usize;
        for a in &steps {
            for b in &steps {
                for c in &steps {
                    for d in &steps {
                        let mut bl = BlockList::new();
                        for (marker, row) in [a, b, c, d] {
                            bl.on_marker(*marker, *row);
                        }
                        bl.assert_invariants();
                        // Row ownership is single-valued, so a row lookup
                        // can never be ambiguous.
                        for row in 0..4 {
                            let owners = bl.blocks().filter(|x| x.contains_row(row)).count();
                            assert!(owners <= 1, "row {row} owned by {owners} blocks");
                            assert_eq!(
                                owners == 1,
                                bl.block_at_row(row).is_some(),
                                "block_at_row disagrees with ownership at row {row}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 20_736);
    }

    #[test]
    fn row_less_blocks_are_not_retained() {
        // `C` + `D` at one row, forever: each pair closes the previous
        // block at its own start row, so every one of them owns nothing.
        // These used to accumulate without bound — unreachable by
        // `block_at_row`, holding no text, and impossible to prune because
        // their rows are on screen rather than evicted.
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        for _ in 0..10_000 {
            bl.on_marker(PromptMarker::CommandExecuted, 0);
            bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(0) }, 0);
        }
        assert_eq!(bl.len(), 1, "row-less blocks accumulated");
        bl.assert_invariants();
        // An exit code alone does not rescue a block that owns no rows:
        // without a command there is nothing for a UI to show.
        assert_eq!(bl.latest().unwrap().exit_code(), Some(0));
    }

    #[test]
    fn a_row_less_block_with_a_command_is_kept() {
        // The complement: a real command that printed nothing and left the
        // cursor where it was is still a command the user ran.
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 4);
        bl.on_marker(PromptMarker::CommandStart, 4);
        bl.set_command("cd /tmp");
        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(0) }, 4);
        bl.on_marker(PromptMarker::PromptStart, 4);

        assert_eq!(bl.len(), 2);
        let kept = bl.first().unwrap();
        assert_eq!(kept.command(), Some("cd /tmp"));
        assert_eq!(kept.start_row(), 4);
        assert_eq!(kept.end_row(), Some(4));
        assert_eq!(kept.exit_code(), Some(0));
        // It owns no rows, so it never shadows the live block.
        assert_eq!(bl.block_at_row(4).unwrap().id(), bl.last().unwrap().id());
        bl.assert_invariants();
    }

    #[test]
    fn navigation_order_is_row_order() {
        // "Jump to the next command" must move down the screen. When
        // marker rows went backwards the list held starts like [0, 10, 5]
        // and `next` walked the viewport backwards.
        let mut bl = BlockList::new();
        for (row, command) in [(0, "a"), (10, "b"), (20, "c")] {
            bl.on_marker(PromptMarker::PromptStart, row);
            bl.on_marker(PromptMarker::CommandStart, row);
            bl.set_command(command);
            bl.on_marker(PromptMarker::CommandExecuted, row);
        }
        // Now a backwards jump, the shape that used to break ordering.
        bl.on_marker(PromptMarker::PromptStart, 5);
        bl.on_marker(PromptMarker::CommandStart, 5);
        bl.set_command("d");
        bl.assert_invariants();

        let starts: Vec<usize> = bl.blocks().map(Block::start_row).collect();
        let mut sorted = starts.clone();
        sorted.sort_unstable();
        assert_eq!(starts, sorted, "list order must be row order");

        // Walking with `next` visits rows in non-decreasing order.
        let mut walked = vec![bl.first().unwrap().start_row()];
        let mut cursor = bl.first().unwrap().id();
        while let Some(block) = bl.next(cursor) {
            walked.push(block.start_row());
            cursor = block.id();
        }
        assert_eq!(walked, starts);
        // ...and `prev` is its exact inverse.
        let mut back = vec![bl.last().unwrap().start_row()];
        let mut cursor = bl.last().unwrap().id();
        while let Some(block) = bl.prev(cursor) {
            back.push(block.start_row());
            cursor = block.id();
        }
        back.reverse();
        assert_eq!(back, starts);
    }

    #[test]
    fn block_at_row_stays_cheap_with_many_blocks() {
        // `index_at_row` used to scan from the oldest block while the row
        // being looked up is essentially always in the newest, so a host
        // resolving one OSC 7 per prompt paid O(blocks^2) over a session:
        // 40k blocks took 1.27s of pure lookup. This is a smoke threshold,
        // not a benchmark — a binary search does the whole loop in
        // milliseconds, and any return to a linear scan blows past it by
        // orders of magnitude even on a slow machine.
        const BLOCKS: usize = 30_000;
        let mut bl = BlockList::new();
        for row in 0..BLOCKS {
            bl.on_marker(PromptMarker::PromptStart, row);
            bl.on_marker(PromptMarker::CommandStart, row);
            bl.set_command("x"); // keep every block, even at zero rows
        }
        assert_eq!(bl.len(), BLOCKS);

        let start = std::time::Instant::now();
        for row in 0..BLOCKS {
            // Worst case for the old scan: always the newest block.
            assert!(bl.block_at_row(BLOCKS - 1 - (row % 3)).is_some());
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "{BLOCKS} row lookups took {elapsed:?}; index_at_row is not O(log n)"
        );
    }

    #[test]
    fn index_at_row_resolves_contested_rows_to_the_newest_owner() {
        // Defence in depth: even if two blocks somehow claimed the same
        // row, the row belongs to whatever wrote it most recently.
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.on_marker(PromptMarker::PromptStart, 5);
        let newest = bl.latest().unwrap().id();
        // Forge an overlap the public API can no longer produce.
        bl.blocks[0].end = Some(9);
        assert_eq!(bl.block_at_row(7).unwrap().id(), newest);
        assert!(
            bl.check_invariants().is_err(),
            "the forgery should be detected"
        );
    }

    // -- Fallback retraction ------------------------------------------------

    #[test]
    fn a_fallback_block_the_host_named_survives_the_first_marker() {
        // A shell that prints an MOTD or ssh banner before its first
        // prompt: retracting unconditionally orphaned that output, so it
        // could not be selected, copied, or found by `search_output`.
        let mut screen = Screen::new(24, 80);
        let mut bl = BlockList::new();
        feed_line(&mut screen, "Welcome to Ubuntu");
        feed_line(&mut screen, "Last login: today");
        bl.observe_row(&screen);
        assert_eq!(bl.len(), 1);
        bl.set_command("login banner");

        bl.on_marker(PromptMarker::PromptStart, 2);
        bl.assert_invariants();

        assert_eq!(bl.len(), 2, "the named fallback block must survive");
        let banner = bl.first().unwrap();
        assert_eq!(banner.start_row(), 0);
        assert_eq!(banner.end_row(), Some(2));
        assert_eq!(banner.command(), Some("login banner"));
        // The banner rows are addressable and searchable again.
        assert_eq!(bl.block_at_row(0).unwrap().id(), banner.id());
        assert_eq!(bl.search_output(&screen, "Welcome to Ubuntu").len(), 1);
    }

    // -- Search semantics ---------------------------------------------------

    #[test]
    fn search_folds_case_beyond_plain_lowercasing() {
        let screen = blank_screen();
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.set_command("grep straße");
        bl.on_marker(PromptMarker::PromptStart, 1);
        bl.set_command("open ﬁle");

        // Plain `to_lowercase` leaves these different; folding does not.
        assert_eq!(bl.search_metadata("STRASSE").len(), 1);
        assert_eq!(bl.search_metadata("strasse").len(), 1);
        assert_eq!(bl.search_metadata("FILE").len(), 1);
        assert_eq!(bl.search_metadata("file").len(), 1);
        // Ordinary case insensitivity still holds, in both entry points.
        assert_eq!(bl.search_metadata("GREP").len(), 1);
        assert!(bl.search_output(&screen, "nothing here").is_empty());

        // Documented residual: a lowercase form that gains a combining
        // mark still does not match the unmarked spelling. This needs
        // Unicode normalization, which needs a dependency.
        let mut turkish = BlockList::new();
        turkish.on_marker(PromptMarker::PromptStart, 0);
        turkish.set_command("echo İstanbul");
        assert_eq!(turkish.search_metadata("İSTANBUL").len(), 1);
        assert_eq!(turkish.search_metadata("istanbul").len(), 0);
    }

    #[test]
    fn an_empty_query_matches_every_block_in_both_searches() {
        // The two entry points used to disagree: `search_metadata("")`
        // quietly filtered to blocks that had a command or cwd, while
        // `search_output(_, "")` returned everything. "" is a substring of
        // every string, so it matches everything; a search function must
        // not smuggle in a second predicate.
        let screen = blank_screen();
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::PromptStart, 0);
        bl.set_command("named");
        bl.on_marker(PromptMarker::PromptStart, 1); // no metadata at all
        assert_eq!(bl.len(), 2);

        assert_eq!(bl.search_metadata("").len(), 2);
        assert_eq!(bl.search_output(&screen, "").len(), 2);
        // Newest first, like every other search result.
        assert_eq!(bl.search_metadata("")[0].id(), bl.last().unwrap().id());
    }

    #[test]
    fn block_ids_are_stable_ordered_and_displayable() {
        let mut bl = BlockList::new();
        run_command(&mut bl, 0, "a", 0, 1);
        run_command(&mut bl, 2, "b", 0, 3);
        let ids: Vec<BlockId> = bl.blocks().map(Block::id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids[0] < ids[1]);
        assert_eq!(ids[0], ids[0]);
        assert_eq!(format!("{}", BlockId(7)), "7");

        use std::collections::HashSet;
        let set: HashSet<BlockId> = ids.iter().copied().collect();
        assert_eq!(set.len(), 2);
    }
}
