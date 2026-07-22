//! The ordered [`BlockList`] and its lifecycle / query operations.

use bb_terminal_model::Screen;

use crate::block::{Block, BlockId, BlockLifecycle, PromptMarker};

/// An ordered list of command-output [`Block`]s for one terminal session.
///
/// Blocks are created and closed by feeding prompt-protocol markers to
/// [`BlockList::on_marker`]; at most one block (always the newest) is open
/// at a time. Blocks are ordered oldest → newest, which is also ascending
/// [`Block::start_row`] and ascending [`BlockId`] order.
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
    /// - [`PromptMarker::PromptStart`] at row `R` closes any still-open
    ///   block (`end_row = R`; a block still
    ///   [`Running`](BlockLifecycle::Running) — a command that never
    ///   reported `D` — becomes [`Finished`](BlockLifecycle::Finished) with
    ///   `exit_code` left `None`) and starts a new
    ///   [`Prompt`](BlockLifecycle::Prompt) block at `start_row = R`.
    ///   A repeated `PromptStart` at the same row as the current prompt
    ///   block (a shell redrawing its prompt) is ignored.
    /// - [`PromptMarker::CommandStart`] moves the current block to
    ///   [`Composing`](BlockLifecycle::Composing).
    /// - [`PromptMarker::CommandExecuted`] moves the current block to
    ///   [`Running`](BlockLifecycle::Running).
    /// - [`PromptMarker::CommandFinished`] moves the current block to
    ///   [`Finished`](BlockLifecycle::Finished) and records the exit code.
    ///   The block's end stays open until the *next* `PromptStart` (Warp
    ///   semantics: the block visually includes the command's output, and
    ///   the next prompt starts the next block).
    ///
    /// Degenerate marker orders never panic: a marker with no open block
    /// creates a block at `cursor_abs_row` already in the implied state,
    /// `CommandExecuted` without `CommandStart` simply jumps the block to
    /// `Running`, and a `CommandStart`/`CommandExecuted` arriving after
    /// `CommandFinished` closes the finished block and starts a new one.
    pub fn on_marker(&mut self, marker: PromptMarker, cursor_abs_row: usize) {
        self.saw_marker = true;
        match marker {
            PromptMarker::PromptStart => {
                if let Some(open) = self.open_block_mut() {
                    if open.lifecycle == BlockLifecycle::Prompt && open.start_row == cursor_abs_row
                    {
                        return; // prompt redraw at the same row
                    }
                    Self::close_block(open, cursor_abs_row);
                }
                self.push_block(cursor_abs_row, BlockLifecycle::Prompt);
            }
            PromptMarker::CommandStart => match self.open_block_mut() {
                Some(open)
                    if matches!(
                        open.lifecycle,
                        BlockLifecycle::Prompt | BlockLifecycle::Composing
                    ) =>
                {
                    open.lifecycle = BlockLifecycle::Composing;
                    open.last_row = open.last_row.max(cursor_abs_row);
                }
                Some(open) => {
                    // B after C/D without a fresh prompt: start a new block.
                    Self::close_block(open, cursor_abs_row);
                    self.push_block(cursor_abs_row, BlockLifecycle::Composing);
                }
                None => self.push_block(cursor_abs_row, BlockLifecycle::Composing),
            },
            PromptMarker::CommandExecuted => match self.open_block_mut() {
                Some(open) if open.lifecycle != BlockLifecycle::Finished => {
                    open.lifecycle = BlockLifecycle::Running;
                    open.last_row = open.last_row.max(cursor_abs_row);
                }
                Some(open) => {
                    // C after D without a fresh prompt: start a new block.
                    Self::close_block(open, cursor_abs_row);
                    self.push_block(cursor_abs_row, BlockLifecycle::Running);
                }
                None => self.push_block(cursor_abs_row, BlockLifecycle::Running),
            },
            PromptMarker::CommandFinished { exit_code } => match self.open_block_mut() {
                Some(open) => {
                    open.lifecycle = BlockLifecycle::Finished;
                    open.exit_code = exit_code;
                    open.last_row = open.last_row.max(cursor_abs_row);
                }
                None => {
                    // D without A: record what we can.
                    self.push_block(cursor_abs_row, BlockLifecycle::Finished);
                    if let Some(block) = self.blocks.last_mut() {
                        block.exit_code = exit_code;
                    }
                }
            },
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
            self.push_block(0, BlockLifecycle::Running);
        }
        let cursor_abs_row = screen.cursor_absolute_row();
        if let Some(open) = self.open_block_mut() {
            open.last_row = open.last_row.max(cursor_abs_row);
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
    pub fn next(&self, id: BlockId) -> Option<&Block> {
        self.index_of(id).and_then(|i| self.blocks.get(i + 1))
    }

    /// The block before the one with `id` (Warp-style jump to the previous
    /// command). `None` if `id` is unknown or already the first block.
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

    /// Blocks whose stored metadata (command / cwd) contains `query`
    /// (case-insensitive substring), newest first.
    pub fn search_metadata(&self, query: &str) -> Vec<&Block> {
        let needle = query.to_lowercase();
        self.blocks
            .iter()
            .rev()
            .filter(|b| {
                b.command
                    .as_deref()
                    .is_some_and(|c| c.to_lowercase().contains(&needle))
                    || b.cwd
                        .as_deref()
                        .is_some_and(|c| c.to_lowercase().contains(&needle))
            })
            .collect()
    }

    /// Full-text search over block *output* via the screen
    /// (case-insensitive substring), newest first. Rows evicted from
    /// scrollback are silently skipped.
    pub fn search_output<'a>(&'a self, screen: &Screen, query: &str) -> Vec<&'a Block> {
        let needle = query.to_lowercase();
        self.blocks
            .iter()
            .rev()
            .filter(|b| {
                Self::collect_text(b, screen)
                    .to_lowercase()
                    .contains(&needle)
            })
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
    pub fn prune_evicted(&mut self, screen: &Screen) -> usize {
        let evicted = screen.scrollback().evicted();
        let before = self.blocks.len();
        self.blocks.retain(|b| match b.end {
            // Keep unless every row (start..end, exclusive end) is evicted.
            Some(end) => end > evicted || b.start_row >= evicted,
            None => true,
        });
        before - self.blocks.len()
    }

    // -------------------------------------------------------------------
    // Internals
    // -------------------------------------------------------------------

    fn push_block(&mut self, start_row: usize, lifecycle: BlockLifecycle) {
        let id = BlockId(self.next_id);
        self.next_id += 1;
        self.blocks.push(Block::new(id, start_row, lifecycle));
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
        (0..screen.rows()).any(|row| {
            screen
                .row_text(screen.absolute_row(row))
                .is_some_and(|text| !text.is_empty())
        })
    }

    /// Index of the block owning `abs_row`, if any.
    fn index_at_row(&self, abs_row: usize) -> Option<usize> {
        self.blocks.iter().position(|b| b.contains_row(abs_row))
    }

    /// The open block, if any. Invariant: only the newest block can be
    /// open (every earlier block was closed when its successor started).
    fn open_block_mut(&mut self) -> Option<&mut Block> {
        self.blocks.last_mut().filter(|b| b.end.is_none())
    }

    /// Close `block` at exclusive end `end_row` (clamped so the end never
    /// precedes the start). A block still `Running` becomes `Finished`
    /// with no exit code — its command never reported `D`.
    fn close_block(block: &mut Block, end_row: usize) {
        block.end = Some(end_row.max(block.start_row));
        if block.lifecycle == BlockLifecycle::Running {
            block.lifecycle = BlockLifecycle::Finished;
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
        let mut out = String::new();
        let mut first = true;
        for abs_row in start..end {
            if let Some(text) = screen.row_text(abs_row) {
                if !first {
                    out.push('\n');
                }
                out.push_str(&text);
                first = false;
            }
        }
        out
    }

    fn index_of(&self, id: BlockId) -> Option<usize> {
        // Ids are allocated monotonically as blocks are pushed, so the vec
        // is sorted by id even after pruning.
        self.blocks.binary_search_by_key(&id, |b| b.id).ok()
    }
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
    fn finished_without_prompt_creates_finished_block() {
        let mut bl = BlockList::new();
        bl.on_marker(PromptMarker::CommandFinished { exit_code: Some(3) }, 7);
        assert_eq!(bl.len(), 1);
        let block = bl.latest().unwrap();
        assert_eq!(block.lifecycle(), BlockLifecycle::Finished);
        assert_eq!(block.start_row(), 7);
        assert_eq!(block.exit_code(), Some(3));
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

        // Open block: extends to the screen's last addressable row; the
        // trailing blank grid rows contribute empty lines.
        let last_id = bl.last().unwrap().id();
        let text = bl.block_text(&screen, last_id).unwrap();
        assert!(text.starts_with("$ rg needle\nsrc/lib.rs: needle found"));

        assert!(bl.block_text(&screen, BlockId(42)).is_none());
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
