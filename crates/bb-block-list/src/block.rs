//! Block identity, lifecycle, prompt-protocol markers, and the [`Block`]
//! record itself.

use std::fmt;

/// Stable identity for a block, unique within a
/// [`BlockList`](crate::BlockList).
///
/// Ids are allocated monotonically and never reused, so they remain valid
/// (as *lookup keys* — the block itself may have been pruned) for the
/// lifetime of the list. They order the same way the blocks were created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u64);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Prompt-protocol marker, host-translated from OSC 133 parser events.
///
/// This crate deliberately does **not** depend on the VT parser: the
/// session glue observes OSC 133 sequences there and maps them onto this
/// enum before calling [`BlockList::on_marker`](crate::BlockList::on_marker).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptMarker {
    /// The shell is about to draw a fresh prompt (`OSC 133;A`).
    PromptStart,
    /// The user started composing a command line (`OSC 133;B`).
    CommandStart,
    /// The command was submitted and is now executing (`OSC 133;C`).
    CommandExecuted,
    /// The command finished (`OSC 133;D`), optionally reporting its exit
    /// code.
    CommandFinished {
        /// Exit code reported by the shell, if any.
        exit_code: Option<i32>,
    },
}

/// Where a block currently is in the prompt → command → output → done
/// progression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockLifecycle {
    /// A prompt has been drawn; no command activity yet.
    Prompt,
    /// The user is composing the command line.
    Composing,
    /// The command is executing and producing output.
    Running,
    /// The command finished (or was implicitly closed by the next prompt).
    Finished,
}

/// One command-output block: a contiguous run of absolute terminal rows
/// grouped under a single command, plus its metadata.
///
/// Rows are identified by the *absolute row ids* of
/// [`bb_terminal_model::Screen`] (see [`Screen::absolute_row`]), which stay
/// stable even after rows scroll into — and are evicted from — scrollback.
///
/// [`Screen::absolute_row`]: bb_terminal_model::Screen::absolute_row
#[derive(Debug, Clone)]
pub struct Block {
    pub(crate) id: BlockId,
    pub(crate) lifecycle: BlockLifecycle,
    pub(crate) start_row: usize,
    /// Exclusive end row; `None` while the block is still growing.
    pub(crate) end: Option<usize>,
    pub(crate) command: Option<String>,
    pub(crate) cwd: Option<String>,
    pub(crate) exit_code: Option<i32>,
    /// Whether this block was synthesized by the no-shell-integration
    /// fallback rather than created by a prompt marker. Such a block is a
    /// *guess*, and the first marker to arrive retracts it.
    pub(crate) implicit: bool,
}

impl Block {
    pub(crate) fn new(id: BlockId, start_row: usize, lifecycle: BlockLifecycle) -> Self {
        Self {
            id,
            lifecycle,
            start_row,
            end: None,
            command: None,
            cwd: None,
            exit_code: None,
            implicit: false,
        }
    }

    /// This block's stable identity.
    pub fn id(&self) -> BlockId {
        self.id
    }

    /// Current lifecycle state.
    pub fn lifecycle(&self) -> BlockLifecycle {
        self.lifecycle
    }

    /// First absolute row of the block (the row the prompt was drawn on).
    /// Inclusive.
    ///
    /// Stable for as long as the block owns any rows. The one exception is
    /// a block *every* row of which is reclaimed by a later block starting
    /// before it (a cursor jump backwards): with nothing left to point at,
    /// it collapses onto the reclaiming row as a zero-row record of the
    /// command that ran. See the invariant on
    /// [`BlockList`](crate::BlockList).
    pub fn start_row(&self) -> usize {
        self.start_row
    }

    /// One past the last absolute row of the block (exclusive), or `None`
    /// while the block is still growing (it is closed by the *next*
    /// [`PromptMarker::PromptStart`]).
    pub fn end_row(&self) -> Option<usize> {
        self.end
    }

    /// The command line the user ran, if the host recorded it via
    /// [`BlockList::set_command`](crate::BlockList::set_command).
    pub fn command(&self) -> Option<&str> {
        self.command.as_deref()
    }

    /// The working directory the command ran in, if the host recorded it
    /// via [`BlockList::set_cwd`](crate::BlockList::set_cwd) (typically
    /// translated from OSC 7).
    pub fn cwd(&self) -> Option<&str> {
        self.cwd.as_deref()
    }

    /// The exit code reported by [`PromptMarker::CommandFinished`], if any.
    pub fn exit_code(&self) -> Option<i32> {
        self.exit_code
    }

    /// Whether the command succeeded (`exit_code == 0`); `None` if no exit
    /// code was ever reported.
    pub fn succeeded(&self) -> Option<bool> {
        self.exit_code.map(|code| code == 0)
    }

    /// Whether `abs_row` falls inside this block: inclusive of
    /// [`Block::start_row`], exclusive of [`Block::end_row`]. A still-open
    /// block owns every row from its start onward.
    pub(crate) fn contains_row(&self, abs_row: usize) -> bool {
        match self.end {
            Some(end) => abs_row >= self.start_row && abs_row < end,
            None => abs_row >= self.start_row,
        }
    }
}
