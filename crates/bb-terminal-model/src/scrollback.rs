//! Bounded scrollback history.

use std::collections::VecDeque;

use crate::cell::Cell;

/// A bounded ring buffer of rows that have scrolled off the top of the
/// primary grid.
///
/// When the buffer is full, pushing a new row evicts the oldest one. The
/// total number of rows ever evicted is tracked ([`Scrollback::evicted`])
/// so that *absolute row ids* stay stable forever: the row at retained
/// index `i` has absolute id `evicted() + i`, and the ids of evicted rows
/// are never reused.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Scrollback {
    rows: VecDeque<Vec<Cell>>,
    max_lines: usize,
    evicted: usize,
}

impl Scrollback {
    /// Create an empty scrollback that retains at most `max_lines` rows.
    ///
    /// A capacity of `0` retains nothing: every pushed row is immediately
    /// counted as evicted.
    pub fn new(max_lines: usize) -> Self {
        Self {
            rows: VecDeque::new(),
            max_lines,
            evicted: 0,
        }
    }

    /// Append a row, evicting the oldest retained row if the buffer is full.
    pub fn push_row(&mut self, row: Vec<Cell>) {
        if self.max_lines == 0 {
            self.evicted += 1;
            return;
        }
        if self.rows.len() >= self.max_lines {
            self.rows.pop_front();
            self.evicted += 1;
        }
        self.rows.push_back(row);
    }

    /// Number of rows currently retained.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no rows are currently retained.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Maximum number of rows this buffer retains.
    pub fn max_lines(&self) -> usize {
        self.max_lines
    }

    /// Total number of rows ever evicted from this buffer.
    pub fn evicted(&self) -> usize {
        self.evicted
    }

    /// Borrow the retained row at `idx`, where `0` is the *oldest* retained
    /// row. Returns `None` if out of range.
    pub fn row(&self, idx: usize) -> Option<&[Cell]> {
        self.rows.get(idx).map(Vec::as_slice)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn row_of(ch: char) -> Vec<Cell> {
        vec![Cell {
            ch,
            ..Cell::default()
        }]
    }

    #[test]
    fn new_scrollback_is_empty() {
        let sb = Scrollback::new(10);
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());
        assert_eq!(sb.max_lines(), 10);
        assert_eq!(sb.evicted(), 0);
        assert!(sb.row(0).is_none());
    }

    #[test]
    fn push_and_read_back_in_order() {
        let mut sb = Scrollback::new(10);
        sb.push_row(row_of('a'));
        sb.push_row(row_of('b'));
        assert_eq!(sb.len(), 2);
        assert!(!sb.is_empty());
        assert_eq!(sb.row(0).unwrap()[0].ch, 'a'); // 0 = oldest
        assert_eq!(sb.row(1).unwrap()[0].ch, 'b');
        assert!(sb.row(2).is_none());
    }

    #[test]
    fn eviction_drops_oldest_and_counts() {
        let mut sb = Scrollback::new(2);
        sb.push_row(row_of('a'));
        sb.push_row(row_of('b'));
        sb.push_row(row_of('c'));
        assert_eq!(sb.len(), 2);
        assert_eq!(sb.evicted(), 1);
        assert_eq!(sb.row(0).unwrap()[0].ch, 'b');
        assert_eq!(sb.row(1).unwrap()[0].ch, 'c');
        sb.push_row(row_of('d'));
        assert_eq!(sb.evicted(), 2);
        assert_eq!(sb.row(0).unwrap()[0].ch, 'c');
    }

    #[test]
    fn zero_capacity_retains_nothing() {
        let mut sb = Scrollback::new(0);
        sb.push_row(row_of('a'));
        sb.push_row(row_of('b'));
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());
        assert_eq!(sb.evicted(), 2);
        assert!(sb.row(0).is_none());
    }
}
