//! A position in the source text. A leaf: it reads nothing of the crate, and both
//! the syntax tree and the diagnostics read it.

/// A range of bytes of the source text: `start` included, `end` excluded.
///
/// Two spans always compare equal, so that two trees compare by what they say and not
/// by where it was written (`parse(write(p)) == p`). Compare `start` and `end` directly
/// where a position matters.
#[derive(Clone, Copy, Debug, Default, Eq)]
pub struct Span {
    /// Byte offset of the first byte.
    pub start: usize,
    /// Byte offset one past the last byte.
    pub end: usize,
}

impl PartialEq for Span {
    fn eq(&self, _other: &Span) -> bool {
        true
    }
}

impl Span {
    /// The span from `start` to `end`.
    pub fn new(start: usize, end: usize) -> Span {
        Span { start, end }
    }
}
