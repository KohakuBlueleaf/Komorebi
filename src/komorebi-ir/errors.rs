//! Diagnostics: what is wrong with a text, where, and how bad.

use std::fmt;

use crate::span::Span;

/// The code of a syntax error: the text does not follow the grammar.
pub const SYNTAX: &str = "K001";
/// The code of a `state` written without its block (settled row 38: always a block).
pub const STATE_NOT_BLOCK: &str = "K002";
/// The code of a text nested deeper than the parser reads (an expression or blocks
/// more than 256 deep): not wrong, beyond this tool.
pub const TOO_DEEP: &str = "K011";

/// How bad a diagnostic is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// The module is refused.
    Error,
    /// The module is accepted; something deserves a look.
    Warning,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        })
    }
}

/// A line and a column, both counted from 1. The column counts characters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Position {
    /// The line, from 1.
    pub line: usize,
    /// The column, from 1, in characters.
    pub col: usize,
}

/// One finding about a text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// The rule's code.
    pub code: &'static str,
    /// What is wrong, in words.
    pub message: String,
    /// How bad it is.
    pub severity: Severity,
    /// The bytes it is about.
    pub span: Span,
    /// Where `span` starts.
    pub start: Position,
    /// Where `span` ends.
    pub end: Position,
}

impl Diagnostic {
    /// An error with `code` and `message` about `span` of `source`.
    pub fn error(source: &str, code: &'static str, message: String, span: Span) -> Diagnostic {
        let index = LineIndex::new(source);
        Diagnostic {
            code,
            message,
            severity: Severity::Error,
            span,
            start: index.position(source, span.start),
            end: index.position(source, span.end),
        }
    }

    /// The diagnostic as `file:line:col: code: text`, the form the command prints.
    pub fn render(&self, file: &str) -> String {
        format!(
            "{}:{}:{}: {}: {}",
            file, self.start.line, self.start.col, self.code, self.message
        )
    }
}

/// Where each line of a text starts, to turn byte offsets into lines and columns.
#[derive(Clone, Debug)]
pub struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    /// The index of `source`.
    pub fn new(source: &str) -> LineIndex {
        let mut starts = vec![0];
        starts.extend(source.match_indices('\n').map(|(at, _)| at + 1));
        LineIndex { starts }
    }

    /// The line and column of byte `offset` of `source` (clamped to the text).
    pub fn position(&self, source: &str, offset: usize) -> Position {
        let mut offset = offset.min(source.len());
        while !source.is_char_boundary(offset) {
            offset -= 1;
        }
        let line = self.starts.partition_point(|&start| start <= offset);
        let start = self.starts[line - 1];
        let col = source[start..offset].chars().count() + 1;
        Position { line, col }
    }
}
