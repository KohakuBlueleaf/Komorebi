//! A frontend's module: the text it printed, and beside it where that text came from.
//! The origin is not written in the text (`.kom` has no syntax for it); the frontend
//! gives its source files and lines, a map from ranges of the text's lines to places of
//! its source, and the real names of the values it gave safe names.
//!
//! The rule from spans to places: after lowering, each anchor whose span starts at line
//! L, column C of the text takes the place of the narrowest range of the map that holds
//! (L, C); an anchor no range holds has no place. The spans are then cleared: a
//! frontend's module has an origin and no spans. A diagnostic of the text is placed by
//! the same rule, at its start.

use crate::deep::on_deep_stack;
use crate::errors::{Diagnostic, LineIndex};
use crate::program;
use crate::program::notes::{Anchor, Named, Origin};

/// What a frontend gives beside its text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Given {
    /// Its source files, resolved; a range names one by its index.
    pub files: Vec<String>,
    /// The source lines its ranges cover: (file, number from 1, the line as written), in
    /// (file, number) order, each once.
    pub lines: Vec<(u32, u32, String)>,
    /// Ranges of the text, each with the place of the source it was printed for.
    pub map: Vec<Mapped>,
    /// (the safe name the text writes, the source's name): the names of values and
    /// parameters only, which the hashes other than the full one leave out.
    pub renames: Vec<(String, String)>,
}

/// One range of the text, from a line and column to a line and column, both held, and
/// the place of the source it was printed for. Columns count characters from 1; a
/// range of whole lines runs from column 1 of its first to the end of its last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapped {
    /// The text's first line, from 1.
    pub text_first: u32,
    /// The column of its first character, from 1.
    pub text_first_col: u32,
    /// The text's last line.
    pub text_last: u32,
    /// The column of its last character.
    pub text_last_col: u32,
    /// The source file, by its index in `files`.
    pub file: u32,
    /// The source's first line, from 1.
    pub first: u32,
    /// The source's last line.
    pub last: u32,
}

impl Mapped {
    /// Where it starts in the text: (line, column).
    pub fn start(&self) -> (u32, u32) {
        (self.text_first, self.text_first_col)
    }

    /// Where it ends in the text, held: (line, column).
    pub fn end(&self) -> (u32, u32) {
        (self.text_last, self.text_last_col)
    }
}

/// A position of the text in words.
fn at((line, col): (u32, u32)) -> String {
    format!("line {line} column {col}")
}

/// A place in the source: a file of the list and its first and last lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Place {
    /// The file, by its index in `files`.
    pub file: u32,
    /// The first line.
    pub first: u32,
    /// The last line.
    pub last: u32,
}

/// Why a text and its origin are refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Refused {
    /// The text is refused: its diagnostics, each with its place when the map has one.
    Text(Vec<(Diagnostic, Option<Place>)>),
    /// The origin is refused: what is wrong with it.
    Origin(String),
}

impl Given {
    /// What is wrong with the map, in words; `None` when nothing is. Each range runs
    /// from line 1, column 1 or later, first to last, in a file of the list; two ranges
    /// are nested or apart, never the same range twice nor crossing.
    pub fn fault(&self) -> Option<String> {
        for m in &self.map {
            let (a, b) = (m.start(), m.end());
            if a.0 == 0 || a.1 == 0 || a > b {
                return Some(format!(
                    "the text's {} to {}: lines and columns count from 1, first to last",
                    at(a),
                    at(b)
                ));
            }
            if m.file as usize >= self.files.len() {
                return Some(format!("a range in file {}, past the files", m.file));
            }
            if m.first == 0 || m.first > m.last {
                return Some(format!(
                    "the source's lines {} to {}: lines count from 1, first to last",
                    m.first, m.last
                ));
            }
        }
        for (i, a) in self.map.iter().enumerate() {
            for b in &self.map[i + 1..] {
                let (a0, a1, b0, b1) = (a.start(), a.end(), b.start(), b.end());
                let apart = a1 < b0 || b1 < a0;
                let nested = (a0 <= b0 && b1 <= a1) || (b0 <= a0 && a1 <= b1);
                if (a0, a1) == (b0, b1) || !(apart || nested) {
                    return Some(format!(
                        "the text's {} to {} and {} to {}: two ranges are nested or \
                         apart, never the same range twice nor crossing",
                        at(a0),
                        at(a1),
                        at(b0),
                        at(b1)
                    ));
                }
            }
        }
        let mut safe: Vec<&str> = self.renames.iter().map(|(s, _)| s.as_str()).collect();
        safe.sort_unstable();
        if let Some(w) = safe.windows(2).find(|w| w[0] == w[1]) {
            return Some(format!("the safe name `{}` renamed twice", w[0]));
        }
        None
    }

    /// The place of line `line`, column `col` of the text: the narrowest range that
    /// holds it (of ranges nested or apart, the one that starts last).
    pub fn place_at(&self, line: u32, col: u32) -> Option<Place> {
        let here = (line, col);
        self.map
            .iter()
            .filter(|m| m.start() <= here && here <= m.end())
            .max_by_key(|m| (m.start(), std::cmp::Reverse(m.end())))
            .map(|m| Place {
                file: m.file,
                first: m.first,
                last: m.last,
            })
    }
}

/// A frontend's text in the writer's own layout (`format`), its origin carried to the
/// text as returned: a range starts where its first character went and ends where its
/// last went (whitespace, commas and backticks aside).
/// The layout moves only whitespace and commas, and takes away the backticks a name
/// does not need (R50), so the other characters, in order, are the same on both sides;
/// a text where they are not is refused.
pub fn layout(source: &str, given: &Given) -> Result<(String, Given), Refused> {
    if let Some(what) = given.fault() {
        return Err(Refused::Origin(what));
    }
    let text = super::format(source).map_err(|d| Refused::Text(placed(given, d)))?;
    let laid = carried(source, &text, given)?;
    Ok((text, laid))
}

/// `given`'s ranges of `source` carried to `text`, the same characters but whitespace,
/// commas and backticks; any other difference refused.
fn carried(source: &str, text: &str, given: &Given) -> Result<Given, Refused> {
    // the characters the layout keeps, each with its (line, column) from 1
    let kept = |t: &str| -> Vec<(char, (u32, u32))> {
        let mut out = Vec::new();
        for (n, line) in (1u32..).zip(t.lines()) {
            for (c, ch) in (1u32..).zip(line.chars()) {
                if !ch.is_whitespace() && ch != ',' && ch != '`' {
                    out.push((ch, (n, c)));
                }
            }
        }
        out
    };
    let (was, now) = (kept(source), kept(text));
    let same = was.len() == now.len() && was.iter().zip(&now).all(|(a, b)| a.0 == b.0);
    if !same {
        return Err(Refused::Origin(
            "the writer's layout changed more than whitespace, commas and backticks".into(),
        ));
    }
    let (first, last) = (
        now.first().map_or((1, 1), |c| c.1),
        now.last().map_or((1, 1), |c| c.1),
    );
    // the character at (line, column) from 1 of a text, if any
    let char_at = |t: &str, (line, col): (u32, u32)| -> Option<char> {
        let line = t.lines().nth((line as usize).checked_sub(1)?)?;
        line.chars().nth((col as usize).checked_sub(1)?)
    };
    // where the first kept character at or after `p` went, and the last at or before;
    // a range that starts (ends) at a name's backtick holds the name's backtick where
    // the layout kept it, as the name's own span does (R50)
    let from = |p: (u32, u32)| {
        let a = was.iter().position(|c| c.1 >= p).map_or(last, |i| now[i].1);
        let tick = (a.0, a.1.saturating_sub(1));
        let both = char_at(source, p) == Some('`') && char_at(text, tick) == Some('`');
        if both {
            tick
        } else {
            a
        }
    };
    let to = |p: (u32, u32)| {
        let b = was
            .iter()
            .rposition(|c| c.1 <= p)
            .map_or(first, |i| now[i].1);
        let tick = (b.0, b.1 + 1);
        let both = char_at(source, p) == Some('`') && char_at(text, tick) == Some('`');
        if both {
            tick
        } else {
            b
        }
    };
    let map = given
        .map
        .iter()
        .map(|m| {
            let a = from(m.start());
            let b = to(m.end()).max(a);
            Mapped {
                text_first: a.0,
                text_first_col: a.1,
                text_last: b.0,
                text_last_col: b.1,
                ..*m
            }
        })
        .collect();
    let laid = Given {
        map,
        ..given.clone()
    };
    if let Some(what) = laid.fault() {
        return Err(Refused::Origin(format!(
            "after the writer's layout, {what}"
        )));
    }
    Ok(laid)
}

/// The program structure of a frontend's text with its origin attached: checked,
/// lowered, derived and verified as `build` does, then placed, renamed and validated.
pub fn build(source: &str, given: &Given) -> Result<program::Module, Refused> {
    if let Some(what) = given.fault() {
        return Err(Refused::Origin(what));
    }
    on_deep_stack(|| {
        let mut module = super::build(source).map_err(|d| Refused::Text(placed(given, d)))?;
        attach(source, &mut module, given)?;
        Ok(module)
    })
}

/// The byte-code program of a frontend's text with its origin attached.
pub fn compile(source: &str, given: &Given) -> Result<Vec<u8>, Refused> {
    build(source, given).map(|module| crate::bytecode::encode(&module))
}

/// Every diagnostic of a frontend's text, each with its place; none when the text and
/// its origin are accepted. A refused origin is an error, not a diagnostic.
pub fn check(source: &str, given: &Given) -> Result<Vec<(Diagnostic, Option<Place>)>, String> {
    match build(source, given) {
        Ok(_) => Ok(Vec::new()),
        Err(Refused::Text(rows)) => Ok(rows),
        Err(Refused::Origin(what)) => Err(what),
    }
}

/// Each diagnostic with the place of its first line.
fn placed(given: &Given, diagnostics: Vec<Diagnostic>) -> Vec<(Diagnostic, Option<Place>)> {
    diagnostics
        .into_iter()
        .map(|d| {
            let line = u32::try_from(d.start.line).ok();
            let col = u32::try_from(d.start.col).ok();
            let place = line.zip(col).and_then(|(l, c)| given.place_at(l, c));
            (d, place)
        })
        .collect()
}

/// The places from the spans, the spans cleared, the renames applied, and the module
/// validated with its origin.
fn attach(source: &str, module: &mut program::Module, given: &Given) -> Result<(), Refused> {
    let index = LineIndex::new(source);
    let mut places: Vec<(Anchor, u32, u32, u32)> = Vec::new();
    for (anchor, span) in &module.notes.spans {
        let at = index.position(source, span.start as usize);
        let line = u32::try_from(at.line).ok();
        let col = u32::try_from(at.col).ok();
        let place = line.zip(col).and_then(|(l, c)| given.place_at(l, c));
        if let Some(p) = place {
            places.push((*anchor, p.file, p.first, p.last));
        }
    }
    places.sort_by_key(|p| p.0);
    places.dedup_by_key(|p| p.0);
    module.notes.spans.clear();
    rename(module, &given.renames)?;
    module.notes.origin = Some(Origin {
        files: given.files.clone(),
        lines: given.lines.clone(),
        places,
    });
    let faults = program::validate(module);
    match faults.into_iter().next() {
        None => Ok(()),
        Some(v) => Err(Refused::Origin(v.what)),
    }
}

/// Each safe name of a value, a slot or a parameter replaced by the source's, all at
/// once (a source's name may be another's safe name); a safe name that names none of
/// them is refused.
fn rename(module: &mut program::Module, renames: &[(String, String)]) -> Result<(), Refused> {
    let mut used = vec![false; renames.len()];
    let mut swap = |name: &mut String| {
        if let Some(i) = renames.iter().position(|(safe, _)| safe == name) {
            name.clone_from(&renames[i].1);
            used[i] = true;
        }
    };
    for named in &mut module.notes.names {
        let (Named::Value(_, _, name) | Named::Local(_, _, name)) = named;
        swap(name);
    }
    for (_, name) in &mut module.notes.param_names {
        swap(name);
    }
    match used.iter().position(|u| !u) {
        None => Ok(()),
        Some(i) => Err(Refused::Origin(format!(
            "the safe name `{}` names no value and no parameter",
            renames[i].0
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layout_that_changes_more_than_whitespace_commas_and_backticks_is_refused() {
        let given = Given {
            files: vec!["a.py".into()],
            lines: vec![(0, 1, "a".into())],
            map: vec![Mapped {
                text_first: 1,
                text_first_col: 1,
                text_last: 1,
                text_last_col: 15,
                file: 0,
                first: 1,
                last: 1,
            }],
            renames: Vec::new(),
        };
        let source = "let a = f(b, c)";
        // split, with a trailing comma: carried to the lines it went to
        let split = carried(source, "let a = f(\n    b,\n    c,\n)", &given).expect("carried");
        assert_eq!((split.map[0].start(), split.map[0].end()), ((1, 1), (4, 1)));
        // a character changed: refused by name
        let changed = carried(source, "let a = f(b; c)", &given);
        let what = "the writer's layout changed more than whitespace, commas and backticks";
        assert!(matches!(changed, Err(Refused::Origin(w)) if w == what));
        // backticks taken away or kept (R50): carried, a range ending at a kept one holding
        // it; a name changed between them is refused
        let quoted = "let `a` = f(`b`, c)";
        let kept = carried(quoted, "let a = f(`b`, c)", &given).expect("carried");
        assert_eq!((kept.map[0].start(), kept.map[0].end()), ((1, 1), (1, 13)));
        let renamed = carried(quoted, "let a = f(`d`, c)", &given);
        assert!(matches!(renamed, Err(Refused::Origin(w)) if w == what));
    }
}
