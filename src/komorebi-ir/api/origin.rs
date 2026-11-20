//! A frontend's module: the text it printed, and beside it where that text came from.
//! The origin is not written in the text (`.kom` has no syntax for it); the frontend
//! gives its source files and lines, a map from ranges of the text's lines to places of
//! its source, and the real names of the values it gave safe names.
//!
//! The rule from spans to places: after lowering, each anchor whose span starts on line
//! L of the text takes the place of the narrowest range of the map that holds L; an
//! anchor no range holds has no place. The spans are then cleared: a frontend's module
//! has an origin and no spans. A diagnostic of the text is placed by the same rule.

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
    /// Ranges of the text's lines, each with the place of the source it was printed for.
    pub map: Vec<Mapped>,
    /// (the safe name the text writes, the source's name): the names of values and
    /// parameters only, which the hashes other than the full one leave out.
    pub renames: Vec<(String, String)>,
}

/// One range of the text's lines and the place of the source it was printed for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapped {
    /// The text's first line, from 1.
    pub text_first: u32,
    /// The text's last line.
    pub text_last: u32,
    /// The source file, by its index in `files`.
    pub file: u32,
    /// The source's first line, from 1.
    pub first: u32,
    /// The source's last line.
    pub last: u32,
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
    /// from line 1 or later, first to last, in a file of the list; two ranges are
    /// nested or apart, never the same lines twice nor crossing.
    pub fn fault(&self) -> Option<String> {
        for m in &self.map {
            if m.text_first == 0 || m.text_first > m.text_last {
                return Some(format!(
                    "the text's lines {} to {}: lines count from 1, first to last",
                    m.text_first, m.text_last
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
                let (a0, a1, b0, b1) = (a.text_first, a.text_last, b.text_first, b.text_last);
                let apart = a1 < b0 || b1 < a0;
                let nested = (a0 <= b0 && b1 <= a1) || (b0 <= a0 && a1 <= b1);
                if (a0, a1) == (b0, b1) || !(apart || nested) {
                    return Some(format!(
                        "the text's lines {a0} to {a1} and {b0} to {b1}: two ranges are \
                         nested or apart, never the same lines twice nor crossing"
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

    /// The place of line `line` of the text: the narrowest range that holds it.
    pub fn place_of(&self, line: u32) -> Option<Place> {
        self.map
            .iter()
            .filter(|m| m.text_first <= line && line <= m.text_last)
            .min_by_key(|m| m.text_last - m.text_first)
            .map(|m| Place {
                file: m.file,
                first: m.first,
                last: m.last,
            })
    }
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
            let place = u32::try_from(d.start.line)
                .ok()
                .and_then(|l| given.place_of(l));
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
        let line = index.position(source, span.start as usize).line;
        let place = u32::try_from(line).ok().and_then(|l| given.place_of(l));
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
