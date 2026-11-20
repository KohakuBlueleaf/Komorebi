//! The origin as today's DSL prints it: a statement's line and its text in a comment,
//! the script's name in a file's first line. The notes hold the lines as written
//! (`program::notes::Origin`); the rules that print them are today's, ported here.

use crate::program::notes::{Anchor, Origin};

/// Where an anchored thing was written: its file and its first and last lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Place {
    /// The file, by its index in the origin's files.
    pub file: u32,
    /// The first line.
    pub first: u32,
    /// The last line.
    pub last: u32,
}

/// The place of `anchor`, if the origin gives one.
pub fn place(origin: &Origin, anchor: Anchor) -> Option<Place> {
    let at = origin
        .places
        .binary_search_by(|(a, _, _, _)| a.cmp(&anchor))
        .ok()?;
    let (_, file, first, last) = origin.places[at];
    Some(Place { file, first, last })
}

/// Line `number` of `file` as written, if the origin holds it.
fn line(origin: &Origin, file: u32, number: u32) -> Option<&str> {
    let at = origin
        .lines
        .binary_search_by(|(f, n, _)| (*f, *n).cmp(&(file, number)))
        .ok()?;
    Some(origin.lines[at].2.as_str())
}

/// The name of a file without its folders, as a generated file's first line gives
/// it (`Path(source).name`).
pub fn file_name(origin: &Origin, file: u32) -> &str {
    let path = origin.files.get(file as usize).map_or("", String::as_str);
    path.rsplit('/').next().unwrap_or(path)
}

/// A statement's text on one line (`compile/source.py`, `line_text`): its lines from
/// `first` to `last`, each stripped, those that are a comment alone left out, joined
/// by a space; then the space after an opening bracket, the space before a closing
/// one and a comma before a closing one taken out, twice.
pub fn line_text(origin: &Origin, file: u32, first: u32, last: u32) -> String {
    let mut kept = Vec::new();
    for number in first..=last {
        let Some(text) = line(origin, file, number) else {
            break;
        };
        let text = text.trim();
        if !text.starts_with('#') {
            kept.push(text);
        }
    }
    let mut text = kept.join(" ");
    for _ in 0..2 {
        text = tidied(&text);
    }
    text
}

/// One pass of `line_text`'s substitution, left to right as the regular expression
/// `([(\[]) | ([)\]])|,([)\]])` matches: each match keeps its bracket alone.
fn tidied(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let next = chars.get(i + 1).copied();
        let c = chars[i];
        if matches!(c, '(' | '[') && next == Some(' ') {
            out.push(c);
            i += 2;
        } else if matches!(c, ' ' | ',') && matches!(next, Some(')' | ']')) {
            out.extend(next);
            i += 2;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// The last line of a loop whose body makes nothing (`pass`, or a drain's call), as
/// today's `line_text` cuts its head: at the line before its body's first when that
/// is after the statement's first, else the whole statement. A drain is the loop
/// today makes on the call's own lines, so its head is the whole statement; a `pass`
/// is the statement's last line (`pass` alone, or after the head's `:`).
pub fn dropped_head_last(origin: &Origin, place: Place) -> u32 {
    let Place { file, first, last } = place;
    let text = line(origin, file, last).unwrap_or("");
    let code = text.split('#').next().unwrap_or("").trim_end();
    let passes = code.strip_suffix("pass").is_some_and(|before| {
        let before = before.trim_end();
        before.is_empty() || before.ends_with(':')
    });
    if passes && last > first {
        last - 1
    } else {
        last
    }
}

/// A statement group's comment (`compile/handler/statements.py`, `statements`):
/// `line N: ` and the statement's text; a statement of more than three lines that
/// holds no body gives its first line and ` ...`. `head_last` is a compound
/// statement's last head line (the line before its body's first).
pub fn statement_comment(origin: &Origin, place: Place, head_last: Option<u32>) -> String {
    let Place { file, first, last } = place;
    if head_last.is_none() && last - first > 2 {
        let text = line(origin, file, first).unwrap_or("").trim();
        return format!("line {first}: {text} ...");
    }
    let last = head_last.map_or(last, |head| head.clamp(first, last));
    format!("line {first}: {}", line_text(origin, file, first, last))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn origin(lines: &[&str]) -> Origin {
        Origin {
            files: vec!["/a/b/script.py".into()],
            lines: lines
                .iter()
                .enumerate()
                .map(|(i, text)| (0, i as u32 + 1, (*text).to_string()))
                .collect(),
            places: Vec::new(),
        }
    }

    #[test]
    fn a_statement_on_lines_is_one_line_with_its_brackets_tidied() {
        let o = origin(&[
            "    out = f(",
            "        # a comment alone",
            "        a,",
            "        b,",
            "    )  # kept",
        ]);
        assert_eq!(line_text(&o, 0, 1, 5), "out = f(a, b)  # kept");
        assert_eq!(file_name(&o, 0), "script.py");
        assert_eq!(file_name(&o, 1), "");
    }

    #[test]
    fn a_long_statement_gives_its_first_line_and_a_compound_its_head() {
        let o = origin(&["x = g(", "1,", "2,", ")", "for a in b:", "    c(a)"]);
        let long = Place {
            file: 0,
            first: 1,
            last: 4,
        };
        assert_eq!(statement_comment(&o, long, None), "line 1: x = g( ...");
        let compound = Place {
            file: 0,
            first: 5,
            last: 6,
        };
        assert_eq!(
            statement_comment(&o, compound, Some(5)),
            "line 5: for a in b:"
        );
    }

    #[test]
    fn a_dropped_loops_head_ends_before_its_pass_and_a_drains_is_all_of_it() {
        let o = origin(&[
            "for _ in s(",
            "    a, b",
            "):",
            "    pass  # nothing",
            "s(",
            "    a, b",
            ").drain()",
            "for _ in s(a):  pass",
            "for _ in s(",
            "    a",
            "): pass",
            "for _ in s(a): pass",
        ]);
        let at = |first, last| Place {
            file: 0,
            first,
            last,
        };
        assert_eq!(dropped_head_last(&o, at(1, 4)), 3);
        assert_eq!(dropped_head_last(&o, at(5, 7)), 7);
        assert_eq!(
            statement_comment(&o, at(5, 7), Some(7)),
            "line 5: s(a, b).drain()"
        );
        assert_eq!(dropped_head_last(&o, at(8, 8)), 8);
        assert_eq!(dropped_head_last(&o, at(9, 11)), 10);
        assert_eq!(dropped_head_last(&o, at(12, 12)), 12);
    }
}
