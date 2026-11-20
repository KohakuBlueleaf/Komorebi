//! The editor files of the language (`packaging/vscode/komorebi-ir/`) keep up with the
//! language. The test reads every word the grammar file quotes, and the words the checker
//! gives a meaning (the built-in names, the formats, the failures), and fails, naming the
//! word, when the TextMate grammar does not write it in any of its patterns. It also checks
//! that the JSON files of the extension parse and say what VS Code needs.
//!
//! It reads text only: no editor and no node are run. The words are cut out of the crate's
//! source by their markers; when a marker moves, the test says which one.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(relative: &str) -> String {
    let path = root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} is readable: {e}", path.display()))
}

// ---- A JSON reader (the crate takes no JSON dependency) -------------------------------

#[derive(Debug)]
enum Json {
    Null,
    Bool(#[allow(dead_code)] bool),
    Num,
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_arr(&self) -> &[Json] {
        match self {
            Json::Arr(items) => items,
            _ => &[],
        }
    }
}

struct Reader<'a> {
    text: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn blank(&mut self) {
        while self.at < self.text.len() && self.text[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn fail<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.at))
    }

    fn eat(&mut self, byte: u8) -> Result<(), String> {
        if self.text.get(self.at) == Some(&byte) {
            self.at += 1;
            Ok(())
        } else {
            self.fail(&format!("expected `{}`", byte as char))
        }
    }

    fn word(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.text[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            self.fail("not a JSON value")
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.blank();
        match self.text.get(self.at) {
            Some(b'{') => {
                self.at += 1;
                let mut members = Vec::new();
                self.blank();
                if self.text.get(self.at) == Some(&b'}') {
                    self.at += 1;
                    return Ok(Json::Obj(members));
                }
                loop {
                    self.blank();
                    let key = self.string()?;
                    self.blank();
                    self.eat(b':')?;
                    let value = self.value()?;
                    members.push((key, value));
                    self.blank();
                    match self.text.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Json::Obj(members));
                        }
                        _ => return self.fail("expected `,` or `}`"),
                    }
                }
            }
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.blank();
                if self.text.get(self.at) == Some(&b']') {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    self.blank();
                    match self.text.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return self.fail("expected `,` or `]`"),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.word("true", Json::Bool(true)),
            Some(b'f') => self.word("false", Json::Bool(false)),
            Some(b'n') => self.word("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => {
                while self
                    .text
                    .get(self.at)
                    .is_some_and(|b| b"+-.eE".contains(b) || b.is_ascii_digit())
                {
                    self.at += 1;
                }
                Ok(Json::Num)
            }
            _ => self.fail("not a JSON value"),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&b) = self.text.get(self.at) else {
                return self.fail("a string is not closed");
            };
            self.at += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let Some(&e) = self.text.get(self.at) else {
                        return self.fail("a string is not closed");
                    };
                    self.at += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let hex = self.text.get(self.at..self.at + 4).unwrap_or(&[]);
                            let code = std::str::from_utf8(hex)
                                .ok()
                                .and_then(|h| u32::from_str_radix(h, 16).ok());
                            let Some(code) = code else {
                                return self.fail("a bad \\u escape");
                            };
                            self.at += 4;
                            let c = char::from_u32(code).unwrap_or('\u{fffd}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return self.fail("a bad escape"),
                    }
                }
                0..=31 => return self.fail("a control character in a string"),
                _ => out.push(b),
            }
        }
        String::from_utf8(out).map_err(|_| "a string is not UTF-8".to_string())
    }
}

fn parse_json(relative: &str) -> Json {
    let text = read(relative);
    let mut reader = Reader {
        text: text.as_bytes(),
        at: 0,
    };
    let value = reader
        .value()
        .unwrap_or_else(|e| panic!("{relative} is not JSON: {e}"));
    reader.blank();
    assert_eq!(
        reader.at,
        text.len(),
        "{relative}: text after the JSON value"
    );
    value
}

// ---- The words of the language ---------------------------------------------------------

/// The text between two markers of a source file; a marker that moved is named.
fn region<'a>(text: &'a str, file: &str, from: &str, to: &str) -> &'a str {
    let Some(start) = text.find(from) else {
        panic!("{file}: the marker `{from}` is gone: update this test's reading of it");
    };
    let rest = &text[start..];
    let Some(end) = rest.find(to) else {
        panic!("{file}: the marker `{to}` is gone: update this test's reading of it");
    };
    &rest[..end]
}

/// Every double-quoted string of `text` that is a word (a letter or `_`, then letters,
/// digits and `_`). A line that starts with `//` is skipped.
fn quoted_words(text: &str) -> BTreeSet<String> {
    let mut words = BTreeSet::new();
    for line in text.lines().filter(|l| !l.trim_start().starts_with("//")) {
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b'"' {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'"' {
                j += if bytes[j] == b'\\' { 2 } else { 1 };
            }
            if let Some(s) = line.get(i + 1..j.min(bytes.len())) {
                let first = s.chars().next();
                if first.is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    words.insert(s.to_string());
                }
            }
            i = j + 1;
        }
    }
    words
}

/// The words of the language, each with where it comes from.
fn language_words() -> Vec<(String, &'static str)> {
    let mut out: Vec<(String, &'static str)> = Vec::new();
    let pest = read("src/komorebi-ir/syntax/komorebi.pest");
    for w in quoted_words(&pest) {
        // `e` and `E` are the exponent of a float, and `_` is a letter of a name: no words.
        if !["e", "E", "_"].contains(&w.as_str()) {
            out.push((w, "syntax/komorebi.pest"));
        }
    }
    let module = read("src/komorebi-ir/check/mod.rs");
    let words = region(
        &module,
        "check/mod.rs",
        "pub fn from_name(text: &str) -> Option<Word>",
        "_ => return None",
    );
    out.extend(quoted_words(words).into_iter().map(|w| (w, "check/mod.rs")));
    let body = read("src/komorebi-ir/check/body.rs");
    for marker in ["PROVIDED: [&str;", "FAILURES: [&str;", "UNARY: [&str;"] {
        let list = region(&body, "check/body.rs", marker, "];");
        out.extend(quoted_words(list).into_iter().map(|w| (w, "check/body.rs")));
    }
    let ops = read("src/komorebi-ir/check/ops.rs");
    let family = region(
        &ops,
        "check/ops.rs",
        "let family = match what {",
        "self.unknown(n);",
    );
    out.extend(
        quoted_words(family)
            .into_iter()
            .map(|w| (w, "check/ops.rs")),
    );
    let format = read("src/komorebi-ir/types/format.rs");
    let names = region(&format, "types/format.rs", "Format::U8 => \"u8\"", "}");
    out.extend(
        quoted_words(names)
            .into_iter()
            .map(|w| (w, "types/format.rs")),
    );
    out
}

/// Every regular expression of a TextMate grammar (`match`, `begin`, `end`, `while`), the
/// escapes of a class (`\b`, `\w`, `\s`) turned into a space so a word beside one is whole.
fn patterns(node: &Json, into: &mut Vec<String>) {
    match node {
        Json::Obj(members) => {
            for (key, value) in members {
                match (key.as_str(), value) {
                    ("match" | "begin" | "end" | "while", Json::Str(re)) => {
                        let mut plain = String::new();
                        let mut chars = re.chars();
                        while let Some(c) = chars.next() {
                            if c == '\\' {
                                chars.next();
                                plain.push(' ');
                            } else {
                                plain.push(c);
                            }
                        }
                        into.push(plain);
                    }
                    _ => patterns(value, into),
                }
            }
        }
        Json::Arr(items) => items.iter().for_each(|i| patterns(i, into)),
        _ => {}
    }
}

/// True when `word` stands in `re` with no letter, digit or `_` against either side.
fn names_word(re: &str, word: &str) -> bool {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    re.match_indices(word).any(|(at, _)| {
        let before = re[..at].chars().next_back();
        let after = re[at + word.len()..].chars().next();
        !before.is_some_and(is_word) && !after.is_some_and(is_word)
    })
}

const TEXTMATE: &str = "packaging/vscode/komorebi-ir/syntaxes/komorebi-ir.tmLanguage.json";

/// The scope roots a rule may give: the ones VS Code's default themes (Dark+, Dark Modern,
/// Light+, Light Modern) colour, and the ones that only carry structure (`meta`) or
/// punctuation. A scope outside them (`constant.other`, `support.constant`) is shown in
/// the editor's plain foreground, which is how a name once went uncoloured.
const ALLOWED_ROOTS: &[&str] = &[
    "comment",
    "constant.character.escape",
    "constant.language",
    "constant.numeric",
    "entity.name",
    "entity.other.attribute-name",
    "keyword",
    "meta",
    "punctuation",
    "storage",
    "string",
    "support.function",
    "support.type",
    "support.variable",
    "variable",
];

/// Every scope name a rule of the grammar gives (a `name`, at any depth below the top).
fn scope_names(node: &Json, top: bool, into: &mut Vec<String>) {
    match node {
        Json::Obj(members) => {
            for (key, value) in members {
                match (key.as_str(), value) {
                    ("name", Json::Str(name)) if !top => into.push(name.clone()),
                    _ => scope_names(value, false, into),
                }
            }
        }
        Json::Arr(items) => items.iter().for_each(|i| scope_names(i, false, into)),
        _ => {}
    }
}

#[test]
fn the_textmate_grammar_names_every_word_of_the_language() {
    let grammar = parse_json(TEXTMATE);
    assert_eq!(
        grammar.get("scopeName").and_then(Json::as_str),
        Some("source.komorebi-ir"),
        "the grammar's scope"
    );
    let mut res = Vec::new();
    patterns(&grammar, &mut res);
    let words = language_words();
    assert!(
        words.len() > 100 && words.iter().any(|(w, _)| w == "handler"),
        "the words of the language were not read ({} found)",
        words.len()
    );
    let missing: Vec<String> = words
        .iter()
        .filter(|(w, _)| !res.iter().any(|re| names_word(re, w)))
        .map(|(w, from)| format!("`{w}` (from {from})"))
        .collect();
    assert!(
        missing.is_empty(),
        "the TextMate grammar does not name: {}",
        missing.join(", ")
    );
}

/// A name between backticks (a word of the text as a name, R50) is coloured as a name:
/// no word's pattern matches inside backticks, and each pattern that names a declared
/// name or a name in an expression takes the backticked form too.
/// The sets of the text's names (`syntax::CALLS`, `CONVERSIONS`, `COMBINATIONS`), which
/// the writer and `K120` read, are the checker's: the calls of its family match and its
/// compute functions, the scalar types, the provided combinations (R50).
#[test]
fn the_texts_sets_of_the_languages_names_are_the_checkers() {
    use komorebi_ir::syntax;
    let set = |words: &[&str]| words.iter().map(|w| w.to_string()).collect::<BTreeSet<_>>();
    let ops = read("src/komorebi-ir/check/ops.rs");
    let body = read("src/komorebi-ir/check/body.rs");
    let mut calls = quoted_words(region(
        &ops,
        "check/ops.rs",
        "let family = match what {",
        "self.unknown(n);",
    ));
    calls.extend(quoted_words(region(
        &body,
        "check/body.rs",
        "UNARY: [&str;",
        "];",
    )));
    assert_eq!(calls, set(&syntax::CALLS));
    let provided = quoted_words(region(&body, "check/body.rs", "PROVIDED: [&str;", "];"));
    assert_eq!(provided, set(&syntax::COMBINATIONS));
    let scalars: Vec<&str> = komorebi_ir::types::ScalarType::ALL
        .iter()
        .map(|s| s.name())
        .collect();
    assert_eq!(set(&scalars), set(&syntax::CONVERSIONS));
}

#[test]
fn the_textmate_grammar_colours_a_name_between_backticks_as_a_name() {
    let mut res = Vec::new();
    patterns(&parse_json(TEXTMATE), &mut res);
    let text = read(TEXTMATE);
    assert!(
        !text.contains("(?<![\\\\w.])"),
        "a word's pattern that would match inside backticks"
    );
    let quoted = "(?:`[A-Za-z_]\\\\w*`|[A-Za-z_]\\\\w*)";
    let declared = text
        .lines()
        .filter(|l| l.contains("^\\\\s*(") && l.contains(quoted))
        .count();
    assert!(
        declared >= 8,
        "the declarations' names ({declared} take backticks)"
    );
    let tree_sitter = read("packaging/tree-sitter-komorebi-ir/grammar.js");
    assert!(
        tree_sitter.contains("/`[A-Za-z_][A-Za-z0-9_]*`/"),
        "tree-sitter's identifier takes a name between backticks"
    );
}

#[test]
fn every_scope_the_grammar_gives_is_one_the_default_themes_colour() {
    let grammar = parse_json(TEXTMATE);
    let mut names = Vec::new();
    scope_names(&grammar, true, &mut names);
    assert!(
        names.len() > 50,
        "the scopes were not read ({} found)",
        names.len()
    );
    let outside: BTreeSet<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|name| {
            !ALLOWED_ROOTS.iter().any(|root| {
                name.strip_prefix(root)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
            })
        })
        .collect();
    assert!(
        outside.is_empty(),
        "scopes outside the roots the default themes colour: {outside:?}"
    );
}

#[test]
fn the_extension_files_parse_and_say_what_vscode_needs() {
    let package = parse_json("packaging/vscode/komorebi-ir/package.json");
    let contributes = package
        .get("contributes")
        .expect("package.json contributes");
    let language = &contributes.get("languages").expect("languages").as_arr()[0];
    assert_eq!(
        language.get("id").and_then(Json::as_str),
        Some("komorebi-ir")
    );
    let extensions: Vec<&str> = language
        .get("extensions")
        .expect("extensions")
        .as_arr()
        .iter()
        .filter_map(Json::as_str)
        .collect();
    assert_eq!(extensions, [".kom"]);
    let grammar = &contributes.get("grammars").expect("grammars").as_arr()[0];
    assert_eq!(
        grammar.get("scopeName").and_then(Json::as_str),
        Some("source.komorebi-ir")
    );
    let base = Path::new("packaging/vscode/komorebi-ir");
    for (node, key) in [(language, "configuration"), (grammar, "path")] {
        let file = node.get(key).and_then(Json::as_str).expect("a file named");
        let file = base.join(file.trim_start_matches("./"));
        assert!(root().join(&file).is_file(), "{} exists", file.display());
    }
    let config = parse_json("packaging/vscode/komorebi-ir/language-configuration.json");
    assert_eq!(
        config
            .get("comments")
            .and_then(|c| c.get("lineComment"))
            .and_then(Json::as_str),
        Some("//")
    );
}
