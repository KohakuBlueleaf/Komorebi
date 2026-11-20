//! The layers of the crate: which top module may read which. The test reads the
//! crate's source, builds the graph of `crate::...` paths between the top modules
//! (comments not counted), and fails on any cycle and on any edge outside the allowed
//! list. The command (`bin/`) and `api` read everything.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// What each top module may read, besides itself.
const ALLOWED: &[(&str, &[&str])] = &[
    ("types", &[]),
    ("span", &[]),
    ("deep", &[]),
    ("errors", &["span"]),
    ("syntax", &["span", "errors", "deep"]),
    ("check", &["syntax", "types", "errors"]),
    ("program", &["types"]),
    ("lower", &["check", "program", "syntax", "types", "deep"]),
    ("derive", &["program", "types"]),
    ("interp", &["types", "program"]),
    ("verify", &["program", "types"]),
    ("bytecode", &["program", "types"]),
    ("backend", &["program", "types", "bytecode"]),
    (
        "bindings",
        &["api", "bytecode", "deep", "errors", "program", "types"],
    ),
];

/// The modules that read everything.
const READ_ALL: &[&str] = &["api"];

fn source_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/komorebi-ir")
}

/// Every `.rs` file under `dir`, in order.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        panic!("{} is readable", dir.display());
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

/// The top module a file belongs to: its folder under the source, or its own name;
/// `None` for `lib.rs` and the command.
fn top_module(path: &Path) -> Option<String> {
    let rel = path.strip_prefix(source_dir()).ok()?;
    let first = rel
        .components()
        .next()?
        .as_os_str()
        .to_string_lossy()
        .into_owned();
    let name = first.strip_suffix(".rs").unwrap_or(&first).to_string();
    (name != "lib" && name != "bin").then_some(name)
}

/// A line without its comment.
fn code_of(line: &str) -> &str {
    match line.find("//") {
        Some(at) => &line[..at],
        None => line,
    }
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The top modules a text names through `crate::`, grouped forms included
/// (`crate::{a::X, b}`).
fn named_modules(code: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = code;
    while let Some(at) = rest.find("crate::") {
        let before = rest[..at].chars().next_back();
        rest = &rest[at + "crate::".len()..];
        if before.is_some_and(is_ident) {
            continue;
        }
        if let Some(group) = rest.strip_prefix('{') {
            let mut depth = 1usize;
            let mut item_start = true;
            let mut name = String::new();
            for c in group.chars() {
                match c {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                if depth == 0 {
                    break;
                }
                if c == ',' && depth == 1 {
                    if !name.is_empty() {
                        out.insert(std::mem::take(&mut name));
                    }
                    item_start = true;
                } else if item_start && is_ident(c) {
                    name.push(c);
                } else if item_start && !name.is_empty() {
                    out.insert(std::mem::take(&mut name));
                    item_start = false;
                }
            }
            if !name.is_empty() {
                out.insert(name);
            }
        } else {
            let name: String = rest.chars().take_while(|c| is_ident(*c)).collect();
            if !name.is_empty() {
                out.insert(name);
            }
        }
    }
    out
}

/// The edges between top modules: who reads whom.
fn graph() -> BTreeMap<String, BTreeSet<String>> {
    let mut files = Vec::new();
    rust_files(&source_dir(), &mut files);
    let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for path in files {
        let Some(from) = top_module(&path) else {
            continue;
        };
        let text = std::fs::read_to_string(&path).expect("a source file is readable");
        // statements may span lines: join the code of every line
        let code: String = text.lines().map(code_of).collect::<Vec<_>>().join("\n");
        let entry = edges.entry(from.clone()).or_default();
        for to in named_modules(&code) {
            if to != from {
                entry.insert(to);
            }
        }
    }
    edges
}

/// A cycle of the graph, if there is one, as the modules along it.
fn cycle(edges: &BTreeMap<String, BTreeSet<String>>) -> Option<Vec<String>> {
    fn visit(
        node: &str,
        edges: &BTreeMap<String, BTreeSet<String>>,
        path: &mut Vec<String>,
        done: &mut BTreeSet<String>,
    ) -> Option<Vec<String>> {
        if let Some(at) = path.iter().position(|n| n == node) {
            let mut found = path[at..].to_vec();
            found.push(node.to_string());
            return Some(found);
        }
        if done.contains(node) {
            return None;
        }
        path.push(node.to_string());
        for next in edges.get(node).into_iter().flatten() {
            if let Some(found) = visit(next, edges, path, done) {
                return Some(found);
            }
        }
        path.pop();
        done.insert(node.to_string());
        None
    }
    let mut done = BTreeSet::new();
    for node in edges.keys() {
        if let Some(found) = visit(node, edges, &mut Vec::new(), &mut done) {
            return Some(found);
        }
    }
    None
}

#[test]
fn the_top_modules_read_each_other_in_no_cycle() {
    let edges = graph();
    if let Some(found) = cycle(&edges) {
        panic!("a cycle of modules: {}", found.join(" -> "));
    }
}

#[test]
fn every_top_module_reads_only_what_it_may() {
    let allowed: BTreeMap<&str, &[&str]> = ALLOWED.iter().copied().collect();
    let mut wrong = Vec::new();
    for (from, tos) in graph() {
        if READ_ALL.contains(&from.as_str()) {
            continue;
        }
        let Some(may) = allowed.get(from.as_str()) else {
            wrong.push(format!("{from}: a top module the list does not name"));
            continue;
        };
        for to in tos {
            if !may.contains(&to.as_str()) {
                wrong.push(format!("{from} reads {to}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn the_reader_finds_grouped_and_plain_paths() {
    let found = named_modules("use crate::{a::X, b, c::{d, e}};\nlet x = crate::f::g();");
    let want: BTreeSet<String> = ["a", "b", "c", "f"].iter().map(|s| s.to_string()).collect();
    assert_eq!(found, want);
    assert!(named_modules("let my_crate::x = 1;").is_empty());
}

#[test]
fn a_cycle_is_found() {
    let edge = |a: &str, b: &[&str]| (a.to_string(), b.iter().map(|s| s.to_string()).collect());
    let looped: BTreeMap<String, BTreeSet<String>> = [
        edge("errors", &["syntax"]),
        edge("syntax", &["errors", "deep"]),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        cycle(&looped),
        Some(vec![
            "errors".to_string(),
            "syntax".to_string(),
            "errors".to_string()
        ])
    );
    let fine: BTreeMap<String, BTreeSet<String>> = [
        edge("errors", &["span"]),
        edge("syntax", &["errors", "span"]),
    ]
    .into_iter()
    .collect();
    assert_eq!(cycle(&fine), None);
}
