//! The examples: every lesson of `examples/ir/lessons/` is the corpus lesson of the same
//! name, each has at least one run script, and every run script under `examples/ir/`
//! gives the report stored beside it (`<script>.report`). A script belongs to the
//! `.kom` of its folder whose stem is the longest prefix of the script's stem.
//!
//! A script whose run is refused (a load refused by name) has as its report the one
//! line `refused: <the refusal>`.
//!
//! `KOMO_IR_BLESS=1` writes each script's report instead of comparing it.

use std::path::{Path, PathBuf};

use komorebi_ir::api::script;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// The files of `dir` with the extension `ext`, in order.
fn files(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        panic!("{} is readable", dir.display());
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == ext))
        .collect();
    paths.sort();
    paths
}

/// The `.run` files under `dir`, in every sub-folder, in order.
fn scripts(dir: &Path) -> Vec<PathBuf> {
    let mut found = files(dir, "run");
    let Ok(entries) = std::fs::read_dir(dir) else {
        panic!("{} is readable", dir.display());
    };
    let mut subs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    subs.sort();
    for sub in subs {
        found.extend(scripts(&sub));
    }
    found
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .map_or(String::new(), |s| s.to_string_lossy().into_owned())
}

/// The `.kom` of the script's folder whose stem is the longest prefix of its stem.
fn module_of(script: &Path) -> Option<PathBuf> {
    let name = stem(script);
    let dir = script.parent()?;
    files(dir, "kom")
        .into_iter()
        .filter(|k| name.starts_with(&stem(k)))
        .max_by_key(|k| stem(k).len())
}

#[test]
fn every_lesson_in_examples_is_the_corpus_lesson() {
    let corpus = root().join("tests/ir/corpus/golden/lessons");
    let examples = root().join("examples/ir/lessons");
    let names = |dir: &Path| -> Vec<String> {
        files(dir, "kom")
            .iter()
            .map(|p| {
                p.file_name()
                    .map_or(String::new(), |n| n.to_string_lossy().into())
            })
            .collect()
    };
    assert_eq!(names(&examples), names(&corpus), "the lessons");
    for name in names(&corpus) {
        let read =
            |dir: &Path| std::fs::read(dir.join(&name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            read(&examples) == read(&corpus),
            "{name} is not the corpus lesson"
        );
    }
}

#[test]
fn every_lesson_has_a_script() {
    let dir = root().join("examples/ir/lessons");
    let scripts = scripts(&dir);
    let lonely: Vec<String> = files(&dir, "kom")
        .iter()
        .filter(|k| !scripts.iter().any(|s| module_of(s).as_ref() == Some(*k)))
        .map(|k| k.display().to_string())
        .collect();
    assert!(lonely.is_empty(), "lessons with no script: {lonely:?}");
}

#[test]
fn every_example_script_gives_its_report() {
    let bless = std::env::var_os("KOMO_IR_BLESS").is_some();
    let all = scripts(&root().join("examples/ir"));
    assert!(!all.is_empty(), "no scripts");
    let mut failed = Vec::new();
    for path in all {
        let at = path.display().to_string();
        let Some(kom) = module_of(&path) else {
            failed.push(format!("{at}: no module in its folder"));
            continue;
        };
        let source = std::fs::read_to_string(&kom).unwrap_or_else(|e| panic!("{at}: {e}"));
        let module = match komorebi_ir::api::build(&source) {
            Ok(m) => m,
            Err(d) => {
                failed.push(format!("{at}: its module does not build: {d:?}"));
                continue;
            }
        };
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{at}: {e}"));
        let parsed = match script::parse(&text) {
            Ok(s) => s,
            Err(e) => {
                failed.push(format!("{at}: {e}"));
                continue;
            }
        };
        // a run refused (a load refused by name) has that refusal as its report
        let report = match script::run(&module, &parsed) {
            Ok(r) => r.print(),
            Err(e) => format!("refused: {e}\n"),
        };
        let stored = path.with_extension("run.report");
        if bless {
            std::fs::write(&stored, &report).unwrap_or_else(|e| panic!("{at}: {e}"));
            continue;
        }
        match std::fs::read_to_string(&stored) {
            Ok(want) if want == report => {}
            Ok(_) => failed.push(format!("{at}: the report is not {}", stored.display())),
            Err(_) => failed.push(format!("{at}: no report {}", stored.display())),
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
