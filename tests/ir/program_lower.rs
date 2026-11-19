//! Lower: the lessons and the fragments that check clean lower to a structure that
//! passes `validate`, and three lessons lower to exactly their printed text in
//! `golden/lowered/`. With `KOMO_IR_BLESS` set, the printed texts are written instead.

use std::path::{Path, PathBuf};

use komorebi_ir::lower::lower;
use komorebi_ir::program::{print, validate};

fn corpus_dir() -> PathBuf {
    match std::env::var_os("KOMO_IR_CORPUS") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/corpus"),
    }
}

/// The `.kom` files of one folder of the golden corpus, in order.
fn modules(sub: &str) -> Vec<(String, String)> {
    let dir = corpus_dir().join("golden").join(sub);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        panic!("the corpus folder {} is readable", dir.display());
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "kom"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path).expect("a corpus file is readable");
            let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
            (stem.unwrap_or_default(), text)
        })
        .collect()
}

/// The printed structure of a module's text, or why there is none.
fn lowered(text: &str) -> Result<String, String> {
    let checked = komorebi_ir::check(text).map_err(|d| format!("does not check: {d:?}"))?;
    let module = lower(&checked).map_err(|e| format!("does not lower: {e:?}"))?;
    let violations = validate(&module);
    if !violations.is_empty() {
        return Err(format!(
            "does not validate: {violations:?}\n{}",
            print(&module)
        ));
    }
    Ok(print(&module))
}

/// Lowers every module of a folder; the failures, by name.
fn failures(sub: &str, skip: &[&str]) -> Vec<String> {
    modules(sub)
        .into_iter()
        .filter(|(name, _)| !skip.contains(&name.as_str()))
        .filter_map(|(name, text)| lowered(&text).err().map(|e| format!("{sub}/{name}: {e}")))
        .collect()
}

/// The lessons with an operation Lower does not read yet: it reads the forms of the
/// text until the checker names each call's and member's operation, and only the
/// forms these lessons do not need are read so far. `t08_trace` waits for a type of
/// a `view` parameter, which names no format; `t13_hinted` for the format of what
/// `matmul` yields.
const LESSONS_WAITING: [&str; 10] = [
    "t02_count",
    "t04_compute",
    "t05_total",
    "t06_rows",
    "t07_views",
    "t08_trace",
    "t10_steps",
    "t12_spread",
    "t13_hinted",
    "t15_handoff",
];

#[test]
fn the_lessons_lower_and_validate() {
    let failed = failures("lessons", &LESSONS_WAITING);
    assert!(failed.is_empty(), "{}", failed.join("\n\n"));
}

#[test]
fn the_waiting_list_is_exact() {
    for (name, text) in modules("lessons") {
        let waits = LESSONS_WAITING.contains(&name.as_str());
        assert_eq!(
            lowered(&text).is_err(),
            waits,
            "{name}: the list of lessons waiting"
        );
    }
}

#[test]
fn three_lessons_lower_to_their_printed_text() {
    let dir = corpus_dir().join("golden").join("lowered");
    for (name, text) in modules("lessons") {
        if !["t01_scale", "t09_walk", "t11_solve"].contains(&name.as_str()) {
            continue;
        }
        let printed = lowered(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        let path = dir.join(format!("{name}.txt"));
        if std::env::var_os("KOMO_IR_BLESS").is_some() {
            std::fs::write(&path, &printed).expect("the golden file is writable");
            continue;
        }
        let expected = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("{} is readable", path.display()));
        assert_eq!(printed, expected, "{name}");
    }
}
