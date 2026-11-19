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

/// The lessons Lower cannot give a structure yet: `t06_rows` for the format of the
/// blocks `arrive` yields, which the checker leaves open.
const LESSONS_WAITING: [&str; 1] = ["t06_rows"];

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

/// The fragments that check clean and that Lower cannot give a structure yet:
/// `state_filled` for a member's count named by no declaration (`K_TILES`), which the
/// checker lets pass.
const FRAGMENTS_WAITING: [&str; 1] = ["state_filled"];

/// The fragments that check clean lower to a structure that passes `validate`.
#[test]
fn the_fragments_that_check_lower_and_validate() {
    let failed: Vec<String> = modules("fragments")
        .into_iter()
        .filter(|(name, _)| !FRAGMENTS_WAITING.contains(&name.as_str()))
        .filter(|(_, text)| komorebi_ir::check(text).is_ok())
        .filter_map(|(name, text)| lowered(&text).err().map(|e| format!("{name}: {e}")))
        .collect();
    assert!(failed.is_empty(), "{}", failed.join("\n\n"));
}

#[test]
fn the_fragments_waiting_are_exact() {
    for (name, text) in modules("fragments") {
        if komorebi_ir::check(&text).is_err() {
            continue;
        }
        let waits = FRAGMENTS_WAITING.contains(&name.as_str());
        assert_eq!(
            lowered(&text).is_err(),
            waits,
            "{name}: the list of fragments waiting"
        );
    }
}
