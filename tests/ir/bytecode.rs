//! The byte-code program: every module the corpus lowers reads back as written; the
//! lessons' three hashes are as recorded; a comment, a hint and an order change exactly
//! the hashes they should; the reader refuses by name what it does not read, and no
//! input makes it panic. A file is loaded only where its text would check: its derived
//! parts are the ones Derive gives again, and Verify's rules hold.
//!
//! The recorded hashes are checked by `cargo test` and by `cargo test --release` alike:
//! the bytes do not depend on the build.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use komorebi_ir::bytecode::{self, decode, encode, hashes, hex, version, Hashes};
use komorebi_ir::program::derive::Cause;
use komorebi_ir::program::{print, Module};

fn golden_dir() -> PathBuf {
    match std::env::var_os("KOMO_IR_CORPUS") {
        Some(dir) => PathBuf::from(dir).join("golden"),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/corpus/golden"),
    }
}

/// The `.kom` files of one folder of the golden corpus, in order: each name and text.
fn modules(sub: &str) -> Vec<(String, String)> {
    let dir = golden_dir().join(sub);
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
            let name = path
                .file_stem()
                .map_or(String::new(), |s| s.to_string_lossy().into_owned());
            let text = std::fs::read_to_string(&path).expect("a corpus file is readable");
            (name, text)
        })
        .collect()
}

/// The module a text lowers to, derived, if it checks and lowers (Verify not run, so
/// that the modules of `invalid/` that lower are read too).
fn lowered(text: &str) -> Option<Module> {
    let parsed = komorebi_ir::parse(text).ok()?;
    let checked = komorebi_ir::check::check_module(text, parsed).ok()?;
    let mut module = komorebi_ir::lower::lower(&checked).ok()?;
    komorebi_ir::derive::derive(&mut module);
    Some(module)
}

/// The module of a text that passes the whole front.
fn built(text: &str) -> Module {
    komorebi_ir::api::build(text).unwrap_or_else(|d| panic!("{d:?}"))
}

#[test]
fn every_module_the_corpus_lowers_reads_back_as_written() {
    let mut read = 0;
    for sub in ["lessons", "reference", "run_fails", "fragments", "invalid"] {
        for (name, text) in modules(sub) {
            let Some(module) = lowered(&text) else {
                continue;
            };
            let bytes = encode(&module);
            let back = decode(&bytes).unwrap_or_else(|r| panic!("{sub}/{name}: {r:?}"));
            assert!(back == module, "{sub}/{name}: the module read back differs");
            assert_eq!(print(&back), print(&module), "{sub}/{name}");
            assert!(
                encode(&back) == bytes,
                "{sub}/{name}: written again, other bytes"
            );
            read += 1;
        }
    }
    // The fifteen lessons, the reference, the two of run_fails and the whole fragments
    // at least.
    assert!(read >= 21, "only {read} modules lowered");
}

#[test]
fn one_module_encoded_twice_gives_the_same_bytes() {
    for (name, text) in modules("lessons") {
        let a = built(&text);
        let b = built(&text);
        assert!(encode(&a) == encode(&b), "{name}: built twice, other bytes");
        assert!(
            encode(&a) == encode(&a),
            "{name}: encoded twice, other bytes"
        );
    }
}

fn hash_line(name: &str, h: &Hashes) -> String {
    format!(
        "{name} {} {} {}",
        hex(&h.semantic),
        hex(&h.build),
        hex(&h.full)
    )
}

/// The three hashes of each lesson, as `hashes.txt` records them; with `KOMO_IR_BLESS`
/// set, the test writes them instead.
#[test]
fn the_lessons_hashes_are_as_recorded() {
    let lines: Vec<String> = modules("lessons")
        .iter()
        .map(|(name, text)| hash_line(name, &hashes(&built(text))))
        .collect();
    let got = lines.join("\n") + "\n";
    let path = golden_dir().join("hashes.txt");
    if std::env::var_os("KOMO_IR_BLESS").is_some() {
        std::fs::write(&path, &got).expect("hashes.txt is writable");
        return;
    }
    let want = std::fs::read_to_string(&path).expect("hashes.txt is readable");
    assert_eq!(want.lines().count(), 15, "fifteen lessons recorded");
    assert!(want == got, "the hashes changed; now:\n{got}");
}

fn lesson(name: &str) -> String {
    modules("lessons")
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, text)| text)
        .unwrap_or_else(|| panic!("the lesson {name}"))
}

/// `text` with `from` replaced by `to`, once; `from` must be there.
fn edited(text: &str, from: &str, to: &str) -> String {
    assert!(text.contains(from), "the lesson has {from:?}");
    text.replacen(from, to, 1)
}

#[test]
fn a_comment_added_changes_the_full_hash_only() {
    let text = lesson("t13_hinted");
    let a = hashes(&built(&text));
    let b = hashes(&built(&edited(
        &text,
        "tile [32, 32]\n",
        "// a comment\ntile [32, 32]\n",
    )));
    assert_eq!(a.semantic, b.semantic);
    assert_eq!(a.build, b.build);
    assert_ne!(a.full, b.full);
}

#[test]
fn a_hint_changed_leaves_the_semantic_hash_only() {
    let text = lesson("t13_hinted");
    let a = hashes(&built(&text));
    let b = hashes(&built(&edited(&text, "\"AAAB\"", "\"AABB\"")));
    assert_eq!(a.semantic, b.semantic);
    assert_ne!(a.build, b.build);
    assert_ne!(a.full, b.full);
}

#[test]
fn two_declarations_swapped_change_every_hash() {
    let text = lesson("t04_compute");
    let gate = "func gate(a: tile<f32>, b: tile<f32>) -> tile<f32> {\n    return silu(a) * b\n}\n";
    let clamp =
        "func clamp01(a: tile<f32>) -> tile<f32> {\n    return minimum(maximum(a, 0.0), 1.0)\n}\n";
    let swapped = edited(
        &text,
        &format!("{gate}\n{clamp}"),
        &format!("{clamp}\n{gate}"),
    );
    let a = hashes(&built(&text));
    let b = hashes(&built(&swapped));
    assert_ne!(a.semantic, b.semantic);
    assert_ne!(a.build, b.build);
    assert_ne!(a.full, b.full);
}

fn refusal(bytes: &[u8]) -> bytecode::Refusal {
    decode(bytes).expect_err("refused")
}

fn with_version(bytes: &[u8], version: u32) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out[4..8].copy_from_slice(&version.to_le_bytes());
    out
}

#[test]
fn another_version_is_refused_by_name() {
    let bytes = encode(&built(&lesson("t01_scale")));
    let newer = refusal(&with_version(&bytes, version::VERSION + 1));
    assert_eq!(newer.code, version::NEWER);
    assert!(
        newer
            .what
            .contains(&format!("version {}", version::VERSION + 1))
            && newer
                .what
                .contains(&format!("version {}", version::VERSION)),
        "{newer:?}"
    );
    assert_eq!(
        refusal(&with_version(&bytes, u32::MAX)).code,
        version::NEWER
    );
    let older = refusal(&with_version(&bytes, version::VERSION - 1));
    assert_eq!(older.code, version::OLDER);
    assert!(older.what.contains("version 0"), "{older:?}");
}

#[test]
fn bytes_the_writer_would_not_write_are_refused() {
    let bytes = encode(&built(&lesson("t01_scale")));
    assert_eq!(refusal(b"").code, version::NOT_BYTE_CODE);
    let mut magic = bytes.clone();
    magic[0] = b'X';
    assert_eq!(refusal(&magic).code, version::NOT_BYTE_CODE);

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(refusal(&trailing).code, version::MALFORMED);

    // The lengths of the parts do not sum to the file.
    let mut lengths = bytes.clone();
    lengths[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(refusal(&lengths).code, version::MALFORMED);

    // A byte moved from S to H: each walk then ends elsewhere than its length.
    let mut moved = bytes.clone();
    let s = u64::from_le_bytes(moved[8..16].try_into().expect("8 bytes"));
    let h = u64::from_le_bytes(moved[16..24].try_into().expect("8 bytes"));
    moved[8..16].copy_from_slice(&(s - 1).to_le_bytes());
    moved[16..24].copy_from_slice(&(h + 1).to_le_bytes());
    assert_eq!(refusal(&moved).code, version::MALFORMED);

    // The module's name given a count of 2^32 - 1 bytes.
    let mut count = bytes.clone();
    count[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
    let r = refusal(&count);
    assert_eq!(r.code, version::MALFORMED);
    assert_eq!(r.at, Some(32));
}

/// Decodes `bytes`, turning a panic into an error that names the case.
fn decode_inside(case: &str, bytes: &[u8]) -> Result<Option<Module>, String> {
    match catch_unwind(AssertUnwindSafe(|| decode(bytes))) {
        Ok(Ok(module)) => Ok(Some(module)),
        Ok(Err(_)) => Ok(None),
        Err(_) => Err(format!("{case}: decode panicked")),
    }
}

#[test]
fn every_prefix_of_a_lesson_is_refused() {
    let mut panics = Vec::new();
    for (name, text) in modules("lessons") {
        let bytes = encode(&built(&text));
        for cut in 0..bytes.len() {
            match catch_unwind(AssertUnwindSafe(|| decode(&bytes[..cut]))) {
                Ok(Err(r)) => assert!(
                    r.code == version::NOT_BYTE_CODE || r.code == version::MALFORMED,
                    "{name}, {cut} bytes: {r:?}"
                ),
                Ok(Ok(_)) => panic!("{name}: {cut} bytes read as a module"),
                Err(_) => panics.push(format!("{name}, {cut} bytes")),
            }
        }
    }
    assert!(panics.is_empty(), "decode panicked: {panics:?}");
}

/// Every byte of `bytes` flipped three ways: the panics of decode and of load, and how
/// many flips read as a module.
fn flips(name: &str, bytes: &[u8]) -> (Vec<String>, usize) {
    let mut panics = Vec::new();
    let mut read = 0;
    for at in 0..bytes.len() {
        for mask in [0x01u8, 0x80, 0xff] {
            let mut flipped = bytes.to_vec();
            flipped[at] ^= mask;
            let case = format!("{name}, byte {at} ^ {mask:#04x}");
            match decode_inside(&case, &flipped) {
                Ok(Some(module)) => {
                    read += 1;
                    if catch_unwind(AssertUnwindSafe(|| komorebi_ir::api::load(&flipped))).is_err()
                    {
                        panics.push(format!("{case}: load panicked"));
                    }
                    assert!(
                        encode(&module) == flipped,
                        "{case}: read, written otherwise"
                    );
                }
                Ok(None) => {}
                Err(panic) => panics.push(panic),
            }
        }
    }
    (panics, read)
}

/// Every byte of every lesson flipped three ways: refused, or read as a module that
/// validates and is written back as exactly the flipped bytes (and that `api::load`
/// derives again and verifies); never a panic. One thread a lesson.
#[test]
fn flipped_bytes_never_panic() {
    let lessons: Vec<(String, Vec<u8>)> = modules("lessons")
        .into_iter()
        .map(|(name, text)| {
            let bytes = encode(&built(&text));
            (name, bytes)
        })
        .collect();
    let results: Vec<(Vec<String>, usize)> = std::thread::scope(|scope| {
        let handles: Vec<_> = lessons
            .iter()
            .map(|(name, bytes)| scope.spawn(move || flips(name, bytes)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a lesson's flips run to their end"))
            .collect()
    });
    let panics: Vec<&String> = results.iter().flat_map(|(p, _)| p).collect();
    assert!(panics.is_empty(), "{} panics: {panics:?}", panics.len());
    // Some flips give another valid module (a name's letter, a span's byte).
    assert!(results.iter().map(|(_, read)| read).sum::<usize>() > 0);
}

/// The tables of `docs/ir/bytecode.md` are the numbers the code writes.
#[test]
fn the_page_gives_the_numbers_the_code_writes() {
    let page = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/ir/bytecode.md");
    let text = std::fs::read_to_string(&page).expect("docs/ir/bytecode.md is readable");
    let mut tables: Vec<(String, Vec<(u8, String)>)> = Vec::new();
    let mut in_tables = false;
    for line in text.lines() {
        if line.starts_with("## ") {
            in_tables = line == "## The numbers";
        } else if in_tables {
            if let Some(name) = line.strip_prefix("### ") {
                tables.push((name.trim_matches('`').to_string(), Vec::new()));
            } else if let Some(row) = line.strip_prefix('|').and_then(|r| r.strip_suffix('|')) {
                let cells: Vec<&str> = row.split('|').map(str::trim).collect();
                if let (Some(table), [n, name]) = (tables.last_mut(), cells.as_slice()) {
                    if let Ok(n) = n.parse::<u8>() {
                        table.1.push((n, name.trim_matches('`').to_string()));
                    }
                }
            }
        }
    }
    let code: Vec<(String, Vec<(u8, String)>)> = version::TABLES
        .iter()
        .map(|(name, t)| {
            (
                name.to_string(),
                t.iter().map(|(n, v)| (*n, v.to_string())).collect(),
            )
        })
        .collect();
    assert_eq!(tables, code);
}

#[test]
fn every_lesson_loads() {
    for (name, text) in modules("lessons") {
        let module = built(&text);
        let loaded =
            komorebi_ir::api::load(&encode(&module)).unwrap_or_else(|r| panic!("{name}: {r:?}"));
        assert!(loaded == module, "{name}: loaded, another module");
    }
}

#[test]
fn derived_parts_that_derive_does_not_give_are_refused() {
    let module = built(&lesson("t13_hinted"));
    let mut forged = module.clone();
    let forward = &mut forged.programs[0].handlers[1];
    assert_eq!(forward.name, "forward");
    let removed = forward.effects.tensors.pop();
    assert!(removed.is_some(), "the handler writes a tensor");
    let bytes = encode(&forged);
    assert!(
        decode(&bytes).is_ok(),
        "validate accepts the forged effects"
    );
    let refusals = komorebi_ir::api::load(&bytes).expect_err("refused");
    assert_eq!(refusals.len(), 1);
    assert_eq!(refusals[0].code, version::DERIVED);
    assert!(
        refusals[0].what.contains("the handler Hinted.forward"),
        "{refusals:?}"
    );

    // A lesson with a requirement a construct causes, that cause taken out (a cause
    // `Stated` is what the text states, and Derive keeps it as it reads it).
    let mut forged = modules("lessons")
        .iter()
        .map(|(_, text)| built(text))
        .find(|m| {
            m.requirements
                .iter()
                .any(|r| r.causes.iter().any(|c| *c != Cause::Stated))
        })
        .expect("a lesson's construct causes a requirement");
    for r in &mut forged.requirements {
        r.causes.retain(|c| *c == Cause::Stated);
    }
    let refusals = komorebi_ir::api::load(&encode(&forged)).expect_err("refused");
    assert_eq!(refusals[0].code, version::DERIVED);
}

#[test]
fn a_module_verify_refuses_is_refused_with_its_findings() {
    for (case, code) in [
        ("k701_race_order", "K701"),
        ("k601_past_unproven", "K601"),
        ("k205_fallback_touches", "K205"),
    ] {
        let text = modules("invalid")
            .into_iter()
            .find(|(n, _)| n == case)
            .map(|(_, text)| text)
            .unwrap_or_else(|| panic!("the case {case}"));
        let module = lowered(&text).unwrap_or_else(|| panic!("{case} lowers"));
        let bytes = encode(&module);
        assert!(decode(&bytes).is_ok(), "{case}: validate accepts it");
        let refusals = komorebi_ir::api::load(&bytes).expect_err("refused");
        assert!(
            refusals.iter().any(|r| r.code == code),
            "{case}: {refusals:?}"
        );
    }
}
