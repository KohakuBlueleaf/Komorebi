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
use komorebi_ir::program::body::InstId;
use komorebi_ir::program::derive::Cause;
use komorebi_ir::program::notes::{Anchor, Origin};
use komorebi_ir::program::{print, BodyRef, Module, Part};
use komorebi_ir::types::{Count, HandlerId, ProgramId};

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
fn a_funcs_hint_leaves_the_semantic_hash_only() {
    let text = lesson("t04_compute");
    let a = hashes(&built(&text));
    let b = hashes(&built(&edited(
        &text,
        "func gram(a: tile<f32>) -> tile<f32> {",
        "func gram(a: tile<f32>) -> tile<f32> @tensix.fidelity(HiFi2) {",
    )));
    assert_eq!(a.semantic, b.semantic);
    assert_ne!(a.build, b.build);
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
    assert!(
        older
            .what
            .contains(&format!("version {}", version::VERSION - 1)),
        "{older:?}"
    );
    assert_eq!(refusal(&with_version(&bytes, 0)).code, version::OLDER);
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

/// A chain call's range (R47) is written and read back: one of numbers, one ending at
/// a module parameter bound at build (its value).
#[test]
fn a_call_s_range_of_the_group_loads_as_written() {
    let text = "module parts

tile [32, 32]
format f32

param HALF: u32 = 3 @bind(build)

program P {
    handler copy(x: tensor<f32>, y: tensor<f32>) @on(group) {
        let out = sink(partition(y.tiles(0, 8), worker.rank, group.size), block = 1)
        for xb in stream(partition(x.tiles(0, 8), worker.rank, group.size), block = 1) {
            out.push(xb)
        }
    }

    chain parts(x: tensor<f32>, y: tensor<f32>) @on(group) {
        call copy(x, y)
        call copy(x, y) @on(group[0:4])
        call copy(x, y) @on(group[1:HALF])
    }
}
";
    let module = built(text);
    let parts: Vec<Option<Part>> = module.programs[0].chains[0]
        .calls
        .iter()
        .map(|c| c.part)
        .collect();
    assert_eq!(
        parts,
        [
            None,
            Some(Part {
                first: Count::Const(0),
                end: Count::Const(4)
            }),
            Some(Part {
                first: Count::Const(1),
                end: Count::Const(3)
            }),
        ]
    );
    let loaded = komorebi_ir::api::load(&encode(&module)).expect("loads");
    assert!(loaded == module, "loaded, another module");
}

/// The corpus module of a message's body (`constructs/c01_message_body`): its bodies,
/// a `send` with `body =`, `m.body.words()` and a `forward` with `as =` are written
/// and read back, and the program loads.
#[test]
fn a_module_with_bodies_round_trips() {
    let (_, text) = modules("constructs")
        .into_iter()
        .find(|(name, _)| name == "c01_message_body")
        .expect("the construct is in the corpus");
    let module = built(&text);
    assert_eq!(
        module.messages.iter().map(|m| m.body).collect::<Vec<_>>(),
        [16, 16]
    );
    let bytes = encode(&module);
    let read = decode(&bytes).expect("read back");
    assert!(read == module, "read back, another module");
    assert_eq!(encode(&read), bytes);
    let loaded = komorebi_ir::api::load(&bytes).expect("loads");
    assert!(loaded == module, "loaded, another module");
}

/// `all_gather` in a handler (a value, with its `publish` list) and over the copies of
/// a replicated tensor (no value) is written, read back and loaded as built.
#[test]
fn a_module_with_all_gather_round_trips() {
    let path = golden_dir().join("constructs/c02_all_gather.kom");
    let text = std::fs::read_to_string(&path).expect("a corpus file is readable");
    let module = built(&text);
    let printed = print(&module);
    assert_eq!(printed.matches("AllGather").count(), 2, "{printed}");
    let bytes = encode(&module);
    let back = decode(&bytes).expect("read back");
    assert!(back == module, "the module read back differs");
    assert_eq!(print(&back), printed);
    assert!(encode(&back) == bytes, "written again, other bytes");
    let loaded = komorebi_ir::api::load(&bytes).expect("loads");
    assert!(loaded == module, "loaded, another module");
}

/// A module of two tile shapes, and declarations of the second (a function, a handler)
/// beside those of the default, is written, read back and loaded as built.
#[test]
fn a_module_with_two_tile_shapes_round_trips() {
    let path = golden_dir().join("constructs/c05_tile_shapes.kom");
    let text = std::fs::read_to_string(&path).expect("a corpus file is readable");
    let module = built(&text);
    let shapes: Vec<(u64, u64)> = module.tiles.iter().map(|t| (t.rows, t.cols)).collect();
    assert_eq!(shapes, [(2, 2), (4, 4)]);
    assert_eq!(
        module.funcs.iter().map(|f| f.tile).collect::<Vec<_>>(),
        [0, 1]
    );
    let handlers = &module.programs[0].handlers;
    assert_eq!(handlers.iter().map(|h| h.tile).collect::<Vec<_>>(), [0, 1]);
    let printed = print(&module);
    assert!(printed.contains("tile [2, 2], [4, 4]"), "{printed}");
    assert_eq!(printed.matches("  tile [4, 4]").count(), 2, "{printed}");
    let bytes = encode(&module);
    let back = decode(&bytes).expect("read back");
    assert!(back == module, "the module read back differs");
    assert_eq!(print(&back), printed);
    assert!(encode(&back) == bytes, "written again, other bytes");
    let loaded = komorebi_ir::api::load(&bytes).expect("loads");
    assert!(loaded == module, "loaded, another module");
}

/// A store of scalars, of each of its three values (a scalar, a state member named
/// whole, the words of a tile), is written, read back and loaded as built.
#[test]
fn a_module_with_a_store_of_scalars_round_trips() {
    let path = golden_dir().join("constructs/c03_store_scalars.kom");
    let text = std::fs::read_to_string(&path).expect("a corpus file is readable");
    let module = built(&text);
    let printed = print(&module);
    assert_eq!(printed.matches("StoreScalars").count(), 3, "{printed}");
    let bytes = encode(&module);
    let back = decode(&bytes).expect("read back");
    assert!(back == module, "the module read back differs");
    assert_eq!(print(&back), printed);
    assert!(encode(&back) == bytes, "written again, other bytes");
    let loaded = komorebi_ir::api::load(&bytes).expect("loads");
    assert!(loaded == module, "loaded, another module");
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

/// T1 with an origin as a frontend records one: a class script's lines, the handler on
/// its `def` line, two instructions on the statement they were made from. The file is a
/// name, not a path of a machine.
fn with_origin() -> Module {
    let mut module = built(&lesson("t01_scale"));
    assert!(
        module.notes.origin.is_none(),
        "a module of .kom text has no origin"
    );
    let body = BodyRef::Handler(ProgramId(0), HandlerId(0));
    module.notes.origin = Some(Origin {
        files: vec!["t1_class.py".into()],
        lines: vec![
            (
                0,
                23,
                "    def run(self, x: km.Tensor, y: km.Tensor, first: int, tiles: int, k: float):"
                    .into(),
            ),
            (
                0,
                24,
                "        src = self.partition(x.pages(first, tiles))".into(),
            ),
            (
                0,
                25,
                "        out = km.Stream(y[src], write=True, block=1)  # one partition".into(),
            ),
        ],
        places: vec![
            (Anchor::Handler(ProgramId(0), HandlerId(0)), 0, 23, 23),
            (Anchor::Inst(body, InstId(5)), 0, 24, 24),
            (Anchor::Inst(body, InstId(6)), 0, 24, 25),
        ],
    });
    module
}

#[test]
fn an_origin_changes_the_full_hash_only() {
    let plain = hashes(&built(&lesson("t01_scale")));
    let module = with_origin();
    assert!(decode(&encode(&module)).is_ok(), "the origin validates");
    let traced = hashes(&module);
    assert_eq!(plain.semantic, traced.semantic);
    assert_eq!(plain.build, traced.build);
    assert_ne!(plain.full, traced.full);
}

#[test]
fn a_module_with_an_origin_round_trips_byte_for_byte() {
    let module = with_origin();
    let bytes = encode(&module);
    let read = decode(&bytes).expect("read back");
    assert_eq!(read.notes.origin, module.notes.origin);
    assert_eq!(read, module);
    assert_eq!(encode(&read), bytes);
}

/// A case of a bad origin: its name, the edit that makes it, what the refusal says.
type OriginCase = (&'static str, fn(&mut Origin), &'static str);

#[test]
fn an_origin_that_does_not_hold_is_refused_by_name() {
    let cases: Vec<OriginCase> = vec![
        (
            "a file past the files",
            |o| o.places[0].1 = 1,
            "past the files",
        ),
        ("a line of no file", |o| o.lines[2].0 = 3, "past the files"),
        ("line 0", |o| o.lines[0].1 = 0, "lines count from 1"),
        ("first after last", |o| o.places[2].2 = 26, "first to last"),
        (
            "a line not held",
            |o| o.places[2].3 = 27,
            "line 26 of file 0, which is not held",
        ),
        (
            "a range of every line",
            |o| o.places[2].3 = u32::MAX,
            "line 26 of file 0, which is not held",
        ),
        (
            "lines out of order",
            |o| o.lines.swap(0, 1),
            "(file, number) order",
        ),
        (
            "an anchor twice",
            |o| o.places[2].0 = o.places[1].0,
            "or one twice",
        ),
        (
            "an anchor of nothing",
            |o| {
                o.places[2].0 =
                    Anchor::Inst(BodyRef::Handler(ProgramId(0), HandlerId(0)), InstId(9999))
            },
            "an anchor that names nothing",
        ),
    ];
    for (case, edit, said) in cases {
        let mut module = with_origin();
        if let Some(origin) = module.notes.origin.as_mut() {
            edit(origin);
        }
        let r = refusal(&encode(&module));
        assert_eq!(r.code, version::INVALID, "{case}: {r:?}");
        assert!(r.what.contains(said), "{case}: {}", r.what);
    }
}

/// Version 6 is frozen: its number tables are held by one digest. A change to a table
/// is a new `VERSION`, with its own digest here.
#[test]
fn version_6_is_frozen() {
    use komorebi_ir::bytecode::hash::{hex, sha256};
    let mut text = String::new();
    for (table, rows) in version::TABLES {
        for (number, name) in rows.iter() {
            text.push_str(&format!("{table}:{number}={name};"));
        }
    }
    assert_eq!(version::VERSION, 6);
    assert_eq!(
        hex(&sha256(text.as_bytes())),
        "317cbf28513ebe1aea6a3eeb1cdf1c46b21268e790fc61255aa5a6c5abbf5a51",
        "{text}"
    );
}
