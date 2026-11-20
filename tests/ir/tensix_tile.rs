//! The tile compiler's port against today's compiler, stage by stage, with no card:
//! today's `tests/ir/tensix/tile_units.py` runs on a corpus of tile functions and
//! writes each one's unit program as data and as today's listing; the port reads the
//! data and must print the same listing byte for byte, costs included, and encode
//! every word of the operation table as today does.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

use komorebi_ir::backend::tensix::tile::unit::{
    decode, encode, Arg, Call, Check, Constant, Count, Insn, Item, Loop, Pass, Product, Runtime,
    Sync, UnitProgram, CPUS, OPS,
};

/// Between the items of a list in one field.
const SEP: char = '\x1f';

/// Today's compiler run on the corpus into a folder: the folder, or none when the
/// tests are told to skip Python.
fn today() -> Option<PathBuf> {
    if std::env::var("KOMO_IR_NO_PYTHON").is_ok_and(|v| v == "1") {
        eprintln!("the tile identity skipped: KOMO_IR_NO_PYTHON=1");
        return None;
    }
    let folder = Path::new(env!("CARGO_TARGET_TMPDIR")).join("tensix-tile");
    let python = std::env::var("KOMO_IR_PYTHON").unwrap_or_else(|_| "python3".into());
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = Command::new(&python)
        .arg(root.join("tests/ir/tensix/tile_units.py"))
        .arg(&folder)
        .env("PYTHONPATH", root.join("src"))
        .output()
        .unwrap_or_else(|e| panic!("{python}: {e} (KOMO_IR_NO_PYTHON=1 skips this test)"));
    assert!(
        out.status.success(),
        "today's tile compiler failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(folder)
}

/// An argument from the data: `i<int>` or `s<name>`.
fn arg(text: &str) -> Arg {
    match text.strip_prefix('i').map(str::parse::<i64>) {
        Some(Ok(n)) => Arg::Num(n),
        _ => Arg::Name(text.strip_prefix('s').unwrap_or(text).to_string()),
    }
}

fn list(field: &str) -> Vec<&str> {
    field.split(SEP).filter(|s| !s.is_empty()).collect()
}

/// The items of a loop nest from lines of (depth, fields), from `at`, at `depth`.
fn items(lines: &[(usize, Vec<&str>)], at: &mut usize, depth: usize) -> Vec<Item> {
    let mut out = Vec::new();
    while let Some((d, f)) = lines.get(*at) {
        if *d < depth {
            break;
        }
        *at += 1;
        let item = match f[0] {
            "LOOP" => {
                let count = match f[3].strip_prefix('n') {
                    Some(n) => Count::Times(n.parse().expect("a count")),
                    None => {
                        let r = list(&f[3][1..]);
                        let value: i64 = r[2].parse().expect("a value");
                        Count::Runtime(Runtime {
                            text: r[1].to_string(),
                            of: Rc::new(move |rows| {
                                assert_eq!(rows, 32, "the data holds the value at 32 rows");
                                value
                            }),
                            guard: r[0] == "1",
                        })
                    }
                };
                let body = items(lines, at, depth + 1);
                Item::Loop(Loop {
                    name: f[2].to_string(),
                    count,
                    body,
                    unroll: f[4] == "1",
                })
            }
            "INSN" => {
                let args: Vec<(&str, Arg)> = list(f[3])
                    .into_iter()
                    .map(|kv| {
                        let (k, v) = kv.split_once('=').expect("a field");
                        (k, arg(v))
                    })
                    .collect();
                let mut insn = Insn::new(f[2], &args, f[4]).unwrap_or_else(|e| panic!("{e}"));
                insn.uses = list(f.get(5).copied().unwrap_or(""))
                    .into_iter()
                    .map(arg)
                    .collect();
                Item::Insn(insn)
            }
            "SYNC" => Item::Sync(Sync {
                cpu: f[2].into(),
                other: f[3].into(),
                text: f[4].into(),
                note: f[5].into(),
            }),
            "PRODUCT" => Item::Product(Product {
                text: f[2].into(),
                note: f[3].into(),
                synced: f[4] == "1",
                words: list(f[5])
                    .into_iter()
                    .map(|w| {
                        let (op, block) = w.split_once(':').expect("a word");
                        let block = (block != "None").then(|| block.parse().expect("a block"));
                        (op.to_string(), block)
                    })
                    .collect(),
                insns: Vec::new(),
                wormhole: Vec::new(),
            }),
            "CALL" => Item::Call(Call {
                text: f[2].into(),
                note: f[3].into(),
            }),
            other => panic!("an item {other}"),
        };
        out.push(item);
    }
    out
}

/// A unit program from today's data.
fn unit(text: &str) -> UnitProgram {
    let mut u = UnitProgram::new("", "");
    let mut lines = text
        .lines()
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .peekable();
    while let Some(f) = lines.next() {
        match f[0] {
            "UNIT" => u = UnitProgram::new(f[1], f[2]),
            "CONST" => {
                let (kind, v) = f[2].split_at(1);
                let value = match kind {
                    "f" => Constant::Float(v.parse().expect("a float")),
                    "i" => Constant::Int(v.parse().expect("an int")),
                    _ => Constant::Text(v.to_string()),
                };
                u.constants.push((f[1].parse().expect("a register"), value));
            }
            "CLOBBER" => u.clobbers.push(f[1].to_string()),
            "INIT" | "TILE" | "PART" => {
                let mut nest = Vec::new();
                while let Some(next) = lines.peek() {
                    let Ok(depth) = next.get(1).map_or("x", |d| *d).parse::<usize>() else {
                        break;
                    };
                    nest.push((depth, lines.next().expect("peeked")));
                }
                let mut at = 0;
                let body = items(&nest, &mut at, 0);
                assert_eq!(at, nest.len(), "every item read");
                let cpu = CPUS.iter().position(|c| *c == f[1]);
                match (f[0], cpu) {
                    ("INIT", Some(i)) => u.init[i] = body,
                    ("TILE", Some(i)) => u.tile[i] = body,
                    _ => u.parts.push((f[1].to_string(), body)),
                }
            }
            "PASS" => u.passes.push(Pass {
                kind: f[1].into(),
                name: f[2].into(),
                done: list(f.get(3).copied().unwrap_or(""))
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            }),
            "CHECK" => u.checks.push(Check {
                rule: f[1].into(),
                text: f[2].into(),
                verdict: f[3].into(),
            }),
            "NOTCHECKED" => u.not_checked.push(f[1].into()),
            other => panic!("a line {other}"),
        }
    }
    u
}

/// The first line where `a` and `b` differ.
fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}:\n  today: {x}\n  rust:  {y}", i + 1);
        }
    }
    format!(
        "{} lines today, {} ported",
        a.lines().count(),
        b.lines().count()
    )
}

#[test]
fn each_unit_programs_listing_and_costs_are_todays() {
    let Some(folder) = today() else { return };
    let refused = std::fs::read_to_string(folder.join("refused.txt")).expect("refused.txt");
    assert_eq!(refused, "", "today refuses none of the corpus");
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&folder)
        .expect("the folder")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "unit"))
        .collect();
    cases.sort();
    assert_eq!(cases.len(), 24, "the corpus: {cases:?}");
    for case in &cases {
        let data = std::fs::read_to_string(case).expect("the data");
        let want = std::fs::read_to_string(case.with_extension("txt")).expect("the listing");
        let got = unit(&data)
            .dump()
            .unwrap_or_else(|e| panic!("{case:?}: {e}"))
            + "\n";
        assert!(
            got == want,
            "{case:?} differs, {}",
            first_difference(&want, &got)
        );
    }
}

#[test]
fn every_word_of_the_table_encodes_and_decodes_as_today() {
    let Some(folder) = today() else { return };
    let text = std::fs::read_to_string(folder.join("encode.txt")).expect("encode.txt");
    let fields = |text: &str| -> Vec<(String, i64)> {
        text.split(',')
            .filter(|s| !s.is_empty())
            .map(|kv| {
                let (k, v) = kv.split_once('=').expect("a field");
                (k.to_string(), v.parse().expect("a value"))
            })
            .collect()
    };
    let (mut words, mut wide) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "WORD" => {
                let args = fields(f[2]);
                let borrowed: Vec<(&str, i64)> =
                    args.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                let word = encode(f[1], &borrowed).unwrap_or_else(|e| panic!("{line}: {e}"));
                assert_eq!(word.to_string(), f[3], "{line}");
                let (name, back) = decode(word).expect("decoded");
                assert_eq!(name, f[4], "{line}");
                let back: Vec<(String, i64)> =
                    back.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
                assert_eq!(back, fields(f[5]), "{line}");
                words += 1;
            }
            "WIDE" => {
                let args = fields(f[2]);
                let borrowed: Vec<(&str, i64)> =
                    args.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                let refused = encode(f[1], &borrowed).expect_err(line);
                assert_eq!(refused.0, f[3], "{line}");
                wide += 1;
            }
            "UNKNOWN" => {
                let word: u32 = f[1].parse().expect("a word");
                assert_eq!(decode(word).is_none().to_string(), f[2].to_lowercase());
            }
            other => panic!("a line {other}"),
        }
    }
    // ten words of each operation of the table; each field one bit too wide
    let fields: usize = OPS.iter().map(|o| o.args.len()).sum();
    assert_eq!((words, wide), (10 * OPS.len(), fields));
}
