//! The tile compiler's port against today's compiler, stage by stage, with no card:
//! today's `tests/ir/tensix/tile_units.py` runs on a corpus of tile functions and
//! writes each one's KernelIR, and its unit program as data and as today's listing.
//! The port must trace the same functions written in the IR (`tile_units.kom`) into
//! the same KernelIR and the same vector-unit function (its SFPI code included), print
//! the same listing from the data byte for byte, costs included, and encode every word
//! of the operation table as today does.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

use komorebi_ir::backend::tensix::tile::chain::{Chain, Source};
use komorebi_ir::backend::tensix::tile::ir::{kernel_ir, Attr, KernelIR};
use komorebi_ir::backend::tensix::tile::lower::{stream_chain, unit_program};
use komorebi_ir::backend::tensix::tile::unit::{
    decode, encode, Arg, Call, Check, Constant, Count, Insn, Item, Loop, Pass, Product, Runtime,
    Sync, UnitProgram, CPUS, OPS,
};

/// Between the items of a list in one field.
const SEP: char = '\x1f';

/// Today's compiler run on the corpus into a folder, once for all the tests (they run
/// at once): the folder, or none when the tests are told to skip Python.
fn today() -> Option<PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER.get_or_init(run_today).clone()
}

/// Today's compiler run on the corpus.
fn run_today() -> Option<PathBuf> {
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
    // `five` reads five tiles: its chain runs as two stages, its unit is refused;
    // `by5`'s chain is refused (each_functions_chain_is_todays)
    let refused: Vec<(&str, &str)> = refused.lines().filter_map(|l| l.split_once('\t')).collect();
    let names: Vec<&str> = refused.iter().map(|r| r.0).collect();
    assert_eq!(names, ["five", "by5"], "{refused:?}");
    let five = "komorebi.dsl.tile: five: the unit program reads at most 4 operand tiles";
    assert!(refused[0].1.starts_with(five), "{}", refused[0].1);
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&folder)
        .expect("the folder")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "unit"))
        .collect();
    cases.sort();
    assert_eq!(cases.len(), 28, "the corpus: {cases:?}");
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

/// The corpus in the IR's text, built.
fn corpus() -> komorebi_ir::program::Module {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/tensix/tile_units.kom");
    let text = std::fs::read_to_string(path).expect("tile_units.kom");
    komorebi_ir::api::build(&text).unwrap_or_else(|e| panic!("{e:?}"))
}

/// The number of the function named `name`.
fn func(module: &komorebi_ir::program::Module, name: &str) -> komorebi_ir::types::FuncId {
    let at = module.funcs.iter().position(|f| f.name == name);
    komorebi_ir::types::FuncId(at.unwrap_or_else(|| panic!("no func {name}")) as u32)
}

#[test]
fn each_function_traces_to_todays_kernel_ir() {
    let Some(folder) = today() else { return };
    let module = corpus();
    let mut cases: Vec<String> = std::fs::read_dir(&folder)
        .expect("the folder")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "ir"))
        .filter_map(|p| Some(p.file_stem()?.to_str()?.to_string()))
        .collect();
    cases.sort();
    let funcs: Vec<&str> = module
        .funcs
        .iter()
        .filter_map(|f| f.name.strip_prefix("t_"))
        .collect();
    let mut sorted = funcs.clone();
    sorted.sort();
    assert_eq!(
        sorted, cases,
        "a func `t_<case>` for each case and no other"
    );
    // `smooth_let` binds its 0.5 once: today's tracer makes a node at each use;
    // `times_on_blocks` is `times` declared on blocks
    let pairs = funcs
        .iter()
        .map(|c| (format!("t_{c}"), c.to_string()))
        .chain([
            ("smooth_let".to_string(), "smooth".to_string()),
            ("times_on_blocks".to_string(), "times".to_string()),
        ]);
    for (f, case) in pairs {
        let want = std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
        let ir = kernel_ir(&module, func(&module, &f), &case)
            .unwrap_or_else(|r| panic!("{f}: {}", r.what));
        let got = ir.dump() + "\n";
        assert!(
            got == want,
            "{f} differs, {}",
            first_difference(&want, &got)
        );
    }
}

#[test]
fn what_the_trace_would_not_make_is_refused_by_name() {
    let text = "module refused\n\ntile [32, 32]\nformat f32, i32\n\n\
        func blocky(a: tile<i32>) -> tile<i32> {\n    return a\n}\n\n\
        func scalars(a: tile<f32>, k: f32) -> tile<f32> {\n    return a * (k * 2.0)\n}\n\n\
        func spread(a: tile<f32>) -> tile<f32> {\n    return a.T\n}\n\n\
        func none(k: f32) -> tile<f32> {\n    return zeros\n}\n";
    let module = komorebi_ir::api::build(text).unwrap_or_else(|e| panic!("{e:?}"));
    for (f, what) in [
        ("blocky", "a tile of I32"),
        ("scalars", "arithmetic on two scalars"),
        ("spread", "the operation Transpose"),
        ("none", "a function of no tile"),
    ] {
        let refusal = kernel_ir(&module, func(&module, f), f).expect_err(f);
        assert_eq!(refusal.code, "K964", "{f}");
        assert!(refusal.what.contains(what), "{f}: {}", refusal.what);
    }
}

/// A chain as tile_units.py prints it, its earlier stages first.
fn chain_lines(ir: &KernelIR, chain: &Chain, lines: &mut Vec<String>) {
    if let Some(prev) = &chain.prev {
        chain_lines(ir, prev, lines);
    }
    let func = &chain.func;
    let pairs = if chain.pairs { "True" } else { "False" };
    lines.push(format!(
        "CHAIN {} out={} pairs={pairs}",
        func.name, chain.out
    ));
    for (k, leaf) in chain.operands.iter().enumerate() {
        let what = match leaf.source {
            Source::Prev => "var".to_string(),
            Source::Node(n) => {
                let node = &ir.arena[n.0];
                match node.attrs.first() {
                    Some((_, Attr::View(v))) if node.kind == "load" => format!("load {}", v.text()),
                    _ => node.kind.clone(),
                }
            }
        };
        lines.push(format!("operand {k} {what} {} None", leaf.kind));
    }
    for (k, s) in chain.values.iter().enumerate() {
        lines.push(format!("value {k} {}", s.key()));
    }
    let ops: Vec<String> = chain.ops.iter().map(|(k, n)| format!("{k}={n}")).collect();
    lines.push(format!("ops {}", ops.join(" ")));
    let fp32 = if func.fp32 { "True" } else { "False" };
    lines.push(format!(
        "fn operands={} scalars={} init={} fp32={fp32}",
        func.operands, func.scalars, func.init
    ));
    lines.extend(func.headers.iter().map(|h| format!("header {h}")));
    lines.extend(func.along.iter().map(|k| format!("along {k} None")));
    lines.extend(func.code.split('\n').map(str::to_string));
}

/// `(line <place>)` of a template's record as `(line ?)`: today's names the Python line
/// that traced the function, the port the IR's place of it (none for IR text).
fn placeless(text: &str) -> String {
    text.lines()
        .map(|l| {
            match (
                l.strip_prefix("template "),
                l.find(" (line "),
                l.find("): "),
            ) {
                (Some(_), Some(open), Some(close)) if open < close => {
                    format!("{} (line ?{}", &l[..open], &l[close..])
                }
                _ => l.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn each_functions_chain_is_todays() {
    let Some(folder) = today() else { return };
    let module = corpus();
    let pairs = module
        .funcs
        .iter()
        .filter_map(|f| Some((f.name.clone(), f.name.strip_prefix("t_")?.to_string())))
        .chain([("smooth_let".to_string(), "smooth".to_string())]);
    let mut compared = 0;
    for (f, case) in pairs {
        let ir = kernel_ir(&module, func(&module, &f), &case)
            .unwrap_or_else(|r| panic!("{f}: {}", r.what));
        let Ok(want) = std::fs::read_to_string(folder.join(format!("{case}.chain"))) else {
            // refused today: the same text, but for the place it names
            let refused = std::fs::read_to_string(folder.join("refused.txt")).expect("refused.txt");
            let today = refused
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{case}\t")))
                .unwrap_or_else(|| panic!("{case}: no chain and no refusal"));
            let error = stream_chain(&ir).expect_err(&f).text();
            assert_eq!(error, today.split(" (at ").next().unwrap_or(today), "{f}");
            compared += 1;
            continue;
        };
        let chain = stream_chain(&ir).unwrap_or_else(|e| panic!("{f}: {}", e.text()));
        let mut lines = Vec::new();
        chain_lines(&ir, &chain, &mut lines);
        lines.extend(
            chain
                .templates
                .iter()
                .map(|(k, v)| format!("template {k}: {v}")),
        );
        let (want, got) = (placeless(&want), placeless(&(lines.join("\n") + "\n")));
        assert!(
            got == want,
            "{f} differs, {}",
            first_difference(&want, &got)
        );
        compared += 1;
    }
    assert_eq!(compared, 31);
}

#[test]
fn each_unit_programs_listing_is_todays() {
    let Some(folder) = today() else { return };
    let module = corpus();
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("cases.txt");
    let refused = std::fs::read_to_string(folder.join("refused.txt")).expect("refused.txt");
    let (mut same, mut not_ported) = (Vec::new(), Vec::new());
    // `times_on_blocks`, `times` declared on blocks, gives `times`'s unit
    let blocks = "times\tfp32\t0\ttimes_on_blocks";
    for line in cases.lines().chain([blocks]) {
        let f: Vec<&str> = line.split('\t').collect();
        let (case, dest, band) = (f[0], f[1], f[2] == "1");
        let name = f.get(3).map_or(format!("t_{case}"), |n| n.to_string());
        let ir = kernel_ir(&module, func(&module, &name), case)
            .unwrap_or_else(|r| panic!("{name}: {}", r.what));
        let made = unit_program(&ir, dest, true, band);
        let today = refused
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{case}\t")));
        if let Some(today) = today {
            // refused today: the same text, but for the place it names (K962)
            let error = made.expect_err(case);
            assert_eq!(
                error.text(),
                today.split(" (at ").next().unwrap_or(today),
                "{case}"
            );
            assert_eq!(error.refusal(None).code, "K962", "{case}");
            continue;
        }
        let unit = match made {
            Ok(unit) => unit,
            Err(e) => {
                let refusal = e.refusal(None);
                assert_eq!(refusal.code, "K964", "{case}: {}", refusal.what);
                not_ported.push(format!("{case}: {}", e.text()));
                continue;
            }
        };
        let want =
            std::fs::read_to_string(folder.join(format!("{case}.txt"))).expect("the listing");
        let got = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
        assert!(
            got == want,
            "{name} differs, {}",
            first_difference(&want, &got)
        );
        same.push(name);
    }
    // the whole-tile arithmetic, `a * k` among it, with and without a load macro;
    // smooth's checks repair its code (an SFPNOP after its constant register's setup)
    let whole = [
        "t_times",
        "t_plus",
        "t_plus_bfp8",
        "t_smooth",
        "t_scaled_sum",
        "t_square",
        "t_neg",
        "t_twice_k",
        "t_chained",
        "times_on_blocks",
    ];
    assert_eq!(same, whole);
    let refused = [
        "plus_band: the band (a body on the live rows)",
        "gate: the vector function sigmoid",
        "silu_gate_bf16: the vector function sigmoid",
        "clamp01: the vector operation min",
        "leaky: the vector operation select",
        "project: a function that is one tile product",
        "exp: the vector function exp",
        "sigmoid: the vector function sigmoid",
        "silu: the vector function sigmoid",
        "tanh: the vector function tanh",
        "erf: the vector function erf",
        "log: the vector function log",
        "sqrt: the vector function sqrt",
        "rsqrt: the vector function rsqrt",
        "recip: the vector function recip",
        "relu: the vector operation max",
        "divide: the vector function recip",
        "floor: the vector operation select",
        "called: the vector operation select",
    ];
    let named: Vec<String> = not_ported
        .iter()
        .map(|l| l.replacen(": not ported", "", 1))
        .collect();
    assert_eq!(named, refused);
}
