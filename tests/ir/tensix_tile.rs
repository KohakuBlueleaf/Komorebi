//! The tile compiler's port against today's compiler, stage by stage, with no card:
//! today's `tests/ir/tensix/tile/tile_units.py` runs on a corpus of tile functions and
//! writes each one's KernelIR, and its unit program as data and as today's listing.
//! The port must trace the same functions written in the IR (`tile_units.kom`) into
//! the same KernelIR and the same vector-unit function (its SFPI code included), print
//! the same listing from the data byte for byte, costs included, and the same header
//! and listing after the emit, and encode every word of the operation table as today
//! does. Today's `tile_wormhole.py`, `tile_checks.py`, `tile_matrix.py` and
//! `tile_rows.py` give the Wormhole forms, the emit and its checks, the hazard checks,
//! the matrix unit's words and the per-row passes, on code of a few words that the
//! corpus does not reach. Today's `tile_rows_form.py` traces functions on a row of n
//! tiles (the row form), written in the IR on blocks in `tile_rows_form.kom`, and
//! makes the row form's unit of each row of 2 tiles from its KernelIR, which the port
//! reads back from today's text.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

use komorebi_ir::backend::tensix::tile::chain::{Chain, Source};
use komorebi_ir::backend::tensix::tile::checks::{
    check_config_nop, check_dst_window, check_mad_readers, check_saved, not_held,
};
use komorebi_ir::backend::tensix::tile::emit::{check_buffers, emit};
use komorebi_ir::backend::tensix::tile::ir::{
    kernel_ir, row_kernel_ir, Attr, Grid, Int, KernelIR, NodeId, Store, TensorParam, View,
};
use komorebi_ir::backend::tensix::tile::lower::{
    common_values, drop_copies, per_row_prepass, stream_chain, unit_program, FIDELITY,
};
use komorebi_ir::backend::tensix::tile::matrix::{tile_products, wormhole_products, Word};
use komorebi_ir::backend::tensix::tile::py::float_repr;
use komorebi_ir::backend::tensix::tile::r#macro::{check_back_door, check_templates};
use komorebi_ir::backend::tensix::tile::rows::{
    check_broadcast, check_lanes, chunk_cost, chunk_program, row_program, Layout, Layouts,
};
use komorebi_ir::backend::tensix::tile::schedule::interleave;
use komorebi_ir::backend::tensix::tile::unit::{
    decode, encode, insns, Arg, Call, Check, Constant, Count, Insn, Item, Loop, Pass, Product,
    Runtime, Sync, UnitError, UnitProgram, CPUS, MATH, OPS,
};
use komorebi_ir::backend::tensix::tile::wormhole::forms;
use komorebi_ir::backend::tensix::unit::{fidelity, row_layout, unit as tile_unit, DstLayout};
use komorebi_ir::program::body::{InstId, Op};
use komorebi_ir::program::notes::{Anchor, Origin};
use komorebi_ir::program::BodyRef;

/// Between the items of a list in one field.
const SEP: char = '\x1f';

/// Today's compiler run on the corpus into a folder, once for all the tests (they run
/// at once): the folder, or none when the tests are told to skip Python.
fn today() -> Option<PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| run_today("tile_units.py", "tensix-tile"))
        .clone()
}

/// Today's Wormhole forms, emit and checks on the units of a few words
/// (`tile_wormhole.py`), once for all the tests.
fn today_words() -> Option<PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| run_today("tile_wormhole.py", "tensix-tile-words"))
        .clone()
}

/// Today's compiler run by `script` into the folder `name`.
fn run_today(script: &str, name: &str) -> Option<PathBuf> {
    if std::env::var("KOMO_IR_NO_PYTHON").is_ok_and(|v| v == "1") {
        eprintln!("the tile identity skipped: KOMO_IR_NO_PYTHON=1");
        return None;
    }
    let folder = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let python = std::env::var("KOMO_IR_PYTHON").unwrap_or_else(|_| "python3".into());
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = Command::new(&python)
        .arg(root.join("tests/ir/tensix/tile").join(script))
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

/// An instruction from its fields: op, fields, note, uses, its run-time immediate.
fn insn(f: &[&str]) -> Insn {
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
    insn.runtime = f.get(6).copied().unwrap_or("").to_string();
    insn
}

/// A product's words of `kind` (its own, Wormhole's), the lines from `at` at `depth`.
fn product_words(
    lines: &[(usize, Vec<&str>)],
    at: &mut usize,
    depth: usize,
    kind: &str,
) -> Vec<Insn> {
    let mut out = Vec::new();
    while let Some((d, f)) = lines.get(*at) {
        if *d != depth || f[0] != kind {
            break;
        }
        out.push(insn(f));
        *at += 1;
    }
    out
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
            "INSN" => Item::Insn(insn(f)),
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
                insns: product_words(lines, at, depth + 1, "PINSN"),
                wormhole: product_words(lines, at, depth + 1, "PWORMHOLE"),
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
            "UNIT" => {
                u = UnitProgram::new(f[1], f[2]);
                u.band = f.get(3) == Some(&"1");
                u.scalars = list(f.get(4).copied().unwrap_or(""))
                    .into_iter()
                    .map(str::to_string)
                    .collect();
            }
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
    // `by5`'s and `four_registers`' chains are refused (each_functions_chain_is_todays)
    let refused: Vec<(&str, &str)> = refused.lines().filter_map(|l| l.split_once('\t')).collect();
    let names: Vec<&str> = refused.iter().map(|r| r.0).collect();
    assert_eq!(names, ["four_registers", "five", "by5"], "{refused:?}");
    let five = "komorebi.dsl.tile: five: the unit program reads at most 4 operand tiles";
    assert!(refused[1].1.starts_with(five), "{}", refused[1].1);
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&folder)
        .expect("the folder")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "unit"))
        .collect();
    cases.sort();
    assert_eq!(cases.len(), 46, "the corpus: {cases:?}");
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
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/tensix/tile/tile_units.kom");
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

/// A listing's templates pass with each `(line <place>)` as `(line ?)`, as
/// `placeless` writes a chain's record.
fn listing_placeless(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for l in text.split('\n') {
        if !l.starts_with("  pass algebra templates: ") {
            out.push(l.to_string());
            continue;
        }
        let mut line = String::new();
        let mut rest = l;
        while let Some(open) = rest.find(" (line ") {
            let after = &rest[open + " (line ".len()..];
            let Some(close) = after.find(')') else { break };
            line.push_str(&rest[..open]);
            line.push_str(" (line ?)");
            rest = &after[close + 1..];
        }
        line.push_str(rest);
        out.push(line);
    }
    out.join("\n")
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
    assert_eq!(compared, 50);
}

/// What a caller reads of a unit that its listing does not print, as
/// `tile_units.py`'s `fields` writes it.
fn unit_fields(unit: &UnitProgram) -> String {
    let list = |v: &[usize]| v.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
    let operands: Vec<&str> = unit
        .operands
        .iter()
        .map(|o| o.as_deref().unwrap_or("-"))
        .collect();
    let products: Vec<String> = unit
        .products
        .iter()
        .map(|p| format!("{}:{}:{}", p.dst, p.a, p.w))
        .collect();
    format!(
        "operands\t{}\nrow\t{}\ncol\t{}\nresults\t{}\nproducts\t{}\nfidelity\t{}\n",
        operands.join(","),
        list(&unit.row_operands),
        list(&unit.col_operands),
        unit.results,
        products.join(","),
        unit.fidelity.as_deref().unwrap_or("-")
    )
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
        let at = fidelity(&module, func(&module, &name)).unwrap_or_else(|r| panic!("{}", r.what));
        let made = unit_program(&ir, dest, true, band, &at);
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
        let want = std::fs::read_to_string(folder.join(format!("{case}.fields"))).expect("fields");
        let got = unit_fields(&unit);
        assert!(
            got == want,
            "{name}'s fields differ, {}",
            first_difference(&want, &got)
        );
        let want =
            std::fs::read_to_string(folder.join(format!("{case}.txt"))).expect("the listing");
        // today's templates pass names the Python line that traced the function
        let want = listing_placeless(&want);
        let got =
            listing_placeless(&(unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n"));
        assert!(
            got == want,
            "{name} differs, {}",
            first_difference(&want, &got)
        );
        // the emit: the header, and the listing with the emit's checks
        let mut unit = unit;
        let words = emit(&mut unit).unwrap_or_else(|e| panic!("{case}: {e}"));
        for (file, got) in [
            ("hpp", words),
            (
                "emitted.txt",
                unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n",
            ),
        ] {
            let want = std::fs::read_to_string(folder.join(format!("{case}.{file}")))
                .unwrap_or_else(|e| panic!("{case}.{file}: {e}"));
            let (want, got) = (listing_placeless(&want), listing_placeless(&got));
            assert!(
                got == want,
                "{name}'s {file} differs, {}",
                first_difference(&want, &got)
            );
        }
        same.push(name);
    }
    let not_emitted = std::fs::read_to_string(folder.join("not_emitted.txt")).expect("the file");
    assert_eq!(not_emitted, "", "today's emit refuses none of the corpus");
    // the whole-tile arithmetic, `a * k` among it, with and without a load macro;
    // smooth's checks repair its code (an SFPNOP after its constant register's setup);
    // exp, recip and sigmoid with their lane conditions; gate's 2-word constants tie
    // for the registers (first loaded first); two reciprocals share their 2.0;
    // min, max and the selects: SFPGT's tests, their inverses, the tests against 0;
    // every vector function, rsqrt's integer constant register among them; a function
    // that is one tile product, at today's default fidelity; products before the vector
    // code (one, one beside a loaded tile, two); a tile's row 0 spread (`.row()`)
    let whole = [
        "t_times",
        "t_plus",
        "t_plus_bfp8",
        "t_gate",
        "t_silu_gate_bf16",
        "t_clamp01",
        "t_leaky",
        "t_smooth",
        "t_project",
        "t_project_hifi2",
        "t_by_row",
        "t_by_col",
        "t_col_prepass",
        "t_col_cheap",
        "t_col_relu",
        "t_col_min",
        "t_col_select",
        "t_col_silu",
        "t_col_poly",
        "t_project_silu",
        "t_project_bias",
        "t_two_products",
        "t_scaled_sum",
        "t_exp",
        "t_sigmoid",
        "t_silu",
        "t_tanh",
        "t_erf",
        "t_log",
        "t_sqrt",
        "t_rsqrt",
        "t_recip",
        "t_relu",
        "t_square",
        "t_neg",
        "t_divide",
        "t_floor",
        "t_twice_k",
        "t_chained",
        "t_two_recips",
        "t_pick_ge",
        "t_pick_le_ne",
        "t_pick_eq",
        "t_zero_tests",
        "t_called",
        "times_on_blocks",
    ];
    assert_eq!(same, whole);
    let refused = ["plus_band: the band (a body on the live rows)"];
    let named: Vec<String> = not_ported
        .iter()
        .map(|l| l.replacen(": not ported", "", 1))
        .collect();
    assert_eq!(named, refused);
}

/// A word as `op fields ; note`.
fn word_text(x: &Insn) -> String {
    let fields: Vec<String> = x.args.iter().map(Arg::to_string).collect();
    format!("{} {} ; {}", x.op.name, fields.join(","), x.note)
}

/// Every word of a function, loop bodies once, a product's both images.
fn all_words<'a>(items: &'a [Item], out: &mut Vec<&'a Insn>) {
    for item in items {
        match item {
            Item::Loop(l) => all_words(&l.body, out),
            Item::Product(p) => out.extend(p.insns.iter().chain(&p.wormhole)),
            Item::Insn(x) => out.push(x),
            Item::Call(_) | Item::Sync(_) => {}
        }
    }
}

/// A check of a unit program.
type UnitCheck = fn(&UnitProgram) -> Result<String, UnitError>;

#[test]
fn wormholes_forms_and_the_emit_with_its_checks_are_todays_on_a_few_words() {
    let Some(folder) = today_words() else { return };
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&folder)
        .expect("the folder")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "unit"))
        .collect();
    cases.sort();
    assert_eq!(cases.len(), 33, "the cases: {cases:?}");
    let checks: [(&str, UnitCheck); 3] = [
        ("one-producer", check_buffers),
        ("back-door", check_back_door),
        ("macro-templates", check_templates),
    ];
    for case in &cases {
        let data = std::fs::read_to_string(case).expect("the data");
        let want = std::fs::read_to_string(case.with_extension("out")).expect("today's");
        let mut made = unit(&data);
        forms(&mut made).unwrap_or_else(|e| panic!("{case:?}: {e}"));
        let mut words = Vec::new();
        for function in [&made.init[MATH], &made.tile[MATH]]
            .into_iter()
            .chain(made.parts.iter().map(|(_, items)| items))
        {
            all_words(function, &mut words);
        }
        let mut out: Vec<String> = words
            .iter()
            .filter(|x| !x.wormhole.is_empty())
            .map(|x| {
                let form: Vec<String> = x.wormhole.iter().map(word_text).collect();
                format!("FORM\t{}\t{}", word_text(x), form.join("\t"))
            })
            .collect();
        for (name, check) in checks {
            let verdict = check(&unit(&data)).unwrap_or_else(|e| format!("refused: {e}"));
            out.push(format!("CHECK\t{name}\t{verdict}"));
        }
        let emitted = emit(&mut unit(&data)).unwrap_or_else(|e| format!("refused: {e}"));
        out.push(format!("EMIT\n{emitted}"));
        let got = out.join("\n") + "\n";
        assert!(
            got == want,
            "{case:?} differs, {}",
            first_difference(&want, &got)
        );
    }
}

/// Today's hazard checks on the units of a few words (`tile_checks.py`), once for all
/// the tests.
fn today_checks() -> Option<PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| run_today("tile_checks.py", "tensix-tile-checks"))
        .clone()
}

/// The `.unit` files of a folder, in name order.
fn unit_files(folder: &Path) -> Vec<PathBuf> {
    let mut cases: Vec<PathBuf> = std::fs::read_dir(folder)
        .expect("the folder")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "unit"))
        .collect();
    cases.sort();
    cases
}

/// A function's `tensix.fidelity` hint: no hint is today's default; a name today's
/// matrix unit does not have is refused with its text (the matrix test pins it); two
/// hints, or one not of one name, are refused by name.
#[test]
fn a_fidelity_hint_is_read_and_an_unknown_one_refused() {
    let given = |hints: &str| {
        let text = format!(
            "module hinted\n\ntile [32, 32]\nformat bf16, f32\n\n\
             func p(x: tile<bf16>, w: tile<bf16>) -> tile<f32>{hints} {{\n\
             \x20   return matmul(x, w)\n}}\n"
        );
        let module = komorebi_ir::api::build(&text).unwrap_or_else(|d| panic!("{d:?}"));
        fidelity(&module, func(&module, "p")).map_err(|r| (r.code, r.what))
    };
    assert_eq!(given(""), Ok("HiFi4".to_string()));
    assert_eq!(given(" @tensix.fidelity(LoFi)"), Ok("LoFi".to_string()));
    let today = tile_products("HiFi5", 1, 1, 0).expect_err("HiFi5").0;
    assert_eq!(
        today,
        "no fidelity 'HiFi5': one of LoFi, HiFi2, HiFi3, HiFi4"
    );
    // today's unit_program refuses it at its entry with its prefix and the name (a
    // module of no origin: no line)
    let whole = format!("komorebi.dsl.tile: p: {today}");
    assert_eq!(given(" @tensix.fidelity(HiFi5)"), Err(("K962", whole)));
    let named = |what: &str| {
        Err((
            "K964",
            format!("the Tensix backend does not take {what} yet"),
        ))
    };
    assert_eq!(
        given(" @tensix.fidelity(HiFi2) @tensix.fidelity(HiFi4)"),
        named("two `tensix.fidelity` hints on a function")
    );
    assert_eq!(
        given(" @tensix.fidelity(\"HiFi2\")"),
        named("a `tensix.fidelity` hint other than of one name")
    );
}

#[test]
fn the_hazard_checks_are_todays_on_a_few_words() {
    let Some(folder) = today_checks() else { return };
    let cases = unit_files(&folder);
    assert_eq!(cases.len(), 15, "the cases: {cases:?}");
    let notes = |code: &[Item]| -> String {
        let words: Vec<String> = code
            .iter()
            .filter_map(|x| match x {
                Item::Insn(x) => Some(format!("{}:{}", x.op.name, x.note)),
                _ => None,
            })
            .collect();
        words.join(" ")
    };
    let refused = |e: UnitError| format!("refused: {e}");
    for case in &cases {
        let data = std::fs::read_to_string(case).expect("the data");
        let want = std::fs::read_to_string(case.with_extension("out")).expect("today's");
        let mut made = unit(&data);
        let flat: Vec<Insn> = insns(&made.tile[MATH]).into_iter().cloned().collect();
        let as_items = || -> Vec<Item> { flat.iter().cloned().map(Item::Insn).collect() };
        let mut out = Vec::new();
        let mut code = as_items();
        let repairs = check_mad_readers(&mut code).unwrap_or_else(|e| panic!("{e}"));
        out.push(format!("MAD\t{repairs}\t{}", notes(&code)));
        let mut code = as_items();
        let repairs = check_config_nop(&mut code).unwrap_or_else(|e| panic!("{e}"));
        out.push(format!("CONFIG\t{repairs}\t{}", notes(&code)));
        let refs: Vec<&Insn> = flat.iter().collect();
        out.push(format!(
            "SAVED\t{}",
            check_saved(&refs).unwrap_or_else(refused)
        ));
        let held: Vec<String> = flat
            .windows(2)
            .map(|p| {
                let written = p[0].writes().unwrap_or_else(|e| panic!("{e}"));
                if not_held(&p[1], &written) {
                    "True"
                } else {
                    "False"
                }
                .to_string()
            })
            .collect();
        out.push(format!("HELD\t{}", held.join(" ")));
        let windows: Vec<String> = (1..=3)
            .map(|tiles| {
                made.checks.clear();
                match check_dst_window(&mut made, tiles) {
                    Ok(()) => made.checks[0].verdict.clone(),
                    Err(e) => refused(e),
                }
            })
            .collect();
        out.push(format!("WINDOW\t{}", windows.join("\t")));
        let got = out.join("\n") + "\n";
        assert!(
            got == want,
            "{case:?} differs, {}",
            first_difference(&want, &got)
        );
    }
}

#[test]
fn the_matrix_units_words_are_todays_at_each_fidelity_and_shape() {
    let Some(folder) = run_today("tile_matrix.py", "tensix-tile-matrix") else {
        return;
    };
    let want = std::fs::read_to_string(folder.join("matrix.out")).expect("today's");
    let lines = |made: Result<Vec<Word>, UnitError>| -> Vec<String> {
        match made {
            Err(e) => vec![format!("refused: {e}")],
            Ok(words) => words
                .iter()
                .map(|((op, block), x)| {
                    let block = block.map_or("None".to_string(), |b| b.to_string());
                    let fields: Vec<String> =
                        x.op.args
                            .iter()
                            .zip(&x.args)
                            .map(|((f, _), v)| format!("{f}={v}"))
                            .collect();
                    format!(
                        "{op} {block} | {} {} ; {}",
                        x.op.name,
                        fields.join(","),
                        x.note
                    )
                })
                .collect(),
        }
    };
    let mut out = Vec::new();
    for fidelity in ["LoFi", "HiFi2", "HiFi3", "HiFi4", "HiFi5"] {
        for (ct, rt, dst) in [(1, 1, 0), (1, 1, 2), (2, 1, 0), (1, 2, 1), (3, 2, 0)] {
            out.push(format!("TILE {fidelity} {ct} {rt} {dst}"));
            out.extend(lines(tile_products(fidelity, ct, rt, dst)));
        }
        for dst in [0, 3] {
            out.push(format!("WORMHOLE {fidelity} {dst}"));
            out.extend(lines(wormhole_products(fidelity, dst)));
        }
    }
    let got = out.join("\n") + "\n";
    assert!(
        got == want,
        "the words differ, {}",
        first_difference(&want, &got)
    );
}

/// A word as `tile_rows.py` writes it: its operation, its fields, its note.
fn row_line(x: &Insn) -> String {
    let fields: Vec<String> =
        x.op.args
            .iter()
            .zip(&x.args)
            .map(|((f, _), v)| format!("{f}={v}"))
            .collect();
    format!("{} {} ; {}", x.op.name, fields.join(" "), x.note)
}

/// A word from its line (a field's value a number, else a name).
fn row_word(line: &str) -> Insn {
    let (head, note) = line.split_once(" ; ").unwrap_or((line, ""));
    let mut parts = head.split(' ');
    let op = parts.next().expect("an operation");
    let args: Vec<(&str, Arg)> = parts
        .map(|f| {
            let (name, value) = f.split_once('=').expect("a field");
            let value = value
                .parse::<i64>()
                .map_or_else(|_| Arg::Name(value.to_string()), Arg::Num);
            (name, value)
        })
        .collect();
    Insn::new(op, &args, note).unwrap_or_else(|e| panic!("{line}: {e}"))
}

#[test]
fn the_per_row_passes_are_todays_on_a_few_words() {
    let Some(folder) = run_today("tile_rows.py", "tensix-tile-rows") else {
        return;
    };
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&folder)
        .expect("the folder")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "in"))
        .collect();
    cases.sort();
    assert_eq!(cases.len(), 8, "the cases: {cases:?}");
    for case in cases {
        let text = std::fs::read_to_string(&case).expect("the case");
        let mut lines = text.lines();
        let rule = lines.next().expect("the rule");
        let mut code: Vec<Insn> = lines.map(row_word).collect();
        let mut pre = None;
        let done = match rule {
            "interleave" => {
                let end = code.len();
                vec![format!(
                    "{} moved",
                    interleave(&mut code, 0, end).expect("interleave")
                )]
            }
            "drop_copies" => drop_copies(&mut code).expect("drop_copies"),
            "common_values" => common_values(&mut code).expect("common_values"),
            _ => {
                let (made, done) = per_row_prepass(&mut code, &[64, 128]).expect("prepass");
                pre = made;
                done
            }
        };
        let mut got = vec![if done.is_empty() {
            "nothing".to_string()
        } else {
            done.join("; ")
        }];
        got.extend(code.iter().map(row_line));
        if let Some(pre) = pre {
            got.push("PRE".into());
            got.extend(pre.iter().map(row_line));
        }
        let got = got.join("\n") + "\n";
        let want = std::fs::read_to_string(case.with_extension("out")).expect("today's");
        assert!(
            got == want,
            "{}: {}",
            case.display(),
            first_difference(&want, &got)
        );
    }
}

/// Two constants read in a run under lane flags and outside it: today loads them in
/// front of the run in a set's order, which no case pins, so the port refuses it by
/// name; one such constant is today's (`outside` in `tile_rows.py`).
#[test]
fn two_constants_before_a_run_are_refused_by_name() {
    let mut code: Vec<Insn> = [
        "SFPLOADI lreg_ind=v1 instr_mod0=0 imm16=16128 ; const",
        "SFPLOADI lreg_ind=v4 instr_mod0=0 imm16=16256 ; const",
        "SFPLOAD lreg_ind=v2 instr_mod0=0 sfpu_addr_mode=0 dest_reg_addr=0 ; Dst + 0",
        "SFPGT imm12_math=0 lreg_c=zero lreg_dest=v2 instr_mod1=1 ; a gt b",
        "SFPMAD lreg_src_a=v1 lreg_src_b=v4 lreg_src_c=zero lreg_dest=v2 instr_mod1=0 ; a b",
        "SFPENCC imm12_math=3 lreg_c=0 lreg_dest=0 instr_mod1=10 ; every lane on",
        "SFPMAD lreg_src_a=v2 lreg_src_b=v1 lreg_src_c=v4 lreg_dest=v3 instr_mod1=0 ; a b + c",
        "SFPSTORE lreg_ind=v3 instr_mod0=0 sfpu_addr_mode=0 dest_reg_addr=0 ; store",
    ]
    .iter()
    .map(|l| row_word(l))
    .collect();
    let before: Vec<String> = code.iter().map(row_line).collect();
    let end = code.len();
    let refused = interleave(&mut code, 0, end).expect_err("refused");
    assert_eq!(
        refused.text(),
        "not ported: the order of the constants before a run is not pinned yet"
    );
    assert_eq!(code.iter().map(row_line).collect::<Vec<_>>(), before);
}

/// Today's trace of the row form's corpus (`tile_rows_form.py`), once for all the tests.
fn today_rows() -> Option<PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| run_today("tile_rows_form.py", "tensix-tile-rows-form"))
        .clone()
}

/// The row form's corpus, `tile_rows_form.kom`.
fn rows_corpus() -> komorebi_ir::program::Module {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/tensix/tile/tile_rows_form.kom");
    let text = std::fs::read_to_string(path).expect("tile_rows_form.kom");
    komorebi_ir::api::build(&text).unwrap_or_else(|e| panic!("{e:?}"))
}

#[test]
fn each_function_on_a_row_traces_to_todays_kernel_ir() {
    let Some(folder) = today_rows() else { return };
    let module = rows_corpus();
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("the cases");
    let mut names: Vec<&str> = Vec::new();
    for line in cases.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (case, width, fields) = (f[0], f[1].parse().ok(), f[2].parse::<usize>());
        let fields = fields.expect("the fields");
        names.push(case);
        let at = func(&module, &format!("t_{case}"));
        let params = module.funcs[at.0 as usize].params.len();
        let resident: Vec<bool> = (0..params).map(|k| k >= params - fields).collect();
        let want = std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
        let ir = row_kernel_ir(&module, at, case, width, &resident)
            .unwrap_or_else(|r| panic!("{case}: {}", r.what));
        let got = ir.dump() + "\n";
        assert!(
            got == want,
            "{case} differs, {}",
            first_difference(&want, &got)
        );
    }
    names.sort();
    let mut funcs: Vec<&str> = module
        .funcs
        .iter()
        .filter_map(|f| f.name.strip_prefix("t_"))
        .collect();
    funcs.sort();
    assert_eq!(funcs, names, "a func `t_<case>` for each case and no other");
}

#[test]
fn a_reduction_on_one_tile_is_refused_with_todays_text_and_line() {
    let Some(folder) = today_rows() else { return };
    let today = std::fs::read_to_string(folder.join("one_tile.refused")).expect("the text");
    let today = today.trim_end();
    // today's line is the reduction's, in the script: the reduction's instruction is
    // placed there and every other one elsewhere
    let line: u32 = today
        .rsplit(':')
        .next()
        .and_then(|l| l.strip_suffix(')'))
        .and_then(|l| l.parse().ok())
        .expect("the line");
    let mut module = rows_corpus();
    let f = func(&module, "one");
    let body = module.funcs[f.0 as usize].body.as_ref().expect("a body");
    let places = (0..body.insts.len())
        .map(|i| {
            let op = &body.insts[i].op;
            let at = if matches!(op, Op::Reduce(..)) {
                line
            } else {
                1000 + i as u32
            };
            let inst = Anchor::Inst(BodyRef::Func(f), InstId(i as u32));
            (inst, 0, at, at)
        })
        .collect();
    module.notes.origin = Some(Origin {
        files: vec!["tests/ir/tensix/tile/tile_rows_form.py".into()],
        lines: Vec::new(),
        places,
    });
    let ir = row_kernel_ir(&module, f, "one", Some(1), &[false]).expect("the trace");
    let refused = unit_program(&ir, "fp32", true, false, FIDELITY).expect_err("refused");
    let refusal = refused.refusal(Some(Anchor::Func(f)));
    assert_eq!((refusal.code, refusal.what.as_str()), ("K962", today));
}

#[test]
fn what_the_row_forms_trace_would_not_make_is_refused_by_name() {
    let text = "module refused\n\ntile [32, 32]\nformat f32\n\n\
        func streamed(x: tile<f32>[2], g: tile<f32>[2]) -> tile<f32>[2] {\n\
        \x20   return x * rsqrt(reduce_sum(x * x, axis = row)).col() * g.row()\n}\n\n\
        func scalar(x: tile<f32>[2], k: f32) -> tile<f32>[2] {\n\
        \x20   return x * reduce_sum(x, axis = row).col() * k\n}\n\n\
        func square(x: tile<f32>[2, 2]) -> tile<f32>[2, 2] {\n\
        \x20   return x * reduce_sum(x, axis = row).col()\n}\n";
    let module = komorebi_ir::api::build(text).unwrap_or_else(|e| panic!("{e:?}"));
    let named = |f: &str, count: Option<u64>, resident: &[bool]| {
        let refusal = row_kernel_ir(&module, func(&module, f), f, count, resident).expect_err(f);
        assert_eq!(refusal.code, "K964", "{f}");
        refusal.what
    };
    let not = |what: &str| format!("the Tensix backend's tile compiler does not take {what} yet");
    let both = [false, false];
    assert_eq!(
        named("streamed", Some(2), &both),
        not("the row form's .row() of a streamed operand")
    );
    assert_eq!(
        named("streamed", None, &both),
        not("a block whose count is not bound at compile")
    );
    assert_eq!(
        named("streamed", Some(4), &both),
        not("a block of 2 tiles called on a row of 4")
    );
    assert_eq!(
        named("streamed", Some(2), &[true, false]),
        not("a resident operand before a streamed one (today's fields are the last parameters)")
    );
    assert_eq!(
        named("streamed", Some(2), &[true, true]),
        not("a row form of no streamed operand")
    );
    assert_eq!(
        named("streamed", Some(2), &[false]),
        not("a call whose operands' sources are not one a parameter")
    );
    assert_eq!(
        named("scalar", Some(2), &both),
        not("a parameter or result of the row form that is not a block of one row")
    );
    assert_eq!(
        named("square", Some(2), &[false]),
        not("a parameter or result of the row form that is not a block of one row")
    );
    // the 1-tile trace takes a row reduction only where every block is of one tile
    let one = kernel_ir(&module, func(&module, "streamed"), "streamed").expect_err("one");
    assert_eq!(
        one.what,
        not("a function on blocks with a product or a reduction (not element by element)")
    );
}

/// Top-level pieces of `text` split at `, ` outside brackets.
fn pieces(text: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0i32, 0);
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'[' | b'(' => depth += 1,
            b']' | b')' => depth -= 1,
            b',' if depth == 0 && bytes.get(i + 1) == Some(&b' ') => {
                out.push(&text[start..i]);
                start = i + 2;
            }
            _ => {}
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// An integer of a view or a grid as today writes it: a number, `unit`, or
/// `tile_rows(x)` / `tiles(x)`.
fn int_of(text: &str) -> Int {
    if let Ok(n) = text.parse() {
        return Int::Const(n);
    }
    if text == "unit" {
        return Int::Unit(None);
    }
    let (what, name) = text
        .strip_suffix(')')
        .and_then(|t| t.split_once('('))
        .unwrap_or_else(|| panic!("an integer {text}"));
    Int::Size {
        what: what.into(),
        name: name.into(),
        dim: None,
    }
}

/// A view as today writes it: `x.tiles[i]` or `x.tiles[i, :]`.
fn view_of(text: &str) -> View {
    let (tensor, rest) = text.split_once(".tiles[").expect("a view");
    let index = rest.strip_suffix(']').expect("a view's end");
    let (index, kind) = match index.strip_suffix(", :") {
        Some(i) => (i, "row"),
        None => (index, "tile"),
    };
    View {
        tensor: tensor.into(),
        kind: kind.into(),
        index: int_of(index),
        cols: None,
    }
}

/// A KernelIR read back from today's dump (the forms the row form's corpus prints):
/// its nodes made in the dump's order, which `nodes` gives again.
fn parse_ir(text: &str) -> KernelIR {
    let mut lines = text.lines();
    let name = lines
        .next()
        .and_then(|l| l.strip_prefix("kernel "))
        .expect("kernel");
    let mut ir = KernelIR::new(name, Vec::new(), Vec::new());
    for line in lines {
        let line = line.trim_start();
        if let Some((side, rest)) = line
            .split_once(' ')
            .filter(|(s, _)| ["in", "out"].contains(s))
        {
            let (tensor, dtype) = rest.trim_start().split_once(": ").expect("a tensor");
            ir.tensors.push(TensorParam {
                name: tensor.into(),
                output: side == "out",
                dtype: Some(dtype.into()),
            });
        } else if let Some(rest) = line.strip_prefix("grid unit < ") {
            let (count, block) = rest.split_once(", block ").expect("a grid");
            ir.grid = Some(Grid {
                count: int_of(count),
                block: block.parse().expect("a block"),
                cols: None,
            });
        } else if let Some(rest) = line.strip_prefix("store ") {
            let (view, value) = rest.split_once(" = %").expect("a store");
            ir.stores.push(Store {
                view: view_of(view),
                value: NodeId(value.parse().expect("a node")),
            });
        } else {
            let (_, made) = line.split_once(" = ").expect("a node");
            let (kind, inner) = made.split_once('(').expect("a node's kind");
            let inner = inner.strip_suffix(')').expect("a node's end");
            let (mut args, mut attrs) = (Vec::new(), Vec::new());
            for piece in pieces(inner) {
                if let Some(n) = piece.strip_prefix('%') {
                    args.push(NodeId(n.parse().expect("an operand")));
                    continue;
                }
                let (k, v) = piece.split_once('=').expect("a field");
                let attr = match (k, v) {
                    (_, "None") => Attr::None,
                    ("view", v) => Attr::View(view_of(v)),
                    ("value", v) => Attr::Float(v.parse().expect("a float")),
                    ("dim", v) => Attr::Int(Int::Const(v.parse().expect("a dim"))),
                    (_, v) => Attr::Text(v.into()),
                };
                attrs.push((k, attr));
            }
            ir.node(kind, args, attrs);
        }
    }
    ir
}

#[test]
fn each_row_of_two_tiles_gets_todays_unit_from_its_kernel_ir() {
    let Some(folder) = today_rows() else { return };
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("the cases");
    let refused = std::fs::read_to_string(folder.join("rows_refused.txt")).expect("refused");
    let (mut same, mut not_ported) = (Vec::new(), Vec::new());
    for line in cases.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f[1] != "2" {
            continue;
        }
        let case = f[0];
        let text = std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
        let ir = parse_ir(&text);
        assert_eq!(ir.dump() + "\n", text, "{case}: the KernelIR read back");
        let made = row_program(&ir, 2, 2, 4, "fp32", false);
        if let Some(today) = refused
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{case}\t")))
        {
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
        let mut unit = match made {
            Ok(unit) => unit,
            Err(e) => {
                assert_eq!(e.refusal(None).code, "K964", "{case}: {}", e.text());
                not_ported.push(format!("{case}: {}", e.text()));
                continue;
            }
        };
        let listing = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
        let words = emit(&mut unit).unwrap_or_else(|e| panic!("{case}: {e}"));
        let emitted = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
        for (file, got) in [("txt", listing), ("hpp", words), ("emitted.txt", emitted)] {
            let want = std::fs::read_to_string(folder.join(format!("{case}.rows.{file}")))
                .unwrap_or_else(|e| panic!("{case}.rows.{file}: {e}"));
            assert!(
                got == want,
                "{case}'s {file} differs, {}",
                first_difference(&want, &got)
            );
        }
        same.push(case);
    }
    // one reduction, with and without a resident row (f32 and bf16), a centred row,
    // a row's largest and smallest values (of a product too, folded by max), a sum of
    // exponentials (the mad-reader's repairs taken back into the tile program), a
    // reciprocal's constant register; the levels: two and three passes, two resident
    // rows, a sum kept for a later level (read directly, or through a value a later
    // sum reads), a sum and a largest value reading .row() (of two resident rows too)
    assert_eq!(
        same,
        [
            "rms_norm",
            "centred",
            "row_min",
            "rms_plain",
            "peak",
            "weighted_bf16",
            "softmax_pair",
            "weighted_sum",
            "peak_square",
            "exp_sum",
            "recip_centred",
            "layer_pair",
            "kept_sum",
            "max_weighted",
            "three_plain",
            "kept_through",
            "two_weights"
        ]
    );
    // two row tensors, more live values than the work registers, more constants than
    // the programmable registers, 5 per-row values, 5 resident rows: today's refusals
    // (K962)
    assert_eq!(refused.lines().count(), 5);
    assert_eq!(not_ported, Vec::<String>::new());
}

#[test]
fn the_row_forms_branches_the_corpus_does_not_reach_are_todays() {
    let Some(folder) = today_rows() else { return };
    let lines = std::fs::read_to_string(folder.join("branches.txt")).expect("the branches");
    let mut not_ported = Vec::new();
    for line in lines.lines() {
        let f: Vec<&str> = line.splitn(4, '\t').collect();
        let (case, width, dest, today) = (f[0], f[1].parse::<i64>().expect("a width"), f[2], f[3]);
        let text = std::fs::read_to_string(folder.join(format!("branch_{case}.ir"))).expect("IR");
        let ir = parse_ir(&text);
        let made = row_program(&ir, width, width, 2 * width, dest, false);
        match (today, made) {
            ("made", Ok(unit)) => {
                let want = folder.join(format!("branch_{case}.rows.txt"));
                let want = std::fs::read_to_string(want).expect("today's unit");
                let got = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                assert!(got == want, "{case}: {}", first_difference(&want, &got));
            }
            (_, Err(e)) if e.refusal(None).code == "K964" => {
                not_ported.push(format!("{case}: {}", e.text()));
            }
            ("made", Err(e)) => panic!("{case}: {}", e.text()),
            (today, Err(e)) => {
                // today's text, but for the place it names (K962)
                assert_eq!(
                    e.text(),
                    today.split(" (at ").next().unwrap_or(today),
                    "{case}"
                );
                assert_eq!(e.refusal(None).code, "K962", "{case}");
            }
            (today, Ok(_)) => panic!("{case}: made, today refuses it: {today}"),
        }
    }
    assert_eq!(not_ported, Vec::<String>::new());
    // the band of the row form (today's `band_groups`) is the last slice; the levels
    // refuse it today
    let text = std::fs::read_to_string(folder.join("rms_norm.ir")).expect("the IR");
    let banded = row_program(&parse_ir(&text), 2, 2, 4, "fp32", true).expect_err("band");
    assert_eq!(banded.text(), "not ported: the row form's band");
    let text = std::fs::read_to_string(folder.join("softmax_pair.ir")).expect("the IR");
    let banded = row_program(&parse_ir(&text), 2, 2, 4, "fp32", true).expect_err("band");
    let today = std::fs::read_to_string(folder.join("band.txt")).expect("today's band");
    let today = today.trim_end();
    assert_eq!(banded.text(), today.split(" (at ").next().unwrap_or(today));
    assert_eq!(banded.refusal(None).code, "K962");
}

/// A call's Dst layout as today's scripts write it.
fn dst_text(layout: &DstLayout) -> String {
    let list = |tiles: &[usize]| {
        let t: Vec<String> = tiles.iter().map(usize::to_string).collect();
        t.join(",")
    };
    let mut lines: Vec<String> = layout
        .params
        .iter()
        .map(|p| format!("param\t{}", list(p)))
        .collect();
    lines.push(format!("result\t{}", list(&layout.result)));
    lines.join("\n") + "\n"
}

#[test]
fn a_units_dst_layout_is_todays_handlers() {
    // the stream form: each unit `unit()` makes of the corpus, against the layout
    // today's handler takes from today's unit
    let Some(folder) = today() else { return };
    let module = corpus();
    let mut made = Vec::new();
    for (i, f) in module.funcs.iter().enumerate() {
        let Some(case) = f.name.strip_prefix("t_") else {
            continue;
        };
        let Ok(unit) = tile_unit(&module, komorebi_ir::types::FuncId(i as u32)) else {
            continue;
        };
        let want = std::fs::read_to_string(folder.join(format!("{case}.dst")))
            .unwrap_or_else(|e| panic!("{case}.dst: {e}"));
        assert_eq!(dst_text(unit.dst_layout()), want, "{case}");
        made.push(case);
    }
    assert!(made.len() >= 20, "{made:?}");
    // the row form, at 2 tiles and in chunks, as today's handler lays out its call
    let Some(folder) = today_rows() else { return };
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("the cases");
    let mut rows = 0;
    for line in cases.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let case = f[0];
        let width: usize = f[1].parse().expect("a width");
        let fields: usize = f[2].parse().expect("the fields");
        let Ok(want) = std::fs::read_to_string(folder.join(format!("{case}.dst"))) else {
            continue;
        };
        let ir = std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
        let params = parse_ir(&ir).tensors.iter().filter(|t| !t.output).count();
        let resident: Vec<bool> = (0..params).map(|k| k + fields >= params).collect();
        assert_eq!(dst_text(&row_layout(&resident, width)), want, "{case}");
        rows += 1;
    }
    assert!(rows >= 20, "{rows}");
}

/// What the handler reads of a chunk unit that its listing does not print, as the
/// script writes it.
fn chunk_fields(unit: &UnitProgram) -> String {
    let bases: Vec<String> = unit.chunk_results.iter().map(i64::to_string).collect();
    format!(
        "results\t{}\nchunk_results\t{}\nfields_in_sums\t{}\n",
        unit.results,
        bases.join(","),
        i64::from(unit.chunk_fields_in_sums)
    )
}

#[test]
fn each_wide_row_gets_todays_chunk_unit_and_cost_from_its_kernel_ir() {
    let Some(folder) = today_rows() else { return };
    let read = |file: &str| std::fs::read_to_string(folder.join(file)).expect(file);
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("the cases");
    let refused = std::fs::read_to_string(folder.join("rows_refused.txt")).expect("refused");
    let costs = std::fs::read_to_string(folder.join("chunk_costs.txt")).expect("the costs");
    let (mut same, mut not_ported) = (Vec::new(), Vec::new());
    for line in cases.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (case, width) = (f[0], f[1].parse::<i64>().expect("a width"));
        let fields = f[2].parse::<i64>().expect("the fields");
        if width <= 2 {
            continue;
        }
        let text = std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
        let ir = parse_ir(&text);
        assert_eq!(ir.dump() + "\n", text, "{case}: the KernelIR read back");
        let made = chunk_program(&ir, width);
        if let Some(today) = refused
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{case}\t")))
        {
            let error = made.expect_err(case);
            assert_eq!(
                error.text(),
                today.split(" (at ").next().unwrap_or(today),
                "{case}"
            );
            assert_eq!(error.refusal(None).code, "K962", "{case}");
            continue;
        }
        let mut unit = match made {
            Ok(unit) => unit,
            Err(e) => {
                assert_eq!(e.refusal(None).code, "K964", "{case}: {}", e.text());
                not_ported.push(format!("{case}: {}", e.text()));
                continue;
            }
        };
        let (count, cycles, copies) =
            chunk_cost(&unit, width, fields).unwrap_or_else(|e| panic!("{case}: {e}"));
        let cost = format!(
            "{case}\t{count}\t{}\t{}",
            float_repr(cycles),
            float_repr(copies)
        );
        assert!(costs.lines().any(|l| l == cost), "{case}: {cost}");
        assert_eq!(
            chunk_fields(&unit),
            read(&format!("{case}.chunk.fields")),
            "{case}"
        );
        let listing = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
        let words = emit(&mut unit).unwrap_or_else(|e| panic!("{case}: {e}"));
        let emitted = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
        for (file, got) in [("txt", listing), ("hpp", words), ("emitted.txt", emitted)] {
            let want = std::fs::read_to_string(folder.join(format!("{case}.chunk.{file}")))
                .unwrap_or_else(|e| panic!("{case}.chunk.{file}: {e}"));
            assert!(
                got == want,
                "{case}'s {file} differs, {}",
                first_difference(&want, &got)
            );
        }
        same.push(case);
    }
    // rms_norm at 3, 4 and 8 tiles (its weight in bf16 at 3), softmax_rows at 3 and 5
    // (an odd row's last chunk), a row's largest value, layer_norm's two levels and two
    // resident rows, a sum reading .row() (and with another sum that does not)
    assert_eq!(
        same,
        [
            "rms_norm_wide",
            "rms_bf16",
            "softmax_rows",
            "row_max",
            "layer_norm",
            "rms_eight",
            "softmax_five",
            "wide_weighted",
            "wide_mixed"
        ]
    );
    // a largest and a smallest value together: today starts them in a set's order
    assert_eq!(
        not_ported,
        [
            "extremes: not ported: the chunk form's start values of a largest and a smallest \
          value together (today makes them in a set's order)"
        ]
    );
    let lines = std::fs::read_to_string(folder.join("chunk_branches.txt")).expect("branches");
    let mut verdicts = 0;
    for line in lines.lines() {
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        let (case, width, today) = (f[0], f[1].parse::<i64>().expect("a width"), f[2]);
        let text = std::fs::read_to_string(folder.join(format!("chunk_{case}.ir"))).expect("IR");
        match (today, chunk_program(&parse_ir(&text), width)) {
            ("made", Ok(unit)) => {
                let want = folder.join(format!("chunk_{case}.chunk.txt"));
                let want = std::fs::read_to_string(want).expect("today's unit");
                let got = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                assert!(got == want, "{case}: {}", first_difference(&want, &got));
                let want = read(&format!("chunk_{case}.chunk.fields"));
                assert_eq!(chunk_fields(&unit), want, "{case}");
            }
            ("made", Err(e)) => panic!("{case}: {}", e.text()),
            (today, Err(e)) => {
                assert_eq!(
                    e.text(),
                    today.split(" (at ").next().unwrap_or(today),
                    "{case}"
                );
                assert_eq!(e.refusal(None).code, "K962", "{case}");
            }
            (today, Ok(_)) => panic!("{case}: made, today refuses it: {today}"),
        }
        verdicts += 1;
    }
    // a row of 2, two results, three (more than Dst holds), five sums, five resident
    // rows, a per-row value alone (part `value`)
    assert_eq!(verdicts, 6);
}

#[test]
fn the_row_forms_checks_are_todays_on_a_few_words() {
    let Some(folder) = today_rows() else { return };
    let lines = std::fs::read_to_string(folder.join("checks.txt")).expect("the checks");
    let layouts = Layouts {
        copy: Layout::Tile(0),
        rows: vec![Layout::Rows(2, 0)],
        sums: Vec::new(),
        cols: vec![Layout::Cols(4)],
    };
    let mut cases = 0;
    for line in lines.lines() {
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        let (case, check, today) = (f[0], f[1], f[2]);
        let words =
            std::fs::read_to_string(folder.join(format!("check_{case}.in"))).expect("the words");
        let code: Vec<Insn> = words.lines().map(row_word).collect();
        let mut unit = UnitProgram::new("c", "fp32");
        let made = if check == "lanes" {
            check_lanes(&mut unit, &code, layouts.rows[0], None)
        } else {
            check_broadcast(&mut unit, &code, &layouts, 2, &[0])
        };
        let got = match made {
            Ok(()) => unit
                .checks
                .last()
                .map(|c| c.verdict.clone())
                .unwrap_or_default(),
            Err(e) => e.0,
        };
        assert_eq!(got, today, "{case}");
        cases += 1;
    }
    assert_eq!(cases, 12);
}
