//! The DSL as a frontend, end to end with no card: a class form made by today's
//! emitter, and by the frontend (`komorebi.dsl.ir`) then the IR and the Tensix backend;
//! each file byte for byte, the kernels and buffers handed to the runtime, the folder.
//! The first lesson (`tensix/t1_class.py`): its module equal to the hand-written one
//! (`tensix/t1_class.kom`, `tensix/t1_places.txt`). The corpus's program 1
//! (`tensix/scalars_line.py`, `Scalars.line`).
//!
//! The Python side (`frontend/made.py`) runs under `python3` (or `KOMO_IR_PYTHON`) with
//! the repository's `src` on its path. Without it the test fails and says so;
//! `KOMO_IR_NO_PYTHON=1` skips it by name.

use std::path::{Path, PathBuf};
use std::process::Command;

use komorebi_ir::api::origin::{Given, Mapped};
use komorebi_ir::api::{self, origin};
use komorebi_ir::backend::tensix::load::{self, thread_header, LoadTensor};
use komorebi_ir::backend::tensix::{self, chip, Tensix};
use komorebi_ir::backend::{Backend, Bound, Build, Expr, Processor, Setting, Target};
use komorebi_ir::bytecode;
use komorebi_ir::bytecode::hash::{hex, sha256};
use komorebi_ir::program::body::InstId;
use komorebi_ir::program::notes::{Anchor, Owner};
use komorebi_ir::program::BodyRef;
use komorebi_ir::program::Module;
use komorebi_ir::types::{HandlerId, ProgramId};

fn here() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir")
}

/// What `made.py` printed: today's build folder, its kernels and buffers (the
/// `KERNEL`, `BUFFER` and `PAIR` lines), and the frontend's text and origin.
struct Made {
    folder: PathBuf,
    printed: Vec<String>,
    text: String,
    given: Given,
}

fn numbers(rest: &str) -> Vec<u32> {
    rest.split(' ')
        .map(|n| n.parse().expect("a number"))
        .collect()
}

/// `made.py` run on a copy of `tensix/<script>.py` in its own folder, for `class`.
fn made(name: &str, script: &str, class: &str) -> Made {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("frontend-{name}"));
    std::fs::create_dir_all(&dir).expect("a folder for the script");
    let file = format!("{script}.py");
    std::fs::copy(here().join("tensix").join(&file), dir.join(&file)).expect("the script copied");
    let python = std::env::var("KOMO_IR_PYTHON").unwrap_or_else(|_| "python3".into());
    let out = Command::new(&python)
        .arg(here().join("frontend/made.py"))
        .args([script, class])
        .current_dir(&dir)
        .env(
            "PYTHONPATH",
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        )
        .output()
        .unwrap_or_else(|e| panic!("{python}: {e} (KOMO_IR_NO_PYTHON=1 skips this test)"));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "made.py failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (head, text) = stdout.split_once("TEXT\n").expect("a TEXT line");
    let mut folder = None;
    let mut printed = Vec::new();
    let mut given = Given::default();
    for line in head.lines() {
        let (kind, rest) = line.split_once(' ').expect("a line of made.py");
        match kind {
            "FOLDER" => folder = Some(PathBuf::from(rest)),
            "KERNEL" | "BUFFER" | "PAIR" => printed.push(line.to_string()),
            "FILE" => given.files.push(rest.to_string()),
            "LINE" => {
                let mut parts = rest.splitn(3, ' ');
                let mut number = || parts.next().expect("a field").parse().expect("a number");
                let (file, n) = (number(), number());
                let text = rest.splitn(3, ' ').nth(2).unwrap_or("").to_string();
                given.lines.push((file, n, text));
            }
            "MAP" => {
                let n = numbers(rest);
                given.map.push(Mapped {
                    text_first: n[0],
                    text_last: n[1],
                    file: n[2],
                    first: n[3],
                    last: n[4],
                });
            }
            "RENAME" => {
                let (safe, real) = rest.split_once(' ').expect("two names");
                given.renames.push((safe.to_string(), real.to_string()));
            }
            _ => panic!("a line of made.py: {line}"),
        }
    }
    Made {
        folder: folder.expect("a FOLDER line"),
        printed,
        text: text.to_string(),
        given,
    }
}

/// The hand-written place table: the handler, its parameters and its instructions.
fn table() -> Vec<(Anchor, u32, u32, u32)> {
    let text = std::fs::read_to_string(here().join("tensix/t1_places.txt")).expect("the table");
    let rows = text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    rows.map(|line| {
        let (kind, rest) = line.split_once(' ').expect("a place line");
        let n = numbers(rest);
        let (p, h) = (ProgramId(n[0]), HandlerId(n[1]));
        let anchor = match kind {
            "handler" => Anchor::Handler(p, h),
            "param" => Anchor::ParamOf(Owner::Handler(p, h), n[2]),
            _ => Anchor::Inst(BodyRef::Handler(p, h), InstId(n[2])),
        };
        (anchor, 0, n[n.len() - 2], n[n.len() - 1])
    })
    .collect()
}

/// The first line where `a` and `b` differ, for the message.
fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}:\n  today:    {x}\n  frontend: {y}", i + 1);
        }
    }
    format!(
        "{} lines today, {} by the frontend",
        a.lines().count(),
        b.lines().count()
    )
}

/// The module's output against today's: each file (the compile's, and the thread
/// header at load from `stand_ins`) against today's folder less its report, the
/// kernels and buffers against `made.py`'s lines, and the folder's name
/// (`<prefix>-<text_key>`).
fn same_as_today(made: &Made, module: &Module, stand_ins: &[LoadTensor], prefix: &str) {
    let mut files = tensix::sources(module).unwrap_or_else(|r| panic!("{r:?}"));
    let header = thread_header(module, stand_ins).unwrap_or_else(|r| panic!("{r:?}"));
    files.push((format!("{prefix}_thread.h"), header));
    files.sort();
    let mut today: Vec<String> = std::fs::read_dir(&made.folder)
        .expect("the build folder")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n != "program.txt")
        .collect();
    today.sort();
    let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, today, "the files");
    for (name, text) in &files {
        let want = std::fs::read_to_string(made.folder.join(name)).expect("today's file");
        assert!(
            *text == want,
            "{name} differs, {}",
            first_difference(&want, text)
        );
    }
    // The kernels the load hands the runtime, field by field as describe hands them on,
    // then the buffers (their formats by the loader's names) and their pairs.
    let target = Target {
        arch: chip::ARCH.to_string(),
        features: Default::default(),
    };
    let output = Tensix
        .compile(module, &target)
        .unwrap_or_else(|r| panic!("{r:?}"));
    let Build::Kernels {
        kernels: built,
        buffers,
    } = &output.build
    else {
        panic!("kernels built at load");
    };
    let loaded = load::kernels(module, built, stand_ins).unwrap_or_else(|r| panic!("{r:?}"));
    let joined = |mut pairs: Vec<(String, String)>| {
        pairs.sort();
        let pairs: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}:{v}")).collect();
        pairs.join(",")
    };
    let mut ours: Vec<String> = loaded
        .iter()
        .map(|k| {
            let (role, number) = match k.processor {
                Processor::Reader => ("reader", 0),
                Processor::Writer => ("writer", 1),
                Processor::Compute => ("compute", 2),
            };
            let args: Vec<String> = k.args.iter().map(u64::to_string).collect();
            let mut options: Vec<(String, String)> = k
                .options
                .iter()
                .map(|(name, v)| {
                    let v = match v {
                        Setting::Bool(b) => b.to_string(),
                        Setting::Text(t) => t.clone(),
                    };
                    (name.clone(), v)
                })
                .collect();
            // the buffers unpacked to Dst, as today's compute option (a Python dict)
            let unpacked: Vec<String> = buffers
                .iter()
                .filter_map(|b| b.unpack.map(|u| format!("{}: '{}'", b.index, u.name())))
                .collect();
            if k.processor == Processor::Compute && !unpacked.is_empty() {
                let dict = format!("{{{}}}", unpacked.join(", "));
                options.push(("unpack_to_dest_mode".to_string(), dict));
            }
            format!(
                "KERNEL {role} {} {number} args={} defines={} options={}",
                k.source,
                args.join(","),
                joined(k.defines.clone()),
                joined(options)
            )
        })
        .collect();
    ours.extend(buffers.iter().map(|b| {
        let dtype = chip::dtype(b.format).unwrap_or("none");
        format!("BUFFER {} {dtype} {} {}", b.index, b.page_bytes, b.pages)
    }));
    ours.extend(
        buffers
            .iter()
            .filter_map(|b| b.pair.filter(|o| b.index < *o).map(|o| (b.index, o)))
            .map(|(a, b)| format!("PAIR {a} {b}")),
    );
    assert_eq!(
        ours, made.printed,
        "the kernels and buffers handed to the runtime"
    );
    // The same files give the same folder (dsl/files.py, text_key).
    let mut digest = Vec::new();
    for (name, text) in &files {
        digest.extend_from_slice(format!("{name}\0{}\0", text.len()).as_bytes());
        digest.extend_from_slice(text.as_bytes());
    }
    let key = &hex(&sha256(&digest))[..16];
    assert_eq!(
        made.folder
            .file_name()
            .map(|n| n.to_string_lossy().into_owned()),
        Some(format!("{prefix}-{key}"))
    );
}

#[test]
fn the_class_through_the_frontend_is_todays_text_byte_for_byte() {
    if std::env::var("KOMO_IR_NO_PYTHON").is_ok_and(|v| v == "1") {
        eprintln!("the frontend's identity skipped: KOMO_IR_NO_PYTHON=1");
        return;
    }
    let made = made("t1", "t1_class", "Scale");

    // The module: the hand-written one's program and hints, its places the table's.
    let module = origin::build(&made.text, &made.given).unwrap_or_else(|r| panic!("{r:?}"));
    let hand = std::fs::read_to_string(here().join("tensix/t1_class.kom")).expect("the .kom");
    let (ours, theirs) = (
        bytecode::hashes(&module),
        bytecode::hashes(&api::build(&hand).expect("the hand module")),
    );
    assert_eq!(ours.semantic, theirs.semantic, "the program");
    assert_eq!(ours.build, theirs.build, "the program and its hints");
    let places = &module.notes.origin.as_ref().expect("an origin").places;
    let handler: Vec<_> = places
        .iter()
        .filter(|(a, ..)| !matches!(a, Anchor::Func(..) | Anchor::Inst(BodyRef::Func(..), _)))
        .filter(|(a, ..)| !matches!(a, Anchor::ParamOf(Owner::Func(..), _)))
        .cloned()
        .collect();
    assert_eq!(handler, table(), "the handler's places");

    // The files, kernels and buffers, the thread header from the stand-in tensors.
    let stand_in = LoadTensor {
        pages: 2048,
        layout: vec![2, 2048],
        dram_page: None,
    };
    same_as_today(&made, &module, &[stand_in.clone(), stand_in], "scale");
}

#[test]
fn program_1_through_the_frontend_is_todays_text_byte_for_byte() {
    if std::env::var("KOMO_IR_NO_PYTHON").is_ok_and(|v| v == "1") {
        eprintln!("the frontend's identity skipped: KOMO_IR_NO_PYTHON=1");
        return;
    }
    // Scalars.line: a word of the rank, streams of t.tiles(first, n) in blocks of 8, a
    // tile expression as a function on blocks, a sink of f32 fed by bf16.
    let made = made("scalars", "scalars_line", "Scalars");
    let module = origin::build(&made.text, &made.given).unwrap_or_else(|r| panic!("{r:?}"));
    // The stand-ins: 768 pages each, a of bf16 pages of 2048 B, out of float32 of 4096.
    let tensor = |page: u64| LoadTensor {
        pages: 768,
        layout: vec![2, page],
        dram_page: None,
    };
    same_as_today(&made, &module, &[tensor(2048), tensor(4096)], "scalars");
    // Its ranges read the rank: a host checks a.tiles(worker.rank * 64, 64) and out's
    // for each rank of the group (today checks nothing here).
    let target = Target {
        arch: chip::ARCH.to_string(),
        features: Default::default(),
    };
    let output = Tensix
        .compile(&module, &target)
        .unwrap_or_else(|r| panic!("{r:?}"));
    let bound = |tensor| Bound {
        tensor,
        first: Expr::Mul(Box::new(Expr::Rank), Box::new(Expr::Const(64))),
        count: Expr::Const(64),
    };
    assert_eq!(output.entries[0].bounds, [bound(0), bound(1)]);
}

#[test]
fn program_2_through_the_frontend_is_todays_text_byte_for_byte() {
    if std::env::var("KOMO_IR_NO_PYTHON").is_ok_and(|v| v == "1") {
        eprintln!("the frontend's identity skipped: KOMO_IR_NO_PYTHON=1");
        return;
    }
    // StepHandlers.sum: a partition's tiles summed into a tile of zeros, the sum
    // reduced over the group (by tree on Tensix), the leader's written to one tile.
    let made = made("steps", "steps_sum", "StepHandlers");
    let module = origin::build(&made.text, &made.given).unwrap_or_else(|r| panic!("{r:?}"));
    // The stand-ins: s of 768 float32 pages of 4096 B, t of one.
    let tensor = |pages: u64| LoadTensor {
        pages,
        layout: vec![2, 4096],
        dram_page: None,
    };
    same_as_today(&made, &module, &[tensor(768), tensor(1)], "stephandlers");
    // Every range is a bound: s's partitioned tiles 0 .. tiles, and t's one tile.
    let target = Target {
        arch: chip::ARCH.to_string(),
        features: Default::default(),
    };
    let output = Tensix
        .compile(&module, &target)
        .unwrap_or_else(|r| panic!("{r:?}"));
    let bound = |tensor, count| Bound {
        tensor,
        first: Expr::Const(0),
        count,
    };
    assert_eq!(
        output.entries[0].bounds,
        [bound(0, Expr::Word(2)), bound(1, Expr::Const(1))]
    );
}

#[test]
fn program_3_through_the_frontend_is_todays_text_byte_for_byte() {
    if std::env::var("KOMO_IR_NO_PYTHON").is_ok_and(|v| v == "1") {
        eprintln!("the frontend's identity skipped: KOMO_IR_NO_PYTHON=1");
        return;
    }
    // StepHandlers.add: a word of the rank times a parameter, two streams zipped in
    // one move, a function of two tiles (`plus`, from the tile compiler) into a sink.
    let made = made("steps_add", "steps_add", "StepHandlers");
    let module = origin::build(&made.text, &made.given).unwrap_or_else(|r| panic!("{r:?}"));
    // The stand-ins: a, b and out of 768 float32 pages of 4096 B each.
    let tensor = LoadTensor {
        pages: 768,
        layout: vec![2, 4096],
        dram_page: None,
    };
    let stand_ins = [tensor.clone(), tensor.clone(), tensor];
    same_as_today(&made, &module, &stand_ins, "stephandlers");
    // Every range is a bound, each rank's: a's, b's and out's tiles first .. first +
    // width, first = worker.rank * width.
    let target = Target {
        arch: chip::ARCH.to_string(),
        features: Default::default(),
    };
    let output = Tensix
        .compile(&module, &target)
        .unwrap_or_else(|r| panic!("{r:?}"));
    let bound = |tensor| Bound {
        tensor,
        first: Expr::Mul(Box::new(Expr::Rank), Box::new(Expr::Word(3))),
        count: Expr::Word(3),
    };
    assert_eq!(output.entries[0].bounds, [bound(0), bound(1), bound(2)]);
}
