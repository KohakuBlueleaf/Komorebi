//! The Tensix backend's files against today's class DSL: the first lesson, made by the
//! Python emitter from `tensix/t1_class.py` (no card, no compiler), and by the Rust
//! emitter from `tensix/t1_class.kom` with the origin the frontend records; each file
//! byte for byte.
//!
//! The Python emitter runs under `python3` (or `KOMO_IR_PYTHON`) with the repository's
//! `src` on its path. Without it the test fails and says so; `KOMO_IR_NO_PYTHON=1`
//! skips it by name.

use std::path::{Path, PathBuf};
use std::process::Command;

use komorebi_ir::backend::tensix::load::{self, thread_header, LoadTensor};
use komorebi_ir::backend::tensix::{self, Tensix};
use komorebi_ir::backend::{
    ArgWord, Backend, Bound, Build, CacheKey, CompileArg, Expr, Processor, Reads, Target,
};
use komorebi_ir::bytecode::hash::{hex, sha256};
use komorebi_ir::bytecode::{decode, encode};
use komorebi_ir::program::body::InstId;
use komorebi_ir::program::notes::{Anchor, Origin};
use komorebi_ir::program::{BodyRef, Module};
use komorebi_ir::types::{HandlerId, ProgramId};

fn here() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/tensix")
}

/// The place table: `handler P H first last` and `inst P H I first last`.
fn places(text: &str) -> Vec<(Anchor, u32, u32, u32)> {
    let mut out = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let n: Vec<u32> = fields[1..]
            .iter()
            .map(|f| f.parse().expect("a number"))
            .collect();
        let anchor = match (fields[0], n.as_slice()) {
            ("handler", [p, h, ..]) => Anchor::Handler(ProgramId(*p), HandlerId(*h)),
            ("inst", [p, h, i, ..]) => {
                Anchor::Inst(BodyRef::Handler(ProgramId(*p), HandlerId(*h)), InstId(*i))
            }
            _ => panic!("a place line: {line}"),
        };
        let [.., first, last] = n.as_slice() else {
            panic!("a place line: {line}");
        };
        out.push((anchor, 0, *first, *last));
    }
    out.sort_by_key(|(a, ..)| *a);
    out
}

/// The module with the origin the frontend gives it: the script at `script`, all its
/// lines held, each anchor at its place in the table.
fn module_from(script: &Path) -> Module {
    let source = std::fs::read_to_string(here().join("t1_class.kom")).expect("the .kom");
    let mut module = komorebi_ir::api::build(&source).unwrap_or_else(|d| panic!("{d:?}"));
    let text = std::fs::read_to_string(script).expect("the script");
    let table = std::fs::read_to_string(here().join("t1_places.txt")).expect("the places");
    module.notes.origin = Some(Origin {
        files: vec![script.display().to_string()],
        lines: text
            .lines()
            .enumerate()
            .map(|(i, line)| (0, i as u32 + 1, line.to_string()))
            .collect(),
        places: places(&table),
    });
    // The origin is one a byte-code file can hold: it validates, and reads back.
    decode(&encode(&module)).unwrap_or_else(|r| panic!("{r:?}"))
}

/// Today's emitter run on a copy of the script in its own folder: the build folder
/// it wrote.
fn today(dir: &Path) -> (PathBuf, String) {
    let python = std::env::var("KOMO_IR_PYTHON").unwrap_or_else(|_| "python3".into());
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let out = Command::new(&python)
        .arg("t1_class.py")
        .current_dir(dir)
        .env("PYTHONPATH", &src)
        .output()
        .unwrap_or_else(|e| panic!("{python}: {e} (KOMO_IR_NO_PYTHON=1 skips this test)"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "today's emitter failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let folder = stdout
        .lines()
        .find_map(|l| l.strip_prefix("FOLDER "))
        .unwrap_or_else(|| panic!("no FOLDER line:\n{stdout}"));
    (PathBuf::from(folder), stdout.into_owned())
}

/// The first line where `a` and `b` differ, for the message.
fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}:\n  today: {x}\n  rust:  {y}", i + 1);
        }
    }
    format!(
        "{} lines today, {} from the Rust emitter",
        a.lines().count(),
        b.lines().count()
    )
}

#[test]
fn the_first_lesson_is_todays_text_byte_for_byte() {
    if std::env::var("KOMO_IR_NO_PYTHON").is_ok_and(|v| v == "1") {
        eprintln!("the identity skipped: KOMO_IR_NO_PYTHON=1");
        return;
    }
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("tensix-t1");
    std::fs::create_dir_all(&dir).expect("a folder for the script");
    let script = dir.join("t1_class.py");
    std::fs::copy(here().join("t1_class.py"), &script).expect("the script copied");
    let (folder, printed) = today(&dir);
    let module = module_from(&script.canonicalize().expect("the script's path"));
    let ours = tensix::sources(&module).unwrap_or_else(|r| panic!("{r:?}"));

    // The compile's files and, at load, the thread header from the stand-in tensors
    // the script makes with (2048 pages each, one layout): today's folder less its
    // report (program.txt, which today's text_key leaves out too).
    let stand_in = LoadTensor {
        pages: 2048,
        layout: vec![2, 2048],
        dram_page: None,
    };
    let stand_ins = [stand_in.clone(), stand_in];
    let header = thread_header(&module, &stand_ins).unwrap_or_else(|r| panic!("{r:?}"));
    let mut ours = ours;
    ours.push(("scale_thread.h".to_string(), header));
    ours.sort();
    let mut theirs: Vec<String> = std::fs::read_dir(&folder)
        .expect("the build folder")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n != "program.txt")
        .collect();
    theirs.sort();
    let names: Vec<&str> = ours.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, theirs, "the files");
    for (name, text) in &ours {
        let want = std::fs::read_to_string(folder.join(name)).expect("today's file");
        assert!(
            *text == want,
            "{name} differs, {}",
            first_difference(&want, text)
        );
    }
    // The kernels the load hands the runtime: today's (the Program's sources,
    // compile arguments, defines and compute options, as describe hands them on) and
    // the output's Build::Kernels with each tensor's layout filled from the same
    // stand-ins, field by field.
    let output = Tensix
        .compile(&module, &target("tensix"))
        .unwrap_or_else(|r| panic!("{r:?}"));
    let Build::Kernels(built) = &output.build else {
        panic!("kernels built at load");
    };
    let loaded = load::kernels(&module, built, &stand_ins).unwrap_or_else(|r| panic!("{r:?}"));
    let joined = |pairs: &[(String, String)]| {
        let mut pairs = pairs.to_vec();
        pairs.sort();
        pairs
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let described: Vec<String> = loaded
        .iter()
        .map(|k| {
            // describe's roles and numbers (session/backend/backend.py, _launch)
            let (role, number) = match k.processor {
                Processor::Reader => ("reader", 0),
                Processor::Writer => ("writer", 1),
                Processor::Compute => ("compute", 2),
            };
            let args: Vec<String> = k.args.iter().map(u64::to_string).collect();
            format!(
                "KERNEL {role} {} {number} args={} defines={} options={}",
                k.source,
                args.join(","),
                joined(&k.defines),
                joined(&k.options)
            )
        })
        .collect();
    let theirs: Vec<&str> = printed
        .lines()
        .filter(|l| l.starts_with("KERNEL "))
        .collect();
    assert_eq!(described, theirs, "the kernels handed to the runtime");

    // The folder's name is today's digest of its files (dsl/files.py, text_key): the
    // same files give the same folder.
    let mut digest = Vec::new();
    for (name, text) in &ours {
        digest.extend_from_slice(format!("{name}\0{}\0", text.len()).as_bytes());
        digest.extend_from_slice(text.as_bytes());
    }
    let key = &hex(&sha256(&digest))[..16];
    assert_eq!(
        folder.file_name().map(|n| n.to_string_lossy().into_owned()),
        Some(format!("scale-{key}"))
    );
}

#[test]
fn every_other_lesson_is_refused_by_name() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/corpus/golden/lessons");
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("the lessons")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "kom"))
        .collect();
    names.sort();
    let mut built = Vec::new();
    for path in names {
        let source = std::fs::read_to_string(&path).expect("a lesson");
        let Ok(module) = komorebi_ir::api::build(&source) else {
            continue; // a lesson the check refuses (T8) has no module
        };
        match tensix::sources(&module) {
            Ok(_) => built.push(path.file_stem().map(|s| s.to_string_lossy().into_owned())),
            Err(refusals) => {
                assert!(!refusals.is_empty());
                for r in refusals {
                    assert_eq!(r.code, "K964", "{}: {r:?}", path.display());
                    assert!(r.what.contains("does not emit"), "{r:?}");
                }
            }
        }
    }
    assert_eq!(built, vec![Some("t01_scale".to_string())]);
}

#[test]
fn without_an_origin_the_files_carry_no_line() {
    let source = std::fs::read_to_string(here().join("t1_class.kom")).expect("the .kom");
    let module = komorebi_ir::api::build(&source).unwrap_or_else(|d| panic!("{d:?}"));
    let files = tensix::sources(&module).unwrap_or_else(|r| panic!("{r:?}"));
    for (name, text) in &files {
        assert!(!text.contains("// line "), "{name}");
        assert!(!text.contains("komorebi.dsl from"), "{name}");
    }
    assert_eq!(tensix::sources(&module), Ok(files), "twice, the same files");
}

/// The class form's module, no origin, with `from` replaced by `to` once.
fn edited(from: &str, to: &str) -> Module {
    let source = std::fs::read_to_string(here().join("t1_class.kom")).expect("the .kom");
    assert!(source.contains(from), "{from}");
    komorebi_ir::api::build(&source.replacen(from, to, 1)).unwrap_or_else(|d| panic!("{d:?}"))
}

#[test]
fn two_partitions_of_one_range_are_one_and_two_that_differ_are_two() {
    let y = "partition(y.tiles(first, tiles), worker.rank, group.size)";
    let file = |module: &Module, name: &str| {
        let files = tensix::sources(module).unwrap_or_else(|r| panic!("{r:?}"));
        files
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, text)| text)
            .expect("the file")
    };
    let partitions = |text: &str| text.matches("km::Partition").count();
    let same = edited(y, y);
    assert_eq!(partitions(&file(&same, "movement.cpp")), 1);
    // y's part from another first word: two ranges, y's stream over its own.
    let moved = edited(
        y,
        "partition(y.tiles(tiles, tiles), worker.rank, group.size)",
    );
    let movement = file(&moved, "movement.cpp");
    assert_eq!(partitions(&movement), 2);
    assert!(movement.contains("const km::Partition dst = km::partition(tiles, km_rank, km_size);"));
    assert!(
        movement.contains("Y(y), (tiles + dst.start), dst.len"),
        "{movement}"
    );
    assert_eq!(
        partitions(&file(&moved, "compute.cpp")),
        1,
        "compute reads x's alone"
    );
    // Another count: the sink is not its loop's stream's count (refused, as today).
    let other = edited(
        y,
        "partition(y.tiles(first, first), worker.rank, group.size)",
    );
    let refused = tensix::sources(&other).expect_err("refused");
    assert!(refused[0].what.contains("another count"), "{refused:?}");
}

fn target(arch: &str) -> Target {
    Target {
        arch: arch.to_string(),
        features: Default::default(),
    }
}

#[test]
fn the_backend_compiles_the_first_lesson_to_kernels_built_at_load() {
    let module = edited("module scale", "module scale");
    let output = Tensix
        .compile(&module, &target("tensix"))
        .unwrap_or_else(|r| panic!("{r:?}"));
    let names: Vec<&str> = output.files.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "compute.cpp",
            "movement.cpp",
            "scale.h",
            "tile_times.hpp",
            "tile_times.txt"
        ]
    );
    let [entry] = output.entries.as_slice() else {
        panic!("one entry");
    };
    assert_eq!((entry.name.as_str(), entry.number), ("Scale.run", 1));
    assert_eq!(
        entry.words,
        [
            ArgWord::TensorAddress(0),
            ArgWord::TensorAddress(1),
            ArgWord::Word(2),
            ArgWord::Word(3),
            ArgWord::Word(4)
        ]
    );
    let bound = |tensor| Bound {
        tensor,
        first: Expr::Word(2),
        count: Expr::Word(3),
    };
    assert_eq!(entry.bounds, [bound(0), bound(1)]);
    let Build::Kernels(kernels) = &output.build else {
        panic!("kernels built at load");
    };
    let layouts = [
        CompileArg::TensorLayout { param: 0 },
        CompileArg::TensorLayout { param: 1 },
    ];
    let shapes: Vec<(&str, Processor, usize)> = kernels
        .iter()
        .map(|k| (k.source.as_str(), k.processor, k.args.len()))
        .collect();
    assert_eq!(
        shapes,
        [
            ("movement.cpp", Processor::Reader, 2),
            ("movement.cpp", Processor::Writer, 2),
            ("compute.cpp", Processor::Compute, 0)
        ]
    );
    assert_eq!(kernels[0].args, layouts);

    // A chip's name is refused at compile; the chip is checked at load, and both
    // chips take the one output (one key).
    for chip in ["blackhole", "wormhole_b0"] {
        let refused = Tensix.compile(&module, &target(chip)).expect_err("a chip");
        assert!(refused[0].what.contains("not a chip"), "{refused:?}");
        assert!(load::chip(&target(chip)).is_ok());
    }
    let again = Tensix
        .compile(&module, &target("tensix"))
        .unwrap_or_else(|r| panic!("{r:?}"));
    assert_eq!(again.key, output.key, "one key for both chips");
    assert!(load::chip(&target("tensix")).is_err());
    let other = Tensix.check(&module, &target("x86_64"));
    assert!(other.iter().any(|r| r.what.contains("not `x86_64`")));
    let mut featured = target("tensix");
    featured.features.insert("avx2".to_string());
    assert!(Tensix.compile(&module, &featured).is_err());
}

#[test]
fn the_key_covers_the_positions_and_names() {
    assert_eq!(Tensix.reads(), Reads::Everything);
    let module = edited("module scale", "module scale");
    let moved = edited("module scale", "\nmodule scale");
    let source = std::fs::read_to_string(here().join("t1_class.kom")).expect("the .kom");
    let renamed =
        komorebi_ir::api::build(&source.replace("src", "part")).unwrap_or_else(|d| panic!("{d:?}"));
    let key = |m: &Module| CacheKey::for_backend(m, &Tensix, &target("tensix"));
    assert_ne!(key(&module), key(&moved), "a moved line");
    assert_ne!(key(&module), key(&renamed), "a value renamed");
    let text = |m: &Module| tensix::sources(m).unwrap_or_else(|r| panic!("{r:?}"));
    let movement = |m: &Module| text(m).into_iter().find(|(n, _)| n == "movement.cpp");
    assert!(movement(&renamed).is_some_and(|(_, t)| t.contains("km::Partition part =")));
}

#[test]
fn the_statement_is_the_chips() {
    let s = Tensix.statement(&target("tensix"));
    assert_eq!(s.tile_shapes.len(), 1);
    assert_eq!(s.tensor_words, 1);
    assert!(s
        .limits
        .contains(&(komorebi_ir::backend::Limit::ArgumentWords, 10)));
    assert!(s
        .limits
        .contains(&(komorebi_ir::backend::Limit::HandlersPerProgram, 239)));
    let machine = komorebi_ir::backend::Machine {
        workers: 120,
        worker_memory: 1 << 20,
        islands: 1,
        options: vec![("depth".to_string(), 100)],
    };
    let at = Tensix.statement_at(&target("blackhole"), &machine, &Default::default());
    assert_eq!(
        (at.workers, at.in_flight, at.local_state_bytes),
        (Some(120), Some(64), Some(1 << 20))
    );
    let built = komorebi_ir::backend::Built {
        dir: Default::default(),
        facts: vec![("l1_used".to_string(), 4096)],
    };
    let machine = komorebi_ir::backend::Machine {
        options: Vec::new(),
        ..machine
    };
    let at = Tensix.statement_at(&target("blackhole"), &machine, &built);
    assert_eq!(
        (at.in_flight, at.local_state_bytes),
        (Some(4), Some((1 << 20) - 4096))
    );
}

#[test]
fn the_version_covers_every_framework_header_on_the_disk() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/komorebi_native/include");
    let mut on_disk = Vec::new();
    let mut folders = vec![root.join("komorebi/abi"), root.join("komorebi/device")];
    while let Some(folder) = folders.pop() {
        for entry in std::fs::read_dir(&folder).expect("a framework folder") {
            let path = entry.expect("an entry").path();
            if path.is_dir() {
                folders.push(path);
            } else if path.extension().is_some_and(|e| e == "h" || e == "hpp") {
                let name = path.strip_prefix(&root).expect("under the root");
                on_disk.push(name.to_string_lossy().into_owned());
            }
        }
    }
    on_disk.sort();
    let mut listed: Vec<String> = tensix::FRAMEWORK
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect();
    listed.sort();
    assert_eq!(listed, on_disk, "the headers the version hashes");
}
