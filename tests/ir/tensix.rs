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
use komorebi_ir::backend::tensix::{self, chip, Tensix};
use komorebi_ir::backend::{
    ArgWord, Backend, Bound, Build, CacheKey, CompileArg, Expr, Processor, Reads, Setting, Target,
};
use komorebi_ir::bytecode::hash::{hex, sha256};
use komorebi_ir::bytecode::{decode, encode};
use komorebi_ir::program::body::InstId;
use komorebi_ir::program::notes::{Anchor, Origin, Owner};
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
            ("param", [p, h, k, ..]) => {
                Anchor::ParamOf(Owner::Handler(ProgramId(*p), HandlerId(*h)), *k)
            }
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
    let Build::Kernels {
        kernels: built,
        buffers,
    } = &output.build
    else {
        panic!("kernels built at load");
    };
    let loaded = load::kernels(&module, built, &stand_ins).unwrap_or_else(|r| panic!("{r:?}"));
    let joined = |pairs: Vec<(String, String)>| {
        let mut pairs = pairs;
        pairs.sort();
        pairs
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    // An option as the script prints it: a bool in lower case, a text as itself.
    let settings = |options: &[(String, Setting)]| {
        options
            .iter()
            .map(|(k, v)| {
                let v = match v {
                    Setting::Bool(b) => b.to_string(),
                    Setting::Text(t) => t.clone(),
                };
                (k.clone(), v)
            })
            .collect::<Vec<_>>()
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
                joined(k.defines.clone()),
                joined(settings(&k.options))
            )
        })
        .collect();
    let theirs: Vec<&str> = printed
        .lines()
        .filter(|l| l.starts_with("KERNEL "))
        .collect();
    assert_eq!(described, theirs, "the kernels handed to the runtime");
    // The buffers, as the Program's halves (description.Buffer: its format by the
    // loader's name of it, chip::dtype), then its pairs (description.BufferPair: mover
    // A's half, mover B's), a pair once, by its first half.
    let made: Vec<String> = buffers
        .iter()
        .map(|b| {
            let dtype = chip::dtype(b.format).unwrap_or("none");
            format!("BUFFER {} {dtype} {} {}", b.index, b.page_bytes, b.pages)
        })
        .chain(
            buffers
                .iter()
                .filter_map(|b| {
                    b.pair
                        .filter(|other| b.index < *other)
                        .map(|o| (b.index, o))
                })
                .map(|(a, b)| format!("PAIR {a} {b}")),
        )
        .collect();
    let theirs: Vec<&str> = printed
        .lines()
        .filter(|l| l.starts_with("BUFFER ") || l.starts_with("PAIR "))
        .collect();
    assert_eq!(made, theirs, "the buffers made at load and their pairs");

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

#[test]
fn the_ranges_and_values_not_built_are_refused_by_name() {
    let streams = "let out = sink(dst, block = 1)\n        for xb in stream(src, block = 1) {";
    let refused = |to: &str| {
        let module = edited(streams, to);
        let refusals = tensix::sources(&module).expect_err("refused");
        assert!(refusals.iter().all(|r| r.code == "K964"), "{refusals:?}");
        refusals
            .into_iter()
            .map(|r| r.what)
            .collect::<Vec<_>>()
            .join("; ")
    };
    // a sink of tiles and a loop over a partition
    let mixed = refused(
        "let out = sink(y.tiles(first, tiles), block = 1)\n        for xb in stream(src, block = 1) {",
    );
    assert!(
        mixed.contains("a sink of a partition and its loop's stream of tiles"),
        "{mixed}"
    );
    // a named value no stream reads
    let unread = refused(
        "let n = first + 1\n        let out = sink(dst, block = 1)\n        for xb in stream(src, block = 1) {",
    );
    assert!(unread.contains("a named value no stream reads"), "{unread}");
    // a read stream whose count is a named value: compute would print it
    let counted = refused(
        "let n = tiles + 1\n        let out = sink(y.tiles(first, n), block = 1)\n        \
         for xb in stream(x.tiles(first, n), block = 1) {",
    );
    assert!(
        counted.contains("a read stream's count of a named value"),
        "{counted}"
    );
    // the same streams of tiles over words: built, no partition printed
    let module = edited(
        streams,
        "let out = sink(y.tiles(first, tiles), block = 1)\n        for xb in stream(x.tiles(first, tiles), block = 1) {",
    );
    let files = tensix::sources(&module).unwrap_or_else(|r| panic!("{r:?}"));
    let movement = &files
        .iter()
        .find(|(n, _)| n == "movement.cpp")
        .expect("movement")
        .1;
    assert!(movement.contains("Y(y), first, tiles,"), "{movement}");
    assert!(!movement.contains("km_size"), "{movement}");
    // a range of a difference is stated as a bound, as every range of a tensor is
    let module = edited(
        streams,
        "let out = sink(y.tiles(first, tiles - first), block = 1)\n        \
         for xb in stream(x.tiles(first, tiles - first), block = 1) {",
    );
    let output = Tensix
        .compile(&module, &target("tensix"))
        .unwrap_or_else(|r| panic!("{r:?}"));
    let less = Expr::Sub(Box::new(Expr::Word(3)), Box::new(Expr::Word(2)));
    let bound = |tensor| Bound {
        tensor,
        first: Expr::Word(2),
        count: less.clone(),
    };
    assert_eq!(output.entries[0].bounds, [bound(0), bound(1)]);
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
    let Build::Kernels { kernels, .. } = &output.build else {
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
        islands: komorebi_ir::types::IslandGraph::one(120),
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

#[test]
fn a_load_takes_its_tensors_by_name_and_names_formats_as_the_loader_does() {
    use komorebi_ir::api::load::load;
    let module = edited("module scale", "module scale");
    let t = LoadTensor {
        pages: 2048,
        layout: vec![2, 2048],
        dram_page: None,
    };
    let given = |names: &[&str]| -> Vec<(String, LoadTensor)> {
        names.iter().map(|n| (n.to_string(), t.clone())).collect()
    };
    let loaded = load(&module, &given(&["y", "x"]), Some("wormhole_b0"), None).expect("loads");
    assert_eq!(loaded.thread.0, "scale_thread.h");
    let ordered = [t.clone(), t.clone()];
    assert_eq!(
        loaded.thread.1,
        thread_header(&module, &ordered).expect("the header")
    );
    assert!(loaded.buffers.iter().all(|b| b.dtype == "bfloat16"));
    assert_eq!(loaded.kernels[0].args, [2, 2048, 2, 2048]);
    let refused =
        |names: &[&str], chip| load(&module, &given(names), chip, None).expect_err("refused");
    assert!(refused(&["x"], None)[0]
        .what
        .contains("no tensor given for `y`"));
    assert!(refused(&["x", "y", "x"], None)[0]
        .what
        .contains("given twice"));
    assert!(refused(&["x", "y", "z"], None)[0]
        .what
        .contains("a tensor `z`"));
    assert!(refused(&["x", "y"], Some("grayskull"))[0]
        .what
        .contains("not `grayskull`"));
}

#[test]
fn a_sum_other_than_program_2s_is_refused_by_name() {
    let source = std::fs::read_to_string(here().join("steps_sum.kom")).expect("the .kom");
    let built = |from: &str, to: &str| {
        assert!(source.contains(from), "{from}");
        komorebi_ir::api::build(&source.replacen(from, to, 1)).unwrap_or_else(|d| panic!("{d:?}"))
    };
    let refused = |from: &str, to: &str| {
        let refusals = tensix::sources(&built(from, to)).expect_err("refused");
        assert!(refusals.iter().all(|r| r.code == "K964"), "{refusals:?}");
        let what: Vec<String> = refusals.into_iter().map(|r| r.what).collect();
        what.join("; ")
    };
    // a reduce's stream in a pattern (of a second u32, `p`) is not built
    let given = source
        .replacen("tiles: u32,\n", "tiles: u32,\n        p: u32,\n", 1)
        .replacen("block = 1) {", "block = 1) @tensix.pattern(p) {", 1);
    let module = komorebi_ir::api::build(&given).unwrap_or_else(|d| panic!("{d:?}"));
    let patterned = tensix::sources(&module).expect_err("refused");
    assert!(
        patterned[0].what.contains("a reduce's stream in a pattern"),
        "{patterned:?}"
    );
    // program 2 as written is built, its leader rank 0
    let files = tensix::sources(&built("", "")).unwrap_or_else(|r| panic!("{r:?}"));
    let movement = &files
        .iter()
        .find(|(n, _)| n == "movement.cpp")
        .expect("movement")
        .1;
    assert!(movement.contains("if (km_rank == 0) {"), "{movement}");
    let max = refused("with = sum", "with = max");
    assert!(max.contains("a reduce with other than sum"), "{max}");
    // (a store outside `if worker.is_leader` is the language's own K405)
    let after = refused(
        "            store t.tiles(0, 1) <- whole\n        }",
        "            store t.tiles(0, 1) <- whole\n        }\n        \
         if worker.is_leader {\n            store t.tiles(0, 1) <- whole\n        }",
    );
    assert!(after.contains("lines after its leader's store"), "{after}");
}

#[test]
fn a_load_sizes_a_reduces_arrivals_from_the_machines_islands() {
    use komorebi_ir::api::load::load;
    use komorebi_ir::backend::Machine;
    use komorebi_ir::types::IslandGraph;
    let machine = |sizes: Vec<usize>| Machine {
        workers: sizes.iter().sum::<usize>() as u64,
        worker_memory: 1 << 20,
        islands: IslandGraph {
            links: Some((1..sizes.len()).map(|i| (i - 1, i)).collect()),
            sizes,
        },
        options: Vec::new(),
    };
    let tensor = |pages| LoadTensor {
        pages,
        layout: vec![2, 4096],
        dram_page: None,
    };
    let source = std::fs::read_to_string(here().join("steps_sum.kom")).expect("the .kom");
    let sum = komorebi_ir::api::build(&source).unwrap_or_else(|d| panic!("{d:?}"));
    let given = vec![("s".to_string(), tensor(768)), ("t".to_string(), tensor(1))];
    let compiled = tensix::sources(&sum).unwrap_or_else(|r| panic!("{r:?}"));
    let compiled = &compiled
        .iter()
        .find(|(n, _)| n == "stephandlers.h")
        .expect("the header")
        .1;
    // none, one island, or two of 6: today's 7 slots, the compile's header itself
    let none = load(&sum, &given, None, None).expect("loads");
    assert_eq!(
        none.header,
        ("stephandlers.h".to_string(), compiled.clone())
    );
    for sizes in [vec![12], vec![6, 6]] {
        assert_eq!(
            load(&sum, &given, None, Some(&machine(sizes))),
            Ok(none.clone())
        );
    }
    // more rounds than 7 on the largest island, or a slot each other island: the
    // arrival's slots in the header and its buffer's pages (`reduce.tree_slots`)
    for (sizes, slots) in [
        (vec![120, 120], 8),
        (vec![60, 60, 60], 8),
        (vec![120; 4], 10),
    ] {
        let on = load(&sum, &given, None, Some(&machine(sizes))).expect("loads");
        let line = format!("constexpr km::Arrival kArrival0 = {{8, {slots}, 1}};");
        assert!(on.header.1.contains(&line), "{}", on.header.1);
        assert_eq!(
            on.header
                .1
                .replace(&line, "constexpr km::Arrival kArrival0 = {8, 7, 1};"),
            *compiled
        );
        let arrival = on.buffers.iter().find(|b| b.index == 8).expect("kArrival0");
        assert_eq!(arrival.pages, slots);
        assert_eq!(on.thread, none.thread);
        assert_eq!(on.kernels, none.kernels);
    }
    // a program with no reduce: the same load on any islands
    let scale = edited("module scale", "module scale");
    let t = LoadTensor {
        pages: 2048,
        layout: vec![2, 2048],
        dram_page: None,
    };
    let both = vec![("x".to_string(), t.clone()), ("y".to_string(), t)];
    assert_eq!(
        load(&scale, &both, None, Some(&machine(vec![120, 120]))),
        load(&scale, &both, None, None)
    );
}

#[test]
fn a_load_refuses_islands_that_are_not_a_graph_of_its_workers_as_a_run_does() {
    use komorebi_ir::api::load::load;
    use komorebi_ir::backend::Machine;
    use komorebi_ir::types::{IslandGraph, IslandRefusal};
    let scale = edited("module scale", "module scale");
    let t = LoadTensor {
        pages: 2048,
        layout: vec![2, 2048],
        dram_page: None,
    };
    let both = vec![("x".to_string(), t.clone()), ("y".to_string(), t)];
    let on = |sizes: &[usize], links: Option<&[(usize, usize)]>| Machine {
        workers: 4,
        worker_memory: 1 << 20,
        islands: IslandGraph {
            sizes: sizes.to_vec(),
            links: links.map(<[(usize, usize)]>::to_vec),
        },
        options: Vec::new(),
    };
    // each graph the interpreter refuses, refused at the backend's load with the
    // type's own reason
    for (machine, why) in [
        (on(&[2, 2], Some(&[])), IslandRefusal::NotConnected),
        (
            on(&[1, 1, 1, 1], Some(&[(0, 1), (2, 3)])),
            IslandRefusal::NotConnected,
        ),
        (
            on(&[2, 1], None),
            IslandRefusal::NotTheWorkers {
                held: 3,
                workers: 4,
            },
        ),
        (on(&[4, 0], None), IslandRefusal::Empty(1)),
        (on(&[2, 2], Some(&[(0, 2)])), IslandRefusal::NoSuchIsland(2)),
    ] {
        let refused = load(&scale, &both, None, Some(&machine)).expect_err("refused");
        assert_eq!(refused.len(), 1, "{refused:?}");
        assert_eq!(refused[0].code, "K966");
        assert_eq!(
            refused[0].what,
            format!("the machine given at load is not a machine: {why}")
        );
    }
    // no links given: every pair connected, one documented meaning for both
    load(&scale, &both, None, Some(&on(&[2, 2], None))).expect("loads");
    load(&scale, &both, None, Some(&on(&[1, 3], Some(&[(1, 0)])))).expect("loads");
}

#[test]
fn a_zip_other_than_program_3s_is_refused_by_name() {
    let source = std::fs::read_to_string(here().join("steps_add.kom")).expect("the .kom");
    let built = |from: &str, to: &str| {
        assert!(source.contains(from), "{from}");
        komorebi_ir::api::build(&source.replacen(from, to, 1)).unwrap_or_else(|d| panic!("{d:?}"))
    };
    let refused = |from: &str, to: &str| {
        let refusals = tensix::sources(&built(from, to)).expect_err("refused");
        assert!(refusals.iter().all(|r| r.code == "K964"), "{refusals:?}");
        let what: Vec<String> = refusals.into_iter().map(|r| r.what).collect();
        what.join("; ")
    };
    // program 3 as written: both read streams in one move, each tile copied to its
    // own Dst tile, two a call
    let files = tensix::sources(&built("", "")).unwrap_or_else(|r| panic!("{r:?}"));
    let file = |name: &str| &files.iter().find(|(n, _)| n == name).expect(name).1;
    assert!(file("movement.cpp").contains("movement.move(km_read0, km_read1, km_write0)"));
    assert!(
        file("compute.cpp").contains("copy_tile(km_half1, km_tile0 + km_slot0, 2 * km_slot0 + 1);")
    );
    let swapped = refused("plus(x, y)", "plus(y, x)");
    assert!(swapped.contains("the loop's tiles in order"), "{swapped}");
    let third = refused(
        "let results = sink(",
        "let c = stream(a.tiles(first, width), block = 1)\n        let results = sink(",
    );
    assert!(third.contains("a third read stream"), "{third}");
    // a pattern's hint names one u32 parameter (here a fifth, `p`), whose word the
    // program then reads no more; on a loop, each stream it walks
    let with_p = |from: &str, to: &str| {
        let given = source.replacen("width: u32,\n", "width: u32,\n        p: u32,\n", 1);
        assert!(given.contains(from), "{from}");
        let module = komorebi_ir::api::build(&given.replacen(from, to, 1))
            .unwrap_or_else(|d| panic!("{d:?}"));
        tensix::sources(&module)
    };
    let refused_p = |from: &str, to: &str| {
        let refusals = with_p(from, to).expect_err("refused");
        assert!(refusals.iter().all(|r| r.code == "K964"), "{refusals:?}");
        let what: Vec<String> = refusals.into_iter().map(|r| r.what).collect();
        what.join("; ")
    };
    let loop_head = "block = 1)) {";
    let named = refused_p(loop_head, "block = 1)) @tensix.pattern(a) {");
    assert!(
        named.contains("other than of one `u32` parameter"),
        "{named}"
    );
    let read = refused_p(
        "let first = worker.rank * width",
        "let first = worker.rank * width @tensix.pattern(p)",
    );
    assert!(
        read.contains("on other than a loop over streams or a sink"),
        "{read}"
    );
    let files =
        with_p(loop_head, "block = 1)) @tensix.pattern(p) {").unwrap_or_else(|r| panic!("{r:?}"));
    let file = |name: &str| &files.iter().find(|(n, _)| n == name).expect(name).1;
    assert!(file("compute.cpp").contains("km::Pattern p = km::Pattern::of(command.a4);"));
    assert!(file("compute.cpp").contains("if (!p.valid()) {"));
    assert!(file("movement.cpp").contains("kRead1, p\n"));
    assert!(!file("stephandlers.h").contains("kRead1Pattern"));
    assert!(file("stephandlers.h").contains("kWrite0Pattern = km::pattern(\"B\")"));
}

#[test]
fn a_counted_loop_other_than_program_5s_repeat_is_refused_by_name() {
    // program 5 as the frontend writes it: a drop repeated, and a copy
    let source = r#"module movement

tile [32, 32]
format bf16

param BLOCK: u32 = 16 @bind(build)

program Movement {
    handler read(
        source: tensor<bf16>,
        tiles: u32,
        pattern: u32,
        repetitions: u32,
    ) @on(group) {
        let pages = source.tiles(worker.rank * tiles, tiles)
        for i in range(repetitions) {
            for _ in stream(pages, block = BLOCK) @tensix.pattern(pattern) {
            }
        }
    }
    handler copy(
        source: tensor<bf16>,
        output: tensor<bf16>,
        tiles: u32,
        pattern: u32,
    ) @on(group) {
        let first = worker.rank * tiles
        let out = sink(output.tiles(first, tiles), block = BLOCK) @tensix.pattern(pattern)
        for tile_ in stream(source.tiles(first, tiles), block = BLOCK) @tensix.pattern(pattern) {
            out.push(tile_)
        }
    }
}
"#;
    let built = |from: &str, to: &str| {
        assert!(source.contains(from), "{from}");
        komorebi_ir::api::build(&source.replacen(from, to, 1)).unwrap_or_else(|d| panic!("{d:?}"))
    };
    let refused = |from: &str, to: &str| {
        let refusals = tensix::sources(&built(from, to)).expect_err("refused");
        assert!(refusals.iter().all(|r| r.code == "K964"), "{refusals:?}");
        let what: Vec<String> = refusals.into_iter().map(|r| r.what).collect();
        what.join("; ")
    };
    let files = tensix::sources(&built("", "")).unwrap_or_else(|r| panic!("{r:?}"));
    let file = |name: &str| &files.iter().find(|(n, _)| n == name).expect(name).1;
    assert!(file("movement.cpp").contains("tiles, kRead0Block, repetitions, kRead0, pattern"));
    assert!(file("movement.cpp").contains("movement.move(km_read0)"));
    assert!(file("compute.cpp").contains("km_block0 < km_blocks0 * repetitions;"));
    assert!(file("compute.cpp").contains("copy_tile(km_half0, km_tile0, 0);"));
    // the nest is a counted loop whose body is exactly one stream loop
    let stream_loop =
        "for _ in stream(pages, block = BLOCK) @tensix.pattern(pattern) {\n            }";
    let other = refused(
        stream_loop,
        &format!("let k = tiles + 1\n            {stream_loop}"),
    );
    assert!(
        other.contains("a counted loop other than around one stream loop"),
        "{other}"
    );
    let two = refused(
        stream_loop,
        &format!("{stream_loop}\n            {stream_loop}"),
    );
    assert!(two.contains("lines after its stream loop"), "{two}");
    let after = refused(
        "        for i in range(repetitions) {\n            for _",
        "        for _ in stream(pages, block = BLOCK) {\n        }\n        for i in range(repetitions) {\n            for _",
    );
    assert!(after.contains("lines after its stream loop"), "{after}");
    // a repeat of a loop with a body (today's repeat moves blocks only)
    let pushed = refused(
        "        for tile_ in stream(source.tiles(first, tiles), block = BLOCK) @tensix.pattern(pattern) {\n            out.push(tile_)\n        }",
        "        let view = source.tiles(first, tiles)\n        for i in range(tiles) {\n            for tile_ in stream(view, block = BLOCK) @tensix.pattern(pattern) {\n                out.push(tile_)\n            }\n        }",
    );
    assert!(
        pushed.contains("a counted loop around a stream loop that pushes"),
        "{pushed}"
    );
}
