//! The concrete backends on the corpus: the CPU backend refuses by name what it does
//! not emit, builds the first lesson, and places a build whole; `api::emit` names a
//! backend and compiles for its default target. Moved from `tests/ir/backend.rs`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use komorebi_ir::backend::{
    ArgWord, Backend, Build, Target, FORMAT_NOT_COMPUTED, NOT_BUILT, NOT_PROVIDED,
    NO_IMPLEMENTATION, OVER_LIMIT,
};
use komorebi_ir::program::Module;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/corpus/golden")
}

fn modules(sub: &str) -> Vec<(String, String)> {
    let dir = golden_dir().join(sub);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        panic!("{} is readable", dir.display());
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "kom"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let name = p
                .file_stem()
                .map_or(String::new(), |s| s.to_string_lossy().into_owned());
            (name, std::fs::read_to_string(&p).expect("readable"))
        })
        .collect()
}

fn built(text: &str) -> Module {
    komorebi_ir::api::build(text).unwrap_or_else(|d| panic!("{d:?}"))
}

fn lesson(name: &str) -> Module {
    let (_, text) = modules("lessons")
        .into_iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("the lesson {name}"));
    built(&text)
}
fn target() -> Target {
    Target {
        arch: "x86_64".into(),
        features: BTreeSet::new(),
    }
}

// ---- The CPU backend ----

/// Whether the tests that run a C++ compiler are skipped: only when `KOMO_IR_NO_CXX=1`
/// says so; without it a missing compiler fails them.
fn no_cxx() -> bool {
    std::env::var("KOMO_IR_NO_CXX").is_ok_and(|v| v == "1")
}

#[test]
fn the_cpu_backend_refuses_by_name_what_it_does_not_emit_yet() {
    let cpu = komorebi_ir::backend::cpu::Cpu;
    let mut emitted = Vec::new();
    for (name, text) in modules("lessons") {
        let module = built(&text);
        let refusals = cpu.check(&module, &target());
        if refusals.is_empty() {
            emitted.push(name);
            continue;
        }
        for r in &refusals {
            assert!(r.at.is_some(), "{name}: {r:?}");
            assert!(
                [
                    NOT_PROVIDED,
                    NO_IMPLEMENTATION,
                    OVER_LIMIT,
                    FORMAT_NOT_COMPUTED,
                    NOT_BUILT
                ]
                .contains(&r.code),
                "{name}: {r:?}"
            );
        }
        assert!(cpu.compile(&module, &target()).is_err(), "{name}");
    }
    assert_eq!(emitted, ["t01_scale"]);
}

#[test]
fn the_cpu_output_of_the_first_lesson_builds() {
    let cpu = komorebi_ir::backend::cpu::Cpu;
    let t01 = lesson("t01_scale");
    let out = cpu.compile(&t01, &target()).expect("t01 is emitted");
    assert_eq!(out, cpu.compile(&t01, &target()).expect("again"), "pure");
    assert_eq!(out.entries.len(), 1);
    assert_eq!(out.entries[0].name, "Scale.run");
    assert_eq!(out.entries[0].number, 0);
    assert_eq!(
        out.entries[0].words,
        [
            ArgWord::TensorAddress(0),
            ArgWord::TensorAddress(1),
            ArgWord::Word(2),
            ArgWord::Word(3),
            ArgWord::Word(4),
        ]
    );
    assert!(out.entries[0].bounds.is_empty());
    assert!(matches!(out.build, Build::Steps(_)));
    let source = out
        .files
        .iter()
        .find(|(n, _)| n == "module.cpp")
        .map(|(_, b)| String::from_utf8_lossy(b).into_owned())
        .expect("the source");
    for i in 0..t01.programs[0].handlers[0].body.insts.len() {
        assert!(
            source.contains(&format!("// i{i} ")),
            "i{i} has its statement"
        );
    }
    if no_cxx() {
        return;
    }
    let cache = Path::new(env!("CARGO_TARGET_TMPDIR")).join("komo-ir-cache");
    let built = komorebi_ir::backend::cpu::build::build(&out, &cache)
        .unwrap_or_else(|e| panic!("{e} (KOMO_IR_NO_CXX=1 skips this)"));
    assert!(komorebi_ir::backend::cpu::build::library(&built).is_file());
    assert!(komorebi_ir::backend::cpu::build::driver(&built).is_file());
    let again = komorebi_ir::backend::cpu::build::build(&out, &cache).expect("found built");
    assert_eq!(built, again, "built once, then found");
}

/// Two builds of one directory: the first put in place is kept whole; the second's
/// staging directory, which only it created, is removed, and nothing else is.
#[test]
fn a_losing_build_leaves_the_winners_directory_whole() {
    use komorebi_ir::backend::cpu::build::place;
    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("place-{}", std::process::id()));
    let dir = base.join("built");
    let write = |name: &'static str| {
        move |staged: &Path| std::fs::write(staged.join(name), name).map_err(|e| e.to_string())
    };
    place(&dir, write("winner")).expect("the first build");
    place(&dir, write("loser")).expect("the second build finds the first in place");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .expect("the directory")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    names.sort();
    assert_eq!(names, ["winner"]);
    assert_eq!(
        std::fs::read_to_string(dir.join("winner")).expect("whole"),
        "winner"
    );
    let left: Vec<String> = std::fs::read_dir(&base)
        .expect("the base")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    assert_eq!(left, ["built"], "no staging directory is left");
}

#[test]
fn a_backend_is_named_and_compiles_for_its_default_target() {
    use komorebi_ir::api::emit::{default_arch, emit, Refused, BACKENDS};
    let t01 = lesson("t01_scale");
    assert_eq!(BACKENDS, ["cpu", "tensix"]);
    assert_eq!(default_arch("tensix"), "tensix");
    assert_eq!(default_arch("cpu"), std::env::consts::ARCH);
    for name in BACKENDS {
        let out = emit(&t01, name, None).unwrap_or_else(|r| panic!("{name}: {r:?}"));
        assert_eq!(
            out,
            emit(&t01, name, Some(default_arch(name))).expect("the same")
        );
    }
    assert_eq!(
        emit(&t01, "gpu", None),
        Err(Refused::NoBackend("gpu".into()))
    );
    match emit(&t01, "tensix", Some("wormhole")) {
        Err(Refused::Backend(r)) => assert_eq!(r[0].code, "K964"),
        other => panic!("not refused: {other:?}"),
    }
}

#[test]
fn the_cpu_backend_refuses_a_function_on_blocks_by_name() {
    let m = built(
        "module blocks\n\ntile [32, 32]\nformat f32\n\n\
         func twice(x: tile<f32>[2], k: f32) -> tile<f32>[2] {\n    return x * k\n}\n",
    );
    let target = Target {
        arch: std::env::consts::ARCH.to_string(),
        features: BTreeSet::new(),
    };
    let refused = komorebi_ir::backend::cpu::Cpu
        .compile(&m, &target)
        .expect_err("refused");
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert_eq!(refused[0].code, "K964");
    assert!(
        refused[0].what.contains("a function on blocks"),
        "{refused:?}"
    );
}

#[test]
fn the_cpu_backend_refuses_a_reduction_along_a_row_of_a_block_by_name() {
    // reading R43's row form: a function on blocks, so refused as one (K964)
    let m = built(
        "module rows\n\ntile [32, 32]\nformat f32\n\n\
         func spread(x: tile<f32>[2]) -> tile<f32>[2] {\n    \
         return reduce_sum(x, axis = row).col()\n}\n",
    );
    let target = Target {
        arch: std::env::consts::ARCH.to_string(),
        features: BTreeSet::new(),
    };
    let refused = komorebi_ir::backend::cpu::Cpu
        .compile(&m, &target)
        .expect_err("refused");
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert_eq!(refused[0].code, "K964");
    assert!(
        refused[0].what.contains("a function on blocks"),
        "{refused:?}"
    );
}

#[test]
fn the_cpu_backend_refuses_a_product_of_blocks_by_name() {
    // reading R45: a function on blocks, so refused as one (K964)
    let m = built(
        "module product\n\ntile [32, 32]\nformat f32\n\n\
         func f(x: tile<f32>[1, 2], w: tile<f32>[2, 1]) -> tile<f32> {\n    \
         return matmul(x, w)\n}\n",
    );
    let target = Target {
        arch: std::env::consts::ARCH.to_string(),
        features: BTreeSet::new(),
    };
    let refused = komorebi_ir::backend::cpu::Cpu
        .compile(&m, &target)
        .expect_err("refused");
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert_eq!(refused[0].code, "K964");
    assert!(
        refused[0].what.contains("a function on blocks"),
        "{refused:?}"
    );
}
