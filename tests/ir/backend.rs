//! The backend interface on the corpus: the shared check refuses by name what a
//! statement does not provide, implement, allow or compute, at compile and at load; the
//! cache key follows the build hash, the backend and the target; and the emitter's walk
//! gives every instruction of every body one event, the constructs nested.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use komorebi_ir::backend::emit::{walk, Event};
use komorebi_ir::backend::{
    check_load, check_statement, Backend, CacheKey, LoadStatement, Machine, Output, Provided,
    Refusal, Statement, Target,
};
use komorebi_ir::backend::{
    Limit, FORMAT_NOT_COMPUTED, NOT_BUILT, NOT_PROVIDED, NO_IMPLEMENTATION, OVER_LIMIT,
};
use komorebi_ir::program::body::Body;
use komorebi_ir::program::notes::Anchor;
use komorebi_ir::program::Module;
use komorebi_ir::types::{Ability, Format, FuncId, ProgramId, TileShape};

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

/// A statement that provides nothing beyond the base six, computes every format and
/// has no limit.
fn bare() -> Statement {
    Statement {
        backend: "test",
        abilities: Vec::new(),
        implementations: Vec::new(),
        limits: Vec::new(),
        formats: Format::ALL.to_vec(),
        formats_document: "docs/ir/backends.md",
        hints: Vec::new(),
        progress: Vec::new(),
    }
}

/// A backend for the load check: a tile takes four bytes an element.
struct Test;

impl Backend for Test {
    fn name(&self) -> &'static str {
        "test"
    }
    fn version(&self) -> String {
        "0".into()
    }
    fn statement(&self, _: &Target) -> Statement {
        bare()
    }
    fn statement_at(&self, _: &Target, machine: &Machine) -> LoadStatement {
        LoadStatement {
            workers: Some(machine.workers),
            islands: Some(machine.islands),
            local_state_bytes: Some(machine.worker_memory),
            ..LoadStatement::default()
        }
    }
    fn tile_bytes(&self, _: Format, shape: TileShape) -> u64 {
        shape.rows * shape.cols * 4
    }
    fn check(&self, program: &Module, target: &Target) -> Vec<Refusal> {
        check_statement(program, &self.statement(target))
    }
    fn compile(&self, _: &Module, _: &Target) -> Result<Output, Vec<Refusal>> {
        Err(Vec::new())
    }
}

fn target() -> Target {
    Target {
        arch: "x86_64".into(),
        features: BTreeSet::new(),
    }
}

#[test]
fn a_requirement_not_provided_is_refused_by_name_at_its_construct() {
    let mut refused = 0;
    for (name, text) in modules("lessons") {
        let module = built(&text);
        let refusals = check_statement(&module, &bare());
        // One refusal for each requirement, none for anything else.
        assert_eq!(
            refusals.len(),
            module.requirements.len(),
            "{name}: {refusals:?}"
        );
        for (r, req) in refusals.iter().zip(&module.requirements) {
            let code = match req.ability {
                Ability::Impl(_) => NO_IMPLEMENTATION,
                _ => NOT_PROVIDED,
            };
            assert_eq!(r.code, code, "{name}: {r:?}");
            assert!(r.at.is_some());
        }
        refused += refusals.len();
    }
    assert!(
        refused >= 4,
        "the lessons require grid, dynamic, islands and an impl"
    );

    let t08 = lesson("t08_trace");
    let r = check_statement(&t08, &bare());
    assert!(
        r.iter()
            .any(|r| r.code == NO_IMPLEMENTATION && matches!(r.at, Some(Anchor::Func(FuncId(_))))),
        "{r:?}"
    );

    let t06 = lesson("t06_rows");
    let r = check_statement(&t06, &bare());
    assert!(
        r.iter()
            .any(|r| r.code == NOT_PROVIDED && r.what.contains("`group = grid`")),
        "{r:?}"
    );
}

#[test]
fn a_statement_that_provides_everything_refuses_nothing() {
    for (name, text) in modules("lessons") {
        let module = built(&text);
        let mut s = bare();
        s.abilities = vec![
            Provided::Grid,
            Provided::Dynamic,
            Provided::Islands,
            Provided::LocalState,
        ];
        s.implementations = module
            .requirements
            .iter()
            .filter_map(|r| match &r.ability {
                Ability::Impl(n) => Some(n.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(check_statement(&module, &s), Vec::new(), "{name}");
    }
}

#[test]
fn a_limit_and_a_format_are_refused_by_name() {
    let t02 = lesson("t02_count");
    let mut s = bare();
    s.limits = vec![(Limit::HandlersPerProgram, 1), (Limit::Parameters, 2)];
    let r = check_statement(&t02, &s);
    assert!(
        r.iter().any(|r| r.code == OVER_LIMIT
            && r.what == "2 handlers in a program, and the test backend allows 1"
            && r.at == Some(Anchor::Program(ProgramId(0)))),
        "{r:?}"
    );
    assert!(
        r.iter()
            .any(|r| r.code == OVER_LIMIT && r.what.starts_with("3 parameters")),
        "{r:?}"
    );

    let t01 = lesson("t01_scale");
    let mut s = bare();
    s.formats = vec![Format::F32];
    let r = check_statement(&t01, &s);
    assert_eq!(r.len(), 1, "{r:?}");
    assert_eq!(r[0].code, FORMAT_NOT_COMPUTED);
    assert!(r[0].what.contains("bf16"), "{r:?}");
}

#[test]
fn islands_and_local_state_are_compared_at_load() {
    let machine = |islands, worker_memory| Machine {
        workers: 8,
        worker_memory,
        islands,
    };
    let t12 = lesson("t12_spread");
    let at = Test.statement_at(&target(), &machine(1, 1 << 20));
    let r = check_load(&t12, &Test, &at);
    assert_eq!(r.len(), 1, "{r:?}");
    assert_eq!(r[0].code, NOT_PROVIDED);
    assert!(r[0].what.starts_with("`islands >= 2`"), "{r:?}");
    let at = Test.statement_at(&target(), &machine(2, 1 << 20));
    assert_eq!(check_load(&t12, &Test, &at), Vec::new());

    // `local_state >= tiles(1)`: one tile of 32 by 32, four bytes an element here.
    let t03 = lesson("t03_gain");
    let at = Test.statement_at(&target(), &machine(1, 4095));
    assert_eq!(check_load(&t03, &Test, &at).len(), 1);
    let at = Test.statement_at(&target(), &machine(1, 4096));
    assert_eq!(check_load(&t03, &Test, &at), Vec::new());
}

#[test]
fn the_cache_key_follows_the_build_hash_the_backend_and_the_target() {
    let (_, text) = modules("lessons")
        .into_iter()
        .find(|(n, _)| n == "t13_hinted")
        .expect("t13");
    let key =
        |text: &str, version: &str, t: &Target| CacheKey::of(&built(text), "test", version, t);
    let t = target();
    let base = key(&text, "1", &t);
    assert_eq!(base, key(&text, "1", &t), "one module, one key");
    assert_eq!(base.hex().len(), 64);
    let commented = text.replacen("tile [32, 32]\n", "// a comment\ntile [32, 32]\n", 1);
    assert_eq!(base, key(&commented, "1", &t), "a comment rebuilds nothing");
    let hinted = text.replacen("\"AAAB\"", "\"AABB\"", 1);
    assert_ne!(base, key(&hinted, "1", &t), "a hint is in the key");
    assert_ne!(
        base,
        key(&text, "2", &t),
        "the backend's version is in the key"
    );
    let mut avx = target();
    avx.features.insert("avx2".into());
    assert_ne!(base, key(&text, "1", &avx), "the target is in the key");
    assert_eq!(avx.canonical(), "x86_64 avx2");
    assert_ne!(
        base,
        CacheKey::of(&built(&text), "other", "1", &t),
        "the backend's name is in the key"
    );
}

fn bodies(m: &Module) -> Vec<&Body> {
    let mut out: Vec<&Body> = m.funcs.iter().filter_map(|f| f.body.as_ref()).collect();
    for p in &m.programs {
        out.extend(p.handlers.iter().map(|h| &h.body));
        for r in &p.runs {
            out.push(&r.seed);
            out.extend(r.on_message.iter().map(|(_, b)| b));
        }
    }
    out.extend(m.controls.iter().map(|c| &c.body));
    out
}

/// Every body of every module the corpus lowers: one event an instruction, in order;
/// each `Close` at the depth of its `Open`; `Else` and `Test` one deeper.
#[test]
fn the_walk_gives_every_instruction_one_event_and_nests() {
    let mut seen = 0;
    for sub in ["lessons", "reference", "run_fails", "fragments"] {
        for (name, text) in modules(sub) {
            let Ok(module) = komorebi_ir::api::build(&text) else {
                continue;
            };
            for body in bodies(&module) {
                let events = walk(body).unwrap_or_else(|e| panic!("{sub}/{name}: {e}"));
                assert_eq!(events.len(), body.insts.len(), "{sub}/{name}");
                let mut open: Vec<usize> = Vec::new();
                for (k, (depth, event)) in events.iter().enumerate() {
                    let id = match event {
                        Event::Open(id, _) => {
                            assert_eq!(*depth, open.len());
                            open.push(*depth);
                            *id
                        }
                        Event::Close(id) => {
                            assert_eq!(open.pop(), Some(*depth), "{sub}/{name}");
                            *id
                        }
                        Event::Else(id) | Event::Test(id, _) => {
                            assert_eq!(*depth, open.len());
                            assert!(*depth > 0);
                            *id
                        }
                        Event::Plain(id, _) => {
                            assert_eq!(*depth, open.len());
                            *id
                        }
                    };
                    assert_eq!(id.0 as usize, k, "{sub}/{name}: in order");
                }
                assert!(open.is_empty());
                seen += events.len();
            }
        }
    }
    assert!(seen > 500, "only {seen} instructions walked");
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
    assert!(built.library().is_file());
    assert!(built.driver().is_file());
    let again = komorebi_ir::backend::cpu::build::build(&out, &cache).expect("found built");
    assert_eq!(built, again, "built once, then found");
}
