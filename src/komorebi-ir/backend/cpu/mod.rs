//! The CPU backend: portable C++17 on the framework in `framework/` (embedded here),
//! one thread for each worker. At this stage it emits what the first lesson uses (a
//! handler on a group over tensors and scalars, views, a partition, streams and sinks
//! of one tile, functions on tiles) and refuses everything else by name (`K964`).
//! How it computes a format is its own document (`docs/ir/backends.md`); nothing here
//! claims a precision.

pub mod build;
pub mod emit;

use crate::bytecode::hash::{hashes, hex};
use crate::program::Module;
use crate::types::{Entry, Format, HandlerId, ProgramId, TileShape};

use super::{
    check_statement, ArgKind, Backend, CacheKey, EntryPoint, LoadStatement, Machine, Output,
    Refusal, Statement, Step, Target,
};

/// The framework's files, as the output carries them.
pub const FRAMEWORK: &[(&str, &str)] = &[
    ("komo_cpu.h", include_str!("framework/komo_cpu.h")),
    ("komo_cpu.hpp", include_str!("framework/komo_cpu.hpp")),
    ("pool.cpp", include_str!("framework/pool.cpp")),
    ("driver.cpp", include_str!("framework/driver.cpp")),
];

/// The emitted source's file name in the output.
pub const SOURCE: &str = "module.cpp";

/// The shared library the build makes.
pub const LIBRARY: &str = "module.so";

/// The driver the build makes: it runs a run script on the module.
pub const DRIVER: &str = "driver";

/// The compiler's flags, before the files.
const FLAGS: &[&str] = &["-std=c++17", "-O2", "-fPIC", "-pthread", "-I."];

/// The CPU backend.
pub struct Cpu;

impl Backend for Cpu {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn version(&self) -> String {
        format!("{}+{}", env!("CARGO_PKG_VERSION"), hex(&framework_hash()))
    }

    fn statement(&self, _target: &Target) -> Statement {
        Statement {
            backend: "cpu",
            abilities: Vec::new(),
            implementations: Vec::new(),
            limits: Vec::new(),
            formats: vec![Format::U32, Format::F32, Format::Bf16],
            formats_document: "docs/ir/backends.md",
            hints: Vec::new(),
            progress: vec!["every worker of a group runs on its own thread"],
        }
    }

    fn statement_at(&self, _target: &Target, machine: &Machine) -> LoadStatement {
        LoadStatement {
            workers: Some(machine.workers),
            islands: Some(1),
            local_state_bytes: Some(machine.worker_memory),
            in_flight: Some(1),
            live_messages: None,
        }
    }

    fn tile_bytes(&self, format: Format, shape: TileShape) -> u64 {
        let width = match format {
            Format::Bf16 => 2,
            _ => format.bytes().map_or(4, |b| b as u64),
        };
        shape.rows * shape.cols * width
    }

    fn check(&self, program: &Module, target: &Target) -> Vec<Refusal> {
        let mut out = check_statement(program, &self.statement(target));
        out.extend(emit::unbuilt_parts(program));
        out
    }

    fn compile(&self, program: &Module, target: &Target) -> Result<Output, Vec<Refusal>> {
        let refusals = self.check(program, target);
        if !refusals.is_empty() {
            return Err(refusals);
        }
        let build_hash = hex(&hashes(program).build);
        let mut files: Vec<(String, Vec<u8>)> = FRAMEWORK
            .iter()
            .map(|(name, text)| ((*name).to_string(), text.as_bytes().to_vec()))
            .collect();
        files.push((
            SOURCE.to_string(),
            emit::source(program, &build_hash).into_bytes(),
        ));
        let mut entries = Vec::new();
        for (p, prog) in program.programs.iter().enumerate() {
            for (h, handler) in prog.handlers.iter().enumerate() {
                entries.push(EntryPoint {
                    name: format!("{}.{}", prog.name, handler.name),
                    entry: Entry::Handler(ProgramId(p as u32), HandlerId(h as u32)),
                    symbol: format!("p{p}_h{h}"),
                    params: handler.params.iter().filter_map(ArgKind::of).collect(),
                });
            }
        }
        Ok(Output {
            key: CacheKey::of(program, self.name(), &self.version(), target),
            files,
            entries,
            build: vec![
                compiler_step(&["-shared", SOURCE, "pool.cpp", "-o", LIBRARY]),
                compiler_step(&[SOURCE, "pool.cpp", "driver.cpp", "-o", DRIVER]),
            ],
        })
    }
}

/// A step of the C++ compiler (`c++`: the build runs `$CXX` in its place when set).
fn compiler_step(rest: &[&str]) -> Step {
    Step {
        program: "c++".to_string(),
        args: FLAGS.iter().chain(rest).map(|a| (*a).to_string()).collect(),
    }
}

/// The hash of the framework's files: a change to them changes the backend's version.
fn framework_hash() -> [u8; 32] {
    let mut bytes = Vec::new();
    for (name, text) in FRAMEWORK {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(text.as_bytes());
        bytes.push(0);
    }
    crate::bytecode::hash::sha256(&bytes)
}
