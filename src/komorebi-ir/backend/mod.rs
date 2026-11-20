//! The backend interface: what a backend states about itself, as data; a check that
//! refuses a program by name; a compile that gives the backend's output, a pure
//! function whose build is data; and the key of that output in a cache. It reads the
//! program structure, the types and the byte code's hashes, and nothing of the front.
//!
//! What a backend states has two halves. `Backend::statement` is what is known at
//! compile, from the target alone: the abilities it provides, its implementations, the
//! limits a program's text can exceed, the formats and tile shapes it computes, the
//! hints it reads. `Backend::statement_at` is what is known only on a machine at load,
//! and after the build: how many workers and islands, how much worker memory for state,
//! how many invocations in flight. The order is compile, build, `statement_at`, then
//! `check_load`; `check` compares a program with the first half before compiling.
//!
//! A backend says which parts of a byte-code file its output depends on (`Reads`): the
//! program and its hints (S, H), or also the positions and names (P), as a backend
//! whose text must carry a script's names and lines does. Its cache key covers exactly
//! those parts, so two files with one key give one output.

pub mod cpu;
pub mod emit;

use std::collections::BTreeSet;

use std::path::PathBuf;

use crate::bytecode::hash::{hashes, hex, sha256, Hashes};
use crate::program::derive::Cause;
use crate::program::notes::Anchor;
use crate::program::Module;
use crate::types::{Ability, Entry, Format, ScalarType, TileShape, Type};

/// A backend refuses a requirement it does not provide (by name, with the construct
/// that needs it).
pub const NOT_PROVIDED: &str = "K960";
/// A backend has no implementation of an external function without a fallback.
pub const NO_IMPLEMENTATION: &str = "K961";
/// A program exceeds one of the backend's limits.
pub const OVER_LIMIT: &str = "K962";
/// A backend does not compute a format the module declares.
pub const FORMAT_NOT_COMPUTED: &str = "K963";
/// A construct the backend does not emit yet.
pub const NOT_BUILT: &str = "K964";
/// A backend does not compute on the module's tile shape.
pub const TILE_NOT_COMPUTED: &str = "K965";

/// The machine a compile is for: only what the output depends on, so that two machines
/// that differ in nothing else share one output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    /// The architecture: `x86_64`, `aarch64`, `riscv64`; `blackhole`, `wormhole_b0`.
    pub arch: String,
    /// The instruction sets the output may use (`avx2`, `fma`); none means portable.
    pub features: BTreeSet<String>,
}

impl Target {
    /// The target as the cache key covers it: the architecture, a space, the features
    /// in order joined by commas.
    pub fn canonical(&self) -> String {
        let features: Vec<&str> = self.features.iter().map(String::as_str).collect();
        format!("{} {}", self.arch, features.join(","))
    }
}

/// What a machine is at load, beyond its target: facts the output reads when it is
/// loaded and never at compile.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Machine {
    /// Its workers.
    pub workers: u64,
    /// The bytes of one worker's memory.
    pub worker_memory: u64,
    /// Its islands.
    pub islands: u64,
    /// The load's named options (`depth`, `max_live`), as data: what each means is the
    /// backend's.
    pub options: Vec<(String, u64)>,
}

/// An ability beyond the base six that a backend provides on a target. Its amounts, if
/// any, are the machine's (`LoadStatement`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provided {
    /// `group = grid`.
    Grid,
    /// `dynamic`.
    Dynamic,
    /// `islands >= n`: more than one island.
    Islands,
    /// `local_state >= ...`: state that lasts between invocations.
    LocalState,
}

/// A limit a program's text can exceed, known at compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// The handlers of one program.
    HandlersPerProgram,
    /// The parameters of one handler, run or chain.
    Parameters,
    /// The argument words of one handler, run or chain: a tensor takes the statement's
    /// `tensor_words`, a scalar of 64 bits two, every other parameter one.
    ArgumentWords,
    /// The fields of one message type.
    MessageFields,
}

impl Limit {
    fn words(self) -> &'static str {
        match self {
            Limit::HandlersPerProgram => "handlers in a program",
            Limit::Parameters => "parameters of a handler, run or chain",
            Limit::ArgumentWords => "argument words of a handler, run or chain",
            Limit::MessageFields => "fields of a message",
        }
    }
}

/// What a backend states about itself on a target: what is known at compile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    /// The backend's name.
    pub backend: &'static str,
    /// The abilities beyond the base six it provides.
    pub abilities: Vec<Provided>,
    /// The implementations of external functions it has, by `impl("name")`.
    pub implementations: Vec<String>,
    /// Its limits on a program's text; a limit not listed is none.
    pub limits: Vec<(Limit, u64)>,
    /// The formats it computes. How it computes each is its own document.
    pub formats: Vec<Format>,
    /// Where that document is: a page of the docs.
    pub formats_document: &'static str,
    /// The tile shapes it computes on; none listed means any.
    pub tile_shapes: Vec<TileShape>,
    /// The argument words one tensor parameter takes (`Limit::ArgumentWords`).
    pub tensor_words: u64,
    /// The hints it reads, by dotted name (`cpu.vector`). Every other hint is ignored.
    pub hints: Vec<&'static str>,
    /// Its forward-progress guarantees, as sentences.
    pub progress: Vec<&'static str>,
}

/// What a backend states about itself on one machine at load: the amounts of its
/// abilities and the limits only the machine fixes. An amount not given is none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadStatement {
    /// The workers a group may take.
    pub workers: Option<u64>,
    /// The islands: what `islands >= n` is compared with.
    pub islands: Option<u64>,
    /// The bytes of one worker's memory that state may take.
    pub local_state_bytes: Option<u64>,
    /// The invocations in flight at once.
    pub in_flight: Option<u64>,
    /// The messages a dynamic run may have live.
    pub live_messages: Option<u64>,
}

/// A refusal by a backend: the code, what, and the construct it is about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// One of the codes above.
    pub code: &'static str,
    /// What is refused, in words.
    pub what: String,
    /// The construct, if one is to blame; `notes.spans` places it in the text.
    pub at: Option<Anchor>,
}

/// How an entry's argument is passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgKind {
    /// A tensor of this format.
    Tensor(Format),
    /// A scalar of this type.
    Scalar(ScalarType),
    /// A boolean.
    Bool,
}

impl ArgKind {
    /// The kind of a parameter of this type, if an invocation can pass one.
    pub fn of(ty: &Type) -> Option<ArgKind> {
        match ty {
            Type::Tensor(f, _) => Some(ArgKind::Tensor(*f)),
            Type::Scalar(s) => Some(ArgKind::Scalar(*s)),
            Type::Bool => Some(ArgKind::Bool),
            _ => None,
        }
    }
}

/// One word of an invocation's arguments, by the parameter it comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgWord {
    /// The address of the tensor passed as parameter `p`.
    TensorAddress(usize),
    /// The tile count of the tensor passed as parameter `p`.
    TensorTiles(usize),
    /// The value of parameter `p`: a scalar's bits (a float as its bits), a boolean as
    /// 0 or 1.
    Word(usize),
}

/// A number computed from an invocation's argument words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// Argument word `i` (an index into `EntryPoint.words`).
    Word(usize),
    /// A constant.
    Const(u64),
    /// A sum.
    Add(Box<Expr>, Box<Expr>),
}

/// What a host checks before it sends an invocation: tiles `first` to
/// `first + count - 1` of the tensor passed as parameter `tensor` lie in it. One that
/// does not ends the invocation failed (`view_end`) without sending it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bound {
    /// The tensor's parameter.
    pub tensor: usize,
    /// The first tile.
    pub first: Expr,
    /// The tiles.
    pub count: Expr,
}

/// One thing of the module an invocation names, in the output: the invocation ABI, the
/// same on every backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryPoint {
    /// `Program.handler`.
    pub name: String,
    /// What it is.
    pub entry: Entry,
    /// Its number in an invocation's command.
    pub number: u32,
    /// Its symbol in the output, where the backend has symbols.
    pub symbol: String,
    /// Its parameters, in order.
    pub params: Vec<ArgKind>,
    /// Its argument words, in word order.
    pub words: Vec<ArgWord>,
    /// What a host checks before sending it; none on a backend that checks itself.
    pub bounds: Vec<Bound>,
}

/// One step of building an output: a program and its arguments, run in the directory
/// the output's files are written to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    /// The program.
    pub program: String,
    /// Its arguments.
    pub args: Vec<String>,
}

/// The processor of a core a kernel runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Processor {
    /// The one that reads into the core.
    Reader,
    /// The one that writes out of it.
    Writer,
    /// The one that computes.
    Compute,
}

/// A compile-time argument of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileArg {
    /// A word known at compile.
    Word(u32),
    /// The layout of the tensor passed as parameter `param`, filled at load.
    TensorLayout {
        /// The parameter.
        param: usize,
    },
}

/// A kernel for the machine's runtime to compile at load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kernel {
    /// Its source: a path in the output's files.
    pub source: String,
    /// The processor it runs on.
    pub processor: Processor,
    /// Its defines, by name.
    pub defines: Vec<(String, String)>,
    /// Its compile-time arguments, in order.
    pub args: Vec<CompileArg>,
    /// Its compile options, by name.
    pub options: Vec<(String, String)>,
}

/// How an output is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Build {
    /// Programs run in order in the directory the output's files are written to.
    Steps(Vec<Step>),
    /// Kernels the machine's runtime compiles at load.
    Kernels(Vec<Kernel>),
}

/// A backend's output: files, the entries, and how to build them; nothing is run and
/// nothing written to make it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    /// Its key.
    pub key: CacheKey,
    /// Each file: its path relative to the build directory, and its bytes.
    pub files: Vec<(String, Vec<u8>)>,
    /// The entries, in the module's order.
    pub entries: Vec<EntryPoint>,
    /// The build.
    pub build: Build,
}

/// What a build made, as `statement_at` is given it: the directory, and the facts the
/// backend measured of it by name (a kernel's bytes).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Built {
    /// The directory that holds what was built.
    pub dir: PathBuf,
    /// Its facts, by name.
    pub facts: Vec<(String, u64)>,
}

/// The parts of a byte-code file a backend's output depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reads {
    /// The program and its hints (S, H): the build hash.
    ProgramAndHints,
    /// Also the positions and names (P): the full hash.
    Everything,
}

impl Reads {
    /// The hash that covers these parts.
    pub fn hash(self, h: &Hashes) -> [u8; 32] {
        match self {
            Reads::ProgramAndHints => h.build,
            Reads::Everything => h.full,
        }
    }
}

/// The key of a backend's output files in a cache: SHA-256 over a tag, the hash of the
/// parts of the module the backend reads (`Reads`), the backend's name and version,
/// and the target. What a build makes of the files is the build's own cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey(pub [u8; 32]);

impl CacheKey {
    /// The key of `backend`'s output for `module` on `target`.
    pub fn for_backend(module: &Module, backend: &dyn Backend, target: &Target) -> CacheKey {
        let hash = backend.reads().hash(&hashes(module));
        CacheKey::of(&hash, backend.name(), &backend.version(), target)
    }

    /// The key over `hash` (the build hash or the full hash, as the backend reads),
    /// from a backend of this name and version on `target`.
    pub fn of(hash: &[u8; 32], backend: &str, version: &str, target: &Target) -> CacheKey {
        let mut bytes = b"komorebi-ir backend\0".to_vec();
        bytes.extend_from_slice(hash);
        for part in [backend, version, &target.canonical()] {
            bytes.extend_from_slice(part.as_bytes());
            bytes.push(0);
        }
        CacheKey(sha256(&bytes))
    }

    /// The key as 64 lowercase hex digits.
    pub fn hex(&self) -> String {
        hex(&self.0)
    }
}

/// A backend: its statement, its check and its compile.
pub trait Backend {
    /// Its name: `cpu`, `tensix`.
    fn name(&self) -> &'static str;
    /// Its version: part of every cache key, so a change to what it emits changes it.
    fn version(&self) -> String;
    /// The parts of a byte-code file its output depends on; its key covers them.
    fn reads(&self) -> Reads;
    /// What it states on a target, known at compile.
    fn statement(&self, target: &Target) -> Statement;
    /// What it states on a machine at load, after the build, given what was built.
    fn statement_at(&self, target: &Target, machine: &Machine, built: &Built) -> LoadStatement;
    /// The bytes a tile of this format and shape takes in its worker memory: how it
    /// stores a tile is its own (a format is a notation).
    fn tile_bytes(&self, format: Format, shape: TileShape) -> u64;
    /// Every refusal of `program` on `target`: the shared check (`check_statement`) and
    /// the backend's own (what it does not emit yet).
    fn check(&self, program: &Module, target: &Target) -> Vec<Refusal>;
    /// Its output for `program` on `target`, or why not.
    fn compile(&self, program: &Module, target: &Target) -> Result<Output, Vec<Refusal>>;
}

/// The construct to blame for a requirement: the first cause that is a construct, else
/// the module (the `requires` block states it).
fn requirement_anchor(causes: &[Cause]) -> Anchor {
    for cause in causes {
        match cause {
            Cause::Stated => {}
            Cause::GridHandler(p, h) => return Anchor::Handler(*p, *h),
            Cause::DynamicRun(Entry::Handler(p, h)) => return Anchor::Handler(*p, *h),
            Cause::DynamicRun(Entry::Run(p, r)) => return Anchor::Run(*p, *r),
            Cause::DynamicRun(Entry::Chain(p, c)) => return Anchor::Chain(*p, *c),
            Cause::ExternWithoutFallback(f) => return Anchor::Func(*f),
        }
    }
    Anchor::Module
}

fn ability_text(a: &Ability) -> String {
    match a {
        Ability::LocalState { amount, tiles } => {
            format!(
                "local_state >= {}({amount})",
                if *tiles { "tiles" } else { "bytes" }
            )
        }
        Ability::Grid => "group = grid".into(),
        Ability::Dynamic => "dynamic".into(),
        Ability::Islands(n) => format!("islands >= {n}"),
        Ability::Impl(name) => format!("impl(\"{name}\")"),
    }
}

/// The shared half of a backend's check at compile: every requirement the statement
/// does not provide (`K960`), every implementation it lacks (`K961`), every limit the
/// program's text exceeds (`K962`), every declared format it does not compute (`K963`),
/// and a tile shape it does not compute on (`K965`).
pub fn check_statement(module: &Module, s: &Statement) -> Vec<Refusal> {
    let mut out = Vec::new();
    for r in &module.requirements {
        let at = Some(requirement_anchor(&r.causes));
        let provided = match &r.ability {
            Ability::Grid => s.abilities.contains(&Provided::Grid),
            Ability::Dynamic => s.abilities.contains(&Provided::Dynamic),
            Ability::Islands(_) => s.abilities.contains(&Provided::Islands),
            Ability::LocalState { .. } => s.abilities.contains(&Provided::LocalState),
            Ability::Impl(name) => {
                if !s.implementations.contains(name) {
                    out.push(Refusal {
                        code: NO_IMPLEMENTATION,
                        what: format!(
                            "the {} backend has no implementation `{name}`, and the external \
                             function has no fallback",
                            s.backend
                        ),
                        at,
                    });
                }
                continue;
            }
        };
        if !provided {
            out.push(Refusal {
                code: NOT_PROVIDED,
                what: format!(
                    "the {} backend does not provide `{}`",
                    s.backend,
                    ability_text(&r.ability)
                ),
                at,
            });
        }
    }
    let words = |list: &[Type]| -> usize {
        list.iter()
            .map(|t| match t {
                Type::Tensor(..) => s.tensor_words as usize,
                Type::Scalar(sc) if sc.bits() > 32 => 2,
                _ => 1,
            })
            .sum()
    };
    let limit = |l: Limit| s.limits.iter().find(|(k, _)| *k == l).map(|(_, n)| *n);
    let over = |l: Limit, count: usize, at: Anchor, out: &mut Vec<Refusal>| {
        if let Some(n) = limit(l) {
            if count as u64 > n {
                out.push(Refusal {
                    code: OVER_LIMIT,
                    what: format!(
                        "{count} {}, and the {} backend allows {n}",
                        l.words(),
                        s.backend
                    ),
                    at: Some(at),
                });
            }
        }
    };
    for (p, program) in module.programs.iter().enumerate() {
        let pid = crate::types::ProgramId(p as u32);
        over(
            Limit::HandlersPerProgram,
            program.handlers.len(),
            Anchor::Program(pid),
            &mut out,
        );
        let params = |list: &[Type], at: Anchor, out: &mut Vec<Refusal>| {
            over(Limit::Parameters, list.len(), at, out);
            over(Limit::ArgumentWords, words(list), at, out);
        };
        for (h, handler) in program.handlers.iter().enumerate() {
            let at = Anchor::Handler(pid, crate::types::HandlerId(h as u32));
            params(&handler.params, at, &mut out);
        }
        for (r, run) in program.runs.iter().enumerate() {
            let at = Anchor::Run(pid, crate::types::RunId(r as u32));
            params(&run.params, at, &mut out);
        }
        for (c, chain) in program.chains.iter().enumerate() {
            let at = Anchor::Chain(pid, crate::types::ChainId(c as u32));
            params(&chain.params, at, &mut out);
        }
    }
    for (m, message) in module.messages.iter().enumerate() {
        let at = Anchor::Message(crate::types::MessageId(m as u32));
        over(Limit::MessageFields, message.fields.len(), at, &mut out);
    }
    if !s.tile_shapes.is_empty() && !s.tile_shapes.contains(&module.tile) {
        out.push(Refusal {
            code: TILE_NOT_COMPUTED,
            what: format!(
                "the {} backend does not compute on tiles of {} by {}",
                s.backend, module.tile.rows, module.tile.cols
            ),
            at: Some(Anchor::Module),
        });
    }
    for f in &module.formats {
        if !s.formats.contains(f) {
            out.push(Refusal {
                code: FORMAT_NOT_COMPUTED,
                what: format!("the {} backend does not compute the format {f}", s.backend),
                at: Some(Anchor::Module),
            });
        }
    }
    out
}

/// The shared half of a backend's check at load, on one machine: `islands >= n` with
/// the islands it has, and `local_state >= ...` with the bytes state may take, where
/// `tiles(n)` is n tiles of the module's tile in the widest of its formats as the
/// backend stores them (`K960`).
pub fn check_load(module: &Module, backend: &dyn Backend, at_load: &LoadStatement) -> Vec<Refusal> {
    let mut out = Vec::new();
    for r in &module.requirements {
        let at = Some(requirement_anchor(&r.causes));
        let (need, have, unit) = match r.ability {
            Ability::Islands(n) => (Some(n), at_load.islands, "islands"),
            Ability::LocalState { amount, tiles } => {
                let need = if tiles {
                    let widest = module
                        .formats
                        .iter()
                        .map(|f| backend.tile_bytes(*f, module.tile))
                        .max()
                        .unwrap_or(0);
                    amount.checked_mul(widest)
                } else {
                    Some(amount)
                };
                (need, at_load.local_state_bytes, "bytes of worker memory")
            }
            _ => continue,
        };
        let enough = match (need, have) {
            (Some(need), Some(have)) => need <= have,
            (_, None) => true,
            (None, Some(_)) => false,
        };
        if !enough {
            out.push(Refusal {
                code: NOT_PROVIDED,
                what: format!(
                    "`{}` on a machine where the {} backend has {} {unit}",
                    ability_text(&r.ability),
                    backend.name(),
                    have.unwrap_or(0)
                ),
                at,
            });
        }
    }
    out
}
