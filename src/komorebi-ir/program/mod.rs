//! The in-memory program structure: what Lower fills from the checked tree, and what
//! Encode, the interpreter and the backends read. Being designed: no stage fills or
//! reads it yet.
//!
//! Three rules hold everywhere:
//!
//! - **Plain data.** No shared pointers and no cycles: a declaration refers to another
//!   by its index in the module's lists (`FuncId`, `ProgramId`, ...), and an
//!   operation to a value by its number in its body (`ValueId`).
//! - **One order.** Every list has one order, stated where the list is declared, so
//!   encoding and hashing are walks in that order and give one result.
//! - **No hardware**, and **no hint and no position inside the program.** Hints, source
//!   positions and the names a body gives its values stand in `Notes`, keyed by
//!   `Anchor`. The semantic hash is a walk of everything but `Module::notes`; the full
//!   hash walks `notes` too. Removing every hint empties a list in `notes` and changes
//!   nothing else, by construction.
//! - **Nothing rewrites a body after Lower and Derive.** The notes and the derived
//!   effects are keyed by instruction; a later stage that wants another form builds
//!   its own from this one.
//!
//! `validate` checks a module against these rules and the operations' types.

pub mod body;
pub mod constant;
pub mod derive;
pub mod notes;
pub mod print;
pub mod validate;

pub use print::print;
pub use validate::{validate, Violation};

use crate::types::{Format, ScalarType, TileShape, Type};
use body::Body;
use derive::{ChainDeps, Effects, Requirement};
use notes::Notes;

pub use crate::types::Count;
use constant::ConstValue;

pub use crate::types::{MessageRef, ParamRef, RecordRef};

/// A function's place in `Module::funcs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FuncId(pub u32);

/// A program's place in `Module::programs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgramId(pub u32);

/// A control program's place in `Module::controls`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ControlId(pub u32);

/// A state member's place in its program's `state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MemberId(pub u32);

/// A handler's place in its program's `handlers`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HandlerId(pub u32);

/// A run's place in its program's `runs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RunId(pub u32);

/// A chain's place in its program's `chains`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChainId(pub u32);

/// What an invocation names: a handler, a run or a chain of one program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Entry {
    /// A handler.
    Handler(ProgramId, HandlerId),
    /// A dynamic run.
    Run(ProgramId, RunId),
    /// A chain.
    Chain(ProgramId, ChainId),
}

/// A body of the module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BodyRef {
    /// A function's body, or an external function's fallback.
    Func(FuncId),
    /// A handler's body.
    Handler(ProgramId, HandlerId),
    /// A run's seed.
    Seed(ProgramId, RunId),
    /// A run's `on M` body, by position.
    On(ProgramId, RunId, u32),
    /// A control program's body.
    Control(ControlId),
}

/// The whole checked module, lowered.
///
/// Its lists are in the order of the canonical text, which is the source's order (the
/// writer never moves a declaration). A reordering of declarations is then a different
/// program for the hash; the note gives the other order (by name) and its cost.
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    /// The module's name.
    pub name: String,
    /// The module's one tile shape.
    pub tile: TileShape,
    /// The formats the module declares, in the order written.
    pub formats: Vec<Format>,
    /// The module's parameters in text order (`ParamRef` is a place in this list),
    /// build parameters included (their value is bound, and
    /// every use of one in a body is that value as a constant).
    pub params: Vec<Param>,
    /// The record types, every declaration of the module in text order, those inside
    /// programs included (`RecordRef` is a place in this list).
    pub records: Vec<Record>,
    /// The message types, every declaration of the module in text order (`MessageRef`
    /// is a place in this list).
    pub messages: Vec<Message>,
    /// The functions: `func`, `combine` and `extern func`.
    pub funcs: Vec<Func>,
    /// The programs.
    pub programs: Vec<Program>,
    /// The control programs.
    pub controls: Vec<Control>,
    /// The requirement set: every ability the module needs, each with what caused it
    /// (Derive), and whether the `requires` block states it.
    pub requirements: Vec<Requirement>,
    /// Hints, source positions and the names of values: everything the semantic hash
    /// leaves out.
    pub notes: Notes,
}

/// When a parameter is fixed.
#[derive(Clone, Debug, PartialEq)]
pub enum Binding {
    /// `@bind(build)`: bound by Lower; every use is this constant.
    Build(ConstValue),
    /// `@bind(load)` with a value, or with `fit` (`None`): fixed when the module is
    /// loaded on a machine.
    Load(Option<ConstValue>),
    /// `@bind(invoke)`: given at each invocation, at most this value (else the
    /// failure `over_bound`).
    Invoke(ConstValue),
}

/// A module parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// Its name: the service binds a load parameter by it.
    pub name: String,
    /// Its scalar type.
    pub ty: ScalarType,
    /// When it is fixed.
    pub binding: Binding,
}

/// A record type: named fields of scalar types, laid out in this order.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    /// Its name.
    pub name: String,
    /// The program it is declared in, if any.
    pub program: Option<ProgramId>,
    /// The fields, in the order declared: a name and a scalar type.
    pub fields: Vec<(String, ScalarType)>,
}

/// What kind of function.
#[derive(Clone, Debug, PartialEq)]
pub enum FuncKind {
    /// `func`: pure compute on tiles.
    Plain,
    /// `combine`: a function a reduce may group in any way; commutative when
    /// declared so.
    Combine {
        /// `@commutative`.
        commutative: bool,
    },
    /// `extern func`: resolved by a backend by its implementation name; `body` is the
    /// fallback when there is one.
    Extern {
        /// `@impl("name")`.
        implementation: String,
        /// The parameters it reads (`@reads`), by position.
        reads: Vec<u32>,
        /// The parameters it writes (`@writes`), by position.
        writes: Vec<u32>,
    },
}

/// A function: parameters, a result, and a body (absent for an external function with
/// no fallback).
#[derive(Clone, Debug, PartialEq)]
pub struct Func {
    /// Its name.
    pub name: String,
    /// What kind.
    pub kind: FuncKind,
    /// The parameter types, in order. A body reads parameter i with `Op::Arg(i)`.
    pub params: Vec<Type>,
    /// The result type.
    pub ret: Type,
    /// The body, or the fallback of an external function.
    pub body: Option<Body>,
}

/// A program: its messages, its state, and what a worker can be asked to do.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    /// Its name.
    pub name: String,
    /// Its state members, every `state` block's members in order, each with its own
    /// attributes resolved (a member's own attribute over its block's).
    pub state: Vec<Member>,
    /// Its handlers.
    pub handlers: Vec<Handler>,
    /// Its dynamic runs.
    pub runs: Vec<Run>,
    /// Its chains.
    pub chains: Vec<Chain>,
}

/// A message type: named scalar fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    /// Its name.
    pub name: String,
    /// The program it is declared in, if any.
    pub program: Option<ProgramId>,
    /// Its fields, in order: a name and a scalar type.
    pub fields: Vec<(String, ScalarType)>,
}

/// What a state member holds.
#[derive(Clone, Debug, PartialEq)]
pub enum MemberKind {
    /// `tile<F>[n]`.
    Tiles(Format),
    /// `u32[n]` and the like.
    Scalars(ScalarType),
}

/// One state member of a program, with its attributes resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    /// Its name: `read_state` reads an exported member by it.
    pub name: String,
    /// What it holds.
    pub kind: MemberKind,
    /// How many elements: a constant, or a load parameter.
    pub count: Count,
    /// `@scope`: only `worker` exists today.
    pub scope: Scope,
    /// `@lifetime`: only `program` exists today.
    pub lifetime: Lifetime,
    /// `@export`.
    pub export: bool,
    /// `@filled(element)`: each element filled on its own.
    pub by_element: bool,
}

/// A state member's scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// `@scope(worker)`: each worker has its own.
    Worker,
}

/// A state member's lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// `@lifetime(program)`: while the program is loaded.
    Program,
}

/// The group shape a handler, a run or a chain runs on.
#[derive(Clone, Debug, PartialEq)]
pub enum On {
    /// `@on(group)`: a line of ranks.
    Group,
    /// `@on(grid[R, C])`.
    Grid(u64, u64),
}

/// A handler: runs once on every worker of the group it is invoked on.
#[derive(Clone, Debug, PartialEq)]
pub struct Handler {
    /// Its name.
    pub name: String,
    /// The parameter types, in order.
    pub params: Vec<Type>,
    /// The scalar result, if any: the leader's value.
    pub ret: Option<ScalarType>,
    /// `@on`.
    pub on: On,
    /// `@exclusive`.
    pub exclusive: bool,
    /// The body.
    pub body: Body,
    /// What the body reads and writes (Derive).
    pub effects: Effects,
}

/// The peers a run declares for `offer`.
#[derive(Clone, Debug, PartialEq)]
pub enum Peers {
    /// No `@peers`.
    None,
    /// `@peers(offsets = [d, ...])`.
    Offsets(Vec<u64>),
    /// `@peers(pairs = [(s, r), ...])`.
    Pairs(Vec<(u64, u64)>),
}

/// A dynamic run: a seed body, and one body for each message type it receives.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// Its name.
    pub name: String,
    /// The parameter types, in order; every body of the run reads them with `Op::Arg`.
    /// A run is invoked on a line of ranks: the text gives it no `@on`.
    pub params: Vec<Type>,
    /// `@peers`.
    pub peers: Peers,
    /// `seed { ... }`.
    pub seed: Body,
    /// `on M(m) { ... }`, in the order written; the arrived message is the body's
    /// `Op::Arrived` value.
    pub on_message: Vec<(MessageRef, Body)>,
    /// What the bodies read and write, together (Derive).
    pub effects: Effects,
}

/// A chain: calls of the program's own handlers, in order, as one invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct Chain {
    /// Its name.
    pub name: String,
    /// The parameter types, in order.
    pub params: Vec<Type>,
    /// `@on`.
    pub on: On,
    /// The calls, in order.
    pub calls: Vec<Call>,
    /// Which call reads what an earlier call wrote (Derive).
    pub deps: ChainDeps,
}

/// One argument of a chain's call.
#[derive(Clone, Debug, PartialEq)]
pub enum CallArg {
    /// The chain's parameter at this position.
    Param(u32),
    /// `past(y)`: the chain's tensor parameter, with the proof that every block read
    /// was written by an earlier call.
    Past(u32),
    /// A constant.
    Const(ConstValue),
}

/// One call of a chain.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    /// The handler called.
    pub handler: HandlerId,
    /// Its arguments, in order.
    pub args: Vec<CallArg>,
}

/// A control program: values, control flow, invocations and their results.
#[derive(Clone, Debug, PartialEq)]
pub struct Control {
    /// Its name.
    pub name: String,
    /// The parameter types, in order.
    pub params: Vec<Type>,
    /// The scalar result, if any.
    pub ret: Option<ScalarType>,
    /// The body.
    pub body: Body,
}
