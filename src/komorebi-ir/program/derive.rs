//! What Derive adds: a body's effects, a chain's dependency relation, and the module's
//! requirement set, each with the construct that caused it.

use super::body::InstId;
use super::BodyRef;
use crate::types::{Ability, Entry, FuncId, HandlerId, MemberId, ProgramId};

/// How a body touches a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// Read through a `stream`.
    Stream,
    /// Read by `v[i]`, `.words()` or `.records(...)`.
    Read,
    /// Written through a `sink`.
    Sink,
    /// Written by `store`.
    Store,
    /// Written by `items.put`.
    Put,
    /// Read by an external function (`@reads`).
    ExternRead,
    /// Written by an external function (`@writes`).
    ExternWrite,
}

/// One access of a body to a tensor: which tensor parameter, through which view, and
/// how. The view is the value an instruction of the body made, so its tiles are known
/// when the arguments are (at the invocation), worker by worker.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorAccess {
    /// The tensor parameter, by position.
    pub param: u32,
    /// The body (a run has several).
    pub body: BodyRef,
    /// The instruction that made the view.
    pub view: InstId,
    /// How.
    pub access: Access,
    /// The instruction that reads or writes through it.
    pub at: InstId,
}

/// How a body touches a state member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateUse {
    /// A read: `s[i]`, `bump`, `put` from it.
    Read,
    /// A write of one element: `s[i] = x` (fills it only `@filled(element)`).
    WriteElement,
    /// Filled whole: `.fill` or `.zero()`.
    FillWhole,
}

/// One use of a state member.
#[derive(Clone, Debug, PartialEq)]
pub struct StateAccess {
    /// The member.
    pub member: MemberId,
    /// The body.
    pub body: BodyRef,
    /// How.
    pub how: StateUse,
    /// The instruction.
    pub at: InstId,
}

/// The effects of a handler or a run (all its bodies): every tensor access and every
/// state use, in the order of the instructions, and the collectives it takes part in.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Effects {
    /// The tensor accesses.
    pub tensors: Vec<TensorAccess>,
    /// The state uses.
    pub state: Vec<StateAccess>,
    /// The instructions that are collectives (each makes an edge between the
    /// group's members).
    pub collectives: Vec<(BodyRef, InstId)>,
    /// The functions it calls, directly or through another function.
    pub calls: Vec<FuncId>,
}

/// One dependency of a chain: a call reads what an earlier call wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainEdge {
    /// The earlier call, by position.
    pub writer: u32,
    /// The later call.
    pub reader: u32,
    /// The chain's tensor parameter they share.
    pub param: u32,
    /// The writer's access (in its handler's effects), by position there.
    pub write: u32,
    /// The reader's access, by position there.
    pub read: u32,
    /// Whether the reader names it with `past`, so the build proved that every block
    /// read was written before.
    pub past: bool,
}

/// A chain's dependency relation: which call reads what which earlier call wrote.
/// The relation is between accesses; which tiles each worker reads is known when the
/// chain's arguments are.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChainDeps {
    /// The edges, by reader, then by writer.
    pub edges: Vec<ChainEdge>,
}

/// What caused a requirement.
#[derive(Clone, Debug, PartialEq)]
pub enum Cause {
    /// The `requires` block states it.
    Stated,
    /// A handler on a grid.
    GridHandler(ProgramId, HandlerId),
    /// A dynamic run.
    DynamicRun(Entry),
    /// An external function with no fallback.
    ExternWithoutFallback(FuncId),
}

/// One requirement and what caused it.
#[derive(Clone, Debug, PartialEq)]
pub struct Requirement {
    /// The ability.
    pub ability: Ability,
    /// Everything that needs it, in the module's order; `Cause::Stated` first when
    /// the `requires` block has the line.
    pub causes: Vec<Cause>,
}
