//! What the semantic hash leaves out: hints, source positions, and the names a body
//! gives its values and slots. Each is attached by an `Anchor` to what it describes.

use super::body::{InstId, LocalId, ValueId};
use super::BodyRef;
use crate::types::{
    ChainId, ControlId, FuncId, HandlerId, MemberId, MessageId, ParamId, ProgramId, RecordId, RunId,
};

/// Where a note is attached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Anchor {
    /// The module.
    Module,
    /// A function.
    Func(FuncId),
    /// A program.
    Program(ProgramId),
    /// A handler.
    Handler(ProgramId, HandlerId),
    /// A run.
    Run(ProgramId, RunId),
    /// A state member.
    Member(ProgramId, MemberId),
    /// A control program.
    Control(ControlId),
    /// A chain.
    Chain(ProgramId, ChainId),
    /// A module parameter.
    Param(ParamId),
    /// A message type.
    Message(MessageId),
    /// A record type.
    Record(RecordId),
    /// A parameter of a function, handler, run, chain or control (its owner, then
    /// its position).
    ParamOf(Owner, u32),
    /// An instruction of a body.
    Inst(BodyRef, InstId),
}

/// The owner of parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Owner {
    /// A function.
    Func(FuncId),
    /// A handler.
    Handler(ProgramId, HandlerId),
    /// A run.
    Run(ProgramId, RunId),
    /// A chain.
    Chain(ProgramId, ChainId),
    /// A control program.
    Control(ControlId),
}

/// An argument of a hint, as written.
#[derive(Clone, Debug, PartialEq)]
pub enum HintArg {
    /// A number, as written.
    Number(String),
    /// A string.
    Text(String),
    /// A name.
    Name(String),
    /// `name = value`.
    Named(String, Box<HintArg>),
    /// `[a, b]`.
    List(Vec<HintArg>),
}

/// A hint: `@backend.name(arguments)`, attached to what it advises.
#[derive(Clone, Debug, PartialEq)]
pub struct Hint {
    /// What it advises.
    pub anchor: Anchor,
    /// Its dotted name: the namespace first.
    pub path: Vec<String>,
    /// Its arguments.
    pub args: Vec<HintArg>,
}

/// A place in the source: bytes from `start` to `end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceSpan {
    /// The first byte.
    pub start: u32,
    /// After the last byte.
    pub end: u32,
}

/// The name the source gives a value or a slot of a body.
#[derive(Clone, Debug, PartialEq)]
pub enum Named {
    /// A `let`, a loop's name, or a parameter read.
    Value(BodyRef, ValueId, String),
    /// A `var`.
    Local(BodyRef, LocalId, String),
}

/// Everything the semantic hash leaves out. Each list is in the order of its anchors
/// (`Anchor`'s order), then in the order written.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Notes {
    /// The hints.
    pub hints: Vec<Hint>,
    /// Where each anchored thing is written.
    pub spans: Vec<(Anchor, SourceSpan)>,
    /// The names of values and slots: in the order of their bodies (`BodyRef`'s
    /// order), a body's values by number, then its slots by number; one name each.
    pub names: Vec<Named>,
    /// The names of parameters (`Anchor::ParamOf`).
    pub param_names: Vec<(Anchor, String)>,
}
