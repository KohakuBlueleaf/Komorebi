//! What the semantic hash leaves out: hints, source positions, the names a body gives
//! its values and slots, and the origin of a frontend's module. Each is attached by an `Anchor` to what it describes.

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
    /// A `call` of a chain, by its position among the chain's calls.
    Call(ProgramId, ChainId, u32),
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

/// Where a module made by a frontend came from: its source files, the lines an anchor
/// covers, and each anchored thing's place. A module read from `.kom` text has none: its
/// places are `spans`. A backend that prints the origin into its output (as comments, as
/// today's DSL does) prints it by its own rule from these facts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Origin {
    /// The source files, as the frontend resolved them; a place names one by its index.
    pub files: Vec<String>,
    /// The lines a place covers: (file, line number from 1, the line as written, its
    /// indentation and comment kept); in (file, number) order, each once.
    pub lines: Vec<(u32, u32, String)>,
    /// Each anchored thing's place: (anchor, file, first line, last line), in the order
    /// of their anchors, each anchor once. A handler's place is its `def` line; an
    /// instruction's is the statement it was made from, which several may share.
    pub places: Vec<(Anchor, u32, u32, u32)>,
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
    /// Where a frontend's module came from; none for a module read from `.kom` text.
    pub origin: Option<Origin>,
}
