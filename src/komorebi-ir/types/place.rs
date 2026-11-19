//! The places of the module's declarations: each declaration is named by its place in
//! its list, in text order, from 0. The checker, Lower, Derive and every later step
//! name a declaration by these.

/// A module parameter (`param`), by its place among the module's parameters in text
/// order, from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId(pub u32);

/// A record type, by its place among the module's record declarations in text order
/// (those inside programs included), from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RecordId(pub u32);

/// A message type, by its place among the module's message declarations in text
/// order (those inside programs included), from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MessageId(pub u32);

/// A function (`func`, `combine` or `extern func`), by its place among the module's
/// functions in text order, from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FuncId(pub u32);

/// A program, by its place among the module's programs in text order, from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgramId(pub u32);

/// A control program, by its place among the module's control programs in text order,
/// from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ControlId(pub u32);

/// A state member, by its place among its program's members (every `state` block's, in
/// order), from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MemberId(pub u32);

/// A handler, by its place among its program's handlers, from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HandlerId(pub u32);

/// A dynamic run, by its place among its program's runs, from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RunId(pub u32);

/// A chain, by its place among its program's chains, from 0.
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
