//! The IR's one vocabulary of types, shared by every step: the element formats, the
//! scalar types, the names of the failures, the tile shape, the type of a value, and
//! what a declaration states beside its types (constants, `@bind`, `@on`, `@peers`,
//! the abilities of `requires`), the places of declarations, and the words of an
//! operation that more than one step names.
//! It reads nothing else of the crate.

pub mod decl;
pub mod failure;
pub mod format;
pub mod op;
pub mod place;
pub mod scalar;
pub mod value;

pub use decl::{Ability, Binding, ConstValue, On, Peers};
pub use failure::Failure;
pub use format::Format;
pub use op::{BinOp, CmpOp, Combiner, ReduceOp, SplitBy, UnaryOp};
pub use place::{
    ChainId, ControlId, Entry, FuncId, HandlerId, MemberId, MessageId, ParamId, ProgramId,
    RecordId, RunId,
};
pub use scalar::ScalarType;
pub use value::{Axis, Count, TileShape, Type};
