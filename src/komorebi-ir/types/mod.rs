//! The IR's one vocabulary of types, shared by every step: the element formats, the
//! scalar types, the names of the failures, the tile shape, and the type of a value.
//! It reads nothing else of the crate.

pub mod failure;
pub mod format;
pub mod scalar;
pub mod value;

pub use failure::Failure;
pub use format::Format;
pub use scalar::ScalarType;
pub use value::{Axis, Count, MessageRef, ParamRef, RecordRef, TileShape, Type};
