//! The types of the IR: element formats, scalars, tiles and blocks of tiles, tensors,
//! views, records and messages, and the values the operations of a body yield.

pub mod format;
pub mod ty;

pub use format::Format;
pub use ty::{Axis, Count, Ty};
