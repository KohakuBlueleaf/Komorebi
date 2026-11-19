//! The reference interpreter. So far its ground: the values (`memory`) and the
//! language's operations on them (`ops`), with no knowledge of the syntax tree.

pub mod memory;
pub mod ops;
