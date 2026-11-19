//! Constants as a body holds them. Every other type comes from `crate::types`.

use crate::types::ScalarType;

/// A constant.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstValue {
    /// An integer of a type, as its bits masked to the width.
    Int(ScalarType, u64),
    /// A float of a type, as written, held as an `f64` (the number the text wrote;
    /// the interpreter holds it as its format holds it).
    Float(ScalarType, f64),
    /// `true` or `false`.
    Bool(bool),
}
