//! The words of an operation that the checker and the program structure both name: the
//! operators, the reductions, how a group is split, and what a collective combines with.

use super::FuncId;

/// A binary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `+`.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/`.
    Div,
    /// `%`.
    Rem,
    /// `&`.
    And,
    /// `|`.
    Or,
    /// `^`.
    Xor,
    /// `<<`.
    Shl,
    /// `>>`.
    Shr,
    /// `min(a, b)`; `minimum(a, b)` on tiles.
    Min,
    /// `max(a, b)`; `maximum(a, b)` on tiles.
    Max,
}

/// A comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    /// `>`.
    Gt,
    /// `<`.
    Lt,
    /// `>=`.
    Ge,
    /// `<=`.
    Le,
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
}

/// A function of one tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// `neg`, and the unary `-` (also on integers).
    Neg,
    /// `abs`.
    Abs,
    /// `relu`.
    Relu,
    /// `square`.
    Square,
    /// `floor`.
    Floor,
    /// `exp`.
    Exp,
    /// `exp2`.
    Exp2,
    /// `tanh`.
    Tanh,
    /// `erf`.
    Erf,
    /// `log`.
    Log,
    /// `sigmoid`.
    Sigmoid,
    /// `silu`.
    Silu,
    /// `sqrt`.
    Sqrt,
    /// `rsqrt`.
    Rsqrt,
    /// `reciprocal`.
    Reciprocal,
}

/// A reduction along an axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// `reduce_sum`.
    Sum,
    /// `reduce_max`.
    Max,
    /// `reduce_min`.
    Min,
}

/// How a group is split: `parts = n` or `size = k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitBy {
    /// `parts = n`.
    Parts,
    /// `size = k`.
    Size,
}

/// What a collective combines with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combiner {
    /// A `combine` function of the module.
    Func(FuncId),
    /// `sum`.
    Sum,
    /// `max`.
    Max,
    /// `min`.
    Min,
    /// `and`.
    And,
    /// `or`.
    Or,
}
