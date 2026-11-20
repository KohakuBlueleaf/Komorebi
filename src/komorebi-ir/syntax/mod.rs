//! The text form: the grammar, the syntax tree, the parser and the writer.

pub mod ast;
pub mod parser;
pub mod writer;

/// The words of the text (the grammar's `keyword`): a name that is one is written
/// between backticks, `` `store` ``, wherever a name is (R50). A name is the name its
/// author gave; the backticks are only how the text writes it. A bare name that is one
/// of the language's where it stands means the language's: an item of the module so
/// named is written between backticks by its kind's set below.
pub const WORDS: [&str; 15] = [
    "let", "var", "if", "else", "for", "in", "while", "return", "store", "consume", "await",
    "true", "false", "past", "fit",
];

/// The words a type is written with: a type named by one (a message or a record) is
/// written between backticks, at its declaration and where a type is.
pub const TYPE_WORDS: [&str; 15] = [
    "tile", "tensor", "view", "bool", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16",
    "f32", "f64",
];

/// The language's calls: a bare name of one where a call stands is the language's, and
/// a function of the module so named is written between backticks (R50).
pub const CALLS: [&str; 43] = [
    "min",
    "max",
    "range",
    "where",
    "maximum",
    "minimum",
    "matmul",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "neg",
    "abs",
    "relu",
    "square",
    "floor",
    "exp",
    "exp2",
    "tanh",
    "erf",
    "log",
    "sigmoid",
    "silu",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "partition",
    "stream",
    "sink",
    "zip",
    "reduce",
    "gather",
    "broadcast",
    "sync",
    "all_reduce",
    "send",
    "forward",
    "offer",
    "invoke",
    "outcome",
    "group",
    "reduce_to",
    "broadcast_from",
    "all_gather",
];

/// The conversions, called by a scalar type's name (`u32(x)`).
pub const CONVERSIONS: [&str; 11] = [
    "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16", "f32", "f64",
];

/// The combinations a `with =` names besides a `combine` of the module.
pub const COMBINATIONS: [&str; 5] = ["sum", "max", "min", "and", "or"];

/// The language's values a bare name stands for: `zeros`, and the words a member is
/// read from (`worker.rank`, `group.size`, `island.count`).
pub const VALUES: [&str; 4] = ["zeros", "worker", "group", "island"];

/// A name a function of the module (a `func`, a `combine`, an `extern func`) is written
/// between backticks with: where a function stands, a call or a `with =`, the bare
/// name is the language's.
pub fn function_word(name: &str) -> bool {
    CALLS.contains(&name) || CONVERSIONS.contains(&name) || COMBINATIONS.contains(&name)
}

/// A name a value of the module (a local, a parameter, a loop's name, a state member, a
/// module constant) is written between backticks with.
pub fn value_word(name: &str) -> bool {
    VALUES.contains(&name)
}

/// A name a type of the module (a record, a message) is written between backticks
/// with.
pub fn type_word(name: &str) -> bool {
    TYPE_WORDS.contains(&name)
}
