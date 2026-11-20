//! The text form: the grammar, the syntax tree, the parser and the writer.

pub mod ast;
pub mod parser;
pub mod writer;

/// The words of the text (the grammar's `keyword`): a name that is one is written
/// between backticks, `` `store` `` (R50). A name is the name its author gave; the
/// backticks are only how the text writes it.
pub const WORDS: [&str; 15] = [
    "let", "var", "if", "else", "for", "in", "while", "return", "store", "consume", "await",
    "true", "false", "past", "fit",
];

/// The words a type is written with: a type named by one (a message or a record) is
/// written between backticks where a type is.
pub const TYPE_WORDS: [&str; 15] = [
    "tile", "tensor", "view", "bool", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16",
    "f32", "f64",
];
