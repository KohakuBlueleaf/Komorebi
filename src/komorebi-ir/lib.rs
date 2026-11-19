//! Komorebi IR: the semantic layer between the Komorebi programming model and the
//! machines that run it. This crate holds its text form (`.kom`): the syntax tree,
//! the parser and the canonical writer; its types; and program compile's checks.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod api;
pub mod check;
pub mod deep;
pub mod derive;
pub mod errors;
pub mod interp;
pub mod lower;
pub mod program;
pub mod span;
pub mod syntax;
pub mod types;
pub mod verify;

pub use api::{check, format, parse, write};
pub use errors::{Diagnostic, Severity};
