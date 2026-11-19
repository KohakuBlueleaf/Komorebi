//! Komorebi IR: the semantic layer between the Komorebi programming model and the
//! machines that run it. This crate holds its text form (`.kom`): the syntax tree,
//! the parser and the canonical writer.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod api;
pub mod deep;
pub mod errors;
pub mod interp;
pub mod syntax;

pub use api::{format, parse, write};
pub use errors::{Diagnostic, Severity};
