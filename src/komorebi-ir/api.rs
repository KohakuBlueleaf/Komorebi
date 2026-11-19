//! The core's entry points. Each runs its work on the stack the depth limit needs
//! (`crate::deep`), whatever thread calls it.

use crate::deep::on_deep_stack;
use crate::errors::Diagnostic;
use crate::syntax::ast::Module;
use crate::syntax::{parser, writer};

/// Parses the text of a module into its syntax tree, or gives the diagnostics that
/// say why the text is not a module.
pub fn parse(source: &str) -> Result<Module, Vec<Diagnostic>> {
    on_deep_stack(|| parser::parse_module(source))
}

/// The canonical text of a syntax tree.
pub fn write(module: &Module) -> String {
    on_deep_stack(|| writer::write_module(module))
}

/// The canonical text of a module's text, or the diagnostics of a text that is not one.
pub fn format(source: &str) -> Result<String, Vec<Diagnostic>> {
    on_deep_stack(|| parser::parse_module(source).map(|module| writer::write_module(&module)))
}
