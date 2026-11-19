//! The core's entry points. Each runs its work on the stack the depth limit needs
//! (`crate::deep`), whatever thread calls it.

use crate::check::{check_module, codes, Checked};
use crate::deep::on_deep_stack;
use crate::errors::Diagnostic;
use crate::span::Span;
use crate::syntax::ast::Module;
use crate::syntax::{parser, writer};
use crate::{derive, lower, verify};

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

/// Parses and checks the text of a module (program compile's checks): the checked
/// module, or every diagnostic that says why it is refused. The rules on what the
/// module does as a whole are Verify's, on the module lowered and derived; a module
/// Lower does not build yet is refused, since those rules were not checked.
pub fn check(source: &str) -> Result<Checked, Vec<Diagnostic>> {
    on_deep_stack(|| {
        let checked =
            parser::parse_module(source).and_then(|module| check_module(source, module))?;
        let mut module = match lower::lower(&checked) {
            Ok(module) => module,
            Err(unlowered) => {
                return Err(unlowered
                    .into_iter()
                    .map(|u| {
                        let message = format!(
                            "the rules on what the module does (K205, K206, K301, K601, K701) \
                             are not checked: Lower does not build this yet ({})",
                            u.what
                        );
                        Diagnostic::error(source, codes::NOT_BUILT, message, u.span)
                    })
                    .collect())
            }
        };
        derive::derive(&mut module);
        let findings = verify::verify(&module);
        if findings.is_empty() {
            return Ok(checked);
        }
        Err(findings
            .into_iter()
            .map(|f| {
                let span = Span::new(f.at.start as usize, f.at.end as usize);
                Diagnostic::error(source, f.code, f.message, span)
            })
            .collect())
    })
}
