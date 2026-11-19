//! The core's entry points. Each runs its work on the stack the depth limit needs
//! (`crate::deep`), whatever thread calls it.

use crate::bytecode::Refusal;
use crate::check::{check_module, codes, Checked};
use crate::deep::on_deep_stack;
use crate::errors::Diagnostic;
use crate::span::Span;
use crate::syntax::ast::Module;
use crate::syntax::{parser, writer};
use crate::{bytecode, derive, lower, program, verify};

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
    on_deep_stack(|| front(source).map(|(checked, _)| checked))
}

/// The program structure of a module's text, checked, lowered, derived and verified:
/// what `compile` encodes; or the diagnostics `check` gives.
pub fn build(source: &str) -> Result<program::Module, Vec<Diagnostic>> {
    on_deep_stack(|| front(source).map(|(_, module)| module))
}

/// The byte-code program of a module's text, or the diagnostics `check` gives.
pub fn compile(source: &str) -> Result<Vec<u8>, Vec<Diagnostic>> {
    on_deep_stack(|| front(source).map(|(_, module)| bytecode::encode(&module)))
}

/// The program structure a byte-code program holds, accepted only where its text would
/// check: read and validated (`bytecode::decode`), its derived parts the ones Derive
/// gives again (else `K955`), and Verify's rules kept (else their findings, with the
/// source bytes they are about).
pub fn load(bytes: &[u8]) -> Result<program::Module, Vec<Refusal>> {
    on_deep_stack(|| {
        let module = bytecode::decode(bytes).map_err(|r| vec![r])?;
        let mut again = module.clone();
        derive::derive(&mut again);
        if let Some(what) = derived_difference(&module, &again) {
            return Err(vec![Refusal {
                code: bytecode::version::DERIVED,
                what: format!("{what} differ from what Derive gives the module"),
                at: None,
            }]);
        }
        let findings = verify::verify(&module);
        if findings.is_empty() {
            return Ok(module);
        }
        Err(findings
            .into_iter()
            .map(|f| Refusal {
                code: f.code,
                what: format!(
                    "{} (source bytes {} to {})",
                    f.message, f.at.start, f.at.end
                ),
                at: None,
            })
            .collect())
    })
}

/// The first derived part in which two modules differ, named; `None` when none does.
fn derived_difference(read: &program::Module, again: &program::Module) -> Option<String> {
    if read.requirements != again.requirements {
        return Some("the requirements".into());
    }
    for (f, g) in read.funcs.iter().zip(&again.funcs) {
        if f.effects != g.effects {
            return Some(format!("the effects of the function {}", f.name));
        }
    }
    for (p, q) in read.programs.iter().zip(&again.programs) {
        for (h, k) in p.handlers.iter().zip(&q.handlers) {
            if h.effects != k.effects {
                return Some(format!("the effects of the handler {}.{}", p.name, h.name));
            }
        }
        for (r, s) in p.runs.iter().zip(&q.runs) {
            if r.effects != s.effects {
                return Some(format!("the effects of the run {}.{}", p.name, r.name));
            }
        }
        for (c, d) in p.chains.iter().zip(&q.chains) {
            if c.deps != d.deps {
                return Some(format!(
                    "the dependencies of the chain {}.{}",
                    p.name, c.name
                ));
            }
        }
    }
    // Derive writes nothing else; any other difference is still one.
    (read != again).then(|| "the derived parts".into())
}

/// The whole front: parse, check, lower, derive, verify.
fn front(source: &str) -> Result<(Checked, program::Module), Vec<Diagnostic>> {
    let checked = parser::parse_module(source).and_then(|module| check_module(source, module))?;
    let mut module = match lower::lower(&checked) {
        Ok(module) => module,
        Err(unlowered) => {
            return Err(unlowered
                .into_iter()
                .map(|u| {
                    let message = format!(
                        "the rules on what the module does (K205, K206, K301, K405, K406, \
                         K601, K701) are not checked: Lower does not build this yet ({})",
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
        return Ok((checked, module));
    }
    Err(findings
        .into_iter()
        .map(|f| {
            let span = Span::new(f.at.start as usize, f.at.end as usize);
            Diagnostic::error(source, f.code, f.message, span)
        })
        .collect())
}
