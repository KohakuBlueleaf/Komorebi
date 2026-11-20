//! The Python face of the core: the extension module `komorebi_ir._rs` (PyO3), which
//! the package `komorebi_ir` (src/komorebi_ir/) wraps. Built only with the feature
//! `python`. Every function calls `api`; nothing here checks, runs or encodes.
//!
//! The crate forbids unsafe code, with this module too: the calls into CPython are
//! PyO3's own (the crates `pyo3` and `pyo3-ffi`), and the code its macros expand to here
//! compiles under `#![forbid(unsafe_code)]`. Nothing here is `unsafe`.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use crate::api;
use crate::bytecode::{self, Refusal};
use crate::deep::on_deep_stack;
use crate::errors::{Diagnostic, Severity};
use crate::program::derive::Cause;
use crate::program::Module;
use crate::types::{Ability, Entry};

create_exception!(
    _rs,
    IrError,
    PyException,
    "A module, a byte-code file or a run script refused; `diagnostics` says why."
);

/// One diagnostic as data: (code, message, line, column, severity, byte). A text's
/// diagnostic has a line and a column; a byte-code file's refusal has the byte.
type Row = (
    String,
    String,
    Option<usize>,
    Option<usize>,
    &'static str,
    Option<u64>,
);

fn severity(s: Severity) -> &'static str {
    match s {
        Severity::Error => "error",
        Severity::Warning => "warning",
    }
}

fn of_text(d: &Diagnostic) -> Row {
    (
        d.code.to_string(),
        d.message.clone(),
        Some(d.start.line),
        Some(d.start.col),
        severity(d.severity),
        None,
    )
}

fn of_file(r: &Refusal) -> Row {
    (
        r.code.to_string(),
        r.what.clone(),
        None,
        None,
        "error",
        r.at,
    )
}

/// The `IrError` of `rows`, its message the first row's.
fn error(py: Python<'_>, what: &str, rows: Vec<Row>) -> PyErr {
    let message = match rows.first() {
        Some((code, message, ..)) => format!("{what}: {code}: {message}"),
        None => what.to_string(),
    };
    let err = IrError::new_err(message);
    let value = err.value(py);
    match PyList::new(py, rows) {
        Ok(list) => match value.setattr("diagnostics", list) {
            Ok(()) => err,
            Err(e) => e,
        },
        Err(e) => e,
    }
}

/// A module from a text (checked, as `komo check` does) or from a byte-code file (read
/// and verified, as `api::load` does).
fn module(py: Python<'_>, program: &Bound<'_, PyAny>) -> PyResult<Module> {
    if let Ok(bytes) = program.cast::<PyBytes>() {
        return api::load(bytes.as_bytes()).map_err(|refusals| {
            error(
                py,
                "the byte-code file is refused",
                refusals.iter().map(of_file).collect(),
            )
        });
    }
    let source: String = program.extract()?;
    api::build(&source).map_err(|diagnostics| {
        error(
            py,
            "the module is refused",
            diagnostics.iter().map(of_text).collect(),
        )
    })
}

/// The canonical text of a module's text.
#[pyfunction]
fn format(py: Python<'_>, source: &str) -> PyResult<String> {
    api::format(source).map_err(|d| {
        error(
            py,
            "the text is not a module",
            d.iter().map(of_text).collect(),
        )
    })
}

/// Every diagnostic of a module's text, as rows; none when it checks.
#[pyfunction]
fn check(source: &str) -> Vec<Row> {
    match api::check(source) {
        Ok(_) => Vec::new(),
        Err(d) => d.iter().map(of_text).collect(),
    }
}

/// The byte-code program of a module's text.
#[pyfunction]
fn compile<'py>(py: Python<'py>, source: &str) -> PyResult<Bound<'py, PyBytes>> {
    match api::compile(source) {
        Ok(bytes) => Ok(PyBytes::new(py, &bytes)),
        Err(d) => Err(error(
            py,
            "the module is refused",
            d.iter().map(of_text).collect(),
        )),
    }
}

/// The three hashes of a module, as hex: (semantic, build, full).
#[pyfunction]
fn hashes(py: Python<'_>, program: &Bound<'_, PyAny>) -> PyResult<(String, String, String)> {
    let m = module(py, program)?;
    let h = bytecode::hashes(&m);
    Ok((
        bytecode::hex(&h.semantic),
        bytecode::hex(&h.build),
        bytecode::hex(&h.full),
    ))
}

fn ability(a: &Ability) -> String {
    match a {
        Ability::LocalState { amount, tiles } => format!(
            "local_state >= {}({amount})",
            if *tiles { "tiles" } else { "bytes" }
        ),
        Ability::Grid => "group = grid".into(),
        Ability::Dynamic => "dynamic".into(),
        Ability::Islands(n) => format!("islands >= {n}"),
        Ability::Impl(name) => format!("impl(\"{name}\")"),
    }
}

/// A construct that causes a requirement, by its kind and name.
fn cause(m: &Module, c: &Cause) -> String {
    let program = |p: crate::types::ProgramId| {
        m.programs
            .get(p.0 as usize)
            .map_or(String::new(), |x| x.name.clone())
    };
    let named = |p: crate::types::ProgramId, kind: &str, i: u32, names: Vec<String>| {
        format!(
            "{kind} {}.{}",
            program(p),
            names.get(i as usize).cloned().unwrap_or_default()
        )
    };
    let handlers = |p: crate::types::ProgramId| {
        m.programs.get(p.0 as usize).map_or(Vec::new(), |x| {
            x.handlers.iter().map(|h| h.name.clone()).collect()
        })
    };
    match c {
        Cause::Stated => "requires".into(),
        Cause::GridHandler(p, h) | Cause::DynamicRun(Entry::Handler(p, h)) => {
            named(*p, "handler", h.0, handlers(*p))
        }
        Cause::DynamicRun(Entry::Run(p, r)) => {
            let names = m.programs.get(p.0 as usize).map_or(Vec::new(), |x| {
                x.runs.iter().map(|r| r.name.clone()).collect()
            });
            named(*p, "run", r.0, names)
        }
        Cause::DynamicRun(Entry::Chain(p, k)) => {
            let names = m.programs.get(p.0 as usize).map_or(Vec::new(), |x| {
                x.chains.iter().map(|c| c.name.clone()).collect()
            });
            named(*p, "chain", k.0, names)
        }
        Cause::ExternWithoutFallback(f) => format!(
            "func {}",
            m.funcs
                .get(f.0 as usize)
                .map_or(String::new(), |x| x.name.clone())
        ),
    }
}

/// The requirement set: each requirement as the `requires` block writes it, with the
/// constructs that cause it (`requires` when the block states it).
#[pyfunction]
fn requirements(
    py: Python<'_>,
    program: &Bound<'_, PyAny>,
) -> PyResult<Vec<(String, Vec<String>)>> {
    let m = module(py, program)?;
    Ok(m.requirements
        .iter()
        .map(|r| {
            (
                ability(&r.ability),
                r.causes.iter().map(|c| cause(&m, c)).collect(),
            )
        })
        .collect())
}

/// Runs a run script (docs/ir/run-scripts.md) on the interpreter: the report's text.
#[pyfunction]
fn run(py: Python<'_>, program: &Bound<'_, PyAny>, script: &str) -> PyResult<String> {
    let m = module(py, program)?;
    let parsed = api::script::parse(script)
        .map_err(|e| error(py, &format!("the run script is refused: {e}"), Vec::new()))?;
    on_deep_stack(|| api::script::run(&m, &parsed).map(|report| report.print()))
        .map_err(|e| error(py, &format!("the run script is refused: {e}"), Vec::new()))
}

/// The extension module.
#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IrError", m.py().get_type::<IrError>())?;
    m.add_function(wrap_pyfunction!(format, m)?)?;
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(hashes, m)?)?;
    m.add_function(wrap_pyfunction!(requirements, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
