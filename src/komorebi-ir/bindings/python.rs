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
use crate::api::origin::{Given, Mapped, Place, Refused};
use crate::backend::{
    ArgKind, ArgWord, Build, ChainArg, ChainCall, CompileArg, EntryPoint, Expr, Kernel, Processor,
    Setting,
};
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

/// One diagnostic as data: (code, message, line, column, severity, byte, place). A
/// text's diagnostic has a line and a column; a byte-code file's refusal has the byte; a
/// frontend's text's diagnostic has the place of its source, (file, first, last).
type Row = (
    String,
    String,
    Option<usize>,
    Option<usize>,
    &'static str,
    Option<u64>,
    Option<(String, u32, u32)>,
);

/// What a frontend gives beside its text, as Python passes it: (files, lines, map,
/// renames), each map range (text's first line, its column, text's last line, its
/// column, file, first, last).
type GivenRow = (
    Vec<String>,
    Vec<(u32, u32, String)>,
    Vec<MapRow>,
    Vec<(String, String)>,
);

/// A map range as Python passes it.
type MapRow = (u32, u32, u32, u32, u32, u32, u32);

fn given((files, lines, map, renames): GivenRow) -> Given {
    let map = map
        .into_iter()
        .map(
            |(text_first, text_first_col, text_last, text_last_col, file, first, last)| Mapped {
                text_first,
                text_first_col,
                text_last,
                text_last_col,
                file,
                first,
                last,
            },
        )
        .collect();
    Given {
        files,
        lines,
        map,
        renames,
    }
}

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
        None,
    )
}

fn of_placed(given: &Given, (d, place): &(Diagnostic, Option<Place>)) -> Row {
    let mut row = of_text(d);
    row.6 = place.and_then(|p| {
        let file = given.files.get(p.file as usize)?;
        Some((file.clone(), p.first, p.last))
    });
    row
}

/// The `IrError` of a refused origin: what is wrong, and no diagnostic.
fn origin_refused(py: Python<'_>, what: &str) -> PyErr {
    error(py, &format!("the origin is refused: {what}"), Vec::new())
}

fn of_file(r: &Refusal) -> Row {
    (
        r.code.to_string(),
        r.what.clone(),
        None,
        None,
        "error",
        r.at,
        None,
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

/// Every diagnostic of a module's text, as rows; none when it checks. With a
/// frontend's origin, each is placed in its source, and a refused origin raises.
#[pyfunction]
#[pyo3(signature = (source, origin=None))]
fn check(py: Python<'_>, source: &str, origin: Option<GivenRow>) -> PyResult<Vec<Row>> {
    let Some(origin) = origin else {
        return Ok(match api::check(source) {
            Ok(_) => Vec::new(),
            Err(d) => d.iter().map(of_text).collect(),
        });
    };
    let given = given(origin);
    match api::origin::check(source, &given) {
        Ok(rows) => Ok(rows.iter().map(|r| of_placed(&given, r)).collect()),
        Err(what) => Err(origin_refused(py, &what)),
    }
}

/// The byte-code program of a module's text; with a frontend's origin, the origin in
/// its notes (`api::origin`).
#[pyfunction]
#[pyo3(signature = (source, origin=None))]
fn compile<'py>(
    py: Python<'py>,
    source: &str,
    origin: Option<GivenRow>,
) -> PyResult<Bound<'py, PyBytes>> {
    let Some(origin) = origin else {
        return match api::compile(source) {
            Ok(bytes) => Ok(PyBytes::new(py, &bytes)),
            Err(d) => Err(error(
                py,
                "the module is refused",
                d.iter().map(of_text).collect(),
            )),
        };
    };
    let given = given(origin);
    match api::origin::compile(source, &given) {
        Ok(bytes) => Ok(PyBytes::new(py, &bytes)),
        Err(Refused::Text(rows)) => Err(error(
            py,
            "the module is refused",
            rows.iter().map(|r| of_placed(&given, r)).collect(),
        )),
        Err(Refused::Origin(what)) => Err(origin_refused(py, &what)),
    }
}

/// A frontend's text in the writer's layout, with its map carried to the lines of the
/// text as returned (`api::origin::layout`): (text, map).
#[pyfunction]
fn layout(py: Python<'_>, source: &str, origin: GivenRow) -> PyResult<(String, Vec<MapRow>)> {
    let given = given(origin);
    match api::origin::layout(source, &given) {
        Ok((text, laid)) => {
            let map = laid
                .map
                .iter()
                .map(|m| {
                    let (a, b) = (m.start(), m.end());
                    (a.0, a.1, b.0, b.1, m.file, m.first, m.last)
                })
                .collect();
            Ok((text, map))
        }
        Err(Refused::Text(rows)) => Err(error(
            py,
            "the text is not a module",
            rows.iter().map(|r| of_placed(&given, r)).collect(),
        )),
        Err(Refused::Origin(what)) => Err(origin_refused(py, &what)),
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

/// The data of an argument expression: `("word", i)`, `("const", n)`, `("add", a, b)`
/// (and `sub`, `mul`, `div`, `rem`).
fn expr<'py>(py: Python<'py>, e: &Expr) -> PyResult<Bound<'py, PyAny>> {
    Ok(match e {
        Expr::Word(i) => ("word", *i).into_pyobject(py)?.into_any(),
        Expr::Const(n) => ("const", *n).into_pyobject(py)?.into_any(),
        Expr::Rank => ("rank",).into_pyobject(py)?.into_any(),
        Expr::Size => ("size",).into_pyobject(py)?.into_any(),
        Expr::Add(a, b) => ("add", expr(py, a)?, expr(py, b)?)
            .into_pyobject(py)?
            .into_any(),
        Expr::Sub(a, b) => ("sub", expr(py, a)?, expr(py, b)?)
            .into_pyobject(py)?
            .into_any(),
        Expr::Mul(a, b) => ("mul", expr(py, a)?, expr(py, b)?)
            .into_pyobject(py)?
            .into_any(),
        Expr::Div(a, b) => ("div", expr(py, a)?, expr(py, b)?)
            .into_pyobject(py)?
            .into_any(),
        Expr::Rem(a, b) => ("rem", expr(py, a)?, expr(py, b)?)
            .into_pyobject(py)?
            .into_any(),
    })
}

fn arg_kind(k: &ArgKind) -> String {
    match k {
        ArgKind::Tensor(f) => format!("tensor<{}>", f.name()),
        ArgKind::Scalar(t) => t.name().to_string(),
        ArgKind::Bool => "bool".into(),
    }
}

fn arg_word(w: &ArgWord) -> (&'static str, usize) {
    match w {
        ArgWord::TensorAddress(i) => ("address", *i),
        ArgWord::TensorTiles(i) => ("tiles", *i),
        ArgWord::Word(i) => ("word", *i),
    }
}

/// One entry point as data: (name, kind, number, symbol, params, words, bounds,
/// patterns, mask_word, calls); a call (handler, args, part, size, width, output,
/// input, ranges, reads, bits).
fn entry_point<'py>(py: Python<'py>, e: &EntryPoint) -> PyResult<Bound<'py, PyAny>> {
    let kind = match e.entry {
        Entry::Handler(..) => "handler",
        Entry::Run(..) => "run",
        Entry::Chain(..) => "chain",
    };
    let params: Vec<String> = e.params.iter().map(arg_kind).collect();
    let words: Vec<(&str, usize)> = e.words.iter().map(arg_word).collect();
    let mut bounds = Vec::new();
    for b in &e.bounds {
        let cols = match &b.cols {
            Some((first, count)) => Some((expr(py, first)?, expr(py, count)?)),
            None => None,
        };
        bounds.push((b.tensor, expr(py, &b.first)?, expr(py, &b.count)?, cols));
    }
    let row = (
        e.name.clone(),
        kind,
        e.number,
        e.symbol.clone(),
        params,
        words,
        bounds,
        e.patterns.clone(),
        e.mask_word,
        e.calls
            .iter()
            .map(|c| chain_call(py, c))
            .collect::<PyResult<Vec<_>>>()?,
    );
    Ok(row.into_pyobject(py)?.into_any())
}

/// One call of a chain as data: (handler, args, part, size, width, output, input,
/// ranges, reads, bits); an argument `("param", p)`, `("int", n)`, `("float", x)` or
/// `("bool", b)`, a width and each range's ends as a bound's.
fn chain_call<'py>(py: Python<'py>, c: &ChainCall) -> PyResult<Bound<'py, PyAny>> {
    let args = c
        .args
        .iter()
        .map(|a| match a {
            ChainArg::Param(p) => ("param", *p).into_pyobject(py).map(Bound::into_any),
            ChainArg::Int(n) => ("int", *n).into_pyobject(py).map(Bound::into_any),
            ChainArg::Float(x) => ("float", f64::from_bits(*x))
                .into_pyobject(py)
                .map(Bound::into_any),
            ChainArg::Bool(b) => ("bool", *b).into_pyobject(py).map(Bound::into_any),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut ranges = Vec::new();
    for (base, step, width) in &c.ranges {
        ranges.push((expr(py, base)?, expr(py, step)?, expr(py, width)?));
    }
    let row = (
        c.handler,
        args,
        c.part,
        c.size,
        expr(py, &c.width)?,
        c.output,
        c.input,
        ranges,
        c.reads.clone(),
        c.bits.clone(),
    );
    Ok(row.into_pyobject(py)?.into_any())
}

/// One kernel as data: (source, processor, defines, args, options); an argument is
/// `("word", n)` or `("layout", parameter)`, an option's value a bool or a text.
fn kernel<'py>(py: Python<'py>, k: &Kernel) -> PyResult<KernelRow<'py>> {
    let processor = match k.processor {
        Processor::Reader => "reader",
        Processor::Writer => "writer",
        Processor::Compute => "compute",
    };
    let args = k
        .args
        .iter()
        .map(|a| match a {
            CompileArg::Word(n) => ("word", u64::from(*n)),
            CompileArg::TensorLayout { param } => ("layout", *param as u64),
        })
        .collect();
    let mut options = Vec::new();
    for (name, value) in &k.options {
        options.push((name.clone(), setting(py, value)?));
    }
    Ok((
        k.source.clone(),
        processor,
        k.defines.clone(),
        args,
        options,
    ))
}

/// A buffer as Python takes it: index, format, page bytes, pages, pair and unpack.
type BufferRow<'a> = (u32, &'a str, u64, u64, Option<u32>, Option<&'a str>);

type KernelRow<'py> = (
    String,
    &'static str,
    Vec<(String, String)>,
    Vec<(&'static str, u64)>,
    Vec<(String, Bound<'py, PyAny>)>,
);

/// A module's output on a backend (`api::emit`), as data: (key, files, entries,
/// kernels, buffers, steps, the files made at load by name); a backend that builds by
/// steps has no kernels, no buffers and no files made at load, one whose machine
/// compiles kernels has no steps.
#[pyfunction]
#[pyo3(signature = (program, backend="tensix", target=None))]
fn emit<'py>(
    py: Python<'py>,
    program: &Bound<'py, PyAny>,
    backend: &str,
    target: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = module(py, program)?;
    let output = match api::emit::emit(&m, backend, target) {
        Ok(output) => output,
        Err(api::emit::Refused::NoBackend(name)) => {
            let names = api::emit::BACKENDS.join(", ");
            return Err(error(
                py,
                &format!("no backend is named {name:?}: one of {names}"),
                Vec::new(),
            ));
        }
        Err(api::emit::Refused::Backend(refusals)) => {
            let rows = refusals
                .iter()
                .map(|r| {
                    let at = r.at.map(|a| format!(" (at {a:?})")).unwrap_or_default();
                    let message = format!("{}{at}", r.what);
                    (r.code.to_string(), message, None, None, "error", None, None)
                })
                .collect();
            return Err(error(py, "the backend refuses the module", rows));
        }
    };
    let files: Vec<(String, Bound<'py, PyBytes>)> = output
        .files
        .iter()
        .map(|(name, bytes)| (name.clone(), PyBytes::new(py, bytes)))
        .collect();
    let mut entries = Vec::new();
    for e in &output.entries {
        entries.push(entry_point(py, e)?);
    }
    let (kernels, buffers, steps, at_load) = match &output.build {
        Build::Kernels {
            kernels,
            buffers,
            at_load,
        } => (
            kernels
                .iter()
                .map(|k| kernel(py, k))
                .collect::<PyResult<_>>()?,
            buffers
                .iter()
                .map(|b| {
                    let unpack = b.unpack.map(|u| u.name());
                    (
                        b.index,
                        b.format.name(),
                        b.page_bytes,
                        b.pages,
                        b.pair,
                        unpack,
                    )
                })
                .collect(),
            Vec::new(),
            at_load.clone(),
        ),
        Build::Steps(steps) => (
            Vec::new(),
            Vec::new(),
            steps
                .iter()
                .map(|s| (s.program.clone(), s.args.clone()))
                .collect(),
            Vec::new(),
        ),
    };
    let kernels: Vec<KernelRow> = kernels;
    let buffers: Vec<BufferRow> = buffers;
    let steps: Vec<(String, Vec<String>)> = steps;
    let row = (
        output.key.hex(),
        files,
        entries,
        kernels,
        buffers,
        steps,
        at_load,
    );
    Ok(row.into_pyobject(py)?.into_any())
}

/// An option's value as Python's: a bool or a str.
fn setting<'py>(py: Python<'py>, value: &Setting) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Setting::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Setting::Text(t) => t.into_pyobject(py)?.into_any(),
    })
}

/// The extension module.
#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IrError", m.py().get_type::<IrError>())?;
    m.add_function(wrap_pyfunction!(format, m)?)?;
    m.add_function(wrap_pyfunction!(layout, m)?)?;
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(hashes, m)?)?;
    m.add_function(wrap_pyfunction!(requirements, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(emit, m)?)?;
    Ok(())
}
