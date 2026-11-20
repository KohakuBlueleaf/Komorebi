//! The C++ printer of the CPU backend: one statement for each instruction, in program
//! order, from the shared walk; values named by their number (`v8`), declarations by
//! their own names; nothing from the notes' positions or names. What it does not print
//! yet is refused by name (`K964`) before anything is printed.

use std::fmt::Write;

use crate::backend::emit::{walk, Event};
use crate::backend::{Refusal, NOT_BUILT};
use crate::bytecode::version::{self as v};
use crate::program::body::{Body, Inst, Op, ValueId};
use crate::program::notes::{Anchor, Owner};
use crate::program::{BodyRef, FuncKind, Module};
use crate::types::{
    BinOp, ConstValue, Count, Format, FuncId, HandlerId, On, ProgramId, ScalarType, Type,
};

/// The element type of a format the framework has.
fn element(f: Format) -> Option<&'static str> {
    match f {
        Format::Bf16 => Some("bf16"),
        Format::F32 => Some("f32"),
        Format::U32 => Some("u32"),
        _ => None,
    }
}

fn scalar(s: ScalarType) -> Option<&'static str> {
    match s {
        ScalarType::U32 => Some("u32"),
        ScalarType::F32 => Some("f32"),
        _ => None,
    }
}

/// The C++ type a value of this type is held in.
fn cpp_type(t: &Type) -> Option<String> {
    match t {
        Type::Scalar(s) => scalar(*s).map(str::to_string),
        Type::Tile(f) => element(*f).map(|e| format!("Tile<{e}>")),
        Type::View(Some(f)) => element(*f).map(|_| "View".to_string()),
        Type::Stream(f, Count::Const(1)) => element(*f).map(|_| "Stream".to_string()),
        Type::Sink(f, Count::Const(1)) => element(*f).map(|_| "Sink".to_string()),
        Type::Tensor(f, false) => element(*f).map(|_| "const komo_tensor&".to_string()),
        _ => None,
    }
}

/// The name of an operation, as the byte code's table writes it.
fn op_name(op: &Op) -> String {
    let text = format!("{op:?}");
    text.split(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap_or_default()
        .to_string()
}

fn val(x: &ValueId) -> String {
    format!("v{}", x.0)
}

/// Where a body is, for its refusals: what it may use.
#[derive(Clone, Copy, PartialEq, Eq)]
enum In {
    Func,
    Handler,
}

/// Why an instruction is not printed yet, if it is not.
fn unbuilt(body: &Body, inst: &Inst, place: In) -> Option<String> {
    let ty = |x: &ValueId| body.values.get(x.0 as usize);
    let def = inst.defs.first().and_then(ty);
    if let Some(t) = def {
        if cpp_type(t).is_none() {
            return Some(format!("a value of type {t}"));
        }
    }
    let ok = match (&inst.op, place) {
        (Op::Arg(_), _) => true,
        (Op::Const(ConstValue::Int(ScalarType::U32, _)), _) => true,
        (Op::Const(ConstValue::Float(ScalarType::F32, x)), _) => (*x as f32).is_finite(),
        (Op::Binary(BinOp::Mul, a, b), _) => {
            matches!(ty(a), Some(Type::Tile(_)))
                && matches!(ty(b), Some(Type::Scalar(ScalarType::F32)))
        }
        (Op::CallFunc(..), _) => true,
        (Op::Return(Some(_)), In::Func) | (Op::Return(None), In::Handler) => true,
        (
            Op::Tiles { .. }
            | Op::Rank
            | Op::GroupSize
            | Op::Partition(..)
            | Op::Stream(..)
            | Op::Sink(..)
            | Op::ForStream { .. }
            | Op::Push(..)
            | Op::End,
            In::Handler,
        ) => true,
        _ => false,
    };
    (!ok).then(|| format!("`{}`", op_name(&inst.op)))
}

fn not_built(what: String, at: Anchor) -> Refusal {
    Refusal {
        code: NOT_BUILT,
        what: format!("the cpu backend does not emit {what} yet"),
        at: Some(at),
    }
}

/// Everything of the module the printer does not print yet, each refused by name at
/// its construct. Empty when the module is printed whole.
pub fn unbuilt_parts(m: &Module) -> Vec<Refusal> {
    let mut out = Vec::new();
    for (i, _) in m.params.iter().enumerate() {
        let at = Anchor::Param(crate::types::ParamId(i as u32));
        out.push(not_built("a module parameter".into(), at));
    }
    for (i, _) in m.records.iter().enumerate() {
        let at = Anchor::Record(crate::types::RecordId(i as u32));
        out.push(not_built("a record type".into(), at));
    }
    for (i, _) in m.messages.iter().enumerate() {
        let at = Anchor::Message(crate::types::MessageId(i as u32));
        out.push(not_built("a message type".into(), at));
    }
    for (i, _) in m.controls.iter().enumerate() {
        let at = Anchor::Control(crate::types::ControlId(i as u32));
        out.push(not_built("a control program".into(), at));
    }
    for (f, func) in m.funcs.iter().enumerate() {
        let fid = FuncId(f as u32);
        let Some(body) = (match func.kind {
            FuncKind::Plain => func.body.as_ref(),
            _ => None,
        }) else {
            out.push(not_built(
                "a `combine` or an external function".into(),
                Anchor::Func(fid),
            ));
            continue;
        };
        for (k, t) in func.params.iter().chain([&func.ret]).enumerate() {
            if !matches!(t, Type::Tile(_) | Type::Scalar(_)) || cpp_type(t).is_none() {
                let at = if k < func.params.len() {
                    Anchor::ParamOf(Owner::Func(fid), k as u32)
                } else {
                    Anchor::Func(fid)
                };
                out.push(not_built(format!("a function's value of type {t}"), at));
            }
        }
        body_parts(body, BodyRef::Func(fid), In::Func, &mut out);
    }
    for (p, prog) in m.programs.iter().enumerate() {
        let pid = ProgramId(p as u32);
        if !prog.state.is_empty() {
            out.push(not_built("state".into(), Anchor::Program(pid)));
        }
        for (r, _) in prog.runs.iter().enumerate() {
            let at = Anchor::Run(pid, crate::types::RunId(r as u32));
            out.push(not_built("a dynamic run".into(), at));
        }
        for (c, _) in prog.chains.iter().enumerate() {
            let at = Anchor::Chain(pid, crate::types::ChainId(c as u32));
            out.push(not_built("a chain".into(), at));
        }
        for (h, handler) in prog.handlers.iter().enumerate() {
            let hid = HandlerId(h as u32);
            if handler.ret.is_some() {
                out.push(not_built(
                    "a handler's result".into(),
                    Anchor::Handler(pid, hid),
                ));
            }
            if handler.on != On::Group {
                out.push(not_built(
                    "a handler on a grid".into(),
                    Anchor::Handler(pid, hid),
                ));
            }
            for (k, t) in handler.params.iter().enumerate() {
                if !matches!(t, Type::Tensor(..) | Type::Scalar(_)) || cpp_type(t).is_none() {
                    let at = Anchor::ParamOf(Owner::Handler(pid, hid), k as u32);
                    out.push(not_built(format!("a parameter of type {t}"), at));
                }
            }
            body_parts(
                &handler.body,
                BodyRef::Handler(pid, hid),
                In::Handler,
                &mut out,
            );
        }
    }
    out
}

fn body_parts(body: &Body, r: BodyRef, place: In, out: &mut Vec<Refusal>) {
    for (i, inst) in body.insts.iter().enumerate() {
        if let Some(what) = unbuilt(body, inst, place) {
            let at = Anchor::Inst(r, crate::program::body::InstId(i as u32));
            out.push(not_built(what, at));
        }
    }
    if let Err(what) = walk(body) {
        out.push(not_built(
            format!("a body whose constructs do not balance ({what})"),
            Anchor::Module,
        ));
    }
}

/// The number of a format or a scalar type in the byte code's tables.
fn format_number(f: Format) -> u8 {
    v::number(v::FORMAT, &format!("{f:?}"))
}

fn scalar_number(s: ScalarType) -> u8 {
    v::number(v::SCALAR_TYPE, &format!("{s:?}"))
}

/// The C++ source of a module `unbuilt_parts` refuses nothing of, with its build hash.
pub fn source(m: &Module, build_hash: &str) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "// Emitted by the Komorebi IR's CPU backend from the module \"{}\"; do not edit.",
        m.name
    );
    let _ = writeln!(s, "// Its build hash: {build_hash}.");
    s.push_str("#include \"komo_cpu.hpp\"\n\n");
    let _ = writeln!(s, "namespace module_{} {{", m.name);
    s.push_str("using namespace komo_cpu;\n");
    let _ = writeln!(
        s,
        "constexpr TileShape TILE{{{}, {}}};",
        m.tile.rows, m.tile.cols
    );
    for (f, func) in m.funcs.iter().enumerate() {
        if let Some(body) = &func.body {
            s.push('\n');
            function(&mut s, f, &func.name, &func.params, &func.ret, body);
        }
    }
    for (p, prog) in m.programs.iter().enumerate() {
        for (h, handler) in prog.handlers.iter().enumerate() {
            s.push('\n');
            let types: Vec<String> = handler.params.iter().map(ToString::to_string).collect();
            let _ = writeln!(
                s,
                "// p{p}.h{h} {}.{}({}) @on(group)",
                prog.name,
                handler.name,
                types.join(", ")
            );
            let _ = writeln!(
                s,
                "static Status p{p}_h{h}(const Worker& w, const Args& a) {{"
            );
            body_text(&mut s, &handler.body, In::Handler);
            s.push_str("    return Status::ok();\n}\n");
        }
    }
    let _ = writeln!(s, "\n}}  // namespace module_{}\n", m.name);
    s.push_str("// The module's entries, by `Program.handler`, each with its parameters.\n");
    let mut entries = Vec::new();
    for (p, prog) in m.programs.iter().enumerate() {
        for (h, handler) in prog.handlers.iter().enumerate() {
            let _ = writeln!(s, "static const komo_cpu::ArgKind p{p}_h{h}_params[] = {{");
            for t in &handler.params {
                let kind = match t {
                    Type::Tensor(f, _) => format!("KOMO_ARG_TENSOR, {}", format_number(*f)),
                    Type::Scalar(sc) => format!("KOMO_ARG_SCALAR, {}", scalar_number(*sc)),
                    _ => "KOMO_ARG_BOOL, 0".to_string(),
                };
                let _ = writeln!(s, "    {{{kind}}},  // {t}");
            }
            s.push_str("};\n");
            entries.push(format!(
                "    {{\"{}.{}\", &module_{}::p{p}_h{h}, {}, p{p}_h{h}_params}},",
                prog.name,
                handler.name,
                m.name,
                handler.params.len()
            ));
        }
    }
    s.push_str("\nstatic const komo_cpu::EntryPoint entries[] = {\n");
    for e in &entries {
        s.push_str(e);
        s.push('\n');
    }
    s.push_str("};\n\n");
    let _ = writeln!(
        s,
        "extern \"C\" const komo_cpu::Module komo_module = {{\n    KOMO_CPU_ABI, \"{}\", {{{}, {}}}, {}, entries\n}};",
        m.name,
        m.tile.rows,
        m.tile.cols,
        entries.len()
    );
    s
}

fn function(s: &mut String, f: usize, name: &str, params: &[Type], ret: &Type, body: &Body) {
    let types: Vec<String> = params.iter().map(ToString::to_string).collect();
    let _ = writeln!(s, "// f{f} {name}({}) -> {ret}", types.join(", "));
    let args: Vec<String> = params
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let ty = cpp_type(t).unwrap_or_default();
            match t {
                Type::Tile(_) => format!("const {ty}& p{i}"),
                _ => format!("{ty} p{i}"),
            }
        })
        .collect();
    let _ = writeln!(
        s,
        "static {} f{f}({}) {{",
        cpp_type(ret).unwrap_or_default(),
        args.join(", ")
    );
    body_text(s, body, In::Func);
    s.push_str("}\n");
}

/// The statements of a body, one for each instruction, indented by depth.
fn body_text(s: &mut String, body: &Body, place: In) {
    let Ok(events) = walk(body) else {
        return;
    };
    let ty = |x: &ValueId| -> String {
        body.values
            .get(x.0 as usize)
            .and_then(cpp_type)
            .unwrap_or_default()
    };
    for (depth, event) in events {
        let pad = "    ".repeat(depth + 1);
        match event {
            Event::Open(id, inst) | Event::Plain(id, inst) => {
                let i = id.0;
                let name = op_name(&inst.op);
                let d = inst.defs.first().map(val).unwrap_or_default();
                let dt = inst.defs.first().map(ty).unwrap_or_default();
                let line = match &inst.op {
                    Op::Arg(k) => {
                        let t = inst
                            .defs
                            .first()
                            .and_then(|x| body.values.get(x.0 as usize));
                        match (place, t) {
                            (In::Func, Some(Type::Tile(_))) => format!("const {dt}& {d} = p{k};"),
                            (In::Func, _) => format!("const {dt} {d} = p{k};"),
                            (_, Some(Type::Tensor(..))) => format!("{dt} {d} = a.tensor({k});"),
                            (_, Some(Type::Scalar(ScalarType::F32))) => {
                                format!("const f32 {d} = a.f32_at({k});")
                            }
                            _ => format!("const {dt} {d} = a.u32_at({k});"),
                        }
                    }
                    Op::Const(ConstValue::Int(_, n)) => format!("const {dt} {d} = {n};"),
                    Op::Const(ConstValue::Float(_, x)) => {
                        format!("const {dt} {d} = {:?}f;", *x as f32)
                    }
                    Op::Binary(BinOp::Mul, a, b) => {
                        format!("{dt} {d} = mul({}, {});", val(a), val(b))
                    }
                    Op::CallFunc(f, args) => {
                        let args: Vec<String> = args.iter().map(val).collect();
                        format!("{dt} {d} = f{}({});", f.0, args.join(", "))
                    }
                    Op::Return(Some(x)) => format!("return {};", val(x)),
                    Op::Return(None) => "return Status::ok();".to_string(),
                    Op::Tiles { tensor, first, n } => format!(
                        "const View {d} = tiles({}, {}, {});",
                        val(tensor),
                        val(first),
                        val(n)
                    ),
                    Op::Rank => format!("const u32 {d} = w.rank;"),
                    Op::GroupSize => format!("const u32 {d} = w.size;"),
                    Op::Partition(x, r, n) => format!(
                        "KOMO_LET(View, {d}, partition({}, {}, {}));",
                        val(x),
                        val(r),
                        val(n)
                    ),
                    Op::Stream(x, b) => {
                        format!(
                            "KOMO_LET(Stream, {d}, stream({}, {}, TILE));",
                            val(x),
                            val(b)
                        )
                    }
                    Op::Sink(x, b) => {
                        format!("KOMO_LET(Sink, {d}, sink({}, {}, TILE));", val(x), val(b))
                    }
                    Op::Push(k, x) => format!("KOMO_DO({}.push({}));", val(k), val(x)),
                    Op::ForStream { stream } => {
                        let _ = writeln!(s, "{pad}for (;;) {{  // i{i} {name}");
                        let _ = writeln!(s, "{pad}    {dt} {d}(TILE);");
                        let _ = writeln!(
                            s,
                            "{pad}    KOMO_LET(bool, more_i{i}, {}.next_tile({d}));",
                            val(stream)
                        );
                        let _ = writeln!(
                            s,
                            "{pad}    if (!more_i{i}) {{\n{pad}        break;\n{pad}    }}"
                        );
                        continue;
                    }
                    _ => continue,
                };
                let _ = writeln!(s, "{pad}{line}  // i{i} {name}");
            }
            Event::Close(id) => {
                let _ = writeln!(s, "{pad}}}  // i{} End", id.0);
            }
            Event::Else(_) | Event::Test(..) => {}
        }
    }
}
