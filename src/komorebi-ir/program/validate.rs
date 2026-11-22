//! The validator: every way a program structure can break its own rules. Lower's
//! result passes it, and a reader of the byte code runs it on what it reads.
//!
//! It checks that the constructs of each body balance; that every operand is defined
//! before its use and not used after its construct's `End`; that the operands' types
//! fit each operation, and the values it defines have the types it gives; that every
//! id is in range; that a body opens with its `Arg`s; that every anchor of the notes,
//! every cause and every derived access names something that exists; and that the
//! notes are in their stated order.

use super::body::{Body, InstId, Op, ScalarsFrom, ValueId};
use super::derive::{Cause, Effects};
use super::notes::{Anchor, Named, Owner};
use super::{BodyRef, CallArg, FuncKind, MemberKind, Module};
use crate::types::{
    BinOp, Binding, Combiner, ConstValue, Count, Entry, Format, FuncId, MessageId, ProgramId,
    ScalarType, Type,
};

/// One violation: where, and what.
#[derive(Clone, Debug, PartialEq)]
pub struct Violation {
    /// The body, when it is in one.
    pub body: Option<BodyRef>,
    /// The instruction, when it is one.
    pub inst: Option<InstId>,
    /// What is wrong.
    pub what: String,
}

/// Every violation of the module, in the module's order. Empty for a valid module.
pub fn validate(module: &Module) -> Vec<Violation> {
    let mut v = Validator {
        module,
        out: Vec::new(),
    };
    v.declarations();
    for (body_ref, body, ctx) in v.bodies() {
        v.body(body_ref, body, &ctx);
    }
    v.derived();
    v.notes();
    v.out
}

/// What a body may refer to.
struct Ctx<'m> {
    /// The owner's parameters, as its body reads them.
    params: std::borrow::Cow<'m, [Type]>,
    /// Whether the body is a function on blocks, typed on blocks (R45).
    on_blocks: bool,
    /// The result `return` gives: `None` for none.
    ret: Option<Type>,
    /// The program the body belongs to.
    program: Option<ProgramId>,
    /// The message an `on M` body receives.
    arrived: Option<MessageId>,
}

/// An open construct.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Open {
    If { seen_else: bool },
    Loop,
    While { tested: bool },
}

struct Validator<'m> {
    module: &'m Module,
    out: Vec<Violation>,
}

/// Whether two formats are one, where among the float formats a format is a
/// notation: two float formats always fit; integer formats must agree, and an integer
/// format never fits a float one.
fn format_fits(f: Format, g: Format) -> bool {
    f == g || (!f.is_integer() && !g.is_integer())
}

fn scalar_fits(s: ScalarType, t: ScalarType) -> bool {
    s == t || (s.is_float() && t.is_float())
}

/// Whether a value of type `got` may stand where `want` is used: the same type, with
/// float formats free.
pub fn fits(got: &Type, want: &Type) -> bool {
    use Type as T;
    let opt = |a: &Option<ScalarType>, b: &Option<ScalarType>| match (a, b) {
        (Some(s), Some(t)) => scalar_fits(*s, *t),
        (None, None) => true,
        _ => false,
    };
    match (got, want) {
        (T::Scalar(s), T::Scalar(t)) | (T::Words(s), T::Words(t)) => scalar_fits(*s, *t),
        (T::Tile(f), T::Tile(g)) => format_fits(*f, *g),
        // a view of a format may be given where a view of no format is declared
        (T::View(_), T::View(None)) => true,
        (T::View(Some(f)), T::View(Some(g))) => format_fits(*f, *g),
        (T::Reduced(f, a, r, c), T::Reduced(g, b, s, d)) => {
            a == b && (r, c) == (s, d) && format_fits(*f, *g)
        }
        (T::Block(f, r, c), T::Block(g, s, d))
        | (T::Stream(f, r, c), T::Stream(g, s, d))
        | (T::Sink(f, r, c), T::Sink(g, s, d)) => (r, c) == (s, d) && format_fits(*f, *g),
        (T::Tensor(f, r), T::Tensor(g, s)) => r == s && format_fits(*f, *g),
        (T::Zip(a, b), T::Zip(c, d)) => fits(a, c) && fits(b, d),
        (T::Handle(a), T::Handle(b)) | (T::Outcome(a), T::Outcome(b)) => opt(a, b),
        _ => got == want,
    }
}

/// The shape of a tile (`[1, 1]`) or of a block (R45): its rows and columns of tiles.
fn shape_of(t: &Type) -> Option<(Count, Count)> {
    match t {
        Type::Tile(_) => Some((Count::Const(1), Count::Const(1))),
        Type::Block(_, r, c) => Some((*r, *c)),
        _ => None,
    }
}

/// A tile or a block of this shape: a tile when it is `[1, 1]`.
fn of_shape(f: Format, (r, c): (Count, Count)) -> Type {
    match (r, c) {
        (Count::Const(1), Count::Const(1)) => Type::Tile(f),
        _ => Type::Block(f, r, c),
    }
}

/// Whether an element-by-element operation on `operands` may give `result`: its tiles
/// (and, in a function on blocks, blocks) or a reduction's values, all of one shape,
/// beside scalars; the result of that shape, a block when an operand is one. `Some`
/// tells whether they are a reduction's values; `None` when they do not fit.
fn elementwise(on_blocks: bool, operands: &[&Type], result: &[&Type]) -> Option<bool> {
    let mut shape = None;
    let (mut block, mut tile, mut axis) = (false, false, None);
    for t in operands {
        let s = match t {
            Type::Scalar(_) => continue,
            Type::Tile(_) => {
                tile = true;
                (Count::Const(1), Count::Const(1))
            }
            Type::Block(_, r, c) if on_blocks => {
                block = true;
                (*r, *c)
            }
            Type::Reduced(_, a, r, c) => {
                if axis.is_some_and(|b| b != *a) {
                    return None;
                }
                axis = Some(*a);
                (*r, *c)
            }
            _ => return None,
        };
        if shape.is_some_and(|first| first != s) {
            return None;
        }
        shape = Some(s);
    }
    let shape = shape?;
    let ok = match (axis, result) {
        (Some(_), _) if tile || block => false,
        (Some(a), [Type::Reduced(_, b, r, c)]) => a == *b && (*r, *c) == shape,
        (None, [Type::Block(_, r, c)]) => block && (*r, *c) == shape,
        (None, [Type::Tile(_)]) => !block && shape == (Count::Const(1), Count::Const(1)),
        _ => false,
    };
    ok.then_some(axis.is_some())
}

fn is_int(t: &Type) -> bool {
    matches!(t, Type::Scalar(s) if !s.is_float())
}

fn scalar_of(c: &ConstValue) -> Type {
    match c {
        ConstValue::Int(s, _) | ConstValue::Float(s, _) => Type::Scalar(*s),
        ConstValue::Bool(_) => Type::Bool,
    }
}

impl<'m> Validator<'m> {
    /// A function on blocks works on each tile, reduces along each row of a block,
    /// spreads that back with `.col()` and a parameter's row 0 with `.row()` (R43), and
    /// multiplies blocks (R45): `.T`, a reduction along a column and any other spread
    /// are violations.
    fn on_blocks_body(&mut self, at: BodyRef, body: &Body) {
        let arg = |v: ValueId| {
            body.insts
                .iter()
                .any(|i| matches!(i.op, Op::Arg(_)) && i.defs.first() == Some(&v))
        };
        for (k, inst) in body.insts.iter().enumerate() {
            let what = match &inst.op {
                Op::Transpose(_) => "`.T`",
                Op::Reduce(_, _, crate::types::Axis::Col) => "a reduction along a column",
                Op::SpreadCol(v) => match body.values.get(v.0 as usize) {
                    Some(Type::Reduced(_, crate::types::Axis::Row, ..)) => continue,
                    _ => "`.col()` of other than a reduction along a row",
                },
                Op::SpreadRow(v) if arg(*v) => continue,
                Op::SpreadRow(_) => "`.row()` of other than a parameter",
                _ => continue,
            };
            self.fail(
                Some(at),
                Some(InstId(k as u32)),
                format!("{what} in a function on blocks"),
            );
        }
    }

    /// A function on tiles called in a function on blocks (R45): its tile parameters
    /// given blocks of one shape, its other parameters their own; it gives its result
    /// on each tile, a block of that shape.
    fn on_each_tile(&self, func: &super::Func, a: &[&Type], d: &[&Type]) -> Result<(), String> {
        let mut shape = None;
        let mut ok = func.params.len() == a.len();
        for (p, x) in func.params.iter().zip(a) {
            match (p, x) {
                (Type::Tile(f), Type::Block(g, r, c)) if format_fits(*g, *f) => {
                    ok &= shape.is_none_or(|s| s == (*r, *c));
                    shape = Some((*r, *c));
                }
                (Type::Tile(_), _) => ok = false,
                _ => ok &= fits(x, p),
            }
        }
        let ok = ok
            && match (shape, &func.ret, d) {
                (Some((r, c)), Type::Tile(f), [Type::Block(g, s, t)]) => {
                    f == g && (r, c) == (*s, *t)
                }
                _ => false,
            };
        if ok {
            Ok(())
        } else {
            Err("a call whose arguments do not fit".into())
        }
    }

    fn fail(&mut self, body: Option<BodyRef>, inst: Option<InstId>, what: impl Into<String>) {
        self.out.push(Violation {
            body,
            inst,
            what: what.into(),
        });
    }

    fn program(&self, p: ProgramId) -> Option<&'m super::Program> {
        self.module.programs.get(p.0 as usize)
    }

    /// Every body of the module, in the module's order, with what it may refer to.
    fn bodies(&self) -> Vec<(BodyRef, &'m Body, Ctx<'m>)> {
        let m = self.module;
        let mut out = Vec::new();
        for (i, f) in m.funcs.iter().enumerate() {
            if let Some(body) = &f.body {
                let ctx = Ctx {
                    params: std::borrow::Cow::Borrowed(&f.params),
                    ret: Some(f.ret.clone()),
                    program: None,
                    arrived: None,
                    on_blocks: f.on_blocks(),
                };
                out.push((BodyRef::Func(FuncId(i as u32)), body, ctx));
            }
        }
        for (p, prog) in m.programs.iter().enumerate() {
            let pid = ProgramId(p as u32);
            for (h, handler) in prog.handlers.iter().enumerate() {
                let ctx = Ctx {
                    params: std::borrow::Cow::Borrowed(&handler.params),
                    ret: handler.ret.map(Type::Scalar),
                    program: Some(pid),
                    arrived: None,
                    on_blocks: false,
                };
                let r = BodyRef::Handler(pid, super::HandlerId(h as u32));
                out.push((r, &handler.body, ctx));
            }
            for (r, run) in prog.runs.iter().enumerate() {
                let rid = super::RunId(r as u32);
                let seed = Ctx {
                    params: std::borrow::Cow::Borrowed(&run.params),
                    ret: None,
                    program: Some(pid),
                    arrived: None,
                    on_blocks: false,
                };
                out.push((BodyRef::Seed(pid, rid), &run.seed, seed));
                for (k, (message, body)) in run.on_message.iter().enumerate() {
                    let ctx = Ctx {
                        params: std::borrow::Cow::Borrowed(&run.params),
                        ret: None,
                        program: Some(pid),
                        arrived: Some(*message),
                        on_blocks: false,
                    };
                    out.push((BodyRef::On(pid, rid, k as u32), body, ctx));
                }
            }
        }
        for (c, control) in m.controls.iter().enumerate() {
            let ctx = Ctx {
                params: std::borrow::Cow::Borrowed(&control.params),
                ret: control.ret.map(Type::Scalar),
                program: None,
                arrived: None,
                on_blocks: false,
            };
            out.push((
                BodyRef::Control(super::ControlId(c as u32)),
                &control.body,
                ctx,
            ));
        }
        out
    }

    fn body_of(&self, r: BodyRef) -> Option<&'m Body> {
        let m = self.module;
        match r {
            BodyRef::Func(f) => m.funcs.get(f.0 as usize)?.body.as_ref(),
            BodyRef::Handler(p, h) => Some(&self.program(p)?.handlers.get(h.0 as usize)?.body),
            BodyRef::Seed(p, r) => Some(&self.program(p)?.runs.get(r.0 as usize)?.seed),
            BodyRef::On(p, r, k) => Some(
                &self
                    .program(p)?
                    .runs
                    .get(r.0 as usize)?
                    .on_message
                    .get(k as usize)?
                    .1,
            ),
            BodyRef::Control(c) => Some(&m.controls.get(c.0 as usize)?.body),
        }
    }

    fn count(&mut self, count: Count, member: bool, what: &str) {
        match count {
            Count::Const(_) => {}
            Count::GroupSize if member => self.fail(
                None,
                None,
                format!("{what}: a member counted by the group's size"),
            ),
            Count::GroupSize => {}
            Count::Param(p) if p.0 as usize >= self.module.params.len() => {
                self.fail(
                    None,
                    None,
                    format!("{what}: parameter {} out of range", p.0),
                );
            }
            Count::Param(_) => {}
        }
    }

    fn declarations(&mut self) {
        let m = self.module;
        let owners = m.records.iter().map(|r| (&r.name, r.program));
        for (name, owner) in owners.chain(m.messages.iter().map(|x| (&x.name, x.program))) {
            if owner.is_some_and(|p| self.program(p).is_none()) {
                self.fail(
                    None,
                    None,
                    format!("{name}: declared in a program out of range"),
                );
            }
        }
        for (i, f) in m.funcs.iter().enumerate() {
            let missing = f.body.is_none() && !matches!(f.kind, FuncKind::Extern { .. });
            if missing {
                self.fail(None, None, format!("function {i} has no body"));
            }
            if f.on_blocks() {
                // a plain function; its body typed on blocks, checked by each operation
                if f.kind != FuncKind::Plain {
                    self.fail(
                        None,
                        None,
                        format!("function {i} on blocks: a plain function"),
                    );
                }
                if let Some(body) = &f.body {
                    self.on_blocks_body(BodyRef::Func(crate::types::FuncId(i as u32)), body);
                }
            }
        }
        for (p, prog) in m.programs.iter().enumerate() {
            let pid = ProgramId(p as u32);
            for member in &prog.state {
                self.count(member.count, true, &member.name);
            }
            for run in &prog.runs {
                for (message, _) in &run.on_message {
                    let owner = m.messages.get(message.0 as usize).map(|x| x.program);
                    if owner != Some(Some(pid)) {
                        self.fail(
                            None,
                            None,
                            format!("run {}: message {} not of its program", run.name, message.0),
                        );
                    }
                }
            }
            for chain in &prog.chains {
                for (k, call) in chain.calls.iter().enumerate() {
                    let Some(handler) = prog.handlers.get(call.handler.0 as usize) else {
                        self.fail(
                            None,
                            None,
                            format!("chain {}: call {k} of no handler", chain.name),
                        );
                        continue;
                    };
                    if handler.params.len() != call.args.len() {
                        self.fail(
                            None,
                            None,
                            format!(
                                "chain {}: call {k} has the wrong count of arguments",
                                chain.name
                            ),
                        );
                        continue;
                    }
                    for (arg, want) in call.args.iter().zip(&handler.params) {
                        let got = match arg {
                            CallArg::Param(i) | CallArg::Past(i) => {
                                chain.params.get(*i as usize).cloned()
                            }
                            CallArg::Const(c) => Some(scalar_of(c)),
                        };
                        let past_ok = !matches!(arg, CallArg::Past(_))
                            || matches!(got, Some(Type::Tensor(..)));
                        if !got.as_ref().is_some_and(|g| fits(g, want)) || !past_ok {
                            self.fail(
                                None,
                                None,
                                format!(
                                    "chain {}: an argument of call {k} does not fit",
                                    chain.name
                                ),
                            );
                        }
                    }
                    if let Some(part) = call.part {
                        let what = format!("chain {}: the range of call {k}", chain.name);
                        for end in [part.first, part.end] {
                            if end == Count::GroupSize {
                                self.fail(
                                    None,
                                    None,
                                    format!("{what}: an end of the group's size"),
                                );
                            }
                            self.count(end, false, &what);
                        }
                        if let (Count::Const(a), Count::Const(b)) = (part.first, part.end) {
                            if a >= b {
                                self.fail(None, None, format!("{what}: no rank"));
                            }
                        }
                    }
                }
            }
        }
    }

    fn body(&mut self, r: BodyRef, body: &'m Body, ctx: &Ctx<'m>) {
        let at = |i: usize| Some(InstId(i as u32));
        // the opening: one Arg for each parameter, then Arrived in an `on` body
        let opening = ctx.params.len();
        for i in 0..ctx.params.len() {
            let ok = matches!(body.insts.get(i), Some(inst) if inst.op == Op::Arg(i as u32) && inst.defs == [ValueId(i as u32)]);
            if !ok {
                self.fail(
                    Some(r),
                    at(i),
                    format!("the body does not open with `Arg {i}`"),
                );
            }
        }
        let arrived = matches!(body.insts.get(opening), Some(inst) if inst.op == Op::Arrived);
        if ctx.arrived.is_some() && !arrived {
            self.fail(
                Some(r),
                at(opening),
                "an `on` body does not open with `Arrived` after its Args",
            );
        }
        let mut live = vec![false; body.values.len()];
        let mut next = 0usize;
        let mut frames: Vec<(Open, Vec<ValueId>)> = Vec::new();
        for (i, inst) in body.insts.iter().enumerate() {
            for operand in operands(&inst.op) {
                let ok = live.get(operand.0 as usize).copied().unwrap_or(false);
                if !ok {
                    self.fail(
                        Some(r),
                        at(i),
                        format!("v{} is not defined here", operand.0),
                    );
                }
            }
            // structure first: a loop's values belong to the frame it opens
            match &inst.op {
                Op::If { .. } => frames.push((Open::If { seen_else: false }, Vec::new())),
                Op::ForRange { .. }
                | Op::ForStream { .. }
                | Op::ForZip { .. }
                | Op::ForArrive { .. } => {
                    frames.push((Open::Loop, Vec::new()));
                }
                Op::While => frames.push((Open::While { tested: false }, Vec::new())),
                Op::Else => match frames.last_mut() {
                    Some((Open::If { seen_else }, defined)) if !*seen_else => {
                        *seen_else = true;
                        for v in defined.drain(..) {
                            live[v.0 as usize] = false;
                        }
                    }
                    _ => self.fail(Some(r), at(i), "an `Else` outside an `If`, or a second one"),
                },
                Op::WhileTest { .. } => match frames.last_mut() {
                    Some((Open::While { tested }, _)) if !*tested => *tested = true,
                    _ => self.fail(
                        Some(r),
                        at(i),
                        "a `WhileTest` outside a `While`, or a second one",
                    ),
                },
                Op::End => match frames.pop() {
                    Some((open, defined)) => {
                        if open == (Open::While { tested: false }) {
                            self.fail(Some(r), at(i), "a `While` closed before its `WhileTest`");
                        }
                        for v in defined {
                            live[v.0 as usize] = false;
                        }
                    }
                    None => self.fail(Some(r), at(i), "an `End` with nothing open"),
                },
                _ => {}
            }
            if let Op::Get(l) | Op::Set(l, _) = &inst.op {
                if l.0 as usize >= body.locals.len() {
                    self.fail(Some(r), at(i), format!("local {} out of range", l.0));
                }
            }
            for def in &inst.defs {
                if def.0 as usize != next || next >= body.values.len() {
                    self.fail(Some(r), at(i), format!("v{} is not the next value", def.0));
                    continue;
                }
                next += 1;
                live[def.0 as usize] = true;
                if let Some((_, defined)) = frames.last_mut() {
                    defined.push(*def);
                }
            }
            let ty = |v: &ValueId| body.values.get(v.0 as usize);
            let operand_types: Option<Vec<&Type>> = operands(&inst.op).iter().map(ty).collect();
            let def_types: Option<Vec<&Type>> = inst.defs.iter().map(ty).collect();
            if let (Some(args), Some(defs)) = (operand_types, def_types) {
                if let Err(what) = self.fit(&inst.op, &args, &defs, ctx, body) {
                    self.fail(Some(r), at(i), what);
                }
            }
        }
        if !frames.is_empty() {
            self.fail(
                Some(r),
                None,
                format!("{} constructs left open", frames.len()),
            );
        }
        if next != body.values.len() {
            self.fail(
                Some(r),
                None,
                format!("{} values typed, {next} defined", body.values.len()),
            );
        }
    }

    /// Whether the operation's operands and its values have the types it gives.
    fn fit(
        &self,
        op: &Op,
        a: &[&Type],
        d: &[&Type],
        ctx: &Ctx<'m>,
        body: &Body,
    ) -> Result<(), String> {
        let m = self.module;
        let u32_ = Type::Scalar(ScalarType::U32);
        let int = |i: usize| a.get(i).is_some_and(|t| is_int(t));
        let ints = |from: usize| (from..a.len()).all(&int);
        let exact = |want: &[Type]| -> Result<(), String> {
            if d.len() == want.len() && d.iter().zip(want).all(|(x, y)| fits(x, y)) {
                Ok(())
            } else {
                Err(format!("defines {d:?}, not {want:?}"))
            }
        };
        let need = |ok: bool, what: &str| if ok { Ok(()) } else { Err(what.to_string()) };
        let tile_or_scalar = |t: &Type| matches!(t, Type::Tile(_) | Type::Scalar(_));
        let views =
            |from: usize| (from..a.len()).all(|i| matches!(a[i], Type::View(_) | Type::Items(_)));
        let prog = ctx.program.and_then(|p| self.program(p));
        let member = |id: super::MemberId| prog.and_then(|p| p.state.get(id.0 as usize));
        let element = |id: super::MemberId| {
            member(id).map(|mm| match mm.kind {
                MemberKind::Tiles(f) => Type::Tile(f),
                MemberKind::Scalars(s) => Type::Scalar(s),
            })
        };
        let message_fields = |id: MessageId| m.messages.get(id.0 as usize).map(|mm| &mm.fields);
        let message_body = |id: MessageId| m.messages.get(id.0 as usize).map_or(0, |mm| mm.body);
        // the bytes of the words a value gives, where the structure shows them: a
        // message's body, the arrived message's; the words of a body or of a view, the
        // body's or a tile's of `u32`
        let value_bytes = |t: &Type, x: ValueId| -> Option<u64> {
            let arrived = ctx.arrived.map(|a| message_body(a) as u64);
            match t {
                Type::Body => arrived,
                Type::Words(_) => {
                    let def = body.insts.iter().find(|inst| inst.defs.contains(&x))?;
                    let Op::Words(v) = def.op else {
                        return None;
                    };
                    match body.values.get(v.0 as usize)? {
                        Type::Body => arrived,
                        Type::View(_) => Some(4 * m.tile.elements()),
                        _ => None,
                    }
                }
                _ => None,
            }
        };
        // `body = src` of a message of type `mm`: given only to a type with a body; a
        // message's body, the words of `u32`, or a state member of `u32`; no more bytes
        // than the type's body where the structure shows them
        let body_fits = |mm: MessageId, src: &Option<ScalarsFrom>, t: Option<&Type>| {
            let Some(src) = src else {
                return Ok(());
            };
            let room = message_body(mm) as u64;
            if room == 0 {
                return Err(format!(
                    "a body given to message {}, which takes none",
                    mm.0
                ));
            }
            let bytes = match (src, t) {
                (ScalarsFrom::Value(x), Some(t @ (Type::Body | Type::Words(ScalarType::U32)))) => {
                    value_bytes(t, *x)
                }
                (ScalarsFrom::Member(id), _) => match member(*id) {
                    Some(mm) if mm.kind == MemberKind::Scalars(ScalarType::U32) => match mm.count {
                        Count::Const(k) => Some(4 * k),
                        _ => None,
                    },
                    _ => return Err("a body from a member that is not of u32 scalars".into()),
                },
                _ => return Err("a body from a value that is not a body or u32 words".into()),
            };
            match bytes {
                Some(b) if b > room => Err(format!(
                    "a body of {b} bytes given to message {}, which takes {room}",
                    mm.0
                )),
                _ => Ok(()),
            }
        };
        let combiner = |c: &Combiner| match c {
            Combiner::Func(f) => {
                matches!(m.funcs.get(f.0 as usize), Some(f) if matches!(f.kind, FuncKind::Combine { .. }))
            }
            _ => true,
        };
        match op {
            Op::Const(c) => exact(&[scalar_of(c)]),
            Op::Arg(i) => match ctx.params.get(*i as usize) {
                Some(t) => exact(std::slice::from_ref(t)),
                None => Err(format!("argument {i} out of range")),
            },
            Op::Param(p) => match m.params.get(p.0 as usize) {
                Some(param) if !matches!(param.binding, Binding::Build(_)) => {
                    exact(&[Type::Scalar(param.ty)])
                }
                Some(_) => Err("a build parameter read as a parameter, not a constant".into()),
                None => Err(format!("parameter {} out of range", p.0)),
            },
            Op::Get(l) => exact(
                body.locals
                    .get(l.0 as usize)
                    .map(std::slice::from_ref)
                    .unwrap_or(&[]),
            ),
            Op::Set(l, _) => need(
                body.locals.get(l.0 as usize).is_some_and(|l| fits(a[0], l)) && d.is_empty(),
                "a `Set` of another type",
            ),
            Op::Binary(op, _, _) => {
                let (x, y) = (a[0], a[1]);
                let r = d.first().copied();
                let ok = match (x, y) {
                    // a handler's `+=` into its var of a block of one row (R48)
                    (Type::Block(_, Count::Const(1), c), Type::Block(_, Count::Const(1), e))
                        if !ctx.on_blocks && *op == BinOp::Add =>
                    {
                        c == e && matches!(r, Some(Type::Block(_, Count::Const(1), g)) if g == c)
                    }
                    (Type::Scalar(s), Type::Scalar(t)) if !s.is_float() || !t.is_float() => {
                        s == t && r == Some(x)
                    }
                    (Type::Scalar(_), Type::Scalar(_)) => {
                        matches!(
                            op,
                            BinOp::Add
                                | BinOp::Sub
                                | BinOp::Mul
                                | BinOp::Div
                                | BinOp::Min
                                | BinOp::Max
                        ) && matches!(r, Some(Type::Scalar(_)))
                    }
                    // a reduction's values with any operation; tiles and blocks, arithmetic
                    _ => match elementwise(ctx.on_blocks, &[x, y], d) {
                        Some(reduced) => {
                            reduced
                                || matches!(
                                    op,
                                    BinOp::Add
                                        | BinOp::Sub
                                        | BinOp::Mul
                                        | BinOp::Div
                                        | BinOp::Min
                                        | BinOp::Max
                                )
                        }
                        None => false,
                    },
                };
                need(
                    ok && d.len() == 1,
                    "the operands of a binary operation do not fit",
                )
            }
            Op::Compare(..) => {
                let ok = matches!(
                    (a[0], a[1]),
                    (Type::Scalar(_), Type::Scalar(_))
                        | (Type::Bool, Type::Bool)
                        | (Type::Reason, Type::Reason)
                );
                need(ok, "a comparison of values that do not compare")?;
                exact(&[Type::Bool])
            }
            Op::LogicAnd(..) | Op::LogicOr(..) => {
                need(
                    a.iter().all(|t| **t == Type::Bool),
                    "a condition that is not a bool",
                )?;
                exact(&[Type::Bool])
            }
            Op::If { .. } | Op::WhileTest { .. } => {
                need(*a[0] == Type::Bool, "a condition that is not a bool")?;
                exact(&[])
            }
            Op::ForRange { .. } => {
                need(int(0), "`range` of a value that is not an integer")?;
                exact(&[a[0].clone()])
            }
            Op::ForStream { .. } => match a[0] {
                Type::Stream(f, r, c) => exact(&[of_shape(*f, (*r, *c))]),
                _ => Err("`for` over a value that is not a stream".into()),
            },
            Op::ForZip { .. } => {
                let blocks: Option<Vec<Type>> = a
                    .iter()
                    .map(|t| match t {
                        Type::Stream(f, r, c) => Some(of_shape(*f, (*r, *c))),
                        _ => None,
                    })
                    .collect();
                match blocks {
                    Some(blocks) if blocks.len() >= 2 => exact(&blocks),
                    _ => Err("`zip` of other than two streams or more".into()),
                }
            }
            Op::ForArrive { .. } => need(
                matches!((a[0], d.first()), (Type::Broadcast(f), Some(Type::Tile(g) | Type::Block(g, ..))) if f == g)
                    && d.len() == 2
                    && matches!(d[1], Type::Scalar(_)),
                "`arrive` of a value that is not a broadcast",
            ),
            Op::While | Op::Else | Op::End => exact(&[]),
            Op::Return(v) => {
                let ok = match (v, &ctx.ret) {
                    (None, _) => true,
                    (Some(_), Some(t)) => fits(a[0], t),
                    (Some(_), None) => false,
                };
                need(
                    ok && d.is_empty(),
                    "a `return` of another type than the result",
                )
            }
            Op::Convert(to, _) => match a[0] {
                Type::Scalar(_) => exact(&[Type::Scalar(*to)]),
                _ => Err("a conversion of what is not a scalar".into()),
            },
            Op::Unary(..) => match (a[0], d) {
                (Type::Scalar(s), [Type::Scalar(t)]) if scalar_fits(*s, *t) => Ok(()),
                _ if elementwise(ctx.on_blocks, &[a[0]], d).is_some() => Ok(()),
                _ => Err("a function of one tile on another value".into()),
            },
            Op::Select { .. } => need(
                a.iter()
                    .all(|t| tile_or_scalar(t) || ctx.on_blocks && shape_of(t).is_some())
                    && elementwise(ctx.on_blocks, a, d) == Some(false),
                "`where` on values that are not tiles or scalars",
            ),
            Op::Reduce(_, _, axis) => match a[0] {
                Type::Tile(f) => {
                    exact(&[Type::Reduced(*f, *axis, Count::Const(1), Count::Const(1))])
                }
                Type::Block(f, r, c) if ctx.on_blocks => exact(&[Type::Reduced(*f, *axis, *r, *c)]),
                _ => Err("a reduction of a value that is not a tile".into()),
            },
            Op::SpreadCol(_) | Op::SpreadRow(_) => {
                let axis = if matches!(op, Op::SpreadCol(_)) {
                    crate::types::Axis::Row
                } else {
                    crate::types::Axis::Col
                };
                match a[0] {
                    Type::Tile(f) => exact(&[Type::Tile(*f)]),
                    Type::Block(..) if ctx.on_blocks => exact(&[a[0].clone()]),
                    Type::Reduced(f, x, r, c) if *x == axis => exact(&[of_shape(*f, (*r, *c))]),
                    _ => Err("`.col()` or `.row()` of a value it does not apply to".into()),
                }
            }
            // of blocks (R45): a `[r, k]` by a `[k, c]` gives a `[r, c]`
            Op::Matmul(..) => {
                let shapes = (shape_of(a[0]), shape_of(a[1]), d);
                let ok = match shapes {
                    (Some((r, k)), Some((k2, c)), [t]) if ctx.on_blocks => {
                        let tile = (r, c) == (Count::Const(1), Count::Const(1));
                        k == k2 && shape_of(t) == Some((r, c)) && matches!(t, Type::Tile(_)) == tile
                    }
                    _ => matches!(
                        (a[0], a[1], d),
                        (Type::Tile(_), Type::Tile(_), [Type::Tile(_)])
                    ),
                };
                need(
                    ok,
                    "`matmul` of values that are not tiles, or blocks that do not fit",
                )
            }
            Op::Transpose(_) => need(
                match (a[0], d) {
                    (Type::Tile(f), [Type::Tile(g)]) => f == g,
                    (Type::Block(f, r, c), [Type::Block(g, s, t)]) => {
                        ctx.on_blocks && f == g && (r, c) == (t, s)
                    }
                    _ => false,
                },
                "`.T` of a value that is not a tile",
            ),
            Op::Zeros => need(
                matches!(d, [Type::Tile(_) | Type::Block(..) | Type::Scalar(_)]),
                "`zeros` of a type that has none",
            ),
            Op::CallFunc(f, _) => match m.funcs.get(f.0 as usize) {
                Some(func) => {
                    let ok = func.params.len() == a.len()
                        && func.params.iter().zip(a).all(|(p, x)| fits(x, p));
                    // a handler's vars of a block of one row given to it (R48)
                    let rows_of_one = a.iter().all(|x| {
                        !matches!(x, Type::Block(..))
                            || matches!(x, Type::Block(_, Count::Const(1), _))
                    });
                    if !ok && (ctx.on_blocks || rows_of_one) && !func.on_blocks() {
                        // a function on tiles on each tile of blocks of one shape (R45)
                        return self.on_each_tile(func, a, d);
                    }
                    need(ok, "a call whose arguments do not fit")?;
                    exact(std::slice::from_ref(&func.ret))
                }
                None => Err(format!("function {} out of range", f.0)),
            },
            Op::Tiles { .. } | Op::Rect { .. } => match a[0] {
                Type::Tensor(f, _) if ints(1) => exact(&[Type::View(Some(*f))]),
                _ => Err("a view of a value that is not a tensor".into()),
            },
            Op::Turn(_) => need(
                matches!((a[0], d), (Type::View(f), [Type::View(g)]) if f == g),
                "`.T` of a value that is not a view",
            ),
            Op::Items { .. } => need(
                matches!(a[0], Type::Tensor(..)) && ints(1) && matches!(d, [Type::Items(_)]),
                "items of a value that is not a tensor",
            ),
            Op::ReadTile(..) => match a[0] {
                Type::View(Some(f)) if int(1) => exact(&[Type::Tile(*f)]),
                // through a view of no format, the tile has its place's format
                Type::View(None) if int(1) => match d {
                    [Type::Tile(_)] => Ok(()),
                    _ => Err("`v[i]` that gives no tile".into()),
                },
                _ => Err("`v[i]` of a value that is not a view".into()),
            },
            Op::Words(_) => match a[0] {
                Type::Body => exact(&[Type::Words(ScalarType::U32)]),
                Type::View(f) => match f.and_then(Format::as_scalar) {
                    Some(s) => exact(&[Type::Words(s)]),
                    None => Err("`.words()` of a view whose format is no scalar type".into()),
                },
                _ => Err("`.words()` of a value that is not a view".into()),
            },
            Op::Word(..) => match a[0] {
                Type::Words(s) if int(1) => exact(&[Type::Scalar(*s)]),
                _ => Err("`words[i]` of a value that is not words".into()),
            },
            Op::Records(_, rec, _) => {
                need(
                    (rec.0 as usize) < m.records.len(),
                    "a record type out of range",
                )?;
                need(
                    matches!(a[0], Type::View(_)) && int(1),
                    "`.records` of a value that is not a view",
                )?;
                exact(&[Type::Records(*rec)])
            }
            Op::RecordAt(..) => match a[0] {
                Type::Records(rec) if int(1) => exact(&[Type::RecordValue(*rec)]),
                _ => Err("`records[i]` of a value that is not records".into()),
            },
            Op::RecordField(_, k) => match a[0] {
                Type::RecordValue(rec) => match m
                    .records
                    .get(rec.0 as usize)
                    .and_then(|r| r.fields.get(*k as usize))
                {
                    Some((_, s)) => exact(&[Type::Scalar(*s)]),
                    None => Err(format!("field {k} out of range")),
                },
                _ => Err("a field of a value that is not a record".into()),
            },
            Op::Put { src, .. } => need(
                matches!(a[0], Type::Items(_)) && int(1) && member(*src).is_some() && d.is_empty(),
                "a `put` that does not fit",
            ),
            Op::Partition(..) => need(
                matches!(a[0], Type::View(_) | Type::Items(_)) && ints(1) && d == [a[0]],
                "a `partition` that does not fit",
            ),
            Op::Stream(..) | Op::Sink(..) => match (a[0], d) {
                (Type::View(f), [Type::Stream(g, ..)])
                    if matches!(op, Op::Stream(..)) && *f == Some(*g) && int(1) =>
                {
                    Ok(())
                }
                (Type::View(f), [Type::Sink(g, ..)])
                    if matches!(op, Op::Sink(..)) && *f == Some(*g) && int(1) =>
                {
                    Ok(())
                }
                _ => Err("a stream or a sink that does not fit".into()),
            },
            Op::Push(..) | Op::Store(..) => {
                let target = if matches!(op, Op::Push(..)) {
                    matches!(a[0], Type::Sink(..))
                } else {
                    matches!(a[0], Type::View(_))
                };
                need(
                    target && matches!(a[1], Type::Tile(_) | Type::Block(..)) && d.is_empty(),
                    "a write that does not fit",
                )
            }
            Op::BlockAt(..) => match a[0] {
                Type::Block(f, Count::Const(1), _) if int(1) => exact(&[Type::Tile(*f)]),
                _ => Err("`block[i]` of a value that is not a block".into()),
            },
            Op::Fill(mm, _) => need(
                matches!(member(*mm).map(|x| &x.kind), Some(MemberKind::Tiles(_)))
                    && matches!(a[0], Type::Stream(..))
                    && d.is_empty(),
                "a `fill` that does not fit",
            ),
            Op::ZeroState(mm) => need(member(*mm).is_some() && d.is_empty(), "state out of range"),
            Op::StateSet(mm, _, _) => need(
                int(0) && element(*mm).as_ref().is_some_and(|e| fits(a[1], e)) && d.is_empty(),
                "a state write that does not fit",
            ),
            Op::StateGet(mm, _) => match element(*mm) {
                Some(t) if int(0) => exact(&[t]),
                _ => Err("a state read that does not fit".into()),
            },
            Op::Bump(mm, _) => match element(*mm) {
                Some(t) if is_int(&t) && int(0) => exact(&[t]),
                _ => Err("a `bump` of a member that is not of integers".into()),
            },
            Op::Rank | Op::WorkerId | Op::Row | Op::Col | Op::GroupSize | Op::IslandCount => {
                exact(&[u32_])
            }
            Op::GroupId(_) | Op::IslandOf(_) | Op::IslandHops(_) => {
                need(int(0), "an id that is not an integer")?;
                exact(&[u32_])
            }
            Op::IsLeader => exact(&[Type::Bool]),
            Op::IslandCrosses(_) => {
                need(int(0), "an id that is not an integer")?;
                exact(&[Type::Bool])
            }
            Op::Split(..) => {
                need(int(0), "a split by a value that is not an integer")?;
                exact(&[Type::SubGroups])
            }
            Op::Leaders(..) => {
                need(int(0), "a split by a value that is not an integer")?;
                exact(&[Type::Leaders])
            }
            Op::ReduceGroup {
                with,
                within,
                among,
                publish,
                ..
            } => {
                let mut k = 1;
                let mut ok = combiner(with);
                if within.is_some() {
                    ok &= *a[k] == Type::SubGroups;
                    k += 1;
                }
                if among.is_some() {
                    ok &= *a[k] == Type::Leaders;
                    k += 1;
                }
                ok &= views(k) && publish.len() == a.len() - k;
                need(ok, "a `reduce` whose operands do not fit")?;
                exact(&[a[0].clone()])
            }
            Op::AllReduce { with, .. } => {
                need(
                    combiner(with) && views(1),
                    "an `all_reduce` whose operands do not fit",
                )?;
                exact(&[a[0].clone()])
            }
            Op::Gather(_) => match a[0] {
                Type::Tile(f) => exact(&[Type::Block(*f, Count::Const(1), Count::GroupSize)]),
                _ => Err("`gather` of a value that is not a tile".into()),
            },
            Op::AllGather { publish, .. } => match a[0] {
                Type::Tile(f) => {
                    need(views(1), "an `all_gather` whose operands do not fit")?;
                    exact(&[Type::Block(*f, Count::Const(1), Count::GroupSize)])
                }
                Type::Tensor(_, true) => need(
                    publish.is_empty() && d.is_empty(),
                    "an `all_gather` over copies that does not fit",
                ),
                _ => Err(
                    "`all_gather` of a value that is neither a tile nor a replicated tensor".into(),
                ),
            },
            Op::Broadcast(..) => {
                need(int(0), "a sender that is not an integer")?;
                need(
                    matches!(d, [Type::Broadcast(_)]),
                    "a broadcast that does not give one broadcast",
                )
            }
            Op::IsSender(_) => {
                need(
                    matches!(a[0], Type::Broadcast(_)),
                    "`is_sender` of a value that is not a broadcast",
                )?;
                exact(&[Type::Bool])
            }
            Op::Cast(..) => need(
                matches!((a[0], a[1]), (Type::Broadcast(f), Type::Tile(g) | Type::Block(g, ..)) if f == g)
                    && (a.len() == 2 || matches!(a[2], Type::Scalar(_)))
                    && d.is_empty(),
                "a `cast` that does not fit",
            ),
            Op::Sync(_) => need(
                views(0) && d.is_empty(),
                "a `sync` of values that are not views",
            ),
            Op::Arrived => match ctx.arrived {
                Some(mm) => exact(&[Type::Message(mm)]),
                None => Err("`Arrived` outside an `on` body".into()),
            },
            Op::MakeMessage(mm, _) => match message_fields(*mm) {
                Some(fields) => {
                    let ok = fields.len() == a.len()
                        && fields
                            .iter()
                            .zip(a)
                            .all(|((_, s), t)| fits(t, &Type::Scalar(*s)));
                    need(ok, "a message whose fields do not fit")?;
                    exact(&[Type::Message(*mm)])
                }
                None => Err(format!("message {} out of range", mm.0)),
            },
            Op::Send {
                body: src, publish, ..
            } => {
                let k = 2 + usize::from(matches!(src, Some(ScalarsFrom::Value(_))));
                need(
                    matches!(a[0], Type::Message(_))
                        && int(1)
                        && a.len() >= k
                        && views(k)
                        && publish.len() == a.len() - k,
                    "a `send` that does not fit",
                )?;
                if let Type::Message(mm) = a[0] {
                    body_fits(*mm, src, a.get(2).copied())?;
                }
                exact(&[Type::Bool])
            }
            Op::Forward { as_type, with, .. } => {
                let Type::Message(mm) = a[0] else {
                    return Err("a `forward` of a value that is not a message".into());
                };
                let target = as_type.unwrap_or(*mm);
                let fields = message_fields(target).ok_or("message out of range")?;
                let ok = with.iter().enumerate().all(|(k, (f, _))| {
                    fields
                        .get(*f as usize)
                        .is_some_and(|(_, s)| fits(a[1 + k], &Type::Scalar(*s)))
                });
                need(
                    ok && int(1 + with.len()) && views(2 + with.len()) && d.is_empty(),
                    "a `forward` that does not fit",
                )?;
                if as_type.is_some() {
                    // every field of the target once, in its order
                    let each = with.len() == fields.len()
                        && with.iter().enumerate().all(|(k, (f, _))| *f as usize == k);
                    need(
                        each,
                        "a `forward` as another type that does not give each of its fields once",
                    )?;
                    let (from, to) = (message_body(*mm), message_body(target));
                    need(
                        from == 0 || to >= from,
                        "a `forward` as another type whose body does not fit it",
                    )?;
                }
                Ok(())
            }
            Op::Consume(_) => need(
                matches!(a[0], Type::Message(_)) && d.is_empty(),
                "`consume` of a value that is not a message",
            ),
            Op::Offer(_, _, src) => {
                let k = 2 + usize::from(matches!(src, Some(ScalarsFrom::Value(_))));
                need(
                    matches!(a[0], Type::Message(_)) && int(1) && a.len() == k,
                    "an `offer` that does not fit",
                )?;
                if let Type::Message(mm) = a[0] {
                    body_fits(*mm, src, a.get(2).copied())?;
                }
                exact(&[Type::Bool])
            }
            Op::Body(_) => match (a[0], ctx.arrived) {
                (Type::Message(mm), Some(arrived)) if *mm == arrived => {
                    need(message_body(*mm) > 0, "the body of a message that has none")?;
                    exact(&[Type::Body])
                }
                _ => Err("the body of a value that is not the arrived message".into()),
            },
            Op::Field(_, k) => match a[0] {
                Type::Message(mm) => match message_fields(*mm).and_then(|f| f.get(*k as usize)) {
                    Some((_, s)) => exact(&[Type::Scalar(*s)]),
                    None => Err(format!("field {k} out of range")),
                },
                _ => Err("a field of a value that is not a message".into()),
            },
            Op::MakeGroup(..) => {
                need(ints(0), "a group of values that are not integers")?;
                exact(&[Type::Group])
            }
            Op::Invoke {
                entry,
                args,
                after,
                after_end,
                ..
            } => {
                let (params, ret) = self.entry(*entry).ok_or("an invocation of nothing")?;
                let n = args.len();
                let ok = params.len() == n
                    && params.iter().zip(a).all(|(p, x)| fits(x, p))
                    && *a[n] == Type::Group
                    && a[n + 1..].iter().all(|t| matches!(t, Type::Handle(_)))
                    && a.len() == n + 1 + after.len() + after_end.len();
                need(ok, "an `invoke` whose operands do not fit")?;
                exact(&[Type::Handle(ret)])
            }
            Op::Await(_) => match a[0] {
                Type::Handle(Some(s)) => exact(&[Type::Scalar(*s)]),
                Type::Handle(None) => exact(&[]),
                _ => Err("`await` of a value that is not a handle".into()),
            },
            Op::Outcome(_) => match a[0] {
                Type::Handle(r) => exact(&[Type::Outcome(*r)]),
                _ => Err("`outcome` of a value that is not a handle".into()),
            },
            Op::Failed(_) | Op::Reason(_) | Op::OutcomeValue(_) => {
                let Type::Outcome(r) = a[0] else {
                    return Err("a part of a value that is not an outcome".into());
                };
                match op {
                    Op::Failed(_) => exact(&[Type::Bool]),
                    Op::Reason(_) => exact(&[Type::Reason]),
                    _ => match r {
                        Some(s) => exact(&[Type::Scalar(*s)]),
                        None => Err("`.value` of an outcome with no result".into()),
                    },
                }
            }
            Op::FailureName(_) => exact(&[Type::Reason]),
            Op::ReduceTo(_, _, with) => need(
                matches!(a[0], Type::Tensor(_, true)) && int(1) && combiner(with) && d.is_empty(),
                "a `reduce_to` that does not fit",
            ),
            Op::BroadcastFrom(..) => need(
                matches!(a[0], Type::Tensor(_, true)) && int(1) && d.is_empty(),
                "a `broadcast_from` that does not fit",
            ),
        }
    }

    /// The parameters and the result of what an invocation names.
    fn entry(&self, e: Entry) -> Option<(&'m [Type], Option<ScalarType>)> {
        match e {
            Entry::Handler(p, h) => {
                let h = self.program(p)?.handlers.get(h.0 as usize)?;
                Some((&h.params, h.ret))
            }
            Entry::Run(p, r) => Some((&self.program(p)?.runs.get(r.0 as usize)?.params, None)),
            Entry::Chain(p, c) => Some((&self.program(p)?.chains.get(c.0 as usize)?.params, None)),
        }
    }

    fn inst_exists(&self, r: BodyRef, i: InstId) -> bool {
        self.body_of(r)
            .is_some_and(|b| (i.0 as usize) < b.insts.len())
    }

    fn effects(&mut self, members: usize, params: usize, e: &Effects) {
        for access in &e.tensors {
            let ok = (access.param as usize) < params
                && self.inst_exists(access.body, access.view)
                && self.inst_exists(access.body, access.at);
            if !ok {
                self.fail(
                    Some(access.body),
                    Some(access.at),
                    "a tensor access that names nothing",
                );
            }
        }
        for s in &e.state {
            if s.member.0 as usize >= members || !self.inst_exists(s.body, s.at) {
                self.fail(Some(s.body), Some(s.at), "a state use that names nothing");
            }
        }
        for (b, i) in &e.collectives {
            if !self.inst_exists(*b, *i) {
                self.fail(Some(*b), Some(*i), "a collective that names nothing");
            }
        }
        for f in &e.calls {
            if f.0 as usize >= self.module.funcs.len() {
                self.fail(None, None, format!("function {} out of range", f.0));
            }
        }
    }

    fn derived(&mut self) {
        let m = self.module;
        for f in &m.funcs {
            self.effects(0, f.params.len(), &f.effects);
        }
        for prog in &m.programs {
            let members = prog.state.len();
            for h in &prog.handlers {
                self.effects(members, h.params.len(), &h.effects);
            }
            for r in &prog.runs {
                self.effects(members, r.params.len(), &r.effects);
            }
            for c in &prog.chains {
                for e in &c.deps.edges {
                    let ok = e.writer < e.reader
                        && (e.reader as usize) < c.calls.len()
                        && (e.param as usize) < c.params.len();
                    if !ok {
                        self.fail(
                            None,
                            None,
                            format!("chain {}: an edge that names nothing", c.name),
                        );
                    }
                }
            }
        }
        for req in &m.requirements {
            for cause in &req.causes {
                let ok = match cause {
                    Cause::Stated => true,
                    Cause::GridHandler(p, h) => self
                        .program(*p)
                        .is_some_and(|x| (h.0 as usize) < x.handlers.len()),
                    Cause::DynamicRun(e) => self.entry(*e).is_some(),
                    Cause::ExternWithoutFallback(f) => (f.0 as usize) < m.funcs.len(),
                };
                if !ok {
                    self.fail(None, None, "a requirement's cause that names nothing");
                }
            }
        }
    }

    fn owner_params(&self, o: Owner) -> Option<usize> {
        let m = self.module;
        Some(match o {
            Owner::Func(f) => m.funcs.get(f.0 as usize)?.params.len(),
            Owner::Handler(p, h) => self.program(p)?.handlers.get(h.0 as usize)?.params.len(),
            Owner::Run(p, r) => self.program(p)?.runs.get(r.0 as usize)?.params.len(),
            Owner::Chain(p, c) => self.program(p)?.chains.get(c.0 as usize)?.params.len(),
            Owner::Control(c) => m.controls.get(c.0 as usize)?.params.len(),
        })
    }

    fn anchor_exists(&self, a: Anchor) -> bool {
        let m = self.module;
        let prog = |p: ProgramId| self.program(p);
        match a {
            Anchor::Module => true,
            Anchor::Func(f) => (f.0 as usize) < m.funcs.len(),
            Anchor::Program(p) => prog(p).is_some(),
            Anchor::Handler(p, h) => prog(p).is_some_and(|x| (h.0 as usize) < x.handlers.len()),
            Anchor::Run(p, r) => prog(p).is_some_and(|x| (r.0 as usize) < x.runs.len()),
            Anchor::Member(p, s) => prog(p).is_some_and(|x| (s.0 as usize) < x.state.len()),
            Anchor::Control(c) => (c.0 as usize) < m.controls.len(),
            Anchor::Chain(p, c) => prog(p).is_some_and(|x| (c.0 as usize) < x.chains.len()),
            Anchor::Call(p, c, k) => prog(p)
                .and_then(|x| x.chains.get(c.0 as usize))
                .is_some_and(|x| (k as usize) < x.calls.len()),
            Anchor::Param(p) => (p.0 as usize) < m.params.len(),
            Anchor::Message(s) => (s.0 as usize) < m.messages.len(),
            Anchor::Record(r) => (r.0 as usize) < m.records.len(),
            Anchor::ParamOf(o, i) => self.owner_params(o).is_some_and(|n| (i as usize) < n),
            Anchor::Inst(b, i) => self.inst_exists(b, i),
        }
    }

    fn notes(&mut self) {
        let notes = &self.module.notes;
        let anchors: Vec<(&str, Vec<Anchor>)> = vec![
            ("hints", notes.hints.iter().map(|h| h.anchor).collect()),
            ("spans", notes.spans.iter().map(|(a, _)| *a).collect()),
            (
                "param_names",
                notes.param_names.iter().map(|(a, _)| *a).collect(),
            ),
        ];
        let mut names = Vec::new();
        for n in &notes.names {
            let (b, key, ok) = match n {
                Named::Value(b, v, _) => (
                    *b,
                    (*b, 0, v.0),
                    self.body_of(*b)
                        .is_some_and(|x| (v.0 as usize) < x.values.len()),
                ),
                Named::Local(b, l, _) => (
                    *b,
                    (*b, 1, l.0),
                    self.body_of(*b)
                        .is_some_and(|x| (l.0 as usize) < x.locals.len()),
                ),
            };
            if !ok {
                self.fail(
                    Some(b),
                    None,
                    "a name of a value or a slot that does not exist",
                );
            }
            names.push(key);
        }
        if names.windows(2).any(|w| w[0] >= w[1]) {
            self.fail(
                None,
                None,
                "names: not in the order of their bodies, values then slots, or named twice",
            );
        }
        for (list, keys) in anchors {
            for a in &keys {
                if !self.anchor_exists(*a) {
                    self.fail(
                        None,
                        None,
                        format!("{list}: an anchor that names nothing: {a:?}"),
                    );
                }
            }
            if keys.windows(2).any(|w| w[0] > w[1]) {
                self.fail(
                    None,
                    None,
                    format!("{list}: not in the order of their anchors"),
                );
            }
        }
        for (a, _) in &notes.param_names {
            if !matches!(a, Anchor::ParamOf(..)) {
                self.fail(None, None, "a parameter's name not on a parameter");
            }
        }
        self.origin();
    }

    /// The origin: every place names an anchor that exists, once and in their order, a
    /// file of the list, and lines from 1, first to last, each held in `lines`; the lines
    /// in (file, number) order, each once, of a file of the list.
    fn origin(&mut self) {
        let Some(origin) = &self.module.notes.origin else {
            return;
        };
        let files = origin.files.len();
        let mut faults = Vec::new();
        for (file, number, _) in &origin.lines {
            if *file as usize >= files {
                faults.push(format!("origin: a line of file {file}, past the files"));
            }
            if *number == 0 {
                faults.push(format!("origin: line 0 of file {file}: lines count from 1"));
            }
        }
        if origin
            .lines
            .windows(2)
            .any(|w| (w[0].0, w[0].1) >= (w[1].0, w[1].1))
        {
            faults.push("origin: lines not in (file, number) order, or a line twice".into());
        }
        for (anchor, file, first, last) in &origin.places {
            if !self.anchor_exists(*anchor) {
                faults.push(format!("origin: an anchor that names nothing: {anchor:?}"));
            }
            if *file as usize >= files {
                faults.push(format!("origin: {anchor:?} in file {file}, past the files"));
            }
            if *first == 0 || first > last {
                faults.push(format!(
                    "origin: {anchor:?} on lines {first} to {last}: lines count from 1, first to last"
                ));
                continue;
            }
            let held = |n: u32| {
                origin
                    .lines
                    .binary_search_by(|(f, k, _)| (*f, *k).cmp(&(*file, n)))
                    .is_ok()
            };
            // Of more consecutive lines than are held, one is not: the walk stops there,
            // so a hostile range is not walked whole.
            let missing = (*first..=*last)
                .take(origin.lines.len() + 1)
                .find(|n| !held(*n));
            if let Some(n) = missing {
                faults.push(format!(
                    "origin: {anchor:?} covers line {n} of file {file}, which is not held"
                ));
            }
        }
        if origin.places.windows(2).any(|w| w[0].0 >= w[1].0) {
            faults.push("origin: places not in the order of their anchors, or one twice".into());
        }
        for what in faults {
            self.fail(None, None, what);
        }
    }
}

/// The values an operation reads, in the order its operands are written.
pub fn operands(op: &Op) -> Vec<ValueId> {
    use Op::*;
    let mut v = Vec::new();
    match op {
        Const(_) | Arg(_) | Param(_) | Get(_) | Else | While | End | Zeros | Rank | IsLeader
        | WorkerId | Row | Col | GroupSize | Arrived | FailureName(_) | IslandCount
        | ZeroState(_) => {}
        Set(_, a)
        | Convert(_, a)
        | Unary(_, a)
        | Reduce(_, a, _)
        | SpreadCol(a)
        | SpreadRow(a)
        | Transpose(a)
        | Turn(a)
        | Words(a)
        | RecordField(a, _)
        | GroupId(a)
        | Split(_, a)
        | Leaders(_, a)
        | Gather(a)
        | Broadcast(_, a)
        | IsSender(a)
        | Consume(a)
        | Field(a, _)
        | Body(a)
        | Await(a)
        | Outcome(a)
        | Failed(a)
        | Reason(a)
        | OutcomeValue(a)
        | IslandOf(a)
        | IslandCrosses(a)
        | IslandHops(a)
        | Fill(_, a)
        | StateGet(_, a)
        | Bump(_, a) => v.push(*a),
        If { cond } | WhileTest { cond } => v.push(*cond),
        ForRange { n } => v.push(*n),
        ForStream { stream } => v.push(*stream),
        ForArrive { broadcast } => v.push(*broadcast),
        ForZip { streams } => v.extend(streams.iter().copied()),
        Binary(_, a, b)
        | Compare(_, a, b)
        | LogicAnd(a, b)
        | LogicOr(a, b)
        | Matmul(a, b)
        | ReadTile(a, b)
        | Word(a, b)
        | RecordAt(a, b)
        | Stream(a, b)
        | Sink(a, b)
        | Push(a, b)
        | Store(a, b)
        | BlockAt(a, b)
        | StateSet(_, a, b)
        | BroadcastFrom(a, b)
        | ReduceTo(a, b, _) => {
            v.push(*a);
            v.push(*b);
        }
        Partition(a, b, c) => v.extend([*a, *b, *c]),
        Return(a) => v.extend(a.iter().copied()),
        Select { a, b, x, y, .. } => v.extend([*a, *b, *x, *y]),
        CallFunc(_, args) | MakeMessage(_, args) | Sync(args) => v.extend(args.iter().copied()),
        Tiles { tensor, first, n } => v.extend([*tensor, *first, *n]),
        Rect { tensor, rows, cols } => {
            v.push(*tensor);
            for r in [rows, cols] {
                v.extend([r.lo, r.hi]);
                v.extend(r.step);
            }
        }
        Items {
            tensor,
            range,
            size,
        } => {
            v.push(*tensor);
            if let Some((f, n)) = range {
                v.extend([*f, *n]);
            }
            v.push(*size);
        }
        Records(a, _, c) => v.extend([*a, *c]),
        Put { items, at, .. } => v.extend([*items, *at]),
        ReduceGroup {
            x,
            within,
            among,
            publish,
            ..
        } => {
            v.push(*x);
            v.extend(within.iter().chain(among.iter()).copied());
            v.extend(publish.iter().copied());
        }
        AllReduce { x, publish, .. } => {
            v.push(*x);
            v.extend(publish.iter().copied());
        }
        AllGather { value, publish } => {
            v.push(*value);
            v.extend(publish.iter().copied());
        }
        Cast(b, x, w) => {
            v.extend([*b, *x]);
            v.extend(w.iter().copied());
        }
        Send {
            message,
            to,
            body,
            publish,
            ..
        } => {
            v.extend([*message, *to]);
            if let Some(ScalarsFrom::Value(x)) = body {
                v.push(*x);
            }
            v.extend(publish.iter().copied());
        }
        Offer(a, b, body) => {
            v.extend([*a, *b]);
            if let Some(ScalarsFrom::Value(x)) = body {
                v.push(*x);
            }
        }
        Forward {
            message,
            with,
            to,
            publish,
            ..
        } => {
            v.push(*message);
            v.extend(with.iter().map(|(_, x)| *x));
            v.push(*to);
            v.extend(publish.iter().copied());
        }
        MakeGroup(first, count) => {
            v.extend(first.iter().copied());
            v.push(*count);
        }
        Invoke {
            args,
            on,
            after,
            after_end,
            ..
        } => {
            v.extend(args.iter().copied());
            v.push(*on);
            v.extend(after.iter().chain(after_end.iter()).copied());
        }
    }
    v
}
