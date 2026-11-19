//! The calls, members and indexes of a body: which operation each one is.
//!
//! This reads the forms of the text. It stands in for the checker's `operation(&Expr)`,
//! which will name the operation of every call and member expression; when that lands,
//! this file walks it instead and reads no form a second time.

use super::body::{Slot, B};
use super::{refuse, Res};
use crate::program::body::{Op, ValueId};
use crate::program::MemberKind;
use crate::syntax::ast::{self, ExprKind, IndexItem};
use crate::types::{Axis, BinOp, Count, Entry, MemberId, ReduceOp, ScalarType, Type, UnaryOp};

/// An operation's arguments: by position, and by name.
struct Args<'e> {
    positional: Vec<&'e ast::Expr>,
    named: Vec<(&'e str, &'e ast::Expr)>,
}

impl<'e> Args<'e> {
    fn of(list: &'e ast::List<ast::Arg>) -> Args<'e> {
        let mut out = Args {
            positional: Vec::new(),
            named: Vec::new(),
        };
        for arg in list.values() {
            match &arg.name {
                Some(n) => out.named.push((n.name.as_str(), &arg.value)),
                None => out.positional.push(&arg.value),
            }
        }
        out
    }

    fn at(&self, i: usize) -> Option<&'e ast::Expr> {
        self.positional.get(i).copied()
    }

    fn named(&self, name: &str) -> Option<&'e ast::Expr> {
        self.named.iter().find(|(n, _)| *n == name).map(|(_, e)| *e)
    }
}

fn unary(name: &str) -> Option<UnaryOp> {
    Some(match name {
        "neg" => UnaryOp::Neg,
        "abs" => UnaryOp::Abs,
        "relu" => UnaryOp::Relu,
        "square" => UnaryOp::Square,
        "floor" => UnaryOp::Floor,
        "exp" => UnaryOp::Exp,
        "exp2" => UnaryOp::Exp2,
        "tanh" => UnaryOp::Tanh,
        "erf" => UnaryOp::Erf,
        "log" => UnaryOp::Log,
        "sigmoid" => UnaryOp::Sigmoid,
        "silu" => UnaryOp::Silu,
        "sqrt" => UnaryOp::Sqrt,
        "rsqrt" => UnaryOp::Rsqrt,
        "reciprocal" => UnaryOp::Reciprocal,
        _ => return None,
    })
}

fn element(kind: MemberKind) -> Type {
    match kind {
        MemberKind::Tiles(f) => Type::Tile(f),
        MemberKind::Scalars(s) => Type::Scalar(s),
    }
}

/// The type of one block of a stream: a tile, or a block of tiles.
fn one_block(t: &Type) -> Option<Type> {
    match t {
        Type::Stream(f, Count::Const(1)) => Some(Type::Tile(*f)),
        Type::Stream(f, c) => Some(Type::Block(*f, *c)),
        _ => None,
    }
}

impl B<'_, '_> {
    fn need<'e>(&self, e: Option<&'e ast::Expr>, at: &ast::Expr, what: &str) -> Res<&'e ast::Expr> {
        match e {
            Some(x) => Ok(x),
            None => refuse(at.span, format!("no {what}")),
        }
    }

    fn values(&mut self, list: Option<&ast::Expr>) -> Res<Vec<ValueId>> {
        match list.map(|e| &e.kind) {
            None => Ok(Vec::new()),
            Some(ExprKind::List(xs)) => xs.values().map(|x| self.expr(x, None)).collect(),
            Some(_) => match list {
                Some(e) => refuse(e.span, "not a list"),
                None => Ok(Vec::new()),
            },
        }
    }

    /// A state member of the body's program, when `e` names one.
    pub(super) fn member(&self, e: &ast::Expr) -> Res<(MemberId, Type)> {
        if let ExprKind::Name(n) = &e.kind {
            if self.lookup(&n.name).is_none() {
                let names = self
                    .prog
                    .and_then(|p| self.cx.program_names.get(p.0 as usize));
                if let Some((id, kind, _)) = names.and_then(|p| p.members.get(&n.name)) {
                    return Ok((*id, element(kind.clone())));
                }
            }
        }
        refuse(e.span, "not a state member")
    }

    /// A word of the language (`worker`, `group`, `island`) when `e` is one.
    fn word<'e>(&self, e: &'e ast::Expr) -> Option<&'e str> {
        match &e.kind {
            ExprKind::Name(n) if self.lookup(&n.name).is_none() => Some(n.name.as_str()),
            _ => None,
        }
    }

    /// What a `for` walks: the opening instruction and the types of its variables.
    pub(super) fn for_iter(&mut self, iter: &ast::Expr) -> Res<(Op, Vec<Type>)> {
        if let ExprKind::Call { callee, args } = &iter.kind {
            let a = Args::of(args);
            match (&callee.kind, self.word(callee)) {
                (ExprKind::Name(_), Some("range")) => {
                    let n = self.expr(self.need(a.at(0), iter, "count")?, None)?;
                    return Ok((Op::ForRange { n }, vec![self.ty(n)]));
                }
                (ExprKind::Name(_), Some("zip")) => {
                    let s = self.expr(self.need(a.at(0), iter, "first stream")?, None)?;
                    let t = self.expr(self.need(a.at(1), iter, "second stream")?, None)?;
                    return match (one_block(&self.ty(s)), one_block(&self.ty(t))) {
                        (Some(x), Some(y)) => Ok((Op::ForZip { a: s, b: t }, vec![x, y])),
                        _ => refuse(iter.span, "`zip` of what is not a stream"),
                    };
                }
                _ => {}
            }
        }
        let s = self.expr(iter, None)?;
        match one_block(&self.ty(s)) {
            Some(t) => Ok((Op::ForStream { stream: s }, vec![t])),
            None => refuse(iter.span, "a `for` over what Lower does not walk"),
        }
    }

    /// A call, a member, an index, a message, an `await`.
    pub(super) fn form(
        &mut self,
        e: &ast::Expr,
        want: Option<ScalarType>,
        used: bool,
    ) -> Res<Option<ValueId>> {
        let span = e.span;
        match &e.kind {
            ExprKind::Member { base, name } => self.member_of(e, base, &name.name).map(Some),
            ExprKind::Index { base, index } => {
                let [IndexItem::At(i)] = index.values().collect::<Vec<_>>()[..] else {
                    return refuse(span, "an index Lower does not read");
                };
                if let Ok((m, t)) = self.member(base) {
                    let i = self.expr(i, Some(ScalarType::U32))?;
                    return Ok(Some(self.emit(Op::StateGet(m, i), t, span)));
                }
                let x = self.expr(base, None)?;
                let i = self.expr(i, Some(ScalarType::U32))?;
                let (op, t) = match self.ty(x) {
                    Type::View(Some(f)) => (Op::ReadTile(x, i), Type::Tile(f)),
                    Type::Block(f, _) => (Op::BlockAt(x, i), Type::Tile(f)),
                    Type::Words(s) => (Op::Word(x, i), Type::Scalar(s)),
                    Type::Records(r) => (Op::RecordAt(x, i), Type::RecordValue(r)),
                    _ => return refuse(span, "an index of what Lower does not index"),
                };
                Ok(Some(self.emit(op, t, span)))
            }
            ExprKind::Init {
                ty: Some(name),
                fields,
            } => {
                let Some(m) = self.cx.message(self.prog, &name.name) else {
                    return refuse(name.span, "no message of that name");
                };
                let types = self.cx.message_fields(m).to_vec();
                let mut got: Vec<Option<ValueId>> = vec![None; types.len()];
                for f in fields.values() {
                    let Some(k) = types.iter().position(|(n, _)| *n == f.name.name) else {
                        return refuse(f.name.span, "no field of that name");
                    };
                    let v = self.expr(&f.value, types.get(k).map(|(_, s)| *s))?;
                    if let Some(slot) = got.get_mut(k) {
                        *slot = Some(v);
                    }
                }
                let Some(values) = got.into_iter().collect::<Option<Vec<ValueId>>>() else {
                    return refuse(span, "a message without every field");
                };
                Ok(Some(self.emit(
                    Op::MakeMessage(m, values),
                    Type::Message(m),
                    span,
                )))
            }
            ExprKind::Await(h) => {
                let Some(Slot::Value(v)) = self.lookup(&h.name) else {
                    return refuse(h.span, "`await` of no handle");
                };
                match self.ty(v) {
                    Type::Handle(Some(s)) => {
                        Ok(Some(self.emit(Op::Await(v), Type::Scalar(s), span)))
                    }
                    Type::Handle(None) => {
                        self.emit0(Op::Await(v), span);
                        Ok(None)
                    }
                    _ => refuse(h.span, "`await` of no handle"),
                }
            }
            ExprKind::Call { callee, args } => self.call(e, callee, args, want, used),
            _ => refuse(span, "an expression Lower does not read"),
        }
    }

    fn member_of(&mut self, e: &ast::Expr, base: &ast::Expr, name: &str) -> Res<ValueId> {
        let span = e.span;
        let u32_ = Type::Scalar(ScalarType::U32);
        if let Some(word) = self.word(base) {
            let found = match (word, name) {
                ("worker", "rank") => Some((Op::Rank, u32_.clone())),
                ("worker", "is_leader") => Some((Op::IsLeader, Type::Bool)),
                ("worker", "id") => Some((Op::WorkerId, u32_.clone())),
                ("worker", "row") => Some((Op::Row, u32_.clone())),
                ("worker", "col") => Some((Op::Col, u32_.clone())),
                ("group", "size") => Some((Op::GroupSize, u32_.clone())),
                ("island", "count") => Some((Op::IslandCount, u32_.clone())),
                _ => None,
            };
            if let Some((op, t)) = found {
                return Ok(self.emit(op, t, span));
            }
        }
        let x = self.expr(base, None)?;
        let field = |fields: &[(String, ScalarType)]| {
            fields
                .iter()
                .position(|(n, _)| n == name)
                .map(|k| (k as u32, fields[k].1))
        };
        let (op, t) = match (self.ty(x), name) {
            (Type::Message(m), _) => match field(self.cx.message_fields(m)) {
                Some((k, s)) => (Op::Field(x, k), Type::Scalar(s)),
                None => return refuse(span, "no field of that name"),
            },
            (Type::RecordValue(r), _) => match field(self.cx.record_fields(r)) {
                Some((k, s)) => (Op::RecordField(x, k), Type::Scalar(s)),
                None => return refuse(span, "no field of that name"),
            },
            (Type::Outcome(_), "failed") => (Op::Failed(x), Type::Bool),
            (Type::Outcome(_), "reason") => (Op::Reason(x), Type::Reason),
            (Type::Outcome(Some(s)), "value") => (Op::OutcomeValue(x), Type::Scalar(s)),
            (Type::Broadcast, "is_sender") => (Op::IsSender(x), Type::Bool),
            (Type::Tile(f), "T") => (Op::Transpose(x), Type::Tile(f)),
            (Type::View(f), "T") => (Op::Turn(x), Type::View(f)),
            _ => return refuse(span, format!("the member `{name}` Lower does not read")),
        };
        Ok(self.emit(op, t, span))
    }

    fn call(
        &mut self,
        e: &ast::Expr,
        callee: &ast::Expr,
        list: &ast::List<ast::Arg>,
        want: Option<ScalarType>,
        used: bool,
    ) -> Res<Option<ValueId>> {
        let span = e.span;
        let a = Args::of(list);
        if let ExprKind::Member { base, name } = &callee.kind {
            return self.method(e, base, &name.name, &a);
        }
        let ExprKind::Name(n) = &callee.kind else {
            return refuse(span, "a call of what Lower does not call");
        };
        let name = n.name.as_str();
        if self.lookup(name).is_none() {
            if let Some((f, sig)) = self.cx.funcs.get(name).cloned() {
                let mut args = Vec::new();
                for (k, x) in a.positional.iter().enumerate() {
                    let want = match sig.params.get(k) {
                        Some(Type::Scalar(s)) => Some(*s),
                        _ => None,
                    };
                    args.push(self.expr(x, want)?);
                }
                let ret = sig.ret.unwrap_or(Type::Bool);
                return Ok(Some(self.emit(Op::CallFunc(f, args), ret, span)));
            }
        }
        let first = |b: &mut Self| -> Res<ValueId> {
            let x = b.need(a.at(0), e, "operand")?;
            b.expr(x, None)
        };
        if let Some(to) = ScalarType::from_name(name) {
            let x = first(self)?;
            return Ok(Some(self.emit(Op::Convert(to, x), Type::Scalar(to), span)));
        }
        if let Some(u) = unary(name) {
            let x = first(self)?;
            let t = self.ty(x);
            return Ok(Some(self.emit(Op::Unary(u, x), t, span)));
        }
        let reduce = match name {
            "reduce_sum" => Some(ReduceOp::Sum),
            "reduce_max" => Some(ReduceOp::Max),
            "reduce_min" => Some(ReduceOp::Min),
            _ => None,
        };
        if let Some(r) = reduce {
            let x = first(self)?;
            let axis = match a.named("axis").map(|x| &x.kind) {
                Some(ExprKind::Name(n)) if n.name == "row" => Axis::Row,
                Some(ExprKind::Name(n)) if n.name == "col" => Axis::Col,
                _ => return refuse(span, "a reduction without its axis"),
            };
            let Type::Tile(f) = self.ty(x) else {
                return refuse(span, "a reduction of what is not a tile");
            };
            return Ok(Some(self.emit(
                Op::Reduce(r, x, axis),
                Type::Reduced(f, axis),
                span,
            )));
        }
        let two = |b: &mut Self, w: Option<ScalarType>| -> Res<(ValueId, ValueId)> {
            let x = b.expr(b.need(a.at(0), e, "operand")?, w)?;
            let y = b.expr(b.need(a.at(1), e, "operand")?, w)?;
            Ok((x, y))
        };
        let v = match name {
            "partition" => {
                let x = first(self)?;
                let r = self.expr(self.need(a.at(1), e, "rank")?, Some(ScalarType::U32))?;
                let s = self.expr(self.need(a.at(2), e, "size")?, Some(ScalarType::U32))?;
                let t = self.ty(x);
                self.emit(Op::Partition(x, r, s), t, span)
            }
            "stream" | "sink" => {
                let x = first(self)?;
                let block = self.need(a.named("block"), e, "block")?;
                let count = self.cx.count(block)?;
                let n = self.expr(block, Some(ScalarType::U32))?;
                let Type::View(Some(f)) = self.ty(x) else {
                    return refuse(span, "a stream or a sink of what is not a view of tiles");
                };
                if name == "stream" {
                    self.emit(Op::Stream(x, n), Type::Stream(f, count), span)
                } else {
                    self.emit(Op::Sink(x, n), Type::Sink(f, count), span)
                }
            }
            "matmul" => {
                let (x, y) = two(self, None)?;
                let t = self.ty(x);
                self.emit(Op::Matmul(x, y), t, span)
            }
            "maximum" | "minimum" | "max" | "min" => {
                let (x, y) = two(self, want)?;
                let op = if name.ends_with("ax") || name == "maximum" {
                    BinOp::Max
                } else {
                    BinOp::Min
                };
                let t = match (self.ty(x), self.ty(y)) {
                    (t @ Type::Tile(_), _) | (_, t @ Type::Tile(_)) => t,
                    (t, _) => t,
                };
                self.emit(Op::Binary(op, x, y), t, span)
            }
            "gather" => {
                let x = first(self)?;
                let Type::Tile(f) = self.ty(x) else {
                    return refuse(span, "`gather` of what is not a tile");
                };
                self.emit(Op::Gather(x), Type::Block(f, Count::GroupSize), span)
            }
            "send" => {
                let m = first(self)?;
                let to = self.expr(self.need(a.named("to"), e, "`to`")?, Some(ScalarType::U32))?;
                let publish = self.values(a.named("publish"))?;
                let op = Op::Send {
                    message: m,
                    to,
                    publish,
                    used,
                };
                self.emit(op, Type::Bool, span)
            }
            "offer" => {
                let m = first(self)?;
                let to = self.expr(self.need(a.named("to"), e, "`to`")?, Some(ScalarType::U32))?;
                self.emit(Op::Offer(m, to), Type::Bool, span)
            }
            "forward" => {
                let m = first(self)?;
                let Type::Message(mt) = self.ty(m) else {
                    return refuse(span, "`forward` of what is not a message");
                };
                let types = self.cx.message_fields(mt).to_vec();
                let mut with = Vec::new();
                if let Some(w) = a.named("with") {
                    let ExprKind::Init { ty: None, fields } = &w.kind else {
                        return refuse(w.span, "`with` that is not fields");
                    };
                    for f in fields.values() {
                        let Some(k) = types.iter().position(|(n, _)| *n == f.name.name) else {
                            return refuse(f.name.span, "no field of that name");
                        };
                        let v = self.expr(&f.value, types.get(k).map(|(_, s)| *s))?;
                        with.push((k as u32, v));
                    }
                }
                let to = self.expr(self.need(a.named("to"), e, "`to`")?, Some(ScalarType::U32))?;
                let publish = self.values(a.named("publish"))?;
                let op = Op::Forward {
                    message: m,
                    with,
                    to,
                    publish,
                };
                self.emit0(op, span);
                return Ok(None);
            }
            "group" => {
                let (first_, count) = match (a.at(0), a.at(1)) {
                    (Some(f), Some(c)) => (Some(f), c),
                    (Some(c), None) => (None, c),
                    _ => return refuse(span, "a group without its count"),
                };
                let f = match first_ {
                    Some(x) => Some(self.expr(x, Some(ScalarType::U32))?),
                    None => None,
                };
                let c = self.expr(count, Some(ScalarType::U32))?;
                self.emit(Op::MakeGroup(f, c), Type::Group, span)
            }
            "invoke" => return self.invoke(e, &a).map(Some),
            "outcome" => {
                let h = first(self)?;
                let Type::Handle(r) = self.ty(h) else {
                    return refuse(span, "`outcome` of no handle");
                };
                self.emit(Op::Outcome(h), Type::Outcome(r), span)
            }
            _ => return refuse(n.span, format!("the call `{name}` Lower does not read yet")),
        };
        Ok(Some(v))
    }

    fn invoke(&mut self, e: &ast::Expr, a: &Args<'_>) -> Res<ValueId> {
        let span = e.span;
        let call = self.need(a.at(0), e, "call")?;
        let ExprKind::Call { callee, args } = &call.kind else {
            return refuse(call.span, "`invoke` of what is not a call");
        };
        let ExprKind::Member { base, name } = &callee.kind else {
            return refuse(callee.span, "`invoke` of what is not `P.h`");
        };
        let ExprKind::Name(p) = &base.kind else {
            return refuse(base.span, "`invoke` of what is not `P.h`");
        };
        let Some(pid) = self.cx.programs.get(&p.name).copied() else {
            return refuse(p.span, "no program of that name");
        };
        let names = self.cx.program_names.get(pid.0 as usize);
        let found = names.and_then(|n| {
            let key = &name.name;
            if let Some((h, s)) = n.handlers.get(key) {
                Some((Entry::Handler(pid, *h), s.clone()))
            } else if let Some((r, s)) = n.runs.get(key) {
                Some((Entry::Run(pid, *r), s.clone()))
            } else {
                n.chains
                    .get(key)
                    .map(|(c, s)| (Entry::Chain(pid, *c), s.clone()))
            }
        });
        let Some((entry, sig)) = found else {
            return refuse(name.span, "no handler, run or chain of that name");
        };
        let mut values = Vec::new();
        for (k, x) in Args::of(args).positional.iter().enumerate() {
            let want = match sig.params.get(k) {
                Some(Type::Scalar(s)) => Some(*s),
                _ => None,
            };
            values.push(self.expr(x, want)?);
        }
        let on = self.expr(self.need(a.named("on"), e, "`on`")?, None)?;
        let after = self.values(a.named("after"))?;
        let after_end = self.values(a.named("after_end"))?;
        let ret = match sig.ret {
            Some(Type::Scalar(s)) => Some(s),
            _ => None,
        };
        let op = Op::Invoke {
            entry,
            args: values,
            on,
            after,
            after_end,
        };
        Ok(self.emit(op, Type::Handle(ret), span))
    }

    fn method(
        &mut self,
        e: &ast::Expr,
        base: &ast::Expr,
        name: &str,
        a: &Args<'_>,
    ) -> Res<Option<ValueId>> {
        let span = e.span;
        if let Ok((m, t)) = self.member(base) {
            return match name {
                "zero" => {
                    self.emit0(Op::ZeroState(m), span);
                    Ok(None)
                }
                "bump" => {
                    let i = self.expr(self.need(a.at(0), e, "index")?, Some(ScalarType::U32))?;
                    Ok(Some(self.emit(Op::Bump(m, i), t, span)))
                }
                "fill" => {
                    let s = self.expr(self.need(a.at(0), e, "stream")?, None)?;
                    self.emit0(Op::Fill(m, s), span);
                    Ok(None)
                }
                _ => refuse(
                    span,
                    format!("the state operation `{name}` Lower does not read"),
                ),
            };
        }
        let x = self.expr(base, None)?;
        match (self.ty(x), name) {
            (Type::Tensor(f, _), "tiles") => {
                let first = self.expr(self.need(a.at(0), e, "first")?, Some(ScalarType::U32))?;
                let n = self.expr(self.need(a.at(1), e, "count")?, Some(ScalarType::U32))?;
                Ok(Some(self.emit(
                    Op::Tiles {
                        tensor: x,
                        first,
                        n,
                    },
                    Type::View(Some(f)),
                    span,
                )))
            }
            (Type::Sink(..), "push") => {
                let t = self.expr(self.need(a.at(0), e, "tile")?, None)?;
                self.emit0(Op::Push(x, t), span);
                Ok(None)
            }
            (Type::Tile(f), "col") => Ok(Some(self.emit(Op::SpreadCol(x), Type::Tile(f), span))),
            (Type::Tile(f), "row") => Ok(Some(self.emit(Op::SpreadRow(x), Type::Tile(f), span))),
            (Type::Reduced(f, Axis::Row), "col") => {
                Ok(Some(self.emit(Op::SpreadCol(x), Type::Tile(f), span)))
            }
            (Type::Reduced(f, Axis::Col), "row") => {
                Ok(Some(self.emit(Op::SpreadRow(x), Type::Tile(f), span)))
            }
            _ => refuse(span, format!("the method `{name}` Lower does not read yet")),
        }
    }
}
