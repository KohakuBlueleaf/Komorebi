//! The operations of a body's expressions, as the checker resolved them.
//!
//! Every call, member, index, operator, message and `await` has its operation from
//! `Checked::operation`, with its operands by part, and Lower walks that: it reads no
//! form of the text a second time. A value's type is the checker's (`type_of`) where
//! it names one this structure keeps; otherwise it follows from the operands.

use super::body::{Slot, B};
use super::{refuse, Res};
use crate::check::op::{OpKind, Resolved};
use crate::check::Name;
use crate::program::body::{Op, Range, ScalarsFrom, ValueId};
use crate::program::{BodyRef, MemberKind};
use crate::syntax::ast::{self, ExprKind};
use crate::types::{Count, Entry, MemberId, MessageId, ScalarType, Type};

fn element(kind: MemberKind) -> Type {
    match kind {
        MemberKind::Tiles(f) => Type::Tile(f),
        MemberKind::Scalars(s) => Type::Scalar(s),
    }
}

/// The type of one block of a stream: a tile, or a block of tiles.
fn one_block(t: &Type) -> Option<Type> {
    match t {
        Type::Stream(f, Count::Const(1), Count::Const(1)) => Some(Type::Tile(*f)),
        Type::Stream(f, r, c) => Some(Type::Block(*f, *r, *c)),
        _ => None,
    }
}

/// The operands of one resolved expression, by part.
struct Parts<'e> {
    at: &'e ast::Expr,
    parts: Vec<Option<&'e ast::Expr>>,
}

impl<'e> Parts<'e> {
    fn of(r: &Resolved, e: &'e ast::Expr) -> Parts<'e> {
        Parts {
            at: e,
            parts: r.operands(e),
        }
    }

    fn get(&self, k: usize) -> Option<&'e ast::Expr> {
        self.parts.get(k).copied().flatten()
    }

    fn need(&self, k: usize) -> Res<&'e ast::Expr> {
        match self.get(k) {
            Some(x) => Ok(x),
            None => refuse(self.at.span, format!("no operand {k}")),
        }
    }
}

impl B<'_, '_> {
    /// The type the checker gives `e`, where it is one this structure keeps as it is.
    pub(super) fn checked(&self, e: &ast::Expr) -> Option<Type> {
        match self.cx.checked.type_of(e)? {
            t @ (Type::Scalar(_)
            | Type::Bool
            | Type::Tile(_)
            | Type::Block(..)
            | Type::Reduced(..)
            | Type::Reason) => Some(t.clone()),
            _ => None,
        }
    }

    /// The checker's scalar type of `e`, a number written in the text.
    pub(super) fn checked_scalar(&self, e: &ast::Expr) -> Option<ScalarType> {
        match self.checked(e)? {
            Type::Scalar(s) => Some(s),
            _ => None,
        }
    }

    fn part(&mut self, p: &Parts<'_>, k: usize) -> Res<ValueId> {
        let x = p.need(k)?;
        self.expr(x)
    }

    fn opt_part(&mut self, p: &Parts<'_>, k: usize) -> Res<Option<ValueId>> {
        match p.get(k) {
            Some(x) => self.expr(x).map(Some),
            None => Ok(None),
        }
    }

    /// The values of a list operand (`publish = [...]`, `after = [...]`).
    fn list_part(&mut self, p: &Parts<'_>, k: usize) -> Res<Vec<ValueId>> {
        match p.get(k) {
            None => Ok(Vec::new()),
            Some(e) => match &e.kind {
                ExprKind::List(xs) => xs.values().map(|x| self.expr(x)).collect(),
                _ => refuse(e.span, "not a list"),
            },
        }
    }

    /// The words a `body = src` operand gives: a state member named whole, by its id;
    /// otherwise the value (a message's body, or the words of a tile).
    fn body_part(&mut self, p: &Parts<'_>, k: usize) -> Res<Option<ScalarsFrom>> {
        let Some(x) = p.get(k) else {
            return Ok(None);
        };
        if let ExprKind::Name(n) = &x.kind {
            if let Some(Name::State { member, .. }) = self.cx.checked.binding(n.span) {
                return Ok(Some(ScalarsFrom::Member(MemberId(*member))));
            }
        }
        Ok(Some(ScalarsFrom::Value(self.expr(x)?)))
    }

    /// A count the text writes in an operand: a number, or a module parameter.
    fn count_part(&self, p: &Parts<'_>, k: usize) -> Res<Count> {
        self.cx.count(p.need(k)?)
    }

    /// The kind and the count of a member of the body's program.
    fn member_kind(&self, m: MemberId, span: ast::Span) -> Res<(MemberKind, Count)> {
        let names = self
            .prog
            .and_then(|p| self.cx.program_names.get(p.0 as usize));
        let found = names.and_then(|p| p.members.values().find(|(id, _, _)| *id == m));
        match found {
            Some((_, kind, count)) => Ok((kind.clone(), *count)),
            None => refuse(span, "no state member of that place"),
        }
    }

    /// The type of an element of a state member.
    pub(super) fn member_element(&self, m: MemberId, span: ast::Span) -> Res<Type> {
        self.member_kind(m, span).map(|(kind, _)| element(kind))
    }

    /// The operation the checker resolved for `e`.
    pub(super) fn resolved(&self, e: &ast::Expr) -> Res<&Resolved> {
        match self.cx.checked.operation(e) {
            Some(r) => Ok(r),
            None => refuse(e.span, "an expression the checker resolved to no operation"),
        }
    }

    /// The target `s[i]` of an assignment: the member and the index.
    pub(super) fn state_target(&mut self, target: &ast::Expr) -> Res<(MemberId, ValueId)> {
        let r = self.resolved(target)?.clone();
        let OpKind::StateSet(m) = r.kind else {
            return refuse(target.span, "an assignment to what is not a state element");
        };
        let p = Parts::of(&r, target);
        let i = self.part(&p, 0)?;
        Ok((m, i))
    }

    /// What a `for` walks: the opening instruction and the types of its variables.
    pub(super) fn for_iter(&mut self, iter: &ast::Expr) -> Res<(Op, Vec<Type>)> {
        let Some(r) = self.cx.checked.operation(iter).cloned() else {
            return self.for_stream(iter);
        };
        let p = Parts::of(&r, iter);
        match r.kind {
            OpKind::ForRange => {
                let n = self.part(&p, 0)?;
                Ok((Op::ForRange { n }, vec![self.ty(n)]))
            }
            OpKind::ForZip => {
                if p.parts.len() < 2 {
                    return refuse(iter.span, "a `zip` of fewer than two streams");
                }
                let mut streams = Vec::new();
                let mut blocks = Vec::new();
                for i in 0..p.parts.len() {
                    let s = self.part(&p, i)?;
                    let Some(block) = one_block(&self.ty(s)) else {
                        return refuse(iter.span, "`zip` of what is not a stream");
                    };
                    streams.push(s);
                    blocks.push(block);
                }
                Ok((Op::ForZip { streams }, blocks))
            }
            OpKind::ForArrive => {
                let br = self.part(&p, 0)?;
                let Type::Broadcast(f) = self.ty(br) else {
                    return refuse(iter.span, "`arrive` of what is not a broadcast");
                };
                // each block the sender casts, and the word that rides with it
                let word = Type::Scalar(ScalarType::U32);
                Ok((Op::ForArrive { broadcast: br }, vec![Type::Tile(f), word]))
            }
            _ => self.for_stream(iter),
        }
    }

    fn for_stream(&mut self, iter: &ast::Expr) -> Res<(Op, Vec<Type>)> {
        let s = self.expr(iter)?;
        match one_block(&self.ty(s)) {
            Some(t) => Ok((Op::ForStream { stream: s }, vec![t])),
            None => refuse(iter.span, "a `for` over what is not a stream"),
        }
    }

    /// The handle named by `await h`.
    fn handle(&self, h: &ast::Ident) -> Res<ValueId> {
        match self.lookup(&h.name) {
            Some(Slot::Value(v)) => Ok(v),
            _ => refuse(h.span, "`await` of no handle"),
        }
    }

    /// The type `as = T` names, with `with` made every field of T in its order: the
    /// value `with` gives, else the arrived message's field of the same name and type,
    /// read with `Field`.
    fn forward_as(
        &mut self,
        message: ValueId,
        t: &ast::Expr,
        with: &mut Vec<(u32, ValueId)>,
    ) -> Res<MessageId> {
        let target = match &t.kind {
            ExprKind::Name(n) => match self.cx.checked.binding(n.span) {
                Some(Name::Message(m)) => *m,
                _ => return refuse(t.span, "a message type the checker did not resolve"),
            },
            _ => return refuse(t.span, "`as` that is not a message type"),
        };
        let Type::Message(from) = self.ty(message) else {
            return refuse(t.span, "a forward of what is not a message");
        };
        let source: Vec<(String, ScalarType)> = self.cx.message_fields(from).to_vec();
        let fields: Vec<(String, ScalarType)> = self.cx.message_fields(target).to_vec();
        let mut all = Vec::with_capacity(fields.len());
        for (k, (name, s)) in fields.iter().enumerate() {
            let k = k as u32;
            if let Some(&(_, v)) = with.iter().find(|(at, _)| *at == k) {
                all.push((k, v));
                continue;
            }
            let Some(j) = source.iter().position(|(n, t)| n == name && t == s) else {
                return refuse(t.span, format!("no value for the field `{name}`"));
            };
            let v = self.emit(Op::Field(message, j as u32), Type::Scalar(*s), t.span);
            all.push((k, v));
        }
        *with = all;
        Ok(target)
    }

    /// The result type of what an invocation names.
    fn entry_result(&self, entry: Entry) -> Option<ScalarType> {
        let Entry::Handler(p, h) = entry else {
            return None;
        };
        let names = self.cx.program_names.get(p.0 as usize)?;
        let (_, sig) = names.handlers.values().find(|(id, _)| *id == h)?;
        match sig.ret {
            Some(Type::Scalar(s)) => Some(s),
            _ => None,
        }
    }

    /// The operation of `e`, resolved by the checker; `used` tells whether its value is
    /// read.
    pub(super) fn form(&mut self, e: &ast::Expr, used: bool) -> Res<Option<ValueId>> {
        let span = e.span;
        let r = self.resolved(e)?.clone();
        let p = Parts::of(&r, e);
        let u32_ = Type::Scalar(ScalarType::U32);
        let known = self.checked(e);
        // the checker's type, or the one the operands give
        let or = |t: Type| known.clone().unwrap_or(t);
        let need_known = |what: &str| match known.clone() {
            Some(t) => Ok(t),
            None => refuse(span, format!("{what} of no type the checker gave")),
        };
        let (op, ty) = match r.kind {
            OpKind::Binary(b) => {
                let x = self.part(&p, 0)?;
                let y = self.part(&p, 1)?;
                let t = match (self.ty(x), self.ty(y)) {
                    (t @ Type::Block(..), _) | (_, t @ Type::Block(..)) => t,
                    (t @ Type::Tile(_), _) | (_, t @ Type::Tile(_)) => t,
                    (t @ Type::Reduced(..), _) | (_, t @ Type::Reduced(..)) => t,
                    (t, _) => t,
                };
                (Op::Binary(b, x, y), or(t))
            }
            OpKind::Compare(c) => {
                let x = self.part(&p, 0)?;
                let y = self.part(&p, 1)?;
                (Op::Compare(c, x, y), Type::Bool)
            }
            OpKind::LogicAnd => (
                Op::LogicAnd(self.part(&p, 0)?, self.part(&p, 1)?),
                Type::Bool,
            ),
            OpKind::LogicOr => (
                Op::LogicOr(self.part(&p, 0)?, self.part(&p, 1)?),
                Type::Bool,
            ),
            OpKind::Unary(u) => {
                let x = self.part(&p, 0)?;
                (Op::Unary(u, x), or(self.ty(x)))
            }
            OpKind::Convert(s) => (Op::Convert(s, self.part(&p, 0)?), Type::Scalar(s)),
            OpKind::Select => {
                let c = p.need(0)?;
                let rc = self.resolved(c)?.clone();
                let OpKind::Compare(cmp) = rc.kind else {
                    return refuse(c.span, "a `where` whose condition is not a comparison");
                };
                let pc = Parts::of(&rc, c);
                let a = self.part(&pc, 0)?;
                let b = self.part(&pc, 1)?;
                let x = self.part(&p, 1)?;
                let y = self.part(&p, 2)?;
                (Op::Select { cmp, a, b, x, y }, or(self.ty(x)))
            }
            OpKind::Reduce(op, axis) => {
                let x = self.part(&p, 0)?;
                // of a block (R45): of each of its tiles, its shape kept
                let t = match self.ty(x) {
                    Type::Tile(f) => Type::Reduced(f, axis, Count::ONE, Count::ONE),
                    Type::Block(f, r, c) => Type::Reduced(f, axis, r, c),
                    _ => return refuse(span, "a reduction of what is not a tile"),
                };
                (Op::Reduce(op, x, axis), t)
            }
            OpKind::SpreadCol | OpKind::SpreadRow => {
                let x = self.part(&p, 0)?;
                let t = match self.ty(x) {
                    Type::Tile(f) | Type::Reduced(f, _, Count::Const(1), Count::Const(1)) => {
                        Type::Tile(f)
                    }
                    Type::Block(f, r, c) | Type::Reduced(f, _, r, c) => Type::Block(f, r, c),
                    _ => return refuse(span, "a spread of what is not a tile"),
                };
                let op = if r.kind == OpKind::SpreadCol {
                    Op::SpreadCol(x)
                } else {
                    Op::SpreadRow(x)
                };
                (op, t)
            }
            OpKind::Matmul => {
                let x = self.part(&p, 0)?;
                let y = self.part(&p, 1)?;
                (Op::Matmul(x, y), or(self.ty(x)))
            }
            OpKind::Transpose => {
                let x = self.part(&p, 0)?;
                // `.T` of a block of r rows of c tiles is a block of c rows of r tiles;
                // of a tile, a tile of the same format
                let ty = match self.ty(x) {
                    Type::Block(f, r, c) => Type::Block(f, c, r),
                    t => or(t),
                };
                (Op::Transpose(x), ty)
            }
            OpKind::Zeros => (Op::Zeros, need_known("`zeros`")?),
            OpKind::CallFunc(f) => {
                let mut args = Vec::new();
                for k in 0..p.parts.len() {
                    args.push(self.part(&p, k)?);
                }
                let ret = self
                    .cx
                    .funcs
                    .values()
                    .find(|(id, _)| *id == f)
                    .and_then(|(_, sig)| sig.ret.clone());
                let Some(ret) = ret else {
                    return refuse(span, "a call of a function with no result");
                };
                // a function on tiles called on blocks (R45): the checker's block
                (Op::CallFunc(f, args), or(ret))
            }
            OpKind::Tiles => {
                let tensor = self.part(&p, 0)?;
                let first = self.part(&p, 1)?;
                let n = self.part(&p, 2)?;
                let Type::Tensor(f, _) = self.ty(tensor) else {
                    return refuse(span, "tiles of what is not a tensor");
                };
                (Op::Tiles { tensor, first, n }, Type::View(Some(f)))
            }
            OpKind::Rect => {
                let tensor = self.part(&p, 0)?;
                let Type::Tensor(f, _) = self.ty(tensor) else {
                    return refuse(span, "tiles of what is not a tensor");
                };
                let rows = Range {
                    lo: self.part(&p, 1)?,
                    hi: self.part(&p, 2)?,
                    step: self.opt_part(&p, 3)?,
                };
                let cols = Range {
                    lo: self.part(&p, 4)?,
                    hi: self.part(&p, 5)?,
                    step: self.opt_part(&p, 6)?,
                };
                (Op::Rect { tensor, rows, cols }, Type::View(Some(f)))
            }
            OpKind::Turn => {
                let x = self.part(&p, 0)?;
                (Op::Turn(x), self.ty(x))
            }
            OpKind::Items => {
                let tensor = self.part(&p, 0)?;
                let range = match (self.opt_part(&p, 1)?, self.opt_part(&p, 2)?) {
                    (Some(first), Some(n)) => Some((first, n)),
                    (None, None) => None,
                    _ => return refuse(span, "`items` with one of `first` and `n`"),
                };
                let count = self.count_part(&p, 3)?;
                let size = self.part(&p, 3)?;
                (
                    Op::Items {
                        tensor,
                        range,
                        size,
                    },
                    Type::Items(count),
                )
            }
            OpKind::ReadTile => {
                let v = self.part(&p, 0)?;
                let i = self.part(&p, 1)?;
                // through a view of no format, the tile has the format its place
                // states, as the checker gives it
                let f = match (self.ty(v), known) {
                    (Type::View(Some(f)), _) | (Type::View(None), Some(Type::Tile(f))) => f,
                    _ => return refuse(span, "a tile whose format neither view nor place gives"),
                };
                (Op::ReadTile(v, i), Type::Tile(f))
            }
            OpKind::Words => {
                let v = self.part(&p, 0)?;
                let s = match self.ty(v) {
                    Type::View(f) => f.and_then(|f| f.as_scalar()),
                    // a message's body, read as u32 scalars
                    Type::Body => Some(ScalarType::U32),
                    _ => None,
                };
                let Some(s) = s else {
                    return refuse(span, "the words of what is not a view of scalars");
                };
                (Op::Words(v), Type::Words(s))
            }
            OpKind::Word => {
                let w = self.part(&p, 0)?;
                let i = self.part(&p, 1)?;
                let Type::Words(s) = self.ty(w) else {
                    return refuse(span, "a word of what is not words");
                };
                (Op::Word(w, i), Type::Scalar(s))
            }
            OpKind::Records(rec) => {
                let v = self.part(&p, 0)?;
                let c = self.part(&p, 1)?;
                (Op::Records(v, rec, c), Type::Records(rec))
            }
            OpKind::RecordAt => {
                let x = self.part(&p, 0)?;
                let i = self.part(&p, 1)?;
                let Type::Records(rec) = self.ty(x) else {
                    return refuse(span, "a record of what is not records");
                };
                (Op::RecordAt(x, i), Type::RecordValue(rec))
            }
            OpKind::RecordField(k) => {
                let x = self.part(&p, 0)?;
                let Type::RecordValue(rec) = self.ty(x) else {
                    return refuse(span, "a field of what is not a record");
                };
                let Some((_, s)) = self.cx.record_fields(rec).get(k as usize) else {
                    return refuse(span, "no field of that place");
                };
                (Op::RecordField(x, k), Type::Scalar(*s))
            }
            OpKind::Field(k) => {
                let x = self.part(&p, 0)?;
                let Type::Message(m) = self.ty(x) else {
                    return refuse(span, "a field of what is not a message");
                };
                let Some((_, s)) = self.cx.message_fields(m).get(k as usize) else {
                    return refuse(span, "no field of that place");
                };
                (Op::Field(x, k), Type::Scalar(*s))
            }
            OpKind::Body => {
                let x = self.part(&p, 0)?;
                let Type::Message(_) = self.ty(x) else {
                    return refuse(span, "the body of what is not a message");
                };
                (Op::Body(x), Type::Body)
            }
            OpKind::Put(src) => {
                let items = self.part(&p, 0)?;
                let at = self.part(&p, 2)?;
                self.emit0(Op::Put { items, src, at }, span);
                return Ok(None);
            }
            OpKind::Partition => {
                let x = self.part(&p, 0)?;
                let rank = self.part(&p, 1)?;
                let size = self.part(&p, 2)?;
                (Op::Partition(x, rank, size), self.ty(x))
            }
            OpKind::Stream | OpKind::Sink => {
                let x = self.part(&p, 0)?;
                let count = self.count_part(&p, 1)?;
                let n = self.part(&p, 1)?;
                // `block = [r, c]` (R45): the rows a part of their own, in the type only
                let rows = match p.get(2) {
                    Some(_) => self.count_part(&p, 2)?,
                    None => Count::ONE,
                };
                let Type::View(Some(f)) = self.ty(x) else {
                    return refuse(span, "a stream or a sink of what is not a view of a format");
                };
                if r.kind == OpKind::Stream {
                    (Op::Stream(x, n), Type::Stream(f, rows, count))
                } else {
                    (Op::Sink(x, n), Type::Sink(f, rows, count))
                }
            }
            OpKind::Push => {
                let out = self.part(&p, 0)?;
                let t = self.part(&p, 1)?;
                self.emit0(Op::Push(out, t), span);
                return Ok(None);
            }
            OpKind::BlockAt => {
                let x = self.part(&p, 0)?;
                let i = self.part(&p, 1)?;
                let Type::Block(f, ..) = self.ty(x) else {
                    return refuse(span, "a tile of what is not a block");
                };
                (Op::BlockAt(x, i), Type::Tile(f))
            }
            OpKind::Fill(m) => {
                let s = self.part(&p, 0)?;
                self.emit0(Op::Fill(m, s), span);
                return Ok(None);
            }
            OpKind::ZeroState(m) => {
                self.emit0(Op::ZeroState(m), span);
                return Ok(None);
            }
            OpKind::StateGet(m) => {
                let i = match self.opt_part(&p, 0)? {
                    Some(i) => i,
                    // a member of one element read by its name: its element 0
                    None => self.constant(super::body::int_const(0, Some(ScalarType::U32)), span),
                };
                (Op::StateGet(m, i), self.member_element(m, span)?)
            }
            OpKind::StateSet(_) => {
                return refuse(span, "a state element written where a value is read")
            }
            OpKind::Bump(m) => {
                let i = self.part(&p, 0)?;
                (Op::Bump(m, i), self.member_element(m, span)?)
            }
            OpKind::Rank => (Op::Rank, u32_),
            OpKind::IsLeader => (Op::IsLeader, Type::Bool),
            OpKind::WorkerId => (Op::WorkerId, u32_),
            OpKind::Row => (Op::Row, u32_),
            OpKind::Col => (Op::Col, u32_),
            OpKind::GroupSize => (Op::GroupSize, u32_),
            OpKind::GroupId => (Op::GroupId(self.part(&p, 0)?), u32_),
            OpKind::Split(by) => (Op::Split(by, self.part(&p, 0)?), Type::SubGroups),
            OpKind::Leaders(by) => (Op::Leaders(by, self.part(&p, 0)?), Type::Leaders),
            OpKind::ReduceGroup(with) => {
                let x = self.part(&p, 0)?;
                let within = self.opt_part(&p, 1)?;
                let among = self.opt_part(&p, 2)?;
                let publish = self.list_part(&p, 3)?;
                let t = or(self.ty(x));
                let op = Op::ReduceGroup {
                    x,
                    with,
                    within,
                    among,
                    publish,
                };
                (op, t)
            }
            OpKind::AllReduce(with) => {
                let x = self.part(&p, 0)?;
                let publish = self.list_part(&p, 1)?;
                (Op::AllReduce { x, with, publish }, or(self.ty(x)))
            }
            OpKind::Gather => {
                let x = self.part(&p, 0)?;
                let Type::Tile(f) = self.ty(x) else {
                    return refuse(span, "`gather` of what is not a tile");
                };
                (Op::Gather(x), Type::Block(f, Count::ONE, Count::GroupSize))
            }
            OpKind::AllGather => {
                let value = self.part(&p, 0)?;
                if matches!(self.body_ref, BodyRef::Control(_)) {
                    // over the islands' copies: it gives no value
                    self.emit0(
                        Op::AllGather {
                            value,
                            publish: Vec::new(),
                        },
                        span,
                    );
                    return Ok(None);
                }
                let publish = self.list_part(&p, 1)?;
                let Type::Tile(f) = self.ty(value) else {
                    return refuse(span, "`all_gather` of what is not a tile");
                };
                let ty = Type::Block(f, Count::ONE, Count::GroupSize);
                (Op::AllGather { value, publish }, ty)
            }
            OpKind::Broadcast(axis) => {
                // the sender is 0 when left out
                let sender = match self.opt_part(&p, 0)? {
                    Some(s) => s,
                    None => self.constant(super::body::int_const(0, Some(ScalarType::U32)), span),
                };
                // the format the casts on it send, which the checker gives
                let Some(t @ Type::Broadcast(_)) = self.cx.checked.type_of(e) else {
                    return refuse(span, "a broadcast that no cast sends on");
                };
                (Op::Broadcast(axis, sender), t.clone())
            }
            OpKind::IsSender => (Op::IsSender(self.part(&p, 0)?), Type::Bool),
            OpKind::Cast => {
                let br = self.part(&p, 0)?;
                let block = self.part(&p, 1)?;
                let word = self.opt_part(&p, 2)?;
                self.emit0(Op::Cast(br, block, word), span);
                return Ok(None);
            }
            OpKind::Sync => {
                let publish = self.list_part(&p, 0)?;
                self.emit0(Op::Sync(publish), span);
                return Ok(None);
            }
            OpKind::MakeMessage(m) => {
                let mut values = Vec::new();
                for k in 0..p.parts.len() {
                    values.push(self.part(&p, k)?);
                }
                (Op::MakeMessage(m, values), Type::Message(m))
            }
            OpKind::Send => {
                let message = self.part(&p, 0)?;
                let to = self.part(&p, 1)?;
                let body = self.body_part(&p, 3)?;
                let publish = self.list_part(&p, 2)?;
                let op = Op::Send {
                    message,
                    to,
                    body,
                    publish,
                    used,
                };
                (op, Type::Bool)
            }
            OpKind::Forward => {
                let message = self.part(&p, 0)?;
                let mut with = Vec::new();
                if let Some(w) = p.get(1) {
                    let ExprKind::Init { fields, .. } = &w.kind else {
                        return refuse(w.span, "`with` that is not fields");
                    };
                    for f in fields.values() {
                        let Some(Name::Field(k)) = self.cx.checked.binding(f.name.span) else {
                            return refuse(f.name.span, "a field the checker did not resolve");
                        };
                        let k = *k;
                        with.push((k, self.expr(&f.value)?));
                    }
                }
                let as_type = match p.get(4) {
                    Some(t) => Some(self.forward_as(message, t, &mut with)?),
                    None => None,
                };
                let to = self.part(&p, 2)?;
                let publish = self.list_part(&p, 3)?;
                let op = Op::Forward {
                    message,
                    as_type,
                    with,
                    to,
                    publish,
                };
                self.emit0(op, span);
                return Ok(None);
            }
            OpKind::Offer => {
                let m = self.part(&p, 0)?;
                let to = self.part(&p, 1)?;
                let body = self.body_part(&p, 2)?;
                (Op::Offer(m, to, body), Type::Bool)
            }
            OpKind::MakeGroup => {
                let first = self.opt_part(&p, 0)?;
                let count = self.part(&p, 1)?;
                (Op::MakeGroup(first, count), Type::Group)
            }
            OpKind::Invoke(entry) => {
                let on = self.part(&p, 0)?;
                let after = self.list_part(&p, 1)?;
                let after_end = self.list_part(&p, 2)?;
                let mut args = Vec::new();
                for k in 3..p.parts.len() {
                    args.push(self.part(&p, k)?);
                }
                let op = Op::Invoke {
                    entry,
                    args,
                    on,
                    after,
                    after_end,
                };
                (op, Type::Handle(self.entry_result(entry)))
            }
            OpKind::Await => {
                let ExprKind::Await(h) = &e.kind else {
                    return refuse(span, "an `await` that is not one");
                };
                let v = self.handle(h)?;
                return match self.ty(v) {
                    Type::Handle(Some(s)) => {
                        Ok(Some(self.emit(Op::Await(v), Type::Scalar(s), span)))
                    }
                    Type::Handle(None) => {
                        self.emit0(Op::Await(v), span);
                        Ok(None)
                    }
                    _ => refuse(h.span, "`await` of no handle"),
                };
            }
            OpKind::Outcome => {
                let h = self.part(&p, 0)?;
                let Type::Handle(ret) = self.ty(h) else {
                    return refuse(span, "`outcome` of no handle");
                };
                (Op::Outcome(h), Type::Outcome(ret))
            }
            OpKind::Failed => (Op::Failed(self.part(&p, 0)?), Type::Bool),
            OpKind::Reason => (Op::Reason(self.part(&p, 0)?), Type::Reason),
            OpKind::OutcomeValue => {
                let o = self.part(&p, 0)?;
                let Type::Outcome(Some(s)) = self.ty(o) else {
                    return refuse(span, "the value of an outcome of no result");
                };
                (Op::OutcomeValue(o), Type::Scalar(s))
            }
            OpKind::IslandCount => (Op::IslandCount, need_known("`island.count`")?),
            OpKind::IslandOf => (Op::IslandOf(self.part(&p, 0)?), need_known("`island.of`")?),
            OpKind::IslandCrosses => (
                Op::IslandCrosses(self.part(&p, 0)?),
                need_known("`island.crosses`")?,
            ),
            OpKind::IslandHops => (
                Op::IslandHops(self.part(&p, 0)?),
                need_known("`island.hops`")?,
            ),
            OpKind::ReduceTo(with) => {
                let t = self.part(&p, 0)?;
                let to = self.part(&p, 1)?;
                self.emit0(Op::ReduceTo(t, to, with), span);
                return Ok(None);
            }
            OpKind::BroadcastFrom => {
                let t = self.part(&p, 0)?;
                let from = self.part(&p, 1)?;
                self.emit0(Op::BroadcastFrom(t, from), span);
                return Ok(None);
            }
            OpKind::ForRange | OpKind::ForZip | OpKind::ForArrive => {
                return refuse(span, "what a `for` walks, where a value is read")
            }
        };
        Ok(Some(self.emit(op, ty, span)))
    }
}
