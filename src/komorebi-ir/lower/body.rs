//! One body: its arguments, its statements and the expressions they evaluate, as a flat
//! list of instructions with structured markers.

use std::collections::{HashMap, HashSet};

use super::{refuse, source_span, Cx, Res};
use crate::program::body::{BinOp, Body, CmpOp, Inst, InstId, LocalId, Op, UnaryOp, ValueId};
use crate::program::notes::{Anchor, Hint, Named};
use crate::program::{BodyRef, MemberKind, MessageRef, ProgramId};
use crate::syntax::ast::{self, AssignOp, ExprKind, Span, Stmt};
use crate::types::{ConstValue, Count, Failure, ScalarType, Type};

/// A number as the text writes it, before it has a type.
#[derive(Clone, Copy, Debug)]
pub(super) enum Lit {
    /// An integer.
    Int(i128),
    /// A float.
    Float(f64),
}

/// The mask of an integer `bits` wide.
fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// An integer constant of type `ty` (with none, `u32`); a float type takes its value.
pub(super) fn int_const(value: i128, ty: Option<ScalarType>) -> ConstValue {
    let ty = ty.unwrap_or(ScalarType::U32);
    if ty.is_float() {
        ConstValue::Float(ty, value as f64)
    } else {
        ConstValue::Int(ty, (value as u64) & mask(ty.bits()))
    }
}

/// A float constant of type `ty` when it is a float type (with none, `f32`).
pub(super) fn float_const(value: f64, ty: Option<ScalarType>) -> ConstValue {
    match ty {
        Some(t) if t.is_float() => ConstValue::Float(t, value),
        _ => ConstValue::Float(ScalarType::F32, value),
    }
}

/// The number a literal (or a negated one) writes.
pub(super) fn lit(e: &ast::Expr) -> Option<Lit> {
    match &e.kind {
        ExprKind::Paren(x) => lit(x),
        ExprKind::Int(t) => Some(Lit::Int(i128::try_from(super::int_value(t)?).ok()?)),
        ExprKind::Float(t) => Some(Lit::Float(t.replace('_', "").parse().ok()?)),
        ExprKind::Neg(x) => match lit(x)? {
            Lit::Int(v) => Some(Lit::Int(-v)),
            Lit::Float(v) => Some(Lit::Float(-v)),
        },
        _ => None,
    }
}

/// A literal as a constant of type `ty`.
pub(super) fn lit_const(l: Lit, ty: Option<ScalarType>) -> ConstValue {
    match l {
        Lit::Int(v) => int_const(v, ty),
        Lit::Float(v) => float_const(v, ty),
    }
}

/// What a name in the body stands for.
#[derive(Clone, Copy)]
pub(super) enum Slot {
    /// A value: a `let`, a loop variable, a parameter, the arrived message.
    Value(ValueId),
    /// A `var`.
    Local(LocalId),
}

/// A body being lowered.
pub(super) struct B<'a, 'c> {
    pub(super) cx: &'a mut Cx<'c>,
    pub(super) prog: Option<ProgramId>,
    body_ref: BodyRef,
    ret: Option<Type>,
    values: Vec<Type>,
    locals: Vec<Type>,
    insts: Vec<Inst>,
    scopes: Vec<HashMap<String, Slot>>,
    /// The values a bare literal defines, with its instruction and its number, so an
    /// operation can give it the type of its other operand.
    literals: HashMap<u32, (usize, Lit)>,
    named: HashSet<u32>,
    /// The type `zeros` stands for where the text writes it: a `var`'s, a store's.
    zeros: Option<Type>,
}

impl<'c> Cx<'c> {
    /// Lowers a body: its `Arg`s, the arrived message of an `on` body, its block.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn body(
        &mut self,
        body_ref: BodyRef,
        prog: Option<ProgramId>,
        params: &ast::List<ast::ParamDecl>,
        types: &[Type],
        ret: Option<Type>,
        arrived: Option<(MessageRef, &ast::Ident)>,
        block: &ast::Block,
    ) -> Res<Body> {
        let mut b = B {
            cx: self,
            prog,
            body_ref,
            ret,
            values: Vec::new(),
            locals: Vec::new(),
            insts: Vec::new(),
            scopes: vec![HashMap::new()],
            literals: HashMap::new(),
            named: HashSet::new(),
            zeros: None,
        };
        for (i, (p, t)) in params.values().zip(types).enumerate() {
            let v = b.emit(Op::Arg(i as u32), t.clone(), p.span);
            b.bind(&p.name.name, Slot::Value(v));
        }
        if let Some((m, name)) = arrived {
            let v = b.emit(Op::Arrived, Type::Message(m), name.span);
            b.bind(&name.name, Slot::Value(v));
            b.name_value(v, &name.name);
        }
        b.block(block)?;
        Ok(Body {
            values: b.values,
            locals: b.locals,
            insts: b.insts,
        })
    }
}

impl B<'_, '_> {
    // ---- the instructions ----

    fn push(&mut self, op: Op, defs: Vec<ValueId>, span: Span) -> usize {
        let at = self.insts.len();
        self.insts.push(Inst { op, defs });
        let anchor = Anchor::Inst(self.body_ref, InstId(at as u32));
        self.cx.spans.push((anchor, source_span(span)));
        at
    }

    /// Emits an operation that defines one value of type `ty`.
    pub(super) fn emit(&mut self, op: Op, ty: Type, span: Span) -> ValueId {
        let v = ValueId(self.values.len() as u32);
        self.values.push(ty);
        self.push(op, vec![v], span);
        v
    }

    /// Emits an operation that defines one value for each of `types`.
    pub(super) fn emit_n(&mut self, op: Op, types: Vec<Type>, span: Span) -> Vec<ValueId> {
        let defs: Vec<ValueId> = types
            .into_iter()
            .map(|t| {
                let v = ValueId(self.values.len() as u32);
                self.values.push(t);
                v
            })
            .collect();
        self.push(op, defs.clone(), span);
        defs
    }

    /// Emits an operation that defines nothing.
    pub(super) fn emit0(&mut self, op: Op, span: Span) {
        self.push(op, Vec::new(), span);
    }

    /// The type of a value.
    pub(super) fn ty(&self, v: ValueId) -> Type {
        self.values.get(v.0 as usize).cloned().unwrap_or(Type::Bool)
    }

    /// The scalar type of a value, if it has one.
    pub(super) fn scalar(&self, v: ValueId) -> Option<ScalarType> {
        match self.ty(v) {
            Type::Scalar(s) => Some(s),
            _ => None,
        }
    }

    /// A constant.
    pub(super) fn constant(&mut self, c: ConstValue, span: Span) -> ValueId {
        let ty = match &c {
            ConstValue::Int(s, _) | ConstValue::Float(s, _) => Type::Scalar(*s),
            ConstValue::Bool(_) => Type::Bool,
        };
        self.emit(Op::Const(c), ty, span)
    }

    /// Gives a bare literal the type `to`, the other operand's.
    fn retype(&mut self, v: ValueId, to: ScalarType) {
        let Some((at, l)) = self.literals.get(&v.0).copied() else {
            return;
        };
        let c = lit_const(l, Some(to));
        if let (Some(inst), Some(slot)) =
            (self.insts.get_mut(at), self.values.get_mut(v.0 as usize))
        {
            *slot = match &c {
                ConstValue::Int(s, _) | ConstValue::Float(s, _) => Type::Scalar(*s),
                ConstValue::Bool(_) => Type::Bool,
            };
            inst.op = Op::Const(c);
        }
    }

    // ---- names ----

    fn bind(&mut self, name: &str, slot: Slot) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), slot);
        }
    }

    pub(super) fn lookup(&self, name: &str) -> Option<Slot> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    fn name_value(&mut self, v: ValueId, name: &str) {
        if self.named.insert(v.0) {
            self.cx
                .names
                .push(Named::Value(self.body_ref, v, name.to_string()));
        }
    }

    fn hints(&mut self, at: usize, attrs: &[ast::Attr]) {
        for attr in attrs.iter().filter(|a| a.is_hint()) {
            self.cx.hints.push(Hint {
                anchor: Anchor::Inst(self.body_ref, InstId(at as u32)),
                path: attr.path.iter().map(|p| p.name.clone()).collect(),
                args: attr
                    .args
                    .as_ref()
                    .map(|args| args.values().map(super::hint_arg).collect())
                    .unwrap_or_default(),
            });
        }
    }

    // ---- statements ----

    fn block(&mut self, block: &ast::Block) -> Res<()> {
        self.scopes.push(HashMap::new());
        let result = block
            .stmts
            .lines
            .iter()
            .try_for_each(|line| self.stmt(&line.node, line.span));
        self.scopes.pop();
        result
    }

    fn stmt(&mut self, stmt: &Stmt, span: Span) -> Res<()> {
        match stmt {
            Stmt::Let {
                name,
                ty,
                value,
                attrs,
            } => {
                let want = match ty {
                    Some(t) => Some(self.cx.scalar_type(t)?),
                    None => None,
                };
                let v = self.expr(value, want)?;
                self.bind(&name.name, Slot::Value(v));
                self.name_value(v, &name.name);
                let at = self.insts.len().saturating_sub(1);
                self.hints(at, attrs);
                Ok(())
            }
            Stmt::Var { name, ty, value } => {
                let t = self.cx.ty(ty, &[], self.prog)?;
                let want = match &t {
                    Type::Scalar(s) => Some(*s),
                    _ => None,
                };
                let l = LocalId(self.locals.len() as u32);
                self.locals.push(t);
                self.zeros = self.locals.get(l.0 as usize).cloned();
                let v = self.expr(value, want);
                self.zeros = None;
                let v = v?;
                self.emit0(Op::Set(l, v), span);
                self.bind(&name.name, Slot::Local(l));
                self.cx
                    .names
                    .push(Named::Local(self.body_ref, l, name.name.clone()));
                Ok(())
            }
            Stmt::Assign { target, op, value } => self.assign(target, *op, value, span),
            Stmt::For {
                vars,
                iter,
                attrs,
                body,
            } => {
                let (op, types) = self.for_iter(iter)?;
                let defs = self.emit_n(op, types, span);
                let at = self.insts.len().saturating_sub(1);
                self.hints(at, attrs);
                self.scopes.push(HashMap::new());
                for (name, v) in vars.iter().zip(&defs) {
                    self.bind(&name.name, Slot::Value(*v));
                    self.name_value(*v, &name.name);
                }
                let result = self.block(body);
                self.scopes.pop();
                result?;
                self.emit0(Op::End, span);
                Ok(())
            }
            Stmt::While { cond, body } => {
                self.emit0(Op::While, span);
                let c = self.expr(cond, None)?;
                self.emit0(Op::WhileTest { cond: c }, cond.span);
                self.block(body)?;
                self.emit0(Op::End, span);
                Ok(())
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let c = self.expr(cond, None)?;
                self.emit0(Op::If { cond: c }, span);
                self.block(then)?;
                if let Some(other) = otherwise {
                    self.emit0(Op::Else, other.span);
                    self.block(other)?;
                }
                self.emit0(Op::End, span);
                Ok(())
            }
            Stmt::Store { target, value } => {
                let t = self.expr(target, None)?;
                self.zeros = match self.ty(t) {
                    Type::View(f) => Some(Type::Tile(f)),
                    _ => None,
                };
                let v = self.expr(value, None);
                self.zeros = None;
                let v = v?;
                self.emit0(Op::Store(t, v), span);
                Ok(())
            }
            Stmt::Consume(name) => match self.lookup(&name.name) {
                Some(Slot::Value(v)) => {
                    self.emit0(Op::Consume(v), span);
                    Ok(())
                }
                _ => refuse(name.span, "`consume` of no message"),
            },
            Stmt::Return(None) => {
                self.emit0(Op::Return(None), span);
                Ok(())
            }
            Stmt::Return(Some(e)) => {
                let want = match &self.ret {
                    Some(Type::Scalar(s)) => Some(*s),
                    _ => None,
                };
                let v = self.expr(e, want)?;
                self.emit0(Op::Return(Some(v)), span);
                Ok(())
            }
            Stmt::Expr(e) => self.eval(e, None, false).map(|_| ()),
        }
    }

    fn assign(
        &mut self,
        target: &ast::Expr,
        op: AssignOp,
        value: &ast::Expr,
        span: Span,
    ) -> Res<()> {
        let bin = match op {
            AssignOp::Set => None,
            AssignOp::Add => Some(BinOp::Add),
            AssignOp::Sub => Some(BinOp::Sub),
        };
        match &target.kind {
            ExprKind::Name(n) => {
                let Some(Slot::Local(l)) = self.lookup(&n.name) else {
                    return refuse(n.span, "an assignment to what is not a `var`");
                };
                let t = self.locals.get(l.0 as usize).cloned().unwrap_or(Type::Bool);
                let want = match &t {
                    Type::Scalar(s) => Some(*s),
                    _ => None,
                };
                let v = match bin {
                    None => self.expr(value, want)?,
                    Some(b) => {
                        let old = self.emit(Op::Get(l), t.clone(), target.span);
                        let x = self.expr(value, want)?;
                        self.emit(Op::Binary(b, old, x), t.clone(), span)
                    }
                };
                self.emit0(Op::Set(l, v), span);
                Ok(())
            }
            ExprKind::Index { base, index } => {
                let (member, element) = self.member(base)?;
                let [ast::IndexItem::At(i)] = index.values().collect::<Vec<_>>()[..] else {
                    return refuse(
                        target.span,
                        "a state element written by more than one index",
                    );
                };
                let i = self.expr(i, Some(ScalarType::U32))?;
                let want = match &element {
                    Type::Scalar(s) => Some(*s),
                    _ => None,
                };
                let v = match bin {
                    None => self.expr(value, want)?,
                    Some(b) => {
                        let old = self.emit(Op::StateGet(member, i), element.clone(), target.span);
                        let x = self.expr(value, want)?;
                        self.emit(Op::Binary(b, old, x), element.clone(), span)
                    }
                };
                self.emit0(Op::StateSet(member, i, v), span);
                Ok(())
            }
            _ => refuse(target.span, "an assignment to what Lower does not write"),
        }
    }

    // ---- expressions ----

    /// An expression that has a value.
    pub(super) fn expr(&mut self, e: &ast::Expr, want: Option<ScalarType>) -> Res<ValueId> {
        match self.eval(e, want, true)? {
            Some(v) => Ok(v),
            None => refuse(e.span, "an expression with no value where one is needed"),
        }
    }

    /// An expression; `used` tells whether its value is read.
    pub(super) fn eval(
        &mut self,
        e: &ast::Expr,
        want: Option<ScalarType>,
        used: bool,
    ) -> Res<Option<ValueId>> {
        if let Some(l) = lit(e) {
            let v = self.constant(lit_const(l, want), e.span);
            if want.is_none() {
                self.literals.insert(v.0, (self.insts.len() - 1, l));
            }
            return Ok(Some(v));
        }
        match &e.kind {
            ExprKind::Bool(b) => Ok(Some(self.constant(ConstValue::Bool(*b), e.span))),
            ExprKind::Paren(x) => self.eval(x, want, used),
            ExprKind::Name(n) => self.name(n).map(Some),
            ExprKind::Neg(x) => {
                let v = self.expr(x, want)?;
                let t = self.ty(v);
                Ok(Some(self.emit(Op::Unary(UnaryOp::Neg, v), t, e.span)))
            }
            ExprKind::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs, want, e.span).map(Some),
            _ => self.form(e, want, used),
        }
    }

    fn name(&mut self, n: &ast::Ident) -> Res<ValueId> {
        match self.lookup(&n.name) {
            Some(Slot::Value(v)) => return Ok(v),
            Some(Slot::Local(l)) => {
                let t = self.locals.get(l.0 as usize).cloned().unwrap_or(Type::Bool);
                return Ok(self.emit(Op::Get(l), t, n.span));
            }
            None => {}
        }
        if let Some((p, ty, value)) = self.cx.params.get(&n.name).cloned() {
            return Ok(match value {
                Some(c) => self.constant(c, n.span),
                None => self.emit(Op::Param(p), Type::Scalar(ty), n.span),
            });
        }
        if n.name == "zeros" {
            if let Some(t) = self.zeros.clone() {
                return Ok(self.emit(Op::Zeros, t, n.span));
            }
        }
        // a member of one element, read by its name: its element 0
        let names = self
            .prog
            .and_then(|p| self.cx.program_names.get(p.0 as usize));
        let one = names
            .and_then(|p| p.members.get(&n.name))
            .and_then(|(m, kind, count)| (*count == Count::Const(1)).then(|| (*m, kind.clone())));
        if let Some((m, kind)) = one {
            let zero = self.constant(int_const(0, Some(ScalarType::U32)), n.span);
            let t = match kind {
                MemberKind::Tiles(f) => Type::Tile(f),
                MemberKind::Scalars(s) => Type::Scalar(s),
            };
            return Ok(self.emit(Op::StateGet(m, zero), t, n.span));
        }
        if let Some(f) = Failure::from_name(&n.name) {
            return Ok(self.emit(Op::FailureName(f), Type::Reason, n.span));
        }
        refuse(n.span, format!("the name {} as a value", n.name))
    }

    fn binary(
        &mut self,
        op: ast::BinOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
        want: Option<ScalarType>,
        span: Span,
    ) -> Res<ValueId> {
        use ast::BinOp as A;
        let logic = matches!(op, A::Or | A::And);
        let compare = matches!(op, A::Eq | A::Ne | A::Lt | A::Le | A::Gt | A::Ge);
        let inner = if logic || compare { None } else { want };
        let a = self.expr(lhs, inner)?;
        let b = self.expr(rhs, inner)?;
        // a bare literal takes the scalar type of the other operand
        match (self.scalar(a), self.scalar(b)) {
            (Some(_), Some(t))
                if self.literals.contains_key(&a.0) && !self.literals.contains_key(&b.0) =>
            {
                self.retype(a, t)
            }
            (Some(t), Some(_))
                if self.literals.contains_key(&b.0) && !self.literals.contains_key(&a.0) =>
            {
                self.retype(b, t)
            }
            _ => {}
        }
        self.literals.remove(&a.0);
        self.literals.remove(&b.0);
        let cmp = |c: CmpOp| Op::Compare(c, a, b);
        let (op, ty) = match op {
            A::Or => (Op::LogicOr(a, b), Type::Bool),
            A::And => (Op::LogicAnd(a, b), Type::Bool),
            A::Eq => (cmp(CmpOp::Eq), Type::Bool),
            A::Ne => (cmp(CmpOp::Ne), Type::Bool),
            A::Lt => (cmp(CmpOp::Lt), Type::Bool),
            A::Le => (cmp(CmpOp::Le), Type::Bool),
            A::Gt => (cmp(CmpOp::Gt), Type::Bool),
            A::Ge => (cmp(CmpOp::Ge), Type::Bool),
            other => {
                let b_op = match other {
                    A::BitOr => BinOp::Or,
                    A::BitXor => BinOp::Xor,
                    A::BitAnd => BinOp::And,
                    A::Shl => BinOp::Shl,
                    A::Shr => BinOp::Shr,
                    A::Add => BinOp::Add,
                    A::Sub => BinOp::Sub,
                    A::Mul => BinOp::Mul,
                    A::Div => BinOp::Div,
                    _ => BinOp::Rem,
                };
                let ty = match (self.ty(a), self.ty(b)) {
                    (t @ Type::Tile(_), _) | (_, t @ Type::Tile(_)) => t,
                    (t @ Type::Reduced(..), _) | (_, t @ Type::Reduced(..)) => t,
                    (t, _) => t,
                };
                (Op::Binary(b_op, a, b), ty)
            }
        };
        Ok(self.emit(op, ty, span))
    }
}
