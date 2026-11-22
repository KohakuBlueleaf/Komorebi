//! The bodies: every function, handler, run, chain and control program, with its names
//! resolved and every expression typed, and the rules on what each kind of body may use.

use std::collections::{HashMap, HashSet};

use super::ty::{Count, Lit, Shape, Ty};
use crate::syntax::ast::{
    Arg, AssignOp, Attr, BinOp, Block, Call, Chain, Control, Expr, ExprKind, Extern, Func, Handler,
    Ident, IndexItem, Item, List, Module, ParamDecl, Program, ProgramItem, Run, RunPart, Span,
    Stmt,
};
use crate::types::{Axis, Failure, Format, ScalarType};

use super::attrs::{named_args, on_grid};
use super::codes;
use super::env::{is_named, parse_int, Declared, Env, Names, Place};
use super::op::OpKind;
use super::views::peers_of;
use super::{Checker, LocalKind, Name, Word};
use crate::types::{MemberId, ParamId, UnaryOp};

/// The kind of body being checked: it decides which operations it may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// A `func` or a `combine`.
    Func,
    /// The `fallback` of an `extern func`.
    Fallback,
    /// A handler.
    Handler,
    /// The `seed` of a dynamic run.
    Seed,
    /// An `on M(m)` body of a dynamic run.
    On,
    /// The arguments of a chain's call.
    Chain,
    /// A control program.
    Control,
}

impl Kind {
    /// True in the bodies that run on workers.
    pub(super) fn on_workers(self) -> bool {
        matches!(self, Kind::Handler | Kind::Seed | Kind::On)
    }
}

/// Which part of the language an operation belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Family {
    /// Values: names, numbers, their operators, `min`, `max`, `range`.
    Value,
    /// The compute family: tile operations and calls of functions.
    Compute,
    /// Operations of a body on workers: views, streams, state, collectives, identity.
    Worker,
    /// `send`, `forward`, `consume`, `offer`, and making a message.
    Dynamic,
    /// `invoke`, `await`, `outcome`, `group(...)`.
    Control,
    /// `reduce_to` and `broadcast_from`.
    Islands,
    /// `island.count` and its kin: a worker's or a control program's.
    IslandQuery,
}

/// A view as the text names it, to compare what a stream reads with what it writes.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ViewKey {
    /// The tensor the view is of.
    pub(super) tensor: String,
    /// The view written out, its names replaced by what they stand for.
    pub(super) text: String,
    /// For `t.tiles(first, n)` with both numbers: the tiles `first..first + n`, and what
    /// wraps them.
    pub(super) range: Option<(String, u64, u64)>,
}

/// A rank `offer` sends to, when the text shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rank {
    /// A constant rank.
    Const(u64),
    /// `(worker.rank + d) % group.size`.
    Offset(u64),
}

/// A name in a body.
#[derive(Clone, Debug)]
pub(super) struct Local {
    pub(super) ty: Ty,
    pub(super) binding: Name,
    pub(super) mutable: bool,
    pub(super) view: Option<ViewKey>,
    pub(super) rank: Option<Rank>,
}

/// The peers a run declares.
#[derive(Clone, Debug, Default)]
pub(super) struct Peers {
    pub(super) offsets: Vec<u64>,
    pub(super) pairs: Vec<(u64, u64)>,
}

/// What one body is checked in.
#[derive(Clone)]
pub(super) struct Cx<'e> {
    pub(super) env: &'e Env,
    pub(super) kind: Kind,
    pub(super) program: Option<String>,
    /// The declared result: `None` for none, `Some(Ty::Unknown)` after an error.
    pub(super) ret: Option<Ty>,
    pub(super) grid: bool,
    pub(super) scopes: Vec<HashMap<String, Local>>,
    /// The views a fallback states with `@reads` and with `@writes`.
    pub(super) reads: HashSet<String>,
    pub(super) writes: HashSet<String>,
    /// The peers of the run (`None` when it declares none).
    pub(super) peers: Option<Peers>,
    /// The tile shape of the body (its declaration's `@tile`, else the default).
    pub(super) tile: Option<(u64, u64)>,
    /// In a run's body: the message types the run has an `on` body for.
    pub(super) received: Option<HashSet<String>>,
}

/// The provided combinations of a reduce.
pub(super) const PROVIDED: [&str; 5] = ["sum", "max", "min", "and", "or"];

/// The names of the failures (`o.reason == name`).
pub(super) const FAILURES: [&str; 10] = [
    "view_end",
    "empty_state",
    "send_refused",
    "capacity",
    "given_up",
    "dependency",
    "not_a_peer",
    "over_bound",
    "divide_by_zero",
    "out_of_range",
];

/// The compute functions on one tile.
pub(super) const UNARY: [&str; 15] = [
    "neg",
    "abs",
    "relu",
    "square",
    "floor",
    "exp",
    "exp2",
    "tanh",
    "erf",
    "log",
    "sigmoid",
    "silu",
    "sqrt",
    "rsqrt",
    "reciprocal",
];

/// One operand of an operation: its name, whether it is given by name, whether it may
/// be left out.
pub(super) struct Operand {
    pub(super) name: &'static str,
    pub(super) named: bool,
    pub(super) optional: bool,
}

pub(super) const fn pos(name: &'static str) -> Operand {
    Operand {
        name,
        named: false,
        optional: false,
    }
}

pub(super) const fn named(name: &'static str) -> Operand {
    Operand {
        name,
        named: true,
        optional: false,
    }
}

pub(super) const fn named_opt(name: &'static str) -> Operand {
    Operand {
        name,
        named: true,
        optional: true,
    }
}

impl Cx<'_> {
    pub(super) fn lookup(&self, name: &str) -> Option<&Local> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    pub(super) fn bind(&mut self, name: &str, local: Local) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), local);
        }
    }

    /// What the state member `name` of the program refers to.
    pub(super) fn member_name(&self, name: &str) -> Option<Name> {
        self.env.ix.member(self.program.as_deref()?, name)
    }

    pub(super) fn state(&self, name: &str) -> Option<&super::env::MemberInfo> {
        let program = self.env.programs.get(self.program.as_deref()?)?;
        program.state.get(name)
    }
}

impl Checker<'_> {
    /// Notes a word of the language (`worker`, `row`, `sum` ...) at its name.
    pub(super) fn note_word(&mut self, n: &Ident) {
        if let Some(word) = Word::from_name(&n.name) {
            self.note_name(n.span, Name::Word(word));
        }
    }

    /// Notes the function `n` names.
    pub(super) fn note_func(&mut self, cx: &Cx, n: &Ident) {
        if let Some(&place) = cx.env.ix.funcs.get(&n.name) {
            self.note_name(n.span, Name::Func(place));
        }
    }
}

/// A local of a type, not a view and not a rank.
pub(super) fn local(ty: Ty, binding: Name, mutable: bool) -> Local {
    Local {
        ty,
        binding,
        mutable,
        view: None,
        rank: None,
    }
}

/// The name an expression is, if it is one.
/// True when `expr` may be an operand of a chain's call: one of the chain's parameters
/// (`params`), `past(y)`, or a constant (a number, a negated number, `true`, `false`,
/// a module parameter bound at build).
fn chain_operand(expr: &Expr, params: &[&str], build: &HashSet<String>) -> bool {
    match &expr.kind {
        ExprKind::Name(name) => params.contains(&name.name.as_str()) || build.contains(&name.name),
        ExprKind::Past(_) | ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => true,
        ExprKind::Neg(inner) => matches!(inner.kind, ExprKind::Int(_) | ExprKind::Float(_)),
        _ => false,
    }
}

/// What is wrong with a chain call's `@on` (R47): it is `group[first:end]`, each end a
/// number or a module parameter bound at build, first below end, and end within the
/// chain's group where its size is stated (`@on(grid[R, C])`).
fn call_part(call: &Call, chain: &[Attr], build: &HashSet<String>) -> Option<(String, Span)> {
    let arg = call
        .attrs
        .iter()
        .filter(|a| is_named(a, "on"))
        .find_map(|a| a.args.as_ref())?
        .values()
        .next()?;
    let form = "a call's `@on` is `group[first:end]`, a range of the chain's group".to_string();
    let ExprKind::Index { base, index } = &arg.value.kind else {
        return Some((form, arg.span));
    };
    let items: Vec<&IndexItem> = index.values().collect();
    let (true, [IndexItem::Slice { lo, hi, step: None }]) = (
        matches!(&base.kind, ExprKind::Name(n) if n.name == "group" && !n.quoted),
        items.as_slice(),
    ) else {
        return Some((form, arg.span));
    };
    let end = |e: &Expr| match &e.kind {
        ExprKind::Int(t) => parse_int(t).map(Some),
        ExprKind::Name(n) if build.contains(&n.name) => Some(None),
        _ => None,
    };
    let (Some(first), Some(last)) = (end(lo), end(hi)) else {
        return Some((
            "an end of a call's range is a number or a module parameter bound at build".into(),
            arg.span,
        ));
    };
    if let (Some(a), Some(b)) = (first, last) {
        if a >= b {
            return Some((
                format!("the range {a}:{b} of a call holds no rank"),
                arg.span,
            ));
        }
    }
    let size = chain
        .iter()
        .filter(|a| is_named(a, "on"))
        .filter_map(|a| a.args.as_ref())
        .flat_map(|args| args.values())
        .find_map(|a| match &a.value.kind {
            ExprKind::Index { base, index }
                if as_name(base).is_some_and(|n| n.name == "grid" && !n.quoted) =>
            {
                let dims: Vec<Option<u64>> = index
                    .values()
                    .map(|i| match i {
                        IndexItem::At(Expr {
                            kind: ExprKind::Int(t),
                            ..
                        }) => parse_int(t),
                        _ => None,
                    })
                    .collect();
                match dims.as_slice() {
                    [Some(r), Some(c)] => r.checked_mul(*c),
                    _ => None,
                }
            }
            _ => None,
        });
    match (last, size) {
        (Some(b), Some(n)) if b > n => Some((
            format!("the range of a call ends at rank {b}, past the chain's group of {n}"),
            arg.span,
        )),
        _ => None,
    }
}

/// Where an expression is read as a value: the value of a `let`, a statement alone,
/// or an operand that takes a value of any type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Read {
    /// The value a `let` binds.
    Bound,
    /// A statement alone: an operation done for its effect is one.
    Alone,
    /// An operand.
    Operand,
}

/// The name of the operation `e` (in no brackets) does, for a diagnostic: `zero` of
/// `c.zero()`, `forward` of `forward(m, ...)`, `await` of `await h`.
fn effect_name(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Call { callee, .. } => match &callee.kind {
            ExprKind::Name(n) | ExprKind::Member { name: n, .. } => n.name.clone(),
            _ => "this".to_string(),
        },
        ExprKind::Await(_) => "await".to_string(),
        _ => "this".to_string(),
    }
}

/// The operands of a call the checker accepted, in the order of the callee's
/// parameters `names`: as written when given by position, by their names when given
/// by name.
pub(crate) fn in_parameter_order<'a>(args: &'a List<Arg>, names: &[String]) -> Vec<&'a Arg> {
    if !args.values().any(|a| a.name.is_some()) {
        return args.values().collect();
    }
    names
        .iter()
        .filter_map(|n| {
            args.values()
                .find(|a| a.name.as_ref().is_some_and(|x| x.name == *n))
        })
        .collect()
}

pub(super) fn as_name(expr: &Expr) -> Option<&Ident> {
    match &expr.kind {
        ExprKind::Name(name) => Some(name),
        ExprKind::Paren(inner) => as_name(inner),
        _ => None,
    }
}

/// The operation of a binary operator.
fn binary_kind(op: BinOp) -> OpKind {
    use crate::types::{BinOp as B, CmpOp as C};
    match op {
        BinOp::Or => OpKind::LogicOr,
        BinOp::And => OpKind::LogicAnd,
        BinOp::Eq => OpKind::Compare(C::Eq),
        BinOp::Ne => OpKind::Compare(C::Ne),
        BinOp::Lt => OpKind::Compare(C::Lt),
        BinOp::Le => OpKind::Compare(C::Le),
        BinOp::Gt => OpKind::Compare(C::Gt),
        BinOp::Ge => OpKind::Compare(C::Ge),
        BinOp::BitOr => OpKind::Binary(B::Or),
        BinOp::BitXor => OpKind::Binary(B::Xor),
        BinOp::BitAnd => OpKind::Binary(B::And),
        BinOp::Shl => OpKind::Binary(B::Shl),
        BinOp::Shr => OpKind::Binary(B::Shr),
        BinOp::Add => OpKind::Binary(B::Add),
        BinOp::Sub => OpKind::Binary(B::Sub),
        BinOp::Mul => OpKind::Binary(B::Mul),
        BinOp::Div => OpKind::Binary(B::Div),
        BinOp::Rem => OpKind::Binary(B::Rem),
    }
}

/// The value of a number written in the text, negated or not.
pub(super) fn number_value(e: &Expr) -> Option<f64> {
    match &e.kind {
        ExprKind::Int(text) => parse_int(text).map(|n| n as f64),
        ExprKind::Float(text) => text.parse::<f64>().ok(),
        ExprKind::Paren(inner) => number_value(inner),
        ExprKind::Neg(inner) => number_value(inner).map(|v| -v),
        _ => None,
    }
}

/// True when the scalar type `t` holds the number `value`: an integer type its
/// whole values in range, a float type every value up to its largest.
pub(super) fn holds(t: ScalarType, value: f64) -> bool {
    if !value.is_finite() {
        return false;
    }
    let largest = match t {
        ScalarType::F16 => 65504.0,
        ScalarType::F32 => f32::MAX as f64,
        ScalarType::F64 => f64::MAX,
        _ => {
            let bits = t.bits() as i32;
            let (lo, hi) = if t.is_signed() {
                (-(2f64.powi(bits - 1)), 2f64.powi(bits - 1) - 1.0)
            } else {
                (0.0, 2f64.powi(bits) - 1.0)
            };
            return value.fract() == 0.0 && value >= lo && value <= hi;
        }
    };
    value.abs() <= largest
}

/// The count an expression stands for: a number written in the text, a module
/// parameter (its value when bound at build), `group.size`, or another value.
pub(super) fn count_of(cx: &Cx, e: &Expr) -> Count {
    if let Some(n) = as_int(e) {
        return Count::Known(n);
    }
    if let Some(n) = as_name(e) {
        if cx.lookup(&n.name).is_none() && cx.state(&n.name).is_none() {
            if let Some(count) = cx.env.ix.param_count(&n.name) {
                return count;
            }
        }
        return Count::Unknown;
    }
    if is_word_member(e, "group", "size") {
        return Count::GroupSize;
    }
    Count::Unknown
}

/// What a name stands for in a constant expression of a body: none where a local or a
/// state member of the body hides it, else the module parameter's.
pub(super) fn constant_name(cx: &Cx, n: &Ident) -> super::consts::Known {
    if cx.lookup(&n.name).is_some() || cx.state(&n.name).is_some() {
        return super::consts::Known::NotParam;
    }
    cx.env.ix.known(&n.name, None)
}

/// Whether `e` names a `var` (a handler's block of one row from `zeros`, R48).
fn block_var(cx: &Cx, e: &Expr) -> bool {
    as_name(e)
        .and_then(|n| cx.lookup(&n.name))
        .is_some_and(|l| l.mutable && matches!(l.ty, Ty::Block(..)))
}

/// Whether `a` and `b` are the same name (or the same number).
fn same_name(a: &Expr, b: &Expr) -> bool {
    match (as_name(a), as_name(b), as_int(a), as_int(b)) {
        (Some(x), Some(y), _, _) => x.name == y.name,
        (_, _, Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// The integer an expression is, if it is a literal.
pub(super) fn as_int(expr: &Expr) -> Option<u64> {
    match &expr.kind {
        ExprKind::Int(text) => parse_int(text),
        ExprKind::Paren(inner) => as_int(inner),
        _ => None,
    }
}

/// True for `base.name` with `base` the word `word`.
pub(super) fn is_word_member(expr: &Expr, word: &str, name: &str) -> bool {
    match &expr.kind {
        ExprKind::Member { base, name: n } => {
            n.name == name && as_name(base).is_some_and(|b| b.name == word && !b.quoted)
        }
        ExprKind::Paren(inner) => is_word_member(inner, word, name),
        _ => false,
    }
}

impl Checker<'_> {
    /// Checks every body of the module.
    pub(crate) fn bodies(&mut self, module: &Module, env: &Env) {
        let mut at = 0;
        for item in module.items.nodes() {
            match item {
                Item::Func(f) | Item::Combine(f) => self.func(env, f),
                Item::Extern(f) => self.fallback(env, f),
                Item::Program(p) => self.program(env, p),
                Item::Control(c) => self.control(env, c),
                Item::Param(p) => {
                    // a module parameter is a scalar (a `bool` is `K109`'s)
                    let declared = env.params.get(&p.name.name);
                    if !matches!(
                        declared,
                        None | Some(Ty::Scalar(_) | Ty::Bool | Ty::Unknown)
                    ) {
                        self.error(
                            codes::WRONG_TYPE,
                            "a module parameter is a scalar".into(),
                            p.ty.span,
                        );
                        at += 1;
                        continue;
                    }
                    if let crate::syntax::ast::ParamValue::Expr(value) = &p.value {
                        let before = self.diagnostics.len();
                        let mut cx = self.cx(env, Kind::Chain, None);
                        let ty = self.expr(&mut cx, value);
                        let want = env.params.get(&p.name.name).cloned().unwrap_or(Ty::Unknown);
                        self.fit(value, &ty, &want, "the parameter's value");
                        // a constant expression, naming only the parameters bound at
                        // build declared before it, of a value its type holds
                        if let (Ty::Scalar(s), true) = (want, self.diagnostics.len() == before) {
                            let place = ParamId(at);
                            let look = |n: &Ident| env.ix.known(&n.name, Some(place));
                            self.constant_at(value, s, "the parameter's value", &look);
                        }
                    }
                    at += 1;
                }
                _ => {}
            }
        }
        self.untyped_zeros();
    }

    /// `K102`: a `zeros` whose place gave it no format (R50: it has the type of its
    /// place).
    fn untyped_zeros(&mut self) {
        let mut at: Vec<(usize, usize)> = self
            .ops
            .iter()
            .filter(|(_, r)| r.kind == OpKind::Zeros)
            .map(|(key, _)| *key)
            .filter(|key| matches!(self.types.get(key), Some(Ty::Tile(None))))
            .collect();
        at.sort_unstable();
        for (start, end) in at {
            self.error(
                codes::WRONG_TYPE,
                "`zeros` takes its type from its place: a declared type, a parameter or a \
                 result"
                    .into(),
                Span::new(start, end),
            );
        }
    }

    pub(super) fn cx<'e>(&mut self, env: &'e Env, kind: Kind, program: Option<String>) -> Cx<'e> {
        self.scope = program.clone();
        Cx {
            env,
            kind,
            program,
            ret: None,
            grid: false,
            scopes: vec![HashMap::new()],
            reads: HashSet::new(),
            writes: HashSet::new(),
            peers: None,
            tile: env.tile,
            received: None,
        }
    }

    /// Binds parameters with the types the declarations were given.
    pub(super) fn bind_params(&mut self, cx: &mut Cx, params: &List<ParamDecl>, types: &[Ty]) {
        for (i, p) in params.values().enumerate() {
            let ty = types.get(i).cloned().unwrap_or(Ty::Unknown);
            cx.bind(&p.name.name, local(ty, Name::Param(i as u32), false));
        }
    }

    pub(super) fn func(&mut self, env: &Env, f: &Func) {
        let mut cx = self.cx(env, Kind::Func, None);
        cx.tile = super::shapes::shape_of(&env.tiles, &f.attrs);
        let sig = env.funcs.get(&f.name.name).cloned();
        let params: Vec<Ty> = sig.as_ref().map(|s| s.params.clone()).unwrap_or_default();
        let ret = sig.map(|s| s.ret).unwrap_or(Ty::Unknown);
        // A function declared on blocks: its body is typed on blocks (R45), each
        // operation's shapes checked; an element-by-element body runs on each tile.
        let on_blocks = params.iter().any(|t| matches!(t, Ty::Block(..)));
        cx.ret = Some(ret);
        self.bind_params(&mut cx, &f.params, &params);
        self.on_blocks = on_blocks;
        self.block(&mut cx, &f.body);
        self.on_blocks = false;
        if on_blocks {
            self.whole_block_ops(&f.name.name, f.body.span);
        }
    }

    /// `K010` for each operation in `span` (a function on blocks) that works on a whole
    /// tile and is named in the language on a block but not built. Built: a reduction
    /// along a row (along each row of the block, R43), `.col()` of its values, `.row()`
    /// of a tile parameter (on its own tile), and `matmul` of blocks (R45).
    fn whole_block_ops(&mut self, name: &str, span: Span) {
        let operand = |r: &super::op::Resolved| r.parts.first().copied().flatten();
        let mut found: Vec<(usize, usize, &'static str)> = self
            .ops
            .iter()
            .filter(|((start, end), _)| *start >= span.start && *end <= span.end)
            .filter_map(|(&(start, end), r)| {
                let of = operand(r).map(|s| (s.start, s.end));
                let what = match r.kind {
                    OpKind::Reduce(_, Axis::Row) => return None,
                    OpKind::Reduce(_, Axis::Col) => "a reduction along a column",
                    OpKind::SpreadCol => match of.and_then(|k| self.types.get(&k)) {
                        Some(Ty::Reduced(_, Axis::Row, _)) => return None,
                        _ => "`.col()` of other than a reduction along a row",
                    },
                    OpKind::SpreadRow => match of.and_then(|k| self.names.get(&k)) {
                        Some(Name::Param(_)) => return None,
                        _ => "`.row()` of other than a tile parameter",
                    },
                    OpKind::Transpose => "`.T`",
                    _ => return None,
                };
                Some((start, end, what))
            })
            .collect();
        found.sort();
        for (start, end, what) in found {
            self.error(
                codes::NOT_BUILT,
                format!(
                    "{what} in `{name}`, a function on blocks, is named in the language and not \
                     built yet: a function on blocks works on each tile with elementwise \
                     operations, reduces along each row of the block and spreads it back \
                     with `.col()`, and multiplies blocks with `matmul`"
                ),
                Span { start, end },
            );
        }
    }

    pub(super) fn fallback(&mut self, env: &Env, f: &Extern) {
        let Some(body) = &f.fallback else {
            return;
        };
        let mut cx = self.cx(env, Kind::Fallback, None);
        cx.tile = super::shapes::shape_of(&env.tiles, &f.attrs);
        let sig = env.funcs.get(&f.name.name).cloned();
        let params: Vec<Ty> = sig.as_ref().map(|s| s.params.clone()).unwrap_or_default();
        cx.ret = Some(sig.map(|s| s.ret).unwrap_or(Ty::Unknown));
        cx.reads = named_args(&f.attrs, "reads");
        cx.writes = named_args(&f.attrs, "writes");
        self.bind_params(&mut cx, &f.params, &params);
        self.block(&mut cx, body);
    }

    pub(super) fn control(&mut self, env: &Env, c: &Control) {
        let mut cx = self.cx(env, Kind::Control, None);
        let params = self.params(env, &c.params);
        cx.ret = c.ret.as_ref().map(|t| self.ty(env, t, Place::Result));
        self.bind_params(&mut cx, &c.params, &params);
        self.block(&mut cx, &c.body);
    }

    pub(super) fn program(&mut self, env: &Env, p: &Program) {
        let name = p.name.name.clone();
        let info = env.programs.get(&name);
        let params_of = |h: &str| -> Vec<Ty> {
            info.and_then(|i| i.invocables.get(h))
                .map(|i| i.params.clone())
                .unwrap_or_default()
        };
        for part in p.body.nodes() {
            match part {
                ProgramItem::Handler(h) => {
                    let mut cx = self.cx(env, Kind::Handler, Some(name.clone()));
                    cx.grid = on_grid(&h.attrs);
                    cx.tile = super::shapes::shape_of(&env.tiles, &h.attrs);
                    cx.ret = info
                        .and_then(|i| i.invocables.get(&h.name.name))
                        .and_then(|i| i.ret.clone());
                    self.bind_params(&mut cx, &h.params, &params_of(&h.name.name));
                    self.handler_body(&mut cx, h);
                }
                ProgramItem::Run(r) => {
                    self.run(env, &name, r, &params_of(&r.name.name));
                }
                ProgramItem::Chain(c) => self.chain(env, &name, c, &params_of(&c.name.name)),
                _ => {}
            }
        }
        // K301 (a member read that no body fills) is Verify's, on the derived effects
    }

    pub(super) fn handler_body(&mut self, cx: &mut Cx, h: &Handler) {
        // A broadcast's arrivals have the format its casts send; where a cast comes
        // after the arrivals in the text, the body is checked again knowing it.
        let again = cx.grid.then(|| {
            (
                cx.clone(),
                self.diagnostics.len(),
                self.undeclared_formats.clone(),
                self.tile_missing,
            )
        });
        self.late.clear();
        self.made.clear();
        self.block(cx, &h.body);
        let known = self.late.iter().any(|at| self.casts.contains_key(at));
        if let (true, Some((mut fresh, errors, formats, tile_missing))) = (known, again) {
            let (start, end) = (h.body.span.start, h.body.span.end);
            let inside = |k: &(usize, usize)| k.0 >= start && k.1 <= end;
            self.diagnostics.truncate(errors);
            self.undeclared_formats = formats;
            self.tile_missing = tile_missing;
            self.types.retain(|k, _| !inside(k));
            self.names.retain(|k, _| !inside(k));
            self.ops.retain(|k, _| !inside(k));
            self.scopes.retain(|k, _| !inside(k));
            // the casts' formats stay known; each cast meets its own again
            self.block(&mut fresh, &h.body);
        }
        let made: Vec<Span> = std::mem::take(&mut self.made);
        let mut seen = HashSet::new();
        for span in made {
            let at = (span.start, span.end);
            if seen.insert(at) && !self.casts.contains_key(&at) {
                self.error(
                    codes::BROADCAST_NO_CAST,
                    "no cast sends on this broadcast: it carries nothing, and its arrivals have \
                     no format"
                        .into(),
                    span,
                );
            }
        }
    }

    pub(super) fn run(&mut self, env: &Env, program: &str, r: &Run, params: &[Ty]) {
        let peers = r
            .attrs
            .iter()
            .find(|a| is_named(a, "peers"))
            .map(|a| peers_of(a, &env.ix));
        // the message types the run receives: one `on` body each
        let mut received = HashSet::new();
        for part in r.body.nodes() {
            if let RunPart::On(on) = part {
                if !received.insert(on.message.name.clone()) {
                    self.error(
                        codes::NAME_TWICE,
                        format!("the run has an `on` body for `{}` already", on.message.name),
                        on.message.span,
                    );
                }
            }
        }
        for part in r.body.nodes() {
            let (kind, block, on) = match part {
                RunPart::Seed(block) => (Kind::Seed, block, None),
                RunPart::On(on) => (Kind::On, &on.body, Some(on)),
            };
            let mut cx = self.cx(env, kind, Some(program.to_string()));
            cx.peers = peers.clone();
            cx.received = Some(received.clone());
            cx.tile = super::shapes::shape_of(&env.tiles, &r.attrs);
            self.bind_params(&mut cx, &r.params, params);
            if let Some(on) = on {
                let fields = env.message(Some(program), &on.message.name);
                if let Some(place) = env.ix.message_ref(Some(program), &on.message.name) {
                    self.note_name(on.message.span, Name::Message(place));
                }
                let ty = if fields.is_some() {
                    Ty::Message(on.message.name.clone())
                } else {
                    self.error(
                        codes::UNKNOWN_NAME,
                        format!("`{}` is not a message of `{program}`", on.message.name),
                        on.message.span,
                    );
                    Ty::Unknown
                };
                self.bare_word(&on.binding, Declared::Value);
                cx.bind(&on.binding.name, local(ty, Name::Arrived, false));
                self.message_once(on);
            }
            self.block(&mut cx, block);
        }
    }

    pub(super) fn chain(&mut self, env: &Env, program: &str, c: &Chain, params: &[Ty]) {
        let mut cx = self.cx(env, Kind::Chain, Some(program.to_string()));
        self.bind_params(&mut cx, &c.params, params);
        let names: Vec<&str> = c.params.values().map(|p| p.name.name.as_str()).collect();
        for call in c.calls.nodes() {
            for arg in call.args.values() {
                if !chain_operand(&arg.value, &names, &env.ix.build_params) {
                    self.error(
                        codes::CHAIN_OPERAND,
                        "an operand of a chain's call is a parameter of the chain, `past(y)` or \
                         a constant (a module parameter bound at build is one)"
                            .into(),
                        arg.value.span,
                    );
                }
            }
            if let Some((what, span)) = call_part(call, &c.attrs, &env.ix.build_params) {
                self.error(codes::CALL_PART, what, span);
            }
            self.chain_call(&mut cx, program, call);
        }
    }

    pub(super) fn chain_call(&mut self, cx: &mut Cx, program: &str, call: &Call) {
        let target = cx
            .env
            .programs
            .get(program)
            .and_then(|p| p.invocables.get(&call.handler.name))
            .filter(|i| i.is_handler)
            .cloned();
        let Some(target) = target else {
            self.error(
                codes::UNKNOWN_NAME,
                format!("`{}` is not a handler of `{program}`", call.handler.name),
                call.handler.span,
            );
            return;
        };
        if let Some(place) = cx.env.ix.invocable_of(program, &call.handler.name) {
            self.note_name(call.handler.span, place);
        }
        if target.ret.is_some() {
            self.error(
                codes::WRONG_TYPE,
                format!(
                    "a chain has no result, and `{}` gives one",
                    call.handler.name
                ),
                call.span,
            );
            return;
        }
        let what = &call.handler.name;
        let Some(args) = self.call_operands(what, call.span, &call.args, &target.names) else {
            return;
        };
        self.positional_call(cx, what, call.span, &args, &target.params);
    }

    /// The operands of a call of a function, a chain's `call` or what `invoke` names,
    /// in the order of the callee's parameters `names`: all by position, or all by
    /// name, each name a parameter (`K101`), each parameter given (`K103`) and given
    /// once (`K112`); some by position and some by name is `K111`. `None` after a
    /// diagnostic.
    pub(super) fn call_operands<'a>(
        &mut self,
        what: &str,
        span: Span,
        args: &'a List<Arg>,
        names: &[String],
    ) -> Option<Vec<&'a Expr>> {
        let first = args.values().next().map(|a| a.name.is_some());
        if let Some(arg) = args.values().find(|a| Some(a.name.is_some()) != first) {
            self.error(
                codes::OPERAND_FORM,
                "the operands of a call are all by position or all by name".into(),
                arg.span,
            );
            return None;
        }
        if first != Some(true) {
            return Some(args.values().map(|a| &a.value).collect());
        }
        let mut given: Vec<Option<&Expr>> = vec![None; names.len()];
        for arg in args.values() {
            let Some(name) = &arg.name else {
                continue;
            };
            match names.iter().position(|n| *n == name.name) {
                None => {
                    self.error(
                        codes::UNKNOWN_NAME,
                        format!("`{}` is not a parameter of `{what}`", name.name),
                        name.span,
                    );
                    return None;
                }
                Some(k) if given[k].is_some() => {
                    self.error(
                        codes::NAME_TWICE,
                        format!("the parameter `{}` is given twice", name.name),
                        arg.span,
                    );
                    return None;
                }
                Some(k) => given[k] = Some(&arg.value),
            }
        }
        let missing: Vec<&str> = names
            .iter()
            .zip(&given)
            .filter(|(_, g)| g.is_none())
            .map(|(n, _)| n.as_str())
            .collect();
        if !missing.is_empty() {
            self.error(
                codes::MISSING_OPERAND,
                format!("`{what}` is not given `{}`", missing.join("`, `")),
                span,
            );
            return None;
        }
        Some(given.into_iter().flatten().collect())
    }

    /// A call's operands, in the order of the callee's parameters (`call_operands`),
    /// checked against the parameters' types. In a function on blocks (R45) a function
    /// on tiles takes blocks of one shape, run on each tile: that shape is given back.
    pub(super) fn positional_call(
        &mut self,
        cx: &mut Cx,
        what: &str,
        span: Span,
        args: &[&Expr],
        params: &[Ty],
    ) -> Option<Shape> {
        if args.len() < params.len() {
            self.error(
                codes::MISSING_OPERAND,
                format!(
                    "`{what}` takes {} operands and {} are given",
                    params.len(),
                    args.len()
                ),
                span,
            );
            return None;
        }
        if args.len() > params.len() {
            self.error(
                codes::WRONG_TYPE,
                format!(
                    "`{what}` takes {} operands and {} are given",
                    params.len(),
                    args.len()
                ),
                span,
            );
            return None;
        }
        let on_tiles = !params.iter().any(|t| matches!(t, Ty::Block(..)));
        let mut lifted: Option<Shape> = None;
        let mut tile_given = false;
        for (arg, want) in args.iter().zip(params) {
            let ty = self.expr(cx, arg);
            tile_given |= matches!((want, &ty), (Ty::Tile(_), Ty::Tile(_)));
            match (want, &ty) {
                // a function on tiles in a function on blocks, or in a handler on its
                // vars of a block of one row (R48): on each tile of one shape
                (Ty::Tile(_), Ty::Block(_, shape))
                    if on_tiles
                        && (self.on_blocks
                            || shape.rows == Count::Known(1) && block_var(cx, arg)) =>
                {
                    match lifted {
                        Some(first) if !first.agrees(*shape) => self.error(
                            codes::WRONG_TYPE,
                            format!(
                                "`{what}` is given blocks of two shapes, {first} and {shape}: \
                                 a function on tiles runs on each tile of blocks of one shape"
                            ),
                            arg.span,
                        ),
                        _ => lifted = Some(*shape),
                    }
                }
                (Ty::Tile(_), Ty::Block(..)) => self.error(
                    codes::BLOCK_TO_TILE,
                    format!("a block of several tiles is given to `{what}`, declared on one tile"),
                    arg.span,
                ),
                (Ty::Block(_, n), Ty::Block(_, m)) => {
                    if !n.agrees(*m) {
                        self.error(
                            codes::WRONG_TYPE,
                            format!(
                                "a block {} is given to `{what}`, declared on a block {}",
                                shape_text(m),
                                shape_text(n)
                            ),
                            arg.span,
                        );
                    }
                }
                _ => {
                    self.fit(arg, &ty, want, &format!("an operand of `{what}`"));
                }
            }
        }
        if let (Some(shape), true) = (lifted, tile_given) {
            self.error(
                codes::WRONG_TYPE,
                format!(
                    "`{what}` is given a tile beside blocks {shape}: a function on tiles runs \
                     on each tile of blocks of one shape"
                ),
                span,
            );
        }
        lifted
    }

    /// Reports `K102` when `ty` cannot stand where `want` is asked.
    pub(super) fn expect_fit(&mut self, ty: &Ty, want: &Ty, span: Span, what: &str) {
        if matches!(ty, Ty::Reduced(..)) && matches!(want, Ty::Tile(_)) {
            self.error(
                codes::REDUCED_VALUE,
                format!("{what} is {ty}, not a tile; `.col()` or `.row()` spreads it into a tile"),
                span,
            );
        } else if !ty.fits(want) {
            // a tile against a block in a function on blocks: a tile is one shape of its own
            let why = match (ty, want) {
                (Ty::Tile(_), Ty::Block(..)) | (Ty::Block(..), Ty::Tile(_)) if self.on_blocks => {
                    " (in a function on blocks a tile is `[1, 1]` and a block is of its shape)"
                }
                _ => "",
            };
            self.error(
                codes::WRONG_TYPE,
                format!("{what} is {ty}, where {want} is asked{why}"),
                span,
            );
        }
    }

    /// Checks that `e`, of type `ty`, may stand where `want` is asked, and gives a
    /// number written in the text (or `zeros`) the type of that place.
    pub(super) fn fit(&mut self, e: &Expr, ty: &Ty, want: &Ty, what: &str) -> Ty {
        self.expect_fit(ty, want, e.span, what);
        self.place(e, ty, want)
    }

    /// Gives a number written in the text the type of its place `want` (a scalar
    /// type), and `zeros` the format of its place (a tile); the type `e` has then.
    pub(super) fn place(&mut self, e: &Expr, ty: &Ty, want: &Ty) -> Ty {
        match (ty, want) {
            (Ty::Number(_), Ty::Scalar(s)) if ty.fits(want) => {
                self.settle(e, *s);
                Ty::Scalar(*s)
            }
            (Ty::Number(_), _) => self.alone(e, ty.clone()),
            (Ty::Tile(None), Ty::Tile(Some(f))) => {
                self.settle_tile(e, *f);
                Ty::Tile(Some(*f))
            }
            (Ty::Tile(Some(f)), Ty::Tile(Some(g))) if f != g && ty.fits(want) => {
                if self.computed(e, *g) {
                    Ty::Tile(Some(*g))
                } else {
                    ty.clone()
                }
            }
            _ => ty.clone(),
        }
    }

    /// R33: a tile a compute operation yields has the format of its place, `g`. True
    /// when `e` is such an operation (in brackets or not), noted so; a name, a read of
    /// data and a function's call keep their own format.
    fn computed(&mut self, e: &Expr, g: Format) -> bool {
        if let ExprKind::Paren(inner) = &e.kind {
            if !self.computed(inner, g) {
                return false;
            }
            self.note_type(e, &Ty::Tile(Some(g)));
            return true;
        }
        let compute = matches!(
            self.ops.get(&(e.span.start, e.span.end)).map(|r| &r.kind),
            Some(
                OpKind::Binary(_)
                    | OpKind::Unary(_)
                    | OpKind::Matmul
                    | OpKind::Select
                    | OpKind::SpreadCol
                    | OpKind::SpreadRow
                    | OpKind::Transpose
            )
        );
        if compute {
            self.note_type(e, &Ty::Tile(Some(g)));
        }
        compute
    }

    /// Two scalar operands of one operation (`what`), which have one type: a number
    /// written in the text takes the other operand's. The type of both, or `Unknown`
    /// after a diagnostic.
    pub(super) fn one_type(
        &mut self,
        what: &str,
        a: &Expr,
        ta: &Ty,
        b: &Expr,
        tb: &Ty,
        span: Span,
    ) -> Ty {
        match (ta, tb) {
            (Ty::Scalar(x), Ty::Scalar(y)) => {
                if x == y {
                    Ty::Scalar(*x)
                } else {
                    self.mixed(what, *x, *y, span);
                    Ty::Unknown
                }
            }
            (Ty::Scalar(_), Ty::Number(_)) => self.fit(b, tb, ta, "an operand"),
            (Ty::Number(_), Ty::Scalar(_)) => self.fit(a, ta, tb, "an operand"),
            (Ty::Number(x), Ty::Number(y)) => {
                if *x == Lit::Point || *y == Lit::Point {
                    Ty::Number(Lit::Point)
                } else {
                    Ty::Number(Lit::Whole)
                }
            }
            _ => Ty::Unknown,
        }
    }

    /// Notes the type `s` for `e`, a number written in the text, and for the numbers
    /// it is made of. A number that `s` cannot hold is `K115`.
    pub(super) fn settle(&mut self, e: &Expr, s: ScalarType) {
        self.settle_checked(e, s, true);
    }

    fn settle_checked(&mut self, e: &Expr, s: ScalarType, check: bool) {
        let key = (e.span.start, e.span.end);
        if !matches!(self.types.get(&key), Some(Ty::Number(_))) {
            return;
        }
        self.types.insert(key, Ty::Scalar(s));
        let value = number_value(e);
        if let (true, Some(value)) = (check, value) {
            if !holds(s, value) {
                self.error(
                    codes::OUT_OF_RANGE,
                    format!(
                        "the number {} has no value in {s}, the type its place gives it",
                        &self.source[e.span.start..e.span.end]
                    ),
                    e.span,
                );
            }
        }
        let check = check && value.is_none();
        match &e.kind {
            ExprKind::Paren(inner) | ExprKind::Neg(inner) => self.settle_checked(inner, s, check),
            ExprKind::Binary { op, lhs, rhs } => {
                self.settle(lhs, s);
                if !matches!(op, BinOp::Shl | BinOp::Shr) {
                    self.settle(rhs, s);
                }
            }
            ExprKind::Call { args, .. } => {
                for arg in args.values() {
                    self.settle(&arg.value, s);
                }
            }
            _ => {}
        }
    }

    /// Notes the format `f` for `e`, a tile of no format yet (`zeros`), and for the
    /// tiles of no format it is made of.
    pub(super) fn settle_tile(&mut self, e: &Expr, f: Format) {
        let key = (e.span.start, e.span.end);
        if !matches!(self.types.get(&key), Some(Ty::Tile(None))) {
            return;
        }
        self.types.insert(key, Ty::Tile(Some(f)));
        match &e.kind {
            ExprKind::Paren(inner) | ExprKind::Neg(inner) => self.settle_tile(inner, f),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.settle_tile(lhs, f);
                self.settle_tile(rhs, f);
            }
            _ => {}
        }
    }

    /// A value whose place gives it no type: a number written in the text takes its
    /// own (`u32` whole, `f32` with a point).
    pub(super) fn alone(&mut self, e: &Expr, ty: Ty) -> Ty {
        match ty {
            Ty::Number(lit) => {
                self.settle(e, lit.alone());
                Ty::Scalar(lit.alone())
            }
            ty => ty,
        }
    }

    /// `K102` where `e`, of type `ty`, is read as a value (`place`) and is none: an
    /// operation that gives no value, a state member of scalars named whole, a list, a
    /// string or a tuple, or what a `for` walks. True when reported.
    pub(super) fn not_a_value(&mut self, cx: &Cx, e: &Expr, ty: &Ty, place: Read) -> bool {
        let mut inner = e;
        while let ExprKind::Paren(x) = &inner.kind {
            inner = x;
        }
        let text = match (&inner.kind, ty) {
            (_, Ty::Void) if place != Read::Alone => {
                format!("`{}` gives no value", effect_name(inner))
            }
            (ExprKind::Name(n), Ty::Scalars(..))
                if cx.lookup(&n.name).is_none() && cx.state(&n.name).is_some() =>
            {
                "a state member of scalars is read by index: `c[i]`".to_string()
            }
            // a member of tiles is given whole only to an operation that takes it
            (ExprKind::Name(n), Ty::Tile(_) | Ty::Block(..))
                if place != Read::Operand
                    && cx.lookup(&n.name).is_none()
                    && cx.state(&n.name).is_some() =>
            {
                "a state member of tiles is read by index: `w[i]`".to_string()
            }
            (_, Ty::TileGrid(_)) => {
                "`t.tiles` names its tiles: `t.tiles(first, n)` or `t.tiles[r0:r1, c0:c1]`"
                    .to_string()
            }
            (ExprKind::List(_), _) => "a list is not a value here".to_string(),
            (ExprKind::Str(_), _) => "a string is not a value here".to_string(),
            (ExprKind::Tuple(_), _) => "a tuple is not a value here".to_string(),
            (_, Ty::Range(_)) => "`range(n)` is what a `for` walks, not a value".to_string(),
            (_, Ty::Zip(_)) => "`zip(...)` is what a `for` walks, not a value".to_string(),
            (_, Ty::Arrive(_)) => "`.arrive()` is what a `for` walks, not a value".to_string(),
            // a stream bound by `let` is walked by a `for` later
            (kind, Ty::Stream(..)) if place != Read::Bound || matches!(kind, ExprKind::Name(_)) => {
                "`stream(...)` is what a `for` walks, not a value".to_string()
            }
            _ => return false,
        };
        self.error(codes::WRONG_TYPE, text, e.span);
        true
    }

    // ---- Statements ---------------------------------------------------------------

    pub(super) fn block(&mut self, cx: &mut Cx, block: &Block) {
        cx.scopes.push(HashMap::new());
        let mut names = Names::default();
        for line in &block.stmts.lines {
            self.stmt(cx, &line.node, line.span, &mut names);
        }
        cx.scopes.pop();
    }

    pub(super) fn stmt(&mut self, cx: &mut Cx, stmt: &Stmt, span: Span, names: &mut Names) {
        match stmt {
            Stmt::Let {
                name, ty, value, ..
            } => {
                names.declare(self, name);
                self.bare_word(name, Declared::Value);
                let mut got = self.expr(cx, value);
                if self.not_a_value(cx, value, &got, Read::Bound) {
                    got = Ty::Unknown;
                }
                let ty = match ty {
                    Some(t) => {
                        let want = self.ty(cx.env, t, Place::Local);
                        let got = self.fit(value, &got, &want, &format!("`{}`", name.name));
                        if matches!(want, Ty::Unknown) {
                            got
                        } else {
                            want
                        }
                    }
                    None => self.alone(value, got),
                };
                let view = self.view_key(cx, value);
                let rank = self.rank_of(cx, value);
                cx.bind(
                    &name.name,
                    Local {
                        ty,
                        binding: Name::Local {
                            decl: name.span,
                            kind: LocalKind::Let,
                        },
                        mutable: false,
                        view,
                        rank,
                    },
                );
            }
            Stmt::Var { name, ty, value } => {
                names.declare(self, name);
                self.bare_word(name, Declared::Value);
                let got = self.expr(cx, value);
                // `var v: tile<F>[n] = zeros`: a block of one row of zeros (R48)
                let zeros =
                    matches!(&value.kind, ExprKind::Name(n) if n.name == "zeros" && !n.quoted);
                let place = if zeros {
                    Place::VarOfZeros
                } else {
                    Place::Local
                };
                // a count the text writes: a number 64 bits hold
                if let Some(c) = ty.count.as_deref() {
                    if matches!(&c.kind, ExprKind::Int(t) if parse_int(t).is_none()) {
                        self.error(
                            codes::WRONG_TYPE,
                            "a count is a constant expression or a module parameter".into(),
                            c.span,
                        );
                    }
                }
                let want = self.ty(cx.env, ty, place);
                if let (true, Ty::Block(Some(_), _)) = (zeros, &want) {
                    self.note_type(value, &want);
                } else {
                    self.fit(value, &got, &want, &format!("`{}`", name.name));
                }
                let binding = Name::Local {
                    decl: name.span,
                    kind: LocalKind::Var,
                };
                cx.bind(&name.name, local(want, binding, true));
            }
            Stmt::Assign { target, op, value } => self.assign(cx, target, *op, value),
            Stmt::For {
                vars, iter, body, ..
            } => self.for_loop(cx, vars, iter, body),
            Stmt::While { cond, body } => {
                if cx.kind != Kind::Control {
                    self.error(
                        codes::WHILE_OUTSIDE_CONTROL,
                        "`while` is allowed in a control body only".into(),
                        span,
                    );
                    return;
                }
                let ty = self.expr(cx, cond);
                self.fit(cond, &ty, &Ty::Bool, "the condition");
                self.block(cx, body);
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let ty = self.expr(cx, cond);
                self.fit(cond, &ty, &Ty::Bool, "the condition");
                self.block(cx, then);
                if let Some(otherwise) = otherwise {
                    self.block(cx, otherwise);
                }
            }
            Stmt::Store { target, value } => self.store(cx, target, value, span),
            Stmt::Consume(name) => {
                if !matches!(cx.kind, Kind::Seed | Kind::On) {
                    self.misplaced(cx, Family::Dynamic, "consume", span);
                    return;
                }
                match cx.lookup(&name.name) {
                    Some(l) if l.binding == Name::Arrived => {
                        self.note_name(name.span, Name::Arrived);
                    }
                    _ => self.error(
                        codes::UNKNOWN_NAME,
                        format!("`{}` is not the message this body handles", name.name),
                        name.span,
                    ),
                }
            }
            Stmt::Return(value) => {
                let ty = match value {
                    Some(value) => self.expr(cx, value),
                    None => Ty::Void,
                };
                match (&cx.ret, value) {
                    (None, Some(value)) => self.error(
                        codes::WRONG_TYPE,
                        "a result is returned from a body that declares none".into(),
                        value.span,
                    ),
                    (Some(want), Some(value)) => {
                        let want = want.clone();
                        self.fit(value, &ty, &want, "the result");
                    }
                    _ => {}
                }
            }
            Stmt::Expr(expr) => {
                let ty = self.expr(cx, expr);
                self.not_a_value(cx, expr, &ty, Read::Alone);
            }
        }
    }

    /// The value of an assignment to a place of type `want`: `=` asks a value of that
    /// type; `+=` and its kin are an operation on the two, whose scalars have one type.
    fn assigned(&mut self, value: &Expr, got: &Ty, want: &Ty, op: AssignOp) {
        if op != AssignOp::Set {
            if let (Ty::Scalar(a), Ty::Scalar(b)) = (want, got) {
                if a != b {
                    self.mixed(op.text(), *a, *b, value.span);
                    return;
                }
            }
        }
        self.fit(value, got, want, "the value");
    }

    /// `K119`: two scalars of different types in one operation.
    pub(super) fn mixed(&mut self, what: &str, a: ScalarType, b: ScalarType, span: Span) {
        self.error(
            codes::MIXED_SCALARS,
            format!(
                "`{what}` takes {a} and {b}; the two operands have one type, and a value is \
                 converted by naming the type: `{a}(x)`"
            ),
            span,
        );
    }

    pub(super) fn assign(&mut self, cx: &mut Cx, target: &Expr, op: AssignOp, value: &Expr) {
        let got = self.expr(cx, value);
        let (root, element) = match &target.kind {
            ExprKind::Name(n) => (n, None),
            ExprKind::Index { base, index } => match as_name(base) {
                Some(n) => (n, Some(index)),
                None => {
                    self.error(
                        codes::ASSIGN_LET,
                        "only a `var` or an element of state is assigned".into(),
                        target.span,
                    );
                    return;
                }
            },
            _ => {
                self.error(
                    codes::ASSIGN_LET,
                    "only a `var` or an element of state is assigned".into(),
                    target.span,
                );
                return;
            }
        };
        if let Some(l) = cx.lookup(&root.name).cloned() {
            self.note_name(root.span, l.binding.clone());
            if !l.mutable || element.is_some() {
                self.error(
                    codes::ASSIGN_LET,
                    format!("`{}` is not a `var` and is not assigned again", root.name),
                    target.span,
                );
                return;
            }
            let want = l.ty.clone();
            self.assigned(value, &got, &want, op);
            // a block `var` (R48): `+=` of a block of its own shape
            if let (AssignOp::Add, Ty::Block(_, shape)) = (op, &want) {
                if !matches!(&got, Ty::Block(_, s) if s == shape) && got != Ty::Unknown {
                    self.error(
                        codes::WRONG_TYPE,
                        format!(
                            "`{}` is {want}; `+=` adds a block of its own shape, and this is \
                             {got}",
                            root.name
                        ),
                        value.span,
                    );
                }
                return;
            }
            if op != AssignOp::Set
                && !(want.is_number() || matches!(want, Ty::Tile(_) | Ty::Unknown))
            {
                self.error(
                    codes::WRONG_TYPE,
                    format!(
                        "`{}` is {want}; `{}` adds numbers or tiles",
                        root.name,
                        op.text()
                    ),
                    target.span,
                );
            }
            return;
        }
        if let Some(member) = cx.state(&root.name).cloned() {
            if let Some(state) = cx.member_name(&root.name) {
                self.note_name(root.span, state);
            }
            let Some(index) = element else {
                self.error(
                    codes::ASSIGN_LET,
                    format!(
                        "the member `{}` is filled with `.fill` or `.zero()`; one of its \
                         elements is assigned",
                        root.name
                    ),
                    target.span,
                );
                return;
            };
            if let (Some(Name::State { member: m, .. }), [IndexItem::At(i)]) = (
                cx.member_name(&root.name),
                index.values().collect::<Vec<_>>().as_slice(),
            ) {
                self.note_op(target.span, OpKind::StateSet(MemberId(m)), &[Some(i)]);
            }
            for item in index.values() {
                if let IndexItem::At(e) = item {
                    let ty = self.expr(cx, e);
                    let ty = self.alone(e, ty);
                    self.fit(e, &ty, &Ty::AnyInt, "an index");
                }
            }
            let end = match &member.ty {
                Ty::Block(
                    _,
                    Shape {
                        cols: Count::Known(n),
                        ..
                    },
                )
                | Ty::Scalars(_, Count::Known(n)) => Some(*n),
                _ => None,
            };
            let at = index.values().next().and_then(|item| match item {
                IndexItem::At(e) => as_int(e),
                IndexItem::Slice { .. } => None,
            });
            if let (Some(end), Some(i)) = (end, at) {
                if i >= end {
                    self.error(
                        codes::OUT_OF_RANGE,
                        format!(
                            "the index {i} is past the end: the member `{}` holds {end}",
                            root.name
                        ),
                        target.span,
                    );
                    return;
                }
            }
            let want = match &member.ty {
                Ty::Block(f, _) => Ty::Tile(*f),
                Ty::Scalars(elem, _) => (**elem).clone(),
                _ => {
                    self.error(
                        codes::WRONG_TYPE,
                        format!("the member `{}` has no elements", root.name),
                        target.span,
                    );
                    return;
                }
            };
            self.assigned(value, &got, &want, op);
            return;
        }
        if cx.env.params.contains_key(&root.name) {
            self.error(
                codes::ASSIGN_LET,
                format!("`{}` is a module parameter and is not assigned", root.name),
                target.span,
            );
            return;
        }
        self.error(
            codes::UNKNOWN_NAME,
            format!("`{}` is not declared", root.name),
            root.span,
        );
    }

    pub(super) fn for_loop(&mut self, cx: &mut Cx, vars: &[Ident], iter: &Expr, body: &Block) {
        let ty = self.expr(cx, iter);
        let elems: Vec<Ty> = match &ty {
            Ty::Stream(elem, _) => vec![(**elem).clone()],
            Ty::Zip(streams) => streams
                .iter()
                .map(|s| match s {
                    Ty::Stream(elem, _) => (**elem).clone(),
                    _ => Ty::Unknown,
                })
                .collect(),
            Ty::Arrive(at) => {
                let format = self.casts.get(at).map(|(f, _)| *f);
                if format.is_none() {
                    self.late.insert(*at);
                }
                vec![Ty::Tile(format), Ty::Scalar(ScalarType::U32)]
            }
            Ty::Range(s) => vec![Ty::Scalar(*s)],
            Ty::Unknown => vars.iter().map(|_| Ty::Unknown).collect(),
            other => {
                self.error(
                    codes::WRONG_TYPE,
                    format!("a `for` walks a stream, a zip, a range or arrivals, not {other}"),
                    iter.span,
                );
                vars.iter().map(|_| Ty::Unknown).collect()
            }
        };
        if elems.len() != vars.len() {
            self.error(
                codes::WRONG_TYPE,
                format!(
                    "this `for` gives {} values a step and names {}",
                    elems.len(),
                    vars.len()
                ),
                iter.span,
            );
        }
        cx.scopes.push(HashMap::new());
        let mut names = Names::default();
        for (i, var) in vars.iter().enumerate() {
            names.declare(self, var);
            self.bare_word(var, Declared::Value);
            let ty = elems.get(i).cloned().unwrap_or(Ty::Unknown);
            let binding = Name::Local {
                decl: var.span,
                kind: LocalKind::Loop,
            };
            cx.bind(&var.name, local(ty, binding, false));
        }
        self.block(cx, body);
        cx.scopes.pop();
    }

    pub(super) fn store(&mut self, cx: &mut Cx, target: &Expr, value: &Expr, span: Span) {
        if cx.kind == Kind::Func {
            self.error(
                codes::FUNC_NOT_COMPUTE,
                "a function body uses the compute family only; `store` is not one".into(),
                span,
            );
            return;
        }
        if !cx.kind.on_workers() && cx.kind != Kind::Fallback {
            self.misplaced(cx, Family::Worker, "store", span);
            return;
        }
        // K205 (a fallback touching what `@reads` and `@writes` do not state) is Verify's
        let want = self.expr(cx, target);
        let got = self.expr(cx, value);
        if !matches!(want, Ty::View { .. } | Ty::Unknown) {
            self.error(
                codes::WRONG_TYPE,
                format!("a store writes a view, and this is {want}"),
                target.span,
            );
            return;
        }
        match &got {
            Ty::Scalar(_) | Ty::Number(_) | Ty::Scalars(..) | Ty::Words(_) => {
                self.store_scalars(cx, &want, value, &got, span)
            }
            Ty::Tile(_) => {
                if let Ty::View {
                    format: Some(f), ..
                } = want
                {
                    self.fit(value, &got, &Ty::Tile(Some(f)), "the tile stored");
                }
            }
            Ty::Unknown => {}
            // a block of one row (R48): its tiles in the view's order, as many as it holds
            Ty::Block(f, shape) if shape.rows == Count::Known(1) => {
                // both counts known at compile: the view's tiles are the block's
                if let (
                    Ty::View {
                        tiles: Count::Known(t),
                        ..
                    },
                    Count::Known(c),
                ) = (&want, shape.cols)
                {
                    if *t != c {
                        self.error(
                            codes::WRONG_TYPE,
                            format!(
                                "a store of a block of {c} tiles into a view of {t}: the view \
                                 holds as many tiles as the block"
                            ),
                            span,
                        );
                        return;
                    }
                }
                if let (
                    Ty::View {
                        format: Some(want), ..
                    },
                    Some(f),
                ) = (want, f)
                {
                    let tile = Ty::Tile(Some(*f));
                    self.expect_fit(&tile, &Ty::Tile(Some(want)), value.span, "the block stored");
                }
            }
            Ty::Reduced(..) => self.error(
                codes::REDUCED_VALUE,
                format!(
                    "a store writes a tile, and this is {got}; `.col()` or `.row()` spreads it \
                     into one"
                ),
                value.span,
            ),
            other => self.error(
                codes::WRONG_TYPE,
                format!("a store writes a tile, and this is {other}"),
                value.span,
            ),
        }
    }

    /// A store of scalars, the mirror of `.words()`: one scalar, a state member of
    /// scalars named whole, or the words of a tile, written into the one tile the view
    /// names (`K102` for a view the text shows to hold more), of a format with stated
    /// bytes (`K208`), of the format's scalar type (`K102`), no more than the tile
    /// holds where the text shows the count (`K115`).
    fn store_scalars(&mut self, cx: &Cx, want: &Ty, value: &Expr, got: &Ty, span: Span) {
        let Ty::View { format, tiles } = want else {
            return;
        };
        if tiles.more_than_one() {
            self.error(
                codes::WRONG_TYPE,
                "a store of scalars writes one tile, and this view holds more".into(),
                span,
            );
            return;
        }
        if !self.stated_bytes(*format, "store", span) {
            return;
        }
        let Some(s) = format.and_then(Format::as_scalar) else {
            return;
        };
        let member = as_name(value)
            .filter(|n| cx.lookup(&n.name).is_none())
            .and_then(|n| cx.member_name(&n.name))
            .is_some();
        let (elem, count) = match got {
            Ty::Scalar(_) | Ty::Number(_) => {
                self.fit(value, got, &Ty::Scalar(s), "the scalar stored");
                return;
            }
            Ty::Scalars(elem, count) if member => ((**elem).clone(), *count),
            Ty::Words(w) => {
                // `v.words()` of a view: the module tile's rows times columns; `b.words()`
                // of a message's body: its body's words; words named by a local: not shown
                let base = match &value.kind {
                    ExprKind::Call { callee, .. } => match &callee.kind {
                        ExprKind::Member { base, .. } => {
                            self.types.get(&(base.span.start, base.span.end)).cloned()
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let count = match base {
                    Some(Ty::View { .. }) => {
                        cx.tile.map_or(Count::Unknown, |(r, c)| Count::Known(r * c))
                    }
                    Some(Ty::Body) => self
                        .arrived_body(cx)
                        .map_or(Count::Unknown, |b| Count::Known(b / 4)),
                    _ => Count::Unknown,
                };
                (w.map_or(Ty::Unknown, Ty::Scalar), count)
            }
            other => {
                self.error(
                    codes::WRONG_TYPE,
                    format!(
                        "a store of scalars writes a scalar, a state member of scalars or the \
                         words of a tile, and this is {other}"
                    ),
                    value.span,
                );
                return;
            }
        };
        if elem != Ty::Unknown && elem != Ty::Scalar(s) {
            self.error(
                codes::WRONG_TYPE,
                format!("the scalars stored are {elem}, and the tensor's format holds {s}"),
                value.span,
            );
            return;
        }
        if let (Count::Known(k), Some((rows, cols))) = (count, cx.tile) {
            if k > rows * cols {
                self.error(
                    codes::OUT_OF_RANGE,
                    format!(
                        "{k} scalars stored into one tile, which holds {}",
                        rows * cols
                    ),
                    value.span,
                );
            }
        }
    }

    // ---- Where an operation may be used ------------------------------------------

    /// Reports an operation used in a body that may not use it, with the code of the
    /// rule it breaks.
    pub(super) fn misplaced(&mut self, cx: &Cx, family: Family, what: &str, span: Span) {
        let (code, text) = match (cx.kind, family) {
            (Kind::Func | Kind::Fallback, _) => (
                codes::FUNC_NOT_COMPUTE,
                format!("a function body uses the compute family only; `{what}` is not one"),
            ),
            (_, Family::Islands) => (
                codes::ISLANDS_IN_HANDLER,
                format!("`{what}` is called from a control program only"),
            ),
            (Kind::Control, _) => (
                codes::CONTROL_NOT_VALUES,
                format!(
                    "a control body uses values, control flow, `invoke`, `await` and results; \
                     `{what}` is not one"
                ),
            ),
            (_, Family::Control) => (
                codes::INVOKE_OUTSIDE_CONTROL,
                format!("`{what}` is used in a control program only"),
            ),
            (_, Family::Dynamic) => (
                codes::SEND_OUTSIDE_RUN,
                format!("`{what}` is used in a dynamic run only"),
            ),
            _ => (
                codes::WRONG_TYPE,
                format!("`{what}` is not used in a chain's call"),
            ),
        };
        self.error(code, text, span);
    }

    /// True when a body of this kind may use an operation of `family`; otherwise reports.
    pub(super) fn allowed(&mut self, cx: &Cx, family: Family, what: &str, span: Span) -> bool {
        let ok = match cx.kind {
            Kind::Func | Kind::Fallback => matches!(family, Family::Value | Family::Compute),
            Kind::Handler => matches!(
                family,
                Family::Value | Family::Compute | Family::Worker | Family::IslandQuery
            ),
            Kind::Seed | Kind::On => matches!(
                family,
                Family::Value
                    | Family::Compute
                    | Family::Worker
                    | Family::IslandQuery
                    | Family::Dynamic
            ),
            Kind::Control => matches!(
                family,
                Family::Value | Family::Control | Family::Islands | Family::IslandQuery
            ),
            Kind::Chain => matches!(family, Family::Value),
        };
        if !ok {
            self.misplaced(cx, family, what, span);
        }
        ok
    }

    // ---- Expressions --------------------------------------------------------------

    /// The type of an expression, with the diagnostics of what it uses.
    pub(super) fn expr(&mut self, cx: &mut Cx, e: &Expr) -> Ty {
        let ty = self.expr_inner(cx, e);
        self.note_type(e, &ty);
        ty
    }

    pub(super) fn expr_inner(&mut self, cx: &mut Cx, e: &Expr) -> Ty {
        match &e.kind {
            ExprKind::Int(_) => Ty::Number(Lit::Whole),
            ExprKind::Float(_) => Ty::Number(Lit::Point),
            ExprKind::Str(_) => Ty::Str,
            ExprKind::Bool(_) => Ty::Bool,
            ExprKind::Name(n) => self.name(cx, n),
            ExprKind::Paren(inner) => self.expr(cx, inner),
            ExprKind::Tuple(items) | ExprKind::List(items) => {
                for item in items.values() {
                    self.expr(cx, item);
                }
                Ty::List
            }
            ExprKind::Init { ty, fields } => match ty {
                Some(name) => self.make_message(cx, name, fields, e.span),
                None => {
                    self.error(
                        codes::WRONG_TYPE,
                        "fields without a type are the `with` of a `forward` only".into(),
                        e.span,
                    );
                    Ty::Unknown
                }
            },
            ExprKind::Member { base, name } => self.member(cx, base, name, e.span),
            ExprKind::Call { callee, args } => self.call(cx, callee, args, e.span),
            ExprKind::Index { base, index } => self.index(cx, base, index, e.span),
            ExprKind::Neg(inner) => {
                if number_value(e).is_none() {
                    self.note_op(e.span, OpKind::Unary(UnaryOp::Neg), &[Some(inner)]);
                }
                let ty = self.expr(cx, inner);
                if !(ty.is_tile_operand() || matches!(ty, Ty::Unknown | Ty::Reduced(..))) {
                    self.error(
                        codes::WRONG_TYPE,
                        format!("`-` takes a number or a tile, not {ty}"),
                        e.span,
                    );
                    return Ty::Unknown;
                }
                ty
            }
            ExprKind::Binary { op, lhs, rhs } => self.binary(cx, *op, lhs, rhs, e.span),
            ExprKind::Await(h) => {
                if !self.allowed(cx, Family::Control, "await", e.span) {
                    return Ty::Unknown;
                }
                self.note_op(e.span, OpKind::Await, &[]);
                match cx.lookup(&h.name).cloned() {
                    Some(l) => {
                        self.note_name(h.span, l.binding);
                        match l.ty {
                            Ty::Handle(ret) => ret.map(|r| *r).unwrap_or(Ty::Void),
                            Ty::Unknown => Ty::Unknown,
                            other => {
                                self.error(
                                    codes::WRONG_TYPE,
                                    format!("`await` takes an invocation's handle, not {other}"),
                                    h.span,
                                );
                                Ty::Unknown
                            }
                        }
                    }
                    None => {
                        self.unknown(h);
                        Ty::Unknown
                    }
                }
            }
            ExprKind::Past(y) => {
                if cx.kind != Kind::Chain {
                    self.error(
                        codes::PAST_OUTSIDE_CHAIN,
                        "`past` is used in a chain's call only".into(),
                        e.span,
                    );
                    return Ty::Unknown;
                }
                let ty = self.name(cx, y);
                if !matches!(ty, Ty::Tensor { .. } | Ty::Unknown) {
                    self.error(
                        codes::WRONG_TYPE,
                        format!("`past` takes a tensor parameter of the chain, not {ty}"),
                        y.span,
                    );
                    return Ty::Unknown;
                }
                ty
            }
        }
    }

    pub(super) fn unknown(&mut self, name: &Ident) {
        self.error(
            codes::UNKNOWN_NAME,
            format!("`{}` is not declared", name.name),
            name.span,
        );
    }

    /// A name read as a value.
    pub(super) fn name(&mut self, cx: &mut Cx, n: &Ident) -> Ty {
        // a bare `zeros` is the language's, whatever the module names (R50)
        if n.name == "zeros" && !n.quoted {
            if !self.allowed(cx, Family::Compute, "zeros", n.span) {
                return Ty::Unknown;
            }
            self.note_name(n.span, Name::Word(Word::Zeros));
            self.note_op(n.span, OpKind::Zeros, &[]);
            // a tile of the format its place gives
            return Ty::Tile(None);
        }
        if let Some(l) = cx.lookup(&n.name).cloned() {
            self.note_name(n.span, l.binding);
            return l.ty;
        }
        if let Some(member) = cx.state(&n.name).cloned() {
            if let Some(state) = cx.member_name(&n.name) {
                // a member of one element read by its name: its element 0
                if let Name::State { member: m, .. } = state {
                    if !matches!(member.ty, Ty::Block(..) | Ty::Scalars(..)) {
                        self.note_op(n.span, OpKind::StateGet(MemberId(m)), &[]);
                    }
                }
                self.note_name(n.span, state);
            }
            return member.ty;
        }
        if let Some(ty) = cx.env.params.get(&n.name) {
            if let Some(&param) = cx.env.ix.params.get(&n.name) {
                self.note_name(n.span, Name::ModuleParam(param));
            }
            return ty.clone();
        }
        self.unknown(n);
        Ty::Unknown
    }

    pub(super) fn binary(
        &mut self,
        cx: &mut Cx,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        span: Span,
    ) -> Ty {
        self.note_op(span, binary_kind(op), &[Some(lhs), Some(rhs)]);
        let l = self.expr(cx, lhs);
        if matches!(op, BinOp::Eq | BinOp::Ne) && matches!(l, Ty::Reason) {
            return self.failure_name(rhs);
        }
        let r = self.expr(cx, rhs);
        let bad = |this: &mut Self| {
            this.error(
                codes::WRONG_TYPE,
                format!("`{}` does not take {l} and {r}", op.text()),
                span,
            );
            Ty::Unknown
        };
        let unknown = matches!(l, Ty::Unknown) || matches!(r, Ty::Unknown);
        if matches!(op, BinOp::Div | BinOp::Rem) && as_int(rhs) == Some(0) {
            let integer = l.is_integer() || matches!(l, Ty::Tile(Some(f)) if f.is_integer());
            if integer {
                self.error(
                    codes::DIVIDE_BY_ZERO,
                    format!("an integer `{}` by zero", op.text()),
                    span,
                );
                return Ty::Unknown;
            }
        }
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            if let ExprKind::Neg(amount) = &rhs.kind {
                if as_int(amount).is_some_and(|n| n > 0) {
                    self.error(
                        codes::OUT_OF_RANGE,
                        "a shift by a negative amount".into(),
                        span,
                    );
                    return Ty::Unknown;
                }
            }
        }
        let scalars = l.is_number() && r.is_number();
        match op {
            BinOp::And | BinOp::Or => {
                if l.fits(&Ty::Bool) && r.fits(&Ty::Bool) {
                    Ty::Bool
                } else {
                    bad(self)
                }
            }
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                if matches!(op, BinOp::Eq | BinOp::Ne)
                    && matches!(l, Ty::Bool)
                    && matches!(r, Ty::Bool)
                {
                    return Ty::Bool;
                }
                if unknown && (l == Ty::Bool || r == Ty::Bool) {
                    return Ty::Bool;
                }
                if scalars {
                    let one = self.one_type(op.text(), lhs, &l, rhs, &r, span);
                    // two numbers written in the text, compared: each alone
                    self.alone(lhs, l.clone());
                    self.alone(rhs, r.clone());
                    return if one == Ty::Unknown {
                        Ty::Unknown
                    } else {
                        Ty::Bool
                    };
                }
                match self.elementwise(op.text(), &[l.clone(), r.clone()], span) {
                    Some(Ty::Unknown) => Ty::Unknown,
                    Some(ty) => {
                        self.beside(&[Some(lhs), Some(rhs)], &ty);
                        Ty::Mask
                    }
                    None => bad(self),
                }
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                if scalars {
                    return self.one_type(op.text(), lhs, &l, rhs, &r, span);
                }
                match self.elementwise(op.text(), &[l.clone(), r.clone()], span) {
                    Some(ty) => {
                        self.beside(&[Some(lhs), Some(rhs)], &ty);
                        ty
                    }
                    None => bad(self),
                }
            }
            BinOp::Shl | BinOp::Shr => {
                // the amount may be of any integer type
                if !(l.fits(&Ty::AnyInt) && r.fits(&Ty::AnyInt)) {
                    return bad(self);
                }
                if matches!(r, Ty::Number(_)) && !matches!(l, Ty::Number(_)) {
                    self.alone(rhs, r.clone());
                }
                if unknown {
                    Ty::Unknown
                } else {
                    l
                }
            }
            _ => {
                let tile = matches!(l, Ty::Tile(Some(f)) if f.is_integer());
                if op == BinOp::Rem && tile && r.fits(&Ty::AnyInt) {
                    self.beside(&[Some(rhs)], &l);
                    return l;
                }
                if l.fits(&Ty::AnyInt) && r.fits(&Ty::AnyInt) {
                    if unknown {
                        Ty::Unknown
                    } else {
                        self.one_type(op.text(), lhs, &l, rhs, &r, span)
                    }
                } else {
                    bad(self)
                }
            }
        }
    }

    /// The kind of an element-by-element operation's result from its operands' types:
    /// a tile, a reduction's values, or `Ty::AnyNumber` for scalars only; `Some(Ty::Unknown)` after a
    /// diagnostic or an operand of unknown type; `None` when an operand is none of these.
    /// A reduction's values mixed with a tile are `K116`; an integer tile with a float
    /// scalar or a float tile is `K117`. In a function on blocks (R45) a block is an
    /// operand too: the tiles, blocks and reductions' values have one shape (`K102`).
    pub(super) fn elementwise(&mut self, what: &str, operands: &[Ty], span: Span) -> Option<Ty> {
        let mut tile: Option<Option<crate::types::Format>> = None;
        let mut axis = None;
        let mut reduced: Option<Format> = None;
        let mut float = false;
        let mut float_tile = false;
        let mut integer_tile = false;
        let mut unknown = false;
        let mut shapes: Vec<Shape> = Vec::new();
        let mut block = false;
        for ty in operands {
            let ty = match ty {
                Ty::Block(f, shape) if self.on_blocks => {
                    shapes.push(*shape);
                    block = true;
                    &Ty::Tile(*f)
                }
                Ty::Tile(_) => {
                    shapes.push(Shape::TILE);
                    ty
                }
                Ty::Reduced(_, _, shape) => {
                    shapes.push(*shape);
                    ty
                }
                _ => ty,
            };
            match ty {
                Ty::Unknown => unknown = true,
                Ty::Scalar(_) | Ty::Number(_) => float |= ty.is_float(),
                Ty::Tile(f) => {
                    match f {
                        Some(f) if f.is_integer() => integer_tile = true,
                        Some(_) => float_tile = true,
                        None => {}
                    }
                    if tile.flatten().is_none() {
                        tile = Some(*f);
                    }
                }
                Ty::Reduced(f, a, _) => {
                    reduced = reduced.or(*f);
                    if axis.is_some_and(|b| b != *a) {
                        self.error(
                            codes::REDUCED_VALUE,
                            format!(
                                "`{what}` mixes values for each row with values for each column"
                            ),
                            span,
                        );
                        return Some(Ty::Unknown);
                    }
                    axis = Some(*a);
                }
                _ => return None,
            }
        }
        if tile.is_some() && axis.is_some() {
            self.error(
                codes::REDUCED_VALUE,
                format!(
                    "`{what}` mixes a reduction's values with a tile; `.col()` or `.row()` \
                     spreads them into a tile first"
                ),
                span,
            );
            return Some(Ty::Unknown);
        }
        if integer_tile && (float || float_tile) {
            self.error(
                codes::INTEGER_WITH_FLOAT,
                format!(
                    "`{what}` takes an integer tile with a float; there is no conversion \
                     between integer and float tiles"
                ),
                span,
            );
            return Some(Ty::Unknown);
        }
        if unknown {
            return Some(Ty::Unknown);
        }
        let shape = shapes.first().copied().unwrap_or(Shape::TILE);
        if let Some(other) = shapes.iter().find(|s| !s.agrees(shape)) {
            self.error(
                codes::WRONG_TYPE,
                format!(
                    "`{what}` takes operands of one shape, element by element: {shape} and \
                     {other}"
                ),
                span,
            );
            return Some(Ty::Unknown);
        }
        if let Some(f) = tile {
            return Some(if block {
                Ty::Block(f, shape)
            } else {
                Ty::Tile(f)
            });
        }
        if let Some(a) = axis {
            return Some(Ty::Reduced(reduced, a, shape));
        }
        Some(Ty::AnyNumber)
    }

    /// Gives the numbers written in the text among `operands` the type of the tile
    /// they are computed with (`result`): the tile's format when it is a scalar type,
    /// `f32` beside a `bf16` or a block-float tile.
    pub(super) fn beside(&mut self, operands: &[Option<&Expr>], result: &Ty) {
        let s = match result {
            Ty::Tile(Some(f)) | Ty::Block(Some(f), _) | Ty::Reduced(Some(f), ..) => {
                f.as_scalar().unwrap_or(ScalarType::F32)
            }
            _ => return,
        };
        for e in operands.iter().flatten() {
            self.settle(e, s);
        }
    }

    /// The name after `o.reason ==`: one of the failures.
    pub(super) fn failure_name(&mut self, rhs: &Expr) -> Ty {
        match as_name(rhs) {
            Some(n) if !n.quoted && FAILURES.contains(&n.name.as_str()) => {
                if let Some(failure) = Failure::from_name(&n.name) {
                    self.note_name(n.span, Name::Failure(failure));
                }
                self.note_type(rhs, &Ty::Reason);
                Ty::Bool
            }
            Some(n) => {
                self.error(
                    codes::UNKNOWN_REASON,
                    format!(
                        "`{}` is not the name of a failure: one of {}",
                        n.name,
                        FAILURES.join(", ")
                    ),
                    n.span,
                );
                Ty::Unknown
            }
            None => {
                self.error(
                    codes::WRONG_TYPE,
                    "`o.reason` is compared with the name of a failure".into(),
                    rhs.span,
                );
                Ty::Unknown
            }
        }
    }

    pub(super) fn member(&mut self, cx: &mut Cx, base: &Expr, name: &Ident, span: Span) -> Ty {
        if let Some(word) = as_name(base) {
            // a bare `worker`, `group` or `island` is the language's (R50), and none
            // has a member of the module's: one between backticks is no member of it
            if !word.quoted {
                let language = matches!(word.name.as_str(), "worker" | "group" | "island");
                if language && name.quoted {
                    self.unknown(name);
                    return Ty::Unknown;
                }
                match word.name.as_str() {
                    "worker" => return self.worker(cx, word, name, span),
                    "group" => {
                        if !self.allowed(cx, Family::Worker, "group.size", span) {
                            return Ty::Unknown;
                        }
                        self.note_word(word);
                        if name.name == "size" {
                            self.note_op(span, OpKind::GroupSize, &[]);
                            return Ty::Scalar(ScalarType::U32);
                        }
                        return self.no_member("group", name);
                    }
                    "island" => {
                        if !self.allowed(cx, Family::IslandQuery, "island.count", span) {
                            return Ty::Unknown;
                        }
                        self.note_word(word);
                        if name.name == "count" {
                            self.note_op(span, OpKind::IslandCount, &[]);
                            return Ty::Scalar(ScalarType::U32);
                        }
                        return self.no_member("island", name);
                    }
                    _ => {}
                }
            }
        }
        let ty = self.expr(cx, base);
        // a member between backticks is the module's only: a field of a message or a
        // record (R50)
        if name.quoted && !matches!(ty, Ty::Message(_) | Ty::Record(_) | Ty::Unknown) {
            self.unknown(name);
            return Ty::Unknown;
        }
        let kind = match (&ty, name.name.as_str()) {
            (Ty::Tile(_), "T") => Some(OpKind::Transpose),
            (Ty::Block(..), "T") if self.on_blocks => Some(OpKind::Transpose),
            (Ty::View { .. }, "T") => Some(OpKind::Turn),
            (Ty::Outcome(_), "failed") => Some(OpKind::Failed),
            (Ty::Outcome(_), "reason") => Some(OpKind::Reason),
            (Ty::Outcome(Some(_)), "value") => Some(OpKind::OutcomeValue),
            (Ty::Broadcast(_), "is_sender") => Some(OpKind::IsSender),
            _ => None,
        };
        if let Some(kind) = kind {
            self.note_op(span, kind, &[Some(base)]);
        }
        match (&ty, name.name.as_str()) {
            (Ty::Unknown, _) => Ty::Unknown,
            (Ty::Tile(f), "T") => Ty::Tile(*f),
            (Ty::Block(f, s), "T") if self.on_blocks => Ty::Block(
                *f,
                Shape {
                    rows: s.cols,
                    cols: s.rows,
                },
            ),
            (Ty::View { format, .. }, "T") => Ty::View {
                format: *format,
                tiles: Count::Unknown,
            },
            (Ty::Tensor { format, .. }, "tiles") => {
                if !self.allowed(cx, Family::Worker, "t.tiles", span) {
                    return Ty::Unknown;
                }
                Ty::TileGrid(*format)
            }
            // a bare `body` after a message is the language's (R50): the body of the
            // message this body handles
            (Ty::Message(m), "body") if !name.quoted => {
                let arrived = as_name(base)
                    .and_then(|n| cx.lookup(&n.name))
                    .is_some_and(|l| l.binding == Name::Arrived);
                if !arrived {
                    return self.no_member(&format!("the message {m}"), name);
                }
                let decl = cx.env.message(cx.program.as_deref(), m);
                if decl.is_some_and(|d| d.body.is_none()) {
                    self.error(
                        codes::WRONG_TYPE,
                        format!("the message {m} has no body"),
                        name.span,
                    );
                    return Ty::Unknown;
                }
                self.note_op(span, OpKind::Body, &[Some(base)]);
                Ty::Body
            }
            (Ty::Message(m), field) => {
                let fields = cx.env.message(cx.program.as_deref(), m).map(|d| &d.fields);
                match fields.and_then(|f| f.iter().enumerate().find(|(_, (n, _))| n == field)) {
                    Some((at, (_, ty))) => {
                        self.note_name(name.span, Name::Field(at as u32));
                        self.note_op(span, OpKind::Field(at as u32), &[Some(base)]);
                        ty.clone()
                    }
                    None => self.no_member(&format!("the message {m}"), name),
                }
            }
            (Ty::Record(r), field) => {
                let fields = cx.env.record(cx.program.as_deref(), r);
                match fields.and_then(|f| f.iter().enumerate().find(|(_, (n, _))| n == field)) {
                    Some((at, (_, ty))) => {
                        self.note_name(name.span, Name::Field(at as u32));
                        self.note_op(span, OpKind::RecordField(at as u32), &[Some(base)]);
                        ty.clone()
                    }
                    None => self.no_member(&format!("the record {r}"), name),
                }
            }
            (Ty::Outcome(_), "failed") => Ty::Bool,
            (Ty::Outcome(_), "reason") => Ty::Reason,
            (Ty::Outcome(Some(value)), "value") => (**value).clone(),
            (Ty::Outcome(None), "value") => {
                self.error(
                    codes::WRONG_TYPE,
                    "the invocation gives no result, so its outcome has no `value`".into(),
                    name.span,
                );
                Ty::Unknown
            }
            (Ty::Broadcast(_), "is_sender") => Ty::Bool,
            _ => self.no_member(&ty.to_string(), name),
        }
    }

    pub(super) fn no_member(&mut self, what: &str, name: &Ident) -> Ty {
        self.error(
            codes::WRONG_TYPE,
            format!("{what} has no `{}`", name.name),
            name.span,
        );
        Ty::Unknown
    }

    pub(super) fn worker(&mut self, cx: &mut Cx, word: &Ident, name: &Ident, span: Span) -> Ty {
        if !self.allowed(cx, Family::Worker, &format!("worker.{}", name.name), span) {
            return Ty::Unknown;
        }
        self.note_word(word);
        let kind = match name.name.as_str() {
            "rank" => Some(OpKind::Rank),
            "id" => Some(OpKind::WorkerId),
            "is_leader" => Some(OpKind::IsLeader),
            "row" => Some(OpKind::Row),
            "col" => Some(OpKind::Col),
            _ => None,
        };
        if let Some(kind) = kind {
            self.note_op(span, kind, &[]);
        }
        match name.name.as_str() {
            "rank" | "id" => Ty::Scalar(ScalarType::U32),
            "is_leader" => Ty::Bool,
            "row" | "col" => {
                if !cx.grid {
                    self.error(
                        codes::ROW_OUTSIDE_GRID,
                        format!("`worker.{}` exists in a handler on a grid only", name.name),
                        span,
                    );
                    return Ty::Unknown;
                }
                Ty::Scalar(ScalarType::U32)
            }
            _ => self.no_member("worker", name),
        }
    }

    pub(super) fn index(
        &mut self,
        cx: &mut Cx,
        base: &Expr,
        index: &List<IndexItem>,
        span: Span,
    ) -> Ty {
        let ty = self.expr(cx, base);
        let items: Vec<&IndexItem> = index.values().collect();
        if let Ty::TileGrid(format) = ty {
            for item in &items {
                match item {
                    IndexItem::Slice { lo, hi, step } => {
                        for e in [Some(lo), Some(hi), step.as_ref()].into_iter().flatten() {
                            let t = self.expr(cx, e);
                            self.expect_fit(&t, &Ty::AnyInt, e.span, "a bound of a rectangle");
                        }
                    }
                    IndexItem::At(e) => {
                        self.error(
                            codes::WRONG_TYPE,
                            "a rectangle of tiles is `t.tiles[r0:r1, c0:c1]`".into(),
                            e.span,
                        );
                        return Ty::Unknown;
                    }
                }
            }
            if items.len() != 2 {
                self.error(
                    codes::WRONG_TYPE,
                    "a rectangle of tiles gives two ranges".into(),
                    span,
                );
                return Ty::Unknown;
            }
            // its tiles, where the compile knows each side: `a:b` of two counts, or
            // `a:a + n`
            let side = |item: &IndexItem| match item {
                IndexItem::Slice { lo, hi, step: None } => {
                    match (count_of(cx, lo), count_of(cx, hi), &hi.kind) {
                        (Count::Known(a), Count::Known(b), _) if b >= a => Some(b - a),
                        (
                            _,
                            _,
                            ExprKind::Binary {
                                op: BinOp::Add,
                                lhs,
                                rhs,
                            },
                        ) if same_name(lhs, lo) => match count_of(cx, rhs) {
                            Count::Known(n) => Some(n),
                            _ => None,
                        },
                        _ => None,
                    }
                }
                _ => None,
            };
            let tiles = match (side(items[0]), side(items[1])) {
                (Some(r), Some(c)) => Count::Known(r * c),
                _ => Count::Unknown,
            };
            if let ExprKind::Member { base: tensor, .. } = &base.kind {
                let mut parts = vec![Some(&**tensor)];
                for item in &items {
                    if let IndexItem::Slice { lo, hi, step } = item {
                        parts.extend([Some(&**lo), Some(&**hi), step.as_deref()]);
                    }
                }
                self.note_op(span, OpKind::Rect, &parts);
            }
            return Ty::View { format, tiles };
        }
        if let Ty::Tensor { .. } = ty {
            self.error(
                codes::TENSOR_INDEXED,
                "a tensor is reached only through a view".into(),
                span,
            );
            return Ty::Unknown;
        }
        let [IndexItem::At(at)] = items.as_slice() else {
            if matches!(ty, Ty::Unknown) {
                return Ty::Unknown;
            }
            self.error(
                codes::WRONG_TYPE,
                format!("{ty} is indexed by one integer"),
                span,
            );
            return Ty::Unknown;
        };
        let state = match as_name(base).and_then(|n| self.names.get(&(n.span.start, n.span.end))) {
            Some(Name::State { member, .. }) => Some(MemberId(*member)),
            _ => None,
        };
        let kind = match (&ty, state) {
            (_, Some(m)) => Some(OpKind::StateGet(m)),
            (Ty::View { .. }, _) => Some(OpKind::ReadTile),
            (Ty::Block(..), _) => Some(OpKind::BlockAt),
            (Ty::Words(_), _) => Some(OpKind::Word),
            (Ty::Records(..), _) => Some(OpKind::RecordAt),
            _ => None,
        };
        match kind {
            Some(OpKind::StateGet(m)) => self.note_op(span, OpKind::StateGet(m), &[Some(at)]),
            Some(kind) => self.note_op(span, kind, &[Some(base), Some(at)]),
            None => {}
        }
        let at_ty = self.expr(cx, at);
        self.expect_fit(&at_ty, &Ty::AnyInt, at.span, "an index");
        if let Ty::Block(_, shape) = &ty {
            if shape.rows != Count::Known(1) {
                self.error(
                    codes::NOT_BUILT,
                    format!(
                        "a tile of a block {shape} by its index is named in the language and not \
                         built yet"
                    ),
                    span,
                );
                return Ty::Unknown;
            }
        }
        let end = match &ty {
            Ty::Block(_, Shape { cols: tiles, .. })
            | Ty::View { tiles, .. }
            | Ty::Scalars(_, tiles) => match tiles {
                Count::Known(n) => Some(*n),
                _ => None,
            },
            Ty::Records(_, Count::Known(n)) => Some(*n),
            Ty::Words(_) => cx.tile.map(|(rows, cols)| rows * cols),
            _ => None,
        };
        if let (Some(end), Some(i)) = (end, as_int(at)) {
            if i >= end {
                self.error(
                    codes::OUT_OF_RANGE,
                    format!("the index {i} is past the end: {ty} holds {end}"),
                    span,
                );
                return Ty::Unknown;
            }
        }
        match ty {
            Ty::Unknown => Ty::Unknown,
            Ty::View { format, .. } => Ty::Tile(format),
            Ty::Block(format, _) => Ty::Tile(format),
            Ty::Scalars(elem, _) => *elem,
            Ty::Words(Some(s)) => Ty::Scalar(s),
            Ty::Words(None) => Ty::Unknown,
            Ty::Records(r, _) => Ty::Record(r),
            other => {
                self.error(codes::WRONG_TYPE, format!("{other} is not indexed"), span);
                Ty::Unknown
            }
        }
    }
}

/// A block's count as a message gives it.
fn shape_text(s: &Shape) -> String {
    if s.rows != Count::Known(1) {
        return format!("of shape {s}");
    }
    match s.cols {
        Count::Known(n) => format!("of {n} tiles"),
        Count::GroupSize => "of `group.size` tiles".into(),
        Count::Param(_) => "of a module parameter's tiles".into(),
        Count::Unknown => "of some tiles".into(),
    }
}
