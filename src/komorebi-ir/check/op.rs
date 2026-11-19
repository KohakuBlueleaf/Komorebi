//! The operation of an expression: which operation of the language a call, a member,
//! an index, an operator, a message or an `await` is, with its operands by part.
//!
//! The kinds carry the names of the program structure's operations, and the words both
//! name (`types::op`), so that Lower maps one to the other by name. A kind holds what the text fixes (an operator, an axis, a
//! declaration's place); the operands are expressions of the text, by the parts of the
//! operation's form, in the order the operations page gives them.

use crate::syntax::ast::{Expr, ExprKind, IndexItem, Span};
use crate::types::{
    Axis, BinOp, CmpOp, Combiner, Entry, FuncId, MemberId, MessageId, RecordId, ReduceOp,
    ScalarType, SplitBy, UnaryOp,
};

/// Which operation an expression is. Each kind names the operation of the program
/// structure it lowers to; the parts each takes are given beside it.
#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    /// `a op b` on scalars or tiles; `min`, `max`, `minimum`, `maximum`. Parts: `a`, `b`.
    Binary(BinOp),
    /// `a cmp b`. Parts: `a`, `b`.
    Compare(CmpOp),
    /// `c && d`. Parts: `c`, `d`.
    LogicAnd,
    /// `c || d`. Parts: `c`, `d`.
    LogicOr,
    /// `neg(a)` and the other functions of one tile; the unary `-` of a value that is
    /// not a number written in the text. Parts: `a`.
    Unary(UnaryOp),
    /// `T(x)`: `x` converted to the scalar type named. Parts: `x`.
    Convert(ScalarType),
    /// `where(c, x, y)`; `c` is a comparison (its own operation). Parts: `c`, `x`, `y`.
    Select,
    /// `reduce_sum(a, axis = ...)` and its kin. Parts: `a`.
    Reduce(ReduceOp, Axis),
    /// `v.col()`. Parts: `v`.
    SpreadCol,
    /// `w.row()`. Parts: `w`.
    SpreadRow,
    /// `matmul(a, b)`. Parts: `a`, `b`.
    Matmul,
    /// `a.T` of a tile. Parts: `a`.
    Transpose,
    /// `zeros`. No parts: its type is its place's.
    Zeros,
    /// A call of a `func`, a `combine` or an `extern func`. Parts: the arguments.
    CallFunc(FuncId),
    /// `t.tiles(first, n)`. Parts: `t`, `first`, `n`.
    Tiles,
    /// `t.tiles[r0:r1(:s), c0:c1(:s)]`. Parts: `t`, `r0`, `r1`, the row step, `c0`,
    /// `c1`, the column step.
    Rect,
    /// `v.T` of a view. Parts: `v`.
    Turn,
    /// `t.items(size = B)` or `t.items(first, n, size = B)`. Parts: `t`, `first`, `n`,
    /// `size`.
    Items,
    /// `v[i]` of a view: one tile, read. Parts: `v`, `i`.
    ReadTile,
    /// `v.words()`. Parts: `v`.
    Words,
    /// `words[i]`. Parts: `words`, `i`.
    Word,
    /// `v.records(R, count = c)`. Parts: `v`, `c`.
    Records(RecordId),
    /// `records[i]`. Parts: `records`, `i`.
    RecordAt,
    /// `record.field`, by the field's position. Parts: `record`.
    RecordField(u32),
    /// `items.put(src, at = i)`; `src` is a state member. Parts: `items`, `src`, `i`.
    Put(MemberId),
    /// `partition(view, rank, size)`. Parts: `view`, `rank`, `size`.
    Partition,
    /// `stream(view, block = n)`. Parts: `view`, `n`.
    Stream,
    /// `sink(view, block = n)`. Parts: `view`, `n`.
    Sink,
    /// `out.push(t)`. Parts: `out`, `t`.
    Push,
    /// `block[i]` of a block of tiles. Parts: `block`, `i`.
    BlockAt,
    /// `s.fill(stream)`. Parts: `stream`.
    Fill(MemberId),
    /// `s.zero()`. No parts.
    ZeroState(MemberId),
    /// `s[i]`; a member of one element read by its name is `s[0]`, with no part.
    /// Parts: `i`, or none.
    StateGet(MemberId),
    /// `s[i] = x`, noted on the target `s[i]`; the value is the statement's. Parts: `i`.
    StateSet(MemberId),
    /// `s.bump(i)`. Parts: `i`.
    Bump(MemberId),
    /// `worker.rank`.
    Rank,
    /// `worker.is_leader`.
    IsLeader,
    /// `worker.id`.
    WorkerId,
    /// `worker.row`.
    Row,
    /// `worker.col`.
    Col,
    /// `group.size`.
    GroupSize,
    /// `group.id(rank)`. Parts: `rank`.
    GroupId,
    /// `group.split(parts = n)` or `(size = k)`. Parts: `n` or `k`.
    Split(SplitBy),
    /// `group.leaders(parts = n)` or `(size = k)`. Parts: `n` or `k`.
    Leaders(SplitBy),
    /// `reduce(x, with = f, within = g, among = g, publish = [v])`. Parts: `x`,
    /// `within`, `among`, `publish`.
    ReduceGroup(Combiner),
    /// `all_reduce(x, with = f, publish = [v])`. Parts: `x`, `publish`.
    AllReduce(Combiner),
    /// `gather(x)`. Parts: `x`.
    Gather,
    /// `broadcast(along = ..., sender = c)`. Parts: `sender`.
    Broadcast(Axis),
    /// `br.is_sender`. Parts: `br`.
    IsSender,
    /// `br.cast(block, word = w)`. Parts: `br`, `block`, `word`.
    Cast,
    /// `sync(publish = [v])`. Parts: `publish`.
    Sync,
    /// `M { field: value, ... }`. Parts: the values, in the type's order of fields.
    MakeMessage(MessageId),
    /// `send(m, to = rank, publish = [v])`. Parts: `m`, `to`, `publish`.
    Send,
    /// `forward(m, with = { ... }, to = rank, publish = [v])`; the fields of `with` are
    /// resolved by name. Parts: `m`, `with`, `to`, `publish`.
    Forward,
    /// `offer(m, to = rank)`. Parts: `m`, `to`.
    Offer,
    /// `m.field` of a message, by the field's position. Parts: `m`.
    Field(u32),
    /// `group(first, count)` or `group(count)`. Parts: `first`, `count`.
    MakeGroup,
    /// `invoke(P.h(args), on = g, after = [a], after_end = [b])`. Parts: `on`, `after`,
    /// `after_end`, then the arguments of `P.h`.
    Invoke(Entry),
    /// `await h`. No parts: `h` is a name, resolved.
    Await,
    /// `outcome(h)`. Parts: `h`.
    Outcome,
    /// `o.failed`. Parts: `o`.
    Failed,
    /// `o.reason`. Parts: `o`.
    Reason,
    /// `o.value`. Parts: `o`.
    OutcomeValue,
    /// `island.count`.
    IslandCount,
    /// `island.of(id)`. Parts: `id`.
    IslandOf,
    /// `island.crosses(id)`. Parts: `id`.
    IslandCrosses,
    /// `island.hops(id)`. Parts: `id`.
    IslandHops,
    /// `reduce_to(t, to = i, with = f)`. Parts: `t`, `to`.
    ReduceTo(Combiner),
    /// `broadcast_from(t, from = i)`. Parts: `t`, `from`.
    BroadcastFrom,
    /// `range(n)`, what a `for` walks. Parts: `n`.
    ForRange,
    /// `zip(s, t)`, what a `for` walks. Parts: `s`, `t`.
    ForZip,
    /// `br.arrive()`, what a `for` walks. Parts: `br`.
    ForArrive,
}

/// The operation of an expression and its operands, as the places of the expressions
/// given; a part left out is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    /// Which operation.
    pub kind: OpKind,
    /// Its operands, by part.
    pub parts: Vec<Option<Span>>,
}

impl Resolved {
    /// The operands of `e`, the expression this was resolved for, by part.
    pub fn operands<'e>(&self, e: &'e Expr) -> Vec<Option<&'e Expr>> {
        self.parts
            .iter()
            .map(|p| p.and_then(|span| find(e, span)))
            .collect()
    }
}

/// The outermost expression inside `e` (or `e`) written at `span`.
fn find(e: &Expr, span: Span) -> Option<&Expr> {
    // spans compare equal whatever they hold: compare the bytes
    if (e.span.start, e.span.end) == (span.start, span.end) {
        return Some(e);
    }
    if span.start < e.span.start || span.end > e.span.end {
        return None;
    }
    let mut found = None;
    each_child(e, &mut |c| {
        if found.is_none() {
            found = find(c, span);
        }
    });
    found
}

/// Calls `f` on each expression `e` is made of.
fn each_child<'e>(e: &'e Expr, f: &mut dyn FnMut(&'e Expr)) {
    match &e.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Bool(_)
        | ExprKind::Name(_)
        | ExprKind::Await(_)
        | ExprKind::Past(_) => {}
        ExprKind::Paren(x) | ExprKind::Neg(x) => f(x),
        ExprKind::Tuple(xs) | ExprKind::List(xs) => xs.values().for_each(f),
        ExprKind::Init { fields, .. } => fields.values().for_each(|i| f(&i.value)),
        ExprKind::Member { base, .. } => f(base),
        ExprKind::Call { callee, args } => {
            f(callee);
            args.values().for_each(|a| f(&a.value));
        }
        ExprKind::Index { base, index } => {
            f(base);
            for item in index.values() {
                match item {
                    IndexItem::At(x) => f(x),
                    IndexItem::Slice { lo, hi, step } => {
                        f(lo);
                        f(hi);
                        if let Some(s) = step {
                            f(s);
                        }
                    }
                }
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
    }
}
