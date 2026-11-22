//! A body: a flat list of typed operations.
//!
//! Each instruction may define one value, numbered in the order of definition
//! (`ValueId`), whose type stands in `Body::values`. A value is defined once (`let`);
//! a `var` is a local slot (`LocalId`) read with `Op::Get` and written with `Op::Set`.
//!
//! Control flow is structured and flat: a construct opens with its instruction and
//! closes with `Op::End`, as in WebAssembly, so the list nests without a tree:
//!
//! ```text
//! if c { A } else { B }      If { cond: c }  A  Else  B  End
//! for i in range(n) { A }    ForRange { n }  -> i      A  End
//! for b in stream(s) { A }   ForStream { stream: s }  -> b   A  End
//! for a, b in zip(s, t) {A}  ForZip { a: s, b: t }  -> (a, b)  A  End
//! for b, w in br.arrive() {} ForArrive { broadcast }  -> (b, w)  A  End
//! while c { A }              While  <ops of c>  WhileTest { cond: c }  A  End
//! ```
//!
//! A value defined inside a construct is not used after its `End`. A loop's values
//! (`i`, `b`, the pair) are the results of its opening instruction.

use crate::types::{
    Axis, BinOp, CmpOp, Combiner, ConstValue, Entry, Failure, FuncId, MemberId, MessageId, ParamId,
    RecordId, ReduceOp, ScalarType, SplitBy, Type, UnaryOp,
};

/// A value's number in its body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

/// A local slot's number in its body: one for each `var`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalId(pub u32);

/// An instruction's place in its body's `insts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstId(pub u32);

/// A body. It opens with one `Op::Arg(i)` for each parameter of its owner, in
/// order, so value i is parameter i; an `on M(m)` body then has `Op::Arrived`.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    /// The type of each value, by `ValueId`.
    pub values: Vec<Type>,
    /// The type of each local slot, by `LocalId`.
    pub locals: Vec<Type>,
    /// The instructions, in program order.
    pub insts: Vec<Inst>,
}

/// One instruction: an operation and the values it defines (none, one, or two for
/// the loops that bind two names).
#[derive(Clone, Debug, PartialEq)]
pub struct Inst {
    /// The operation.
    pub op: Op,
    /// The values it defines.
    pub defs: Vec<ValueId>,
}

/// A range of a rectangle view: `lo:hi` or `lo:hi:step`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    /// The first.
    pub lo: ValueId,
    /// The bound, not included.
    pub hi: ValueId,
    /// The step, when written.
    pub step: Option<ValueId>,
}

/// Where the words of a message's body come from (`body = src` on a `send` or an
/// `offer`): a value of `Type::Body` or `Type::Words(u32)`, or a state member of `u32`
/// scalars named whole, by its id (as `Op::Put` names its source).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarsFrom {
    /// A message's body, or the words of a tile.
    Value(ValueId),
    /// A state member of `u32` scalars, read whole.
    Member(MemberId),
}

/// One operation: one for each row of the operations page, plus the structure of
/// control flow. Operands are values of the same body.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    // ---- values ----
    /// A constant.
    Const(ConstValue),
    /// Parameter i of the body's function, handler, run, chain or control.
    Arg(u32),
    /// A load or invoke parameter of the module (a build parameter is a `Const`).
    Param(ParamId),
    /// Reads a `var`.
    Get(LocalId),
    /// Writes a `var` (`x = e`; `x += e` is a `Get`, a `Binary` and a `Set`).
    Set(LocalId, ValueId),
    /// `a op b` on two scalars, or on tiles and scalars (`+ - * /`, `minimum`,
    /// `maximum`); the types of the operands say which.
    Binary(BinOp, ValueId, ValueId),
    /// `a cmp b`.
    Compare(CmpOp, ValueId, ValueId),
    /// `c && d`.
    LogicAnd(ValueId, ValueId),
    /// `c || d`.
    LogicOr(ValueId, ValueId),
    // ---- control flow ----
    /// Opens `if`.
    If {
        /// The condition.
        cond: ValueId,
    },
    /// Between the two branches of an `if`.
    Else,
    /// Opens `for i in range(n)`; defines `i`.
    ForRange {
        /// The count.
        n: ValueId,
    },
    /// Opens `for b in stream`; defines `b`.
    ForStream {
        /// The stream.
        stream: ValueId,
    },
    /// Opens `for a, b in zip(s, t)` (two streams or more); defines a block of each.
    ForZip {
        /// The streams, in order.
        streams: Vec<ValueId>,
    },
    /// Opens `for b, w in br.arrive()`; defines the block and its word.
    ForArrive {
        /// The broadcast.
        broadcast: ValueId,
    },
    /// Opens `while`: the condition's instructions follow, then `WhileTest`.
    While,
    /// Ends the condition of a `while`: the loop goes on while `cond` holds.
    WhileTest {
        /// The condition.
        cond: ValueId,
    },
    /// Closes the innermost open construct.
    End,
    /// `return e` or `return`.
    Return(Option<ValueId>),
    /// `f32(n)`, `u32(x)`: a scalar converted to the named scalar type.
    Convert(ScalarType, ValueId),
    // ---- compute on tiles ----
    /// `neg(a)`, `exp(a)` and the other functions of one tile.
    Unary(UnaryOp, ValueId),
    /// `where(a cmp b, x, y)`.
    Select {
        /// The comparison.
        cmp: CmpOp,
        /// Its left operand.
        a: ValueId,
        /// Its right operand.
        b: ValueId,
        /// Where it holds.
        x: ValueId,
        /// Elsewhere.
        y: ValueId,
    },
    /// `reduce_sum(a, axis = row)` and its kin: a value for each row (or column).
    Reduce(ReduceOp, ValueId, Axis),
    /// `v.col()`: a value for each row made a tile; on a tile, column 0 spread.
    SpreadCol(ValueId),
    /// `w.row()`: a value for each column made a tile; on a tile, row 0 spread.
    SpreadRow(ValueId),
    /// `matmul(a, b)`.
    Matmul(ValueId, ValueId),
    /// `a.T` of a tile.
    Transpose(ValueId),
    /// `zeros`, of the type of its value.
    Zeros,
    /// A call of a `func`, a `combine` or an `extern func`.
    CallFunc(FuncId, Vec<ValueId>),
    // ---- views of a tensor ----
    /// `t.tiles(first, n)`.
    Tiles {
        /// The tensor.
        tensor: ValueId,
        /// The first tile.
        first: ValueId,
        /// How many.
        n: ValueId,
    },
    /// `t.tiles[r0:r1(:s), c0:c1(:s)]`.
    Rect {
        /// The tensor.
        tensor: ValueId,
        /// The rows of tiles.
        rows: Range,
        /// The columns of tiles.
        cols: Range,
    },
    /// `v.T` of a rectangle view.
    Turn(ValueId),
    /// `t.items(size = B)`, or `t.items(first, n, size = B)`.
    Items {
        /// The tensor.
        tensor: ValueId,
        /// `first` and `n`, when given.
        range: Option<(ValueId, ValueId)>,
        /// The bytes of an item.
        size: ValueId,
    },
    /// `v[i]`: one tile, read.
    ReadTile(ValueId, ValueId),
    /// `v.words()`.
    Words(ValueId),
    /// `words[i]`.
    Word(ValueId, ValueId),
    /// `v.records(R, count = c)`.
    Records(ValueId, RecordId, ValueId),
    /// `records[i]`.
    RecordAt(ValueId, ValueId),
    /// `record.field`, by the field's position.
    RecordField(ValueId, u32),
    /// `items.put(src, at = i)`.
    Put {
        /// The items view.
        items: ValueId,
        /// The state member written from.
        src: MemberId,
        /// The item.
        at: ValueId,
    },
    /// `partition(view, rank, size)`.
    Partition(ValueId, ValueId, ValueId),
    /// `stream(view, block = n)`.
    Stream(ValueId, ValueId),
    /// `sink(view, block = n)`.
    Sink(ValueId, ValueId),
    /// `out.push(t)`.
    Push(ValueId, ValueId),
    /// `store view <- value`.
    Store(ValueId, ValueId),
    /// `block[i]` of a block of tiles (`every[r]` of a gather).
    BlockAt(ValueId, ValueId),
    // ---- state ----
    /// `s.fill(stream)`.
    Fill(MemberId, ValueId),
    /// `s.zero()`.
    ZeroState(MemberId),
    /// `s[i] = x`.
    StateSet(MemberId, ValueId, ValueId),
    /// `s[i]`.
    StateGet(MemberId, ValueId),
    /// `s.bump(i)`.
    Bump(MemberId, ValueId),
    // ---- identity ----
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
    /// `group.id(rank)`.
    GroupId(ValueId),
    /// `group.split(parts = n)` or `(size = k)`.
    Split(SplitBy, ValueId),
    /// `group.leaders(parts = n)` or `(size = k)`.
    Leaders(SplitBy, ValueId),
    // ---- collectives ----
    /// `reduce(x, with = f, within = g, among = g, publish = [v])`.
    ReduceGroup {
        /// The operand.
        x: ValueId,
        /// The combiner.
        with: Combiner,
        /// `within`.
        within: Option<ValueId>,
        /// `among`.
        among: Option<ValueId>,
        /// `publish`.
        publish: Vec<ValueId>,
    },
    /// `all_reduce(x, with = f, publish = [v])`: in a handler, over the group; in a
    /// control program, over the islands' copies.
    AllReduce {
        /// The operand.
        x: ValueId,
        /// The combiner.
        with: Combiner,
        /// `publish`.
        publish: Vec<ValueId>,
    },
    /// `gather(x)`.
    Gather(ValueId),
    /// `broadcast(along = row or col, sender = c)`.
    Broadcast(Axis, ValueId),
    /// `br.is_sender`.
    IsSender(ValueId),
    /// `br.cast(block, word = w)`.
    Cast(ValueId, ValueId, Option<ValueId>),
    /// `sync(publish = [v])`.
    Sync(Vec<ValueId>),
    // ---- dynamic runs ----
    /// The message that arrived, in an `on M(m)` body.
    Arrived,
    /// `M { field: value, ... }`, the fields in the type's order.
    MakeMessage(MessageId, Vec<ValueId>),
    /// `send(M { ... }, to = rank, body = src, publish = [v])`.
    Send {
        /// The message.
        message: ValueId,
        /// The rank.
        to: ValueId,
        /// `body`: the words the message's body carries, when given.
        body: Option<ScalarsFrom>,
        /// `publish`.
        publish: Vec<ValueId>,
        /// Whether the program uses the result: an unused refusal is `send_refused`.
        used: bool,
    },
    /// `forward(m, as = T, with = { ... }, to = rank, publish = [v])`.
    Forward {
        /// The arrived message.
        message: ValueId,
        /// `as`: the message type it goes on as, when given.
        as_type: Option<MessageId>,
        /// The fields changed: position and value. With `as_type`, every field of
        /// the target type, in its order.
        with: Vec<(u32, ValueId)>,
        /// The rank.
        to: ValueId,
        /// `publish`.
        publish: Vec<ValueId>,
    },
    /// `consume m`.
    Consume(ValueId),
    /// `offer(M { ... }, to = rank, body = src)`.
    Offer(ValueId, ValueId, Option<ScalarsFrom>),
    /// `m.field`, by the field's position.
    Field(ValueId, u32),
    /// `m.body`: the body of the arrived message.
    Body(ValueId),
    // ---- control programs ----
    /// `group(first, count)` or `group(count)`.
    MakeGroup(Option<ValueId>, ValueId),
    /// `invoke(P.h(args), on = g, after = [a], after_end = [b])`.
    Invoke {
        /// What is invoked.
        entry: Entry,
        /// Its arguments.
        args: Vec<ValueId>,
        /// The group.
        on: ValueId,
        /// `after`.
        after: Vec<ValueId>,
        /// `after_end`.
        after_end: Vec<ValueId>,
    },
    /// `await h`.
    Await(ValueId),
    /// `outcome(h)`.
    Outcome(ValueId),
    /// `o.failed`.
    Failed(ValueId),
    /// `o.reason`.
    Reason(ValueId),
    /// `o.value`.
    OutcomeValue(ValueId),
    /// A failure's name, compared with `o.reason`.
    FailureName(Failure),
    // ---- islands ----
    /// `island.count`.
    IslandCount,
    /// `island.of(id)`.
    IslandOf(ValueId),
    /// `island.crosses(id)`.
    IslandCrosses(ValueId),
    /// `island.hops(id)`.
    IslandHops(ValueId),
    /// `reduce_to(t, to = i, with = f)`.
    ReduceTo(ValueId, ValueId, Combiner),
    /// `broadcast_from(t, from = i)`.
    BroadcastFrom(ValueId, ValueId),
}
