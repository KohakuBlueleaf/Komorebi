//! What a handler's body is, in the terms today's class DSL compiles: its command
//! words, its named int values, the one partition its streams share or the range
//! each streams, and a loop over a read stream, or over two zipped, that either pushes
//! a tile function's result into a write stream, sums its tiles into a `var` reduced
//! over the group and stored by the leader, or adds the products of the two streams'
//! blocks into a `var` each rank stores. A body of another form is refused by name
//! (K964).

use std::collections::HashMap;

use crate::backend::{Refusal, NOT_BUILT};
use crate::program::body::{Body, InstId, LocalId, Op, ValueId};
use crate::program::notes::{Anchor, HintArg, Named, Owner};
use crate::program::{BodyRef, FuncKind, Module};
use crate::types::{
    BinOp, Combiner, ConstValue, Count, Format, FuncId, HandlerId, On, ProgramId, ScalarType, Type,
};

/// How a command word is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordKind {
    /// A tensor: its address word.
    Tensor,
    /// A `u32`.
    U32,
    /// An `f32`: its bits.
    F32,
    /// A `u32` a stream's `@tensix.pattern` names: a pattern's word, which only
    /// chooses the mover of each block (`km::Pattern::of`); the body reads it not.
    Pattern,
}

/// A handler parameter, one command word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Word {
    /// Its name (the notes'; `p<i>` when they give none).
    pub name: String,
    /// How it is read.
    pub kind: WordKind,
    /// A tensor's element format; none for a scalar.
    pub format: Option<Format>,
}

/// An unsigned int the kernels compute from the command, as today's text prints it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Int {
    /// A `u32` word, by its index.
    Word(usize),
    /// A number.
    Const(u64),
    /// The worker's rank in the group (`km_rank`).
    Rank,
    /// The group's size (`km_size`).
    Size,
    /// A named value: an index into the plan's `locals`.
    Local(usize),
    /// `(a op b)`: `+`, `-`, `*`, `/` or `%`.
    Bin(char, Box<Int>, Box<Int>),
}

/// The tiles from `lo` to `hi` (not included), when the compile knows how many: two
/// numbers, or `hi` written `lo + n`.
pub fn span(lo: &Int, hi: &Int) -> Option<u64> {
    match (lo, hi) {
        (Int::Const(a), Int::Const(b)) => b.checked_sub(*a),
        (lo, Int::Bin('+', a, b)) if **a == *lo => match **b {
            Int::Const(n) => Some(n),
            _ => None,
        },
        _ => None,
    }
}

/// A rectangle of a tensor's tiles, `t.tiles[r0:r1, c0:c1]`, row-major in a row of the
/// tensor's width (which the load gives): its rows' ends and its columns'.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rect {
    /// `r0` and `r1`.
    pub rows: (Int, Int),
    /// `c0` and `c1`.
    pub cols: (Int, Int),
}

impl Rect {
    /// The rows and the columns, when the compile knows them.
    pub fn sides(&self) -> (Option<u64>, Option<u64>) {
        (
            span(&self.rows.0, &self.rows.1),
            span(&self.cols.0, &self.cols.1),
        )
    }
}

impl Int {
    /// Calls `f` on each part of the int, itself included.
    pub fn each(&self, f: &mut impl FnMut(&Int)) {
        f(self);
        if let Int::Bin(_, a, b) = self {
            a.each(f);
            b.each(f);
        }
    }
}

/// A named int value (`let first = worker.rank * 64`): a line of the movers' case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Local {
    /// Its name (the notes'; `v<i>` when they give none).
    pub name: String,
    /// What it is.
    pub value: Int,
    /// The instruction that makes it.
    pub inst: InstId,
}

/// A partition the kernels print: `partition(t.tiles(first, n), rank, size)`, by its
/// ints; which partitions are one is the layout rule `shared`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    /// Its name: the first partition's.
    pub name: String,
    /// `first`.
    pub first: Int,
    /// `n`.
    pub n: Int,
    /// The first partition's instruction.
    pub inst: InstId,
}

/// The tiles a stream moves: a partition's part, `t.tiles(first, n)` as it is, or a
/// rectangle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Range {
    /// An index into the plan's `parts`.
    Part(usize),
    /// Tiles `first` to `first + n - 1`.
    Tiles {
        /// The first tile.
        first: Int,
        /// The tiles.
        n: Int,
    },
    /// `t.tiles[r0:r1, c0:c1]`.
    Rect(Rect),
}

/// A stream of whole tiles over its range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Moved {
    /// The tensor's word.
    pub tensor: usize,
    /// Its block.
    pub block: u64,
    /// The instruction that opens it: the sink, or the loop over the stream.
    pub inst: InstId,
    /// Its tiles.
    pub range: Range,
    /// The word of its `@tensix.pattern`; none: the role's pattern.
    pub pattern: Option<usize>,
}

/// The loop's body: `out.push(f(x, y, k))`, the loop's tiles in order, then `f32` words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    /// The tile function.
    pub func: FuncId,
    /// Its name.
    pub name: String,
    /// Its tile operands: the loop's tiles, one from each read stream, in order.
    pub tiles: usize,
    /// The words of its scalar arguments, after the tiles.
    pub scalars: Vec<usize>,
    /// The call's instruction.
    pub inst: InstId,
}

/// What a handler's loop makes of its read stream's tiles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shape {
    /// `out.push(f(tile, k))` into a write stream of the same count (the first lesson),
    /// or `out.push(tile)`, the loop's block itself, which compute copies (program 5).
    Mapped {
        /// The write stream (the sink).
        write: Moved,
        /// The loop's call; none: the block is pushed as it is.
        call: Option<Call>,
        /// The push.
        push: InstId,
    },
    /// A loop over one read stream whose body is empty: its blocks moved and dropped,
    /// maybe in a counted loop around it, today's repeat (program 5, reading R46).
    Dropped,
    /// `total += tile` from `zeros`, the total reduced over the group, the leader's
    /// result stored (program 2).
    Summed(Sum),
    /// `acc += f(x, w)` from `zeros`, f the product of the two streams' blocks, each
    /// rank's result stored (program 6, reading R45).
    Product(Product),
}

/// The products of two streams' blocks added into a tile: `var acc: tile<f32> =
/// zeros`, `acc += f(xb, wb)` in the loop over the two zipped, f `matmul` of its
/// parameters (`[1, k]` by `[k, 1]`), and `store t.tiles(first, 1) <- acc` after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Product {
    /// The `zeros` the tile starts at.
    pub zeros: InstId,
    /// The call of f in the loop.
    pub call: Call,
    /// The store: its tensor and tiles, one block of one tile.
    pub store: Moved,
    /// f's fidelity (`@tensix.fidelity`, else the backend's).
    pub fidelity: String,
    /// The columns of tiles of the result (acc's tiles).
    pub width: u64,
}

/// A sum of a stream's tiles: `var total: tile<f32> = zeros`, `total += tile` in the
/// loop, `let whole = reduce(total, with = sum)`, `if worker.is_leader { store ... }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sum {
    /// The `zeros` the total starts at.
    pub zeros: InstId,
    /// The add in the loop.
    pub add: InstId,
    /// The reduce.
    pub reduce: InstId,
    /// The `if worker.is_leader`.
    pub leader: InstId,
    /// The store in it: its tensor and tiles, one block of one tile.
    pub store: Moved,
}

/// One handler, as the kernels print it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The program.
    pub program: ProgramId,
    /// The handler.
    pub handler: HandlerId,
    /// Its name.
    pub name: String,
    /// Its command words, in order.
    pub words: Vec<Word>,
    /// Its named int values, in the order they are written.
    pub locals: Vec<Local>,
    /// The partitions, in the order they are written: one where the streams' are
    /// the same range (`shared`), else the write stream's and the read stream's.
    pub parts: Vec<Part>,
    /// The read streams the loop walks: one, or the two it zips, in order.
    pub reads: Vec<Moved>,
    /// What the loop makes of them.
    pub shape: Shape,
    /// The count of the counted loop around the stream loop (today's repeat); none:
    /// once.
    pub repeat: Option<Int>,
}

impl Plan {
    /// The tiles the handler writes: its write stream's, or its store's; none when
    /// it drops what it reads.
    pub fn written(&self) -> Option<&Moved> {
        match &self.shape {
            Shape::Mapped { write, .. } => Some(write),
            Shape::Summed(sum) => Some(&sum.store),
            Shape::Product(product) => Some(&product.store),
            Shape::Dropped => None,
        }
    }

    /// The first instruction of the loop's body (whose line ends the loop's head); a
    /// loop with an empty body, its own.
    pub fn looped(&self) -> InstId {
        match &self.shape {
            Shape::Mapped {
                call: Some(call), ..
            } => call.inst,
            Shape::Mapped { push, .. } => *push,
            Shape::Summed(sum) => sum.add,
            Shape::Product(product) => product.call.inst,
            Shape::Dropped => self.reads[0].inst,
        }
    }
}

/// The name the notes give value `v` of `body`, if any.
pub fn value_name(module: &Module, body: BodyRef, v: ValueId) -> Option<&str> {
    module.notes.names.iter().find_map(|n| match n {
        Named::Value(b, x, name) if *b == body && *x == v => Some(name.as_str()),
        _ => None,
    })
}

/// The name the notes give parameter `i` of `owner`, if any.
pub fn param_name(module: &Module, owner: Owner, i: u32) -> Option<&str> {
    module
        .notes
        .param_names
        .iter()
        .find_map(|(a, name)| (*a == Anchor::ParamOf(owner, i)).then_some(name.as_str()))
}

/// The refusal of what is not built, at `at`.
fn not_built(what: &str, at: Anchor) -> Refusal {
    Refusal {
        code: NOT_BUILT,
        what: format!("the Tensix backend does not emit {what} yet"),
        at: Some(at),
    }
}

/// The plan of handler `h` of program `p`, or why it is not built.
pub fn plan(module: &Module, p: ProgramId, h: HandlerId) -> Result<Plan, Refusal> {
    let at = Anchor::Handler(p, h);
    let Some(handler) = module
        .programs
        .get(p.0 as usize)
        .and_then(|program| program.handlers.get(h.0 as usize))
    else {
        return Err(not_built("a handler that does not exist", at));
    };
    if handler.on != On::Group {
        return Err(not_built("a handler on a grid", at));
    }
    if handler.ret.is_some() || handler.exclusive {
        return Err(not_built("a handler with a result or `@exclusive`", at));
    }
    let mut words = Vec::new();
    for (i, ty) in handler.params.iter().enumerate() {
        let kind = match ty {
            Type::Tensor(_, false) => WordKind::Tensor,
            Type::Scalar(ScalarType::U32) => WordKind::U32,
            Type::Scalar(ScalarType::F32) => WordKind::F32,
            _ => {
                return Err(not_built(
                    "a parameter other than a tensor, a u32 or a f32",
                    Anchor::ParamOf(Owner::Handler(p, h), i as u32),
                ))
            }
        };
        let name = param_name(module, Owner::Handler(p, h), i as u32)
            .map_or_else(|| format!("p{i}"), str::to_string);
        let format = match ty {
            Type::Tensor(f, _) => Some(*f),
            _ => None,
        };
        words.push(Word { name, kind, format });
    }
    let body = BodyRef::Handler(p, h);
    let patterns = patterns(module, body, &mut words)?;
    let found = Walk::new(module, &handler.body, body, &words).run()?;
    let Found {
        locals,
        parts,
        mut reads,
        mut shape,
        repeat,
    } = found;
    if let (Some((inst, _)), true) = (
        patterns.first(),
        matches!(shape, Shape::Summed(_) | Shape::Product(_)),
    ) {
        let what = match shape {
            Shape::Product(_) => "a product's stream in a pattern the command gives",
            _ => "a reduce's stream in a pattern the command gives",
        };
        return Err(not_built(what, Anchor::Inst(body, *inst)));
    }
    // the loop's hint is each stream's it walks (zipped, the loop opens the first)
    for (inst, word) in &patterns {
        if reads.iter().any(|r| r.inst == *inst) {
            reads.iter_mut().for_each(|r| r.pattern = Some(*word));
        } else if let Shape::Mapped { write, .. } = &mut shape {
            if write.inst != *inst {
                return Err(not_built(
                    "a `@tensix.pattern` on other than a loop over streams or a sink",
                    Anchor::Inst(body, *inst),
                ));
            }
            write.pattern = Some(*word);
        }
    }
    Ok(Plan {
        program: p,
        handler: h,
        name: handler.name.clone(),
        words,
        locals,
        parts,
        reads,
        shape,
        repeat,
    })
}

/// The `@tensix.pattern(p)` hints of a handler's body, (the instruction, p's word):
/// each names one `u32` parameter, which becomes a pattern's word (the program does
/// not read it). Any other `@tensix.pattern` is refused by name.
fn patterns(
    module: &Module,
    body: BodyRef,
    words: &mut [Word],
) -> Result<Vec<(InstId, usize)>, Refusal> {
    let mut out = Vec::new();
    for hint in &module.notes.hints {
        let Anchor::Inst(at, inst) = hint.anchor else {
            continue;
        };
        if at != body || hint.path != ["tensix", "pattern"] {
            continue;
        }
        let word = match hint.args.as_slice() {
            [HintArg::Name(name)] => words.iter().position(|w| {
                &w.name == name && matches!(w.kind, WordKind::U32 | WordKind::Pattern)
            }),
            _ => None,
        };
        let Some(word) = word else {
            return Err(not_built(
                "a `@tensix.pattern` other than of one `u32` parameter of the handler",
                hint.anchor,
            ));
        };
        words[word].kind = WordKind::Pattern;
        out.push((inst, word));
    }
    Ok(out)
}

/// The layout rule for partitions (today's DSL writes one partition and indexes both
/// tensors with it, `y[src]`): partitions that take the same values (the same `first`
/// and `n`, by `worker.rank` and `group.size`, the only rank and size the walk takes)
/// are one range whatever their tensors, printed once, as the first of them written;
/// partitions that differ in a value are printed each. Gives the partitions to print
/// and, for each given one, its index among them.
pub fn shared(parts: &[Part]) -> (Vec<Part>, Vec<usize>) {
    let mut order: Vec<usize> = (0..parts.len()).collect();
    order.sort_by_key(|&i| parts[i].inst);
    let mut out: Vec<Part> = Vec::new();
    let mut index = vec![0; parts.len()];
    for i in order {
        let part = &parts[i];
        match out
            .iter()
            .position(|p| (&p.first, &p.n) == (&part.first, &part.n))
        {
            Some(at) => index[i] = at,
            None => {
                index[i] = out.len();
                out.push(part.clone());
            }
        }
    }
    (out, index)
}

/// What a value of the body is, as far as the plan reads it.
#[derive(Clone, Debug)]
enum Val {
    Arg(usize),
    Int(Int),
    Tiles {
        tensor: usize,
        first: Int,
        n: Int,
    },
    Part(usize, Part),
    Rect {
        tensor: usize,
        rect: Rect,
    },
    Stream,
    Sink,
    /// The loop's tile from read stream `i`.
    Tile(usize),
    Called,
    Zeros,
    Total,
    Added,
    Reduced,
    Leader,
}

/// The tiles a stream was opened on, before the partitions are shared.
#[derive(Clone, Debug)]
enum Over {
    Part(Part),
    Tiles(Int, Int),
    Rect(Rect),
}

/// What the walk found.
struct Found {
    locals: Vec<Local>,
    parts: Vec<Part>,
    reads: Vec<Moved>,
    shape: Shape,
    repeat: Option<Int>,
}

/// What the walk found of a sum, as it is read.
#[derive(Default)]
struct Summing {
    local: Option<LocalId>,
    zeros_seen: bool,
    zeros: Option<InstId>,
    add: Option<InstId>,
    summed: bool,
    reduce: Option<InstId>,
    leader: Option<InstId>,
    store: Option<Moved>,
    closed: bool,
    /// The add is of a call's result (a product), not of the loop's tile.
    product: bool,
}

/// The body read instruction by instruction: arguments, ints of words, numbers and
/// the rank (a named one a local), partitions of `t.tiles(first, n)` by rank and
/// size, one sink, one or two streams, each of a partition or of tiles, the loop over
/// the stream (or the two zipped) whose body calls a tile function on the loop's tiles
/// (or blocks) and `f32` words and pushes the result into the sink, the loop's end,
/// and nothing after; or,
/// with no sink, a `var` of `zeros` the loop adds each tile to, its reduce with `sum`
/// after the loop and the leader's store of the result, and nothing after.
struct Walk<'m> {
    module: &'m Module,
    body: &'m Body,
    at: BodyRef,
    words: &'m [Word],
    vals: HashMap<ValueId, Val>,
    locals: Vec<Local>,
    sink: Option<(ValueId, Moved, Over)>,
    /// The read streams opened: each its value, its stream and its tiles.
    streams: Vec<(ValueId, Moved, Over)>,
    /// The streams the loop walks, by their place in `streams`.
    walked: Vec<usize>,
    call: Option<Call>,
    looped: bool,
    pushed: bool,
    /// The push, when it pushes the loop's block itself.
    copied: Option<InstId>,
    ended: bool,
    sum: Summing,
    /// A counted loop around the stream loop: its count; and whether it is closed.
    repeat: Option<Int>,
    repeat_closed: bool,
}

impl<'m> Walk<'m> {
    fn new(module: &'m Module, body: &'m Body, at: BodyRef, words: &'m [Word]) -> Self {
        Walk {
            module,
            body,
            at,
            words,
            vals: HashMap::new(),
            locals: Vec::new(),
            sink: None,
            streams: Vec::new(),
            walked: Vec::new(),
            call: None,
            looped: false,
            pushed: false,
            copied: None,
            ended: false,
            sum: Summing::default(),
            repeat: None,
            repeat_closed: false,
        }
    }

    fn val(&self, v: ValueId) -> Option<Val> {
        self.vals.get(&v).cloned()
    }

    /// The int `v` is: a `u32` word, a number, the rank or size, a named value, or
    /// an operation of these.
    fn int(&self, v: ValueId) -> Option<Int> {
        match self.val(v)? {
            Val::Arg(i) if self.words.get(i).is_some_and(|w| w.kind == WordKind::U32) => {
                Some(Int::Word(i))
            }
            Val::Int(i) => Some(i),
            _ => None,
        }
    }

    /// The word `v` is, if it is an argument of `kind`.
    fn word(&self, v: ValueId, kind: WordKind) -> Option<usize> {
        match self.val(v) {
            Some(Val::Arg(i)) if self.words.get(i).is_some_and(|w| w.kind == kind) => Some(i),
            _ => None,
        }
    }

    fn run(mut self) -> Result<Found, Refusal> {
        for (i, inst) in self.body.insts.iter().enumerate() {
            let id = InstId(i as u32);
            let anchor = Anchor::Inst(self.at, id);
            let closes_repeat = matches!(inst.op, Op::End) && self.repeat.is_some();
            if self.ended && self.sum.local.is_none() && !(closes_repeat && !self.repeat_closed) {
                return Err(not_built("a handler's lines after its stream loop", anchor));
            }
            // a counted loop whose body is exactly one stream loop (R46): its stream
            let in_repeat = self.repeat.is_some() && !self.looped;
            let stream_part = matches!(
                inst.op,
                Op::Stream(..) | Op::ForStream { .. } | Op::Param(_) | Op::Const(_) | Op::Arg(_)
            );
            if in_repeat && !stream_part {
                return Err(not_built(
                    "a counted loop other than around one stream loop, today's repeat",
                    anchor,
                ));
            }
            if self.sum.closed {
                let what = if self.sum.product {
                    "a handler's lines after its store"
                } else {
                    "a handler's lines after its leader's store"
                };
                return Err(not_built(what, anchor));
            }
            let val = self
                .step(&inst.op, id)
                .map_err(|what| not_built(&what, anchor))?;
            if let (Some(v), Some(val)) = (inst.defs.first(), val) {
                self.vals.insert(*v, val);
            }
        }
        let at = match self.at {
            BodyRef::Handler(p, h) => Anchor::Handler(p, h),
            _ => Anchor::Module,
        };
        if self.sum.local.is_some() {
            return self.summed(at);
        }
        if self.sum.zeros_seen {
            return Err(not_built("a zeros other than a var's start", at));
        }
        if self.repeat.is_some() && !self.repeat_closed {
            return Err(not_built("a counted loop not closed", at));
        }
        if self.ended && self.sink.is_none() && self.call.is_none() && !self.pushed {
            return self.dropped(at);
        }
        if self.repeat.is_some() {
            return Err(not_built(
                "a counted loop around a stream loop that pushes (a repeat of a loop with a body)",
                at,
            ));
        }
        let call = self.call.take();
        let (Some((_, write, wover)), Some(push), true) = (
            self.sink.take(),
            call.as_ref().map(|c| c.inst).or(self.copied),
            self.ended,
        ) else {
            return Err(not_built(
                "a handler other than a stream loop that pushes a tile function's result, or its \
                 block, into a sink",
                at,
            ));
        };
        let (reads, overs) = self.walked_streams();
        if reads.iter().any(|r| r.block != write.block) {
            return Err(not_built("a sink in blocks other than its loop's", at));
        }
        if overs
            .iter()
            .chain([&wover])
            .any(|o| matches!(o, Over::Rect(_)))
        {
            return Err(not_built(
                "a sink or a pushing loop's stream of a rectangle",
                at,
            ));
        }
        let count = |o: &Over| match o {
            Over::Part(p) => p.n.clone(),
            Over::Tiles(_, n) => n.clone(),
            Over::Rect(_) => Int::Const(0),
        };
        if overs.iter().any(|o| count(o) != count(&wover)) {
            return Err(not_built(
                "a sink of another count than its loop's stream",
                at,
            ));
        }
        let all: Vec<Over> = std::iter::once(wover).chain(overs).collect();
        let (parts, ranges) = if all.iter().all(|o| matches!(o, Over::Part(_))) {
            let given: Vec<Part> = all
                .into_iter()
                .filter_map(|o| match o {
                    Over::Part(p) => Some(p),
                    Over::Tiles(..) | Over::Rect(_) => None,
                })
                .collect();
            let (parts, index) = shared(&given);
            (
                parts,
                index.into_iter().map(Range::Part).collect::<Vec<_>>(),
            )
        } else if all.iter().all(|o| matches!(o, Over::Tiles(..))) {
            let ranges = all
                .into_iter()
                .filter_map(|o| match o {
                    Over::Tiles(first, n) => Some(Range::Tiles { first, n }),
                    Over::Part(_) | Over::Rect(_) => None,
                })
                .collect();
            (Vec::new(), ranges)
        } else {
            return Err(not_built(
                "a sink of a partition and its loop's stream of tiles, or the other way",
                at,
            ));
        };
        let mut ranges = ranges.into_iter();
        let write = Moved {
            range: ranges.next().unwrap_or(Range::Part(0)),
            ..write
        };
        let reads: Vec<Moved> = reads
            .into_iter()
            .zip(ranges)
            .map(|(read, range)| Moved { range, ..read })
            .collect();
        let moved: Vec<&Moved> = std::iter::once(&write).chain(&reads).collect();
        self.locals_read(&parts, &moved, at)?;
        Ok(Found {
            locals: self.locals,
            parts,
            reads,
            shape: Shape::Mapped { write, call, push },
            repeat: None,
        })
    }

    /// The handler whose stream loop drops its blocks, maybe repeated (R46).
    fn dropped(&mut self, at: Anchor) -> Result<Found, Refusal> {
        let (reads, overs) = self.walked_streams();
        if reads.len() != 1 {
            return Err(not_built("a loop that drops two streams' blocks", at));
        }
        let range = match overs.into_iter().next() {
            Some(Over::Tiles(first, n)) => Range::Tiles { first, n },
            _ => {
                return Err(not_built(
                    "a dropped stream of other than t.tiles(first, n)",
                    at,
                ))
            }
        };
        let reads = vec![Moved {
            range,
            ..reads[0].clone()
        }];
        let moved: Vec<&Moved> = reads.iter().collect();
        self.locals_read(&[], &moved, at)?;
        Ok(Found {
            locals: std::mem::take(&mut self.locals),
            parts: Vec::new(),
            reads,
            shape: Shape::Dropped,
            repeat: self.repeat.clone(),
        })
    }

    /// The loop at `id` over `streams` (values of read streams, in order): each read
    /// stream is walked by it, once.
    fn walk_streams(&mut self, streams: &[ValueId], id: InstId) -> Result<(), String> {
        let found: Option<Vec<usize>> = streams
            .iter()
            .map(|v| self.streams.iter().position(|(s, _, _)| s == v))
            .collect();
        match found {
            Some(walked) if !self.looped && walked.len() == self.streams.len() => {
                for &i in &walked {
                    self.streams[i].1.inst = id;
                }
                self.walked = walked;
                self.looped = true;
                Ok(())
            }
            _ => Err("a loop other than one over each read stream, once".into()),
        }
    }

    /// The streams the loop walks, in the loop's order: each stream and its tiles.
    fn walked_streams(&mut self) -> (Vec<Moved>, Vec<Over>) {
        let walked = std::mem::take(&mut self.walked);
        walked
            .into_iter()
            .map(|i| {
                let (_, moved, over) = self.streams[i].clone();
                (moved, over)
            })
            .unzip()
    }

    /// Refuses a named value no stream reads (no line today's text prints) and a read
    /// stream's count of a named value.
    /// `moved` is the written tiles, then the read streams.
    fn locals_read(&self, parts: &[Part], moved: &[&Moved], at: Anchor) -> Result<(), Refusal> {
        let mut used = Vec::new();
        let ints = parts.iter().flat_map(|p| [&p.first, &p.n]).chain(
            moved.iter().map(|m| &m.range).flat_map(|r| match r {
                Range::Tiles { first, n } => vec![first, n],
                Range::Rect(rect) => vec![&rect.rows.0, &rect.rows.1, &rect.cols.0, &rect.cols.1],
                Range::Part(_) => Vec::new(),
            }),
        );
        for int in ints {
            int.each(&mut |i| {
                if let Int::Local(l) = i {
                    used.push(*l);
                }
            });
        }
        for (l, local) in self.locals.iter().enumerate().rev() {
            local.value.each(&mut |i| {
                if let Int::Local(k) = i {
                    if used.contains(&l) {
                        used.push(*k);
                    }
                }
            });
        }
        if let Some(local) = (0..self.locals.len())
            .rev()
            .find(|l| !used.contains(l))
            .map(|l| &self.locals[l])
        {
            return Err(not_built(
                "a named value no stream reads",
                Anchor::Inst(self.at, local.inst),
            ));
        }
        let mut counted = false;
        for read in &moved[1..] {
            if let Range::Tiles { n, .. } = &read.range {
                n.each(&mut |i| counted |= matches!(i, Int::Local(_)));
            }
        }
        if counted {
            return Err(not_built("a read stream's count of a named value", at));
        }
        Ok(())
    }

    /// A sum's plan: its stream of a partition or of tiles, its total of `f32` tiles
    /// of one a block, its reduce and the leader's store; every named value read.
    fn summed(mut self, at: Anchor) -> Result<Found, Refusal> {
        if self.sum.product {
            return self.product(at);
        }
        let sum = std::mem::take(&mut self.sum);
        if self.sink.is_some() {
            return Err(not_built("a sink in a handler of a sum", at));
        }
        let (reads, overs) = self.walked_streams();
        let (Ok([read]), Ok([over]), Some(zeros), Some(add), Some(reduce), Some(leader)) = (
            <[Moved; 1]>::try_from(reads),
            <[Over; 1]>::try_from(overs),
            sum.zeros,
            sum.add,
            sum.reduce,
            sum.leader,
        ) else {
            return Err(not_built(
                "a handler other than a stream loop summed into a var, reduced and stored by the leader",
                at,
            ));
        };
        let (Some(store), true) = (sum.store, sum.closed) else {
            return Err(not_built("a reduce other than one the leader stores", at));
        };
        let format = self.words.get(read.tensor).and_then(|w| w.format);
        if format != Some(Format::F32) || read.block != 1 {
            return Err(not_built(
                "a sum of a stream other than of f32 tiles one a block",
                at,
            ));
        }
        let (parts, read) = match over {
            Over::Part(part) => (
                vec![part],
                Moved {
                    range: Range::Part(0),
                    ..read
                },
            ),
            Over::Tiles(first, n) => (
                Vec::new(),
                Moved {
                    range: Range::Tiles { first, n },
                    ..read
                },
            ),
            Over::Rect(_) => return Err(not_built("a sum of a rectangle's stream", at)),
        };
        self.locals_read(&parts, &[&store, &read], at)?;
        Ok(Found {
            locals: self.locals,
            parts,
            reads: vec![read],
            repeat: None,
            shape: Shape::Summed(Sum {
                zeros,
                add,
                reduce,
                leader,
                store,
            }),
        })
    }

    /// A product's plan (program 6, reading R45): two streams zipped, of rectangles or
    /// of tiles, the first in blocks of k tiles (`[1, k]`) and the second of k rows of
    /// one tile (`[k, 1]`), both `bf16`; the call of a function that is `matmul` of its
    /// two parameters, its result an `f32` tile added to the `var` of zeros; the store
    /// of that tile into one `f32` tile; every named value read.
    fn product(mut self, at: Anchor) -> Result<Found, Refusal> {
        let sum = std::mem::take(&mut self.sum);
        let call = self.call.take();
        if self.sink.is_some() {
            return Err(not_built("a sink in a handler of a product", at));
        }
        let (reads, overs) = self.walked_streams();
        let (Ok([x, w]), Ok([xo, wo]), Some(zeros), Some(call), true) = (
            <[Moved; 2]>::try_from(reads),
            <[Over; 2]>::try_from(overs),
            sum.zeros,
            call,
            sum.summed,
        ) else {
            return Err(not_built(
                "a product other than of two zipped streams' blocks added into a var",
                at,
            ));
        };
        let (Some(store), true) = (sum.store, sum.closed) else {
            return Err(not_built(
                "a product's var other than stored after its loop",
                at,
            ));
        };
        let k = self.product_function(call.func, at)?;
        let format = |m: &Moved| self.words.get(m.tensor).and_then(|w| w.format);
        if format(&x) != Some(Format::Bf16) || format(&w) != Some(Format::Bf16) {
            return Err(not_built(
                "a product of streams other than of bf16 tiles",
                at,
            ));
        }
        if format(&store) != Some(Format::F32) {
            return Err(not_built(
                "a product stored into a tensor other than f32",
                at,
            ));
        }
        if k != Some(x.block) || w.block != x.block {
            return Err(not_built(
                "a product whose function's shapes are not its streams' blocks ([1, k] by [k, 1])",
                at,
            ));
        }
        let range = |o: Over| match o {
            Over::Rect(rect) => Ok(Range::Rect(rect)),
            Over::Tiles(first, n) => Ok(Range::Tiles { first, n }),
            Over::Part(_) => Err(not_built("a product's stream of a partition", at)),
        };
        let reads = vec![
            Moved {
                range: range(xo)?,
                ..x
            },
            Moved {
                range: range(wo)?,
                ..w
            },
        ];
        let fidelity = super::unit::fidelity(self.module, call.func)?;
        self.locals_read(&[], &[&store, &reads[0], &reads[1]], at)?;
        Ok(Found {
            locals: self.locals,
            parts: Vec::new(),
            reads,
            repeat: None,
            shape: Shape::Product(Product {
                zeros,
                call,
                store,
                fidelity,
                width: 1,
            }),
        })
    }

    /// Whether function `f` is the product of its two parameters, `return matmul(x,
    /// w)`, x `[1, k]` (or a tile) and w `[k, 1]` (or a tile), its result an `f32`
    /// tile: its k when it is, else refused at `at`.
    fn product_function(&self, f: FuncId, at: Anchor) -> Result<Option<u64>, Refusal> {
        let func = self.module.funcs.get(f.0 as usize);
        let body = func.and_then(|func| func.body.as_ref());
        let (Some(func), Some(body)) = (func, body) else {
            return Err(not_built("a product's function with no body", at));
        };
        let ops: Vec<&Op> = body.insts.iter().map(|i| &i.op).collect();
        let defs = |i: usize| {
            body.insts
                .get(i)
                .and_then(|inst| inst.defs.first().copied())
        };
        let product = match ops.as_slice() {
            [Op::Arg(0), Op::Arg(1), Op::Matmul(a, b), Op::Return(Some(r))] => {
                Some(*a) == defs(0) && Some(*b) == defs(1) && Some(*r) == defs(2)
            }
            _ => false,
        };
        let shape = |t: &Type| match t {
            Type::Tile(Format::Bf16) => Some((1, 1)),
            Type::Block(Format::Bf16, Count::Const(r), Count::Const(c)) => Some((*r, *c)),
            _ => None,
        };
        let shapes: Vec<Option<(u64, u64)>> = func.params.iter().map(shape).collect();
        match (product, shapes.as_slice(), &func.ret) {
            (true, [Some((1, k)), Some((rows, 1))], Type::Tile(Format::F32)) if k == rows => {
                Ok(Some(*k))
            }
            _ => Err(not_built(
                "a function in `acc +=` other than `matmul` of a [1, k] and a [k, 1] of bf16 \
                 giving an f32 tile",
                Anchor::Func(f),
            )),
        }
    }

    /// What one instruction is, or why it is not built.
    fn step(&mut self, op: &Op, id: InstId) -> Result<Option<Val>, String> {
        Ok(match op {
            Op::Arg(a) => Some(Val::Arg(*a as usize)),
            Op::Const(ConstValue::Int(_, x)) => Some(Val::Int(Int::Const(*x))),
            Op::Rank => Some(Val::Int(Int::Rank)),
            Op::GroupSize => Some(Val::Int(Int::Size)),
            Op::Binary(BinOp::Add, a, b)
                if matches!(self.val(*a), Some(Val::Total))
                    && matches!(self.val(*b), Some(Val::Tile(0))) =>
            {
                if !self.looped || self.ended || self.sum.add.is_some() {
                    return Err("an add of tiles other than the total's in the loop, once".into());
                }
                self.sum.add = Some(id);
                Some(Val::Added)
            }
            // a product's: the call's result added (program 6)
            Op::Binary(BinOp::Add, a, b)
                if matches!(self.val(*a), Some(Val::Total))
                    && matches!(self.val(*b), Some(Val::Called)) =>
            {
                if !self.looped || self.ended || self.sum.add.is_some() {
                    return Err(
                        "an add of a call's result other than the var's in the loop, once".into(),
                    );
                }
                self.sum.add = Some(id);
                self.sum.product = true;
                Some(Val::Added)
            }
            Op::Zeros if !self.looped && self.sum.zeros.is_none() => {
                self.sum.zeros_seen = true;
                Some(Val::Zeros)
            }
            Op::Set(l, v) => {
                let tile = self.body.locals.get(l.0 as usize) == Some(&Type::Tile(Format::F32));
                match (self.val(*v), self.sum.local) {
                    (Some(Val::Zeros), None) if tile && !self.looped => {
                        self.sum.local = Some(*l);
                        self.sum.zeros = Some(id);
                    }
                    (Some(Val::Added), Some(local)) if local == *l && !self.sum.summed => {
                        self.sum.summed = true;
                    }
                    _ => {
                        return Err(
                            "a var other than a tile<f32> of zeros the loop adds its tiles to"
                                .into(),
                        )
                    }
                }
                None
            }
            Op::Get(l) if self.sum.local == Some(*l) => Some(Val::Total),
            Op::ReduceGroup {
                x,
                with,
                within,
                among,
                publish,
            } => {
                let total = matches!(self.val(*x), Some(Val::Total));
                if !self.ended || !total || self.sum.reduce.is_some() {
                    return Err("a reduce other than of the loop's total, once, after it".into());
                }
                if *with != Combiner::Sum {
                    return Err("a reduce with other than sum".into());
                }
                if within.is_some() || among.is_some() || !publish.is_empty() {
                    return Err("a reduce with within, among or publish".into());
                }
                self.sum.reduce = Some(id);
                Some(Val::Reduced)
            }
            Op::IsLeader if self.sum.reduce.is_some() => Some(Val::Leader),
            Op::If { cond } => {
                let leader = matches!(self.val(*cond), Some(Val::Leader));
                if !leader || self.sum.leader.is_some() {
                    return Err(
                        "an if other than `if worker.is_leader` after the reduce, once".into(),
                    );
                }
                self.sum.leader = Some(id);
                None
            }
            Op::Store(view, v) if self.sum.product => {
                let total = matches!(self.val(*v), Some(Val::Total));
                let Some(Val::Tiles { tensor, first, n }) = self.val(*view) else {
                    return Err("a store other than into t.tiles(first, n) of a parameter".into());
                };
                if !total || !self.ended || self.sum.store.is_some() || n != Int::Const(1) {
                    return Err(
                        "a store other than of the product's var into one tile, after its loop, \
                         once"
                            .into(),
                    );
                }
                self.sum.store = Some(Moved {
                    tensor,
                    block: 1,
                    inst: id,
                    range: Range::Tiles { first, n },
                    pattern: None,
                });
                self.sum.closed = true;
                None
            }
            Op::Store(view, v) => {
                let reduced = matches!(self.val(*v), Some(Val::Reduced));
                let inside = self.sum.leader.is_some() && !self.sum.closed;
                let Some(Val::Tiles { tensor, first, n }) = self.val(*view) else {
                    return Err("a store other than into t.tiles(first, n) of a parameter".into());
                };
                if !reduced || !inside || self.sum.store.is_some() || n != Int::Const(1) {
                    return Err(
                        "a store other than of the reduce's tile into one tile, by the leader, once"
                            .into(),
                    );
                }
                self.sum.store = Some(Moved {
                    tensor,
                    block: 1,
                    inst: id,
                    range: Range::Tiles { first, n },
                    pattern: None,
                });
                None
            }
            Op::End if self.sum.leader.is_some() && self.sum.store.is_some() => {
                self.sum.closed = true;
                None
            }
            Op::Binary(op, a, b) => {
                let op = match op {
                    BinOp::Add => '+',
                    BinOp::Sub => '-',
                    BinOp::Mul => '*',
                    BinOp::Div => '/',
                    BinOp::Rem => '%',
                    _ => return Err("an int operation other than + - * / %".into()),
                };
                let (Some(a), Some(b)) = (self.int(*a), self.int(*b)) else {
                    return Err("an operation other than of u32 words, numbers and the rank".into());
                };
                let value = Int::Bin(op, Box::new(a), Box::new(b));
                let defined = self
                    .body
                    .insts
                    .get(id.0 as usize)
                    .and_then(|i| i.defs.first().copied());
                match defined.and_then(|v| value_name(self.module, self.at, v)) {
                    Some(name) => {
                        self.locals.push(Local {
                            name: name.to_string(),
                            value,
                            inst: id,
                        });
                        Some(Val::Int(Int::Local(self.locals.len() - 1)))
                    }
                    None => Some(Val::Int(value)),
                }
            }
            Op::Tiles { tensor, first, n } => {
                let (Some(tensor), Some(first), Some(n)) = (
                    self.word(*tensor, WordKind::Tensor),
                    self.int(*first),
                    self.int(*n),
                ) else {
                    return Err("a view other than t.tiles(first, n) of a parameter".into());
                };
                Some(Val::Tiles { tensor, first, n })
            }
            Op::Rect { tensor, rows, cols } => {
                let end = |v: ValueId| self.int(v);
                let tensor = self.word(*tensor, WordKind::Tensor);
                let ends = (end(rows.lo), end(rows.hi), end(cols.lo), end(cols.hi));
                let (Some(tensor), (Some(r0), Some(r1), Some(c0), Some(c1)), None, None) =
                    (tensor, ends, rows.step, cols.step)
                else {
                    return Err(
                        "a rectangle other than t.tiles[r0:r1, c0:c1] of a parameter, its ends \
                         of u32 words, numbers and the rank"
                            .into(),
                    );
                };
                let rect = Rect {
                    rows: (r0, r1),
                    cols: (c0, c1),
                };
                if rect.sides().0.is_none() || rect.sides().1.is_none() {
                    return Err(
                        "a rectangle whose rows or columns the compile does not know (each \
                         side two numbers, or `a:a + n`)"
                            .into(),
                    );
                }
                Some(Val::Rect { tensor, rect })
            }
            Op::Partition(view, rank, size) => {
                match (self.val(*view), self.int(*rank), self.int(*size)) {
                    (Some(Val::Tiles { tensor, first, n }), Some(Int::Rank), Some(Int::Size)) => {
                        let defined = self
                            .body
                            .insts
                            .get(id.0 as usize)
                            .and_then(|i| i.defs.first().copied());
                        let name = defined
                            .and_then(|v| value_name(self.module, self.at, v))
                            .map_or_else(
                                || format!("v{}", defined.map_or(0, |v| v.0)),
                                str::to_string,
                            );
                        let part = Part {
                            name,
                            first,
                            n,
                            inst: id,
                        };
                        Some(Val::Part(tensor, part))
                    }
                    _ => {
                        return Err(
                            "a partition other than of a view by worker.rank and group.size".into(),
                        )
                    }
                }
            }
            Op::Sink(view, block) | Op::Stream(view, block) => {
                let rows = self
                    .body
                    .insts
                    .get(id.0 as usize)
                    .and_then(|i| i.defs.first())
                    .and_then(|v| self.body.values.get(v.0 as usize));
                let rows = match rows {
                    Some(Type::Stream(_, r, _) | Type::Sink(_, r, _)) => *r,
                    _ => Count::ONE,
                };
                let block = match self.int(*block) {
                    Some(Int::Const(b)) => b,
                    _ => return Err("a stream in a block that is not a constant".into()),
                };
                let (tensor, over) = match self.val(*view) {
                    Some(Val::Part(tensor, part)) => (tensor, Over::Part(part)),
                    Some(Val::Tiles { tensor, first, n }) => (tensor, Over::Tiles(first, n)),
                    Some(Val::Rect { tensor, rect }) => (tensor, Over::Rect(rect)),
                    _ => {
                        return Err(
                            "a stream other than of a partition, of t.tiles(first, n) or of a \
                             rectangle"
                                .into(),
                        )
                    }
                };
                // blocks of r rows of c tiles: today's blocks of r c tiles of a rectangle of
                // c columns, walked row by row (R45)
                let block = match (rows, &over) {
                    (Count::Const(1), _) => block,
                    (Count::Const(r), Over::Rect(rect))
                        if rect.sides().1 == Some(block) && matches!(op, Op::Stream(..)) =>
                    {
                        r * block
                    }
                    _ => {
                        return Err(
                            "a stream in blocks of more than one row of tiles other than of a \
                             rectangle of the block's columns"
                                .into(),
                        )
                    }
                };
                let moved = Moved {
                    tensor,
                    block,
                    inst: id,
                    range: Range::Part(0),
                    pattern: None,
                };
                if matches!(op, Op::Stream(..)) {
                    if self.streams.len() == 2 {
                        return Err("a third read stream".into());
                    }
                    let defined = self
                        .body
                        .insts
                        .get(id.0 as usize)
                        .and_then(|i| i.defs.first().copied());
                    let Some(v) = defined else {
                        return Err("a stream that defines nothing".into());
                    };
                    self.streams.push((v, moved, over));
                    Some(Val::Stream)
                } else {
                    if self.sink.is_some() {
                        return Err("a second write stream".into());
                    }
                    let defined = self
                        .body
                        .insts
                        .get(id.0 as usize)
                        .and_then(|i| i.defs.first().copied());
                    let Some(v) = defined else {
                        return Err("a sink that defines nothing".into());
                    };
                    self.sink = Some((v, moved, over));
                    Some(Val::Sink)
                }
            }
            Op::ForStream { stream } => {
                self.walk_streams(&[*stream], id)?;
                Some(Val::Tile(0))
            }
            Op::ForZip { a, b } => {
                self.walk_streams(&[*a, *b], id)?;
                // the second tile: the first is the instruction's first value
                let defs = &self.body.insts[id.0 as usize].defs;
                if let Some(v) = defs.get(1) {
                    self.vals.insert(*v, Val::Tile(1));
                }
                Some(Val::Tile(0))
            }
            Op::CallFunc(f, args) => {
                let plain = self
                    .module
                    .funcs
                    .get(f.0 as usize)
                    .filter(|func| func.kind == FuncKind::Plain);
                let tiles = self.walked.len();
                let in_order = args.len() >= tiles
                    && args[..tiles]
                        .iter()
                        .enumerate()
                        .all(|(i, t)| matches!(self.val(*t), Some(Val::Tile(j)) if i == j));
                let scalars: Option<Vec<usize>> = args[tiles.min(args.len())..]
                    .iter()
                    .map(|s| self.word(*s, WordKind::F32))
                    .collect();
                let (Some(func), true, None, true, Some(scalars)) =
                    (plain, self.looped, self.call.as_ref(), in_order, scalars)
                else {
                    return Err(
                        "a call other than f(the loop's tiles in order, f32 words), once, in the loop"
                            .into(),
                    );
                };
                self.call = Some(Call {
                    func: *f,
                    name: func.name.clone(),
                    tiles,
                    scalars,
                    inst: id,
                });
                Some(Val::Called)
            }
            Op::Push(s, v) => {
                let to_sink = self.sink.as_ref().is_some_and(|(sv, _, _)| sv == s);
                // the call's result, or the loop's block itself (one stream walked)
                let block = matches!(self.val(*v), Some(Val::Tile(0))) && self.walked.len() == 1;
                let called = matches!(self.val(*v), Some(Val::Called));
                if self.pushed || !to_sink || !(called || block) {
                    return Err(
                        "a push other than of the call's tile or the loop's block into the \
                         sink, once"
                            .into(),
                    );
                }
                if block {
                    self.copied = Some(id);
                }
                self.pushed = true;
                None
            }
            Op::ForRange { n } => {
                let count = self.int(*n);
                let (None, false, Some(count)) = (&self.repeat, self.looped, count) else {
                    return Err(
                        "a counted loop other than one of u32 words and numbers around the \
                         stream loop"
                            .into(),
                    );
                };
                self.repeat = Some(count);
                None
            }
            Op::End if self.ended && self.repeat.is_some() && !self.repeat_closed => {
                self.repeat_closed = true;
                None
            }
            // a loop whose body is a push or a sum; an empty one drops its blocks
            Op::End
                if self.looped
                    && !self.ended
                    && (self.pushed || self.sum.summed || self.call.is_none()) =>
            {
                self.ended = true;
                None
            }
            other => {
                let name = format!("{other:?}");
                let name = name.split(['(', ' ', '{']).next().unwrap_or_default();
                return Err(format!("the operation {name} in a handler"));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(name: &str, first: usize, n: usize, inst: u32) -> Part {
        Part {
            name: name.into(),
            first: Int::Word(first),
            n: Int::Word(n),
            inst: InstId(inst),
        }
    }

    #[test]
    fn two_equal_partitions_are_one_and_two_that_differ_are_two() {
        let (one, index) = shared(&[part("dst", 2, 3, 12), part("src", 2, 3, 8)]);
        assert_eq!(one, vec![part("src", 2, 3, 8)]);
        assert_eq!(index, vec![0, 0]);
        let (two, index) = shared(&[part("dst", 3, 3, 12), part("src", 2, 3, 8)]);
        assert_eq!(two, vec![part("src", 2, 3, 8), part("dst", 3, 3, 12)]);
        assert_eq!(index, vec![1, 0]);
        let (two, _) = shared(&[part("dst", 2, 4, 12), part("src", 2, 3, 8)]);
        assert_eq!(two.len(), 2, "a count that differs");
    }
}
