//! The workers of a loaded program and the bodies they run.
//!
//! A worker is an identity, `worker.id`, with its own state: one list of members for
//! each program of the module. A body in progress is a `Frame`: plain data (the body, a
//! position, the values, the `var` slots, the open loops), so a body can stop at a
//! meeting and go on later without a thread of its own. A worker's call stack in one
//! invocation is a `Thread` of frames; `run` steps it until it returns or faults.
//!
//! One body at a time on a worker, to its end: `run` takes the worker's state for the
//! whole of a step, and nothing else writes it meanwhile.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::memory::{
    Block, Failure, Fault, Format, Heap, Member, Races, RecordType, Reduced, Scalar, Sink, Steps,
    Stream, TensorId, TensorPlace, Tile, View, Zip,
};
use super::ops::{self, Bin, Cmp, Element, Operand, ReducedOperand, Unary};
use crate::program::body::{Body, Inst, InstId, Op, ValueId};
use crate::program::{BodyRef, MemberKind, Module};
use crate::types::{
    Axis, BinOp, CmpOp, Combiner, ConstValue, Count, Entry, FuncId, MemberId, MessageId, Peers,
    ProgramId, ReduceOp, SplitBy, Type, UnaryOp,
};

/// A value a body computes with, as the interpreter holds it.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A scalar, or a boolean.
    Scalar(Scalar),
    /// A tile.
    Tile(Tile),
    /// A value for each row or column.
    Reduced(Reduced),
    /// A block of tiles.
    Block(Block),
    /// A tensor, by its place on the heap.
    Tensor(TensorId),
    /// A view of a tensor.
    View(View),
    /// A stream over a view.
    Stream(Stream),
    /// Two streams walked together.
    Zip(Zip),
    /// A write stream over a view.
    Sink(Sink),
    /// The scalars of one tile: `v.words()`.
    Words(Vec<Scalar>),
    /// Records read from one tile, each its fields in order.
    Records(Vec<Vec<Scalar>>),
    /// One record: its fields in order.
    Record(Vec<Scalar>),
    /// The name of a failure.
    Reason(Failure),
    /// `group.split(...)`: the group in contiguous sub-groups.
    SubGroups(Split),
    /// `group.leaders(...)`: the leaders of those sub-groups.
    Leaders(Split),
    /// What a collective gives a rank it does not deliver to (reading R34, Q1): any
    /// use of it is an invalid program.
    Absent,
    /// A message of a dynamic run: its type and its fields, in the type's order.
    Message(MessageId, Vec<Scalar>),
    /// `group(first, count)` in a control program: the first worker and the count.
    Group(usize, usize),
    /// What `invoke` yields in a control program: the invocation's number in the order
    /// the program made them.
    Handle(usize),
    /// `outcome(h)`: how an invocation ended.
    Outcome(Outcome),
    /// `broadcast(along, sender)`: the line this worker is on, and whether it sends.
    Broadcast {
        /// The line's channel.
        line: LineKey,
        /// `b.is_sender`.
        sender: bool,
    },
}

/// How an invocation ended, as a control program reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    /// `o.failed`.
    pub failed: bool,
    /// `o.reason`: the failure's name, when it failed.
    pub reason: Option<Failure>,
    /// `o.value`: the result, when it completed with one.
    pub value: Option<Scalar>,
}

/// What a control program asks of the run; the frame waits for the answer
/// (`Thread::deliver`).
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// `invoke(...)`: an invocation to make; the answer is its handle.
    Invoke {
        /// What is invoked.
        entry: Entry,
        /// The arguments: tensors and scalars.
        args: Vec<Value>,
        /// The first worker and the count.
        group: (usize, usize),
        /// `after`: the handles that must have completed.
        after: Vec<usize>,
        /// `after_end`: the handles that must have ended.
        after_end: Vec<usize>,
    },
    /// `await h`: the result; a failed `h` fails the control program.
    Await(usize),
    /// `outcome(h)`: how `h` ended.
    Outcome(usize),
    /// A collective over the copies of a replicated tensor (its copy 0, as passed).
    Copies(TensorId, OverCopies),
}

/// A broadcast's line: which `broadcast` the frame made (the k-th it ran), along
/// which axis (0 for a row, 1 for a column), and which row or column.
pub type LineKey = (u32, u8, usize);

/// What the sender of one line has cast: each tile and its word, in order, and
/// whether its body has left the block that made the broadcast.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Line {
    /// The tiles cast, each with its word.
    pub items: Vec<(Tile, Scalar, Vec<u64>)>,
    /// The sender left the block: `arrive` ends after the last item.
    pub closed: bool,
}

/// The lines of one invocation's broadcasts.
pub type Lines = BTreeMap<LineKey, Line>;

/// How a group is split: into `n` parts, or into parts of `n` workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    /// `parts = n` (else `size = n`).
    pub parts: bool,
    /// The n.
    pub n: usize,
}

impl Split {
    /// The sub-groups of a group of `size` ranks, each a range of ranks, in order.
    /// `parts = n` follows `partition`'s law: the first `size mod n` parts have one
    /// rank more. `size = k` gives parts of k ranks, the last one shorter.
    pub fn ranges(self, size: usize) -> Result<Vec<std::ops::Range<usize>>, Fault> {
        if self.n == 0 {
            return Err(Failure::OutOfRange.into());
        }
        let mut out = Vec::new();
        if self.parts {
            let (each, extra) = (size / self.n, size % self.n);
            let mut start = 0;
            for k in 0..self.n {
                let len = each + usize::from(k < extra);
                if len > 0 {
                    out.push(start..start + len);
                }
                start += len;
            }
        } else {
            let mut start = 0;
            while start < size {
                let end = start.saturating_add(self.n).min(size);
                out.push(start..end);
                start = end;
            }
        }
        Ok(out)
    }
}

/// A collective a frame has reached, with this rank's operand.
#[derive(Clone, Debug, PartialEq)]
pub enum Collective {
    /// `reduce(x, with = f, within = g, among = g)`: the combination at the leader of
    /// the members.
    Reduce {
        /// This rank's operand.
        x: Value,
        /// What combines.
        with: Combiner,
        /// `within`: the sub-groups.
        within: Option<Split>,
        /// `among`: the sub-groups whose leaders are the members.
        among: Option<Split>,
    },
    /// `all_reduce(x, with = f)`: the combination at every member.
    AllReduce {
        /// This rank's operand.
        x: Value,
        /// What combines.
        with: Combiner,
    },
    /// `gather(x)`: every member's value at the leader, by rank.
    Gather {
        /// This rank's operand.
        x: Value,
    },
    /// `sync(publish = [...])`: a point every member reaches.
    Sync,
}

/// One worker: its position among the loaded program's workers, and its state.
#[derive(Clone, Debug, PartialEq)]
pub struct Worker {
    /// `worker.id`.
    pub id: usize,
    /// The state members of each program of the module, by `ProgramId`, each in its
    /// program's order.
    pub state: Vec<Vec<Member>>,
}

impl Worker {
    /// A worker whose every member is empty: what loading the module gives.
    pub fn load(id: usize, module: &Module, params: &[Option<Scalar>]) -> Result<Worker, Fault> {
        let mut state = Vec::new();
        for program in &module.programs {
            let mut members = Vec::new();
            for m in &program.state {
                let n = count(m.count, params)?;
                members.push(match &m.kind {
                    MemberKind::Tiles(f) => {
                        Member::tiles(&m.name, *f, module.tile, n, m.by_element)
                    }
                    MemberKind::Scalars(s) => Member::scalars(&m.name, s.format(), n, m.by_element)
                        .ok_or_else(|| Fault::undefined(format!("a member of {}", s.format())))?,
                });
            }
            state.push(members);
        }
        Ok(Worker { id, state })
    }
}

/// A count of a declaration: a constant, or a module parameter's value.
pub fn count(c: Count, params: &[Option<Scalar>]) -> Result<usize, Fault> {
    let n = match c {
        Count::Const(n) => n,
        Count::Param(p) => match params.get(p.0 as usize).copied().flatten() {
            Some(Scalar::Int(_, bits)) => bits,
            _ => return Err(Fault::undefined(format!("parameter {} has no value", p.0))),
        },
        Count::GroupSize => return Err(Fault::undefined("a declaration counted by a group")),
    };
    usize::try_from(n).map_err(|_| Failure::Capacity.into())
}

/// Where a body runs in its invocation: its rank, the group's size, and the grid's
/// shape on `@on(grid[R, C])`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Place {
    /// `worker.rank`.
    pub rank: usize,
    /// `group.size`.
    pub size: usize,
    /// The first worker's id: `group.id(r)` is `first + r`.
    pub first: usize,
    /// Rows and columns of the grid, if the group is one.
    pub grid: Option<(usize, usize)>,
}

/// The structure of a body's control flow, worked out once: for each opening marker
/// its `End`, for each `If` its `Else`, and for each `End` its opening.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Shape {
    end_of: Vec<Option<usize>>,
    else_of: Vec<Option<usize>>,
    opener_of: Vec<Option<usize>>,
    /// For each instruction, the `Else` or `End` that closes the block it stands in;
    /// none at the body's top.
    closer_of: Vec<Option<usize>>,
}

impl Shape {
    /// The shape of `body`; a body whose markers do not balance is not defined.
    pub fn of(body: &Body) -> Result<Shape, Fault> {
        let n = body.insts.len();
        let mut shape = Shape {
            end_of: vec![None; n],
            else_of: vec![None; n],
            opener_of: vec![None; n],
            closer_of: vec![None; n],
        };
        let mut open: Vec<usize> = Vec::new();
        for (i, inst) in body.insts.iter().enumerate() {
            match inst.op {
                Op::If { .. }
                | Op::While
                | Op::ForRange { .. }
                | Op::ForStream { .. }
                | Op::ForZip { .. }
                | Op::ForArrive { .. } => open.push(i),
                Op::Else => match open.last() {
                    Some(&o) => shape.else_of[o] = Some(i),
                    None => return Err(Fault::undefined("an `else` with no `if`")),
                },
                Op::End => match open.pop() {
                    Some(o) => {
                        shape.end_of[o] = Some(i);
                        shape.opener_of[i] = Some(o);
                    }
                    None => return Err(Fault::undefined("an `end` with nothing open")),
                },
                _ => {}
            }
        }
        if !open.is_empty() {
            return Err(Fault::undefined("a construct with no `end`"));
        }
        // the innermost block of each instruction: a `then` part closes at its `else`
        let mut stack: Vec<usize> = Vec::new();
        for i in 0..n {
            if shape.opener_of.get(i).copied().flatten().is_some() {
                stack.pop();
            }
            let closer = stack.last().and_then(|&o| {
                let end = shape.end_of.get(o).copied().flatten();
                match shape.else_of.get(o).copied().flatten() {
                    Some(e) if i < e => Some(e),
                    _ => end,
                }
            });
            if let Some(slot) = shape.closer_of.get_mut(i) {
                *slot = closer;
            }
            if shape.end_of.get(i).copied().flatten().is_some() {
                stack.push(i);
            }
        }
        Ok(shape)
    }
}

/// The shapes of every body of a module, worked out at load.
pub type Shapes = HashMap<BodyRef, Shape>;

/// Every body of a module, with its place.
pub fn bodies(module: &Module) -> Vec<(BodyRef, &Body)> {
    let mut out = Vec::new();
    for (f, func) in module.funcs.iter().enumerate() {
        if let Some(b) = &func.body {
            out.push((BodyRef::Func(crate::types::FuncId(f as u32)), b));
        }
    }
    for (p, program) in module.programs.iter().enumerate() {
        let pid = ProgramId(p as u32);
        for (h, handler) in program.handlers.iter().enumerate() {
            let hid = crate::types::HandlerId(h as u32);
            out.push((BodyRef::Handler(pid, hid), &handler.body));
        }
        for (r, run) in program.runs.iter().enumerate() {
            let rid = crate::types::RunId(r as u32);
            out.push((BodyRef::Seed(pid, rid), &run.seed));
            for (k, (_, b)) in run.on_message.iter().enumerate() {
                out.push((BodyRef::On(pid, rid, k as u32), b));
            }
        }
    }
    for (c, control) in module.controls.iter().enumerate() {
        out.push((
            BodyRef::Control(crate::types::ControlId(c as u32)),
            &control.body,
        ));
    }
    out
}

/// The body at a place of the module.
pub fn body_at(module: &Module, at: BodyRef) -> Option<&Body> {
    match at {
        BodyRef::Func(f) => module.funcs.get(f.0 as usize)?.body.as_ref(),
        BodyRef::Handler(p, h) => Some(
            &module
                .programs
                .get(p.0 as usize)?
                .handlers
                .get(h.0 as usize)?
                .body,
        ),
        BodyRef::Seed(p, r) => Some(
            &module
                .programs
                .get(p.0 as usize)?
                .runs
                .get(r.0 as usize)?
                .seed,
        ),
        BodyRef::On(p, r, k) => Some(
            &module
                .programs
                .get(p.0 as usize)?
                .runs
                .get(r.0 as usize)?
                .on_message
                .get(k as usize)?
                .1,
        ),
        BodyRef::Control(c) => Some(&module.controls.get(c.0 as usize)?.body),
    }
}

/// What a body runs against: the module, its shapes, the module parameters' values,
/// and the heap.
pub struct Context<'a> {
    /// The module.
    pub module: &'a Module,
    /// The shapes of its bodies.
    pub shapes: &'a Shapes,
    /// Each module parameter's value, by `ParamId` (`None` before it is bound).
    pub params: &'a [Option<Scalar>],
    /// The tensors.
    pub heap: &'a mut Heap,
    /// The broadcasts' lines of the invocation.
    pub lines: &'a mut Lines,
    /// The tiles the invocation read, and those it opened for writing.
    pub accesses: &'a mut Accesses,
    /// A dynamic run's messages in flight.
    pub mail: &'a mut Mail,
    /// The record that finds races.
    pub races: &'a mut Races,
    /// The invocation running, by its number in the record.
    pub inv: usize,
    /// The islands of the run.
    pub islands: &'a IslandMap,
}

/// The islands of a run as the queries see them: the island holding each worker, and
/// the hops of a shortest path between every two islands.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IslandMap {
    /// The island of each worker, by id.
    pub of: Vec<usize>,
    /// `hops[a][b]`: the length of a shortest path from island a to island b.
    pub hops: Vec<Vec<u64>>,
}

impl IslandMap {
    /// One island holding `workers` workers.
    pub fn one(workers: usize) -> IslandMap {
        IslandMap {
            of: vec![0; workers],
            hops: vec![vec![0]],
        }
    }

    /// How many islands.
    pub fn count(&self) -> usize {
        self.hops.len()
    }

    /// The island holding worker `id`; past the loaded workers, an invalid program.
    pub fn island(&self, id: usize) -> Result<usize, Fault> {
        self.of
            .get(id)
            .copied()
            .ok_or_else(|| Fault::undefined(format!("worker {id} is not one of the run's")))
    }
}

/// A collective over the copies of a replicated tensor, made by a control program.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverCopies {
    /// `reduce_to(t, to = i, with = f)`.
    ReduceTo {
        /// The island whose copy takes the fold.
        to: usize,
        /// The combiner.
        with: Combiner,
    },
    /// `broadcast_from(t, from = i)`.
    BroadcastFrom {
        /// The island whose copy every copy becomes.
        from: usize,
    },
    /// `all_reduce(t, with = f)`.
    AllReduce {
        /// The combiner.
        with: Combiner,
    },
}

impl OverCopies {
    /// Its name in the language.
    pub fn name(&self) -> &'static str {
        match self {
            OverCopies::ReduceTo { .. } => "reduce_to",
            OverCopies::BroadcastFrom { .. } => "broadcast_from",
            OverCopies::AllReduce { .. } => "all_reduce",
        }
    }
}

/// A message on its way: from which rank, its number among that rank's sends, and to
/// which rank.
#[derive(Clone, Debug, PartialEq)]
pub struct Letter {
    /// The sender's rank.
    pub from: usize,
    /// How many messages the sender had sent before it.
    pub seq: u64,
    /// The receiver's rank.
    pub to: usize,
    /// The message.
    pub message: Value,
    /// The sender's clock when it sent it.
    pub clock: Vec<u64>,
    /// The places it publishes, each with what the sender saw of its writes.
    pub publications: Vec<((usize, usize), Vec<u64>)>,
}

/// A dynamic run's delivery state: what was sent this round, the offer slots, the
/// live count, and how the body that runs handles its message.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mail {
    /// Sent or forwarded this round, in the order they were.
    pub outbox: Vec<Letter>,
    /// The offer slots, by (sender, receiver).
    pub slots: BTreeMap<(usize, usize), Letter>,
    /// Messages sent or offered and not yet consumed.
    pub live: u64,
    /// The bound on new work: a `send` with `live` at it is refused. None: no bound.
    pub bound: Option<u64>,
    /// Each rank's count of messages sent, forwarded or offered.
    pub sent: Vec<u64>,
    /// How many times the running body consumed or forwarded its message.
    pub handled: u32,
    /// The run's `@peers`.
    pub peers: Peers,
}

impl Mail {
    /// The delivery state of a run on `size` ranks.
    pub fn new(size: usize, peers: Peers, bound: Option<u64>) -> Mail {
        Mail {
            sent: vec![0; size],
            peers,
            bound,
            ..Mail::default()
        }
    }

    fn letter(
        &mut self,
        from: usize,
        to: usize,
        message: Value,
        races: &mut Races,
        publications: Vec<((usize, usize), Vec<u64>)>,
    ) -> Letter {
        let seq = self.sent.get(from).copied().unwrap_or(0);
        if let Some(n) = self.sent.get_mut(from) {
            *n += 1;
        }
        Letter {
            from,
            seq,
            to,
            message,
            clock: races.release(from),
            publications,
        }
    }
}

/// The tiles an invocation touched, each as (rank, tensor, tile): those a rank read,
/// and those inside a view it opened for writing (a sink's view, a store's target, a
/// put's items), whether or not each was written.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Accesses {
    /// The tiles read.
    pub read: BTreeSet<(usize, usize, usize)>,
    /// The tiles of views opened for writing.
    pub opened: BTreeSet<(usize, usize, usize)>,
}

impl Accesses {
    fn note(&mut self, opened: bool, rank: usize, tensor: TensorId, tiles: Vec<usize>) {
        let set = if opened {
            &mut self.opened
        } else {
            &mut self.read
        };
        set.extend(tiles.into_iter().map(|t| (rank, tensor.0, t)));
    }
}

/// The tiles the views of a `publish` list name, as (tensor, tile).
fn published(
    cx: &Context<'_>,
    frame: &Frame,
    list: &[ValueId],
) -> Result<Vec<(usize, usize)>, Fault> {
    let mut out = Vec::new();
    for v in list {
        let v = view(frame, *v)?;
        out.extend(
            ops::view_tiles(cx.heap, v)?
                .into_iter()
                .map(|n| (v.tensor.0, n)),
        );
    }
    Ok(out)
}

/// Accesses of `rank` to places, through the race record: a race is not a program.
fn touch(
    cx: &mut Context<'_>,
    rank: usize,
    at: (BodyRef, InstId),
    places: Vec<TensorPlace>,
    write: bool,
) -> Result<(), Fault> {
    for p in places {
        cx.races
            .access(cx.inv, rank, p, write, at)
            .map_err(Fault::Undefined)?;
    }
    Ok(())
}

/// The whole tiles of a view, as places; none for a view past its tensor, which
/// faults where it is used.
fn tile_places(cx: &Context<'_>, v: &View) -> Vec<TensorPlace> {
    ops::view_tiles(cx.heap, v)
        .map(|tiles| {
            tiles
                .into_iter()
                .map(|n| TensorPlace::tile(v.tensor, n))
                .collect()
        })
        .unwrap_or_default()
}

/// Notes the tiles of a view `rank` read (`opened` false) or opened for writing. A
/// view that names no tile of its tensor faults where it is used, not here.
fn note_view(cx: &mut Context<'_>, rank: usize, v: &View, opened: bool) {
    if let Ok(tiles) = ops::view_tiles(cx.heap, v) {
        cx.accesses.note(opened, rank, v.tensor, tiles);
    }
}

/// Notes, and checks for races, the tiles the next block of a stream reads, at
/// `rank`.
fn read_next(
    cx: &mut Context<'_>,
    rank: usize,
    at: (BodyRef, InstId),
    s: &Stream,
) -> Result<(), Fault> {
    let Ok(tiles) = ops::next_tiles(s) else {
        return Ok(());
    };
    let places = tiles
        .iter()
        .map(|n| TensorPlace::tile(s.view.tensor, *n))
        .collect();
    cx.accesses.note(false, rank, s.view.tensor, tiles);
    touch(cx, rank, at, places, false)
}

/// An open loop of a frame.
#[derive(Clone, Debug, PartialEq)]
enum Loop {
    /// `for i in range(n)`: the next value and the bound, of a format.
    Range {
        opener: usize,
        next: u64,
        n: u64,
        format: Format,
    },
    /// `for b in stream(...)`.
    Stream { opener: usize, stream: Stream },
    /// `for a, b in zip(...)`.
    Zip { opener: usize, zip: Zip },
    /// `for t, w in b.arrive()`: the line, and the next item to take.
    Arrive {
        opener: usize,
        line: LineKey,
        next: usize,
    },
}

/// A body in progress.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    at: BodyRef,
    pc: usize,
    args: Vec<Value>,
    values: Vec<Option<Value>>,
    locals: Vec<Option<Value>>,
    loops: Vec<Loop>,
    /// Where the caller takes this frame's result, when it is a function's.
    result_to: Option<ValueId>,
    /// How many `broadcast`s the frame has made.
    made: u32,
    /// The lines this frame sends on, each with the instruction that closes the block
    /// that made it (none: the body's end).
    sending: Vec<(Option<usize>, LineKey)>,
    /// The message that arrived, in an `on M(m)` body.
    arrived: Option<Value>,
    /// A call of a function on blocks: its body runs on each tile in turn.
    per_tile: Option<Box<PerTile>>,
    /// A call of a function on blocks whose body reduces along a row (R43) or
    /// multiplies blocks (R45): the body runs once on the whole blocks, each
    /// elementwise operation tile by tile.
    whole: bool,
}

/// A call of a function on blocks under way (`Func::on_blocks`): the arguments as
/// given, the tile the body runs on now, and the result's tiles so far.
#[derive(Clone, Debug, PartialEq)]
struct PerTile {
    func: FuncId,
    args: Vec<Value>,
    at: usize,
    count: usize,
    /// The rows of tiles of the blocks given: the result's.
    rows: usize,
    done: Vec<Tile>,
}

impl PerTile {
    /// The arguments of the body on tile `at`: a block's tile `at`, any other as given.
    fn args(&self) -> Vec<Value> {
        self.args
            .iter()
            .map(|a| match a {
                Value::Block(b) => Value::Tile(b.tiles[self.at].clone()),
                other => other.clone(),
            })
            .collect()
    }
}

impl Frame {
    /// The frame, given the message that arrived for its `on M(m)` body.
    pub fn with_arrived(mut self, message: Value) -> Frame {
        self.arrived = Some(message);
        self
    }

    /// A frame at the start of the body `at`, given its arguments.
    pub fn new(module: &Module, at: BodyRef, args: Vec<Value>) -> Result<Frame, Fault> {
        let body =
            body_at(module, at).ok_or_else(|| Fault::undefined(format!("no body {at:?}")))?;
        Ok(Frame {
            at,
            pc: 0,
            args,
            values: vec![None; body.values.len()],
            locals: vec![None; body.locals.len()],
            loops: Vec::new(),
            result_to: None,
            made: 0,
            sending: Vec::new(),
            arrived: None,
            per_tile: None,
            whole: false,
        })
    }
}

/// A worker's call stack in one invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct Thread {
    frames: Vec<Frame>,
    /// Instructions run so far.
    pub steps: u64,
}

/// How a thread stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum Stop {
    /// The outermost body returned, with its value if it gave one.
    Returned(Option<Value>),
    /// An instruction faulted: the fault, and where.
    Faulted {
        /// The fault.
        fault: Fault,
        /// The body and the instruction.
        at: (BodyRef, InstId),
    },
    /// A control program asks something of the run, and waits for the answer.
    Control {
        /// What it asks.
        request: Request,
        /// The body and the instruction.
        at: (BodyRef, InstId),
    },
    /// The frame waits at `arrive` for a cast that has not come; run it again later.
    Waiting {
        /// The body and the instruction.
        at: (BodyRef, InstId),
    },
    /// The frame reached a collective and waits for its value (`deliver`).
    Meeting {
        /// The collective, with this rank's operand.
        collective: Collective,
        /// The tiles it publishes, as (tensor, tile).
        publish: Vec<(usize, usize)>,
        /// The body and the instruction.
        at: (BodyRef, InstId),
    },
}

impl Thread {
    /// The instructions run so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// A thread that runs `frame` as its outermost body.
    pub fn new(frame: Frame) -> Thread {
        Thread {
            frames: vec![frame],
            steps: 0,
        }
    }

    /// Runs until the outermost body returns or an instruction faults. `budget` is the
    /// count of instructions the thread may run in all; past it, the failure
    /// `given_up`.
    pub fn run(
        &mut self,
        worker: &mut Worker,
        place: Place,
        cx: &mut Context<'_>,
        budget: u64,
    ) -> Stop {
        loop {
            let Some(frame) = self.frames.last_mut() else {
                return Stop::Returned(None);
            };
            let at = frame.at;
            let pc = frame.pc;
            if self.steps >= budget {
                return Stop::Faulted {
                    fault: Failure::GivenUp.into(),
                    at: (at, InstId(pc as u32)),
                };
            }
            self.steps += 1;
            match step(frame, worker, place, cx) {
                Ok(Next::Go) => {}
                Ok(Next::Wait) => {
                    // waiting is no step
                    self.steps -= 1;
                    return Stop::Waiting {
                        at: (at, InstId(pc as u32)),
                    };
                }
                Ok(Next::Ask(request)) => {
                    return Stop::Control {
                        request,
                        at: (at, InstId(pc as u32)),
                    }
                }
                Ok(Next::Meet(collective, publish)) => {
                    return Stop::Meeting {
                        collective,
                        publish,
                        at: (at, InstId(pc as u32)),
                    }
                }
                Ok(Next::Call(callee)) => self.frames.push(*callee),
                Ok(Next::Return(value)) => {
                    let mut done = self.frames.pop();
                    // a function on blocks: the tile's result kept, the next tile run
                    let mut value = value;
                    if let Some(mut state) = done.as_mut().and_then(|f| f.per_tile.take()) {
                        let to = done.as_ref().and_then(|f| f.result_to);
                        let Some(Value::Tile(tile)) = value else {
                            return Stop::Faulted {
                                fault: Fault::undefined(
                                    "a function on blocks that returned no tile",
                                ),
                                at: (at, InstId(pc as u32)),
                            };
                        };
                        state.done.push(tile);
                        state.at += 1;
                        if state.at < state.count {
                            let next =
                                Frame::new(cx.module, BodyRef::Func(state.func), state.args());
                            match next {
                                Ok(mut next) => {
                                    next.result_to = to;
                                    next.per_tile = Some(state);
                                    self.frames.push(next);
                                    continue;
                                }
                                Err(fault) => {
                                    return Stop::Faulted {
                                        fault,
                                        at: (at, InstId(pc as u32)),
                                    }
                                }
                            }
                        }
                        value = Some(Value::Block(Block {
                            tiles: state.done,
                            rows: state.rows,
                        }));
                    }
                    let Some(caller) = self.frames.last_mut() else {
                        return Stop::Returned(value);
                    };
                    let to = done.and_then(|f| f.result_to);
                    match (to, value) {
                        (Some(to), Some(v)) => {
                            if let Some(slot) = caller.values.get_mut(to.0 as usize) {
                                *slot = Some(v);
                            }
                        }
                        _ => {
                            return Stop::Faulted {
                                fault: Fault::undefined("a function that returned no value"),
                                at: (caller.at, InstId(caller.pc.saturating_sub(1) as u32)),
                            }
                        }
                    }
                }
                Err(fault) => {
                    return Stop::Faulted {
                        fault,
                        at: (at, InstId(pc as u32)),
                    }
                }
            }
        }
    }
}

impl Thread {
    /// Gives the frame that stopped at a collective the value it arrives with (none
    /// for a `sync`), so it can go on.
    pub fn deliver(&mut self, module: &Module, value: Option<Value>) -> Result<(), Fault> {
        let frame = self
            .frames
            .last_mut()
            .ok_or_else(|| Fault::undefined("a delivery to a thread that ended"))?;
        let body = body_at(module, frame.at).ok_or_else(|| Fault::undefined("no body"))?;
        let inst = frame
            .pc
            .checked_sub(1)
            .and_then(|i| body.insts.get(i))
            .ok_or_else(|| Fault::undefined("a delivery with no collective"))?;
        match (inst.defs.first(), value) {
            (Some(d), Some(v)) => {
                if let Some(slot) = frame.values.get_mut(d.0 as usize) {
                    *slot = Some(v);
                }
                Ok(())
            }
            (None, None) => Ok(()),
            _ => Err(Fault::undefined("a collective's value where none is taken")),
        }
    }
}

/// What a step leads to.
enum Next {
    Go,
    Call(Box<Frame>),
    Return(Option<Value>),
    Meet(Collective, Vec<(usize, usize)>),
    Ask(Request),
    Wait,
}

fn bin(op: BinOp) -> Bin {
    match op {
        BinOp::Add => Bin::Add,
        BinOp::Sub => Bin::Sub,
        BinOp::Mul => Bin::Mul,
        BinOp::Div => Bin::Div,
        BinOp::Rem => Bin::Rem,
        BinOp::And => Bin::And,
        BinOp::Or => Bin::Or,
        BinOp::Xor => Bin::Xor,
        BinOp::Shl => Bin::Shl,
        BinOp::Shr => Bin::Shr,
        BinOp::Min => Bin::Min,
        BinOp::Max => Bin::Max,
    }
}

fn cmp(op: CmpOp) -> Cmp {
    match op {
        CmpOp::Gt => Cmp::Gt,
        CmpOp::Lt => Cmp::Lt,
        CmpOp::Ge => Cmp::Ge,
        CmpOp::Le => Cmp::Le,
        CmpOp::Eq => Cmp::Eq,
        CmpOp::Ne => Cmp::Ne,
    }
}

fn unary(op: UnaryOp) -> Unary {
    match op {
        UnaryOp::Neg => Unary::Neg,
        UnaryOp::Abs => Unary::Abs,
        UnaryOp::Relu => Unary::Relu,
        UnaryOp::Square => Unary::Square,
        UnaryOp::Floor => Unary::Floor,
        UnaryOp::Exp => Unary::Exp,
        UnaryOp::Exp2 => Unary::Exp2,
        UnaryOp::Tanh => Unary::Tanh,
        UnaryOp::Erf => Unary::Erf,
        UnaryOp::Log => Unary::Log,
        UnaryOp::Sigmoid => Unary::Sigmoid,
        UnaryOp::Silu => Unary::Silu,
        UnaryOp::Sqrt => Unary::Sqrt,
        UnaryOp::Rsqrt => Unary::Rsqrt,
        UnaryOp::Reciprocal => Unary::Reciprocal,
    }
}

/// A constant as a scalar.
pub fn constant(c: ConstValue) -> Result<Scalar, Fault> {
    match c {
        ConstValue::Bool(b) => Ok(Scalar::Bool(b)),
        ConstValue::Int(s, bits) => Ok(Scalar::Int(s.format(), bits)),
        ConstValue::Float(s, v) => Scalar::float(s.format(), v)
            .ok_or_else(|| Fault::undefined(format!("a float constant of {}", s.name()))),
    }
}

fn get(frame: &Frame, v: ValueId) -> Result<&Value, Fault> {
    match frame.values.get(v.0 as usize).and_then(Option::as_ref) {
        Some(Value::Absent) => Err(Fault::undefined(format!(
            "v{}, a collective's value this rank was not given, is used",
            v.0
        ))),
        Some(x) => Ok(x),
        None => Err(Fault::undefined(format!(
            "v{} read before it is defined",
            v.0
        ))),
    }
}

/// A collective's operand: the value as it stands, an absent one included. A rank
/// that is not a member passes the collective and its operand is never read; a
/// member's absent operand is refused where the meeting completes.
fn operand_of(frame: &Frame, v: ValueId) -> Result<Value, Fault> {
    frame
        .values
        .get(v.0 as usize)
        .cloned()
        .flatten()
        .ok_or_else(|| Fault::undefined(format!("v{} read before it is defined", v.0)))
}

fn message_of(frame: &Frame, v: ValueId) -> Result<Value, Fault> {
    match get(frame, v)? {
        m @ Value::Message(..) => Ok(m.clone()),
        other => Err(Fault::undefined(format!(
            "{} where a message is read",
            kind(other)
        ))),
    }
}

/// A receiver's rank: one of the group, else `out_of_range`.
fn rank_to(frame: &Frame, v: ValueId, place: Place) -> Result<usize, Fault> {
    let to = index(frame, v)?;
    if to >= place.size {
        return Err(Failure::OutOfRange.into());
    }
    Ok(to)
}

fn handle(frame: &Frame, v: ValueId) -> Result<usize, Fault> {
    match get(frame, v)? {
        Value::Handle(h) => Ok(*h),
        other => Err(Fault::undefined(format!(
            "{} where a handle is read",
            kind(other)
        ))),
    }
}

fn handles(frame: &Frame, vs: &[ValueId]) -> Result<Vec<usize>, Fault> {
    vs.iter().map(|v| handle(frame, *v)).collect()
}

fn outcome(frame: &Frame, v: ValueId) -> Result<Outcome, Fault> {
    match get(frame, v)? {
        Value::Outcome(o) => Ok(*o),
        other => Err(Fault::undefined(format!(
            "{} where an outcome is read",
            kind(other)
        ))),
    }
}

/// Marks ended every line the frame sends on whose block closes at `at` (all of them
/// when the body ends: `None`).
fn close(frame: &mut Frame, lines: &mut Lines, at: Option<usize>) {
    frame.sending.retain(|(closer, key)| {
        let ends = at.is_none() || *closer == at;
        if ends {
            if let Some(line) = lines.get_mut(key) {
                line.closed = true;
            }
        }
        !ends
    });
}

fn split_of(frame: &Frame, v: Option<ValueId>, leaders: bool) -> Result<Option<Split>, Fault> {
    let Some(v) = v else {
        return Ok(None);
    };
    match (get(frame, v)?, leaders) {
        (Value::SubGroups(s), false) | (Value::Leaders(s), true) => Ok(Some(*s)),
        (other, _) => Err(Fault::undefined(format!(
            "{} where sub-groups are read",
            kind(other)
        ))),
    }
}

fn scalar(frame: &Frame, v: ValueId) -> Result<Scalar, Fault> {
    match get(frame, v)? {
        Value::Scalar(s) => Ok(*s),
        other => Err(Fault::undefined(format!(
            "{} where a scalar is read",
            kind(other)
        ))),
    }
}

fn truth(frame: &Frame, v: ValueId) -> Result<bool, Fault> {
    match scalar(frame, v)? {
        Scalar::Bool(b) => Ok(b),
        _ => Err(Fault::undefined("a number where a condition is read")),
    }
}

/// An integer scalar as an index or a count: a negative one, or one past the
/// machine's addresses, is `out_of_range`.
fn index(frame: &Frame, v: ValueId) -> Result<usize, Fault> {
    let s = scalar(frame, v)?;
    let n = s
        .as_i128()
        .ok_or_else(|| Fault::undefined("a float where an index is read"))?;
    usize::try_from(n).map_err(|_| Failure::OutOfRange.into())
}

fn view(frame: &Frame, v: ValueId) -> Result<&View, Fault> {
    match get(frame, v)? {
        Value::View(x) => Ok(x),
        other => Err(Fault::undefined(format!(
            "{} where a view is read",
            kind(other)
        ))),
    }
}

fn tensor(frame: &Frame, v: ValueId) -> Result<TensorId, Fault> {
    match get(frame, v)? {
        Value::Tensor(t) => Ok(*t),
        other => Err(Fault::undefined(format!(
            "{} where a tensor is read",
            kind(other)
        ))),
    }
}

fn stream(frame: &Frame, v: ValueId) -> Result<&Stream, Fault> {
    match get(frame, v)? {
        Value::Stream(s) => Ok(s),
        other => Err(Fault::undefined(format!(
            "{} where a stream is read",
            kind(other)
        ))),
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Scalar(_) => "a scalar",
        Value::Tile(_) => "a tile",
        Value::Reduced(_) => "a value for each row or column",
        Value::Block(_) => "a block",
        Value::Tensor(_) => "a tensor",
        Value::View(_) => "a view",
        Value::Stream(_) => "a stream",
        Value::Zip(_) => "a zip",
        Value::Sink(_) => "a sink",
        Value::Words(_) => "words",
        Value::Records(_) => "records",
        Value::Record(_) => "a record",
        Value::Reason(_) => "a failure's name",
        Value::SubGroups(_) => "sub-groups",
        Value::Leaders(_) => "the leaders of sub-groups",
        Value::Absent => "a value this rank was not given",
        Value::Broadcast { .. } => "a broadcast",
        Value::Group(..) => "a group",
        Value::Message(..) => "a message",
        Value::Handle(_) => "an invocation's handle",
        Value::Outcome(_) => "an outcome",
    }
}

fn operand(v: &Value) -> Result<Operand<'_>, Fault> {
    match v {
        Value::Tile(t) => Ok(Operand::Tile(t)),
        Value::Scalar(s) => Ok(Operand::Scalar(*s)),
        other => Err(Fault::undefined(format!(
            "{} as a tile operand",
            kind(other)
        ))),
    }
}

fn reduced_operand(v: &Value) -> Result<ReducedOperand<'_>, Fault> {
    match v {
        Value::Reduced(r) => Ok(ReducedOperand::Reduced(r)),
        Value::Scalar(s) => Ok(ReducedOperand::Scalar(*s)),
        other => Err(Fault::undefined(format!(
            "{} beside a reduced value",
            kind(other)
        ))),
    }
}

fn as_block(v: &Value) -> Result<Block, Fault> {
    match v {
        Value::Tile(t) => Ok(Block::row(vec![t.clone()])),
        Value::Block(b) => Ok(b.clone()),
        other => Err(Fault::undefined(format!(
            "{} where tiles are written",
            kind(other)
        ))),
    }
}

/// `a op b` on any two operands the language allows.
pub fn binary(op: BinOp, a: &Value, b: &Value) -> Result<Value, Fault> {
    if let Some(shape) = block_shape(&[a, b])? {
        return tile_by_tile(shape, |i| {
            binary(op, &tile_at(a, i), &tile_at(b, i))?.into_tile()
        });
    }
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => {
            ops::scalar_binary(bin(op), *x, *y).map(Value::Scalar)
        }
        (Value::Reduced(_), _) | (_, Value::Reduced(_)) => {
            ops::reduced_binary(bin(op), reduced_operand(a)?, reduced_operand(b)?)
                .map(Value::Reduced)
        }
        _ => ops::binary(bin(op), operand(a)?, operand(b)?).map(Value::Tile),
    }
}

/// The shape of the blocks among `values` (their tiles and rows), when any is a block:
/// in a function on blocks run whole (R43, R45), an elementwise operation then works
/// tile by tile, on blocks of one shape.
fn block_shape(values: &[&Value]) -> Result<Option<(usize, usize)>, Fault> {
    let mut shapes = values.iter().filter_map(|v| match v {
        Value::Block(b) => Some((b.tiles.len(), b.rows)),
        _ => None,
    });
    let Some(first) = shapes.next() else {
        return Ok(None);
    };
    if shapes.any(|s| s != first) {
        return Err(Fault::undefined(
            "an elementwise operation on blocks of two shapes",
        ));
    }
    Ok(Some(first))
}

/// Tile `i` of a block, any other value as it is (a scalar goes to every tile).
fn tile_at(v: &Value, i: usize) -> Value {
    match v {
        Value::Block(b) => b.tiles.get(i).cloned().map_or(Value::Absent, Value::Tile),
        other => other.clone(),
    }
}

/// The block of `n` tiles in `rows` rows `f` gives, one an index.
fn tile_by_tile(
    (n, rows): (usize, usize),
    f: impl Fn(usize) -> Result<Tile, Fault>,
) -> Result<Value, Fault> {
    let tiles = (0..n).map(f).collect::<Result<Vec<Tile>, Fault>>()?;
    Ok(Value::Block(Block { tiles, rows }))
}

impl Value {
    /// The tile this is, else a fault.
    fn into_tile(self) -> Result<Tile, Fault> {
        match self {
            Value::Tile(t) => Ok(t),
            other => Err(Fault::undefined(format!(
                "{} where a block's tile is made",
                kind(&other)
            ))),
        }
    }
}

/// Whether function `f`'s body reduces along a row (R43) or multiplies blocks (R45):
/// it runs on whole blocks.
fn runs_whole(module: &Module, f: FuncId) -> bool {
    let body = module.funcs.get(f.0 as usize).and_then(|x| x.body.as_ref());
    body.is_some_and(|b| {
        b.insts
            .iter()
            .any(|i| matches!(i.op, Op::Reduce(_, _, Axis::Row) | Op::Matmul(..)))
    })
}

/// The negation of a scalar: `0 - x`, wrapped on an integer.
fn negate(s: Scalar) -> Result<Scalar, Fault> {
    match s {
        Scalar::F32(f, v) => Ok(Scalar::F32(f, -v)),
        Scalar::F64(v) => Ok(Scalar::F64(-v)),
        Scalar::Int(f, _) => ops::scalar_binary(Bin::Sub, Scalar::Int(f, 0), s),
        Scalar::Bool(_) => Err(Fault::undefined("the negation of a boolean")),
    }
}

fn member(worker: &mut Worker, at: BodyRef, m: MemberId) -> Result<&mut Member, Fault> {
    let p = match at {
        BodyRef::Handler(p, _) | BodyRef::Seed(p, _) | BodyRef::On(p, _, _) => p,
        _ => return Err(Fault::undefined("state outside a program")),
    };
    worker
        .state
        .get_mut(p.0 as usize)
        .and_then(|s| s.get_mut(m.0 as usize))
        .ok_or_else(|| Fault::undefined(format!("no member {}", m.0)))
}

fn element_value(e: Element) -> Value {
    match e {
        Element::Tile(t) => Value::Tile(t),
        Element::Scalar(s) => Value::Scalar(s),
    }
}

fn element_of(v: &Value) -> Result<Element, Fault> {
    match v {
        Value::Tile(t) => Ok(Element::Tile(t.clone())),
        Value::Scalar(s) => Ok(Element::Scalar(*s)),
        other => Err(Fault::undefined(format!(
            "{} written to state",
            kind(other)
        ))),
    }
}

/// The value a loop gives for one block of a stream: a tile where the body reads one,
/// else the block.
fn block_value(mut block: Block, ty: Option<&Type>) -> Value {
    if matches!(ty, Some(Type::Tile(_))) && block.tiles.len() == 1 {
        if let Some(tile) = block.tiles.pop() {
            return Value::Tile(tile);
        }
    }
    Value::Block(block)
}

fn int_of(format: Format, n: u64) -> Scalar {
    Scalar::Int(format, n)
}

/// Runs one instruction of `frame`.
fn step(
    frame: &mut Frame,
    worker: &mut Worker,
    place: Place,
    cx: &mut Context<'_>,
) -> Result<Next, Fault> {
    let module = cx.module;
    let body = body_at(module, frame.at).ok_or_else(|| Fault::undefined("no body"))?;
    let shape = cx
        .shapes
        .get(&frame.at)
        .ok_or_else(|| Fault::undefined("a body with no shape"))?;
    let Some(inst) = body.insts.get(frame.pc) else {
        // falling off the end of a body returns nothing
        close(frame, cx.lines, None);
        return Ok(Next::Return(None));
    };
    let pc = frame.pc;
    let here = (frame.at, InstId(pc as u32));
    // leaving a block closes the lines made in it
    match inst.op {
        Op::Else | Op::End => close(frame, cx.lines, Some(pc)),
        Op::Return(_) => close(frame, cx.lines, None),
        _ => {}
    }
    frame.pc += 1;
    let Inst { op, defs } = inst;
    let def = |frame: &mut Frame, k: usize, v: Value| -> Result<(), Fault> {
        let d = defs
            .get(k)
            .ok_or_else(|| Fault::undefined("an operation that defines no value"))?;
        let slot = frame
            .values
            .get_mut(d.0 as usize)
            .ok_or_else(|| Fault::undefined("a value past the body's"))?;
        *slot = Some(v);
        Ok(())
    };
    let def_ty = |k: usize| defs.get(k).and_then(|d| body.values.get(d.0 as usize));
    let end_of = |i: usize| {
        shape
            .end_of
            .get(i)
            .copied()
            .flatten()
            .ok_or_else(|| Fault::undefined("a construct with no end"))
    };
    let value = match op {
        // ---- values ----
        Op::Const(c) => Value::Scalar(constant(*c)?),
        Op::Arg(i) => frame
            .args
            .get(*i as usize)
            .cloned()
            .ok_or_else(|| Fault::undefined(format!("no argument {i}")))?,
        Op::Param(p) => Value::Scalar(
            cx.params
                .get(p.0 as usize)
                .copied()
                .flatten()
                .ok_or_else(|| Fault::undefined(format!("parameter {} has no value", p.0)))?,
        ),
        Op::Get(l) => frame
            .locals
            .get(l.0 as usize)
            .cloned()
            .flatten()
            .ok_or_else(|| Fault::undefined(format!("l{} read before it is set", l.0)))?,
        Op::Set(l, v) => {
            let x = get(frame, *v)?.clone();
            let slot = frame
                .locals
                .get_mut(l.0 as usize)
                .ok_or_else(|| Fault::undefined("a slot past the body's"))?;
            *slot = Some(x);
            return Ok(Next::Go);
        }
        Op::Binary(o, a, b) => binary(*o, get(frame, *a)?, get(frame, *b)?)?,
        Op::Compare(c, a, b) => match (get(frame, *a)?, get(frame, *b)?) {
            // `o.reason == capacity`: failures' names compare equal or not
            (Value::Reason(x), Value::Reason(y)) => match c {
                CmpOp::Eq => Value::Scalar(Scalar::Bool(x == y)),
                CmpOp::Ne => Value::Scalar(Scalar::Bool(x != y)),
                _ => return Err(Fault::undefined("an order between failures' names")),
            },
            _ => Value::Scalar(Scalar::Bool(ops::compare(
                cmp(*c),
                scalar(frame, *a)?,
                scalar(frame, *b)?,
            )?)),
        },
        Op::LogicAnd(a, b) => Value::Scalar(Scalar::Bool(truth(frame, *a)? && truth(frame, *b)?)),
        Op::LogicOr(a, b) => Value::Scalar(Scalar::Bool(truth(frame, *a)? || truth(frame, *b)?)),
        Op::Convert(s, v) => Value::Scalar(ops::convert_scalar(&scalar(frame, *v)?, *s)?),
        // ---- control flow ----
        Op::If { cond } => {
            if !truth(frame, *cond)? {
                frame.pc = match shape.else_of.get(pc).copied().flatten() {
                    Some(e) => e + 1,
                    None => end_of(pc)? + 1,
                };
            }
            return Ok(Next::Go);
        }
        Op::Else => {
            // the `then` part ended: past the construct
            let opener = (0..pc)
                .rev()
                .find(|&o| shape.else_of.get(o).copied().flatten() == Some(pc))
                .ok_or_else(|| Fault::undefined("an `else` with no `if`"))?;
            frame.pc = end_of(opener)? + 1;
            return Ok(Next::Go);
        }
        Op::While => return Ok(Next::Go),
        Op::WhileTest { cond } => {
            if !truth(frame, *cond)? {
                let opener = (0..pc)
                    .rev()
                    .find(|&o| {
                        matches!(body.insts.get(o).map(|i| &i.op), Some(Op::While))
                            && shape
                                .end_of
                                .get(o)
                                .copied()
                                .flatten()
                                .is_some_and(|e| e > pc)
                    })
                    .ok_or_else(|| Fault::undefined("a test outside a `while`"))?;
                frame.pc = end_of(opener)? + 1;
            }
            return Ok(Next::Go);
        }
        Op::ForRange { n } => {
            let s = scalar(frame, *n)?;
            let Scalar::Int(format, _) = s else {
                return Err(Fault::undefined("`range` of what is not an integer"));
            };
            let bound = u64::try_from(s.as_i128().unwrap_or(0)).unwrap_or(0);
            if bound == 0 {
                frame.pc = end_of(pc)? + 1;
            } else {
                def(frame, 0, Value::Scalar(int_of(format, 0)))?;
                frame.loops.push(Loop::Range {
                    opener: pc,
                    next: 1,
                    n: bound,
                    format,
                });
            }
            return Ok(Next::Go);
        }
        Op::ForStream { stream: s } => {
            let mut st = stream(frame, *s)?.clone();
            read_next(cx, place.rank, here, &st)?;
            match ops::next_block(cx.heap, &mut st)? {
                None => frame.pc = end_of(pc)? + 1,
                Some(block) => {
                    def(frame, 0, block_value(block, def_ty(0)))?;
                    frame.loops.push(Loop::Stream {
                        opener: pc,
                        stream: st,
                    });
                }
            }
            return Ok(Next::Go);
        }
        Op::ForZip { a, b } => {
            let mut z = ops::zip(stream(frame, *a)?.clone(), stream(frame, *b)?.clone())?;
            read_next(cx, place.rank, here, &z.a)?;
            read_next(cx, place.rank, here, &z.b)?;
            match ops::next_pair(cx.heap, &mut z)? {
                None => frame.pc = end_of(pc)? + 1,
                Some((x, y)) => {
                    def(frame, 0, block_value(x, def_ty(0)))?;
                    def(frame, 1, block_value(y, def_ty(1)))?;
                    frame.loops.push(Loop::Zip { opener: pc, zip: z });
                }
            }
            return Ok(Next::Go);
        }
        Op::End => {
            let opener = shape
                .opener_of
                .get(pc)
                .copied()
                .flatten()
                .ok_or_else(|| Fault::undefined("an `end` with nothing open"))?;
            let open = body.insts.get(opener).map(|i| (&i.op, &i.defs));
            match open {
                Some((Op::While, _)) => frame.pc = opener + 1,
                // the opener takes the next item
                Some((Op::ForArrive { .. }, _)) => frame.pc = opener,
                Some((Op::ForRange { .. } | Op::ForStream { .. } | Op::ForZip { .. }, odefs)) => {
                    let odefs = odefs.clone();
                    let set = |frame: &mut Frame, k: usize, v: Value| -> Result<(), Fault> {
                        let d = odefs
                            .get(k)
                            .ok_or_else(|| Fault::undefined("a loop with no variable"))?;
                        if let Some(slot) = frame.values.get_mut(d.0 as usize) {
                            *slot = Some(v);
                        }
                        Ok(())
                    };
                    let ty = |k: usize| odefs.get(k).and_then(|d| body.values.get(d.0 as usize));
                    let more = match frame.loops.pop() {
                        Some(Loop::Range {
                            opener: o,
                            next,
                            n,
                            format,
                        }) if o == opener => {
                            if next < n {
                                set(frame, 0, Value::Scalar(int_of(format, next)))?;
                                frame.loops.push(Loop::Range {
                                    opener: o,
                                    next: next + 1,
                                    n,
                                    format,
                                });
                                true
                            } else {
                                false
                            }
                        }
                        Some(Loop::Stream {
                            opener: o,
                            mut stream,
                        }) if o == opener => {
                            read_next(cx, place.rank, here, &stream)?;
                            match ops::next_block(cx.heap, &mut stream)? {
                                Some(block) => {
                                    set(frame, 0, block_value(block, ty(0)))?;
                                    frame.loops.push(Loop::Stream { opener: o, stream });
                                    true
                                }
                                None => false,
                            }
                        }
                        Some(Loop::Zip { opener: o, mut zip }) if o == opener => {
                            read_next(cx, place.rank, here, &zip.a)?;
                            read_next(cx, place.rank, here, &zip.b)?;
                            match ops::next_pair(cx.heap, &mut zip)? {
                                Some((x, y)) => {
                                    set(frame, 0, block_value(x, ty(0)))?;
                                    set(frame, 1, block_value(y, ty(1)))?;
                                    frame.loops.push(Loop::Zip { opener: o, zip });
                                    true
                                }
                                None => false,
                            }
                        }
                        _ => return Err(Fault::undefined("a loop's end out of order")),
                    };
                    frame.pc = if more { opener + 1 } else { pc + 1 };
                }
                _ => {}
            }
            return Ok(Next::Go);
        }
        Op::Return(v) => {
            let value = match v {
                Some(v) => Some(get(frame, *v)?.clone()),
                None => None,
            };
            return Ok(Next::Return(value));
        }
        // ---- compute on tiles ----
        Op::Unary(u, x) => match get(frame, *x)? {
            Value::Tile(t) => Value::Tile(ops::unary(unary(*u), t)?),
            Value::Block(b) => tile_by_tile((b.tiles.len(), b.rows), |i| {
                ops::unary(unary(*u), &b.tiles[i])
            })?,
            Value::Reduced(r) => Value::Reduced(ops::reduced_unary(unary(*u), r)?),
            Value::Scalar(s) if *u == UnaryOp::Neg => Value::Scalar(negate(*s)?),
            other => return Err(Fault::undefined(format!("{u:?} of {}", kind(other)))),
        },
        Op::Select { cmp: c, a, b, x, y } => {
            let parts = [*a, *b, *x, *y].map(|v| get(frame, v));
            let [a, b, x, y] = parts;
            let (a, b, x, y) = (a?, b?, x?, y?);
            let select = |a: &Value, b: &Value, x: &Value, y: &Value| {
                ops::select(cmp(*c), operand(a)?, operand(b)?, operand(x)?, operand(y)?)
            };
            match block_shape(&[a, b, x, y])? {
                Some(shape) => tile_by_tile(shape, |i| {
                    select(
                        &tile_at(a, i),
                        &tile_at(b, i),
                        &tile_at(x, i),
                        &tile_at(y, i),
                    )
                })?,
                None => Value::Tile(select(a, b, x, y)?),
            }
        }
        Op::Reduce(r, x, axis) => {
            let kind = match r {
                ReduceOp::Sum => ops::Reduce::Sum,
                ReduceOp::Max => ops::Reduce::Max,
                ReduceOp::Min => ops::Reduce::Min,
            };
            match (get(frame, *x)?, axis) {
                (Value::Tile(t), _) => Value::Reduced(ops::reduce(kind, t, *axis)?),
                // along each row of the block's tiles (R43, R45)
                (Value::Block(b), Axis::Row) => Value::Reduced(ops::reduce_block(kind, b)?),
                _ => return Err(Fault::undefined("a reduction of what is not a tile")),
            }
        }
        Op::SpreadCol(x) => match get(frame, *x)? {
            // each row's value over every tile of its row of the block (R43, R45)
            Value::Reduced(r) => match r.cols {
                Some(cols) => Value::Block(ops::col_block(r, cols)?),
                None => Value::Tile(ops::col(r)?),
            },
            Value::Tile(t) => Value::Tile(ops::spread_col(t)?),
            other => return Err(Fault::undefined(format!(".col() of {}", kind(other)))),
        },
        Op::SpreadRow(x) => match get(frame, *x)? {
            Value::Reduced(r) => Value::Tile(ops::row(r)?),
            // each tile's row 0 down its own tile (R43)
            Value::Block(b) => {
                tile_by_tile((b.tiles.len(), b.rows), |i| ops::spread_row(&b.tiles[i]))?
            }
            Value::Tile(t) => Value::Tile(ops::spread_row(t)?),
            other => return Err(Fault::undefined(format!(".row() of {}", kind(other)))),
        },
        Op::Matmul(a, b) => match (get(frame, *a)?, get(frame, *b)?) {
            (Value::Tile(x), Value::Tile(y)) => Value::Tile(ops::matmul(x, y)?),
            // of blocks (R45), a tile one of one row of one tile: `[1, 1]` is a tile
            (x @ (Value::Tile(_) | Value::Block(_)), y @ (Value::Tile(_) | Value::Block(_))) => {
                let product = ops::matmul_blocks(&as_block(x)?, &as_block(y)?)?;
                block_value(product, def_ty(0))
            }
            _ => return Err(Fault::undefined("`matmul` of what is not two tiles")),
        },
        Op::Transpose(x) => match get(frame, *x)? {
            Value::Tile(t) => Value::Tile(ops::transpose(t)?),
            other => return Err(Fault::undefined(format!(".T of {}", kind(other)))),
        },
        Op::Zeros => match def_ty(0) {
            Some(Type::Tile(f)) => Value::Tile(ops::zeros(*f, module.tile)),
            _ => return Err(Fault::undefined("`zeros` of what is not a tile")),
        },
        Op::CallFunc(f, args) => {
            let func = module
                .funcs
                .get(f.0 as usize)
                .ok_or_else(|| Fault::undefined("no such function"))?;
            if func.body.is_none() {
                return Err(Fault::undefined(format!(
                    "`{}` has no fallback, and the interpreter has no implementation of it",
                    func.name
                )));
            }
            let mut values = Vec::with_capacity(args.len());
            for a in args {
                values.push(get(frame, *a)?.clone());
            }
            let blocks = values.iter().any(|v| matches!(v, Value::Block(_)));
            if func.on_blocks() && runs_whole(module, *f) {
                // a reduction along a row or a product: the body once, on the whole blocks
                let mut callee = Frame::new(module, BodyRef::Func(*f), values)?;
                callee.result_to = defs.first().copied();
                callee.whole = true;
                return Ok(Next::Call(Box::new(callee)));
            }
            if func.on_blocks() || blocks {
                // each tile of the blocks in turn (a one-tile function called on a
                // function on blocks' whole blocks, too); a shorter last block, a shorter
                // result
                let shapes: Vec<(usize, usize)> = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::Block(b) => Some((b.tiles.len(), b.rows)),
                        _ => None,
                    })
                    .collect();
                let (count, rows) = shapes.first().copied().unwrap_or((0, 1));
                if shapes.iter().any(|&s| s != (count, rows)) {
                    return Err(Fault::undefined(format!(
                        "`{}` given blocks of different shapes {shapes:?}",
                        func.name
                    )));
                }
                if count == 0 {
                    Value::Block(Block {
                        tiles: Vec::new(),
                        rows,
                    })
                } else {
                    let state = PerTile {
                        func: *f,
                        args: values,
                        at: 0,
                        count,
                        rows,
                        done: Vec::new(),
                    };
                    let mut callee = Frame::new(module, BodyRef::Func(*f), state.args())?;
                    callee.result_to = defs.first().copied();
                    callee.per_tile = Some(Box::new(state));
                    return Ok(Next::Call(Box::new(callee)));
                }
            } else {
                let mut callee = Frame::new(module, BodyRef::Func(*f), values)?;
                callee.result_to = defs.first().copied();
                return Ok(Next::Call(Box::new(callee)));
            }
        }
        // ---- views of a tensor ----
        Op::Tiles {
            tensor: t,
            first,
            n,
        } => Value::View(ops::tiles(
            cx.heap,
            tensor(frame, *t)?,
            index(frame, *first)?,
            index(frame, *n)?,
        )?),
        Op::Rect {
            tensor: t,
            rows,
            cols,
        } => {
            let steps = |frame: &Frame, r: &crate::program::body::Range| -> Result<Steps, Fault> {
                Ok(Steps {
                    start: index(frame, r.lo)?,
                    end: index(frame, r.hi)?,
                    step: match r.step {
                        Some(s) => index(frame, s)?,
                        None => 1,
                    },
                })
            };
            Value::View(ops::rect(
                cx.heap,
                tensor(frame, *t)?,
                steps(frame, rows)?,
                steps(frame, cols)?,
            )?)
        }
        Op::Turn(v) => Value::View(ops::turn(view(frame, *v)?)?),
        Op::Items {
            tensor: t,
            range,
            size,
        } => {
            let range = match range {
                Some((f, n)) => Some((index(frame, *f)?, index(frame, *n)?)),
                None => None,
            };
            Value::View(ops::items(
                cx.heap,
                tensor(frame, *t)?,
                range,
                index(frame, *size)?,
            )?)
        }
        Op::ReadTile(v, i) => {
            let (v, i) = (view(frame, *v)?, index(frame, *i)?);
            if let Ok(tiles) = ops::view_tiles(cx.heap, v) {
                let one: Vec<usize> = tiles.get(i).map(|t| vec![*t]).unwrap_or_default();
                let places = one
                    .iter()
                    .map(|n| TensorPlace::tile(v.tensor, *n))
                    .collect();
                cx.accesses.note(false, place.rank, v.tensor, one);
                touch(cx, place.rank, here, places, false)?;
            }
            Value::Tile(ops::read(cx.heap, v, i)?)
        }
        Op::Words(v) => {
            let v = view(frame, *v)?;
            note_view(cx, place.rank, v, false);
            touch(cx, place.rank, here, tile_places(cx, v), false)?;
            Value::Words(ops::words(cx.heap, v)?)
        }
        Op::Word(w, i) => {
            let Value::Words(words) = get(frame, *w)? else {
                return Err(Fault::undefined("a word of what is not words"));
            };
            let i = index(frame, *i)?;
            Value::Scalar(*words.get(i).ok_or(Fault::Failure(Failure::OutOfRange))?)
        }
        Op::Records(v, r, c) => {
            let rec = module
                .records
                .get(r.0 as usize)
                .ok_or_else(|| Fault::undefined("no such record"))?;
            let ty = RecordType {
                name: rec.name.clone(),
                fields: rec
                    .fields
                    .iter()
                    .map(|(n, s)| (n.clone(), s.format()))
                    .collect(),
            };
            let v = view(frame, *v)?;
            note_view(cx, place.rank, v, false);
            touch(cx, place.rank, here, tile_places(cx, v), false)?;
            Value::Records(ops::records(cx.heap, v, &ty, index(frame, *c)?)?)
        }
        Op::RecordAt(x, i) => {
            let Value::Records(records) = get(frame, *x)? else {
                return Err(Fault::undefined("a record of what is not records"));
            };
            let i = index(frame, *i)?;
            Value::Record(
                records
                    .get(i)
                    .cloned()
                    .ok_or(Fault::Failure(Failure::OutOfRange))?,
            )
        }
        Op::RecordField(x, k) => {
            let Value::Record(fields) = get(frame, *x)? else {
                return Err(Fault::undefined("a field of what is not a record"));
            };
            Value::Scalar(
                *fields
                    .get(*k as usize)
                    .ok_or_else(|| Fault::undefined("no such field"))?,
            )
        }
        Op::Put { items, src, at } => {
            let v = view(frame, *items)?.clone();
            let i = index(frame, *at)?;
            let m = member(worker, frame.at, *src)?.clone();
            note_view(cx, place.rank, &v, true);
            let places = ops::item_places(cx.heap, &v, i).unwrap_or_default();
            touch(cx, place.rank, here, places, true)?;
            ops::put(cx.heap, &v, &m, i)?;
            return Ok(Next::Go);
        }
        Op::Partition(v, r, s) => Value::View(ops::partition(
            view(frame, *v)?,
            index(frame, *r)?,
            index(frame, *s)?,
        )?),
        Op::Stream(v, n) => {
            // the rows of `block = [r, c]` (R45), in the stream's type
            let rows = match def_ty(0) {
                Some(Type::Stream(_, Count::Const(r), _)) => *r as usize,
                _ => 1,
            };
            Value::Stream(ops::stream_blocks(
                view(frame, *v)?,
                index(frame, *n)?,
                rows,
            )?)
        }
        Op::Sink(v, n) => {
            let v = view(frame, *v)?;
            note_view(cx, place.rank, v, true);
            Value::Sink(ops::sink(v, index(frame, *n)?)?)
        }
        Op::Push(s, t) => {
            let block = as_block(get(frame, *t)?)?;
            let Some(Some(Value::Sink(sink))) = frame.values.get_mut(s.0 as usize) else {
                return Err(Fault::undefined("a push to what is not a sink"));
            };
            let places = ops::push_tiles(sink, block.tiles.len())
                .unwrap_or_default()
                .into_iter()
                .map(|n| TensorPlace::tile(sink.view.tensor, n))
                .collect();
            touch(cx, place.rank, here, places, true)?;
            ops::push(cx.heap, sink, &block)?;
            return Ok(Next::Go);
        }
        Op::Store(v, x) => {
            let block = as_block(get(frame, *x)?)?;
            let v = view(frame, *v)?;
            note_view(cx, place.rank, v, true);
            touch(cx, place.rank, here, tile_places(cx, v), true)?;
            ops::store(cx.heap, v, &block)?;
            return Ok(Next::Go);
        }
        Op::BlockAt(b, i) => {
            let Value::Block(block) = get(frame, *b)? else {
                return Err(Fault::undefined("a tile of what is not a block"));
            };
            let i = index(frame, *i)?;
            Value::Tile(
                block
                    .tiles
                    .get(i)
                    .cloned()
                    .ok_or(Fault::Failure(Failure::OutOfRange))?,
            )
        }
        // ---- state ----
        Op::Fill(m, s) => {
            let mut st = stream(frame, *s)?.clone();
            ops::fill(cx.heap, member(worker, frame.at, *m)?, &mut st)?;
            return Ok(Next::Go);
        }
        Op::ZeroState(m) => {
            ops::zero(member(worker, frame.at, *m)?);
            return Ok(Next::Go);
        }
        Op::StateSet(m, i, x) => {
            let i = index(frame, *i)?;
            let e = element_of(get(frame, *x)?)?;
            ops::set(member(worker, frame.at, *m)?, i, e)?;
            return Ok(Next::Go);
        }
        Op::StateGet(m, i) => {
            let i = index(frame, *i)?;
            element_value(ops::get(member(worker, frame.at, *m)?, i)?)
        }
        Op::Bump(m, i) => {
            let i = index(frame, *i)?;
            Value::Scalar(ops::bump(member(worker, frame.at, *m)?, i)?)
        }
        // ---- identity ----
        Op::Rank => Value::Scalar(int_of(Format::U32, place.rank as u64)),
        Op::IsLeader => Value::Scalar(Scalar::Bool(place.rank == 0)),
        Op::WorkerId => Value::Scalar(int_of(Format::U32, worker.id as u64)),
        Op::Row | Op::Col => {
            let (_, cols) = place
                .grid
                .ok_or_else(|| Fault::undefined("a row or a column outside a grid"))?;
            let cols = cols.max(1);
            let n = if matches!(op, Op::Row) {
                place.rank / cols
            } else {
                place.rank % cols
            };
            Value::Scalar(int_of(Format::U32, n as u64))
        }
        Op::GroupSize => Value::Scalar(int_of(Format::U32, place.size as u64)),
        Op::GroupId(r) => {
            let r = index(frame, *r)?;
            if r >= place.size {
                return Err(Failure::OutOfRange.into());
            }
            Value::Scalar(int_of(Format::U32, (place.first + r) as u64))
        }
        Op::FailureName(f) => Value::Reason(*f),
        // ---- islands ----
        Op::IslandCount => Value::Scalar(int_of(Format::U32, cx.islands.count() as u64)),
        Op::IslandOf(id) => {
            let island = cx.islands.island(index(frame, *id)?)?;
            Value::Scalar(int_of(Format::U32, island as u64))
        }
        Op::IslandCrosses(id) | Op::IslandHops(id) => {
            // the asker's own island: a control program runs on none
            if matches!(frame.at, BodyRef::Control(_)) {
                return Err(Fault::undefined(
                    "`island.crosses` or `island.hops` in a control program, which has no island",
                ));
            }
            let mine = cx.islands.island(worker.id)?;
            let theirs = cx.islands.island(index(frame, *id)?)?;
            if matches!(op, Op::IslandCrosses(_)) {
                Value::Scalar(Scalar::Bool(mine != theirs))
            } else {
                let hops = cx
                    .islands
                    .hops
                    .get(mine)
                    .and_then(|h| h.get(theirs))
                    .copied()
                    .ok_or_else(|| Fault::undefined("an island with no path"))?;
                Value::Scalar(int_of(Format::U32, hops))
            }
        }
        Op::Split(by, n) => Value::SubGroups(Split {
            parts: *by == SplitBy::Parts,
            n: index(frame, *n)?,
        }),
        Op::Leaders(by, n) => Value::Leaders(Split {
            parts: *by == SplitBy::Parts,
            n: index(frame, *n)?,
        }),
        // ---- collectives: the frame waits at a meeting ----
        Op::ReduceGroup {
            x,
            with,
            within,
            among,
            publish,
        } => {
            let publish = published(cx, frame, publish)?;
            let within = split_of(frame, *within, false)?;
            let among = split_of(frame, *among, true)?;
            if within.is_some() && among.is_some() {
                return Err(Fault::undefined("a `reduce` both within and among"));
            }
            let c = Collective::Reduce {
                x: operand_of(frame, *x)?,
                with: *with,
                within,
                among,
            };
            return Ok(Next::Meet(c, publish));
        }
        Op::AllReduce { x, with, .. } if matches!(frame.at, BodyRef::Control(_)) => {
            // in a control program: over the copies of a replicated tensor
            let Value::Tensor(t) = get(frame, *x)? else {
                return Err(Fault::undefined("an `all_reduce` of what is not a tensor"));
            };
            return Ok(Next::Ask(Request::Copies(
                *t,
                OverCopies::AllReduce { with: *with },
            )));
        }
        Op::ReduceTo(x, to, with) => {
            let Value::Tensor(t) = get(frame, *x)? else {
                return Err(Fault::undefined("a `reduce_to` of what is not a tensor"));
            };
            let t = *t;
            let to = index(frame, *to)?;
            return Ok(Next::Ask(Request::Copies(
                t,
                OverCopies::ReduceTo { to, with: *with },
            )));
        }
        Op::BroadcastFrom(x, from) => {
            let Value::Tensor(t) = get(frame, *x)? else {
                return Err(Fault::undefined(
                    "a `broadcast_from` of what is not a tensor",
                ));
            };
            let t = *t;
            let from = index(frame, *from)?;
            return Ok(Next::Ask(Request::Copies(
                t,
                OverCopies::BroadcastFrom { from },
            )));
        }
        Op::AllReduce { x, with, publish } => {
            let publish = published(cx, frame, publish)?;
            let c = Collective::AllReduce {
                x: operand_of(frame, *x)?,
                with: *with,
            };
            return Ok(Next::Meet(c, publish));
        }
        Op::Gather(x) => {
            let c = Collective::Gather {
                x: operand_of(frame, *x)?,
            };
            return Ok(Next::Meet(c, Vec::new()));
        }
        Op::Sync(publish) => {
            let publish = published(cx, frame, publish)?;
            return Ok(Next::Meet(Collective::Sync, publish));
        }
        // ---- dynamic runs: messages ----
        Op::Arrived => frame
            .arrived
            .clone()
            .ok_or_else(|| Fault::undefined("no message arrived for this body"))?,
        Op::MakeMessage(ty, fields) => {
            let mut values = Vec::with_capacity(fields.len());
            for f in fields {
                values.push(scalar(frame, *f)?);
            }
            Value::Message(*ty, values)
        }
        Op::Field(m, k) => match get(frame, *m)? {
            Value::Message(_, fields) => Value::Scalar(
                *fields
                    .get(*k as usize)
                    .ok_or_else(|| Fault::undefined("no such field"))?,
            ),
            other => return Err(Fault::undefined(format!("a field of {}", kind(other)))),
        },
        Op::Send {
            message,
            to,
            used,
            publish,
        } => {
            let m = message_of(frame, *message)?;
            let to = rank_to(frame, *to, place)?;
            let refused = cx.mail.bound.is_some_and(|b| cx.mail.live >= b);
            if refused && !*used {
                return Err(Failure::SendRefused.into());
            }
            if !refused {
                let tiles = published(cx, frame, publish)?;
                let publications = cx.races.publish(place.rank, &tiles);
                cx.mail.live += 1;
                let letter = cx.mail.letter(place.rank, to, m, cx.races, publications);
                cx.mail.outbox.push(letter);
            }
            if defs.is_empty() {
                return Ok(Next::Go);
            }
            Value::Scalar(Scalar::Bool(!refused))
        }
        Op::Forward {
            message,
            with,
            to,
            publish,
        } => {
            let Value::Message(ty, mut fields) = message_of(frame, *message)? else {
                return Err(Fault::undefined("`forward` of what is not a message"));
            };
            for (k, v) in with {
                let x = scalar(frame, *v)?;
                let slot = fields
                    .get_mut(*k as usize)
                    .ok_or_else(|| Fault::undefined("no such field"))?;
                *slot = x;
            }
            let to = rank_to(frame, *to, place)?;
            let tiles = published(cx, frame, publish)?;
            let publications = cx.races.publish(place.rank, &tiles);
            cx.mail.handled += 1;
            let m = Value::Message(ty, fields);
            let letter = cx.mail.letter(place.rank, to, m, cx.races, publications);
            cx.mail.outbox.push(letter);
            return Ok(Next::Go);
        }
        Op::Consume(m) => {
            message_of(frame, *m)?;
            cx.mail.handled += 1;
            cx.mail.live = cx.mail.live.saturating_sub(1);
            return Ok(Next::Go);
        }
        Op::Offer(message, to) => {
            let m = message_of(frame, *message)?;
            let to = rank_to(frame, *to, place)?;
            let (from, size) = (place.rank, place.size.max(1) as u64);
            let peer = match &cx.mail.peers {
                Peers::None => false,
                Peers::Offsets(ds) => ds
                    .iter()
                    .any(|d| (from as u64 + d % size) % size == to as u64),
                Peers::Pairs(ps) => ps.iter().any(|&(s, r)| (s, r) == (from as u64, to as u64)),
            };
            if !peer {
                return Err(Failure::NotAPeer.into());
            }
            let taken = cx.mail.slots.contains_key(&(from, to));
            if !taken {
                let letter = cx.mail.letter(from, to, m, cx.races, Vec::new());
                cx.mail.slots.insert((from, to), letter);
                cx.mail.live += 1;
            }
            if defs.is_empty() {
                return Ok(Next::Go);
            }
            Value::Scalar(Scalar::Bool(!taken))
        }
        // ---- control programs: requests the run answers ----
        Op::MakeGroup(first, count) => {
            // a group given as a count only: the run chooses the first worker, 0
            let first = match first {
                Some(f) => index(frame, *f)?,
                None => 0,
            };
            Value::Group(first, index(frame, *count)?)
        }
        Op::Invoke {
            entry,
            args,
            on,
            after,
            after_end,
        } => {
            let Value::Group(first, count) = get(frame, *on)? else {
                return Err(Fault::undefined("`invoke` on what is not a group"));
            };
            let group = (*first, *count);
            let mut values = Vec::with_capacity(args.len());
            for a in args {
                values.push(get(frame, *a)?.clone());
            }
            return Ok(Next::Ask(Request::Invoke {
                entry: *entry,
                args: values,
                group,
                after: handles(frame, after)?,
                after_end: handles(frame, after_end)?,
            }));
        }
        Op::Await(h) => return Ok(Next::Ask(Request::Await(handle(frame, *h)?))),
        Op::Outcome(h) => return Ok(Next::Ask(Request::Outcome(handle(frame, *h)?))),
        Op::Failed(o) => Value::Scalar(Scalar::Bool(outcome(frame, *o)?.failed)),
        Op::Reason(o) => match outcome(frame, *o)?.reason {
            Some(r) => Value::Reason(r),
            None => {
                return Err(Fault::undefined(
                    "the reason of an invocation that completed",
                ))
            }
        },
        Op::OutcomeValue(o) => match outcome(frame, *o)?.value {
            Some(v) => Value::Scalar(v),
            None => {
                return Err(Fault::undefined(
                    "the value of an invocation that failed or gave none",
                ))
            }
        },
        // ---- broadcast: a line from one sender, cast without waiting ----
        Op::Broadcast(axis, sender) => {
            let (_, cols) = place
                .grid
                .ok_or_else(|| Fault::undefined("a broadcast outside a grid"))?;
            let cols = cols.max(1);
            let (row, col) = (place.rank / cols, place.rank % cols);
            let s = index(frame, *sender)?;
            let (a, line, at, across) = match axis {
                Axis::Row => (0, row, col, cols),
                Axis::Col => (1, col, row, place.size / cols),
            };
            // the sender is a column of the row (a row of the column)
            if s >= across {
                return Err(Failure::OutOfRange.into());
            }
            let key = (frame.made, a, line);
            frame.made += 1;
            cx.lines.entry(key).or_default();
            if at == s {
                let closer = shape.closer_of.get(pc).copied().flatten();
                frame.sending.push((closer, key));
            }
            Value::Broadcast {
                line: key,
                sender: at == s,
            }
        }
        Op::IsSender(b) => match get(frame, *b)? {
            Value::Broadcast { sender, .. } => Value::Scalar(Scalar::Bool(*sender)),
            other => return Err(Fault::undefined(format!("`is_sender` of {}", kind(other)))),
        },
        Op::Cast(b, x, w) => {
            let Value::Broadcast { line, sender } = get(frame, *b)? else {
                return Err(Fault::undefined("`cast` on what is not a broadcast"));
            };
            if !*sender {
                return Err(Fault::undefined(
                    "a `cast` by a worker that is not the sender",
                ));
            }
            let line = *line;
            // a cast that leaves the word out sends 0
            let word = match w {
                Some(w) => scalar(frame, *w)?,
                None => int_of(Format::U32, 0),
            };
            let tiles = match get(frame, *x)? {
                Value::Tile(t) => vec![t.clone()],
                Value::Block(b) => b.tiles.clone(),
                other => return Err(Fault::undefined(format!("a `cast` of {}", kind(other)))),
            };
            let open = cx.lines.entry(line).or_default();
            if open.closed {
                return Err(Fault::undefined("a `cast` on a line that has ended"));
            }
            let clock = cx.races.release(place.rank);
            let open = cx.lines.entry(line).or_default();
            open.items
                .extend(tiles.into_iter().map(|t| (t, word, clock.clone())));
            return Ok(Next::Go);
        }
        Op::ForArrive { broadcast } => {
            let next = match frame.loops.last() {
                Some(Loop::Arrive { opener, next, .. }) if *opener == pc => *next,
                _ => {
                    let Value::Broadcast { line, .. } = get(frame, *broadcast)? else {
                        return Err(Fault::undefined("`arrive` on what is not a broadcast"));
                    };
                    let line = *line;
                    frame.loops.push(Loop::Arrive {
                        opener: pc,
                        line,
                        next: 0,
                    });
                    0
                }
            };
            let Some(Loop::Arrive { line, .. }) = frame.loops.last() else {
                return Err(Fault::undefined("an `arrive` with no loop"));
            };
            let line = *line;
            let open = cx.lines.get(&line).cloned().unwrap_or_default();
            match open.items.get(next) {
                Some((tile, word, clock)) => {
                    // the cast is before the arrival
                    cx.races.acquire(place.rank, clock);
                    def(frame, 0, Value::Tile(tile.clone()))?;
                    def(frame, 1, Value::Scalar(*word))?;
                    if let Some(Loop::Arrive { next: n, .. }) = frame.loops.last_mut() {
                        *n += 1;
                    }
                }
                None if open.closed => {
                    frame.loops.pop();
                    frame.pc = end_of(pc)? + 1;
                }
                None => {
                    frame.pc = pc;
                    return Ok(Next::Wait);
                }
            }
            return Ok(Next::Go);
        }
    };
    def(frame, 0, value)?;
    Ok(Next::Go)
}
