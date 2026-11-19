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

use std::collections::HashMap;

use super::memory::{
    Block, Failure, Fault, Format, Heap, Member, RecordType, Reduced, Scalar, Sink, Steps, Stream,
    TensorId, Tile, View, Zip,
};
use super::ops::{self, Bin, Cmp, Element, Operand, ReducedOperand, Unary};
use crate::program::body::{Body, Inst, InstId, Op, ValueId};
use crate::program::{BodyRef, MemberKind, Module};
use crate::types::{
    BinOp, CmpOp, Combiner, ConstValue, Count, MemberId, ProgramId, ReduceOp, SplitBy, Type,
    UnaryOp,
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
}

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
}

impl Shape {
    /// The shape of `body`; a body whose markers do not balance is not defined.
    pub fn of(body: &Body) -> Result<Shape, Fault> {
        let n = body.insts.len();
        let mut shape = Shape {
            end_of: vec![None; n],
            else_of: vec![None; n],
            opener_of: vec![None; n],
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
        if open.is_empty() {
            Ok(shape)
        } else {
            Err(Fault::undefined("a construct with no `end`"))
        }
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
}

impl Frame {
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
    /// The frame reached a collective and waits for its value (`deliver`).
    Meeting {
        /// The collective, with this rank's operand.
        collective: Collective,
        /// The body and the instruction.
        at: (BodyRef, InstId),
    },
}

impl Thread {
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
                Ok(Next::Meet(collective)) => {
                    return Stop::Meeting {
                        collective,
                        at: (at, InstId(pc as u32)),
                    }
                }
                Ok(Next::Call(callee)) => self.frames.push(*callee),
                Ok(Next::Return(value)) => {
                    let done = self.frames.pop();
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
    Meet(Collective),
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
        Value::Tile(t) => Ok(Block {
            tiles: vec![t.clone()],
        }),
        Value::Block(b) => Ok(b.clone()),
        other => Err(Fault::undefined(format!(
            "{} where tiles are written",
            kind(other)
        ))),
    }
}

/// `a op b` on any two operands the language allows.
pub fn binary(op: BinOp, a: &Value, b: &Value) -> Result<Value, Fault> {
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
        return Ok(Next::Return(None));
    };
    let pc = frame.pc;
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
        Op::Compare(c, a, b) => Value::Scalar(Scalar::Bool(ops::compare(
            cmp(*c),
            scalar(frame, *a)?,
            scalar(frame, *b)?,
        )?)),
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
                        }) if o == opener => match ops::next_block(cx.heap, &mut stream)? {
                            Some(block) => {
                                set(frame, 0, block_value(block, ty(0)))?;
                                frame.loops.push(Loop::Stream { opener: o, stream });
                                true
                            }
                            None => false,
                        },
                        Some(Loop::Zip { opener: o, mut zip }) if o == opener => {
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
            Value::Reduced(r) => Value::Reduced(ops::reduced_unary(unary(*u), r)?),
            Value::Scalar(s) if *u == UnaryOp::Neg => Value::Scalar(negate(*s)?),
            other => return Err(Fault::undefined(format!("{u:?} of {}", kind(other)))),
        },
        Op::Select { cmp: c, a, b, x, y } => Value::Tile(ops::select(
            cmp(*c),
            operand(get(frame, *a)?)?,
            operand(get(frame, *b)?)?,
            operand(get(frame, *x)?)?,
            operand(get(frame, *y)?)?,
        )?),
        Op::Reduce(r, x, axis) => {
            let Value::Tile(t) = get(frame, *x)? else {
                return Err(Fault::undefined("a reduction of what is not a tile"));
            };
            let kind = match r {
                ReduceOp::Sum => ops::Reduce::Sum,
                ReduceOp::Max => ops::Reduce::Max,
                ReduceOp::Min => ops::Reduce::Min,
            };
            Value::Reduced(ops::reduce(kind, t, *axis)?)
        }
        Op::SpreadCol(x) => match get(frame, *x)? {
            Value::Reduced(r) => Value::Tile(ops::col(r)?),
            Value::Tile(t) => Value::Tile(ops::spread_col(t)?),
            other => return Err(Fault::undefined(format!(".col() of {}", kind(other)))),
        },
        Op::SpreadRow(x) => match get(frame, *x)? {
            Value::Reduced(r) => Value::Tile(ops::row(r)?),
            Value::Tile(t) => Value::Tile(ops::spread_row(t)?),
            other => return Err(Fault::undefined(format!(".row() of {}", kind(other)))),
        },
        Op::Matmul(a, b) => match (get(frame, *a)?, get(frame, *b)?) {
            (Value::Tile(x), Value::Tile(y)) => Value::Tile(ops::matmul(x, y)?),
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
            let mut callee = Frame::new(module, BodyRef::Func(*f), values)?;
            callee.result_to = defs.first().copied();
            return Ok(Next::Call(Box::new(callee)));
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
        Op::ReadTile(v, i) => Value::Tile(ops::read(cx.heap, view(frame, *v)?, index(frame, *i)?)?),
        Op::Words(v) => Value::Words(ops::words(cx.heap, view(frame, *v)?)?),
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
            Value::Records(ops::records(
                cx.heap,
                view(frame, *v)?,
                &ty,
                index(frame, *c)?,
            )?)
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
            ops::put(cx.heap, &v, &m, i)?;
            return Ok(Next::Go);
        }
        Op::Partition(v, r, s) => Value::View(ops::partition(
            view(frame, *v)?,
            index(frame, *r)?,
            index(frame, *s)?,
        )?),
        Op::Stream(v, n) => Value::Stream(ops::stream(view(frame, *v)?, index(frame, *n)?)?),
        Op::Sink(v, n) => Value::Sink(ops::sink(view(frame, *v)?, index(frame, *n)?)?),
        Op::Push(s, t) => {
            let block = as_block(get(frame, *t)?)?;
            let Some(Some(Value::Sink(sink))) = frame.values.get_mut(s.0 as usize) else {
                return Err(Fault::undefined("a push to what is not a sink"));
            };
            ops::push(cx.heap, sink, &block)?;
            return Ok(Next::Go);
        }
        Op::Store(v, x) => {
            let block = as_block(get(frame, *x)?)?;
            ops::store(cx.heap, view(frame, *v)?, &block)?;
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
            ..
        } => {
            let within = split_of(frame, *within, false)?;
            let among = split_of(frame, *among, true)?;
            if within.is_some() && among.is_some() {
                return Err(Fault::undefined("a `reduce` both within and among"));
            }
            return Ok(Next::Meet(Collective::Reduce {
                x: operand_of(frame, *x)?,
                with: *with,
                within,
                among,
            }));
        }
        Op::AllReduce { x, with, .. } => {
            return Ok(Next::Meet(Collective::AllReduce {
                x: operand_of(frame, *x)?,
                with: *with,
            }))
        }
        Op::Gather(x) => {
            return Ok(Next::Meet(Collective::Gather {
                x: operand_of(frame, *x)?,
            }))
        }
        Op::Sync(_) => return Ok(Next::Meet(Collective::Sync)),
        // ---- the parts that run in later stages ----
        other => {
            return Err(Fault::undefined(format!(
                "{} is not run by this interpreter yet",
                op_name(other)
            )))
        }
    };
    def(frame, 0, value)?;
    Ok(Next::Go)
}

/// A short name of an operation, for a report.
fn op_name(op: &Op) -> String {
    let text = format!("{op:?}");
    text.split(['(', ' ', '{']).next().unwrap_or("").to_string()
}
