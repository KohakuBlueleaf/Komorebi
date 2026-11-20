//! What a handler's body is, in the terms today's class DSL compiles: its command
//! words, its named int values, the one partition its streams share or the range
//! each streams, a write stream, a read stream whose loop pushes a tile function's
//! result. A body of another form is refused by name (K964) until it is built.

use std::collections::HashMap;

use crate::backend::{Refusal, NOT_BUILT};
use crate::program::body::{Body, InstId, Op, ValueId};
use crate::program::notes::{Anchor, Named, Owner};
use crate::program::{BodyRef, FuncKind, Module};
use crate::types::{BinOp, ConstValue, Format, FuncId, HandlerId, On, ProgramId, ScalarType, Type};

/// How a command word is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordKind {
    /// A tensor: its address word.
    Tensor,
    /// A `u32`.
    U32,
    /// An `f32`: its bits.
    F32,
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
    /// `(a op b)`: `+`, `-` or `*`.
    Bin(char, Box<Int>, Box<Int>),
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

/// The tiles a stream moves: a partition's part, or `t.tiles(first, n)` as it is.
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
}

/// The loop's body: `out.push(f(tile, scalar))`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    /// The tile function.
    pub func: FuncId,
    /// Its name.
    pub name: String,
    /// The word of its scalar argument.
    pub scalar: usize,
    /// The call's instruction.
    pub inst: InstId,
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
    /// The write stream (the sink).
    pub write: Moved,
    /// The read stream (the loop's).
    pub read: Moved,
    /// The loop's call.
    pub call: Call,
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
    let found = Walk::new(module, &handler.body, body, &words).run()?;
    let Found {
        locals,
        parts,
        write,
        read,
        call,
    } = found;
    Ok(Plan {
        program: p,
        handler: h,
        name: handler.name.clone(),
        words,
        locals,
        parts,
        write,
        read,
        call,
    })
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
    Tiles { tensor: usize, first: Int, n: Int },
    Part(usize, Part),
    Stream,
    Sink,
    Tile,
    Called,
}

/// The tiles a stream was opened on, before the partitions are shared.
#[derive(Clone, Debug)]
enum Over {
    Part(Part),
    Tiles(Int, Int),
}

/// What the walk found.
struct Found {
    locals: Vec<Local>,
    parts: Vec<Part>,
    write: Moved,
    read: Moved,
    call: Call,
}

/// The body read instruction by instruction: arguments, ints of words, numbers and
/// the rank (a named one a local), partitions of `t.tiles(first, n)` by rank and
/// size, one sink, one stream, each of a partition or of tiles, the loop over the
/// stream whose body calls a tile function on the loop's tile (or block) and a `f32`
/// word and pushes the result into the sink, the loop's end, and nothing after.
struct Walk<'m> {
    module: &'m Module,
    body: &'m Body,
    at: BodyRef,
    words: &'m [Word],
    vals: HashMap<ValueId, Val>,
    locals: Vec<Local>,
    sink: Option<(ValueId, Moved, Over)>,
    stream: Option<(Moved, Over)>,
    call: Option<Call>,
    looped: bool,
    pushed: bool,
    ended: bool,
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
            stream: None,
            call: None,
            looped: false,
            pushed: false,
            ended: false,
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
            if self.ended {
                return Err(not_built("a handler's lines after its stream loop", anchor));
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
        let (Some((_, write, wover)), Some((read, rover)), Some(call), true) =
            (self.sink, self.stream, self.call, self.ended)
        else {
            return Err(not_built(
                "a handler other than a stream loop that pushes a tile function's result into a sink",
                at,
            ));
        };
        if write.block != read.block {
            return Err(not_built("a sink in blocks other than its loop's", at));
        }
        let (parts, write, read) = match (wover, rover) {
            (Over::Part(w), Over::Part(r)) => {
                if w.n != r.n {
                    return Err(not_built(
                        "a sink of another count than its loop's stream",
                        at,
                    ));
                }
                let (parts, index) = shared(&[w, r]);
                let write = Moved {
                    range: Range::Part(index[0]),
                    ..write
                };
                let read = Moved {
                    range: Range::Part(index[1]),
                    ..read
                };
                (parts, write, read)
            }
            (Over::Tiles(wf, wn), Over::Tiles(rf, rn)) => {
                if wn != rn {
                    return Err(not_built(
                        "a sink of another count than its loop's stream",
                        at,
                    ));
                }
                let write = Moved {
                    range: Range::Tiles { first: wf, n: wn },
                    ..write
                };
                let read = Moved {
                    range: Range::Tiles { first: rf, n: rn },
                    ..read
                };
                (Vec::new(), write, read)
            }
            _ => {
                return Err(not_built(
                    "a sink of a partition and its loop's stream of tiles, or the other way",
                    at,
                ))
            }
        };
        // a named value no stream reads: no line today's text prints
        let mut used = Vec::new();
        let ints = parts.iter().flat_map(|p| [&p.first, &p.n]).chain(
            [&write.range, &read.range]
                .into_iter()
                .flat_map(|r| match r {
                    Range::Tiles { first, n } => vec![first, n],
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
        if let Range::Tiles { n, .. } = &read.range {
            n.each(&mut |i| counted |= matches!(i, Int::Local(_)));
        }
        if counted {
            return Err(not_built("a read stream's count of a named value", at));
        }
        Ok(Found {
            locals: self.locals,
            parts,
            write,
            read,
            call,
        })
    }

    /// What one instruction is, or why it is not built.
    fn step(&mut self, op: &Op, id: InstId) -> Result<Option<Val>, String> {
        Ok(match op {
            Op::Arg(a) => Some(Val::Arg(*a as usize)),
            Op::Const(ConstValue::Int(_, x)) => Some(Val::Int(Int::Const(*x))),
            Op::Rank => Some(Val::Int(Int::Rank)),
            Op::GroupSize => Some(Val::Int(Int::Size)),
            Op::Binary(op, a, b) => {
                let op = match op {
                    BinOp::Add => '+',
                    BinOp::Sub => '-',
                    BinOp::Mul => '*',
                    _ => return Err("an int operation other than + - *".into()),
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
                let block = match self.int(*block) {
                    Some(Int::Const(b)) => b,
                    _ => return Err("a stream in a block that is not a constant".into()),
                };
                let (tensor, over) = match self.val(*view) {
                    Some(Val::Part(tensor, part)) => (tensor, Over::Part(part)),
                    Some(Val::Tiles { tensor, first, n }) => (tensor, Over::Tiles(first, n)),
                    _ => {
                        return Err(
                            "a stream other than of a partition or of t.tiles(first, n)".into()
                        )
                    }
                };
                let moved = Moved {
                    tensor,
                    block,
                    inst: id,
                    range: Range::Part(0),
                };
                if matches!(op, Op::Stream(..)) {
                    if self.stream.is_some() {
                        return Err("a second read stream".into());
                    }
                    self.stream = Some((moved, over));
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
                if self.looped || !matches!(self.val(*stream), Some(Val::Stream)) {
                    return Err("a loop other than one over the read stream".into());
                }
                if let Some((m, _)) = self.stream.as_mut() {
                    m.inst = id;
                }
                self.looped = true;
                Some(Val::Tile)
            }
            Op::CallFunc(f, args) => {
                let plain = self
                    .module
                    .funcs
                    .get(f.0 as usize)
                    .filter(|func| func.kind == FuncKind::Plain);
                let ([t, s], Some(func)) = (args.as_slice(), plain) else {
                    return Err("a call other than f(the loop's tile, a f32 word)".into());
                };
                let scalar = self.word(*s, WordKind::F32);
                let (true, None, true, Some(scalar)) = (
                    self.looped,
                    self.call.as_ref(),
                    matches!(self.val(*t), Some(Val::Tile)),
                    scalar,
                ) else {
                    return Err(
                        "a call other than f(the loop's tile, a f32 word), once, in the loop"
                            .into(),
                    );
                };
                self.call = Some(Call {
                    func: *f,
                    name: func.name.clone(),
                    scalar,
                    inst: id,
                });
                Some(Val::Called)
            }
            Op::Push(s, v) => {
                let to_sink = self.sink.as_ref().is_some_and(|(sv, _, _)| sv == s);
                if self.pushed || !to_sink || !matches!(self.val(*v), Some(Val::Called)) {
                    return Err("a push other than of the call's tile into the sink, once".into());
                }
                self.pushed = true;
                None
            }
            Op::End if self.looped && self.pushed => {
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
