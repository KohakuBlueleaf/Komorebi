//! What a handler's body is, in the terms today's class DSL compiles: its command
//! words, the one partition its streams share, a write stream, a read stream whose
//! loop pushes a tile function's result. A body of another form is refused by name
//! (K964) until it is built.

use std::collections::HashMap;

use crate::backend::{Refusal, NOT_BUILT};
use crate::program::body::{Body, InstId, Op, ValueId};
use crate::program::notes::{Anchor, Named, Owner};
use crate::program::{BodyRef, FuncKind, Module};
use crate::types::{ConstValue, FuncId, HandlerId, On, ProgramId, ScalarType, Type};

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
}

/// A partition the kernels print: `partition(t.tiles(first, n), rank, size)`, by its
/// words; which partitions are one is the layout rule `shared`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    /// Its name: the first partition's.
    pub name: String,
    /// The word of `first`.
    pub first: usize,
    /// The word of `n`.
    pub n: usize,
    /// The first partition's instruction.
    pub inst: InstId,
}

/// A stream of whole tiles over the partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moved {
    /// The tensor's word.
    pub tensor: usize,
    /// Its block.
    pub block: u64,
    /// The instruction that opens it: the sink, or the loop over the stream.
    pub inst: InstId,
    /// Its partition: an index into the plan's `parts`.
    pub part: usize,
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
        words.push(Word { name, kind });
    }
    let body = BodyRef::Handler(p, h);
    let found = Walk::new(module, &handler.body, body, &words).run()?;
    let Found {
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
        parts,
        write,
        read,
        call,
    })
}

/// The layout rule for partitions (today's DSL writes one partition and indexes both
/// tensors with it, `y[src]`): partitions that take the same values (the same `first`
/// and `n` words, by `worker.rank` and `group.size`, the only rank and size the walk
/// takes) are one range whatever their tensors, printed once, as the first of them
/// written; partitions that differ in a value are printed each. Gives the partitions
/// to print and, for each given one, its index among them.
pub fn shared(parts: &[Part]) -> (Vec<Part>, Vec<usize>) {
    let mut order: Vec<usize> = (0..parts.len()).collect();
    order.sort_by_key(|&i| parts[i].inst);
    let mut out: Vec<Part> = Vec::new();
    let mut index = vec![0; parts.len()];
    for i in order {
        let part = &parts[i];
        match out
            .iter()
            .position(|p| (p.first, p.n) == (part.first, part.n))
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
#[derive(Clone, Copy, Debug)]
enum Val {
    Arg(usize),
    Const(u64),
    Rank,
    Size,
    Tiles {
        tensor: usize,
        first: usize,
        n: usize,
    },
    Part {
        tensor: usize,
        first: usize,
        n: usize,
        inst: InstId,
    },
    Stream,
    Sink,
    Tile,
    Called,
}

/// What the walk found.
struct Found {
    parts: Vec<Part>,
    write: Moved,
    read: Moved,
    call: Call,
}

/// The body read instruction by instruction: arguments, partitions of `t.tiles(first,
/// n)` by rank and size, one sink, one stream, the loop over it whose body calls a
/// tile function on the loop's tile and a `f32` word and pushes the result into the
/// sink, the loop's end, and nothing after.
struct Walk<'m> {
    module: &'m Module,
    body: &'m Body,
    at: BodyRef,
    words: &'m [Word],
    vals: HashMap<ValueId, Val>,
    sink: Option<(ValueId, Moved, (usize, usize, InstId))>,
    stream: Option<(Moved, (usize, usize, InstId))>,
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
            sink: None,
            stream: None,
            call: None,
            looped: false,
            pushed: false,
            ended: false,
        }
    }

    fn val(&self, v: ValueId) -> Option<Val> {
        self.vals.get(&v).copied()
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
        let (Some((_, write, wpart)), Some((read, rpart)), Some(call), true) =
            (self.sink, self.stream, self.call, self.ended)
        else {
            return Err(not_built(
                "a handler other than a stream loop that pushes a tile function's result into a sink",
                at,
            ));
        };
        if wpart.1 != rpart.1 {
            return Err(not_built(
                "a sink of another count than its loop's stream",
                at,
            ));
        }
        if write.block != read.block {
            return Err(not_built("a sink in blocks other than its loop's", at));
        }
        let named = |(first, n, inst): (usize, usize, InstId)| {
            let defined = self
                .body
                .insts
                .get(inst.0 as usize)
                .and_then(|i| i.defs.first().copied());
            let name = defined
                .and_then(|v| value_name(self.module, self.at, v))
                .map_or_else(
                    || format!("v{}", defined.map_or(0, |v| v.0)),
                    str::to_string,
                );
            Part {
                name,
                first,
                n,
                inst,
            }
        };
        let (parts, index) = shared(&[named(wpart), named(rpart)]);
        Ok(Found {
            parts,
            write: Moved {
                part: index[0],
                ..write
            },
            read: Moved {
                part: index[1],
                ..read
            },
            call,
        })
    }

    /// What one instruction is, or why it is not built.
    fn step(&mut self, op: &Op, id: InstId) -> Result<Option<Val>, String> {
        Ok(match op {
            Op::Arg(a) => Some(Val::Arg(*a as usize)),
            Op::Const(ConstValue::Int(_, x)) => Some(Val::Const(*x)),
            Op::Rank => Some(Val::Rank),
            Op::GroupSize => Some(Val::Size),
            Op::Tiles { tensor, first, n } => {
                let (Some(tensor), Some(first), Some(n)) = (
                    self.word(*tensor, WordKind::Tensor),
                    self.word(*first, WordKind::U32),
                    self.word(*n, WordKind::U32),
                ) else {
                    return Err("a view other than t.tiles(first, n) of parameters".into());
                };
                Some(Val::Tiles { tensor, first, n })
            }
            Op::Partition(view, rank, size) => {
                match (self.val(*view), self.val(*rank), self.val(*size)) {
                    (Some(Val::Tiles { tensor, first, n }), Some(Val::Rank), Some(Val::Size)) => {
                        Some(Val::Part {
                            tensor,
                            first,
                            n,
                            inst: id,
                        })
                    }
                    _ => {
                        return Err(
                            "a partition other than of a view by worker.rank and group.size".into(),
                        )
                    }
                }
            }
            Op::Sink(view, block) | Op::Stream(view, block) => {
                let (
                    Some(Val::Part {
                        tensor,
                        first,
                        n,
                        inst,
                    }),
                    Some(Val::Const(block)),
                ) = (self.val(*view), self.val(*block))
                else {
                    return Err("a stream other than of a partition in a constant block".into());
                };
                let moved = Moved {
                    tensor,
                    block,
                    inst: id,
                    part: 0,
                };
                if matches!(op, Op::Stream(..)) {
                    if self.stream.is_some() {
                        return Err("a second read stream".into());
                    }
                    self.stream = Some((moved, (first, n, inst)));
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
                    self.sink = Some((v, moved, (first, n, inst)));
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
            first,
            n,
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
