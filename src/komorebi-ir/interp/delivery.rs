//! Delivery between the workers of one invocation. A collective is a meeting: each
//! member's frame stops at it with its operand, and when the last member arrives the
//! value is made once, from the operands in rank order, and handed to the members it
//! goes to. The order of arrival never enters: a combination folds from the lowest
//! rank to the highest, `((x0 f x1) f x2) ...`, which keeps "the left operand holds
//! lower ranks" for a combine that is not commutative.

use super::memory::{Block, Fault, Scalar};
use super::ops::{self, Bin};
use super::workers::{binary, Collective, Context, Frame, Place, Stop, Thread, Value, Worker};
use crate::program::BodyRef;
use crate::types::{BinOp, Combiner};

/// The ranks that meet at a collective reached at `rank` of a group of `size`, in
/// order; `None` when `rank` is no member and passes it (the ranks that are not
/// leaders, at a `reduce` among leaders).
pub fn members(c: &Collective, rank: usize, size: usize) -> Result<Option<Vec<usize>>, Fault> {
    match c {
        Collective::Reduce {
            within: Some(split),
            ..
        } => {
            let parts = split.ranges(size)?;
            let part = parts
                .into_iter()
                .find(|r| r.contains(&rank))
                .ok_or_else(|| Fault::undefined("a rank in no sub-group"))?;
            Ok(Some(part.collect()))
        }
        Collective::Reduce {
            among: Some(split), ..
        } => {
            let leaders: Vec<usize> = split.ranges(size)?.into_iter().map(|r| r.start).collect();
            Ok(leaders.contains(&rank).then_some(leaders))
        }
        _ => Ok(Some((0..size).collect())),
    }
}

/// `a f b` under a combiner. A `combine` function runs its body; `budget` bounds its
/// steps.
pub fn combine(
    with: Combiner,
    a: Value,
    b: Value,
    cx: &mut Context<'_>,
    budget: u64,
) -> Result<Value, Fault> {
    let op = match with {
        Combiner::Sum => BinOp::Add,
        Combiner::Max => BinOp::Max,
        Combiner::Min => BinOp::Min,
        Combiner::And | Combiner::Or => return logic(with, a, b),
        Combiner::Func(f) => {
            let frame = Frame::new(cx.module, BodyRef::Func(f), vec![a, b])?;
            let mut thread = Thread::new(frame);
            // a function reads no state and no identity
            let mut nobody = Worker {
                id: 0,
                state: Vec::new(),
            };
            let place = Place {
                rank: 0,
                size: 1,
                first: 0,
                grid: None,
            };
            return match thread.run(&mut nobody, place, cx, budget) {
                Stop::Returned(Some(v)) => Ok(v),
                Stop::Faulted { fault, .. } => Err(fault),
                _ => Err(Fault::undefined("a combine function that gave no value")),
            };
        }
    };
    binary(op, &a, &b)
}

/// `and` and `or` as combiners: on booleans, and bit by bit on integers.
fn logic(with: Combiner, a: Value, b: Value) -> Result<Value, Fault> {
    let and = with == Combiner::And;
    match (a, b) {
        (Value::Scalar(Scalar::Bool(x)), Value::Scalar(Scalar::Bool(y))) => {
            Ok(Value::Scalar(Scalar::Bool(if and {
                x && y
            } else {
                x || y
            })))
        }
        (Value::Scalar(x), Value::Scalar(y)) => {
            let op = if and { Bin::And } else { Bin::Or };
            ops::scalar_binary(op, x, y).map(Value::Scalar)
        }
        _ => Err(Fault::undefined("`and` or `or` of what is not scalars")),
    }
}

/// The values a complete meeting hands out: for each member, in the order given, the
/// value its frame goes on with (none for a `sync`). The operands are the members'
/// collectives in rank order.
pub fn meet(
    arrivals: Vec<Collective>,
    cx: &mut Context<'_>,
    budget: u64,
) -> Result<Vec<Option<Value>>, Fault> {
    let n = arrivals.len();
    let absent = |c: &Collective| match c {
        Collective::Reduce { x, .. }
        | Collective::AllReduce { x, .. }
        | Collective::Gather { x } => *x == Value::Absent,
        Collective::Sync => false,
    };
    if arrivals.iter().any(absent) {
        return Err(Fault::undefined(
            "a collective's operand that this rank was not given",
        ));
    }
    let Some(first) = arrivals.first() else {
        return Ok(Vec::new());
    };
    let leader_only = |v: Value| -> Vec<Option<Value>> {
        let mut out = vec![Some(Value::Absent); n];
        if let Some(slot) = out.first_mut() {
            *slot = Some(v);
        }
        out
    };
    match first.clone() {
        Collective::Reduce { with, .. } | Collective::AllReduce { with, .. } => {
            let all = matches!(first, Collective::AllReduce { .. });
            let mut acc: Option<Value> = None;
            for c in arrivals {
                let x = match c {
                    Collective::Reduce { x, .. } | Collective::AllReduce { x, .. } => x,
                    _ => return Err(Fault::undefined("two kinds of collective at one meeting")),
                };
                acc = Some(match acc {
                    None => x,
                    Some(a) => combine(with, a, x, cx, budget)?,
                });
            }
            let v = acc.ok_or_else(|| Fault::undefined("a meeting of no member"))?;
            Ok(if all {
                vec![Some(v); n]
            } else {
                leader_only(v)
            })
        }
        Collective::Gather { .. } => {
            let mut tiles = Vec::with_capacity(n);
            for c in arrivals {
                match c {
                    Collective::Gather { x: Value::Tile(t) } => tiles.push(t),
                    _ => return Err(Fault::undefined("`gather` of what is not a tile")),
                }
            }
            Ok(leader_only(Value::Block(Block { tiles })))
        }
        Collective::Sync => Ok(vec![None; n]),
    }
}
