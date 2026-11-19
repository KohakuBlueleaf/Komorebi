//! Verify: the rules that need what a module does as a whole, read off the program
//! structure after Derive: the effects of each handler and run, and the chains'
//! dependencies. Each finding is placed through the notes' spans, at the instruction
//! the rule is about.
//!
//! Verify reads the structure and writes nothing; `api` turns its findings into
//! diagnostics.

use std::collections::HashMap;

use crate::program::body::{Body, InstId, Op, ValueId};
use crate::program::derive::{Access, Effects, StateUse};
use crate::program::notes::{Anchor, Owner, SourceSpan};
use crate::program::{BodyRef, CallArg, FuncKind, Module};
use crate::types::{
    BinOp, ChainId, ConstValue, ControlId, Entry, FuncId, HandlerId, ProgramId, RunId,
};

/// A stream that writes where it reads, on a view other than exactly the view read.
pub const STREAM_OVERLAP: &str = "K206";
/// A state member that some body reads and no body of the program fills.
pub const EMPTY_STATE: &str = "K301";
/// A fallback body that touches what `@reads` and `@writes` do not state.
pub const FALLBACK_TOUCHES: &str = "K205";
/// A `past` whose reads the build cannot show an earlier call wrote.
pub const PAST_UNPROVEN: &str = "K601";
/// A race: two invocations with no edge between them on the same tiles, one writing.
pub const RACE: &str = "K701";
/// A value a collective gives only the leader, used where other ranks run.
pub const LEADER_VALUE: &str = "K405";
/// A collective under a branch on the rank that not every member takes.
pub const COLLECTIVE_BRANCH: &str = "K406";

/// One rule a module breaks, and where.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    /// The rule's code.
    pub code: &'static str,
    /// What is wrong, in words.
    pub message: String,
    /// The place in the source it is about.
    pub at: SourceSpan,
}

/// The rules `module` (lowered and derived) breaks, in the order of the module.
pub fn verify(module: &Module) -> Vec<Finding> {
    let v = Verifier {
        module,
        spans: module.notes.spans.iter().copied().collect(),
    };
    let mut out = Vec::new();
    v.fallbacks(&mut out);
    for (p, prog) in module.programs.iter().enumerate() {
        let pid = ProgramId(p as u32);
        for (h, handler) in prog.handlers.iter().enumerate() {
            let owner = Owner::Handler(pid, HandlerId(h as u32));
            v.stream_overlap(&handler.effects, owner, &mut out);
            v.ranks(
                BodyRef::Handler(pid, HandlerId(h as u32)),
                &handler.body,
                &mut out,
            );
        }
        for (r, run) in prog.runs.iter().enumerate() {
            let owner = Owner::Run(pid, RunId(r as u32));
            v.stream_overlap(&run.effects, owner, &mut out);
        }
        v.empty_state(pid, &mut out);
        for c in 0..prog.chains.len() {
            v.past(pid, ChainId(c as u32), &mut out);
        }
    }
    for c in 0..module.controls.len() {
        v.races(ControlId(c as u32), &mut out);
    }
    out.sort_by_key(|f| (f.at.start, f.code));
    out
}

struct Verifier<'m> {
    module: &'m Module,
    spans: HashMap<Anchor, SourceSpan>,
}

impl Verifier<'_> {
    fn body(&self, r: BodyRef) -> Option<&Body> {
        let prog = |p: ProgramId| self.module.programs.get(p.0 as usize);
        match r {
            BodyRef::Func(f) => self.module.funcs.get(f.0 as usize)?.body.as_ref(),
            BodyRef::Handler(p, h) => Some(&prog(p)?.handlers.get(h.0 as usize)?.body),
            BodyRef::Seed(p, r) => Some(&prog(p)?.runs.get(r.0 as usize)?.seed),
            BodyRef::On(p, r, k) => Some(
                &prog(p)?
                    .runs
                    .get(r.0 as usize)?
                    .on_message
                    .get(k as usize)?
                    .1,
            ),
            BodyRef::Control(c) => Some(&self.module.controls.get(c.0 as usize)?.body),
        }
    }

    fn span(&self, r: BodyRef, at: InstId) -> SourceSpan {
        self.spans
            .get(&Anchor::Inst(r, at))
            .copied()
            .unwrap_or(SourceSpan { start: 0, end: 0 })
    }

    /// The name the source gives parameter `k` of `owner`.
    fn param_name(&self, owner: Owner, k: u32) -> String {
        let anchor = Anchor::ParamOf(owner, k);
        self.module
            .notes
            .param_names
            .iter()
            .find(|(a, _)| *a == anchor)
            .map_or_else(|| format!("parameter {k}"), |(_, n)| n.clone())
    }

    /// `K206`: a `push` to a sink, inside a loop that streams from a view of the same
    /// tensor, where the two views are not the same view and their tiles may meet.
    fn stream_overlap(&self, effects: &Effects, owner: Owner, out: &mut Vec<Finding>) {
        for w in effects.tensors.iter().filter(|a| a.access == Access::Sink) {
            let Some(body) = self.body(w.body) else {
                continue;
            };
            let values = Values::of(body);
            let overlap = effects.tensors.iter().any(|r| {
                r.access == Access::Stream
                    && r.body == w.body
                    && r.param == w.param
                    && inside(body, r.at, w.at)
                    && !values.exact(r.view, w.view)
                    && values.may_meet(r.view, w.view)
            });
            if overlap {
                out.push(Finding {
                    code: STREAM_OVERLAP,
                    message: format!(
                        "this stream writes `{}`, which overlaps the view it reads other than \
                         exactly",
                        self.param_name(owner, w.param)
                    ),
                    at: self.span(w.body, w.at),
                });
            }
        }
    }

    /// `K301`: a state member that a body of the program reads and that no body fills:
    /// whole, or element by element when it is `@filled(element)`. Reported at its
    /// first read.
    fn empty_state(&self, pid: ProgramId, out: &mut Vec<Finding>) {
        let Some(prog) = self.module.programs.get(pid.0 as usize) else {
            return;
        };
        let uses: Vec<_> = prog
            .handlers
            .iter()
            .map(|h| &h.effects)
            .chain(prog.runs.iter().map(|r| &r.effects))
            .flat_map(|e| e.state.iter())
            .collect();
        for (m, member) in prog.state.iter().enumerate() {
            let mine = || uses.iter().filter(|u| u.member.0 as usize == m);
            let filled = mine().any(|u| {
                u.how == StateUse::FillWhole
                    || (member.by_element && u.how == StateUse::WriteElement)
            });
            if filled {
                continue;
            }
            let first = mine()
                .filter(|u| u.how == StateUse::Read)
                .map(|u| self.span(u.body, u.at))
                .min_by_key(|s| s.start);
            if let Some(at) = first {
                out.push(Finding {
                    code: EMPTY_STATE,
                    message: format!(
                        "`{}` is read and no body of `{}` fills it (`.fill`, `.zero()`, or an \
                         element write to a member marked `@filled(element)`)",
                        member.name, prog.name
                    ),
                    at,
                });
            }
        }
    }
}

impl Verifier<'_> {
    /// `K601`: each tensor a call reads through `past(y)` was written, every tile of
    /// it, by an earlier call of the chain. The proof has two forms: the reader's view
    /// is the writer's own view (structurally equal instructions fed by the same
    /// arguments of the two calls), for every worker and every value of the
    /// arguments; or both views name a range of tiles the text fixes, with no
    /// partition, the reader's inside the writer's. What is not shown so is refused.
    fn past(&self, pid: ProgramId, cid: ChainId, out: &mut Vec<Finding>) {
        let Some(prog) = self.module.programs.get(pid.0 as usize) else {
            return;
        };
        let Some(chain) = prog.chains.get(cid.0 as usize) else {
            return;
        };
        let handler = |c: usize| {
            chain
                .calls
                .get(c)
                .and_then(|call| prog.handlers.get(call.handler.0 as usize))
        };
        for (c, call) in chain.calls.iter().enumerate() {
            let Some(reader) = handler(c) else {
                continue;
            };
            for (k, arg) in call.args.iter().enumerate() {
                let CallArg::Past(y) = arg else {
                    continue;
                };
                let reads = reader
                    .effects
                    .tensors
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.param == k as u32 && !writes(a.access));
                for (r, read) in reads {
                    let proven = chain.deps.edges.iter().any(|e| {
                        if e.reader as usize != c || e.read as usize != r || !e.past {
                            return false;
                        }
                        let Some(writer) = handler(e.writer as usize) else {
                            return false;
                        };
                        let Some(write) = writer.effects.tensors.get(e.write as usize) else {
                            return false;
                        };
                        let wcall = &chain.calls[e.writer as usize];
                        let (rv, wv) = (
                            Values::with_args(&reader.body, constants(&call.args)),
                            Values::with_args(&writer.body, constants(&wcall.args)),
                        );
                        let given = |i: u32, j: u32| match (
                            call.args.get(i as usize),
                            wcall.args.get(j as usize),
                        ) {
                            (Some(a), Some(b)) => given_alike(a, b),
                            _ => false,
                        };
                        let (Some(x), Some(z)) = (rv.made(read.view), wv.made(write.view)) else {
                            return false;
                        };
                        same(&rv, x, &wv, z, &given, false)
                            || match (rv.range_of(x), wv.range_of(z)) {
                                (Some(a), Some(b)) => {
                                    a.parts.is_empty()
                                        && b.parts.is_empty()
                                        && b.lo <= a.lo
                                        && a.hi <= b.hi
                                }
                                _ => false,
                            }
                    });
                    if !proven {
                        let owner = Owner::Chain(pid, cid);
                        out.push(Finding {
                            code: PAST_UNPROVEN,
                            message: format!(
                                "the build cannot show that every tile call {} (`{}`) reads of \
                                 `{}` through `past` was written by an earlier call",
                                c + 1,
                                reader.name,
                                self.param_name(owner, *y)
                            ),
                            at: self.call_span(pid, cid, c as u32),
                        });
                    }
                }
            }
        }
    }

    fn call_span(&self, pid: ProgramId, cid: ChainId, c: u32) -> SourceSpan {
        self.spans
            .get(&Anchor::Call(pid, cid, c))
            .copied()
            .unwrap_or(SourceSpan { start: 0, end: 0 })
    }
}

/// One invocation of a handler in a control program, as far as the text fixes it.
struct Invocation<'m> {
    /// The instruction.
    at: InstId,
    /// The handler, when a handler is invoked.
    handler: Option<&'m crate::program::Handler>,
    /// The arguments: values of the control program.
    args: Vec<ValueId>,
    /// The branches it stands in: each `if` by its instruction, and whether it is in
    /// the `else`.
    arms: Vec<(usize, bool)>,
}

impl Invocation<'_> {
    /// Whether the two can both run in one pass: not when they stand on the two arms
    /// of one `if`.
    fn together(&self, other: &Invocation<'_>) -> bool {
        !self
            .arms
            .iter()
            .any(|(k, arm)| other.arms.iter().any(|(j, other)| k == j && arm != other))
    }
}

impl Verifier<'_> {
    /// `K701`, where it can be shown before the run: two invocations of a control
    /// program with no edge between them (`after`, `after_end`, or an `await` of the
    /// one before the other is invoked, or a path of such) that touch the same tiles
    /// of one tensor, one of them writing. The tiles of an invocation are those of all
    /// its workers: a view `partition(v, worker.rank, group.size)` covers `v`. Two
    /// invocations on the two arms of one `if` never both run, and are no race. What the
    /// text does not fix is left to the run: an invocation in a loop against itself on
    /// the next turn, and an `await` on one arm of an `if`, which counts here as an edge
    /// on every path (so the build may miss a race there, and the run finds it).
    fn races(&self, cid: ControlId, out: &mut Vec<Finding>) {
        let Some(control) = self.module.controls.get(cid.0 as usize) else {
            return;
        };
        let body = &control.body;
        let cv = Values::of(body);
        let mut invocations = Vec::new();
        // the invocations each one comes after, directly
        let mut before: Vec<Vec<usize>> = Vec::new();
        let mut awaited: Vec<usize> = Vec::new();
        let place = |invocations: &[Invocation], v: &ValueId| {
            invocations
                .iter()
                .position(|i| body.insts[i.at.0 as usize].defs.first() == Some(v))
        };
        // the constructs open around each instruction: an `if` (by its instruction, and
        // whether in its `else`), or another construct
        let mut open: Vec<Option<(usize, bool)>> = Vec::new();
        for (i, inst) in body.insts.iter().enumerate() {
            match &inst.op {
                Op::If { .. } => open.push(Some((i, false))),
                Op::Else => {
                    if let Some(Some((k, _))) = open.pop() {
                        open.push(Some((k, true)));
                    }
                }
                Op::ForRange { .. }
                | Op::ForStream { .. }
                | Op::ForZip { .. }
                | Op::ForArrive { .. }
                | Op::While => open.push(None),
                Op::End => {
                    open.pop();
                }
                _ => {}
            }
            match &inst.op {
                Op::Invoke {
                    entry,
                    args,
                    after,
                    after_end,
                    ..
                } => {
                    let mut edges: Vec<usize> = after
                        .iter()
                        .chain(after_end)
                        .filter_map(|v| place(&invocations, v))
                        .collect();
                    edges.extend(awaited.iter().copied());
                    before.push(edges);
                    let handler = match entry {
                        Entry::Handler(p, h) => self
                            .module
                            .programs
                            .get(p.0 as usize)
                            .and_then(|prog| prog.handlers.get(h.0 as usize)),
                        _ => None,
                    };
                    invocations.push(Invocation {
                        at: InstId(i as u32),
                        handler,
                        args: args.clone(),
                        arms: open.iter().flatten().copied().collect(),
                    });
                }
                Op::Await(h) => awaited.extend(place(&invocations, h)),
                _ => {}
            }
        }
        let ordered = |a: usize, b: usize| -> bool {
            // whether a is before b by a path of edges
            let mut stack = vec![b];
            let mut seen = vec![false; before.len()];
            while let Some(x) = stack.pop() {
                for &y in &before[x] {
                    if y == a {
                        return true;
                    }
                    if !seen[y] {
                        seen[y] = true;
                        stack.push(y);
                    }
                }
            }
            false
        };
        for (b, later) in invocations.iter().enumerate() {
            let Some(handler) = later.handler else {
                continue;
            };
            let raced = (0..b).find(|&a| {
                let earlier = &invocations[a];
                earlier.together(later) && !ordered(a, b) && self.meet(&cv, earlier, later)
            });
            if let Some(a) = raced {
                let earlier = invocations[a].handler.map_or("", |x| x.name.as_str());
                out.push(Finding {
                    code: RACE,
                    message: format!(
                        "a race: this invocation of `{}` and the earlier one of `{earlier}` touch \
                         the same tiles of a tensor, one of them writing, with no edge between \
                         them (`after`, `after_end`, or an `await` before)",
                        handler.name
                    ),
                    at: self.span(BodyRef::Control(cid), later.at),
                });
            }
        }
    }

    /// Whether two invocations touch the same tiles of one tensor, one writing.
    fn meet(&self, cv: &Values<'_>, a: &Invocation<'_>, b: &Invocation<'_>) -> bool {
        let (Some(ha), Some(hb)) = (a.handler, b.handler) else {
            return false;
        };
        let tiles = |inv: &Invocation<'_>, body: &Body, view: InstId| -> Option<(u64, u64)> {
            let values = Values::with_args(body, inv.args.iter().map(|v| cv.number(*v)).collect());
            let r = values.range_of(values.made(view)?)?;
            let whole = r.parts.iter().all(|(rank, size)| {
                matches!(values.op(*rank), Some(Op::Rank))
                    && matches!(values.op(*size), Some(Op::GroupSize))
            });
            whole.then_some((r.lo, r.hi))
        };
        ha.effects.tensors.iter().any(|x| {
            hb.effects.tensors.iter().any(|y| {
                if !(writes(x.access) || writes(y.access)) {
                    return false;
                }
                let (Some(tx), Some(ty)) =
                    (a.args.get(x.param as usize), b.args.get(y.param as usize))
                else {
                    return false;
                };
                if !cv.same(*tx, *ty) {
                    return false;
                }
                match (tiles(a, &ha.body, x.view), tiles(b, &hb.body, y.view)) {
                    (Some((l1, h1)), Some((l2, h2))) => l1 < h2 && l2 < h1,
                    _ => false,
                }
            })
        })
    }
}

impl Verifier<'_> {
    /// `K205`: an external function's fallback touches only what `@reads` and
    /// `@writes` state: it writes only what `@writes` states, and reads only what one
    /// of the two states.
    fn fallbacks(&self, out: &mut Vec<Finding>) {
        for (f, func) in self.module.funcs.iter().enumerate() {
            let FuncKind::Extern {
                reads,
                writes: written,
                ..
            } = &func.kind
            else {
                continue;
            };
            let id = FuncId(f as u32);
            for a in &func.effects.tensors {
                let name = || self.param_name(Owner::Func(id), a.param);
                let message = if writes(a.access) {
                    (!written.contains(&a.param)).then(|| {
                        format!(
                            "the fallback writes `{}`, which `@writes` does not state",
                            name()
                        )
                    })
                } else {
                    (!written.contains(&a.param) && !reads.contains(&a.param)).then(|| {
                        format!(
                            "the fallback reads `{}`, which neither `@reads` nor `@writes` states",
                            name()
                        )
                    })
                };
                if let Some(message) = message {
                    out.push(Finding {
                        code: FALLBACK_TOUCHES,
                        message,
                        at: self.span(a.body, a.at),
                    });
                }
            }
        }
    }
}

/// One construct around an instruction: a branch, with its condition and whether
/// the instruction is in its `else`, or a loop.
#[derive(Clone, Copy)]
enum Frame {
    Branch(ValueId, bool),
    Loop,
}

impl Verifier<'_> {
    /// The two rules on ranks, where the body shows them. `K405`: a value `reduce` or
    /// `gather` gives only the leader is used where a rank that is not the leader runs:
    /// outside every branch on the rank, or in the `else` of `worker.is_leader`; as the
    /// operand of a `reduce ... among` it is used by the leaders only. `K406`: a
    /// collective inside a branch whose condition depends on the rank.
    fn ranks(&self, r: BodyRef, body: &Body, out: &mut Vec<Finding>) {
        let values = Values::of(body);
        let mut frames: Vec<Frame> = Vec::new();
        let mut leader_only: HashMap<ValueId, InstId> = HashMap::new();
        for (i, inst) in body.insts.iter().enumerate() {
            let at = InstId(i as u32);
            let on_rank = |f: &Frame| match f {
                Frame::Branch(c, _) => values.on_rank(*c),
                Frame::Loop => false,
            };
            let non_leader_runs = !frames.iter().any(on_rank)
                || frames.iter().any(|f| {
                    matches!(f, Frame::Branch(c, true) if matches!(values.op(*c), Some(Op::IsLeader)))
                });
            // the operands this instruction uses
            let used: Vec<ValueId> = match &inst.op {
                Op::ReduceGroup {
                    x,
                    among: Some(_),
                    within,
                    publish,
                    ..
                } => within
                    .iter()
                    .chain(publish)
                    .copied()
                    .filter(|v| v != x)
                    .collect(),
                op => crate::program::validate::operands(op),
            };
            if non_leader_runs {
                for v in used {
                    if leader_only.remove(&v).is_some() {
                        out.push(Finding {
                            code: LEADER_VALUE,
                            message: "this uses a value that `reduce` or `gather` gives only the \
                                      leader, where other ranks run: use it inside \
                                      `if worker.is_leader`"
                                .into(),
                            at: self.span(r, at),
                        });
                    }
                }
            }
            match &inst.op {
                Op::ReduceGroup { .. } | Op::Gather(_) => {
                    if let Some(v) = inst.defs.first() {
                        leader_only.insert(*v, at);
                    }
                }
                _ => {}
            }
            let collective = matches!(
                inst.op,
                Op::ReduceGroup { .. }
                    | Op::AllReduce { .. }
                    | Op::Gather(_)
                    | Op::Broadcast(..)
                    | Op::Sync(_)
            );
            if collective && frames.iter().any(on_rank) {
                out.push(Finding {
                    code: COLLECTIVE_BRANCH,
                    message: "every member of the group takes part in a collective: this one is \
                              under a branch on the rank, which not every member takes"
                        .into(),
                    at: self.span(r, at),
                });
            }
            match &inst.op {
                Op::If { cond } => frames.push(Frame::Branch(*cond, false)),
                Op::Else => {
                    if let Some(Frame::Branch(c, _)) = frames.pop() {
                        frames.push(Frame::Branch(c, true));
                    }
                }
                Op::ForRange { .. }
                | Op::ForStream { .. }
                | Op::ForZip { .. }
                | Op::ForArrive { .. }
                | Op::While => frames.push(Frame::Loop),
                Op::End => {
                    frames.pop();
                }
                _ => {}
            }
        }
    }
}

/// Whether an access writes.
fn writes(a: Access) -> bool {
    matches!(
        a,
        Access::Sink | Access::Store | Access::Put | Access::ExternWrite
    )
}

/// The constants a call gives its handler's parameters, by position.
fn constants(args: &[CallArg]) -> Vec<Option<u64>> {
    args.iter()
        .map(|a| match a {
            CallArg::Const(ConstValue::Int(_, n)) => Some(*n),
            _ => None,
        })
        .collect()
}

/// Whether two arguments of a chain's calls give the same value: the same parameter of
/// the chain (`past` or not), or the same constant.
fn given_alike(a: &CallArg, b: &CallArg) -> bool {
    let param = |x: &CallArg| match x {
        CallArg::Param(p) | CallArg::Past(p) => Some(*p),
        CallArg::Const(_) => None,
    };
    match (a, b) {
        (CallArg::Const(x), CallArg::Const(y)) => x == y,
        _ => param(a).is_some() && param(a) == param(b),
    }
}

/// Whether instruction `at` is inside the loop `open` opens.
fn inside(body: &Body, open: InstId, at: InstId) -> bool {
    let mut depth = 0u32;
    for (i, inst) in body.insts.iter().enumerate().skip(open.0 as usize) {
        if i == at.0 as usize {
            return depth > 0;
        }
        match inst.op {
            Op::If { .. }
            | Op::ForRange { .. }
            | Op::ForStream { .. }
            | Op::ForZip { .. }
            | Op::ForArrive { .. }
            | Op::While => depth += 1,
            Op::End => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    false
}

/// The values of one body, by the instruction that defines each, with the constants
/// its parameters are given where they are known.
struct Values<'b> {
    body: &'b Body,
    def: HashMap<ValueId, usize>,
    args: Vec<Option<u64>>,
}

/// The tiles a view names, where the text fixes them: the range `lo..hi` of a
/// `t.tiles(first, n)` with both numbers known, inside the partitions written around
/// it (each a rank and a size, outermost last).
struct Range {
    lo: u64,
    hi: u64,
    parts: Vec<(ValueId, ValueId)>,
}

impl<'b> Values<'b> {
    fn of(body: &'b Body) -> Values<'b> {
        let mut def = HashMap::new();
        for (i, inst) in body.insts.iter().enumerate() {
            for v in &inst.defs {
                def.insert(*v, i);
            }
        }
        Values {
            body,
            def,
            args: Vec::new(),
        }
    }

    /// The values of `body`, its parameters given the constants `args` (by position).
    fn with_args(body: &'b Body, args: Vec<Option<u64>>) -> Values<'b> {
        Values {
            args,
            ..Values::of(body)
        }
    }

    fn op(&self, v: ValueId) -> Option<&'b Op> {
        self.def.get(&v).map(|i| &self.body.insts[*i].op)
    }

    /// The value the instruction `at` defines.
    fn made(&self, at: InstId) -> Option<ValueId> {
        self.body.insts.get(at.0 as usize)?.defs.first().copied()
    }

    /// Whether the views two instructions make are the same view: the same value, or
    /// structurally equal instructions fed by the same values.
    fn exact(&self, a: InstId, b: InstId) -> bool {
        match (self.made(a), self.made(b)) {
            (Some(x), Some(y)) => self.same(x, y),
            _ => false,
        }
    }

    /// Whether two values are equal by how they are made: the same value, or the
    /// same pure operation on values equal so.
    fn same(&self, x: ValueId, y: ValueId) -> bool {
        same(self, x, self, y, &|i, j| i == j, true)
    }

    /// Whether a value depends on the rank, as far as the body shows: a value of the
    /// worker's identity, or an operation on one.
    fn on_rank(&self, v: ValueId) -> bool {
        match self.op(v) {
            Some(Op::Rank | Op::IsLeader | Op::WorkerId | Op::Row | Op::Col) => true,
            Some(
                op @ (Op::Binary(..)
                | Op::Compare(..)
                | Op::LogicAnd(..)
                | Op::LogicOr(..)
                | Op::Unary(..)
                | Op::Convert(..)),
            ) => crate::program::validate::operands(op)
                .into_iter()
                .any(|x| self.on_rank(x)),
            _ => false,
        }
    }

    /// A whole number the body fixes: a constant, or an operation on such.
    fn number(&self, v: ValueId) -> Option<u64> {
        match self.op(v)? {
            Op::Const(ConstValue::Int(_, n)) => Some(*n),
            Op::Arg(i) => self.args.get(*i as usize).copied().flatten(),
            Op::Binary(op, a, b) => {
                let (a, b) = (self.number(*a)?, self.number(*b)?);
                match op {
                    BinOp::Add => a.checked_add(b),
                    BinOp::Sub => a.checked_sub(b),
                    BinOp::Mul => a.checked_mul(b),
                    BinOp::Div => a.checked_div(b),
                    BinOp::Rem => a.checked_rem(b),
                    BinOp::Min => Some(a.min(b)),
                    BinOp::Max => Some(a.max(b)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The tiles the view an instruction makes names, where the text fixes them.
    fn range(&self, at: InstId) -> Option<Range> {
        self.range_of(self.made(at)?)
    }

    fn range_of(&self, v: ValueId) -> Option<Range> {
        match self.op(v)? {
            Op::Tiles { first, n, .. } => {
                let lo = self.number(*first)?;
                let hi = lo.checked_add(self.number(*n)?)?;
                Some(Range {
                    lo,
                    hi,
                    parts: Vec::new(),
                })
            }
            Op::Partition(inner, rank, size) => {
                let mut r = self.range_of(*inner)?;
                r.parts.push((*rank, *size));
                Some(r)
            }
            _ => None,
        }
    }

    /// Whether the tiles of two views may meet: they do not only where both ranges
    /// are fixed, under the same partitions, and apart.
    fn may_meet(&self, a: InstId, b: InstId) -> bool {
        let (Some(x), Some(y)) = (self.range(a), self.range(b)) else {
            return true;
        };
        let same_parts = x.parts.len() == y.parts.len()
            && x.parts
                .iter()
                .zip(&y.parts)
                .all(|((r1, s1), (r2, s2))| self.same(*r1, *r2) && self.same(*s1, *s2));
        !same_parts || (x.lo < y.hi && y.lo < x.hi)
    }
}

/// Whether value `x` of one body and value `y` of another (or the same) are equal by
/// how they are made: the same value, or the same pure operation on values equal so.
/// `args` says which parameters of the two bodies are given the same value; `one`, that
/// both are the same run of one body, where a value is equal to itself.
fn same(
    a: &Values<'_>,
    x: ValueId,
    b: &Values<'_>,
    y: ValueId,
    args: &dyn Fn(u32, u32) -> bool,
    one: bool,
) -> bool {
    if one && x == y {
        return true;
    }
    let (Some(p), Some(q)) = (a.op(x), b.op(y)) else {
        return false;
    };
    let s = |u: &ValueId, v: &ValueId| same(a, *u, b, *v, args, one);
    let range = |r: &crate::program::body::Range, t: &crate::program::body::Range| {
        s(&r.lo, &t.lo)
            && s(&r.hi, &t.hi)
            && match (r.step, t.step) {
                (None, None) => true,
                (Some(u), Some(v)) => s(&u, &v),
                _ => false,
            }
    };
    match (p, q) {
        (Op::Const(u), Op::Const(v)) => u == v,
        (Op::Arg(i), Op::Arg(j)) => args(*i, *j),
        (Op::Param(u), Op::Param(v)) => u == v,
        (Op::Rank, Op::Rank) | (Op::GroupSize, Op::GroupSize) => true,
        (Op::Binary(o, a1, a2), Op::Binary(p, b1, b2)) => o == p && s(a1, b1) && s(a2, b2),
        (
            Op::Tiles {
                tensor: t1,
                first: f1,
                n: n1,
            },
            Op::Tiles {
                tensor: t2,
                first: f2,
                n: n2,
            },
        ) => s(t1, t2) && s(f1, f2) && s(n1, n2),
        (
            Op::Rect {
                tensor: t1,
                rows: r1,
                cols: c1,
            },
            Op::Rect {
                tensor: t2,
                rows: r2,
                cols: c2,
            },
        ) => s(t1, t2) && range(r1, r2) && range(c1, c2),
        (Op::Turn(u), Op::Turn(v)) => s(u, v),
        (
            Op::Items {
                tensor: t1,
                range: g1,
                size: z1,
            },
            Op::Items {
                tensor: t2,
                range: g2,
                size: z2,
            },
        ) => {
            s(t1, t2)
                && s(z1, z2)
                && match (g1, g2) {
                    (None, None) => true,
                    (Some((u1, u2)), Some((v1, v2))) => s(u1, v1) && s(u2, v2),
                    _ => false,
                }
        }
        (Op::Partition(v1, r1, z1), Op::Partition(v2, r2, z2)) => {
            s(v1, v2) && s(r1, r2) && s(z1, z2)
        }
        _ => false,
    }
}
