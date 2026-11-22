//! A loaded module and what runs on it: the invocations a host makes (a handler, a
//! chain, or a dynamic run in rounds of messages), in the order it makes them, each
//! run whole before the next; the control programs, whose invocations run in the
//! order the program makes them; and the run's report.
//!
//! An invocation of a handler runs its body once on every worker of its group. The
//! workers' frames are stepped one after another in rank order; each runs until it
//! returns or faults. The invocation's end is its leader's result, or the failure of
//! the lowest rank that faulted. A tensor write goes straight to the heap, and every
//! read and write of a place goes through the race record (`memory::Races`): an access
//! with no edge of B1 to an earlier one on the same place, one of them a write, is a
//! race, an invalid program, and the run stops there.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};

use super::delivery;
use super::memory::{
    Failure, Fault, Filled, Format, Heap, MemberData, Races, Scalar, Tensor, TensorId, TensorPlace,
    Tile,
};
use super::ops;
use super::workers::{
    bodies, constant, Accesses, Collective, Context, Frame, IslandMap, Letter, Lines, Mail, Marks,
    Outcome, OverCopies, Place, Request, Shape, Shapes, Stop, Thread, Value, Worker,
};
use crate::program::body::InstId;
use crate::program::{BodyRef, CallArg, Module};
use crate::types::{
    Ability, Binding, ChainId, Combiner, ConstValue, ControlId, Entry, HandlerId, On, ProgramId,
    RunId, Type,
};
use crate::types::{IslandGraph, IslandRefusal, NOT_A_MACHINE};

/// The options of a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    /// How many instructions one invocation may run on each worker before it gives up
    /// by name (`given_up`).
    pub step_budget: u64,
    /// The bytes of a worker's memory, which a `fit` parameter fills.
    pub worker_memory: u64,
    /// The bound on a dynamic run's new work: a `send` while this many messages are
    /// live is refused. None: no bound.
    pub live_bound: Option<u64>,
    /// The islands of the run. None: one island holding every worker.
    pub islands: Option<IslandGraph>,
}

/// The island map of a graph: each worker's island, and the hops of a shortest path
/// between every two islands (`IslandGraph::hops`).
fn island_map(g: &IslandGraph, workers: usize) -> Result<IslandMap, IslandRefusal> {
    let hops = g.hops(workers)?;
    let of = g
        .sizes
        .iter()
        .enumerate()
        .flat_map(|(i, s)| std::iter::repeat_n(i, *s))
        .collect();
    Ok(IslandMap { of, hops })
}

impl Default for Options {
    fn default() -> Options {
        Options {
            step_budget: 1 << 32,
            worker_memory: 1 << 20,
            live_bound: None,
            islands: None,
        }
    }
}

/// Why the interpreter refuses to load a module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A requirement the interpreter does not meet, as the text states it.
    Requirement(String),
    /// A load parameter with no value, by name.
    Unbound(String),
    /// A parameter's value that does not fit it, by name.
    BadValue(String),
    /// The structure is not one the interpreter can run.
    Invalid(String),
    /// The run's islands, by why.
    Islands(IslandRefusal),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Requirement(r) => write!(f, "requirement not met: {r}"),
            Refusal::Unbound(p) => write!(f, "load parameter with no value: {p}"),
            Refusal::BadValue(p) => write!(f, "a value that does not fit the parameter {p}"),
            Refusal::Invalid(w) => write!(f, "not a program the interpreter runs: {w}"),
            Refusal::Islands(why) => write!(f, "{NOT_A_MACHINE}: {}", why.sentence()),
        }
    }
}

/// An argument a host gives an invocation.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    /// A tensor of the heap.
    Tensor(TensorId),
    /// A scalar.
    Scalar(Scalar),
}

/// How an invocation ended.
#[derive(Clone, Debug, PartialEq)]
pub enum End {
    /// Completed, with the leader's result if the handler has one.
    Completed(Option<Scalar>),
    /// Failed by a named reason, at a rank and an instruction (none when it failed
    /// before any body ran).
    Failed {
        /// The reason.
        reason: Failure,
        /// The rank that failed, and where.
        at: Option<(usize, BodyRef, InstId)>,
    },
    /// Not defined: what the program did is not a program (program compile refuses
    /// every one it can show), at a rank and an instruction.
    Invalid {
        /// What.
        what: String,
        /// The rank, and where.
        at: Option<(usize, BodyRef, InstId)>,
    },
}

/// One line of the run's report: an invocation, or a control program, and its end.
#[derive(Clone, Debug, PartialEq)]
pub struct Ended {
    /// The line's number in the run, from 0.
    pub number: usize,
    /// What it named, as `Program.handler`, or `control name`.
    pub entry: String,
    /// The first worker and the count of its group; none for a control program.
    pub group: Option<(usize, usize)>,
    /// How it ended.
    pub end: End,
}

/// A run's report: each invocation's end, in the order they ended, and what the host
/// asked it to show (a tensor, an exported state member), where it asked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// The invocations' ends.
    pub ended: Vec<Ended>,
    /// What the host asked to show.
    pub shown: Vec<Shown>,
}

/// A tensor or an exported state member, as it was when the host asked to see it.
#[derive(Clone, Debug, PartialEq)]
pub struct Shown {
    /// How many ends the report had when it was asked for: it prints after them.
    pub after: usize,
    /// Its heading: `tensor NAME FORMAT [ROWS, COLS]` or `state Program.member`.
    pub heading: String,
    /// One line for each tile (`tile i: ...`) or each worker (`worker id: ...`).
    pub lines: Vec<String>,
}

/// Elements as text, in order, a run of one value written once with its count
/// (`0.0*1024`), `_` for an element not filled.
fn run_text(elements: impl IntoIterator<Item = String>) -> String {
    let mut runs: Vec<(String, usize)> = Vec::new();
    for e in elements {
        match runs.last_mut() {
            Some((last, n)) if *last == e => *n += 1,
            _ => runs.push((e, 1)),
        }
    }
    runs.iter()
        .map(|(e, n)| {
            if *n == 1 {
                e.clone()
            } else {
                format!("{e}*{n}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// A tile's elements as text: an integer as itself, a float as the shortest text
/// that reads back to its value as the interpreter holds it (`f32`, or `f64`).
fn tile_text(tile: &Tile) -> String {
    if let Some(ints) = tile.to_ints() {
        return run_text(ints.into_iter().map(|v| v.to_string()));
    }
    let wide = tile.format == Format::F64;
    match tile.to_f64() {
        Some(v) => run_text(v.into_iter().map(|x| {
            if wide {
                format!("{x:?}")
            } else {
                format!("{:?}", x as f32)
            }
        })),
        None => "?".into(),
    }
}

fn place_text(at: Option<(usize, BodyRef, InstId)>) -> String {
    match at {
        None => String::new(),
        Some((rank, body, inst)) => {
            format!(
                " at rank {rank}, {} i{}",
                crate::program::print::body_ref(&body),
                inst.0
            )
        }
    }
}

/// A scalar as the report writes it.
pub fn scalar_text(s: &Scalar) -> String {
    match *s {
        Scalar::Int(f, bits) => format!("{}:{}", super::memory::int_value(f, bits), f.name()),
        Scalar::F32(f, v) => format!("{v:?}:{}", f.name()),
        Scalar::F64(v) => format!("{v:?}:f64"),
        Scalar::Bool(b) => b.to_string(),
    }
}

impl Report {
    /// The report's printed form: one line for each invocation, and each thing shown
    /// as its heading and its lines indented by two spaces, where it was asked for.
    pub fn print(&self) -> String {
        let mut out = String::new();
        let show = |out: &mut String, at: usize| {
            for s in self.shown.iter().filter(|s| s.after == at) {
                let _ = writeln!(out, "{}", s.heading);
                for line in &s.lines {
                    let _ = writeln!(out, "  {line}");
                }
            }
        };
        for (i, e) in self.ended.iter().enumerate() {
            show(&mut out, i);
            let end = match &e.end {
                End::Completed(None) => "completed".to_string(),
                End::Completed(Some(v)) => format!("completed {}", scalar_text(v)),
                End::Failed { reason, at } => {
                    format!("failed {}{}", reason.name(), place_text(*at))
                }
                End::Invalid { what, at } => {
                    format!("invalid: {what}{}", place_text(*at))
                }
            };
            let group = match e.group {
                Some((first, count)) => format!(" group({first}, {count})"),
                None => String::new(),
            };
            let _ = writeln!(out, "#{} {}{group}: {end}", e.number, e.entry);
        }
        show(&mut out, self.ended.len());
        out
    }
}

/// A module loaded on the interpreter's workers, with its tensors.
pub struct Machine<'m> {
    module: &'m Module,
    shapes: Shapes,
    params: Vec<Option<Scalar>>,
    /// The tensors.
    pub heap: Heap,
    /// The workers of the loaded program, by `worker.id`.
    pub workers: Vec<Worker>,
    options: Options,
    /// What has run so far.
    pub report: Report,
    /// The record that finds races, over every invocation of the run.
    races: Races,
    /// The islands of the run.
    islands: IslandMap,
    /// The copies of each replicated tensor, one for each island, by its copy 0.
    copies: BTreeMap<usize, Vec<TensorId>>,
    /// The places of each replicated tensor's copies written since they were last
    /// equal, with the islands that wrote them.
    marks: Marks,
}

/// The bytes the interpreter holds an element of a format in: every float format but
/// `f64` is held as `f32`.
fn held_bytes(f: Format) -> u64 {
    match super::memory::held(f) {
        super::memory::Held::F32 => 4,
        super::memory::Held::F64 => 8,
        super::memory::Held::Int => u64::from(f.bits().unwrap_or(64) / 8).max(1),
    }
}

fn scalar_of(c: ConstValue) -> Result<Scalar, Refusal> {
    constant(c).map_err(|f| Refusal::Invalid(f.to_string()))
}

impl<'m> Machine<'m> {
    /// Loads `module` on `workers` workers. `bindings` give load parameters their
    /// values by name; a `fit` parameter that is not given takes the largest value
    /// with which every program's state fits `options.worker_memory`.
    pub fn load(
        module: &'m Module,
        workers: usize,
        bindings: &[(&str, Scalar)],
        options: Options,
    ) -> Result<Machine<'m>, Refusal> {
        let islands = match &options.islands {
            None => IslandMap::one(workers),
            Some(g) => island_map(g, workers).map_err(Refusal::Islands)?,
        };
        for r in &module.requirements {
            match &r.ability {
                Ability::Impl(name) => {
                    let has_fallback = module.funcs.iter().any(|f| {
                        f.body.is_some()
                            && matches!(&f.kind, crate::program::FuncKind::Extern { implementation, .. } if implementation == name)
                    });
                    if !has_fallback {
                        return Err(Refusal::Requirement(format!("impl(\"{name}\")")));
                    }
                }
                Ability::Islands(n) if (islands.count() as u64) < *n => {
                    return Err(Refusal::Requirement(format!("islands >= {n}")));
                }
                _ => {}
            }
        }
        let mut params: Vec<Option<Scalar>> = Vec::with_capacity(module.params.len());
        let mut fits = Vec::new();
        for (i, p) in module.params.iter().enumerate() {
            let given = bindings.iter().find(|(n, _)| *n == p.name).map(|(_, v)| *v);
            let value = match p.binding {
                Binding::Build(c) => Some(scalar_of(c)?),
                Binding::Load(declared) => match (given, declared) {
                    (Some(v), _) => Some(v),
                    (None, Some(c)) => Some(scalar_of(c)?),
                    (None, None) => {
                        fits.push(i);
                        None
                    }
                },
                Binding::Invoke(c) => Some(scalar_of(c)?),
            };
            if let Some(v) = value {
                if v.format() != Some(p.ty.format()) {
                    return Err(Refusal::BadValue(p.name.clone()));
                }
            }
            params.push(value);
        }
        for i in fits {
            let n = Machine::fit(module, &params, i, options.worker_memory)?;
            if let (Some(slot), Some(p)) = (params.get_mut(i), module.params.get(i)) {
                *slot = Scalar::int(p.ty.format(), i128::from(n));
            }
        }
        let mut shapes = Shapes::new();
        for (at, body) in bodies(module) {
            let shape = Shape::of(body).map_err(|f| Refusal::Invalid(f.to_string()))?;
            shapes.insert(at, shape);
        }
        let mut list = Vec::with_capacity(workers);
        for id in 0..workers {
            list.push(
                Worker::load(id, module, &params).map_err(|f| Refusal::Invalid(f.to_string()))?,
            );
        }
        Ok(Machine {
            module,
            shapes,
            params,
            heap: Heap::default(),
            workers: list,
            options,
            report: Report::default(),
            races: Races::default(),
            islands,
            copies: BTreeMap::new(),
            marks: Marks::default(),
        })
    }

    /// The largest value of the `fit` parameter `i` with which every program's state
    /// fits `memory` bytes.
    fn fit(
        module: &Module,
        params: &[Option<Scalar>],
        i: usize,
        memory: u64,
    ) -> Result<u64, Refusal> {
        let tile = module.tile.rows.saturating_mul(module.tile.cols);
        let mut worst: Option<u64> = None;
        for program in &module.programs {
            let (mut fixed, mut per) = (0u64, 0u64);
            for m in &program.state {
                let each = match &m.kind {
                    crate::program::MemberKind::Tiles(f) => tile.saturating_mul(held_bytes(*f)),
                    crate::program::MemberKind::Scalars(s) => held_bytes(s.format()),
                };
                match m.count {
                    crate::types::Count::Param(p) if p.0 as usize == i => {
                        per = per.saturating_add(each)
                    }
                    c => {
                        let n = super::workers::count(c, params)
                            .map_err(|f| Refusal::Invalid(f.to_string()))?;
                        fixed = fixed.saturating_add(each.saturating_mul(n as u64));
                    }
                }
            }
            if let Some(n) = memory.saturating_sub(fixed).checked_div(per) {
                worst = Some(worst.map_or(n, |w| w.min(n)));
            }
        }
        let largest = module
            .params
            .get(i)
            .and_then(|p| p.ty.format().bits())
            .map_or(u64::MAX, super::memory::mask);
        match worst {
            // no state is counted by it: its type's largest value
            None => Ok(largest),
            Some(n) if n > 0 => Ok(n.min(largest)),
            Some(_) => Err(Refusal::Requirement(
                "local_state: no room for the state".into(),
            )),
        }
    }

    /// The module loaded.
    pub fn module(&self) -> &'m Module {
        self.module
    }

    /// Puts a tensor on the heap.
    pub fn tensor(&mut self, tensor: Tensor) -> TensorId {
        self.heap.add(tensor)
    }

    /// Puts a replicated tensor on the heap: one copy for each island, each holding
    /// `tensor` at first. Gives copy 0, which names the tensor in an argument.
    pub fn replicated(&mut self, tensor: Tensor) -> TensorId {
        let ids: Vec<TensorId> = (0..self.islands.count())
            .map(|_| self.heap.add(tensor.clone()))
            .collect();
        let first = ids.first().copied().unwrap_or(TensorId(usize::MAX));
        self.marks.copies(&ids);
        self.copies.insert(first.0, ids);
        first
    }

    /// The copy of island `island` of the replicated tensor `t` (named by copy 0).
    pub fn copy(&self, t: TensorId, island: usize) -> Option<TensorId> {
        self.copies.get(&t.0).and_then(|c| c.get(island)).copied()
    }

    /// Shows tensor `t` in the report as it is now: each tile's elements. A replicated
    /// tensor (named by its copy 0) is shown copy by copy.
    pub fn show_tensor(&mut self, t: TensorId) -> Result<(), String> {
        let copies = self.copies.get(&t.0).cloned();
        let each: Vec<(TensorId, Option<usize>)> = match &copies {
            Some(c) => c.iter().enumerate().map(|(i, x)| (*x, Some(i))).collect(),
            None => vec![(t, None)],
        };
        for (id, copy) in each {
            let tensor = self.heap.get(id).map_err(|f| f.to_string())?;
            let copy = copy.map(|i| format!(" copy {i}")).unwrap_or_default();
            let heading = format!(
                "tensor {}{copy} {} [{}, {}]",
                tensor.name,
                tensor.format.name(),
                tensor.shape.0,
                tensor.shape.1
            );
            let lines = tensor
                .tiles
                .iter()
                .enumerate()
                .map(|(i, tile)| format!("tile {i}: {}", tile_text(tile)))
                .collect();
            let after = self.report.ended.len();
            self.report.shown.push(Shown {
                after,
                heading,
                lines,
            });
        }
        Ok(())
    }

    /// Shows the exported state member `Program.member` in the report as it is now,
    /// worker by worker. An element not filled of a member filled element by element
    /// is `_`; a member filled whole that is not filled shows what it holds, followed
    /// by `(not filled)`.
    pub fn show_state(&mut self, name: &str) -> Result<(), String> {
        let (program, member) = name
            .split_once('.')
            .ok_or_else(|| format!("{name} is not Program.member"))?;
        let (p, prog) = self
            .module
            .programs
            .iter()
            .enumerate()
            .find(|(_, x)| x.name == program)
            .ok_or_else(|| format!("no program {program}"))?;
        let (m, decl) = prog
            .state
            .iter()
            .enumerate()
            .find(|(_, x)| x.name == member)
            .ok_or_else(|| format!("no state member {name}"))?;
        if !decl.export {
            return Err(format!("{name} is not exported"));
        }
        let mut lines = Vec::with_capacity(self.workers.len());
        for w in &self.workers {
            let held = w
                .state
                .get(p)
                .and_then(|s| s.get(m))
                .ok_or_else(|| format!("worker {} has no {name}", w.id))?;
            // an element not filled is `_`, where filling is element by element; a
            // member filled whole that is not filled shows what it holds, marked
            let by_element = matches!(held.filled, Filled::Element(_));
            let filled = |i: usize| !by_element || held.is_filled(i);
            let text = match &held.data {
                MemberData::Scalars(v) => run_text(v.iter().enumerate().map(|(i, s)| {
                    if filled(i) {
                        scalar_text(s)
                    } else {
                        "_".into()
                    }
                })),
                MemberData::Tiles(v) => v
                    .iter()
                    .enumerate()
                    .map(|(i, tile)| {
                        if filled(i) {
                            format!("[{}]", tile_text(tile))
                        } else {
                            "_".into()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
            };
            let mark = if by_element || held.is_filled(0) {
                ""
            } else {
                " (not filled)"
            };
            lines.push(format!("worker {}: {text}{mark}", w.id));
        }
        let after = self.report.ended.len();
        self.report.shown.push(Shown {
            after,
            heading: format!("state {name}"),
            lines,
        });
        Ok(())
    }

    /// Whether `t` is a copy of a replicated tensor other than its copy 0.
    fn later_copy(&self, t: TensorId) -> bool {
        self.copies
            .values()
            .any(|c| c.iter().skip(1).any(|x| *x == t))
    }

    /// The arguments as the worker `id` sees them: a replicated tensor is the copy of
    /// its island.
    fn local(&self, values: &[Value], params: &[Type], id: usize) -> Vec<Value> {
        let island = self.islands.of.get(id).copied().unwrap_or(0);
        values
            .iter()
            .zip(params)
            .map(|(v, t)| match (v, t) {
                (Value::Tensor(x), Type::Tensor(_, true)) => {
                    Value::Tensor(self.copy(*x, island).unwrap_or(*x))
                }
                _ => v.clone(),
            })
            .collect()
    }

    /// The handler named `Program.handler`.
    pub fn handler(&self, name: &str) -> Option<Entry> {
        let (p, h) = name.split_once('.')?;
        let pid = self.module.programs.iter().position(|x| x.name == p)?;
        let hid = self.module.programs[pid]
            .handlers
            .iter()
            .position(|x| x.name == h)?;
        Some(Entry::Handler(ProgramId(pid as u32), HandlerId(hid as u32)))
    }

    /// The handler, chain or run named `Program.name`.
    pub fn entry(&self, name: &str) -> Option<Entry> {
        if let Some(e) = self.handler(name) {
            return Some(e);
        }
        let (p, n) = name.split_once('.')?;
        let pid = self.module.programs.iter().position(|x| x.name == p)?;
        let program = self.module.programs.get(pid)?;
        let pid = ProgramId(pid as u32);
        if let Some(c) = program.chains.iter().position(|x| x.name == n) {
            return Some(Entry::Chain(pid, ChainId(c as u32)));
        }
        let r = program.runs.iter().position(|x| x.name == n)?;
        Some(Entry::Run(pid, RunId(r as u32)))
    }

    /// Invokes the handler or chain `name` (`Program.name`) on the workers `first ..
    /// first + count`, with `args` in order and the `@bind(invoke)` parameters given
    /// by name (a parameter not given takes its declared value). Runs it whole and
    /// gives its end, which the report also records.
    pub fn invoke(
        &mut self,
        name: &str,
        args: &[Arg],
        group: (usize, usize),
        invoke_params: &[(&str, Scalar)],
    ) -> End {
        let end = match self.entry(name) {
            Some(entry) => {
                // after everything the host has seen end
                let before = self.races.all();
                let mut accesses = Accesses::default();
                self.invoke_here(entry, args, group, invoke_params, &before, &mut accesses)
            }
            None => End::Invalid {
                what: format!("no handler, chain or run {name}"),
                at: None,
            },
        };
        self.record(name.to_string(), Some(group), end)
    }

    /// A chain: one invocation whose calls run in the written order, each on every
    /// worker of the group, or of its range `@on(group[a:b])` (R47), and whole before
    /// the next. A call that does not complete ends the chain with its end, and the
    /// later calls do not run.
    fn invoke_chain(
        &mut self,
        p: ProgramId,
        c: ChainId,
        args: &[Arg],
        group: (usize, usize),
        invoke_params: &[(&str, Scalar)],
        before: &BTreeSet<usize>,
    ) -> End {
        let invalid = |what: String| End::Invalid { what, at: None };
        let module = self.module;
        let Some(chain) = module
            .programs
            .get(p.0 as usize)
            .and_then(|x| x.chains.get(c.0 as usize))
        else {
            return invalid("no such chain".into());
        };
        if let Err(what) = self.arg_values(args, &chain.params) {
            return invalid(what);
        }
        let mut opened = BTreeSet::new();
        let mut prior = before.clone();
        for (k, call) in chain.calls.iter().enumerate() {
            let mut call_args = Vec::with_capacity(call.args.len());
            for a in &call.args {
                let arg = match a {
                    // `past(y)` is the tensor as the earlier calls wrote it
                    CallArg::Param(i) | CallArg::Past(i) => args.get(*i as usize).cloned(),
                    CallArg::Const(k) => constant(*k).ok().map(Arg::Scalar),
                };
                match arg {
                    Some(arg) => call_args.push(arg),
                    None => return invalid("a call's argument that names nothing".into()),
                }
            }
            // the call's range is its group: rank 0 is the range's first rank
            let (from, count) = match call.part {
                None => (0, group.1),
                Some(part) => {
                    let ends = super::workers::count(part.first, &self.params)
                        .and_then(|a| Ok((a, super::workers::count(part.end, &self.params)?)));
                    match ends {
                        Ok((a, b)) if a < b && b <= group.1 => (a, b - a),
                        Ok((a, b)) => {
                            return invalid(format!(
                                "the range {a}:{b} of call {k} is not inside the chain's \
                                 group of {}",
                                group.1
                            ))
                        }
                        Err(f) => return invalid(f.to_string()),
                    }
                }
            };
            let mut accesses = Accesses::default();
            let begun = self.races.count();
            let end = self.invoke_here(
                Entry::Handler(p, call.handler),
                &call_args,
                (group.0 + from, count),
                invoke_params,
                &prior,
                &mut accesses,
            );
            // ranks of the chain's group, for the proof of `past`
            let shift = |set: BTreeSet<(usize, usize, usize)>| {
                set.into_iter()
                    .map(|(rank, x, tile)| (rank + from, x, tile))
                    .collect::<BTreeSet<_>>()
            };
            accesses.read = shift(std::mem::take(&mut accesses.read));
            accesses.opened = shift(std::mem::take(&mut accesses.opened));
            // a call is after every earlier call of the chain
            if self.races.count() > begun {
                prior.insert(begun);
            }
            if end != End::Completed(None) {
                return match end {
                    End::Completed(Some(_)) => invalid("a chain's call that gave a result".into()),
                    other => other,
                };
            }
            // the build's proof of `past(y)`, checked worker by worker: each tile a
            // rank read of y in this call lies inside a view that same rank opened for
            // writing in an earlier call
            for a in &call.args {
                let CallArg::Past(i) = a else { continue };
                let Some(Arg::Tensor(y)) = args.get(*i as usize) else {
                    return invalid("`past` of what is not a tensor".into());
                };
                let unproven = accesses
                    .read
                    .iter()
                    .find(|(rank, x, tile)| *x == y.0 && !opened.contains(&(*rank, *x, *tile)));
                if let Some((rank, _, tile)) = unproven {
                    return invalid(format!(
                        "call {k} reads at rank {rank} tile {tile} of `past` parameter {i}, \
                         which no earlier call opened for writing at that rank"
                    ));
                }
            }
            opened.extend(accesses.opened);
        }
        End::Completed(None)
    }

    /// A dynamic run: `seed` on every worker in rank order (round 0), then rounds
    /// until no message is left. In each round every worker, in rank order, takes the
    /// messages it holds at the round's start: its offer slots by sender rank, then
    /// its queue in (sender rank, sender's send count) order, each by the body of its
    /// type. What is sent in a round is delivered at the round's end. Each body must
    /// consume or forward its message exactly once. The first fault, by (round, rank,
    /// order in the round), ends the run; the rest is dropped.
    fn invoke_run(
        &mut self,
        p: ProgramId,
        r: RunId,
        args: &[Arg],
        (first, count): (usize, usize),
        invoke_params: &[(&str, Scalar)],
        before: &BTreeSet<usize>,
    ) -> End {
        let invalid = |what: String| End::Invalid { what, at: None };
        let module = self.module;
        let Some(run) = module
            .programs
            .get(p.0 as usize)
            .and_then(|x| x.runs.get(r.0 as usize))
        else {
            return invalid("no such run".into());
        };
        if count == 0
            || first
                .checked_add(count)
                .is_none_or(|end| end > self.workers.len())
        {
            return End::Failed {
                reason: Failure::Capacity,
                at: None,
            };
        }
        let params = match self.invoke_params(invoke_params) {
            Ok(x) => x,
            Err(end) => return end,
        };
        let values = match self.arg_values(args, &run.params) {
            Ok(v) => v,
            Err(what) => return invalid(what),
        };
        let local: Vec<Vec<Value>> = (0..count)
            .map(|rank| self.local(&values, &run.params, first + rank))
            .collect();
        let Some(group) = self.workers.get_mut(first..first + count) else {
            return invalid("a worker past the loaded ones".into());
        };
        let inv = self.races.begin(before, count, self.report.ended.len());
        let mut lines = Lines::new();
        let mut mail = Mail::new(count, run.peers.clone(), self.options.live_bound);
        let mut cx = Context {
            module,
            shapes: &self.shapes,
            params: &params,
            heap: &mut self.heap,
            lines: &mut lines,
            accesses: &mut Accesses::default(),
            mail: &mut mail,
            races: &mut self.races,
            inv,
            islands: &self.islands,
            marks: &mut self.marks,
        };
        let budget = self.options.step_budget;
        let mut spent = vec![0u64; count];
        let place = |rank| Place {
            rank,
            size: count,
            first,
            grid: None,
        };
        let failed = |rank: usize, fault: Fault, at: (BodyRef, InstId)| match fault {
            Fault::Failure(reason) => End::Failed {
                reason,
                at: Some((rank, at.0, at.1)),
            },
            Fault::Undefined(what) => End::Invalid {
                what,
                at: Some((rank, at.0, at.1)),
            },
        };
        let seed = BodyRef::Seed(p, r);
        for (rank, worker) in group.iter_mut().enumerate() {
            let left = budget.saturating_sub(spent.get(rank).copied().unwrap_or(0));
            match run_body(
                &mut cx,
                worker,
                place(rank),
                seed,
                local.get(rank).cloned().unwrap_or_default(),
                None,
                left,
            ) {
                Ok(steps) => add(&mut spent, rank, steps),
                Err((fault, at)) => return failed(rank, fault, at),
            }
        }
        let mut queues: Vec<Vec<Letter>> = vec![Vec::new(); count];
        loop {
            // what was sent last round arrives, by sender rank and send count
            let mut letters = std::mem::take(&mut cx.mail.outbox);
            letters.sort_by_key(|l| (l.from, l.seq));
            for l in letters {
                if let Some(q) = queues.get_mut(l.to) {
                    q.push(l);
                }
            }
            if queues.iter().all(Vec::is_empty) && cx.mail.slots.is_empty() {
                return if cx.mail.live == 0 {
                    End::Completed(None)
                } else {
                    invalid("a message neither consumed nor forwarded".into())
                };
            }
            // each worker takes what it holds at the round's start
            let mut held: Vec<Vec<Letter>> = Vec::with_capacity(count);
            for (rank, queue) in queues.iter_mut().enumerate() {
                let mut mine: Vec<Letter> = Vec::new();
                let slots: Vec<(usize, usize)> = cx
                    .mail
                    .slots
                    .keys()
                    .filter(|(_, to)| *to == rank)
                    .copied()
                    .collect();
                for key in slots {
                    if let Some(m) = cx.mail.slots.remove(&key) {
                        mine.push(m);
                    }
                }
                mine.append(queue);
                held.push(mine);
            }
            for (rank, (worker, messages)) in group.iter_mut().zip(held).enumerate() {
                for letter in messages {
                    let ty = match &letter.message {
                        Value::Message(ty, ..) => *ty,
                        _ => return invalid("a letter that is not a message".into()),
                    };
                    // the send is before the body that receives it, and its
                    // publications arrive with it
                    cx.races.acquire(rank, &letter.clock);
                    cx.races.receive(rank, &letter.publications);
                    let m = letter.message;
                    let Some(k) = run.on_message.iter().position(|(t, _)| *t == ty) else {
                        return invalid("a message no body of the run receives".into());
                    };
                    let at = BodyRef::On(p, r, k as u32);
                    cx.mail.handled = 0;
                    let left = budget.saturating_sub(spent.get(rank).copied().unwrap_or(0));
                    match run_body(
                        &mut cx,
                        worker,
                        place(rank),
                        at,
                        local.get(rank).cloned().unwrap_or_default(),
                        Some(m),
                        left,
                    ) {
                        Ok(steps) => add(&mut spent, rank, steps),
                        Err((fault, at)) => return failed(rank, fault, at),
                    }
                    if cx.mail.handled != 1 {
                        let what = if cx.mail.handled == 0 {
                            "a message body that neither consumes nor forwards its message"
                        } else {
                            "a message body that consumes or forwards its message twice"
                        };
                        // at the body's end
                        let end = run
                            .on_message
                            .get(k)
                            .map_or(0, |(_, b)| b.insts.len() as u32);
                        return failed(rank, Fault::undefined(what), (at, InstId(end)));
                    }
                }
            }
        }
    }

    /// The parameters of one invocation: the module's, with the `@bind(invoke)` ones
    /// the host gives (a parameter not given keeps its declared value).
    fn invoke_params(&self, invoke_params: &[(&str, Scalar)]) -> Result<Vec<Option<Scalar>>, End> {
        let invalid = |what: String| End::Invalid { what, at: None };
        let mut params = self.params.clone();
        for (i, p) in self.module.params.iter().enumerate() {
            let Binding::Invoke(declared) = p.binding else {
                continue;
            };
            let Some((_, v)) = invoke_params.iter().find(|(n, _)| *n == p.name) else {
                continue;
            };
            let (Some(given), Ok(Scalar::Int(_, top))) = (v.as_i128(), constant(declared)) else {
                return Err(invalid(format!(
                    "a value of {} that is not an integer",
                    p.name
                )));
            };
            if v.format() != Some(p.ty.format()) {
                return Err(invalid(format!("a value of {} of another type", p.name)));
            }
            if given < 0 || given as u128 > u128::from(top) {
                return Err(End::Failed {
                    reason: Failure::OverBound,
                    at: None,
                });
            }
            if let Some(slot) = params.get_mut(i) {
                *slot = Some(*v);
            }
        }
        Ok(params)
    }

    /// The arguments as a body reads them, each checked against its parameter.
    fn arg_values(&self, args: &[Arg], params: &[Type]) -> Result<Vec<Value>, String> {
        if args.len() != params.len() {
            return Err(format!(
                "{} arguments for {} parameters",
                args.len(),
                params.len()
            ));
        }
        let mut values = Vec::with_capacity(args.len());
        for (a, t) in args.iter().zip(params) {
            let ok = match (a, t) {
                (Arg::Tensor(id), Type::Tensor(f, replicated)) => {
                    let is = self.copies.contains_key(&id.0);
                    if is != *replicated || self.later_copy(*id) {
                        return Err(if *replicated {
                            "a tensor that is not replicated, for a `@replicated` parameter".into()
                        } else {
                            "a replicated tensor, for a parameter that is not `@replicated`".into()
                        });
                    }
                    self.heap.get(*id).is_ok_and(|x| x.format == *f)
                }
                (Arg::Scalar(s), Type::Scalar(st)) => s.format() == Some(st.format()),
                _ => false,
            };
            if !ok {
                return Err(format!("an argument that does not fit {t:?}"));
            }
            values.push(match a {
                Arg::Tensor(id) => Value::Tensor(*id),
                Arg::Scalar(s) => Value::Scalar(*s),
            });
        }
        Ok(values)
    }

    /// Runs the control program `name` with `args`. Its body runs on a frame of its
    /// own; each `invoke` is recorded, and run when an `await` or an `outcome` needs
    /// it, the invocations always in the order the program made them. An `after` on
    /// one that did not complete fails by `dependency` without starting. When the
    /// program has ended, every invocation it made that has not run is run to its
    /// end. The report gets a line for each invocation as it ends, and one for the
    /// program as it ends. An invocation that is not a program stops the run there.
    pub fn control(&mut self, name: &str, args: &[Arg]) -> End {
        let label = format!("control {name}");
        let module = self.module;
        let Some((c, control)) = module
            .controls
            .iter()
            .enumerate()
            .find(|(_, x)| x.name == name)
        else {
            let what = format!("no control program {name}");
            return self.record(label, None, End::Invalid { what, at: None });
        };
        let at = BodyRef::Control(ControlId(c as u32));
        let frame = self
            .arg_values(args, &control.params)
            .and_then(|values| Frame::new(module, at, values).map_err(|f| f.to_string()));
        let mut thread = match frame {
            Ok(f) => Thread::new(f),
            Err(what) => return self.record(label, None, End::Invalid { what, at: None }),
        };
        let mut issued: Vec<Issued> = Vec::new();
        // everything the host saw end before the program; the handles it has seen end
        let base = self.races.all();
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        // the collectives over copies it has made, by their numbers in the race record
        let mut done: BTreeSet<usize> = BTreeSet::new();
        let end = loop {
            let stop = {
                let mut lines = Lines::new();
                let mut cx = Context {
                    module,
                    shapes: &self.shapes,
                    params: &self.params,
                    heap: &mut self.heap,
                    lines: &mut lines,
                    accesses: &mut Accesses::default(),
                    mail: &mut Mail::default(),
                    // a control program touches no tensor
                    races: &mut Races::default(),
                    inv: 0,
                    islands: &self.islands,
                    marks: &mut Marks::default(),
                };
                // a control program has no worker: it reads no state and no identity
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
                thread.run(&mut nobody, place, &mut cx, self.options.step_budget)
            };
            let (request, at) = match stop {
                Stop::Returned(v) => {
                    break match (v, control.ret) {
                        (Some(Value::Scalar(s)), Some(_)) => End::Completed(Some(s)),
                        (None, None) => End::Completed(None),
                        _ => End::Invalid {
                            what: "a result that does not fit the control program".into(),
                            at: None,
                        },
                    }
                }
                Stop::Faulted { fault, at } => break fault_end(fault, at),
                Stop::Meeting { at, .. } | Stop::Waiting { at } => {
                    let what = "a collective or an `arrive` in a control program";
                    break fault_end(Fault::undefined(what), at);
                }
                Stop::Control { request, at } => (request, at),
            };
            let answer = match request {
                Request::Invoke {
                    entry,
                    args,
                    group,
                    after,
                    after_end,
                } => {
                    let given = args.len();
                    let mut host = Vec::with_capacity(given);
                    for a in args {
                        match a {
                            Value::Tensor(id) => host.push(Arg::Tensor(id)),
                            Value::Scalar(s) => host.push(Arg::Scalar(s)),
                            _ => {}
                        }
                    }
                    let n = issued.len();
                    if host.len() != given || after.iter().chain(&after_end).any(|h| *h >= n) {
                        let what = "an invocation's argument or edge that is not one";
                        break fault_end(Fault::undefined(what), at);
                    }
                    issued.push(Issued {
                        entry,
                        args: host,
                        group,
                        after,
                        after_end,
                        seen: seen.clone(),
                        made_after: done.clone(),
                        end: None,
                        inv: None,
                    });
                    Some(Value::Handle(n))
                }
                Request::Copies(tensor, over) => {
                    // after what the program has seen end, and its earlier collectives
                    let mut before = base.clone();
                    for h in &seen {
                        before.extend(issued.get(*h).and_then(|x| x.inv));
                    }
                    before.extend(done.iter().copied());
                    match self.over_copies(&label, tensor, over, &before, at) {
                        Ok(n) => {
                            done.insert(n);
                            // `all_reduce` gives its operand, the tensor; the others nothing
                            matches!(over, OverCopies::AllReduce { .. })
                                .then_some(Value::Tensor(tensor))
                        }
                        Err(end) => break end,
                    }
                }
                Request::Await(h) | Request::Outcome(h) => {
                    let waited = self.run_issued(&mut issued, h, &base);
                    // what the program makes from here is after h's end
                    seen.insert(h);
                    match (waited, matches!(request, Request::Await(_))) {
                        (Err(invalid), _) => break invalid,
                        (Ok(End::Completed(v)), true) => v.map(Value::Scalar),
                        // `await` of a failed invocation fails the control program
                        (Ok(End::Failed { reason, .. }), true) => {
                            break End::Failed {
                                reason,
                                at: Some((0, at.0, at.1)),
                            }
                        }
                        (Ok(End::Completed(value)), false) => Some(Value::Outcome(Outcome {
                            failed: false,
                            reason: None,
                            value,
                        })),
                        (Ok(End::Failed { reason, .. }), false) => Some(Value::Outcome(Outcome {
                            failed: true,
                            reason: Some(reason),
                            value: None,
                        })),
                        (Ok(invalid @ End::Invalid { .. }), _) => break invalid,
                    }
                }
            };
            if let Err(fault) = thread.deliver(module, answer) {
                break fault_end(fault, at);
            }
        };
        let end = self.record(label, None, end);
        // the invocations it made go on by themselves
        if !matches!(end, End::Invalid { .. }) {
            if let Some(last) = issued.len().checked_sub(1) {
                let _ = self.run_issued(&mut issued, last, &base);
            }
        }
        end
    }

    /// A collective over the copies of the replicated tensor `t`, made by the control
    /// program `label` at `at`, after `before`. It folds tile by tile, from the left in
    /// island order, and has its own line in the report and its own entry in the race
    /// record (one actor). Gives that entry's number, or the end that stops the program.
    fn over_copies(
        &mut self,
        label: &str,
        t: TensorId,
        over: OverCopies,
        before: &BTreeSet<usize>,
        at: (BodyRef, InstId),
    ) -> Result<usize, End> {
        let entry = format!("{label} {}", over.name());
        let invalid = |what: &str| End::Invalid {
            what: what.into(),
            at: Some((0, at.0, at.1)),
        };
        let Some(copies) = self.copies.get(&t.0).cloned() else {
            let end = invalid("a collective over the copies of a tensor that is not replicated");
            return Err(self.record(entry, None, end));
        };
        let (reads, writes): (Vec<TensorId>, Vec<TensorId>) = match over {
            OverCopies::ReduceTo { to, .. } => match copies.get(to) {
                Some(x) => (copies.clone(), vec![*x]),
                None => return Err(self.record(entry, None, invalid("`to` names no island"))),
            },
            OverCopies::BroadcastFrom { from } => match copies.get(from) {
                Some(x) => (
                    vec![*x],
                    copies.iter().copied().filter(|c| c != x).collect(),
                ),
                None => return Err(self.record(entry, None, invalid("`from` names no island"))),
            },
            OverCopies::AllReduce { .. } | OverCopies::AllGather => {
                (copies.clone(), copies.clone())
            }
        };
        let n = self.races.begin(before, 1, self.report.ended.len());
        let tiles = match self.heap.get(t) {
            Ok(x) => x.tiles.len(),
            Err(f) => return Err(self.record(entry, None, invalid(&f.to_string()))),
        };
        for (list, write) in [(&reads, false), (&writes, true)] {
            for c in list {
                for k in 0..tiles {
                    let place = TensorPlace::tile(*c, k);
                    if let Err(what) = self.races.access(n, 0, place, write, at) {
                        return Err(self.record(entry, None, invalid(&what)));
                    }
                }
            }
        }
        match self.fold_copies(&copies, over, tiles) {
            Ok(()) => {
                match over {
                    // it writes island `to`'s copy only: island `to` wrote every place
                    OverCopies::ReduceTo { to, .. } => self.marks.all_by(t, tiles, to),
                    // the copies are equal again
                    _ => self.marks.clear(t),
                }
                self.record(entry, None, End::Completed(None));
                Ok(n)
            }
            Err(fault) => Err(self.record(entry, None, fault_end(fault, at))),
        }
    }

    /// The values of a collective over copies, written to the heap.
    fn fold_copies(
        &mut self,
        copies: &[TensorId],
        over: OverCopies,
        tiles: usize,
    ) -> Result<(), Fault> {
        if over == OverCopies::AllGather {
            return self.gather_copies(copies);
        }
        let module = self.module;
        let mut lines = Lines::new();
        let mut cx = Context {
            module,
            shapes: &self.shapes,
            params: &self.params,
            heap: &mut self.heap,
            lines: &mut lines,
            accesses: &mut Accesses::default(),
            mail: &mut Mail::default(),
            // a combine body is compute only
            races: &mut Races::default(),
            inv: 0,
            islands: &self.islands,
            marks: &mut Marks::default(),
        };
        let budget = self.options.step_budget;
        let (with, into): (Option<Combiner>, Vec<TensorId>) = match over {
            OverCopies::ReduceTo { to, with } => {
                (Some(with), copies.get(to).into_iter().copied().collect())
            }
            OverCopies::AllReduce { with } => (Some(with), copies.to_vec()),
            OverCopies::BroadcastFrom { from } => {
                (None, copies.get(from).into_iter().copied().collect())
            }
            OverCopies::AllGather => return Err(Fault::undefined("an `all_gather` as a fold")),
        };
        for k in 0..tiles {
            let tile_of = |cx: &Context<'_>, c: TensorId| -> Result<Tile, Fault> {
                cx.heap
                    .get(c)?
                    .tiles
                    .get(k)
                    .cloned()
                    .ok_or_else(|| Fault::undefined("a copy of another shape"))
            };
            let value = match with {
                Some(with) => {
                    let mut acc: Option<Value> = None;
                    for c in copies {
                        let x = Value::Tile(tile_of(&cx, *c)?);
                        acc = Some(match acc {
                            None => x,
                            Some(a) => delivery::combine(with, a, x, &mut cx, budget)?,
                        });
                    }
                    match acc {
                        Some(Value::Tile(t)) => t,
                        _ => return Err(Fault::undefined("a fold of copies that is not a tile")),
                    }
                }
                None => match into.first() {
                    Some(from) => tile_of(&cx, *from)?,
                    None => return Err(Fault::undefined("no copy to broadcast")),
                },
            };
            let targets: Vec<TensorId> = match over {
                OverCopies::BroadcastFrom { .. } => copies.to_vec(),
                _ => into.clone(),
            };
            for c in targets {
                let tensor = cx.heap.get_mut(c)?;
                let held = ops::convert(&value, tensor.format)?;
                match tensor.tiles.get_mut(k) {
                    Some(slot) => *slot = held,
                    None => return Err(Fault::undefined("a copy of another shape")),
                }
            }
        }
        Ok(())
    }

    /// `all_gather` over `copies`: each place written since the copies were last equal
    /// takes, on every copy, its value in the copy of an island that wrote it; a place
    /// no island wrote is left as it is (the copies agree there).
    fn gather_copies(&mut self, copies: &[TensorId]) -> Result<(), Fault> {
        let Some(&first) = copies.first() else {
            return Ok(());
        };
        // the language leaves the choice open among the islands that wrote the place;
        // the interpreter takes the lowest, so that one program and input give one
        // report
        let marked: Vec<((usize, usize, usize), usize)> = self
            .marks
            .of(first)
            .map(|m| {
                m.iter()
                    .filter_map(|(p, islands)| islands.first().map(|i| (*p, *i)))
                    .collect()
            })
            .unwrap_or_default();
        let shape = || Fault::undefined("a copy of another shape");
        for ((tile, lo, hi), island) in marked {
            let from = *copies
                .get(island)
                .ok_or_else(|| Fault::undefined("a write by an island with no copy"))?;
            let value = self
                .heap
                .get(from)?
                .tiles
                .get(tile)
                .cloned()
                .ok_or_else(shape)?;
            for &c in copies.iter().filter(|c| **c != from) {
                let slot = self
                    .heap
                    .get_mut(c)?
                    .tiles
                    .get_mut(tile)
                    .ok_or_else(shape)?;
                if hi == usize::MAX {
                    *slot = value.clone();
                    continue;
                }
                // part of a tile: the bytes of an item
                let bytes = || Fault::undefined("bytes");
                let src = value.to_bytes().ok_or_else(bytes)?;
                let mut dst = slot.to_bytes().ok_or_else(bytes)?;
                match (dst.get_mut(lo..hi), src.get(lo..hi)) {
                    (Some(d), Some(s)) => d.copy_from_slice(s),
                    _ => return Err(shape()),
                }
                *slot = Tile::from_bytes(slot.format, slot.shape, &dst).ok_or_else(bytes)?;
            }
        }
        Ok(())
    }

    /// Runs the invocations a control program made, in order, up to `upto`, and gives
    /// its end; an invocation that is not a program stops there, and is given.
    fn run_issued(
        &mut self,
        issued: &mut [Issued],
        upto: usize,
        base: &BTreeSet<usize>,
    ) -> Result<End, End> {
        for i in 0..=upto {
            let Some(inv) = issued.get(i) else {
                return Err(End::Invalid {
                    what: "a handle that names no invocation".into(),
                    at: None,
                });
            };
            if inv.end.is_some() {
                continue;
            }
            let failed_before = inv.after.iter().any(|a| {
                !matches!(
                    issued.get(*a).and_then(|x| x.end.as_ref()),
                    Some(End::Completed(_))
                )
            });
            let (entry, args, group) = (inv.entry, inv.args.clone(), inv.group);
            // after what the program saw end before it made this one, and its edges
            let mut before = base.clone();
            for h in inv.seen.iter().chain(&inv.after).chain(&inv.after_end) {
                before.extend(issued.get(*h).and_then(|x| x.inv));
            }
            before.extend(inv.made_after.iter().copied());
            let begun = self.races.count();
            let end = if failed_before {
                End::Failed {
                    reason: Failure::Dependency,
                    at: None,
                }
            } else {
                let mut accesses = Accesses::default();
                self.invoke_here(entry, &args, group, &[], &before, &mut accesses)
            };
            // a chain is known by its last call
            let ran = (self.races.count() > begun).then(|| self.races.count() - 1);
            let name = self.entry_name(entry);
            let end = self.record(name, Some(group), end);
            if let Some(inv) = issued.get_mut(i) {
                inv.end = Some(end.clone());
                inv.inv = ran;
            }
            if matches!(end, End::Invalid { .. }) {
                return Err(end);
            }
        }
        issued
            .get(upto)
            .and_then(|x| x.end.clone())
            .ok_or_else(|| End::Invalid {
                what: "a handle that names no invocation".into(),
                at: None,
            })
    }

    /// Adds a line to the report, and gives the end back.
    fn record(&mut self, entry: String, group: Option<(usize, usize)>, end: End) -> End {
        let number = self.report.ended.len();
        self.report.ended.push(Ended {
            number,
            entry,
            group,
            end: end.clone(),
        });
        end
    }

    /// An entry's name, as `Program.handler`.
    fn entry_name(&self, entry: Entry) -> String {
        let (p, name) = match entry {
            Entry::Handler(p, h) => (
                p,
                self.module
                    .programs
                    .get(p.0 as usize)
                    .and_then(|x| x.handlers.get(h.0 as usize))
                    .map(|x| x.name.clone()),
            ),
            Entry::Run(p, r) => (
                p,
                self.module
                    .programs
                    .get(p.0 as usize)
                    .and_then(|x| x.runs.get(r.0 as usize))
                    .map(|x| x.name.clone()),
            ),
            Entry::Chain(p, c) => (
                p,
                self.module
                    .programs
                    .get(p.0 as usize)
                    .and_then(|x| x.chains.get(c.0 as usize))
                    .map(|x| x.name.clone()),
            ),
        };
        let program = self
            .module
            .programs
            .get(p.0 as usize)
            .map(|x| x.name.as_str());
        format!(
            "{}.{}",
            program.unwrap_or("?"),
            name.as_deref().unwrap_or("?")
        )
    }

    fn invoke_here(
        &mut self,
        entry: Entry,
        args: &[Arg],
        (first, count): (usize, usize),
        invoke_params: &[(&str, Scalar)],
        before: &BTreeSet<usize>,
        accesses: &mut Accesses,
    ) -> End {
        let invalid = |what: String| End::Invalid { what, at: None };
        let name = self.entry_name(entry);
        let (p, h) = match entry {
            Entry::Handler(p, h) => (p, h),
            Entry::Chain(p, c) => {
                return self.invoke_chain(p, c, args, (first, count), invoke_params, before)
            }
            Entry::Run(p, r) => {
                let group = (first, count);
                // a run is no chain's call: what it reads is checked by nothing after it
                return self.invoke_run(p, r, args, group, invoke_params, before);
            }
        };
        let module = self.module;
        let Some(handler) = module
            .programs
            .get(p.0 as usize)
            .and_then(|x| x.handlers.get(h.0 as usize))
        else {
            return invalid(format!("no handler {name}"));
        };
        if count == 0
            || first
                .checked_add(count)
                .is_none_or(|end| end > self.workers.len())
        {
            return End::Failed {
                reason: Failure::Capacity,
                at: None,
            };
        }
        let grid = match handler.on {
            On::Group => None,
            On::Grid(r, c) => {
                if r.checked_mul(c) != Some(count as u64) {
                    return invalid(format!("a grid of {r} by {c} on {count} workers"));
                }
                Some((r as usize, c as usize))
            }
        };
        // the parameters this invocation binds
        let params = match self.invoke_params(invoke_params) {
            Ok(x) => x,
            Err(end) => return end,
        };
        let values = match self.arg_values(args, &handler.params) {
            Ok(v) => v,
            Err(what) => return invalid(what),
        };
        let at = BodyRef::Handler(p, h);
        let mut threads = Vec::with_capacity(count);
        for rank in 0..count {
            let local = self.local(&values, &handler.params, first + rank);
            match Frame::new(module, at, local) {
                Ok(f) => threads.push(Thread::new(f)),
                Err(f) => return invalid(f.to_string()),
            }
        }
        let Some(group) = self.workers.get_mut(first..first + count) else {
            return invalid("a worker past the loaded ones".into());
        };
        let inv = self.races.begin(before, count, self.report.ended.len());
        let mut lines = Lines::new();
        let mut cx = Context {
            module,
            shapes: &self.shapes,
            params: &params,
            heap: &mut self.heap,
            lines: &mut lines,
            accesses,
            mail: &mut Mail::default(),
            races: &mut self.races,
            inv,
            islands: &self.islands,
            marks: &mut self.marks,
        };
        let place = |rank| Place {
            rank,
            size: count,
            first,
            grid,
        };
        match run_group(&mut cx, group, threads, place, self.options.step_budget) {
            GroupEnd::Faulted(rank, Fault::Failure(reason), body, inst) => End::Failed {
                reason,
                at: Some((rank, body, inst)),
            },
            GroupEnd::Faulted(rank, Fault::Undefined(what), body, inst) => End::Invalid {
                what,
                at: Some((rank, body, inst)),
            },
            GroupEnd::Stuck(rank, what, body, inst) => End::Invalid {
                what: what.into(),
                at: Some((rank, body, inst)),
            },
            GroupEnd::Completed(result) => match (result, handler.ret) {
                (Some(Value::Scalar(s)), Some(_)) => End::Completed(Some(s)),
                (None, None) => End::Completed(None),
                (None, Some(_)) => invalid("the leader returned no result".into()),
                (Some(_), _) => invalid("a result of another type".into()),
            },
        }
    }
}

/// Adds a body's steps to its rank's count.
fn add(spent: &mut [u64], rank: usize, steps: u64) {
    if let Some(s) = spent.get_mut(rank) {
        *s = s.saturating_add(steps);
    }
}

/// Runs one body of a dynamic run on a worker, to its end: the steps it took, or its
/// fault and where. A run's body waits for nothing: a collective, an `arrive` or a
/// control program's request in it is not a program.
fn run_body(
    cx: &mut Context<'_>,
    worker: &mut Worker,
    place: Place,
    at: BodyRef,
    args: Vec<Value>,
    arrived: Option<Value>,
    budget: u64,
) -> Result<u64, (Fault, (BodyRef, InstId))> {
    let frame = Frame::new(cx.module, at, args).map_err(|f| (f, (at, InstId(0))))?;
    let frame = match arrived {
        Some(m) => frame.with_arrived(m),
        None => frame,
    };
    let mut thread = Thread::new(frame);
    match thread.run(worker, place, cx, budget) {
        Stop::Returned(_) => Ok(thread.steps()),
        Stop::Faulted { fault, at } => Err((fault, at)),
        Stop::Meeting { at, .. } | Stop::Waiting { at } | Stop::Control { at, .. } => Err((
            Fault::undefined("a collective, an `arrive` or a request in a run's body"),
            at,
        )),
    }
}

/// An invocation a control program made, and its end once it has run.
struct Issued {
    entry: Entry,
    args: Vec<Arg>,
    group: (usize, usize),
    /// `after`: each must have completed.
    after: Vec<usize>,
    /// `after_end`: each must have ended, which it has, since invocations run in the
    /// order they were made; the edge orders them for the race record.
    after_end: Vec<usize>,
    /// The handles the program had seen end when it made this one.
    seen: BTreeSet<usize>,
    /// The collectives over copies the program had made before it, by their numbers
    /// in the race record.
    made_after: BTreeSet<usize>,
    end: Option<End>,
    /// Its number in the race record, once it has run.
    inv: Option<usize>,
}

/// The end a fault gives, at rank 0 of a control program.
fn fault_end(fault: Fault, at: (BodyRef, InstId)) -> End {
    match fault {
        Fault::Failure(reason) => End::Failed {
            reason,
            at: Some((0, at.0, at.1)),
        },
        Fault::Undefined(what) => End::Invalid {
            what,
            at: Some((0, at.0, at.1)),
        },
    }
}

/// How the frames of one group ended.
enum GroupEnd {
    /// Every frame returned; the leader's value.
    Completed(Option<Value>),
    /// The lowest rank that faulted, its fault and where.
    Faulted(usize, Fault, BodyRef, InstId),
    /// No frame faulted, and some wait at a collective or an `arrive` that cannot
    /// complete: the lowest such rank, what it waits at, and where.
    Stuck(usize, &'static str, BodyRef, InstId),
}

/// Where one rank's frame stands.
enum Standing {
    Ready,
    Waiting(BodyRef, InstId),
    Arriving(BodyRef, InstId),
    Done(Option<Value>),
    Faulted(Fault, BodyRef, InstId),
}

/// A member at a meeting: its rank, its collective, where, and the tiles it publishes.
type Arrival = (usize, Collective, (BodyRef, InstId), Vec<(usize, usize)>);

/// One collective being met: who meets, and who has arrived, with what.
struct Meeting {
    members: Vec<usize>,
    arrived: Vec<Arrival>,
}

/// Runs the frames of one group, rank by rank, each until it returns, faults, or waits
/// at a collective or an `arrive`; a collective completes when its last member
/// arrives, and an `arrive` goes on when its sender casts or ends. Repeats until no
/// frame can go on. Thread timing never enters: frames run in rank order and a meeting
/// is keyed by how many collectives each frame has reached.
fn run_group(
    cx: &mut Context<'_>,
    workers: &mut [Worker],
    mut threads: Vec<Thread>,
    place: impl Fn(usize) -> Place,
    budget: u64,
) -> GroupEnd {
    let n = threads.len();
    let mut standing: Vec<Standing> = (0..n).map(|_| Standing::Ready).collect();
    let mut reached = vec![0usize; n];
    let mut meetings: BTreeMap<(usize, usize), Meeting> = BTreeMap::new();
    let module = cx.module;
    loop {
        let mut moved = false;
        for rank in 0..n {
            while matches!(
                standing.get(rank),
                Some(Standing::Ready | Standing::Arriving(..))
            ) {
                let (Some(thread), Some(worker)) = (threads.get_mut(rank), workers.get_mut(rank))
                else {
                    break;
                };
                let before = thread.steps();
                let stop = thread.run(worker, place(rank), cx, budget);
                if thread.steps() != before || !matches!(stop, Stop::Waiting { .. }) {
                    moved = true;
                }
                let now = match stop {
                    Stop::Control { at, .. } => {
                        let what = "a control program's request in a handler";
                        Standing::Faulted(Fault::undefined(what), at.0, at.1)
                    }
                    Stop::Waiting { at } => {
                        // the others go first; it is run again when they have moved
                        if let Some(s) = standing.get_mut(rank) {
                            *s = Standing::Arriving(at.0, at.1);
                        }
                        break;
                    }
                    Stop::Returned(v) => Standing::Done(v),
                    Stop::Faulted { fault, at } => Standing::Faulted(fault, at.0, at.1),
                    Stop::Meeting {
                        collective,
                        publish,
                        at,
                    } => {
                        let seq = reached.get(rank).copied().unwrap_or(0);
                        if let Some(r) = reached.get_mut(rank) {
                            *r += 1;
                        }
                        match delivery::members(&collective, rank, n) {
                            Err(fault) => Standing::Faulted(fault, at.0, at.1),
                            // not a member: it passes, with no value
                            Ok(None) => match thread.deliver(module, Some(Value::Absent)) {
                                Ok(()) => Standing::Ready,
                                Err(fault) => Standing::Faulted(fault, at.0, at.1),
                            },
                            Ok(Some(members)) => {
                                let key = (seq, members.first().copied().unwrap_or(0));
                                let m = meetings.entry(key).or_insert_with(|| Meeting {
                                    members,
                                    arrived: Vec::new(),
                                });
                                m.arrived.push((rank, collective, at, publish));
                                if m.arrived.len() == m.members.len() {
                                    if let Some(m) = meetings.remove(&key) {
                                        complete(cx, m, &mut threads, &mut standing, budget);
                                    }
                                    // this rank's standing was set by the meeting
                                    continue;
                                }
                                Standing::Waiting(at.0, at.1)
                            }
                        }
                    }
                };
                if let Some(s) = standing.get_mut(rank) {
                    *s = now;
                }
            }
        }
        if !moved {
            break;
        }
    }
    let mut fault = None;
    let mut stuck = None;
    let mut leader = None;
    for (rank, s) in standing.into_iter().enumerate() {
        match s {
            Standing::Faulted(f, body, inst) if fault.is_none() => {
                fault = Some(GroupEnd::Faulted(rank, f, body, inst));
            }
            Standing::Waiting(body, inst) if stuck.is_none() => {
                let what = "a collective that not every member reached";
                stuck = Some(GroupEnd::Stuck(rank, what, body, inst));
            }
            Standing::Arriving(body, inst) if stuck.is_none() => {
                let what = "an `arrive` that no cast and no end of its sender reaches";
                stuck = Some(GroupEnd::Stuck(rank, what, body, inst));
            }
            Standing::Done(v) if rank == 0 => leader = v,
            _ => {}
        }
    }
    fault.or(stuck).unwrap_or(GroupEnd::Completed(leader))
}

/// Completes a meeting whose every member has arrived: the members must stand at one
/// instruction; the value is made once and each member's frame goes on with its part.
fn complete(
    cx: &mut Context<'_>,
    meeting: Meeting,
    threads: &mut [Thread],
    standing: &mut [Standing],
    budget: u64,
) {
    let Meeting { mut arrived, .. } = meeting;
    arrived.sort_by_key(|(rank, ..)| *rank);
    let Some(&(lowest, _, at, _)) = arrived.first() else {
        return;
    };
    let mut set = |rank: usize, s: Standing| {
        if let Some(slot) = standing.get_mut(rank) {
            *slot = s;
        }
    };
    if arrived.iter().any(|(_, _, a, _)| *a != at) {
        let what = "the members of a meeting reached different collectives";
        set(
            lowest,
            Standing::Faulted(Fault::undefined(what), at.0, at.1),
        );
        return;
    }
    let ranks: Vec<usize> = arrived.iter().map(|(r, ..)| *r).collect();
    // B1's collective edge: every member's arrival is before the value where it
    // arrives (every member at a `sync`); the views named with `publish` go with it
    let mut clock = Vec::new();
    let mut publications = Vec::new();
    for (r, _, _, publish) in &arrived {
        publications.extend(cx.races.publish(*r, publish));
        let c = cx.races.release(*r);
        join_clocks(&mut clock, &c);
    }
    let collectives = arrived.into_iter().map(|(_, c, ..)| c).collect();
    let values = match delivery::meet(collectives, cx, budget) {
        Ok(v) => v,
        Err(fault) => {
            set(lowest, Standing::Faulted(fault, at.0, at.1));
            return;
        }
    };
    for (rank, value) in ranks.into_iter().zip(values) {
        // a member left absent receives nothing, so no edge reaches it
        if !matches!(value, Some(Value::Absent)) {
            cx.races.acquire(rank, &clock);
            cx.races.receive(rank, &publications);
        }
        let delivered = match threads.get_mut(rank) {
            Some(thread) => thread.deliver(cx.module, value),
            None => Err(Fault::undefined("a member with no frame")),
        };
        match delivered {
            Ok(()) => set(rank, Standing::Ready),
            Err(fault) => set(rank, Standing::Faulted(fault, at.0, at.1)),
        }
    }
}

/// A clock made the join of itself and another.
fn join_clocks(into: &mut Vec<u64>, other: &[u64]) {
    if into.len() < other.len() {
        into.resize(other.len(), 0);
    }
    for (a, b) in into.iter_mut().zip(other) {
        *a = (*a).max(*b);
    }
}
