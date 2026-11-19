//! A loaded module and what runs on it: the invocations a host makes, in the order it
//! makes them, each run whole before the next; the control programs, whose
//! invocations run in the order the program makes them; and the run's report.
//!
//! An invocation of a handler runs its body once on every worker of its group. The
//! workers' frames are stepped one after another in rank order; each runs until it
//! returns or faults. The invocation's end is its leader's result, or the failure of
//! the lowest rank that faulted. A tensor write goes straight to the heap: anything
//! that could see it before the invocation's completion, with no edge to it, would be
//! a race.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};

use super::delivery;
use super::memory::{Failure, Fault, Format, Heap, Scalar, Tensor, TensorId};
use super::workers::{
    bodies, constant, Accesses, Collective, Context, Frame, Lines, Outcome, Place, Request, Shape,
    Shapes, Stop, Thread, Value, Worker,
};
use crate::program::body::InstId;
use crate::program::{BodyRef, CallArg, Module};
use crate::types::{
    Ability, Binding, ChainId, ConstValue, ControlId, Entry, HandlerId, On, ProgramId, RunId, Type,
};

/// The options of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// How many instructions one invocation may run on each worker before it gives up
    /// by name (`given_up`).
    pub step_budget: u64,
    /// The bytes of a worker's memory, which a `fit` parameter fills.
    pub worker_memory: u64,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            step_budget: 1 << 32,
            worker_memory: 1 << 20,
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
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Requirement(r) => write!(f, "requirement not met: {r}"),
            Refusal::Unbound(p) => write!(f, "load parameter with no value: {p}"),
            Refusal::BadValue(p) => write!(f, "a value that does not fit the parameter {p}"),
            Refusal::Invalid(w) => write!(f, "not a program the interpreter runs: {w}"),
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

/// A run's report: each invocation's end, in the order they were made.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// The invocations' ends.
    pub ended: Vec<Ended>,
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
    /// The report's printed form: one line for each invocation.
    pub fn print(&self) -> String {
        let mut out = String::new();
        for e in &self.ended {
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
                Ability::Islands(n) if *n > 1 => {
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
                self.invoke_here(entry, args, group, invoke_params, &mut Accesses::default())
            }
            None => End::Invalid {
                what: format!("no handler, chain or run {name}"),
                at: None,
            },
        };
        self.record(name.to_string(), Some(group), end)
    }

    /// A chain: one invocation whose calls run in the written order, each on every
    /// worker of the group and whole before the next. A call that does not complete
    /// ends the chain with its end, and the later calls do not run.
    fn invoke_chain(
        &mut self,
        p: ProgramId,
        c: ChainId,
        args: &[Arg],
        group: (usize, usize),
        invoke_params: &[(&str, Scalar)],
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
            let mut accesses = Accesses::default();
            let end = self.invoke_here(
                Entry::Handler(p, call.handler),
                &call_args,
                group,
                invoke_params,
                &mut accesses,
            );
            if end != End::Completed(None) {
                return match end {
                    End::Completed(Some(_)) => invalid("a chain's call that gave a result".into()),
                    other => other,
                };
            }
            // the build's proof of `past(y)`, checked: each tile this call read of y
            // lies inside a view an earlier call opened for writing
            for a in &call.args {
                let CallArg::Past(i) = a else { continue };
                let Some(Arg::Tensor(y)) = args.get(*i as usize) else {
                    return invalid("`past` of what is not a tensor".into());
                };
                let unproven = accesses
                    .read
                    .iter()
                    .find(|(x, tile)| *x == y.0 && !opened.contains(&(*x, *tile)));
                if let Some((_, tile)) = unproven {
                    return invalid(format!(
                        "call {k} reads tile {tile} of `past` parameter {i}, which no earlier \
                         call opened for writing"
                    ));
                }
            }
            opened.extend(accesses.opened);
        }
        End::Completed(None)
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
                (Arg::Tensor(id), Type::Tensor(f, _)) => {
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
                        end: None,
                    });
                    Some(Value::Handle(n))
                }
                Request::Await(h) | Request::Outcome(h) => {
                    let waited = self.run_issued(&mut issued, h);
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
                let _ = self.run_issued(&mut issued, last);
            }
        }
        end
    }

    /// Runs the invocations a control program made, in order, up to `upto`, and gives
    /// its end; an invocation that is not a program stops there, and is given.
    fn run_issued(&mut self, issued: &mut [Issued], upto: usize) -> Result<End, End> {
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
            let end = if failed_before {
                End::Failed {
                    reason: Failure::Dependency,
                    at: None,
                }
            } else {
                self.invoke_here(entry, &args, group, &[], &mut Accesses::default())
            };
            let name = self.entry_name(entry);
            let end = self.record(name, Some(group), end);
            if let Some(inv) = issued.get_mut(i) {
                inv.end = Some(end.clone());
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
        accesses: &mut Accesses,
    ) -> End {
        let invalid = |what: String| End::Invalid { what, at: None };
        let name = self.entry_name(entry);
        let (p, h) = match entry {
            Entry::Handler(p, h) => (p, h),
            Entry::Chain(p, c) => {
                return self.invoke_chain(p, c, args, (first, count), invoke_params)
            }
            Entry::Run(..) => {
                return invalid(format!("{name}: a run is not run by this interpreter yet"))
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
        let mut params = self.params.clone();
        for (i, p) in module.params.iter().enumerate() {
            let Binding::Invoke(declared) = p.binding else {
                continue;
            };
            let Some((_, v)) = invoke_params.iter().find(|(n, _)| *n == p.name) else {
                continue;
            };
            let (Some(given), Ok(Scalar::Int(_, top))) = (v.as_i128(), constant(declared)) else {
                return invalid(format!("a value of {} that is not an integer", p.name));
            };
            if v.format() != Some(p.ty.format()) {
                return invalid(format!("a value of {} of another type", p.name));
            }
            if given < 0 || given as u128 > u128::from(top) {
                return End::Failed {
                    reason: Failure::OverBound,
                    at: None,
                };
            }
            if let Some(slot) = params.get_mut(i) {
                *slot = Some(*v);
            }
        }
        let values = match self.arg_values(args, &handler.params) {
            Ok(v) => v,
            Err(what) => return invalid(what),
        };
        let at = BodyRef::Handler(p, h);
        let mut threads = Vec::with_capacity(count);
        for _ in 0..count {
            match Frame::new(module, at, values.clone()) {
                Ok(f) => threads.push(Thread::new(f)),
                Err(f) => return invalid(f.to_string()),
            }
        }
        let Some(group) = self.workers.get_mut(first..first + count) else {
            return invalid("a worker past the loaded ones".into());
        };
        let mut lines = Lines::new();
        let mut cx = Context {
            module,
            shapes: &self.shapes,
            params: &params,
            heap: &mut self.heap,
            lines: &mut lines,
            accesses,
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

/// An invocation a control program made, and its end once it has run.
struct Issued {
    entry: Entry,
    args: Vec<Arg>,
    group: (usize, usize),
    /// `after`; `after_end` needs nothing more, since invocations run in the order
    /// they were made.
    after: Vec<usize>,
    end: Option<End>,
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

/// One collective being met: who meets, and who has arrived, with what.
struct Meeting {
    members: Vec<usize>,
    arrived: Vec<(usize, Collective, (BodyRef, InstId))>,
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
                    Stop::Meeting { collective, at } => {
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
                                m.arrived.push((rank, collective, at));
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
    arrived.sort_by_key(|(rank, _, _)| *rank);
    let Some(&(lowest, _, at)) = arrived.first() else {
        return;
    };
    let mut set = |rank: usize, s: Standing| {
        if let Some(slot) = standing.get_mut(rank) {
            *slot = s;
        }
    };
    if arrived.iter().any(|(_, _, a)| *a != at) {
        let what = "the members of a meeting reached different collectives";
        set(
            lowest,
            Standing::Faulted(Fault::undefined(what), at.0, at.1),
        );
        return;
    }
    let ranks: Vec<usize> = arrived.iter().map(|(r, _, _)| *r).collect();
    let collectives = arrived.into_iter().map(|(_, c, _)| c).collect();
    let values = match delivery::meet(collectives, cx, budget) {
        Ok(v) => v,
        Err(fault) => {
            set(lowest, Standing::Faulted(fault, at.0, at.1));
            return;
        }
    };
    for (rank, value) in ranks.into_iter().zip(values) {
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
