//! A loaded module and what runs on it: the invocations a host makes, in the order it
//! makes them, each run whole before the next, and the run's report.
//!
//! An invocation of a handler runs its body once on every worker of its group. The
//! workers' frames are stepped one after another in rank order; each runs until it
//! returns or faults. The invocation's end is its leader's result, or the failure of
//! the lowest rank that faulted. A tensor write goes straight to the heap: anything
//! that could see it before the invocation's completion, with no edge to it, would be
//! a race.

use std::fmt::{self, Write as _};

use super::memory::{Failure, Fault, Format, Heap, Scalar, Tensor, TensorId};
use super::workers::{
    bodies, constant, Context, Frame, Place, Shape, Shapes, Stop, Thread, Value, Worker,
};
use crate::program::body::InstId;
use crate::program::{BodyRef, Module};
use crate::types::{Ability, Binding, ConstValue, Entry, HandlerId, On, ProgramId, Type};

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

/// One line of the run's report: an invocation and its end.
#[derive(Clone, Debug, PartialEq)]
pub struct Ended {
    /// The invocation's number in the run, from 0.
    pub number: usize,
    /// What it named, as `Program.handler`.
    pub entry: String,
    /// The first worker and the count of its group.
    pub group: (usize, usize),
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
            let _ = writeln!(
                out,
                "#{} {} group({}, {}): {end}",
                e.number, e.entry, e.group.0, e.group.1
            );
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

    /// Invokes the handler `name` (`Program.handler`) on the workers `first .. first +
    /// count`, with `args` in order and the `@bind(invoke)` parameters given by name
    /// (a parameter not given takes its declared value). Runs it whole and gives its
    /// end, which the report also records.
    pub fn invoke(
        &mut self,
        name: &str,
        args: &[Arg],
        group: (usize, usize),
        invoke_params: &[(&str, Scalar)],
    ) -> End {
        let end = self.invoke_here(name, args, group, invoke_params);
        let number = self.report.ended.len();
        self.report.ended.push(Ended {
            number,
            entry: name.to_string(),
            group,
            end: end.clone(),
        });
        end
    }

    fn invoke_here(
        &mut self,
        name: &str,
        args: &[Arg],
        (first, count): (usize, usize),
        invoke_params: &[(&str, Scalar)],
    ) -> End {
        let invalid = |what: String| End::Invalid { what, at: None };
        let Some(Entry::Handler(p, h)) = self.handler(name) else {
            return invalid(format!("no handler {name}"));
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
        // the arguments, against the handler's parameters
        if args.len() != handler.params.len() {
            return invalid(format!(
                "{} arguments for {} parameters",
                args.len(),
                handler.params.len()
            ));
        }
        let mut values = Vec::with_capacity(args.len());
        for (a, t) in args.iter().zip(&handler.params) {
            let ok = match (a, t) {
                (Arg::Tensor(id), Type::Tensor(f, _)) => {
                    self.heap.get(*id).is_ok_and(|x| x.format == *f)
                }
                (Arg::Scalar(s), Type::Scalar(st)) => s.format() == Some(st.format()),
                _ => false,
            };
            if !ok {
                return invalid(format!("an argument that does not fit {t:?}"));
            }
            values.push(match a {
                Arg::Tensor(id) => Value::Tensor(*id),
                Arg::Scalar(s) => Value::Scalar(*s),
            });
        }
        let at = BodyRef::Handler(p, h);
        let mut cx = Context {
            module,
            shapes: &self.shapes,
            params: &params,
            heap: &mut self.heap,
        };
        let mut result = None;
        let mut first_fault: Option<(usize, Fault, BodyRef, InstId)> = None;
        for rank in 0..count {
            let place = Place {
                rank,
                size: count,
                first,
                grid,
            };
            let frame = match Frame::new(module, at, values.clone()) {
                Ok(f) => f,
                Err(f) => return invalid(f.to_string()),
            };
            let Some(worker) = self.workers.get_mut(first + rank) else {
                return invalid("a worker past the loaded ones".into());
            };
            let mut thread = Thread::new(frame);
            match thread.run(worker, place, &mut cx, self.options.step_budget) {
                Stop::Returned(v) => {
                    if rank == 0 {
                        result = v;
                    }
                }
                Stop::Faulted { fault, at } => {
                    if first_fault.is_none() {
                        first_fault = Some((rank, fault, at.0, at.1));
                    }
                }
            }
        }
        match first_fault {
            Some((rank, Fault::Failure(reason), body, inst)) => End::Failed {
                reason,
                at: Some((rank, body, inst)),
            },
            Some((rank, Fault::Undefined(what), body, inst)) => End::Invalid {
                what,
                at: Some((rank, body, inst)),
            },
            None => match (result, handler.ret) {
                (Some(Value::Scalar(s)), Some(_)) => End::Completed(Some(s)),
                (None, None) => End::Completed(None),
                (None, Some(_)) => invalid("the leader returned no result".into()),
                (Some(_), _) => invalid("a result of another type".into()),
            },
        }
    }
}
