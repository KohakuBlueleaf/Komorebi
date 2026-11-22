//! Derive: what a lowered module does, read off its bodies once. Each handler's and
//! run's effects (the tensors it reads and writes, through which view, and where; the
//! state it uses; its collectives; the functions it calls), an external function's
//! fallback's effects on its view parameters, each chain's dependency relation, and
//! the module's requirement set with what caused each requirement.
//!
//! Derive writes `effects`, `deps` and `requirements` and nothing else: no body is
//! rewritten, so every note keyed by an instruction stays true.

use std::collections::{BTreeSet, HashMap};

use crate::program::body::{Body, InstId, Op, ScalarsFrom, ValueId};
use crate::program::derive::{
    Access, Cause, ChainDeps, ChainEdge, Effects, Requirement, StateAccess, StateUse, TensorAccess,
};
use crate::program::{BodyRef, CallArg, FuncKind, Module};
use crate::types::{Ability, Entry, FuncId, HandlerId, On, ProgramId, RunId, Type};

/// Fills the effects of every handler, run and external function's fallback, the
/// dependencies of every chain, and the requirement set.
pub fn derive(module: &mut Module) {
    let calls = call_closure(module);
    let fallback_effects: Vec<Effects> = module
        .funcs
        .iter()
        .enumerate()
        .map(|(f, func)| match (&func.kind, &func.body) {
            (FuncKind::Extern { .. }, Some(body)) => {
                effects(module, &calls, &[(BodyRef::Func(FuncId(f as u32)), body)])
            }
            _ => Effects::default(),
        })
        .collect();
    for (func, e) in module.funcs.iter_mut().zip(fallback_effects) {
        func.effects = e;
    }
    for p in 0..module.programs.len() {
        let pid = ProgramId(p as u32);
        let Some(prog) = module.programs.get(p) else {
            continue;
        };
        let handler_effects: Vec<Effects> = prog
            .handlers
            .iter()
            .enumerate()
            .map(|(h, handler)| {
                let r = BodyRef::Handler(pid, HandlerId(h as u32));
                effects(module, &calls, &[(r, &handler.body)])
            })
            .collect();
        let run_effects: Vec<Effects> = prog
            .runs
            .iter()
            .enumerate()
            .map(|(r, run)| {
                let rid = RunId(r as u32);
                let mut bodies = vec![(BodyRef::Seed(pid, rid), &run.seed)];
                for (k, (_, body)) in run.on_message.iter().enumerate() {
                    bodies.push((BodyRef::On(pid, rid, k as u32), body));
                }
                effects(module, &calls, &bodies)
            })
            .collect();
        let chain_deps: Vec<ChainDeps> = prog
            .chains
            .iter()
            .map(|chain| deps(&chain.calls, &handler_effects))
            .collect();
        if let Some(prog) = module.programs.get_mut(p) {
            for (h, e) in prog.handlers.iter_mut().zip(handler_effects) {
                h.effects = e;
            }
            for (r, e) in prog.runs.iter_mut().zip(run_effects) {
                r.effects = e;
            }
            for (c, d) in prog.chains.iter_mut().zip(chain_deps) {
                c.deps = d;
            }
        }
    }
    module.requirements = requirements(module);
}

/// For each function, every function it calls, directly or through another.
fn call_closure(module: &Module) -> Vec<BTreeSet<u32>> {
    let direct: Vec<BTreeSet<u32>> = module
        .funcs
        .iter()
        .map(|f| {
            f.body
                .iter()
                .flat_map(|b| b.insts.iter())
                .filter_map(|i| match i.op {
                    Op::CallFunc(g, _) => Some(g.0),
                    _ => None,
                })
                .collect()
        })
        .collect();
    let mut all = direct.clone();
    // a fixed point: at most one round for each function
    for _ in 0..module.funcs.len() {
        let mut changed = false;
        for set in all.iter_mut() {
            let reach: Vec<u32> = set
                .iter()
                .flat_map(|g| direct.get(*g as usize).into_iter().flatten().copied())
                .collect();
            for g in reach {
                changed |= set.insert(g);
            }
        }
        if !changed {
            break;
        }
    }
    all
}

/// Where a value comes from: a tensor parameter, and the instruction that made the
/// view of it (none for the tensor itself).
#[derive(Clone, Copy)]
struct Origin {
    param: u32,
    view: Option<InstId>,
}

/// The effects of the bodies of one handler or run, in their order.
fn effects(module: &Module, calls: &[BTreeSet<u32>], bodies: &[(BodyRef, &Body)]) -> Effects {
    let mut out = Effects::default();
    let mut called = BTreeSet::new();
    for (r, body) in bodies {
        let mut origin: HashMap<u32, Origin> = HashMap::new();
        let of = |origin: &HashMap<u32, Origin>, v: &ValueId| origin.get(&v.0).copied();
        for (i, inst) in body.insts.iter().enumerate() {
            let at = InstId(i as u32);
            let def = inst.defs.first().map(|v| v.0);
            let access = |out: &mut Effects, o: Option<Origin>, how: Access| {
                if let Some(Origin {
                    param,
                    view: Some(view),
                }) = o
                {
                    out.tensors.push(TensorAccess {
                        param,
                        body: *r,
                        view,
                        access: how,
                        at,
                    });
                }
            };
            let state = |out: &mut Effects, member, how| {
                out.state.push(StateAccess {
                    member,
                    body: *r,
                    how,
                    at,
                })
            };
            match &inst.op {
                Op::Arg(p) => {
                    let ty = body.values.get(def.unwrap_or(u32::MAX) as usize);
                    // a view parameter (a function's) is the view its `Arg` makes
                    let view = match ty {
                        Some(Type::Tensor(..)) => Some(None),
                        Some(Type::View(_)) => Some(Some(at)),
                        _ => None,
                    };
                    if let (Some(view), Some(d)) = (view, def) {
                        origin.insert(d, Origin { param: *p, view });
                    }
                }
                // a view: the instruction that makes it is the view's
                Op::Tiles { tensor: x, .. }
                | Op::Rect { tensor: x, .. }
                | Op::Items { tensor: x, .. }
                | Op::Partition(x, _, _)
                | Op::Turn(x) => {
                    if let (Some(o), Some(d)) = (of(&origin, x), def) {
                        origin.insert(
                            d,
                            Origin {
                                param: o.param,
                                view: Some(at),
                            },
                        );
                    }
                }
                // a stream or a sink carries its view
                Op::Stream(x, _) | Op::Sink(x, _) | Op::Words(x) | Op::Records(x, _, _) => {
                    if let (Some(o), Some(d)) = (of(&origin, x), def) {
                        origin.insert(d, o);
                    }
                    if matches!(inst.op, Op::Words(_) | Op::Records(..)) {
                        access(&mut out, of(&origin, x), Access::Read);
                    }
                }
                Op::ForStream { stream } => access(&mut out, of(&origin, stream), Access::Stream),
                Op::ForZip { streams } => {
                    for s in streams {
                        access(&mut out, of(&origin, s), Access::Stream);
                    }
                }
                Op::Push(s, _) => access(&mut out, of(&origin, s), Access::Sink),
                Op::ReadTile(x, _) => access(&mut out, of(&origin, x), Access::Read),
                Op::Store(x, _) => access(&mut out, of(&origin, x), Access::Store),
                Op::Put { items, src, .. } => {
                    state(&mut out, *src, StateUse::Read);
                    access(&mut out, of(&origin, items), Access::Put);
                }
                // a body given from a state member reads it whole, as a `put`'s source
                Op::Send {
                    body: Some(ScalarsFrom::Member(m)),
                    ..
                }
                | Op::Offer(_, _, Some(ScalarsFrom::Member(m))) => {
                    state(&mut out, *m, StateUse::Read)
                }
                Op::Fill(m, _) | Op::ZeroState(m) => state(&mut out, *m, StateUse::FillWhole),
                Op::StateGet(m, _) | Op::Bump(m, _) => state(&mut out, *m, StateUse::Read),
                Op::StateSet(m, _, _) => state(&mut out, *m, StateUse::WriteElement),
                Op::ReduceGroup { .. }
                | Op::AllReduce { .. }
                | Op::Gather(_)
                | Op::Broadcast(..)
                | Op::Sync(_)
                | Op::ReduceTo(..)
                | Op::BroadcastFrom(..) => out.collectives.push((*r, at)),
                Op::CallFunc(f, args) => {
                    called.insert(f.0);
                    called.extend(calls.get(f.0 as usize).into_iter().flatten().copied());
                    if let Some(FuncKind::Extern { reads, writes, .. }) =
                        module.funcs.get(f.0 as usize).map(|x| &x.kind)
                    {
                        for k in reads {
                            let o = args.get(*k as usize).and_then(|v| of(&origin, v));
                            access(&mut out, o, Access::ExternRead);
                        }
                        for k in writes {
                            let o = args.get(*k as usize).and_then(|v| of(&origin, v));
                            access(&mut out, o, Access::ExternWrite);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out.calls = called.into_iter().map(FuncId).collect();
    out
}

fn writes(a: Access) -> bool {
    matches!(
        a,
        Access::Sink | Access::Store | Access::Put | Access::ExternWrite
    )
}

/// A chain's dependencies: for each call, each earlier call that wrote a tensor
/// parameter of the chain it reads.
fn deps(calls: &[crate::program::Call], handlers: &[Effects]) -> ChainDeps {
    // the chain's parameter each argument of a call passes, and whether by `past`
    let passed = |call: &crate::program::Call, k: u32| match call.args.get(k as usize) {
        Some(CallArg::Param(p)) => Some((*p, false)),
        Some(CallArg::Past(p)) => Some((*p, true)),
        _ => None,
    };
    let mut edges = Vec::new();
    for (reader, rc) in calls.iter().enumerate() {
        let Some(re) = handlers.get(rc.handler.0 as usize) else {
            continue;
        };
        for (writer, wc) in calls.iter().enumerate().take(reader) {
            let Some(we) = handlers.get(wc.handler.0 as usize) else {
                continue;
            };
            for (w, wa) in we.tensors.iter().enumerate() {
                if !writes(wa.access) {
                    continue;
                }
                let Some((param, _)) = passed(wc, wa.param) else {
                    continue;
                };
                for (r, ra) in re.tensors.iter().enumerate() {
                    if writes(ra.access) {
                        continue;
                    }
                    if let Some((p, past)) = passed(rc, ra.param) {
                        if p == param {
                            edges.push(ChainEdge {
                                writer: writer as u32,
                                reader: reader as u32,
                                param,
                                write: w as u32,
                                read: r as u32,
                                past,
                            });
                        }
                    }
                }
            }
        }
    }
    ChainDeps { edges }
}

/// The requirement set: the stated ones in the order written, then each one the
/// module needs and does not state, in the order of its first cause; each with every
/// cause, `Stated` first.
fn requirements(module: &Module) -> Vec<Requirement> {
    let mut out: Vec<Requirement> = module
        .requirements
        .iter()
        .filter(|r| r.causes.contains(&Cause::Stated))
        .map(|r| Requirement {
            ability: r.ability.clone(),
            causes: vec![Cause::Stated],
        })
        .collect();
    let mut add =
        |ability: Ability, cause: Cause| match out.iter_mut().find(|r| r.ability == ability) {
            Some(r) => r.causes.push(cause),
            None => out.push(Requirement {
                ability,
                causes: vec![cause],
            }),
        };
    for (f, func) in module.funcs.iter().enumerate() {
        if let (FuncKind::Extern { implementation, .. }, None) = (&func.kind, &func.body) {
            add(
                Ability::Impl(implementation.clone()),
                Cause::ExternWithoutFallback(FuncId(f as u32)),
            );
        }
    }
    for (p, prog) in module.programs.iter().enumerate() {
        let pid = ProgramId(p as u32);
        for (h, handler) in prog.handlers.iter().enumerate() {
            if matches!(handler.on, On::Grid(..)) {
                add(Ability::Grid, Cause::GridHandler(pid, HandlerId(h as u32)));
            }
        }
        for r in 0..prog.runs.len() {
            add(
                Ability::Dynamic,
                Cause::DynamicRun(Entry::Run(pid, RunId(r as u32))),
            );
        }
    }
    out
}
