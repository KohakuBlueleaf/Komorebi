//! The place of every declaration in text order, by name: what a [`Name`] refers to.
//! A name declared twice keeps its first place (the second is an error already).

use std::collections::{HashMap, HashSet};

use super::attrs::word_args;
use super::consts::{eval, Known, Num};
use super::ty::Count;
use super::Name;
use crate::syntax::ast::{Item, Module, ParamValue, ProgramItem, TypeKind};
use crate::types::{MessageId, ParamId, RecordId, ScalarType};

/// The places of the module's declarations.
#[derive(Clone, Debug, Default)]
pub(crate) struct Index {
    /// The module's parameters.
    pub params: HashMap<String, ParamId>,
    /// The module's parameters bound at build (`@bind(build)`): constants.
    pub build_params: HashSet<String>,
    /// The whole-number values of the parameters bound at build.
    pub build_values: HashMap<String, u64>,
    /// Each parameter's value as its constant expression gives it, by `ParamId` (every
    /// declaration, one declared twice included); `None` for `= fit` and for a value
    /// that is not a constant expression.
    pub values: Vec<Option<Num>>,
    /// The functions: `func`, `combine` and `extern func`, in one list.
    pub funcs: HashMap<String, u32>,
    /// The programs.
    pub programs: HashMap<String, u32>,
    /// The record types: the module's under `None`, a program's under its name.
    records: HashMap<(Option<String>, String), RecordId>,
    /// The message types, as the record types.
    messages: HashMap<(Option<String>, String), MessageId>,
    /// A program's state members, every state block's in order.
    members: HashMap<(String, String), u32>,
    /// A program's handlers, runs and chains, each by its place among its kind.
    invocables: HashMap<(String, String), Name>,
}

impl Index {
    /// The places of `module`'s declarations.
    pub(crate) fn of(module: &Module) -> Index {
        let mut ix = Index::default();
        let mut counts = Counts::default();
        let params: Vec<&crate::syntax::ast::Param> = module
            .items
            .nodes()
            .filter_map(|item| match item {
                Item::Param(p) => Some(p),
                _ => None,
            })
            .collect();
        for (k, p) in params.iter().enumerate() {
            ix.params
                .entry(p.name.name.clone())
                .or_insert(ParamId(k as u32));
            if word_args(&p.attrs, "bind").contains("build") {
                ix.build_params.insert(p.name.name.clone());
            }
        }
        // each value in text order, naming only the parameters declared before it
        for (k, p) in params.iter().enumerate() {
            let float = matches!(&p.ty.kind, TypeKind::Scalar(s)
                if ScalarType::from_name(s.name()).is_some_and(ScalarType::is_float));
            let place = ParamId(k as u32);
            let value = match &p.value {
                ParamValue::Expr(e) => eval(e, float, &|n| ix.known(&n.name, Some(place))).ok(),
                ParamValue::Fit(_) => None,
            };
            ix.values.push(value);
            let first = ix.params.get(&p.name.name) == Some(&place);
            if let (true, true, Some(Num::Int(n))) =
                (first, ix.build_params.contains(&p.name.name), value)
            {
                if let Ok(n) = u64::try_from(n) {
                    ix.build_values.insert(p.name.name.clone(), n);
                }
            }
        }
        for item in module.items.nodes() {
            match item {
                Item::Message(m) => ix.message(&mut counts, None, &m.name.name),
                Item::Record(r) => ix.record(&mut counts, None, &r.name.name),
                Item::Func(f) | Item::Combine(f) => ix.func(&mut counts, &f.name.name),
                Item::Extern(f) => ix.func(&mut counts, &f.name.name),
                Item::Program(p) => {
                    let program = p.name.name.clone();
                    let place = counts.programs;
                    counts.programs += 1;
                    ix.programs.entry(program.clone()).or_insert(place);
                    let mut members = 0;
                    let (mut handlers, mut runs, mut chains) = (0, 0, 0);
                    for part in p.body.nodes() {
                        let scope = Some(program.clone());
                        match part {
                            ProgramItem::Message(m) => ix.message(&mut counts, scope, &m.name.name),
                            ProgramItem::Record(r) => ix.record(&mut counts, scope, &r.name.name),
                            ProgramItem::State(s) => {
                                for member in s.members.nodes() {
                                    let key = (program.clone(), member.name.name.clone());
                                    ix.members.entry(key).or_insert(members);
                                    members += 1;
                                }
                            }
                            ProgramItem::Handler(h) => {
                                let name = Name::Handler {
                                    program: place,
                                    index: handlers,
                                };
                                handlers += 1;
                                ix.invocable(&program, &h.name.name, name);
                            }
                            ProgramItem::Run(r) => {
                                let name = Name::Run {
                                    program: place,
                                    index: runs,
                                };
                                runs += 1;
                                ix.invocable(&program, &r.name.name, name);
                            }
                            ProgramItem::Chain(c) => {
                                let name = Name::Chain {
                                    program: place,
                                    index: chains,
                                };
                                chains += 1;
                                ix.invocable(&program, &c.name.name, name);
                            }
                        }
                    }
                }
                Item::Param(_)
                | Item::Tile(_)
                | Item::Format(_)
                | Item::Requires(_)
                | Item::Control(_) => {}
            }
        }
        ix
    }

    fn message(&mut self, counts: &mut Counts, scope: Option<String>, name: &str) {
        let place = MessageId(counts.messages);
        counts.messages += 1;
        self.messages
            .entry((scope, name.to_string()))
            .or_insert(place);
    }

    fn record(&mut self, counts: &mut Counts, scope: Option<String>, name: &str) {
        let place = RecordId(counts.records);
        counts.records += 1;
        self.records
            .entry((scope, name.to_string()))
            .or_insert(place);
    }

    fn func(&mut self, counts: &mut Counts, name: &str) {
        let place = counts.funcs;
        counts.funcs += 1;
        self.funcs.entry(name.to_string()).or_insert(place);
    }

    fn invocable(&mut self, program: &str, name: &str, place: Name) {
        self.invocables
            .entry((program.to_string(), name.to_string()))
            .or_insert(place);
    }

    /// The message type `name` as `program` sees it: its own, else the module's.
    pub(crate) fn message_ref(&self, program: Option<&str>, name: &str) -> Option<MessageId> {
        program
            .and_then(|p| self.messages.get(&(Some(p.to_string()), name.to_string())))
            .or_else(|| self.messages.get(&(None, name.to_string())))
            .copied()
    }

    /// The record type `name` as `program` sees it: its own, else the module's.
    pub(crate) fn record_ref(&self, program: Option<&str>, name: &str) -> Option<RecordId> {
        program
            .and_then(|p| self.records.get(&(Some(p.to_string()), name.to_string())))
            .or_else(|| self.records.get(&(None, name.to_string())))
            .copied()
    }

    /// The count a module parameter stands for: its value when it is bound at build,
    /// else the parameter.
    pub(crate) fn param_count(&self, name: &str) -> Option<Count> {
        if let Some(&n) = self.build_values.get(name) {
            return Some(Count::Known(n));
        }
        self.params.get(name).map(|&p| Count::Param(p))
    }

    /// What the name `name` stands for in a constant expression: a module parameter
    /// bound at build with its value, one bound otherwise, one declared at or after
    /// `before` (the parameter whose value names it), or none.
    pub(crate) fn known(&self, name: &str, before: Option<ParamId>) -> Known {
        let Some(&p) = self.params.get(name) else {
            return Known::NotParam;
        };
        if before.is_some_and(|b| p.0 >= b.0) {
            return Known::Later;
        }
        if self.build_params.contains(name) {
            Known::Build(self.values.get(p.0 as usize).copied().flatten())
        } else {
            Known::Unbuilt
        }
    }

    /// The state member `name` of `program`.
    pub(crate) fn member(&self, program: &str, name: &str) -> Option<Name> {
        let member = *self.members.get(&(program.to_string(), name.to_string()))?;
        let program = *self.programs.get(program)?;
        Some(Name::State { program, member })
    }

    /// The handler, run or chain `name` of `program`.
    pub(crate) fn invocable_of(&self, program: &str, name: &str) -> Option<Name> {
        self.invocables
            .get(&(program.to_string(), name.to_string()))
            .cloned()
    }
}

/// How many of each kind of declaration are seen so far.
#[derive(Default)]
struct Counts {
    funcs: u32,
    programs: u32,
    records: u32,
    messages: u32,
}
