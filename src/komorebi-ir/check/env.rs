//! The declarations of a module, collected before any body is checked: formats,
//! parameters, messages, records, functions, programs and what each program holds.

use std::collections::{HashMap, HashSet};

use super::ty::{Count, Ty};
use crate::syntax::ast::{
    Field, Ident, Item, List, Module, ParamDecl, ProgramItem, Span, Type, TypeKind,
};
use crate::types::Format;

use super::codes;
use super::Checker;

/// Where a type is written; it decides which types are allowed there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// A state member: a count `[n]` may follow a scalar or a tile type.
    StateMember,
    /// A parameter of a `func`, a `combine` or an `extern func`.
    FuncParam,
    /// A parameter of a handler, a run, a chain or a control program.
    Param,
    /// A field of a message or a record.
    Field,
    /// The result of a function, a handler or a control program.
    Result,
    /// The type of a `let`, a `var` or a module parameter.
    Local,
}

/// What kind of function a name declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FuncKind {
    /// `func`.
    Func,
    /// `combine`.
    Combine,
    /// `extern func`.
    Extern,
}

/// A function's signature.
#[derive(Clone, Debug)]
pub(crate) struct FuncSig {
    /// What kind of function.
    pub kind: FuncKind,
    /// The parameters' types.
    pub params: Vec<Ty>,
    /// The result's type.
    pub ret: Ty,
}

/// A state member of a program.
#[derive(Clone, Debug)]
pub(crate) struct MemberInfo {
    /// Its type.
    pub ty: Ty,
    /// `@filled(element)`: each element is filled once written.
    pub by_element: bool,
}

/// Something a control program can invoke: a handler, a run or a chain.
#[derive(Clone, Debug)]
pub(crate) struct Invocable {
    /// The parameters' types.
    pub params: Vec<Ty>,
    /// The result's type, when it has one.
    pub ret: Option<Ty>,
    /// True for a handler (only a handler can be a chain's call).
    pub is_handler: bool,
}

/// What a program declares.
#[derive(Clone, Debug, Default)]
pub(crate) struct ProgramInfo {
    /// Its message types: field names and types, in order.
    pub messages: HashMap<String, Vec<(String, Ty)>>,
    /// Its record types.
    pub records: HashMap<String, Vec<(String, Ty)>>,
    /// Its state members.
    pub state: HashMap<String, MemberInfo>,
    /// Its handlers, runs and chains.
    pub invocables: HashMap<String, Invocable>,
}

/// The module's declarations.
#[derive(Clone, Debug, Default)]
pub(crate) struct Env {
    /// The formats of the `format` lines.
    pub formats: HashSet<Format>,
    /// The module's parameters and their types.
    pub params: HashMap<String, Ty>,
    /// The module's message types.
    pub messages: HashMap<String, Vec<(String, Ty)>>,
    /// The module's record types.
    pub records: HashMap<String, Vec<(String, Ty)>>,
    /// The module's functions.
    pub funcs: HashMap<String, FuncSig>,
    /// The module's programs.
    pub programs: HashMap<String, ProgramInfo>,
    /// The module's tile shape, rows and columns, when its `tile` line gives one.
    pub tile: Option<(u64, u64)>,
    /// The bytes of each record type whose fields are all scalars.
    pub record_bytes: HashMap<String, u64>,
}

impl Env {
    /// The fields of a message type visible in `program` (its own, then the module's).
    pub fn message(&self, program: Option<&str>, name: &str) -> Option<&Vec<(String, Ty)>> {
        program
            .and_then(|p| self.programs.get(p))
            .and_then(|p| p.messages.get(name))
            .or_else(|| self.messages.get(name))
    }

    /// The fields of a record type visible in `program`.
    pub fn record(&self, program: Option<&str>, name: &str) -> Option<&Vec<(String, Ty)>> {
        program
            .and_then(|p| self.programs.get(p))
            .and_then(|p| p.records.get(name))
            .or_else(|| self.records.get(name))
    }
}

/// Names declared in one scope, to find one declared twice.
#[derive(Default)]
pub(crate) struct Names {
    seen: HashSet<String>,
}

impl Names {
    /// Records `name`; reports it when the scope already has it.
    pub fn declare(&mut self, checker: &mut Checker, name: &Ident) {
        if !self.seen.insert(name.name.clone()) {
            checker.error(
                codes::NAME_TWICE,
                format!("`{}` is declared twice", name.name),
                name.span,
            );
        }
    }
}

impl Checker<'_> {
    /// Collects the module's declarations and checks the types they write.
    pub(crate) fn collect(&mut self, module: &Module) -> Env {
        let mut env = Env::default();
        for item in module.items.nodes() {
            if let Item::Format(decl) = item {
                for name in &decl.formats {
                    match Format::from_name(&name.name) {
                        Some(format) => {
                            env.formats.insert(format);
                        }
                        None => self.error(
                            codes::UNKNOWN_FORMAT,
                            format!("`{}` is not a format", name.name),
                            name.span,
                        ),
                    }
                }
            }
        }
        // Record and message names first, so a type may name one declared later.
        for item in module.items.nodes() {
            match item {
                Item::Message(m) => {
                    self.message_names.insert(m.name.name.clone());
                }
                Item::Record(r) => {
                    self.record_names.insert(r.name.name.clone());
                    if let Some(bytes) = record_bytes(&r.fields) {
                        env.record_bytes.insert(r.name.name.clone(), bytes);
                    }
                }
                Item::Tile(t) => {
                    if env.tile.is_none() {
                        if let (Some(rows), Some(cols)) =
                            (parse_int(&t.rows.text), parse_int(&t.cols.text))
                        {
                            env.tile = Some((rows, cols));
                        }
                    }
                }
                Item::Program(p) => {
                    for part in p.body.nodes() {
                        match part {
                            ProgramItem::Message(m) => {
                                self.message_names.insert(m.name.name.clone());
                            }
                            ProgramItem::Record(r) => {
                                self.record_names.insert(r.name.name.clone());
                                if let Some(bytes) = record_bytes(&r.fields) {
                                    env.record_bytes.insert(r.name.name.clone(), bytes);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        let mut names = Names::default();
        for item in module.items.nodes() {
            match item {
                Item::Tile(_) | Item::Format(_) | Item::Requires(_) => {}
                Item::Param(p) => {
                    names.declare(self, &p.name);
                    let ty = self.ty(&env, &p.ty, Place::Local);
                    env.params.insert(p.name.name.clone(), ty);
                }
                Item::Message(m) => {
                    names.declare(self, &m.name);
                    let fields = self.fields(&env, &m.fields, true);
                    env.messages.insert(m.name.name.clone(), fields);
                }
                Item::Record(r) => {
                    names.declare(self, &r.name);
                    let fields = self.fields(&env, &r.fields, false);
                    env.records.insert(r.name.name.clone(), fields);
                }
                Item::Func(f) | Item::Combine(f) => {
                    names.declare(self, &f.name);
                    let kind = if matches!(item, Item::Func(_)) {
                        FuncKind::Func
                    } else {
                        FuncKind::Combine
                    };
                    let params = self.func_params(&env, &f.params, kind);
                    let ret = self.ty(&env, &f.ret, Place::Result);
                    env.funcs
                        .insert(f.name.name.clone(), FuncSig { kind, params, ret });
                }
                Item::Extern(f) => {
                    names.declare(self, &f.name);
                    let params = self.func_params(&env, &f.params, FuncKind::Extern);
                    let ret = self.ty(&env, &f.ret, Place::Result);
                    let kind = FuncKind::Extern;
                    env.funcs
                        .insert(f.name.name.clone(), FuncSig { kind, params, ret });
                }
                Item::Program(p) => {
                    names.declare(self, &p.name);
                    env.programs
                        .insert(p.name.name.clone(), ProgramInfo::default());
                }
                Item::Control(c) => {
                    names.declare(self, &c.name);
                }
            }
        }
        for item in module.items.nodes() {
            if let Item::Program(p) = item {
                let info = self.program_info(&env, &p.body.nodes().collect::<Vec<_>>());
                env.programs.insert(p.name.name.clone(), info);
            }
        }
        env
    }

    fn program_info(&mut self, env: &Env, parts: &[&ProgramItem]) -> ProgramInfo {
        let mut info = ProgramInfo::default();
        let mut names = Names::default();
        for part in parts {
            match part {
                ProgramItem::Message(m) => {
                    names.declare(self, &m.name);
                    let fields = self.fields(env, &m.fields, true);
                    info.messages.insert(m.name.name.clone(), fields);
                }
                ProgramItem::Record(r) => {
                    names.declare(self, &r.name);
                    let fields = self.fields(env, &r.fields, false);
                    info.records.insert(r.name.name.clone(), fields);
                }
                ProgramItem::State(s) => {
                    for member in s.members.nodes() {
                        names.declare(self, &member.name);
                        let ty = self.ty(env, &member.ty, Place::StateMember);
                        let by_element = member
                            .attrs
                            .iter()
                            .chain(s.attrs.iter())
                            .any(|a| a.path.len() == 1 && a.path[0].name == "filled");
                        info.state
                            .insert(member.name.name.clone(), MemberInfo { ty, by_element });
                    }
                }
                ProgramItem::Handler(h) => {
                    names.declare(self, &h.name);
                    let params = self.params(env, &h.params);
                    let ret = h.ret.as_ref().map(|t| self.ty(env, t, Place::Result));
                    let entry = Invocable {
                        params,
                        ret,
                        is_handler: true,
                    };
                    info.invocables.insert(h.name.name.clone(), entry);
                }
                ProgramItem::Run(r) => {
                    names.declare(self, &r.name);
                    let params = self.params(env, &r.params);
                    let entry = Invocable {
                        params,
                        ret: None,
                        is_handler: false,
                    };
                    info.invocables.insert(r.name.name.clone(), entry);
                }
                ProgramItem::Chain(c) => {
                    names.declare(self, &c.name);
                    let params = self.params(env, &c.params);
                    let entry = Invocable {
                        params,
                        ret: None,
                        is_handler: false,
                    };
                    info.invocables.insert(c.name.name.clone(), entry);
                }
            }
        }
        info
    }

    /// The types of a handler's, run's, chain's or control program's parameters.
    pub(crate) fn params(&mut self, env: &Env, params: &List<ParamDecl>) -> Vec<Ty> {
        let mut names = Names::default();
        let mut out = Vec::new();
        for p in params.values() {
            names.declare(self, &p.name);
            let mut ty = self.ty(env, &p.ty, Place::Param);
            if let Ty::Tensor { replicated, .. } = &mut ty {
                *replicated = p.attrs.iter().any(|a| is_named(a, "replicated"));
            }
            out.push(ty);
        }
        out
    }

    fn func_params(&mut self, env: &Env, params: &List<ParamDecl>, kind: FuncKind) -> Vec<Ty> {
        let mut names = Names::default();
        let mut out = Vec::new();
        for p in params.values() {
            names.declare(self, &p.name);
            let ty = self.ty(env, &p.ty, Place::FuncParam);
            if kind != FuncKind::Extern && matches!(ty, Ty::View { .. }) {
                self.error(
                    codes::FUNC_TAKES_VIEW,
                    format!(
                        "`{}` takes a view; a func is pure and takes no view",
                        p.name.name
                    ),
                    p.span,
                );
            }
            out.push(ty);
        }
        out
    }

    fn fields(&mut self, env: &Env, fields: &List<Field>, message: bool) -> Vec<(String, Ty)> {
        let mut names = Names::default();
        let mut out = Vec::new();
        for f in fields.values() {
            names.declare(self, &f.name);
            let ty = self.ty(env, &f.ty, Place::Field);
            if message
                && f.ty.count.is_none()
                && !matches!(ty, Ty::Int | Ty::Float | Ty::Unknown)
                && !matches!(f.ty.kind, TypeKind::Bool)
            {
                self.error(
                    codes::MESSAGE_FIELD,
                    format!(
                        "the field `{}` is {ty}; a message's fields are scalars",
                        f.name.name
                    ),
                    f.span,
                );
            }
            out.push((f.name.name.clone(), ty));
        }
        out
    }

    /// The type a written type stands for, with the diagnostics of the place it is in.
    pub(crate) fn ty(&mut self, env: &Env, ty: &Type, place: Place) -> Ty {
        let base = match &ty.kind {
            TypeKind::Scalar(s) => {
                if s.name().starts_with('f') {
                    Ty::Float
                } else {
                    Ty::Int
                }
            }
            TypeKind::Bool => {
                if place != Place::Local {
                    let what = match place {
                        Place::StateMember => "of state",
                        Place::Field => "of a field",
                        Place::Result => "of a result",
                        _ => "of a parameter",
                    };
                    self.error(
                        codes::BOOL_PLACE,
                        format!("`bool` is not a type {what}"),
                        ty.span,
                    );
                }
                Ty::Bool
            }
            TypeKind::Tile { format } => {
                self.tile_needed(env, ty.span);
                Ty::Tile(self.format(env, format))
            }
            TypeKind::Tensor { format } => {
                self.tile_needed(env, ty.span);
                Ty::Tensor {
                    format: self.format(env, format),
                    replicated: false,
                }
            }
            TypeKind::View => {
                self.tile_needed(env, ty.span);
                Ty::View {
                    format: None,
                    tiles: Count::Unknown,
                }
            }
            TypeKind::Named(name) => {
                if self.message_names.contains(&name.name) {
                    Ty::Message(name.name.clone())
                } else if self.record_names.contains(&name.name) {
                    Ty::Record(name.name.clone())
                } else {
                    self.error(
                        codes::UNKNOWN_NAME,
                        format!("`{}` is not a declared type", name.name),
                        name.span,
                    );
                    Ty::Unknown
                }
            }
        };
        let Some(count) = &ty.count else {
            return base;
        };
        let n = match &count.kind {
            crate::syntax::ast::ExprKind::Int(text) => {
                parse_int(text).map(Count::Known).unwrap_or(Count::Unknown)
            }
            _ => Count::Unknown,
        };
        match place {
            Place::StateMember => match base {
                Ty::Tile(f) => Ty::Block(f, n),
                Ty::Unknown => Ty::Unknown,
                other => Ty::Scalars(Box::new(other), n),
            },
            Place::FuncParam if matches!(base, Ty::Tile(_)) => {
                self.error(
                    codes::NOT_BUILT,
                    "a function on a block of several tiles is named in the language and not \
                     built yet"
                        .into(),
                    ty.span,
                );
                Ty::Unknown
            }
            _ => {
                self.error(
                    codes::WRONG_TYPE,
                    "a count `[n]` follows a type only in a state member".into(),
                    ty.span,
                );
                Ty::Unknown
            }
        }
    }

    /// `K118`: a tile, a tensor or a view in a module with no `tile` line, reported at
    /// its first use.
    fn tile_needed(&mut self, env: &Env, span: Span) {
        if env.tile.is_none() && !self.tile_missing {
            self.tile_missing = true;
            self.error(
                codes::TILE_NOT_DECLARED,
                "a tile, a tensor or a view is used and the module has no `tile` line (its \
                 first use)"
                    .into(),
                span,
            );
        }
    }

    fn format(&mut self, env: &Env, name: &Ident) -> Option<Format> {
        match Format::from_name(&name.name) {
            None => {
                self.error(
                    codes::UNKNOWN_FORMAT,
                    format!("`{}` is not a format", name.name),
                    name.span,
                );
                None
            }
            Some(format) => {
                if !env.formats.contains(&format) && self.undeclared_formats.insert(format) {
                    self.error(
                        codes::FORMAT_NOT_DECLARED,
                        format!(
                            "the format `{format}` is used and not declared in `format` (its \
                             first use)"
                        ),
                        name.span,
                    );
                }
                Some(format)
            }
        }
    }
}

/// The bytes of a record whose fields are all scalars, in declared order with no gap.
fn record_bytes(fields: &List<Field>) -> Option<u64> {
    let mut total = 0;
    for f in fields.values() {
        let TypeKind::Scalar(s) = &f.ty.kind else {
            return None;
        };
        if f.ty.count.is_some() {
            return None;
        }
        total += Format::from_name(s.name())?.bytes()? as u64;
    }
    Some(total)
}

/// True for an attribute with this one undotted name.
pub(crate) fn is_named(attr: &crate::syntax::ast::Attr, name: &str) -> bool {
    attr.path.len() == 1 && attr.path[0].name == name
}

/// The value of an integer literal as written, decimal or `0x` hexadecimal.
pub(crate) fn parse_int(text: &str) -> Option<u64> {
    match text.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => text.parse().ok(),
    }
}
