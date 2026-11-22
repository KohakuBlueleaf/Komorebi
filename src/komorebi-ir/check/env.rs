//! The declarations of a module, collected before any body is checked: formats,
//! parameters, messages, records, functions, programs and what each program holds.

use std::collections::{HashMap, HashSet};

use super::ty::{Count, Shape, Ty};
use crate::syntax;
use crate::syntax::ast::{
    Field, Ident, Item, List, Module, ParamDecl, ProgramItem, Span, Type, TypeKind,
};
use crate::types::{Format, ScalarType};

use super::codes;
use super::consts::Num;
use super::index::Index;
use super::Checker;

/// Why the rows of a block are refused: a number written in the text or a module
/// parameter bound at build (R45); one bound at load or at the invocation, or
/// `group.size`, is not built.
pub(super) const ROWS_BUILT: &str = "the rows of a block `[r, c]` are a number written in the \
     text or a module parameter bound at build (one bound at load or at the invocation, or \
     `group.size`, there is not built)";

/// Where a type is written; it decides which types are allowed there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// A state member: a count `[n]` may follow a scalar or a tile type.
    StateMember,
    /// A parameter of a `func`, a `combine` or an `extern func`: a count `[n]` may
    /// follow a tile type (a block; a `func`'s only).
    FuncParam,
    /// A parameter of a handler, a run, a chain or a control program.
    Param,
    /// A field of a message or a record.
    Field,
    /// The result of a handler or a control program.
    Result,
    /// The result of a `func`, a `combine` or an `extern func`: as a parameter's.
    FuncResult,
    /// The type of a `let`, a `var` or a module parameter.
    Local,
    /// The type of a `var` of `zeros`: a count `[n]` may follow a tile type, a block of
    /// one row of n tiles (R48: today's field one handler's compute fills, reads and
    /// frees).
    VarOfZeros,
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

/// The names of declared parameters, in order.
fn param_names(params: &List<ParamDecl>) -> Vec<String> {
    params.values().map(|p| p.name.name.clone()).collect()
}

/// A function's signature.
#[derive(Clone, Debug)]
pub(crate) struct FuncSig {
    /// What kind of function.
    pub kind: FuncKind,
    /// The parameters' types.
    pub params: Vec<Ty>,
    /// The parameters' names, in order.
    pub names: Vec<String>,
    /// The result's type.
    pub ret: Ty,
}

/// A state member of a program.
#[derive(Clone, Debug)]
pub(crate) struct MemberInfo {
    /// Its type.
    pub ty: Ty,
}

/// Something a control program can invoke: a handler, a run or a chain.
#[derive(Clone, Debug)]
pub(crate) struct Invocable {
    /// The parameters' types.
    pub params: Vec<Ty>,
    /// The parameters' names, in order.
    pub names: Vec<String>,
    /// The result's type, when it has one.
    pub ret: Option<Ty>,
    /// True for a handler (only a handler can be a chain's call).
    pub is_handler: bool,
}

/// A message type: its fields and the bytes of its body.
#[derive(Clone, Debug, Default)]
pub(crate) struct MessageInfo {
    /// Field names and types, in order.
    pub fields: Vec<(String, Ty)>,
    /// The bytes of its body (`@body(n)`), when it has one.
    pub body: Option<u64>,
}

/// What a program declares.
#[derive(Clone, Debug, Default)]
pub(crate) struct ProgramInfo {
    /// Its message types.
    pub messages: HashMap<String, MessageInfo>,
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
    pub messages: HashMap<String, MessageInfo>,
    /// The module's record types.
    pub records: HashMap<String, Vec<(String, Ty)>>,
    /// The module's functions.
    pub funcs: HashMap<String, FuncSig>,
    /// The module's programs.
    pub programs: HashMap<String, ProgramInfo>,
    /// The module's default tile shape, rows and columns, when its first `tile` line
    /// gives one.
    pub tile: Option<(u64, u64)>,
    /// The shapes of every `tile` line, in order, each once (the first is the default).
    pub tiles: Vec<(u64, u64)>,
    /// The shape of each function, by its place among the functions.
    pub func_tiles: HashMap<u32, (u64, u64)>,
    /// The shape of each state member of tiles, by its program's place and its own.
    pub member_tiles: HashMap<(u32, u32), (u64, u64)>,
    /// The bytes of each record type whose fields are all scalars.
    pub record_bytes: HashMap<String, u64>,
    /// The place of every declaration, by name.
    pub ix: Index,
}

impl Env {
    /// A message type visible in `program` (its own, then the module's).
    pub fn message(&self, program: Option<&str>, name: &str) -> Option<&MessageInfo> {
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

/// The kind of an item a name declares, for the language's names where that kind
/// stands (R50).
#[derive(Clone, Copy)]
pub(crate) enum Declared {
    /// A `func`, a `combine`, an `extern func`.
    Function,
    /// A local, a parameter, a loop's name, a state member, a module constant.
    Value,
    /// A record, a message.
    Type,
}

impl Checker<'_> {
    /// Refuses an item of the module declared by a bare name that is the language's
    /// where its kind stands: it is written between backticks (R50).
    pub(crate) fn bare_word(&mut self, name: &Ident, kind: Declared) {
        let (word, what): (fn(&str) -> bool, &str) = match kind {
            Declared::Function => (syntax::function_word, "function"),
            Declared::Value => (syntax::value_word, "value"),
            Declared::Type => (syntax::type_word, "type"),
        };
        if !name.quoted && word(&name.name) {
            self.error(
                codes::BARE_WORD,
                format!(
                    "`{0}` is the language's where a {what} stands: a {what} of the module \
                     named `{0}` is written between backticks, at its declaration and at \
                     every use",
                    name.name
                ),
                name.span,
            );
        }
    }

    /// Collects the module's declarations and checks the types they write.
    pub(crate) fn collect(&mut self, module: &Module) -> Env {
        let mut env = Env {
            ix: Index::of(module),
            ..Env::default()
        };
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
                    self.bare_word(&p.name, Declared::Value);
                    let ty = self.ty(&env, &p.ty, Place::Local);
                    env.params.insert(p.name.name.clone(), ty);
                }
                Item::Message(m) => {
                    names.declare(self, &m.name);
                    self.bare_word(&m.name, Declared::Type);
                    let fields = self.fields(&env, &m.fields, true);
                    let body = body_bytes(&m.attrs);
                    env.messages
                        .insert(m.name.name.clone(), MessageInfo { fields, body });
                }
                Item::Record(r) => {
                    names.declare(self, &r.name);
                    self.bare_word(&r.name, Declared::Type);
                    let fields = self.fields(&env, &r.fields, false);
                    env.records.insert(r.name.name.clone(), fields);
                }
                Item::Func(f) | Item::Combine(f) => {
                    names.declare(self, &f.name);
                    self.bare_word(&f.name, Declared::Function);
                    let kind = if matches!(item, Item::Func(_)) {
                        FuncKind::Func
                    } else {
                        FuncKind::Combine
                    };
                    let params = self.func_params(&env, &f.params, kind);
                    let ret = self.ty(&env, &f.ret, Place::FuncResult);
                    self.on_blocks(&f.name.name, kind, &params, &ret, f.span);
                    let names = param_names(&f.params);
                    env.funcs.insert(
                        f.name.name.clone(),
                        FuncSig {
                            kind,
                            params,
                            names,
                            ret,
                        },
                    );
                }
                Item::Extern(f) => {
                    names.declare(self, &f.name);
                    self.bare_word(&f.name, Declared::Function);
                    let params = self.func_params(&env, &f.params, FuncKind::Extern);
                    let ret = self.ty(&env, &f.ret, Place::FuncResult);
                    let kind = FuncKind::Extern;
                    self.on_blocks(&f.name.name, kind, &params, &ret, f.span);
                    let names = param_names(&f.params);
                    env.funcs.insert(
                        f.name.name.clone(),
                        FuncSig {
                            kind,
                            params,
                            names,
                            ret,
                        },
                    );
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
        self.collect_shapes(module, &mut env);
        env
    }

    fn program_info(&mut self, env: &Env, parts: &[&ProgramItem]) -> ProgramInfo {
        let mut info = ProgramInfo::default();
        let mut names = Names::default();
        for part in parts {
            match part {
                ProgramItem::Message(m) => {
                    names.declare(self, &m.name);
                    self.bare_word(&m.name, Declared::Type);
                    let fields = self.fields(env, &m.fields, true);
                    let body = body_bytes(&m.attrs);
                    info.messages
                        .insert(m.name.name.clone(), MessageInfo { fields, body });
                }
                ProgramItem::Record(r) => {
                    names.declare(self, &r.name);
                    self.bare_word(&r.name, Declared::Type);
                    let fields = self.fields(env, &r.fields, false);
                    info.records.insert(r.name.name.clone(), fields);
                }
                ProgramItem::State(s) => {
                    for member in s.members.nodes() {
                        names.declare(self, &member.name);
                        self.bare_word(&member.name, Declared::Value);
                        // tiles or scalars, with a count or not (`bool` is `K109`'s)
                        let ty = match &member.ty.kind {
                            TypeKind::Tensor { .. } | TypeKind::View | TypeKind::Named(_) => {
                                self.error(
                                    codes::WRONG_TYPE,
                                    "a state member is tiles or scalars".into(),
                                    member.ty.span,
                                );
                                Ty::Unknown
                            }
                            _ => self.ty(env, &member.ty, Place::StateMember),
                        };
                        info.state
                            .insert(member.name.name.clone(), MemberInfo { ty });
                    }
                }
                ProgramItem::Handler(h) => {
                    names.declare(self, &h.name);
                    let params = self.params(env, &h.params);
                    let ret = h.ret.as_ref().map(|t| self.ty(env, t, Place::Result));
                    let entry = Invocable {
                        params,
                        names: param_names(&h.params),
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
                        names: param_names(&r.params),
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
                        names: param_names(&c.params),
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
            self.bare_word(&p.name, Declared::Value);
            let mut ty = self.ty(env, &p.ty, Place::Param);
            if let Ty::Tensor { replicated, .. } = &mut ty {
                *replicated = p.attrs.iter().any(|a| is_named(a, "replicated"));
            }
            out.push(ty);
        }
        out
    }

    /// The rule of a function declared on blocks (`tile<F>[r, c]`): a `func` only
    /// (`K010` for a `combine` or an `extern func`). Its body is typed on blocks, each
    /// operation's shapes checked there (R45).
    fn on_blocks(&mut self, name: &str, kind: FuncKind, params: &[Ty], ret: &Ty, span: Span) {
        let blocks = params.iter().chain(std::iter::once(ret));
        if !blocks.clone().any(|t| matches!(t, Ty::Block(..))) {
            return;
        }
        if kind != FuncKind::Func {
            self.error(
                codes::NOT_BUILT,
                format!(
                    "`{name}`: a `combine` or an `extern func` on a block of several tiles is \
                     named in the language and not built yet"
                ),
                span,
            );
        }
    }

    fn func_params(&mut self, env: &Env, params: &List<ParamDecl>, kind: FuncKind) -> Vec<Ty> {
        let mut names = Names::default();
        let mut out = Vec::new();
        for p in params.values() {
            names.declare(self, &p.name);
            self.bare_word(&p.name, Declared::Value);
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
            // a bare `body` after a message is the language's (R50)
            if message && !f.name.quoted && f.name.name == "body" {
                self.error(
                    codes::BARE_WORD,
                    "`body` is the language's after a message: a field of a message named \
                     `body` is written between backticks, at its declaration and at every use"
                        .into(),
                    f.name.span,
                );
            }
            let ty = self.ty(env, &f.ty, Place::Field);
            if f.ty.count.is_none()
                && !matches!(ty, Ty::Scalar(_) | Ty::Unknown)
                && !matches!(f.ty.kind, TypeKind::Bool)
            {
                let whose = if message { "a message's" } else { "a record's" };
                self.error(
                    codes::MESSAGE_FIELD,
                    format!(
                        "the field `{}` is {ty}; {whose} fields are scalars",
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
            TypeKind::Scalar(s) => match ScalarType::from_name(s.name()) {
                Some(s) => Ty::Scalar(s),
                None => Ty::Unknown,
            },
            TypeKind::Bool => {
                if place != Place::Local {
                    let what = match place {
                        Place::StateMember => "of state",
                        Place::Field => "of a field",
                        Place::Result | Place::FuncResult => "of a result",
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
            crate::syntax::ast::ExprKind::Name(name) => match env.ix.param_count(&name.name) {
                Some(n) => n,
                None => {
                    self.error(
                        codes::UNKNOWN_NAME,
                        format!("`{}` is not a declared parameter", name.name),
                        name.span,
                    );
                    return Ty::Unknown;
                }
            },
            // a constant expression
            _ => {
                let look = |n: &Ident| env.ix.known(&n.name, None);
                match self.constant_at(count, ScalarType::U32, "a count", &look) {
                    Some(Num::Int(n)) => Count::Known(n as u64),
                    _ => return Ty::Unknown,
                }
            }
        };
        // `tile<F>[r, c]` (R45): r rows of c tiles, r a number the text writes or a
        // module parameter bound at build (its value known at compile)
        let rows = match ty.rows.as_deref().map(|r| (&r.kind, r.span)) {
            None => Count::Known(1),
            Some((crate::syntax::ast::ExprKind::Int(text), span)) => match parse_int(text) {
                Some(r) if r > 0 => Count::Known(r),
                _ => {
                    self.error(
                        codes::WRONG_TYPE,
                        "a block has one row or more".into(),
                        span,
                    );
                    return Ty::Unknown;
                }
            },
            Some((crate::syntax::ast::ExprKind::Name(name), span)) => {
                match env.ix.param_count(&name.name) {
                    Some(Count::Known(r)) if r > 0 => Count::Known(r),
                    Some(Count::Known(_)) => {
                        self.error(
                            codes::WRONG_TYPE,
                            "a block has one row or more".into(),
                            span,
                        );
                        return Ty::Unknown;
                    }
                    Some(_) => {
                        self.error(codes::WRONG_TYPE, ROWS_BUILT.into(), span);
                        return Ty::Unknown;
                    }
                    None => {
                        self.error(
                            codes::UNKNOWN_NAME,
                            format!("`{}` is not a declared parameter", name.name),
                            name.span,
                        );
                        return Ty::Unknown;
                    }
                }
            }
            Some((_, span)) => {
                self.error(codes::WRONG_TYPE, ROWS_BUILT.into(), span);
                return Ty::Unknown;
            }
        };
        let shape = Shape { rows, cols: n };
        if rows != Count::Known(1) && !matches!(place, Place::FuncParam | Place::FuncResult) {
            self.error(
                codes::NOT_BUILT,
                "a block of several rows `[r, c]` other than a function's parameter or result \
                 is named in the language and not built yet"
                    .into(),
                ty.span,
            );
            return Ty::Unknown;
        }
        match place {
            Place::StateMember => match base {
                Ty::Tile(f) => Ty::Block(f, shape),
                Ty::Unknown => Ty::Unknown,
                other => Ty::Scalars(Box::new(other), n),
            },
            Place::FuncParam | Place::FuncResult | Place::VarOfZeros => match base {
                Ty::Tile(f) => Ty::Block(f, shape),
                Ty::Unknown => Ty::Unknown,
                _ if place == Place::VarOfZeros => {
                    self.error(
                        codes::WRONG_TYPE,
                        "a count `[n]` follows the type of a `var` of `zeros` only after a \
                         tile type"
                            .into(),
                        ty.span,
                    );
                    Ty::Unknown
                }
                _ => {
                    self.error(
                        codes::WRONG_TYPE,
                        "a count `[n]` follows a type of a function only after a tile type".into(),
                        ty.span,
                    );
                    Ty::Unknown
                }
            },
            _ => {
                self.error(
                    codes::WRONG_TYPE,
                    "a count `[n]` follows a type only in a state member, or a tile type of a function"
                        .into(),
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

/// The bytes of a message's body, from its `@body(n)`: n when it is one whole number
/// written in the text, above 0 and a multiple of 4 (otherwise `K004`, and none).
pub(crate) fn body_bytes(attrs: &[crate::syntax::ast::Attr]) -> Option<u64> {
    let attr = attrs.iter().find(|a| is_named(a, "body"))?;
    let args = attr.args.as_ref()?;
    if args.len() != 1 {
        return None;
    }
    let arg = args.values().next()?;
    let crate::syntax::ast::ExprKind::Int(text) = &arg.value.kind else {
        return None;
    };
    match parse_int(text) {
        Some(n) if arg.name.is_none() && n > 0 && n % 4 == 0 => Some(n),
        _ => None,
    }
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
