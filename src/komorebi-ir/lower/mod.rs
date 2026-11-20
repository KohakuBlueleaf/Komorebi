//! Lower: a module that checks clean, to the program structure (`crate::program`).
//!
//! Declarations are numbered in text order: the module's parameters, its record and
//! message types (those inside programs included), its functions, programs and control
//! programs. Each body becomes a flat list of instructions in the order the text
//! evaluates it. Hints, source positions and the names of values go to the notes.
//! Effects, chain relations and requirements beyond the stated ones are Derive's.
//!
//! Lower relies on the check: on a module that did not check clean it may refuse with
//! an `Unlowered`, and it never panics.

mod body;
mod forms;

use std::collections::HashMap;

use crate::check::Checked;
use crate::program::derive::{Cause, ChainDeps, Effects, Requirement};
use crate::program::notes::{Anchor, Hint, HintArg, Notes, Owner, SourceSpan};
use crate::program::{self as pg, BodyRef, CallArg, FuncKind, Lifetime, MemberKind, Scope};
use crate::syntax::ast::{self, Attr, ExprKind, Item, ProgramItem, Span};
use crate::types::{
    Ability, Binding, ChainId, ConstValue, ControlId, Count, Format, FuncId, HandlerId, MemberId,
    MessageId, On, ParamId, Peers, ProgramId, RecordId, RunId, ScalarType, TileShape, Type,
};

/// Why a construct was not lowered.
#[derive(Clone, Debug, PartialEq)]
pub struct Unlowered {
    /// Where it is written.
    pub span: Span,
    /// What it is.
    pub what: String,
}

type Res<T> = Result<T, Unlowered>;

fn refuse<T>(span: Span, what: impl Into<String>) -> Res<T> {
    Err(Unlowered {
        span,
        what: what.into(),
    })
}

/// The program structure of a module that checks clean, or what could not be lowered.
/// Runs on a deep stack, as every entry point of the crate does.
pub fn lower(checked: &Checked) -> Result<pg::Module, Vec<Unlowered>> {
    crate::deep::on_deep_stack(|| lower_here(checked))
}

fn lower_here(checked: &Checked) -> Result<pg::Module, Vec<Unlowered>> {
    let mut cx = Cx::new(checked);
    let mut errors = Vec::new();
    if let Err(e) = cx.declare() {
        return Err(vec![e]);
    }
    let module = cx.module(&mut errors);
    if errors.is_empty() {
        Ok(module)
    } else {
        Err(errors)
    }
}

/// A record or message type: the program it is declared in, its name, its fields.
type Declared = (Option<ProgramId>, String, Vec<(String, ScalarType)>);

/// A function's, handler's or chain's signature, for calls and invocations.
#[derive(Clone)]
struct Signature {
    params: Vec<Type>,
    ret: Option<Type>,
}

/// What a program declares, by name.
#[derive(Default)]
struct ProgramNames {
    handlers: HashMap<String, (HandlerId, Signature)>,
    runs: HashMap<String, (RunId, Signature)>,
    chains: HashMap<String, (ChainId, Signature)>,
    members: HashMap<String, (MemberId, MemberKind, Count)>,
}

/// The declarations of the module, by name, and the notes gathered.
struct Cx<'c> {
    checked: &'c Checked,
    params: HashMap<String, (ParamId, ScalarType, Option<ConstValue>)>,
    records: Vec<Declared>,
    messages: Vec<Declared>,
    funcs: HashMap<String, (FuncId, Signature)>,
    programs: HashMap<String, ProgramId>,
    program_names: Vec<ProgramNames>,
    spans: Vec<(Anchor, SourceSpan)>,
    hints: Vec<Hint>,
    param_names: Vec<(Anchor, String)>,
    names: Vec<crate::program::notes::Named>,
}

/// A span as the notes keep it.
fn source_span(span: Span) -> SourceSpan {
    let at = |x: usize| u32::try_from(x).unwrap_or(u32::MAX);
    SourceSpan {
        start: at(span.start),
        end: at(span.end),
    }
}

/// An integer literal's value: decimal, `0x`, `0o` or `0b`, with `_` between digits.
pub(crate) fn int_value(text: &str) -> Option<u128> {
    let clean: String = text.chars().filter(|c| *c != '_').collect();
    let (digits, radix) = if let Some(rest) = clean.strip_prefix("0x") {
        (rest, 16)
    } else if let Some(rest) = clean.strip_prefix("0o") {
        (rest, 8)
    } else if let Some(rest) = clean.strip_prefix("0b") {
        (rest, 2)
    } else {
        (clean.as_str(), 10)
    };
    u128::from_str_radix(digits, radix).ok()
}

/// An attribute's name, when it is a plain one (`@on`, not a hint).
fn attr_name(attr: &Attr) -> Option<&str> {
    match attr.path.as_slice() {
        [one] => Some(one.name.as_str()),
        _ => None,
    }
}

fn has_attr(attrs: &[Attr], name: &str) -> bool {
    attrs.iter().any(|a| attr_name(a) == Some(name))
}

fn attr_args<'a>(attrs: &'a [Attr], name: &str) -> Vec<&'a ast::Arg> {
    attrs
        .iter()
        .filter(|a| attr_name(a) == Some(name))
        .filter_map(|a| a.args.as_ref())
        .flat_map(|args| args.values())
        .collect()
}

/// A hint's argument as the notes keep it.
fn hint_arg(arg: &ast::Arg) -> HintArg {
    let value = hint_value(&arg.value);
    match &arg.name {
        Some(name) => HintArg::Named(name.name.clone(), Box::new(value)),
        None => value,
    }
}

fn hint_value(e: &ast::Expr) -> HintArg {
    match &e.kind {
        ExprKind::Int(t) | ExprKind::Float(t) => HintArg::Number(t.clone()),
        ExprKind::Str(t) => HintArg::Text(unquote(t)),
        ExprKind::Name(n) => HintArg::Name(n.name.clone()),
        ExprKind::Bool(b) => HintArg::Name(b.to_string()),
        ExprKind::List(xs) | ExprKind::Tuple(xs) => {
            HintArg::List(xs.values().map(hint_value).collect())
        }
        ExprKind::Neg(x) => match hint_value(x) {
            HintArg::Number(t) => HintArg::Number(format!("-{t}")),
            other => other,
        },
        _ => HintArg::Name(String::new()),
    }
}

/// A string literal's text without its quotes.
fn unquote(text: &str) -> String {
    text.trim_start_matches('"')
        .trim_end_matches('"')
        .to_string()
}

/// The constant a literal (or a negated one) writes, of type `ty`.
fn literal(e: &ast::Expr, ty: Option<ScalarType>) -> Option<ConstValue> {
    match &e.kind {
        ExprKind::Bool(b) => Some(ConstValue::Bool(*b)),
        _ => body::lit(e).map(|l| body::lit_const(l, ty)),
    }
}

impl<'c> Cx<'c> {
    fn new(checked: &'c Checked) -> Cx<'c> {
        Cx {
            checked,
            params: HashMap::new(),
            records: Vec::new(),
            messages: Vec::new(),
            funcs: HashMap::new(),
            programs: HashMap::new(),
            program_names: Vec::new(),
            spans: Vec::new(),
            hints: Vec::new(),
            param_names: Vec::new(),
            names: Vec::new(),
        }
    }

    fn items(&self) -> impl Iterator<Item = &'c Item> {
        self.checked.module.items.nodes()
    }

    /// A record or message type by name: the program's own first, then the module's.
    fn find(list: &[Declared], prog: Option<ProgramId>, name: &str) -> Option<u32> {
        let at = |owner: Option<ProgramId>| {
            list.iter()
                .position(|(p, n, _)| *p == owner && n == name)
                .map(|i| i as u32)
        };
        prog.and_then(|p| at(Some(p))).or_else(|| at(None))
    }

    fn record(&self, prog: Option<ProgramId>, name: &str) -> Option<RecordId> {
        Cx::find(&self.records, prog, name).map(RecordId)
    }

    fn message(&self, prog: Option<ProgramId>, name: &str) -> Option<MessageId> {
        Cx::find(&self.messages, prog, name).map(MessageId)
    }

    fn message_fields(&self, m: MessageId) -> &[(String, ScalarType)] {
        self.messages
            .get(m.0 as usize)
            .map_or(&[], |x| x.2.as_slice())
    }

    fn record_fields(&self, r: RecordId) -> &[(String, ScalarType)] {
        self.records
            .get(r.0 as usize)
            .map_or(&[], |x| x.2.as_slice())
    }

    fn scalar(s: ast::Scalar) -> Option<ScalarType> {
        ScalarType::from_name(s.name())
    }

    fn format(&self, id: &ast::Ident) -> Res<Format> {
        match Format::from_name(&id.name) {
            Some(f) => Ok(f),
            None => refuse(id.span, format!("no format named {}", id.name)),
        }
    }

    /// A count the text writes: a number, or a module parameter.
    fn count(&self, e: &ast::Expr) -> Res<Count> {
        match &e.kind {
            ExprKind::Int(t) => match int_value(t).and_then(|v| u64::try_from(v).ok()) {
                Some(n) => Ok(Count::Const(n)),
                None => refuse(e.span, "a count too large"),
            },
            ExprKind::Paren(x) => self.count(x),
            ExprKind::Name(n) => match self.params.get(&n.name) {
                Some((_, _, Some(ConstValue::Int(_, v)))) => Ok(Count::Const(*v)),
                Some((p, _, _)) => Ok(Count::Param(*p)),
                None => refuse(e.span, "a count that is neither a number nor a parameter"),
            },
            _ => refuse(e.span, "a count that is neither a number nor a parameter"),
        }
    }

    fn scalar_type(&self, ty: &ast::Type) -> Res<ScalarType> {
        match &ty.kind {
            ast::TypeKind::Scalar(s) if ty.count.is_none() => match Cx::scalar(*s) {
                Some(s) => Ok(s),
                None => refuse(ty.span, "no such scalar type"),
            },
            _ => refuse(ty.span, "not a scalar type"),
        }
    }

    /// The type a declaration writes.
    fn ty(&self, ty: &ast::Type, attrs: &[Attr], prog: Option<ProgramId>) -> Res<Type> {
        let count = match &ty.count {
            Some(c) => Some(self.count(c)?),
            None => None,
        };
        Ok(match (&ty.kind, count) {
            (ast::TypeKind::Scalar(_), None) => Type::Scalar(self.scalar_type(ty)?),
            (ast::TypeKind::Bool, None) => Type::Bool,
            (ast::TypeKind::Tile { format }, None) => Type::Tile(self.format(format)?),
            (ast::TypeKind::Tile { format }, Some(c)) => {
                let rows = match &ty.rows {
                    Some(r) => self.count(r)?,
                    None => Count::Const(1),
                };
                Type::Block(self.format(format)?, rows, c)
            }
            (ast::TypeKind::Tensor { format }, None) => {
                Type::Tensor(self.format(format)?, has_attr(attrs, "replicated"))
            }
            // `view` as written names no format
            (ast::TypeKind::View, None) => Type::View(None),
            (ast::TypeKind::Named(n), None) => {
                if let Some(r) = self.record(prog, &n.name) {
                    Type::RecordValue(r)
                } else if let Some(m) = self.message(prog, &n.name) {
                    Type::Message(m)
                } else {
                    return refuse(n.span, format!("no type named {}", n.name));
                }
            }
            _ => return refuse(ty.span, "a type with a count where none is allowed"),
        })
    }

    fn params(&self, list: &ast::List<ast::ParamDecl>, prog: Option<ProgramId>) -> Res<Vec<Type>> {
        list.values()
            .map(|p| self.ty(&p.ty, &p.attrs, prog))
            .collect()
    }

    fn ret(&self, ty: &Option<ast::Type>) -> Res<Option<ScalarType>> {
        match ty {
            Some(t) => Ok(Some(self.scalar_type(t)?)),
            None => Ok(None),
        }
    }

    fn fields(&self, list: &ast::List<ast::Field>) -> Res<Vec<(String, ScalarType)>> {
        list.values()
            .map(|f| Ok((f.name.name.clone(), self.scalar_type(&f.ty)?)))
            .collect()
    }

    /// The first pass: every declaration's place and signature, by name.
    fn declare(&mut self) -> Res<()> {
        // record and message types first: parameters and fields may name them
        let mut program_index = 0u32;
        for item in self.items() {
            match item {
                Item::Record(r) => {
                    let fields = self.fields(&r.fields)?;
                    self.records.push((None, r.name.name.clone(), fields));
                }
                Item::Message(m) => {
                    let fields = self.fields(&m.fields)?;
                    self.messages.push((None, m.name.name.clone(), fields));
                }
                Item::Program(p) => {
                    let pid = ProgramId(program_index);
                    program_index += 1;
                    self.programs.insert(p.name.name.clone(), pid);
                    for part in p.body.nodes() {
                        match part {
                            ProgramItem::Record(r) => {
                                let fields = self.fields(&r.fields)?;
                                self.records.push((Some(pid), r.name.name.clone(), fields));
                            }
                            ProgramItem::Message(m) => {
                                let fields = self.fields(&m.fields)?;
                                self.messages.push((Some(pid), m.name.name.clone(), fields));
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        let mut param_index = 0u32;
        let mut func_index = 0u32;
        for item in self.items() {
            match item {
                Item::Param(p) => {
                    let ty = self.scalar_type(&p.ty)?;
                    let build = attr_args(&p.attrs, "bind")
                        .iter()
                        .any(|a| matches!(&a.value.kind, ExprKind::Name(n) if n.name == "build"));
                    let value = match &p.value {
                        ast::ParamValue::Expr(e) if build => literal(e, Some(ty)),
                        _ => None,
                    };
                    self.params
                        .insert(p.name.name.clone(), (ParamId(param_index), ty, value));
                    param_index += 1;
                }
                Item::Func(f) | Item::Combine(f) => {
                    let sig = Signature {
                        params: self.params(&f.params, None)?,
                        ret: Some(self.ty(&f.ret, &[], None)?),
                    };
                    self.funcs
                        .insert(f.name.name.clone(), (FuncId(func_index), sig));
                    func_index += 1;
                }
                Item::Extern(f) => {
                    let sig = Signature {
                        params: self.params(&f.params, None)?,
                        ret: Some(self.ty(&f.ret, &[], None)?),
                    };
                    self.funcs
                        .insert(f.name.name.clone(), (FuncId(func_index), sig));
                    func_index += 1;
                }
                Item::Program(p) => {
                    let pid = self.programs.get(&p.name.name).copied();
                    let mut names = ProgramNames::default();
                    let (mut h, mut r, mut k, mut s) = (0u32, 0u32, 0u32, 0u32);
                    for part in p.body.nodes() {
                        match part {
                            ProgramItem::Handler(x) => {
                                let sig = Signature {
                                    params: self.params(&x.params, pid)?,
                                    ret: self.ret(&x.ret)?.map(Type::Scalar),
                                };
                                names
                                    .handlers
                                    .insert(x.name.name.clone(), (HandlerId(h), sig));
                                h += 1;
                            }
                            ProgramItem::Run(x) => {
                                let sig = Signature {
                                    params: self.params(&x.params, pid)?,
                                    ret: None,
                                };
                                names.runs.insert(x.name.name.clone(), (RunId(r), sig));
                                r += 1;
                            }
                            ProgramItem::Chain(x) => {
                                let sig = Signature {
                                    params: self.params(&x.params, pid)?,
                                    ret: None,
                                };
                                names.chains.insert(x.name.name.clone(), (ChainId(k), sig));
                                k += 1;
                            }
                            ProgramItem::State(state) => {
                                for m in state.members.nodes() {
                                    let (kind, count) = self.member_kind(&m.ty)?;
                                    names
                                        .members
                                        .insert(m.name.name.clone(), (MemberId(s), kind, count));
                                    s += 1;
                                }
                            }
                            _ => {}
                        }
                    }
                    self.program_names.push(names);
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn member_kind(&self, ty: &ast::Type) -> Res<(MemberKind, Count)> {
        let count = match &ty.count {
            Some(c) => self.count(c)?,
            None => Count::Const(1),
        };
        let kind = match &ty.kind {
            ast::TypeKind::Scalar(s) => match Cx::scalar(*s) {
                Some(s) => MemberKind::Scalars(s),
                None => return refuse(ty.span, "no such scalar type"),
            },
            ast::TypeKind::Tile { format } => MemberKind::Tiles(self.format(format)?),
            _ => return refuse(ty.span, "a state member that is neither tiles nor scalars"),
        };
        Ok((kind, count))
    }

    /// Notes a span, the hints of `attrs`, and the names of `params` of an owner.
    fn note_item(&mut self, anchor: Anchor, span: Span, attrs: &[Attr]) {
        self.spans.push((anchor, source_span(span)));
        for attr in attrs.iter().filter(|a| a.is_hint()) {
            self.hints.push(Hint {
                anchor,
                path: attr.path.iter().map(|p| p.name.clone()).collect(),
                args: attr
                    .args
                    .as_ref()
                    .map(|args| args.values().map(hint_arg).collect())
                    .unwrap_or_default(),
            });
        }
    }

    fn note_params(&mut self, owner: Owner, list: &ast::List<ast::ParamDecl>) {
        for (i, p) in list.values().enumerate() {
            let anchor = Anchor::ParamOf(owner, i as u32);
            self.note_item(anchor, p.span, &p.attrs);
            self.param_names.push((anchor, p.name.name.clone()));
        }
    }

    fn on(&self, attrs: &[Attr]) -> Res<On> {
        for arg in attr_args(attrs, "on") {
            if let ExprKind::Index { base, index } = &arg.value.kind {
                if matches!(&base.kind, ExprKind::Name(n) if n.name == "grid") {
                    let dims: Vec<Option<u64>> = index
                        .values()
                        .map(|i| match i {
                            ast::IndexItem::At(e) => match &e.kind {
                                ExprKind::Int(t) => {
                                    int_value(t).and_then(|v| u64::try_from(v).ok())
                                }
                                _ => None,
                            },
                            ast::IndexItem::Slice { .. } => None,
                        })
                        .collect();
                    return match dims.as_slice() {
                        [Some(r), Some(c)] => Ok(On::Grid(*r, *c)),
                        _ => refuse(arg.span, "a grid that is not two numbers"),
                    };
                }
            }
        }
        Ok(On::Group)
    }

    fn peers(&self, attrs: &[Attr]) -> Res<Peers> {
        let number = |e: &ast::Expr| match &e.kind {
            ExprKind::Int(t) => int_value(t).and_then(|v| u64::try_from(v).ok()),
            _ => None,
        };
        if let Some(arg) = attr_args(attrs, "peers").first() {
            let name = arg.name.as_ref().map(|n| n.name.as_str());
            let ExprKind::List(items) = &arg.value.kind else {
                return refuse(arg.span, "peers that are not a list");
            };
            match name {
                Some("offsets") => {
                    let offsets: Option<Vec<u64>> = items.values().map(number).collect();
                    return offsets
                        .map(Peers::Offsets)
                        .ok_or(())
                        .or_else(|_| refuse(arg.span, "an offset that is not a number"));
                }
                Some("pairs") => {
                    let pairs: Option<Vec<(u64, u64)>> = items
                        .values()
                        .map(|p| match &p.kind {
                            ExprKind::Tuple(xs) => {
                                let v: Vec<&ast::Expr> = xs.values().collect();
                                match v.as_slice() {
                                    [a, b] => Some((number(a)?, number(b)?)),
                                    _ => None,
                                }
                            }
                            _ => None,
                        })
                        .collect();
                    return pairs
                        .map(Peers::Pairs)
                        .ok_or(())
                        .or_else(|_| refuse(arg.span, "a pair that is not two numbers"));
                }
                _ => return refuse(arg.span, "peers neither offsets nor pairs"),
            }
        }
        Ok(Peers::None)
    }

    fn func_kind(
        &self,
        f_attrs: &[Attr],
        params: &ast::List<ast::ParamDecl>,
        ext: bool,
        combine: bool,
    ) -> FuncKind {
        if combine {
            return FuncKind::Combine {
                commutative: has_attr(f_attrs, "commutative"),
            };
        }
        if !ext {
            return FuncKind::Plain;
        }
        let names: Vec<&str> = params.values().map(|p| p.name.name.as_str()).collect();
        let positions = |attr: &str| -> Vec<u32> {
            attr_args(f_attrs, attr)
                .iter()
                .filter_map(|a| match &a.value.kind {
                    ExprKind::Name(n) => names.iter().position(|x| *x == n.name).map(|i| i as u32),
                    _ => None,
                })
                .collect()
        };
        let implementation = attr_args(f_attrs, "impl")
            .first()
            .and_then(|a| match &a.value.kind {
                ExprKind::Str(t) => Some(unquote(t)),
                _ => None,
            })
            .unwrap_or_default();
        FuncKind::Extern {
            implementation,
            reads: positions("reads"),
            writes: positions("writes"),
        }
    }

    /// The second pass: every declaration and body, in text order.
    fn module(&mut self, errors: &mut Vec<Unlowered>) -> pg::Module {
        let m = &self.checked.module;
        let mut out = pg::Module {
            name: m.name.name.clone(),
            tile: TileShape { rows: 0, cols: 0 },
            formats: Vec::new(),
            params: Vec::new(),
            records: Vec::new(),
            messages: Vec::new(),
            funcs: Vec::new(),
            programs: Vec::new(),
            controls: Vec::new(),
            requirements: Vec::new(),
            notes: Notes::default(),
        };
        self.note_item(Anchor::Module, m.span, &[]);
        out.records = self
            .records
            .iter()
            .map(|(p, n, f)| pg::Record {
                name: n.clone(),
                program: *p,
                fields: f.clone(),
            })
            .collect();
        out.messages = self
            .messages
            .iter()
            .map(|(p, n, f)| pg::Message {
                name: n.clone(),
                program: *p,
                fields: f.clone(),
            })
            .collect();
        let (mut record_at, mut message_at) = (0u32, 0u32);
        for item in self.items() {
            let result = match item {
                Item::Tile(t) => {
                    let n =
                        |l: &ast::IntLit| int_value(&l.text).and_then(|v| u64::try_from(v).ok());
                    match (n(&t.rows), n(&t.cols)) {
                        (Some(rows), Some(cols)) => {
                            out.tile = TileShape { rows, cols };
                            Ok(())
                        }
                        _ => refuse(t.span, "a tile shape too large"),
                    }
                }
                Item::Format(f) => f
                    .formats
                    .iter()
                    .try_for_each(|id| self.format(id).map(|x| out.formats.push(x))),
                Item::Requires(r) => {
                    for req in r.body.nodes() {
                        let ability = match req {
                            ast::Requirement::LocalState { unit, amount, .. } => {
                                int_value(&amount.text)
                                    .and_then(|v| u64::try_from(v).ok())
                                    .map(|amount| Ability::LocalState {
                                        amount,
                                        tiles: *unit == ast::StateUnit::Tiles,
                                    })
                            }
                            ast::Requirement::Group {
                                shape: ast::GroupShape::Grid,
                                ..
                            } => Some(Ability::Grid),
                            ast::Requirement::Group { .. } => None,
                            ast::Requirement::Dynamic { .. } => Some(Ability::Dynamic),
                            ast::Requirement::Islands { count, .. } => int_value(&count.text)
                                .and_then(|v| u64::try_from(v).ok())
                                .map(Ability::Islands),
                            ast::Requirement::Impl { name, .. } => {
                                Some(Ability::Impl(unquote(&name.text)))
                            }
                        };
                        if let Some(ability) = ability {
                            out.requirements.push(Requirement {
                                ability,
                                causes: vec![Cause::Stated],
                            });
                        }
                    }
                    Ok(())
                }
                Item::Param(p) => self.param(p).map(|x| {
                    let at = ParamId(out.params.len() as u32);
                    self.note_item(Anchor::Param(at), p.span, &p.attrs);
                    out.params.push(x);
                }),
                Item::Message(x) => {
                    self.note_item(Anchor::Message(MessageId(message_at)), x.span, &x.attrs);
                    message_at += 1;
                    Ok(())
                }
                Item::Record(x) => {
                    self.note_item(Anchor::Record(RecordId(record_at)), x.span, &[]);
                    record_at += 1;
                    Ok(())
                }
                Item::Func(f) | Item::Combine(f) => {
                    let combine = matches!(item, Item::Combine(_));
                    let id = FuncId(out.funcs.len() as u32);
                    self.note_item(Anchor::Func(id), f.span, &f.attrs);
                    self.note_params(Owner::Func(id), &f.params);
                    self.func(
                        id,
                        &f.name,
                        &f.params,
                        &f.ret,
                        Some(&f.body),
                        self.func_kind(&f.attrs, &f.params, false, combine),
                    )
                    .map(|x| out.funcs.push(x))
                }
                Item::Extern(f) => {
                    let id = FuncId(out.funcs.len() as u32);
                    self.note_item(Anchor::Func(id), f.span, &f.attrs);
                    self.note_params(Owner::Func(id), &f.params);
                    let kind = self.func_kind(&f.attrs, &f.params, true, false);
                    self.func(id, &f.name, &f.params, &f.ret, f.fallback.as_ref(), kind)
                        .map(|x| out.funcs.push(x))
                }
                Item::Program(p) => {
                    let pid = ProgramId(out.programs.len() as u32);
                    self.note_item(Anchor::Program(pid), p.span, &[]);
                    let program = self.program(pid, p, &mut record_at, &mut message_at, errors);
                    out.programs.push(program);
                    Ok(())
                }
                Item::Control(c) => {
                    let id = ControlId(out.controls.len() as u32);
                    self.note_item(Anchor::Control(id), c.span, &c.attrs);
                    self.note_params(Owner::Control(id), &c.params);
                    self.control(id, c).map(|x| out.controls.push(x))
                }
            };
            if let Err(e) = result {
                errors.push(e);
            }
        }
        let mut notes = Notes {
            hints: std::mem::take(&mut self.hints),
            spans: std::mem::take(&mut self.spans),
            names: std::mem::take(&mut self.names),
            param_names: std::mem::take(&mut self.param_names),
            origin: None,
        };
        notes.hints.sort_by_key(|h| h.anchor);
        notes.spans.sort_by_key(|(a, _)| *a);
        notes.param_names.sort_by_key(|(a, _)| *a);
        notes.names.sort_by_key(|n| match n {
            crate::program::notes::Named::Value(b, v, _) => (*b, 0, v.0),
            crate::program::notes::Named::Local(b, l, _) => (*b, 1, l.0),
        });
        out.notes = notes;
        out
    }

    fn param(&self, p: &ast::Param) -> Res<pg::Param> {
        let ty = self.scalar_type(&p.ty)?;
        let how = attr_args(&p.attrs, "bind")
            .first()
            .and_then(|a| match &a.value.kind {
                ExprKind::Name(n) => Some(n.name.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let value = match &p.value {
            ast::ParamValue::Expr(e) => match literal(e, Some(ty)) {
                Some(v) => Some(v),
                None => return refuse(e.span, "a parameter's value that is not a constant"),
            },
            ast::ParamValue::Fit(_) => None,
        };
        let binding = match (how.as_str(), value) {
            ("build", Some(v)) => Binding::Build(v),
            ("load", v) => Binding::Load(v),
            ("invoke", Some(v)) => Binding::Invoke(v),
            _ => return refuse(p.span, "a parameter bound in no way Lower knows"),
        };
        Ok(pg::Param {
            name: p.name.name.clone(),
            ty,
            binding,
        })
    }

    fn func(
        &mut self,
        id: FuncId,
        name: &ast::Ident,
        params: &ast::List<ast::ParamDecl>,
        ret: &ast::Type,
        body: Option<&ast::Block>,
        kind: FuncKind,
    ) -> Res<pg::Func> {
        let params_t = self.params(params, None)?;
        let ret_t = self.ty(ret, &[], None)?;
        // A function on blocks: its body is typed on blocks, each parameter its own (R45).
        let body = match body {
            Some(b) => Some(self.body(BodyRef::Func(id), None, params, &params_t, None, b)?),
            None => None,
        };
        Ok(pg::Func {
            name: name.name.clone(),
            kind,
            params: params_t,
            ret: ret_t,
            body,
            effects: Effects::default(),
        })
    }

    fn control(&mut self, id: ControlId, c: &ast::Control) -> Res<pg::Control> {
        let params = self.params(&c.params, None)?;
        let ret = self.ret(&c.ret)?;
        let body = self.body(
            BodyRef::Control(id),
            None,
            &c.params,
            &params,
            None,
            &c.body,
        )?;
        Ok(pg::Control {
            name: c.name.name.clone(),
            params,
            ret,
            body,
        })
    }

    fn program(
        &mut self,
        pid: ProgramId,
        p: &ast::Program,
        record_at: &mut u32,
        message_at: &mut u32,
        errors: &mut Vec<Unlowered>,
    ) -> pg::Program {
        let mut out = pg::Program {
            name: p.name.name.clone(),
            state: Vec::new(),
            handlers: Vec::new(),
            runs: Vec::new(),
            chains: Vec::new(),
        };
        let prog = Some(pid);
        for part in p.body.nodes() {
            let result: Res<()> = match part {
                ProgramItem::Message(x) => {
                    self.note_item(Anchor::Message(MessageId(*message_at)), x.span, &x.attrs);
                    *message_at += 1;
                    Ok(())
                }
                ProgramItem::Record(x) => {
                    self.note_item(Anchor::Record(RecordId(*record_at)), x.span, &[]);
                    *record_at += 1;
                    Ok(())
                }
                ProgramItem::State(state) => state.members.nodes().try_for_each(|m| {
                    let (kind, count) = self.member_kind(&m.ty)?;
                    let attrs: Vec<&Attr> = state.attrs.iter().chain(m.attrs.iter()).collect();
                    let named = |n: &str| attrs.iter().any(|a| attr_name(a) == Some(n));
                    let filled = attrs
                        .iter()
                        .filter(|a| attr_name(a) == Some("filled"))
                        .filter_map(|a| a.args.as_ref())
                        .flat_map(|x| x.values())
                        .any(|a| matches!(&a.value.kind, ExprKind::Name(n) if n.name == "element"));
                    let id = MemberId(out.state.len() as u32);
                    self.note_item(Anchor::Member(pid, id), m.span, &m.attrs);
                    out.state.push(pg::Member {
                        name: m.name.name.clone(),
                        kind,
                        count,
                        scope: Scope::Worker,
                        lifetime: Lifetime::Program,
                        export: named("export"),
                        by_element: filled,
                    });
                    Ok(())
                }),
                ProgramItem::Handler(h) => {
                    let id = HandlerId(out.handlers.len() as u32);
                    self.note_item(Anchor::Handler(pid, id), h.span, &h.attrs);
                    self.note_params(Owner::Handler(pid, id), &h.params);
                    (|| {
                        let params = self.params(&h.params, prog)?;
                        let ret = self.ret(&h.ret)?;
                        let body = self.body(
                            BodyRef::Handler(pid, id),
                            prog,
                            &h.params,
                            &params,
                            None,
                            &h.body,
                        )?;
                        out.handlers.push(pg::Handler {
                            name: h.name.name.clone(),
                            params,
                            ret,
                            on: self.on(&h.attrs)?,
                            exclusive: has_attr(&h.attrs, "exclusive"),
                            body,
                            effects: Effects::default(),
                        });
                        Ok(())
                    })()
                }
                ProgramItem::Run(r) => {
                    let id = RunId(out.runs.len() as u32);
                    self.note_item(Anchor::Run(pid, id), r.span, &r.attrs);
                    self.note_params(Owner::Run(pid, id), &r.params);
                    (|| {
                        let params = self.params(&r.params, prog)?;
                        let mut seed = None;
                        let mut on_message = Vec::new();
                        for part in r.body.nodes() {
                            match part {
                                ast::RunPart::Seed(b) => {
                                    seed = Some(self.body(
                                        BodyRef::Seed(pid, id),
                                        prog,
                                        &r.params,
                                        &params,
                                        None,
                                        b,
                                    )?);
                                }
                                ast::RunPart::On(on) => {
                                    let Some(m) = self.message(prog, &on.message.name) else {
                                        return refuse(on.message.span, "no message of that name");
                                    };
                                    let k = on_message.len() as u32;
                                    let arrived = Some((m, &on.binding));
                                    let body = self.body(
                                        BodyRef::On(pid, id, k),
                                        prog,
                                        &r.params,
                                        &params,
                                        arrived,
                                        &on.body,
                                    )?;
                                    on_message.push((m, body));
                                }
                            }
                        }
                        out.runs.push(pg::Run {
                            name: r.name.name.clone(),
                            params,
                            peers: self.peers(&r.attrs)?,
                            seed: seed.unwrap_or_else(|| pg::body::Body {
                                values: Vec::new(),
                                locals: Vec::new(),
                                insts: Vec::new(),
                            }),
                            on_message,
                            effects: Effects::default(),
                        });
                        Ok(())
                    })()
                }
                ProgramItem::Chain(c) => {
                    let id = ChainId(out.chains.len() as u32);
                    self.note_item(Anchor::Chain(pid, id), c.span, &c.attrs);
                    self.note_params(Owner::Chain(pid, id), &c.params);
                    for (k, call) in c.calls.nodes().enumerate() {
                        let anchor = Anchor::Call(pid, id, k as u32);
                        self.spans.push((anchor, source_span(call.span)));
                    }
                    self.chain(pid, c).map(|x| out.chains.push(x))
                }
            };
            if let Err(e) = result {
                errors.push(e);
            }
        }
        out
    }

    fn chain(&self, pid: ProgramId, c: &ast::Chain) -> Res<pg::Chain> {
        let prog = Some(pid);
        let params = self.params(&c.params, prog)?;
        let names: Vec<&str> = c.params.values().map(|p| p.name.name.as_str()).collect();
        let position = |n: &ast::Ident| names.iter().position(|x| *x == n.name).map(|i| i as u32);
        let handlers = self.program_names.get(pid.0 as usize).map(|p| &p.handlers);
        let mut calls = Vec::new();
        for call in c.calls.nodes() {
            let Some((handler, sig)) = handlers.and_then(|h| h.get(&call.handler.name)) else {
                return refuse(call.handler.span, "no handler of that name");
            };
            let mut args = Vec::new();
            for (k, arg) in call.args.values().enumerate() {
                let want = match sig.params.get(k) {
                    Some(Type::Scalar(s)) => Some(*s),
                    _ => None,
                };
                let a = match &arg.value.kind {
                    // a chain's parameter, else a module parameter bound at build: its
                    // value is a constant
                    ExprKind::Name(n) => {
                        position(n)
                            .map(CallArg::Param)
                            .or_else(|| match self.params.get(&n.name) {
                                Some((_, _, Some(c))) => Some(CallArg::Const(*c)),
                                _ => None,
                            })
                    }
                    ExprKind::Past(n) => position(n).map(CallArg::Past),
                    _ => literal(&arg.value, want).map(CallArg::Const),
                };
                match a {
                    Some(a) => args.push(a),
                    None => {
                        return refuse(
                            arg.span,
                            "a chain's argument that is no parameter and no constant",
                        )
                    }
                }
            }
            calls.push(pg::Call {
                handler: *handler,
                args,
            });
        }
        Ok(pg::Chain {
            name: c.name.name.clone(),
            params,
            on: self.on(&c.attrs)?,
            calls,
            deps: ChainDeps::default(),
        })
    }
}
