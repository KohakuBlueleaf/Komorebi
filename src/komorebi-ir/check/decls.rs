//! The declarations of a module that checks clean, each with its attributes read: what
//! Lower lays out, in text order, so that the places of `types::place` index these
//! lists.

use crate::syntax::ast::{
    self, Attr, ExprKind, IndexItem, Item, Module, ParamValue, ProgramItem, Requirement, Span,
    StateUnit, TypeKind,
};
use crate::types::{
    Ability, Binding, ConstValue, Count, Format, On, Peers, ProgramId, ScalarType, TileShape, Type,
};

use super::consts::{self, Num};
use super::env::{is_named, parse_int};
use super::index::Index;
use super::ty::Count as Known;

/// Every declaration of the module, in text order.
#[derive(Clone, Debug, PartialEq)]
pub struct Decls {
    /// The `tile` line; `None` for a module that declares nothing a tile is part of.
    pub tile: Option<TileShape>,
    /// The `format` line's formats, in the order written.
    pub formats: Vec<Format>,
    /// The lines of the `requires` block, as stated, in the order written; `group = line`
    /// is every target's and no ability.
    pub requires: Vec<Ability>,
    /// The module's parameters: `ParamId` is a place in this list.
    pub params: Vec<Param>,
    /// The record types, the programs' own included: `RecordId` is a place here.
    pub records: Vec<Fields>,
    /// The message types, the programs' own included: `MessageId` is a place here.
    pub messages: Vec<Fields>,
    /// The functions, `func`, `combine` and `extern func`: `FuncId` is a place here.
    pub funcs: Vec<Function>,
    /// The programs: `ProgramId` is a place here.
    pub programs: Vec<ProgramDecl>,
    /// The control programs: `ControlId` is a place here.
    pub controls: Vec<ControlDecl>,
}

/// A module parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// Its name.
    pub name: String,
    /// Its scalar type.
    pub ty: ScalarType,
    /// When it is fixed, with its value.
    pub binding: Binding,
}

/// A record or a message type.
#[derive(Clone, Debug, PartialEq)]
pub struct Fields {
    /// Its name.
    pub name: String,
    /// The program it is declared in, if any.
    pub program: Option<ProgramId>,
    /// Its fields in the order declared: a name and a scalar type.
    pub fields: Vec<(String, ScalarType)>,
}

/// What kind of function.
#[derive(Clone, Debug, PartialEq)]
pub enum FunctionKind {
    /// `func`.
    Func,
    /// `combine`, and whether `@commutative`.
    Combine {
        /// `@commutative`.
        commutative: bool,
    },
    /// `extern func`.
    Extern {
        /// `@impl("name")`.
        implementation: String,
        /// The parameters `@reads` names, by position.
        reads: Vec<u32>,
        /// The parameters `@writes` names, by position.
        writes: Vec<u32>,
        /// Whether it has a fallback body.
        fallback: bool,
    },
}

/// A function's declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    /// Its name.
    pub name: String,
    /// What kind.
    pub kind: FunctionKind,
    /// Its parameters' types, in order.
    pub params: Vec<Type>,
    /// Its result's type.
    pub ret: Type,
}

/// What a state member holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    /// Tiles of a format.
    Tiles(Format),
    /// Scalars of a type.
    Scalars(ScalarType),
}

/// A state member, with its attributes and its block's resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    /// Its name.
    pub name: String,
    /// What it holds.
    pub element: Element,
    /// How many: 1 for a member declared with no count.
    pub count: Count,
    /// `@export`.
    pub export: bool,
    /// `@filled(element)`.
    pub by_element: bool,
}

/// A handler's declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct HandlerDecl {
    /// Its name.
    pub name: String,
    /// Its parameters' types, in order.
    pub params: Vec<Type>,
    /// Its scalar result, if any.
    pub ret: Option<ScalarType>,
    /// `@on`.
    pub on: On,
    /// `@exclusive`.
    pub exclusive: bool,
}

/// A dynamic run's declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct RunDecl {
    /// Its name.
    pub name: String,
    /// Its parameters' types, in order.
    pub params: Vec<Type>,
    /// `@peers`.
    pub peers: Peers,
}

/// A chain's declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainDecl {
    /// Its name.
    pub name: String,
    /// Its parameters' types, in order.
    pub params: Vec<Type>,
    /// `@on`.
    pub on: On,
}

/// A program's declaration: its state, and what a worker can be asked to do.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramDecl {
    /// Its name.
    pub name: String,
    /// Its state members, every `state` block's in order: `MemberId` is a place here.
    pub members: Vec<Member>,
    /// Its handlers: `HandlerId` is a place here.
    pub handlers: Vec<HandlerDecl>,
    /// Its dynamic runs: `RunId` is a place here.
    pub runs: Vec<RunDecl>,
    /// Its chains: `ChainId` is a place here.
    pub chains: Vec<ChainDecl>,
}

/// A control program's declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlDecl {
    /// Its name.
    pub name: String,
    /// Its parameters' types, in order.
    pub params: Vec<Type>,
    /// Its scalar result, if any.
    pub ret: Option<ScalarType>,
}

/// The declarations of `module`, which checked clean; `Err` with the place of a
/// declaration they cannot describe.
pub(crate) fn of(module: &Module, ix: &Index) -> Result<Decls, Span> {
    let mut d = Decls {
        tile: None,
        formats: Vec::new(),
        requires: Vec::new(),
        params: Vec::new(),
        records: Vec::new(),
        messages: Vec::new(),
        funcs: Vec::new(),
        programs: Vec::new(),
        controls: Vec::new(),
    };
    let fields = |list: &ast::List<ast::Field>| -> Result<Vec<(String, ScalarType)>, Span> {
        list.values()
            .map(|f| Ok((f.name.name.clone(), scalar(&f.ty)?)))
            .collect()
    };
    for item in module.items.nodes() {
        let at = d.programs.len() as u32;
        match item {
            Item::Tile(t) => {
                let rows = parse_int(&t.rows.text).ok_or(t.span)?;
                let cols = parse_int(&t.cols.text).ok_or(t.span)?;
                d.tile = Some(TileShape { rows, cols });
            }
            Item::Format(f) => {
                for name in &f.formats {
                    d.formats
                        .push(Format::from_name(&name.name).ok_or(name.span)?);
                }
            }
            Item::Requires(r) => {
                for line in r.body.nodes() {
                    if let Some(ability) = ability(line)? {
                        d.requires.push(ability);
                    }
                }
            }
            Item::Param(p) => {
                let value = ix.values.get(d.params.len()).copied().flatten();
                d.params.push(param(p, value)?);
            }
            Item::Record(r) => d.records.push(Fields {
                name: r.name.name.clone(),
                program: None,
                fields: fields(&r.fields)?,
            }),
            Item::Message(m) => d.messages.push(Fields {
                name: m.name.name.clone(),
                program: None,
                fields: fields(&m.fields)?,
            }),
            Item::Func(f) | Item::Combine(f) => {
                let kind = if matches!(item, Item::Combine(_)) {
                    FunctionKind::Combine {
                        commutative: has(&f.attrs, "commutative"),
                    }
                } else {
                    FunctionKind::Func
                };
                d.funcs.push(Function {
                    name: f.name.name.clone(),
                    kind,
                    params: params(ix, &f.params, None)?,
                    ret: ty(ix, &f.ret, &[], None)?,
                });
            }
            Item::Extern(f) => {
                let names: Vec<&str> = f.params.values().map(|p| p.name.name.as_str()).collect();
                let positions = |attr: &str| -> Vec<u32> {
                    args(&f.attrs, attr)
                        .filter_map(|a| match &a.value.kind {
                            ExprKind::Name(n) => {
                                names.iter().position(|x| *x == n.name).map(|i| i as u32)
                            }
                            _ => None,
                        })
                        .collect()
                };
                let implementation = args(&f.attrs, "impl")
                    .find_map(|a| match &a.value.kind {
                        ExprKind::Str(t) => Some(t.trim_matches('"').to_string()),
                        _ => None,
                    })
                    .ok_or(f.span)?;
                d.funcs.push(Function {
                    name: f.name.name.clone(),
                    kind: FunctionKind::Extern {
                        implementation,
                        reads: positions("reads"),
                        writes: positions("writes"),
                        fallback: f.fallback.is_some(),
                    },
                    params: params(ix, &f.params, None)?,
                    ret: ty(ix, &f.ret, &[], None)?,
                });
            }
            Item::Program(p) => {
                let program = Some(p.name.name.as_str());
                let mut decl = ProgramDecl {
                    name: p.name.name.clone(),
                    members: Vec::new(),
                    handlers: Vec::new(),
                    runs: Vec::new(),
                    chains: Vec::new(),
                };
                for part in p.body.nodes() {
                    match part {
                        ProgramItem::Record(r) => d.records.push(Fields {
                            name: r.name.name.clone(),
                            program: Some(ProgramId(at)),
                            fields: fields(&r.fields)?,
                        }),
                        ProgramItem::Message(m) => d.messages.push(Fields {
                            name: m.name.name.clone(),
                            program: Some(ProgramId(at)),
                            fields: fields(&m.fields)?,
                        }),
                        ProgramItem::State(s) => {
                            for m in s.members.nodes() {
                                decl.members.push(member(ix, m, &s.attrs)?);
                            }
                        }
                        ProgramItem::Handler(h) => decl.handlers.push(HandlerDecl {
                            name: h.name.name.clone(),
                            params: params(ix, &h.params, program)?,
                            ret: h.ret.as_ref().map(scalar).transpose()?,
                            on: on(ix, &h.attrs)?,
                            exclusive: has(&h.attrs, "exclusive"),
                        }),
                        ProgramItem::Run(r) => decl.runs.push(RunDecl {
                            name: r.name.name.clone(),
                            params: params(ix, &r.params, program)?,
                            peers: peers(ix, &r.attrs)?,
                        }),
                        ProgramItem::Chain(c) => decl.chains.push(ChainDecl {
                            name: c.name.name.clone(),
                            params: params(ix, &c.params, program)?,
                            on: on(ix, &c.attrs)?,
                        }),
                    }
                }
                d.programs.push(decl);
            }
            Item::Control(c) => d.controls.push(ControlDecl {
                name: c.name.name.clone(),
                params: params(ix, &c.params, None)?,
                ret: c.ret.as_ref().map(scalar).transpose()?,
            }),
        }
    }
    Ok(d)
}

/// The arguments of the attributes named `name`.
fn args<'a>(attrs: &'a [Attr], name: &'a str) -> impl Iterator<Item = &'a ast::Arg> + 'a {
    attrs
        .iter()
        .filter(move |a| is_named(a, name))
        .filter_map(|a| a.args.as_ref())
        .flat_map(|list| list.values())
}

fn has(attrs: &[Attr], name: &str) -> bool {
    attrs.iter().any(|a| is_named(a, name))
}

/// True when an attribute `name` among `attrs` gives the word `word`, bare (R50).
fn gives(attrs: &[Attr], name: &str, word: &str) -> bool {
    args(attrs, name)
        .any(|a| matches!(&a.value.kind, ExprKind::Name(n) if n.name == word && !n.quoted))
}

/// The value of a constant expression at a `u32` place: a count, a rank, a grid side.
fn number(ix: &Index, e: &ast::Expr) -> Option<u64> {
    consts::count(e, &|n| ix.known(&n.name, None))
}

/// A requirement line as an ability; `None` for `group = line`.
fn ability(line: &Requirement) -> Result<Option<Ability>, Span> {
    Ok(Some(match line {
        Requirement::LocalState { unit, amount, span } => Ability::LocalState {
            amount: parse_int(&amount.text).ok_or(*span)?,
            tiles: *unit == StateUnit::Tiles,
        },
        Requirement::Group { shape, .. } => match shape {
            ast::GroupShape::Line => return Ok(None),
            ast::GroupShape::Grid => Ability::Grid,
        },
        Requirement::Dynamic { .. } => Ability::Dynamic,
        Requirement::Islands { count, span } => {
            Ability::Islands(parse_int(&count.text).ok_or(*span)?)
        }
        Requirement::Impl { name, .. } => Ability::Impl(name.text.clone()),
    }))
}

/// A scalar type written with no count.
fn scalar(t: &ast::Type) -> Result<ScalarType, Span> {
    match &t.kind {
        TypeKind::Scalar(s) if t.count.is_none() => ScalarType::from_name(s.name()).ok_or(t.span),
        _ => Err(t.span),
    }
}

fn format(id: &ast::Ident) -> Result<Format, Span> {
    Format::from_name(&id.name).ok_or(id.span)
}

/// A count a type writes: a number, a module parameter (its value when bound at
/// build), or a constant expression.
fn count(ix: &Index, e: &ast::Expr) -> Result<Count, Span> {
    let known = match &e.kind {
        ExprKind::Int(t) => parse_int(t).map(Known::Known),
        ExprKind::Name(n) => ix.param_count(&n.name),
        _ => number(ix, e).map(Known::Known),
    };
    match known {
        Some(Known::Known(n)) => Ok(Count::Const(n)),
        Some(Known::Param(p)) => Ok(Count::Param(p)),
        _ => Err(e.span),
    }
}

/// The type a declaration writes, as `program` sees the names of types.
fn ty(ix: &Index, t: &ast::Type, attrs: &[Attr], program: Option<&str>) -> Result<Type, Span> {
    let n = t.count.as_deref().map(|c| count(ix, c)).transpose()?;
    // the rows of `tile<F>[r, c]` (R45): a number the text writes or a module parameter
    // bound at build; `[n]` is one row
    let rows = match t.rows.as_deref() {
        Some(r) => match count(ix, r)? {
            Count::Const(r) if r > 0 => Count::Const(r),
            _ => return Err(r.span),
        },
        None => Count::Const(1),
    };
    Ok(match (&t.kind, n) {
        (TypeKind::Scalar(_), None) => Type::Scalar(scalar(t)?),
        (TypeKind::Bool, None) => Type::Bool,
        (TypeKind::Tile { format: f }, None) => Type::Tile(format(f)?),
        (TypeKind::Tile { format: f }, Some(n)) => Type::Block(format(f)?, rows, n),
        (TypeKind::Tensor { format: f }, None) => {
            Type::Tensor(format(f)?, has(attrs, "replicated"))
        }
        (TypeKind::View, None) => Type::View(None),
        (TypeKind::Named(name), None) => {
            if let Some(r) = ix.record_ref(program, &name.name) {
                Type::RecordValue(r)
            } else if let Some(m) = ix.message_ref(program, &name.name) {
                Type::Message(m)
            } else {
                return Err(name.span);
            }
        }
        _ => return Err(t.span),
    })
}

fn params(
    ix: &Index,
    list: &ast::List<ast::ParamDecl>,
    program: Option<&str>,
) -> Result<Vec<Type>, Span> {
    list.values()
        .map(|p| ty(ix, &p.ty, &p.attrs, program))
        .collect()
}

/// A parameter: its type, and when it is fixed with its value, the value its
/// constant expression gives (`value`).
fn param(p: &ast::Param, value: Option<Num>) -> Result<Param, Span> {
    let s = scalar(&p.ty)?;
    let value = match &p.value {
        ParamValue::Expr(e) => Some(match &e.kind {
            ExprKind::Bool(b) => ConstValue::Bool(*b),
            _ => value.and_then(|v| at_type(v, s)).ok_or(e.span)?,
        }),
        ParamValue::Fit(_) => None,
    };
    let binding = match value {
        Some(v) if gives(&p.attrs, "bind", "build") => Binding::Build(v),
        v if gives(&p.attrs, "bind", "load") => Binding::Load(v),
        Some(v) if gives(&p.attrs, "bind", "invoke") => Binding::Invoke(v),
        _ => return Err(p.span),
    };
    Ok(Param {
        name: p.name.name.clone(),
        ty: s,
        binding,
    })
}

/// The value of a constant expression at the type `s`; an integer is held as its bits
/// masked to the type's width.
fn at_type(v: Num, s: ScalarType) -> Option<ConstValue> {
    match v {
        Num::Float(v) if s.is_float() => Some(ConstValue::Float(s, v)),
        Num::Int(v) if !s.is_float() => {
            let bits = s.bits();
            let mask = if bits >= 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            Some(ConstValue::Int(s, (v as u64) & mask))
        }
        _ => None,
    }
}

/// A state member, with its attributes and its block's.
fn member(ix: &Index, m: &ast::StateMember, block: &[Attr]) -> Result<Member, Span> {
    let attrs: Vec<Attr> = block.iter().chain(m.attrs.iter()).cloned().collect();
    let element = match &m.ty.kind {
        TypeKind::Tile { format: f } => Element::Tiles(format(f)?),
        TypeKind::Scalar(s) => Element::Scalars(ScalarType::from_name(s.name()).ok_or(m.ty.span)?),
        _ => return Err(m.ty.span),
    };
    let count = match m.ty.count.as_deref() {
        Some(c) => count(ix, c)?,
        None => Count::Const(1),
    };
    Ok(Member {
        name: m.name.name.clone(),
        element,
        count,
        export: has(&attrs, "export"),
        by_element: gives(&attrs, "filled", "element"),
    })
}

/// `@on(group)` or `@on(grid[R, C])`, each side a constant expression.
fn on(ix: &Index, attrs: &[Attr]) -> Result<On, Span> {
    for arg in args(attrs, "on") {
        if let ExprKind::Index { base, index } = &arg.value.kind {
            if matches!(&base.kind, ExprKind::Name(n) if n.name == "grid" && !n.quoted) {
                let dims: Vec<Option<u64>> = index
                    .values()
                    .map(|i| match i {
                        IndexItem::At(e) => number(ix, e).filter(|n| *n > 0),
                        IndexItem::Slice { .. } => None,
                    })
                    .collect();
                return match dims.as_slice() {
                    [Some(r), Some(c)] => Ok(On::Grid(*r, *c)),
                    _ => Err(arg.span),
                };
            }
        }
    }
    Ok(On::Group)
}

/// `@peers(offsets = [...])` or `@peers(pairs = [(s, r), ...])`, each number a constant
/// expression.
fn peers(ix: &Index, attrs: &[Attr]) -> Result<Peers, Span> {
    let Some(arg) = args(attrs, "peers").next() else {
        return Ok(Peers::None);
    };
    let ExprKind::List(items) = &arg.value.kind else {
        return Err(arg.span);
    };
    match arg
        .name
        .as_ref()
        .filter(|n| !n.quoted)
        .map(|n| n.name.as_str())
    {
        Some("offsets") => items
            .values()
            .map(|e| number(ix, e))
            .collect::<Option<Vec<u64>>>()
            .map(Peers::Offsets)
            .ok_or(arg.span),
        Some("pairs") => items
            .values()
            .map(|p| match &p.kind {
                ExprKind::Tuple(xs) => match xs.values().collect::<Vec<_>>().as_slice() {
                    [a, b] => Some((number(ix, a)?, number(ix, b)?)),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Option<Vec<(u64, u64)>>>()
            .map(Peers::Pairs)
            .ok_or(arg.span),
        _ => Err(arg.span),
    }
}
