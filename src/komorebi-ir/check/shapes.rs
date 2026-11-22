//! Several tile shapes in one module: the `tile` lines, `@tile(R, C)` on a declaration,
//! and what must agree between the bodies of two shapes (a call of a function, a
//! `combine` a reduce names, a state member of tiles).

use std::collections::HashMap;

use crate::syntax::ast::{Attr, ExprKind, Item, Module, ProgramItem, RunPart, Span, TypeKind};

use super::codes;
use super::env::{is_named, parse_int, Env, FuncKind};
use super::{Checker, Name};

/// A tile shape: rows and columns.
pub(crate) type Shape = (u64, u64);

/// The shapes of the module's `tile` lines, in order, each once: the first is the
/// default shape.
pub(crate) fn tile_lines(module: &Module) -> Vec<Shape> {
    let mut out = Vec::new();
    for item in module.items.nodes() {
        if let Item::Tile(t) = item {
            if let (Some(rows), Some(cols)) = (parse_int(&t.rows.text), parse_int(&t.cols.text)) {
                if !out.contains(&(rows, cols)) {
                    out.push((rows, cols));
                }
            }
        }
    }
    out
}

/// What `@tile` among `attrs` gives: `None` when there is none, `Some(None)` when its
/// arguments are not two whole numbers written in the text, else the shape.
pub(crate) fn tile_attr(attrs: &[Attr]) -> Option<Option<Shape>> {
    let attr = attrs.iter().find(|a| is_named(a, "tile"))?;
    let Some(args) = attr.args.as_ref() else {
        return Some(None);
    };
    let numbers: Vec<Option<u64>> = args
        .values()
        .map(|arg| match (&arg.name, &arg.value.kind) {
            (None, ExprKind::Int(text)) => parse_int(text),
            _ => None,
        })
        .collect();
    Some(match numbers.as_slice() {
        [Some(rows), Some(cols)] => Some((*rows, *cols)),
        _ => None,
    })
}

/// The place among `lines` of the shape of a declaration with `attrs`: the one its
/// `@tile` names, else (none, or one the checker refuses) the default, 0.
pub(crate) fn tile_index(lines: &[Shape], attrs: &[Attr]) -> u32 {
    tile_attr(attrs)
        .flatten()
        .and_then(|s| lines.iter().position(|l| *l == s))
        .map_or(0, |i| i as u32)
}

/// The shape of a declaration with `attrs` in a module of the shapes `lines`; none when
/// the module has no `tile` line.
pub(crate) fn shape_of(lines: &[Shape], attrs: &[Attr]) -> Option<Shape> {
    lines.get(tile_index(lines, attrs) as usize).copied()
}

fn text(s: Shape) -> String {
    format!("[{}, {}]", s.0, s.1)
}

impl Checker<'_> {
    /// The shapes of the declarations, kept in `env`: each function's by its place, and
    /// each state member's by its program and its place.
    pub(crate) fn collect_shapes(&mut self, module: &Module, env: &mut Env) {
        env.tiles = tile_lines(module);
        let lines = env.tiles.clone();
        for item in module.items.nodes() {
            match item {
                Item::Func(f) | Item::Combine(f) => {
                    if let (Some(&place), Some(s)) =
                        (env.ix.funcs.get(&f.name.name), shape_of(&lines, &f.attrs))
                    {
                        env.func_tiles.insert(place, s);
                    }
                }
                Item::Extern(f) => {
                    if let (Some(&place), Some(s)) =
                        (env.ix.funcs.get(&f.name.name), shape_of(&lines, &f.attrs))
                    {
                        env.func_tiles.insert(place, s);
                    }
                }
                Item::Program(p) => {
                    for part in p.body.nodes() {
                        let ProgramItem::State(state) = part else {
                            continue;
                        };
                        // a member of scalars has no shape
                        let tiles = state
                            .members
                            .nodes()
                            .filter(|m| matches!(m.ty.kind, TypeKind::Tile { .. }));
                        for m in tiles {
                            // the member's own `@tile`, else its block's
                            let attrs: Vec<Attr> =
                                m.attrs.iter().chain(&state.attrs).cloned().collect();
                            let Some(s) = shape_of(&lines, &attrs) else {
                                continue;
                            };
                            if let Some(Name::State { program, member }) =
                                env.ix.member(&p.name.name, &m.name.name)
                            {
                                env.member_tiles.insert((program, member), s);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// `K112`: a `tile` line whose shape an earlier one declares.
    pub(crate) fn tile_twice(&mut self, module: &Module) {
        let mut seen: Vec<Shape> = Vec::new();
        for item in module.items.nodes() {
            let Item::Tile(t) = item else {
                continue;
            };
            let (Some(rows), Some(cols)) = (parse_int(&t.rows.text), parse_int(&t.cols.text))
            else {
                continue;
            };
            if seen.contains(&(rows, cols)) {
                self.error(
                    codes::NAME_TWICE,
                    format!("the tile shape {} is declared twice", text((rows, cols))),
                    t.span,
                );
            } else {
                seen.push((rows, cols));
            }
        }
    }

    /// `@tile(R, C)` on a declaration that may carry it: two whole numbers written in
    /// the text (`K004`), a shape among the module's `tile` lines (`K101`).
    pub(crate) fn tile_args(&mut self, attrs: &[Attr], lines: &[Shape]) {
        let Some(attr) = attrs.iter().find(|a| is_named(a, "tile")) else {
            return;
        };
        match tile_attr(attrs).flatten() {
            None => self.error(
                codes::ATTRIBUTE_ARGUMENT,
                "`@tile` takes two arguments, the rows and the columns of a tile, whole numbers \
                 written in the text"
                    .into(),
                attr.span,
            ),
            Some(s) if !lines.contains(&s) => self.error(
                codes::UNKNOWN_NAME,
                format!("the module declares no tile shape {}", text(s)),
                attr.span,
            ),
            Some(_) => {}
        }
    }

    /// `K102`, where two shapes must agree: a body that calls a function, names a
    /// `combine` for a reduce, or reads or writes a state member of tiles of another
    /// shape than its own. A control program is free (it applies a `combine` in the
    /// function's own shape), as is a chain.
    pub(crate) fn shape_uses(&mut self, module: &Module, env: &Env) {
        let lines = &env.tiles;
        let mut bodies: Vec<(Shape, Span)> = Vec::new();
        for item in module.items.nodes() {
            match item {
                Item::Func(f) | Item::Combine(f) => {
                    bodies.extend(shape_of(lines, &f.attrs).map(|s| (s, f.body.span)));
                }
                Item::Extern(f) => {
                    if let (Some(s), Some(body)) = (shape_of(lines, &f.attrs), &f.fallback) {
                        bodies.push((s, body.span));
                    }
                }
                Item::Program(p) => {
                    for part in p.body.nodes() {
                        match part {
                            ProgramItem::Handler(h) => {
                                bodies.extend(shape_of(lines, &h.attrs).map(|s| (s, h.body.span)));
                            }
                            ProgramItem::Run(r) => {
                                let Some(s) = shape_of(lines, &r.attrs) else {
                                    continue;
                                };
                                for part in r.body.nodes() {
                                    let block = match part {
                                        RunPart::Seed(b) => b,
                                        RunPart::On(on) => &on.body,
                                    };
                                    bodies.push((s, block.span));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        let names: HashMap<u32, &str> =
            env.ix.funcs.iter().map(|(n, &p)| (p, n.as_str())).collect();
        let mut found: Vec<(Span, String)> = Vec::new();
        for (own, body) in bodies {
            for (&(start, end), name) in &self.names {
                if start < body.start || end > body.end {
                    continue;
                }
                let at = Span { start, end };
                match *name {
                    Name::Func(place) => {
                        let Some(&theirs) = env.func_tiles.get(&place).filter(|t| **t != own)
                        else {
                            continue;
                        };
                        let fname = names.get(&place).copied().unwrap_or("");
                        let combine = env
                            .funcs
                            .get(fname)
                            .is_some_and(|f| f.kind == FuncKind::Combine);
                        let what = if combine {
                            "the `combine`"
                        } else {
                            "the function"
                        };
                        found.push((
                            at,
                            format!(
                                "{what} `{fname}` computes in tiles of {}, and this body in tiles \
                                 of {}",
                                text(theirs),
                                text(own)
                            ),
                        ));
                    }
                    Name::State { program, member } => {
                        let Some(&theirs) = env
                            .member_tiles
                            .get(&(program, member))
                            .filter(|t| **t != own)
                        else {
                            continue;
                        };
                        found.push((
                            at,
                            format!(
                                "this state member holds tiles of {} (its `state` block's \
                                 shape), and this body computes in tiles of {}",
                                text(theirs),
                                text(own)
                            ),
                        ));
                    }
                    _ => {}
                }
            }
        }
        found.sort_by_key(|(at, _)| (at.start, at.end));
        for (at, message) in found {
            self.error(codes::WRONG_TYPE, message, at);
        }
    }
}
