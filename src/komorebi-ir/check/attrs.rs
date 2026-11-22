//! The rules on the module's declarations: attributes (known, in place, present),
//! `= fit`, what `requires` must state, and what is named and not built yet there.

use std::collections::HashSet;

use crate::syntax::ast::{
    Attr, Block, Expr, ExprKind, GroupShape, IndexItem, Item, List, Module, ParamDecl, ParamValue,
    ProgramItem, Requirement, Span, Stmt, TypeKind,
};

use super::codes;
use super::consts::Num;
use super::env::{body_bytes, is_named};
use super::index::Index;
use super::Checker;
use crate::types::ScalarType;

/// The language's own undotted attributes. A dotted name is a hint and never an error.
const KNOWN: [&str; 16] = [
    "on",
    "exclusive",
    "scope",
    "lifetime",
    "export",
    "filled",
    "bind",
    "commutative",
    "reads",
    "writes",
    "impl",
    "replicated",
    "dynamic",
    "peers",
    "body",
    "tile",
];

/// The attributes whose operands are words of the language (`@on(group)`,
/// `@scope(worker)`, `@bind(load)`): a name between backticks is the module's only and
/// means nothing there (R50).
const TAKE_WORDS: [&str; 5] = ["on", "scope", "lifetime", "filled", "bind"];

/// True when `@on(...)` among `attrs` names a grid: the bare word (R50).
pub(crate) fn on_grid(attrs: &[Attr]) -> bool {
    attrs.iter().filter(|a| is_named(a, "on")).any(|a| {
        a.args.as_ref().is_some_and(|args| {
            args.values().any(|arg| match &arg.value.kind {
                ExprKind::Index { base, .. } => {
                    matches!(&base.kind, ExprKind::Name(n) if n.name == "grid" && !n.quoted)
                }
                ExprKind::Name(n) => n.name == "grid" && !n.quoted,
                _ => false,
            })
        })
    })
}

/// The bare words an attribute such as `@bind(load)` gives: a name between backticks
/// is the module's, never the word (R50).
pub(crate) fn word_args(attrs: &[Attr], name: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for attr in attrs.iter().filter(|a| is_named(a, name)) {
        if let Some(args) = &attr.args {
            for arg in args.values() {
                if let ExprKind::Name(n) = &arg.value.kind {
                    if !n.quoted {
                        out.insert(n.name.clone());
                    }
                }
            }
        }
    }
    out
}

/// The names an attribute such as `@reads(a, b)` gives.
pub(crate) fn named_args(attrs: &[Attr], name: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for attr in attrs.iter().filter(|a| is_named(a, name)) {
        if let Some(args) = &attr.args {
            for arg in args.values() {
                if let ExprKind::Name(n) = &arg.value.kind {
                    out.insert(n.name.clone());
                }
            }
        }
    }
    out
}

/// The string of `@impl("...")`, when given.
fn impl_name(attrs: &[Attr]) -> Option<String> {
    let attr = attrs.iter().find(|a| is_named(a, "impl"))?;
    let arg = attr.args.as_ref()?.values().next()?;
    match &arg.value.kind {
        ExprKind::Str(text) => Some(unquote(text)),
        _ => None,
    }
}

/// A string literal's text without its quotes.
fn unquote(text: &str) -> String {
    text.trim_start_matches('"')
        .trim_end_matches('"')
        .to_string()
}

/// The one word each of `@scope`, `@lifetime` and `@filled` takes, and the words
/// `@bind` takes.
fn words_taken(name: &str) -> &'static [&'static str] {
    match name {
        "scope" => &["worker"],
        "lifetime" => &["program"],
        "filled" => &["element"],
        "bind" => &["build", "load", "invoke"],
        _ => &[],
    }
}

/// True when `value` is the bare word `word`.
fn is_word(value: &Expr, word: &str) -> bool {
    matches!(&value.kind, ExprKind::Name(n) if n.name == word && !n.quoted)
}

/// The sides R and C when `value` is `grid[R, C]`, each an expression (the rule on
/// constant expressions gives each its code).
fn grid_sides(value: &Expr) -> Option<[&Expr; 2]> {
    let ExprKind::Index { base, index } = &value.kind else {
        return None;
    };
    let sides: Vec<&Expr> = index
        .values()
        .filter_map(|item| match item {
            IndexItem::At(e) => Some(e),
            IndexItem::Slice { .. } => None,
        })
        .collect();
    match sides.as_slice() {
        [r, c] if is_word(base, "grid") && index.len() == 2 => Some([r, c]),
        _ => None,
    }
}

/// True when an operand of `attr` is a name between backticks: `K101` reports it
/// already, by its own name.
fn has_quoted(attr: &Attr) -> bool {
    attr.args.iter().flat_map(|args| args.values()).any(|arg| {
        let label = arg.name.as_ref().is_some_and(|n| n.quoted);
        let word = match &arg.value.kind {
            ExprKind::Name(n) => n.quoted,
            ExprKind::Index { base, .. } => {
                matches!(&base.kind, ExprKind::Name(n) if n.quoted)
            }
            _ => false,
        };
        label || word
    })
}

impl Checker<'_> {
    /// Checks the arguments of the attributes in `checked` among `attrs` (`K004`):
    /// `@scope(worker)`, `@lifetime(program)`, `@filled(element)`, `@bind` with one of
    /// `build`, `load`, `invoke`, `@on(group)` or `@on(grid[R, C])` (R and C constant
    /// expressions, each above 0, given their codes by `ix`), and no list at all after
    /// `@exclusive` and `@export`.
    fn attr_args(&mut self, attrs: &[Attr], checked: &[&str], ix: &Index) {
        for attr in attrs.iter().filter(|a| !a.is_hint()) {
            let name = attr.path[0].name.as_str();
            if !checked.contains(&name) || has_quoted(attr) {
                continue;
            }
            let fits = match name {
                "exclusive" | "export" => attr.args.is_none(),
                _ => match attr.args.as_ref() {
                    Some(args) if args.len() == 1 => {
                        let arg = args.values().next().expect("one operand");
                        arg.name.is_none()
                            && match name {
                                "on" => is_word(&arg.value, "group") || self.grid(&arg.value, ix),
                                _ => words_taken(name).iter().any(|w| is_word(&arg.value, w)),
                            }
                    }
                    _ => false,
                },
            };
            if !fits {
                let takes = match name {
                    "exclusive" | "export" => "no argument list".to_string(),
                    "on" => "one argument, `group` or `grid[R, C]`".to_string(),
                    "bind" => "one argument, one of `build`, `load`, `invoke`".to_string(),
                    _ => format!("one argument, `{}`", words_taken(name).join("`, `")),
                };
                self.error(
                    codes::ATTRIBUTE_ARGUMENT,
                    format!("`@{name}` takes {takes}"),
                    attr.span,
                );
            }
        }
    }

    /// Checks the attributes in one place: each undotted one known (`K003`) and allowed
    /// there (`K006`).
    fn attrs_in(&mut self, attrs: &[Attr], allowed: &[&str], place: &str) {
        for attr in attrs.iter().filter(|a| !a.is_hint()) {
            let name = attr.path[0].name.as_str();
            if !KNOWN.contains(&name) {
                self.error(
                    codes::UNKNOWN_ATTRIBUTE,
                    format!("`@{name}` is not an attribute of the language"),
                    attr.span,
                );
            } else if !allowed.contains(&name) {
                self.error(
                    codes::ATTRIBUTE_MISPLACED,
                    format!("`@{name}` has no meaning on {place}"),
                    attr.span,
                );
            }
            // an operand's name is the language's, as is a word where the attribute
            // takes words: one between backticks is the module's only, and no item of
            // the module means anything there (R50)
            let words = TAKE_WORDS.contains(&name);
            for arg in attr.args.iter().flat_map(|args| args.values()) {
                if let Some(label) = arg.name.as_ref().filter(|n| n.quoted) {
                    self.unknown(label);
                }
                let word = match &arg.value.kind {
                    ExprKind::Name(n) => Some(n),
                    ExprKind::Index { base, .. } => match &base.kind {
                        ExprKind::Name(n) => Some(n),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(n) = word.filter(|n| words && n.quoted) {
                    self.unknown(n);
                }
            }
        }
    }

    /// True when `value` is `grid[R, C]`; each side a constant expression above 0 that
    /// fits a `u32`, reported where it is not.
    fn grid(&mut self, value: &Expr, ix: &Index) -> bool {
        let Some(sides) = grid_sides(value) else {
            return false;
        };
        let look = |n: &crate::syntax::ast::Ident| ix.known(&n.name, None);
        for side in sides {
            let what = "a side of a grid";
            if let Some(Num::Int(0)) = self.constant_at(side, ScalarType::U32, what, &look) {
                self.error(
                    codes::OUT_OF_RANGE,
                    "a side of a grid is above 0".into(),
                    side.span,
                );
            }
        }
        true
    }

    /// `@peers(offsets = [d, ...])` or `@peers(pairs = [(s, r), ...])`: each number a
    /// constant expression that fits a `u32`, reported where it is not.
    fn peer_numbers(&mut self, attrs: &[Attr], ix: &Index) {
        let look = |n: &crate::syntax::ast::Ident| ix.known(&n.name, None);
        for arg in attrs
            .iter()
            .filter(|a| is_named(a, "peers"))
            .filter_map(|a| a.args.as_ref())
            .flat_map(|args| args.values())
        {
            let ExprKind::List(items) = &arg.value.kind else {
                continue;
            };
            for item in items.values() {
                let numbers: Vec<&Expr> = match &item.kind {
                    ExprKind::Tuple(pair) => pair.values().collect(),
                    _ => vec![item],
                };
                for e in numbers {
                    self.constant_at(e, ScalarType::U32, "a peer", &look);
                }
            }
        }
    }

    /// Reports a declaration without an attribute it must carry (`K007`).
    fn require_attr(&mut self, attrs: &[Attr], name: &str, what: &str, span: Span) {
        if !attrs.iter().any(|a| is_named(a, name)) {
            self.error(
                codes::ATTRIBUTE_MISSING,
                format!("{what} has no `@{name}`; nothing is supplied silently"),
                span,
            );
        }
    }

    fn param_attrs(&mut self, params: &List<ParamDecl>) {
        for p in params.values() {
            self.attrs_in(&p.attrs, &["replicated"], "a parameter");
            let tensor = matches!(p.ty.kind, TypeKind::Tensor { .. });
            if !tensor {
                if let Some(attr) = p.attrs.iter().find(|a| is_named(a, "replicated")) {
                    self.error(
                        codes::ATTRIBUTE_MISPLACED,
                        format!(
                            "`@replicated` marks a tensor, and `{}` is not one",
                            p.name.name
                        ),
                        attr.span,
                    );
                }
            }
        }
    }

    fn block_attrs(&mut self, block: &Block) {
        for stmt in block.stmts.nodes() {
            match stmt {
                Stmt::Let { attrs, .. } => self.attrs_in(attrs, &[], "a `let`"),
                Stmt::For { attrs, body, .. } => {
                    self.attrs_in(attrs, &[], "a `for`");
                    self.block_attrs(body);
                }
                Stmt::While { body, .. } => self.block_attrs(body),
                Stmt::If {
                    then, otherwise, ..
                } => {
                    self.block_attrs(then);
                    if let Some(otherwise) = otherwise {
                        self.block_attrs(otherwise);
                    }
                }
                _ => {}
            }
        }
    }

    /// `@body(n)` on a message: one whole number written in the text, above 0 and a
    /// multiple of 4 (`K004`).
    fn message_attrs(&mut self, attrs: &[Attr]) {
        self.attrs_in(attrs, &["body"], "a message");
        for attr in attrs.iter().filter(|a| is_named(a, "body")) {
            if has_quoted(attr) {
                continue;
            }
            if body_bytes(std::slice::from_ref(attr)).is_none() {
                self.error(
                    codes::ATTRIBUTE_ARGUMENT,
                    "`@body` takes one argument, a whole number of bytes written in the text, \
                     above 0 and a multiple of 4"
                        .into(),
                    attr.span,
                );
            }
        }
    }

    /// The rules on the declarations of the module.
    pub(crate) fn module_rules(&mut self, module: &Module, ix: &Index) {
        let mut has_dynamic = false;
        let mut has_grid = false;
        let mut impls: HashSet<String> = HashSet::new();
        let lines = super::shapes::tile_lines(module);
        for item in module.items.nodes() {
            if let Item::Requires(r) = item {
                for req in r.body.nodes() {
                    match req {
                        Requirement::Dynamic { .. } => has_dynamic = true,
                        Requirement::Group {
                            shape: GroupShape::Grid,
                            ..
                        } => has_grid = true,
                        Requirement::Impl { name, .. } => {
                            impls.insert(unquote(&name.text));
                        }
                        _ => {}
                    }
                }
            }
        }
        self.tile_twice(module);
        for item in module.items.nodes() {
            match item {
                Item::Tile(_) | Item::Format(_) | Item::Requires(_) => {}
                Item::Param(p) => {
                    self.attrs_in(&p.attrs, &["bind"], "a module parameter");
                    self.attr_args(&p.attrs, &["bind"], ix);
                    // before the declaration is described, which has no `bool` parameter
                    if matches!(p.ty.kind, TypeKind::Bool) {
                        self.error(
                            codes::BOOL_PLACE,
                            "`bool` is not a type of a parameter".into(),
                            p.ty.span,
                        );
                    }
                    self.require_attr(&p.attrs, "bind", "a parameter", p.span);
                    if matches!(p.value, ParamValue::Fit(_)) {
                        let bind = word_args(&p.attrs, "bind");
                        let has_bind = p.attrs.iter().any(|a| is_named(a, "bind"));
                        if has_bind && !bind.contains("load") {
                            self.error(
                                codes::FIT_NOT_AT_LOAD,
                                format!(
                                    "`{}` is `= fit` and is not bound at load (`@bind(load)`)",
                                    p.name.name
                                ),
                                p.span,
                            );
                        }
                    }
                }
                Item::Message(m) => self.message_attrs(&m.attrs),
                Item::Record(_) => {}
                Item::Func(f) => {
                    self.attrs_in(&f.attrs, &["tile"], "a func");
                    self.tile_args(&f.attrs, &lines);
                    self.param_attrs(&f.params);
                    self.block_attrs(&f.body);
                }
                Item::Combine(f) => {
                    self.attrs_in(&f.attrs, &["commutative", "tile"], "a combine");
                    self.tile_args(&f.attrs, &lines);
                    self.param_attrs(&f.params);
                    self.block_attrs(&f.body);
                }
                Item::Extern(f) => {
                    self.attrs_in(
                        &f.attrs,
                        &["reads", "writes", "impl", "tile"],
                        "an extern func",
                    );
                    self.tile_args(&f.attrs, &lines);
                    self.param_attrs(&f.params);
                    let stated: HashSet<String> = named_args(&f.attrs, "reads")
                        .union(&named_args(&f.attrs, "writes"))
                        .cloned()
                        .collect();
                    let unstated: Vec<&str> = f
                        .params
                        .values()
                        .filter(|p| matches!(p.ty.kind, TypeKind::View))
                        .filter(|p| !stated.contains(&p.name.name))
                        .map(|p| p.name.name.as_str())
                        .collect();
                    if !unstated.is_empty() {
                        self.error(
                            codes::EXTERN_VIEW_UNSTATED,
                            format!(
                                "`{}` takes the view `{}` and states it in neither `@reads` \
                                 nor `@writes`",
                                f.name.name,
                                unstated.join("`, `")
                            ),
                            f.span,
                        );
                    }
                    match &f.fallback {
                        Some(body) => self.block_attrs(body),
                        None => {
                            if let Some(name) = impl_name(&f.attrs) {
                                if !impls.contains(&name) {
                                    self.error(
                                        codes::REQUIREMENT_MISSING,
                                        format!(
                                            "`{}` has no fallback and needs its implementation: \
                                             add `impl(\"{name}\")` to `requires`",
                                            f.name.name
                                        ),
                                        f.span,
                                    );
                                }
                            }
                        }
                    }
                }
                Item::Program(p) => {
                    for part in p.body.nodes() {
                        self.program_item_rules(part, has_dynamic, has_grid, ix, &lines);
                    }
                }
                Item::Control(c) => {
                    self.attrs_in(&c.attrs, &[], "a control program");
                    self.param_attrs(&c.params);
                    self.block_attrs(&c.body);
                }
            }
        }
    }

    fn program_item_rules(
        &mut self,
        part: &ProgramItem,
        has_dynamic: bool,
        has_grid: bool,
        ix: &Index,
        lines: &[super::shapes::Shape],
    ) {
        match part {
            ProgramItem::Message(m) => self.message_attrs(&m.attrs),
            ProgramItem::Record(_) => {}
            ProgramItem::State(s) => {
                self.attrs_in(&s.attrs, &["scope", "lifetime", "tile"], "a state block");
                self.tile_args(&s.attrs, lines);
                self.attr_args(&s.attrs, &["scope", "lifetime"], ix);
                let missing: Vec<&str> = ["scope", "lifetime"]
                    .into_iter()
                    .filter(|name| !s.attrs.iter().any(|a| is_named(a, name)))
                    .collect();
                if !missing.is_empty() {
                    self.error(
                        codes::ATTRIBUTE_MISSING,
                        format!(
                            "a state has no `@{}`; nothing is supplied silently",
                            missing.join("` and no `@")
                        ),
                        s.span,
                    );
                }
                for member in s.members.nodes() {
                    self.attrs_in(&member.attrs, &["export", "filled"], "a state member");
                    self.attr_args(&member.attrs, &["export", "filled"], ix);
                }
            }
            ProgramItem::Handler(h) => {
                self.attrs_in(&h.attrs, &["on", "exclusive", "tile"], "a handler");
                self.tile_args(&h.attrs, lines);
                self.attr_args(&h.attrs, &["on", "exclusive"], ix);
                self.require_attr(&h.attrs, "on", "a handler", h.span);
                self.param_attrs(&h.params);
                self.block_attrs(&h.body);
                if on_grid(&h.attrs) && !has_grid {
                    self.error(
                        codes::REQUIREMENT_MISSING,
                        format!(
                            "the handler `{}` is on a grid: add `group = grid` to `requires`",
                            h.name.name
                        ),
                        h.span,
                    );
                }
            }
            ProgramItem::Run(r) => {
                self.attrs_in(&r.attrs, &["dynamic", "peers", "tile"], "a run");
                self.tile_args(&r.attrs, lines);
                self.peer_numbers(&r.attrs, ix);
                self.require_attr(&r.attrs, "dynamic", "a run", r.span);
                self.param_attrs(&r.params);
                for part in r.body.nodes() {
                    match part {
                        crate::syntax::ast::RunPart::Seed(b) => self.block_attrs(b),
                        crate::syntax::ast::RunPart::On(on) => self.block_attrs(&on.body),
                    }
                }
                if !has_dynamic {
                    self.error(
                        codes::REQUIREMENT_MISSING,
                        format!(
                            "`{}` is a dynamic run: add `dynamic` to `requires`",
                            r.name.name
                        ),
                        r.span,
                    );
                }
            }
            ProgramItem::Chain(c) => {
                self.attrs_in(&c.attrs, &["on"], "a chain");
                self.attr_args(&c.attrs, &["on"], ix);
                self.require_attr(&c.attrs, "on", "a chain", c.span);
                for call in c.calls.nodes() {
                    self.attrs_in(&call.attrs, &["on"], "a chain's call");
                }
                self.param_attrs(&c.params);
                if on_grid(&c.attrs) && !has_grid {
                    self.error(
                        codes::REQUIREMENT_MISSING,
                        format!(
                            "the chain `{}` is on a grid: add `group = grid` to `requires`",
                            c.name.name
                        ),
                        c.span,
                    );
                }
            }
        }
    }
}
