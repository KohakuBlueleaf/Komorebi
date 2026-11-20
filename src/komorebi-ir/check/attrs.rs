//! The rules on the module's declarations: attributes (known, in place, present),
//! `= fit`, what `requires` must state, and what is named and not built yet there.

use std::collections::HashSet;

use crate::syntax::ast::{
    Attr, Block, ExprKind, GroupShape, Item, List, Module, ParamDecl, ParamValue, ProgramItem,
    Requirement, Span, Stmt, TypeKind,
};

use super::codes;
use super::env::is_named;
use super::Checker;

/// The language's own undotted attributes. A dotted name is a hint and never an error.
const KNOWN: [&str; 15] = [
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
];

/// True when `@on(...)` among `attrs` names a grid.
pub(crate) fn on_grid(attrs: &[Attr]) -> bool {
    attrs.iter().filter(|a| is_named(a, "on")).any(|a| {
        a.args.as_ref().is_some_and(|args| {
            args.values().any(|arg| match &arg.value.kind {
                ExprKind::Index { base, .. } => {
                    matches!(&base.kind, ExprKind::Name(n) if n.name == "grid")
                }
                ExprKind::Name(n) => n.name == "grid",
                _ => false,
            })
        })
    })
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

impl Checker<'_> {
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

    fn message_attrs(&mut self, attrs: &[Attr]) {
        self.attrs_in(attrs, &["body"], "a message");
        if let Some(attr) = attrs.iter().find(|a| is_named(a, "body")) {
            self.error(
                codes::NOT_BUILT,
                "a message's body (`@body`) is named in the language and not built yet".into(),
                attr.span,
            );
        }
    }

    /// The rules on the declarations of the module.
    pub(crate) fn module_rules(&mut self, module: &Module) {
        let mut has_dynamic = false;
        let mut has_grid = false;
        let mut impls: HashSet<String> = HashSet::new();
        let mut tiles = 0;
        for item in module.items.nodes() {
            match item {
                Item::Requires(r) => {
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
                Item::Tile(t) => {
                    tiles += 1;
                    if tiles > 1 {
                        self.error(
                            codes::NOT_BUILT,
                            "several tile shapes in one module are named in the language and \
                             not built yet"
                                .into(),
                            t.span,
                        );
                    }
                }
                _ => {}
            }
        }
        for item in module.items.nodes() {
            match item {
                Item::Tile(_) | Item::Format(_) | Item::Requires(_) => {}
                Item::Param(p) => {
                    self.attrs_in(&p.attrs, &["bind"], "a module parameter");
                    self.require_attr(&p.attrs, "bind", "a parameter", p.span);
                    if matches!(p.value, ParamValue::Fit(_)) {
                        let bind = named_args(&p.attrs, "bind");
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
                    self.attrs_in(&f.attrs, &[], "a func");
                    self.param_attrs(&f.params);
                    self.block_attrs(&f.body);
                }
                Item::Combine(f) => {
                    self.attrs_in(&f.attrs, &["commutative"], "a combine");
                    self.param_attrs(&f.params);
                    self.block_attrs(&f.body);
                }
                Item::Extern(f) => {
                    self.attrs_in(&f.attrs, &["reads", "writes", "impl"], "an extern func");
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
                        self.program_item_rules(part, has_dynamic, has_grid);
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

    fn program_item_rules(&mut self, part: &ProgramItem, has_dynamic: bool, has_grid: bool) {
        match part {
            ProgramItem::Message(m) => self.message_attrs(&m.attrs),
            ProgramItem::Record(_) => {}
            ProgramItem::State(s) => {
                self.attrs_in(&s.attrs, &["scope", "lifetime"], "a state block");
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
                }
            }
            ProgramItem::Handler(h) => {
                self.attrs_in(&h.attrs, &["on", "exclusive"], "a handler");
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
                self.attrs_in(&r.attrs, &["dynamic", "peers"], "a run");
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
