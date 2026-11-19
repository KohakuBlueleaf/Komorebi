//! Program compile's checks: names and types, then the rules of the language that
//! need no derived effects. Each rule has its code (`codes`); `docs/ir/checks.md`
//! gives the sentence of the specification that states it.
//!
//! A module that checks clean gives a [`Checked`]: the syntax tree with the type of
//! every expression and what every name stands for, for the steps that follow.

pub mod codes;

mod attrs;
mod body;
mod env;
mod messages;
mod ops;
pub mod ty;
mod views;

use std::collections::{HashMap, HashSet};

use crate::errors::Diagnostic;
use crate::syntax::ast::{Expr, Module, Span};
use crate::types::Format;
use ty::Ty;

/// What a name in a body stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    /// A `let`, a `var` or a loop variable of the body.
    Local,
    /// A parameter of the function, handler, run, chain or control program.
    Param,
    /// A state member of the program.
    State,
    /// A parameter of the module.
    ModuleParam,
    /// A function of the module.
    Func,
    /// A program of the module (in `P.h(...)` of an `invoke`).
    Program,
    /// The message an `on M(m)` body handles.
    Message,
    /// A word of an operation: `worker`, `group`, `island`, `row`, `col`, `zeros`, a
    /// provided combination (`sum` ...), a failure's name, a record type.
    Word,
}

/// A module that checks clean, with what the checker found out about it.
#[derive(Clone, Debug)]
pub struct Checked {
    /// The syntax tree.
    pub module: Module,
    types: HashMap<(usize, usize), Ty>,
    names: HashMap<(usize, usize), Binding>,
}

impl Checked {
    /// The type of an expression of `module`.
    pub fn type_of(&self, expr: &Expr) -> Option<&Ty> {
        self.types.get(&(expr.span.start, expr.span.end))
    }

    /// What the name at `span` (a name expression of `module`) stands for.
    pub fn binding(&self, span: Span) -> Option<&Binding> {
        self.names.get(&(span.start, span.end))
    }
}

/// Checks a parsed module: the module with its types, or every diagnostic found.
pub fn check_module(source: &str, module: Module) -> Result<Checked, Vec<Diagnostic>> {
    let mut checker = Checker {
        source,
        diagnostics: Vec::new(),
        message_names: HashSet::new(),
        record_names: HashSet::new(),
        types: HashMap::new(),
        names: HashMap::new(),
        state_use: StateReads::default(),
        undeclared_formats: HashSet::new(),
    };
    let env = checker.collect(&module);
    checker.module_rules(&module);
    checker.bodies(&module, &env);
    if checker.diagnostics.is_empty() {
        Ok(Checked {
            module,
            types: checker.types,
            names: checker.names,
        })
    } else {
        let mut diagnostics = checker.diagnostics;
        diagnostics.sort_by_key(|d| (d.start, d.code));
        Err(diagnostics)
    }
}

/// The state of one check of a module.
pub(crate) struct Checker<'a> {
    source: &'a str,
    diagnostics: Vec<Diagnostic>,
    /// The names of the message types, the module's and the programs'.
    message_names: HashSet<String>,
    /// The names of the record types.
    record_names: HashSet<String>,
    types: HashMap<(usize, usize), Ty>,
    names: HashMap<(usize, usize), Binding>,
    /// What the body being checked does with its program's state: the members it
    /// reads (with where), and the members it fills.
    state_use: StateReads,
    /// The formats used and not declared, each reported at its first use.
    undeclared_formats: HashSet<Format>,
}

/// The members a body reads, each with where, and the members it fills.
pub(crate) type StateReads = (Vec<(String, Span)>, HashSet<String>);

impl Checker<'_> {
    /// Reports an error with `code` at `span`.
    pub(crate) fn error(&mut self, code: &'static str, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::error(self.source, code, message, span));
    }

    /// Notes the type of an expression.
    pub(crate) fn note_type(&mut self, expr: &Expr, ty: &Ty) {
        self.types
            .insert((expr.span.start, expr.span.end), ty.clone());
    }

    /// Notes what a name stands for.
    pub(crate) fn note_name(&mut self, span: Span, binding: Binding) {
        self.names.insert((span.start, span.end), binding);
    }
}

#[cfg(test)]
mod tests {
    use super::ty::Ty;
    use crate::check;
    use crate::syntax::ast::{ExprKind, Item, ProgramItem, Stmt};

    use super::Binding;

    /// The codes a text gives, in order.
    fn codes(text: &str) -> Vec<&'static str> {
        match check(text) {
            Ok(_) => Vec::new(),
            Err(diagnostics) => diagnostics.iter().map(|d| d.code).collect(),
        }
    }

    const STREAM: &str = "module m\n\ntile [32, 32]\nformat f32\n\nprogram P {\n    \
        handler h(x: tensor<f32>) @on(group) {\n        let out = sink(x.tiles(WRITE), block = 1)\n        \
        for xb in stream(x.tiles(0, 32), block = 1) {\n            out.push(xb)\n        }\n    }\n}\n";

    #[test]
    fn a_stream_may_write_apart_from_what_it_reads() {
        assert_eq!(
            codes(&STREAM.replace("WRITE", "32, 32")),
            Vec::<&str>::new()
        );
        assert_eq!(codes(&STREAM.replace("WRITE", "0, 32")), Vec::<&str>::new());
        assert_eq!(codes(&STREAM.replace("WRITE", "16, 32")), vec!["K206"]);
    }

    const OFFER: &str = "module m\n\ntile [32, 32]\nformat u32\n\nrequires {\n    dynamic\n}\n\n\
        program P {\n    message T { s: u32 }\n\n    run r() @dynamic @peers(pairs = [(0, 5), (5, 0)]) {\n        \
        seed {\n            let went = offer(T { s: 0 }, to = RANK)\n        }\n    }\n}\n";

    #[test]
    fn an_offer_to_a_pair() {
        assert_eq!(codes(&OFFER.replace("RANK", "5")), Vec::<&str>::new());
        assert_eq!(codes(&OFFER.replace("RANK", "3")), vec!["K502"]);
        // a rank the text does not show is the run's to find
        assert_eq!(
            codes(&OFFER.replace("RANK", "worker.id")),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn a_checked_module_is_typed_and_resolved() {
        let text = "module m\n\ntile [32, 32]\nformat f32\n\nprogram P {\n    \
            handler h(x: tensor<f32>, n: u32) -> u32 @on(group) {\n        \
            let v = x.tiles(0, n)\n        return n\n    }\n}\n";
        let checked = check(text).unwrap_or_else(|d| panic!("{d:?}"));
        let Some(Item::Program(p)) = checked.module.items.nodes().last() else {
            panic!("the module ends with its program");
        };
        let Some(ProgramItem::Handler(h)) = p.body.nodes().next() else {
            panic!("the program holds a handler");
        };
        let stmts: Vec<&Stmt> = h.body.stmts.nodes().collect();
        let Stmt::Let { value, .. } = stmts[0] else {
            panic!("a let first");
        };
        assert!(matches!(checked.type_of(value), Some(Ty::View { .. })));
        let Stmt::Return(Some(n)) = stmts[1] else {
            panic!("a return second");
        };
        assert_eq!(checked.type_of(n), Some(&Ty::Int));
        let ExprKind::Name(name) = &n.kind else {
            panic!("a name");
        };
        assert_eq!(checked.binding(name.span), Some(&Binding::Param));
    }

    const FUNC: &str = "module m\n\ntile [32, 32]\nformat f32, u32\n\n\
        func f(a: tile<f32>, n: tile<u32>) -> tile<f32> {\n    return BODY\n}\n";

    fn func_codes(body: &str) -> Vec<&'static str> {
        codes(&FUNC.replace("BODY", body))
    }

    #[test]
    fn a_reduction_is_spread_by_its_own_axis() {
        assert_eq!(
            func_codes("a * reduce_sum(a, axis = row).col()"),
            Vec::<&str>::new()
        );
        assert_eq!(
            func_codes("a * reduce_sum(a, axis = col).row()"),
            Vec::<&str>::new()
        );
        assert_eq!(
            func_codes("a * exp(reduce_max(a, axis = row) * 2.0).col()"),
            Vec::<&str>::new()
        );
        assert_eq!(
            func_codes("a * reduce_sum(a, axis = row).row()"),
            vec!["K116"]
        );
        assert_eq!(
            func_codes("matmul(a, reduce_sum(a, axis = row))"),
            vec!["K116"]
        );
        assert_eq!(func_codes("reduce_sum(a, axis = row)"), vec!["K116"]);
        assert_eq!(
            func_codes("(reduce_sum(a, axis = row) + reduce_sum(a, axis = col)).col()"),
            vec!["K116"]
        );
    }

    #[test]
    fn integer_and_float_tiles() {
        assert_eq!(func_codes("a * 2"), Vec::<&str>::new());
        assert_eq!(func_codes("a / 0.0"), Vec::<&str>::new());
        assert_eq!(
            codes(
                &FUNC
                    .replace("-> tile<f32>", "-> tile<u32>")
                    .replace("BODY", "n * 3 % 7")
            ),
            Vec::<&str>::new()
        );
        assert_eq!(func_codes("a + n"), vec!["K117"]);
        assert_eq!(
            codes(
                &FUNC
                    .replace("-> tile<f32>", "-> tile<u32>")
                    .replace("BODY", "n / 0")
            ),
            vec!["K114"]
        );
    }
}
