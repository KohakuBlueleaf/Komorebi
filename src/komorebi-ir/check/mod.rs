//! Program compile's checks: names and types, then the rules of the language that
//! need no derived effects. Each rule has its code (`codes`); `docs/ir/checks.md`
//! gives the sentence of the specification that states it.
//!
//! A module that checks clean gives a [`Checked`]: the syntax tree with the type of
//! every expression and what every name stands for, for the steps that follow.

pub mod codes;

mod attrs;
mod body;
pub mod decls;
mod env;
mod index;
mod messages;
pub mod op;
mod ops;
pub mod ty;
mod views;

use std::collections::{HashMap, HashSet};

use crate::errors::Diagnostic;
use crate::syntax::ast::{Expr, Module, Span};
use crate::types::{self as t, Failure, Format, MessageId, ParamId, RecordId, Type};
pub use decls::Decls;
use index::Index;
use op::{OpKind, Resolved};
use ty::{Count, Ty};

/// What a name in a body stands for: the declaration it refers to, by its place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Name {
    /// A `let`, a `var` or a loop's variable of the body, with the span of the name
    /// that declares it.
    Local {
        /// Where the declaring name is written.
        decl: Span,
        /// Which of the three.
        kind: LocalKind,
    },
    /// The parameter at this position of the function, handler, run, chain or control
    /// program the body belongs to.
    Param(u32),
    /// The message an `on M(m)` body handles: its `m`.
    Arrived,
    /// A parameter of the module.
    ModuleParam(ParamId),
    /// A state member: the program's place among the programs, and the member's among
    /// the program's members (every state block's, in order).
    State {
        /// The program.
        program: u32,
        /// The member.
        member: u32,
    },
    /// A function: its place among the module's `func`, `combine` and `extern func`.
    Func(u32),
    /// A program, by its place.
    Program(u32),
    /// A handler of a program, by its place among the program's handlers.
    Handler {
        /// The program.
        program: u32,
        /// The handler.
        index: u32,
    },
    /// A dynamic run of a program, by its place among the program's runs.
    Run {
        /// The program.
        program: u32,
        /// The run.
        index: u32,
    },
    /// A chain of a program, by its place among the program's chains.
    Chain {
        /// The program.
        program: u32,
        /// The chain.
        index: u32,
    },
    /// A record type.
    Record(RecordId),
    /// A message type.
    Message(MessageId),
    /// A field of a message or a record, by its position.
    Field(u32),
    /// A failure's name.
    Failure(Failure),
    /// A word of the language.
    Word(Word),
    /// The name of a provided operation (`partition`, `stream`, `reduce` ...).
    Operation,
}

/// Which kind of local a name declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalKind {
    /// `let`: a value.
    Let,
    /// `var`: a slot, assigned again.
    Var,
    /// A `for` loop's variable: a value the loop gives each step.
    Loop,
}

/// A word of the language that names no declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Word {
    /// `worker`.
    Worker,
    /// `group`.
    Group,
    /// `island`.
    Island,
    /// `zeros`.
    Zeros,
    /// `row` (an axis or a direction).
    Row,
    /// `col`.
    Col,
    /// A provided combination of a reduce: `sum`.
    Sum,
    /// `max`.
    Max,
    /// `min`.
    Min,
    /// `and`.
    And,
    /// `or`.
    Or,
}

impl Word {
    /// The word written `text`, if it is one.
    pub fn from_name(text: &str) -> Option<Word> {
        Some(match text {
            "worker" => Word::Worker,
            "group" => Word::Group,
            "island" => Word::Island,
            "zeros" => Word::Zeros,
            "row" => Word::Row,
            "col" => Word::Col,
            "sum" => Word::Sum,
            "max" => Word::Max,
            "min" => Word::Min,
            "and" => Word::And,
            "or" => Word::Or,
            _ => return None,
        })
    }
}

/// A module that checks clean, with what the checker found out about it.
#[derive(Clone, Debug)]
pub struct Checked {
    /// The syntax tree.
    pub module: Module,
    /// Every declaration, its attributes read.
    pub decls: Decls,
    types: HashMap<(usize, usize), Type>,
    names: HashMap<(usize, usize), Name>,
    ops: HashMap<(usize, usize), Resolved>,
}

impl Checked {
    /// The operation `expr` (an expression of `module`) is, with its operands by part:
    /// a call, a member, an index, an operator, a message, an `await`, `zeros`, or a
    /// state member of one element read by its name. `None` for a name, a number, a
    /// list, or an expression whose value is given by what it names.
    pub fn operation(&self, expr: &Expr) -> Option<&Resolved> {
        self.ops.get(&(expr.span.start, expr.span.end))
    }

    /// The type of a value expression of `module`. `None` for what is no value: a
    /// name of a function, a program or a word, a state member as a whole, a
    /// comparison of tiles inside `where`, what a `for` walks, a list.
    pub fn type_of(&self, expr: &Expr) -> Option<&Type> {
        self.types.get(&(expr.span.start, expr.span.end))
    }

    /// What the name at `span` (a name expression of `module`) stands for.
    pub fn binding(&self, span: Span) -> Option<&Name> {
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
        ops: HashMap::new(),
        state_use: StateReads::default(),
        undeclared_formats: HashSet::new(),
        casts: HashMap::new(),
        late: HashSet::new(),
        tile_missing: false,
        scope: None,
        scopes: HashMap::new(),
    };
    let env = checker.collect(&module);
    checker.module_rules(&module);
    checker.bodies(&module, &env);
    // A number written in the text that no place gave a type has its own (R28).
    for ty in checker.types.values_mut() {
        if let Ty::Number(lit) = ty {
            *ty = Ty::Scalar(lit.alone());
        }
    }
    let decls = if checker.diagnostics.is_empty() {
        match decls::of(&module, &env.ix) {
            Ok(decls) => Some(decls),
            Err(span) => {
                checker.error(
                    codes::NOT_BUILT,
                    "this declaration is named in the language and not described yet".into(),
                    span,
                );
                None
            }
        }
    } else {
        None
    };
    if let (true, Some(decls)) = (checker.diagnostics.is_empty(), decls) {
        let types = checker
            .types
            .iter()
            .filter_map(|(key, ty)| {
                let scope = checker.scopes.get(key).cloned().flatten();
                public(ty, scope.as_deref(), &env.ix, &checker.casts).map(|t| (*key, t))
            })
            .collect();
        Ok(Checked {
            module,
            decls,
            types,
            names: checker.names,
            ops: checker.ops,
        })
    } else {
        let mut diagnostics = checker.diagnostics;
        diagnostics.sort_by_key(|d| (d.start, d.code));
        Err(diagnostics)
    }
}

/// The format each broadcast's casts send, by the place of its call.
type Casts = HashMap<(usize, usize), (Format, Span)>;

/// The type of a value, as every later step names it, from the checker's own; `None`
/// for what is no value. `program` is where the expression is: a message or a record
/// type is found by its name there.
fn public(ty: &Ty, program: Option<&str>, ix: &Index, casts: &Casts) -> Option<Type> {
    let count = |n: &Count| match *n {
        Count::Known(n) => Some(t::Count::Const(n)),
        Count::Param(p) => Some(t::Count::Param(p)),
        Count::GroupSize => Some(t::Count::GroupSize),
        Count::Unknown => None,
    };
    Some(match ty {
        Ty::Scalar(s) => Type::Scalar(*s),
        Ty::Bool => Type::Bool,
        Ty::Tile(f) => Type::Tile((*f)?),
        Ty::Reduced(f, axis) => Type::Reduced((*f)?, *axis),
        Ty::Block(f, n) => Type::Block((*f)?, count(n)?),
        Ty::Tensor { format, replicated } => Type::Tensor((*format)?, *replicated),
        Ty::View { format, .. } => Type::View(*format),
        Ty::Items(_, bytes) => Type::Items(count(bytes)?),
        Ty::Words(s) => Type::Words((*s)?),
        Ty::Records(r, _) => Type::Records(ix.record_ref(program, r)?),
        Ty::Record(r) => Type::RecordValue(ix.record_ref(program, r)?),
        Ty::Message(m) => Type::Message(ix.message_ref(program, m)?),
        Ty::Stream(elem, _) => match &**elem {
            Ty::Tile(f) => Type::Stream((*f)?, t::Count::Const(1)),
            Ty::Block(f, n) => Type::Stream((*f)?, count(n)?),
            _ => return None,
        },
        Ty::Zip(streams) => match streams.as_slice() {
            [a, b] => Type::Zip(
                Box::new(public(a, program, ix, casts)?),
                Box::new(public(b, program, ix, casts)?),
            ),
            _ => return None,
        },
        Ty::Sink(f, n) => Type::Sink((*f)?, count(n)?),
        Ty::SubGroups => Type::SubGroups,
        Ty::Leaders => Type::Leaders,
        Ty::Broadcast(at) => Type::Broadcast(casts.get(at)?.0),
        Ty::Group => Type::Group,
        Ty::Handle(ret) => Type::Handle(result(ret.as_deref())?),
        Ty::Outcome(ret) => Type::Outcome(result(ret.as_deref())?),
        Ty::Reason => Type::Reason,
        _ => return None,
    })
}

/// An invocation's result: none, or a scalar; `None` for another type.
fn result(ret: Option<&Ty>) -> Option<Option<crate::types::ScalarType>> {
    match ret {
        None | Some(Ty::Void) => Some(None),
        Some(Ty::Scalar(s)) => Some(Some(*s)),
        Some(_) => None,
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
    names: HashMap<(usize, usize), Name>,
    ops: HashMap<(usize, usize), Resolved>,
    /// What the body being checked does with its program's state: the members it
    /// reads (with where), and the members it fills.
    state_use: StateReads,
    /// The formats used and not declared, each reported at its first use.
    undeclared_formats: HashSet<Format>,
    /// Whether `K118` (no `tile` line) is reported.
    tile_missing: bool,
    /// The program whose body is being checked, if any.
    scope: Option<String>,
    /// The program each expression's type was noted in, where it was in one: a
    /// message or a record type is found by its name there.
    scopes: HashMap<(usize, usize), Option<String>>,
    /// The format each broadcast's casts send (by the place of its call), with the
    /// first cast.
    casts: Casts,
    /// The broadcasts whose arrivals were walked before a cast on them gave a format.
    late: HashSet<(usize, usize)>,
}

/// The members a body reads, each with where, and the members it fills.
pub(crate) type StateReads = (Vec<(String, Span)>, HashSet<String>);

impl Checker<'_> {
    /// Notes that a cast on the broadcast at `at` sends tiles of `format`: the first
    /// gives the broadcast its format, and another must send the same (`K403`).
    pub(crate) fn cast_format(&mut self, at: (usize, usize), format: Format, span: Span) {
        match self.casts.get(&at) {
            None => {
                self.casts.insert(at, (format, span));
            }
            Some((first, _)) if *first == format => {}
            Some((first, _)) => {
                let first = *first;
                self.error(
                    codes::BROADCAST_FORMATS,
                    format!(
                        "this cast sends {format} tiles and an earlier cast on the same \
                         broadcast sends {first}: one broadcast carries one format"
                    ),
                    span,
                );
            }
        }
    }

    /// Reports an error with `code` at `span`.
    pub(crate) fn error(&mut self, code: &'static str, message: String, span: Span) {
        self.diagnostics
            .push(Diagnostic::error(self.source, code, message, span));
    }

    /// Notes the operation of the expression at `span`, with its operands by part.
    pub(crate) fn note_op(&mut self, span: Span, kind: OpKind, parts: &[Option<&Expr>]) {
        let parts = parts.iter().map(|p| p.map(|e| e.span)).collect();
        self.ops
            .insert((span.start, span.end), Resolved { kind, parts });
    }

    /// Notes the type of an expression.
    pub(crate) fn note_type(&mut self, expr: &Expr, ty: &Ty) {
        let key = (expr.span.start, expr.span.end);
        self.types.insert(key, ty.clone());
        if self.scope.is_some() {
            self.scopes.insert(key, self.scope.clone());
        }
    }

    /// Notes what a name stands for.
    pub(crate) fn note_name(&mut self, span: Span, name: Name) {
        self.names.insert((span.start, span.end), name);
    }
}

#[cfg(test)]
mod tests {
    use crate::check;
    use crate::syntax::ast::{ExprKind, Item, ProgramItem, Stmt};
    use crate::types::Type;

    use super::Name;

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
        assert_eq!(
            checked.type_of(value),
            Some(&Type::View(Some(crate::types::Format::F32)))
        );
        let Stmt::Return(Some(n)) = stmts[1] else {
            panic!("a return second");
        };
        assert_eq!(
            checked.type_of(n),
            Some(&Type::Scalar(crate::types::ScalarType::U32))
        );
        let ExprKind::Name(name) = &n.kind else {
            panic!("a name");
        };
        assert_eq!(checked.binding(name.span), Some(&Name::Param(1)));
    }

    const CHAIN: &str = "module m\n\ntile [32, 32]\nformat f32\n\n\
        param LIMIT: u32 = 64 @bind(build)\nparam BLOCK: u32 = 8 @bind(invoke)\n\nprogram P {\n    \
        handler h(x: tensor<f32>, n: u32) @on(group) {\n        \
        for xb in stream(partition(x.tiles(0, n), worker.rank, group.size), block = 1) {\n        }\n    }\n\n    \
        chain c(x: tensor<f32>) @on(group) {\n        call h(x, ARG)\n    }\n}\n";

    #[test]
    fn a_chain_call_takes_a_parameter_bound_at_build() {
        assert_eq!(codes(&CHAIN.replace("ARG", "LIMIT")), Vec::<&str>::new());
        assert_eq!(codes(&CHAIN.replace("ARG", "16")), Vec::<&str>::new());
        // bound at the invocation: not a constant, and the structure has no form for it yet
        assert_eq!(codes(&CHAIN.replace("ARG", "BLOCK")), vec!["K603"]);
    }

    /// The type of the value of each `let` of the control program `c` in `text`.
    fn let_types(text: &str) -> Vec<Type> {
        let checked = check(text).unwrap_or_else(|d| panic!("{d:?}"));
        let Some(Item::Control(c)) = checked.module.items.nodes().last() else {
            panic!("the module ends with its control program");
        };
        c.body
            .stmts
            .nodes()
            .filter_map(|s| match s {
                Stmt::Let { value, .. } => checked.type_of(value).cloned(),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn each_let_names_its_operation_and_operands() {
        use super::op::OpKind;
        use crate::types::ScalarType::U8;
        use crate::types::{BinOp, CmpOp, UnaryOp};
        let text = "module m\n\ntile [32, 32]\nformat f32\n\n\
            control c(a: u32, b: u32) {\n    let x = a + b\n    let y = min(a, b)\n    \
            let z = u8(a)\n    let w = a < 3\n    let v = -a\n    let u = group(2, a)\n}\n";
        let checked = check(text).unwrap_or_else(|d| panic!("{d:?}"));
        let Some(Item::Control(c)) = checked.module.items.nodes().last() else {
            panic!("the module ends with its control program");
        };
        let found: Vec<(OpKind, Vec<&str>)> = c
            .body
            .stmts
            .nodes()
            .filter_map(|s| match s {
                Stmt::Let { value, .. } => {
                    let r = checked.operation(value)?;
                    let parts = r
                        .operands(value)
                        .iter()
                        .map(|p| p.map_or("-", |e| &text[e.span.start..e.span.end]))
                        .collect();
                    Some((r.kind.clone(), parts))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            found,
            vec![
                (OpKind::Binary(BinOp::Add), vec!["a", "b"]),
                (OpKind::Binary(BinOp::Min), vec!["a", "b"]),
                (OpKind::Convert(U8), vec!["a"]),
                (OpKind::Compare(CmpOp::Lt), vec!["a", "3"]),
                (OpKind::Unary(UnaryOp::Neg), vec!["a"]),
                (OpKind::MakeGroup, vec!["2", "a"]),
            ]
        );
    }

    #[test]
    fn a_tile_an_operation_yields_has_the_format_of_its_place() {
        use crate::types::Format;
        let text = "module m\n\ntile [32, 32]\nformat f32, bf16, i32, u32\n\n\
            func f(x: tile<bf16>, w: tile<bf16>) -> tile<f32> {\n    return matmul(x, w)\n}\n";
        let checked = check(text).unwrap_or_else(|d| panic!("{d:?}"));
        let Some(Item::Func(f)) = checked.module.items.nodes().last() else {
            panic!("the module ends with its function");
        };
        let Some(Stmt::Return(Some(value))) = f.body.stmts.nodes().next() else {
            panic!("a return");
        };
        assert_eq!(checked.type_of(value), Some(&Type::Tile(Format::F32)));
        // integer formats agree
        let text = "module m\n\ntile [32, 32]\nformat i32, u32\n\n\
            func g(x: tile<i32>) -> tile<u32> {\n    return x + x\n}\n";
        assert_eq!(codes(text), vec!["K102"]);
    }

    #[test]
    fn a_broadcast_carries_the_format_its_casts_send() {
        use crate::types::Format;
        // the arrivals come first in the text: the body is checked again
        let text = "module m\n\ntile [32, 32]\nformat bf16\nrequires {\n    group = grid\n}\n\n\
            program P {\n    handler h(x: tensor<bf16>, y: tensor<bf16>) @on(grid[2, 2]) {\n        \
            let b = broadcast(along = row, sender = 0)\n        \
            if worker.col != 0 {\n            let out = sink(y.tiles(0, 1), block = 1)\n            \
            for xb, tag in b.arrive() {\n                out.push(xb)\n            }\n        \
            } else {\n            for xb in stream(x.tiles(0, 1), block = 1) {\n                \
            b.cast(xb, word = 0)\n            }\n        }\n    }\n}\n";
        let checked = check(text).unwrap_or_else(|d| panic!("{d:?}"));
        let Some(Item::Program(p)) = checked.module.items.nodes().last() else {
            panic!("the module ends with its program");
        };
        let Some(ProgramItem::Handler(h)) = p.body.nodes().next() else {
            panic!("the program holds a handler");
        };
        let Some(Stmt::Let { value, .. }) = h.body.stmts.nodes().next() else {
            panic!("a let first");
        };
        assert_eq!(checked.type_of(value), Some(&Type::Broadcast(Format::Bf16)));
        let Some(Stmt::If { then, .. }) = h.body.stmts.nodes().nth(1) else {
            panic!("an if second");
        };
        let Some(Stmt::For { body, .. }) = then.stmts.nodes().nth(1) else {
            panic!("the arrivals walked");
        };
        let Some(Stmt::Expr(push)) = body.stmts.nodes().next() else {
            panic!("a push");
        };
        let ExprKind::Call { args, .. } = &push.kind else {
            panic!("a call");
        };
        let xb = &args.values().next().expect("the block pushed").value;
        assert_eq!(checked.type_of(xb), Some(&Type::Tile(Format::Bf16)));
    }

    #[test]
    fn the_declarations_are_read_with_their_attributes() {
        use super::decls::{Element, FunctionKind};
        use crate::types::{
            Ability, Binding, ConstValue, Count, Format, On, ParamId, Peers, ScalarType, TileShape,
        };
        let text = "module m\n\ntile [32, 32]\nformat f32\n\n\
            requires {\n    group = grid\n    dynamic\n    local_state >= tiles(2)\n}\n\n\
            param N: u32 = 4 @bind(build)\nparam ROOM: u32 = fit @bind(load)\n\
            param B: i32 = -2 @bind(invoke)\n\n\
            message Job {\n    n: u32\n}\n\n\
            combine add(a: tile<f32>, b: tile<f32>) -> tile<f32> @commutative {\n    return a + b\n}\n\n\
            program P {\n    state @scope(worker) @lifetime(program) {\n        \
            w: tile<f32>[ROOM]\n        seen: u32[N] @export\n    }\n\n    \
            handler h(x: tensor<f32>) -> u32 @on(grid[2, 4]) @exclusive {\n        return 0\n    }\n\n    \
            run r(x: tensor<f32>) @dynamic @peers(offsets = [1]) {\n        seed {\n        }\n\n        \
            on Job(m) {\n            consume m\n        }\n    }\n}\n";
        let checked = check(text).unwrap_or_else(|d| panic!("{d:?}"));
        let d = &checked.decls;
        assert_eq!(d.tile, Some(TileShape { rows: 32, cols: 32 }));
        assert_eq!(d.formats, vec![Format::F32]);
        assert_eq!(
            d.requires,
            vec![
                Ability::Grid,
                Ability::Dynamic,
                Ability::LocalState {
                    amount: 2,
                    tiles: true
                }
            ]
        );
        let bindings: Vec<&Binding> = d.params.iter().map(|p| &p.binding).collect();
        assert_eq!(
            bindings,
            vec![
                &Binding::Build(ConstValue::Int(ScalarType::U32, 4)),
                &Binding::Load(None),
                &Binding::Invoke(ConstValue::Int(ScalarType::I32, 0xffff_fffe)),
            ]
        );
        assert_eq!(
            d.messages[0].fields,
            vec![("n".to_string(), ScalarType::U32)]
        );
        assert_eq!(d.funcs[0].kind, FunctionKind::Combine { commutative: true });
        let p = &d.programs[0];
        assert_eq!(p.members[0].element, Element::Tiles(Format::F32));
        assert_eq!(p.members[0].count, Count::Param(ParamId(1)));
        assert_eq!(p.members[1].count, Count::Const(4));
        assert!(p.members[1].export);
        assert_eq!(p.handlers[0].on, On::Grid(2, 4));
        assert!(p.handlers[0].exclusive);
        assert_eq!(p.handlers[0].ret, Some(ScalarType::U32));
        assert_eq!(p.runs[0].peers, Peers::Offsets(vec![1]));
    }

    #[test]
    fn a_number_has_the_type_of_its_place() {
        use crate::types::ScalarType::{F32, I32, U32, U8};
        let text = "module m\n\ntile [32, 32]\nformat f32\n\n\
            control c(a: i32, b: u8) {\n    let w = 5\n    let x = 2.5\n    let y = a + 1\n    \
            let z = 1 + b\n    let v: u8 = 7\n    let u = f32(a) * 0.5\n    let t = i32(b) - a\n}\n";
        assert_eq!(
            let_types(text),
            [U32, F32, I32, U8, U8, F32, I32].map(Type::Scalar).to_vec()
        );
    }

    #[test]
    fn two_scalars_of_different_types_are_refused() {
        let text = "module m\n\ntile [32, 32]\nformat f32\n\n\
            control c(a: i32, b: u32) {\n    let x = EXPR\n}\n";
        assert_eq!(codes(&text.replace("EXPR", "a + b")), vec!["K119"]);
        assert_eq!(codes(&text.replace("EXPR", "a < b")), vec!["K119"]);
        assert_eq!(codes(&text.replace("EXPR", "a << b")), Vec::<&str>::new());
        assert_eq!(
            codes(&text.replace("EXPR", "a + i32(b)")),
            Vec::<&str>::new()
        );
        // a number with a point is no integer's
        assert_eq!(codes(&text.replace("EXPR", "u8(2.5)")), vec!["K102"]);
    }

    #[test]
    fn a_number_its_place_cannot_hold_is_refused() {
        let text = "module m\n\ntile [32, 32]\nformat f32\n\n\
            control c(b: u8) {\n    let x = EXPR\n}\n";
        let refused = |expr: &str| codes(&text.replace("EXPR", expr));
        // no place: u32
        assert_eq!(refused("-1"), vec!["K115"]);
        assert_eq!(refused("0xffffffff"), Vec::<&str>::new());
        // the other operand's type
        assert_eq!(refused("b + 255"), Vec::<&str>::new());
        assert_eq!(refused("b + 300"), vec!["K115"]);
        // a conversion's number takes the target type as its place
        assert_eq!(refused("u32(-1)"), vec!["K115"]);
        assert_eq!(refused("i8(-128)"), Vec::<&str>::new());
        assert_eq!(refused("u8(256)"), vec!["K115"]);
        assert_eq!(refused("f16(70000.0)"), vec!["K115"]);
        // a value that is not a number written in the text is converted
        assert_eq!(refused("u8(u32(b) + 300)"), Vec::<&str>::new());
        assert_eq!(
            codes(&text.replace("let x = EXPR", "let x: i32 = -1")),
            Vec::<&str>::new()
        );
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
