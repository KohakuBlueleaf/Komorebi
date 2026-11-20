//! Round trips of the text form: text to tree to text, and tree to text to tree.
//!
//! The corpus is read from `tests/ir/corpus/` (or the folder `KOMO_IR_CORPUS` names).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use komorebi_ir::syntax::ast::*;
use komorebi_ir::{parse, write};

fn corpus_dir() -> PathBuf {
    match std::env::var_os("KOMO_IR_CORPUS") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/corpus"),
    }
}

fn kom_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            kom_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "kom") {
            out.push(path);
        }
    }
}

fn corpus(sub: &str) -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();
    kom_files(&corpus_dir().join(sub), &mut files);
    files
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path).expect("a corpus file is readable");
            (path, text)
        })
        .collect()
}

fn lessons() -> Vec<(PathBuf, String)> {
    let lessons = corpus("golden/lessons");
    assert_eq!(
        lessons.len(),
        15,
        "the fifteen IR lessons under {}",
        corpus_dir().join("golden/lessons").display()
    );
    lessons
}

fn assert_round_trip(module: &Module, what: &str) {
    let text = write(module);
    let again = match parse(&text) {
        Ok(again) => again,
        Err(d) => panic!("{what}: the written text does not parse: {d:?}\n{text}"),
    };
    assert!(
        &again == module,
        "{what}: parse(write(p)) != p\n{text}\n{module:#?}\n{again:#?}"
    );
    assert_eq!(write(&again), text, "{what}: the writer is not stable");
}

/// A `func` takes hints after its head, as a `combine` and an `extern func` do: read,
/// written back the same, and read again to the same tree.
#[test]
fn a_func_with_a_hint_round_trips() {
    let text = "module hinted\n\ntile [32, 32]\nformat bf16, f32\n\n\
                func project(x: tile<bf16>, w: tile<bf16>) -> tile<f32> @tensix.fidelity(HiFi2) {\n\
                \x20   return matmul(x, w)\n}\n";
    let module = parse(text).unwrap_or_else(|d| panic!("{d:?}"));
    let Some(Item::Func(func)) = module.items.nodes().find(|i| matches!(i, Item::Func(_))) else {
        panic!("no func: {module:#?}");
    };
    let path: Vec<&str> = func.attrs[0].path.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(path, ["tensix", "fidelity"]);
    assert_eq!(write(&module), text, "the canonical text written back");
    assert_round_trip(&module, "a func with a hint");
}

/// Every lesson is written in the canonical layout: `komo fmt` gives it back.
#[test]
fn lessons_write_back() {
    for (path, text) in lessons() {
        let module = parse(&text).unwrap_or_else(|d| panic!("{}: {d:?}", path.display()));
        assert_eq!(write(&module), text, "{}", path.display());
    }
}

/// Every fragment of the specification, made a module, parses, is written back byte
/// for byte (the folder holds them in the canonical layout), and round-trips.
#[test]
fn fragments_write_back() {
    let fragments = corpus("golden/fragments");
    assert!(fragments.len() >= 17, "the fragments are there");
    for (path, text) in fragments {
        let what = path.display().to_string();
        let module = parse(&text).unwrap_or_else(|d| panic!("{what}: {d:?}"));
        assert_eq!(write(&module), text, "{what}");
        assert_round_trip(&module, &what);
    }
}

#[test]
fn corpus_round_trip() {
    let mut parsed = 0;
    for (path, text) in corpus("golden") {
        let what = path.display().to_string();
        let must_parse = !what.contains("invalid");
        match parse(&text) {
            Ok(module) => {
                assert_round_trip(&module, &what);
                parsed += 1;
            }
            Err(d) => assert!(!must_parse, "{what}: {d:?}"),
        }
    }
    assert!(parsed >= 15, "the corpus holds at least the lessons");
}

/// A text that is not a module gives at least one diagnostic inside the text, and
/// never panics; a text that is one round-trips.
fn check_any_text(text: &str) -> bool {
    let result = catch_unwind(AssertUnwindSafe(|| parse(text)));
    let Ok(result) = result else {
        panic!("the parser panicked on:\n{text}");
    };
    match result {
        Ok(module) => {
            assert_round_trip(&module, "a mangled text");
            false
        }
        Err(diagnostics) => {
            assert!(!diagnostics.is_empty());
            let lines = text.split('\n').count();
            for d in &diagnostics {
                assert!(d.start.line >= 1 && d.start.line <= lines, "{d:?}");
                assert!(
                    d.span.start <= text.len() && d.span.end <= text.len(),
                    "{d:?}"
                );
            }
            true
        }
    }
}

#[test]
fn prefixes_never_panic() {
    let lessons = lessons();
    for (path, text) in lessons.iter().filter(|(path, _)| {
        let name = path.file_name().unwrap().to_string_lossy();
        name.starts_with("t05") || name.starts_with("t09")
    }) {
        let (mut refused, mut modules) = (0, 0);
        for (at, _) in text.char_indices() {
            if check_any_text(&text[..at]) {
                refused += 1;
            } else {
                modules += 1;
            }
        }
        eprintln!(
            "{}: {} prefixes, {refused} refused, {modules} are modules",
            path.display(),
            refused + modules
        );
        assert!(refused > 0);
    }
}

/// A small generator of pseudo-random numbers (xorshift), so the tests need no crate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

#[test]
fn mangled_texts_never_panic() {
    let lessons = lessons();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let pieces = [
        "(", ")", "{", "}", "[", "]", ",", ":", "=", "@", "//", "\n", " ", "-", "<-", "->", "\"",
        "x", "0", "1.5", "let ", "if ", "else", "state ", "on ", ".", "<", ">", "|", "&",
    ];
    let mut refused = 0;
    let total = 3000;
    for k in 0..total {
        let (_, text) = &lessons[k % lessons.len()];
        let mut chars: Vec<char> = text.chars().collect();
        for _ in 0..1 + rng.below(4) {
            let at = rng.below(chars.len());
            match rng.below(3) {
                0 => {
                    chars.remove(at);
                }
                1 => {
                    let piece: Vec<char> = rng.pick(&pieces).chars().collect();
                    chars.splice(at..at, piece);
                }
                _ => chars[at] = *rng.pick(&['(', '}', ' ', '\n', 'a', '0', '@', '"', '/']),
            }
        }
        let mangled: String = chars.into_iter().collect();
        if check_any_text(&mangled) {
            refused += 1;
        }
    }
    eprintln!("{total} mangled lessons, {refused} refused");
    assert!(refused > total / 2);
}

/// A module whose control body is `let a = <expr>`, inside `blocks` nested `if a { }`.
fn deep_module(expr: &str, blocks: usize) -> String {
    let mut body = format!("let a = {expr}");
    for _ in 0..blocks {
        body = format!("if a {{\n{body}\n}}");
    }
    format!("module m\n\ncontrol c() {{\n{body}\n}}\n")
}

/// The code of the one diagnostic of a refused text.
fn refusal(text: &str) -> &'static str {
    match parse(text) {
        Ok(_) => "accepted",
        Err(d) => d[0].code,
    }
}

/// Expressions of depth `n`: the operand counts one, and each bracket, sign, member
/// and operator of a chain one more.
fn deep_exprs(n: usize) -> Vec<String> {
    vec![
        format!("{}x{}", "(".repeat(n - 1), ")".repeat(n - 1)),
        format!("{}x", "-".repeat(n - 1)),
        format!("x{}", ".m".repeat(n - 1)),
        format!("x{}", " + 1".repeat(n - 1)),
        format!("{}x{}", "[".repeat(n - 1), "]".repeat(n - 1)),
        format!("{}x{}", "f(".repeat(n - 1), ")".repeat(n - 1)),
    ]
}

#[test]
fn depth_256_is_read_and_257_is_k011() {
    for e in deep_exprs(256) {
        let text = deep_module(&e, 0);
        let module = parse(&text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
        assert_round_trip(&module, "an expression 256 deep");
    }
    for e in deep_exprs(257) {
        assert_eq!(refusal(&deep_module(&e, 0)), "K011", "{e}");
    }
    // the body is the first block: 255 more make 256
    assert_eq!(refusal(&deep_module("x", 255)), "accepted");
    assert_eq!(refusal(&deep_module("x", 256)), "K011");
}

#[test]
fn deepest_accepted_text_round_trips() {
    // blocks 256 deep around an expression 256 deep: parsed, written, compared and
    // dropped from a test thread's small stack
    for e in deep_exprs(256) {
        let text = deep_module(&e, 255);
        let module = parse(&text).unwrap_or_else(|d| panic!("{d:?}"));
        assert_round_trip(&module, "the deepest accepted text");
        // a clone is the caller's business: on the crate's deep stack it is safe
        let same = komorebi_ir::deep::on_deep_stack(|| module.clone() == module);
        assert!(same);
    }
}

#[test]
fn far_deeper_texts_are_k011() {
    for n in [1000, 100_000] {
        for e in deep_exprs(n) {
            assert_eq!(refusal(&deep_module(&e, 0)), "K011");
        }
        assert_eq!(refusal(&deep_module("x", n)), "K011");
    }
}

#[test]
fn a_long_flat_list_is_no_depth() {
    let terms: Vec<String> = (0..5000).map(|k| k.to_string()).collect();
    let text = deep_module(&format!("f({})", terms.join(", ")), 0);
    let module = parse(&text).unwrap_or_else(|d| panic!("{d:?}"));
    assert_round_trip(&module, "a call of 5,000 operands");
}

// ---- The generator of random trees ---------------------------------------------------

const NAMES: &[&str] = &[
    "a", "b", "x", "y", "acc", "part", "worker", "group", "rank", "size", "tiles", "with", "to",
    "on", "seed", "M", "Visit", "Job", "go", "state", "call",
];

struct Gen {
    rng: Rng,
}

fn sp() -> Span {
    Span::default()
}

impl Gen {
    fn ident(&mut self) -> Ident {
        Ident {
            name: self.rng.pick(NAMES).to_string(),
            span: sp(),
        }
    }

    fn comment(&mut self) -> Comment {
        Comment {
            text: format!("// note {}", self.rng.below(100)),
            span: sp(),
        }
    }

    fn maybe_comment(&mut self) -> Option<Comment> {
        self.rng.chance(15).then(|| self.comment())
    }

    fn comments(&mut self) -> Vec<Comment> {
        let n = if !self.rng.chance(15) {
            0
        } else {
            1 + self.rng.below(2)
        };
        (0..n).map(|_| self.comment()).collect()
    }

    /// Leading lines for a part: no blank first when `first`, never two blanks in a row.
    fn leads(&mut self, first: bool) -> Vec<Lead> {
        let mut out = Vec::new();
        for _ in 0..self.rng.below(3) {
            if self.rng.chance(50) {
                let blank_ok = !(first && out.is_empty()) && out.last() != Some(&Lead::Blank);
                if blank_ok {
                    out.push(Lead::Blank);
                }
            } else {
                out.push(Lead::Comment(self.comment()));
            }
        }
        out
    }

    fn lines<T>(
        &mut self,
        n: usize,
        keep_first_blank: bool,
        mut part: impl FnMut(&mut Gen, usize) -> T,
    ) -> Lines<T> {
        let open = self.maybe_comment();
        let lines = (0..n)
            .map(|k| {
                let leading = self.leads(k == 0 && !keep_first_blank);
                let node = part(self, k);
                Line {
                    trivia: Trivia {
                        leading,
                        trailing: self.maybe_comment(),
                    },
                    node,
                    span: sp(),
                }
            })
            .collect();
        let mut end = self.leads(n == 0 && !keep_first_blank);
        while end.last() == Some(&Lead::Blank) {
            end.pop();
        }
        Lines { open, lines, end }
    }

    fn list<T>(&mut self, n: usize, mut element: impl FnMut(&mut Gen) -> T) -> List<T> {
        let open = self.maybe_comment();
        let entries = (0..n)
            .map(|_| {
                let leading = self.comments();
                let value = element(self);
                Entry {
                    leading,
                    value,
                    trailing: self.maybe_comment(),
                }
            })
            .collect();
        List {
            open,
            entries,
            end: self.comments(),
            span: sp(),
        }
    }

    fn e(&self, kind: ExprKind) -> Expr {
        Expr { kind, span: sp() }
    }

    fn paren(&self, e: Expr) -> Expr {
        self.e(ExprKind::Paren(Box::new(e)))
    }

    /// An expression that may stand before `.`, `(` or `[`.
    fn postfix_base(&mut self, depth: usize) -> Expr {
        let e = self.expr(depth);
        match e.kind {
            ExprKind::Binary { .. } | ExprKind::Neg(_) => self.paren(e),
            _ => e,
        }
    }

    fn literal(&mut self) -> Expr {
        let kind = match self.rng.below(5) {
            0 => ExprKind::Int(self.rng.pick(&["0", "42", "0xffff", "4096"]).to_string()),
            1 => ExprKind::Float(self.rng.pick(&["1.0", "0.5", "1e-6", "2.5E+3"]).to_string()),
            2 => ExprKind::Str(self.rng.pick(&["kohaku.accel", "AAAB", ""]).to_string()),
            3 => ExprKind::Bool(self.rng.chance(50)),
            _ => ExprKind::Name(self.ident()),
        };
        self.e(kind)
    }

    fn arg(&mut self, depth: usize) -> Arg {
        Arg {
            name: self.rng.chance(40).then(|| self.ident()),
            value: self.expr(depth),
            span: sp(),
        }
    }

    fn expr(&mut self, depth: usize) -> Expr {
        if depth == 0 {
            return self.literal();
        }
        let d = depth - 1;
        match self.rng.below(14) {
            0 => self.literal(),
            1 => {
                let inner = self.expr(d);
                self.paren(inner)
            }
            2 => {
                let n = 2 + self.rng.below(2);
                let list = self.list(n, |g| g.expr(d));
                self.e(ExprKind::Tuple(list))
            }
            3 => {
                let n = self.rng.below(3);
                let list = self.list(n, |g| g.expr(d));
                self.e(ExprKind::List(list))
            }
            4 => {
                let ty = self.rng.chance(50).then(|| self.ident());
                let n = 1 + self.rng.below(3);
                let fields = self.list(n, |g| FieldInit {
                    name: g.ident(),
                    value: g.expr(d),
                    span: sp(),
                });
                self.e(ExprKind::Init { ty, fields })
            }
            5 => {
                let base = Box::new(self.postfix_base(d));
                let name = self.ident();
                self.e(ExprKind::Member { base, name })
            }
            6 | 7 => {
                let callee = Box::new(self.postfix_base(d));
                let n = self.rng.below(4);
                let args = self.list(n, |g| g.arg(d));
                self.e(ExprKind::Call { callee, args })
            }
            8 => {
                let base = Box::new(self.postfix_base(d));
                let n = 1 + self.rng.below(2);
                let index = self.list(n, |g| {
                    if g.rng.chance(50) {
                        IndexItem::At(g.expr(d))
                    } else {
                        IndexItem::Slice {
                            lo: Box::new(g.expr(d)),
                            hi: Box::new(g.expr(d)),
                            step: g.rng.chance(30).then(|| Box::new(g.expr(d))),
                        }
                    }
                });
                self.e(ExprKind::Index { base, index })
            }
            9 => {
                let operand = self.expr(d);
                let operand = match operand.kind {
                    ExprKind::Binary { .. } => self.paren(operand),
                    _ => operand,
                };
                self.e(ExprKind::Neg(Box::new(operand)))
            }
            10 => {
                let name = self.ident();
                self.e(ExprKind::Await(name))
            }
            11 => {
                let name = self.ident();
                self.e(ExprKind::Past(name))
            }
            _ => {
                let op = *self.rng.pick(&BinOp::ALL);
                let lhs = self.expr(d);
                let rhs = self.expr(d);
                let lhs = match &lhs.kind {
                    ExprKind::Binary { op: o, .. } if o.precedence() < op.precedence() => {
                        self.paren(lhs)
                    }
                    _ => lhs,
                };
                let rhs = match &rhs.kind {
                    ExprKind::Binary { op: o, .. } if o.precedence() <= op.precedence() => {
                        self.paren(rhs)
                    }
                    _ => rhs,
                };
                self.e(ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
            }
        }
    }

    fn lvalue(&mut self) -> Expr {
        let name = self.ident();
        let mut e = self.e(ExprKind::Name(name));
        for _ in 0..self.rng.below(3) {
            e = if self.rng.chance(50) {
                let name = self.ident();
                self.e(ExprKind::Member {
                    base: Box::new(e),
                    name,
                })
            } else {
                let index = self.list(1, |g| IndexItem::At(g.expr(1)));
                self.e(ExprKind::Index {
                    base: Box::new(e),
                    index,
                })
            };
        }
        e
    }

    fn attrs(&mut self) -> Vec<Attr> {
        (0..self.rng.below(3))
            .map(|_| {
                let parts = 1 + self.rng.below(2);
                let path = (0..parts).map(|_| self.ident()).collect();
                let args = self.rng.chance(60).then(|| {
                    let n = self.rng.below(3);
                    self.list(n, |g| g.arg(1))
                });
                Attr {
                    path,
                    args,
                    span: sp(),
                }
            })
            .collect()
    }

    fn scalar(&mut self) -> Type {
        Type {
            kind: TypeKind::Scalar(*self.rng.pick(&Scalar::ALL)),
            count: None,
            span: sp(),
        }
    }

    fn ty(&mut self) -> Type {
        let format = self.ident();
        let (kind, countable) = match self.rng.below(6) {
            0 => (TypeKind::Scalar(*self.rng.pick(&Scalar::ALL)), true),
            1 => (TypeKind::Bool, false),
            2 => (TypeKind::Tile { format }, true),
            3 => (TypeKind::Tensor { format }, false),
            4 => (TypeKind::View, false),
            _ => (TypeKind::Named(format), false),
        };
        let count = (countable && self.rng.chance(40)).then(|| Box::new(self.expr(1)));
        Type {
            kind,
            count,
            span: sp(),
        }
    }

    fn params(&mut self) -> List<ParamDecl> {
        let n = self.rng.below(4);
        self.list(n, |g| ParamDecl {
            name: g.ident(),
            ty: g.ty(),
            attrs: g.attrs(),
            span: sp(),
        })
    }

    fn block(&mut self, depth: usize) -> Block {
        let n = self.rng.below(4);
        Block {
            stmts: self.lines(n, false, |g, _| g.stmt(depth)),
            span: sp(),
        }
    }

    fn stmt(&mut self, depth: usize) -> Stmt {
        let nested = depth > 0;
        let choice = self.rng.below(if nested { 13 } else { 10 });
        match choice {
            0 => Stmt::Let {
                name: self.ident(),
                ty: self.rng.chance(40).then(|| self.ty()),
                value: self.expr(3),
                attrs: self.attrs(),
            },
            1 => Stmt::Var {
                name: self.ident(),
                ty: self.ty(),
                value: self.expr(2),
            },
            2 | 3 => Stmt::Assign {
                target: self.lvalue(),
                op: *self
                    .rng
                    .pick(&[AssignOp::Set, AssignOp::Add, AssignOp::Sub]),
                value: self.expr(3),
            },
            4 => Stmt::Store {
                target: self.expr(2),
                value: self.expr(2),
            },
            5 => Stmt::Consume(self.ident()),
            6 => Stmt::Return(self.rng.chance(70).then(|| self.expr(2))),
            7..=9 => Stmt::Expr(self.expr(3)),
            10 => Stmt::For {
                vars: (0..1 + self.rng.below(2)).map(|_| self.ident()).collect(),
                iter: self.expr(2),
                attrs: self.attrs(),
                body: self.block(depth - 1),
            },
            11 => Stmt::While {
                cond: self.expr(2),
                body: self.block(depth - 1),
            },
            _ => Stmt::If {
                cond: self.expr(2),
                then: self.block(depth - 1),
                otherwise: self.rng.chance(50).then(|| self.block(depth - 1)),
            },
        }
    }

    fn int(&mut self) -> IntLit {
        IntLit {
            text: self.rng.pick(&["1", "32", "64", "0x10"]).to_string(),
            span: sp(),
        }
    }

    fn fields(&mut self) -> List<Field> {
        let n = 1 + self.rng.below(4);
        self.list(n, |g| Field {
            name: g.ident(),
            ty: g.ty(),
            span: sp(),
        })
    }

    fn requirement(&mut self) -> Requirement {
        match self.rng.below(5) {
            0 => Requirement::LocalState {
                unit: *self.rng.pick(&[StateUnit::Tiles, StateUnit::Bytes]),
                amount: self.int(),
                span: sp(),
            },
            1 => Requirement::Group {
                shape: *self.rng.pick(&[GroupShape::Line, GroupShape::Grid]),
                span: sp(),
            },
            2 => Requirement::Dynamic { span: sp() },
            3 => Requirement::Islands {
                count: self.int(),
                span: sp(),
            },
            _ => Requirement::Impl {
                name: StrLit {
                    text: "vendor.thing".into(),
                    span: sp(),
                },
                span: sp(),
            },
        }
    }

    fn func(&mut self, attrs: bool) -> Func {
        Func {
            name: self.ident(),
            params: self.params(),
            ret: self.ty(),
            attrs: if attrs { self.attrs() } else { Vec::new() },
            body: self.block(2),
            span: sp(),
        }
    }

    fn program_item(&mut self) -> ProgramItem {
        match self.rng.below(7) {
            0 => ProgramItem::Message(Message {
                name: self.ident(),
                attrs: self.attrs(),
                fields: self.fields(),
                span: sp(),
            }),
            1 => ProgramItem::Record(Record {
                name: self.ident(),
                fields: self.fields(),
                span: sp(),
            }),
            2 => {
                let n = self.rng.below(3);
                ProgramItem::State(State {
                    attrs: self.attrs(),
                    members: self.lines(n, false, |g, _| StateMember {
                        name: g.ident(),
                        ty: g.ty(),
                        attrs: g.attrs(),
                        span: sp(),
                    }),
                    span: sp(),
                })
            }
            3 | 4 => ProgramItem::Handler(Handler {
                name: self.ident(),
                params: self.params(),
                ret: self.rng.chance(40).then(|| self.scalar()),
                attrs: self.attrs(),
                body: self.block(2),
                span: sp(),
            }),
            5 => {
                let n = 1 + self.rng.below(3);
                ProgramItem::Run(Run {
                    name: self.ident(),
                    params: self.params(),
                    attrs: self.attrs(),
                    body: self.lines(n, false, |g, k| {
                        if k == 0 {
                            RunPart::Seed(g.block(1))
                        } else {
                            RunPart::On(OnBlock {
                                message: g.ident(),
                                binding: g.ident(),
                                body: g.block(1),
                                span: sp(),
                            })
                        }
                    }),
                    span: sp(),
                })
            }
            _ => {
                let n = self.rng.below(3);
                ProgramItem::Chain(Chain {
                    name: self.ident(),
                    params: self.params(),
                    attrs: self.attrs(),
                    calls: self.lines(n, false, |g, _| {
                        let k = g.rng.below(4);
                        Call {
                            handler: g.ident(),
                            args: g.list(k, |g| g.arg(1)),
                            span: sp(),
                        }
                    }),
                    span: sp(),
                })
            }
        }
    }

    fn item(&mut self) -> Item {
        match self.rng.below(11) {
            0 => Item::Tile(TileDecl {
                rows: self.int(),
                cols: self.int(),
                span: sp(),
            }),
            1 => Item::Format(FormatDecl {
                formats: (0..1 + self.rng.below(3)).map(|_| self.ident()).collect(),
                span: sp(),
            }),
            2 => {
                let n = self.rng.below(4);
                Item::Requires(Requires {
                    body: self.lines(n, false, |g, _| g.requirement()),
                    span: sp(),
                })
            }
            3 => Item::Param(Param {
                name: self.ident(),
                ty: self.ty(),
                value: if self.rng.chance(30) {
                    ParamValue::Fit(sp())
                } else {
                    ParamValue::Expr(self.expr(2))
                },
                attrs: self.attrs(),
                span: sp(),
            }),
            4 => Item::Message(Message {
                name: self.ident(),
                attrs: self.attrs(),
                fields: self.fields(),
                span: sp(),
            }),
            5 => Item::Func(self.func(false)),
            6 => Item::Combine(self.func(true)),
            7 => Item::Extern(Extern {
                name: self.ident(),
                params: self.params(),
                ret: self.ty(),
                attrs: self.attrs(),
                fallback: self.rng.chance(50).then(|| self.block(1)),
                span: sp(),
            }),
            8 | 9 => {
                let n = self.rng.below(4);
                Item::Program(Program {
                    name: self.ident(),
                    body: self.lines(n, false, |g, _| g.program_item()),
                    span: sp(),
                })
            }
            _ => Item::Control(Control {
                name: self.ident(),
                params: self.params(),
                ret: self.rng.chance(40).then(|| self.scalar()),
                attrs: self.attrs(),
                body: self.block(2),
                span: sp(),
            }),
        }
    }

    fn module(&mut self) -> Module {
        let mut leading = self.leads(true);
        if leading.first() == Some(&Lead::Blank) {
            leading.remove(0);
        }
        let n = self.rng.below(5);
        let mut items = self.lines(n, true, |g, _| g.item());
        items.open = None;
        Module {
            trivia: Trivia {
                leading,
                trailing: self.maybe_comment(),
            },
            name: self.ident(),
            items,
            span: sp(),
        }
    }
}

#[test]
fn generated_trees_round_trip() {
    let count = 4000;
    let mut g = Gen {
        rng: Rng(0x2545_f491_4f6c_dd1d),
    };
    for k in 0..count {
        let module = g.module();
        assert_round_trip(&module, &format!("generated tree {k}"));
    }
}
