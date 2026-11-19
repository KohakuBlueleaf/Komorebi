//! The golden test of program compile's checks: every module of `invalid/` gives
//! exactly the diagnostics its `.expect` file lists, and the lessons, the reference's
//! modules, the modules of `run_fails/` and the fragments that are whole modules check
//! clean.
//!
//! The corpus is read from `tests/ir/corpus/` (or the folder `KOMO_IR_CORPUS` names).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use komorebi_ir::check;

/// The cases whose rule needs the effects a handler derives (what it reads and
/// writes), which a later step of the build gives. Until then they check clean.
const WAITING_FOR_DERIVE: [&str; 3] = ["k601_past_unproven", "k701_race_halves", "k701_race_order"];

fn corpus_dir() -> PathBuf {
    match std::env::var_os("KOMO_IR_CORPUS") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/corpus"),
    }
}

/// The `.kom` files of one folder of the golden corpus, in order.
fn modules(sub: &str) -> Vec<(PathBuf, String)> {
    let dir = corpus_dir().join("golden").join(sub);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        panic!("the corpus folder {} is readable", dir.display());
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "kom"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path).expect("a corpus file is readable");
            (path, text)
        })
        .collect()
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// The diagnostics of a text as `.expect` lines, `CODE LINE`, in order.
fn found(text: &str) -> Vec<String> {
    match check(text) {
        Ok(_) => Vec::new(),
        Err(diagnostics) => {
            let mut lines: Vec<String> = diagnostics
                .iter()
                .map(|d| format!("{} {}", d.code, d.start.line))
                .collect();
            lines.sort();
            lines
        }
    }
}

/// The lines of the `.expect` file beside a module, in order.
fn expected(path: &Path) -> Vec<String> {
    let expect = path.with_extension("expect");
    let text = std::fs::read_to_string(&expect)
        .unwrap_or_else(|_| panic!("{} is readable", expect.display()));
    let mut lines: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
    lines.sort();
    lines
}

fn assert_clean(sub: &str, at_least: usize) {
    let modules = modules(sub);
    assert!(
        modules.len() >= at_least,
        "{sub} holds at least {at_least} modules"
    );
    let mut faults = Vec::new();
    for (path, text) in &modules {
        if let Err(diagnostics) = check(text) {
            for d in diagnostics {
                faults.push(d.render(&path.display().to_string()));
            }
        }
    }
    assert!(
        faults.is_empty(),
        "{sub} must check clean:\n{}",
        faults.join("\n")
    );
}

#[test]
fn lessons_check_clean() {
    assert_clean("lessons", 15);
}

#[test]
fn reference_checks_clean() {
    assert_clean("reference", 1);
}

#[test]
fn run_fails_check_clean() {
    assert_clean("run_fails", 2);
}

/// The fragments that are whole modules with their header (`tile`, `format`,
/// `requires`). The other fragments name what the specification declares around them,
/// and are only read and written back.
const WHOLE_FRAGMENTS: [&str; 4] = [
    "count_program",
    "forward_program",
    "sweep_halves",
    "tile_functions",
];

#[test]
fn whole_fragments_check_clean() {
    let fragments = modules("fragments");
    let mut faults = Vec::new();
    for name in WHOLE_FRAGMENTS {
        let Some((path, text)) = fragments.iter().find(|(p, _)| stem(p) == name) else {
            panic!("the fragment {name} is in the corpus");
        };
        if let Err(diagnostics) = check(text) {
            for d in diagnostics {
                faults.push(d.render(&path.display().to_string()));
            }
        }
    }
    assert!(
        faults.is_empty(),
        "these fragments must check clean:\n{}",
        faults.join("\n")
    );
}

#[test]
fn invalid_cases_give_their_diagnostics() {
    let modules = modules("invalid");
    assert!(modules.len() >= 58, "invalid holds the catalogue's cases");
    let mut faults = Vec::new();
    for (path, text) in &modules {
        if WAITING_FOR_DERIVE.contains(&stem(path).as_str()) {
            continue;
        }
        let (want, got) = (expected(path), found(text));
        if want != got {
            faults.push(format!(
                "{}: expected {want:?}, found {got:?}",
                path.display()
            ));
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

/// A case waiting for the derived effects is refused for its one reason only: until
/// the effects exist, it gives no diagnostic at all.
#[test]
fn cases_waiting_for_derive_give_nothing_else() {
    for name in WAITING_FOR_DERIVE {
        let path = corpus_dir()
            .join("golden/invalid")
            .join(format!("{name}.kom"));
        let text = std::fs::read_to_string(&path).expect("the case is readable");
        assert_eq!(found(&text), Vec::<String>::new(), "{}", path.display());
    }
}

#[test]
#[ignore = "waits for derive/: the races and `past` need the handlers' derived effects"]
fn cases_waiting_for_derive_give_their_diagnostics() {
    for name in WAITING_FOR_DERIVE {
        let path = corpus_dir()
            .join("golden/invalid")
            .join(format!("{name}.kom"));
        let text = std::fs::read_to_string(&path).expect("the case is readable");
        assert_eq!(found(&text), expected(&path), "{}", path.display());
    }
}

/// Every diagnostic lies inside the text.
fn assert_inside(text: &str) {
    let result = catch_unwind(AssertUnwindSafe(|| check(text)));
    let Ok(result) = result else {
        panic!("the checker panicked on:\n{text}");
    };
    if let Err(diagnostics) = result {
        assert!(!diagnostics.is_empty());
        let lines = text.split('\n').count();
        for d in &diagnostics {
            assert!(d.start.line >= 1 && d.start.line <= lines, "{d:?}");
            assert!(d.span.end <= text.len(), "{d:?}");
        }
    }
}

/// A small generator of pseudo-random numbers (xorshift), so the test needs no crate.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// Words of the language, swapped for one another so the mangled modules still parse
/// and reach the checks.
const WORDS: [&str; 24] = [
    "x",
    "y",
    "tiles",
    "first",
    "acc",
    "xb",
    "worker",
    "group",
    "rank",
    "size",
    "stream",
    "sink",
    "partition",
    "reduce",
    "send",
    "consume",
    "forward",
    "invoke",
    "await",
    "store",
    "zeros",
    "m",
    "row",
    "block",
];

#[test]
fn mangled_modules_never_panic() {
    let mut texts: Vec<String> = Vec::new();
    for sub in ["lessons", "invalid", "run_fails"] {
        texts.extend(modules(sub).into_iter().map(|(_, text)| text));
    }
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let total = 4000;
    for k in 0..total {
        let text = &texts[k % texts.len()];
        let mut words: Vec<String> = text.split(' ').map(String::from).collect();
        for _ in 0..1 + rng.below(3) {
            let at = rng.below(words.len());
            let word = &words[at];
            let start = word.find(|c: char| c.is_ascii_alphabetic());
            if let Some(start) = start {
                let end = word[start..]
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .map_or(word.len(), |e| start + e);
                let swapped = WORDS[rng.below(WORDS.len())];
                words[at] = format!("{}{}{}", &word[..start], swapped, &word[end..]);
            }
        }
        assert_inside(&words.join(" "));
    }
}

#[test]
fn deep_expressions_are_checked() {
    for n in [64, 256] {
        let expr = format!("{}1{}", "(".repeat(n - 1), ")".repeat(n - 1));
        let text = format!("module m\n\ncontrol c() -> u32 {{\n    return {expr}\n}}\n");
        assert_eq!(found(&text), Vec::<String>::new());
        let sum = format!("1{}", " + 1".repeat(n - 1));
        let text = format!("module m\n\ncontrol c() -> u32 {{\n    return {sum}\n}}\n");
        assert_eq!(found(&text), Vec::<String>::new());
    }
}

/// Every expression of the bodies of a module, in text order.
mod walk {
    use komorebi_ir::syntax::ast::{
        Block, Expr, ExprKind, IndexItem, Item, Module, ProgramItem, RunPart, Stmt,
    };

    pub fn module(m: &Module, out: &mut Vec<Expr>) {
        for item in m.items.nodes() {
            match item {
                Item::Func(f) | Item::Combine(f) => block(&f.body, out),
                Item::Extern(f) => {
                    if let Some(b) = &f.fallback {
                        block(b, out);
                    }
                }
                Item::Control(c) => block(&c.body, out),
                Item::Program(p) => {
                    for part in p.body.nodes() {
                        match part {
                            ProgramItem::Handler(h) => block(&h.body, out),
                            ProgramItem::Run(r) => {
                                for part in r.body.nodes() {
                                    match part {
                                        RunPart::Seed(b) => block(b, out),
                                        RunPart::On(on) => block(&on.body, out),
                                    }
                                }
                            }
                            ProgramItem::Chain(c) => {
                                for call in c.calls.nodes() {
                                    for arg in call.args.values() {
                                        expr(&arg.value, out);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// The places of what a body assigns (`x = e`, `s[i] += e`) and of the expressions
    /// written as statements, whose value is not used.
    pub fn targets(m: &Module, out: &mut std::collections::HashSet<(usize, usize)>) {
        let mut exprs = Vec::new();
        module(m, &mut exprs);
        let mut blocks = Vec::new();
        for item in m.items.nodes() {
            match item {
                Item::Func(f) | Item::Combine(f) => blocks.push(&f.body),
                Item::Extern(f) => blocks.extend(f.fallback.as_ref()),
                Item::Control(c) => blocks.push(&c.body),
                Item::Program(p) => {
                    for part in p.body.nodes() {
                        match part {
                            ProgramItem::Handler(h) => blocks.push(&h.body),
                            ProgramItem::Run(r) => {
                                for part in r.body.nodes() {
                                    match part {
                                        RunPart::Seed(b) => blocks.push(b),
                                        RunPart::On(on) => blocks.push(&on.body),
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        while let Some(b) = blocks.pop() {
            for s in b.stmts.nodes() {
                match s {
                    Stmt::Assign { target, .. } | Stmt::Expr(target) => {
                        out.insert((target.span.start, target.span.end));
                    }
                    Stmt::For { body, .. } | Stmt::While { body, .. } => blocks.push(body),
                    Stmt::If {
                        then, otherwise, ..
                    } => {
                        blocks.push(then);
                        blocks.extend(otherwise.as_ref());
                    }
                    _ => {}
                }
            }
        }
    }

    fn block(b: &Block, out: &mut Vec<Expr>) {
        for s in b.stmts.nodes() {
            match s {
                Stmt::Let { value, .. } | Stmt::Var { value, .. } => expr(value, out),
                Stmt::Assign { target, value, .. } | Stmt::Store { target, value } => {
                    expr(target, out);
                    expr(value, out);
                }
                Stmt::For { iter, body, .. } => {
                    expr(iter, out);
                    block(body, out);
                }
                Stmt::While { cond, body } => {
                    expr(cond, out);
                    block(body, out);
                }
                Stmt::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    expr(cond, out);
                    block(then, out);
                    if let Some(b) = otherwise {
                        block(b, out);
                    }
                }
                Stmt::Return(Some(e)) | Stmt::Expr(e) => expr(e, out),
                Stmt::Return(None) | Stmt::Consume(_) => {}
            }
        }
    }

    fn expr(e: &Expr, out: &mut Vec<Expr>) {
        out.push(e.clone());
        match &e.kind {
            ExprKind::Paren(inner) | ExprKind::Neg(inner) => expr(inner, out),
            ExprKind::Tuple(items) | ExprKind::List(items) => {
                for item in items.values() {
                    expr(item, out);
                }
            }
            ExprKind::Init { fields, .. } => {
                for f in fields.values() {
                    expr(&f.value, out);
                }
            }
            ExprKind::Member { base, .. } => expr(base, out),
            ExprKind::Call { callee, args } => {
                expr(callee, out);
                for arg in args.values() {
                    expr(&arg.value, out);
                }
            }
            ExprKind::Index { base, index } => {
                expr(base, out);
                for item in index.values() {
                    match item {
                        IndexItem::At(at) => expr(at, out),
                        IndexItem::Slice { lo, hi, step } => {
                            expr(lo, out);
                            expr(hi, out);
                            if let Some(step) = step {
                                expr(step, out);
                            }
                        }
                    }
                }
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                expr(lhs, out);
                expr(rhs, out);
            }
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Bool(_)
            | ExprKind::Name(_)
            | ExprKind::Await(_)
            | ExprKind::Past(_) => {}
        }
    }
}

/// The names of the bodies of the clean modules that the result leaves unresolved.
#[test]
fn every_name_of_a_clean_module_is_resolved() {
    let mut missing = Vec::new();
    let mut seen = 0;
    for sub in ["lessons", "reference", "run_fails"] {
        for (path, text) in modules(sub) {
            let Ok(checked) = check(&text) else {
                continue;
            };
            let mut exprs = Vec::new();
            walk::module(&checked.module, &mut exprs);
            for e in &exprs {
                let name = match &e.kind {
                    komorebi_ir::syntax::ast::ExprKind::Name(n)
                    | komorebi_ir::syntax::ast::ExprKind::Await(n)
                    | komorebi_ir::syntax::ast::ExprKind::Past(n) => n,
                    _ => continue,
                };
                seen += 1;
                if checked.binding(name.span).is_none() {
                    missing.push(format!(
                        "{} `{}` at byte {}",
                        stem(&path),
                        name.name,
                        name.span.start
                    ));
                }
            }
        }
    }
    assert!(seen > 500, "the walk sees the bodies ({seen} names)");
    assert!(
        missing.is_empty(),
        "unresolved names:\n{}",
        missing.join("\n")
    );
}

/// The calls, members, indexes, operators, messages and `await`s of the bodies of the
/// clean modules that the result gives no operation.
#[test]
fn every_operation_of_a_clean_module_is_resolved() {
    use komorebi_ir::syntax::ast::ExprKind;
    let mut missing = Vec::new();
    let mut seen = 0;
    for sub in ["lessons", "reference", "run_fails"] {
        for (path, text) in modules(sub) {
            let Ok(checked) = check(&text) else {
                continue;
            };
            let mut exprs = Vec::new();
            walk::module(&checked.module, &mut exprs);
            // a method's callee, the `P.h(...)` an `invoke` names, and the `t.tiles` of
            // a rectangle are parts of the operation around them
            let mut parts = std::collections::HashSet::new();
            for e in &exprs {
                if let ExprKind::Index { base, .. } = &e.kind {
                    if matches!(&base.kind, ExprKind::Member { name, .. } if name.name == "tiles") {
                        parts.insert((base.span.start, base.span.end));
                    }
                }
                if let ExprKind::Call { callee, args } = &e.kind {
                    parts.insert((callee.span.start, callee.span.end));
                    if matches!(&callee.kind, ExprKind::Name(n) if n.name == "invoke") {
                        if let Some(call) = args.values().next() {
                            parts.insert((call.value.span.start, call.value.span.end));
                        }
                    }
                }
            }
            for e in &exprs {
                if parts.contains(&(e.span.start, e.span.end)) {
                    continue;
                }
                let operation = match &e.kind {
                    ExprKind::Call { .. }
                    | ExprKind::Member { .. }
                    | ExprKind::Index { .. }
                    | ExprKind::Binary { .. }
                    | ExprKind::Await(_)
                    | ExprKind::Init { ty: Some(_), .. } => true,
                    ExprKind::Neg(inner) => {
                        !matches!(inner.kind, ExprKind::Int(_) | ExprKind::Float(_))
                    }
                    _ => false,
                };
                if !operation {
                    continue;
                }
                seen += 1;
                if let Some(r) = checked.operation(e) {
                    let given = r.parts.iter().flatten().count();
                    let found = r.operands(e).iter().flatten().count();
                    if given != found {
                        missing.push(format!(
                            "{} `{}`: {found} of its {given} operands found",
                            stem(&path),
                            &text[e.span.start..e.span.end],
                        ));
                    }
                }
                if checked.operation(e).is_none() {
                    missing.push(format!(
                        "{} `{}` at byte {}",
                        stem(&path),
                        &text[e.span.start..e.span.end],
                        e.span.start
                    ));
                }
            }
        }
    }
    assert!(seen > 300, "the walk sees the bodies ({seen} operations)");
    assert!(
        missing.is_empty(),
        "unresolved operations:\n{}",
        missing.join("\n")
    );
}

/// The value expressions of the bodies of the clean modules that the result gives no
/// type. What is no value has none: a name of a function, a program, a type, a word,
/// or a whole state member; what is assigned; an expression written as a statement; a
/// callee; a list; what a `for` walks; the comparison of a `where`; the `t.tiles` of a
/// rectangle. One value is open: the block a broadcast delivers, whose format the text
/// does not state (`for xb, tag in b.arrive()`).
#[test]
fn every_value_of_a_clean_module_has_its_type() {
    use komorebi_ir::check::op::OpKind;
    use komorebi_ir::check::Name;
    use komorebi_ir::syntax::ast::ExprKind;
    let mut missing = Vec::new();
    let mut seen = 0;
    for sub in ["lessons", "reference", "run_fails"] {
        for (path, text) in modules(sub) {
            let Ok(checked) = check(&text) else {
                continue;
            };
            let mut exprs = Vec::new();
            walk::module(&checked.module, &mut exprs);
            // a callee, the `P.h(...)` of an `invoke`, the comparison of a `where` and the
            // `t.tiles` of a rectangle are parts of what is around them; what is assigned
            // and a statement's expression are no value
            let mut parts = std::collections::HashSet::new();
            walk::targets(&checked.module, &mut parts);
            for e in &exprs {
                if let Some(r) = checked.operation(e) {
                    if matches!(r.kind, OpKind::Select | OpKind::Invoke(_)) {
                        let first = match r.kind {
                            OpKind::Select => r.operands(e).first().copied().flatten(),
                            _ => match &e.kind {
                                ExprKind::Call { args, .. } => {
                                    args.values().next().map(|a| &a.value)
                                }
                                _ => None,
                            },
                        };
                        if let Some(x) = first {
                            parts.insert((x.span.start, x.span.end));
                        }
                    }
                }
                match &e.kind {
                    ExprKind::Call { callee, .. } => {
                        parts.insert((callee.span.start, callee.span.end));
                    }
                    ExprKind::Index { base, .. } => {
                        if matches!(&base.kind, ExprKind::Member { name, .. } if name.name == "tiles")
                        {
                            parts.insert((base.span.start, base.span.end));
                        }
                    }
                    _ => {}
                }
            }
            for e in &exprs {
                if parts.contains(&(e.span.start, e.span.end)) {
                    continue;
                }
                let no_value = match &e.kind {
                    ExprKind::Name(n) => matches!(
                        checked.binding(n.span),
                        Some(
                            Name::Word(_)
                                | Name::Func(_)
                                | Name::Program(_)
                                | Name::Record(_)
                                | Name::Message(_)
                                | Name::State { .. }
                        )
                    ),
                    ExprKind::List(_) | ExprKind::Tuple(_) => true,
                    _ => matches!(
                        checked.operation(e).map(|r| &r.kind),
                        Some(OpKind::ForRange | OpKind::ForZip | OpKind::ForArrive)
                    ),
                };
                if no_value {
                    continue;
                }
                seen += 1;
                if checked.type_of(e).is_none() {
                    missing.push(format!(
                        "{} `{}` at byte {}",
                        stem(&path),
                        &text[e.span.start..e.span.end],
                        e.span.start
                    ));
                }
            }
        }
    }
    assert!(seen > 800, "the walk sees the bodies ({seen} values)");
    let open = ["t06_rows `xb` at byte 586"];
    assert_eq!(missing, open, "values with no type");
}
