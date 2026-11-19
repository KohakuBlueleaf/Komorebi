//! The golden test of program compile's checks: every module of `invalid/` gives
//! exactly the diagnostics its `.expect` file lists, and the lessons, the reference's
//! modules, the modules of `run_fails/` and the fragments that are whole modules check
//! clean.
//! The whole front ends on every prefix of every lesson and on every lesson with one
//! byte changed: diagnostics or clean, never a panic and never a hang.
//!
//! The corpus is read from `tests/ir/corpus/` (or the folder `KOMO_IR_CORPUS` names).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use komorebi_ir::check;

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
const WHOLE_FRAGMENTS: [&str; 3] = ["count_program", "forward_program", "tile_functions"];

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

/// The number of texts the front is given for a lesson of `n` bytes: every prefix, then
/// the lesson with one byte changed at every position, three ways.
fn mutation_count(n: usize) -> usize {
    n + 3 * n
}

/// The text number `k` of a lesson: a prefix (`k` below its length), or the lesson with
/// one byte changed (the next character, `{`, `"`: a neighbour, an unclosed block, an
/// unclosed string). The lessons are ASCII, so each is a text.
fn mutation(text: &str, k: usize) -> String {
    let n = text.len();
    if k < n {
        return text[..k].to_string();
    }
    let (at, way) = ((k - n) / 3, (k - n) % 3);
    let mut bytes = text.as_bytes().to_vec();
    bytes[at] = [bytes[at] ^ 0x01, b'{', b'"'][way];
    String::from_utf8(bytes).expect("an ASCII lesson stays ASCII")
}

/// The whole front (`api::check`: parse, check, lower, derive, verify) on every prefix of
/// every lesson and on every lesson with one byte changed, three ways: diagnostics or
/// clean, never a panic, and never a hang (all texts must end within the time below).
/// Twelve threads share the texts; each runs them on one deep stack.
#[test]
fn the_front_never_panics_on_a_cut_or_changed_lesson() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::{Duration, Instant};
    let lessons: Vec<(String, String)> = modules("lessons")
        .into_iter()
        .map(|(path, text)| (stem(&path), text))
        .collect();
    assert!(lessons.iter().all(|(_, text)| text.is_ascii()));
    let mut starts = Vec::new();
    let mut total = 0;
    for (_, text) in &lessons {
        starts.push(total);
        total += mutation_count(text.len());
    }
    let shared = Arc::new((lessons, starts, AtomicUsize::new(0)));
    let (done, finished) = mpsc::channel();
    let mut current: Vec<Arc<Mutex<usize>>> = Vec::new();
    for _ in 0..12 {
        let at = Arc::new(Mutex::new(usize::MAX));
        current.push(Arc::clone(&at));
        let shared = Arc::clone(&shared);
        let done = done.clone();
        std::thread::spawn(move || {
            let (lessons, starts, next) = &*shared;
            let panics = komorebi_ir::deep::on_deep_stack(|| {
                let mut panics = Vec::new();
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= total {
                        break;
                    }
                    if let Ok(mut at) = at.lock() {
                        *at = g;
                    }
                    let l = starts.partition_point(|s| *s <= g) - 1;
                    let (name, text) = &lessons[l];
                    let t = mutation(text, g - starts[l]);
                    if catch_unwind(AssertUnwindSafe(|| assert_inside(&t))).is_err() {
                        panics.push(format!("{name}, text {}:\n{t}", g - starts[l]));
                    }
                }
                panics
            });
            let _ = done.send(panics);
        });
    }
    drop(done);
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut panics = Vec::new();
    for _ in 0..current.len() {
        let left = deadline.saturating_duration_since(Instant::now());
        match finished.recv_timeout(left) {
            Ok(p) => panics.extend(p),
            Err(_) => {
                let (lessons, starts, _) = &*shared;
                let at: Vec<String> = current
                    .iter()
                    .filter_map(|at| at.lock().ok().map(|a| *a))
                    .filter(|g| *g < total)
                    .map(|g| {
                        let l = starts.partition_point(|s| *s <= g) - 1;
                        format!("{} text {}", lessons[l].0, g - starts[l])
                    })
                    .collect();
                panic!("the front did not end in 600 s; still at {at:?}");
            }
        }
    }
    assert!(
        panics.is_empty(),
        "{} panics:\n{}",
        panics.len(),
        panics.join("\n---\n")
    );
}

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
/// rectangle.
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
    assert!(
        missing.is_empty(),
        "values with no type:\n{}",
        missing.join("\n")
    );
}

/// The program of `k701_race_order`, with the control program `CONTROL`.
const ARMS: &str = "module arms

tile [32, 32]
format f32

program P {
    handler write(x: tensor<f32>) @on(group) {
        let part = partition(x.tiles(0, 64), worker.rank, group.size)
        let out = sink(part, block = 1)
        for xb in stream(part, block = 1) {
            out.push(xb * 2.0)
        }
    }

    handler read(x: tensor<f32>) -> u32 @on(group) {
        var blocks: u32 = 0
        for xb in stream(partition(x.tiles(0, 64), worker.rank, group.size), block = 1) {
            blocks += 1
        }
        return blocks
    }
}

CONTROL
";

#[test]
fn two_invocations_on_the_two_arms_of_one_if_are_no_race() {
    let control = "control order(x: tensor<f32>, pick: u32) {
    if pick == 0 {
        let a = invoke(P.write(x), on = group(0, 8))
    } else {
        let b = invoke(P.read(x), on = group(8, 8))
    }
}";
    let text = ARMS.replace("CONTROL", control);
    assert_eq!(found(&text), Vec::<String>::new());
}

#[test]
fn a_race_inside_one_arm_is_refused() {
    let control = "control order(x: tensor<f32>, pick: u32) {
    if pick == 0 {
        let a = invoke(P.write(x), on = group(0, 8))
        let b = invoke(P.read(x), on = group(8, 8))
    }
}";
    let text = ARMS.replace("CONTROL", control);
    assert_eq!(found(&text), vec!["K701 27".to_string()]);
}

/// Every instruction of every body, and every call of every chain, of the modules the
/// corpus lowers has its place in the notes: a finding of Verify is never placed by
/// default.
#[test]
fn every_instruction_and_call_has_its_position() {
    use komorebi_ir::program::notes::Anchor;
    use komorebi_ir::program::BodyRef;
    use komorebi_ir::types::{ChainId, ControlId, FuncId, HandlerId, ProgramId, RunId};
    let mut seen = 0;
    let mut missing = Vec::new();
    for sub in ["lessons", "reference", "run_fails", "fragments"] {
        for (path, text) in modules(sub) {
            let Ok(checked) = komorebi_ir::check::check_module(
                &text,
                komorebi_ir::parse(&text).expect("a corpus module parses"),
            ) else {
                continue;
            };
            let Ok(m) = komorebi_ir::lower::lower(&checked) else {
                continue;
            };
            let anchors: std::collections::HashSet<Anchor> =
                m.notes.spans.iter().map(|(a, _)| *a).collect();
            let mut bodies: Vec<(BodyRef, &komorebi_ir::program::body::Body)> = Vec::new();
            for (f, func) in m.funcs.iter().enumerate() {
                if let Some(b) = &func.body {
                    bodies.push((BodyRef::Func(FuncId(f as u32)), b));
                }
            }
            for (p, prog) in m.programs.iter().enumerate() {
                let pid = ProgramId(p as u32);
                for (h, handler) in prog.handlers.iter().enumerate() {
                    bodies.push((BodyRef::Handler(pid, HandlerId(h as u32)), &handler.body));
                }
                for (r, run) in prog.runs.iter().enumerate() {
                    let rid = RunId(r as u32);
                    bodies.push((BodyRef::Seed(pid, rid), &run.seed));
                    for (k, (_, b)) in run.on_message.iter().enumerate() {
                        bodies.push((BodyRef::On(pid, rid, k as u32), b));
                    }
                }
                for (c, chain) in prog.chains.iter().enumerate() {
                    for k in 0..chain.calls.len() {
                        seen += 1;
                        let anchor = Anchor::Call(pid, ChainId(c as u32), k as u32);
                        if !anchors.contains(&anchor) {
                            missing.push(format!("{}: {anchor:?}", stem(&path)));
                        }
                    }
                }
            }
            for (c, control) in m.controls.iter().enumerate() {
                bodies.push((BodyRef::Control(ControlId(c as u32)), &control.body));
            }
            for (r, body) in bodies {
                for i in 0..body.insts.len() {
                    seen += 1;
                    let anchor = Anchor::Inst(r, komorebi_ir::program::body::InstId(i as u32));
                    if !anchors.contains(&anchor) {
                        missing.push(format!("{}: {anchor:?}", stem(&path)));
                    }
                }
            }
        }
    }
    assert!(seen > 1000, "the walk sees the bodies ({seen})");
    assert!(missing.is_empty(), "no position:\n{}", missing.join("\n"));
}
