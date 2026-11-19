//! The golden test of program compile's checks: every module of `invalid/` gives
//! exactly the diagnostics its `.expect` file lists, and the lessons, the reference's
//! modules and the modules of `run_fails/` check clean.
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
