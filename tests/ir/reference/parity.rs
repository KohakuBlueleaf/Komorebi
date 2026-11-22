//! Parity: every run script in `tests/ir/parity/` runs on the interpreter (`komo run`'s
//! report) and on the CPU backend's driver, and the two reports are the same by the
//! rules of what is the same between backends: each invocation completed, failed or was
//! invalid on both; a failure's reason too, unless the script says
//! `# parity: any failure` (several reasons possible, and which is reported is the
//! backend's); every tensor read equal element by element inside the tensor, except one
//! a failed invocation was given, whose writes are not compared. Floats are compared
//! exactly: the scripts choose values every backend computes exactly.
//!
//! Without a C++ compiler the test fails and says so; `KOMO_IR_NO_CXX=1` skips it.

use std::path::{Path, PathBuf};

use komorebi_ir::api::script::{self, Line, ScriptArg};
use komorebi_ir::backend::cpu::{build, Cpu};
use komorebi_ir::backend::{Backend, Target};
use komorebi_ir::types::TileShape;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir")
}

/// An invocation's end, as both reports write it.
#[derive(Debug, PartialEq)]
enum End {
    Completed,
    Failed(String),
    Invalid,
}

/// A tensor read: its name, its shape in elements, and each tile's elements.
type Shown = (String, (usize, usize), Vec<Vec<f64>>);

/// A report read back: the ends, and each tensor read.
#[derive(Debug, Default)]
struct Read {
    ends: Vec<(String, End)>,
    tensors: Vec<Shown>,
}

fn value(word: &str) -> f64 {
    match word {
        "NaN" | "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => word
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("`{word}` is not a number")),
    }
}

fn elements(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    for run in text.split(", ") {
        let (v, n) = match run.rsplit_once('*') {
            Some((v, n)) => (v, n.parse::<usize>().expect("a count")),
            None => (run, 1),
        };
        out.extend(std::iter::repeat_n(value(v), n));
    }
    out
}

fn read_report(text: &str) -> Read {
    let mut r = Read::default();
    for line in text.lines() {
        if let Some(tile) = line.strip_prefix("  tile ") {
            let (_, rest) = tile.split_once(": ").expect("tile i: ...");
            let last = r.tensors.last_mut().expect("a tile after its tensor");
            last.2.push(elements(rest));
        } else if let Some(t) = line.strip_prefix("tensor ") {
            let words: Vec<&str> = t.splitn(3, ' ').collect();
            let shape = words[2].trim_matches(['[', ']']);
            let (rows, cols) = shape.split_once(", ").expect("[rows, cols]");
            r.tensors.push((
                words[0].to_string(),
                (rows.parse().expect("rows"), cols.parse().expect("cols")),
                Vec::new(),
            ));
        } else if line.starts_with('#') {
            let (head, end) = line.split_once(": ").expect("#n entry: end");
            let end = if end == "completed" || end.starts_with("completed ") {
                End::Completed
            } else if let Some(f) = end.strip_prefix("failed ") {
                End::Failed(f.split(' ').next().unwrap_or_default().to_string())
            } else {
                End::Invalid
            };
            r.ends.push((head.to_string(), end));
        } else {
            panic!("`{line}` is not a line of a report");
        }
    }
    r
}

/// Whether element `k` of tile `i` lies inside a tensor of `shape`.
fn inside(shape: (usize, usize), tile: TileShape, i: usize, k: usize) -> bool {
    let (tr, tc) = (tile.rows as usize, tile.cols as usize);
    let grid_cols = shape.1.div_ceil(tc);
    let r = (i / grid_cols) * tr + k / tc;
    let c = (i % grid_cols) * tc + k % tc;
    r < shape.0 && c < shape.1
}

fn compare(name: &str, text: &str, tile: TileShape, reference: &Read, cpu: &Read) {
    let any_failure = text.lines().any(|l| l.trim() == "# parity: any failure");
    let lines = script::parse(text).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(reference.ends.len(), cpu.ends.len(), "{name}: the ends");
    // The tensors each invocation was given, in the order of the ends.
    let given: Vec<Vec<&str>> = lines
        .lines
        .iter()
        .filter_map(|l| match l {
            Line::Invoke { args, .. } => Some(
                args.iter()
                    .filter_map(|a| match a {
                        ScriptArg::Tensor(t) => Some(t.as_str()),
                        ScriptArg::Scalar(_) => None,
                    })
                    .collect(),
            ),
            _ => None,
        })
        .collect();
    let mut not_compared: Vec<&str> = Vec::new();
    for (k, ((head, a), (head_b, b))) in reference.ends.iter().zip(&cpu.ends).enumerate() {
        assert_eq!(head, head_b, "{name}: end {k}");
        match (a, b) {
            (End::Failed(_), End::Failed(_)) if any_failure => {}
            _ => assert_eq!(a, b, "{name}: {head}"),
        }
        if *a != End::Completed {
            not_compared.extend(given.get(k).into_iter().flatten());
        }
    }
    // A tensor a failed invocation was given is not compared in any read of the
    // script (the scripts read it only after that invocation).
    assert_eq!(
        reference.tensors.len(),
        cpu.tensors.len(),
        "{name}: the reads"
    );
    for ((t, shape, tiles), (t_b, shape_b, tiles_b)) in reference.tensors.iter().zip(&cpu.tensors) {
        assert_eq!((t, shape), (t_b, shape_b), "{name}");
        assert_eq!(tiles.len(), tiles_b.len(), "{name}: tiles of {t}");
        if not_compared.contains(&t.as_str()) {
            continue;
        }
        for (i, (x, y)) in tiles.iter().zip(tiles_b).enumerate() {
            assert_eq!(x.len(), y.len(), "{name}: tile {i} of {t}");
            for (k, (u, v)) in x.iter().zip(y).enumerate() {
                if inside(*shape, tile, i, k) {
                    assert!(
                        u == v || (u.is_nan() && v.is_nan()),
                        "{name}: {t}, tile {i}, element {k}: the interpreter {u}, cpu {v}"
                    );
                }
            }
        }
    }
}

#[test]
fn every_script_gives_the_same_report_on_the_interpreter_and_the_cpu() {
    if std::env::var("KOMO_IR_NO_CXX").is_ok_and(|v| v == "1") {
        eprintln!("parity skipped: KOMO_IR_NO_CXX=1");
        return;
    }
    let cache = Path::new(env!("CARGO_TARGET_TMPDIR")).join("komo-ir-cache");
    let target = Target {
        arch: std::env::consts::ARCH.to_string(),
        features: Default::default(),
    };
    let dir = root().join("parity");
    let mut scripts: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("tests/ir/parity")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "run"))
        .collect();
    scripts.sort();
    assert!(scripts.len() >= 6, "{scripts:?}");
    for path in scripts {
        let name = path
            .file_name()
            .map_or(String::new(), |n| n.to_string_lossy().into_owned());
        let lesson = name.split('_').take(2).collect::<Vec<_>>().join("_");
        let source =
            std::fs::read_to_string(root().join(format!("corpus/golden/lessons/{lesson}.kom")))
                .unwrap_or_else(|e| panic!("{name}: the lesson {lesson}: {e}"));
        let module = komorebi_ir::api::build(&source).unwrap_or_else(|d| panic!("{d:?}"));
        let text = std::fs::read_to_string(&path).expect("the script");

        let parsed = script::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        let report = script::run(&module, &parsed).unwrap_or_else(|e| panic!("{name}: {e}"));
        let reference = read_report(&report.print());
        assert!(
            !reference.tensors.is_empty(),
            "{name}: the script reads nothing"
        );

        let output = Cpu
            .compile(&module, &target)
            .unwrap_or_else(|r| panic!("{name}: {r:?}"));
        let built = build::build(&output, &cache)
            .unwrap_or_else(|e| panic!("{name}: {e} (KOMO_IR_NO_CXX=1 skips this test)"));
        let run = std::process::Command::new(build::driver(&built))
            .arg(&path)
            .output()
            .expect("the driver runs");
        assert!(
            run.status.success(),
            "{name}: the driver: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let cpu = read_report(&String::from_utf8_lossy(&run.stdout));
        compare(&name, &text, module.default_tile(), &reference, &cpu);
    }
}

/// The comparison is not vacuous: an element, a reason or an end that differs is found,
/// and an element past the tensor's edge or a tensor a failed invocation was given is
/// not compared.
#[test]
fn the_comparison_finds_a_difference() {
    let tile = TileShape { rows: 2, cols: 2 };
    let text = "tensor y bf16 2 3\ninvoke P.h y on 0 1\nread y\n";
    let report = |end: &str, tile0: &str, tile1: &str| {
        read_report(&format!(
            "#0 P.h group(0, 1): {end}\ntensor y bf16 [2, 3]\n  tile 0: {tile0}\n  tile 1: {tile1}\n"
        ))
    };
    let base = report("completed", "1.0*4", "2.0, 9.0, 2.0, 9.0");
    let same = report("completed", "1.0*4", "2.0, 0.0, 2.0, 0.0");
    compare("past the edge", text, tile, &base, &same);
    let run = |a: Read, b: Read, t: &'static str| {
        std::panic::catch_unwind(move || compare("case", t, tile, &a, &b)).is_err()
    };
    let element = report("completed", "1.0, 1.0, 1.0, 1.5", "2.0*4");
    assert!(run(report("completed", "1.0*4", "2.0*4"), element, text));
    let failed = report("failed view_end", "1.0*4", "2.0*4");
    assert!(run(report("completed", "1.0*4", "2.0*4"), failed, text));
    let (a, b) = (
        report("failed view_end", "1.0*4", "2.0*4"),
        report("failed capacity", "5.0*4", "2.0*4"),
    );
    assert!(run(a, b, text), "another reason");
    let (a, b) = (
        report("failed view_end", "1.0*4", "2.0*4"),
        report("failed capacity", "5.0*4", "2.0*4"),
    );
    let any = "# parity: any failure\ntensor y bf16 2 3\ninvoke P.h y on 0 1\nread y\n";
    assert!(!run(a, b, any), "any failure, and y not compared after it");
}
