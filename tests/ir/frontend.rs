//! The DSL as a frontend, end to end with no card: the class form of the first lesson
//! (`tensix/t1_class.py`) made by today's emitter, and by the frontend (`komorebi.dsl.ir`)
//! then the IR and the Tensix backend; the module equal to the hand-written one
//! (`tensix/t1_class.kom`, `tensix/t1_places.txt`), and each file byte for byte.
//!
//! The Python side (`frontend/made.py`) runs under `python3` (or `KOMO_IR_PYTHON`) with
//! the repository's `src` on its path. Without it the test fails and says so;
//! `KOMO_IR_NO_PYTHON=1` skips it by name.

use std::path::{Path, PathBuf};
use std::process::Command;

use komorebi_ir::api::origin::{Given, Mapped};
use komorebi_ir::api::{self, origin};
use komorebi_ir::backend::tensix;
use komorebi_ir::backend::tensix::load::{thread_header, LoadTensor};
use komorebi_ir::bytecode;
use komorebi_ir::bytecode::hash::{hex, sha256};
use komorebi_ir::program::body::InstId;
use komorebi_ir::program::notes::{Anchor, Owner};
use komorebi_ir::program::BodyRef;
use komorebi_ir::types::{HandlerId, ProgramId};

fn here() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir")
}

/// What `made.py` printed: today's build folder, and the frontend's text and origin.
struct Made {
    folder: PathBuf,
    text: String,
    given: Given,
}

fn numbers(rest: &str) -> Vec<u32> {
    rest.split(' ')
        .map(|n| n.parse().expect("a number"))
        .collect()
}

fn made(dir: &Path) -> Made {
    let python = std::env::var("KOMO_IR_PYTHON").unwrap_or_else(|_| "python3".into());
    let out = Command::new(&python)
        .arg(here().join("frontend/made.py"))
        .current_dir(dir)
        .env(
            "PYTHONPATH",
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        )
        .output()
        .unwrap_or_else(|e| panic!("{python}: {e} (KOMO_IR_NO_PYTHON=1 skips this test)"));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "made.py failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (head, text) = stdout.split_once("TEXT\n").expect("a TEXT line");
    let mut folder = None;
    let mut given = Given::default();
    for line in head.lines() {
        let (kind, rest) = line.split_once(' ').expect("a line of made.py");
        match kind {
            "FOLDER" => folder = Some(PathBuf::from(rest)),
            "FILE" => given.files.push(rest.to_string()),
            "LINE" => {
                let mut parts = rest.splitn(3, ' ');
                let mut number = || parts.next().expect("a field").parse().expect("a number");
                let (file, n) = (number(), number());
                let text = rest.splitn(3, ' ').nth(2).unwrap_or("").to_string();
                given.lines.push((file, n, text));
            }
            "MAP" => {
                let n = numbers(rest);
                given.map.push(Mapped {
                    text_first: n[0],
                    text_last: n[1],
                    file: n[2],
                    first: n[3],
                    last: n[4],
                });
            }
            "RENAME" => {
                let (safe, real) = rest.split_once(' ').expect("two names");
                given.renames.push((safe.to_string(), real.to_string()));
            }
            _ => panic!("a line of made.py: {line}"),
        }
    }
    Made {
        folder: folder.expect("a FOLDER line"),
        text: text.to_string(),
        given,
    }
}

/// The hand-written place table: the handler, its parameters and its instructions.
fn table() -> Vec<(Anchor, u32, u32, u32)> {
    let text = std::fs::read_to_string(here().join("tensix/t1_places.txt")).expect("the table");
    let rows = text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    rows.map(|line| {
        let (kind, rest) = line.split_once(' ').expect("a place line");
        let n = numbers(rest);
        let (p, h) = (ProgramId(n[0]), HandlerId(n[1]));
        let anchor = match kind {
            "handler" => Anchor::Handler(p, h),
            "param" => Anchor::ParamOf(Owner::Handler(p, h), n[2]),
            _ => Anchor::Inst(BodyRef::Handler(p, h), InstId(n[2])),
        };
        (anchor, 0, n[n.len() - 2], n[n.len() - 1])
    })
    .collect()
}

/// The first line where `a` and `b` differ, for the message.
fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}:\n  today:    {x}\n  frontend: {y}", i + 1);
        }
    }
    format!(
        "{} lines today, {} by the frontend",
        a.lines().count(),
        b.lines().count()
    )
}

#[test]
fn the_class_through_the_frontend_is_todays_text_byte_for_byte() {
    if std::env::var("KOMO_IR_NO_PYTHON").is_ok_and(|v| v == "1") {
        eprintln!("the frontend's identity skipped: KOMO_IR_NO_PYTHON=1");
        return;
    }
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("frontend-t1");
    std::fs::create_dir_all(&dir).expect("a folder for the script");
    std::fs::copy(here().join("tensix/t1_class.py"), dir.join("t1_class.py"))
        .expect("the script copied");
    let made = made(&dir);

    // The module: the hand-written one's program and hints, its places the table's.
    let module = origin::build(&made.text, &made.given).unwrap_or_else(|r| panic!("{r:?}"));
    let hand = std::fs::read_to_string(here().join("tensix/t1_class.kom")).expect("the .kom");
    let (ours, theirs) = (
        bytecode::hashes(&module),
        bytecode::hashes(&api::build(&hand).expect("the hand module")),
    );
    assert_eq!(ours.semantic, theirs.semantic, "the program");
    assert_eq!(ours.build, theirs.build, "the program and its hints");
    let places = &module.notes.origin.as_ref().expect("an origin").places;
    let handler: Vec<_> = places
        .iter()
        .filter(|(a, ..)| !matches!(a, Anchor::Func(..) | Anchor::Inst(BodyRef::Func(..), _)))
        .filter(|(a, ..)| !matches!(a, Anchor::ParamOf(Owner::Func(..), _)))
        .cloned()
        .collect();
    assert_eq!(handler, table(), "the handler's places");

    // The files: the Tensix backend's, and the thread header at load from the stand-in
    // tensors, against today's folder less its report.
    let mut files = tensix::sources(&module).unwrap_or_else(|r| panic!("{r:?}"));
    let stand_in = LoadTensor {
        pages: 2048,
        layout: vec![2, 2048],
        dram_page: None,
    };
    let header =
        thread_header(&module, &[stand_in.clone(), stand_in]).unwrap_or_else(|r| panic!("{r:?}"));
    files.push(("scale_thread.h".to_string(), header));
    files.sort();
    let mut today: Vec<String> = std::fs::read_dir(&made.folder)
        .expect("the build folder")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n != "program.txt")
        .collect();
    today.sort();
    let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, today, "the files");
    for (name, text) in &files {
        let want = std::fs::read_to_string(made.folder.join(name)).expect("today's file");
        assert!(
            *text == want,
            "{name} differs, {}",
            first_difference(&want, text)
        );
    }
    // The same files give the same folder (dsl/files.py, text_key).
    let mut digest = Vec::new();
    for (name, text) in &files {
        digest.extend_from_slice(format!("{name}\0{}\0", text.len()).as_bytes());
        digest.extend_from_slice(text.as_bytes());
    }
    let key = &hex(&sha256(&digest))[..16];
    assert_eq!(
        made.folder
            .file_name()
            .map(|n| n.to_string_lossy().into_owned()),
        Some(format!("scale-{key}"))
    );
}
