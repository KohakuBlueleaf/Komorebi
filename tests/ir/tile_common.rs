//! The tile tests' shared helpers: today's scripts run once a test target, the
//! KernelIR read back from today's text, and the texts the tests compare.

use std::path::{Path, PathBuf};
use std::process::Command;

use komorebi_ir::backend::tensix::tile::ir::{
    Attr, Grid, Int, KernelIR, NodeId, Store, TensorParam, View,
};
use komorebi_ir::backend::tensix::tile::unit::{Arg, Insn};
use komorebi_ir::backend::tensix::unit::DstLayout;

/// Today's compiler run by `script` into the folder `name`.
pub fn run_today(script: &str, name: &str) -> Option<PathBuf> {
    if std::env::var("KOMO_IR_NO_PYTHON").is_ok_and(|v| v == "1") {
        eprintln!("the tile identity skipped: KOMO_IR_NO_PYTHON=1");
        return None;
    }
    let folder = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let python = std::env::var("KOMO_IR_PYTHON").unwrap_or_else(|_| "python3".into());
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = Command::new(&python)
        .arg(root.join("tests/ir/tensix/tile").join(script))
        .arg(&folder)
        .env("PYTHONPATH", root.join("src"))
        .output()
        .unwrap_or_else(|e| panic!("{python}: {e} (KOMO_IR_NO_PYTHON=1 skips this test)"));
    assert!(
        out.status.success(),
        "today's tile compiler failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(folder)
}

/// The first line where `a` and `b` differ.
pub fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}:\n  today: {x}\n  rust:  {y}", i + 1);
        }
    }
    format!(
        "{} lines today, {} ported",
        a.lines().count(),
        b.lines().count()
    )
}

/// A word from its line (a field's value a number, else a name).
pub fn row_word(line: &str) -> Insn {
    let (head, note) = line.split_once(" ; ").unwrap_or((line, ""));
    let mut parts = head.split(' ');
    let op = parts.next().expect("an operation");
    let args: Vec<(&str, Arg)> = parts
        .map(|f| {
            let (name, value) = f.split_once('=').expect("a field");
            let value = value
                .parse::<i64>()
                .map_or_else(|_| Arg::Name(value.to_string()), Arg::Num);
            (name, value)
        })
        .collect();
    Insn::new(op, &args, note).unwrap_or_else(|e| panic!("{line}: {e}"))
}

/// Today's trace of the row form's corpus (`tile_rows_form.py`), once for all the tests.
pub fn today_rows() -> Option<PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| {
            run_today(
                "tile_rows_form.py",
                concat!("tile-rows-form-", env!("CARGO_CRATE_NAME")),
            )
        })
        .clone()
}

/// Top-level pieces of `text` split at `, ` outside brackets.
pub fn pieces(text: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (Vec::new(), 0i32, 0);
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'[' | b'(' => depth += 1,
            b']' | b')' => depth -= 1,
            b',' if depth == 0 && bytes.get(i + 1) == Some(&b' ') => {
                out.push(&text[start..i]);
                start = i + 2;
            }
            _ => {}
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// An integer of a view or a grid as today writes it: a number, `unit`, or
/// `tile_rows(x)` / `tiles(x)`.
pub fn int_of(text: &str) -> Int {
    if let Ok(n) = text.parse() {
        return Int::Const(n);
    }
    if text == "unit" {
        return Int::Unit(None);
    }
    let (what, name) = text
        .strip_suffix(')')
        .and_then(|t| t.split_once('('))
        .unwrap_or_else(|| panic!("an integer {text}"));
    Int::Size {
        what: what.into(),
        name: name.into(),
        dim: None,
    }
}

/// A view as today writes it: `x.tiles[i]` or `x.tiles[i, :]`.
pub fn view_of(text: &str) -> View {
    let (tensor, rest) = text.split_once(".tiles[").expect("a view");
    let index = rest.strip_suffix(']').expect("a view's end");
    let (index, kind) = match index.strip_suffix(", :") {
        Some(i) => (i, "row"),
        None => (index, "tile"),
    };
    View {
        tensor: tensor.into(),
        kind: kind.into(),
        index: int_of(index),
        cols: None,
    }
}

/// A KernelIR read back from today's dump (the forms the row form's corpus prints):
/// its nodes made in the dump's order, which `nodes` gives again.
pub fn parse_ir(text: &str) -> KernelIR {
    let mut lines = text.lines();
    let name = lines
        .next()
        .and_then(|l| l.strip_prefix("kernel "))
        .expect("kernel");
    let mut ir = KernelIR::new(name, Vec::new(), Vec::new());
    for line in lines {
        let line = line.trim_start();
        if let Some((side, rest)) = line
            .split_once(' ')
            .filter(|(s, _)| ["in", "out"].contains(s))
        {
            let (tensor, dtype) = rest.trim_start().split_once(": ").expect("a tensor");
            ir.tensors.push(TensorParam {
                name: tensor.into(),
                output: side == "out",
                dtype: Some(dtype.into()),
            });
        } else if let Some(rest) = line.strip_prefix("grid unit < ") {
            let (count, block) = rest.split_once(", block ").expect("a grid");
            ir.grid = Some(Grid {
                count: int_of(count),
                block: block.parse().expect("a block"),
                cols: None,
            });
        } else if let Some(rest) = line.strip_prefix("store ") {
            let (view, value) = rest.split_once(" = %").expect("a store");
            ir.stores.push(Store {
                view: view_of(view),
                value: NodeId(value.parse().expect("a node")),
            });
        } else {
            let (_, made) = line.split_once(" = ").expect("a node");
            let (kind, inner) = made.split_once('(').expect("a node's kind");
            let inner = inner.strip_suffix(')').expect("a node's end");
            let (mut args, mut attrs) = (Vec::new(), Vec::new());
            for piece in pieces(inner) {
                if let Some(n) = piece.strip_prefix('%') {
                    args.push(NodeId(n.parse().expect("an operand")));
                    continue;
                }
                let (k, v) = piece.split_once('=').expect("a field");
                let attr = match (k, v) {
                    (_, "None") => Attr::None,
                    ("view", v) => Attr::View(view_of(v)),
                    ("value", v) => Attr::Float(v.parse().expect("a float")),
                    ("dim", v) => Attr::Int(Int::Const(v.parse().expect("a dim"))),
                    (_, v) => Attr::Text(v.into()),
                };
                attrs.push((k, attr));
            }
            ir.node(kind, args, attrs);
        }
    }
    ir
}

/// A call's Dst layout as today's scripts write it.
pub fn dst_text(layout: &DstLayout) -> String {
    let list = |tiles: &[usize]| {
        let t: Vec<String> = tiles.iter().map(usize::to_string).collect();
        t.join(",")
    };
    let mut lines: Vec<String> = layout
        .params
        .iter()
        .map(|p| format!("param\t{}", list(p)))
        .collect();
    lines.push(format!("result\t{}", list(&layout.result)));
    lines.join("\n") + "\n"
}

/// The number of the function named `name`.
pub fn func(module: &komorebi_ir::program::Module, name: &str) -> komorebi_ir::types::FuncId {
    let at = module.funcs.iter().position(|f| f.name == name);
    komorebi_ir::types::FuncId(at.unwrap_or_else(|| panic!("no func {name}")) as u32)
}
