//! The row form (today's `komorebi.dsl.tile.forms.rows`): a row of `width` whole tiles
//! in Dst, one exact reduction along the row, a per-row value computed from it, and the
//! stored row computed from the row, that value broadcast with `.col()` and a resident
//! row broadcast with `.row()`. Layout rule: a per-row value is lanes of one tile.
//! `row_program` is ported; its levels (`level_program`), the chunk form and the band
//! are refused by name.

use std::collections::{BTreeSet, HashMap};

use super::chain::{Chain, ChainError, Compiler, Source};
use super::checks::{check_dst_window, hazard_checks_on, STALL_SFPU, TILE, WAIT_MATH};
use super::ir::{Attr, Int, KernelIR, Node, NodeId};
use super::lower::{
    allocate, call, fold_negation, fuse_mad, hoist, insn, items, maximum, minimum, pass,
    plan_error, sync, value, Body, Reads, NOT_CHECKED, TILE_DTYPES,
};
use super::schedule::interleave;
use super::unit::{
    Arg, Check, Insn, Item, UnitError, UnitProgram, MATH, PACK, PROGRAMMABLE, UNPACK,
};
use super::vector::{FnResult, VId, Vs};

/// The row groups, 4 rows each: (h, q), rows 16 h + 4 q .. + 3.
const GROUPS: [(i64, i64); 8] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
];
/// SFPSHFT2's mode: rotate lanes right by one inside each group of 8.
const ROTATE: i64 = 3;
/// The reductions along the row.
const REDUCTIONS: [&str; 3] = ["sum", "max", "min"];
/// fp32 tiles in all of Dst: the row form takes it whole.
pub const DST_TILES: i64 = 8;

/// Where a value's vectors live in Dst. `Tile`: vector (t, h, c, q, p) of tile t, rows
/// 16 h + 4 q .. + 3, columns 16 c + p + 2 k; `Rows`: per-row slot (h, q), lane l
/// holding row 16 h + 4 q + l / 8 in all 8 lanes of its lane row; `Cols`: per-column
/// slot (t, c, p), lane l holding column 32 t + 16 c + 2 (l % 8) + p in every row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// A row's tiles from Dst tile `0`.
    Tile(i64),
    /// The per-row slots in Dst tile `0`, at `1` in the tile.
    Rows(i64, i64),
    /// The per-column vectors in Dst tile `0`.
    Cols(i64),
}

impl Layout {
    /// The Dst address of the vector a tile vector (t, h, c, q, p) meets.
    pub fn address(&self, (t, h, c, q, p): (i64, i64, i64, i64, i64)) -> i64 {
        match *self {
            Layout::Tile(tile) => TILE * tile + TILE * t + 32 * h + 16 * c + 4 * q + 2 * p,
            Layout::Rows(tile, offset) => TILE * tile + 4 * (4 * h + q) + offset,
            Layout::Cols(tile) => TILE * tile + 16 * c + 8 * t + 2 * p,
        }
    }
}

/// The three layouts of a row: the row's (`copy`), the per-row slots (`bcast_col`)
/// and the per-column vectors (`bcast_row`).
#[derive(Clone, Copy, Debug)]
pub struct Layouts {
    /// The row's.
    pub copy: Layout,
    /// The per-row slots'.
    pub rows: Layout,
    /// The per-column vectors'.
    pub cols: Layout,
}

impl Layouts {
    /// Each layout's address of a vector, in today's order of the layouts.
    fn all(&self, at: (i64, i64, i64, i64, i64)) -> [i64; 3] {
        [
            self.copy.address(at),
            self.rows.address(at),
            self.cols.address(at),
        ]
    }
}

/// A node's field `name`.
fn attr<'a>(n: &'a Node, name: &str) -> Option<&'a Attr> {
    n.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

/// The view kind of a load or a resident row.
fn view_kind(n: &Node) -> Option<&str> {
    match attr(n, "view") {
        Some(Attr::View(v)) => Some(v.kind.as_str()),
        _ => None,
    }
}

/// The node `id` of `ir`.
fn node(ir: &KernelIR, id: NodeId) -> Result<&Node, ChainError> {
    ir.arena
        .get(id.0)
        .ok_or_else(|| ChainError::NotPorted("a node of no kernel".into()))
}

/// The form of every node `ir`'s stores read (today's `_Shapes`): `tile`, `row`, `col`
/// (one value per row), `bcol` (a per-row value broadcast) or `scalar`, or today's
/// refusal; in the order of `nodes`, each operand's before its user's.
pub fn row_shapes(ir: &KernelIR) -> Result<Vec<Option<&'static str>>, ChainError> {
    let mut of: Vec<Option<&'static str>> = vec![None; ir.arena.len()];
    for id in ir.nodes() {
        let n = node(ir, id)?;
        let made = row_shape(n, &n.args.iter().map(|a| of[a.0]).collect::<Vec<_>>(), ir)?;
        of[id.0] = Some(made);
    }
    Ok(of)
}

/// The form of node `n` whose operands' forms are `args`.
fn row_shape(
    n: &Node,
    args: &[Option<&'static str>],
    ir: &KernelIR,
) -> Result<&'static str, ChainError> {
    let k = n.kind.as_str();
    let plan = |message: &str| ChainError::Plan(message.into(), n.line.clone());
    match k {
        "load" | "resident" => {
            return Ok(if view_kind(n) == Some("row") {
                "row"
            } else {
                "tile"
            })
        }
        "const" | "param" | "size" => return Ok("scalar"),
        "slot" | "word" | "state" | "slice" | "cat" => {
            return Err(ChainError::NotPorted(format!("the row form's {k}")))
        }
        _ => {}
    }
    let args: Vec<&str> = args.iter().map(|a| a.unwrap_or("?")).collect();
    let first = args.first().copied().unwrap_or("?");
    if k == "reduce" {
        if first != "row" {
            return Err(plan("tile.reduce_sum reduces a row: load x.tiles[r, :]"));
        }
        return Ok("col");
    }
    if k == "col" {
        let resident = n
            .args
            .first()
            .and_then(|a| ir.arena.get(a.0))
            .is_some_and(|a| a.kind == "resident");
        if !["col", "tile"].contains(&first) && resident {
            return Err(plan(
                ".col() of a field read whole is not lowered in the row form \
                 (.col() broadcasts a reduced value or a tile)",
            ));
        }
        if !["col", "tile"].contains(&first) {
            return Err(plan(".col() broadcasts a reduced value or a tile"));
        }
        return Ok(if first == "col" { "bcol" } else { "tile" });
    }
    let has = |s: &str| args.contains(&s);
    if has("row") || has("bcol") {
        if has("col") || has("tile") {
            return Err(plan(
                "a row block meets a column block: broadcast it with .col()",
            ));
        }
        return Ok("row");
    }
    if has("tile") {
        if has("col") {
            return Err(plan(
                "a tile meets a reduced value: broadcast it with .col()",
            ));
        }
        return Ok("tile");
    }
    if has("col") {
        return Ok("col");
    }
    Ok("scalar")
}

/// Refused unless `red` is reduce_sum, reduce_max or reduce_min, exact, along the row.
fn reduced(red: &Node) -> Result<&'static str, ChainError> {
    let op = match attr(red, "op") {
        Some(Attr::Text(op)) => REDUCTIONS.iter().find(|o| *o == op).copied(),
        _ => None,
    };
    let exact = matches!(attr(red, "exact"), Some(Attr::Text(t)) if t == "True");
    let along = matches!(attr(red, "dim"), Some(Attr::Int(Int::Const(-1))));
    match op {
        Some(op) if exact && along => Ok(op),
        _ => Err(plan_error(
            "the row form's reduction is reduce_sum, reduce_max or reduce_min(..., \
             exact=True) along the row"
                .into(),
        )),
    }
}

/// acc and v combined by reduction `op`: their sum, larger or smaller.
fn fold(b: &mut Body, op: &str, acc: Arg, v: Arg, note: &str) -> Result<Arg, ChainError> {
    match op {
        "max" => maximum(b, acc, v),
        "min" => minimum(b, acc, v),
        _ => b.add(acc, v, note),
    }
}

/// The 8 lanes of each lane row of `acc` combined by `op` (rotations by 1, 2 and 4
/// lanes), the value in every lane.
fn lanes(b: &mut Body, op: &str, mut acc: Arg) -> Result<Arg, ChainError> {
    for k in [1, 2, 4] {
        let mut turned = acc.clone();
        for _ in 0..k {
            let args = [("lreg_src_c", turned), ("instr_mod1", Arg::Num(ROTATE))];
            turned = b.op("SFPSHFT2", &format!("lanes turned (step {k})"), &args)?;
        }
        acc = fold(b, op, acc, turned, &format!("sum of {} lanes", 2 * k))?;
    }
    Ok(acc)
}

/// (t, c, p) of every vector of one row group across the row.
fn vectors(width: i64) -> Vec<(i64, i64, i64)> {
    let mut out = Vec::new();
    for t in 0..width {
        for c in 0..2 {
            for p in 0..2 {
                out.push((t, c, p));
            }
        }
    }
    out
}

/// A vector's place as today prints its tuple.
fn tuple((t, h, c, q, p): (i64, i64, i64, i64, i64)) -> String {
    format!("({t}, {h}, {c}, {q}, {p})")
}

/// The vector function of `chain` and its result.
fn result_of(chain: &Chain) -> Result<(Vs, VId), ChainError> {
    let mut vs = chain.vs.clone();
    let FnResult::One(result) = chain.func.result else {
        return Err(ChainError::NotPorted(
            "a row function of even and odd results".into(),
        ));
    };
    let id = vs.wrap(result);
    Ok((vs, id))
}

/// Per row group: f of every vector of the row summed lane by lane (a product into a
/// multiply-add), the 8 lanes of each lane row summed by rotations, the per-row value
/// computed and stored to its slot. Returns each pair of groups' words (their range in
/// the code as made), for the interleave pass.
fn stat_phase(
    b: &mut Body,
    reduced: &Chain,
    per_row: &Chain,
    rows: Layout,
    width: i64,
    op: &str,
) -> Result<Vec<(usize, usize)>, ChainError> {
    let (vs, top) = result_of(reduced)?;
    let (stat_vs, stat) = result_of(per_row)?;
    let top_node = vs
        .get(top)
        .ok_or_else(|| ChainError::NotPorted("a vector value of no function".into()))?
        .clone();
    let (mut pairs, mut start) = (Vec::new(), 0);
    for (h, q) in GROUPS {
        if q % 2 == 0 {
            start = b.code.len();
        }
        let mut acc: Option<Arg> = None;
        for (t, c, p) in vectors(width) {
            let at = (t, h, c, q, p);
            b.reads = Some(Reads {
                address: Layout::Tile(0).address(at),
                note: format!("row {}", tuple(at)),
                ..Reads::default()
            });
            let mad = top_node.kind == "mul" && op == "sum";
            acc = Some(match (acc.take(), &top_node.args[..]) {
                (Some(sum), [x, m]) if mad => {
                    let mut memo = Vec::new();
                    let a = value(b, &vs, *x, &mut memo)?;
                    let m = value(b, &vs, *m, &mut memo)?;
                    b.mad_of(a, m, sum, "sum += f", false, false, None)?
                }
                (sum, _) => {
                    let v = value(b, &vs, top, &mut Vec::new())?;
                    match sum {
                        None => v,
                        Some(sum) => fold(b, op, sum, v, "sum += f")?,
                    }
                }
            });
        }
        b.reads = None;
        let Some(acc) = acc else {
            return Err(ChainError::NotPorted("a row of no vector".into()));
        };
        let acc = lanes(b, op, acc)?;
        b.reads = Some(Reads {
            all: Some(acc),
            ..Reads::default()
        });
        let made = value(b, &stat_vs, stat, &mut Vec::new())?;
        b.reads = None;
        let note = format!("per-row value, group {h} {q}");
        b.store(made, rows.address((0, h, 0, q, 0)), &note)?;
        if q % 2 == 1 {
            pairs.push((start, b.code.len()));
        }
    }
    Ok(pairs)
}

/// Per half of the row: its 4 per-row slots loaded once; per vector column (t, c, p)
/// the per-column vector loaded once; the stored function of two row groups at a time
/// written over the row. Returns each pair's words (their range in the code as made).
fn apply_phase(
    b: &mut Body,
    stored: &Chain,
    layouts: Layouts,
    width: i64,
) -> Result<Vec<(usize, usize)>, ChainError> {
    let kinds: Vec<&'static str> = stored.operands.iter().map(|l| l.kind).collect();
    let (vs, result) = result_of(stored)?;
    let mut pairs = Vec::new();
    for h in 0..2 {
        let mut slots: Vec<(i64, Arg)> = Vec::new();
        for (t, c, p) in vectors(width) {
            let mut column: Option<Arg> = None;
            for pair in [[0, 1], [2, 3]] {
                let start = b.code.len();
                for q in pair {
                    let at = (t, h, c, q, p);
                    let address = layouts.copy.address(at);
                    b.reads = Some(Reads {
                        all: None,
                        kinds: kinds.clone(),
                        address,
                        note: format!("row {}", tuple(at)),
                        loaded: Vec::new(),
                        slot: (q, layouts.rows.address(at), format!("per-row slot {h} {q}")),
                        slots: std::mem::take(&mut slots),
                        column_at: (layouts.cols.address(at), format!("per-column {t} {c} {p}")),
                        column: column.take(),
                    });
                    let made = value(b, &vs, result, &mut Vec::new());
                    if let Some(r) = b.reads.take() {
                        slots = r.slots;
                        column = r.column;
                    }
                    b.store(made?, address, &format!("row {}", tuple(at)))?;
                }
                pairs.push((start, b.code.len()));
            }
        }
    }
    Ok(pairs)
}

/// (t, h, c, q, p) of a row vector's Dst address, or none outside the row.
fn index(address: i64, width: i64) -> Option<(i64, i64, i64, i64, i64)> {
    if !(0..TILE * width).contains(&address) {
        return None;
    }
    let (t, rest) = (address / TILE, address % TILE);
    let (h, rest) = (rest / 32, rest % 32);
    let (c, rest) = (rest / 16, rest % 16);
    let (q, rest) = (rest / 4, rest % 4);
    Some((t, h, c, q, rest / 2))
}

/// A number field of `x`.
fn num(x: &Insn, field: &str) -> i64 {
    x.arg(field).and_then(Arg::num).unwrap_or(0)
}

/// A register field of `x`.
fn reg(x: &Insn, field: &str) -> Arg {
    x.arg(field).cloned().unwrap_or(Arg::Num(0))
}

/// A set of addresses as today prints a sorted list.
fn listed(set: &BTreeSet<i64>) -> String {
    let parts: Vec<String> = set.iter().map(i64::to_string).collect();
    format!("[{}]", parts.join(", "))
}

/// broadcast: every stored row vector was computed from the row vector at its own
/// address, the per-row slot of its row group and the per-column vector of its columns
/// (before allocation).
pub fn check_broadcast(
    unit: &mut UnitProgram,
    code: &[Insn],
    layouts: Layouts,
    width: i64,
) -> Result<(), UnitError> {
    let mut sources: HashMap<Arg, BTreeSet<i64>> = HashMap::new();
    let (mut checked, mut depth) = (0, 0);
    for x in code {
        if x.sets_flags() {
            depth = if x.op.name == "SFPENCC" { 0 } else { depth + 1 };
        }
        if x.op.name == "SFPLOAD" {
            if let Some(w) = x.writes()?.into_iter().next() {
                sources.insert(w, BTreeSet::from([num(x, "dest_reg_addr")]));
            }
        } else if x.op.name == "SFPSTORE" {
            let address = num(x, "dest_reg_addr");
            let Some(at) = index(address, width) else {
                continue;
            };
            let want: BTreeSet<i64> = layouts.all(at).into_iter().collect();
            let got = sources
                .get(&reg(x, "lreg_ind"))
                .cloned()
                .unwrap_or_default();
            let own = layouts.copy.address(at);
            let slots = BTreeSet::from([layouts.rows.address(at)]);
            let alone = !got.is_empty() && got.is_subset(&slots);
            if !got.is_subset(&want) || !got.contains(&own) && !alone {
                return Err(UnitError(format!(
                    "broadcast: the row vector at Dst + {address} reads {}, not its own slots {}",
                    listed(&got),
                    listed(&want)
                )));
            }
            checked += 1;
        } else {
            let mut got = BTreeSet::new();
            for v in x.reads() {
                if let Some(s) = sources.get(&v) {
                    got.extend(s.iter().copied());
                }
            }
            for v in x.writes()? {
                // a write under lane flags keeps what the other lanes held
                let mut made = got.clone();
                if depth > 0 {
                    if let Some(s) = sources.get(&v) {
                        made.extend(s.iter().copied());
                    }
                }
                sources.insert(v, made);
            }
        }
    }
    unit.checks.push(Check {
        rule: "broadcast".into(),
        text: "a stored row vector meets the per-row slot of its rows and the per-column \
               vector of its columns"
            .into(),
        verdict: format!("ok: {checked} vectors followed back to their loads"),
    });
    Ok(())
}

/// lanes: a value stored into the per-row tile is complete, every lane of a lane row
/// holding the sum of all 8 (a + a turned by its span doubles the span; values the same
/// in every lane have span 8). Before register allocation.
pub fn check_lanes(unit: &mut UnitProgram, code: &[Insn], rows: Layout) -> Result<(), UnitError> {
    let Layout::Rows(tile, _) = rows else {
        return Err(UnitError(
            "lanes: the per-row values are not in slots".into(),
        ));
    };
    let mut span: HashMap<Arg, i64> = HashMap::new();
    let mut turned: HashMap<Arg, (Arg, i64)> = HashMap::new();
    let mut copied: HashMap<Arg, Arg> = HashMap::new();
    let mut stores = 0;
    for x in code {
        let reads: Vec<Arg> = x
            .reads()
            .into_iter()
            .map(|r| copied.get(&r).cloned().unwrap_or(r))
            .collect();
        if x.op.name == "SFPLOAD" {
            // a per-row slot holds complete values
            let whole = num(x, "dest_reg_addr").div_euclid(TILE) == tile;
            if let Some(w) = x.writes()?.into_iter().next() {
                span.insert(w, if whole { 8 } else { 1 });
            }
            continue;
        }
        if x.op.name == "SFPSTORE" {
            let address = num(x, "dest_reg_addr");
            if address.div_euclid(TILE) == tile {
                let got = span.get(&reg(x, "lreg_ind")).copied().unwrap_or(8);
                if got != 8 {
                    return Err(UnitError(format!(
                        "lanes: the per-row value stored at Dst + {address} sums {got} of the 8 lanes of a row"
                    )));
                }
                stores += 1;
            }
            continue;
        }
        let out = x.writes()?;
        for w in &out {
            copied.remove(w);
        }
        let Some(first) = out.first().cloned() else {
            continue;
        };
        let read0 = reads.first().cloned().unwrap_or(Arg::Num(0));
        if x.op.name == "SFPSHFT2" {
            let (base, shift) = turned.get(&read0).cloned().unwrap_or((read0, 0));
            span.insert(first.clone(), span.get(&base).copied().unwrap_or(8));
            turned.insert(first, (base, shift + 1));
            continue;
        }
        if x.op.name == "SFPMOV" && span.contains_key(&read0) && num(x, "instr_mod1") == 0 {
            copied.insert(first, read0);
            continue;
        }
        let values: Vec<&Arg> = reads
            .iter()
            .filter(|r| matches!(r, Arg::Name(_)) && span.contains_key(*r))
            .collect();
        let mut width = values.iter().map(|v| span[*v]).min().unwrap_or(8);
        for a in &values {
            for b in &values {
                if turned.get(*b) == Some(&((*a).clone(), span[*a])) {
                    width = 2 * span[*a];
                }
            }
        }
        for w in out {
            // a swap writes both its registers
            span.insert(w, width);
        }
    }
    unit.checks.push(Check {
        rule: "lanes".into(),
        text: "a per-row value is summed over all 8 lanes of its row before a store \
               broadcasts it"
            .into(),
        verdict: format!("ok: {stores} slots, each complete"),
    });
    Ok(())
}

/// The SFPNOPs the mad-reader check inserted (its verdict `repaired: n SFPNOP`).
fn repairs(unit: &UnitProgram) -> usize {
    unit.checks
        .iter()
        .find(|c| c.rule == "mad-reader")
        .and_then(|c| c.verdict.strip_prefix("repaired: "))
        .and_then(|v| v.split(' ').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// The levels of a row (several reductions, `.col()` or resident rows, or a reduction
/// reading `.row()`): not ported yet.
fn levels() -> ChainError {
    ChainError::NotPorted(
        "the row form's levels (several reductions, .col() or resident rows, or a \
         reduction that reads .row())"
            .into(),
    )
}

/// The unit program of row kernel `ir` over a row of `width` tiles in Dst tiles 0 ..
/// width - 1 (the result stored over them), its per-row value in Dst tile `rows_tile`
/// and its per-column operand read from Dst tile `cols_tile` (today's `row_program`);
/// `band`: the body's live rows a runtime argument, not ported yet.
pub fn row_program(
    ir: &KernelIR,
    width: i64,
    rows_tile: i64,
    cols_tile: i64,
    dest: &str,
    band: bool,
) -> Result<UnitProgram, ChainError> {
    if dest != "fp32" {
        return Err(ChainError::Plan(
            "the unit program is built for fp32 Dst".into(),
            None,
        ));
    }
    if !(1..=2).contains(&width) {
        return Err(plan_error(
            "the row form takes rows of 1 or 2 tiles (a per-column tile holds 2)".into(),
        ));
    }
    let typed = ir
        .tensors
        .iter()
        .all(|t| t.dtype.as_deref().is_some_and(|d| TILE_DTYPES.contains(&d)));
    if !typed {
        return Err(plan_error(format!(
            "{}: tensors are fp32 or of the B family ({})",
            ir.name,
            TILE_DTYPES[1..].join(", ")
        )));
    }
    let [store] = &ir.stores[..] else {
        return Err(plan_error(format!(
            "{}: the row form takes one store",
            ir.name
        )));
    };
    let shapes = row_shapes(ir)?;
    let nodes = ir.nodes();
    let of_kind = |k: &str| -> Vec<NodeId> {
        nodes
            .iter()
            .copied()
            .filter(|n| ir.arena.get(n.0).is_some_and(|x| x.kind == k))
            .collect()
    };
    let reduces = of_kind("reduce");
    if reduces.is_empty() {
        return Err(plan_error(format!(
            "{}: the row form takes a reduction",
            ir.name
        )));
    }
    let mut ops = Vec::new();
    for &r in &reduces {
        ops.push(reduced(node(ir, r)?)?);
    }
    let cols = of_kind("col");
    let mut stats = Vec::new();
    for &c in &cols {
        let Some(&a) = node(ir, c)?.args.first() else {
            return Err(ChainError::NotPorted("a .col() of no operand".into()));
        };
        if shapes.get(a.0).copied().flatten() != Some("col") {
            return Err(plan_error(
                "a per-row value meets the row through .col()".into(),
            ));
        }
        stats.push(a);
    }
    let mut tensors: Vec<&str> = Vec::new();
    let mut row = None;
    for &n in &nodes {
        let x = node(ir, n)?;
        if x.kind == "load" && view_kind(x) == Some("row") {
            if let Some(Attr::View(v)) = attr(x, "view") {
                if !tensors.contains(&v.tensor.as_str()) {
                    tensors.push(&v.tensor);
                }
                row.get_or_insert(v.tensor.clone());
            }
        }
    }
    let (Some(row), 1) = (row, tensors.len()) else {
        return Err(plan_error(
            "the row form reads one row tensor (loaded once)".into(),
        ));
    };
    let residents = of_kind("resident");
    if reduces.len() > 1 || cols.len() != 1 || residents.len() > 1 {
        return Err(levels());
    }
    let (red, stat, op) = (reduces[0], stats[0], ops[0]);
    let Some(&summed) = node(ir, red)?.args.first() else {
        return Err(ChainError::NotPorted("a reduction of no operand".into()));
    };
    let mut compiler = Compiler::rows(ir, shapes, red, stat);
    let reduced = compiler.chain(summed)?;
    if reduced.operands.iter().any(|l| l.kind == "bcast_row") {
        // the stat phase reads the row alone: a reduction of a resident row's columns
        // is the levels' (their sums read it by .row())
        return Err(levels());
    }
    let per_row = compiler.chain(stat)?;
    let stored = compiler.chain(store.value)?;
    for c in [&reduced, &per_row, &stored] {
        if !c.values.is_empty() || c.prev.is_some() {
            return Err(plan_error(format!(
                "{}: the row form takes no runtime scalars and one stage",
                ir.name
            )));
        }
    }
    let layouts = Layouts {
        copy: Layout::Tile(0),
        rows: Layout::Rows(rows_tile, 0),
        cols: Layout::Cols(cols_tile),
    };
    for c in [&reduced, &stored] {
        for leaf in &c.operands {
            let kind = match leaf.source {
                Source::Node(n) => node(ir, n)?.kind.as_str(),
                Source::Prev => "prev",
            };
            if leaf.kind == "copy" && kind != "load" {
                return Err(plan_error("the row's own operand is the loaded row".into()));
            }
            if leaf.kind == "bcast_row" && kind != "resident" {
                return Err(plan_error(
                    ".row() broadcasts a resident row (a weight)".into(),
                ));
            }
        }
    }
    let mut body = Body::default();
    let mut unit = UnitProgram::new(&ir.name, dest);
    unit.operands = vec![Some(row)];
    let mut pairs = stat_phase(&mut body, &reduced, &per_row, layouts.rows, width, op)?;
    pairs.extend(apply_phase(&mut body, &stored, layouts, width)?);
    // each word marked, so that a pair's words are found again after the passes
    for (i, x) in body.code.iter_mut().enumerate() {
        x.mark = i + 1;
    }
    let done = fold_negation(&mut body.code)?;
    unit.passes.push(pass("algebra", "fold_negation", done));
    let done = fuse_mad(&mut body.code)?;
    unit.passes.push(pass("algebra", "fuse_mad", done));
    unit.passes.push(pass(
        "layout",
        "lanes",
        vec![format!(
            "the per-row value: 8 vectors of lanes in tile {rows_tile}; the per-column \
             value: {} vectors in tile {cols_tile}",
            4 * width
        )],
    ));
    check_lanes(&mut unit, &body.code, layouts.rows)?;
    let mut moved = 0;
    for &(start, end) in &pairs {
        let at: Vec<usize> = (0..body.code.len())
            .filter(|&i| (start + 1..=end).contains(&body.code[i].mark))
            .collect();
        let (Some(&first), Some(&last)) = (at.first(), at.last()) else {
            return Err(ChainError::NotPorted("a pair of no word".into()));
        };
        moved += interleave(&mut body.code, first, last + 1)?;
    }
    unit.passes.push(pass(
        "schedule",
        "interleave",
        vec![format!("{moved} instructions moved in {}", pairs.len())],
    ));
    check_broadcast(&mut unit, &body.code, layouts, width)?;
    let (mut setup, mut head, mut tail) = (Vec::new(), Vec::new(), Vec::new());
    let done = hoist(&mut body.code, &mut setup, &mut head, &mut tail)?;
    if !head.is_empty() {
        return Err(ChainError::NotPorted(
            "LReg 11 set by the row form's body".into(),
        ));
    }
    unit.passes.push(pass("schedule", "hoist", done));
    let done = allocate(&mut body.code, &body.prog, &[])?;
    unit.passes.push(pass("layout", "allocate", done));
    allocate(&mut setup, &body.prog, &[])?;
    unit.constants = body
        .prog
        .iter()
        .zip(PROGRAMMABLE)
        .map(|((_, v), r)| (r, v.clone()))
        .collect();
    let mut init = vec![Item::Insn(insn(
        "SFPCONFIG",
        &[("imm16_math", 0), ("config_dest", 15), ("instr_mod1", 1)],
        "lane configuration",
    )?)];
    init.extend(items(setup));
    unit.init[MATH] = init;
    let mut copies: Vec<String> = (0..width)
        .map(|t| format!("copy_tile(x, {t}, dst + {t})"))
        .collect();
    copies.push(format!("copy_tile(cols, 0, dst + {cols_tile})"));
    let mut unpack = vec![sync(
        "unpack",
        "reader",
        "cb_wait_front(x, width)",
        "the row's tiles",
    )];
    unpack.extend(
        copies
            .iter()
            .map(|c| call(c, "into Dst (fp32, unpacked to Dst)")),
    );
    unit.tile[UNPACK] = unpack;
    let mut math = vec![sync(
        "math",
        "pack",
        "tile_regs_acquire()",
        "a free Dst half",
    )];
    math.extend(
        copies
            .iter()
            .map(|c| call(c, "the math part of the copy (none to Dst)")),
    );
    math.push(call(
        "ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst)",
        "Dst base of tile dst (RISC-V writes)",
    ));
    math.push(Item::Insn(insn(
        "STALLWAIT",
        &[("stall_res", STALL_SFPU), ("wait_res", WAIT_MATH)],
        "SFPU waits for the matrix unit",
    )?));
    math.extend(items(body.code.clone()));
    math.push(sync(
        "math",
        "pack",
        "tile_regs_commit()",
        "hands Dst to the pack CPU",
    ));
    unit.tile[MATH] = math;
    let mut pack = vec![sync(
        "pack",
        "math",
        "tile_regs_wait()",
        "Dst from the math CPU",
    )];
    pack.extend(
        (0..width).map(|t| call(&format!("pack_tile(dst + {t}, out)"), "the row's tile t")),
    );
    pack.push(sync(
        "pack",
        "math",
        "tile_regs_release()",
        "frees the Dst half",
    ));
    unit.tile[PACK] = pack;
    let mut code = items(std::mem::take(&mut body.code));
    hazard_checks_on(&mut unit, &mut code)?;
    // the checks' repairs went into the code: the tile program takes it again (as
    // today, from the end less the code's length and the mad-reader's repairs)
    let repaired = repairs(&unit);
    let math = &mut unit.tile[MATH];
    let end = math.len().saturating_sub(1);
    let start = (end + repaired).saturating_sub(code.len()).min(end);
    math.splice(start..end, code);
    if band {
        return Err(ChainError::NotPorted("the row form's band".into()));
    }
    unit.not_checked = NOT_CHECKED
        .iter()
        .filter(|r| !r.contains("load-macro"))
        .map(|r| r.to_string())
        .collect();
    check_dst_window(&mut unit, DST_TILES)?;
    Ok(unit)
}
