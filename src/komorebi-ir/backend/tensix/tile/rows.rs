//! The row form (today's `komorebi.dsl.tile.forms.rows`): a row of `width` whole tiles
//! in Dst, one exact reduction along the row, a per-row value computed from it, and the
//! stored row computed from the row, that value broadcast with `.col()` and a resident
//! row broadcast with `.row()`. Layout rule: a per-row value is lanes of one tile.
//! `row_program` and its levels (`level_program`) are ported; the chunk form and the
//! band are refused by name.

use std::collections::{BTreeSet, HashMap};

use super::chain::{Chain, ChainError, Compiler, Source, SLOTS};
use super::checks::{check_dst_window, hazard_checks_on, STALL_SFPU, TILE, WAIT_MATH};
use super::ir::{Attr, Int, KernelIR, Node, NodeId};
use super::lower::{
    allocate, call, fold_negation, fuse_mad, hoist, insn, items, maximum, minimum, pass,
    plan_error, sync, value, Body, Read, Reads, NOT_CHECKED, TILE_DTYPES,
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
/// The per-row slots of a row group in the rows tile (Dst rows).
const ROW_SLOTS: [i64; 4] = [0, 2, 32, 34];
/// What a level's reduction reads.
const SUMMED: [&str; 3] = ["copy", "bcast_col", "bcast_row"];

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

/// The layouts of a row: the row's (`copy`), each per-row value's slot (`bcast_col`)
/// and each resident row's per-column vectors (`bcast_row`).
#[derive(Clone, Debug)]
pub struct Layouts {
    /// The row's.
    pub copy: Layout,
    /// The per-row slots', one a value.
    pub rows: Vec<Layout>,
    /// The per-column vectors', one a resident row.
    pub cols: Vec<Layout>,
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

/// `vs`'s value `id` with operand k read by `ops[k]`, a load once under each key of
/// `cache` (kept there for the caller to drop where today's dict of loaded values ends).
fn read_value(
    b: &mut Body,
    vs: &Vs,
    id: VId,
    ops: Vec<Read>,
    cache: &mut Vec<(String, Arg)>,
    memo: &mut Vec<(VId, Arg)>,
) -> Result<Arg, ChainError> {
    b.reads = Some(Reads {
        ops,
        cache: std::mem::take(cache),
    });
    let made = value(b, vs, id, memo);
    if let Some(r) = b.reads.take() {
        *cache = r.cache;
    }
    made
}

/// Drops from `cache` the values of keys starting with `scope`.
fn drop_scope(cache: &mut Vec<(String, Arg)>, scope: &str) {
    cache.retain(|(k, _)| !k.starts_with(scope));
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
            let (address, note) = (Layout::Tile(0).address(at), format!("row {}", tuple(at)));
            // each operand loaded once a vector (today's dict by operand)
            let ops: Vec<Read> = (0..reduced.operands.len())
                .map(|i| Read::Load(format!("row{i}"), address, note.clone()))
                .collect();
            let mut cache = Vec::new();
            let mad = top_node.kind == "mul" && op == "sum";
            acc = Some(match (acc.take(), &top_node.args[..]) {
                (Some(sum), [x, m]) if mad => {
                    let mut memo = Vec::new();
                    let a = read_value(b, &vs, *x, ops.clone(), &mut cache, &mut memo)?;
                    let m = read_value(b, &vs, *m, ops, &mut cache, &mut memo)?;
                    b.mad_of(a, m, sum, "sum += f", false, false, None)?
                }
                (sum, _) => {
                    let v = read_value(b, &vs, top, ops, &mut cache, &mut Vec::new())?;
                    match sum {
                        None => v,
                        Some(sum) => fold(b, op, sum, v, "sum += f")?,
                    }
                }
            });
        }
        let Some(acc) = acc else {
            return Err(ChainError::NotPorted("a row of no vector".into()));
        };
        let acc = lanes(b, op, acc)?;
        let ops = vec![Read::Value(acc); per_row.operands.len()];
        let made = read_value(b, &stat_vs, stat, ops, &mut Vec::new(), &mut Vec::new())?;
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
    layouts: &Layouts,
    width: i64,
) -> Result<Vec<(usize, usize)>, ChainError> {
    let (vs, result) = result_of(stored)?;
    let (Some(&slot), Some(&cols)) = (layouts.rows.first(), layouts.cols.first()) else {
        return Err(ChainError::NotPorted("a row form of no slot".into()));
    };
    let mut pairs = Vec::new();
    for h in 0..2 {
        // the slots loaded once a half, the column once a vector column, the row's
        // operands once a vector
        let mut cache = Vec::new();
        for (t, c, p) in vectors(width) {
            for pair in [[0, 1], [2, 3]] {
                let start = b.code.len();
                for q in pair {
                    let at = (t, h, c, q, p);
                    let address = layouts.copy.address(at);
                    let ops = stored
                        .operands
                        .iter()
                        .enumerate()
                        .map(|(i, l)| match l.kind {
                            "bcast_col" => Read::Load(
                                format!("slot{q}"),
                                slot.address(at),
                                format!("per-row slot {h} {q}"),
                            ),
                            "bcast_row" => Read::Load(
                                "column".into(),
                                cols.address(at),
                                format!("per-column {t} {c} {p}"),
                            ),
                            _ => {
                                Read::Load(format!("row{i}"), address, format!("row {}", tuple(at)))
                            }
                        })
                        .collect();
                    let made = read_value(b, &vs, result, ops, &mut cache, &mut Vec::new());
                    drop_scope(&mut cache, "row");
                    b.store(made?, address, &format!("row {}", tuple(at)))?;
                }
                pairs.push((start, b.code.len()));
            }
            drop_scope(&mut cache, "column");
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
    layouts: &Layouts,
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
            let mut want = BTreeSet::from([layouts.copy.address(at)]);
            let slots: BTreeSet<i64> = layouts.rows.iter().map(|l| l.address(at)).collect();
            want.extend(&slots);
            want.extend(layouts.cols.iter().map(|l| l.address(at)));
            let got = sources
                .get(&reg(x, "lreg_ind"))
                .cloned()
                .unwrap_or_default();
            let own = layouts.copy.address(at);
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
        return level_program(ir, width, rows_tile, cols_tile, shapes, row, band);
    }
    let (red, stat, op) = (reduces[0], stats[0], ops[0]);
    let Some(&summed) = node(ir, red)?.args.first() else {
        return Err(ChainError::NotPorted("a reduction of no operand".into()));
    };
    let mut compiler = Compiler::rows(ir, shapes.clone(), vec![red, stat], SLOTS);
    let reduced = compiler.chain(summed)?;
    if reduced.operands.iter().any(|l| l.kind == "bcast_row") {
        // the stat phase reads the row alone: a reduction of a resident row's columns
        // is the levels' (their sums read it by .row())
        return level_program(ir, width, rows_tile, cols_tile, shapes, row, band);
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
        rows: vec![Layout::Rows(rows_tile, 0)],
        cols: vec![Layout::Cols(cols_tile)],
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
    let rows = Layout::Rows(rows_tile, 0);
    let mut pairs = stat_phase(&mut body, &reduced, &per_row, rows, width, op)?;
    pairs.extend(apply_phase(&mut body, &stored, &layouts, width)?);
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
    check_lanes(&mut unit, &body.code, rows)?;
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
    check_broadcast(&mut unit, &body.code, &layouts, width)?;
    let mut copies: Vec<String> = (0..width)
        .map(|t| format!("copy_tile(x, {t}, dst + {t})"))
        .collect();
    copies.push(format!("copy_tile(cols, 0, dst + {cols_tile})"));
    finish(unit, body, width, &copies, band)
}

/// The row form's unit from its body checked for broadcasts on: the hoist, allocation,
/// the tile program around the code (`copies` into Dst), the hazard checks with their
/// repairs taken back, and the Dst window; `band` refused by name.
fn finish(
    mut unit: UnitProgram,
    mut body: Body,
    width: i64,
    copies: &[String],
    band: bool,
) -> Result<UnitProgram, ChainError> {
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

/// The nodes of `kinds` under `from`, not descending past a node of `stop` (`from`
/// itself excepted), in today's order of finding them.
fn below(
    ir: &KernelIR,
    from: NodeId,
    kinds: &[&str],
    stop: &[&str],
) -> Result<Vec<NodeId>, ChainError> {
    let (mut found, mut seen, mut todo) = (Vec::new(), BTreeSet::new(), vec![from]);
    while let Some(n) = todo.pop() {
        if !seen.insert(n.0) {
            continue;
        }
        let x = node(ir, n)?;
        if kinds.contains(&x.kind.as_str()) {
            found.push(n);
        }
        if !stop.contains(&x.kind.as_str()) || n == from {
            todo.extend(x.args.iter().copied());
        }
    }
    Ok(found)
}

/// Each reduction's level (a pass over the row: one more than the per-row values its
/// argument reads by .col()) and each per-row value's (its last reduction's), taken in
/// the order of `nodes` (each operand's before its user's).
fn level_of(
    ir: &KernelIR,
    nodes: &[NodeId],
    reduces: &[NodeId],
    stats: &[NodeId],
) -> Result<HashMap<NodeId, i64>, ChainError> {
    let mut level: HashMap<NodeId, i64> = HashMap::new();
    let read = |level: &HashMap<NodeId, i64>, n: NodeId| {
        level
            .get(&n)
            .copied()
            .ok_or_else(|| ChainError::NotPorted("a per-row value read before it is made".into()))
    };
    for &n in nodes {
        let made = if reduces.contains(&n) {
            let Some(&summed) = node(ir, n)?.args.first() else {
                return Err(ChainError::NotPorted("a reduction of no operand".into()));
            };
            let mut top = -1;
            for c in below(ir, summed, &["col"], &["col"])? {
                let Some(&p) = node(ir, c)?.args.first() else {
                    return Err(ChainError::NotPorted("a .col() of no operand".into()));
                };
                top = top.max(read(&level, p)?);
            }
            1 + top
        } else if stats.contains(&n) {
            let mut top = None;
            for r in below(ir, n, &["reduce"], &["reduce"])? {
                top = Some(top.unwrap_or(0).max(read(&level, r)?));
            }
            top.ok_or_else(|| ChainError::NotPorted("a per-row value of no reduction".into()))?
        } else {
            continue;
        };
        level.insert(n, made);
    }
    Ok(level)
}

/// The layout of `n` in `of`, or none.
fn layout_of(of: &[(NodeId, Layout)], n: NodeId) -> Result<Layout, ChainError> {
    of.iter()
        .find(|(k, _)| *k == n)
        .map(|(_, l)| *l)
        .ok_or_else(|| ChainError::NotPorted("a per-row or per-column value of no slot".into()))
}

/// The node a leaf reads.
fn source(leaf: &super::chain::Leaf) -> Result<NodeId, ChainError> {
    match leaf.source {
        Source::Node(n) => Ok(n),
        Source::Prev => Err(ChainError::NotPorted(
            "a stage's result in the row form".into(),
        )),
    }
}

/// The row form with several reductions in levels, a pass over the row a level: a
/// level's sums may read the per-row values of the levels before it by .col(); each
/// per-row value kept across a level (and each sum a later value reads) in a slot of the
/// rows tile; one per-column tile a resident row (Dst tile `cols_tile` + k). Today's
/// `level_program`; `band` refused with today's text.
fn level_program(
    ir: &KernelIR,
    width: i64,
    rows_tile: i64,
    cols_tile: i64,
    shapes: Vec<Option<&'static str>>,
    row: String,
    band: bool,
) -> Result<UnitProgram, ChainError> {
    if band {
        return Err(plan_error(format!(
            "{}: the band takes the row form of one reduction",
            ir.name
        )));
    }
    let [store] = &ir.stores[..] else {
        return Err(plan_error(format!(
            "{}: the row form takes one store",
            ir.name
        )));
    };
    let nodes = ir.nodes();
    let mut reduces: Vec<NodeId> = Vec::new();
    let mut stats: Vec<NodeId> = Vec::new();
    let mut residents: Vec<NodeId> = Vec::new();
    for &n in &nodes {
        let x = node(ir, n)?;
        match x.kind.as_str() {
            "reduce" => reduces.push(n),
            "resident" if !residents.contains(&n) => residents.push(n),
            "col" => {
                let Some(&a) = x.args.first() else {
                    return Err(ChainError::NotPorted("a .col() of no operand".into()));
                };
                if !stats.contains(&a) {
                    stats.push(a);
                }
            }
            _ => {}
        }
    }
    let level = level_of(ir, &nodes, &reduces, &stats)?;
    let at = |n: NodeId| level.get(&n).copied().unwrap_or(0);
    let mut kept = Vec::new();
    for &r in &reduces {
        if stats.contains(&r) {
            continue;
        }
        let mut read_later = false;
        for &p in &stats {
            if below(ir, p, &["reduce"], &["reduce"])?.contains(&r) && at(p) > at(r) {
                read_later = true;
            }
        }
        if read_later {
            kept.push(r);
        }
    }
    let held: Vec<NodeId> = stats.iter().chain(&kept).copied().collect();
    if held.len() > ROW_SLOTS.len() {
        return Err(plan_error(format!(
            "{}: {} per-row values kept across the row's passes; the rows tile holds {} \
             (a slot a value a row group)",
            ir.name,
            held.len(),
            ROW_SLOTS.len()
        )));
    }
    let fields = residents.len() as i64;
    if cols_tile + fields > DST_TILES {
        return Err(plan_error(format!(
            "{}: {fields} resident rows; Dst's {DST_TILES} fp32 tiles hold {} per-column \
             tiles after the row's",
            ir.name,
            DST_TILES - cols_tile
        )));
    }
    let slot: Vec<(NodeId, Layout)> = held
        .iter()
        .zip(ROW_SLOTS)
        .map(|(&v, offset)| (v, Layout::Rows(rows_tile, offset)))
        .collect();
    let cols: Vec<(NodeId, Layout)> = residents
        .iter()
        .zip(cols_tile..)
        .map(|(&r, tile)| (r, Layout::Cols(tile)))
        .collect();
    let mut leaves = reduces.clone();
    leaves.extend(&held);
    let limit = SLOTS + residents.len() + stats.len();
    let mut compiler = Compiler::rows(ir, shapes, leaves, limit);
    let mut summed = Vec::new();
    for &r in &reduces {
        let Some(&a) = node(ir, r)?.args.first() else {
            return Err(ChainError::NotPorted("a reduction of no operand".into()));
        };
        summed.push((r, compiler.chain(a)?));
    }
    let mut made = Vec::new();
    for &p in &stats {
        if node(ir, p)?.kind != "reduce" {
            made.push((p, compiler.chain(p)?));
        }
    }
    let stored = compiler.chain(store.value)?;
    for (_, c) in &summed {
        if c.operands.iter().any(|l| !SUMMED.contains(&l.kind)) {
            return Err(plan_error(format!(
                "{}: a sum along the row reads the row, per-row values and resident rows",
                ir.name
            )));
        }
    }
    for c in summed.iter().chain(&made).map(|(_, c)| c).chain([&stored]) {
        if !c.values.is_empty() || c.prev.is_some() {
            return Err(plan_error(format!(
                "{}: the row form takes no runtime scalars and one stage",
                ir.name
            )));
        }
        for leaf in &c.operands {
            let resident = node(ir, source(leaf)?)?.kind == "resident";
            if leaf.kind == "copy" && resident || leaf.kind == "bcast_row" && !resident {
                return Err(plan_error(
                    ".row() broadcasts a resident row (a weight)".into(),
                ));
            }
        }
    }
    let mut ops = Vec::new();
    for &r in &reduces {
        ops.push((r, reduced(node(ir, r)?)?));
    }
    let mut body = Body::default();
    let mut unit = UnitProgram::new(&ir.name, "fp32");
    unit.operands = vec![Some(row)];
    let mut pairs = Vec::new();
    let last = reduces.iter().map(|&r| at(r)).max().unwrap_or(0);
    for l in 0..=last {
        let mine: Vec<(NodeId, &'static str)> =
            ops.iter().copied().filter(|&(r, _)| at(r) == l).collect();
        let done: Vec<NodeId> = stats.iter().copied().filter(|&p| at(p) == l).collect();
        let phase = Level {
            summed: &summed,
            made: &made,
            slot: &slot,
            cols: &cols,
        };
        pairs.extend(phase.run(&mut body, &mine, &done, width)?);
    }
    pairs.extend(apply_slots(&mut body, &stored, &slot, &cols, width)?);
    // each word marked, so that a pair's words are found again after the passes
    for (i, x) in body.code.iter_mut().enumerate() {
        x.mark = i + 1;
    }
    let done = fold_negation(&mut body.code)?;
    unit.passes.push(pass("algebra", "fold_negation", done));
    let done = fuse_mad(&mut body.code)?;
    unit.passes.push(pass("algebra", "fuse_mad", done));
    let levels = 1 + level.values().copied().max().unwrap_or(0);
    unit.passes.push(pass(
        "layout",
        "lanes",
        vec![format!(
            "{} sum(s) in {levels} pass(es) over the row; {} per-row value(s) in slots of \
             tile {rows_tile}; {fields} per-column value(s) in tile(s) from {cols_tile}",
            reduces.len(),
            held.len()
        )],
    ));
    check_lanes(&mut unit, &body.code, Layout::Rows(rows_tile, 0))?;
    let mut moved = 0;
    for &(start, end) in &pairs {
        let at: Vec<usize> = (0..body.code.len())
            .filter(|&i| (start + 1..=end).contains(&body.code[i].mark))
            .collect();
        if let (Some(&first), Some(&last)) = (at.first(), at.last()) {
            moved += interleave(&mut body.code, first, last + 1)?;
        }
    }
    unit.passes.push(pass(
        "schedule",
        "interleave",
        vec![format!("{moved} instructions moved in {}", pairs.len())],
    ));
    let layouts = Layouts {
        copy: Layout::Tile(0),
        rows: slot.iter().map(|(_, l)| *l).collect(),
        cols: cols.iter().map(|(_, l)| *l).collect(),
    };
    check_broadcast(&mut unit, &body.code, &layouts, width)?;
    let mut copies: Vec<String> = (0..width)
        .map(|t| format!("copy_tile(x, {t}, dst + {t})"))
        .collect();
    copies.extend((0..fields).map(|k| format!("copy_tile(cols{k}, 0, dst + {})", cols_tile + k)));
    finish(unit, body, width, &copies, false)
}

/// What a level's phase reads: each reduction's chain, each per-row value's, the
/// per-row slots and the resident rows' per-column tiles.
struct Level<'a> {
    summed: &'a [(NodeId, Chain)],
    made: &'a [(NodeId, Chain)],
    slot: &'a [(NodeId, Layout)],
    cols: &'a [(NodeId, Layout)],
}

impl Level<'_> {
    /// A level, per row group: the level's sums (`mine`, each with its reduction) over
    /// every vector of the row (a product into a multiply-add), their lanes summed by
    /// rotations, then the per-row values the level completes (`done`) and the kept sums
    /// stored to their slots. Returns each pair of groups' words (their range in the
    /// code as made).
    fn run(
        &self,
        b: &mut Body,
        mine: &[(NodeId, &'static str)],
        done: &[NodeId],
        width: i64,
    ) -> Result<Vec<(usize, usize)>, ChainError> {
        let chain_of = |of: &[(NodeId, Chain)], n: NodeId| {
            of.iter()
                .find(|(k, _)| *k == n)
                .map(|(_, c)| c.clone())
                .ok_or_else(|| ChainError::NotPorted("a per-row value of no chain".into()))
        };
        let (mut pairs, mut start) = (Vec::new(), 0);
        for (h, q) in GROUPS {
            if q % 2 == 0 {
                start = b.code.len();
            }
            // a per-row value of an earlier level, loaded once a group
            let kept = |n: NodeId| -> Result<Read, ChainError> {
                Ok(Read::Load(
                    format!("held{}", n.0),
                    layout_of(self.slot, n)?.address((0, h, 0, q, 0)),
                    format!("per-row slot, group {h} {q}"),
                ))
            };
            let mut cache = Vec::new();
            let mut accs: Vec<(NodeId, Arg)> = Vec::new();
            for (t, c, p) in vectors(width) {
                let at = (t, h, c, q, p);
                let address = Layout::Tile(0).address(at);
                for &(r, op) in mine {
                    let chain = chain_of(self.summed, r)?;
                    let mut reads = Vec::new();
                    for leaf in &chain.operands {
                        reads.push(match leaf.kind {
                            "bcast_col" => kept(source(leaf)?)?,
                            "bcast_row" => {
                                let n = source(leaf)?;
                                let at = layout_of(self.cols, n)?.address(at);
                                Read::Load(format!("col{}", n.0), at, format!("per-column {at}"))
                            }
                            _ => Read::Load("row".into(), address, format!("row {address}")),
                        });
                    }
                    let (vs, top) = result_of(&chain)?;
                    let top_node = vs
                        .get(top)
                        .ok_or_else(|| {
                            ChainError::NotPorted("a vector value of no function".into())
                        })?
                        .clone();
                    let acc = accs.iter().position(|(k, _)| *k == r);
                    let mad = top_node.kind == "mul" && op == "sum";
                    let made = match (acc, &top_node.args[..]) {
                        (Some(i), [x, m]) if mad => {
                            let mut memo = Vec::new();
                            let a = read_value(b, &vs, *x, reads.clone(), &mut cache, &mut memo)?;
                            let m = read_value(b, &vs, *m, reads, &mut cache, &mut memo)?;
                            let sum = accs[i].1.clone();
                            b.mad_of(a, m, sum, "sum += f", false, false, None)?
                        }
                        (acc, _) => {
                            let v = read_value(b, &vs, top, reads, &mut cache, &mut Vec::new())?;
                            match acc {
                                None => v,
                                Some(i) => fold(b, op, accs[i].1.clone(), v, "sum += f")?,
                            }
                        }
                    };
                    match acc {
                        Some(i) => accs[i].1 = made,
                        None => accs.push((r, made)),
                    }
                }
                drop_scope(&mut cache, "row");
                drop_scope(&mut cache, "col");
            }
            for &(r, op) in mine {
                let Some(i) = accs.iter().position(|(k, _)| *k == r) else {
                    return Err(ChainError::NotPorted("a row of no vector".into()));
                };
                let acc = lanes(b, op, accs[i].1.clone())?;
                accs[i].1 = acc.clone();
                // read by .col() itself, or by a later level's value
                if let Ok(slot) = layout_of(self.slot, r) {
                    let note = format!("per-row sum, group {h} {q}");
                    b.store(acc, slot.address((0, h, 0, q, 0)), &note)?;
                }
            }
            for &p in done {
                let Ok(chain) = chain_of(self.made, p) else {
                    continue; // a sum read by .col() itself: stored above
                };
                let mut reads = Vec::new();
                for leaf in &chain.operands {
                    let n = source(leaf)?;
                    reads.push(match accs.iter().find(|(k, _)| *k == n) {
                        Some((_, v)) => Read::Value(v.clone()),
                        None => kept(n)?,
                    });
                }
                let (vs, result) = result_of(&chain)?;
                let value = read_value(b, &vs, result, reads, &mut cache, &mut Vec::new())?;
                let note = format!("per-row value {h} {q}");
                b.store(
                    value,
                    layout_of(self.slot, p)?.address((0, h, 0, q, 0)),
                    &note,
                )?;
            }
            if q % 2 == 1 {
                pairs.push((start, b.code.len()));
            }
        }
        Ok(pairs)
    }
}

/// Per vector column (t, c, p) and pair of row groups: each per-row value loaded from
/// its slot once a group, each per-column vector once a vector column, the stored
/// result written over the row (today's `_apply_slots` of one result at tile 0).
/// Returns each pair's words (their range in the code as made).
fn apply_slots(
    b: &mut Body,
    stored: &Chain,
    slot: &[(NodeId, Layout)],
    cols: &[(NodeId, Layout)],
    width: i64,
) -> Result<Vec<(usize, usize)>, ChainError> {
    let (vs, result) = result_of(stored)?;
    let mut pairs = Vec::new();
    for h in 0..2 {
        for (t, c, p) in vectors(width) {
            let mut cache = Vec::new();
            for pair in [[0, 1], [2, 3]] {
                let start = b.code.len();
                for q in pair {
                    let at = (t, h, c, q, p);
                    let address = Layout::Tile(0).address(at);
                    let mut reads = Vec::new();
                    for leaf in &stored.operands {
                        reads.push(match leaf.kind {
                            "bcast_col" => {
                                let n = source(leaf)?;
                                Read::Load(
                                    format!("slot{}", n.0),
                                    layout_of(slot, n)?.address(at),
                                    format!("per-row slot {h} {q}"),
                                )
                            }
                            "bcast_row" => {
                                let n = source(leaf)?;
                                Read::Load(
                                    format!("col{}", n.0),
                                    layout_of(cols, n)?.address(at),
                                    format!("per-column {t} {c} {p}"),
                                )
                            }
                            _ => Read::Load("row".into(), address, format!("row {}", tuple(at))),
                        });
                    }
                    let made = read_value(b, &vs, result, reads, &mut cache, &mut Vec::new());
                    drop_scope(&mut cache, "row");
                    drop_scope(&mut cache, "slot");
                    let note = format!("row {} of tile 0", tuple(at));
                    b.store(made?, Layout::Tile(0).address(at), &note)?;
                }
                pairs.push((start, b.code.len()));
            }
        }
    }
    Ok(pairs)
}
