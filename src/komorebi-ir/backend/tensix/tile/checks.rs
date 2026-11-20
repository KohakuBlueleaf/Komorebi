//! The hazard checks of a lowered unit program (today's
//! `komorebi.dsl.tile.hazards.checks`): Dst window reads, the MAD readers, the
//! configuration NOPs, the constant registers and the saved register, lane
//! predication, the stall before the first load and every word's encoding. Two of them
//! repair the code (an SFPNOP, a STALLWAIT) and say so; the others refuse.

use std::collections::HashMap;

use super::unit::{
    encode, insns, Arg, Check, Insn, Item, Kind, UnitError, UnitProgram, PROGRAMMABLE, SAVED, WORK,
};

/// Dst address units of a tile in either Dst: 4 faces of 16 rows.
pub const TILE: i64 = 64;
/// The readers of a MAD-class result that the unit does not hold back (rule mad-reader:
/// the ISA's SFPMAD page; SFPGT, SFPLE from tt-metal's ckernel_sfpu_recip.h).
pub const NOT_HELD: [&str; 7] = [
    "SFPIADD", "SFPSHFT", "SFPAND", "SFPOR", "SFPSWAP", "SFPGT", "SFPLE",
];
/// SFPSHFT2 lane modes: their reads are not seen at all.
pub const SHFT2_LANES: [i64; 3] = [2, 3, 4];
/// STALLWAIT's resource: the vector unit stalls.
pub const STALL_SFPU: i64 = 0x100;
/// STALLWAIT's condition: until the matrix unit is done.
pub const WAIT_MATH: i64 = 0x10;
/// -1.0 in bf16: LReg 11's value, restored.
pub const NEG_ONE_BF16: i64 = 0xBF80;

fn at(insn: &Insn, field: &str) -> Arg {
    insn.arg(field).cloned().unwrap_or(Arg::Num(0))
}

fn num(insn: &Insn, field: &str) -> i64 {
    at(insn, field).num().unwrap_or(0)
}

/// Python's `repr` of a list of LReg numbers.
fn listed(values: &[i64]) -> String {
    let items: Vec<String> = values.iter().map(i64::to_string).collect();
    format!("[{}]", items.join(", "))
}

fn nop(note: &str) -> Result<Insn, UnitError> {
    Insn::new("SFPNOP", &[], note)
}

/// Every Insn of a loop nest in issue order, loop bodies once (today's `_order`).
pub fn order(items: &[Item]) -> Vec<&Insn> {
    insns(items)
}

/// The number of a register the code names (`L<n>`), refused where today's `int()`
/// raises.
pub fn named_register(s: &str) -> Result<i64, UnitError> {
    s.get(1..)
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| UnitError(format!("register name {s} names no register")))
}

/// {register: [(first, last)]}: where a register the code names (`L<n>`) holds a
/// value, from each write to its last read before the next write.
pub fn named_spans(code: &[Insn]) -> Result<HashMap<i64, Vec<(usize, usize)>>, UnitError> {
    let mut spans: HashMap<i64, Vec<(usize, usize)>> = HashMap::new();
    let work = i64::from(WORK);
    for (i, x) in code.iter().enumerate() {
        for v in x.reads() {
            if let Arg::Name(s) = &v {
                if s.starts_with('L') {
                    let r = named_register(s)?;
                    match spans.get_mut(&r) {
                        Some(list) if r < work && !list.is_empty() => {
                            if let Some(last) = list.last_mut() {
                                last.1 = i;
                            }
                        }
                        _ if r < work => {
                            spans.insert(r, vec![(i, i)]);
                        }
                        _ => {}
                    }
                }
            }
        }
        for v in x.writes()? {
            if let Arg::Name(s) = &v {
                if s.starts_with('L') {
                    let r = named_register(s)?;
                    if r < work {
                        let list = spans.entry(r).or_default();
                        if list.last().is_none_or(|l| l.1 != i) {
                            list.push((i, i));
                        }
                    }
                }
            }
        }
    }
    Ok(spans)
}

/// True when `insn` reads (or half-writes) a register of `written` the cycle after a
/// multiply-add and the unit does not stall it (the ISA's SFPMAD page).
pub fn not_held(insn: &Insn, written: &[Arg]) -> bool {
    let overlap = insn.reads().iter().any(|r| written.contains(r));
    let name = insn.op.name;
    if NOT_HELD.contains(&name) || name == "SFPLOADI" {
        return overlap;
    }
    if name == "SFPSHFT2" {
        return SHFT2_LANES.contains(&num(insn, "instr_mod1")) && overlap;
    }
    if matches!(name, "SFPSETSGN" | "SFPSETEXP" | "SFPSETMAN") {
        return num(insn, "instr_mod1") & 1 == 0 && written.contains(&at(insn, "lreg_dest"));
    }
    if name == "SFPCONFIG" {
        return written
            .iter()
            .any(|w| *w == Arg::Num(0) || *w == Arg::Name("L0".into()));
    }
    false
}

fn is_mad(item: &Item) -> Option<&Insn> {
    match item {
        Item::Insn(x) if x.op.kind == Kind::Mad => Some(x),
        _ => None,
    }
}

/// mad-reader: an SFPNOP between a multiply-add and a reader the unit does not hold
/// back, across the block's wrap too (a loop body's last instruction runs before its
/// first). Returns the repairs made.
pub fn check_mad_readers(code: &mut Vec<Item>) -> Result<usize, UnitError> {
    let mut repairs = 0;
    let mut i = 1;
    while i < code.len() {
        let written = match is_mad(&code[i - 1]) {
            Some(prev) => Some(prev.writes()?),
            None => None,
        };
        if let (Some(written), Item::Insn(cur)) = (written, &code[i]) {
            if not_held(cur, &written) {
                code.insert(i, Item::Insn(nop("hazard mad-reader: MAD result")?));
                repairs += 1;
            }
        }
        i += 1;
    }
    let wrap = match (code.last().and_then(is_mad), code.first()) {
        (Some(last), Some(Item::Insn(first))) => not_held(first, &last.writes()?),
        _ => false,
    };
    if wrap {
        code.push(Item::Insn(nop("hazard mad-reader: MAD result, the wrap")?));
        repairs += 1;
    }
    Ok(repairs)
}

/// config-nop: an SFPCONFIG writing LReg 11 to 14 is followed by an SFPNOP. Returns
/// the repairs made.
pub fn check_config_nop(code: &mut Vec<Item>) -> Result<usize, UnitError> {
    let mut repairs = 0;
    for i in (0..code.len()).rev() {
        let config = matches!(&code[i], Item::Insn(x)
            if x.op.name == "SFPCONFIG" && (11..=14).contains(&num(x, "config_dest")));
        if !config {
            continue;
        }
        let nop_after = matches!(code.get(i + 1), Some(Item::Insn(x)) if x.op.name == "SFPNOP");
        if !nop_after {
            code.insert(
                i + 1,
                Item::Insn(nop("hazard config-nop: after SFPCONFIG")?),
            );
            repairs += 1;
        }
    }
    Ok(repairs)
}

/// LReg 11 in the body's code `flat`: set before its first read, restored to -1.0
/// (SFPLOADI of 0xBF80 into LReg 0, then SFPCONFIG to 11) after its last. Returns the
/// check's text.
pub fn check_saved(flat: &[&Insn]) -> Result<String, UnitError> {
    let saved = Arg::Num(i64::from(SAVED));
    let sets: Vec<usize> = (0..flat.len())
        .filter(|&i| flat[i].op.name == "SFPCONFIG" && at(flat[i], "config_dest") == saved)
        .collect();
    let Some(&last) = sets.last() else {
        return Ok("LReg 11 not written".into());
    };
    let reads: Vec<usize> = (0..flat.len())
        .filter(|&i| flat[i].reads().contains(&saved))
        .collect();
    let loads: Vec<usize> = (0..last)
        .filter(|&i| flat[i].op.name == "SFPLOADI" && at(flat[i], "lreg_ind") == Arg::Num(0))
        .collect();
    let mut restore = false;
    if let Some(&load) = loads.last() {
        let form = (num(flat[load], "instr_mod0"), num(flat[load], "imm16"));
        let mut kept = true;
        for x in &flat[load + 1..last] {
            if x.writes()?.contains(&Arg::Num(0)) {
                kept = false;
            }
        }
        restore = form == (0, NEG_ONE_BF16) && kept;
    }
    if !restore || reads.last().is_some_and(|&r| r > last) {
        return Err(UnitError(
            "LReg 11 is read after its restore, or not restored to -1.0".into(),
        ));
    }
    let early = reads
        .first()
        .is_some_and(|&r| sets.first().is_some_and(|&s| r < s));
    if (!reads.is_empty() && sets.len() < 2) || early {
        return Err(UnitError("LReg 11 is read before the body sets it".into()));
    }
    Ok(format!(
        "LReg 11 set by the body, read {} times, restored to -1.0 after",
        reads.len()
    ))
}

fn check(rule: &str, text: &str, verdict: String) -> Check {
    Check {
        rule: rule.into(),
        text: text.into(),
        verdict,
    }
}

fn repaired(n: usize) -> String {
    if n == 0 {
        "ok".into()
    } else {
        format!("repaired: {n} SFPNOP")
    }
}

/// The per-row loop body of a unit's math program: the face loop's first item.
fn row_of(tile: &mut [Item]) -> Option<&mut Vec<Item>> {
    let faces = tile.iter_mut().find_map(|x| match x {
        Item::Loop(l) => Some(l),
        _ => None,
    })?;
    match faces.body.first_mut()? {
        Item::Loop(row) => Some(&mut row.body),
        _ => None,
    }
}

/// The hazard checks of `unit`'s math program, its per-row loop body (the face loop's
/// row loop) not yet replayed (today's `_checks`): each records its verdict, a repair
/// changes the code, a refusal is an error.
pub fn hazard_checks(unit: &mut UnitProgram) -> Result<(), UnitError> {
    let row = row_of(&mut unit.tile[1])
        .ok_or_else(|| UnitError(format!("{}: no per-row loop", unit.name)))?;
    let n = check_mad_readers(row)?;
    let mad = check(
        "mad-reader",
        "SFPNOP between a MAD and a reader not held back",
        repaired(n),
    );
    unit.checks.push(mad);
    let mut n = check_config_nop(&mut unit.init[1])?;
    if let Some(row) = row_of(&mut unit.tile[1]) {
        n += check_config_nop(row)?;
    }
    let config = check(
        "config-nop",
        "SFPNOP after an SFPCONFIG of LReg 11 to 14",
        repaired(n),
    );
    unit.checks.push(config);
    let saved = check_saved(&order(&unit.tile[1]))?;
    let mut written: Vec<i64> = insns(&unit.init[1])
        .iter()
        .filter(|x| x.op.name == "SFPCONFIG")
        .map(|x| num(x, "config_dest"))
        .filter(|r| (11..=14).contains(r))
        .collect();
    written.sort_unstable();
    let mut read: Vec<i64> = insns(&unit.tile[1])
        .iter()
        .flat_map(|x| x.reads())
        .filter_map(|r| r.num())
        .filter(|r| PROGRAMMABLE.iter().any(|p| i64::from(*p) == *r))
        .collect();
    read.sort_unstable();
    read.dedup();
    if read.iter().any(|r| !written.contains(r)) {
        return Err(UnitError(format!(
            "LReg {} read but not set by the setup",
            listed(&read)
        )));
    }
    if written.contains(&i64::from(SAVED)) {
        return Err(UnitError(
            "LReg 11 is written by the setup: it is set and restored by the body".into(),
        ));
    }
    unit.clobbers = written.iter().map(|r| format!("LReg{r}")).collect();
    unit.checks.push(check(
        "constants",
        "constant registers set before use, LReg 11 restored to -1.0",
        format!(
            "ok: LReg {} set by the setup and declared clobbered; {saved}",
            listed(&written)
        ),
    ));
    let mut depth = 0;
    if let Some(row) = row_of(&mut unit.tile[1]) {
        for x in insns(row) {
            if x.sets_flags() {
                depth = if x.op.name == "SFPENCC" { 0 } else { depth + 1 };
            }
        }
    }
    if depth != 0 {
        return Err(UnitError(
            "a condition is open at the end of the row code".into(),
        ));
    }
    unit.checks.push(check(
        "predication",
        "lane predication restored (every lane on) at the end of the code",
        "ok".into(),
    ));
    let flat = order(&unit.tile[1]);
    let first = flat
        .iter()
        .position(|x| matches!(x.op.name, "SFPLOAD" | "SFPLOADMACRO"))
        .ok_or_else(|| UnitError(format!("{}: the vector code loads nothing", unit.name)))?;
    let stalled = flat[..first]
        .iter()
        .any(|x| x.op.name == "STALLWAIT" && num(x, "stall_res") & STALL_SFPU != 0);
    if !stalled {
        let args = [
            ("stall_res", Arg::Num(STALL_SFPU)),
            ("wait_res", Arg::Num(WAIT_MATH)),
        ];
        let stall = Insn::new("STALLWAIT", &args, "SFPU waits for the matrix unit")?;
        unit.tile[1].insert(0, Item::Insn(stall));
    }
    unit.checks.push(check(
        "dst-load",
        "matrix-unit writes to Dst finished before the first SFPLOAD",
        if stalled {
            "ok: STALLWAIT(SFPU on MATH) first".into()
        } else {
            "repaired: STALLWAIT".into()
        },
    ));
    let mut built = 0;
    for x in insns(&unit.init[1]).into_iter().chain(insns(&unit.tile[1])) {
        let mut args = Vec::new();
        for ((field, _), value) in x.op.args.iter().zip(&x.args) {
            let n = value
                .num()
                .ok_or_else(|| UnitError(format!("{} {field}={value}: not a number", x.op.name)))?;
            args.push((*field, n));
        }
        encode(x.op.name, &args)?;
        if !x.runtime.is_empty() {
            built += 1;
        }
    }
    let verdict = if built > 0 {
        format!(
            "ok: every word encoded; {built} immediate(s) of run-time scalars built on the core (TT_SFPLOADI of a 16-bit half)"
        )
    } else {
        "ok: every word encoded (all immediates are compile-time)".into()
    };
    unit.checks.push(check(
        "immediates",
        "every immediate and register field inside its width",
        verdict,
    ));
    Ok(())
}

/// dst-window: the vector code's loads and stores address Dst tiles below `tiles`,
/// those its caller gives a call (a walk of rows covers the one tile its address
/// names: 4 faces of 16 rows). Refused past them.
pub fn check_dst_window(unit: &mut UnitProgram, tiles: i64) -> Result<(), UnitError> {
    let mut items: Vec<&Item> = unit.tile[1].iter().collect();
    for (_, part) in &unit.parts {
        items.extend(part.iter());
    }
    let owned: Vec<Item> = items.into_iter().cloned().collect();
    let highest = insns(&owned)
        .iter()
        .filter(|x| matches!(x.op.name, "SFPLOAD" | "SFPSTORE" | "SFPLOADMACRO"))
        .map(|x| num(x, "dest_reg_addr"))
        .max()
        .unwrap_or(-1);
    let reach = highest.div_euclid(TILE) + 1;
    if reach > tiles {
        return Err(UnitError(format!(
            "{}: the vector code reaches Dst tile {}; a call is given tiles 0 .. {}",
            unit.name,
            reach - 1,
            tiles - 1
        )));
    }
    unit.checks.push(check(
        "dst-window",
        "Dst accesses inside a call's tiles",
        format!("{reach} of {tiles}"),
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::tensix::tile::unit::{op_named, Count, Loop};

    /// `op` with its fields 0 but `set`.
    fn word(op: &str, set: &[(&str, i64)]) -> Insn {
        let fields = op_named(op).map_or(&[][..], |o| o.args);
        let args: Vec<(&str, Arg)> = fields
            .iter()
            .map(|(n, _)| {
                let v = set.iter().find(|(s, _)| s == n).map_or(0, |(_, v)| *v);
                (*n, Arg::Num(v))
            })
            .collect();
        Insn::new(op, &args, "").expect("a word")
    }

    fn mad(dest: i64) -> Insn {
        let f = [
            ("lreg_src_a", 3),
            ("lreg_src_b", 4),
            ("lreg_src_c", 9),
            ("lreg_dest", dest),
        ];
        word("SFPMAD", &f)
    }

    fn iadd(read: i64) -> Insn {
        word(
            "SFPIADD",
            &[("lreg_c", read), ("lreg_dest", 5), ("instr_mod1", 5)],
        )
    }

    fn notes(code: &[Item]) -> Vec<String> {
        insns(code)
            .iter()
            .map(|x| format!("{}:{}", x.op.name, x.note))
            .collect()
    }

    // Each test's expectations are today's (hazards/checks.py) on the same words.

    #[test]
    fn a_reader_not_held_back_gets_an_sfpnop_and_so_does_the_wrap() {
        let mov = word("SFPMOV", &[("lreg_c", 2), ("lreg_dest", 7)]);
        let mut code: Vec<Item> = [iadd(2), mad(1), iadd(1), mad(2), mov]
            .into_iter()
            .map(Item::Insn)
            .collect();
        assert_eq!(check_mad_readers(&mut code), Ok(1));
        let want = [
            "SFPIADD:",
            "SFPMAD:",
            "SFPNOP:hazard mad-reader: MAD result",
            "SFPIADD:",
            "SFPMAD:",
            "SFPMOV:",
        ];
        assert_eq!(notes(&code), want);
        let mut code = vec![Item::Insn(iadd(2)), Item::Insn(mad(2))];
        assert_eq!(check_mad_readers(&mut code), Ok(1));
        assert_eq!(
            notes(&code),
            [
                "SFPIADD:",
                "SFPMAD:",
                "SFPNOP:hazard mad-reader: MAD result, the wrap"
            ]
        );
    }

    #[test]
    fn what_the_unit_holds_back() {
        let one = [Arg::Num(1)];
        let sgn = |m| {
            word(
                "SFPSETSGN",
                &[("lreg_c", 3), ("lreg_dest", 1), ("instr_mod1", m)],
            )
        };
        assert!(not_held(&sgn(0), &one));
        assert!(!not_held(&sgn(1), &one));
        assert!(not_held(
            &word("SFPCONFIG", &[("config_dest", 12)]),
            &[Arg::Num(0)]
        ));
        let shft2 = [("lreg_src_c", 1), ("lreg_dest", 1), ("instr_mod1", 3)];
        assert!(not_held(&word("SFPSHFT2", &shft2), &one));
        assert!(!not_held(
            &word("SFPMOV", &[("lreg_c", 1), ("lreg_dest", 2)]),
            &one
        ));
    }

    #[test]
    fn lreg_11_is_set_before_its_reads_and_restored_after() {
        let ld = |imm| word("SFPLOADI", &[("lreg_ind", 0), ("imm16", imm)]);
        let cfg = || word("SFPCONFIG", &[("config_dest", 11)]);
        let read = || {
            let f = [
                ("lreg_src_a", 11),
                ("lreg_src_b", 1),
                ("lreg_src_c", 9),
                ("lreg_dest", 2),
            ];
            word("SFPMAD", &f)
        };
        let clobber = word("SFPMOV", &[("lreg_c", 1), ("lreg_dest", 0)]);
        let restored = "LReg 11 is read after its restore, or not restored to -1.0";
        let cases: Vec<(Vec<Insn>, Result<&str, &str>)> = vec![
            (
                vec![ld(0x3F00), cfg(), read(), ld(0xBF80), cfg()],
                Ok("LReg 11 set by the body, read 1 times, restored to -1.0 after"),
            ),
            (vec![read()], Ok("LReg 11 not written")),
            (vec![ld(0x3F00), cfg(), read()], Err(restored)),
            (
                vec![read(), ld(0x3F00), cfg(), ld(0xBF80), cfg()],
                Err("LReg 11 is read before the body sets it"),
            ),
            (
                vec![ld(0x3F00), cfg(), read(), ld(0xBF80), clobber, cfg()],
                Err(restored),
            ),
        ];
        for (code, want) in cases {
            let flat: Vec<&Insn> = code.iter().collect();
            let got = check_saved(&flat).map_err(|e| e.0);
            assert_eq!(got, want.map(str::to_string).map_err(str::to_string));
        }
    }

    #[test]
    fn the_vector_code_stays_inside_its_calls_tiles() {
        let mut unit = UnitProgram::new("w", "fp32");
        let store = word("SFPSTORE", &[("dest_reg_addr", 130)]);
        unit.tile[1] = vec![
            Item::Insn(word("SFPLOAD", &[])),
            Item::Loop(Loop {
                name: "x".into(),
                count: Count::Times(2),
                body: vec![Item::Insn(store)],
                unroll: false,
            }),
        ];
        let refused = check_dst_window(&mut unit, 2).map_err(|e| e.0);
        let want = "w: the vector code reaches Dst tile 2; a call is given tiles 0 .. 1";
        assert_eq!(refused, Err(want.to_string()));
        assert_eq!(check_dst_window(&mut unit, 3), Ok(()));
        unit.tile[1].clear();
        assert_eq!(check_dst_window(&mut unit, 1), Ok(()));
        let verdicts: Vec<&str> = unit.checks.iter().map(|c| c.verdict.as_str()).collect();
        assert_eq!(verdicts, ["3 of 3", "0 of 1"]);
    }
}
