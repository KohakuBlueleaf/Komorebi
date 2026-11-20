//! Schedule rule "load-macro chain" (today's `komorebi.dsl.tile.macro`): a row of
//! loads, at most one operation a sub-unit (simple, MAD, round) and one store of the
//! result where an operand was loaded becomes one SFPLOADMACRO a row, beside plain
//! loads of the others. A row that does not fit is refused with the reason.
//!
//! The checks of the back door and of the templates run at the emit. Not ported: the
//! max under a load macro (`max_macro`, `macro_setup`); the stream form's unit
//! program does not use it.

use std::collections::BTreeSet;

use super::unit::{insns, op_named, Arg, Insn, Item, UnitError, UnitProgram, MATH};

/// The sub-unit each instruction a macro slot may arm runs on (the ISA's allocation
/// table); flag and lane-movement instructions are left out (see `fit`).
pub const UNITS: [(&str, &[&str]); 3] = [
    (
        "simple",
        &[
            "SFPABS",
            "SFPAND",
            "SFPARECIP",
            "SFPCAST",
            "SFPDIVP2",
            "SFPEXEXP",
            "SFPEXMAN",
            "SFPIADD",
            "SFPLZ",
            "SFPMOV",
            "SFPNOT",
            "SFPOR",
            "SFPSETEXP",
            "SFPSETMAN",
            "SFPSETSGN",
            "SFPSHFT",
            "SFPXOR",
        ],
    ),
    ("mad", &["SFPADD", "SFPADDI", "SFPMAD", "SFPMUL", "SFPMULI"]),
    ("round", &["SFP_STOCH_RND", "SFPSHFT2"]),
];
/// A sequence word holds one byte a slot, from bit 0: simple, MAD, round, store.
pub const SLOT_SHIFT: [(&str, u32); 4] = [("simple", 0), ("mad", 8), ("round", 16), ("store", 24)];
/// Slot selector: SFPSTORE of the loaded register.
pub const STORE_SLOT: i64 = 3;
/// Selectors 4 .. 7: instruction template 0 .. 3.
pub const TEMPLATE_SLOT: i64 = 4;
/// The loaded register replaces VB (else VC).
pub const PATCH_VB: i64 = 0x80;
/// Sequences, and instruction templates.
pub const MACROS: i64 = 4;
/// A word whose VD is 12 + k stores instruction template k.
pub const BACK_DOOR: i64 = 12;
/// The latest a macro's store may come after its load.
pub const MAX_DELAY: i64 = 7;
/// The loaded register is LReg 0 .. 3 (the macro's two low VD bits).
pub const LOADED: i64 = 4;
/// Plain-loaded operands rotate in LReg 4 .. 7.
pub const FIRST_PLAIN: i64 = 4;
/// VA * VB: the operands may swap.
pub const COMMUTES: [&str; 3] = ["SFPMAD", "SFPMUL", "SFPADD"];

/// Why a row is not one macro: today's `Refused` (a pass's record), or today's
/// failure that is no refusal, refused by name.
#[derive(Clone, Debug, PartialEq)]
pub enum NoFit {
    /// The rule does not take the row, and why.
    Refused(String),
    /// Today raises.
    Error(UnitError),
}

impl From<UnitError> for NoFit {
    fn from(e: UnitError) -> NoFit {
        NoFit::Error(e)
    }
}

/// One instruction of a macro's chain: the instruction, its sub-unit, the field the
/// loaded register patches, its delay after the load.
#[derive(Clone, Debug)]
pub struct Link {
    /// The instruction (its operands swapped when the chain value was VA).
    pub insn: Insn,
    /// Its sub-unit.
    pub unit: &'static str,
    /// The field patched.
    pub slot: &'static str,
    /// Its delay.
    pub delay: i64,
}

/// A row as a macro: the load it carries, the other loads, the chain, the store's
/// delay, the rotations of the loaded register and of the plain-loaded ones.
#[derive(Clone, Debug)]
pub struct Plan {
    /// The load the macro carries.
    pub carried: Insn,
    /// The other loads.
    pub others: Vec<Insn>,
    /// The chain.
    pub chain: Vec<Link>,
    /// The store's delay.
    pub store_delay: i64,
    /// Registers the loaded value rotates over.
    pub loaded_turns: i64,
    /// Registers each plain-loaded operand rotates over.
    pub plain_turns: i64,
}

impl Plan {
    /// Rows in one turn of every rotation.
    pub fn rows(&self) -> i64 {
        let (a, b) = (self.loaded_turns, self.plain_turns);
        let (mut x, mut y) = (a, b);
        while y != 0 {
            (x, y) = (y, x % y);
        }
        if x == 0 {
            0
        } else {
            a / x * b
        }
    }
}

fn arg(insn: &Insn, name: &str) -> Arg {
    insn.arg(name).cloned().unwrap_or(Arg::Num(0))
}

/// Python's `repr` of a list of register arguments.
fn listed(values: &[Arg]) -> String {
    let items: Vec<String> = values
        .iter()
        .map(|v| match v {
            Arg::Num(n) => n.to_string(),
            Arg::Name(s) => super::py::str_repr(s),
        })
        .collect();
    format!("[{}]", items.join(", "))
}

fn latency(insn: &Insn, unit: &str) -> i64 {
    if unit == "mad" {
        return 2;
    }
    if insn.op.name == "SFPSHFT2" && matches!(arg(insn, "instr_mod1").num(), Some(2..=4)) {
        return 2;
    }
    1
}

/// The load-macro plan of allocated row code `row` (LReg numbers), or why not; like
/// today's, a product whose chain value is VA has its operands swapped in `row` even
/// when the row is refused after.
pub fn fit(row: &mut [Insn]) -> Result<Plan, NoFit> {
    let refused = |s: String| NoFit::Refused(s);
    let loads: Vec<usize> = (0..row.len())
        .filter(|&i| row[i].op.name == "SFPLOAD")
        .collect();
    let stores: Vec<usize> = (0..row.len())
        .filter(|&i| row[i].op.name == "SFPSTORE")
        .collect();
    let work: Vec<usize> = (0..row.len())
        .filter(|&i| !matches!(row[i].op.name, "SFPLOAD" | "SFPSTORE"))
        .collect();
    if stores.len() != 1 {
        return Err(refused(format!(
            "the row stores {} values; the store slot stores one",
            stores.len()
        )));
    }
    let store = row[stores[0]].clone();
    let address = arg(&store, "dest_reg_addr");
    let Some(&carried) = loads
        .iter()
        .find(|&&i| arg(&row[i], "dest_reg_addr") == address)
    else {
        return Err(refused(
            "the result goes where no operand was loaded; the macro's store writes the row its load read"
                .into(),
        ));
    };
    let mut plan = Plan {
        carried: row[carried].clone(),
        others: loads
            .iter()
            .filter(|&&i| i != carried)
            .map(|&i| row[i].clone())
            .collect(),
        chain: Vec::new(),
        store_delay: 0,
        loaded_turns: 1,
        plain_turns: 1,
    };
    let plain: Vec<Arg> = plan.others.iter().map(|x| arg(x, "lreg_ind")).collect();
    if work.is_empty() {
        return Err(refused("the row computes nothing".into()));
    }
    let of: Vec<Option<&'static str>> = work
        .iter()
        .map(|&i| {
            UNITS
                .iter()
                .find(|(_, ops)| ops.contains(&row[i].op.name))
                .map(|(u, _)| *u)
        })
        .collect();
    let count = |u: &str| of.iter().filter(|x| **x == Some(u)).count();
    if work.len() > 3 || UNITS.iter().any(|(u, _)| count(u) > 1) {
        return Err(refused(format!(
            "the row computes with {} instructions ({} MAD-class); a macro arms at most one a sub-unit (simple, MAD, round)",
            work.len(),
            count("mad")
        )));
    }
    let mut value = arg(&plan.carried, "lreg_ind");
    let mut delay = 0;
    for (&i, unit) in work.iter().zip(&of) {
        let insn = &row[i];
        let name = insn.op.name;
        if insn.sets_flags() || matches!(name, "SFPENCC" | "SFPGT" | "SFPSETCC") {
            return Err(refused(format!(
                "{name} sets lane flags; a macro's slots run unguarded"
            )));
        }
        let Some(unit) = *unit else {
            return Err(refused(format!(
                "{name} is not an instruction a macro slot arms"
            )));
        };
        if name == "SFPSHFT2" && matches!(arg(insn, "instr_mod1").num(), Some(0..=2)) {
            return Err(refused(
                "an SFPSHFT2 register shuffle touches LReg 0 .. 3".into(),
            ));
        }
        let Some(&dest) = insn.op.writes.first() else {
            return Err(NoFit::Error(UnitError(format!(
                "{name} writes no register"
            ))));
        };
        if insn.read_fields().contains(&dest) {
            return Err(refused(format!(
                "{name} reads its destination, which a template fixes"
            )));
        }
        let slot = patched(&mut row[i], &value)?;
        let insn = &row[i];
        let others: Vec<Arg> = insn.reads().into_iter().filter(|r| *r != value).collect();
        for r in &others {
            match r {
                Arg::Num(n) if !plain.contains(r) && *n < 9 => {
                    return Err(refused(format!(
                        "{name} reads LReg {}: a template's other registers are constants or a plain load of the same row",
                        listed(&others)
                    )))
                }
                Arg::Num(_) => {}
                Arg::Name(s) if !plain.contains(r) => {
                    // today compares a name with a number and raises
                    return Err(NoFit::Error(UnitError(format!(
                        "{name} reads {s}, a name, after the layout pass"
                    ))));
                }
                Arg::Name(_) => {}
            }
        }
        plan.chain.push(Link {
            insn: insn.clone(),
            unit,
            slot,
            delay,
        });
        delay += latency(insn, unit);
        value = arg(insn, dest);
    }
    if arg(&store, "lreg_ind") != value {
        return Err(refused(
            "the row stores a value other than the chain's last".into(),
        ));
    }
    plan.store_delay = delay;
    if delay > MAX_DELAY {
        return Err(refused(format!(
            "the store comes {delay} cycles after the load; at most 7"
        )));
    }
    rotations(&mut plan)?;
    Ok(plan)
}

/// The field the loaded register patches (VC, or VB with PATCH_VB); swaps a product's
/// operands when the chain value is VA. Refused when neither fits.
fn patched(insn: &mut Insn, value: &Arg) -> Result<&'static str, NoFit> {
    let name = insn.op.name;
    if insn.reads().iter().filter(|r| *r == value).count() != 1 {
        return Err(NoFit::Refused(format!(
            "{name} does not read the previous result exactly once"
        )));
    }
    if let (Some(a), Some(b)) = (
        insn.arg("lreg_src_a").cloned(),
        insn.arg("lreg_src_b").cloned(),
    ) {
        if a == *value {
            if !COMMUTES.contains(&name) {
                return Err(NoFit::Refused(format!(
                    "{name} reads the previous result as VA"
                )));
            }
            insn.set("lreg_src_a", b);
            insn.set("lreg_src_b", value.clone());
        }
        if arg(insn, "lreg_src_b") == *value {
            return Ok("lreg_src_b");
        }
        return Ok("lreg_src_c");
    }
    if insn.arg("lreg_c") == Some(value) {
        return Ok("lreg_c");
    }
    Err(NoFit::Refused(format!(
        "{name} reads the previous result in no patched field"
    )))
}

/// How many registers the loaded value and each plain-loaded operand rotate over, so
/// that no load overwrites a register a scheduled instruction has yet to read: a row
/// issues 1 + len(others) instructions.
fn rotations(plan: &mut Plan) -> Result<(), NoFit> {
    let width = 1 + plan.others.len() as i64;
    let mut turns = 1;
    while turns * width <= plan.store_delay + 1 {
        turns += 1;
    }
    plan.loaded_turns = turns;
    let plain: Vec<Arg> = plan.others.iter().map(|x| arg(x, "lreg_ind")).collect();
    let last_read = plan
        .chain
        .iter()
        .filter(|l| l.insn.reads().iter().any(|r| plain.contains(r)))
        .map(|l| l.delay)
        .max();
    let mut turns = 1;
    if let Some(last) = last_read {
        while turns * width <= width + last {
            turns += 1;
        }
    }
    plan.plain_turns = turns;
    let templates = plan.chain.len() as i64 * plan.plain_turns;
    if plan.loaded_turns > LOADED || plan.others.len() as i64 * turns > 8 - FIRST_PLAIN {
        return Err(NoFit::Refused(
            "the rotation needs more registers than the macro can name".into(),
        ));
    }
    if templates > MACROS || plan.plain_turns > MACROS {
        return Err(NoFit::Refused(format!(
            "{templates} instruction templates; the unit holds 4"
        )));
    }
    Ok(())
}

fn shift(slot: &str) -> u32 {
    SLOT_SHIFT
        .iter()
        .find(|(s, _)| *s == slot)
        .map_or(0, |(_, n)| *n)
}

fn word(name: &str, args: &[(&str, i64)], note: &str) -> Result<Insn, UnitError> {
    let args: Vec<(&str, Arg)> = args.iter().map(|(n, v)| (*n, Arg::Num(*v))).collect();
    Insn::new(name, &args, note)
}

fn nop(note: &str) -> Result<Insn, UnitError> {
    word("SFPNOP", &[], note)
}

/// `insn`'s operation with its fields, `changed` set over them, and `note`.
fn remade(insn: &Insn, changed: &[(&str, Arg)], note: &str) -> Result<Insn, UnitError> {
    let mut args: Vec<(&str, Arg)> = insn
        .op
        .args
        .iter()
        .map(|(n, _)| (*n, arg(insn, n)))
        .collect();
    for (name, value) in changed {
        if let Some(slot) = args.iter_mut().find(|(n, _)| n == name) {
            slot.1 = value.clone();
        }
    }
    Insn::new(insn.op.name, &args, note)
}

fn registers(plan: &Plan, variant: i64) -> Vec<i64> {
    (0..plan.others.len() as i64)
        .map(|k| FIRST_PLAIN + k * plan.plain_turns + variant)
        .collect()
}

/// The instructions that write the templates (through the back door: VD 12 + k stores
/// the word), the sequences and the store format, once per stream; LaneConfig must
/// leave the back door open (the setup's first SFPCONFIG writes it 0).
pub fn setup(plan: &Plan) -> Result<Vec<Insn>, UnitError> {
    let mut out = vec![nop("LaneConfig settles")?, nop("")?];
    for v in 0..plan.plain_turns {
        let mut sequence = (STORE_SLOT | plan.store_delay << 3) << shift("store");
        for (k, link) in plan.chain.iter().enumerate() {
            let index = k as i64 * plan.plain_turns + v;
            let insn = &link.insn;
            let op = op_named(insn.op.name)
                .ok_or_else(|| UnitError(format!("no operation {}", insn.op.name)))?;
            let vd = *op
                .writes
                .first()
                .ok_or_else(|| UnitError(format!("{} writes no register", op.name)))?;
            let mut changed = vec![(link.slot, Arg::Num(0)), (vd, Arg::Num(BACK_DOOR + index))];
            for (x, reg) in plan.others.iter().zip(registers(plan, v)) {
                for (name, _) in op.args {
                    // VD holds the back-door index, not an operand to rename
                    if *name != link.slot && *name != vd && arg(insn, name) == arg(x, "lreg_ind") {
                        changed.push((name, Arg::Num(reg)));
                    }
                }
            }
            out.push(remade(
                insn,
                &changed,
                &format!("template {index}: {}", insn.note),
            )?);
            let mut bits = (TEMPLATE_SLOT + index) | (link.delay << 3);
            if link.slot == "lreg_src_b" {
                bits |= PATCH_VB;
            }
            sequence |= bits << shift(link.unit);
        }
        out.push(word(
            "SFPLOADI",
            &[
                ("lreg_ind", 0),
                ("instr_mod0", 10),
                ("imm16", sequence & 0xFFFF),
            ],
            &format!("sequence {v} low"),
        )?);
        out.push(word(
            "SFPLOADI",
            &[
                ("lreg_ind", 0),
                ("instr_mod0", 8),
                ("imm16", sequence >> 16),
            ],
            &format!("sequence {v} high"),
        )?);
        out.push(word(
            "SFPCONFIG",
            &[("imm16_math", 0), ("config_dest", 4 + v), ("instr_mod1", 0)],
            &format!("sequence {v}"),
        )?);
        out.push(nop("")?);
        out.push(nop("")?);
    }
    let mask = ((1 << plan.plain_turns) - 1) << 4;
    out.push(word(
        "SFPCONFIG",
        &[("imm16_math", mask), ("config_dest", 8), ("instr_mod1", 1)],
        "macro stores use their load's format",
    )?);
    out.push(nop("")?);
    out.push(nop("")?);
    Ok(out)
}

/// The code of `plan.rows()` rows: each row's plain loads, then its macro, whose
/// address mode `addr_mod` steps Dst to the next row group.
pub fn rows(plan: &Plan, addr_mod: i64) -> Result<Vec<Insn>, UnitError> {
    let mut out = Vec::new();
    for r in 0..plan.rows() {
        let (v, loaded) = (r % plan.plain_turns, r % plan.loaded_turns);
        for (x, reg) in plan.others.iter().zip(registers(plan, v)) {
            out.push(remade(x, &[("lreg_ind", Arg::Num(reg))], &x.note)?);
        }
        let carried = &plan.carried;
        let changed = [
            ("lreg_ind", Arg::Num(v << 2 | loaded)),
            ("sfpu_addr_mode", Arg::Num(addr_mod)),
        ];
        let made = remade(carried, &changed, &format!("macro {v}: load, chain, store"))?;
        let args: Vec<(&str, Arg)> = made
            .op
            .args
            .iter()
            .map(|(n, _)| (*n, arg(&made, n)))
            .collect();
        out.push(Insn::new("SFPLOADMACRO", &args, &made.note)?);
    }
    Ok(out)
}

/// SFPNOPs after the last macro until its store has run.
pub fn drain(plan: &Plan) -> Result<Vec<Insn>, UnitError> {
    (0..plan.store_delay + 2)
        .map(|_| nop("load-macro drain"))
        .collect()
}

/// Field `field` of `x` as a number (none when it has no such field), or a register's
/// name that today compares with a number, refused by name.
fn number(x: &Insn, field: &str) -> Result<Option<i64>, UnitError> {
    match x.arg(field) {
        None => Ok(None),
        Some(Arg::Num(n)) => Ok(Some(*n)),
        Some(value) => Err(UnitError(format!(
            "{} {field}={value}: not a number",
            x.op.name
        ))),
    }
}

/// The field a vector word's VD is in: `lreg_dest`, else `lreg_ind`, if it has one.
fn vd_field(x: &Insn) -> Option<&'static str> {
    ["lreg_dest", "lreg_ind"]
        .into_iter()
        .find(|n| x.arg(n).is_some())
}

/// (k, word) for each word of `items` (loops looked into) that writes VD 12 + k: on
/// the vector unit such a word stores instruction template k, whatever else it does
/// (SFPCONFIG writes the constant registers by its own field).
pub fn template_writes(items: &[Item]) -> Result<Vec<(i64, &Insn)>, UnitError> {
    let mut out = Vec::new();
    for x in insns(items) {
        if !x.op.name.starts_with("SFP") || x.op.name == "SFPCONFIG" {
            continue;
        }
        let Some(name) = vd_field(x) else { continue };
        if !x.op.writes.contains(&name) {
            continue;
        }
        if let Some(v) = number(x, name)? {
            if (BACK_DOOR..BACK_DOOR + MACROS).contains(&v) {
                out.push((v - BACK_DOOR, x));
            }
        }
    }
    Ok(out)
}

/// Each macro sequence set and the templates it runs (none: a word not known).
type Sequences = Vec<(i64, Option<BTreeSet<i64>>)>;

/// {sequence: the templates it runs} as `init` writes them (SFPCONFIG 4 + k from
/// LReg 0, its halves from SFPLOADI modes 10 and 8); none for a word not known.
fn sequences(init: &[Item]) -> Result<Sequences, UnitError> {
    let (mut low, mut high) = (None, None);
    let mut out: Sequences = Vec::new();
    for x in insns(init) {
        let name = x.op.name;
        if name == "SFPLOADI" && x.arg("lreg_ind") == Some(&Arg::Num(0)) {
            match number(x, "instr_mod0")? {
                Some(10) => low = number(x, "imm16")?,
                Some(8) => high = number(x, "imm16")?,
                _ => (low, high) = (None, None),
            }
        } else if x.writes()?.contains(&Arg::Num(0)) {
            (low, high) = (None, None);
        } else if name == "SFPCONFIG" && number(x, "instr_mod1")? == Some(0) {
            let k = number(x, "config_dest")?.unwrap_or(0) - 4;
            if !(0..MACROS).contains(&k) {
                continue;
            }
            let runs = match (low, high) {
                (Some(low), Some(high)) => {
                    let word = high << 16 | low;
                    let slots = ["simple", "mad", "round"].map(|u| word >> shift(u) & 7);
                    Some(
                        slots
                            .iter()
                            .filter(|s| **s >= TEMPLATE_SLOT)
                            .map(|s| s - TEMPLATE_SLOT)
                            .collect(),
                    )
                }
                _ => None,
            };
            match out.iter_mut().find(|(j, _)| *j == k) {
                Some(entry) => entry.1 = runs,
                None => out.push((k, runs)),
            }
        }
    }
    Ok(out)
}

/// Python's `repr` of a list of numbers.
fn numbers(values: &[i64]) -> String {
    let items: Vec<String> = values.iter().map(i64::to_string).collect();
    format!("[{}]", items.join(", "))
}

/// The math CPU's setup and its later functions: the tile's code and the parts.
fn math(unit: &UnitProgram) -> (&[Item], Vec<&[Item]>) {
    let mut tile: Vec<&[Item]> = vec![&unit.tile[MATH]];
    tile.extend(unit.parts.iter().map(|(_, items)| items.as_slice()));
    (&unit.init[MATH], tile)
}

/// Rule "back door": the init leaves LaneConfig's back door open, so a vector word
/// with VD 12 + k (but SFPCONFIG and SFPLOADMACRO) stores template k and runs
/// nothing; only the init's template stores have one. The check's text, or the
/// refusal.
pub fn check_back_door(unit: &UnitProgram) -> Result<String, UnitError> {
    let mut stores = 0;
    let (init, tile) = math(unit);
    for (at_init, functions) in [(true, vec![init]), (false, tile)] {
        for x in functions.into_iter().flat_map(insns) {
            let name = x.op.name;
            if !name.starts_with("SFP") || matches!(name, "SFPCONFIG" | "SFPLOADMACRO") {
                continue;
            }
            let Some(Arg::Num(v)) = vd_field(x).and_then(|f| x.arg(f)) else {
                continue;
            };
            if *v < BACK_DOOR {
                continue;
            }
            if at_init && x.note.starts_with("template") {
                stores += 1;
                continue;
            }
            return Err(UnitError(format!(
                "{}: {name} ({}) has VD {v}: with the back door open it stores template {} and runs nothing (a VD of 12 or more only in a macro's template store)",
                unit.name,
                x.note,
                v - BACK_DOOR
            )));
        }
    }
    Ok(format!(
        "ok: a VD of 12 or more only in the init's {stores} template store(s)"
    ))
}

/// Rule "macro templates": each template a macro of `unit` runs is the last word of
/// its init to write VD 12 + k and is setup's, and no word of the tile's code (it
/// repeats a tile: the next tile's macros would run that word) writes it. The
/// check's text, or the refusal.
pub fn check_templates(unit: &UnitProgram) -> Result<String, UnitError> {
    let (init, tile) = math(unit);
    let mut used = BTreeSet::new();
    for x in tile.iter().flat_map(|t| insns(t)) {
        if x.op.name == "SFPLOADMACRO" {
            let ind = number(x, "lreg_ind")?.unwrap_or(0);
            used.insert(ind >> 2);
        }
    }
    let used: Vec<i64> = used.into_iter().collect();
    if used.is_empty() {
        return Ok("ok: no load macro".to_string());
    }
    let sequences = sequences(init)?;
    let mut stored: Vec<(i64, &Insn)> = Vec::new();
    for (k, x) in template_writes(init)? {
        match stored.iter_mut().find(|(j, _)| *j == k) {
            Some(entry) => entry.1 = x,
            None => stored.push((k, x)),
        }
    }
    let name = &unit.name;
    for k in &used {
        let Some((_, Some(runs))) = sequences.iter().find(|(j, _)| j == k) else {
            return Err(UnitError(format!(
                "{name}: macro {k}'s sequence is not set by the init"
            )));
        };
        for t in runs {
            let last = stored.iter().find(|(j, _)| j == t).map(|(_, x)| *x);
            if !last.is_some_and(|x| x.note.starts_with("template")) {
                return Err(UnitError(format!(
                    "{name}: macro {k} runs template {t}, which the init does not store (it would run an earlier body's)"
                )));
            }
            for items in &tile {
                if let Some((_, x)) = template_writes(items)?.into_iter().find(|(j, _)| j == t) {
                    return Err(UnitError(format!(
                        "{name}: {} ({}) writes VD {} in the tile's code: template {t} of macro {k} after the first tile",
                        x.op.name,
                        x.note,
                        BACK_DOOR + t
                    )));
                }
            }
        }
    }
    Ok(format!(
        "ok: macros {} run templates their init stores, unwritten after",
        numbers(&used)
    ))
}
