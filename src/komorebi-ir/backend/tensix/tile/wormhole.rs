//! Wormhole's forms of the words it would run otherwise (today's
//! `komorebi.dsl.tile.hazards.wormhole`): `forms` gives each such word of the unit's
//! math CPU its Wormhole words, which compute the same there; the emitter writes them
//! under ARCH_WORMHOLE, and `wormhole_run` (today's `unit.wormhole_run`) walks them for
//! the SFPNOPs and the replay counts a Wormhole core needs. A word without a form
//! stays as it is.
//!
//! Today keys the words by the objects' identities; the port keys them by their
//! places in memory, which do not move while a pass reads them.

use std::collections::{HashMap, HashSet};

use super::super::chip::{self, ChipWord};
use super::lower::CC_RESET;
use super::r#macro::FIRST_PLAIN;
use super::unit::{
    Arg, Count, Insn, Item, Loop, Product, UnitError, UnitProgram, FIXED, IADD_NO_CC, IADD_SUB,
    MATH, SHIFT_IMM_VC,
};

/// The LRegs a copy may take (7: Blackhole's indirect SrcA, Dst register).
const WORK: std::ops::Range<i64> = 0..7;
/// SFPMOV `instr_mod1`: the copy's sign complemented.
const COMPSIGN: i64 = 1;
/// SFPSHFT `instr_mod1` on Wormhole: `lreg_dest` shifted by the immediate.
const SHIFT_IMM: i64 = 1;
/// NEGATE_VA, NEGATE_VC: the fields a copy may negate (the product's sign: either).
const NEGATED: [(i64, &[&str]); 2] = [(1, &["lreg_src_a", "lreg_src_b"]), (2, &["lreg_src_c"])];
/// The MAD unit's source fields.
const SOURCES: [&str; 3] = ["lreg_src_a", "lreg_src_b", "lreg_src_c"];
/// A load macro row's plain-loaded registers.
const PLAIN: std::ops::Range<i64> = FIRST_PLAIN..8;
/// The words that end a scan for a dead register (a stack of flags the scan does not
/// follow).
const UNKNOWN: [&str; 5] = [
    "SFPLOADMACRO",
    "SFPCONFIG",
    "SFPPUSHC",
    "SFPPOPC",
    "SFPCOMPC",
];
/// The constant 0's LReg.
const ZERO: i64 = FIXED[0].1 as i64;
/// SFPSETCC `instr_mod1`: the integer view < 0, != 0, >= 0.
const LT0: i64 = 0;
const NE0: i64 = 2;
const GTE0: i64 = 4;
/// SFPARECIP's estimate on Wormhole: p(m) ~ -1 / 2m on [1, 2) (cubic, bf16 coefficients
/// for SFPMULI / SFPADDI, highest first; relative error 2.5e-3, the LUT's 5.6e-3):
/// 0.1104, -0.6641, 1.4688, -1.4141.
const RECIP_P: [i64; 4] = [0x3DE2, 0xBF2A, 0x3FBC, 0xBFB5];
/// Blackhole's SFPCAST `instr_mod1`: integer abs, to sign-magnitude.
const INT_ABS: i64 = 2;
const INT_SM: i64 = 3;
/// MAD words that NEGATE_VA, NEGATE_VC apply to.
const MADS: [&str; 3] = ["SFPMAD", "SFPADD", "SFPMUL"];

/// Where a word is: its identity while the unit is read.
fn id(x: &Insn) -> usize {
    x as *const Insn as usize
}

/// Field `field` of `x` (0 without it: every field of the unit's math words is a
/// number, `forms` checks it first).
fn n(x: &Insn, field: &str) -> i64 {
    x.arg(field).and_then(Arg::num).unwrap_or(0)
}

/// `x`'s LReg numbers among `args`.
fn numbers(args: &[Arg]) -> Vec<i64> {
    args.iter().filter_map(Arg::num).collect()
}

/// `insn` as a chip's check reads it (today's `unit.chip_word`).
pub fn chip_word(insn: &Insn) -> Result<ChipWord, UnitError> {
    let mut args = Vec::new();
    for ((name, _), value) in insn.op.args.iter().zip(&insn.args) {
        let Some(v) = value.num() else {
            return Err(UnitError(format!(
                "{} {name}={value}: not a number",
                insn.op.name
            )));
        };
        args.push((name.to_string(), v));
    }
    Ok(ChipWord {
        op: insn.op.name.to_string(),
        args,
        note: insn.note.clone(),
        kind: insn.op.kind.name().to_string(),
        reads: numbers(&insn.reads()),
        writes: numbers(&insn.writes()?),
    })
}

/// An item of a function as the walks read it: a loop nest's item, or one of the
/// words a word or a product holds.
#[derive(Clone, Copy)]
enum Elem<'a> {
    Loop(&'a Loop),
    Product(&'a Product),
    Insn(&'a Insn),
    Other,
}

fn elems(items: &[Item]) -> Vec<Elem<'_>> {
    items
        .iter()
        .map(|item| match item {
            Item::Loop(l) => Elem::Loop(l),
            Item::Product(p) => Elem::Product(p),
            Item::Insn(x) => Elem::Insn(x),
            Item::Call(_) | Item::Sync(_) => Elem::Other,
        })
        .collect()
}

fn words_of(insns: &[Insn]) -> Vec<Elem<'_>> {
    insns.iter().map(Elem::Insn).collect()
}

/// A product's words a Wormhole core runs: its own, else Blackhole's.
fn chip_insns(p: &Product) -> &[Insn] {
    if p.wormhole.is_empty() {
        &p.insns
    } else {
        &p.wormhole
    }
}

/// Where a Wormhole core runs one function's words otherwise than Blackhole's order
/// (`chip::wormhole_hold`): `nops` the words an SFPNOP goes before, `counts` each
/// REPLAY's count with the recorded SFPNOPs; `tails` each replaying REPLAY's recorded
/// words past the 32 entries, issued after it.
#[derive(Clone, Debug, Default)]
pub struct WormholeRun<'a> {
    /// The words an SFPNOP goes before, by identity.
    pub nops: HashSet<usize>,
    /// Each REPLAY's count, by identity.
    pub counts: HashMap<usize, i64>,
    /// Each replaying REPLAY's words past the replay buffer, by identity.
    pub tails: HashMap<usize, Vec<&'a Insn>>,
}

/// What a REPLAY recorded: its words (Wormhole's), those kept in the buffer, their
/// count with their SFPNOPs, and the words an SFPNOP goes before.
type Recorded<'a> = (Vec<&'a Insn>, usize, i64, Vec<&'a Insn>);

struct Walk<'a> {
    run: WormholeRun<'a>,
    recorded: HashMap<i64, Recorded<'a>>,
}

/// Whether `insn` after any of `before` is held otherwise on Wormhole.
fn held(before: &[&Insn], insn: &Insn) -> Result<bool, UnitError> {
    let word = chip_word(insn)?;
    for b in before {
        if chip::wormhole_hold(&chip_word(b)?, &word).is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The words of `a`, then those of `b` not in it.
fn union<'a>(a: &[&'a Insn], b: &[&'a Insn]) -> Vec<&'a Insn> {
    let mut out: Vec<&Insn> = Vec::new();
    for x in a.iter().chain(b) {
        if !out.iter().any(|y| id(y) == id(x)) {
            out.push(x);
        }
    }
    out
}

impl<'a> Walk<'a> {
    fn walk(
        &mut self,
        items: &[Elem<'a>],
        mut before: Vec<&'a Insn>,
        keep: bool,
    ) -> Result<Vec<&'a Insn>, UnitError> {
        let mut i = 0;
        while i < items.len() {
            match items[i] {
                Elem::Loop(l) => {
                    let body = elems(&l.body);
                    let once = self.walk(&body, before.clone(), false)?;
                    let last = self.walk(&body, union(&before, &once), keep)?;
                    let skipped = matches!(l.count, Count::Runtime(_) | Count::Times(0));
                    before = if skipped { union(&last, &before) } else { last };
                }
                Elem::Product(p) => {
                    before = self.walk(&words_of(chip_insns(p)), before, keep)?;
                }
                Elem::Insn(x) if x.op.name == "REPLAY" => {
                    let (start, count) = (n(x, "start_idx"), n(x, "len"));
                    let loading = n(x, "load_mode") != 0;
                    if loading {
                        let end = i + 1 + count as usize;
                        let words: Vec<&Insn> = items
                            .get(i + 1..end)
                            .unwrap_or(&[])
                            .iter()
                            .filter_map(|e| match e {
                                Elem::Insn(w) if w.op.name != "REPLAY" => Some(*w),
                                _ => None,
                            })
                            .collect();
                        if words.len() < count as usize {
                            return Err(UnitError(format!(
                                "REPLAY of {count}: not followed by its words"
                            )));
                        }
                        let words: Vec<&Insn> = words
                            .into_iter()
                            .flat_map(|w| {
                                if w.wormhole.is_empty() {
                                    vec![w]
                                } else {
                                    w.wormhole.iter().collect()
                                }
                            })
                            .collect();
                        let mut added = Vec::new();
                        for (b, w) in words.iter().zip(words.iter().skip(1)) {
                            if held(&[*b], w)? {
                                added.push(*w);
                            }
                        }
                        let (kept, total) = fitted(&words, &added);
                        self.recorded.insert(start, (words, kept, total, added));
                        i += count as usize;
                    }
                    let Some((words, kept, total, added)) = self.recorded.get(&start).cloned()
                    else {
                        return Err(UnitError(format!(
                            "REPLAY of {count} from entry {start}: nothing recorded there"
                        )));
                    };
                    let (Some(first), Some(last)) = (words.first(), words.last()) else {
                        return Err(UnitError(format!(
                            "REPLAY of {count} from entry {start}: no word recorded"
                        )));
                    };
                    if keep {
                        self.run.counts.insert(id(x), total);
                        self.run.nops.extend(added.iter().map(|w| id(w)));
                        if held(&before, first)? {
                            self.run.nops.insert(id(x));
                        }
                        if kept < words.len() && !loading {
                            self.run.tails.insert(id(x), words[kept..].to_vec());
                        }
                    }
                    before = vec![*last];
                }
                Elem::Insn(x) if !x.wormhole.is_empty() => {
                    before = self.walk(&words_of(&x.wormhole), before, keep)?;
                }
                Elem::Insn(x) => {
                    if keep && held(&before, x)? {
                        self.run.nops.insert(id(x));
                    }
                    before = vec![x];
                }
                Elem::Other => {}
            }
            i += 1;
        }
        Ok(before)
    }
}

/// One function's run on Wormhole (docs/reference/tile.md, On Wormhole): a word's
/// predecessors are every word that can run just before it (a REPLAY, a call, a sync
/// are none); a region over 32 entries with its SFPNOPs records its longest prefix in
/// them. Refused when a recorded word's SFPNOP is outside its count.
pub fn wormhole_run(items: &[Item]) -> Result<WormholeRun<'_>, UnitError> {
    let mut walk = Walk {
        run: WormholeRun::default(),
        recorded: HashMap::new(),
    };
    walk.walk(&elems(items), Vec::new(), true)?;
    for (words, kept, total, _) in walk.recorded.values() {
        let inside = words[..*kept]
            .iter()
            .filter(|w| walk.run.nops.contains(&id(w)))
            .count() as i64;
        if inside != total - *kept as i64 {
            let op = words.first().map_or("", |w| w.op.name);
            return Err(UnitError(format!(
                "a word recorded by REPLAY is given an SFPNOP elsewhere on Wormhole ({op}, {} words): its count would not hold it",
                words.len()
            )));
        }
    }
    Ok(walk.run)
}

/// (the recorded words kept, their count with their SFPNOPs): the longest prefix of
/// `words` within Wormhole's replay buffer (`chip::REPLAY_MOST` entries).
fn fitted(words: &[&Insn], added: &[&Insn]) -> (usize, i64) {
    let nops: HashSet<usize> = added.iter().map(|w| id(w)).collect();
    let held = |w: &&Insn| nops.contains(&id(w));
    let mut kept = 0;
    for i in 0..words.len() {
        let size = i as i64 + 1 + words[1..=i].iter().filter(|w| held(w)).count() as i64;
        if size > chip::REPLAY_MOST {
            break;
        }
        kept = i + 1;
    }
    let inside = words.get(1..kept).unwrap_or(&[]);
    (
        kept,
        kept as i64 + inside.iter().filter(|w| held(w)).count() as i64,
    )
}

/// Every word of a function, loop bodies once, a product's both images.
fn words<'a>(items: &'a [Item], out: &mut Vec<&'a Insn>) {
    for item in items {
        match item {
            Item::Loop(l) => words(&l.body, out),
            Item::Product(p) => out.extend(p.insns.iter().chain(&p.wormhole)),
            Item::Insn(x) => out.push(x),
            Item::Call(_) | Item::Sync(_) => {}
        }
    }
}

/// The math CPU's functions: its setup, its tile, its parts.
fn functions(unit: &UnitProgram) -> Vec<&[Item]> {
    let mut out: Vec<&[Item]> = vec![&unit.init[MATH], &unit.tile[MATH]];
    out.extend(unit.parts.iter().map(|(_, items)| items.as_slice()));
    out
}

/// Each word of `unit`'s math functions that Wormhole runs otherwise given its form
/// there (once); a copy goes to the word's own destination, an LReg 0 .. 6 no word of
/// the unit names, or one dead after the word on every path from it (a recorded word:
/// after each play).
pub fn forms(unit: &mut UnitProgram) -> Result<(), UnitError> {
    let made = {
        let functions = functions(unit);
        let mut all = Vec::new();
        for f in &functions {
            words(f, &mut all);
        }
        for x in &all {
            chip_word(x)?; // every field a number
        }
        let mut named = HashSet::new();
        for x in &all {
            named.extend(numbers(&x.reads()));
            named.extend(numbers(&x.writes()?));
        }
        let free: Vec<i64> = WORK.filter(|r| !named.contains(r)).collect();
        let mut made = templates(&all)?;
        let (code, starts) = flow(&functions)?;
        // id: (the word, the registers dead after it), in the order first reached
        let mut places: Vec<(&Insn, Vec<i64>)> = Vec::new();
        for (i, step) in code.iter().enumerate() {
            let Step::Insn(x) = step else { continue };
            if !x.wormhole.is_empty() || made.contains_key(&id(x)) {
                continue;
            }
            let mut dead = Vec::new();
            for r in WORK {
                if is_dead(r, &code, i + 1, x.sets_flags(), &starts)? {
                    dead.push(r);
                }
            }
            match places.iter_mut().find(|(y, _)| id(y) == id(x)) {
                Some((_, was)) => was.retain(|r| dead.contains(r)),
                None => places.push((x, dead)),
            }
        }
        for (x, dead) in places {
            let mut registers = free.clone();
            registers.extend(dead);
            made.insert(id(x), form(x, &registers)?);
        }
        made
    };
    let mut set = |x: &mut Insn| {
        if let Some(form) = made.get(&id(x)) {
            x.wormhole.clone_from(form);
        }
    };
    let math = std::iter::once(&mut unit.init[MATH])
        .chain(std::iter::once(&mut unit.tile[MATH]))
        .chain(unit.parts.iter_mut().map(|(_, items)| items));
    for items in math {
        apply(items, &mut set);
    }
    Ok(())
}

/// `set` on every word `words` reads, in place.
fn apply(items: &mut [Item], set: &mut impl FnMut(&mut Insn)) {
    for item in items {
        match item {
            Item::Loop(l) => apply(&mut l.body, set),
            Item::Product(p) => p
                .insns
                .iter_mut()
                .chain(&mut p.wormhole)
                .for_each(&mut *set),
            Item::Insn(x) => set(x),
            Item::Call(_) | Item::Sync(_) => {}
        }
    }
}

/// Whether `x` stores a load macro's template (VD 12 + k) and runs nothing.
fn stored(x: &Insn) -> bool {
    n(x, "lreg_dest") >= chip::TEMPLATE_VD && !UNKNOWN.contains(&x.op.name)
}

/// A copy of `x` with `changed` fields, its note, and none of Wormhole's words; its
/// registers beside its fields and its run-time immediate kept when asked.
fn remade(x: &Insn, changed: &[(&str, i64)], note: &str, uses: bool, runtime: bool) -> Insn {
    let mut out = x.clone();
    for (field, value) in changed {
        out.set(field, Arg::Num(*value));
    }
    out.note = note.to_string();
    out.wormhole = Vec::new();
    if !uses {
        out.uses = Vec::new();
    }
    if !runtime {
        out.runtime = String::new();
    }
    out
}

/// A template negating a plain-loaded operand (LReg 4 .. 7: NEGATE_VC's VC,
/// NEGATE_VA's either factor) stored without the negation, each plain load of that
/// register followed by its sign-complemented copy; only where templates negating it
/// alone read it and plain loads alone write it. The forms, by the words' identities.
fn templates(words: &[&Insn]) -> Result<HashMap<usize, Vec<Insn>>, UnitError> {
    let mut out = HashMap::new();
    let mut negating: Vec<(&Insn, Vec<i64>)> = Vec::new();
    for x in words {
        if !x.wormhole.is_empty() || !stored(x) || !MADS.contains(&x.op.name) {
            continue;
        }
        let mod1 = n(x, "instr_mod1");
        let bits: Vec<&[&str]> = NEGATED
            .iter()
            .filter(|(b, _)| mod1 & b != 0)
            .map(|(_, fields)| *fields)
            .collect();
        let plain: Vec<Option<i64>> = bits
            .iter()
            .map(|fields| fields.iter().map(|f| n(x, f)).find(|v| PLAIN.contains(v)))
            .collect();
        let mut distinct: Vec<i64> = plain.iter().flatten().copied().collect();
        distinct.sort_unstable();
        distinct.dedup();
        if !bits.is_empty() && plain.iter().all(Option::is_some) && distinct.len() == plain.len() {
            negating.push((x, distinct));
        }
    }
    for (x, registers) in &negating {
        for r in registers {
            for y in words {
                if id(y) == id(x) {
                    continue;
                }
                let reads = y.reads().contains(&Arg::Num(*r));
                if !reads && !y.writes()?.contains(&Arg::Num(*r)) {
                    continue;
                }
                if stored(y) {
                    let negates = negating
                        .iter()
                        .any(|(z, regs)| id(z) == id(y) && regs.contains(r));
                    if !negates {
                        return Ok(out); // a template that reads it as it is
                    }
                } else if y.op.name == "SFPLOADMACRO" {
                    continue; // it runs the templates
                } else if reads || y.op.name != "SFPLOAD" {
                    return Ok(out); // a plain word that reads or makes it
                }
            }
        }
    }
    for (x, _) in &negating {
        let mod1 = n(x, "instr_mod1") & !3;
        out.insert(
            id(x),
            vec![remade(x, &[("instr_mod1", mod1)], &x.note, true, true)],
        );
    }
    let loaded: Vec<i64> = negating.iter().flat_map(|(_, r)| r.clone()).collect();
    for y in words {
        let r = n(y, "lreg_ind");
        if y.op.name == "SFPLOAD" && loaded.contains(&r) && y.wormhole.is_empty() {
            let note = "Wormhole: negated for a template";
            let load = remade(y, &[], &y.note, false, false);
            out.insert(id(y), vec![load, mov(r, r, COMPSIGN, note)?]);
        }
    }
    Ok(out)
}

/// One step of the math functions in run order (`flow`'s).
#[derive(Clone, Copy)]
enum Step<'a> {
    /// A word.
    Insn(&'a Insn),
    /// A word the scan does not follow.
    Unknown,
    /// A loop's start: its body from here, or none of it (a count of 0) to `end`.
    Skip(usize),
    /// A loop's end: its next pass from `start`, or on after it.
    Back(usize),
    /// A math function's end: the next call's start, every lane on.
    End,
}

/// The words of `functions` in run order, one list (a replay: its recorded words; a
/// call, a sync, a product: no vector-unit word), and where the tile function and
/// each part start (what may run after a function's end).
fn flow<'a>(functions: &[&'a [Item]]) -> Result<(Vec<Step<'a>>, Vec<usize>), UnitError> {
    let (mut code, mut starts) = (Vec::new(), Vec::new());
    for (k, function) in functions.iter().enumerate() {
        if k > 0 {
            starts.push(code.len());
        }
        flat(function, &mut HashMap::new(), &mut code)?;
        code.push(Step::End);
    }
    Ok((code, starts))
}

fn flat<'a>(
    items: &'a [Item],
    recorded: &mut HashMap<i64, Vec<Step<'a>>>,
    code: &mut Vec<Step<'a>>,
) -> Result<(), UnitError> {
    for (i, item) in items.iter().enumerate() {
        match item {
            Item::Loop(l) => {
                let skip = code.len();
                code.push(Step::Skip(0));
                let start = code.len();
                flat(&l.body, recorded, code)?;
                code.push(Step::Back(start));
                code[skip] = Step::Skip(code.len());
            }
            Item::Insn(x) if x.op.name == "REPLAY" => {
                let (start, count) = (n(x, "start_idx"), n(x, "len") as usize);
                if n(x, "load_mode") != 0 {
                    let end = (i + 1 + count).min(items.len());
                    let mut words = Vec::new();
                    for w in items.get(i + 1..end).unwrap_or(&[]) {
                        let Item::Insn(w) = w else {
                            return Err(UnitError(format!(
                                "REPLAY of {count}: records an item that is not a word"
                            )));
                        };
                        words.push(Step::Insn(w));
                    }
                    recorded.insert(start, words);
                } else {
                    let played = recorded.get(&start).cloned();
                    code.extend(played.unwrap_or_else(|| vec![Step::Unknown]));
                }
            }
            Item::Insn(x) => code.push(Step::Insn(x)),
            Item::Call(_) | Item::Sync(_) | Item::Product(_) => {}
        }
    }
    Ok(())
}

/// Whether the scan stops at `x` (an SFPCONFIG of LReg 11 .. 14 writes that register
/// alone).
fn unknown(x: &Insn) -> bool {
    let constant = x.op.name == "SFPCONFIG" && (11..=14).contains(&n(x, "config_dest"));
    UNKNOWN.contains(&x.op.name) && !constant
}

/// `register` not read on any path from `code[at]` before a write to every lane a copy
/// took: with the flags unchanged since the copy, or after an SFPENCC that turns every
/// lane on (`masked`: the flags changed by the word itself).
fn is_dead(
    register: i64,
    code: &[Step],
    at: usize,
    masked: bool,
    starts: &[usize],
) -> Result<bool, UnitError> {
    let mut paths = vec![(at, masked)];
    let mut seen = HashSet::new();
    while let Some((i, mut masked)) = paths.pop() {
        if !seen.insert((i, masked)) {
            continue;
        }
        let Some(step) = code.get(i) else {
            return Err(UnitError(format!(
                "the scan for LReg {register} runs past the code"
            )));
        };
        let x = match *step {
            Step::End => {
                paths.extend(starts.iter().map(|s| (*s, false)));
                continue;
            }
            Step::Back(start) => {
                paths.extend([(start, masked), (i + 1, masked)]);
                continue;
            }
            Step::Skip(end) => {
                paths.extend([(i + 1, masked), (end, masked)]);
                continue;
            }
            Step::Unknown => return Ok(false),
            Step::Insn(x) => x,
        };
        if unknown(x) || x.reads().contains(&Arg::Num(register)) {
            return Ok(false);
        }
        let every = (
            x.arg("imm12_math").and_then(Arg::num),
            x.arg("instr_mod1").and_then(Arg::num),
        ) == (Some(CC_RESET.0), Some(CC_RESET.1));
        if x.op.name == "SFPENCC" && every {
            masked = false;
        } else if x.sets_flags() {
            masked = true;
        } else if x.writes()?.contains(&Arg::Num(register)) && !masked {
            continue; // this path writes it
        }
        paths.push((i + 1, masked));
    }
    Ok(true)
}

/// A word of the register form: `op` of `c` into `d`.
fn word(op: &str, c: i64, d: i64, mod1: i64, imm: i64, note: &str) -> Result<Insn, UnitError> {
    let args = [
        ("imm12_math", Arg::Num(imm)),
        ("lreg_c", Arg::Num(c)),
        ("lreg_dest", Arg::Num(d)),
        ("instr_mod1", Arg::Num(mod1)),
    ];
    Insn::new(op, &args, note)
}

/// SFPMOV of `source` into `target`.
fn mov(source: i64, target: i64, mod1: i64, note: &str) -> Result<Insn, UnitError> {
    word("SFPMOV", source, target, mod1, 0, note)
}

/// A MAD word's fields: `a * b + c` into `d`.
fn mad(op: &str, a: i64, b: i64, c: i64, d: i64, note: &str) -> Result<Insn, UnitError> {
    let args = [
        ("lreg_src_a", Arg::Num(a)),
        ("lreg_src_b", Arg::Num(b)),
        ("lreg_src_c", Arg::Num(c)),
        ("lreg_dest", Arg::Num(d)),
        ("instr_mod1", Arg::Num(0)),
    ];
    Insn::new(op, &args, note)
}

/// Wormhole's words for `x`, or none (the same word, or no form: a load macro's
/// template store, which runs nothing and stores this one word).
fn form(x: &Insn, free: &[i64]) -> Result<Vec<Insn>, UnitError> {
    if stored(x) {
        return Ok(Vec::new());
    }
    let (op, mod1) = (x.op.name, n(x, "instr_mod1"));
    match op {
        "SFPSHFT" if mod1 == SHIFT_IMM_VC => {
            let (c, d) = (n(x, "lreg_c"), n(x, "lreg_dest"));
            let shift = remade(
                x,
                &[("lreg_c", 0), ("instr_mod1", SHIFT_IMM)],
                &x.note,
                true,
                false,
            );
            let mut out = Vec::new();
            if c != d {
                out.push(mov(c, d, 0, "Wormhole: the shifted value")?);
            }
            out.push(shift);
            Ok(out)
        }
        _ if MADS.contains(&op) && mod1 & 3 != 0 => negated(x, free),
        "SFPCAST" if mod1 == INT_ABS => Ok(vec![word(
            "SFPABS",
            n(x, "lreg_src_c"),
            n(x, "lreg_dest"),
            0,
            0,
            "",
        )?]),
        "SFPCAST" if mod1 == INT_SM => sign_magnitude(x, free),
        "SFPGT" if mod1 == 1 => less(x, free),
        "SFPARECIP" if mod1 == 0 => recip(x, free),
        _ => Ok(Vec::new()),
    }
}

/// The first of `free` that is none of `taken`.
fn other(free: &[i64], taken: &[i64]) -> Option<i64> {
    free.iter().copied().find(|r| !taken.contains(r))
}

/// Two of `free` but `taken`, or none.
fn two(free: &[i64], taken: &[i64]) -> Option<(i64, i64)> {
    let pick: Vec<i64> = free
        .iter()
        .copied()
        .filter(|r| !taken.contains(r))
        .take(2)
        .collect();
    match pick[..] {
        [a, b] => Some((a, b)),
        _ => None,
    }
}

/// SFPGT's flags, VC < VD as sign-magnitude integers (FP32's total order): against 0
/// the sign (SFPSETCC); else SFPSETCC of r < 0, r made without the flags (VC - VD,
/// negated where VC < 0, VC where the signs differ: r ^ ((VC ^ VD) & (r ^ VC))), or
/// with one register, by the lanes of the two signs (the flags kept on their stack).
fn less(x: &Insn, free: &[i64]) -> Result<Vec<Insn>, UnitError> {
    let (c, d) = (n(x, "lreg_c"), n(x, "lreg_dest"));
    if d == ZERO {
        return Ok(vec![word("SFPSETCC", c, 0, LT0, 0, &x.note)?]);
    }
    if c == ZERO {
        // 0 < VD: VD >= 0 and VD != 0 (the second in the first's lanes)
        return Ok(vec![
            word("SFPSETCC", d, 0, GTE0, 0, &x.note)?,
            word("SFPSETCC", d, 0, NE0, 0, "")?,
        ]);
    }
    let Some((a, b)) = two(free, &[c, d]) else {
        return less_by_lanes(x, other(free, &[c, d]));
    };
    let (sub, add) = (IADD_SUB | IADD_NO_CC, IADD_NO_CC);
    let shift = (-31i64) & 0xFFF;
    Ok(vec![
        mov(c, a, 0, "Wormhole: VC's sign")?,
        word("SFPSHFT", 0, a, SHIFT_IMM, shift, "")?,
        word("SFPIADD", ZERO, a, sub, 0, "m = -sign: 0 or -1")?,
        mov(d, b, 0, "Wormhole: VD")?,
        word("SFPIADD", c, b, sub, 0, "r = VC - VD")?,
        word("SFPIADD", a, b, add, 0, "")?,
        word("SFPXOR", a, b, 0, 0, "r negated where VC < 0")?,
        mov(c, a, 0, "")?,
        word("SFPXOR", d, a, 0, 0, "VC ^ VD")?,
        word("SFPXOR", c, b, 0, 0, "")?,
        word("SFPAND", b, a, 0, 0, "")?,
        word("SFPXOR", c, b, 0, 0, "")?,
        word("SFPXOR", a, b, 0, 0, "VC's sign where the signs differ")?,
        word("SFPSETCC", b, 0, LT0, 0, &x.note)?,
    ])
}

/// r in register `a` (dead in every lane) lane by lane, every lane on: VC - VD where
/// both are positive, VC where VC >= 0 > VD or VD >= 0 > VC, VD - VC where both are
/// negative (the magnitudes' order reversed); then the flags of before (SFPPUSHC,
/// SFPPOPC) and SFPSETCC of r < 0 in their lanes, as SFPGT sets them.
fn less_by_lanes(x: &Insn, a: Option<i64>) -> Result<Vec<Insn>, UnitError> {
    let Some(a) = a else {
        return Ok(Vec::new()); // no register: Wormhole runs the word as it is
    };
    let (c, d) = (n(x, "lreg_c"), n(x, "lreg_dest"));
    let (imm, mod1) = CC_RESET;
    let sub = IADD_SUB | IADD_NO_CC;
    Ok(vec![
        word("SFPPUSHC", 0, 0, 0, 0, "Wormhole: the lanes' flags kept")?,
        word("SFPENCC", 0, 0, mod1, imm, "every lane on")?,
        mov(d, a, 0, "VD")?,
        word("SFPSETCC", c, 0, GTE0, 0, "VC >= 0")?,
        word("SFPIADD", c, a, sub, 0, "r = VC - VD")?,
        word("SFPSETCC", d, 0, LT0, 0, "VD < 0")?,
        mov(c, a, 0, "r = VC (not below)")?,
        word("SFPENCC", 0, 0, mod1, imm, "every lane on")?,
        word("SFPSETCC", c, 0, LT0, 0, "VC < 0")?,
        mov(c, a, 0, "r = VC (below)")?,
        word("SFPSETCC", d, 0, LT0, 0, "VD < 0")?,
        word("SFPIADD", d, a, sub, 0, "r = VD - VC")?,
        word("SFPPOPC", 0, 0, 0, 0, "the lanes' flags of before")?,
        word("SFPSETCC", a, 0, LT0, 0, &x.note)?,
    ])
}

/// SFPARECIP (sign + a 7-bit estimate of 1 / |x|) as p(m) 2^(128 - e): m = x's
/// mantissa in [1, 2), p of RECIP_P, 2^(128 - e) = ~x with its mantissa cleared (x's
/// sign flipped, p < 0: the product's is x's). As the LUT, inf below 2^-126 and 0 at
/// inf and NaN; 1 / x up to 2^127 (the LUT: 0 from 2^126).
fn recip(x: &Insn, free: &[i64]) -> Result<Vec<Insn>, UnitError> {
    let (c, d) = (n(x, "lreg_c"), n(x, "lreg_dest"));
    let picked = if c != d {
        other(free, &[c, d]).map(|t| (d, t))
    } else {
        two(free, &[c])
    };
    let Some((m, t)) = picked else {
        return Ok(Vec::new()); // no registers: Wormhole runs the word as it is
    };
    let mut out = vec![
        word(
            "SFPSETEXP",
            c,
            m,
            1,
            127,
            "Wormhole: x's mantissa in [1, 2)",
        )?,
        word("SFPSETSGN", m, m, 1, 0, "m = |that|")?,
        mov(m, t, 0, "p(m)")?,
    ];
    for (k, coefficient) in RECIP_P.iter().enumerate() {
        let op = if k == 0 { "SFPMULI" } else { "SFPADDI" };
        let args = [
            ("imm16_math", Arg::Num(*coefficient)),
            ("lreg_dest", Arg::Num(t)),
            ("instr_mod1", Arg::Num(0)),
        ];
        out.push(Insn::new(op, &args, "")?);
        if 0 < k && k < RECIP_P.len() - 1 {
            out.push(mad("SFPMUL", t, m, ZERO, t, "")?);
        }
        if k == RECIP_P.len() - 2 {
            out.push(word("SFPNOT", c, m, 0, 0, "~x: 2^(128 - e)")?);
        }
    }
    out.push(word("SFPSETMAN", m, m, 1, 0, "")?);
    let mut last = mad("SFPMUL", t, m, ZERO, d, &x.note)?;
    last.uses.clone_from(&x.uses);
    out.push(last);
    Ok(out)
}

/// Blackhole's two's complement to sign-magnitude cast: the integer magnitude
/// (SFPABS) into a copy register, and onto it the sign of a copy of the value.
fn sign_magnitude(x: &Insn, free: &[i64]) -> Result<Vec<Insn>, UnitError> {
    let (c, d) = (n(x, "lreg_src_c"), n(x, "lreg_dest"));
    let Some(t) = other(free, &[d]) else {
        return Ok(Vec::new()); // no register: Wormhole runs the word as it is
    };
    let mut out = Vec::new();
    if c != d {
        out.push(mov(c, d, 0, "Wormhole: the sign's source")?);
    }
    out.push(word("SFPABS", c, t, 0, 0, "Wormhole: the magnitude")?);
    let mut sign = word("SFPSETSGN", t, d, 0, 0, &x.note)?;
    sign.uses.clone_from(&x.uses);
    out.push(sign);
    Ok(out)
}

/// NEGATE_VA, NEGATE_VC as a sign-complemented copy each, read in the field's place
/// (the product's or the addend's sign changed exactly as Blackhole's). Without the
/// registers, the addend -0 (NEGATE_VC of the constant 0) is read as 0: the one result
/// that differs is a product of -0, +0 on Wormhole and -0 on Blackhole.
fn negated(x: &Insn, free: &[i64]) -> Result<Vec<Insn>, UnitError> {
    let mod1 = n(x, "instr_mod1");
    let bits: Vec<i64> = NEGATED
        .iter()
        .map(|(b, _)| *b)
        .filter(|b| mod1 & b != 0)
        .collect();
    let mut registers = vec![n(x, "lreg_dest")];
    for r in free {
        if !registers.contains(r) {
            registers.push(*r);
        }
    }
    let fields = |x: &Insn| SOURCES.map(|f| (f, n(x, f)));
    let mut found = copies(fields(x), &bits, &registers)?;
    if found.is_none() && bits.contains(&2) && n(x, "lreg_src_c") == ZERO {
        let rest: Vec<i64> = bits.iter().copied().filter(|b| *b != 2).collect();
        found = copies(fields(x), &rest, &registers)?;
    }
    let Some((args, mut words)) = found else {
        return Ok(Vec::new()); // no register: Wormhole runs the word as it is
    };
    let mut changed: Vec<(&str, i64)> = args.to_vec();
    changed.push(("instr_mod1", mod1 & !3));
    words.push(remade(x, &changed, &x.note, true, true));
    Ok(words)
}

/// The MAD unit's source fields and their registers.
type Sources = [(&'static str, i64); 3];

/// (the fields after the copies, the copies' SFPMOVs) for the negations `bits`, each
/// copy in one of `registers` that the word reads in no other field; or none.
fn copies(
    args: Sources,
    bits: &[i64],
    registers: &[i64],
) -> Result<Option<(Sources, Vec<Insn>)>, UnitError> {
    let Some((first, rest)) = bits.split_first() else {
        return Ok(Some((args, Vec::new())));
    };
    let fields = NEGATED
        .iter()
        .find(|(b, _)| b == first)
        .map_or(&[][..], |(_, f)| *f);
    for field in fields {
        let value = args.iter().find(|(f, _)| f == field).map_or(0, |(_, v)| *v);
        let others: Vec<i64> = args
            .iter()
            .filter(|(f, _)| f != field)
            .map(|(_, v)| *v)
            .collect();
        for r in registers {
            if others.contains(r) {
                continue;
            }
            let mut moved = args;
            for (f, v) in moved.iter_mut() {
                if f == field {
                    *v = *r;
                }
            }
            let left: Vec<i64> = registers.iter().copied().filter(|x| x != r).collect();
            if let Some((done, mut words)) = copies(moved, rest, &left)? {
                let note = format!("Wormhole: -{field}");
                words.insert(0, mov(value, *r, COMPSIGN, &note)?);
                return Ok(Some((done, words)));
            }
        }
    }
    Ok(None)
}
