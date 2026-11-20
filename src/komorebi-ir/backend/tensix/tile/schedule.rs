//! The schedule pass of a unit program's vector code (today's
//! `komorebi.dsl.tile.schedule`): a run of instructions reordered so a reader of a MAD
//! result waits behind other ready words.
//!
//! Each instruction's reads and writes are taken once, before the pass: it only
//! reorders, so they hold for it (the pressure checks recount them).

use std::collections::{HashMap, HashSet};

use super::chain::ChainError;
use super::unit::{Arg, Insn, Kind, WORK};

/// Each instruction's (reads, writes), by its place in the code before the pass.
type Uses = Vec<(Vec<Arg>, Vec<Arg>)>;

/// True for a value that takes a work register (not a fixed or constant one).
fn work(value: &Arg) -> bool {
    matches!(value, Arg::Name(n) if n.starts_with('v'))
}

/// Schedule: reorders `code[start..end]` (a run under lane flags moves as one) so a
/// reader of the MAD result issued just before it waits behind another ready word; a
/// constant's loads go in front of its first reader. Returns how many moved. A run
/// that two or more constants are loaded in front of is refused by name: today takes
/// their order from a set, which no case pins yet.
pub fn interleave(code: &mut Vec<Insn>, start: usize, end: usize) -> Result<usize, ChainError> {
    let end = end.min(code.len());
    let start = start.min(end);
    let mut uses = Uses::new();
    for x in code.iter() {
        uses.push((x.reads(), x.writes()?));
    }
    let mut order: Vec<usize> = (0..code.len()).collect();
    let moved = schedule(code, &uses, &mut order, start, end)?;
    let before = code.clone();
    *code = order.iter().map(|&i| before[i].clone()).collect();
    Ok(moved)
}

/// The pass on `order` (the code's instructions by their place before it).
fn schedule(
    code: &[Insn],
    uses: &Uses,
    order: &mut Vec<usize>,
    start: usize,
    end: usize,
) -> Result<usize, ChainError> {
    let reads = |i: usize| &uses[i].0;
    let writes = |i: usize| &uses[i].1;
    let whole: Vec<usize> = order[start..end].to_vec();
    let mut loaded: HashMap<&Arg, bool> = HashMap::new();
    for &x in order.iter() {
        for v in writes(x) {
            let was = *loaded.get(v).unwrap_or(&true);
            loaded.insert(v, was && code[x].op.name == "SFPLOADI");
        }
    }
    let mut consts: Vec<(Arg, Vec<usize>)> = Vec::new();
    for &x in &whole {
        if code[x].op.name != "SFPLOADI" {
            continue;
        }
        let Some(v) = code[x].arg("lreg_ind") else {
            continue;
        };
        if work(v) && loaded.get(v).copied().unwrap_or(false) {
            match consts.iter_mut().find(|(c, _)| c == v) {
                Some((_, ws)) => ws.push(x),
                None => consts.push((v.clone(), vec![x])),
            }
        }
    }
    let is_const = |v: &Arg| consts.iter().any(|(c, _)| c == v);
    // a constant register's setup (loads into L0, SFPCONFIG) stays whole, in front
    let mut setup = Vec::new();
    for (i, &x) in whole.iter().enumerate() {
        if code[x].op.name != "SFPCONFIG" {
            continue;
        }
        let mut j = i;
        while j > 0 && code[whole[j - 1]].op.name == "SFPLOADI" {
            if code[whole[j - 1]].arg("lreg_ind") != Some(&Arg::Name("L0".into())) {
                break;
            }
            j -= 1;
        }
        setup.extend_from_slice(&whole[j..=i]);
    }
    let held: HashSet<usize> = consts
        .iter()
        .flat_map(|(_, ws)| ws.iter().copied())
        .chain(setup.iter().copied())
        .collect();
    let block: Vec<usize> = whole
        .iter()
        .copied()
        .filter(|x| !held.contains(x))
        .collect();
    let Some(units) = units_of(code, &block) else {
        return Ok(0);
    };
    let mut preds: Vec<HashSet<usize>> = Vec::new();
    for (i, unit) in units.iter().enumerate() {
        let mine: HashSet<usize> = (0..i)
            .filter(|&j| {
                units[j]
                    .iter()
                    .any(|&x| unit.iter().any(|&y| depends(code, uses, x, y)))
            })
            .collect();
        preds.push(mine);
    }
    let mut height = vec![1usize; units.len()];
    for i in (0..units.len()).rev() {
        for j in i + 1..units.len() {
            if preds[j].contains(&i) {
                height[i] = height[i].max(height[j] + 1);
            }
        }
    }
    let later: HashSet<Arg> = order[end..]
        .iter()
        .flat_map(|&x| reads(x).iter().cloned())
        .collect();
    let written: HashSet<&Arg> = block.iter().flat_map(|&x| writes(x)).collect();
    let mut outside: HashSet<Arg> = block
        .iter()
        .flat_map(|&x| reads(x))
        .filter(|v| work(v) && !written.contains(v) && !is_const(v))
        .cloned()
        .collect();
    for &x in &order[..start] {
        outside.extend(writes(x).iter().filter(|v| later.contains(*v)).cloned());
    }
    let mut left: HashMap<Arg, i64> = HashMap::new();
    for &x in &block {
        for v in reads(x) {
            *left.entry(v.clone()).or_insert(0) += 1;
        }
    }
    let start_live: HashSet<Arg> = outside
        .into_iter()
        .filter(|v| left.get(v).is_some_and(|n| *n != 0) || later.contains(v))
        .collect();
    // a constant loaded at its reader is live beside the estimate: a tighter bound
    let mut picked = None;
    for limit in (3..=WORK as usize).rev() {
        let mut state = State {
            uses,
            later: &later,
        };
        let issued = state.order(
            code,
            &units,
            &preds,
            &height,
            start_live.clone(),
            left.clone(),
            limit,
        );
        let mut placed: Vec<&Arg> = Vec::new();
        let mut out = Vec::new();
        for &i in &issued {
            // a constant read under lane flags and outside them: loaded in front of
            // the run (loaded in it, the lanes the run leaves off keep stale values)
            let ahead: Vec<&Arg> = read_outside(&units[i], &consts, order, uses)
                .into_iter()
                .filter(|v| !placed.contains(v))
                .collect();
            if ahead.len() > 1 {
                return Err(ChainError::NotPorted(
                    "the order of the constants before a run is not pinned yet".into(),
                ));
            }
            for v in ahead {
                if !placed.contains(&v) {
                    if let Some((c, ws)) = consts.iter().find(|(c, _)| c == v) {
                        out.extend_from_slice(ws);
                        placed.push(c);
                    }
                }
            }
            for &x in &units[i] {
                for v in reads(x) {
                    if !placed.contains(&v) {
                        if let Some((c, ws)) = consts.iter().find(|(c, _)| c == v) {
                            out.extend_from_slice(ws);
                            placed.push(c);
                        }
                    }
                }
                out.push(x);
            }
        }
        let unread = consts
            .iter()
            .filter(|(c, _)| !placed.contains(&c))
            .flat_map(|(_, ws)| ws.iter().copied());
        let run: Vec<usize> = setup.iter().copied().chain(unread).chain(out).collect();
        order.splice(start..end, run);
        let end_now = order.len() - (code.len() - end);
        if pressure(order, uses) <= WORK as usize {
            picked = Some(issued);
            break;
        }
        order.splice(start..end_now, whole.iter().copied());
    }
    Ok(match picked {
        Some(issued) => issued.iter().enumerate().filter(|(k, i)| k != *i).count(),
        None => 0,
    })
}

/// What the issue order reads besides the units.
struct State<'a> {
    uses: &'a Uses,
    later: &'a HashSet<Arg>,
}

impl State<'_> {
    /// The issue order of `units`: the tallest ready one that keeps at most `limit`
    /// values live, else the one leaving fewest; not a reader of the MAD just issued.
    #[allow(clippy::too_many_arguments)]
    fn order(
        &mut self,
        code: &[Insn],
        units: &[Vec<usize>],
        preds: &[HashSet<usize>],
        height: &[usize],
        mut live: HashSet<Arg>,
        mut left: HashMap<Arg, i64>,
        limit: usize,
    ) -> Vec<usize> {
        let mut order = Vec::new();
        let mut issued: HashSet<usize> = HashSet::new();
        let mut previous: Option<usize> = None;
        while order.len() < units.len() {
            let mut ready: Vec<usize> = (0..units.len())
                .filter(|i| !issued.contains(i) && preds[*i].is_subset(&issued))
                .collect();
            ready.sort_by_key(|&i| (std::cmp::Reverse(height[i]), i));
            let after: HashMap<usize, usize> = ready
                .iter()
                .map(|&i| (i, self.unit_after(&units[i], &live, &mut left, false).len()))
                .collect();
            let fits: Vec<usize> = ready
                .iter()
                .copied()
                .filter(|i| after[i] <= limit)
                .collect();
            let ready = if fits.is_empty() {
                let mut by = ready.clone();
                by.sort_by_key(|i| after[i]);
                by
            } else {
                fits
            };
            let Some(&first) = ready.first() else {
                break;
            };
            let mut pick = first;
            if let Some(p) = previous.filter(|&p| code[p].op.kind == Kind::Mad) {
                let written = &self.uses[p].1;
                for &i in &ready {
                    let head = &self.uses[units[i][0]].0;
                    if !written.iter().any(|w| head.contains(w)) {
                        pick = i;
                        break;
                    }
                }
            }
            live = self.unit_after(&units[pick], &live, &mut left, true);
            order.push(pick);
            issued.insert(pick);
            previous = units[pick].last().copied();
        }
        order
    }

    /// The live work values once every instruction of `unit` has issued; with
    /// `issue`, `left` counts their reads off.
    fn unit_after(
        &self,
        unit: &[usize],
        live: &HashSet<Arg>,
        left: &mut HashMap<Arg, i64>,
        issue: bool,
    ) -> HashSet<Arg> {
        let mut out = live.clone();
        let mut counts = left.clone();
        for &x in unit {
            for v in &self.uses[x].0 {
                *counts.entry(v.clone()).or_insert(0) -= 1;
            }
            out = self.after(x, &out, &counts);
        }
        if issue {
            *left = counts;
        }
        out
    }

    /// The live work values once instruction `x` has issued (`left`: reads still to
    /// come in the block, its own counted off).
    fn after(&self, x: usize, live: &HashSet<Arg>, left: &HashMap<Arg, i64>) -> HashSet<Arg> {
        let mut out = live.clone();
        for v in &self.uses[x].0 {
            let rest = left.get(v).copied().unwrap_or(0);
            if rest <= 0 && !self.later.contains(v) {
                out.remove(v);
            }
        }
        out.extend(self.uses[x].1.iter().filter(|v| work(v)).cloned());
        out
    }
}

/// The constants of `consts` a run under lane flags (`unit`) reads that an instruction
/// outside it reads too, in the order the run reads them; none for one instruction.
fn read_outside<'a>(
    unit: &[usize],
    consts: &'a [(Arg, Vec<usize>)],
    code: &[usize],
    uses: &Uses,
) -> Vec<&'a Arg> {
    if unit.len() < 2 {
        return Vec::new();
    }
    let mut out: Vec<&Arg> = Vec::new();
    for &x in unit {
        for v in &uses[x].0 {
            let Some((c, _)) = consts.iter().find(|(c, _)| c == v) else {
                continue;
            };
            let elsewhere = code
                .iter()
                .any(|y| !unit.contains(y) && uses[*y].0.contains(c));
            if elsewhere && !out.contains(&c) {
                out.push(c);
            }
        }
    }
    out
}

/// The block as units that move whole: an instruction, or a run under lane flags
/// (none when a run does not close inside the block).
fn units_of(code: &[Insn], block: &[usize]) -> Option<Vec<Vec<usize>>> {
    let mut units = Vec::new();
    let mut i = 0;
    while i < block.len() {
        if !code[block[i]].sets_flags() {
            units.push(vec![block[i]]);
            i += 1;
            continue;
        }
        let (mut depth, mut j) = (0, i);
        loop {
            let x = &code[*block.get(j)?];
            if x.sets_flags() {
                depth = if x.op.name == "SFPENCC" { 0 } else { depth + 1 };
            }
            j += 1;
            if depth == 0 {
                break;
            }
        }
        units.push(block[i..j].to_vec());
        i = j;
    }
    Some(units)
}

/// True when instruction `y` must stay after `x` (registers or a Dst row).
fn depends(code: &[Insn], uses: &Uses, x: usize, y: usize) -> bool {
    let ((xr, xw), (yr, yw)) = (&uses[x], &uses[y]);
    let meets = |a: &[Arg], b: &[Arg]| a.iter().any(|v| b.contains(v));
    let (a, b) = (code[x].arg("dest_reg_addr"), code[y].arg("dest_reg_addr"));
    let same = a.is_some() && b.is_some() && a == b;
    let touches = code[x].op.name == "SFPSTORE" || code[y].op.name == "SFPSTORE";
    meets(xw, yr) || meets(xr, yw) || meets(xw, yw) || (same && touches)
}

/// The most work registers the allocator needs at one instruction of the code in
/// `order` (a value whose last read is an instruction frees its register for that
/// instruction's result).
fn pressure(order: &[usize], uses: &Uses) -> usize {
    let mut first: Vec<(&Arg, usize)> = Vec::new();
    let mut last: HashMap<&Arg, usize> = HashMap::new();
    for (i, &x) in order.iter().enumerate() {
        for v in uses[x].1.iter().filter(|v| work(v)) {
            if !first.iter().any(|(f, _)| *f == v) {
                first.push((v, i));
            }
        }
        for v in uses[x].0.iter().filter(|v| work(v)) {
            last.insert(v, i);
        }
    }
    let mut starts = vec![0i64; order.len() + 2];
    let mut ends = vec![0i64; order.len() + 2];
    for (v, i) in first {
        starts[i] += 1;
        let mut end = last.get(v).copied().unwrap_or(i);
        // written there: held through it
        if end == i || order.get(end).is_some_and(|&x| uses[x].1.contains(v)) {
            end += 1;
        }
        ends[end] += 1;
    }
    let (mut most, mut live) = (0i64, 0i64);
    for i in 0..order.len() {
        live += starts[i] - ends[i];
        most = most.max(live);
    }
    usize::try_from(most).unwrap_or(0)
}
