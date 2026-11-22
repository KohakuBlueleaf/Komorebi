//! The unit program's math CPU to C++ (today's `komorebi.dsl.tile.emit`): a header with
//! `<name>_init()` (once per stream) and `<name>(dst)` (one tile in Dst tile `dst`), one
//! coprocessor instruction a line, for a compute kernel to include and call on its math
//! thread, both chips' words in it (Wormhole's under ARCH_WORMHOLE). No kernel
//! skeleton, no movers; the waits around it are the caller's. The emit runs the last
//! checks and adds them to the unit's listing.

use std::collections::BTreeSet;

use super::super::chip;
use super::r#macro::{check_back_door, check_templates};
use super::unit::{
    encode, insns, Arg, Check, Count, Insn, Item, Product, UnitError, UnitProgram, CPUS, MATH,
};
use super::wormhole::{forms, wormhole_run, WormholeRun};

/// Wormhole's ADDR_MOD_SET_Base set around a function's vector words, as tt-llk's
/// `_llk_math_eltwise_sfpu_start_` sets it.
const BASE_ON: [&str; 1] = ["TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1);"];
/// And cleared around its products, as `_done_` clears it.
const BASE_OFF: [&str; 2] = [
    "TTI_STALLWAIT(ckernel::p_stall::STALL_CFG, ckernel::p_stall::WAIT_SFPU);",
    "TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 0);",
];

/// A field's C++ text: its tt-llk name (`chip::STALL_NAMES`, STALLWAIT's) or its
/// number.
fn arg_text(op: &str, name: &str, value: i64) -> String {
    let named = chip::STALL_NAMES
        .iter()
        .find(|(field, number, _)| op == "STALLWAIT" && *field == name && *number == value);
    named.map_or_else(|| value.to_string(), |(_, _, text)| text.to_string())
}

/// `item`'s fields as numbers, in its operation's order.
fn fields(item: &Insn) -> Result<Vec<(&'static str, i64)>, UnitError> {
    let mut out = Vec::new();
    for ((name, _), value) in item.op.args.iter().zip(&item.args) {
        let Some(v) = value.num() else {
            return Err(UnitError(format!(
                "{} {name}={value}: not a number",
                item.op.name
            )));
        };
        out.push((*name, v));
    }
    Ok(out)
}

/// Field `name` of `item` (0 without it).
fn n(item: &Insn, name: &str) -> i64 {
    item.arg(name).and_then(Arg::num).unwrap_or(0)
}

/// `item` a vector word with an address mode of 4 or more.
fn based(item: &Insn) -> bool {
    chip::BASED.contains(&item.op.name) && n(item, "sfpu_addr_mode") >= chip::ADDR_MODS
}

/// Lines for Wormhole alone (`wormhole`), else `blackhole`'s, by ARCH_WORMHOLE.
fn for_wormhole(pad: &str, wormhole: &[String], blackhole: &[String]) -> Vec<String> {
    let mut out = vec!["#if defined(ARCH_WORMHOLE)".to_string()];
    out.extend(wormhole.iter().map(|l| format!("{pad}{l}")));
    if !blackhole.is_empty() {
        out.push("#else".to_string());
        out.extend(blackhole.iter().map(|l| format!("{pad}{l}")));
    }
    out.push("#endif".to_string());
    out
}

fn texts(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|l| l.to_string()).collect()
}

/// The TTI_ line of `item`, its fields checked as `encode` checks them.
fn word(item: &Insn) -> Result<String, UnitError> {
    let fields = fields(item)?;
    encode(item.op.name, &fields)?;
    let op = item.op.name;
    if fields.is_empty() {
        // a macro without arguments (TTI_SFPNOP) takes no parentheses
        return Ok(format!("TTI_{op};"));
    }
    let args: Vec<String> = fields.iter().map(|(n, v)| arg_text(op, n, *v)).collect();
    Ok(format!("TTI_{op}({});", args.join(", ")))
}

/// TT_<op> of a word whose immediate the caller gives (`item.runtime`, its C++): the
/// word built on the core; its other fields checked as TTI_'s are.
fn runtime(item: &Insn) -> Result<String, UnitError> {
    let fields = fields(item)?;
    encode(item.op.name, &fields)?;
    let args: Vec<String> = fields
        .iter()
        .map(|(n, v)| {
            if *n == "imm16" {
                item.runtime.clone()
            } else {
                v.to_string()
            }
        })
        .collect();
    Ok(format!("TT_{}({})", item.op.name, args.join(", ")))
}

/// `item` with field `name` set to `value`, its note kept, and none of Wormhole's words.
fn with(item: &Insn, name: &str, value: i64) -> Insn {
    let mut out = item.clone();
    out.set(name, Arg::Num(value));
    out.uses = Vec::new();
    out.runtime = String::new();
    out.wormhole = Vec::new();
    out
}

/// An item as the lines read it: a loop nest's, or a word that a word or a product
/// holds.
#[derive(Clone, Copy)]
enum Elem<'a> {
    Item(&'a Item),
    Insn(&'a Insn),
}

fn of_items(items: &[Item]) -> Vec<Elem<'_>> {
    items.iter().map(Elem::Item).collect()
}

fn of_insns<'a>(insns: impl IntoIterator<Item = &'a Insn>) -> Vec<Elem<'a>> {
    insns.into_iter().map(Elem::Insn).collect()
}

/// The run of one function on Wormhole, less a word whose SFPNOP is written already.
#[derive(Clone)]
struct Run<'a> {
    run: WormholeRun<'a>,
}

impl<'a> Run<'a> {
    fn without(&self, x: &Insn) -> Run<'a> {
        let mut run = self.run.clone();
        run.nops.remove(&(x as *const Insn as usize));
        Run { run }
    }
}

fn id(x: &Insn) -> usize {
    x as *const Insn as usize
}

/// `item` a REPLAY whose count on Wormhole holds its recorded SFPNOPs, or that issues
/// recorded words past the replay buffer after it.
fn recount(item: &Insn, wormhole: Option<&Run>) -> bool {
    let Some(w) = wormhole else { return false };
    let changed = w
        .run
        .counts
        .get(&id(item))
        .is_some_and(|c| *c != n(item, "len"));
    changed || w.run.tails.contains_key(&id(item))
}

/// A function's lines: with the base set around them when a word needs it, and
/// Wormhole's SFPNOPs and replay counts (`wormhole_run`).
fn body(items: &[Item], depth: usize) -> Result<Vec<String>, UnitError> {
    let any_based = insns(items).into_iter().any(based);
    let run = Run {
        run: wormhole_run(items)?,
    };
    let lines = lines(&of_items(items), depth, any_based, Some(&run))?;
    if !any_based {
        return Ok(lines);
    }
    let pad = "    ".repeat(depth);
    let mut out = for_wormhole(&pad, &texts(&BASE_ON), &[]);
    out.extend(lines);
    out.extend(for_wormhole(&pad, &texts(&BASE_OFF), &[]));
    Ok(out)
}

fn lines(
    items: &[Elem],
    depth: usize,
    any_based: bool,
    wormhole: Option<&Run>,
) -> Result<Vec<String>, UnitError> {
    let pad = "    ".repeat(depth);
    let mut out = Vec::new();
    for item in items {
        let x: &Insn = match *item {
            Elem::Insn(x) | Elem::Item(Item::Insn(x)) => x,
            Elem::Item(Item::Loop(l)) => {
                let var = &l.name;
                let count = match &l.count {
                    Count::Runtime(r) if r.guard => {
                        out.push(format!("{pad}if ({}) {{", r.text));
                        out.extend(lines(&of_items(&l.body), depth + 1, any_based, wormhole)?);
                        out.push(format!("{pad}}}"));
                        continue;
                    }
                    Count::Runtime(r) => r.text.clone(),
                    Count::Times(n) => {
                        if l.unroll {
                            out.push(format!("#pragma GCC unroll {n}"));
                        }
                        n.to_string()
                    }
                };
                out.push(format!(
                    "{pad}for (uint32_t {var} = 0; {var} < {count}; ++{var}) {{"
                ));
                out.extend(lines(&of_items(&l.body), depth + 1, any_based, wormhole)?);
                out.push(format!("{pad}}}"));
                continue;
            }
            Elem::Item(Item::Call(c)) => {
                out.push(format!("{pad}{};", c.text));
                continue;
            }
            Elem::Item(Item::Product(p)) if !p.insns.is_empty() => {
                out.extend(product(p, depth, any_based, wormhole)?);
                continue;
            }
            Elem::Item(Item::Product(p)) => {
                return Err(UnitError(format!(
                    "no C++ for the product {:?} of no words",
                    p.text
                )));
            }
            Elem::Item(Item::Sync(_)) => continue,
        };
        if !x.wormhole.is_empty() {
            // Wormhole's own words
            out.push("#if defined(ARCH_WORMHOLE)".to_string());
            out.extend(lines(&of_insns(&x.wormhole), depth, any_based, wormhole)?);
            out.push("#else".to_string());
            let mut plain = x.clone();
            plain.wormhole = Vec::new();
            out.extend(lines(&[Elem::Insn(&plain)], depth, any_based, None)?);
            out.push("#endif".to_string());
        } else if let Some(w) = wormhole.filter(|w| w.run.nops.contains(&id(x))) {
            // a hold Wormhole does not make
            out.extend(for_wormhole(&pad, &texts(&["TTI_SFPNOP;"]), &[]));
            out.extend(lines(
                &[Elem::Insn(x)],
                depth,
                any_based,
                Some(&w.without(x)),
            )?);
        } else if recount(x, wormhole) {
            let w = wormhole.map(|w| &w.run);
            let Some(count) = w.and_then(|w| w.counts.get(&id(x))) else {
                return Err(UnitError(format!(
                    "REPLAY of {} words: no count on Wormhole",
                    n(x, "len")
                )));
            };
            let form = with(x, "len", *count);
            let tail = w
                .and_then(|w| w.tails.get(&id(x)))
                .cloned()
                .unwrap_or_default();
            out.push("#if defined(ARCH_WORMHOLE)".to_string());
            out.push(format!("{pad}{}", word(&form)?));
            out.extend(lines(&of_insns(tail), depth, any_based, wormhole)?);
            out.push("#else".to_string());
            out.push(format!("{pad}{}", word(x)?));
            out.push("#endif".to_string());
        } else if !x.runtime.is_empty() {
            out.push(format!("{pad}{};", runtime(x)?));
        } else if based(x) {
            let mode = n(x, "sfpu_addr_mode") - chip::ADDR_MODS;
            let form = with(x, "sfpu_addr_mode", mode);
            out.extend(for_wormhole(&pad, &[word(&form)?], &[word(x)?]));
        } else {
            out.push(format!("{pad}{}", word(x)?));
        }
    }
    Ok(out)
}

/// A product's lines: its words, the base cleared around them (the matrix unit's
/// address modes are 0 to 3), Wormhole's own image under ARCH_WORMHOLE.
fn product(
    p: &Product,
    depth: usize,
    any_based: bool,
    wormhole: Option<&Run>,
) -> Result<Vec<String>, UnitError> {
    let pad = "    ".repeat(depth);
    let mut out = vec![format!("{pad}// {}: {} words", p.text, p.insns.len())];
    if any_based {
        out.extend(for_wormhole(&pad, &texts(&BASE_OFF), &[]));
    }
    if p.wormhole.is_empty() {
        out.extend(lines(&of_insns(&p.insns), depth, false, wormhole)?);
    } else {
        // Wormhole's own image, in the modes its init sets
        out.push("#if defined(ARCH_WORMHOLE)".to_string());
        out.extend(lines(&of_insns(&p.wormhole), depth, false, wormhole)?);
        out.push("#else".to_string());
        out.extend(lines(&of_insns(&p.insns), depth, false, None)?);
        out.push("#endif".to_string());
    }
    if any_based {
        out.extend(for_wormhole(&pad, &texts(&BASE_ON), &[]));
    }
    Ok(out)
}

/// The math CPU's own items: the syncs and the library units around the vector code
/// (copy_tile, the products, tile_regs_*) are the caller's.
fn math_part(items: &[Item]) -> Vec<Item> {
    items
        .iter()
        .filter(|x| match x {
            Item::Sync(_) => false,
            Item::Product(p) => !p.insns.is_empty(),
            Item::Call(c) => !c.text.starts_with("copy_tile"),
            Item::Insn(_) | Item::Loop(_) => true,
        })
        .cloned()
        .collect()
}

/// Every item of a loop nest but the loops, in order.
fn flat<'a>(items: &'a [Item], out: &mut Vec<&'a Item>) {
    for x in items {
        match x {
            Item::Loop(l) => flat(&l.body, out),
            other => out.push(other),
        }
    }
}

/// A word character, as Python's `\w` reads text.
fn word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// A match of `prefix(\w+)` at byte `start` of `text`: its end and the name.
fn name_at(text: &str, start: usize, prefix: &str) -> Option<(usize, String)> {
    let rest = text.get(start..)?.strip_prefix(prefix)?;
    let name: String = rest.chars().take_while(|c| word_char(*c)).collect();
    let end = start + prefix.len() + name.len();
    (!name.is_empty()).then_some((end, name))
}

/// A match of `pack_tile\([^,]+,\s*(\w+)\)` at byte `start` of `text`: its end and
/// the name.
fn pack_tile_at(text: &str, start: usize) -> Option<(usize, String)> {
    let rest = text.get(start..)?.strip_prefix("pack_tile(")?;
    let comma = rest.find(',')?;
    if comma == 0 {
        return None;
    }
    let after = &rest[comma + 1..];
    let spaces = after.len() - after.trim_start().len();
    let tail = &after[spaces..];
    let name: String = tail.chars().take_while(|c| word_char(*c)).collect();
    if name.is_empty() || !tail[name.len()..].starts_with(')') {
        return None;
    }
    let end = start + "pack_tile(".len() + comma + 1 + spaces + name.len() + 1;
    Some((end, name))
}

/// A pattern matched at a byte of a text: the match's end and its name.
type Pattern<'a> = &'a dyn Fn(&str, usize) -> Option<(usize, String)>;

/// The names of `text`'s matches, left to right, a match's end the next search's start
/// (`re.findall`), of the first of `patterns` that matches at each place.
fn find_all(text: &str, patterns: &[Pattern]) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < text.len() {
        match patterns.iter().find_map(|p| p(text, at)) {
            Some((end, name)) => {
                out.push(name);
                at = end;
            }
            None => at += text[at..].chars().next().map_or(1, char::len_utf8),
        }
    }
    out
}

/// The buffers `text` waits for (`cb_wait_front\((\w+)`).
fn waited(text: &str) -> Vec<String> {
    find_all(text, &[&|t, at| name_at(t, at, "cb_wait_front(")])
}

/// The buffers `text` pushes (`cb_push_back\((\w+)|pack_tile\([^,]+,\s*(\w+)\)`).
fn pushed(text: &str) -> Vec<String> {
    find_all(
        text,
        &[&|t, at| name_at(t, at, "cb_push_back("), &pack_tile_at],
    )
}

/// Rule "one producer": each buffer of the unit's table (the unpack CPU's waits name
/// who pushes; the pack CPU's packs and pushes are its own) has one producer: a second
/// one's pages never reach a wait (each push writes its own count into the buffer's
/// received counter). The check's text, or the refusal.
pub fn check_buffers(unit: &UnitProgram) -> Result<String, UnitError> {
    let mut producers: Vec<(String, BTreeSet<String>)> = Vec::new();
    let mut add = |name: String, who: &str| match producers.iter_mut().find(|(n, _)| *n == name) {
        Some((_, set)) => {
            set.insert(who.to_string());
        }
        None => producers.push((name, BTreeSet::from([who.to_string()]))),
    };
    for (cpu, (init, tile)) in CPUS.iter().zip(unit.init.iter().zip(&unit.tile)) {
        if *cpu == "math" {
            continue;
        }
        let mut items = Vec::new();
        flat(init, &mut items);
        flat(tile, &mut items);
        for x in items {
            match (*cpu, x) {
                ("unpack", Item::Sync(s)) => {
                    for name in waited(&s.text) {
                        add(name, &s.other);
                    }
                }
                ("pack", Item::Sync(_) | Item::Call(_) | Item::Product(_)) => {
                    let text = match x {
                        Item::Sync(s) => &s.text,
                        Item::Call(c) => &c.text,
                        Item::Product(p) => &p.text,
                        Item::Insn(_) | Item::Loop(_) => continue,
                    };
                    for name in pushed(text) {
                        add(name, "pack");
                    }
                }
                _ => {}
            }
        }
    }
    if let Some((name, who)) = producers.iter().find(|(_, who)| who.len() > 1) {
        let who: Vec<&str> = who.iter().map(String::as_str).collect();
        return Err(UnitError(format!(
            "{}: buffer {name} is pushed by {}: one producer a buffer (a state kept in place is neither pushed nor popped)",
            unit.name,
            who.join(" and ")
        )));
    }
    producers.sort_by(|a, b| a.0.cmp(&b.0));
    let table: Vec<String> = producers
        .iter()
        .map(|(name, who)| {
            let who: String = who.iter().map(String::as_str).collect();
            format!("{name} by {who}")
        })
        .collect();
    let table = if table.is_empty() {
        "no buffer".to_string()
    } else {
        table.join(", ")
    };
    Ok(format!("ok: {table}"))
}

/// The digits from byte `at` of `text`.
fn digits(text: &[u8], at: usize) -> usize {
    text.get(at..)
        .map_or(0, |t| t.iter().take_while(|c| c.is_ascii_digit()).count())
}

/// A word boundary of `text` before char index `at` (`\b`).
fn boundary(text: &[char], at: usize) -> bool {
    let before = at > 0 && text.get(at - 1).is_some_and(|c| word_char(*c));
    let after = text.get(at).is_some_and(|c| word_char(*c));
    before != after
}

/// The first float literal or type of `code` (today's `FLOAT`,
/// `\bfloat\b|\bdouble\b|\d\.\d*f?\b|\d+e[-+]?\d+`): compute code carries no float
/// (rule no-float).
fn float_in(code: &str) -> Option<String> {
    let chars: Vec<char> = code.chars().collect();
    let digit = |i: usize| chars.get(i).is_some_and(char::is_ascii_digit);
    let has = |i: usize, word: &str| {
        word.chars()
            .enumerate()
            .all(|(k, c)| chars.get(i + k) == Some(&c))
    };
    let take = |i: usize, j: usize| chars[i..j].iter().collect::<String>();
    let ascii: Vec<u8> = chars
        .iter()
        .map(|c| if c.is_ascii() { *c as u8 } else { b' ' })
        .collect();
    for i in 0..chars.len() {
        for word in ["float", "double"] {
            let j = i + word.chars().count();
            if boundary(&chars, i) && has(i, word) && boundary(&chars, j) {
                return Some(word.to_string());
            }
        }
        if digit(i) && chars.get(i + 1) == Some(&'.') {
            let most = digits(&ascii, i + 2);
            for k in (0..=most).rev() {
                let j = i + 2 + k;
                if chars.get(j) == Some(&'f') && boundary(&chars, j + 1) {
                    return Some(take(i, j + 1));
                }
                if boundary(&chars, j) {
                    return Some(take(i, j));
                }
            }
        }
        if digit(i) {
            let j = i + digits(&ascii, i);
            if chars.get(j) == Some(&'e') {
                let sign = usize::from(matches!(chars.get(j + 1), Some('-' | '+')));
                let more = digits(&ascii, j + 1 + sign);
                if more > 0 {
                    return Some(take(i, j + 1 + sign + more));
                }
            }
        }
    }
    None
}

/// The C++ header of `unit`'s math CPU, Wormhole's forms of its words given first, its
/// last checks added to its listing; refuses text that carries a float, a macro whose
/// template the init does not store or the tile's code overwrites, a word with a VD of
/// 12 or more that is no template store, and a buffer with two producers.
pub fn emit(unit: &mut UnitProgram) -> Result<String, UnitError> {
    forms(unit)?;
    let check = |rule: &str, text: &str, verdict: String| Check {
        rule: rule.to_string(),
        text: text.to_string(),
        verdict,
    };
    let templates = check_templates(unit)?;
    unit.checks.push(check(
        "macro-templates",
        "each macro's template stored, kept",
        templates,
    ));
    let back_door = check_back_door(unit)?;
    unit.checks.push(check(
        "back-door",
        "a VD of 12 or more only in a template store",
        back_door,
    ));
    let buffers = check_buffers(unit)?;
    unit.checks.push(check(
        "one-producer",
        "one pushing CPU or kernel a buffer",
        buffers,
    ));
    let name = &unit.name;
    let mut signature = "const uint32_t dst".to_string();
    if unit.band {
        signature += ", const uint32_t rows";
    }
    for s in &unit.scalars {
        signature += &format!(", const uint32_t {s}");
    }
    let clobbers = if unit.clobbers.is_empty() {
        "no constant register".to_string()
    } else {
        unit.clobbers.join(", ")
    };
    let mut lines = vec![
        format!(
            "// {name}: emitted by komorebi.dsl.tile from its unit program ({} Dst); math CPU only.",
            unit.dest
        ),
        format!("// Writes {clobbers}; the caller acquires Dst and waits."),
    ];
    if unit.band {
        lines.push("// rows: the live rows (1 .. 32), run in bands of 4.".to_string());
    }
    for s in &unit.scalars {
        lines.push(format!("// {s}: a run-time scalar, a float32's bits."));
    }
    let fidelity = unit.fidelity.as_deref().unwrap_or("None");
    for p in &unit.products {
        lines.push(format!(
            "// Product into Dst tile dst + {}: {} @ {} at {fidelity}; the caller unpacks llk_unpack_AB_matmul({}, {}, ...).",
            p.dst, p.a, p.w, p.a, p.w
        ));
    }
    lines.extend(texts(&[
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        "#ifdef TRISC_MATH",
        "namespace km_tile {",
        "",
    ]));
    lines.push(format!("    inline void {name}_init() {{"));
    lines.extend(body(&unit.init[MATH], 2)?);
    lines.push("    }".to_string());
    lines.push(String::new());
    lines.push(format!("    inline void {name}({signature}) {{"));
    lines.extend(body(&math_part(&unit.tile[MATH]), 2)?);
    lines.push("    }".to_string());
    lines.push(String::new());
    for (part, items) in &unit.parts {
        lines.push(format!(
            "    // after the caller's {part} products: {name}_{part}(dst)"
        ));
        lines.push(format!(
            "    inline void {name}_{part}(const uint32_t dst) {{"
        ));
        lines.extend(body(&math_part(items), 2)?);
        lines.push("    }".to_string());
        lines.push(String::new());
    }
    lines.extend(texts(&["}  // namespace km_tile", "#endif", ""]));
    let text = lines.join("\n");
    let code: Vec<&str> = lines
        .iter()
        .map(String::as_str)
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect();
    if let Some(found) = float_in(&code.join("\n")) {
        return Err(UnitError(format!(
            "the emitted body carries a float ({})",
            super::py::str_repr(&found)
        )));
    }
    unit.checks.push(check(
        "no-float",
        "no float in compute code",
        "ok: none in the text".to_string(),
    ));
    Ok(text)
}
