//! The Tensix backend's emitter: a module to the C++ today's class DSL writes for the
//! same program (`komorebi.dsl`), on Komorebi's device framework, text for text. It
//! reads the program, its hints and its positions (S, H and P): today's text names
//! the script, each statement's line and the names of values, from the notes' origin
//! and names. These are pure functions of the module; nothing is run or written.
//!
//! Built: one program of handlers on a group, each a loop over a read stream of a
//! partition of `t.tiles(first, n)` that pushes a tile function's result into a sink
//! of the same partition (the first lesson); or that sums its `f32` tiles, the sum
//! reduced over the group by tree and stored by the leader, rank 0 (program 2); or
//! that adds the products of two streams' blocks into a tile each rank stores
//! (program 6). Everything else is refused by name (K964).
//!
//! A stream of a rectangle walks it in rows of its tensor's width, which the load
//! gives: the movement source is then made at load (`at_load`, `sources_at_load`).

pub mod chip;
pub mod load;
pub mod origin;
pub mod plan;
pub mod tile;
pub mod unit;

use crate::backend::{
    check_statement, ArgKind, ArgWord, Backend, Bound, Buffer, Build, Built, CacheKey, CompileArg,
    EntryPoint, Expr, Kernel, LoadStatement, Machine, Output, Processor, Reads, Refusal, Setting,
    Statement, Target, Unpack, NOT_BUILT,
};
use crate::bytecode::hash::{hex, sha256};
use crate::program::body::InstId;
use crate::program::notes::{Anchor, Origin};
use crate::program::{BodyRef, CallArg, Module};
use crate::types::{Ability, Entry, Format, IslandGraph, TileShape};
use crate::types::{ChainId, FuncId, HandlerId, ProgramId};

use self::origin::{place, statement_comment, Place};
use self::plan::{
    Fields, Int, Moved, Part, Plan, Product, Range, Rect, Shape, Stored, Sum, WordKind,
};
use self::unit::Unit;

/// One indent of the generated text (`program_sources.INDENT`).
const INDENT: &str = "    ";

/// Buffer indices: read pairs from 0, write pairs from 16 (`compile/layout.py`).
const FIRST_READ: u32 = 0;
/// The first index of a write pair.
const FIRST_WRITE: u32 = 16;
/// The blocks a read half holds: double buffering on each mover and one more each
/// (`layout.READ_ITEMS`).
const READ_ITEMS: u64 = 4;
/// The blocks a write half holds (`layout.WRITE_ITEMS`).
const WRITE_ITEMS: u64 = 2;
/// A read stream's pattern when the class states none (`page_streams.READ_PATTERN`).
const READ_PATTERN: &str = "AAAB";
/// A write stream's (`page_streams.WRITE_PATTERN`).
const WRITE_PATTERN: &str = "B";
/// The tiles of one Dst half a call takes one of (`tiles/functions.py`, `DST_HALF`):
/// a call of one Dst tile runs this many tiles an acquire.
const DST_HALF: u64 = 4;
/// The first index of a buffer other than a pair: an arrival (`layout.FIRST_OTHER`).
const FIRST_OTHER: u32 = 8;
/// A tree reduce's arrival slots over the command's group on one island: a slot a
/// round (`reduce.TREE_ROUNDS`, `km::kTreeRounds`).
const TREE_ROUNDS: u64 = 7;

/// A tree reduce's arrival slots over the command's group on `islands`, the one
/// statement of today's rule (`src/komorebi/dsl/compile/tiles/reduce.py:898`,
/// `tree_slots`, its `size` none): `TREE_ROUNDS` on one island; over several, the
/// largest island's rounds and a slot each other island, whichever is more.
pub(crate) fn tree_slots(islands: &IslandGraph) -> u64 {
    let parts = islands.count() as u64;
    if parts <= 1 {
        return TREE_ROUNDS;
    }
    let most = islands.sizes.iter().copied().max().unwrap_or(1) as u64;
    let rounds = u64::from(64 - most.saturating_sub(1).leading_zeros()).max(1);
    TREE_ROUNDS.max(rounds + parts - 1)
}

/// The compute kernel's includes (`program_sources.COMPUTE_INCLUDES`), in order, each
/// with the need that brings it (none: every program's); the needs this emitter makes.
const COMPUTE_INCLUDES: &[(&str, Option<&str>)] = &[
    ("api/compute/cb_api.h", None),
    ("api/compute/compute_kernel_api.h", None),
    ("api/compute/compute_kernel_hw_startup.h", None),
    ("api/compute/eltwise_binary_sfpu.h", Some("add")),
    ("api/compute/eltwise_unary/fill.h", Some("fill")),
    ("api/compute/matmul.h", Some("matmul")),
    ("api/compute/pack.h", None),
    ("api/compute/reg_api.h", None),
    ("api/compute/tile_move_copy.h", None),
    ("komorebi/device/movement/km_movement.hpp", None),
    ("komorebi/device/movement/km_port.hpp", Some("port")),
    ("komorebi/device/km_worker.hpp", None),
];

/// What the handlers need of the includes (`compiled.includes`): a sum's add, fill
/// and port, a product's fill and matmul; their store's ask.
fn needs(plans: &[Plan]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for plan in plans {
        let wanted: &[&str] = match &plan.shape {
            Shape::Summed(_) => &["add", "fill", "port", "ask"],
            Shape::Product(_) => &["fill", "matmul", "ask"],
            // a field's sums start at the first block's product: no fill
            Shape::Fields(_) => &["matmul", "ask"],
            _ => &[],
        };
        for need in wanted {
            if !out.contains(need) {
                out.push(*need);
            }
        }
    }
    out
}

/// `name` with each `_`-separated part capitalized (`files.camel`).
fn camel(name: &str) -> String {
    name.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(c) => c
                    .to_uppercase()
                    .chain(chars.flat_map(char::to_lowercase))
                    .collect(),
                None => String::new(),
            }
        })
        .collect()
}

/// A buffer pair, as `scale.h` declares it.
struct Pair {
    name: String,
    index: u32,
    block: u64,
    /// Its pattern constant's letters; none when every stream of it is in a pattern
    /// its command gives (today's `buffer.patterns`, which then has none).
    pattern: Option<&'static str>,
    what: String,
    /// Its pages' format: the stream's tensor's.
    format: Option<Format>,
    /// Each half's pages: the role's items of a block (`layout.pair`).
    pages: u64,
}

/// What the files print of the origin: the script's name and each place's comment.
struct Lines<'m> {
    origin: Option<&'m Origin>,
}

impl Lines<'_> {
    /// `// <Program>, made by ...` as each file's first line gives it.
    fn made(&self, program: &str) -> String {
        match self.origin {
            Some(o) => format!(
                "{program}, made by komorebi.dsl from {}:",
                origin::file_name(o, o.places.first().map_or(0, |p| p.1))
            ),
            None => format!("{program}, made by komorebi-ir:"),
        }
    }

    /// The place of `anchor`.
    fn place(&self, anchor: Anchor) -> Option<Place> {
        self.origin.and_then(|o| place(o, anchor))
    }

    /// A statement's comment, `line N: text`; none without an origin.
    fn comment(&self, anchor: Anchor, head_last: Option<u32>) -> Option<String> {
        let o = self.origin?;
        let at = place(o, anchor)?;
        Some(statement_comment(o, at, head_last))
    }

    /// ` line N` of a handler's case, `, line N` of its header line.
    fn handler_line(&self, p: ProgramId, h: HandlerId) -> Option<u32> {
        self.place(Anchor::Handler(p, h)).map(|at| at.first)
    }
}

/// The plans of `module`'s handlers and the units of the tile functions they call;
/// or every refusal of what is not built.
fn planned(module: &Module) -> Result<(Vec<Plan>, Vec<Unit>), Vec<Refusal>> {
    let refused = |what: &str, at: Anchor| Refusal {
        code: NOT_BUILT,
        what: format!("the Tensix backend does not emit {what} yet"),
        at: Some(at),
    };
    let mut refusals = Vec::new();
    if module.programs.len() != 1 {
        refusals.push(refused(
            "a module of other than one program",
            Anchor::Module,
        ));
    }
    if !module.controls.is_empty() {
        refusals.push(refused("a control program", Anchor::Module));
    }
    if (module.tile.rows, module.tile.cols) != (32, 32) {
        refusals.push(refused("a tile other than [32, 32]", Anchor::Module));
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }
    let p = ProgramId(0);
    let program = &module.programs[0];
    if !program.state.is_empty() || !program.runs.is_empty() {
        return Err(vec![refused(
            "a program with state or runs",
            Anchor::Program(p),
        )]);
    }
    let mut plans = Vec::new();
    for h in 0..program.handlers.len() {
        match plan::plan(module, p, HandlerId(h as u32)) {
            Ok(plan) => plans.push(plan),
            Err(r) => refusals.push(r),
        }
    }
    if plans.is_empty() && refusals.is_empty() {
        refusals.push(refused("a program of no handler", Anchor::Program(p)));
    }
    refusals.extend(chain_refusals(module, &plans));
    for plan in &plans {
        if let Err(r) = bounds(plan) {
            refusals.push(r);
        }
    }
    // the fidelity is the compute kernel's: one for the program's products
    // (`products.stream_product`)
    let mut fidelity: Option<&str> = None;
    for plan in &plans {
        let (func, own) = match &plan.shape {
            Shape::Product(product) => (product.call.func, product.fidelity.as_str()),
            Shape::Fields(fields) => (fields.sums[0].call.func, fields.fidelity.as_str()),
            _ => continue,
        };
        match fidelity {
            Some(first) if first != own => {
                let name = |f: FuncId| module.funcs[f.0 as usize].name.clone();
                refusals.push(Refusal {
                    code: NOT_BUILT,
                    what: format!(
                        "{}'s fidelity is {own} and another product's {first}: the fidelity \
                         is the compute kernel's",
                        name(func),
                    ),
                    at: Some(Anchor::Func(func)),
                });
            }
            Some(_) => {}
            None => fidelity = Some(own),
        }
    }
    let mut units: Vec<Unit> = Vec::new();
    for plan in &plans {
        let func = match &plan.shape {
            Shape::Mapped {
                call: Some(call), ..
            } => call.func,
            Shape::Fields(Fields {
                stored: Stored::Gate(gate),
                ..
            }) => gate.func,
            _ => continue,
        };
        match unit::unit(module, func) {
            Ok(u) => units.push(u),
            Err(r) => refusals.push(r),
        }
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }
    Ok((plans, units))
}

/// What a chain of the program asks that today's walk is not emitted for yet, each
/// refused by name until a program needs it (R49): a chain on islands, an earlier
/// input (`past`), a call whose handler reduces over its group (a reduce's tree).
fn chain_refusals(module: &Module, plans: &[Plan]) -> Vec<Refusal> {
    let p = ProgramId(0);
    let program = &module.programs[0];
    let mut out = Vec::new();
    let refused = |what: String, at: Anchor| Refusal {
        code: NOT_BUILT,
        what: format!("the Tensix backend does not emit {what} yet"),
        at: Some(at),
    };
    if !program.chains.is_empty()
        && module
            .requirements
            .iter()
            .any(|r| matches!(r.ability, Ability::Islands(_)))
    {
        out.push(refused("a chain on islands".into(), Anchor::Program(p)));
    }
    for (c, chain) in program.chains.iter().enumerate() {
        let c = ChainId(c as u32);
        for (k, call) in chain.calls.iter().enumerate() {
            let at = Anchor::Call(p, c, k as u32);
            if call.args.iter().any(|a| matches!(a, CallArg::Past(_))) {
                out.push(refused(
                    format!(
                        "a chain's call with an earlier input (`past`), call {k} of `{}`",
                        chain.name
                    ),
                    at,
                ));
            }
            let plan = plans.iter().find(|plan| plan.handler == call.handler);
            if let Some(Shape::Summed(_)) = plan.map(|plan| &plan.shape) {
                out.push(refused(
                    format!(
                        "a chain's call of a handler that reduces over its group (a reduce's \
                         tree), call {k} of `{}`",
                        chain.name
                    ),
                    at,
                ));
            }
        }
    }
    out
}

/// The generated files of `module` the compile makes: each name and text, in name
/// order; or every refusal of what is not built. A file made at load (`at_load`) is
/// not among them.
pub fn sources(module: &Module) -> Result<Vec<(String, String)>, Vec<Refusal>> {
    let (plans, units) = planned(module)?;
    let program = &module.programs[0];
    let lines = Lines {
        origin: module.notes.origin.as_ref(),
    };
    let name = &program.name;
    let header = format!("{}.h", name.to_lowercase());
    let mut files = vec![
        (header.clone(), sized_header(module, &plans, TREE_ROUNDS)),
        (
            "compute.cpp".to_string(),
            compute_text(name, &header, &plans, &units, &lines, chained(module)),
        ),
    ];
    if !walks_rectangles(&plans) {
        let movement = movement_text(name, &header, &plans, &lines, &[], chained(module))
            .map_err(|r| vec![r])?;
        files.push(("movement.cpp".to_string(), movement));
    }
    for u in units {
        files.push((format!("tile_{}.hpp", u.name), u.words));
        files.push((format!("tile_{}.txt", u.name), u.listing));
    }
    files.sort();
    Ok(files)
}

/// Whether a handler streams a rectangle, whose walk needs its tensor's width.
fn walks_rectangles(plans: &[Plan]) -> bool {
    plans
        .iter()
        .flat_map(|p| p.reads.iter().chain(p.written()))
        .any(|r| matches!(r.range, Range::Rect(_)))
}

/// The files of `module` the load makes, by name: the movement source when a handler
/// streams a rectangle (its walk is in rows of the tensor's width, a fact of the
/// load); none else.
pub fn at_load(module: &Module) -> Result<Vec<String>, Vec<Refusal>> {
    let (plans, _) = planned(module)?;
    Ok(if walks_rectangles(&plans) {
        vec!["movement.cpp".to_string()]
    } else {
        Vec::new()
    })
}

/// The files the load makes (`at_load`) given each tensor parameter's shape in tiles
/// (rows, columns) by name; a tensor a rectangle walks given no shape, or a rectangle
/// whose known ends do not fit its tensor, is refused by name.
pub fn sources_at_load(
    module: &Module,
    shapes: &[(String, Option<(u64, u64)>)],
) -> Result<Vec<(String, String)>, Vec<Refusal>> {
    let (plans, _) = planned(module)?;
    if !walks_rectangles(&plans) {
        return Ok(Vec::new());
    }
    let program = &module.programs[0];
    let lines = Lines {
        origin: module.notes.origin.as_ref(),
    };
    let header = format!("{}.h", program.name.to_lowercase());
    let movement = movement_text(
        &program.name,
        &header,
        &plans,
        &lines,
        shapes,
        chained(module),
    )
    .map_err(|r| vec![r])?;
    Ok(vec![("movement.cpp".to_string(), movement)])
}

/// The header of `plans` with each tree reduce's arrival of `slots` slots: the
/// compile's on one island, a load's on its machine's (`load::sized`).
fn sized_header(module: &Module, plans: &[Plan], slots: u64) -> String {
    let lines = Lines {
        origin: module.notes.origin.as_ref(),
    };
    let name = &module.programs[0].name;
    let pairs = pairs(plans, &lines);
    let others = others(plans, &lines, slots);
    header_text(name, plans, &pairs, &others, &lines, chained(module))
}

/// Whether the program declares a chain: its kernels then walk a chain's records
/// (`km_chain.hpp`), as today's class with a `km.chain` method (R49).
fn chained(module: &Module) -> bool {
    module
        .programs
        .first()
        .is_some_and(|p| !p.chains.is_empty())
}

/// The header (`<program>.h`) and the buffers of `module` on a machine of `islands`
/// (none: one island): the arrivals sized from its graph (`tree_slots`), the rest the
/// compile's. The slots are a constant of the header, as today's make prints them
/// from its session.
pub(crate) fn sized(
    module: &Module,
    islands: Option<&IslandGraph>,
) -> Result<load::Sized, Vec<Refusal>> {
    let (plans, _) = planned(module)?;
    let slots = islands.map_or(TREE_ROUNDS, tree_slots);
    let name = format!("{}.h", module.programs[0].name.to_lowercase());
    let header = sized_header(module, &plans, slots);
    Ok(((name, header), buffers(&plans, slots, chained(module))))
}

/// The buffer pairs in the order the streams open (`layout.pair`): each handler's
/// streams numbered from 0 by role, a pair of one name shared across handlers; read
/// pairs from index 0, write pairs from 16, two side by side.
fn pairs(plans: &[Plan], lines: &Lines) -> Vec<Pair> {
    let mut out: Vec<Pair> = Vec::new();
    for plan in plans {
        let body = BodyRef::Handler(plan.program, plan.handler);
        let mut opened: Vec<_> = plan
            .reads
            .iter()
            .enumerate()
            .map(|(i, read)| (read.inst, "Read", i, read, READ_PATTERN, true))
            .collect();
        if let Shape::Mapped { write, .. } = &plan.shape {
            opened.push((write.inst, "Write", 0, write, WRITE_PATTERN, false));
        }
        opened.sort_by_key(|o| o.0);
        for (inst, role, number, moved, pattern, looped) in opened {
            let block = moved.block;
            let name = format!("k{role}{number}");
            let pattern = moved.pattern.is_none().then_some(pattern);
            if let Some(pair) = out.iter_mut().find(|p| p.name == name) {
                pair.pattern = pair.pattern.or(pattern);
                continue;
            }
            let first = if role == "Read" {
                FIRST_READ
            } else {
                FIRST_WRITE
            };
            let mut index = first;
            while out
                .iter()
                .any(|p| p.index == index || p.index + 1 == index || p.index == index + 1)
            {
                index += 2;
            }
            let anchor = Anchor::Inst(body, inst);
            let head = if looped { head_last(lines, plan) } else { None };
            let what = lines
                .comment(anchor, head)
                .unwrap_or_else(|| format!("{name}: a {} stream", role.to_lowercase()));
            let items = if role == "Read" {
                READ_ITEMS
            } else {
                WRITE_ITEMS
            };
            out.push(Pair {
                name,
                index,
                block,
                pattern,
                what,
                format: plan.words.get(moved.tensor).and_then(|w| w.format),
                pages: items * block,
            });
        }
    }
    out
}

/// A buffer other than a pair (`layout.other_buffer`, `layout.out_buffer`): a sum's
/// arrival, `{index, slots, pages a slot}`, and its out buffer, the reduce's tile.
struct Other {
    name: String,
    index: u32,
    /// An arrival's slots; none for an out buffer or a field.
    slots: Option<u64>,
    pages: u64,
    what: String,
    /// A field of sums (a product's block var): its pages stated, unpacked as float32.
    field: bool,
    /// Its format: an out buffer's is its tensor's, the others' float32.
    format: Format,
}

/// The sums' buffers in the order they are made: each reduce's arrival, from index 8,
/// `slots` of one tile (`tree_slots`); then its out buffer, from 16, two writes.
fn others(plans: &[Plan], lines: &Lines, slots: u64) -> Vec<Other> {
    let mut out: Vec<Other> = Vec::new();
    let free = |out: &[Other], first: u32| {
        (first..)
            .find(|i| out.iter().all(|o| o.index != *i))
            .unwrap_or(first)
    };
    for plan in plans {
        if let (Shape::Fields(fields), Some(name)) = (&plan.shape, out_rows(plans, plan)) {
            // the rows the vars' sums are packed into, written by mover B: one buffer a
            // width and format (`layout.out_buffer`, `writes.write_sums`)
            let c = fields.width;
            if let Some(held) = out.iter_mut().find(|o| o.name == name) {
                held.pages = held.pages.max(c);
                continue;
            }
            let body = BodyRef::Handler(plan.program, plan.handler);
            let what = lines
                .comment(Anchor::Inst(body, fields.store.inst), None)
                .unwrap_or_else(|| format!("{name}: a store's"));
            let index = free(&out, FIRST_WRITE);
            out.push(Other {
                name,
                index,
                slots: None,
                pages: c,
                what,
                field: false,
                format: store_format(plan, &fields.store),
            });
            continue;
        }
        if let Shape::Product(product) = &plan.shape {
            // the tile each rank stores, packed by compute and written by mover B
            let body = BodyRef::Handler(plan.program, plan.handler);
            let what = lines
                .comment(Anchor::Inst(body, product.store.inst), None)
                .unwrap_or_else(|| "kOut0: a store's".to_string());
            let index = free(&out, FIRST_WRITE);
            out.push(Other {
                name: "kOut0".to_string(),
                index,
                slots: None,
                pages: WRITE_ITEMS,
                what,
                field: false,
                format: Format::F32,
            });
            continue;
        }
        let Shape::Summed(sum) = &plan.shape else {
            continue;
        };
        let anchor = Anchor::Inst(BodyRef::Handler(plan.program, plan.handler), sum.reduce);
        let what = |name: &str| {
            lines
                .comment(anchor, None)
                .unwrap_or_else(|| format!("{name}: a reduce's"))
        };
        let index = free(&out, FIRST_OTHER);
        out.push(Other {
            name: "kArrival0".to_string(),
            index,
            slots: Some(slots),
            pages: slots,
            what: what("kArrival0"),
            field: false,
            format: Format::F32,
        });
        let index = free(&out, FIRST_WRITE);
        out.push(Other {
            name: "kOut0".to_string(),
            index,
            slots: None,
            pages: WRITE_ITEMS,
            what: what("kOut0"),
            field: false,
            format: Format::F32,
        });
    }
    // the fields the vars' sums are added into, after the handlers' (`sums_field`)
    for plan in plans {
        let Shape::Fields(fields) = &plan.shape else {
            continue;
        };
        let c = fields.width;
        for sum in &fields.sums {
            let index = free(&out, FIRST_OTHER);
            out.push(Other {
                name: format!("kField{}", camel(&sum.field.ident)),
                index,
                slots: None,
                pages: c,
                what: format!("field {}: float32, 1 x {c} x 32 x 32", sum.field.ident),
                field: true,
                format: Format::F32,
            });
        }
    }
    out
}

/// The format of the tensor `moved` writes, bf16 when it states none.
fn store_format(plan: &Plan, moved: &Moved) -> Format {
    plan.words
        .get(moved.tensor)
        .and_then(|w| w.format)
        .unwrap_or(Format::Bf16)
}

/// The name of the rows buffer a plan of `Fields` packs into (`kOutRows<c>`): one a
/// width and format, a second format of a width named with a `b` more
/// (`layout.out_buffer`); none for another plan.
fn out_rows(plans: &[Plan], plan: &Plan) -> Option<String> {
    let mut named: Vec<(String, u64, Format)> = Vec::new();
    for p in plans {
        let Shape::Fields(fields) = &p.shape else {
            continue;
        };
        let (c, format) = (fields.width, store_format(p, &fields.store));
        let base = format!("kOutRows{c}");
        let held = named
            .iter()
            .find(|(n, w, f)| n.starts_with(&base) && *w == c && *f == format)
            .map(|(n, ..)| n.clone());
        let name = held.unwrap_or_else(|| {
            let mut name = base.clone();
            while named.iter().any(|(n, ..)| *n == name) {
                name.push('b');
            }
            named.push((name.clone(), c, format));
            name
        });
        if std::ptr::eq(p, plan) {
            return Some(name);
        }
    }
    None
}

/// Whether compute unpacks buffers of more than one format, and packs into buffers of
/// more than one (`program_sources.mixed_formats`): the reads, the arrivals and the
/// fields it unpacks, the outs, the writes and the fields of sums it packs into.
fn mixed_formats(plans: &[Plan], others: &[Other]) -> (bool, bool) {
    let mut unpacked: Vec<Format> = Vec::new();
    let mut packed: Vec<Format> = Vec::new();
    let add = |list: &mut Vec<Format>, f: Format| {
        if !list.contains(&f) {
            list.push(f);
        }
    };
    for pair in pairs(plans, &Lines { origin: None }) {
        let format = pair.format.unwrap_or(Format::Bf16);
        if pair.name.starts_with("kRead") {
            add(&mut unpacked, format);
        } else {
            add(&mut packed, format);
        }
    }
    for other in others {
        if other.slots.is_some() || other.field {
            add(&mut unpacked, Format::F32);
        }
        if other.slots.is_none() {
            add(&mut packed, other.format);
        }
    }
    (unpacked.len() > 1, packed.len() > 1)
}

/// The line of instruction `inst` of `plan` in the source, else its number.
fn line_of(lines: &Lines, plan: &Plan, inst: InstId) -> u32 {
    let body = BodyRef::Handler(plan.program, plan.handler);
    lines
        .place(Anchor::Inst(body, inst))
        .map_or(inst.0, |p| p.first)
}

/// The last head line of a compound statement: the line before its body's first.
fn head_last(lines: &Lines, plan: &Plan) -> Option<u32> {
    if let Shape::Dropped = plan.shape {
        // a loop whose body is one line that makes nothing (`pass`): its head ends
        // the line before the statement's last
        let body = BodyRef::Handler(plan.program, plan.handler);
        let place = lines.place(Anchor::Inst(body, plan.reads[0].inst))?;
        return Some(place.last.saturating_sub(1).max(place.first));
    }
    head_before(lines, plan, plan.looped())
}

/// The line before the place of `inner`, the first instruction of a body.
fn head_before(lines: &Lines, plan: &Plan, inner: InstId) -> Option<u32> {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let inner = lines.place(Anchor::Inst(body, inner))?;
    Some(inner.first.saturating_sub(1))
}

/// The header both kernels include (`program.header`).
fn header_text(
    name: &str,
    plans: &[Plan],
    pairs: &[Pair],
    others: &[Other],
    lines: &Lines,
    chained: bool,
) -> String {
    let mut out = vec![
        format!("// {}", lines.made(name)),
        "// the handlers, buffers and command words its kernels share.".to_string(),
        "#pragma once".to_string(),
        String::new(),
        "#include <stdint.h>".to_string(),
        String::new(),
        "#include \"komorebi/device/movement/km_movement.hpp\"".to_string(),
    ];
    if needs(plans).contains(&"port") {
        out.push("#include \"komorebi/device/movement/km_port.hpp\"".to_string());
    }
    out.push(String::new());
    for (number, plan) in plans.iter().enumerate() {
        let words: Vec<String> = plan
            .words
            .iter()
            .enumerate()
            .map(|(i, w)| format!("a{i} {}", w.name))
            .collect();
        let words = if words.is_empty() {
            "no words".to_string()
        } else {
            words.join(", ")
        };
        let line = lines
            .handler_line(plan.program, plan.handler)
            .map_or(String::new(), |l| format!(", line {l}"));
        out.push(format!("// {}{line}: {words}.", plan.name.to_uppercase()));
        out.push(format!(
            "constexpr uint32_t kHandler{} = {};",
            camel(&plan.name),
            number + 1
        ));
    }
    if chained {
        // the walk's handler, after the program's (`program.chain_prepared`)
        out.push("// The chain's walk: a chain command's records, each its handler's case.".into());
        out.push(format!(
            "constexpr uint32_t kHandlerKmChain = {};",
            plans.len() + 1
        ));
    }
    for pair in pairs {
        out.push(String::new());
        out.push(format!("// {}", pair.what));
        out.push(format!(
            "constexpr km::BufferPair {} = {{{}, {}}};",
            pair.name,
            pair.index,
            pair.index + 1
        ));
        out.push(format!(
            "constexpr uint32_t {}Block = {};",
            pair.name, pair.block
        ));
        if let Some(pattern) = pair.pattern {
            out.push(format!(
                "constexpr km::Pattern {}Pattern = km::pattern(\"{pattern}\");",
                pair.name
            ));
        }
    }
    for other in others {
        out.push(String::new());
        out.push(format!("// {}", other.what));
        out.push(match other.slots {
            Some(slots) => format!(
                "constexpr km::Arrival {} = {{{}, {slots}, {}}};",
                other.name,
                other.index,
                other.pages / slots
            ),
            None => format!("constexpr uint32_t {} = {};", other.name, other.index),
        });
        if other.field {
            out.push(format!(
                "constexpr uint32_t {}Pages = {};",
                other.name, other.pages
            ));
        }
    }
    if chained {
        out.push(String::new());
        out.push(format!("// {CHAIN_AREA_WHAT}"));
        out.push(format!(
            "constexpr uint32_t kChainArea = {};",
            chip::CHAIN_BUFFER
        ));
    }
    let (unpack, pack) = mixed_formats(plans, others);
    let or_none = |mixed: bool, call: &str| if mixed { call } else { "((void)0)" }.to_string();
    out.extend([
        String::new(),
        "// Compute's formats: set again before each unpack and pack when its buffers".into(),
        "// have more than one.".into(),
        format!(
            "#define KM_UNPACK_FORMAT_A(cb) {}",
            or_none(unpack, "reconfig_data_format_srca(cb)")
        ),
        format!(
            "#define KM_UNPACK_FORMATS(a, b) {}",
            or_none(unpack, "reconfig_data_format((a), (b))")
        ),
        format!(
            "#define KM_PACK_FORMAT(cb) {}",
            or_none(pack, "pack_reconfig_data_format(cb)")
        ),
    ]);
    out.join("\n") + "\n"
}

/// Lines of `text` at `depth` indents (an empty line stays empty).
fn indented(out: &mut Vec<String>, depth: usize, text: &str) {
    for line in text.split('\n') {
        if line.is_empty() {
            out.push(String::new());
        } else {
            out.push(format!("{}{line}", INDENT.repeat(depth)));
        }
    }
}

/// The tensor parameters' names in the order the handlers take them, each once.
pub(crate) fn tensors(plans: &[Plan]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for plan in plans {
        for w in &plan.words {
            if w.kind == WordKind::Tensor && !out.contains(&w.name) {
                out.push(w.name.clone());
            }
        }
    }
    out
}

/// A command word's line, read as its kind is.
fn word_line(plan: &Plan, i: usize) -> String {
    let w = &plan.words[i];
    match w.kind {
        WordKind::F32 => format!(
            "const float {} = __builtin_bit_cast(float, command.a{i});",
            w.name
        ),
        WordKind::Tensor | WordKind::U32 => {
            format!("const uint32_t {} = command.a{i};", w.name)
        }
        WordKind::Pattern => format!(
            "const km::Pattern {} = km::Pattern::of(command.a{i});",
            w.name
        ),
    }
}

/// A stream's pattern as the kernels print it: its pair's constant, or the word of
/// its `@tensix.pattern`.
fn pattern_text(plan: &Plan, moved: &Moved, pair: &str) -> String {
    moved
        .pattern
        .map_or_else(|| format!("{pair}Pattern"), |w| plan.words[w].name.clone())
}

/// An int as the kernels print it: a word's or a value's name, a number, the rank or
/// size, `(a op b)`.
fn int_text(plan: &Plan, int: &Int) -> String {
    match int {
        Int::Word(i) => plan.words[*i].name.clone(),
        Int::Const(n) => n.to_string(),
        Int::Rank => "km_rank".to_string(),
        Int::Size => "km_size".to_string(),
        Int::Local(l) => plan.locals[*l].name.clone(),
        Int::Bin(op, a, b) => format!("({} {op} {})", int_text(plan, a), int_text(plan, b)),
    }
}

/// What a kernel's lines read of `ints`, the named values they read included: the
/// words, the named values, and whether the rank and the size.
#[derive(Default)]
struct Uses {
    words: Vec<usize>,
    locals: Vec<usize>,
    rank: bool,
    size: bool,
    /// The group (`km_group`), which a reduce reads.
    group: bool,
}

impl Uses {
    fn of<'a>(plan: &Plan, ints: impl IntoIterator<Item = &'a Int>) -> Uses {
        let mut out = Uses::default();
        let mut pending: Vec<Int> = ints.into_iter().cloned().collect();
        while let Some(int) = pending.pop() {
            int.each(&mut |i| match i {
                Int::Word(w) if !out.words.contains(w) => out.words.push(*w),
                Int::Rank => out.rank = true,
                Int::Size => out.size = true,
                Int::Local(l) if !out.locals.contains(l) => {
                    out.locals.push(*l);
                    pending.push(plan.locals[*l].value.clone());
                }
                _ => {}
            });
        }
        out
    }
}

/// The ints a range reads.
fn range_ints<'p>(plan: &'p Plan, range: &'p Range) -> Vec<&'p Int> {
    match range {
        Range::Part(p) => vec![&plan.parts[*p].first, &plan.parts[*p].n],
        Range::Tiles { first, n } => vec![first, n],
        Range::Rect(rect) => vec![&rect.rows.0, &rect.rows.1, &rect.cols.0, &rect.cols.1],
    }
}

/// A rectangle's tiles in the kernels' terms (`ranges.rectangle`, `stream_cxx`): its
/// first tile in rows of the tensor's `width`, and either one run of tiles (a row, or
/// rows the tensor's width) or the walk of its rows in blocks of `block` tiles.
fn rect_tiles(plan: &Plan, rect: &Rect, width: u64, block: u64) -> (String, Option<String>) {
    let (rows, cols) = rect.sides();
    let (rows, cols) = (rows.unwrap_or(0), cols.unwrap_or(0));
    let first = format!(
        "(({}) * {width} + ({}))",
        int_text(plan, &rect.rows.0),
        int_text(plan, &rect.cols.0)
    );
    if rows == 1 || cols == width {
        return (format!("{first}, {}", rows * cols), None);
    }
    let per = block / cols.max(1);
    let blocks = format!("(({rows}) + {}) / {per}", per.saturating_sub(1));
    let walk = format!(
        "km::Walk{{{first}, {blocks}, {}, {per}, {width}, {cols}, 1, {cols}, {block}, ({rows}) - \
         ({blocks} - 1) * {per}, {cols}}}",
        per * width
    );
    (
        walk,
        Some(format!(
            "// the rectangle's rows, {cols} tiles each, {width} apart"
        )),
    )
}

/// A case's opening lines: the rank and size where the kernel reads them, then the
/// words it reads.
fn case_head(out: &mut Vec<String>, plan: &Plan, me: &str, reads: &Uses) {
    if reads.rank {
        indented(
            out,
            4,
            &format!("const uint32_t km_rank = {me}.me() - km::first(command);"),
        );
    }
    if reads.size {
        indented(out, 4, "const uint32_t km_size = km::count(command);");
    }
    if reads.group {
        indented(
            out,
            4,
            &format!("const km::Group km_group = km::group_of(command, {me}.me());"),
        );
    }
    for i in 0..plan.words.len() {
        if reads.words.contains(&i) {
            indented(out, 4, &word_line(plan, i));
        }
    }
}

/// `// comment` at depth 4, if the origin gives one.
fn comment(out: &mut Vec<String>, depth: usize, text: Option<String>) {
    if let Some(text) = text {
        indented(out, depth, &format!("// {text}"));
    }
}

/// A partition's line.
fn part_line(plan: &Plan, part: &Part) -> String {
    format!(
        "const km::Partition {} = km::partition({}, km_rank, km_size);",
        part.name,
        int_text(plan, &part.n)
    )
}

/// The switch's case line of a handler.
fn case_line(plan: &Plan, lines: &Lines) -> String {
    let line = lines
        .handler_line(plan.program, plan.handler)
        .map_or(String::new(), |l| format!("  // line {l}"));
    format!("case kHandler{}: {{{line}", camel(&plan.name))
}

/// The movement source both movers build (`program_sources.movement_source`); a
/// rectangle's walk in rows of its tensor's width, from `shapes` (each tensor's rows
/// and columns of tiles, by name, as the load gives them).
fn movement_text(
    name: &str,
    header: &str,
    plans: &[Plan],
    lines: &Lines,
    shapes: &[(String, Option<(u64, u64)>)],
    chained: bool,
) -> Result<String, Refusal> {
    let mut out = vec![format!("// {}", lines.made(name))];
    if chained {
        out.extend([
            "// the movement source of both movers (mover A: RISCV_0, mover B: RISCV_1),".into(),
            "// with the chain's walk (km_chain.hpp).".into(),
        ]);
    } else {
        out.push(
            "// the movement source of both movers (mover A: RISCV_0, mover B: RISCV_1)."
                .to_string(),
        );
    }
    out.extend([
        "#include <stdint.h>".to_string(),
        String::new(),
        "#include \"api/dataflow/dataflow_api.h\"".to_string(),
        "#include \"komorebi/device/movement/km_movement.hpp\"".to_string(),
    ]);
    let needs = needs(plans);
    if needs.contains(&"port") {
        out.push("#include \"komorebi/device/movement/km_port.hpp\"".to_string());
    }
    if needs.contains(&"ask") {
        out.push("#include \"komorebi/device/tensors/km_ask.hpp\"".to_string());
    }
    if chained {
        out.push(format!("#include \"{CHAIN_INCLUDE}\""));
    }
    out.extend([format!("#include \"{header}\""), String::new()]);
    if chained {
        out.push(CHAIN_HELPERS.to_string());
    }
    out.push("void kernel_main() {".to_string());
    let mut previous: Option<String> = None;
    for t in tensors(plans) {
        let offset = previous.map_or("0".to_string(), |p| {
            format!("{p}::next_compile_time_args_offset()")
        });
        indented(
            &mut out,
            1,
            &format!(
                "using {} [[maybe_unused]] = km::Tensor<{offset}>;  // {t}",
                camel(&t)
            ),
        );
        previous = Some(camel(&t));
    }
    indented(&mut out, 1, "km::Movement movement;");
    if chained {
        out.push("#if defined(COMPILE_FOR_BRISC)  // mover A: the chain's words zero".into());
        indented(&mut out, 1, "km::chain_start();");
        out.push("#endif".into());
    }
    let cases = out.len();
    indented(
        &mut out,
        1,
        "movement.run([&](const km_command_t& command) {",
    );
    indented(&mut out, 2, "switch (km::handler(command)) {");
    for plan in plans {
        let body = BodyRef::Handler(plan.program, plan.handler);
        let start = out.len();
        indented(&mut out, 3, &case_line(plan, lines));
        let written = plan.written();
        let ints = plan
            .reads
            .iter()
            .chain(written)
            .flat_map(|r| range_ints(plan, &r.range))
            .chain(plan.repeat.as_ref());
        let mut reads = Uses::of(plan, ints);
        reads.words.extend(plan.reads.iter().map(|r| r.tensor));
        reads.words.extend(written.map(|w| w.tensor));
        let patterns = plan.reads.iter().chain(written).filter_map(|m| m.pattern);
        reads.words.extend(patterns);
        if !plan.parts.is_empty() {
            (reads.rank, reads.size) = (true, true);
        }
        if let Shape::Summed(_) = plan.shape {
            // the reduce reads the group; the leader is rank 0
            (reads.rank, reads.group) = (true, true);
        }
        case_head(&mut out, plan, "movement", &reads);
        if let Shape::Fields(fields) = &plan.shape {
            // mover B counts the sums its write waits for (`sums_field`)
            out.push("#if !defined(COMPILE_FOR_BRISC)  // mover B".to_string());
            for sum in &fields.sums {
                indented(
                    &mut out,
                    4,
                    &format!(
                        "uint32_t km_sums_{} = 0;  // {}'s",
                        sum.field.ident, sum.field.source
                    ),
                );
            }
            out.push("#endif".to_string());
        }
        let stream = |kind: &str, var: &str, pair: &str, moved: &Moved| {
            let t = &plan.words[moved.tensor].name;
            let at = Anchor::Inst(body, moved.inst);
            let mut walked = None;
            let tiles = match &moved.range {
                Range::Rect(rect) => {
                    let shape = shapes.iter().find(|(n, _)| n == t).and_then(|(_, s)| *s);
                    let Some((rows, width)) = shape else {
                        return Err(Refusal {
                            code: NOT_BUILT,
                            what: format!(
                                "the tensor `{t}` is given with no shape in tiles, and its \
                                 rectangle is walked in rows of its width"
                            ),
                            at: Some(at),
                        });
                    };
                    fits(plan, rect, t, (rows, width), at)?;
                    let (tiles, comment) = rect_tiles(plan, rect, width, moved.block);
                    walked = comment.map(|c| (c, tiles.clone()));
                    tiles
                }
                Range::Part(p) => {
                    // a partition's first tile: its range's first plus its start
                    let part = &plan.parts[*p];
                    let start = match &part.first {
                        Int::Const(0) => format!("{}.start", part.name),
                        first => format!("({} + {}.start)", int_text(plan, first), part.name),
                    };
                    format!("{start}, {}.len", part.name)
                }
                Range::Tiles { first, n } => {
                    format!("{}, {}", int_text(plan, first), int_text(plan, n))
                }
            };
            let pattern = pattern_text(plan, moved, pair);
            // a read stream's repeat: the counted loop around its loop (R46)
            let repeat = match (&plan.repeat, kind) {
                (Some(r), "km::kMoveRead") => int_text(plan, r),
                _ => "1".to_string(),
            };
            Ok(match walked {
                Some((comment, walk)) => format!(
                    "{comment}\nconst km::Stream {var}(\n{INDENT}{kind}, {}({t}), {walk}, {repeat}, {pair}, {pattern}\n);",
                    camel(t),
                ),
                None => format!(
                    "const km::Stream {var}(\n{INDENT}{kind}, {}({t}), {tiles}, {pair}Block, {repeat}, {pair}, {pattern}\n);",
                    camel(t),
                ),
            })
        };
        // The statements in the order written, a comment where the place changes.
        let mut statements: Vec<(InstId, String, Option<u32>)> = plan
            .parts
            .iter()
            .map(|part| (part.inst, part_line(plan, part), None))
            .collect();
        for (l, local) in plan.locals.iter().enumerate() {
            if reads.locals.contains(&l) {
                let text = format!(
                    "const uint32_t {} = {};",
                    local.name,
                    int_text(plan, &local.value)
                );
                statements.push((local.inst, text, None));
            }
        }
        if let Shape::Mapped { write, .. } = &plan.shape {
            statements.push((
                write.inst,
                stream("km::kMoveWrite", "km_write0", "kWrite0", write)?,
                None,
            ));
        }
        for (i, read) in plan.reads.iter().enumerate() {
            statements.push((
                read.inst,
                stream(
                    "km::kMoveRead",
                    &format!("km_read{i}"),
                    &format!("kRead{i}"),
                    read,
                )?,
                head_last(lines, plan),
            ));
        }
        statements.sort_by_key(|s| s.0);
        let mut at: Option<Place> = None;
        for (inst, text, head) in statements {
            let anchor = Anchor::Inst(body, inst);
            let here = lines.place(anchor);
            if here.is_none() || here != at {
                comment(&mut out, 4, lines.comment(anchor, head));
            }
            at = here;
            indented(&mut out, 4, &text);
        }
        match &plan.shape {
            Shape::Product(product) => movement_product(&mut out, plan, product, lines),
            Shape::Fields(fields) => {
                let Range::Rect(rect) = &fields.store.range else {
                    unreachable!("a store of block vars is into a rectangle (plan)");
                };
                let t = &plan.words[fields.store.tensor].name;
                let shape = shapes.iter().find(|(n, _)| n == t).and_then(|(_, s)| *s);
                let at = Anchor::Inst(body, fields.store.inst);
                let Some(shape) = shape else {
                    return Err(Refusal {
                        code: NOT_BUILT,
                        what: format!(
                            "the tensor `{t}` is given with no shape in tiles, and its \
                             rectangle is written in rows of its width"
                        ),
                        at: Some(at),
                    });
                };
                fits(plan, rect, t, shape, at)?;
                let rows = out_rows(plans, plan).unwrap_or_default();
                movement_fields(&mut out, plan, fields, rect, shape.1, &rows, lines);
            }
            Shape::Mapped { .. } => {
                let moved: Vec<String> = (0..plan.reads.len())
                    .map(|i| format!("km_read{i}"))
                    .chain(["km_write0".to_string()])
                    .collect();
                indented(
                    &mut out,
                    4,
                    &format!(
                        "if (!movement.move({})) {{\n    return false;\n}}",
                        moved.join(", ")
                    ),
                );
            }
            Shape::Summed(sum) => movement_sum(&mut out, plan, sum, lines),
            Shape::Dropped => indented(
                &mut out,
                4,
                "if (!movement.move(km_read0)) {\n    return false;\n}",
            ),
        }
        indented(&mut out, 4, "return true;");
        indented(&mut out, 3, "}");
        if chained {
            let case = out.split_off(start);
            out.extend(chain_case(plan, lines, case));
        }
    }
    indented(&mut out, 3, "default:");
    indented(&mut out, 4, "return true;");
    indented(&mut out, 2, "}");
    if chained {
        // one copy of the cases, for a plain command and a chain's records
        let switch: Vec<String> = out
            .split_off(cases + 1)
            .into_iter()
            .map(|l| l.strip_prefix(INDENT).map_or(l.clone(), str::to_string))
            .collect();
        out.truncate(cases);
        indented(
            &mut out,
            1,
            "const auto km_case = [&](const km_command_t& command)",
        );
        indented(&mut out, 3, "__attribute__((noinline)) -> bool {");
        out.extend(switch);
        indented(&mut out, 1, "};");
        indented(
            &mut out,
            1,
            "movement.run([&](const km_command_t& command) {",
        );
        indented(
            &mut out,
            2,
            "if (km::handler(command) != kHandlerKmChain) {",
        );
        indented(&mut out, 3, "return km_case(command);");
        indented(&mut out, 2, "}");
        out.push("#if defined(COMPILE_FOR_BRISC)  // mover A".into());
        indented(&mut out, 2, "km::chain_reader(movement, command, km_case);");
        indented(&mut out, 2, "return true;");
        out.push("#else  // mover B".into());
        indented(
            &mut out,
            2,
            "return km::chain_writer(km::ChainShape{}, movement, command, km_case);",
        );
        out.push("#endif".into());
    }
    indented(&mut out, 1, "});");
    out.push("}".to_string());
    Ok(out.join("\n") + "\n")
}

/// The device header of the chain's walk, which both kernels of a chained program
/// include.
const CHAIN_INCLUDE: &str = "komorebi/device/movement/km_chain.hpp";

/// The movement source's helpers of the chain's walk (`program_sources.CHAIN_HELPERS`):
/// a record's wait for its input before a read, and its end's marks on mover B.
const CHAIN_HELPERS: &str = "namespace {

    bool km_chain_waited = false;  // this record's input waited for

    // Before a line reading tensors `bits` of a chain's record: its input's marks
    // waited for once, unless `mask` (the chain's written tensors) has none of them.
    __attribute__((noinline)) bool km_chain_read(
        km::Movement& movement, uint32_t mask, uint32_t bits
    ) {
        if (km_chain_waited || !km::chain_now.on || (mask & bits) == 0) {
            return true;
        }
        km_chain_waited = true;
        return km::chain_input(km::ChainShape{}, movement);
    }

    // A record's end: on mover B each group's root marks the next record's cores once
    // its block is written.
    __attribute__((noinline)) bool km_chain_end(
        km::Movement& movement, const km_command_t& command
    ) {
        if (!km::chain_now.on) {
            return true;
        }
#if defined(COMPILE_FOR_BRISC)  // mover A
        return true;
#else  // mover B
        const uint32_t size = km::ChainShape{}.size(movement, km::chain_now.now);
        if ((movement.me() - km::first(command)) % size != 0) {
            return true;
        }
        const km::ChainDirect none = {0, 0, 0, 0};
        return km::chain_mark(km::ChainShape{}, movement, command, none, false);
#endif
    }

}  // namespace
";

/// A handler's movement case in a chained program (`program_sources.chain_cases`):
/// the record's flag cleared first; before each line that reads a tensor, the wait
/// for the record's input, held by one gate over a loop's streams in step
/// (`chain_gated`), else `km_chain_read`; the end's marks in place of `return true`.
/// A tensor's bit is its place among the handler's tensor words; the mask word is the
/// first after the handler's own (none past the tenth: the whole wait).
fn chain_case(plan: &Plan, lines: &Lines, case: Vec<String>) -> Vec<String> {
    let tensors: Vec<usize> = (0..plan.words.len())
        .filter(|&i| plan.words[i].kind == WordKind::Tensor)
        .collect();
    let bit = |tensor: usize| 1u32 << tensors.iter().position(|&t| t == tensor).unwrap_or(0);
    let mask = match plan.words.len() {
        n if n < 10 => format!("command.a{n}"),
        _ => "~0u".to_string(),
    };
    // each read's statement line: the bits of its tensors, its streams
    let mut reads: Vec<(u32, u32, Vec<usize>)> = Vec::new();
    for (i, read) in plan.reads.iter().enumerate() {
        let line = line_of(lines, plan, read.inst);
        match reads.iter_mut().find(|r| r.0 == line) {
            Some(r) => {
                r.1 |= bit(read.tensor);
                r.2.push(i);
            }
            None => reads.push((line, bit(read.tensor), vec![i])),
        }
    }
    let pad = |text: &str| {
        let n = text.len() - text.trim_start().len();
        text[..n].to_string()
    };
    let head = case[0].clone();
    let mut body: Vec<String> = case[1..case.len() - 2].to_vec();
    let mut out: Vec<String> = Vec::new();
    let mut found: Vec<u32> = Vec::new();
    let mut at = 0;
    while at < body.len() {
        let text = body[at].clone();
        out.push(text.clone());
        let line = text
            .trim_start()
            .strip_prefix("// line ")
            .and_then(|l| l.split(':').next())
            .and_then(|n| n.parse::<u32>().ok());
        let read = line.and_then(|l| reads.iter().find(|r| r.0 == l));
        if let (Some(line), Some((_, bits, streams))) = (line, read) {
            let indent = pad(&text);
            found.push(line);
            let gate = (streams.len() > 1)
                .then(|| chain_gated(plan, &mut body, at, streams, &mask, *bits, &bit))
                .flatten();
            match gate {
                Some(gate) => out.push(format!("{indent}{gate}")),
                None => {
                    out.push(format!(
                        "{indent}if (!km_chain_read(movement, {mask}, {bits}u)) {{"
                    ));
                    out.push(format!("{indent}{INDENT}return false;"));
                    out.push(format!("{indent}}}"));
                }
            }
        }
        at += 1;
    }
    if reads.iter().any(|r| !found.contains(&r.0)) {
        // a read's line not seen: the wait first
        let indent = INDENT.repeat(4);
        out.splice(
            0..0,
            [
                format!("{indent}if (!km_chain_read(movement, ~0u, ~0u)) {{"),
                format!("{indent}{INDENT}return false;"),
                format!("{indent}}}"),
            ],
        );
    }
    let mut case = vec![head];
    indented(&mut case, 4, "km_chain_waited = false;");
    case.extend(out);
    indented(&mut case, 4, "return km_chain_end(movement, command);");
    indented(&mut case, 3, "}");
    case
}

/// The gate of a loop over `streams` in step whose comment is at `at` in `body`: its
/// move's read streams each held by it, rewritten in `body`, and the gate's line;
/// none when the loop's move is not found before the next statement.
fn chain_gated(
    plan: &Plan,
    body: &mut [String],
    at: usize,
    streams: &[usize],
    mask: &str,
    bits: u32,
    bit: &dyn Fn(usize) -> u32,
) -> Option<String> {
    let gate = format!("km_gate{}", streams[0]);
    for text in body.iter_mut().skip(at + 1) {
        if text.trim_start().starts_with("// line ") {
            return None;
        }
        if !text.contains("movement.move(") {
            continue;
        }
        let mut moved = text.clone();
        for &i in streams {
            let var = format!("km_read{i}");
            let held = format!("km::Gated({var}, {gate}, {}u)", bit(plan.reads[i].tensor));
            let words: Vec<&str> = moved.split(&var).collect();
            let alone = words.len() == 2
                && !words[0].ends_with(|c: char| c.is_alphanumeric() || c == '_')
                && !words[1].starts_with(|c: char| c.is_alphanumeric() || c == '_');
            if !alone {
                return None;
            }
            moved = words.join(&held);
        }
        *text = moved;
        return Some(format!(
            "km::ChainGate {gate} = km::chain_gate(km::ChainShape{{}}, movement, {mask}, \
             {bits}u, km_chain_waited);"
        ));
    }
    None
}

/// Refuses at `at` a rectangle of tensor `t` (rows and columns of tiles `shape`) whose
/// ends the compile knows and that does not fit it; ends of the run are checked at
/// submit (the entry's rectangles).
fn fits(plan: &Plan, rect: &Rect, t: &str, shape: (u64, u64), at: Anchor) -> Result<(), Refusal> {
    for ((lo, hi), held, what) in [
        (&rect.rows, shape.0, "rows"),
        (&rect.cols, shape.1, "columns"),
    ] {
        let end = match (lo, hi) {
            (_, Int::Const(hi)) => Some(*hi),
            _ => None,
        };
        if end.is_some_and(|end| end > held) {
            return Err(Refusal {
                code: NOT_BUILT,
                what: format!(
                    "the rectangle's {what} {}:{} of `{t}`, which has {held} {what} of tiles \
                     at this load",
                    int_text(plan, lo),
                    int_text(plan, hi)
                ),
                at: Some(at),
            });
        }
    }
    Ok(())
}

/// A product's movement after its streams (`writes.write_moved`): mover B writes the
/// tile compute packs of the var, each rank its own.
fn movement_product(out: &mut Vec<String>, plan: &Plan, product: &Product, lines: &Lines) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let moved: Vec<String> = (0..plan.reads.len())
        .map(|i| format!("km_read{i}"))
        .collect();
    indented(
        out,
        4,
        &format!(
            "if (!movement.move({})) {{\n    return false;\n}}",
            moved.join(", ")
        ),
    );
    let store = &product.store;
    comment(out, 4, lines.comment(Anchor::Inst(body, store.inst), None));
    out.push("#if !defined(COMPILE_FOR_BRISC)  // mover B".to_string());
    let t = &plan.words[store.tensor].name;
    let ints = range_ints(plan, &store.range);
    let (first, n) = (int_text(plan, ints[0]), int_text(plan, ints[1]));
    indented(
        out,
        4,
        &format!(
            "if (!movement.wait_front(kOut0, {n}) ||\n    \
             !km::write_pages(movement, command, {}({t}), {first}, {n}, get_read_ptr(kOut0))) {{\n    \
             return false;\n}}\nnoc_async_writes_flushed();\ncb_pop_front(kOut0, {n});",
            camel(t)
        ),
    );
    out.push("#endif".to_string());
}

/// The movement of products into block vars after their streams (`writes.write_sums`,
/// `write_function_of_sums`): mover B counts the loop's sums of each var, and when
/// there were some writes the rectangle a row at a time from `rows`, the tensor `width`
/// tiles wide.
fn movement_fields(
    out: &mut Vec<String>,
    plan: &Plan,
    fields: &Fields,
    rect: &Rect,
    width: u64,
    rows: &str,
    lines: &Lines,
) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let (_, _, p) = compute_count(plan);
    let moved: Vec<String> = (0..plan.reads.len())
        .map(|i| format!("km_read{i}"))
        .collect();
    indented(
        out,
        4,
        &format!(
            "if (!movement.move({})) {{\n    return false;\n}}",
            moved.join(", ")
        ),
    );
    let sums: Vec<String> = fields
        .sums
        .iter()
        .map(|s| format!("km_sums_{}", s.field.ident))
        .collect();
    out.push("#if !defined(COMPILE_FOR_BRISC)  // mover B".to_string());
    for sum in &sums {
        indented(
            out,
            4,
            &format!("{sum} += ({p} + kRead0Block - 1) / kRead0Block;"),
        );
    }
    out.push("#endif".to_string());
    let store = &fields.store;
    comment(out, 4, lines.comment(Anchor::Inst(body, store.inst), None));
    out.push("#if !defined(COMPILE_FOR_BRISC)  // mover B".to_string());
    let stored: Vec<&String> = match &fields.stored {
        Stored::Var(k) => vec![&sums[*k]],
        Stored::Gate(gate) => gate.vars.iter().map(|k| &sums[*k]).collect(),
    };
    let made: Vec<String> = stored.iter().map(|s| format!("{s} > 0")).collect();
    let t = &plan.words[store.tensor].name;
    let line = line_of(lines, plan, store.inst);
    let (r, cols) = rect.sides();
    let (r, c) = (r.unwrap_or(1), cols.unwrap_or(0));
    let first = format!(
        "(({}) * {width} + ({}))",
        int_text(plan, &rect.rows.0),
        int_text(plan, &rect.cols.0)
    );
    let resets: Vec<String> = stored.iter().map(|s| format!("{s} = 0;")).collect();
    // a row's step is the rectangle's columns, as today's (`row_step`): one row only
    indented(
        out,
        4,
        &format!(
            "if ({}) {{\n    \
             for (uint32_t km_row{line} = 0; km_row{line} < {r}; km_row{line}++) {{\n        \
             if (!movement.wait_front({rows}, {c}) ||\n            \
             !km::write_pages(movement, command, {}({t}), {first} + km_row{line} * {c}, {c}, get_read_ptr({rows}))) {{\n            \
             return false;\n        }}\n        \
             noc_async_writes_flushed();\n        \
             cb_pop_front({rows}, {c});\n    }}\n}}\n{}",
            made.join(" && "),
            camel(t),
            resets.join("\n")
        ),
    );
    out.push("#endif".to_string());
}

/// A sum's movement after its stream (`reduce.tile_reduce`, `writes.write_moved`): mover
/// A takes the arrivals, mover B sends the total up the tree; the leader, rank 0, has
/// mover B write the reduce's tile.
fn movement_sum(out: &mut Vec<String>, plan: &Plan, sum: &Sum, lines: &Lines) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let reduce = "km::kReduceTree";
    indented(
        out,
        4,
        "if (!movement.move(km_read0)) {\n    return false;\n}",
    );
    comment(out, 4, lines.comment(Anchor::Inst(body, sum.reduce), None));
    out.push("#if defined(COMPILE_FOR_BRISC)  // mover A".to_string());
    indented(
        out,
        4,
        &format!(
            "if (!km::reduce_arrive(movement, command, kArrival0, km_group, {reduce})) {{\n    return false;\n}}"
        ),
    );
    out.push("#else  // mover B".to_string());
    indented(
        out,
        4,
        &format!(
            "if (!km::reduce_emit(movement, command, kArrival0, km_group, {reduce}, kOut0, 1)) {{\n    return false;\n}}"
        ),
    );
    out.push("#endif".to_string());
    let head = head_before(lines, plan, sum.store.inst);
    comment(out, 4, lines.comment(Anchor::Inst(body, sum.leader), head));
    indented(out, 4, "if (km_rank == 0) {");
    comment(
        out,
        5,
        lines.comment(Anchor::Inst(body, sum.store.inst), None),
    );
    out.push("#if !defined(COMPILE_FOR_BRISC)  // mover B".to_string());
    let t = &plan.words[sum.store.tensor].name;
    let ints = range_ints(plan, &sum.store.range);
    let (first, n) = (int_text(plan, ints[0]), int_text(plan, ints[1]));
    indented(
        out,
        5,
        &format!(
            "if (!km::reduce_result(movement, kOut0, {n}) ||\n    \
             !km::write_pages(movement, command, {}({t}), {first}, {n}, get_read_ptr(kOut0))) {{\n    \
             return false;\n}}\nnoc_async_writes_flushed();\ncb_pop_front(kOut0, {n});",
            camel(t)
        ),
    );
    out.push("#endif".to_string());
    indented(out, 4, "}");
}

/// The compute source (`program.compute_source`): each case's loop over the read
/// stream's blocks, its tiles `DST_HALF` an acquire, the call on each and the packs.
fn compute_text(
    name: &str,
    header: &str,
    plans: &[Plan],
    units: &[Unit],
    lines: &Lines,
    chained: bool,
) -> String {
    let mut out = vec![
        format!("// {}", lines.made(name)),
        "// the compute kernel: the tiles of its streams, in block order.".to_string(),
        "#include <stdint.h>".to_string(),
        String::new(),
    ];
    let needs = needs(plans);
    out.extend(
        COMPUTE_INCLUDES
            .iter()
            .filter(|(_, need)| need.is_none_or(|n| needs.contains(&n)))
            .map(|(i, _)| format!("#include \"{i}\"")),
    );
    let (unpack, pack) = mixed_formats(plans, &others(plans, lines, TREE_ROUNDS));
    if unpack || pack {
        out.push("#include \"api/compute/reconfig_data_format.h\"".to_string());
    }
    out.extend(
        units
            .iter()
            .map(|u| format!("#include \"tile_{}.hpp\"", u.name)),
    );
    if chained {
        out.push(format!("#include \"{CHAIN_INCLUDE}\""));
    }
    out.push(format!("#include \"{header}\""));
    out.push(String::new());
    out.push("void kernel_main() {".to_string());
    indented(&mut out, 1, "km::Compute compute;");
    // the first read and the first written buffer (a sum's out buffer)
    let rows = plans.first().and_then(|p| out_rows(plans, p));
    let written = match plans.first().map(|p| &p.shape) {
        Some(Shape::Fields(_)) => rows.as_deref().unwrap_or("kOut0"),
        Some(Shape::Summed(_) | Shape::Product(_)) => "kOut0",
        _ => "kWrite0.mover_a",
    };
    indented(
        &mut out,
        1,
        &format!("compute_kernel_hw_startup(kRead0.mover_a, {written});"),
    );
    if chained {
        // one copy of the cases, for a plain command and a chain's records
        indented(
            &mut out,
            1,
            "const auto km_case = [&](const km_command_t& command)",
        );
        indented(&mut out, 3, "__attribute__((noinline)) {");
    } else {
        indented(
            &mut out,
            1,
            "compute.run([&](const km_command_t& command) {",
        );
    }
    indented(&mut out, 2, "switch (km::handler(command)) {");
    // each mapped plan's unit, in order (`planned`)
    let mut called = units.iter();
    for plan in plans {
        indented(&mut out, 3, &case_line(plan, lines));
        match &plan.shape {
            Shape::Mapped {
                call: Some(call),
                write,
                ..
            } => {
                let unit = called.next().expect("a unit for each mapped plan");
                compute_mapped(&mut out, plan, Some((call, unit)), write, lines);
            }
            Shape::Mapped { write, .. } => {
                compute_mapped(&mut out, plan, None, write, lines);
            }
            Shape::Summed(sum) => compute_sum(&mut out, plan, sum, lines),
            Shape::Product(product) => compute_product(&mut out, plan, product, lines),
            Shape::Fields(fields) => {
                let unit = match &fields.stored {
                    Stored::Gate(_) => called.next(),
                    Stored::Var(_) => None,
                };
                let rows = out_rows(plans, plan).unwrap_or_default();
                compute_fields(&mut out, plan, fields, unit, &rows, lines);
            }
            Shape::Dropped => compute_dropped(&mut out, plan, lines),
        }
        indented(&mut out, 4, "break;");
        indented(&mut out, 3, "}");
    }
    indented(&mut out, 3, "default:");
    indented(&mut out, 4, "break;");
    indented(&mut out, 2, "}");
    if chained {
        indented(&mut out, 1, "};");
        indented(
            &mut out,
            1,
            "compute.run([&](const km_command_t& command) {",
        );
        indented(
            &mut out,
            2,
            "if (km::handler(command) == kHandlerKmChain) {",
        );
        indented(&mut out, 3, "km::chain_compute(compute, command, km_case);");
        indented(&mut out, 2, "} else {");
        indented(&mut out, 3, "km_case(command);");
        indented(&mut out, 2, "}");
    }
    indented(&mut out, 1, "});");
    out.push("}".to_string());
    out.join("\n") + "\n"
}

/// The read stream's count and what compute reads for it: its partition's length,
/// or its tiles' count.
fn compute_count(plan: &Plan) -> (Uses, Option<&Part>, String) {
    match &plan.reads[0].range {
        Range::Part(p) => {
            let part = &plan.parts[*p];
            let mut reads = Uses::of(plan, [&part.n]);
            (reads.rank, reads.size) = (true, true);
            (reads, Some(part), format!("{}.len", part.name))
        }
        Range::Tiles { n, .. } => (Uses::of(plan, [n]), None, int_text(plan, n)),
        Range::Rect(rect) => {
            let (rows, cols) = rect.sides();
            let tiles = rows.unwrap_or(0) * cols.unwrap_or(0);
            (Uses::default(), None, tiles.to_string())
        }
    }
}

/// A product's compute case (`products.stream_product`): the var in Dst tile 0 from
/// zeros; the matrix unit set once for the loop; each tile k of the first stream's
/// block by row k of the second's added into it (`matmul_block`); each block freed;
/// the var packed into the out buffer.
fn compute_product(out: &mut Vec<String>, plan: &Plan, product: &Product, lines: &Lines) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let (reads, _, p) = compute_count(plan);
    case_head(out, plan, "compute", &reads);
    indented(out, 4, "bool km_ok = true;");
    comment(
        out,
        4,
        lines.comment(Anchor::Inst(body, product.zeros), None),
    );
    indented(
        out,
        4,
        "tile_regs_acquire();\nfill_tile_init();\nfill_tile(0, 0.0f);",
    );
    comment(
        out,
        4,
        lines.comment(
            Anchor::Inst(body, plan.reads[0].inst),
            head_last(lines, plan),
        ),
    );
    let width = product.width;
    indented(
        out,
        4,
        &format!(
            "const uint32_t km_blocks0 = ({p} + kRead0Block - 1) / kRead0Block;\n\
             // the zeros' fills (vector unit) done before the matrix unit adds\n\
             MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n\
             KM_UNPACK_FORMATS(kRead1.mover_a, kRead0.mover_a);\n\
             matmul_block_init(kRead0.mover_a, kRead1.mover_a, 0, {width}, 1, kRead0Block);\n\
             for (uint32_t km_block0 = 0; km_ok && km_block0 < km_blocks0; km_block0++) {{"
        ),
    );
    for i in 0..plan.reads.len() {
        indented(
            out,
            5,
            &format!(
                "const uint32_t km_half{i} = km::block_half(kRead{i}, kRead{i}Pattern, km_block0);\n\
                 km_ok = compute.wait_front(km_half{i}, kRead{i}Block);\n\
                 if (!km_ok) {{\n    break;\n}}"
            ),
        );
    }
    indented(
        out,
        5,
        &format!(
            "const uint32_t km_valid0 = km_block0 % km_blocks0 + 1 == km_blocks0 ? {p} - (km_blocks0 - 1) * kRead0Block : kRead0Block;\n\
             for (uint32_t km_tile0 = 0; km_tile0 < km_valid0; km_tile0++) {{"
        ),
    );
    comment(
        out,
        6,
        lines.comment(Anchor::Inst(body, product.call.inst), None),
    );
    indented(
        out,
        6,
        &format!(
            "matmul_block(km_half0, km_half1, km_tile0, km_tile0 * {width}, 0, 0, {width}, 1, kRead0Block);"
        ),
    );
    indented(out, 5, "}");
    for i in 0..plan.reads.len() {
        indented(out, 5, &format!("cb_pop_front(km_half{i}, kRead{i}Block);"));
    }
    indented(out, 4, "}");
    comment(
        out,
        4,
        lines.comment(Anchor::Inst(body, product.store.inst), None),
    );
    indented(
        out,
        4,
        "tile_regs_commit();\n\
         tile_regs_wait();\n\
         if (km_ok && compute.reserve_back(kOut0, 1)) {\n    \
         KM_PACK_FORMAT(kOut0);\n    \
         pack_tile(0, kOut0);\n    \
         cb_push_back(kOut0, 1);\n\
         }\n\
         tile_regs_release();",
    );
}

/// The compute case of products into block vars (`stream_field_product`,
/// `writes.write_sums`, `write_function_of_sums`): each block's product of each var by
/// sub-columns of Dst's half, packed into its field with the packer's L1 accumulation
/// (the first block overwrites); the matrix unit set for each product other than the
/// first and set back. After the loop, the stored var's field copied to Dst and packed
/// into `rows`, or each tile of the vars' fields copied to Dst, the function's unit run
/// on it and its tile packed.
fn compute_fields(
    out: &mut Vec<String>,
    plan: &Plan,
    fields: &Fields,
    unit: Option<&Unit>,
    rows: &str,
    lines: &Lines,
) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let (reads, _, p) = compute_count(plan);
    let c = fields.width;
    // the product's k, a number (`stream_field_product`)
    let inner = plan.reads[0].block;
    // the widest sub-column that divides the field and fits Dst's half
    let step = (1..=DST_HALF)
        .rev()
        .find(|w| c.is_multiple_of(*w))
        .unwrap_or(1);
    let buffer = |k: usize| format!("kField{}", camel(&fields.sums[k].field.ident));
    let sums = |k: usize| format!("km_sums_{}", fields.sums[k].field.ident);
    let room = |k: usize| format!("km_room_sums_{}", fields.sums[k].field.ident);
    case_head(out, plan, "compute", &reads);
    indented(out, 4, "bool km_ok = true;");
    for (k, sum) in fields.sums.iter().enumerate() {
        let src = &sum.field.source;
        indented(
            out,
            4,
            &format!(
                "uint32_t {} = 0;  // {src}'s\nbool {} = true;  // pack holds {src}'s pages",
                sums(k),
                room(k)
            ),
        );
    }
    comment(
        out,
        4,
        lines.comment(
            Anchor::Inst(body, plan.reads[0].inst),
            head_last(lines, plan),
        ),
    );
    let init = |w: usize| {
        format!(
            "KM_UNPACK_FORMATS(kRead{w}.mover_a, kRead0.mover_a);\n\
             matmul_block_init(kRead0.mover_a, kRead{w}.mover_a, 0, {step}, 1, {inner});"
        )
    };
    let first = init(fields.sums[0].operands.1);
    indented(
        out,
        4,
        &format!(
            "const uint32_t km_blocks0 = ({p} + kRead0Block - 1) / kRead0Block;\n\
             {first}\n\
             KM_PACK_FORMAT({});\n\
             for (uint32_t km_block0 = 0; km_ok && km_block0 < km_blocks0; km_block0++) {{",
            buffer(0)
        ),
    );
    for i in 0..plan.reads.len() {
        indented(
            out,
            5,
            &format!(
                "const uint32_t km_half{i} = km::block_half(kRead{i}, kRead{i}Pattern, km_block0);\n\
                 km_ok = compute.wait_front(km_half{i}, kRead{i}Block);\n\
                 if (!km_ok) {{\n    break;\n}}"
            ),
        );
    }
    indented(
        out,
        5,
        &format!(
            "const uint32_t km_valid0 = km_block0 % km_blocks0 + 1 == km_blocks0 ? {p} - (km_blocks0 - 1) * kRead0Block : kRead0Block;"
        ),
    );
    let indent = |text: String| {
        text.lines()
            .map(|l| format!("    {l}\n"))
            .collect::<String>()
    };
    for (k, sum) in fields.sums.iter().enumerate() {
        comment(
            out,
            5,
            lines.comment(Anchor::Inst(body, sum.call.inst), None),
        );
        let add = line_of(lines, plan, sum.call.inst);
        let (column, kk) = (format!("km_sub_column{add}"), format!("km_k{add}"));
        let w = sum.operands.1;
        let own = init(w);
        // a product other than the first sets the matrix unit for its own and back
        let (own, restore) = if own == first {
            (String::new(), String::new())
        } else {
            (indent(own), indent(first.clone()))
        };
        let packs: Vec<String> = (0..step)
            .map(|j| {
                format!(
                    "            pack_tile<true>({j}, {}, {column} + {j});",
                    buffer(k)
                )
            })
            .collect();
        let (b, n, r) = (buffer(k), sums(k), room(k));
        indented(
            out,
            5,
            &format!(
                "if (km_ok && {n} == 0) {{  // a new sum: the field's pages held\n    \
                 {r} = {r} && compute.reserve_back({b}, {b}Pages);\n\
                 }}\n\
                 if (km_ok) {{\n\
                 {own}    \
                 if ({r}) {{\n        \
                 pack_reconfig_l1_acc({n} > 0 ? 1 : 0);\n    \
                 }}\n    \
                 for (uint32_t {column} = 0; {column} < {c}; {column} += {step}) {{\n        \
                 tile_regs_acquire();\n        \
                 for (uint32_t {kk} = 0; {kk} < km_valid0; {kk}++) {{\n            \
                 matmul_block(km_half0, km_half{w}, {kk}, {kk} * {c} + {column}, 0, 0, {step}, 1, {inner});\n        \
                 }}\n        \
                 tile_regs_commit();\n        \
                 tile_regs_wait();\n        \
                 if ({r}) {{\n{}\n        }}\n        \
                 tile_regs_release();\n    \
                 }}\n    \
                 if ({r}) {{\n        \
                 pack_reconfig_l1_acc(0);\n    \
                 }}\n\
                 {restore}    \
                 {n}++;\n\
                 }}",
                packs.join("\n")
            ),
        );
    }
    for i in 0..plan.reads.len() {
        indented(out, 5, &format!("cb_pop_front(km_half{i}, kRead{i}Block);"));
    }
    indented(out, 4, "}");
    let store = fields.store.inst;
    comment(out, 4, lines.comment(Anchor::Inst(body, store), None));
    let line = line_of(lines, plan, store);
    let (tile, out_room) = (format!("km_tile{line}"), format!("km_room{line}"));
    let held = |k: usize| {
        let (b, n, r) = (buffer(k), sums(k), room(k));
        format!(
            "if (km_ok && {n} > 0) {{\n    \
             if ({r}) {{  // pack's: its pages held\n        \
             cb_push_back({b}, {b}Pages);\n    \
             }}\n    \
             km_ok = compute.wait_front({b}, {b}Pages);\n\
             }}"
        )
    };
    match (&fields.stored, unit) {
        (Stored::Gate(gate), Some(unit)) => {
            let text = lines.comment(Anchor::Inst(body, store), None);
            let what = text
                .as_deref()
                .and_then(|t| t.split_once(": ").map(|(_, rest)| rest.to_string()))
                .unwrap_or_default();
            let mut text = vec![format!("// {what}: {} of the fields' sums", gate.name)];
            text.extend(gate.vars.iter().map(|k| held(*k)));
            let made: Vec<String> = gate
                .vars
                .iter()
                .map(|k| format!("{} > 0", sums(*k)))
                .collect();
            let firsts = buffer(gate.vars[0]);
            let copies: Vec<String> = gate
                .vars
                .iter()
                .enumerate()
                .map(|(dst, k)| format!("        copy_tile({}, {tile}, {dst});", buffer(*k)))
                .collect();
            let pops: Vec<String> = gate
                .vars
                .iter()
                .map(|k| format!("    cb_pop_front({b}, {b}Pages);", b = buffer(*k)))
                .collect();
            text.push(format!(
                "if (km_ok && {}) {{\n    \
                 KM_UNPACK_FORMAT_A({firsts});\n    \
                 copy_init({firsts});\n    \
                 MATH((km_tile::{u}_init()));  // {name}'s setup\n    \
                 bool {out_room} = true;  // pack's alone: it guards the packs only\n    \
                 for (uint32_t {tile} = 0; km_ok && {tile} < {c}; {tile}++) {{\n        \
                 tile_regs_acquire();\n{}\n        \
                 MATH((km_tile::{u}(0)));\n        \
                 tile_regs_commit();\n        \
                 tile_regs_wait();\n        \
                 {out_room} = {out_room} && compute.reserve_back({rows}, 1);\n        \
                 if ({out_room}) {{\n            \
                 KM_PACK_FORMAT({rows});\n            \
                 pack_tile(0, {rows});\n            \
                 cb_push_back({rows}, 1);\n        \
                 }}\n        \
                 tile_regs_release();\n    \
                 }}\n{}\n\
                 }}",
                made.join(" && "),
                copies.join("\n"),
                pops.join("\n"),
                u = unit.name,
                name = gate.name,
            ));
            text.extend(gate.vars.iter().map(|k| format!("{} = 0;", sums(*k))));
            indented(out, 4, &text.join("\n"));
        }
        (Stored::Var(k), _) => {
            let k = *k;
            let (b, n) = (buffer(k), sums(k));
            let src = &fields.sums[k].field.source;
            let copies: Vec<String> = (0..step)
                .map(|j| format!("        copy_tile({b}, {tile} + {j}, {j});"))
                .collect();
            let row_packs: Vec<String> = (0..step)
                .map(|j| format!("            pack_tile({j}, {rows});"))
                .collect();
            let format = match store_format(plan, &fields.store) {
                Format::F32 => "float32",
                Format::Bf16 => "bfloat16",
                _ => "its format",
            };
            indented(
                out,
                4,
                &format!(
                    "// {src}'s sums, packed again in {format}\n{}\n\
                     if (km_ok && {n} > 0) {{\n    \
                     KM_UNPACK_FORMAT_A({b});\n    \
                     copy_init({b});\n    \
                     bool {out_room} = true;  // pack's alone: it guards the packs only\n    \
                     for (uint32_t {tile} = 0; km_ok && {tile} < {c}; {tile} += {step}) {{\n        \
                     tile_regs_acquire();\n{}\n        \
                     tile_regs_commit();\n        \
                     tile_regs_wait();\n        \
                     {out_room} = {out_room} && compute.reserve_back({rows}, {step});\n        \
                     if ({out_room}) {{\n            \
                     KM_PACK_FORMAT({rows});\n{}\n            \
                     cb_push_back({rows}, {step});\n        \
                     }}\n        \
                     tile_regs_release();\n    \
                     }}\n    \
                     cb_pop_front({b}, {b}Pages);\n\
                     }}\n\
                     {n} = 0;",
                    held(k),
                    copies.join("\n"),
                    row_packs.join("\n")
                ),
            );
        }
        (Stored::Gate(_), None) => unreachable!("a unit for each function of block vars (planned)"),
    }
}

/// A sum's compute case (`dst_tiles`, `reduce.tile_reduce`): the total in Dst tile 0
/// from zeros, each tile copied to Dst tile 1 and added; the arrivals added alike in
/// the reduce, the result packed into the out buffer.
fn compute_sum(out: &mut Vec<String>, plan: &Plan, sum: &Sum, lines: &Lines) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let (mut reads, part, p) = compute_count(plan);
    (reads.rank, reads.group) = (true, true);
    case_head(out, plan, "compute", &reads);
    indented(out, 4, "bool km_ok = true;");
    if let Some(part) = part {
        comment(out, 4, lines.comment(Anchor::Inst(body, part.inst), None));
        indented(out, 4, &part_line(plan, part));
    }
    comment(out, 4, lines.comment(Anchor::Inst(body, sum.zeros), None));
    indented(
        out,
        4,
        "tile_regs_acquire();\nfill_tile_init();\nfill_tile(0, 0.0f);",
    );
    comment(
        out,
        4,
        lines.comment(
            Anchor::Inst(body, plan.reads[0].inst),
            head_last(lines, plan),
        ),
    );
    indented(
        out,
        4,
        &format!(
            "const uint32_t km_blocks0 = ({p} + kRead0Block - 1) / kRead0Block;\n\
             for (uint32_t km_block0 = 0; km_ok && km_block0 < km_blocks0; km_block0++) {{"
        ),
    );
    indented(
        out,
        5,
        &format!(
            "const uint32_t km_half0 = km::block_half(kRead0, kRead0Pattern, km_block0);\n\
             km_ok = compute.wait_front(km_half0, kRead0Block);\n\
             if (!km_ok) {{\n    break;\n}}\n\
             const uint32_t km_valid0 = km_block0 % km_blocks0 + 1 == km_blocks0 ? {p} - (km_blocks0 - 1) * kRead0Block : kRead0Block;\n\
             for (uint32_t km_tile0 = 0; km_tile0 < km_valid0; km_tile0++) {{"
        ),
    );
    comment(out, 6, lines.comment(Anchor::Inst(body, sum.add), None));
    indented(
        out,
        6,
        "KM_UNPACK_FORMAT_A(km_half0);\n\
         copy_init(km_half0);\n\
         copy_tile(km_half0, km_tile0, 1);\n\
         add_binary_tile_init();\n\
         add_binary_tile(0, 1, 0);",
    );
    indented(out, 5, "}\ncb_pop_front(km_half0, kRead0Block);");
    indented(out, 4, "}");
    comment(out, 4, lines.comment(Anchor::Inst(body, sum.reduce), None));
    indented(
        out,
        4,
        "km_ok = km_ok && km::reduce(compute, kArrival0, km_group, km::kReduceTree, [&](uint32_t km_slot) {\n    \
         KM_UNPACK_FORMAT_A(kArrival0.cb);\n    \
         copy_init(kArrival0.cb);\n    \
         copy_tile(kArrival0.cb, km::slot_page(kArrival0, km_slot), 1);\n    \
         add_binary_tile_init();\n    \
         add_binary_tile(0, 1, 0);\n\
         });\n\
         tile_regs_commit();\n\
         tile_regs_wait();\n\
         if (km_ok && compute.reserve_back(kOut0, 1)) {\n    \
         KM_PACK_FORMAT(kOut0);\n    \
         pack_tile(0, kOut0);\n    \
         cb_push_back(kOut0, 1);\n\
         }\n\
         tile_regs_release();",
    );
}

/// A mapped case's compute: the loop over the read streams' blocks (zipped: each
/// waited for and freed in turn), its tiles a batch an acquire, the call on each and
/// the packs. A call takes the Dst tiles its unit states (`Unit::dst_tiles`): a batch
/// is the Dst half's tiles over that (`functions.py`, `DST_HALF // count`).
fn compute_mapped(
    out: &mut Vec<String>,
    plan: &Plan,
    called: Option<(&plan::Call, &Unit)>,
    written: &Moved,
    lines: &Lines,
) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let (mut reads, part, count) = compute_count(plan);
    if let Some((call, _)) = called {
        reads.words.extend(&call.scalars);
    }
    reads
        .words
        .extend(plan.reads.iter().chain([written]).filter_map(|m| m.pattern));
    case_head(out, plan, "compute", &reads);
    indented(out, 4, "bool km_ok = true;");
    if let Some(part) = part {
        comment(out, 4, lines.comment(Anchor::Inst(body, part.inst), None));
        indented(out, 4, &part_line(plan, part));
    }
    comment(
        out,
        4,
        lines.comment(
            Anchor::Inst(body, plan.reads[0].inst),
            head_last(lines, plan),
        ),
    );
    pattern_checks(out, plan);
    let p = &count;
    let out_pattern = pattern_text(plan, written, "kWrite0");
    let Some((call, unit)) = called else {
        compute_copied(out, plan, p, &out_pattern, lines);
        return;
    };
    let f = &call.name;
    let dst = unit.dst_tiles() as u64;
    let slots = DST_HALF / dst;
    let words: String = call
        .scalars
        .iter()
        .map(|w| format!(", __builtin_bit_cast(uint32_t, {})", plan.words[*w].name))
        .collect();
    indented(
        out,
        4,
        &format!(
            "const uint32_t km_blocks0 = ({p} + kRead0Block - 1) / kRead0Block;\n\
             KM_UNPACK_FORMAT_A(kRead0.mover_a);\n\
             copy_init(kRead0.mover_a);\n\
             MATH((km_tile::{f}_init()));  // {f}'s setup\n\
             for (uint32_t km_block0 = 0; km_ok && km_block0 < km_blocks0; km_block0++) {{"
        ),
    );
    for (i, read) in plan.reads.iter().enumerate() {
        let pattern = pattern_text(plan, read, &format!("kRead{i}"));
        indented(
            out,
            5,
            &format!(
                "const uint32_t km_half{i} = km::block_half(kRead{i}, {pattern}, km_block0);\n\
                 km_ok = compute.wait_front(km_half{i}, kRead{i}Block);\n\
                 if (!km_ok) {{\n    break;\n}}"
            ),
        );
    }
    indented(
        out,
        5,
        &format!(
            "const uint32_t km_valid0 = km_block0 % km_blocks0 + 1 == km_blocks0 ? {p} - (km_blocks0 - 1) * kRead0Block : kRead0Block;\n\
             const uint32_t km_out0 = km::block_half(kWrite0, {out_pattern}, km_block0);\n\
             const bool km_room0 = compute.reserve_back(km_out0, kWrite0Block);\n\
             for (uint32_t km_tile0 = 0; km_tile0 < km_valid0; km_tile0 += {slots}) {{"
        ),
    );
    comment(out, 6, lines.comment(Anchor::Inst(body, call.inst), None));
    let copies: String = (0..plan.reads.len())
        .map(|i| {
            format!("{INDENT}copy_tile(km_half{i}, km_tile0 + km_slot0, {dst} * km_slot0 + {i});\n")
        })
        .collect();
    indented(
        out,
        6,
        &format!(
            "const uint32_t km_batch0 = km_valid0 - km_tile0 < {slots} ? km_valid0 - km_tile0 : {slots};\n\
             tile_regs_acquire();\n\
             for (uint32_t km_slot0 = 0; km_slot0 < km_batch0; km_slot0++) {{\n\
             {copies}\
             {INDENT}MATH((km_tile::{f}({dst} * km_slot0{words})));\n\
             }}\n\
             tile_regs_commit();\n\
             tile_regs_wait();\n\
             if (km_room0) {{\n\
             {INDENT}KM_PACK_FORMAT(km_out0);\n\
             {INDENT}for (uint32_t km_slot0 = 0; km_slot0 < km_batch0; km_slot0++) {{\n\
             {INDENT}{INDENT}pack_tile<true>({dst} * km_slot0, km_out0, km_tile0 + km_slot0);\n\
             {INDENT}}}\n\
             }}\n\
             tile_regs_release();"
        ),
    );
    indented(out, 5, "}");
    indented(
        out,
        5,
        "if (km_room0) {\n    cb_push_back(km_out0, kWrite0Block);\n}",
    );
    for i in 0..plan.reads.len() {
        indented(out, 5, &format!("cb_pop_front(km_half{i}, kRead{i}Block);"));
    }
    indented(out, 4, "}");
}

/// A pattern the command gives is checked before any block (`page_streams`): an
/// invalid word ends the case, each read stream's word once, in order.
fn pattern_checks(out: &mut Vec<String>, plan: &Plan) {
    let mut checked = Vec::new();
    for w in plan.reads.iter().filter_map(|m| m.pattern) {
        if !checked.contains(&w) {
            checked.push(w);
            let name = &plan.words[w].name;
            indented(out, 4, &format!("if (!{name}.valid()) {{\n    return;\n}}"));
        }
    }
}

/// The blocks of the read stream: each waited for (its half by the stream's pattern)
/// and the case left when the run ends.
fn wait_block(out: &mut Vec<String>, plan: &Plan) {
    let read = &plan.reads[0];
    let pattern = pattern_text(plan, read, "kRead0");
    indented(
        out,
        5,
        &format!(
            "const uint32_t km_half0 = km::block_half(kRead0, {pattern}, km_block0);\n\
             km_ok = compute.wait_front(km_half0, kRead0Block);\n\
             if (!km_ok) {{\n    break;\n}}"
        ),
    );
}

/// A copy's compute (`out.push(tile)`, today's copy of a block): each tile of the
/// block copied to Dst tile 0 and packed into the out half, one an acquire.
fn compute_copied(out: &mut Vec<String>, plan: &Plan, p: &str, out_pattern: &str, lines: &Lines) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    indented(
        out,
        4,
        &format!(
            "const uint32_t km_blocks0 = ({p} + kRead0Block - 1) / kRead0Block;\n\
             for (uint32_t km_block0 = 0; km_ok && km_block0 < km_blocks0; km_block0++) {{"
        ),
    );
    wait_block(out, plan);
    indented(
        out,
        5,
        &format!(
            "const uint32_t km_valid0 = km_block0 % km_blocks0 + 1 == km_blocks0 ? {p} - (km_blocks0 - 1) * kRead0Block : kRead0Block;\n\
             const uint32_t km_out0 = km::block_half(kWrite0, {out_pattern}, km_block0);\n\
             const bool km_room0 = compute.reserve_back(km_out0, kWrite0Block);\n\
             KM_UNPACK_FORMAT_A(km_half0);\n\
             copy_init(km_half0);\n\
             for (uint32_t km_tile0 = 0; km_tile0 < km_valid0; km_tile0++) {{"
        ),
    );
    comment(
        out,
        6,
        lines.comment(Anchor::Inst(body, plan.looped()), None),
    );
    indented(
        out,
        6,
        "tile_regs_acquire();\n\
         copy_tile(km_half0, km_tile0, 0);\n\
         tile_regs_commit();\n\
         tile_regs_wait();\n\
         if (km_room0) {\n    \
         KM_PACK_FORMAT(km_out0);\n    \
         pack_tile<true>(0, km_out0, km_tile0);\n\
         }\n\
         tile_regs_release();",
    );
    indented(out, 5, "}");
    indented(
        out,
        5,
        "if (km_room0) {\n    cb_push_back(km_out0, kWrite0Block);\n}\n\
         cb_pop_front(km_half0, kRead0Block);",
    );
    indented(out, 4, "}");
}

/// A dropped stream's compute (`for _ in stream(...) { }`, maybe repeated): each block
/// waited for and freed, its count times the repeat.
fn compute_dropped(out: &mut Vec<String>, plan: &Plan, lines: &Lines) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let (mut reads, _, count) = compute_count(plan);
    reads
        .words
        .extend(plan.reads.iter().filter_map(|m| m.pattern));
    let repeat = plan.repeat.as_ref();
    let more = Uses::of(plan, repeat);
    reads.words.extend(more.words);
    reads.locals.extend(more.locals);
    case_head(out, plan, "compute", &reads);
    indented(out, 4, "bool km_ok = true;");
    comment(
        out,
        4,
        lines.comment(
            Anchor::Inst(body, plan.reads[0].inst),
            head_last(lines, plan),
        ),
    );
    pattern_checks(out, plan);
    let total = match repeat {
        Some(r) => format!("km_blocks0 * {}", int_text(plan, r)),
        None => "km_blocks0".to_string(),
    };
    indented(
        out,
        4,
        &format!(
            "const uint32_t km_blocks0 = ({count} + kRead0Block - 1) / kRead0Block;\n\
             for (uint32_t km_block0 = 0; km_ok && km_block0 < {total}; km_block0++) {{"
        ),
    );
    wait_block(out, plan);
    indented(out, 5, "cb_pop_front(km_half0, kRead0Block);");
    indented(out, 4, "}");
}

/// The device framework's headers the generated text is written against (under
/// `src/komorebi_native/include/`): a change to one changes the backend's version, so
/// its outputs are made again (tt-metal's own cache is not trusted to see them).
pub const FRAMEWORK: &[(&str, &[u8])] = &[
    (
        "komorebi/abi/km_abi.h",
        include_bytes!("../../../komorebi_native/include/komorebi/abi/km_abi.h"),
    ),
    (
        "komorebi/abi/km_chain.h",
        include_bytes!("../../../komorebi_native/include/komorebi/abi/km_chain.h"),
    ),
    (
        "komorebi/abi/km_copy.h",
        include_bytes!("../../../komorebi_native/include/komorebi/abi/km_copy.h"),
    ),
    (
        "komorebi/abi/km_graph.h",
        include_bytes!("../../../komorebi_native/include/komorebi/abi/km_graph.h"),
    ),
    (
        "komorebi/abi/km_stream.h",
        include_bytes!("../../../komorebi_native/include/komorebi/abi/km_stream.h"),
    ),
    (
        "komorebi/device/core/km_command.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/core/km_command.hpp"),
    ),
    (
        "komorebi/device/core/km_completion.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/core/km_completion.hpp"),
    ),
    (
        "komorebi/device/core/km_core.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/core/km_core.hpp"),
    ),
    (
        "komorebi/device/core/km_group.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/core/km_group.hpp"),
    ),
    (
        "komorebi/device/core/km_island.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/core/km_island.hpp"),
    ),
    (
        "komorebi/device/core/km_program.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/core/km_program.hpp"),
    ),
    (
        "komorebi/device/core/km_tf32.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/core/km_tf32.hpp"),
    ),
    (
        "komorebi/device/core/km_trace.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/core/km_trace.hpp"),
    ),
    (
        "komorebi/device/core/km_x280.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/core/km_x280.hpp"),
    ),
    (
        "komorebi/device/km_dst.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/km_dst.hpp"),
    ),
    (
        "komorebi/device/km_worker.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/km_worker.hpp"),
    ),
    (
        "komorebi/device/links/km_far.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/links/km_far.hpp"),
    ),
    (
        "komorebi/device/links/km_far_serve.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/links/km_far_serve.hpp"),
    ),
    (
        "komorebi/device/links/km_link.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/links/km_link.hpp"),
    ),
    (
        "komorebi/device/links/km_links_a_core.hpp",
        include_bytes!(
            "../../../komorebi_native/include/komorebi/device/links/km_links_a_core.hpp"
        ),
    ),
    (
        "komorebi/device/links/km_pages.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/links/km_pages.hpp"),
    ),
    (
        "komorebi/device/links/km_transport.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/links/km_transport.hpp"),
    ),
    (
        "komorebi/device/messages/km_deliver.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/messages/km_deliver.hpp"),
    ),
    (
        "komorebi/device/messages/km_messages.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/messages/km_messages.hpp"),
    ),
    (
        "komorebi/device/movement/km_broadcast.hpp",
        include_bytes!(
            "../../../komorebi_native/include/komorebi/device/movement/km_broadcast.hpp"
        ),
    ),
    (
        "komorebi/device/movement/km_chain.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/movement/km_chain.hpp"),
    ),
    (
        "komorebi/device/movement/km_movement.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/movement/km_movement.hpp"),
    ),
    (
        "komorebi/device/movement/km_port.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/movement/km_port.hpp"),
    ),
    (
        "komorebi/device/movement/km_rect.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/movement/km_rect.hpp"),
    ),
    (
        "komorebi/device/movement/km_tree.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/movement/km_tree.hpp"),
    ),
    (
        "komorebi/device/overlays/km_overlay_core.hpp",
        include_bytes!(
            "../../../komorebi_native/include/komorebi/device/overlays/km_overlay_core.hpp"
        ),
    ),
    (
        "komorebi/device/overlays/km_overlay_entry.hpp",
        include_bytes!(
            "../../../komorebi_native/include/komorebi/device/overlays/km_overlay_entry.hpp"
        ),
    ),
    (
        "komorebi/device/overlays/km_overlay.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/overlays/km_overlay.hpp"),
    ),
    (
        "komorebi/device/overlays/km_overlay_part.hpp",
        include_bytes!(
            "../../../komorebi_native/include/komorebi/device/overlays/km_overlay_part.hpp"
        ),
    ),
    (
        "komorebi/device/overlays/km_overlay_records.hpp",
        include_bytes!(
            "../../../komorebi_native/include/komorebi/device/overlays/km_overlay_records.hpp"
        ),
    ),
    (
        "komorebi/device/tensors/km_ask.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/tensors/km_ask.hpp"),
    ),
    (
        "komorebi/device/tensors/km_extent.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/tensors/km_extent.hpp"),
    ),
    (
        "komorebi/device/tensors/km_placement.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/tensors/km_placement.hpp"),
    ),
    (
        "komorebi/device/tensors/km_region.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/tensors/km_region.hpp"),
    ),
    (
        "komorebi/device/tensors/km_table_cache.hpp",
        include_bytes!(
            "../../../komorebi_native/include/komorebi/device/tensors/km_table_cache.hpp"
        ),
    ),
    (
        "komorebi/device/tensors/km_table.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/tensors/km_table.hpp"),
    ),
    (
        "komorebi/device/tensors/km_transfer.hpp",
        include_bytes!("../../../komorebi_native/include/komorebi/device/tensors/km_transfer.hpp"),
    ),
];

/// The Tensix backend: the C++ today's class DSL writes, built by tt-metal at load.
pub struct Tensix;

impl Tensix {
    /// The hash of what the output depends on besides the module and the backend's
    /// code: the framework's headers.
    fn tables_hash() -> [u8; 32] {
        let mut bytes = Vec::new();
        for (name, text) in FRAMEWORK.iter().copied() {
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(text);
            bytes.push(0);
        }
        sha256(&bytes)
    }
}

impl Backend for Tensix {
    fn name(&self) -> &'static str {
        "tensix"
    }

    fn version(&self) -> String {
        format!(
            "{}+{}",
            env!("CARGO_PKG_VERSION"),
            hex(&Tensix::tables_hash())
        )
    }

    /// Today's text names the script, each statement's line and the values' names:
    /// the output depends on the positions and names too, so the key is the full hash.
    fn reads(&self) -> Reads {
        Reads::Everything
    }

    fn statement(&self, _target: &Target) -> Statement {
        Statement {
            backend: "tensix",
            abilities: chip::ABILITIES.to_vec(),
            implementations: Vec::new(),
            limits: chip::LIMITS.to_vec(),
            formats: chip::FORMATS.to_vec(),
            formats_document: "docs/ir/backends.md",
            tile_shapes: vec![chip::TILE],
            tensor_words: chip::TENSOR_WORDS,
            hints: vec!["tensix.pattern"],
            progress: chip::PROGRESS.to_vec(),
        }
    }

    /// The workers and islands are the host's (`session.ranks`: the grid less the
    /// dispatcher); in flight, the load's `depth` up to the ring's slots; live
    /// messages, the load's `max_live`; local state, a worker's memory (the chip's L1,
    /// today's `l1=`) less what the build uses of it (`Built`'s `l1_used`, when a
    /// build reports it). The chip is checked at load (`load::chip`).
    fn statement_at(&self, _target: &Target, machine: &Machine, built: &Built) -> LoadStatement {
        let option = |name: &str| {
            machine
                .options
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| *v)
        };
        let used = built
            .facts
            .iter()
            .find(|(n, _)| n == "l1_used")
            .map_or(0, |(_, v)| *v);
        LoadStatement {
            workers: Some(machine.workers),
            islands: Some(machine.islands.count() as u64),
            local_state_bytes: Some(machine.worker_memory.saturating_sub(used)),
            in_flight: Some(option("depth").unwrap_or(chip::DEPTH).min(chip::RING_SLOTS)),
            live_messages: Some(option("max_live").unwrap_or(chip::MAX_LIVE)),
        }
    }

    fn tile_bytes(&self, format: Format, shape: TileShape) -> u64 {
        chip::tile_bytes(format, shape)
    }

    fn check(&self, program: &Module, target: &Target) -> Vec<Refusal> {
        let mut out = check_statement(program, &self.statement(target));
        if let Some(refusal) = chip::compile_target(target) {
            out.push(refusal);
        }
        if let Err(refusals) = planned(program) {
            out.extend(refusals);
        }
        out
    }

    fn compile(&self, program: &Module, target: &Target) -> Result<Output, Vec<Refusal>> {
        let refusals = self.check(program, target);
        if !refusals.is_empty() {
            return Err(refusals);
        }
        let (plans, _) = planned(program)?;
        let files = sources(program)?
            .into_iter()
            .map(|(name, text)| (name, text.into_bytes()))
            .collect();
        Ok(Output {
            key: CacheKey::for_backend(program, self, target),
            files,
            entries: entries(program, &plans)?,
            build: Build::Kernels {
                kernels: kernels(&plans),
                buffers: buffers(&plans, TREE_ROUNDS, chained(program)),
                at_load: at_load(program)?,
            },
        })
    }
}

/// Each handler's entry: its number in the command (from 1, `compiler.handler_variants`),
/// its words (a tensor its address, a scalar its value's bits) and the bounds a host
/// checks before sending: each stream's tiles `first .. first + n` in its tensor.
fn entries(module: &Module, plans: &[Plan]) -> Result<Vec<EntryPoint>, Vec<Refusal>> {
    let program = &module.programs[0];
    plans
        .iter()
        .enumerate()
        .map(|(i, plan)| {
            let handler = &program.handlers[plan.handler.0 as usize];
            let words = plan
                .words
                .iter()
                .enumerate()
                .map(|(p, w)| match w.kind {
                    WordKind::Tensor => ArgWord::TensorAddress(p),
                    WordKind::U32 | WordKind::F32 | WordKind::Pattern => ArgWord::Word(p),
                })
                .collect();
            let bounds = bounds(plan).map_err(|r| vec![r])?;
            Ok(EntryPoint {
                name: format!("{}.{}", program.name, plan.name),
                entry: Entry::Handler(plan.program, plan.handler),
                number: i as u32 + 1,
                symbol: format!("kHandler{}", camel(&plan.name)),
                params: handler.params.iter().filter_map(ArgKind::of).collect(),
                words,
                bounds,
                patterns: (plan.words.iter().enumerate())
                    .filter(|(_, w)| w.kind == WordKind::Pattern)
                    .map(|(p, _)| p)
                    .collect(),
            })
        })
        .collect()
}

/// Every range of a handler's tensors as a bound a host checks: a partition's whole
/// range, or a stream's tiles (each rank's, when it reads the rank). A range that
/// cannot be stated is refused by name: no range of a tensor goes unchecked.
fn bounds(plan: &Plan) -> Result<Vec<Bound>, Refusal> {
    let mut bounds = Vec::new();
    for moved in plan.reads.iter().chain(plan.written()) {
        let at = Anchor::Inst(BodyRef::Handler(plan.program, plan.handler), moved.inst);
        let side = |(lo, hi): &(Int, Int)| -> Result<(Expr, Expr), Refusal> {
            let lo = host_expr(plan, lo, at)?;
            let many = Expr::Sub(Box::new(host_expr(plan, hi, at)?), Box::new(lo.clone()));
            Ok((lo, many))
        };
        let bound = match &moved.range {
            Range::Part(p) => Bound {
                tensor: moved.tensor,
                first: host_expr(plan, &plan.parts[*p].first, at)?,
                count: host_expr(plan, &plan.parts[*p].n, at)?,
                cols: None,
            },
            Range::Tiles { first, n } => Bound {
                tensor: moved.tensor,
                first: host_expr(plan, first, at)?,
                count: host_expr(plan, n, at)?,
                cols: None,
            },
            Range::Rect(rect) => {
                let (first, count) = side(&rect.rows)?;
                Bound {
                    tensor: moved.tensor,
                    first,
                    count,
                    cols: Some(side(&rect.cols)?),
                }
            }
        };
        if !bounds.contains(&bound) {
            bounds.push(bound);
        }
    }
    bounds.sort_by_key(|b| b.tensor);
    Ok(bounds)
}

/// An int as a host computes it from the command's words and group: words, numbers,
/// the rank and size, `+ - * / %` and named values of these.
fn host_expr(plan: &Plan, int: &Int, at: Anchor) -> Result<Expr, Refusal> {
    let both = |a: &Int, b: &Int| -> Result<_, Refusal> {
        Ok((
            Box::new(host_expr(plan, a, at)?),
            Box::new(host_expr(plan, b, at)?),
        ))
    };
    Ok(match int {
        Int::Word(i) => Expr::Word(*i),
        Int::Const(n) => Expr::Const(*n),
        Int::Rank => Expr::Rank,
        Int::Size => Expr::Size,
        Int::Local(l) => host_expr(plan, &plan.locals[*l].value, at)?,
        Int::Bin('+', a, b) => both(a, b).map(|(a, b)| Expr::Add(a, b))?,
        Int::Bin('-', a, b) => both(a, b).map(|(a, b)| Expr::Sub(a, b))?,
        Int::Bin('*', a, b) => both(a, b).map(|(a, b)| Expr::Mul(a, b))?,
        Int::Bin('/', a, b) => both(a, b).map(|(a, b)| Expr::Div(a, b))?,
        Int::Bin('%', a, b) => both(a, b).map(|(a, b)| Expr::Rem(a, b))?,
        Int::Bin(op, ..) => {
            return Err(Refusal {
                code: NOT_BUILT,
                what: format!(
                    "the Tensix backend does not emit a range of `{op}`: the range cannot \
                     be stated as a bound yet"
                ),
                at: Some(at),
            })
        }
    })
}

/// The buffers: each pair's two halves, in the order the pairs are made, a half the
/// pair's pages of its tensor's tiles (`layout.pair`, `BufferPair.halves`); a float32
/// read stream's unpacked to Dst as float32 (`page_streams`, `unpack_fp32`).
fn buffers(plans: &[Plan], slots: u64, chained: bool) -> Vec<Buffer> {
    let lines = Lines { origin: None };
    let mut out = Vec::new();
    for pair in pairs(plans, &lines) {
        // A stream's word is a tensor's by the walk (`plan`), so its format is known.
        let format = pair.format.unwrap_or(Format::Bf16);
        let read = pair.name.starts_with("kRead");
        let unpack = (read && format == Format::F32).then_some(Unpack::ToDestFp32);
        for (index, other) in [(pair.index, pair.index + 1), (pair.index + 1, pair.index)] {
            out.push(Buffer {
                index,
                format,
                page_bytes: chip::tile_bytes(format, chip::TILE),
                pages: pair.pages,
                pair: Some(other),
                unpack,
            });
        }
    }
    // a sum's arrivals are unpacked to Dst as float32; its out buffer is packed
    for other in others(plans, &lines, slots) {
        let unpacked = other.slots.is_some() || other.field;
        out.push(Buffer {
            index: other.index,
            format: other.format,
            page_bytes: chip::tile_bytes(other.format, chip::TILE),
            pages: other.pages,
            pair: None,
            unpack: unpacked.then_some(Unpack::ToDestFp32),
        });
    }
    if chained {
        // the chain's words: one page of `uint32` (`program.chain_prepared`)
        out.push(Buffer {
            index: chip::CHAIN_BUFFER,
            format: Format::U32,
            page_bytes: chip::CHAIN_PAGE,
            pages: 1,
            pair: None,
            unpack: None,
        });
    }
    out
}

/// The chain's buffer's line in the header, as today's `Buffer.what` gives it.
const CHAIN_AREA_WHAT: &str = "the chain's words (km_chain.hpp): marks, credits, the records' ring";

/// The three kernels (`session/backend/backend.py`, `describe`): the movement source
/// on the reader and on the writer, each tensor's layout as its compile arguments in
/// order (`km::Tensor<...>`); the compute source with today's compute options
/// (`program.compute_options`: a bf16 class's Dst is fp32, its products' fidelity or
/// `HiFi4`).
fn kernels(plans: &[Plan]) -> Vec<Kernel> {
    let fidelity = plans
        .iter()
        .find_map(|p| match &p.shape {
            Shape::Product(product) => Some(product.fidelity.clone()),
            Shape::Fields(fields) => Some(fields.fidelity.clone()),
            _ => None,
        })
        .unwrap_or_else(|| tile::lower::FIDELITY.to_string());
    // each tensor of the program, in the order the handlers first take them (today's
    // `km::Tensor<...>` chain, one for all handlers)
    let layouts: Vec<CompileArg> = (0..tensors(plans).len())
        .map(|param| CompileArg::TensorLayout { param })
        .collect();
    let mover = |processor| Kernel {
        source: "movement.cpp".to_string(),
        processor,
        defines: Vec::new(),
        args: layouts.clone(),
        options: Vec::new(),
    };
    vec![
        mover(Processor::Reader),
        mover(Processor::Writer),
        Kernel {
            source: "compute.cpp".to_string(),
            processor: Processor::Compute,
            defines: Vec::new(),
            args: Vec::new(),
            options: vec![
                ("fp32_dest_acc_en".to_string(), Setting::Bool(true)),
                ("math_fidelity".to_string(), Setting::Text(fidelity)),
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_range_that_cannot_be_a_bound_is_refused_by_name() {
        // the parts of a plan a bound reads: its words (none here) and named values
        let moved = Moved {
            tensor: 0,
            block: 1,
            inst: InstId(0),
            range: Range::Part(0),
            pattern: None,
        };
        let plan = Plan {
            program: ProgramId(0),
            handler: HandlerId(0),
            name: "run".into(),
            words: Vec::new(),
            locals: Vec::new(),
            parts: Vec::new(),
            reads: vec![moved.clone()],
            shape: Shape::Mapped {
                write: moved,
                call: Some(plan::Call {
                    func: crate::types::FuncId(0),
                    name: "f".into(),
                    tiles: 1,
                    scalars: vec![0],
                    inst: InstId(0),
                }),
                push: InstId(0),
            },
            repeat: None,
        };
        let at = Anchor::Handler(ProgramId(0), HandlerId(0));
        let halved = Int::Bin('/', Box::new(Int::Word(3)), Box::new(Int::Const(2)));
        let stated = host_expr(&plan, &halved, at).expect("a quotient");
        assert_eq!(
            stated,
            Expr::Div(Box::new(Expr::Word(3)), Box::new(Expr::Const(2)))
        );
        // an operation the plan does not make is refused, not dropped
        let other = Int::Bin('&', Box::new(Int::Word(3)), Box::new(Int::Const(2)));
        let refused = host_expr(&plan, &other, at).expect_err("refused");
        assert_eq!(refused.code, NOT_BUILT);
        assert!(
            refused.what.contains("cannot be stated as a bound yet"),
            "{refused:?}"
        );
        let less = Int::Bin('-', Box::new(Int::Word(3)), Box::new(Int::Word(2)));
        let stated = host_expr(&plan, &less, at).expect("a difference");
        assert_eq!(
            stated,
            Expr::Sub(Box::new(Expr::Word(3)), Box::new(Expr::Word(2)))
        );
    }
}
