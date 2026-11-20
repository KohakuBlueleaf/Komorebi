//! The Tensix backend's emitter: a module to the C++ today's class DSL writes for the
//! same program (`komorebi.dsl`), on Komorebi's device framework, text for text. It
//! reads the program, its hints and its positions (S, H and P): today's text names
//! the script, each statement's line and the names of values, from the notes' origin
//! and names. These are pure functions of the module; nothing is run or written.
//!
//! Built: one program of handlers on a group, each a loop over a read stream of a
//! partition of `t.tiles(first, n)` that pushes a tile function's result into a sink
//! of the same partition (the first lesson); or that sums its `f32` tiles, the sum
//! reduced over the group by tree and stored by the leader, rank 0 (program 2).
//! Everything else is refused by name (K964).

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
use crate::program::{BodyRef, Module};
use crate::types::{Entry, Format, TileShape};
use crate::types::{HandlerId, ProgramId};

use self::origin::{place, statement_comment, Place};
use self::plan::{Int, Moved, Part, Plan, Range, Shape, Sum, WordKind};
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
pub(crate) const TREE_ROUNDS: u64 = 7;

/// The compute kernel's includes (`program_sources.COMPUTE_INCLUDES`), in order, each
/// with the need that brings it (none: every program's); the needs this emitter makes.
const COMPUTE_INCLUDES: &[(&str, Option<&str>)] = &[
    ("api/compute/cb_api.h", None),
    ("api/compute/compute_kernel_api.h", None),
    ("api/compute/compute_kernel_hw_startup.h", None),
    ("api/compute/eltwise_binary_sfpu.h", Some("add")),
    ("api/compute/eltwise_unary/fill.h", Some("fill")),
    ("api/compute/pack.h", None),
    ("api/compute/reg_api.h", None),
    ("api/compute/tile_move_copy.h", None),
    ("komorebi/device/movement/km_movement.hpp", None),
    ("komorebi/device/movement/km_port.hpp", Some("port")),
    ("komorebi/device/km_worker.hpp", None),
];

/// What the handlers need of the includes (`compiled.includes`): a sum's add, fill
/// and port; its store's ask.
fn needs(plans: &[Plan]) -> Vec<&'static str> {
    if plans.iter().any(|p| matches!(p.shape, Shape::Summed(_))) {
        vec!["add", "fill", "port", "ask"]
    } else {
        Vec::new()
    }
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
    pattern: &'static str,
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
    if !program.state.is_empty() || !program.runs.is_empty() || !program.chains.is_empty() {
        return Err(vec![refused(
            "a program with state, runs or chains",
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
    if plans.len() > 1 {
        refusals.push(refused(
            "a program of more than one handler",
            Anchor::Program(p),
        ));
    }
    for plan in &plans {
        if let Err(r) = bounds(plan) {
            refusals.push(r);
        }
    }
    let mut units: Vec<Unit> = Vec::new();
    for plan in &plans {
        let Shape::Mapped { call, .. } = &plan.shape else {
            continue;
        };
        match unit::unit(module, call.func) {
            Ok(u) => units.push(u),
            Err(r) => refusals.push(r),
        }
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }
    Ok((plans, units))
}

/// The generated files of `module`: each name and text, in name order; or every
/// refusal of what is not built.
pub fn sources(module: &Module) -> Result<Vec<(String, String)>, Vec<Refusal>> {
    let (plans, units) = planned(module)?;
    let program = &module.programs[0];
    let lines = Lines {
        origin: module.notes.origin.as_ref(),
    };
    let name = &program.name;
    let header = format!("{}.h", name.to_lowercase());
    let pairs = pairs(&plans, &lines);
    let others = others(&plans, &lines);
    let mut files = vec![
        (
            header.clone(),
            header_text(name, &plans, &pairs, &others, &lines),
        ),
        (
            "movement.cpp".to_string(),
            movement_text(name, &header, &plans, &lines),
        ),
        (
            "compute.cpp".to_string(),
            compute_text(name, &header, &plans, &units, &lines),
        ),
    ];
    for u in units {
        files.push((format!("tile_{}.hpp", u.name), u.words));
        files.push((format!("tile_{}.txt", u.name), u.listing));
    }
    files.sort();
    Ok(files)
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
            if out.iter().any(|p| p.name == name) {
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
    /// An arrival's slots; none for an out buffer.
    slots: Option<u64>,
    pages: u64,
    what: String,
}

/// The sums' buffers in the order they are made: each reduce's arrival, from index 8,
/// its slots a tree's rounds of one tile; then its out buffer, from 16, two writes.
fn others(plans: &[Plan], lines: &Lines) -> Vec<Other> {
    let mut out: Vec<Other> = Vec::new();
    for plan in plans {
        let Shape::Summed(sum) = &plan.shape else {
            continue;
        };
        let anchor = Anchor::Inst(BodyRef::Handler(plan.program, plan.handler), sum.reduce);
        let what = |name: &str| {
            lines
                .comment(anchor, None)
                .unwrap_or_else(|| format!("{name}: a reduce's"))
        };
        let free = |out: &[Other], first: u32| {
            (first..)
                .find(|i| out.iter().all(|o| o.index != *i))
                .unwrap_or(first)
        };
        let index = free(&out, FIRST_OTHER);
        out.push(Other {
            name: "kArrival0".to_string(),
            index,
            slots: Some(TREE_ROUNDS),
            pages: TREE_ROUNDS,
            what: what("kArrival0"),
        });
        let index = free(&out, FIRST_WRITE);
        out.push(Other {
            name: "kOut0".to_string(),
            index,
            slots: None,
            pages: WRITE_ITEMS,
            what: what("kOut0"),
        });
    }
    out
}

/// The last head line of a compound statement: the line before its body's first.
fn head_last(lines: &Lines, plan: &Plan) -> Option<u32> {
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
        out.push(format!(
            "constexpr km::Pattern {}Pattern = km::pattern(\"{}\");",
            pair.name, pair.pattern
        ));
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
    }
    out.extend(
        [
            "",
            "// Compute's formats: set again before each unpack and pack when its buffers",
            "// have more than one.",
            "#define KM_UNPACK_FORMAT_A(cb) ((void)0)",
            "#define KM_UNPACK_FORMATS(a, b) ((void)0)",
            "#define KM_PACK_FORMAT(cb) ((void)0)",
        ]
        .map(str::to_string),
    );
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
    }
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
    }
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

/// The movement source both movers build (`program_sources.movement_source`).
fn movement_text(name: &str, header: &str, plans: &[Plan], lines: &Lines) -> String {
    let mut out = vec![
        format!("// {}", lines.made(name)),
        "// the movement source of both movers (mover A: RISCV_0, mover B: RISCV_1).".to_string(),
        "#include <stdint.h>".to_string(),
        String::new(),
        "#include \"api/dataflow/dataflow_api.h\"".to_string(),
        "#include \"komorebi/device/movement/km_movement.hpp\"".to_string(),
    ];
    let needs = needs(plans);
    if needs.contains(&"port") {
        out.push("#include \"komorebi/device/movement/km_port.hpp\"".to_string());
    }
    if needs.contains(&"ask") {
        out.push("#include \"komorebi/device/tensors/km_ask.hpp\"".to_string());
    }
    out.extend([
        format!("#include \"{header}\""),
        String::new(),
        "void kernel_main() {".to_string(),
    ]);
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
    indented(
        &mut out,
        1,
        "movement.run([&](const km_command_t& command) {",
    );
    indented(&mut out, 2, "switch (km::handler(command)) {");
    for plan in plans {
        let body = BodyRef::Handler(plan.program, plan.handler);
        indented(&mut out, 3, &case_line(plan, lines));
        let written = plan.written();
        let ints = plan
            .reads
            .iter()
            .flat_map(|r| range_ints(plan, &r.range))
            .chain(range_ints(plan, &written.range));
        let mut reads = Uses::of(plan, ints);
        reads.words.extend(plan.reads.iter().map(|r| r.tensor));
        reads.words.push(written.tensor);
        if !plan.parts.is_empty() {
            (reads.rank, reads.size) = (true, true);
        }
        if let Shape::Summed(_) = plan.shape {
            // the reduce reads the group; the leader is rank 0
            (reads.rank, reads.group) = (true, true);
        }
        case_head(&mut out, plan, "movement", &reads);
        let stream = |kind: &str, var: &str, pair: &str, moved: &Moved| {
            let t = &plan.words[moved.tensor].name;
            let tiles = match &moved.range {
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
            format!(
                "const km::Stream {var}(\n{INDENT}{kind}, {}({t}), {tiles}, {pair}Block, 1, {pair}, {pair}Pattern\n);",
                camel(t),
            )
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
                stream("km::kMoveWrite", "km_write0", "kWrite0", write),
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
                ),
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
        }
        indented(&mut out, 4, "return true;");
        indented(&mut out, 3, "}");
    }
    indented(&mut out, 3, "default:");
    indented(&mut out, 4, "return true;");
    indented(&mut out, 2, "}");
    indented(&mut out, 1, "});");
    out.push("}".to_string());
    out.join("\n") + "\n"
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
fn compute_text(name: &str, header: &str, plans: &[Plan], units: &[Unit], lines: &Lines) -> String {
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
    out.extend(
        units
            .iter()
            .map(|u| format!("#include \"tile_{}.hpp\"", u.name)),
    );
    out.push(format!("#include \"{header}\""));
    out.push(String::new());
    out.push("void kernel_main() {".to_string());
    indented(&mut out, 1, "km::Compute compute;");
    // the first read and the first written buffer (a sum's out buffer)
    let written = match plans.first().map(|p| &p.shape) {
        Some(Shape::Summed(_)) => "kOut0",
        _ => "kWrite0.mover_a",
    };
    indented(
        &mut out,
        1,
        &format!("compute_kernel_hw_startup(kRead0.mover_a, {written});"),
    );
    indented(
        &mut out,
        1,
        "compute.run([&](const km_command_t& command) {",
    );
    indented(&mut out, 2, "switch (km::handler(command)) {");
    // each mapped plan's unit, in order (`planned`)
    let mut called = units.iter();
    for plan in plans {
        indented(&mut out, 3, &case_line(plan, lines));
        match &plan.shape {
            Shape::Mapped { call, .. } => {
                let unit = called.next().expect("a unit for each mapped plan");
                compute_mapped(&mut out, plan, call, unit, lines);
            }
            Shape::Summed(sum) => compute_sum(&mut out, plan, sum, lines),
        }
        indented(&mut out, 4, "break;");
        indented(&mut out, 3, "}");
    }
    indented(&mut out, 3, "default:");
    indented(&mut out, 4, "break;");
    indented(&mut out, 2, "}");
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
    call: &plan::Call,
    unit: &Unit,
    lines: &Lines,
) {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let f = &call.name;
    let (mut reads, part, count) = compute_count(plan);
    reads.words.extend(&call.scalars);
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
    let p = &count;
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
             const uint32_t km_out0 = km::block_half(kWrite0, kWrite0Pattern, km_block0);\n\
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
            hints: Vec::new(),
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
                buffers: buffers(&plans),
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
                    WordKind::U32 | WordKind::F32 => ArgWord::Word(p),
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
            })
        })
        .collect()
}

/// Every range of a handler's tensors as a bound a host checks: a partition's whole
/// range, or a stream's tiles (each rank's, when it reads the rank). A range that
/// cannot be stated is refused by name: no range of a tensor goes unchecked.
fn bounds(plan: &Plan) -> Result<Vec<Bound>, Refusal> {
    let mut bounds = Vec::new();
    for moved in plan.reads.iter().chain([plan.written()]) {
        let (first, n) = match &moved.range {
            Range::Part(p) => (&plan.parts[*p].first, &plan.parts[*p].n),
            Range::Tiles { first, n } => (first, n),
        };
        let at = Anchor::Inst(BodyRef::Handler(plan.program, plan.handler), moved.inst);
        let bound = Bound {
            tensor: moved.tensor,
            first: host_expr(plan, first, at)?,
            count: host_expr(plan, n, at)?,
        };
        if !bounds.contains(&bound) {
            bounds.push(bound);
        }
    }
    bounds.sort_by_key(|b| b.tensor);
    Ok(bounds)
}

/// An int as a host computes it from the command's words and group: words, numbers,
/// the rank and size, `+ - *` and named values of these; another is refused at `at`.
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
fn buffers(plans: &[Plan]) -> Vec<Buffer> {
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
    for other in others(plans, &lines) {
        let arrival = other.slots.is_some();
        out.push(Buffer {
            index: other.index,
            format: Format::F32,
            page_bytes: chip::tile_bytes(Format::F32, chip::TILE),
            pages: other.pages,
            pair: None,
            unpack: arrival.then_some(Unpack::ToDestFp32),
        });
    }
    out
}

/// The three kernels (`session/backend/backend.py`, `describe`): the movement source
/// on the reader and on the writer, each tensor's layout as its compile arguments in
/// order (`km::Tensor<...>`); the compute source with today's compute options
/// (`program.compute_options`: a bf16 class's Dst is fp32, `HiFi4`).
fn kernels(plans: &[Plan]) -> Vec<Kernel> {
    let mut params = Vec::new();
    for plan in plans {
        for (p, w) in plan.words.iter().enumerate() {
            if w.kind == WordKind::Tensor && !params.contains(&p) {
                params.push(p);
            }
        }
    }
    let layouts: Vec<CompileArg> = params
        .into_iter()
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
                (
                    "math_fidelity".to_string(),
                    Setting::Text("HiFi4".to_string()),
                ),
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
                call: plan::Call {
                    func: crate::types::FuncId(0),
                    name: "f".into(),
                    tiles: 1,
                    scalars: vec![0],
                    inst: InstId(0),
                },
            },
        };
        let at = Anchor::Handler(ProgramId(0), HandlerId(0));
        let halved = Int::Bin('/', Box::new(Int::Word(3)), Box::new(Int::Const(2)));
        let refused = host_expr(&plan, &halved, at).expect_err("refused");
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
