//! The Tensix backend's emitter: a module to the C++ today's class DSL writes for the
//! same program (`komorebi.dsl`), on Komorebi's device framework, text for text. It
//! reads the program, its hints and its positions (S, H and P): today's text names
//! the script, each statement's line and the names of values, from the notes' origin
//! and names. These are pure functions of the module; nothing is run or written.
//!
//! Built: one program of handlers on a group, each a loop over a read stream of a
//! partition of `t.tiles(first, n)` that pushes a tile function's result into a sink
//! of the same partition (the first lesson); the tile function a tile times a `f32`.
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
    Statement, Target, NOT_BUILT,
};
use crate::bytecode::hash::{hex, sha256};
use crate::program::body::InstId;
use crate::program::notes::{Anchor, Origin};
use crate::program::{BodyRef, Module};
use crate::types::{Entry, Format, TileShape};
use crate::types::{HandlerId, ProgramId};

use self::origin::{place, statement_comment, Place};
use self::plan::{Int, Moved, Part, Plan, Range, WordKind};
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

/// The compute kernel's includes every program has (`program_sources.COMPUTE_INCLUDES`
/// whose need is none), in order.
const COMPUTE_INCLUDES: &[&str] = &[
    "api/compute/cb_api.h",
    "api/compute/compute_kernel_api.h",
    "api/compute/compute_kernel_hw_startup.h",
    "api/compute/pack.h",
    "api/compute/reg_api.h",
    "api/compute/tile_move_copy.h",
    "komorebi/device/movement/km_movement.hpp",
    "komorebi/device/km_worker.hpp",
];

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
    let mut units: Vec<Unit> = Vec::new();
    for plan in &plans {
        match unit::unit(module, plan.call.func) {
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
    let mut files = vec![
        (header.clone(), header_text(name, &plans, &pairs, &lines)),
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
        let mut opened = [
            (plan.write.inst, "Write", &plan.write, WRITE_PATTERN, false),
            (plan.read.inst, "Read", &plan.read, READ_PATTERN, true),
        ];
        opened.sort_by_key(|o| o.0);
        for (inst, role, moved, pattern, looped) in opened {
            let block = moved.block;
            let name = format!("k{role}0");
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

/// The last head line of a handler's loop: the line before its body's first.
fn head_last(lines: &Lines, plan: &Plan) -> Option<u32> {
    let body = BodyRef::Handler(plan.program, plan.handler);
    let inner = lines.place(Anchor::Inst(body, plan.call.inst))?;
    Some(inner.first.saturating_sub(1))
}

/// The header both kernels include (`program.header`).
fn header_text(name: &str, plans: &[Plan], pairs: &[Pair], lines: &Lines) -> String {
    let mut out = vec![
        format!("// {}", lines.made(name)),
        "// the handlers, buffers and command words its kernels share.".to_string(),
        "#pragma once".to_string(),
        String::new(),
        "#include <stdint.h>".to_string(),
        String::new(),
        "#include \"komorebi/device/movement/km_movement.hpp\"".to_string(),
        String::new(),
    ];
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
        format!("#include \"{header}\""),
        String::new(),
        "void kernel_main() {".to_string(),
    ];
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
        let ints = range_ints(plan, &plan.read.range)
            .into_iter()
            .chain(range_ints(plan, &plan.write.range));
        let mut reads = Uses::of(plan, ints);
        reads.words.extend([plan.read.tensor, plan.write.tensor]);
        if !plan.parts.is_empty() {
            (reads.rank, reads.size) = (true, true);
        }
        case_head(&mut out, plan, "movement", &reads);
        let stream = |kind: &str, var: &str, pair: &str, moved: &Moved| {
            let t = &plan.words[moved.tensor].name;
            let tiles = match &moved.range {
                Range::Part(p) => {
                    let part = &plan.parts[*p];
                    let first = int_text(plan, &part.first);
                    format!("({first} + {p}.start), {p}.len", p = part.name)
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
        statements.push((
            plan.write.inst,
            stream("km::kMoveWrite", "km_write0", "kWrite0", &plan.write),
            None,
        ));
        statements.push((
            plan.read.inst,
            stream("km::kMoveRead", "km_read0", "kRead0", &plan.read),
            head_last(lines, plan),
        ));
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
        indented(
            &mut out,
            4,
            "if (!movement.move(km_read0, km_write0)) {\n    return false;\n}",
        );
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

/// The compute source (`program.compute_source`): each case's loop over the read
/// stream's blocks, its tiles `DST_HALF` an acquire, the call on each and the packs.
fn compute_text(name: &str, header: &str, plans: &[Plan], units: &[Unit], lines: &Lines) -> String {
    let mut out = vec![
        format!("// {}", lines.made(name)),
        "// the compute kernel: the tiles of its streams, in block order.".to_string(),
        "#include <stdint.h>".to_string(),
        String::new(),
    ];
    out.extend(COMPUTE_INCLUDES.iter().map(|i| format!("#include \"{i}\"")));
    out.extend(
        units
            .iter()
            .map(|u| format!("#include \"tile_{}.hpp\"", u.name)),
    );
    out.push(format!("#include \"{header}\""));
    out.push(String::new());
    out.push("void kernel_main() {".to_string());
    indented(&mut out, 1, "km::Compute compute;");
    indented(
        &mut out,
        1,
        "compute_kernel_hw_startup(kRead0.mover_a, kWrite0.mover_a);",
    );
    indented(
        &mut out,
        1,
        "compute.run([&](const km_command_t& command) {",
    );
    indented(&mut out, 2, "switch (km::handler(command)) {");
    for plan in plans {
        let body = BodyRef::Handler(plan.program, plan.handler);
        let f = &plan.call.name;
        indented(&mut out, 3, &case_line(plan, lines));
        // the read stream's count: its partition's length, or its tiles' count
        let (mut reads, part, count) = match &plan.read.range {
            Range::Part(p) => {
                let part = &plan.parts[*p];
                let mut reads = Uses::of(plan, [&part.n]);
                (reads.rank, reads.size) = (true, true);
                (reads, Some(part), format!("{}.len", part.name))
            }
            Range::Tiles { n, .. } => (Uses::of(plan, [n]), None, int_text(plan, n)),
        };
        reads.words.push(plan.call.scalar);
        case_head(&mut out, plan, "compute", &reads);
        indented(&mut out, 4, "bool km_ok = true;");
        if let Some(part) = part {
            comment(
                &mut out,
                4,
                lines.comment(Anchor::Inst(body, part.inst), None),
            );
            indented(&mut out, 4, &part_line(plan, part));
        }
        comment(
            &mut out,
            4,
            lines.comment(Anchor::Inst(body, plan.read.inst), head_last(lines, plan)),
        );
        let p = &count;
        let k = &plan.words[plan.call.scalar].name;
        indented(
            &mut out,
            4,
            &format!(
                "const uint32_t km_blocks0 = ({p} + kRead0Block - 1) / kRead0Block;\n\
                 KM_UNPACK_FORMAT_A(kRead0.mover_a);\n\
                 copy_init(kRead0.mover_a);\n\
                 MATH((km_tile::{f}_init()));  // {f}'s setup\n\
                 for (uint32_t km_block0 = 0; km_ok && km_block0 < km_blocks0; km_block0++) {{"
            ),
        );
        indented(
            &mut out,
            5,
            &format!(
                "const uint32_t km_half0 = km::block_half(kRead0, kRead0Pattern, km_block0);\n\
                 km_ok = compute.wait_front(km_half0, kRead0Block);\n\
                 if (!km_ok) {{\n    break;\n}}\n\
                 const uint32_t km_valid0 = km_block0 % km_blocks0 + 1 == km_blocks0 ? {p} - (km_blocks0 - 1) * kRead0Block : kRead0Block;\n\
                 const uint32_t km_out0 = km::block_half(kWrite0, kWrite0Pattern, km_block0);\n\
                 const bool km_room0 = compute.reserve_back(km_out0, kWrite0Block);\n\
                 for (uint32_t km_tile0 = 0; km_tile0 < km_valid0; km_tile0 += {DST_HALF}) {{"
            ),
        );
        comment(
            &mut out,
            6,
            lines.comment(Anchor::Inst(body, plan.call.inst), None),
        );
        indented(
            &mut out,
            6,
            &format!(
                "const uint32_t km_batch0 = km_valid0 - km_tile0 < {DST_HALF} ? km_valid0 - km_tile0 : {DST_HALF};\n\
                 tile_regs_acquire();\n\
                 for (uint32_t km_slot0 = 0; km_slot0 < km_batch0; km_slot0++) {{\n\
                 {INDENT}copy_tile(km_half0, km_tile0 + km_slot0, 1 * km_slot0 + 0);\n\
                 {INDENT}MATH((km_tile::{f}(1 * km_slot0, __builtin_bit_cast(uint32_t, {k}))));\n\
                 }}\n\
                 tile_regs_commit();\n\
                 tile_regs_wait();\n\
                 if (km_room0) {{\n\
                 {INDENT}KM_PACK_FORMAT(km_out0);\n\
                 {INDENT}for (uint32_t km_slot0 = 0; km_slot0 < km_batch0; km_slot0++) {{\n\
                 {INDENT}{INDENT}pack_tile<true>(1 * km_slot0, km_out0, km_tile0 + km_slot0);\n\
                 {INDENT}}}\n\
                 }}\n\
                 tile_regs_release();"
            ),
        );
        indented(&mut out, 5, "}");
        indented(
            &mut out,
            5,
            "if (km_room0) {\n    cb_push_back(km_out0, kWrite0Block);\n}\ncb_pop_front(km_half0, kRead0Block);",
        );
        indented(&mut out, 4, "}");
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
    /// The hash of what the output depends on besides the module: the tile units'
    /// templates and the framework's headers.
    fn tables_hash() -> [u8; 32] {
        let mut bytes = Vec::new();
        for (name, text) in unit::TEMPLATES
            .iter()
            .map(|(n, t)| (*n, t.as_bytes()))
            .chain(FRAMEWORK.iter().copied())
        {
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
            islands: Some(machine.islands),
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
            entries: entries(program, &plans),
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
fn entries(module: &Module, plans: &[Plan]) -> Vec<EntryPoint> {
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
            // A range a host can compute from the words: a partition's whole range, or
            // a stream's tiles; one that reads the rank is each worker's, not stated.
            let mut bounds = Vec::new();
            for moved in [&plan.read, &plan.write] {
                let (first, n) = match &moved.range {
                    Range::Part(p) => (&plan.parts[*p].first, &plan.parts[*p].n),
                    Range::Tiles { first, n } => (first, n),
                };
                let (Some(first), Some(count)) = (host_expr(plan, first), host_expr(plan, n))
                else {
                    continue;
                };
                let bound = Bound {
                    tensor: moved.tensor,
                    first,
                    count,
                };
                if !bounds.contains(&bound) {
                    bounds.push(bound);
                }
            }
            bounds.sort_by_key(|b| b.tensor);
            EntryPoint {
                name: format!("{}.{}", program.name, plan.name),
                entry: Entry::Handler(plan.program, plan.handler),
                number: i as u32 + 1,
                symbol: format!("kHandler{}", camel(&plan.name)),
                params: handler.params.iter().filter_map(ArgKind::of).collect(),
                words,
                bounds,
            }
        })
        .collect()
}

/// An int as a host computes it from the command's words, if it can: words, numbers,
/// sums and named values of these.
fn host_expr(plan: &Plan, int: &Int) -> Option<Expr> {
    match int {
        Int::Word(i) => Some(Expr::Word(*i)),
        Int::Const(n) => Some(Expr::Const(*n)),
        Int::Local(l) => host_expr(plan, &plan.locals[*l].value),
        Int::Bin('+', a, b) => Some(Expr::Add(
            Box::new(host_expr(plan, a)?),
            Box::new(host_expr(plan, b)?),
        )),
        _ => None,
    }
}

/// The buffers: each pair's two halves, in the order the pairs are made, a half the
/// pair's pages of its tensor's tiles (`layout.pair`, `BufferPair.halves`).
fn buffers(plans: &[Plan]) -> Vec<Buffer> {
    let lines = Lines { origin: None };
    let mut out = Vec::new();
    for pair in pairs(plans, &lines) {
        // A stream's word is a tensor's by the walk (`plan`), so its format is known.
        let format = pair.format.unwrap_or(Format::Bf16);
        for (index, other) in [(pair.index, pair.index + 1), (pair.index + 1, pair.index)] {
            out.push(Buffer {
                index,
                format,
                page_bytes: chip::tile_bytes(format, chip::TILE),
                pages: pair.pages,
                pair: Some(other),
            });
        }
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
