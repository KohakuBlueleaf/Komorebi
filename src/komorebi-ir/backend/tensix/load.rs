//! What the Tensix backend writes at load, from the tensors a load is given: the C
//! header an X280 thread includes to submit the program's commands
//! (`<program>_thread.h`, today's `dsl/thread/class_thread.py`, `header_text`). It
//! holds each tensor's page count and kind, which a compile does not know; so it is
//! not one of the compiled files and not in the compile's key. A load also checks the
//! machine it is on against what the compile sized (`machine`).

use crate::backend::{
    CompileArg, Kernel, Machine, Processor, Refusal, Setting, Target, NOT_A_MACHINE, NOT_BUILT,
};
use crate::program::notes::Anchor;
use crate::program::{BodyRef, Module};

use super::chip::Chip;

use super::plan::{Plan, Shape, WordKind};
use super::Lines;

/// A tensor a load is given for a tensor parameter, as the header reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadTensor {
    /// Its pages.
    pub pages: u64,
    /// Its layout as the kernels' compile arguments take it (`CompileArg::TensorLayout`,
    /// today's `tensor_args`: the accessor's words),
    /// with whatever else makes two tensors of one kind: tensors of one layout share a
    /// kind, numbered from 1 in parameter order.
    pub layout: Vec<u64>,
    /// The bytes of a page in DRAM, for a tensor a thread may fetch and store; none
    /// for one it may not (no `<program>_<tensor>_dram` function then).
    pub dram_page: Option<u64>,
}

/// The chip a load's target names: `blackhole` or `wormhole_b0`; any other is refused
/// by name. A compile names no chip (`chip::ARCH`); the chip is checked here.
pub fn chip(target: &Target) -> Result<Chip, Refusal> {
    Chip::of(&target.arch).ok_or_else(|| Refusal {
        code: NOT_BUILT,
        what: format!(
            "the Tensix backend loads on blackhole or wormhole_b0, not `{}`",
            target.arch
        ),
        at: None,
    })
}

/// C words a parameter's name may be: its argument takes a `_` after it
/// (`class_thread.C_WORDS`).
const C_WORDS: &[&str] = &[
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
    "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register",
    "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while",
];

/// A parameter's name as a C argument (`class_thread.argument`).
fn argument(name: &str) -> String {
    if C_WORDS.contains(&name) || name.starts_with("km_") {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

/// The names of `module`'s tensor parameters in the order a load takes its tensors:
/// as the handlers take them, each once.
pub fn tensor_names(module: &Module) -> Result<Vec<String>, Vec<Refusal>> {
    let (plans, _) = super::planned(module)?;
    Ok(super::tensors(&plans))
}

/// The thread header of `module` for a load given `tensors`, one for each tensor
/// parameter in the order the handlers take them; or every refusal of the module.
pub fn thread_header(module: &Module, tensors: &[LoadTensor]) -> Result<String, Vec<Refusal>> {
    let (plans, _) = super::planned(module)?;
    let lines = Lines {
        origin: module.notes.origin.as_ref(),
    };
    let name = module.programs.first().map_or("", |p| p.name.as_str());
    let names = super::tensors(&plans);
    if names.len() != tensors.len() {
        return Err(vec![Refusal {
            code: NOT_BUILT,
            what: format!(
                "a thread header for {} tensors given; the program takes {}",
                tensors.len(),
                names.len()
            ),
            at: None,
        }]);
    }
    Ok(text(name, &plans, &names, tensors, &lines))
}

/// What a load refuses of `machine` for `module`: islands that are not a graph of its
/// workers (`IslandGraph::hops`, as a run's load refuses them); a machine of more than
/// one island when a reduce's arrivals are sized for one (a tree's rounds; today's
/// make sizes them from a session's graph, which this backend does not read yet).
pub fn machine(module: &Module, machine: &Machine) -> Result<(), Vec<Refusal>> {
    let workers = usize::try_from(machine.workers).unwrap_or(usize::MAX);
    if let Err(why) = machine.islands.hops(workers) {
        return Err(vec![Refusal {
            code: NOT_A_MACHINE,
            what: why.sentence(),
            at: None,
        }]);
    }
    let (plans, _) = super::planned(module)?;
    let islands = machine.islands.count();
    if islands <= 1 {
        return Ok(());
    }
    let refusals: Vec<Refusal> = plans
        .iter()
        .filter_map(|plan| match &plan.shape {
            Shape::Summed(sum) => Some(Refusal {
                code: NOT_BUILT,
                what: format!(
                    "a reduce's arrivals are sized for one island ({} slots, a tree's \
                     rounds); a load on a machine of {islands} islands does not size them \
                     from the machine's graph yet",
                    super::TREE_ROUNDS
                ),
                at: Some(Anchor::Inst(
                    BodyRef::Handler(plan.program, plan.handler),
                    sum.reduce,
                )),
            }),
            Shape::Mapped { .. } => None,
        })
        .collect();
    if refusals.is_empty() {
        Ok(())
    } else {
        Err(refusals)
    }
}

/// A kernel as a load hands it to the chip's runtime: a compiled `Kernel` with each
/// tensor's layout filled in from the tensors given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedKernel {
    /// Its source: a path in the output's files.
    pub source: String,
    /// The processor it runs on.
    pub processor: Processor,
    /// Its defines, by name.
    pub defines: Vec<(String, String)>,
    /// Its compile-time arguments, each tensor's layout in its place.
    pub args: Vec<u64>,
    /// Its compile options, by name.
    pub options: Vec<(String, Setting)>,
}

/// The kernels of `module`'s output as a load given `tensors` (one for each tensor
/// parameter, as `thread_header` takes them) hands them on: a `TensorLayout{param}`
/// becomes that tensor's layout words (today's `tensor_args`), a word stays itself.
pub fn kernels(
    module: &Module,
    built: &[Kernel],
    tensors: &[LoadTensor],
) -> Result<Vec<LoadedKernel>, Vec<Refusal>> {
    let (plans, _) = super::planned(module)?;
    let names = super::tensors(&plans);
    let refused = |what: String| {
        vec![Refusal {
            code: NOT_BUILT,
            what,
            at: None,
        }]
    };
    if names.len() != tensors.len() {
        return Err(refused(format!(
            "kernels for {} tensors given; the program takes {}",
            tensors.len(),
            names.len()
        )));
    }
    let layout = |param: usize| {
        plans
            .first()
            .and_then(|plan| plan.words.get(param))
            .and_then(|word| names.iter().position(|n| *n == word.name))
            .and_then(|at| tensors.get(at))
            .map(|t| t.layout.clone())
    };
    let mut out = Vec::new();
    for kernel in built {
        let mut args = Vec::new();
        for arg in &kernel.args {
            match *arg {
                CompileArg::Word(w) => args.push(u64::from(w)),
                CompileArg::TensorLayout { param } => match layout(param) {
                    Some(words) => args.extend(words),
                    None => {
                        return Err(refused(format!(
                            "a tensor layout of parameter {param}, which is no tensor given"
                        )))
                    }
                },
            }
        }
        out.push(LoadedKernel {
            source: kernel.source.clone(),
            processor: kernel.processor,
            defines: kernel.defines.clone(),
            args,
            options: kernel.options.clone(),
        });
    }
    Ok(out)
}

/// The header's text.
fn text(
    name: &str,
    plans: &[Plan],
    names: &[String],
    tensors: &[LoadTensor],
    lines: &Lines,
) -> String {
    let (upper, lower) = (name.to_uppercase(), name.to_lowercase());
    let made = lines.made(name);
    let mut out = vec![
        format!("/* {made}"),
        " * its handlers' numbers and each command word's place (a0 is 0), for a".to_string(),
        " * thread's commands (kmx_submit). */".to_string(),
        "#pragma once".to_string(),
    ];
    for (number, plan) in plans.iter().enumerate() {
        let handler = format!("{upper}_{}", plan.name.to_uppercase());
        let line = lines
            .handler_line(plan.program, plan.handler)
            .map_or(String::new(), |l| format!(", line {l}"));
        out.push(String::new());
        out.push(format!("/* {}{line}. */", plan.name.to_uppercase()));
        out.push(format!(
            "#define {upper}_HANDLER_{} {}u",
            plan.name.to_uppercase(),
            number + 1
        ));
        for (place, word) in plan.words.iter().enumerate() {
            out.push(format!(
                "#define {handler}_{} {place}u",
                word.name.to_uppercase()
            ));
        }
        out.push(format!("#define {handler}_WORDS {}u", plan.words.len()));
    }
    let program = format!("kProgram{}", super::camel(&lower));
    out.extend(
        [
            "",
            "/* For the thread's C, from the same build: the program field, the class's",
            " * constants, the load's words, the regions, and a function per handler that",
            " * fills its command for kmx_submit. */",
            "#include \"kmx.h\"",
            "",
        ]
        .map(str::to_string),
    );
    out.push(format!(
        "/* The program field: {program} in a run of several programs, else 0. */"
    ));
    out.push(format!("#ifdef {program}"));
    out.push(format!("#define {upper}_PROGRAM {program}"));
    out.push("#else".to_string());
    out.push(format!("#define {upper}_PROGRAM 0u"));
    out.push("#endif".to_string());
    if !names.is_empty() {
        tensor_lines(&mut out, &upper, &lower, names, tensors);
        kind_lines(&mut out, &upper, &lower, names, tensors);
    }
    for plan in plans {
        out.push(String::new());
        out.extend(fill_function(plan, &upper, &lower, lines));
        out.extend(fill_with(plan, &upper, &lower, names, tensors, lines));
    }
    out.push(String::new());
    out.extend([
        format!("/* Cores [first, first + count) hold {lower} (kmx_load): KMX_OK, or KMX_BAD"),
        " * with the refusal in `event`. */".to_string(),
        format!("static inline int {lower}_load("),
        "    uint32_t first, uint32_t count, uint64_t deadline, kmx_event* event".to_string(),
        ") {".to_string(),
        format!("    return kmx_load({upper}_PROGRAM, first, count, deadline, event);"),
        "}".to_string(),
    ]);
    out.join("\n") + "\n"
}

/// The load's words: each tensor's place and pages, the word reader, and a DRAM
/// helper for each tensor a thread may fetch and store (`class_thread.tensor_lines`).
fn tensor_lines(
    out: &mut Vec<String>,
    upper: &str,
    lower: &str,
    names: &[String],
    tensors: &[LoadTensor],
) {
    out.extend([
        String::new(),
        "/* The load's words, after the thread's own: each tensor of the make, its".to_string(),
        " * address; its pages. */".to_string(),
        format!("#define {upper}_WORDS_OWN 0u"),
    ]);
    for (place, (name, tensor)) in names.iter().zip(tensors).enumerate() {
        let big = name.to_uppercase();
        out.push(format!(
            "#define {upper}_WORD_{big} ({upper}_WORDS_OWN + {place}u)"
        ));
        out.push(format!("#define {upper}_{big}_PAGES {}u", tensor.pages));
    }
    out.extend([
        format!("static inline uint32_t {lower}_word(uint32_t place) {{"),
        "    return kmx_args_get()->words[place];".to_string(),
        "}".to_string(),
    ]);
    for (name, tensor) in names.iter().zip(tensors) {
        let Some(page) = tensor.dram_page else {
            continue;
        };
        out.extend([
            format!("/* {name}'s pages from `first`, for kmx_fetch and kmx_store. */"),
            format!("static inline kmx_dram {lower}_{name}_dram(uint32_t first) {{"),
            "    const kmx_dram dram = {".to_string(),
            format!(
                "        {lower}_word({upper}_WORD_{}),",
                name.to_uppercase()
            ),
            "        KM_COPY_INTERLEAVED,".to_string(),
            format!("        {page}u,"),
            "        first,".to_string(),
            "    };".to_string(),
            "    return dram;".to_string(),
            "}".to_string(),
        ]);
    }
}

/// Each tensor's kind: tensors of one layout share one, numbered from 1 in order
/// (`class_thread.kinds_of`).
fn kinds(tensors: &[LoadTensor]) -> Vec<u64> {
    let mut seen: Vec<&Vec<u64>> = Vec::new();
    tensors
        .iter()
        .map(|t| match seen.iter().position(|l| **l == t.layout) {
            Some(at) => at as u64 + 1,
            None => {
                seen.push(&t.layout);
                seen.len() as u64
            }
        })
        .collect()
}

/// A tensor's kind and pages by its word's place (`class_thread.tensor_kinds`).
fn kind_lines(
    out: &mut Vec<String>,
    upper: &str,
    lower: &str,
    names: &[String],
    tensors: &[LoadTensor],
) {
    let kinds = kinds(tensors);
    out.push(
        "/* A load tensor's kind (page bytes, format, layout, place; 0: no tensor) and".to_string(),
    );
    out.push(" * pages, by its word's place: what <handler>_with checks and gives. */".to_string());
    for function in ["kind", "pages"] {
        out.push(format!(
            "static inline uint32_t {lower}_{function}(uint32_t place) {{"
        ));
        out.push("    switch (place) {".to_string());
        for (name, kind) in names.iter().zip(&kinds) {
            let big = name.to_uppercase();
            let given = if function == "kind" {
                format!("{kind}u")
            } else {
                format!("{upper}_{big}_PAGES")
            };
            out.push(format!("    case {upper}_WORD_{big}: return {given};"));
        }
        out.extend(["    default: return 0u;", "    }", "}"].map(str::to_string));
    }
}

/// The C function that fills a handler's command: the load's tensors by their words,
/// the other parameters given (`class_thread.fill_function`). Gives the lines and the
/// signature's names.
fn fill(plan: &Plan, upper: &str, lower: &str, lines: &Lines) -> (Vec<String>, Vec<String>) {
    let (mut arguments, mut said, mut body) = (Vec::new(), Vec::new(), Vec::new());
    for (place, word) in plan.words.iter().enumerate() {
        let to = format!("    km_command->a{place} = ");
        if word.kind == WordKind::Tensor {
            body.push(format!(
                "{to}{lower}_word({upper}_WORD_{});",
                word.name.to_uppercase()
            ));
            said.push(format!("{} the make's tensor", word.name));
        } else {
            let a = argument(&word.name);
            body.push(format!("{to}{a};"));
            arguments.push(a);
            said.push(word.name.clone());
        }
    }
    let function = format!("{lower}_{}", plan.name);
    let mut signature: Vec<String> = [
        "km_command_t* km_command",
        "uint32_t km_flags",
        "uint32_t km_first",
        "uint32_t km_count",
    ]
    .map(str::to_string)
    .to_vec();
    signature.extend(arguments.iter().map(|a| format!("uint32_t {a}")));
    let line = lines
        .handler_line(plan.program, plan.handler)
        .map_or(String::new(), |l| format!(", line {l}"));
    let said = if said.is_empty() {
        "no words".to_string()
    } else {
        said.join(", ")
    };
    let mut out = vec![
        format!("/* {}{line}: {said}. */", plan.name.to_uppercase()),
        format!("static inline void {function}("),
    ];
    out.extend(
        signature
            .iter()
            .take(signature.len() - 1)
            .map(|s| format!("    {s},")),
    );
    out.extend(signature.last().map(|s| format!("    {s}")));
    out.push(") {".to_string());
    out.push(format!(
        "    kmx_command(km_command, {upper}_PROGRAM, {upper}_HANDLER_{}, km_first, km_count);",
        plan.name.to_uppercase()
    ));
    out.push("    km_command->hdr |= km_flags << 8;".to_string());
    out.extend(body);
    out.push("}".to_string());
    let given = signature
        .iter()
        .map(|s| s.split_whitespace().last().unwrap_or("").to_string())
        .collect();
    (out, given)
}

/// `fill_function`'s lines.
fn fill_function(plan: &Plan, upper: &str, lower: &str, lines: &Lines) -> Vec<String> {
    fill(plan, upper, lower, lines).0
}

/// `<handler>_with`: the handler's function on the tensors the thread names by their
/// words' places, `KMX_BAD` for a tensor of another kind than the load's
/// (`class_thread.fill_with`).
fn fill_with(
    plan: &Plan,
    upper: &str,
    lower: &str,
    names: &[String],
    tensors: &[LoadTensor],
    lines: &Lines,
) -> Vec<String> {
    let places: Vec<(usize, &str)> = plan
        .words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.kind == WordKind::Tensor)
        .map(|(place, w)| (place, w.name.as_str()))
        .collect();
    if places.is_empty() {
        return Vec::new();
    }
    let kinds = kinds(tensors);
    let kind_of = |name: &str| {
        names
            .iter()
            .position(|n| n == name)
            .and_then(|i| kinds.get(i))
            .copied()
            .unwrap_or(0)
    };
    let (_, given) = fill(plan, upper, lower, lines);
    let function = format!("{lower}_{}", plan.name);
    let named: Vec<String> = places
        .iter()
        .map(|(_, p)| format!("{}_place", argument(p)))
        .collect();
    let mut body = Vec::new();
    for ((_, parameter), place_name) in places.iter().zip(&named) {
        body.extend([
            format!(
                "    if ({lower}_kind({place_name}) != {}u) {{",
                kind_of(parameter)
            ),
            "        return KMX_BAD;".to_string(),
            "    }".to_string(),
        ]);
    }
    body.push(format!("    {function}({});", given.join(", ")));
    for ((place, _), place_name) in places.iter().zip(&named) {
        body.push(format!(
            "    km_command->a{place} = {lower}_word({place_name});"
        ));
    }
    let mut signature: Vec<String> = [
        "km_command_t* km_command",
        "uint32_t km_flags",
        "uint32_t km_first",
    ]
    .map(str::to_string)
    .to_vec();
    signature.extend(
        given
            .iter()
            .skip(3)
            .chain(&named)
            .map(|n| format!("uint32_t {n}")),
    );
    let mut out = vec![
        String::new(),
        format!(
            "/* {} on the tensors the thread names, each by its",
            plan.name.to_uppercase()
        ),
        format!(" * word's place ({upper}_WORD_<name>); KMX_BAD for one of another kind. */"),
        format!("static inline int {function}_with("),
    ];
    out.extend(
        signature
            .iter()
            .take(signature.len() - 1)
            .map(|s| format!("    {s},")),
    );
    out.extend(signature.last().map(|s| format!("    {s}")));
    out.push(") {".to_string());
    out.extend(body);
    out.push("    return KMX_OK;".to_string());
    out.push("}".to_string());
    out
}
