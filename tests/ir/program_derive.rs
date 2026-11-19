//! Derive: the effects of handlers and runs, the dependencies of a chain, and the
//! requirement set with its causes, read off lowered modules.

use komorebi_ir::derive::derive;
use komorebi_ir::lower::lower;
use komorebi_ir::program::body::InstId;
use komorebi_ir::program::derive::{
    Access, Cause, ChainEdge, Requirement, StateAccess, StateUse, TensorAccess,
};
use komorebi_ir::program::notes::Anchor;
use komorebi_ir::program::{validate, BodyRef, CallArg, Module};
use komorebi_ir::types::{
    Ability, ChainId, ConstValue, Entry, FuncId, HandlerId, MemberId, ProgramId, RunId, ScalarType,
};

fn derived(text: &str) -> Module {
    let checked = komorebi_ir::check(text).unwrap_or_else(|d| panic!("{d:?}"));
    let mut module = lower(&checked).unwrap_or_else(|e| panic!("{e:?}"));
    derive(&mut module);
    let violations = validate(&module);
    assert!(violations.is_empty(), "{violations:?}");
    module
}

fn lesson(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/ir/corpus/golden/lessons")
        .join(format!("{name}.kom"));
    std::fs::read_to_string(path).expect("the lesson is readable")
}

fn access(param: u32, body: BodyRef, view: u32, access: Access, at: u32) -> TensorAccess {
    TensorAccess {
        param,
        body,
        view: InstId(view),
        access,
        at: InstId(at),
    }
}

const P0: ProgramId = ProgramId(0);

#[test]
fn a_handler_streams_one_tensor_into_another() {
    let m = derived(&lesson("t01_scale"));
    let h0 = BodyRef::Handler(P0, HandlerId(0));
    let e = &m.programs[0].handlers[0].effects;
    assert_eq!(
        e.tensors,
        vec![
            access(0, h0, 8, Access::Stream, 17),
            access(1, h0, 12, Access::Sink, 19)
        ]
    );
    assert_eq!(e.calls, vec![FuncId(0)]);
    assert!(e.state.is_empty() && e.collectives.is_empty());
    assert_eq!(m.requirements, Vec::<Requirement>::new());
}

#[test]
fn one_view_both_read_and_written() {
    let m = derived(&lesson("t11_solve"));
    let h0 = BodyRef::Handler(P0, HandlerId(0));
    assert_eq!(
        m.programs[0].handlers[0].effects.tensors,
        vec![
            access(0, h0, 6, Access::Stream, 11),
            access(0, h0, 6, Access::Sink, 14)
        ]
    );
    let h1 = BodyRef::Handler(P0, HandlerId(1));
    assert_eq!(
        m.programs[0].handlers[1].effects.tensors,
        vec![access(0, h1, 8, Access::Stream, 11)]
    );
}

#[test]
fn a_run_uses_its_state_and_needs_dynamic() {
    let m = derived(&lesson("t09_walk"));
    let s0 = MemberId(0);
    let use_ = |body, how, at| StateAccess {
        member: s0,
        body,
        how,
        at: InstId(at),
    };
    let seed = BodyRef::Seed(P0, RunId(0));
    let done = BodyRef::On(P0, RunId(0), 1);
    assert_eq!(
        m.programs[0].runs[0].effects.state,
        vec![
            use_(seed, StateUse::FillWhole, 1),
            use_(done, StateUse::Read, 3),
            use_(done, StateUse::WriteElement, 6),
        ]
    );
    assert_eq!(
        m.requirements,
        vec![Requirement {
            ability: Ability::Dynamic,
            causes: vec![Cause::Stated, Cause::DynamicRun(Entry::Run(P0, RunId(0)))],
        }]
    );
}

const CHAIN: &str = "module c

tile [32, 32]
format bf16

program P {
    handler copy(x: tensor<bf16>, y: tensor<bf16>, n: u32) @on(group) {
        let out = sink(y.tiles(0, n), block = 1)
        for b in stream(x.tiles(0, n), block = 1) {
            out.push(b)
        }
    }

    chain twice(a: tensor<bf16>, b: tensor<bf16>, c: tensor<bf16>, n: u32) @on(group) {
        call copy(a, b, n)
        call copy(PAST, c, n)
    }
}
";

#[test]
fn a_chain_reads_what_an_earlier_call_wrote() {
    for (arg, past) in [("b", false), ("past(b)", true)] {
        let m = derived(&CHAIN.replace("PAST", arg));
        assert_eq!(
            m.programs[0].chains[0].deps.edges,
            vec![ChainEdge {
                writer: 0,
                reader: 1,
                param: 1,
                write: 1,
                read: 0,
                past,
            }],
            "{arg}"
        );
    }
    // no shared tensor, no edge
    let m = derived(&CHAIN.replace("PAST", "a"));
    assert!(m.programs[0].chains[0].deps.edges.is_empty());
}

#[test]
fn a_chain_passes_a_parameter_bound_at_build_as_its_constant() {
    let text = CHAIN
        .replace("PAST", "b")
        .replace(
            "format bf16\n",
            "format bf16\n\nparam LIMIT: u32 = 64 @bind(build)\n",
        )
        .replace("call copy(b, c, n)", "call copy(b, c, LIMIT)");
    let m = derived(&text);
    assert_eq!(
        m.programs[0].chains[0].calls[1].args[2],
        CallArg::Const(ConstValue::Int(ScalarType::U32, 64))
    );
}

#[test]
fn each_call_of_a_chain_has_its_span() {
    let text = lesson("t10_steps");
    let m = derived(&text);
    let calls: Vec<&str> = m
        .notes
        .spans
        .iter()
        .filter_map(|(a, s)| match a {
            Anchor::Call(ProgramId(0), ChainId(0), _) => text.get(s.start as usize..s.end as usize),
            _ => None,
        })
        .collect();
    assert_eq!(
        calls,
        [
            "call scale(x, y, k, n)",
            "call plus(z, past(y), s, n)",
            "call total(s, t, n)"
        ]
    );
}

const FALLBACKS: &str = "module fallbacks

tile [32, 32]
format f32

func plain(t: tile<f32>) -> tile<f32> {
    return t
}

extern func fill(v: view, t: tile<f32>) -> tile<f32> @writes(v) @impl(\"x.fill\") fallback {
    store v <- t
    return t
}
";

#[test]
fn a_fallback_has_its_effects_on_its_view_parameters() {
    let m = derived(FALLBACKS);
    assert_eq!(m.funcs[0].effects.tensors, Vec::<TensorAccess>::new());
    let a = &m.funcs[1].effects.tensors;
    assert_eq!(a.len(), 1);
    assert_eq!((a[0].param, a[0].access), (0, Access::Store));
    assert_eq!(a[0].body, BodyRef::Func(FuncId(1)));
    // the view is the parameter's `Arg`
    assert_eq!(a[0].view, InstId(0));
}

#[test]
fn derive_twice_is_derive_once() {
    let mut m = derived(&lesson("t09_walk"));
    let once = m.clone();
    derive(&mut m);
    assert_eq!(m, once);
}

#[test]
fn every_lesson_that_lowers_also_derives_and_validates() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/corpus/golden/lessons");
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("the lessons are readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    let mut derived_count = 0;
    for path in entries {
        let text = std::fs::read_to_string(&path).expect("a lesson is readable");
        let checked = komorebi_ir::check(&text).unwrap_or_else(|d| panic!("{d:?}"));
        // the lessons that do not lower to a valid structure yet are the lower test's
        let Ok(mut module) = lower(&checked) else {
            continue;
        };
        if !validate(&module).is_empty() {
            continue;
        }
        derive(&mut module);
        let violations = validate(&module);
        assert!(violations.is_empty(), "{}: {violations:?}", path.display());
        derived_count += 1;
    }
    assert!(derived_count >= 15, "{derived_count} lessons derived");
}
