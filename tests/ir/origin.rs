//! A frontend's text with its origin beside it (`api::origin`): the places the spans
//! give, by the narrowest range of the map; the spans cleared; the renames of values and
//! parameters; the diagnostics placed; and the refusals of an origin.

use komorebi_ir::api::origin::{Given, Mapped, Place, Refused};
use komorebi_ir::api::{self, origin};
use komorebi_ir::bytecode;
use komorebi_ir::program::body::InstId;
use komorebi_ir::program::notes::{Anchor, Owner};
use komorebi_ir::program::BodyRef;
use komorebi_ir::types::{HandlerId, ProgramId};

const T1: &str = include_str!("corpus/golden/lessons/t01_scale.kom");

/// One range: the text's lines `a` to `b`, printed for the source's `first` to `last`.
fn range(a: u32, b: u32, first: u32, last: u32) -> Mapped {
    Mapped {
        text_first: a,
        text_last: b,
        file: 0,
        first,
        last,
    }
}

/// T1 as the frontend prints it for the class form's script (`def run` on line 18):
/// the handler's lines on its `def`, each statement's lines on the statement's, the
/// loop on its lines 21 to 22 and its push on 22.
fn t1_given() -> Given {
    Given {
        files: vec!["/work/t1_class.py".into()],
        lines: (18..=22).map(|n| (0, n, format!("line {n}"))).collect(),
        map: vec![
            range(11, 24, 18, 18),
            range(18, 18, 19, 19),
            range(19, 20, 20, 20),
            range(21, 23, 21, 22),
            range(22, 22, 22, 22),
        ],
        renames: Vec::new(),
    }
}

fn handler() -> (ProgramId, HandlerId) {
    (ProgramId(0), HandlerId(0))
}

/// The class form's module (tests/ir/tensix/): the lesson with two lines of comment
/// first, so each range of `t1_given` two lines further.
const CLASS_FORM: &str = include_str!("tensix/t1_class.kom");

/// The class form's place table (tests/ir/tensix/t1_places.txt): `handler P H`,
/// `param P H K` and `inst P H I`, each then its first and last lines.
fn table() -> Vec<(Anchor, u32, u32, u32)> {
    let text = include_str!("tensix/t1_places.txt");
    let rows = text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    rows.map(|line| {
        let (kind, rest) = line.split_once(' ').expect("a place line");
        let n: Vec<u32> = rest
            .split(' ')
            .map(|f| f.parse().expect("a number"))
            .collect();
        let (p, h) = (ProgramId(n[0]), HandlerId(n[1]));
        let anchor = match kind {
            "handler" => Anchor::Handler(p, h),
            "param" => Anchor::ParamOf(Owner::Handler(p, h), n[2]),
            _ => Anchor::Inst(BodyRef::Handler(p, h), InstId(n[2])),
        };
        (anchor, 0, n[n.len() - 2], n[n.len() - 1])
    })
    .collect()
}

#[test]
fn the_spans_give_the_places_of_the_class_form() {
    let mut given = t1_given();
    for m in &mut given.map {
        (m.text_first, m.text_last) = (m.text_first + 2, m.text_last + 2);
    }
    let module = origin::build(CLASS_FORM, &given).expect("accepted");
    assert!(module.notes.spans.is_empty());
    let found = module.notes.origin.as_ref().expect("an origin");
    assert_eq!(found.files, ["/work/t1_class.py"]);
    assert_eq!(found.lines.len(), 5);
    // The handler, its parameters and its instructions: the class form's table.
    assert_eq!(found.places, table());
}

#[test]
fn an_origin_changes_only_the_full_hash_and_round_trips() {
    let plain = api::build(T1).expect("accepted");
    let module = origin::build(T1, &t1_given()).expect("accepted");
    let (a, b) = (bytecode::hashes(&plain), bytecode::hashes(&module));
    assert_eq!(a.semantic, b.semantic);
    assert_eq!(a.build, b.build);
    assert_ne!(a.full, b.full);
    let bytes = origin::compile(T1, &t1_given()).expect("accepted");
    assert_eq!(bytes, bytecode::encode(&module));
    let loaded = api::load(&bytes).expect("loaded");
    assert_eq!(loaded.notes.origin, module.notes.origin);
    assert_eq!(bytecode::encode(&loaded), bytes);
}

#[test]
fn renames_give_the_source_names_of_values_and_parameters() {
    let mut given = t1_given();
    // The source's names may be the IR's words, and one safe name another's real one.
    given.renames = vec![
        ("tiles".into(), "group".into()),
        ("src".into(), "stream".into()),
        ("dst".into(), "src".into()),
    ];
    let module = origin::build(T1, &given).expect("accepted");
    let (p, h) = handler();
    let tiles = Anchor::ParamOf(Owner::Handler(p, h), 3);
    let name = |a| {
        module
            .notes
            .param_names
            .iter()
            .find(|(b, _)| *b == a)
            .map(|(_, n)| n.clone())
    };
    assert_eq!(name(tiles).as_deref(), Some("group"));
    let values: Vec<&str> = module
        .notes
        .names
        .iter()
        .map(|n| match n {
            komorebi_ir::program::notes::Named::Value(_, _, s)
            | komorebi_ir::program::notes::Named::Local(_, _, s) => s.as_str(),
        })
        .collect();
    assert!(values.contains(&"stream") && values.contains(&"src"));
    assert!(!values.contains(&"dst"));
    let plain = origin::build(T1, &t1_given()).expect("accepted");
    let (a, b) = (bytecode::hashes(&plain), bytecode::hashes(&module));
    assert_eq!((a.semantic, a.build), (b.semantic, b.build));
    assert_ne!(a.full, b.full);
}

#[test]
fn a_diagnostic_is_placed_on_the_class_line() {
    let broken = T1.replace("out.push(times(xb, k))", "out.push(times(xb, q))");
    let rows = origin::check(&broken, &t1_given()).expect("the origin is accepted");
    assert!(!rows.is_empty());
    for (d, place) in &rows {
        assert_eq!(d.start.line, 22, "{}", d.message);
        assert_eq!(
            *place,
            Some(Place {
                file: 0,
                first: 22,
                last: 22
            })
        );
    }
    assert_eq!(origin::check(T1, &t1_given()), Ok(Vec::new()));
}

/// The refusal of `given` beside T1, in words.
fn refused(given: &Given) -> String {
    match origin::build(T1, given) {
        Err(Refused::Origin(what)) => what,
        other => panic!("not an origin refusal: {other:?}"),
    }
}

#[test]
fn an_origin_is_refused_by_name() {
    let mut g = t1_given();
    g.map.push(range(20, 21, 20, 20));
    assert!(refused(&g).contains("never the same lines twice nor crossing"));
    let mut g = t1_given();
    g.map.push(range(18, 18, 19, 19));
    assert!(refused(&g).contains("never the same lines twice nor crossing"));
    let mut g = t1_given();
    g.map[1].file = 1;
    assert!(refused(&g).contains("past the files"));
    let mut g = t1_given();
    g.map[1].first = 0;
    assert!(refused(&g).contains("lines count from 1"));
    let mut g = t1_given();
    g.renames = vec![("nothing".into(), "x".into())];
    assert!(refused(&g).contains("`nothing` names no value and no parameter"));
    let mut g = t1_given();
    g.renames = vec![("src".into(), "a".into()), ("src".into(), "b".into())];
    assert!(refused(&g).contains("renamed twice"));
    // A place on a line the origin does not hold: the module's own rule (validate).
    let mut g = t1_given();
    g.lines.retain(|(_, n, _)| *n != 20);
    assert!(refused(&g).contains("which is not held"));
    // The text refused before the origin is looked at past its map.
    assert!(matches!(
        origin::build("module", &t1_given()),
        Err(Refused::Text(_))
    ));
}
