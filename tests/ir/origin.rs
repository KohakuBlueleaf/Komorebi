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

/// One range of `text`: its whole lines `a` to `b` (column 1 of `a` to the end of
/// `b`), printed for the source's `first` to `last`.
fn range(text: &str, a: u32, b: u32, first: u32, last: u32) -> Mapped {
    let end = text
        .lines()
        .nth(b as usize - 1)
        .map_or(0, |l| l.chars().count());
    Mapped {
        text_first: a,
        text_first_col: 1,
        text_last: b,
        text_last_col: (end as u32).max(1),
        file: 0,
        first,
        last,
    }
}

/// T1 as the frontend prints it for the class form's script (`def run` on line 18):
/// the handler's lines on its `def`, each statement's lines on the statement's, the
/// loop on its lines 21 to 22 and its push on 22; in `text`, `shift` lines further.
fn t1_given_in(text: &str, shift: u32) -> Given {
    let r = |a: u32, b: u32, first, last| range(text, a + shift, b + shift, first, last);
    Given {
        files: vec!["/work/t1_class.py".into()],
        lines: (18..=22).map(|n| (0, n, format!("line {n}"))).collect(),
        map: vec![
            r(11, 24, 18, 18),
            r(18, 18, 19, 19),
            r(19, 20, 20, 20),
            r(21, 23, 21, 22),
            r(22, 22, 22, 22),
        ],
        renames: Vec::new(),
    }
}

fn t1_given() -> Given {
    t1_given_in(T1, 0)
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
    let given = t1_given_in(CLASS_FORM, 2);
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
    g.map.push(range(T1, 20, 21, 20, 20));
    assert!(refused(&g).contains("never the same range twice nor crossing"));
    let mut g = t1_given();
    g.map.push(range(T1, 18, 18, 19, 19));
    assert!(refused(&g).contains("never the same range twice nor crossing"));
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

#[test]
fn the_writers_layout_carries_each_range_to_where_its_text_went() {
    // T1 with its indentation doubled, laid out: T1's own text, and each anchor in the
    // place it had before the layout
    let spread = T1.replace("\n    ", "\n        ");
    let given = t1_given_in(&spread, 0);
    let (text, laid) = origin::layout(&spread, &given).expect("laid out");
    assert_eq!(text, T1);
    let before = origin::build(&spread, &given).expect("accepted");
    let after = origin::build(&text, &laid).expect("accepted");
    assert_eq!(after.notes.origin, before.notes.origin);
    // a range starts where its first character went, past the new indentation
    assert_eq!(laid.map[1].start(), (18, 9));
}

/// A frontend's text with every name it was given between backticks (R50), as the
/// class frontend writes it: a function `square` called beside the language's
/// `square`, a value `group` beside the language's `group.size`, a var `store` (a
/// field's var named by a word of the text) and names that need none.
const QUOTED: &str = "module `quoted`

tile [32, 32]
format f32

func `square`(`a`: tile<f32>) -> tile<f32> {
    return square(`a`) * `a`
}

program `P` {
    handler `run`(`x`: tensor<f32>, `y`: tensor<f32>, `group`: u32) @on(group) {
        let `src` = partition(`x`.tiles(0, `group`), worker.rank, group.size)
        let `dst` = partition(`y`.tiles(0, `group`), worker.rank, group.size)
        let `out` = sink(`dst`, block = 1)
        var `store`: tile<f32> = zeros
        for `xb` in stream(`src`, block = 1) {
            `store` += `square`(`xb`)
            `out`.push(`store`)
        }
    }
}
";

/// The range of `text` that holds exactly the `n`th (from 0) `word`, as (line, column)
/// from 1, both ends held.
fn word_at(text: &str, word: &str, n: usize) -> Mapped {
    let (line, col) = text
        .lines()
        .enumerate()
        .flat_map(|(i, l)| l.match_indices(word).map(move |(c, _)| (i, c)))
        .nth(n)
        .expect("the word");
    let first = l1(line, col);
    let last = (first.0, first.1 + word.chars().count() as u32 - 1);
    Mapped {
        text_first: first.0,
        text_first_col: first.1,
        text_last: last.0,
        text_last_col: last.1,
        file: 0,
        first: 3,
        last: 3,
    }
}

fn l1(line: usize, col: usize) -> (u32, u32) {
    (line as u32 + 1, col as u32 + 1)
}

/// The text a range holds (on one line).
fn held(text: &str, m: &Mapped) -> String {
    let line = text
        .lines()
        .nth(m.text_first as usize - 1)
        .expect("the line");
    line.chars()
        .skip(m.text_first_col as usize - 1)
        .take((m.text_last_col - m.text_first_col + 1) as usize)
        .collect()
}

/// Whether two texts with their origins compile to the same byte code, bytes compared.
fn same_program(a: (&str, &Given), b: (&str, &Given)) -> bool {
    match (origin::compile(a.0, a.1), origin::compile(b.0, b.1)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

#[test]
fn the_layout_keeps_the_backticks_a_name_needs_and_the_byte_code_is_the_raw_texts() {
    let lines = QUOTED.lines().count() as u32;
    let given = Given {
        files: vec!["/work/quoted.py".into()],
        lines: vec![(0, 1, "line 1".into()), (0, 3, "line 3".into())],
        map: vec![
            range(QUOTED, 1, lines, 1, 1),
            // a note on the call of the module's `square`, which keeps its backticks,
            // and one on the loop's `xb`, which needs none
            word_at(QUOTED, "`square`", 1),
            word_at(QUOTED, "`xb`", 1),
        ],
        renames: Vec::new(),
    };
    let (text, laid) = origin::layout(QUOTED, &given).expect("laid out");
    // kept where the language has a name of that spelling where the name stands
    for kept in [
        "func `square`(a: tile<f32>) -> tile<f32> {",
        "    return square(a) * a",
        "    handler run(x: tensor<f32>, y: tensor<f32>, `group`: u32) @on(group) {",
        "partition(x.tiles(0, `group`), worker.rank, group.size)",
        "        var `store`: tile<f32> = zeros",
        "            `store` += `square`(xb)",
        "            out.push(`store`)",
    ] {
        assert!(text.contains(kept), "{kept}\n{text}");
    }
    assert!(text.starts_with("module quoted\n") && text.contains("program P {"));
    // a range on a name still holds the name after the layout, with its backticks
    // where they were kept (as the name's own span does), without where taken away
    assert_eq!(held(&text, &laid.map[1]), "`square`");
    assert_eq!(held(&text, &laid.map[2]), "xb");
    // the laid-out text is the same program as the raw one, bytes compared
    assert!(same_program((QUOTED, &given), (&text, &laid)));
    // a writer that dropped a needed pair would be caught: the call of the module's
    // `square` read as the language's, or the value `group` declared bare (K120)
    let call = text.replace("`square`(xb)", "square(xb)");
    assert!(!same_program((QUOTED, &given), (&call, &laid)));
    let value = text.replace("`group`", "group");
    assert!(!same_program((QUOTED, &given), (&value, &laid)));
}
