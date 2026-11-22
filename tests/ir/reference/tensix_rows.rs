//! The tile compiler's row form against today's, with no card: today's
//! `tests/ir/tensix/tile/tile_rows_form.py` traces functions on a row of n tiles,
//! written in the IR on blocks in `tile_rows_form.kom`, and makes the row form's unit
//! of each row of 2 tiles (one reduction, or in levels) from its KernelIR, which the
//! port reads back from today's text, and its band; the branches the corpus does not
//! reach, the row form's checks and the band's runs on a few words, and where a call
//! holds its operands in Dst.

use std::path::Path;

use komorebi_ir::backend::tensix::tile::emit::emit;
use komorebi_ir::backend::tensix::tile::ir::{kernel_ir, row_kernel_ir};
use komorebi_ir::backend::tensix::tile::lower::{unit_program, FIDELITY};
use komorebi_ir::backend::tensix::tile::py::float_repr;
use komorebi_ir::backend::tensix::tile::rows::{
    band_groups, check_broadcast, check_lanes, row_program, Layout, Layouts,
};
use komorebi_ir::backend::tensix::tile::unit::{Count, Insn, Item, UnitProgram, MATH};
use komorebi_ir::backend::tensix::unit::row_layout;
use komorebi_ir::program::body::{InstId, Op};
use komorebi_ir::program::notes::{Anchor, Origin};
use komorebi_ir::program::BodyRef;

#[allow(dead_code)] // each target takes a part of the shared helpers
#[path = "tile_common.rs"]
mod common;
use common::*;

/// The row form's corpus, `tile_rows_form.kom`.
fn rows_corpus() -> komorebi_ir::program::Module {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir/tensix/tile/tile_rows_form.kom");
    let text = std::fs::read_to_string(path).expect("tile_rows_form.kom");
    komorebi_ir::api::build(&text).unwrap_or_else(|e| panic!("{e:?}"))
}

#[test]
fn each_function_on_a_row_traces_to_todays_kernel_ir() {
    let Some(folder) = today_rows() else { return };
    let module = rows_corpus();
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("the cases");
    let mut names: Vec<&str> = Vec::new();
    for line in cases.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (case, width, fields) = (f[0], f[1].parse().ok(), f[2].parse::<usize>());
        let fields = fields.expect("the fields");
        names.push(case);
        let at = func(&module, &format!("t_{case}"));
        let params = module.funcs[at.0 as usize].params.len();
        let resident: Vec<bool> = (0..params).map(|k| k >= params - fields).collect();
        let want = std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
        let ir = row_kernel_ir(&module, at, case, width, &resident)
            .unwrap_or_else(|r| panic!("{case}: {}", r.what));
        let got = ir.dump() + "\n";
        assert!(
            got == want,
            "{case} differs, {}",
            first_difference(&want, &got)
        );
    }
    names.sort();
    let mut funcs: Vec<&str> = module
        .funcs
        .iter()
        .filter_map(|f| f.name.strip_prefix("t_"))
        .collect();
    funcs.sort();
    assert_eq!(funcs, names, "a func `t_<case>` for each case and no other");
}

#[test]
fn a_reduction_on_one_tile_is_refused_with_todays_text_and_line() {
    let Some(folder) = today_rows() else { return };
    let today = std::fs::read_to_string(folder.join("one_tile.refused")).expect("the text");
    let today = today.trim_end();
    // today's line is the reduction's, in the script: the reduction's instruction is
    // placed there and every other one elsewhere
    let line: u32 = today
        .rsplit(':')
        .next()
        .and_then(|l| l.strip_suffix(')'))
        .and_then(|l| l.parse().ok())
        .expect("the line");
    let mut module = rows_corpus();
    let f = func(&module, "one");
    let body = module.funcs[f.0 as usize].body.as_ref().expect("a body");
    let places = (0..body.insts.len())
        .map(|i| {
            let op = &body.insts[i].op;
            let at = if matches!(op, Op::Reduce(..)) {
                line
            } else {
                1000 + i as u32
            };
            let inst = Anchor::Inst(BodyRef::Func(f), InstId(i as u32));
            (inst, 0, at, at)
        })
        .collect();
    module.notes.origin = Some(Origin {
        files: vec!["tests/ir/tensix/tile/tile_rows_form.py".into()],
        lines: Vec::new(),
        places,
    });
    let ir = row_kernel_ir(&module, f, "one", Some(1), &[false]).expect("the trace");
    let refused = unit_program(&ir, "fp32", true, false, FIDELITY).expect_err("refused");
    let refusal = refused.refusal(Some(Anchor::Func(f)));
    assert_eq!((refusal.code, refusal.what.as_str()), ("K962", today));
}

#[test]
fn what_the_row_forms_trace_would_not_make_is_refused_by_name() {
    let text = "module refused\n\ntile [32, 32]\nformat f32\n\n\
        func streamed(x: tile<f32>[2], g: tile<f32>[2]) -> tile<f32>[2] {\n\
        \x20   return x * rsqrt(reduce_sum(x * x, axis = row)).col() * g.row()\n}\n\n\
        func scalar(x: tile<f32>[2], k: f32) -> tile<f32>[2] {\n\
        \x20   return x * reduce_sum(x, axis = row).col() * k\n}\n\n\
        func `square`(x: tile<f32>[2, 2]) -> tile<f32>[2, 2] {\n\
        \x20   return x * reduce_sum(x, axis = row).col()\n}\n";
    let module = komorebi_ir::api::build(text).unwrap_or_else(|e| panic!("{e:?}"));
    let named = |f: &str, count: Option<u64>, resident: &[bool]| {
        let refusal = row_kernel_ir(&module, func(&module, f), f, count, resident).expect_err(f);
        assert_eq!(refusal.code, "K964", "{f}");
        refusal.what
    };
    let not = |what: &str| format!("the Tensix backend's tile compiler does not take {what} yet");
    let both = [false, false];
    assert_eq!(
        named("streamed", Some(2), &both),
        not("the row form's .row() of a streamed operand")
    );
    assert_eq!(
        named("streamed", None, &both),
        not("a block whose count is not bound at compile")
    );
    assert_eq!(
        named("streamed", Some(4), &both),
        not("a block of 2 tiles called on a row of 4")
    );
    assert_eq!(
        named("streamed", Some(2), &[true, false]),
        not("a resident operand before a streamed one (today's fields are the last parameters)")
    );
    assert_eq!(
        named("streamed", Some(2), &[true, true]),
        not("a row form of no streamed operand")
    );
    assert_eq!(
        named("streamed", Some(2), &[false]),
        not("a call whose operands' sources are not one a parameter")
    );
    assert_eq!(
        named("scalar", Some(2), &both),
        not("a parameter or result of the row form that is not a block of one row")
    );
    assert_eq!(
        named("square", Some(2), &[false]),
        not("a parameter or result of the row form that is not a block of one row")
    );
    // the 1-tile trace takes a row reduction only where every block is of one tile
    let one = kernel_ir(&module, func(&module, "streamed"), "streamed").expect_err("one");
    assert_eq!(
        one.what,
        not("a function on blocks with a product or a reduction (not element by element)")
    );
}

#[test]
fn each_row_of_two_tiles_gets_todays_unit_from_its_kernel_ir() {
    let Some(folder) = today_rows() else { return };
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("the cases");
    let refused = std::fs::read_to_string(folder.join("rows_refused.txt")).expect("refused");
    let (mut same, mut not_ported) = (Vec::new(), Vec::new());
    // each case on a thread of its own (the identities are long in debug), the
    // outcomes taken in the cases' order: none (not this test's), the case made
    // whole, or the text of a refusal by name
    let (folder, refused) = (&folder, &refused);
    let outcomes: Vec<Option<Result<&str, String>>> = std::thread::scope(|s| {
        let runs: Vec<_> = (cases.lines())
            .map(|line| {
                s.spawn(move || {
                    let f: Vec<&str> = line.split('\t').collect();
                    if f[1] != "2" {
                        return None;
                    }
                    let case = f[0];
                    let text =
                        std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
                    let ir = parse_ir(&text);
                    assert_eq!(ir.dump() + "\n", text, "{case}: the KernelIR read back");
                    let made = row_program(&ir, 2, 2, 4, "fp32", false);
                    if let Some(today) = refused
                        .lines()
                        .find_map(|l| l.strip_prefix(&format!("{case}\t")))
                    {
                        // refused today: the same text, but for the place it names (K962)
                        let error = made.expect_err(case);
                        assert_eq!(
                            error.text(),
                            today.split(" (at ").next().unwrap_or(today),
                            "{case}"
                        );
                        assert_eq!(error.refusal(None).code, "K962", "{case}");
                        return None;
                    }
                    let mut unit = match made {
                        Ok(unit) => unit,
                        Err(e) => {
                            assert_eq!(e.refusal(None).code, "K964", "{case}: {}", e.text());
                            return Some(Err(format!("{case}: {}", e.text())));
                        }
                    };
                    let listing = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                    let words = emit(&mut unit).unwrap_or_else(|e| panic!("{case}: {e}"));
                    let emitted = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                    for (file, got) in [("txt", listing), ("hpp", words), ("emitted.txt", emitted)]
                    {
                        let want =
                            std::fs::read_to_string(folder.join(format!("{case}.rows.{file}")))
                                .unwrap_or_else(|e| panic!("{case}.rows.{file}: {e}"));
                        assert!(
                            got == want,
                            "{case}'s {file} differs, {}",
                            first_difference(&want, &got)
                        );
                    }
                    Some(Ok(case))
                })
            })
            .collect();
        (runs.into_iter())
            .map(|r| r.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    for outcome in outcomes.into_iter().flatten() {
        match outcome {
            Ok(case) => same.push(case),
            Err(text) => not_ported.push(text),
        }
    }
    // one reduction, with and without a resident row (f32 and bf16), a centred row,
    // a row's largest and smallest values (of a product too, folded by max), a sum of
    // exponentials (the mad-reader's repairs taken back into the tile program), a
    // reciprocal's constant register; the levels: two and three passes, two resident
    // rows, a sum kept for a later level (read directly, or through a value a later
    // sum reads), a sum and a largest value reading .row() (of two resident rows too)
    assert_eq!(
        same,
        [
            "rms_norm",
            "centred",
            "row_min",
            "rms_plain",
            "peak",
            "weighted_bf16",
            "softmax_pair",
            "weighted_sum",
            "peak_square",
            "exp_sum",
            "recip_centred",
            "layer_pair",
            "kept_sum",
            "max_weighted",
            "three_plain",
            "kept_through",
            "two_weights"
        ]
    );
    // two row tensors, more live values than the work registers, more constants than
    // the programmable registers, 5 per-row values, 5 resident rows: today's refusals
    // (K962)
    assert_eq!(refused.lines().count(), 5);
    assert_eq!(not_ported, Vec::<String>::new());
}

#[test]
fn the_row_forms_branches_the_corpus_does_not_reach_are_todays() {
    let Some(folder) = today_rows() else { return };
    let lines = std::fs::read_to_string(folder.join("branches.txt")).expect("the branches");
    let mut not_ported = Vec::new();
    for line in lines.lines() {
        let f: Vec<&str> = line.splitn(4, '\t').collect();
        let (case, width, dest, today) = (f[0], f[1].parse::<i64>().expect("a width"), f[2], f[3]);
        let text = std::fs::read_to_string(folder.join(format!("branch_{case}.ir"))).expect("IR");
        let ir = parse_ir(&text);
        let made = row_program(&ir, width, width, 2 * width, dest, false);
        match (today, made) {
            ("made", Ok(unit)) => {
                let want = folder.join(format!("branch_{case}.rows.txt"));
                let want = std::fs::read_to_string(want).expect("today's unit");
                let got = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                assert!(got == want, "{case}: {}", first_difference(&want, &got));
            }
            (_, Err(e)) if e.refusal(None).code == "K964" => {
                not_ported.push(format!("{case}: {}", e.text()));
            }
            ("made", Err(e)) => panic!("{case}: {}", e.text()),
            (today, Err(e)) => {
                // today's text, but for the place it names (K962)
                assert_eq!(
                    e.text(),
                    today.split(" (at ").next().unwrap_or(today),
                    "{case}"
                );
                assert_eq!(e.refusal(None).code, "K962", "{case}");
            }
            (today, Ok(_)) => panic!("{case}: made, today refuses it: {today}"),
        }
    }
    assert_eq!(not_ported, Vec::<String>::new());
    // the levels refuse the band today
    let text = std::fs::read_to_string(folder.join("softmax_pair.ir")).expect("the IR");
    let banded = row_program(&parse_ir(&text), 2, 2, 4, "fp32", true).expect_err("band");
    let today = std::fs::read_to_string(folder.join("band.txt")).expect("today's band");
    let today = today.trim_end();
    assert_eq!(banded.text(), today.split(" (at ").next().unwrap_or(today));
    assert_eq!(banded.refusal(None).code, "K962");
}

/// A unit's math cost at 0 .. 33 live rows, a line each: the rows, the instructions and
/// the cycles as today writes them.
fn band_costs(unit: &UnitProgram) -> String {
    (0..34)
        .map(|r| {
            let cost = unit
                .cost(MATH, r)
                .unwrap_or_else(|e| panic!("{r} rows: {e}"));
            let cycles = cost.cycles.map_or("None".to_string(), float_repr);
            format!("{r}\t{}\t{cycles}\n", cost.instructions)
        })
        .collect()
}

#[test]
fn each_row_of_two_tiles_gets_todays_band() {
    let Some(folder) = today_rows() else { return };
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("the cases");
    let refused = std::fs::read_to_string(folder.join("band_refused.txt")).expect("refused");
    let (folder, refused) = (&folder, &refused);
    // each case on a thread of its own, the outcomes taken in the cases' order: none
    // (not this test's, or refused today with the same text), or the case made whole
    let same: Vec<&str> = std::thread::scope(|s| {
        let runs: Vec<_> = (cases.lines())
            .map(|line| {
                s.spawn(move || {
                    let f: Vec<&str> = line.split('\t').collect();
                    if f[1] != "2" {
                        return None;
                    }
                    let case = f[0];
                    let text =
                        std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
                    let made = row_program(&parse_ir(&text), 2, 2, 4, "fp32", true);
                    if let Some(today) = refused
                        .lines()
                        .find_map(|l| l.strip_prefix(&format!("{case}\t")))
                    {
                        // refused today: the same text, but for the place it names (K962)
                        let error = made.expect_err(case);
                        assert_eq!(
                            error.text(),
                            today.split(" (at ").next().unwrap_or(today),
                            "{case}"
                        );
                        assert_eq!(error.refusal(None).code, "K962", "{case}");
                        return None;
                    }
                    let mut unit = made.unwrap_or_else(|e| panic!("{case}: {}", e.text()));
                    let listing = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                    let costs = band_costs(&unit);
                    let words = emit(&mut unit).unwrap_or_else(|e| panic!("{case}: {e}"));
                    let emitted = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                    for (file, got) in [
                        ("band.txt", listing),
                        ("band", costs),
                        ("band.hpp", words),
                        ("band.emitted.txt", emitted),
                    ] {
                        let want = std::fs::read_to_string(folder.join(format!("{case}.{file}")))
                            .unwrap_or_else(|e| panic!("{case}.{file}: {e}"));
                        assert!(
                            got == want,
                            "{case}'s {file} differs, {}",
                            first_difference(&want, &got)
                        );
                    }
                    Some(case)
                })
            })
            .collect();
        (runs.into_iter())
            .filter_map(|r| r.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    // the row form of one reduction, with and without a resident row (f32 and bf16):
    // its runs of one band under `if`, the cost at each count of live rows
    assert_eq!(
        same,
        [
            "rms_norm",
            "centred",
            "row_min",
            "rms_plain",
            "peak",
            "weighted_bf16",
            "peak_square"
        ]
    );
    // two row tensors, the levels (12), lane flags (a sum of exponentials, a
    // reciprocal), more live values than the work registers: today's refusals (K962)
    assert_eq!(refused.lines().count(), 15);
}

#[test]
fn the_band_runs_are_todays_on_a_few_words() {
    let Some(folder) = today_rows() else { return };
    let lines = std::fs::read_to_string(folder.join("band_words.txt")).expect("the runs");
    let mut cases = 0;
    for line in lines.lines() {
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        let (case, offset, today) = (f[0], f[1].parse::<i64>().expect("an offset"), f[2]);
        let words =
            std::fs::read_to_string(folder.join(format!("band_{case}.in"))).expect("the words");
        let code: Vec<Insn> = words.lines().map(row_word).collect();
        cases += 1;
        let (items, done) = match band_groups(code, Layout::Rows(2, offset)) {
            Ok(made) => made,
            Err(e) => {
                assert_eq!(e.text(), today, "{case}");
                continue;
            }
        };
        assert_eq!(done.join("; "), today, "{case}");
        let mut got = String::new();
        for item in &items {
            match item {
                Item::Insn(x) => got += &format!("{}\n", row_line(x)),
                Item::Loop(l) => {
                    let Count::Runtime(r) = &l.count else {
                        panic!("{case}: a band of a count known at the build");
                    };
                    let passes: Vec<String> = (0..34).map(|n| (r.of)(n).to_string()).collect();
                    got += &format!("{} if {}: {}\n", l.name, r.text, passes.join(" "));
                    for x in &l.body {
                        let Item::Insn(x) = x else {
                            panic!("{case}: a band of more than words");
                        };
                        got += &format!("  {}\n", row_line(x));
                    }
                }
                _ => panic!("{case}: {item:?}"),
            }
        }
        let want =
            std::fs::read_to_string(folder.join(format!("band_{case}.out"))).expect("the runs");
        assert!(got == want, "{case}: {}", first_difference(&want, &got));
    }
    // an SFPNOP after a run's last MAD (none after the last run's), words before any
    // store's band, a value two stores need, the per-row slots (at an offset too), a
    // store past them, a word writing two values of two bands, lane flags refused (set
    // or pushed)
    assert_eq!(cases, 9);
}

#[test]
fn the_row_forms_checks_are_todays_on_a_few_words() {
    let Some(folder) = today_rows() else { return };
    let lines = std::fs::read_to_string(folder.join("checks.txt")).expect("the checks");
    let layouts = Layouts {
        copy: Layout::Tile(0),
        rows: vec![Layout::Rows(2, 0)],
        sums: Vec::new(),
        cols: vec![Layout::Cols(4)],
    };
    let mut cases = 0;
    for line in lines.lines() {
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        let (case, check, today) = (f[0], f[1], f[2]);
        let words =
            std::fs::read_to_string(folder.join(format!("check_{case}.in"))).expect("the words");
        let code: Vec<Insn> = words.lines().map(row_word).collect();
        let mut unit = UnitProgram::new("c", "fp32");
        let made = if check == "lanes" {
            check_lanes(&mut unit, &code, layouts.rows[0], None)
        } else {
            check_broadcast(&mut unit, &code, &layouts, 2, &[0])
        };
        let got = match made {
            Ok(()) => unit
                .checks
                .last()
                .map(|c| c.verdict.clone())
                .unwrap_or_default(),
            Err(e) => e.0,
        };
        assert_eq!(got, today, "{case}");
        cases += 1;
    }
    assert_eq!(cases, 12);
}

#[test]
fn a_row_forms_dst_layout_is_todays_handlers() {
    // at 2 tiles and in chunks, as today's handler lays out a call of the row form
    let Some(folder) = today_rows() else { return };
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("the cases");
    let mut rows = 0;
    for line in cases.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let case = f[0];
        let width: usize = f[1].parse().expect("a width");
        let fields: usize = f[2].parse().expect("the fields");
        let Ok(want) = std::fs::read_to_string(folder.join(format!("{case}.dst"))) else {
            continue;
        };
        let ir = std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
        let params = parse_ir(&ir).tensors.iter().filter(|t| !t.output).count();
        let resident: Vec<bool> = (0..params).map(|k| k + fields >= params).collect();
        assert_eq!(dst_text(&row_layout(&resident, width)), want, "{case}");
        rows += 1;
    }
    assert!(rows >= 20, "{rows}");
}
