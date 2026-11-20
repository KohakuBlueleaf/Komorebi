//! The tile compiler's row form over more than 2 tiles (the chunk form) against
//! today's, with no card: today's `tests/ir/tensix/tile/tile_rows_form.py` makes the
//! chunk form's unit of each wider row of its corpus and of a few branches from its
//! KernelIR, which the port reads back from today's text: each unit whole, its cost
//! and the fields its listing does not print.

use komorebi_ir::backend::tensix::tile::emit::emit;
use komorebi_ir::backend::tensix::tile::py::float_repr;
use komorebi_ir::backend::tensix::tile::rows::{chunk_cost, chunk_program};
use komorebi_ir::backend::tensix::tile::unit::UnitProgram;

#[allow(dead_code)] // each target takes a part of the shared helpers
#[path = "tile_common.rs"]
mod common;
use common::*;

/// What the handler reads of a chunk unit that its listing does not print, as the
/// script writes it.
fn chunk_fields(unit: &UnitProgram) -> String {
    let bases: Vec<String> = unit.chunk_results.iter().map(i64::to_string).collect();
    format!(
        "results\t{}\nchunk_results\t{}\nfields_in_sums\t{}\n",
        unit.results,
        bases.join(","),
        i64::from(unit.chunk_fields_in_sums)
    )
}

#[test]
fn each_wide_row_gets_todays_chunk_unit_and_cost_from_its_kernel_ir() {
    let Some(folder) = today_rows() else { return };
    let read = |file: &str| std::fs::read_to_string(folder.join(file)).expect(file);
    let cases = std::fs::read_to_string(folder.join("cases.txt")).expect("the cases");
    let refused = std::fs::read_to_string(folder.join("rows_refused.txt")).expect("refused");
    let costs = std::fs::read_to_string(folder.join("chunk_costs.txt")).expect("the costs");
    let (mut same, mut not_ported) = (Vec::new(), Vec::new());
    // each case on a thread of its own (the identities are long in debug), the
    // outcomes taken in the cases' order: none (not this test's), the case made
    // whole, or the text of a refusal by name
    let (folder, refused, costs, read) = (&folder, &refused, &costs, &read);
    let outcomes: Vec<Option<Result<&str, String>>> = std::thread::scope(|s| {
        let runs: Vec<_> = (cases.lines())
            .map(|line| {
                s.spawn(move || {
                    let f: Vec<&str> = line.split('\t').collect();
                    let (case, width) = (f[0], f[1].parse::<i64>().expect("a width"));
                    let fields = f[2].parse::<i64>().expect("the fields");
                    if width <= 2 {
                        return None;
                    }
                    let text =
                        std::fs::read_to_string(folder.join(format!("{case}.ir"))).expect("the IR");
                    let ir = parse_ir(&text);
                    assert_eq!(ir.dump() + "\n", text, "{case}: the KernelIR read back");
                    let made = chunk_program(&ir, width);
                    if let Some(today) = refused
                        .lines()
                        .find_map(|l| l.strip_prefix(&format!("{case}\t")))
                    {
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
                    let (count, cycles, copies) =
                        chunk_cost(&unit, width, fields).unwrap_or_else(|e| panic!("{case}: {e}"));
                    let cost = format!(
                        "{case}\t{count}\t{}\t{}",
                        float_repr(cycles),
                        float_repr(copies)
                    );
                    assert!(costs.lines().any(|l| l == cost), "{case}: {cost}");
                    assert_eq!(
                        chunk_fields(&unit),
                        read(&format!("{case}.chunk.fields")),
                        "{case}"
                    );
                    let listing = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                    let words = emit(&mut unit).unwrap_or_else(|e| panic!("{case}: {e}"));
                    let emitted = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                    for (file, got) in [("txt", listing), ("hpp", words), ("emitted.txt", emitted)]
                    {
                        let want =
                            std::fs::read_to_string(folder.join(format!("{case}.chunk.{file}")))
                                .unwrap_or_else(|e| panic!("{case}.chunk.{file}: {e}"));
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
    // rms_norm at 3, 4 and 8 tiles (its weight in bf16 at 3), softmax_rows at 3 and 5
    // (an odd row's last chunk), a row's largest value, layer_norm's two levels and two
    // resident rows, a sum reading .row() (and with another sum that does not)
    assert_eq!(
        same,
        [
            "rms_norm_wide",
            "rms_bf16",
            "softmax_rows",
            "row_max",
            "layer_norm",
            "rms_eight",
            "softmax_five",
            "wide_weighted",
            "wide_mixed"
        ]
    );
    // a largest and a smallest value together: today starts them in a set's order
    assert_eq!(
        not_ported,
        [
            "extremes: not ported: the chunk form's start values of a largest and a smallest \
          value together (today makes them in a set's order)"
        ]
    );
    let lines = std::fs::read_to_string(folder.join("chunk_branches.txt")).expect("branches");
    let mut verdicts = 0;
    for line in lines.lines() {
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        let (case, width, today) = (f[0], f[1].parse::<i64>().expect("a width"), f[2]);
        let text = std::fs::read_to_string(folder.join(format!("chunk_{case}.ir"))).expect("IR");
        match (today, chunk_program(&parse_ir(&text), width)) {
            ("made", Ok(unit)) => {
                let want = folder.join(format!("chunk_{case}.chunk.txt"));
                let want = std::fs::read_to_string(want).expect("today's unit");
                let got = unit.dump().unwrap_or_else(|e| panic!("{case}: {e}")) + "\n";
                assert!(got == want, "{case}: {}", first_difference(&want, &got));
                let want = read(&format!("chunk_{case}.chunk.fields"));
                assert_eq!(chunk_fields(&unit), want, "{case}");
            }
            ("made", Err(e)) => panic!("{case}: {}", e.text()),
            (today, Err(e)) => {
                assert_eq!(
                    e.text(),
                    today.split(" (at ").next().unwrap_or(today),
                    "{case}"
                );
                assert_eq!(e.refusal(None).code, "K962", "{case}");
            }
            (today, Ok(_)) => panic!("{case}: made, today refuses it: {today}"),
        }
        verdicts += 1;
    }
    // a row of 2, two results, three (more than Dst holds), five sums, five resident
    // rows, a per-row value alone (part `value`)
    assert_eq!(verdicts, 6);
}
