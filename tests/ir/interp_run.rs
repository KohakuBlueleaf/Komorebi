//! The interpreter running the lessons' handlers on small tensors, each result checked
//! against numbers written in the test; the failures by name, with their rank and
//! instruction; and one run made twice, which gives the same report.

use komorebi_ir::derive::derive;
use komorebi_ir::interp::control::{Arg, End, Machine, Options, Refusal};
use komorebi_ir::interp::memory::{Failure, Format, Scalar, Tensor, TensorId, Tile};
use komorebi_ir::lower::lower;
use komorebi_ir::program::Module;

fn lesson(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/ir/corpus/golden/lessons")
        .join(format!("{name}.kom"));
    std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{} is readable", path.display()))
}

fn module(text: &str) -> Module {
    let checked = komorebi_ir::check(text).unwrap_or_else(|d| panic!("{d:?}"));
    let mut m = lower(&checked).unwrap_or_else(|e| panic!("{e:?}"));
    derive(&mut m);
    m
}

/// A module whose handlers are invalid on purpose, built by the checker without
/// Verify, then lowered and derived: the interpreter must still catch at the run what
/// the build cannot always show, so its test runs what program compile refuses.
fn module_unverified(text: &str) -> Module {
    let parsed = komorebi_ir::parse(text).unwrap_or_else(|d| panic!("{d:?}"));
    let checked =
        komorebi_ir::check::check_module(text, parsed).unwrap_or_else(|d| panic!("{d:?}"));
    let mut m = lower(&checked).unwrap_or_else(|e| panic!("{e:?}"));
    derive(&mut m);
    m
}

fn u32s(v: u64) -> Scalar {
    Scalar::Int(Format::U32, v)
}

/// A tensor of `rows` by `cols` tiles of the module's tile, tile `i` (row after row of
/// tiles) holding `value(i, k)` at element k.
fn tensor(
    m: &mut Machine<'_>,
    format: Format,
    (rows, cols): (usize, usize),
    value: impl Fn(usize, usize) -> f64,
) -> TensorId {
    let shape = m.module().tile;
    let (h, w) = (shape.rows as usize, shape.cols as usize);
    let mut t = Tensor::zeros("t", format, rows * h, cols * w, shape);
    for (i, tile) in t.tiles.iter_mut().enumerate() {
        let values: Vec<f64> = (0..h * w).map(|k| value(i, k)).collect();
        *tile = if format.is_integer() {
            let ints: Vec<i128> = values.iter().map(|v| *v as i128).collect();
            Tile::from_ints(format, shape, &ints).expect("a tile of integers")
        } else {
            Tile::from_f64(format, shape, &values).expect("a tile of floats")
        };
    }
    m.tensor(t)
}

fn tile_f64(m: &Machine<'_>, t: TensorId, i: usize) -> Vec<f64> {
    m.heap.tensors[t.0].tiles[i].to_f64().expect("a float tile")
}

fn tile_ints(m: &Machine<'_>, t: TensorId, i: usize) -> Vec<i128> {
    m.heap.tensors[t.0].tiles[i]
        .to_ints()
        .expect("an integer tile")
}

fn all(values: &[f64], want: f64) -> bool {
    values
        .iter()
        .all(|v| (v - want).abs() <= 1e-6 * want.abs().max(1.0))
}

#[test]
fn t1_scales_every_tile() {
    let m = module(&lesson("t01_scale"));
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::Bf16, (5, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::Bf16, (5, 1), |_, _| 0.0);
    let args = [
        Arg::Tensor(x),
        Arg::Tensor(y),
        Arg::Scalar(u32s(0)),
        Arg::Scalar(u32s(5)),
        Arg::Scalar(Scalar::F32(Format::F32, 2.0)),
    ];
    assert_eq!(
        run.invoke("Scale.run", &args, (0, 3), &[]),
        End::Completed(None)
    );
    for i in 0..5 {
        assert!(all(&tile_f64(&run, y, i), 2.0 * (i + 1) as f64), "tile {i}");
    }
}

#[test]
fn t2_counts_blocks_and_binds_an_invocation_parameter() {
    let m = module(&lesson("t02_count"));
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::U32, (20, 1), |_, k| k as f64);
    let args = [Arg::Tensor(x), Arg::Scalar(u32s(0)), Arg::Scalar(u32s(20))];
    // five tiles a rank: one block of eight, ragged; three blocks of two
    assert_eq!(
        run.invoke("Count.blocks", &args, (0, 4), &[]),
        End::Completed(Some(u32s(1)))
    );
    assert_eq!(
        run.invoke("Count.blocks", &args, (0, 4), &[("BLOCK", u32s(2))]),
        End::Completed(Some(u32s(3)))
    );
    assert_eq!(
        run.invoke("Count.blocks", &args, (0, 4), &[("BLOCK", u32s(9))]),
        End::Failed {
            reason: Failure::OverBound,
            at: None
        }
    );
    // words 0 to 15 of tile 0, rotated and folded: 2295
    let args = [Arg::Tensor(x), Arg::Scalar(u32s(0))];
    assert_eq!(
        run.invoke("Count.checksum", &args, (0, 4), &[]),
        End::Completed(Some(u32s(2295)))
    );
}

#[test]
fn t3_state_stays_between_invocations_and_empty_state_fails() {
    let m = module(&lesson("t03_gain"));
    let mut run = Machine::load(&m, 4, &[("ROOM", u32s(2))], Options::default()).expect("loads");
    let g = tensor(&mut run, Format::Bf16, (2, 1), |i, _| [3.0, 5.0][i]);
    let x = tensor(&mut run, Format::Bf16, (4, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::Bf16, (4, 1), |_, _| 0.0);
    let fill = [Arg::Tensor(g), Arg::Scalar(u32s(0))];
    assert_eq!(
        run.invoke("Gain.fill", &fill, (0, 2), &[]),
        End::Completed(None)
    );
    let apply = [
        Arg::Tensor(x),
        Arg::Tensor(y),
        Arg::Scalar(u32s(0)),
        Arg::Scalar(u32s(4)),
        Arg::Scalar(u32s(1)),
    ];
    assert_eq!(
        run.invoke("Gain.apply", &apply, (0, 2), &[]),
        End::Completed(None)
    );
    for i in 0..4 {
        assert!(all(&tile_f64(&run, y, i), 5.0 * (i + 1) as f64), "tile {i}");
    }
    // `calls` of worker 0: bumped once, then the tiles written
    let calls = &run.workers[0].state[0][1];
    let komorebi_ir::interp::memory::MemberData::Scalars(v) = &calls.data else {
        panic!("calls holds scalars");
    };
    assert_eq!(v[..2], [u32s(1), u32s(4)]);
    // workers 2 and 3 were never filled: the lowest of them fails, at the `bump`
    let end = run.invoke("Gain.apply", &apply, (0, 4), &[]);
    let End::Failed {
        reason: Failure::EmptyState,
        at: Some((2, _, _)),
    } = end
    else {
        panic!("{end:?}");
    };
}

#[test]
fn t4_a_function_runs_on_every_pair_of_blocks() {
    let m = module(&lesson("t04_compute"));
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let a = tensor(&mut run, Format::F32, (4, 1), |_, _| 1.0);
    let b = tensor(&mut run, Format::F32, (4, 1), |_, _| 2.0);
    let out = tensor(&mut run, Format::F32, (2, 1), |_, _| 0.0);
    let args = [
        Arg::Tensor(a),
        Arg::Tensor(b),
        Arg::Tensor(out),
        Arg::Scalar(u32s(0)),
        Arg::Scalar(u32s(4)),
    ];
    assert_eq!(
        run.invoke("Compute.gated", &args, (0, 2), &[]),
        End::Completed(None)
    );
    // silu(1) * 2 = 1.4621171572600098, two blocks a rank
    for rank in 0..2 {
        assert!(all(
            &tile_f64(&run, out, rank),
            2.0 * 1.462_117_157_260_009_8
        ));
    }
}

#[test]
fn t7_rectangles_turned_walks_records_and_items() {
    let m = module(&lesson("t07_views"));
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let image = tensor(&mut run, Format::Bf16, (8, 8), |i, _| i as f64);
    let out = tensor(&mut run, Format::Bf16, (16, 1), |_, _| 0.0);
    let crop = [
        Arg::Tensor(image),
        Arg::Tensor(out),
        Arg::Scalar(u32s(2)),
        Arg::Scalar(u32s(1)),
    ];
    assert_eq!(
        run.invoke("Views.crop", &crop, (0, 4), &[]),
        End::Completed(None)
    );
    for k in 0..16 {
        let want = ((2 + k / 4) * 8 + 1 + k % 4) as f64;
        assert!(all(&tile_f64(&run, out, k), want), "crop tile {k}");
    }
    let turn = [Arg::Tensor(image), Arg::Tensor(out)];
    assert_eq!(
        run.invoke("Views.turn", &turn, (0, 4), &[]),
        End::Completed(None)
    );
    for k in 0..16 {
        let want = ((k % 4) * 8 + k / 4) as f64;
        assert!(all(&tile_f64(&run, out, k), want), "turned tile {k}");
    }
    // record e of the table's tile 0 is words 3e, 3e + 1, 3e + 2
    let table = tensor(&mut run, Format::U32, (1, 1), |_, k| k as f64);
    let results = tensor(&mut run, Format::U32, (1, 1), |_, _| 0.0);
    let lookup = [
        Arg::Tensor(table),
        Arg::Tensor(results),
        Arg::Scalar(u32s(5)),
    ];
    assert_eq!(
        run.invoke("Views.lookup", &lookup, (0, 3), &[]),
        End::Completed(None)
    );
    let words = tile_ints(&run, results, 0);
    for rank in 0..3 {
        assert_eq!(
            words[4 * rank..4 * rank + 4],
            [15, 16, 17, rank as i128],
            "item {rank}"
        );
    }
}

#[test]
fn t8_needs_an_implementation_the_interpreter_does_not_have() {
    let m = module(&lesson("t08_trace"));
    let refused = Machine::load(&m, 1, &[], Options::default()).err();
    assert_eq!(
        refused,
        Some(Refusal::Requirement(
            "impl(\"kohaku.accel.ray_box\")".into()
        ))
    );
}

const FALLBACK: &str = "module fallback

tile [32, 32]
format f32

extern func smooth(a: tile<f32>) -> tile<f32> @impl(\"vendor.smooth\") fallback {
    return a * 0.5 + abs(a) * 0.5
}

program P {
    handler run(x: tensor<f32>, y: tensor<f32>) @on(group) {
        store y.tiles(worker.rank, 1) <- smooth(x.tiles(worker.rank, 1)[0])
    }
}
";

#[test]
fn t8_an_external_function_runs_its_fallback() {
    let m = module(FALLBACK);
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (2, 1), |i, _| [-2.0, 3.0][i]);
    let y = tensor(&mut run, Format::F32, (2, 1), |_, _| 9.0);
    assert_eq!(
        run.invoke("P.run", &[Arg::Tensor(x), Arg::Tensor(y)], (0, 2), &[]),
        End::Completed(None)
    );
    assert!(all(&tile_f64(&run, y, 0), 0.0));
    assert!(all(&tile_f64(&run, y, 1), 3.0));
}

/// T3's two invocations and the failing third, as a report.
fn gain_report() -> String {
    let m = module(&lesson("t03_gain"));
    let mut run = Machine::load(&m, 4, &[("ROOM", u32s(2))], Options::default()).expect("loads");
    let g = tensor(&mut run, Format::Bf16, (2, 1), |i, _| [3.0, 5.0][i]);
    let x = tensor(&mut run, Format::Bf16, (4, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::Bf16, (4, 1), |_, _| 0.0);
    run.invoke(
        "Gain.fill",
        &[Arg::Tensor(g), Arg::Scalar(u32s(0))],
        (0, 2),
        &[],
    );
    let apply = [
        Arg::Tensor(x),
        Arg::Tensor(y),
        Arg::Scalar(u32s(0)),
        Arg::Scalar(u32s(4)),
        Arg::Scalar(u32s(1)),
    ];
    run.invoke("Gain.apply", &apply, (0, 2), &[]);
    run.invoke("Gain.apply", &apply, (0, 4), &[]);
    run.invoke("Gain.apply", &apply, (2, 4), &[]);
    run.report.print()
}

#[test]
fn the_report_has_one_printed_form() {
    assert_eq!(
        gain_report(),
        "#0 Gain.fill group(0, 2): completed\n\
         #1 Gain.apply group(0, 2): completed\n\
         #2 Gain.apply group(0, 4): failed empty_state at rank 2, p0.h1 i6\n\
         #3 Gain.apply group(2, 4): failed capacity\n"
    );
}

#[test]
fn the_same_run_twice_gives_the_same_report() {
    assert_eq!(gain_report(), gain_report());
}

#[test]
fn a_loop_past_the_step_budget_gives_up_by_name() {
    let text = "module spin

tile [32, 32]
format u32

program P {
    handler spin() -> u32 @on(group) {
        var n: u32 = 0
        for i in range(4000000000) {
            n = n ^ i
        }
        return n
    }
}
";
    let m = module(text);
    let options = Options {
        step_budget: 1000,
        ..Options::default()
    };
    let mut run = Machine::load(&m, 1, &[], options).expect("loads");
    let End::Failed {
        reason: Failure::GivenUp,
        at: Some((0, _, _)),
    } = run.invoke("P.spin", &[], (0, 1), &[])
    else {
        panic!("gives up");
    };
}

fn t5_args(x: TensorId, y: TensorId) -> [Arg; 4] {
    [
        Arg::Tensor(x),
        Arg::Tensor(y),
        Arg::Scalar(u32s(0)),
        Arg::Scalar(u32s(10)),
    ]
}

#[test]
fn t5_four_ways_to_combine_over_the_group() {
    let m = module(&lesson("t05_total"));
    let mut run = Machine::load(&m, 8, &[], Options::default()).expect("loads");
    // tile i holds i + 1: ranks 0 and 1 take two tiles, the others one
    let x = tensor(&mut run, Format::F32, (10, 1), |i, _| (i + 1) as f64);
    for handler in ["Total.any_order", "Total.by_fours", "Total.rank_order"] {
        let y = tensor(&mut run, Format::F32, (1, 1), |_, _| 0.0);
        assert_eq!(
            run.invoke(handler, &t5_args(x, y), (0, 8), &[]),
            End::Completed(None)
        );
        assert!(all(&tile_f64(&run, y, 0), 55.0), "{handler}");
    }
    // the largest tile holds 10: every tile scaled by 1/10
    let y = tensor(&mut run, Format::F32, (10, 1), |_, _| 0.0);
    assert_eq!(
        run.invoke("Total.by_peak", &t5_args(x, y), (0, 8), &[]),
        End::Completed(None)
    );
    for i in 0..10 {
        assert!(
            all(&tile_f64(&run, y, i), (i + 1) as f64 / 10.0),
            "tile {i}"
        );
    }
}

const ORDERED: &str = "module ordered

tile [32, 32]
format f32

combine twice_then(a: tile<f32>, b: tile<f32>) -> tile<f32> {
    return a * 2.0 + b
}

program P {
    handler fold(x: tensor<f32>, y: tensor<f32>) @on(group) {
        let whole = reduce(x.tiles(worker.rank, 1)[0], with = twice_then)
        if worker.is_leader {
            store y.tiles(0, 1) <- whole
        }
    }

    handler leader_only(x: tensor<f32>, y: tensor<f32>) @on(group) {
        if worker.is_leader {
            let whole = reduce(x.tiles(0, 1)[0], with = sum)
            store y.tiles(0, 1) <- whole
        }
    }

    handler everyone_reads(x: tensor<f32>, y: tensor<f32>) @on(group) {
        let whole = reduce(x.tiles(worker.rank, 1)[0], with = sum)
        store y.tiles(worker.rank, 1) <- whole
    }
}
";

#[test]
fn a_combine_that_is_not_commutative_folds_from_the_lowest_rank() {
    let m = module_unverified(ORDERED);
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (4, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::F32, (4, 1), |_, _| 0.0);
    let args = [Arg::Tensor(x), Arg::Tensor(y)];
    assert_eq!(
        run.invoke("P.fold", &args, (0, 4), &[]),
        End::Completed(None)
    );
    // ((1 * 2 + 2) * 2 + 3) * 2 + 4
    assert!(all(&tile_f64(&run, y, 0), 26.0));
    // a collective only the leader reaches never completes
    let End::Invalid {
        at: Some((0, _, _)),
        ..
    } = run.invoke("P.leader_only", &args, (0, 4), &[])
    else {
        panic!("a collective not every member reached");
    };
    // a rank that is not the leader holds no value of a `reduce`: using it is invalid
    let End::Invalid {
        at: Some((1, _, _)),
        ..
    } = run.invoke("P.everyone_reads", &args, (0, 4), &[])
    else {
        panic!("a use of a value the rank was not given");
    };
}

#[test]
fn t6_each_row_receives_its_senders_tiles() {
    let m = module(&lesson("t06_rows"));
    let mut run = Machine::load(&m, 32, &[], Options::default()).expect("loads");
    // two tiles a row: row r sends tiles 2r and 2r + 1, which hold 2r + 1 and 2r + 2
    let x = tensor(&mut run, Format::Bf16, (8, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::Bf16, (64, 1), |_, _| 0.0);
    let args = [Arg::Tensor(x), Arg::Tensor(y), Arg::Scalar(u32s(2))];
    assert_eq!(
        run.invoke("Rows.spread", &args, (0, 32), &[]),
        End::Completed(None)
    );
    for row in 0..4 {
        for col in 0..8 {
            for k in 0..2 {
                // the sender, in column 0, writes nothing
                let want = if col == 0 {
                    0.0
                } else {
                    (row * 2 + k + 1) as f64
                };
                let i = (row * 8 + col) * 2 + k;
                assert!(
                    all(&tile_f64(&run, y, i), want),
                    "row {row} col {col} tile {k}"
                );
            }
        }
    }
}

const LINES: &str = "module lines

tile [32, 32]
format f32

requires {
    group = grid
}

program P {
    handler twice(x: tensor<f32>, y: tensor<f32>) @on(grid[1, 4]) {
        for i in range(2) {
            let b = broadcast(along = row, sender = 0)
            if b.is_sender {
                b.cast(x.tiles(i, 1)[0], word = i)
            } else {
                for t, w in b.arrive() {
                    store y.tiles(worker.rank * 2 + w, 1) <- t
                }
            }
        }
    }

    handler crossed(x: tensor<f32>, y: tensor<f32>) @on(grid[1, 2]) {
        let b = broadcast(along = row, sender = 0)
        let c = broadcast(along = row, sender = 1)
        if b.is_sender {
            for t, w in c.arrive() {
                b.cast(x.tiles(0, 1)[0], word = w)
            }
        } else {
            for t, w in b.arrive() {
                c.cast(x.tiles(1, 1)[0], word = w)
            }
        }
    }

    handler far(x: tensor<f32>, y: tensor<f32>) @on(grid[1, 2]) {
        let b = broadcast(along = row, sender = 2)
        if b.is_sender {
            b.cast(x.tiles(0, 1)[0])
        }
    }
}
";

#[test]
fn a_line_ends_where_its_sender_leaves_the_block_that_made_it() {
    let m = module(LINES);
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (2, 1), |i, _| (i + 1) as f64 * 10.0);
    let y = tensor(&mut run, Format::F32, (8, 1), |_, _| 0.0);
    let args = [Arg::Tensor(x), Arg::Tensor(y)];
    // each pass of the loop makes a new line, which ends with the pass
    assert_eq!(
        run.invoke("P.twice", &args, (0, 4), &[]),
        End::Completed(None)
    );
    for rank in 0..4 {
        for i in 0..2 {
            let want = if rank == 0 {
                0.0
            } else {
                (i + 1) as f64 * 10.0
            };
            assert!(
                all(&tile_f64(&run, y, rank * 2 + i), want),
                "rank {rank} pass {i}"
            );
        }
    }
    // each waits at `arrive` on the line the other sends on, which ends only with
    // the other's body: the lowest rank is reported
    let End::Invalid {
        what,
        at: Some((0, _, _)),
    } = run.invoke("P.crossed", &args, (0, 2), &[])
    else {
        panic!("two arrives that wait on each other");
    };
    assert!(what.contains("arrive"), "{what}");
    // a row of two columns has no column 2
    let End::Failed {
        reason: Failure::OutOfRange,
        at: Some((0, _, _)),
    } = run.invoke("P.far", &args, (0, 2), &[])
    else {
        panic!("a sender outside the line");
    };
}

#[test]
fn t11_a_control_program_decides_from_a_result() {
    let m = module(&lesson("t11_solve"));
    let mut run = Machine::load(&m, 60, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::Bf16, (64, 1), |i, _| (i + 1) as f64);
    // the first norm is the leader's count of blocks: 64 tiles on 60 ranks give rank
    // 0 two, below 64, so one round
    let args = [Arg::Tensor(x), Arg::Scalar(u32s(5))];
    assert_eq!(run.control("refine", &args), End::Completed(Some(u32s(1))));
    for i in 0..64 {
        assert!(all(&tile_f64(&run, x, i), (i + 1) as f64 / 2.0), "tile {i}");
    }
    // no round at all
    let args = [Arg::Tensor(x), Arg::Scalar(u32s(0))];
    assert_eq!(run.control("refine", &args), End::Completed(Some(u32s(0))));
    assert_eq!(
        run.report.print(),
        "#0 Solver.step group(0, 60): completed\n\
         #1 Solver.norm group(0, 60): completed 2:u32\n\
         #2 control refine: completed 1:u32\n\
         #3 control refine: completed 0:u32\n"
    );
}

#[test]
fn t16_a_failed_step_is_tidied_and_retried_smaller() {
    let m = module(&lesson("t16_retry"));
    let mut run = Machine::load(&m, 8, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (64, 1), |i, _| (i + 1) as f64);
    let s = tensor(&mut run, Format::F32, (64, 1), |_, _| 7.0);
    // 4096 tiles of a tensor of 64: the view ends past the tensor; the retry copies
    // 64 tiles, 8 for each rank
    let end = run.control("careful", &[Arg::Tensor(x), Arg::Tensor(s)]);
    assert_eq!(end, End::Completed(Some(u32s(8))));
    for i in 0..64 {
        assert!(all(&tile_f64(&run, s, i), (i + 1) as f64), "tile {i}");
    }
    let lines: Vec<String> = run.report.print().lines().map(String::from).collect();
    assert_eq!(lines.len(), 4);
    assert!(lines[0] == "#0 Work.step group(0, 8): failed view_end at rank 0, p0.h0 i4");
    assert_eq!(
        lines[1..],
        [
            "#1 Work.tidy group(0, 8): completed",
            "#2 Work.step group(0, 8): completed 8:u32",
            "#3 control careful: completed 8:u32",
        ]
    );
}

const AFTER: &str = "module after

tile [32, 32]
format f32

program W {
    handler take(x: tensor<f32>, n: u32) -> u32 @on(group) {
        let v = x.tiles(0, n)
        return 1
    }
}

control strict(x: tensor<f32>) -> u32 {
    let a = invoke(W.take(x, 100), on = group(0, 2))
    let b = invoke(W.take(x, 1), on = group(0, 2), after = [a])
    let c = invoke(W.take(x, 1), on = group(2))
    return await b
}

control reads(x: tensor<f32>) -> u32 {
    let a = invoke(W.take(x, 100), on = group(0, 2))
    let o = outcome(a)
    if o.failed {
        if o.reason == view_end {
            return 7
        }
    }
    return 0
}
";

#[test]
fn an_after_on_a_failed_invocation_fails_by_dependency() {
    let m = module(AFTER);
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (4, 1), |_, _| 1.0);
    // `await` of the failed `b` fails the program; `c`, made before, still runs
    let End::Failed {
        reason: Failure::Dependency,
        at: Some((0, _, _)),
    } = run.control("strict", &[Arg::Tensor(x)])
    else {
        panic!("a dependency failure");
    };
    assert_eq!(
        run.control("reads", &[Arg::Tensor(x)]),
        End::Completed(Some(u32s(7)))
    );
    let printed = run.report.print();
    let lines: Vec<&str> = printed.lines().collect();
    assert!(
        lines[0].starts_with("#0 W.take group(0, 2): failed view_end at rank 0"),
        "{printed}"
    );
    assert_eq!(lines[1], "#1 W.take group(0, 2): failed dependency");
    assert!(
        lines[2].starts_with("#2 control strict: failed dependency at rank 0, c0 i"),
        "{printed}"
    );
    assert_eq!(lines[3], "#3 W.take group(0, 2): completed 1:u32");
    assert!(
        lines[4].starts_with("#4 W.take group(0, 2): failed view_end"),
        "{printed}"
    );
    assert_eq!(lines[5], "#5 control reads: completed 7:u32");
}

#[test]
fn t10_a_chain_runs_its_calls_in_order() {
    let m = module(&lesson("t10_steps"));
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::Bf16, (8, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::Bf16, (8, 1), |_, _| 0.0);
    let z = tensor(&mut run, Format::Bf16, (8, 1), |_, _| 10.0);
    let s = tensor(&mut run, Format::Bf16, (8, 1), |_, _| 0.0);
    let t = tensor(&mut run, Format::Bf16, (1, 1), |_, _| 0.0);
    let args = |n: u64| {
        [
            Arg::Tensor(x),
            Arg::Tensor(y),
            Arg::Tensor(z),
            Arg::Tensor(s),
            Arg::Tensor(t),
            Arg::Scalar(Scalar::F32(Format::F32, 2.0)),
            Arg::Scalar(u32s(n)),
        ]
    };
    assert_eq!(
        run.invoke("Steps.steps", &args(8), (0, 4), &[]),
        End::Completed(None)
    );
    for i in 0..8 {
        // y = 2x; s = z + y
        assert!(all(&tile_f64(&run, y, i), 2.0 * (i + 1) as f64), "y {i}");
        assert!(
            all(&tile_f64(&run, s, i), 10.0 + 2.0 * (i + 1) as f64),
            "s {i}"
        );
    }
    // the sum of s: 8 * 10 + 2 * (1 + ... + 8)
    assert!(all(&tile_f64(&run, t, 0), 152.0));
    // 16 tiles of tensors of 8: the first call fails, and the others do not run
    let t2 = tensor(&mut run, Format::Bf16, (1, 1), |_, _| 0.0);
    let mut late = args(16);
    late[4] = Arg::Tensor(t2);
    let End::Failed {
        reason: Failure::ViewEnd,
        at: Some((0, _, _)),
    } = run.invoke("Steps.steps", &late, (0, 4), &[])
    else {
        panic!("the first call ends past the tensor");
    };
    assert!(all(&tile_f64(&run, t2, 0), 0.0));
    let printed = run.report.print();
    let lines: Vec<&str> = printed.lines().collect();
    assert_eq!(lines[0], "#0 Steps.steps group(0, 4): completed");
    // i5: `y.tiles(0, n)`, the first view the first call makes
    assert_eq!(
        lines[1],
        "#1 Steps.steps group(0, 4): failed view_end at rank 0, p0.h0 i5"
    );
}

const PAST: &str = "module past

tile [32, 32]
format f32

program P {
    handler copy(x: tensor<f32>, y: tensor<f32>, n: u32) @on(group) {
        let out = sink(partition(y.tiles(0, n), worker.rank, group.size), block = 1)
        for xb in stream(partition(x.tiles(0, n), worker.rank, group.size), block = 1) {
            out.push(xb)
        }
    }

    chain short(x: tensor<f32>, y: tensor<f32>, s: tensor<f32>) @on(group) {
        call copy(x, y, 4)
        call copy(past(y), s, 8)
    }

    chain enough(x: tensor<f32>, y: tensor<f32>, s: tensor<f32>) @on(group) {
        call copy(x, y, 8)
        call copy(past(y), s, 8)
    }
}
";

#[test]
fn a_past_that_reads_what_no_earlier_call_opened_is_invalid() {
    // the build refuses it; the run, given it anyway, finds it
    let codes: Vec<&str> = komorebi_ir::check(PAST)
        .err()
        .unwrap_or_default()
        .iter()
        .map(|d| d.code)
        .collect();
    assert_eq!(codes, ["K601"]);
    let m = module_unverified(PAST);
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (8, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::F32, (8, 1), |_, _| 0.0);
    let s = tensor(&mut run, Format::F32, (8, 1), |_, _| 0.0);
    let args = [Arg::Tensor(x), Arg::Tensor(y), Arg::Tensor(s)];
    assert_eq!(
        run.invoke("P.enough", &args, (0, 2), &[]),
        End::Completed(None)
    );
    assert_eq!(
        run.invoke("P.short", &args, (0, 2), &[]),
        End::Invalid {
            what: "call 1 reads tile 4 of `past` parameter 1, which no earlier call opened \
                   for writing"
                .into(),
            at: None
        }
    );
}
