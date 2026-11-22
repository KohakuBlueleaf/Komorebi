//! The interpreter running the lessons' handlers on small tensors, each result checked
//! against numbers written in the test; the failures by name, with their rank and
//! instruction; and one run made twice, which gives the same report.

use komorebi_ir::derive::derive;
use komorebi_ir::interp::control::{Arg, End, Machine, Options, Refusal};
use komorebi_ir::interp::memory::{Failure, Format, Scalar, Tensor, TensorId, Tile, TileShape};
use komorebi_ir::lower::lower;
use komorebi_ir::program::body::InstId;
use komorebi_ir::program::{BodyRef, Module};
use komorebi_ir::types::{Count, HandlerId, ParamId, ProgramId, RunId, Type};
use komorebi_ir::types::{IslandGraph, IslandRefusal};

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
    let shape = m.module().default_tile();
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

/// A handler that counts the blocks of a stream of `x`'s tiles `[0:2, 0:2]` in blocks
/// of `[2, 2]`, with `R` bound at load beside it.
const ROWS: &str = "module rows

tile [32, 32]
format f32

param R: u32 = 2 @bind(load)

program Rows {
    handler blocks(x: tensor<f32>) -> u32 @on(group) {
        var walked: u32 = 0
        for xb in stream(x.tiles[0:2, 0:2], block = [2, 2]) {
            walked += 1
        }
        return walked
    }
}
";

#[test]
fn a_streams_rows_bound_at_load_are_the_parameters_value() {
    // program compile refuses rows bound at load in the text (K102), so the stream's
    // type is given the parameter `R` by hand in the program structure
    let mut m = module(ROWS);
    let mut turned = 0;
    for ty in &mut m.programs[0].handlers[0].body.values {
        if let Type::Stream(_, rows, _) = ty {
            *rows = Count::Param(ParamId(0));
            turned += 1;
        }
    }
    assert_eq!(turned, 1);
    let mut run = Machine::load(&m, 1, &[("R", u32s(2))], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (2, 2), |i, _| i as f64);
    // one block of 2 rows by 2 tiles; rows taken as 1 would give two blocks
    assert_eq!(
        run.invoke("Rows.blocks", &[Arg::Tensor(x)], (0, 1), &[]),
        End::Completed(Some(u32s(1)))
    );
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

const PAST: &str = "module `past`

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

    // each rank copies the part of its neighbour
    handler shift(y: tensor<f32>, s: tensor<f32>, n: u32) @on(group) {
        let next = (worker.rank + 1) % group.size
        let out = sink(partition(s.tiles(0, n), worker.rank, group.size), block = 1)
        for yb in stream(partition(y.tiles(0, n), next, group.size), block = 1) {
            out.push(yb)
        }
    }

    chain crossed(x: tensor<f32>, y: tensor<f32>, s: tensor<f32>) @on(group) {
        call copy(x, y, 8)
        call shift(past(y), s, 8)
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
    // `short` and `crossed`
    assert_eq!(codes, ["K601", "K601"]);
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
            // rank 0 opened tiles 0 and 1 (its half of 4) and reads 0 to 3 (its half of 8)
            what: "call 1 reads at rank 0 tile 2 of `past` parameter 1, which no earlier \
                   call opened for writing at that rank"
                .into(),
            at: None
        }
    );
}

#[test]
fn a_past_is_proven_worker_by_worker() {
    let m = module_unverified(PAST);
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (8, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::F32, (8, 1), |_, _| 0.0);
    let s = tensor(&mut run, Format::F32, (8, 1), |_, _| 0.0);
    let args = [Arg::Tensor(x), Arg::Tensor(y), Arg::Tensor(s)];
    // the first call opens all 8 tiles of y between the two ranks, but each reads
    // its neighbour's half: rank 0 opened 0 to 3 and reads 4 to 7
    assert_eq!(
        run.invoke("P.crossed", &args, (0, 2), &[]),
        End::Invalid {
            what: "call 1 reads at rank 0 tile 4 of `past` parameter 1, which no earlier \
                   call opened for writing at that rank"
                .into(),
            at: None
        }
    );
}

const PARTS: &str = "module parts

tile [32, 32]
format f32

program P {
    handler copy(x: tensor<f32>, y: tensor<f32>, n: u32) @on(group) {
        let out = sink(partition(y.tiles(0, n), worker.rank, group.size), block = 1)
        for xb in stream(partition(x.tiles(0, n), worker.rank, group.size), block = 1) {
            out.push(xb)
        }
    }

    chain halves(x: tensor<f32>, y: tensor<f32>, s: tensor<f32>) @on(group) {
        call copy(x, y, 8)
        call copy(y, s, 8) @on(group[1:3])
    }
}
";

#[test]
fn a_call_on_a_range_of_the_chain_s_group_runs_on_it_as_its_group() {
    let m = module(PARTS);
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (8, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::F32, (8, 1), |_, _| 0.0);
    let s = tensor(&mut run, Format::F32, (8, 1), |_, _| 0.0);
    let args = [Arg::Tensor(x), Arg::Tensor(y), Arg::Tensor(s)];
    assert_eq!(
        run.invoke("P.halves", &args, (0, 4), &[]),
        End::Completed(None)
    );
    // the second call's group is ranks 1 and 2: its group.size is 2, so its two
    // ranks copy all 8 tiles, 4 each (on a group of 4 they would copy 2 to 5)
    for i in 0..8 {
        assert!(all(&tile_f64(&run, s, i), (i + 1) as f64), "s {i}");
    }
    // on a group of 2 the range 1:3 is past it: refused at the invocation
    assert_eq!(
        run.invoke("P.halves", &args, (0, 2), &[]),
        End::Invalid {
            what: "the range 1:3 of call 1 is not inside the chain's group of 2".into(),
            at: None
        }
    );
}

/// Program 7's chain at small sizes (examples/ir/p7_swiglu.kom with D 2, H 4, WIDTH 2
/// and CHUNK 2; the gate `h g + h` in place of silu, so each number is exact): `forward`
/// as the frontend writes it, and `half`, whose up runs on rank 0 only.
const SWIGLU_CHAIN: &str = "module swiglu

tile [32, 32]
format bf16, f32

param D: u32 = 2 @bind(build)
param H: u32 = 4 @bind(build)
param WIDTH: u32 = 2 @bind(build)
param CHUNK: u32 = 2 @bind(build)

func project(x: tile<bf16>[CHUNK], w: tile<bf16>[CHUNK, WIDTH]) -> tile<f32>[WIDTH] {
    return matmul(x, w)
}

func gate(h: tile<f32>, g: tile<f32>) -> tile<f32> {
    return h * g + h
}

program SwiGLU {
    handler up(x: tensor<bf16>, w1: tensor<bf16>, w3: tensor<bf16>, hidden: tensor<bf16>) @on(group) {
        var h: tile<f32>[WIDTH] = zeros
        var g: tile<f32>[WIDTH] = zeros
        let first = worker.rank * WIDTH
        for xb, w1b, w3b in zip(stream(x.tiles[0:1, 0:D], block = CHUNK), stream(w1.tiles[0:D, first:first + WIDTH], block = [CHUNK, WIDTH]), stream(w3.tiles[0:D, first:first + WIDTH], block = [CHUNK, WIDTH])) {
            h += project(xb, w1b)
            g += project(xb, w3b)
        }
        store hidden.tiles[0:1, first:first + WIDTH] <- gate(h, g)
    }
    handler down(hidden: tensor<bf16>, w2: tensor<bf16>, y: tensor<f32>) @on(group) {
        var out: tile<f32>[WIDTH] = zeros
        let first = worker.rank * WIDTH
        for hb, wb in zip(stream(hidden.tiles[0:1, 0:H], block = CHUNK), stream(w2.tiles[0:H, first:first + WIDTH], block = [CHUNK, WIDTH])) {
            out += project(hb, wb)
        }
        store y.tiles[0:1, first:first + WIDTH] <- out
    }
    chain forward(x: tensor<bf16>, w1: tensor<bf16>, w3: tensor<bf16>, w2: tensor<bf16>, hidden: tensor<bf16>, y: tensor<f32>) @on(group) {
        call up(x, w1, w3, hidden)
        call down(hidden, w2, y) @on(group[0:1])
    }
    chain half(x: tensor<bf16>, w1: tensor<bf16>, w3: tensor<bf16>, w2: tensor<bf16>, hidden: tensor<bf16>, y: tensor<f32>) @on(group) {
        call up(x, w1, w3, hidden) @on(group[0:1])
        call down(hidden, w2, y) @on(group[0:1])
    }
}
";

#[test]
fn program_7s_chain_runs_up_then_down_on_its_part_of_the_group() {
    let m = module(SWIGLU_CHAIN);
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    // x's tiles hold 1/32 and 2/32; W1's tile (k, j) holds j + 1, W3's 2; W2's tile
    // (k, j) j + 1; the hidden and y start at -1
    let x = tensor(&mut run, Format::Bf16, (1, 2), |i, _| (i + 1) as f64 / 32.0);
    let w1 = tensor(&mut run, Format::Bf16, (2, 4), |i, _| (i % 4 + 1) as f64);
    let w3 = tensor(&mut run, Format::Bf16, (2, 4), |_, _| 2.0);
    let w2 = tensor(&mut run, Format::Bf16, (4, 2), |i, _| (i % 2 + 1) as f64);
    let hidden = tensor(&mut run, Format::Bf16, (1, 4), |_, _| -1.0);
    let y = tensor(&mut run, Format::F32, (1, 2), |_, _| -1.0);
    let args = [x, w1, w3, w2, hidden, y].map(Arg::Tensor);
    assert_eq!(
        run.invoke("SwiGLU.forward", &args, (0, 2), &[]),
        End::Completed(None)
    );
    let exact = |run: &Machine<'_>, t, i, want: f64| {
        let got = tile_f64(run, t, i);
        assert!(
            got.iter().all(|v| *v == want),
            "tile {i}: {} for {want}",
            got[0]
        );
    };
    // By hand: up on ranks 0 and 1, h's tile j 32 (1/32 + 2/32)(j + 1) = 3 (j + 1), g's
    // 32 (3/32) 2 = 6, the gate h g + h = 21 (j + 1): the hidden 21, 42, 63, 84; down
    // on rank 0 alone (group[0:1], a group of 1), y's tile j the sum over k of
    // 32 hidden_k (j + 1) = 32 x 210 (j + 1): 6720 and 13440
    for (j, want) in [21.0, 42.0, 63.0, 84.0].into_iter().enumerate() {
        exact(&run, hidden, j, want);
    }
    exact(&run, y, 0, 6720.0);
    exact(&run, y, 1, 13440.0);
    // `half`: up on rank 0 only writes the hidden's tiles 0 and 1; down reads all four,
    // tiles 2 and 3 as they were before the chain (-1): 32 (21 + 42 - 1 - 1)(j + 1) =
    // 1952 (j + 1)
    let hidden = tensor(&mut run, Format::Bf16, (1, 4), |_, _| -1.0);
    let y = tensor(&mut run, Format::F32, (1, 2), |_, _| -1.0);
    let args = [x, w1, w3, w2, hidden, y].map(Arg::Tensor);
    assert_eq!(
        run.invoke("SwiGLU.half", &args, (0, 2), &[]),
        End::Completed(None)
    );
    for (j, want) in [21.0, 42.0, -1.0, -1.0].into_iter().enumerate() {
        exact(&run, hidden, j, want);
    }
    exact(&run, y, 0, 1952.0);
    exact(&run, y, 1, 3904.0);
}

fn scalars(run: &Machine<'_>, worker: usize, member: usize) -> Vec<Scalar> {
    match &run.workers[worker].state[0][member].data {
        komorebi_ir::interp::memory::MemberData::Scalars(v) => v.clone(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn t9_visits_walk_the_ranks_and_end_at_worker_0() {
    let m = module(&lesson("t09_walk"));
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    assert_eq!(
        run.invoke("Walk.walk", &[Arg::Scalar(u32s(2))], (0, 4), &[]),
        End::Completed(None)
    );
    // eight Visits, two from each rank, each ends as one Done at worker 0
    assert_eq!(scalars(&run, 0, 0)[0], u32s(8));
    // the last Done in (sender rank, send count) order is rank 3's second: Visit 0
    // from rank 3, which passed ranks 3, 0, 1, 2 and so totals 6
    assert_eq!(scalars(&run, 0, 0)[1], u32s(6));
    for w in 1..4 {
        assert_eq!(scalars(&run, w, 0), [u32s(0), u32s(0)], "worker {w}");
    }
    // with room for seven live messages, the eighth `send` of the seed (rank 3's
    // second) is refused, and nothing uses its result
    let options = Options {
        live_bound: Some(7),
        ..Options::default()
    };
    let mut run = Machine::load(&m, 4, &[], options).expect("loads");
    let End::Failed {
        reason: Failure::SendRefused,
        at: Some((3, komorebi_ir::program::BodyRef::Seed(..), _)),
    } = run.invoke("Walk.walk", &[Arg::Scalar(u32s(2))], (0, 4), &[])
    else {
        panic!("the eighth send is refused");
    };
}

#[test]
fn t15_a_job_passes_along_the_ranks() {
    let m = module(&lesson("t15_handoff"));
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (8, 1), |_, _| 1.0);
    let args = [Arg::Tensor(x), Arg::Scalar(u32s(8))];
    assert_eq!(
        run.invoke("Relay.pass", &args, (0, 4), &[]),
        End::Completed(None)
    );
    // four hops, each adds 1 to every tile
    for i in 0..8 {
        assert!(all(&tile_f64(&run, x, i), 5.0), "tile {i}");
    }
    // the job reached rank r with hops 3 - r, and wrote its worker's id there
    for w in 0..4 {
        let done = scalars(&run, w, 0);
        assert_eq!(done[3 - w], u32s(w as u64), "worker {w}");
    }
    // the halo: each rank doubles its two tiles, then sums its neighbour's two
    assert_eq!(
        run.invoke("Relay.halo", &args, (0, 4), &[]),
        End::Completed(None)
    );
    for w in 0..4 {
        let komorebi_ir::interp::memory::MemberData::Tiles(edge) = &run.workers[w].state[0][1].data
        else {
            panic!("tiles");
        };
        assert!(all(&edge[0].to_f64().expect("floats"), 20.0), "worker {w}");
    }
}

const RING: &str = "module ring

tile [32, 32]
format f32

requires {
    dynamic
}

program R {
    message Tick { step: u32, gap: u32 }

    state @scope(worker) @lifetime(program) {
        seen: u32[1]
    }

    run ring(gap: u32) @dynamic @peers(offsets = [1]) {
        seed {
            seen.zero()
            if worker.is_leader {
                let went = offer(Tick { step: 0, gap: gap }, to = (worker.rank + gap) % group.size)
            }
        }

        on Tick(m) {
            seen.bump(0)
            if m.step < 5 {
                let went = offer(
                    Tick { step: m.step + 1, gap: m.gap },
                    to = (worker.rank + m.gap) % group.size,
                )
            }
            consume m
        }
    }

    run keep(n: u32) @dynamic {
        seed {
            seen.zero()
            send(Tick { step: 0, gap: n }, to = worker.rank)
        }

        on Tick(m) {
            seen.bump(0)
            consume m
        }
    }
}
";

#[test]
fn an_offer_goes_to_a_declared_peer_only() {
    let m = module(RING);
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    // a tick goes round the ring six times: ranks 1, 2, 3, 0, 1, 2
    assert_eq!(
        run.invoke("R.ring", &[Arg::Scalar(u32s(1))], (0, 4), &[]),
        End::Completed(None)
    );
    let seen: Vec<Scalar> = (0..4).map(|w| scalars(&run, w, 0)[0]).collect();
    assert_eq!(seen, [u32s(1), u32s(2), u32s(2), u32s(1)]);
    // rank 2 is no peer of rank 0
    let End::Failed {
        reason: Failure::NotAPeer,
        at: Some((0, _, _)),
    } = run.invoke("R.ring", &[Arg::Scalar(u32s(2))], (0, 4), &[])
    else {
        panic!("an offer to a rank that is not a peer");
    };
    assert_eq!(
        run.invoke("R.keep", &[Arg::Scalar(u32s(0))], (0, 4), &[]),
        End::Completed(None)
    );
}

fn construct(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/ir/corpus/golden/constructs")
        .join(format!("{name}.kom"));
    std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{} is readable", path.display()))
}

#[test]
fn a_body_given_to_a_send_is_read_where_the_message_arrives() {
    let m = module(&construct("c01_message_body"));
    let mut run = Machine::load(&m, 3, &[], Options::default()).expect("loads");
    assert_eq!(
        run.invoke("Bodies.trace", &[Arg::Scalar(u32s(1))], (0, 3), &[]),
        End::Completed(None)
    );
    // each worker's Hit came from the rank before it, whose note was [1, rank]
    for w in 0..3 {
        let sender = ((w + 2) % 3) as u64;
        assert_eq!(
            scalars(&run, w, 1),
            [u32s(0), u32s(1 + sender), u32s(sender), u32s(1)],
            "worker {w}"
        );
    }
}

#[test]
fn a_forward_as_another_type_carries_the_fields_and_the_body() {
    let m = module(&construct("c01_message_body"));
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    assert_eq!(
        run.invoke("Bodies.trace", &[Arg::Scalar(u32s(2))], (0, 2), &[]),
        End::Completed(None)
    );
    // two Rays from the other rank, each forwarded as a Hit; the last is Ray 1, whose
    // body was [2, sender]: its id and its body's word 1 arrive in `on Hit`
    for w in 0..2 {
        let sender = ((w + 1) % 2) as u64;
        let seen = scalars(&run, w, 1);
        assert_eq!(seen[0], u32s(1), "worker {w}: the carried id");
        assert_eq!(
            seen[1],
            u32s(2 + sender),
            "worker {w}: the sum `with` gives"
        );
        assert_eq!(seen[2], u32s(sender), "worker {w}: the carried body");
        assert_eq!(seen[3], u32s(2), "worker {w}: two Hits");
    }
}

const PASS_ON: &str = "module relay

tile [4, 4]
format u32

requires {
    dynamic
}

program P {
    message Ray @body(8) { id: u32 }

    state @scope(worker) @lifetime(program) {
        note: u32[2]
        seen: u32[5]  @export
    }

    run go(n: u32) @dynamic {
        seed {
            note.zero()
            seen.zero()
            note[0] = n
            note[1] = worker.rank + 10
            send(Ray { id: worker.rank }, to = worker.rank, body = note)
        }

        on Ray(m) {
            let w = m.body.words()
            let k = seen.bump(4)
            seen[k * 2] = w[0]
            seen[k * 2 + 1] = w[1]
            if k == 0 {
                send(Ray { id: m.id }, to = worker.rank, body = m.body)
            }
            consume m
        }
    }
}
";

#[test]
fn a_body_passed_on_in_a_new_send() {
    let m = module(PASS_ON);
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    assert_eq!(
        run.invoke("P.go", &[Arg::Scalar(u32s(7))], (0, 2), &[]),
        End::Completed(None)
    );
    // the second arrival reads the words the first one carried
    for w in 0..2u64 {
        assert_eq!(
            scalars(&run, w as usize, 1),
            [u32s(7), u32s(w + 10), u32s(7), u32s(w + 10), u32s(2)],
            "worker {w}"
        );
    }
}

const OFFER_BODY: &str = "module ring

tile [4, 4]
format u32

requires {
    dynamic
}

program R {
    message Tick @body(4) { step: u32 }

    state @scope(worker) @lifetime(program) {
        note: u32[1]
        seen: u32[2]
    }

    run ring(gap: u32) @dynamic @peers(offsets = [1]) {
        seed {
            note.zero()
            seen.zero()
            note[0] = worker.rank + 100
            if worker.is_leader {
                let went = offer(Tick { step: 0 }, to = (worker.rank + gap) % group.size, body = note)
            }
        }

        on Tick(m) {
            let w = m.body.words()
            seen[0] = w[0]
            seen.bump(1)
            consume m
        }
    }
}
";

#[test]
fn an_offer_carries_a_body() {
    let m = module(OFFER_BODY);
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    assert_eq!(
        run.invoke("R.ring", &[Arg::Scalar(u32s(1))], (0, 4), &[]),
        End::Completed(None)
    );
    // the leader's offer reaches rank 1 with the leader's note
    let seen: Vec<Vec<Scalar>> = (0..4).map(|w| scalars(&run, w, 1)).collect();
    assert_eq!(seen[1], [u32s(100), u32s(1)]);
    for w in [0, 2, 3] {
        assert_eq!(seen[w], [u32s(0), u32s(0)], "worker {w}");
    }
}

const TOO_LONG: &str = "module long

tile [4, 4]
format u32

param ROOM: u32 = 8 @bind(load)

requires {
    dynamic
}

program L {
    message Ray @body(16) { id: u32 }

    state @scope(worker) @lifetime(program) {
        big: u32[ROOM]
    }

    run go(n: u32) @dynamic {
        seed {
            big.zero()
            send(Ray { id: n }, to = worker.rank, body = big)
        }

        on Ray(m) {
            consume m
        }
    }
}
";

#[test]
fn a_body_longer_than_the_message_takes_fails_out_of_range() {
    let m = module(TOO_LONG);
    let mut run = Machine::load(&m, 1, &[], Options::default()).expect("loads");
    // eight words are 32 bytes, and a Ray takes 16
    let End::Failed {
        reason: Failure::OutOfRange,
        at: Some((0, BodyRef::Seed(..), _)),
    } = run.invoke("L.go", &[Arg::Scalar(u32s(0))], (0, 1), &[])
    else {
        panic!("a body longer than the message takes");
    };
    // four words fit
    let mut run = Machine::load(&m, 1, &[("ROOM", u32s(4))], Options::default()).expect("loads");
    assert_eq!(
        run.invoke("L.go", &[Arg::Scalar(u32s(0))], (0, 1), &[]),
        End::Completed(None)
    );
}

const NO_WORDS: &str = "module empty

tile [4, 4]
format u32

requires {
    dynamic
}

program E {
    message Ray @body(16) { id: u32 }

    state @scope(worker) @lifetime(program) {
        seen: u32[1]
    }

    run go(n: u32) @dynamic {
        seed {
            seen.zero()
            send(Ray { id: n }, to = worker.rank)
        }

        on Ray(m) {
            let w = m.body.words()
            seen[0] = w[0]
            consume m
        }
    }
}
";

#[test]
fn a_word_past_the_bodys_end_fails_out_of_range() {
    let m = module(NO_WORDS);
    let mut run = Machine::load(&m, 1, &[], Options::default()).expect("loads");
    // a send with no `body =`: the body is empty, and its word 0 is past its end
    let End::Failed {
        reason: Failure::OutOfRange,
        at: Some((0, BodyRef::On(..), _)),
    } = run.invoke("E.go", &[Arg::Scalar(u32s(0))], (0, 1), &[])
    else {
        panic!("a word past the body's end");
    };
}

#[test]
fn a_scalar_stored_is_the_tiles_first_word_and_the_rest_are_zero() {
    let m = module(&construct("c03_store_scalars"));
    let mut run = Machine::load(&m, 3, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::U32, (7, 1), |_, _| 0.0);
    let y = tensor(&mut run, Format::U32, (3, 1), |_, _| 5.0);
    let args = [Arg::Tensor(x), Arg::Tensor(y), Arg::Scalar(u32s(7))];
    assert_eq!(
        run.invoke("Counted.blocks", &args, (0, 3), &[]),
        End::Completed(None)
    );
    // seven tiles over three workers: three, two, two blocks
    for (w, count) in [3, 2, 2].into_iter().enumerate() {
        let mut want = vec![0i128; 16];
        want[0] = count;
        assert_eq!(tile_ints(&run, y, w), want, "worker {w}");
    }
}

#[test]
fn a_state_member_stored_is_the_tiles_first_words() {
    let m = module(&construct("c03_store_scalars"));
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let y = tensor(&mut run, Format::U32, (2, 1), |_, _| 5.0);
    assert_eq!(
        run.invoke("Counted.notes", &[Arg::Tensor(y)], (0, 2), &[]),
        End::Completed(None)
    );
    for w in 0..2 {
        let mut want = vec![0i128; 16];
        want[..4].copy_from_slice(&[w as i128, 2, 7, 9]);
        assert_eq!(tile_ints(&run, y, w), want, "worker {w}");
    }
}

#[test]
fn the_words_of_a_tile_stored_copy_the_tile() {
    let m = module(&construct("c03_store_scalars"));
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::U32, (2, 1), |i, k| (100 * i + k) as f64);
    let y = tensor(&mut run, Format::U32, (2, 1), |_, _| 5.0);
    assert_eq!(
        run.invoke(
            "Counted.copy",
            &[Arg::Tensor(x), Arg::Tensor(y)],
            (0, 2),
            &[]
        ),
        End::Completed(None)
    );
    for w in 0..2 {
        assert_eq!(tile_ints(&run, y, w), tile_ints(&run, x, w), "worker {w}");
    }
}

const TOO_MANY: &str = "module many

tile [4, 4]
format u32

param ROOM: u32 = 32 @bind(load)

program M {
    state @scope(worker) @lifetime(program) {
        note: u32[ROOM]
    }

    handler put(y: tensor<u32>) @on(group) {
        note.zero()
        store y.tiles(worker.rank, 1) <- note
    }
}
";

#[test]
fn a_parameters_value_is_its_expression() {
    let m = module(
        "module sized

tile [4, 4]
format u32

param D: u32 = 3 @bind(build)
param H: u32 = 2 * D + 1 @bind(build)

program S {
    handler height(x: tensor<u32>) -> u32 @on(group) {
        return H
    }
}
",
    );
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::U32, (1, 1), |_, _| 0.0);
    assert_eq!(
        run.invoke("S.height", &[Arg::Tensor(x)], (0, 2), &[]),
        End::Completed(Some(u32s(7)))
    );
}

const PLACED: &str = "module placed

tile [4, 4]
format u32

requires {
    group = grid
}

param R: u32 = 2 @bind(build)
param C: u32 = 2 @bind(build)

program G {
    handler place(y: tensor<u32>) @on(grid[SIDES]) {
        store y.tiles(worker.rank, 1) <- worker.row * 100 + worker.col
    }
}
";

#[test]
fn a_grid_given_by_expressions_runs_as_its_numbers_do() {
    let tiles = |sides: &str| {
        let m = module(&PLACED.replace("SIDES", sides));
        let mut run = Machine::load(&m, 8, &[], Options::default()).expect("loads");
        let y = tensor(&mut run, Format::U32, (8, 1), |_, _| 5.0);
        assert_eq!(
            run.invoke("G.place", &[Arg::Tensor(y)], (0, 8), &[]),
            End::Completed(None)
        );
        (0..8)
            .map(|i| tile_ints(&run, y, i)[0])
            .collect::<Vec<i128>>()
    };
    let by_numbers = tiles("2, 4");
    assert_eq!(by_numbers, [0, 1, 2, 3, 100, 101, 102, 103]);
    assert_eq!(tiles("R, 2 * C"), by_numbers);
}

const PEERED: &str = "module peered

tile [4, 4]
format u32

requires {
    dynamic
}

param G: u32 = 2 @bind(build)
param Z: u32 = 0 @bind(build)

program R {
    message Tick { step: u32, gap: u32 }

    state @scope(worker) @lifetime(program) {
        seen: u32[1]
    }

    run ring(gap: u32) @dynamic @peers(offsets = [G - 1]) {
        seed {
            seen.zero()
            if worker.is_leader {
                let went = offer(Tick { step: 0, gap: gap }, to = (worker.rank + gap) % group.size)
            }
        }

        on Tick(m) {
            seen.bump(0)
            consume m
        }
    }

    run pair(to: u32) @dynamic @peers(pairs = [(Z, G + 1), (G + 1, Z)]) {
        seed {
            seen.zero()
            if worker.is_leader {
                let went = offer(Tick { step: 0, gap: 0 }, to = to)
            }
        }

        on Tick(m) {
            seen.bump(0)
            consume m
        }
    }
}
";

#[test]
fn peers_given_by_parameters_are_the_numbers() {
    let m = module(PEERED);
    assert_eq!(
        m.programs[0]
            .runs
            .iter()
            .map(|r| r.peers.clone())
            .collect::<Vec<_>>(),
        [
            komorebi_ir::types::Peers::Offsets(vec![1]),
            komorebi_ir::types::Peers::Pairs(vec![(0, 3), (3, 0)]),
        ]
    );
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    // the offset G - 1 is 1: a tick to the next rank arrives
    assert_eq!(
        run.invoke("R.ring", &[Arg::Scalar(u32s(1))], (0, 4), &[]),
        End::Completed(None)
    );
    assert_eq!(scalars(&run, 1, 0), [u32s(1)]);
    // rank 2 is no peer of rank 0
    let End::Failed {
        reason: Failure::NotAPeer,
        at: Some((0, _, _)),
    } = run.invoke("R.ring", &[Arg::Scalar(u32s(2))], (0, 4), &[])
    else {
        panic!("an offer to a rank that is not a peer");
    };
    // the pair (Z, G + 1) is (0, 3)
    assert_eq!(
        run.invoke("R.pair", &[Arg::Scalar(u32s(3))], (0, 4), &[]),
        End::Completed(None)
    );
    assert_eq!(scalars(&run, 3, 0), [u32s(1)]);
    let End::Failed {
        reason: Failure::NotAPeer,
        at: Some((0, _, _)),
    } = run.invoke("R.pair", &[Arg::Scalar(u32s(1))], (0, 4), &[])
    else {
        panic!("an offer to a rank that is not a pair");
    };
}

#[test]
fn a_count_given_by_an_expression() {
    let m = module(
        "module counted

tile [4, 4]
format u32

param N: u32 = 4 @bind(build)

program C {
    state @scope(worker) @lifetime(program) {
        seen: u32[N * 2] @filled(element)
    }

    handler last(x: tensor<u32>) -> u32 @on(group) {
        seen[N * 2 - 1] = worker.rank + 40
        return seen[N * 2 - 1]
    }
}
",
    );
    assert_eq!(m.programs[0].state[0].count, Count::Const(8));
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::U32, (1, 1), |_, _| 0.0);
    assert_eq!(
        run.invoke("C.last", &[Arg::Tensor(x)], (0, 2), &[]),
        End::Completed(Some(u32s(40)))
    );
    assert_eq!(scalars(&run, 1, 0)[7], u32s(41));
}

#[test]
fn scalars_past_the_tiles_end_fail_out_of_range() {
    let m = module(TOO_MANY);
    // 32 scalars, and a tile of [4, 4] holds 16
    let mut run = Machine::load(&m, 1, &[], Options::default()).expect("loads");
    let y = tensor(&mut run, Format::U32, (1, 1), |_, _| 5.0);
    let End::Failed {
        reason: Failure::OutOfRange,
        at: Some((0, BodyRef::Handler(..), _)),
    } = run.invoke("M.put", &[Arg::Tensor(y)], (0, 1), &[])
    else {
        panic!("more scalars than the tile holds");
    };
    // sixteen fit
    let mut run = Machine::load(&m, 1, &[("ROOM", u32s(16))], Options::default()).expect("loads");
    let y = tensor(&mut run, Format::U32, (1, 1), |_, _| 5.0);
    assert_eq!(
        run.invoke("M.put", &[Arg::Tensor(y)], (0, 1), &[]),
        End::Completed(None)
    );
    assert_eq!(tile_ints(&run, y, 0), vec![0i128; 16]);
}

#[test]
fn a_message_body_must_handle_its_message_once() {
    use komorebi_ir::program::body::Op;
    use komorebi_ir::types::MemberId;
    // the checker refuses both on every path (K501): the body is edited after Lower
    let m = module(RING);
    let insts = |m: &Module| m.programs[0].runs[1].on_message[0].1.insts.clone();
    let consume = insts(&m)
        .iter()
        .position(|i| matches!(i.op, Op::Consume(_)))
        .expect("a consume");
    let bump = insts(&m)
        .iter()
        .position(|i| matches!(i.op, Op::Bump(..)))
        .expect("a bump");
    let mut none = m.clone();
    none.programs[0].runs[1].on_message[0].1.insts[consume].op = Op::ZeroState(MemberId(0));
    let mut twice = m.clone();
    twice.programs[0].runs[1].on_message[0].1.insts[bump].op = insts(&m)[consume].op.clone();
    for (m, said) in [(none, "neither consumes nor forwards"), (twice, "twice")] {
        let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
        let End::Invalid {
            what,
            at: Some((0, komorebi_ir::program::BodyRef::On(..), _)),
        } = run.invoke("R.keep", &[Arg::Scalar(u32s(0))], (0, 4), &[])
        else {
            panic!("{said}");
        };
        assert!(what.contains(said), "{what}");
    }
}

const RACES: &str = "module races

tile [32, 32]
format f32

requires {
    dynamic
}

program Sweep {
    message Job { first: u32, tiles: u32, hops: u32 }

    handler scale(x: tensor<f32>, first: u32, tiles: u32) @on(group) {
        let part = partition(x.tiles(first, tiles), worker.rank, group.size)
        let out = sink(part, block = 1)
        for xb in stream(part, block = 1) {
            out.push(xb * 0.5)
        }
    }

    // each rank doubles its part, then adds its neighbour's part into its own
    handler halo(x: tensor<f32>, y: tensor<f32>, tiles: u32) @on(group) {
        let mine = partition(x.tiles(0, tiles), worker.rank, group.size)
        let out = sink(mine, block = 1)
        for xb in stream(mine, block = 1) {
            out.push(xb * 2.0)
        }
        sync(publish = [mine])
        let next = (worker.rank + 1) % group.size
        let ys = sink(partition(y.tiles(0, tiles), worker.rank, group.size), block = 1)
        for xb in stream(partition(x.tiles(0, tiles), next, group.size), block = 1) {
            ys.push(xb)
        }
    }

    // the same, met at an all_reduce that publishes nothing: ordered, not seen
    handler bare(x: tensor<f32>, y: tensor<f32>, tiles: u32) -> u32 @on(group) {
        let mine = partition(x.tiles(0, tiles), worker.rank, group.size)
        let out = sink(mine, block = 1)
        for xb in stream(mine, block = 1) {
            out.push(xb * 2.0)
        }
        let n = all_reduce(worker.rank, with = max)
        let next = (worker.rank + 1) % group.size
        let ys = sink(partition(y.tiles(0, tiles), worker.rank, group.size), block = 1)
        for xb in stream(partition(x.tiles(0, tiles), next, group.size), block = 1) {
            ys.push(xb)
        }
        return n
    }

    // the job passes once from rank 0 to rank 1; `late` writes after publishing
    run pass(x: tensor<f32>, late: u32) @dynamic {
        seed {
            if worker.is_leader {
                send(Job { first: 0, tiles: 1, hops: 1 }, to = worker.rank)
            }
        }
        on Job(m) {
            let part = x.tiles(m.first, m.tiles)
            let out = sink(part, block = 1)
            for xb in stream(part, block = 1) {
                out.push(xb + 1.0)
            }
            if m.hops == 0 {
                consume m
            } else {
                forward(m, with = { hops: m.hops - 1 }, to = worker.rank + 1, publish = [part])
                if late == 1 {
                    var zero: tile<f32> = zeros
                    store part <- zero
                }
            }
        }
    }
}

control halves(x: tensor<f32>) {
    let a = invoke(Sweep.scale(x, 0, 8), on = group(0, 2))
    let b = invoke(Sweep.scale(x, 8, 8), on = group(2, 2))
    let c = invoke(Sweep.scale(x, 4, 8), on = group(4, 2))
    await a
    await b
    await c
}

control ordered(x: tensor<f32>) {
    let a = invoke(Sweep.scale(x, 0, 8), on = group(0, 2))
    let b = invoke(Sweep.scale(x, 8, 8), on = group(2, 2))
    let c = invoke(Sweep.scale(x, 4, 8), on = group(4, 2), after = [a, b])
    await c
}

control seen(x: tensor<f32>) {
    let a = invoke(Sweep.scale(x, 0, 8), on = group(0, 2))
    await a
    let c = invoke(Sweep.scale(x, 4, 8), on = group(4, 2))
    await c
}
";

/// The `what` of an invalid end.
fn invalid_what(end: End) -> String {
    match end {
        End::Invalid { what, .. } => what,
        other => panic!("an invalid end, not {other:?}"),
    }
}

#[test]
fn b2_halves_c_races_a_and_an_edge_or_an_await_orders_it() {
    let m = module_unverified(RACES);
    let mut run = Machine::load(&m, 6, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (16, 1), |_, _| 1.0);
    // c's rank 0 reads tile 4 first, which a's rank 1 wrote; nothing orders a and c
    let race = "a race on tensor 0 tile 4: a write at #0 rank 1 (p0.h0 i14) and a read at \
                #2 rank 0 (p0.h0 i11): no completion, end or chain edge orders the two \
                invocations";
    assert_eq!(
        run.control("halves", &[Arg::Tensor(x)]),
        End::Invalid {
            what: race.into(),
            at: Some((0, BodyRef::Handler(ProgramId(0), HandlerId(0)), InstId(11))),
        }
    );
    let printed = run.report.print();
    let lines: Vec<&str> = printed.lines().collect();
    assert_eq!(lines[0], "#0 Sweep.scale group(0, 2): completed");
    assert_eq!(lines[1], "#1 Sweep.scale group(2, 2): completed");
    assert_eq!(
        lines[2],
        format!("#2 Sweep.scale group(4, 2): invalid: {race} at rank 0, p0.h0 i11")
    );
    assert!(
        lines[3].starts_with("#3 control halves: invalid: a race"),
        "{printed}"
    );
    // `after = [a, b]`, or an `await a` before c is made, orders them
    let mut run = Machine::load(&m, 6, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (16, 1), |_, _| 1.0);
    assert_eq!(
        run.control("ordered", &[Arg::Tensor(x)]),
        End::Completed(None)
    );
    assert_eq!(run.control("seen", &[Arg::Tensor(x)]), End::Completed(None));
    // a halves 0 to 7 twice, b 8 to 15 once, c 4 to 11 twice
    for i in 0..16 {
        let want = [0.25, 0.0625, 0.125, 0.5][i / 4];
        assert!(all(&tile_f64(&run, x, i), want), "tile {i}");
    }
    // invocations the host makes one after another are ordered
    let mut run = Machine::load(&m, 6, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (16, 1), |_, _| 1.0);
    for (first, group) in [(0, (0, 2)), (8, (2, 2)), (4, (4, 2))] {
        let args = [
            Arg::Tensor(x),
            Arg::Scalar(u32s(first)),
            Arg::Scalar(u32s(8)),
        ];
        assert_eq!(
            run.invoke("Sweep.scale", &args, group, &[]),
            End::Completed(None)
        );
    }
}

#[test]
fn a_write_is_seen_on_another_rank_only_through_its_publication() {
    let m = module_unverified(RACES);
    let mut run = Machine::load(&m, 4, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (8, 1), |i, _| i as f64);
    let y = tensor(&mut run, Format::F32, (8, 1), |_, _| 0.0);
    let args = [Arg::Tensor(x), Arg::Tensor(y), Arg::Scalar(u32s(8))];
    // `sync(publish = [mine])`: each rank's part reaches the rank before it
    assert_eq!(
        run.invoke("Sweep.halo", &args, (0, 4), &[]),
        End::Completed(None)
    );
    for i in 0..8 {
        let want = 2.0 * ((i + 2) % 8) as f64;
        assert!(all(&tile_f64(&run, y, i), want), "tile {i}");
    }
    // the same met at an `all_reduce` that publishes nothing: ordered, but not seen
    assert_eq!(
        invalid_what(run.invoke("Sweep.bare", &args, (0, 4), &[])),
        "a race on tensor 0 tile 2: a write at #1 rank 1 (p0.h2 i15) and a read at #1 \
         rank 0 (p0.h2 i37): no publication of the place reaches the second rank"
    );
}

#[test]
fn b11_a_store_after_its_forward_races_the_receivers_read() {
    let m = module_unverified(RACES);
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (2, 1), |_, _| 1.0);
    let pass = |late| [Arg::Tensor(x), Arg::Scalar(u32s(late))];
    // the forward publishes tile 0; rank 1 reads and writes it after
    assert_eq!(
        run.invoke("Sweep.pass", &pass(0), (0, 2), &[]),
        End::Completed(None)
    );
    assert!(all(&tile_f64(&run, x, 0), 3.0));
    // rank 0 stores tile 0 after the publication: rank 1's read does not see it
    assert_eq!(
        run.invoke("Sweep.pass", &pass(1), (0, 2), &[]),
        End::Invalid {
            what: "a race on tensor 0 tile 0: a write at #1 rank 0 (p0.r0.on0 i34) and a \
                   read at #1 rank 1 (p0.r0.on0 i10): no publication of the place reaches \
                   the second rank"
                .into(),
            at: Some((1, BodyRef::On(ProgramId(0), RunId(0), 0), InstId(10))),
        }
    );
}

const ITEMS: &str = "module items

tile [32, 32]
format u32

program Marks {
    state @scope(worker) @lifetime(program) {
        hit: u32[4] @filled(element)
    }

    handler mark(results: tensor<u32>, at: u32) @on(group) {
        hit[0] = at
        hit[1] = at
        hit[2] = at
        hit[3] = at
        results.items(size = 16).put(hit, at = at)
    }
}

control apart(r: tensor<u32>) {
    let a = invoke(Marks.mark(r, 0), on = group(0, 1))
    let b = invoke(Marks.mark(r, 1), on = group(1, 1))
    await a
    await b
}

control same(r: tensor<u32>) {
    let a = invoke(Marks.mark(r, 1), on = group(0, 1))
    let b = invoke(Marks.mark(r, 1), on = group(1, 1))
    await a
    await b
}
";

#[test]
fn an_item_is_its_own_place() {
    let m = module_unverified(ITEMS);
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let r = tensor(&mut run, Format::U32, (1, 1), |_, _| 0.0);
    // two items of one tile, put by unordered invocations: no race
    assert_eq!(
        run.control("apart", &[Arg::Tensor(r)]),
        End::Completed(None)
    );
    assert_eq!(tile_ints(&run, r, 0)[..8], [0, 0, 0, 0, 1, 1, 1, 1]);
    // one item put by both
    let what = invalid_what(run.control("same", &[Arg::Tensor(r)]));
    assert!(
        what.starts_with("a race on tensor 0 tile 0: a write at #3 rank 0"),
        "{what}"
    );
}

fn invalid_case(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/ir/corpus/golden/invalid")
        .join(format!("{name}.kom"));
    std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{} is readable", path.display()))
}

#[test]
fn the_corpus_races_the_build_refuses_are_races_at_the_run() {
    // halves: c's rank 0 reads tile 16 first, which a's rank 4 wrote
    let m = module_unverified(&invalid_case("k701_race_halves"));
    let mut run = Machine::load(&m, 24, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::Bf16, (64, 1), |_, _| 1.0);
    let what = invalid_what(run.control("halves", &[Arg::Tensor(x)]));
    assert!(
        what.starts_with("a race on tensor 0 tile 16: a write at #0 rank 4 ")
            && what.contains(" and a read at #2 rank 0 ")
            && what.ends_with(": no completion, end or chain edge orders the two invocations"),
        "{what}"
    );
    // order: b reads what a wrote, with no edge; c and d alone would be fine
    let m = module_unverified(&invalid_case("k701_race_order"));
    let mut run = Machine::load(&m, 16, &[], Options::default()).expect("loads");
    let x = tensor(&mut run, Format::F32, (64, 1), |_, _| 1.0);
    let y = tensor(&mut run, Format::F32, (64, 1), |_, _| 1.0);
    let what = invalid_what(run.control("order", &[Arg::Tensor(x), Arg::Tensor(y)]));
    assert!(
        what.starts_with("a race on tensor 0 tile 0: a write at #0 rank 0 ")
            && what.contains(" and a read at #1 rank 0 "),
        "{what}"
    );
    let printed = run.report.print();
    let lines: Vec<&str> = printed.lines().collect();
    // the race stops the run: c and d never run
    assert_eq!(lines.len(), 3, "{printed}");
    assert_eq!(lines[0], "#0 P.write group(0, 8): completed");
    assert!(
        lines[1].starts_with("#1 P.read group(8, 8): invalid: a race"),
        "{printed}"
    );
    assert!(
        lines[2].starts_with("#2 control order: invalid: a race"),
        "{printed}"
    );
}

fn on_islands(sizes: &[usize], links: Option<&[(usize, usize)]>) -> Options {
    Options {
        islands: Some(IslandGraph {
            sizes: sizes.to_vec(),
            links: links.map(<[(usize, usize)]>::to_vec),
        }),
        ..Options::default()
    }
}

/// What `locate` wrote on each worker: count, own island, crosses, hops to the next.
fn located(sizes: &[usize], links: Option<&[(usize, usize)]>) -> Vec<Vec<Scalar>> {
    let m = module(&lesson("t12_spread"));
    let workers = sizes.iter().sum();
    let mut run = Machine::load(&m, workers, &[], on_islands(sizes, links)).expect("loads");
    assert_eq!(
        run.invoke("Spread.locate", &[], (0, workers), &[]),
        End::Completed(None)
    );
    (0..workers).map(|w| scalars(&run, w, 0)).collect()
}

fn u32x4(v: [u64; 4]) -> Vec<Scalar> {
    v.iter().map(|x| u32s(*x)).collect()
}

#[test]
fn t12_the_queries_walk_the_island_graph() {
    // two islands, every pair connected: ranks 1 and 3 cross to their next
    let seen = located(&[2, 2], None);
    assert_eq!(seen[0], u32x4([2, 0, 0, 0]));
    assert_eq!(seen[1], u32x4([2, 0, 1, 1]));
    assert_eq!(seen[2], u32x4([2, 1, 0, 0]));
    assert_eq!(seen[3], u32x4([2, 1, 1, 1]));
    // a line of three: worker 5 on island 2 reaches worker 0 on island 0 in two hops
    let seen = located(&[2, 2, 2], Some(&[(0, 1), (1, 2)]));
    assert_eq!(seen[1], u32x4([3, 0, 1, 1]));
    assert_eq!(seen[3], u32x4([3, 1, 1, 1]));
    assert_eq!(seen[5], u32x4([3, 2, 1, 2]));
    // neither a line nor complete: 1 is joined to 0, 2 and 3; from 2 to 3 is 2 hops,
    // from 3 back to 0 is 2, from 1 to 2 is 1
    let seen = located(&[1, 1, 1, 1], Some(&[(0, 1), (1, 2), (1, 3)]));
    assert_eq!(seen[0], u32x4([4, 0, 1, 1]));
    assert_eq!(seen[1], u32x4([4, 1, 1, 1]));
    assert_eq!(seen[2], u32x4([4, 2, 1, 2]));
    assert_eq!(seen[3], u32x4([4, 3, 1, 2]));
}

#[test]
fn t12_islands_that_are_not_a_connected_graph_are_refused_by_name() {
    let m = module(&lesson("t12_spread"));
    let load = |sizes: &[usize], links: Option<&[(usize, usize)]>| {
        let workers = 4;
        Machine::load(&m, workers, &[], on_islands(sizes, links)).err()
    };
    assert_eq!(
        load(&[2, 2], Some(&[])),
        Some(Refusal::Islands(IslandRefusal::NotConnected))
    );
    // the code and the text a backend's load gives for the same graph
    assert_eq!(
        load(&[2, 2], Some(&[])).map(|r| r.to_string()).as_deref(),
        Some("K966: the machine given at load is not a machine: the islands are not connected")
    );
    assert_eq!(
        load(&[1, 1, 1, 1], Some(&[(0, 1), (2, 3)])),
        Some(Refusal::Islands(IslandRefusal::NotConnected))
    );
    assert_eq!(
        load(&[2, 1], None),
        Some(Refusal::Islands(IslandRefusal::NotTheWorkers {
            held: 3,
            workers: 4
        }))
    );
    assert_eq!(
        load(&[4, 0], None),
        Some(Refusal::Islands(IslandRefusal::Empty(1)))
    );
    assert_eq!(
        load(&[2, 2], Some(&[(0, 2)])),
        Some(Refusal::Islands(IslandRefusal::NoSuchIsland(2)))
    );
    // `islands >= 2` on the one island a run has with no option
    assert_eq!(
        Machine::load(&m, 4, &[], Options::default()).err(),
        Some(Refusal::Requirement("islands >= 2".into()))
    );
}

#[test]
fn t12_each_island_writes_its_copy_and_merge_makes_them_one() {
    let m = module(&lesson("t12_spread"));
    let mut run = Machine::load(&m, 4, &[], on_islands(&[2, 2], None)).expect("loads");
    let x = tensor(&mut run, Format::F32, (4, 1), |i, _| (i + 1) as f64);
    let shape = m.default_tile();
    let w = run.replicated(Tensor::zeros("w", Format::F32, 4 * 32, 32, shape));
    let args = [Arg::Tensor(x), Arg::Tensor(w), Arg::Scalar(u32s(4))];
    assert_eq!(
        run.invoke("Spread.learn", &args, (0, 4), &[]),
        End::Completed(None)
    );
    let copy = |run: &Machine<'_>, i| run.copy(w, i).expect("a copy");
    let sums = |run: &Machine<'_>, i| -> Vec<f64> {
        (0..4).map(|k| tile_f64(run, copy(run, i), k)[0]).collect()
    };
    // workers 0 and 1 wrote island 0's copy, workers 2 and 3 island 1's
    assert_eq!(sums(&run, 0), [1.0, 2.0, 0.0, 0.0]);
    assert_eq!(sums(&run, 1), [0.0, 0.0, 3.0, 4.0]);
    assert_eq!(
        run.control("merge", &[Arg::Tensor(w)]),
        End::Completed(None)
    );
    assert_eq!(sums(&run, 0), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(sums(&run, 1), [1.0, 2.0, 3.0, 4.0]);
    for i in 0..2 {
        for k in 0..4 {
            assert!(all(&tile_f64(&run, copy(&run, i), k), (k + 1) as f64));
        }
    }
    assert_eq!(
        run.report.print(),
        "#0 Spread.learn group(0, 4): completed\n\
         #1 control merge reduce_to: completed\n\
         #2 control merge broadcast_from: completed\n\
         #3 control merge: completed\n"
    );
    // a plain tensor where a replicated one is declared, and the reverse
    let plain = [Arg::Tensor(x), Arg::Tensor(x), Arg::Scalar(u32s(4))];
    assert_eq!(
        invalid_what(run.invoke("Spread.learn", &plain, (0, 4), &[])),
        "a tensor that is not replicated, for a `@replicated` parameter"
    );
    let reverse = [Arg::Tensor(w), Arg::Tensor(w), Arg::Scalar(u32s(4))];
    assert_eq!(
        invalid_what(run.invoke("Spread.learn", &reverse, (0, 4), &[])),
        "a replicated tensor, for a parameter that is not `@replicated`"
    );
}

const COPIES: &str = "module copies

tile [32, 32]
format f32

requires {
    islands >= 2
}

combine add(a: tile<f32>, b: tile<f32>) -> tile<f32> @commutative {
    return a + b
}

program Spread {
    handler learn(x: tensor<f32>, w: tensor<f32> @replicated, tiles: u32) @on(group) {
        let src = partition(x.tiles(0, tiles), worker.rank, group.size)
        var acc: tile<f32> = zeros
        for xb in stream(src, block = 1) {
            acc += xb
        }
        store w.tiles(worker.rank, 1) <- acc
    }
}

control far(w: tensor<f32> @replicated) {
    reduce_to(w, to = 2, with = add)
}

control total(w: tensor<f32> @replicated) {
    all_reduce(w, with = add)
}

// the reduce reads the copies the invocation writes, with no await between
control hasty(x: tensor<f32>, w: tensor<f32> @replicated) {
    let a = invoke(Spread.learn(x, w, 4), on = group(0, 4))
    reduce_to(w, to = 0, with = add)
    await a
}
";

#[test]
fn a_collective_over_copies_is_ordered_like_an_invocation() {
    let m = module_unverified(COPIES);
    let mut run = Machine::load(&m, 4, &[], on_islands(&[2, 2], None)).expect("loads");
    let shape = m.default_tile();
    let w = run.replicated(Tensor::zeros("w", Format::F32, 4 * 32, 32, shape));
    // all_reduce: every copy the sum of both, from 1 in each to 2
    let ones = run.replicated(Tensor::zeros("o", Format::F32, 4 * 32, 32, shape));
    for i in 0..2 {
        let c = run.copy(ones, i).expect("a copy");
        for t in run.heap.tensors[c.0].tiles.iter_mut() {
            *t = Tile::from_f64(Format::F32, shape, &[1.0; 1024]).expect("a tile");
        }
    }
    assert_eq!(
        run.control("total", &[Arg::Tensor(ones)]),
        End::Completed(None)
    );
    for i in 0..2 {
        assert!(all(
            &tile_f64(&run, run.copy(ones, i).expect("a copy"), 3),
            2.0
        ));
    }
    // `to = 2` on two islands
    let End::Invalid {
        what,
        at: Some((0, _, _)),
    } = run.control("far", &[Arg::Tensor(w)])
    else {
        panic!("an invalid end at the control's instruction");
    };
    assert_eq!(what, "`to` names no island");
    // the invocation made before the reduce, and not awaited, races it: the reduce
    // wrote copy 0 (which replaces its own read of it in the record)
    let mut run = Machine::load(&m, 4, &[], on_islands(&[2, 2], None)).expect("loads");
    let x = tensor(&mut run, Format::F32, (4, 1), |i, _| (i + 1) as f64);
    let w2 = run.replicated(Tensor::zeros("w", Format::F32, 4 * 32, 32, shape));
    let what = invalid_what(run.control("hasty", &[Arg::Tensor(x), Arg::Tensor(w2)]));
    assert!(
        what.starts_with("a race on tensor 1 tile 0: a write at #0 rank 0 (c2 i")
            && what.contains("and a write at #1 rank 0 (p0.h0 i")
            && what.ends_with("no completion, end or chain edge orders the two invocations"),
        "{what}"
    );
}

/// A replicated tensor of `tiles` tiles of the module's tile, one tile wide, every
/// copy holding `value(i)` in tile i.
fn replicated(m: &mut Machine<'_>, tiles: usize, value: impl Fn(usize) -> f64) -> TensorId {
    let shape = m.module().default_tile();
    let (h, w) = (shape.rows as usize, shape.cols as usize);
    let mut t = Tensor::zeros("w", Format::F32, tiles * h, w, shape);
    for (i, tile) in t.tiles.iter_mut().enumerate() {
        *tile = Tile::from_f64(Format::F32, shape, &vec![value(i); h * w]).expect("a tile");
    }
    m.replicated(t)
}

/// Element 0 of each tile of island `i`'s copy of `w`.
fn copy_firsts(run: &Machine<'_>, w: TensorId, i: usize) -> Vec<f64> {
    let c = run.copy(w, i).expect("a copy");
    (0..run.heap.tensors[c.0].tiles.len())
        .map(|k| tile_f64(run, c, k)[0])
        .collect()
}

#[test]
fn all_gather_gives_every_rank_the_groups_tiles_in_rank_order() {
    let m = module(&construct("c02_all_gather"));
    let mut run = Machine::load(&m, 3, &[], on_islands(&[2, 1], None)).expect("loads");
    let x = tensor(&mut run, Format::F32, (3, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::F32, (9, 1), |_, _| 0.0);
    assert_eq!(
        run.invoke(
            "Gathered.everyone",
            &[Arg::Tensor(x), Arg::Tensor(y)],
            (0, 3),
            &[]
        ),
        End::Completed(None)
    );
    for r in 0..3 {
        for k in 0..3 {
            assert!(
                all(&tile_f64(&run, y, r * 3 + k), (k + 1) as f64),
                "rank {r}, tile {k}"
            );
        }
    }
}

#[test]
fn all_gather_over_copies_gives_every_copy_what_each_island_wrote() {
    let m = module(&construct("c02_all_gather"));
    let mut run = Machine::load(&m, 4, &[], on_islands(&[2, 2], None)).expect("loads");
    let x = tensor(&mut run, Format::F32, (4, 1), |i, _| (i + 1) as f64);
    let w = replicated(&mut run, 4, |_| 0.0);
    assert_eq!(
        run.invoke(
            "Gathered.learn",
            &[Arg::Tensor(x), Arg::Tensor(w)],
            (0, 4),
            &[]
        ),
        End::Completed(None)
    );
    assert_eq!(copy_firsts(&run, w, 0), [1.0, 2.0, 0.0, 0.0]);
    assert_eq!(copy_firsts(&run, w, 1), [0.0, 0.0, 3.0, 4.0]);
    assert_eq!(
        run.control("spread", &[Arg::Tensor(w)]),
        End::Completed(None)
    );
    for i in 0..2 {
        for k in 0..4 {
            let c = run.copy(w, i).expect("a copy");
            assert!(
                all(&tile_f64(&run, c, k), (k + 1) as f64),
                "copy {i}, tile {k}"
            );
        }
    }
    assert_eq!(
        run.report.print(),
        "#0 Gathered.learn group(0, 4): completed\n\
         #1 control spread all_gather: completed\n\
         #2 control spread: completed\n"
    );
}

#[test]
fn all_gather_leaves_a_place_no_island_wrote() {
    let m = module(&construct("c02_all_gather"));
    let mut run = Machine::load(&m, 4, &[], on_islands(&[2, 2], None)).expect("loads");
    let x = tensor(&mut run, Format::F32, (4, 1), |i, _| (i + 1) as f64);
    // the fifth tile holds 7 on every copy, and no worker writes it
    let w = replicated(&mut run, 5, |i| if i == 4 { 7.0 } else { 0.0 });
    assert_eq!(
        run.invoke(
            "Gathered.learn",
            &[Arg::Tensor(x), Arg::Tensor(w)],
            (0, 4),
            &[]
        ),
        End::Completed(None)
    );
    assert_eq!(
        run.control("spread", &[Arg::Tensor(w)]),
        End::Completed(None)
    );
    for i in 0..2 {
        assert_eq!(
            copy_firsts(&run, w, i),
            [1.0, 2.0, 3.0, 4.0, 7.0],
            "copy {i}"
        );
        let c = run.copy(w, i).expect("a copy");
        assert!(all(&tile_f64(&run, c, 4), 7.0));
    }
}

const GATHER_AFTER_REDUCE: &str = "module folded

tile [4, 4]
format f32

requires {
    islands >= 2
}

control fold_then_spread(w: tensor<f32> @replicated) {
    reduce_to(w, to = 1, with = sum)
    all_gather(w)
}
";

#[test]
fn all_gather_after_reduce_to_spreads_the_copy_written() {
    let m = module(GATHER_AFTER_REDUCE);
    let mut run = Machine::load(&m, 4, &[], on_islands(&[2, 2], None)).expect("loads");
    let w = replicated(&mut run, 2, |_| 0.0);
    // copy 0 holds 1 and copy 1 holds 2 everywhere, set from the host: no island
    // wrote them
    for (i, v) in [(0, 1.0), (1, 2.0)] {
        let c = run.copy(w, i).expect("a copy");
        for t in run.heap.tensors[c.0].tiles.iter_mut() {
            *t = Tile::from_f64(Format::F32, m.default_tile(), &[v; 16]).expect("a tile");
        }
    }
    assert_eq!(
        run.control("fold_then_spread", &[Arg::Tensor(w)]),
        End::Completed(None)
    );
    // copy 1's fold is 3 everywhere, and every copy equals it
    for i in 0..2 {
        for k in 0..2 {
            let c = run.copy(w, i).expect("a copy");
            assert!(all(&tile_f64(&run, c, k), 3.0), "copy {i}, tile {k}");
        }
    }
    assert_eq!(
        run.report.print(),
        "#0 control fold_then_spread reduce_to: completed\n\
         #1 control fold_then_spread all_gather: completed\n\
         #2 control fold_then_spread: completed\n"
    );
}

#[test]
fn a_place_written_on_two_islands_takes_the_lowest_islands_value() {
    let m = module(&construct("c02_all_gather"));
    // two islands of one worker each: `learn` invoked once on each island's worker
    // writes tile 0 (its rank's) of that island's copy, so no two workers of one
    // island write it
    let mut run = Machine::load(&m, 2, &[], on_islands(&[1, 1], None)).expect("loads");
    let x0 = tensor(&mut run, Format::F32, (1, 1), |_, _| 10.0);
    let x1 = tensor(&mut run, Format::F32, (1, 1), |_, _| 20.0);
    let w = replicated(&mut run, 4, |_| 0.0);
    for (x, first) in [(x0, 0), (x1, 1)] {
        assert_eq!(
            run.invoke(
                "Gathered.learn",
                &[Arg::Tensor(x), Arg::Tensor(w)],
                (first, 1),
                &[]
            ),
            End::Completed(None)
        );
    }
    assert_eq!(copy_firsts(&run, w, 0), [10.0, 0.0, 0.0, 0.0]);
    assert_eq!(copy_firsts(&run, w, 1), [20.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        run.control("spread", &[Arg::Tensor(w)]),
        End::Completed(None)
    );
    for i in 0..2 {
        assert_eq!(copy_firsts(&run, w, i), [10.0, 0.0, 0.0, 0.0], "copy {i}");
    }
}

#[test]
fn all_gather_makes_the_copies_equal_for_a_later_all_gather() {
    let m = module(&construct("c02_all_gather"));
    let mut run = Machine::load(&m, 4, &[], on_islands(&[2, 2], None)).expect("loads");
    let x = tensor(&mut run, Format::F32, (4, 1), |i, _| (i + 1) as f64);
    let w = replicated(&mut run, 4, |_| 0.0);
    assert_eq!(
        run.invoke(
            "Gathered.learn",
            &[Arg::Tensor(x), Arg::Tensor(w)],
            (0, 4),
            &[]
        ),
        End::Completed(None)
    );
    assert_eq!(
        run.control("spread", &[Arg::Tensor(w)]),
        End::Completed(None)
    );
    let once: Vec<Vec<Vec<f64>>> = (0..2)
        .map(|i| {
            let c = run.copy(w, i).expect("a copy");
            (0..4).map(|k| tile_f64(&run, c, k)).collect()
        })
        .collect();
    assert_eq!(
        run.control("spread", &[Arg::Tensor(w)]),
        End::Completed(None)
    );
    let twice: Vec<Vec<Vec<f64>>> = (0..2)
        .map(|i| {
            let c = run.copy(w, i).expect("a copy");
            (0..4).map(|k| tile_f64(&run, c, k)).collect()
        })
        .collect();
    assert_eq!(once, twice);
    assert_eq!(copy_firsts(&run, w, 0), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(copy_firsts(&run, w, 1), [1.0, 2.0, 3.0, 4.0]);
}

/// T12 run whole, with its state and its replicated tensor shown on request.
fn spread_report() -> String {
    let m = module(&lesson("t12_spread"));
    let mut run = Machine::load(&m, 4, &[], on_islands(&[2, 2], None)).expect("loads");
    let x = tensor(&mut run, Format::F32, (4, 1), |i, _| (i + 1) as f64);
    let w = run.replicated(Tensor::zeros(
        "w",
        Format::F32,
        4 * 32,
        32,
        m.default_tile(),
    ));
    run.show_state("Spread.seen").expect("exported");
    run.invoke("Spread.locate", &[], (0, 4), &[]);
    run.show_state("Spread.seen").expect("exported");
    let args = [Arg::Tensor(x), Arg::Tensor(w), Arg::Scalar(u32s(4))];
    run.invoke("Spread.learn", &args, (0, 4), &[]);
    run.show_tensor(w).expect("a tensor");
    run.control("merge", &[Arg::Tensor(w)]);
    run.show_tensor(w).expect("a tensor");
    run.report.print()
}

#[test]
fn the_report_shows_tensors_and_exported_state_on_request() {
    let printed = spread_report();
    assert_eq!(
        printed,
        "state Spread.seen\n\
         \x20 worker 0: _*4\n\
         \x20 worker 1: _*4\n\
         \x20 worker 2: _*4\n\
         \x20 worker 3: _*4\n\
         #0 Spread.locate group(0, 4): completed\n\
         state Spread.seen\n\
         \x20 worker 0: 2:u32, 0:u32*3\n\
         \x20 worker 1: 2:u32, 0:u32, 1:u32*2\n\
         \x20 worker 2: 2:u32, 1:u32, 0:u32*2\n\
         \x20 worker 3: 2:u32, 1:u32*3\n\
         #1 Spread.learn group(0, 4): completed\n\
         tensor w copy 0 f32 [128, 32]\n\
         \x20 tile 0: 1.0*1024\n\
         \x20 tile 1: 2.0*1024\n\
         \x20 tile 2: 0.0*1024\n\
         \x20 tile 3: 0.0*1024\n\
         tensor w copy 1 f32 [128, 32]\n\
         \x20 tile 0: 0.0*1024\n\
         \x20 tile 1: 0.0*1024\n\
         \x20 tile 2: 3.0*1024\n\
         \x20 tile 3: 4.0*1024\n\
         #2 control merge reduce_to: completed\n\
         #3 control merge broadcast_from: completed\n\
         #4 control merge: completed\n\
         tensor w copy 0 f32 [128, 32]\n\
         \x20 tile 0: 1.0*1024\n\
         \x20 tile 1: 2.0*1024\n\
         \x20 tile 2: 3.0*1024\n\
         \x20 tile 3: 4.0*1024\n\
         tensor w copy 1 f32 [128, 32]\n\
         \x20 tile 0: 1.0*1024\n\
         \x20 tile 1: 2.0*1024\n\
         \x20 tile 2: 3.0*1024\n\
         \x20 tile 3: 4.0*1024\n"
    );
    // the same run twice: the same text, byte for byte
    assert_eq!(spread_report(), printed);
    // only an exported member is shown
    let m = module(&lesson("t07_views"));
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    assert_eq!(
        run.show_state("Views.hit"),
        Err("Views.hit is not exported".into())
    );
    assert_eq!(
        run.show_state("Views.nothing"),
        Err("no state member Views.nothing".into())
    );
    // a member filled element by element: `_` where nothing was written
    let m = module(
        "module marks

tile [32, 32]
format f32

program Marks {
    state @scope(worker) @lifetime(program) {
        some: u32[3] @filled(element) @export
    }

    handler mark() @on(group) {
        some[1] = worker.rank + 7
    }
}
",
    );
    let mut marks = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    marks.invoke("Marks.mark", &[], (0, 2), &[]);
    marks.show_state("Marks.some").expect("exported");
    assert_eq!(
        marks.report.print(),
        "#0 Marks.mark group(0, 2): completed\n\
         state Marks.some\n\
         \x20 worker 0: _, 7:u32, _\n\
         \x20 worker 1: _, 8:u32, _\n"
    );
    // an integer tensor, element by element
    let t = tensor(&mut run, Format::U32, (1, 1), |_, k| (k % 3) as f64);
    run.show_tensor(t).expect("a tensor");
    let printed = run.report.print();
    let tile = printed.lines().nth(1).expect("a tile line");
    assert!(tile.starts_with("  tile 0: 0, 1, 2, 0, 1, 2,"), "{tile}");
    assert!(tile.ends_with(", 0"), "{tile}");
}

/// T13's `drive` on 64 workers: y and the printed report.
fn hinted_run(text: &str) -> (Vec<Vec<f64>>, String) {
    let m = module(text);
    let mut run = Machine::load(&m, 64, &[], Options::default()).expect("loads");
    let w = tensor(&mut run, Format::Bf16, (1, 1), |_, _| 0.5);
    let x = tensor(&mut run, Format::Bf16, (64, 1), |i, _| (i + 1) as f64);
    let y = tensor(&mut run, Format::F32, (64, 1), |_, _| 0.0);
    let args = [Arg::Tensor(w), Arg::Tensor(x), Arg::Tensor(y)];
    assert_eq!(run.control("drive", &args), End::Completed(None));
    let tiles = (0..64).map(|i| tile_f64(&run, y, i)).collect();
    (tiles, run.report.print())
}

#[test]
fn t13_hints_change_nothing_a_run_gives() {
    let text = lesson("t13_hinted");
    let (y, printed) = hinted_run(&text);
    // each element of y's tile i is 32 products (i + 1) * 0.5
    for (i, tile) in y.iter().enumerate() {
        assert!(all(tile, 16.0 * (i + 1) as f64), "tile {i}");
    }
    assert_eq!(
        printed,
        "#0 Hinted.load group(0, 64): completed\n\
         #1 Hinted.forward group(0, 64): completed\n\
         #2 control drive: completed\n"
    );
    // the same text with every hint taken out
    let bare = text
        .replace(" @tensix.on_demand", "")
        .replace(" @tensix.pattern(\"AAAB\")", "")
        .replace(" @cpu.vector(512)", "")
        .replace(" @place.control(\"near\")", "");
    assert!(!bare.contains("@tensix") && !bare.contains("@cpu") && !bare.contains("@place"));
    assert_eq!(hinted_run(&bare), (y, printed));
}

/// A function declared on blocks of 3 tiles (reading R40): its body runs on each tile,
/// a scalar going to every tile; over 7 tiles the last block is 1 tile, its result 1.
const ON_BLOCKS: &str = "module blocks

tile [32, 32]
format f32

func shifted(x: tile<f32>[3], k: f32) -> tile<f32>[3] {
    return x * k + 1.0
}

program Blocks {
    handler run(a: tensor<f32>, out: tensor<f32>, k: f32) @on(group) {
        let results = sink(out.tiles(0, 7), block = 3)
        for x in stream(a.tiles(0, 7), block = 3) {
            results.push(shifted(x, k))
        }
    }
}
";

#[test]
fn a_function_on_blocks_runs_on_each_tile_and_a_last_block_is_shorter() {
    let m = module(ON_BLOCKS);
    let mut run = Machine::load(&m, 1, &[], Options::default()).expect("loads");
    let a = tensor(&mut run, Format::F32, (1, 7), |i, _| (i + 1) as f64);
    let out = tensor(&mut run, Format::F32, (1, 7), |_, _| -1.0);
    let args = [
        Arg::Tensor(a),
        Arg::Tensor(out),
        Arg::Scalar(Scalar::F32(Format::F32, 2.0)),
    ];
    assert_eq!(
        run.invoke("Blocks.run", &args, (0, 1), &[]),
        End::Completed(None)
    );
    // tile i holds i + 1: 2 (i + 1) + 1 = 3, 5, 7, 9, 11, 13 and, the last block's
    // one tile, 15
    let want = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
    for (i, w) in want.into_iter().enumerate() {
        assert!(all(&tile_f64(&run, out, i), w), "tile {i}");
    }
}

#[test]
fn a_function_on_blocks_calls_functions_on_blocks_of_their_shapes_and_on_tiles() {
    // R45: in a function on blocks a function on blocks takes blocks of its shapes, and
    // a function on tiles runs on each tile of blocks of one shape
    let text = |call: &str| {
        format!(
            "module blocks\n\ntile [32, 32]\nformat f32\n\n\
             func twice(x: tile<f32>[4]) -> tile<f32>[4] {{\n    return x * 2.0\n}}\n\n\
             func one(x: tile<f32>) -> tile<f32> {{\n    return x + 1.0\n}}\n\n\
             func many(x: tile<f32>[4], t: tile<f32>) -> tile<f32>[4] {{\n    return {call}\n}}\n"
        )
    };
    for call in ["twice(x)", "one(x)", "twice(one(x)) - x"] {
        komorebi_ir::check(&text(call)).unwrap_or_else(|d| panic!("{call}: {d:?}"));
    }
    let why = "(in a function on blocks a tile is `[1, 1]` and a block is of its shape)";
    for (call, what) in [
        ("twice(t)", why),
        ("one(t)", why),
        (
            "x + t",
            "takes operands of one shape, element by element: [4] and [1]",
        ),
    ] {
        let found = komorebi_ir::check(&text(call)).expect_err(call);
        assert!(!found.is_empty(), "{call}");
        for d in &found {
            assert_eq!(d.code, "K102", "{call}: {d:?}");
            assert!(d.message.contains(what), "{call}: {d:?}");
        }
    }
}

/// A function on blocks of `n` tiles that reduces along a row (reading R43): the
/// reduction runs along the whole block, `.col()` spreads each row's value over every
/// tile, `.row()` spreads each tile's own row 0, `* 2.0` and `+` work tile by tile. A
/// stream of one tile a block gives tiles: the same body, on one tile.
fn row_form(n: usize) -> String {
    let t = if n == 1 {
        "tile<f32>".to_string()
    } else {
        format!("tile<f32>[{n}]")
    };
    format!(
        "module rows

tile [32, 32]
format f32

func spread(x: {t}) -> {t} {{
    return reduce_sum(x * 2.0, axis = row).col() + x.row()
}}

program Rows {{
    handler run(a: tensor<f32>, out: tensor<f32>) @on(group) {{
        let results = sink(out.tiles(0, 2), block = {n})
        for x in stream(a.tiles(0, 2), block = {n}) {{
            results.push(spread(x))
        }}
    }}
}}
"
    )
}

/// The two tiles of `row_form(n)` run on a tensor whose tile 0 holds r + c at row r,
/// column c, and tile 1 holds 1: the result's two tiles, element by element.
fn row_form_run(n: usize) -> (Vec<f64>, Vec<f64>) {
    let m = module(&row_form(n));
    let mut run = Machine::load(&m, 1, &[], Options::default()).expect("loads");
    let a = tensor(&mut run, Format::F32, (1, 2), |i, k| {
        if i == 0 {
            (k / 32 + k % 32) as f64
        } else {
            1.0
        }
    });
    let out = tensor(&mut run, Format::F32, (1, 2), |_, _| -1.0);
    let args = [Arg::Tensor(a), Arg::Tensor(out)];
    assert_eq!(
        run.invoke("Rows.run", &args, (0, 1), &[]),
        End::Completed(None)
    );
    (tile_f64(&run, out, 0), tile_f64(&run, out, 1))
}

#[test]
fn a_reduction_along_a_row_in_a_function_on_blocks_runs_along_the_whole_block() {
    // a block of 2 tiles: row r of the 64 elements side by side doubled is
    // 2 (32 r + 496 + 32) = 64 r + 1056; tile 0 adds its row 0 (c at column c), tile 1
    // its own (1)
    let (first, second) = row_form_run(2);
    for k in 0..32 * 32 {
        let (r, c) = ((k / 32) as f64, (k % 32) as f64);
        assert_eq!(first[k], 64.0 * r + 1056.0 + c, "tile 0, element {k}");
        assert_eq!(second[k], 64.0 * r + 1057.0, "tile 1, element {k}");
    }
    // blocks of 1 tile: each tile's own rows, 2 (32 r + 496) + c and 2 x 32 + 1
    let (first, second) = row_form_run(1);
    for k in 0..32 * 32 {
        let (r, c) = ((k / 32) as f64, (k % 32) as f64);
        assert_eq!(first[k], 64.0 * r + 992.0 + c, "tile 0, element {k}");
        assert_eq!(second[k], 65.0, "tile 1, element {k}");
    }
}

#[test]
fn in_a_function_on_blocks_what_works_on_a_whole_tile_but_a_row_is_not_built() {
    // R43 builds a reduction along a row, `.col()` of its values and `.row()` of a tile
    // parameter; a reduction along a column, `.col()` of a tile, `.row()` of another
    // tile, `matmul` and `.T` stay named and not built (K010)
    let body = |line: &str| {
        format!(
            "module rows\n\ntile [32, 32]\nformat f32\n\n\
             func f(x: tile<f32>[2], y: tile<f32>[2]) -> tile<f32>[2] {{\n    {line}\n}}\n"
        )
    };
    let built = [
        "return reduce_max(x, axis = row).col() - x",
        "return reduce_min(x + y, axis = row).col() * y.row()",
    ];
    for line in built {
        komorebi_ir::check(&body(line)).unwrap_or_else(|d| panic!("{line}: {d:?}"));
    }
    let refused = [
        (
            "return reduce_sum(x, axis = col).row()",
            "a reduction along a column",
        ),
        (
            "return x.col()",
            "`.col()` of other than a reduction along a row",
        ),
        (
            "return (x * 2.0).row()",
            "`.row()` of other than a tile parameter",
        ),
        ("return x.T", "`.T`"),
    ];
    for (line, what) in refused {
        let found = komorebi_ir::check(&body(line)).expect_err(line);
        assert!(
            found
                .iter()
                .any(|d| d.code == "K010" && d.message.starts_with(what)),
            "{line}: {found:?}"
        );
    }
}

/// A handler that pushes `f(x, w)`, the stream of `a`'s first two tiles in blocks of
/// `[1, 2]` zipped with the stream of the tiles `[0:2, 0:c]` of `w` in blocks of
/// `[2, c]` (reading R45: a block is a matrix of tiles).
fn product_of_blocks(func: &str, c: usize, out: usize) -> String {
    format!(
        "module product

tile [32, 32]
format f32

{func}

program Product {{
    handler run(a: tensor<f32>, w: tensor<f32>, out: tensor<f32>) @on(group) {{
        let results = sink(out.tiles(0, {out}), block = {out})
        let rows = stream(a.tiles(0, 2), block = 2)
        let cols = stream(w.tiles[0:2, 0:{c}], block = [2, {c}])
        for x, y in zip(rows, cols) {{
            results.push(f(x, y))
        }}
    }}
}}
"
    )
}

/// `product_of_blocks` run: `a`'s tile 0 holds 1 and its tile 1 holds the element's
/// row; `w`, a grid of 2 x `c` tiles, is `w(i, j)` at its tile (i, j), element (k, c).
/// The result's tiles.
fn product_run(text: &str, c: usize, out: usize, w: impl Fn(usize, usize) -> f64) -> Vec<Vec<f64>> {
    let m = module(text);
    let mut run = Machine::load(&m, 1, &[], Options::default()).expect("loads");
    let a = tensor(&mut run, Format::F32, (1, 2), |i, k| {
        if i == 0 {
            1.0
        } else {
            (k / 32) as f64
        }
    });
    let w = tensor(&mut run, Format::F32, (2, c), w);
    let o = tensor(&mut run, Format::F32, (1, out), |_, _| -1.0);
    let args = [Arg::Tensor(a), Arg::Tensor(w), Arg::Tensor(o)];
    assert_eq!(
        run.invoke("Product.run", &args, (0, 1), &[]),
        End::Completed(None)
    );
    (0..out).map(|i| tile_f64(&run, o, i)).collect()
}

#[test]
fn a_product_of_a_row_of_two_tiles_by_a_column_of_two_is_the_sum_of_the_tiles_products() {
    let func = "func f(x: tile<f32>[1, 2], w: tile<f32>[2, 1]) -> tile<f32> {\n    \
                return matmul(x, w)\n}";
    // w's tile 0 holds its element's row k, its tile 1 holds 1. By hand: x0 w0 at
    // (r, c) is the sum over k of 1 * k, 496; x1 w1 is the sum over k of r * 1, 32 r.
    // The other pairing (x0 w1 + x1 w0) would give 32 + 496 r.
    let w = |i: usize, k: usize| if i == 0 { (k / 32) as f64 } else { 1.0 };
    let tiles = product_run(&product_of_blocks(func, 1, 1), 1, 1, w);
    for (k, got) in tiles[0].iter().enumerate() {
        let r = (k / 32) as f64;
        assert_eq!(*got, 496.0 + 32.0 * r, "element {k}");
    }
}

#[test]
fn a_product_of_a_row_of_two_tiles_by_two_rows_of_three_is_a_row_of_three() {
    let func = "func f(x: tile<f32>[1, 2], w: tile<f32>[2, 3]) -> tile<f32>[1, 3] {\n    \
                return matmul(x, w)\n}";
    // w's tile (0, j) holds j + 1, its tile (1, j) holds 1 (tile i of the grid is row
    // i / 3, column i % 3). By hand, out tile j at (r, c): the sum over k of
    // 1 * (j + 1), 32 (j + 1), and of r * 1, 32 r.
    let w = |i: usize, _: usize| if i < 3 { (i + 1) as f64 } else { 1.0 };
    let tiles = product_run(&product_of_blocks(func, 3, 3), 3, 3, w);
    for (j, tile) in tiles.iter().enumerate() {
        for (k, got) in tile.iter().enumerate() {
            let r = (k / 32) as f64;
            assert_eq!(
                *got,
                32.0 * (j + 1) as f64 + 32.0 * r,
                "tile {j}, element {k}"
            );
        }
    }
}

#[test]
fn a_reduction_along_a_row_of_a_block_of_two_rows_runs_along_each_row() {
    // `f(y: [1, 2], x: [2, 2])`: x's two rows of tiles reduced along a row each and
    // spread back, then y times that (reading R45)
    let func = "func f(y: tile<f32>[1, 2], x: tile<f32>[2, 2]) -> tile<f32>[1, 2] {\n    \
                return matmul(y, reduce_sum(x, axis = row).col())\n}";
    let text = product_of_blocks(func, 2, 2);
    // x's tiles hold 1, 2 (row 0) and 3, 4 (row 1); y's tile 0 holds 1, its tile 1
    // the element's row. By hand: row 0 of the block sums to 32 (1 + 2) = 96 at every
    // element of its tiles, row 1 to 32 (3 + 4) = 224. Out tile j at (r, c): the sum
    // over k of 1 * 96 and of r * 224, 3072 + 7168 r. One reduction along the whole
    // block (320 everywhere) would give 10240 + 10240 r.
    let x = |i: usize, _: usize| (i + 1) as f64;
    let tiles = product_run(&text, 2, 2, x);
    for (j, tile) in tiles.iter().enumerate() {
        for (k, got) in tile.iter().enumerate() {
            let r = (k / 32) as f64;
            assert_eq!(*got, 3072.0 + 7168.0 * r, "tile {j}, element {k}");
        }
    }
}

#[test]
fn the_shapes_of_blocks_are_checked_and_their_rows_are_known_at_build() {
    // R45: `matmul` of blocks takes the left's columns of tiles as the right's rows;
    // the rows of a block are a number written or a module parameter bound at build
    // (program 6); a sink of several rows is not built (K010)
    let func = |sig: &str, body: &str| {
        format!(
            "module shapes\n\ntile [32, 32]\nformat f32\n\nparam n: u32 = 2 @bind(build)\n\
             param m: u32 = 2 @bind(load)\nparam k: u32 = 2 @bind(invoke)\n\n\
             func f({sig} {{\n    return {body}\n}}\n"
        )
    };
    let built = [
        (
            "x: tile<f32>[1, 2], w: tile<f32>[2, 1]) -> tile<f32>",
            "matmul(x, w)",
        ),
        (
            "x: tile<f32>[2, 2], w: tile<f32>[2, 3]) -> tile<f32>[2, 3]",
            "matmul(x, w) + 1.0",
        ),
        (
            "x: tile<f32>[2, 3]) -> tile<f32>[2, 3]",
            "reduce_sum(x, axis = row).col() * x",
        ),
        // rows of a parameter bound at build: its value, known at compile
        (
            "x: tile<f32>[n], w: tile<f32>[n, 1]) -> tile<f32>",
            "matmul(x, w)",
        ),
    ];
    for (sig, body) in built {
        komorebi_ir::check(&func(sig, body)).unwrap_or_else(|d| panic!("{body}: {d:?}"));
    }
    let refused = [
        (
            "x: tile<f32>[1, 2], w: tile<f32>[3, 1]) -> tile<f32>",
            "matmul(x, w)",
            "the left's columns of tiles are the right's rows",
        ),
        (
            "x: tile<f32>[2, 2], w: tile<f32>[1, 4]) -> tile<f32>[2, 2]",
            "x + w",
            "takes operands of one shape, element by element: [2, 2] and [4]",
        ),
        (
            "x: tile<f32>[m, 2]) -> tile<f32>[m, 2]",
            "x",
            "a number written in the text or a module parameter bound at build",
        ),
        (
            "x: tile<f32>[k, 2]) -> tile<f32>[k, 2]",
            "x",
            "a module parameter bound at build (one bound at load or at the invocation",
        ),
        (
            "x: tile<f32>[n, 3], w: tile<f32>[2, 1]) -> tile<f32>",
            "matmul(x, w)",
            "the left's columns of tiles are the right's rows",
        ),
    ];
    for (sig, body, what) in refused {
        let found = komorebi_ir::check(&func(sig, body)).expect_err(body);
        assert!(
            found
                .iter()
                .any(|d| d.code == "K102" && d.message.contains(what)),
            "{body}: {found:?}"
        );
    }
    let sink = "module shapes\n\ntile [32, 32]\nformat f32\n\n\
                program P {\n    handler run(out: tensor<f32>) @on(group) {\n        \
                let s = sink(out.tiles(0, 4), block = [2, 2])\n    }\n}\n";
    let found = komorebi_ir::check(sink).expect_err("a sink of two rows");
    assert!(found.iter().any(|d| d.code == "K010"), "{found:?}");
}

#[test]
fn program_6_sums_each_ranks_products_of_a_row_of_x_and_a_column_of_w() {
    // Small.read as the frontend writes it (reading R45): rank r N + n reads X's row r
    // in blocks of CHUNK and W's column n in blocks of [CHUNK, 1], and adds the
    // products into a tile of zeros, which it stores at Y's tile call M N + rank
    let m = module(
        "module small

tile [32, 32]
format bf16, f32

param M_TILES: u32 = 4 @bind(build)
param K_TILES: u32 = 28 @bind(build)
param N_TILES: u32 = 4 @bind(build)
param CHUNK: u32 = 4 @bind(build)

func project(x: tile<bf16>[CHUNK], w: tile<bf16>[CHUNK, 1]) -> tile<f32> {
    return matmul(x, w)
}

program Small {
    handler read(x: tensor<bf16>, w: tensor<bf16>, y: tensor<f32>, call: u32) @on(group) {
        let r = worker.rank / N_TILES
        let n = worker.rank % N_TILES
        var acc: tile<f32> = zeros
        for xb, wb in zip(stream(x.tiles[r:r + 1, 0:K_TILES], block = CHUNK), stream(w.tiles[0:K_TILES, n:n + 1], block = [CHUNK, 1])) {
            acc += project(xb, wb)
        }
        store y.tiles(((call * M_TILES) * N_TILES) + worker.rank, 1) <- acc
    }
}
",
    );
    let mut run = Machine::load(&m, 16, &[], Options::default()).expect("loads");
    // X's tile (r, k) holds r + 1, W's tile (k, n) holds k + n; Y two calls' worth
    let x = tensor(&mut run, Format::Bf16, (4, 28), |i, _| (i / 28 + 1) as f64);
    let w = tensor(&mut run, Format::Bf16, (28, 4), |i, _| {
        (i / 4 + i % 4) as f64
    });
    let y = tensor(&mut run, Format::F32, (8, 4), |_, _| -1.0);
    let args = [
        Arg::Tensor(x),
        Arg::Tensor(w),
        Arg::Tensor(y),
        Arg::Scalar(u32s(1)),
    ];
    assert_eq!(
        run.invoke("Small.read", &args, (0, 16), &[]),
        End::Completed(None)
    );
    // By hand: each element of Y's tile (r, n) is the sum over k of 32 (r + 1)(k + n),
    // 32 (r + 1)(378 + 28 n); call 1 writes tiles 16 to 31, call 0's stay -1
    for rank in 0..16 {
        let (r, n) = ((rank / 4) as f64, (rank % 4) as f64);
        let want = 32.0 * (r + 1.0) * (378.0 + 28.0 * n);
        let got = tile_f64(&run, y, 16 + rank);
        assert!(
            got.iter().all(|v| *v == want),
            "rank {rank}: {} for {want}",
            got[0]
        );
        assert!(
            tile_f64(&run, y, rank).iter().all(|v| *v == -1.0),
            "tile {rank}"
        );
    }
}

#[test]
fn program_7s_down_adds_each_ranks_products_into_a_block_var_and_stores_it() {
    // SwiGLU.down as the frontend writes it (reading R48): the field `out` the
    // handler's var of a block of one row from zeros; rank r adds the products of the
    // hidden's row, in blocks of [1, CHUNK], by W2's columns r WIDTH to r WIDTH +
    // WIDTH - 1, in blocks of [CHUNK, WIDTH], and stores the var to y's row there
    let m = module(
        "module swiglu

tile [32, 32]
format bf16, f32

param H: u32 = 6 @bind(build)
param WIDTH: u32 = 4 @bind(build)
param CHUNK: u32 = 2 @bind(build)

func project(x: tile<bf16>[CHUNK], w: tile<bf16>[CHUNK, WIDTH]) -> tile<f32>[WIDTH] {
    return matmul(x, w)
}

program SwiGLU {
    handler down(hidden: tensor<bf16>, w2: tensor<bf16>, y: tensor<f32>) @on(group) {
        var out: tile<f32>[WIDTH] = zeros
        let first = worker.rank * WIDTH
        for hb, wb in zip(stream(hidden.tiles[0:1, 0:H], block = CHUNK), stream(w2.tiles[0:H, first:first + WIDTH], block = [CHUNK, WIDTH])) {
            out += project(hb, wb)
        }
        store y.tiles[0:1, first:first + WIDTH] <- out
    }
}
",
    );
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    // the hidden's tile (0, k) holds k + 1, W2's tile (k, j) holds j + 1; y is 1 x 8
    let hidden = tensor(&mut run, Format::Bf16, (1, 6), |i, _| (i + 1) as f64);
    let w2 = tensor(&mut run, Format::Bf16, (6, 8), |i, _| (i % 8 + 1) as f64);
    let y = tensor(&mut run, Format::F32, (1, 8), |_, _| -1.0);
    let args = [Arg::Tensor(hidden), Arg::Tensor(w2), Arg::Tensor(y)];
    assert_eq!(
        run.invoke("SwiGLU.down", &args, (0, 2), &[]),
        End::Completed(None)
    );
    // By hand: y's tile j at each element is the sum over k of 32 (k + 1)(j + 1),
    // 32 x 21 (j + 1) = 672 (j + 1); ranks 0 and 1 write tiles 0-3 and 4-7
    for j in 0..8 {
        let want = 672.0 * (j + 1) as f64;
        let got = tile_f64(&run, y, j);
        assert!(
            got.iter().all(|v| *v == want),
            "tile {j}: {} for {want}",
            got[0]
        );
    }
}

#[test]
fn program_7s_up_zips_three_streams_and_stores_a_function_of_two_block_vars() {
    // SwiGLU.up (reading R48): x's row by W1's and W3's columns, three streams zipped,
    // added into two vars of a block of one row; `gate(h, g)`, the function on one
    // tile run on each tile of the two, stored to the rectangle
    let m = module(
        "module swiglu

tile [32, 32]
format bf16, f32

param D: u32 = 2 @bind(build)
param WIDTH: u32 = 2 @bind(build)
param CHUNK: u32 = 2 @bind(build)

func gate(h: tile<f32>, g: tile<f32>) -> tile<f32> {
    return h * g + h
}

func project(x: tile<bf16>[CHUNK], w: tile<bf16>[CHUNK, WIDTH]) -> tile<f32>[WIDTH] {
    return matmul(x, w)
}

program SwiGLU {
    handler up(x: tensor<bf16>, w1: tensor<bf16>, w3: tensor<bf16>, hidden: tensor<f32>) @on(group) {
        var h: tile<f32>[WIDTH] = zeros
        var g: tile<f32>[WIDTH] = zeros
        let first = worker.rank * WIDTH
        for xb, w1b, w3b in zip(stream(x.tiles[0:1, 0:D], block = CHUNK), stream(w1.tiles[0:D, first:first + WIDTH], block = [CHUNK, WIDTH]), stream(w3.tiles[0:D, first:first + WIDTH], block = [CHUNK, WIDTH])) {
            h += project(xb, w1b)
            g += project(xb, w3b)
        }
        store hidden.tiles[0:1, first:first + WIDTH] <- gate(h, g)
    }
}
",
    );
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    // x's tiles hold 1/32 and 2/32; W1's tile (k, j) holds j + 1, W3's 2
    let x = tensor(&mut run, Format::Bf16, (1, 2), |i, _| (i + 1) as f64 / 32.0);
    let w1 = tensor(&mut run, Format::Bf16, (2, 4), |i, _| (i % 4 + 1) as f64);
    let w3 = tensor(&mut run, Format::Bf16, (2, 4), |_, _| 2.0);
    let hidden = tensor(&mut run, Format::F32, (1, 4), |_, _| -1.0);
    let args = [x, w1, w3, hidden].map(Arg::Tensor);
    assert_eq!(
        run.invoke("SwiGLU.up", &args, (0, 2), &[]),
        End::Completed(None)
    );
    // By hand: h's tile j holds 32 (1/32 + 2/32)(j + 1) = 3 (j + 1), g's 32 (3/32) 2
    // = 6; the gate gives h g + h = 21 (j + 1): 21, 42, 63, 84
    for (j, want) in [21.0, 42.0, 63.0, 84.0].into_iter().enumerate() {
        let got = tile_f64(&run, hidden, j);
        assert!(
            got.iter().all(|v| *v == want),
            "tile {j}: {} for {want}",
            got[0]
        );
    }
}

// ---- several tile shapes in one module ----

/// A tensor of 8 by 8 elements stored in tiles of `shape`, element (r, c) being
/// `value(r, c)`.
fn grid8(m: &mut Machine<'_>, shape: TileShape, value: impl Fn(usize, usize) -> f64) -> TensorId {
    let mut t = Tensor::zeros("t", Format::F32, 8, 8, shape);
    let (h, w) = (shape.rows as usize, shape.cols as usize);
    let cols = t.grid.1;
    for (i, tile) in t.tiles.iter_mut().enumerate() {
        let values: Vec<f64> = (0..h * w)
            .map(|k| value((i / cols) * h + k / w, (i % cols) * w + k % w))
            .collect();
        *tile = Tile::from_f64(Format::F32, shape, &values).expect("a tile of floats");
    }
    m.tensor(t)
}

/// The 8 by 8 elements of tensor `t`, row by row, whatever tiles it is stored in.
fn elements8(m: &Machine<'_>, t: TensorId) -> Vec<f64> {
    let tensor = &m.heap.tensors[t.0];
    let (h, w) = (tensor.tile.rows as usize, tensor.tile.cols as usize);
    let mut out = Vec::with_capacity(64);
    for r in 0..8 {
        for c in 0..8 {
            let tile = &tensor.tiles[(r / h) * tensor.grid.1 + c / w];
            out.push(tile.to_f64().expect("a float tile")[(r % h) * w + c % w]);
        }
    }
    out
}

const S2: TileShape = TileShape { rows: 2, cols: 2 };

#[test]
fn a_tensor_read_in_another_tile_shape_holds_the_same_elements() {
    let m = module(&construct("c05_tile_shapes"));
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let x = grid8(&mut run, S2, |r, c| (r * 8 + c) as f64);
    let (y1, y2) = (
        grid8(&mut run, S2, |_, _| 0.0),
        grid8(&mut run, S2, |_, _| 0.0),
    );
    // 4 tiles of [4, 4] and 16 of [2, 2] are the same 64 elements
    let coarse = [Arg::Tensor(x), Arg::Tensor(y1), Arg::Scalar(u32s(4))];
    assert_eq!(
        run.invoke("Shaped.coarse", &coarse, (0, 2), &[]),
        End::Completed(None)
    );
    let fine = [Arg::Tensor(x), Arg::Tensor(y2), Arg::Scalar(u32s(16))];
    assert_eq!(
        run.invoke("Shaped.fine", &fine, (0, 2), &[]),
        End::Completed(None)
    );
    let want: Vec<f64> = (0..64).map(|k| 2.0 * k as f64).collect();
    assert_eq!(elements8(&run, y1), want);
    assert_eq!(elements8(&run, y2), want);
}

#[test]
fn a_write_in_one_shape_is_read_in_another() {
    let m = module(&construct("c05_tile_shapes"));
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let x = grid8(&mut run, S2, |r, c| (r * 8 + c) as f64);
    let y = grid8(&mut run, S2, |_, _| 0.0);
    let z = grid8(&mut run, S2, |_, _| 0.0);
    let fine = [Arg::Tensor(x), Arg::Tensor(y), Arg::Scalar(u32s(16))];
    assert_eq!(
        run.invoke("Shaped.fine", &fine, (0, 2), &[]),
        End::Completed(None)
    );
    let coarse = [Arg::Tensor(y), Arg::Tensor(z), Arg::Scalar(u32s(4))];
    assert_eq!(
        run.invoke("Shaped.coarse", &coarse, (0, 2), &[]),
        End::Completed(None)
    );
    let want: Vec<f64> = (0..64).map(|k| 4.0 * k as f64).collect();
    assert_eq!(elements8(&run, z), want);
}

/// Two handlers of two shapes, the tensors stored in the default `[4, 4]`: a tile of
/// `fine` (`[2, 2]`) is part of a stored tile, one place for each of its rows.
const SHAPES_RACE: &str = "module shapes_race

tile [4, 4]
tile [2, 2]
format f32

program P {
    handler fine(x: tensor<f32>, y: tensor<f32>, first: u32, tiles: u32) @on(group) @tile(2, 2) {
        let out = sink(partition(y.tiles(first, tiles), worker.rank, group.size), block = 1)
        for xb in stream(partition(x.tiles(first, tiles), worker.rank, group.size), block = 1) {
            out.push(xb + xb)
        }
    }

    handler coarse(x: tensor<f32>, y: tensor<f32>, first: u32, tiles: u32) @on(group) {
        let out = sink(partition(y.tiles(first, tiles), worker.rank, group.size), block = 1)
        for xb in stream(partition(x.tiles(first, tiles), worker.rank, group.size), block = 1) {
            out.push(xb + xb)
        }
    }
}

// rows 0 and 1 in tiles of [2, 2]; rows 0 to 3, columns 0 to 3, in one tile of [4, 4]
control overlap(x: tensor<f32>, y: tensor<f32>) {
    let a = invoke(P.fine(x, y, 0, 4), on = group(0, 1))
    let b = invoke(P.coarse(x, y, 0, 1), on = group(1, 1))
    await a
    await b
}

// rows 4 to 7 in tiles of [2, 2]; rows 0 to 3 in tiles of [4, 4]
control apart(x: tensor<f32>, y: tensor<f32>) {
    let a = invoke(P.fine(x, y, 8, 8), on = group(0, 1))
    let b = invoke(P.coarse(x, y, 0, 2), on = group(1, 1))
    await a
    await b
}
";

#[test]
fn two_shapes_on_the_same_elements_with_no_order_race() {
    // the build compares no views of two shapes: the run finds the race
    let m = module(SHAPES_RACE);
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let shape = TileShape { rows: 4, cols: 4 };
    let x = grid8(&mut run, shape, |r, c| (r * 8 + c) as f64);
    let y = grid8(&mut run, shape, |_, _| 0.0);
    let what = invalid_what(run.control("overlap", &[Arg::Tensor(x), Arg::Tensor(y)]));
    assert!(
        what.starts_with("a race on tensor 1 tile 0: a write at #0 rank 0 (p0.h0 i"),
        "{what}"
    );
    assert!(what.contains("and a write at #1 rank 0 (p0.h1 i"), "{what}");
}

#[test]
fn two_shapes_on_separate_elements_do_not_race() {
    let m = module(SHAPES_RACE);
    let mut run = Machine::load(&m, 2, &[], Options::default()).expect("loads");
    let shape = TileShape { rows: 4, cols: 4 };
    let x = grid8(&mut run, shape, |r, c| (r * 8 + c) as f64);
    let y = grid8(&mut run, shape, |_, _| 0.0);
    assert_eq!(
        run.control("apart", &[Arg::Tensor(x), Arg::Tensor(y)]),
        End::Completed(None)
    );
    let want: Vec<f64> = (0..64).map(|k| 2.0 * k as f64).collect();
    assert_eq!(elements8(&run, y), want);
}

#[test]
fn a_tensor_not_whole_tiles_of_a_shape_is_invalid() {
    let m = module(&construct("c05_tile_shapes"));
    let mut run = Machine::load(&m, 1, &[], Options::default()).expect("loads");
    // 6 by 6 is whole tiles of [2, 2], not of [4, 4]
    let x = run.tensor(Tensor::zeros("x", Format::F32, 6, 6, S2));
    let y = run.tensor(Tensor::zeros("y", Format::F32, 6, 6, S2));
    let args = [Arg::Tensor(x), Arg::Tensor(y), Arg::Scalar(u32s(1))];
    assert_eq!(
        run.invoke("Shaped.coarse", &args, (0, 1), &[]),
        End::Invalid {
            what: "a tensor whose padded shape is not whole tiles of [4, 4]".into(),
            at: None,
        }
    );
    let args = [Arg::Tensor(x), Arg::Tensor(y), Arg::Scalar(u32s(9))];
    assert_eq!(
        run.invoke("Shaped.fine", &args, (0, 1), &[]),
        End::Completed(None)
    );
}

#[test]
fn all_gather_between_shapes_copies_the_elements_written() {
    // stored in [4, 4]: a write in tiles of [2, 2] marks parts of stored tiles, row by
    // row, and `all_gather` copies those elements only: two islands' writes to two
    // halves of one stored tile both reach every copy
    let m = module(
        "module gathered_shapes

tile [4, 4]
tile [2, 2]
format f32

requires {
    islands >= 2
}

program G {
    handler put(x: tensor<f32>, w: tensor<f32> @replicated, first: u32, tiles: u32) @on(group) @tile(2, 2) {
        let out = sink(partition(w.tiles(first, tiles), worker.rank, group.size), block = 1)
        for xb in stream(partition(x.tiles(first, tiles), worker.rank, group.size), block = 1) {
            out.push(xb)
        }
    }

    handler take(w: tensor<f32> @replicated, y: tensor<f32>, tiles: u32) @on(group) {
        let out = sink(partition(y.tiles(0, tiles), worker.rank, group.size), block = 1)
        for wb in stream(partition(w.tiles(0, tiles), worker.rank, group.size), block = 1) {
            out.push(wb)
        }
    }
}

control spread(w: tensor<f32> @replicated) {
    all_gather(w)
}
",
    );
    let mut run = Machine::load(&m, 2, &[], on_islands(&[1, 1], None)).expect("loads");
    let shape = TileShape { rows: 4, cols: 4 };
    let x = grid8(&mut run, shape, |r, c| (r * 8 + c) as f64);
    let y = grid8(&mut run, shape, |_, _| -1.0);
    let w = run.replicated(Tensor::zeros("w", Format::F32, 8, 8, shape));
    // island 0 writes tiles 0 and 1 of [2, 2]: rows 0 and 1, columns 0 to 3; island 1
    // tiles 4 and 5: rows 2 and 3, columns 0 to 3; both in stored tile 0
    for (first, worker) in [(0, 0), (4, 1)] {
        let put = [
            Arg::Tensor(x),
            Arg::Tensor(w),
            Arg::Scalar(u32s(first)),
            Arg::Scalar(u32s(2)),
        ];
        assert_eq!(
            run.invoke("G.put", &put, (worker, 1), &[]),
            End::Completed(None)
        );
    }
    assert_eq!(
        run.control("spread", &[Arg::Tensor(w)]),
        End::Completed(None)
    );
    // island 1 reads its copy in tiles of [4, 4]
    let take = [Arg::Tensor(w), Arg::Tensor(y), Arg::Scalar(u32s(4))];
    assert_eq!(
        run.invoke("G.take", &take, (1, 1), &[]),
        End::Completed(None)
    );
    let want: Vec<f64> = (0..64)
        .map(|k| {
            if k / 8 < 4 && k % 8 < 4 {
                k as f64
            } else {
                0.0
            }
        })
        .collect();
    assert_eq!(elements8(&run, y), want);
    for island in 0..2 {
        assert_eq!(
            elements8(&run, run.copy(w, island).expect("a copy")),
            want,
            "copy {island}"
        );
    }
}
