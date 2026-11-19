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
