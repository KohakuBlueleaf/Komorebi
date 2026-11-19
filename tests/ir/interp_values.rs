//! The interpreter's values and operations, tested alone: for each operation a few
//! cases whose numbers are written by hand, and the laws as properties over generated
//! sizes.

use komorebi_ir::interp::memory::{
    Block, Failure, Fault, Format, Heap, Member, RecordType, Scalar, Steps, Tensor, TensorId, Tile,
    TileShape, View,
};
use komorebi_ir::interp::ops::{
    self, Axis, Bin, Cmp, Element, Operand, Reduce, ReducedOperand, Unary,
};

const S22: TileShape = TileShape { rows: 2, cols: 2 };

fn f32s(values: &[f64]) -> Tile {
    Tile::from_f64(Format::F32, S22, values).expect("four values")
}

fn tile_of(format: Format, values: &[f64]) -> Tile {
    Tile::from_f64(format, S22, values).expect("four values")
}

fn ints(format: Format, values: &[i128]) -> Tile {
    Tile::from_ints(format, S22, values).expect("four values")
}

fn u32s(v: u64) -> Scalar {
    Scalar::Int(Format::U32, v)
}

fn int(format: Format, v: i128) -> Scalar {
    Scalar::int(format, v).expect("an integer format")
}

fn values(t: &Tile) -> Vec<f64> {
    t.to_f64().expect("a float tile")
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

const VIEW_END: Fault = Fault::Failure(Failure::ViewEnd);
const EMPTY: Fault = Fault::Failure(Failure::EmptyState);
const OUT: Fault = Fault::Failure(Failure::OutOfRange);
const DIV0: Fault = Fault::Failure(Failure::DivideByZero);

/// A heap with one f32 tensor of `rows` by `cols` tiles of 2 by 2, tile `i` holding
/// `i` in every element.
fn numbered(rows: usize, cols: usize) -> (Heap, TensorId) {
    let mut t = Tensor::zeros("t", Format::F32, rows * 2, cols * 2, S22);
    for (i, tile) in t.tiles.iter_mut().enumerate() {
        *tile = tile_of(Format::F32, &[i as f64; 4]);
    }
    let mut heap = Heap::default();
    let id = heap.add(t);
    (heap, id)
}

/// The number each tile of a view holds, in the view's order.
fn numbers_of(heap: &Heap, v: &View) -> Vec<usize> {
    (0..v.len())
        .map(|i| values(&ops::read(heap, v, i).expect("inside the view"))[0] as usize)
        .collect()
}

// ---- Scalars ----

#[test]
fn integers_wrap() {
    let add = ops::scalar_binary(Bin::Add, int(Format::U8, 250), int(Format::U8, 10));
    assert_eq!(add, Ok(int(Format::U8, 4)));
    let sub = ops::scalar_binary(Bin::Sub, u32s(1), u32s(2));
    assert_eq!(sub, Ok(u32s(0xffff_ffff)));
    let mul = ops::scalar_binary(Bin::Mul, int(Format::I8, 100), int(Format::I8, 3));
    assert_eq!(mul.map(|s| s.as_i128()), Ok(Some(44))); // 300 - 256
}

#[test]
fn integer_division_rounds_toward_zero() {
    let div = |a, b| ops::scalar_binary(Bin::Div, int(Format::I32, a), int(Format::I32, b));
    let rem = |a, b| ops::scalar_binary(Bin::Rem, int(Format::I32, a), int(Format::I32, b));
    assert_eq!(div(7, 2), Ok(int(Format::I32, 3)));
    assert_eq!(div(-7, 2), Ok(int(Format::I32, -3)));
    assert_eq!(rem(-7, 2), Ok(int(Format::I32, -1)));
    assert_eq!(rem(7, -2), Ok(int(Format::I32, 1)));
    // the one quotient out of range wraps
    let min = ops::scalar_binary(Bin::Div, int(Format::I8, -128), int(Format::I8, -1));
    assert_eq!(min, Ok(int(Format::I8, -128)));
}

#[test]
fn integer_division_by_zero_is_a_fault() {
    for op in [Bin::Div, Bin::Rem] {
        assert_eq!(ops::scalar_binary(op, u32s(5), u32s(0)), Err(DIV0));
    }
    let t = ints(Format::U32, &[1, 2, 3, 4]);
    let z = ints(Format::U32, &[1, 1, 0, 1]);
    let r = ops::binary(Bin::Div, Operand::Tile(&t), Operand::Tile(&z));
    assert_eq!(r, Err(DIV0));
}

#[test]
fn bits_and_shifts() {
    let b = |op, x, y| ops::scalar_binary(op, u32s(x), u32s(y));
    assert_eq!(b(Bin::And, 0b1100, 0b1010), Ok(u32s(0b1000)));
    assert_eq!(b(Bin::Or, 0b1100, 0b1010), Ok(u32s(0b1110)));
    assert_eq!(b(Bin::Xor, 0b1100, 0b1010), Ok(u32s(0b0110)));
    assert_eq!(b(Bin::Shl, 0x8000_0001, 1), Ok(u32s(2)));
    assert_eq!(b(Bin::Shr, 0x8000_0000, 31), Ok(u32s(1)));
    assert_eq!(b(Bin::Shl, 1, 32), Ok(u32s(0)));
    let signed = ops::scalar_binary(Bin::Shr, int(Format::I16, -8), int(Format::I16, 1));
    assert_eq!(signed, Ok(int(Format::I16, -4)));
    let negative = ops::scalar_binary(Bin::Shl, int(Format::I32, 1), int(Format::I32, -1));
    assert_eq!(negative, Err(OUT));
    let out = ops::scalar_binary(Bin::Shr, int(Format::I16, -8), int(Format::I16, 40));
    assert_eq!(out, Ok(int(Format::I16, -1)));
    // T2's checksum step: ((sum << 1) | (sum >> 15)) ^ w
    let sum = 0x0000_8001u64;
    let step = ((sum << 1) | (sum >> 15)) ^ 0x10;
    let l = b(Bin::Shl, sum, 1).unwrap();
    let r = b(Bin::Shr, sum, 15).unwrap();
    let or = ops::scalar_binary(Bin::Or, l, r).unwrap();
    assert_eq!(ops::scalar_binary(Bin::Xor, or, u32s(0x10)), Ok(u32s(step)));
}

#[test]
fn min_and_max() {
    assert_eq!(
        ops::scalar_binary(Bin::Min, u32s(7), u32s(4096)),
        Ok(u32s(7))
    );
    let max = ops::scalar_binary(Bin::Max, int(Format::I32, -3), int(Format::I32, 2));
    assert_eq!(max, Ok(int(Format::I32, 2)));
    let fmin = ops::scalar_binary(
        Bin::Min,
        Scalar::F32(Format::F32, 1.5),
        Scalar::F32(Format::F32, -2.0),
    );
    assert_eq!(fmin, Ok(Scalar::F32(Format::F32, -2.0)));
}

#[test]
fn comparisons() {
    assert_eq!(
        ops::compare(Cmp::Lt, int(Format::I32, -1), int(Format::I32, 0)),
        Ok(true)
    );
    assert_eq!(ops::compare(Cmp::Gt, u32s(0xffff_ffff), u32s(0)), Ok(true));
    assert_eq!(
        ops::compare(Cmp::Eq, Scalar::Bool(false), Scalar::Bool(false)),
        Ok(true)
    );
    assert_eq!(
        ops::compare(Cmp::Ne, Scalar::Bool(true), Scalar::Bool(false)),
        Ok(true)
    );
    assert!(ops::compare(Cmp::Lt, Scalar::Bool(true), Scalar::Bool(false)).is_err());
    let mixed = ops::compare(Cmp::Ge, Scalar::F32(Format::Bf16, 2.0), Scalar::F64(2.0));
    assert_eq!(mixed, Ok(true));
    assert!(ops::compare(Cmp::Eq, u32s(1), Scalar::F64(1.0)).is_err());
}

// ---- Compute on tiles ----

#[test]
fn tile_arithmetic() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let b = f32s(&[0.5, -1.0, 2.0, 8.0]);
    let (ta, tb) = (Operand::Tile(&a), Operand::Tile(&b));
    let r = |op| values(&ops::binary(op, ta, tb).unwrap());
    assert_eq!(r(Bin::Add), vec![1.5, 1.0, 5.0, 12.0]);
    assert_eq!(r(Bin::Sub), vec![0.5, 3.0, 1.0, -4.0]);
    assert_eq!(r(Bin::Mul), vec![0.5, -2.0, 6.0, 32.0]);
    assert_eq!(r(Bin::Div), vec![2.0, -2.0, 1.5, 0.5]);
    // T1: a * k with a scalar k
    let k = Operand::Scalar(Scalar::F32(Format::F32, 3.0));
    let bf = tile_of(Format::Bf16, &[1.0, 2.0, 3.0, 4.0]);
    let times = ops::binary(Bin::Mul, Operand::Tile(&bf), k).unwrap();
    assert_eq!(times.format, Format::Bf16);
    assert_eq!(values(&times), vec![3.0, 6.0, 9.0, 12.0]);
    // a float tile divided by zero is IEEE's
    let z = f32s(&[0.0; 4]);
    let inf = values(&ops::binary(Bin::Div, ta, Operand::Tile(&z)).unwrap());
    assert!(inf.iter().all(|x| x.is_infinite()));
}

#[test]
fn integer_tiles_wrap() {
    let a = ints(Format::U8, &[200, 1, 7, 0]);
    let b = ints(Format::U8, &[100, 2, 2, 1]);
    let sum = ops::binary(Bin::Add, Operand::Tile(&a), Operand::Tile(&b)).unwrap();
    assert_eq!(sum.to_ints(), Some(vec![44, 3, 9, 1]));
    let diff = ops::binary(Bin::Sub, Operand::Tile(&a), Operand::Tile(&b)).unwrap();
    assert_eq!(diff.to_ints(), Some(vec![100, 255, 5, 255]));
    let q = ops::binary(Bin::Div, Operand::Tile(&a), Operand::Tile(&b)).unwrap();
    assert_eq!(q.to_ints(), Some(vec![2, 0, 3, 0]));
    let f = f32s(&[1.0; 4]);
    assert!(ops::binary(Bin::Add, Operand::Tile(&a), Operand::Tile(&f)).is_err());
    // an integer tile takes no float scalar
    let half = Operand::Scalar(Scalar::F32(Format::F32, 0.5));
    assert!(matches!(
        ops::binary(Bin::Mul, Operand::Tile(&a), half),
        Err(Fault::Undefined(_))
    ));
    // a float tile takes an integer scalar by its value: `x / tiles`
    let x = f32s(&[3.0, 6.0, 9.0, 1.5]);
    let tiles = Operand::Scalar(u32s(3));
    assert_eq!(
        values(&ops::binary(Bin::Div, Operand::Tile(&x), tiles).unwrap()),
        vec![1.0, 2.0, 3.0, 0.5]
    );
}

#[test]
fn mixed_formats() {
    let a = tile_of(Format::Bf16, &[1.0, 2.0, 3.0, 4.0]);
    let b = tile_of(Format::Bfp8, &[0.25, 0.25, 0.25, 0.25]);
    let r = ops::binary(Bin::Add, Operand::Tile(&a), Operand::Tile(&b)).unwrap();
    assert_eq!(r.format, Format::Bf16);
    assert_eq!(values(&r), vec![1.25, 2.25, 3.25, 4.25]);
    let w = tile_of(Format::F64, &[1e-10, 0.0, 0.0, 0.0]);
    let wide = ops::binary(Bin::Add, Operand::Tile(&a), Operand::Tile(&w)).unwrap();
    assert_eq!(wide.format, Format::F64);
    assert_eq!(values(&wide)[0], 1.0 + 1e-10);
}

#[test]
fn elementwise_functions() {
    let a = f32s(&[-1.5, 0.0, 2.25, -0.5]);
    let r = |f| values(&ops::unary(f, &a).unwrap());
    assert_eq!(r(Unary::Neg), vec![1.5, 0.0, -2.25, 0.5]);
    assert_eq!(r(Unary::Abs), vec![1.5, 0.0, 2.25, 0.5]);
    assert_eq!(r(Unary::Relu), vec![0.0, 0.0, 2.25, 0.0]);
    assert_eq!(r(Unary::Square), vec![2.25, 0.0, 5.0625, 0.25]);
    assert_eq!(r(Unary::Floor), vec![-2.0, 0.0, 2.0, -1.0]);
    let i = ints(Format::I8, &[-128, -3, 5, 127]);
    let neg = ops::unary(Unary::Neg, &i).unwrap();
    assert_eq!(neg.to_ints(), Some(vec![-128, 3, -5, -127]));
    let sq = ops::unary(Unary::Square, &i).unwrap();
    assert_eq!(sq.to_ints(), Some(vec![0, 9, 25, 1])); // 16384 and 16129 wrap
    assert!(ops::unary(Unary::Exp, &i).is_err());
}

#[test]
fn elementary_functions() {
    let a = f32s(&[0.0, 1.0, 4.0, 0.25]);
    let r = |f| values(&ops::unary(f, &a).unwrap());
    let e = std::f64::consts::E;
    assert!(close(
        &r(Unary::Exp),
        &[1.0, e, e.powi(4), e.powf(0.25)],
        1e-5
    ));
    assert!(close(
        &r(Unary::Exp2),
        &[1.0, 2.0, 16.0, 2f64.powf(0.25)],
        1e-6
    ));
    assert!(close(
        &r(Unary::Tanh),
        &[0.0, 0.761_594_2, 0.999_329_3, 0.244_918_7],
        1e-6
    ));
    assert!(close(&r(Unary::Sqrt), &[0.0, 1.0, 2.0, 0.5], 0.0));
    assert!(close(&r(Unary::Rsqrt)[1..], &[1.0, 0.5, 2.0], 0.0));
    assert!(close(&r(Unary::Reciprocal)[1..], &[1.0, 0.25, 4.0], 0.0));
    assert!(close(
        &r(Unary::Log)[1..],
        &[0.0, 4f64.ln(), 0.25f64.ln()],
        1e-6
    ));
    assert!(close(
        &r(Unary::Sigmoid),
        &[0.5, 0.731_058_6, 0.982_013_8, 0.562_176_5],
        1e-6
    ));
    assert!(close(
        &r(Unary::Silu),
        &[0.0, 0.731_058_6, 3.928_055_2, 0.140_544_1],
        1e-6
    ));
    assert!(close(
        &r(Unary::Erf),
        &[0.0, 0.842_700_8, 1.0, 0.276_326_4],
        1e-6
    ));
}

#[test]
fn erf_values() {
    // from tables of the error function
    let cases = [
        (0.1, 0.112_462_916_018_284_9),
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (2.0, 0.995_322_265_018_952_7),
        (2.5, 0.999_593_047_982_555),
        (3.0, 0.999_977_909_503_001_4),
        (-1.5, -0.966_105_146_475_310_7),
    ];
    for (x, want) in cases {
        let got = ops::erf(x);
        assert!((got - want).abs() < 1e-13, "erf({x}) = {got}, not {want}");
    }
    assert_eq!(ops::erf(10.0), 1.0);
}

#[test]
fn maximum_minimum() {
    let a = f32s(&[-1.0, 0.5, 2.0, 0.0]);
    let zero = Operand::Scalar(Scalar::F32(Format::F32, 0.0));
    let one = Operand::Scalar(Scalar::F32(Format::F32, 1.0));
    // T4's clamp01: minimum(maximum(a, 0.0), 1.0)
    let lo = ops::binary(Bin::Max, Operand::Tile(&a), zero).unwrap();
    let both = ops::binary(Bin::Min, Operand::Tile(&lo), one).unwrap();
    assert_eq!(values(&both), vec![0.0, 0.5, 1.0, 0.0]);
    let b = f32s(&[0.0, 1.0, 1.0, -1.0]);
    let max = ops::binary(Bin::Max, Operand::Tile(&a), Operand::Tile(&b)).unwrap();
    assert_eq!(values(&max), vec![0.0, 1.0, 2.0, 0.0]);
}

#[test]
fn where_selects() {
    // T4's leaky: where(a > 0.0, a, a * 0.01)
    let a = f32s(&[-2.0, 3.0, 0.0, 1.0]);
    let small = ops::binary(
        Bin::Mul,
        Operand::Tile(&a),
        Operand::Scalar(Scalar::F32(Format::F32, 0.5)),
    )
    .unwrap();
    let zero = Operand::Scalar(Scalar::F32(Format::F32, 0.0));
    let r = ops::select(
        Cmp::Gt,
        Operand::Tile(&a),
        zero,
        Operand::Tile(&a),
        Operand::Tile(&small),
    );
    assert_eq!(values(&r.unwrap()), vec![-1.0, 3.0, 0.0, 1.0]);
    // scalars for the two values
    let b = f32s(&[1.0, 5.0, 5.0, 9.0]);
    let r = ops::select(
        Cmp::Eq,
        Operand::Tile(&b),
        Operand::Scalar(Scalar::F32(Format::F32, 5.0)),
        Operand::Scalar(Scalar::F32(Format::F32, 1.0)),
        Operand::Scalar(Scalar::F32(Format::F32, 0.0)),
    );
    assert_eq!(values(&r.unwrap()), vec![0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn reduce_along_an_axis() {
    let a = f32s(&[1.0, 2.0, 3.0, 5.0]);
    let sum_rows = ops::reduce(Reduce::Sum, &a, Axis::Row).unwrap();
    assert_eq!(sum_rows.axis, Axis::Row);
    assert_eq!(
        values(&ops::col(&sum_rows).unwrap()),
        vec![3.0, 3.0, 8.0, 8.0]
    );
    let max_cols = ops::reduce(Reduce::Max, &a, Axis::Col).unwrap();
    assert_eq!(
        values(&ops::row(&max_cols).unwrap()),
        vec![3.0, 5.0, 3.0, 5.0]
    );
    let min_rows = ops::reduce(Reduce::Min, &a, Axis::Row).unwrap();
    assert_eq!(
        values(&ops::col(&min_rows).unwrap()),
        vec![1.0, 1.0, 3.0, 3.0]
    );
    let i = ints(Format::U8, &[200, 100, 1, 2]);
    let wrap = ops::reduce(Reduce::Sum, &i, Axis::Row).unwrap();
    assert_eq!(ops::col(&wrap).unwrap().to_ints(), Some(vec![44, 44, 3, 3]));
    // `.col()` belongs to a value for each row, `.row()` to one for each column
    assert!(matches!(ops::row(&sum_rows), Err(Fault::Undefined(_))));
    assert!(matches!(ops::col(&max_cols), Err(Fault::Undefined(_))));
}

#[test]
fn reduced_values_are_not_tiles() {
    let a = f32s(&[1.0, 2.0, 3.0, 5.0]);
    let rows = ops::reduce(Reduce::Sum, &a, Axis::Row).unwrap();
    let cols = ops::reduce(Reduce::Sum, &a, Axis::Col).unwrap();
    let r = |v| ReducedOperand::Reduced(v);
    let half = ReducedOperand::Scalar(Scalar::F32(Format::F32, 0.5));
    // with a scalar, and with another of its kind
    let scaled = ops::reduced_binary(Bin::Mul, r(&rows), half).unwrap();
    assert_eq!(
        values(&ops::col(&scaled).unwrap()),
        vec![1.5, 1.5, 4.0, 4.0]
    );
    let twice = ops::reduced_binary(Bin::Add, r(&rows), r(&rows)).unwrap();
    assert_eq!(
        values(&ops::col(&twice).unwrap()),
        vec![6.0, 6.0, 16.0, 16.0]
    );
    let root = ops::reduced_unary(Unary::Sqrt, &twice).unwrap();
    assert_eq!(values(&ops::col(&root).unwrap())[2], 4.0);
    // a value for each row with one for each column is not defined
    assert!(matches!(
        ops::reduced_binary(Bin::Add, r(&rows), r(&cols)),
        Err(Fault::Undefined(_))
    ));
}

#[test]
fn spread_col_and_row() {
    let v = f32s(&[3.0, 9.0, 8.0, 9.0]);
    assert_eq!(
        values(&ops::spread_col(&v).unwrap()),
        vec![3.0, 3.0, 8.0, 8.0]
    );
    assert_eq!(
        values(&ops::spread_row(&v).unwrap()),
        vec![3.0, 9.0, 3.0, 9.0]
    );
    // T4's rms_norm on one row of [3, 4]: a * rsqrt(mean(a * a)).col()
    let a = f32s(&[3.0, 4.0, 0.0, 0.0]);
    let sq = ops::binary(Bin::Mul, Operand::Tile(&a), Operand::Tile(&a)).unwrap();
    let ms = ops::reduce(Reduce::Sum, &sq, Axis::Row).unwrap();
    let mean = ops::reduced_binary(
        Bin::Mul,
        ReducedOperand::Reduced(&ms),
        ReducedOperand::Scalar(Scalar::F32(Format::F32, 0.5)),
    )
    .unwrap();
    let r = ops::reduced_unary(Unary::Rsqrt, &mean).unwrap();
    let n = ops::binary(
        Bin::Mul,
        Operand::Tile(&a),
        Operand::Tile(&ops::col(&r).unwrap()),
    )
    .unwrap();
    let want = [3.0 / 12.5f64.sqrt(), 4.0 / 12.5f64.sqrt()];
    assert!(close(&values(&n)[..2], &want, 1e-6));
}

#[test]
fn matmul_by_hand() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let b = f32s(&[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        values(&ops::matmul(&a, &b).unwrap()),
        vec![19.0, 22.0, 43.0, 50.0]
    );
    // T4's gram: matmul(a, a.T)
    let g = ops::matmul(&a, &ops::transpose(&a).unwrap()).unwrap();
    assert_eq!(values(&g), vec![5.0, 11.0, 11.0, 25.0]);
    let row = Tile::from_f64(Format::F32, TileShape { rows: 1, cols: 2 }, &[1.0, 1.0]).unwrap();
    assert_eq!(values(&ops::matmul(&row, &a).unwrap()), vec![4.0, 6.0]);
    assert!(ops::matmul(&a, &row).is_err());
    let i = ints(Format::U8, &[16, 0, 0, 1]);
    assert_eq!(
        ops::matmul(&i, &i).unwrap().to_ints(),
        Some(vec![0, 0, 0, 1])
    );
}

#[test]
fn tile_transpose() {
    let a = Tile::from_f64(
        Format::F32,
        TileShape { rows: 2, cols: 3 },
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let t = ops::transpose(&a).unwrap();
    assert_eq!(t.shape, TileShape { rows: 3, cols: 2 });
    assert_eq!(values(&t), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(ops::transpose(&t).unwrap(), a);
}

#[test]
fn zeros_tile() {
    assert_eq!(values(&ops::zeros(Format::Bfp8, S22)), vec![0.0; 4]);
    assert_eq!(ops::zeros(Format::I16, S22).to_ints(), Some(vec![0; 4]));
}

#[test]
fn writing_another_format_keeps_values() {
    // no rounding to bf16's precision: 1 + 2^-20 stays as it is
    let fine = 1.0 + 2f64.powi(-20);
    let a = f32s(&[fine, 0.0, 0.0, 0.0]);
    let mut heap = Heap::default();
    let t = heap.add(Tensor::zeros("y", Format::Bf16, 2, 2, S22));
    let v = ops::tiles(&heap, t, 0, 1).unwrap();
    ops::store(&mut heap, &v, &Block { tiles: vec![a] }).unwrap();
    let back = ops::read(&heap, &v, 0).unwrap();
    assert_eq!(back.format, Format::Bf16);
    assert_eq!(values(&back)[0], fine);
    let i = ints(Format::U32, &[1, 2, 3, 4]);
    assert!(ops::convert(&i, Format::F32).is_err());
    assert_eq!(
        ops::convert(&ints(Format::U16, &[65535, 1, 2, 3]), Format::I16)
            .unwrap()
            .to_ints(),
        Some(vec![-1, 1, 2, 3])
    );
}

// ---- Views of a tensor ----

#[test]
fn tiles_view_and_view_end() {
    let (heap, t) = numbered(2, 3);
    let v = ops::tiles(&heap, t, 2, 3).unwrap();
    assert_eq!(numbers_of(&heap, &v), vec![2, 3, 4]);
    assert_eq!(ops::tiles(&heap, t, 4, 2).map(|v| v.len()), Ok(2));
    assert_eq!(ops::tiles(&heap, t, 5, 2), Err(VIEW_END));
    assert_eq!(ops::tiles(&heap, t, usize::MAX, 2), Err(VIEW_END));
}

#[test]
fn rectangle_walk() {
    // a grid of 3 by 4 tiles, numbered row by row
    let (heap, t) = numbered(3, 4);
    let s = |start, end, step| Steps { start, end, step };
    let v = ops::rect(&heap, t, s(1, 3, 1), s(0, 4, 2)).unwrap();
    assert_eq!(numbers_of(&heap, &v), vec![4, 6, 8, 10]);
    let w = ops::rect(&heap, t, s(0, 3, 2), s(1, 4, 1)).unwrap();
    assert_eq!(numbers_of(&heap, &w), vec![1, 2, 3, 9, 10, 11]);
    assert_eq!(ops::rect(&heap, t, s(0, 4, 1), s(0, 1, 1)), Err(VIEW_END));
    assert!(ops::rect(&heap, t, s(0, 2, 0), s(0, 1, 1)).is_err());
}

#[test]
fn turned_rectangle() {
    let (heap, t) = numbered(3, 4);
    let s = |start, end| Steps {
        start,
        end,
        step: 1,
    };
    let v = ops::rect(&heap, t, s(0, 2), s(1, 4)).unwrap();
    assert_eq!(numbers_of(&heap, &v), vec![1, 2, 3, 5, 6, 7]);
    let turned = ops::turn(&v).unwrap();
    assert_eq!(numbers_of(&heap, &turned), vec![1, 5, 2, 6, 3, 7]);
    let run = ops::tiles(&heap, t, 0, 2).unwrap();
    assert!(ops::turn(&run).is_err());
}

#[test]
fn items_views() {
    let mut heap = Heap::default();
    // u32 tiles of 2 by 2: 16 bytes a tile, 3 tiles: 48 bytes
    let t = heap.add(Tensor::zeros("r", Format::U32, 2, 6, S22));
    assert_eq!(ops::items(&heap, t, None, 16).map(|v| v.len()), Ok(3));
    assert_eq!(ops::items(&heap, t, None, 20).map(|v| v.len()), Ok(2));
    assert_eq!(
        ops::items(&heap, t, Some((1, 2)), 16).map(|v| v.len()),
        Ok(2)
    );
    assert_eq!(ops::items(&heap, t, Some((2, 2)), 16), Err(VIEW_END));
    let f = heap.add(Tensor::zeros("b", Format::Bf16, 2, 2, S22));
    assert!(matches!(
        ops::items(&heap, f, None, 16),
        Err(Fault::Undefined(_))
    ));
}

#[test]
fn read_one_tile() {
    let (heap, t) = numbered(1, 4);
    let v = ops::tiles(&heap, t, 1, 2).unwrap();
    assert_eq!(values(&ops::read(&heap, &v, 1).unwrap()), vec![2.0; 4]);
    assert_eq!(ops::read(&heap, &v, 2), Err(OUT));
}

#[test]
fn words_of_one_tile() {
    let mut heap = Heap::default();
    let t = heap.add(Tensor::zeros("x", Format::U32, 2, 4, S22));
    heap.tensors[t.0].tiles[1] = ints(Format::U32, &[7, 8, 9, 10]);
    let v = ops::tiles(&heap, t, 1, 1).unwrap();
    assert_eq!(
        ops::words(&heap, &v),
        Ok(vec![u32s(7), u32s(8), u32s(9), u32s(10)])
    );
    let b = heap.add(Tensor::zeros("b", Format::Bf16, 2, 2, S22));
    let bv = ops::tiles(&heap, b, 0, 1).unwrap();
    assert!(ops::words(&heap, &bv).is_err());
}

#[test]
fn records_of_one_tile() {
    // T7's Node { lo: u32, hi: u32, child: u32 }, from a tile of 3 by 4 u32 words
    let shape = TileShape { rows: 3, cols: 4 };
    let mut heap = Heap::default();
    let t = heap.add(Tensor::zeros("table", Format::U32, 3, 4, shape));
    let words: Vec<i128> = (0..12).map(|i| i * 10).collect();
    heap.tensors[t.0].tiles[0] = Tile::from_ints(Format::U32, shape, &words).unwrap();
    let node = RecordType {
        name: "Node".into(),
        fields: vec![
            ("lo".into(), Format::U32),
            ("hi".into(), Format::U32),
            ("child".into(), Format::U32),
        ],
    };
    let v = ops::tiles(&heap, t, 0, 1).unwrap();
    let recs = ops::records(&heap, &v, &node, 4).unwrap();
    assert_eq!(recs[0], vec![u32s(0), u32s(10), u32s(20)]);
    assert_eq!(recs[3], vec![u32s(90), u32s(100), u32s(110)]);
    assert_eq!(ops::records(&heap, &v, &node, 5), Err(OUT));
    // fields of two widths, with no gap: u8 then u16 then u8 over bytes 0, 1, 2, ...
    let mut bytes_heap = Heap::default();
    let b = bytes_heap.add(Tensor::zeros("bytes", Format::U8, 2, 2, S22));
    bytes_heap.tensors[b.0].tiles[0] = ints(Format::U8, &[1, 2, 3, 4]);
    let mixed = RecordType {
        name: "M".into(),
        fields: vec![
            ("a".into(), Format::U8),
            ("b".into(), Format::U16),
            ("c".into(), Format::U8),
        ],
    };
    let bv = ops::tiles(&bytes_heap, b, 0, 1).unwrap();
    let got = ops::records(&bytes_heap, &bv, &mixed, 1).unwrap();
    assert_eq!(
        got[0],
        vec![
            int(Format::U8, 1),
            int(Format::U16, 0x0302),
            int(Format::U8, 4)
        ]
    );
}

#[test]
fn put_writes_one_item() {
    // T7's lookup: hit (u32[4], by element) put as one 16-byte item at the rank
    let mut heap = Heap::default();
    let t = heap.add(Tensor::zeros("results", Format::U32, 2, 4, S22));
    let mut hit = Member::scalars("hit", Format::U32, 4, true).unwrap();
    let all = ops::items(&heap, t, None, 16).unwrap();
    assert_eq!(ops::put(&mut heap, &all, &hit, 1), Err(EMPTY));
    for (i, v) in [5u64, 6, 7, 1].into_iter().enumerate() {
        ops::set(&mut hit, i, Element::Scalar(u32s(v))).unwrap();
    }
    ops::put(&mut heap, &all, &hit, 1).unwrap();
    assert_eq!(heap.tensors[t.0].tiles[0].to_ints(), Some(vec![0; 4]));
    assert_eq!(heap.tensors[t.0].tiles[1].to_ints(), Some(vec![5, 6, 7, 1]));
    assert_eq!(ops::put(&mut heap, &all, &hit, 2), Err(OUT));
    let wide = ops::items(&heap, t, None, 8).unwrap();
    assert!(ops::put(&mut heap, &wide, &hit, 0).is_err());
}

#[test]
fn partition_by_hand() {
    let (heap, t) = numbered(1, 10);
    let v = ops::tiles(&heap, t, 0, 10).unwrap();
    // 10 tiles over 4 ranks: 3, 3, 2, 2
    let parts: Vec<Vec<usize>> = (0..4)
        .map(|r| numbers_of(&heap, &ops::partition(&v, r, 4).unwrap()))
        .collect();
    assert_eq!(
        parts,
        vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7], vec![8, 9]]
    );
    // 2 tiles over 4 ranks: 1, 1, 0, 0
    let two = ops::tiles(&heap, t, 4, 2).unwrap();
    let lens: Vec<usize> = (0..4)
        .map(|r| ops::partition(&two, r, 4).unwrap().len())
        .collect();
    assert_eq!(lens, vec![1, 1, 0, 0]);
    assert!(ops::partition(&v, 4, 4).is_err());
}

#[test]
fn stream_blocks_and_ragged_end() {
    let (heap, t) = numbered(1, 5);
    let v = ops::tiles(&heap, t, 0, 5).unwrap();
    let mut s = ops::stream(&v, 2).unwrap();
    let mut seen = Vec::new();
    while let Some(block) = ops::next_block(&heap, &mut s).unwrap() {
        seen.push(
            block
                .tiles
                .iter()
                .map(|t| values(t)[0] as usize)
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(seen, vec![vec![0, 1], vec![2, 3], vec![4]]);
    assert!(ops::stream(&v, 0).is_err());
}

#[test]
fn zip_two_streams() {
    let (heap, t) = numbered(1, 6);
    let a = ops::tiles(&heap, t, 0, 3).unwrap();
    let b = ops::tiles(&heap, t, 3, 3).unwrap();
    let (sa, sb) = (ops::stream(&a, 1).unwrap(), ops::stream(&b, 1).unwrap());
    let mut z = ops::zip(sa, sb).unwrap();
    let mut pairs = Vec::new();
    while let Some((x, y)) = ops::next_pair(&heap, &mut z).unwrap() {
        pairs.push((values(&x.tiles[0])[0], values(&y.tiles[0])[0]));
    }
    assert_eq!(pairs, vec![(0.0, 3.0), (1.0, 4.0), (2.0, 5.0)]);
    // streams of different lengths: out_of_range, when the zip is made
    let c = ops::tiles(&heap, t, 0, 2).unwrap();
    let (sa, sc) = (ops::stream(&a, 1).unwrap(), ops::stream(&c, 1).unwrap());
    assert_eq!(ops::zip(sa, sc), Err(OUT));
    // the same number of blocks of different sizes walk together
    let (s2, s3) = (ops::stream(&c, 2).unwrap(), ops::stream(&a, 3).unwrap());
    assert!(ops::zip(s2, s3).is_ok());
}

#[test]
fn sink_writes_in_order() {
    let (mut heap, t) = numbered(1, 4);
    let v = ops::tiles(&heap, t, 1, 3).unwrap();
    let mut out = ops::sink(&v, 2).unwrap();
    let seven = f32s(&[7.0; 4]);
    let block = Block {
        tiles: vec![seven.clone(), seven.clone()],
    };
    ops::push(&mut heap, &mut out, &block).unwrap();
    // the ragged end: a block of two, of which only the one tile named is written
    ops::push(&mut heap, &mut out, &block).unwrap();
    let all = ops::tiles(&heap, t, 0, 4).unwrap();
    assert_eq!(numbers_of(&heap, &all), vec![0, 7, 7, 7]);
    assert_eq!(ops::push(&mut heap, &mut out, &block), Err(VIEW_END));
}

#[test]
fn store_one_tile() {
    let (mut heap, t) = numbered(1, 3);
    let v = ops::tiles(&heap, t, 2, 1).unwrap();
    ops::store(
        &mut heap,
        &v,
        &Block {
            tiles: vec![f32s(&[9.0; 4])],
        },
    )
    .unwrap();
    let all = ops::tiles(&heap, t, 0, 3).unwrap();
    assert_eq!(numbers_of(&heap, &all), vec![0, 1, 9]);
    let two = ops::tiles(&heap, t, 0, 2).unwrap();
    assert!(ops::store(
        &mut heap,
        &two,
        &Block {
            tiles: vec![f32s(&[1.0; 4])]
        }
    )
    .is_err());
}

// ---- State ----

#[test]
fn fill_fills_whole() {
    let (heap, t) = numbered(1, 4);
    let mut table = Member::tiles("table", Format::Bf16, S22, 3, false);
    assert_eq!(ops::get(&table, 0), Err(EMPTY));
    let mut s = ops::stream(&ops::tiles(&heap, t, 1, 3).unwrap(), 1).unwrap();
    ops::fill(&heap, &mut table, &mut s).unwrap();
    let Ok(Element::Tile(second)) = ops::get(&table, 1) else {
        panic!("a tile")
    };
    assert_eq!(
        (second.format, values(&second)),
        (Format::Bf16, vec![2.0; 4])
    );
    let mut short = ops::stream(&ops::tiles(&heap, t, 0, 2).unwrap(), 1).unwrap();
    let mut other = Member::tiles("other", Format::F32, S22, 3, false);
    assert_eq!(ops::fill(&heap, &mut other, &mut short), Err(OUT));
    assert_eq!(ops::get(&other, 0), Err(EMPTY));
}

#[test]
fn zero_fills() {
    let mut calls = Member::scalars("calls", Format::U32, 4, false).unwrap();
    assert_eq!(ops::get(&calls, 2), Err(EMPTY));
    ops::zero(&mut calls);
    assert_eq!(ops::get(&calls, 2), Ok(Element::Scalar(u32s(0))));
    let mut by_element = Member::tiles("w", Format::F32, S22, 2, true);
    ops::zero(&mut by_element);
    assert!(ops::get(&by_element, 1).is_ok());
}

#[test]
fn element_writes() {
    // a member filled whole: a write is allowed and does not fill it
    let mut calls = Member::scalars("calls", Format::U32, 4, false).unwrap();
    ops::set(&mut calls, 1, Element::Scalar(u32s(9))).unwrap();
    assert_eq!(ops::get(&calls, 1), Err(EMPTY));
    // by element: each element readable once written
    let mut hit = Member::scalars("hit", Format::U32, 4, true).unwrap();
    ops::set(&mut hit, 2, Element::Scalar(u32s(5))).unwrap();
    assert_eq!(ops::get(&hit, 2), Ok(Element::Scalar(u32s(5))));
    assert_eq!(ops::get(&hit, 1), Err(EMPTY));
    assert_eq!(ops::get(&hit, 4), Err(OUT));
    assert!(ops::set(&mut hit, 0, Element::Scalar(int(Format::I32, 1))).is_err());
}

#[test]
fn bump_reads_then_adds_one() {
    let mut calls = Member::scalars("calls", Format::U8, 2, false).unwrap();
    assert_eq!(ops::bump(&mut calls, 0), Err(EMPTY));
    ops::zero(&mut calls);
    ops::set(&mut calls, 1, Element::Scalar(int(Format::U8, 255))).unwrap();
    assert_eq!(ops::bump(&mut calls, 0), Ok(int(Format::U8, 0)));
    assert_eq!(ops::get(&calls, 0), Ok(Element::Scalar(int(Format::U8, 1))));
    assert_eq!(ops::bump(&mut calls, 1), Ok(int(Format::U8, 255)));
    assert_eq!(ops::get(&calls, 1), Ok(Element::Scalar(int(Format::U8, 0))));
}

// ---- Laws, over generated sizes ----

/// Views of every kind over a grid: runs, rectangles with steps, turned ones, items.
fn views_of(heap: &Heap, t: TensorId, items_t: TensorId) -> Vec<View> {
    let mut out = Vec::new();
    let tensor = heap.get(t).unwrap();
    let (rows, cols) = tensor.grid;
    for first in 0..tensor.tile_count() {
        for n in 0..=(tensor.tile_count() - first) {
            out.push(ops::tiles(heap, t, first, n).unwrap());
        }
    }
    for step in 1..=3 {
        for r0 in 0..rows {
            for c0 in 0..cols {
                let s = |start, end| Steps { start, end, step };
                let v = ops::rect(heap, t, s(r0, rows), s(c0, cols)).unwrap();
                out.push(ops::turn(&v).unwrap());
                out.push(v);
            }
        }
    }
    out.push(ops::items(heap, items_t, None, 4).unwrap());
    out.push(ops::items(heap, items_t, Some((3, 29)), 4).unwrap());
    out
}

#[test]
fn partition_covers_exactly_once() {
    let (heap, t) = numbered(4, 5);
    let mut heap = heap;
    let items_t = heap.add(Tensor::zeros("i", Format::U32, 4, 16, S22));
    let mut checked = 0;
    for v in views_of(&heap, t, items_t) {
        for size in 1..=13 {
            let mut next = 0;
            let mut lens = Vec::new();
            for rank in 0..size {
                let part = ops::partition(&v, rank, size).unwrap();
                // contiguous, in rank order: this part starts where the last ended
                let whole: Vec<String> = describe(&heap, &v);
                let mine: Vec<String> = describe(&heap, &part);
                assert_eq!(mine[..], whole[next..next + mine.len()]);
                next += mine.len();
                lens.push(mine.len());
            }
            assert_eq!(next, v.len(), "every place once");
            let (lo, hi) = (lens.iter().min().unwrap(), lens.iter().max().unwrap());
            assert!(hi - lo <= 1);
            assert!(
                lens.windows(2).all(|w| w[0] >= w[1]),
                "the first ranks get more"
            );
            checked += 1;
        }
    }
    assert!(checked > 1000);
}

/// What a view names, as text: its tile numbers, or its item numbers.
fn describe(heap: &Heap, v: &View) -> Vec<String> {
    match &v.kind {
        komorebi_ir::interp::memory::ViewKind::Tiles(_) => numbers_of(heap, v)
            .iter()
            .map(|n| format!("t{n}"))
            .collect(),
        komorebi_ir::interp::memory::ViewKind::Items { first, n, .. } => {
            (*first..first + n).map(|i| format!("i{i}")).collect()
        }
    }
}

#[test]
fn turned_twice_is_the_same_view() {
    for rows in 1..6 {
        for cols in 1..6 {
            let (heap, t) = numbered(rows, cols);
            for step in 1..4 {
                let s = |end| Steps {
                    start: 0,
                    end,
                    step,
                };
                let v = ops::rect(&heap, t, s(rows), s(cols)).unwrap();
                let back = ops::turn(&ops::turn(&v).unwrap()).unwrap();
                assert_eq!(back, v);
                assert_eq!(numbers_of(&heap, &back), numbers_of(&heap, &v));
            }
        }
    }
}

#[test]
fn a_stream_into_a_sink_of_the_same_view_writes_the_same_tiles() {
    for (rows, cols) in [(1, 7), (3, 4), (5, 2)] {
        let (heap, t) = numbered(rows, cols);
        let mut heap = heap;
        let before = heap.clone();
        let dst = heap.add(Tensor::zeros("d", Format::F32, rows * 2, cols * 2, S22));
        for block in 1..5 {
            let s = |end| Steps {
                start: 0,
                end,
                step: 1,
            };
            let src = ops::turn(&ops::rect(&heap, t, s(rows), s(cols)).unwrap()).unwrap();
            for v in [src, ops::tiles(&heap, t, 1, rows * cols - 1).unwrap()] {
                // in place: the view written is the view read (B8, point 12)
                let mut input = ops::stream(&v, block).unwrap();
                let mut out = ops::sink(&v, block).unwrap();
                while let Some(b) = ops::next_block(&heap, &mut input).unwrap() {
                    ops::push(&mut heap, &mut out, &b).unwrap();
                }
                assert_eq!(heap.tensors[t.0], before.tensors[t.0]);
                // and into the same view of another tensor of the same shape
                let other = View {
                    tensor: dst,
                    kind: v.kind.clone(),
                };
                let mut input = ops::stream(&v, block).unwrap();
                let mut out = ops::sink(&other, block).unwrap();
                while let Some(b) = ops::next_block(&heap, &mut input).unwrap() {
                    ops::push(&mut heap, &mut out, &b).unwrap();
                }
                assert_eq!(numbers_of(&heap, &other), numbers_of(&heap, &v));
            }
        }
    }
}

#[test]
fn words_on_more_than_one_tile_is_refused() {
    let mut heap = Heap::default();
    let t = heap.add(Tensor::zeros("x", Format::U32, 2, 8, S22));
    for n in 0..=4 {
        let v = ops::tiles(&heap, t, 0, n).unwrap();
        let words = ops::words(&heap, &v);
        let shape = TileShape { rows: 2, cols: 2 };
        let node = RecordType {
            name: "N".into(),
            fields: vec![("a".into(), Format::U32)],
        };
        let recs = ops::records(&heap, &v, &node, 1);
        if n == 1 {
            assert_eq!(words.map(|w| w.len()), Ok(shape.elements() as usize));
            assert!(recs.is_ok());
        } else {
            assert!(matches!(words, Err(Fault::Undefined(_))), "{n} tiles");
            assert!(matches!(recs, Err(Fault::Undefined(_))), "{n} tiles");
        }
    }
}

#[test]
fn malformed_values_are_faults_not_panics() {
    // a tile whose elements do not match its shape, built by hand
    let bad = Tile {
        format: Format::F32,
        shape: S22,
        elements: komorebi_ir::interp::memory::Elements::F32(vec![1.0]),
    };
    let good = f32s(&[1.0; 4]);
    let (b, g) = (Operand::Tile(&bad), Operand::Tile(&good));
    assert!(ops::binary(Bin::Add, b, g).is_err());
    assert!(ops::unary(Unary::Exp, &bad).is_err());
    assert!(ops::select(Cmp::Gt, b, g, g, g).is_err());
    assert!(ops::reduce(Reduce::Sum, &bad, Axis::Row).is_err());
    assert!(ops::spread_col(&bad).is_err());
    assert!(ops::spread_row(&bad).is_err());
    assert!(ops::matmul(&bad, &good).is_err());
    assert!(ops::transpose(&bad).is_err());
    assert!(ops::convert(&bad, Format::Bf16).is_err());
    let mut heap = Heap::default();
    let t = heap.add(Tensor::zeros("t", Format::F32, 2, 2, S22));
    let v = ops::tiles(&heap, t, 0, 1).unwrap();
    assert!(ops::store(&mut heap, &v, &Block { tiles: vec![bad] }).is_err());
    assert!(ops::read(
        &heap,
        &View {
            tensor: TensorId(9),
            kind: v.kind.clone()
        },
        0
    )
    .is_err());
}

#[test]
fn ten_failure_names() {
    let all = [
        Failure::ViewEnd,
        Failure::EmptyState,
        Failure::SendRefused,
        Failure::Capacity,
        Failure::GivenUp,
        Failure::Dependency,
        Failure::NotAPeer,
        Failure::OverBound,
        Failure::DivideByZero,
        Failure::OutOfRange,
    ];
    let names: Vec<&str> = all.iter().map(|f| f.name()).collect();
    assert_eq!(
        names,
        [
            "view_end",
            "empty_state",
            "send_refused",
            "capacity",
            "given_up",
            "dependency",
            "not_a_peer",
            "over_bound",
            "divide_by_zero",
            "out_of_range",
        ]
    );
}
