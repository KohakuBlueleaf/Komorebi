//! The language's operations on values: integer and float scalars, compute on tiles,
//! views of a tensor, streams and sinks, and a worker's state. Each operation returns
//! its value or a `Fault`; none panics.

use super::memory::{
    int_value, mask, Block, Elements, Failure, Fault, Filled, Format, Heap, Held, Member,
    MemberData, RecordType, Scalar, Sink, Steps, Stream, Tensor, TensorId, Tile, TileShape, View,
    ViewKind, Walk,
};

// ---- Scalars ----

/// A binary operator on numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bin {
    /// `+`.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/`: integer division rounded toward zero; on floats, division.
    Div,
    /// `%`: the remainder of `/` on integers.
    Rem,
    /// `&` on integers.
    And,
    /// `|` on integers.
    Or,
    /// `^` on integers.
    Xor,
    /// `<<` on integers.
    Shl,
    /// `>>` on integers: logical on an unsigned format, arithmetic on a signed one.
    Shr,
    /// `min(a, b)`, and `minimum(a, b)` on tiles.
    Min,
    /// `max(a, b)`, and `maximum(a, b)` on tiles.
    Max,
}

/// A comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    /// `>`.
    Gt,
    /// `<`.
    Lt,
    /// `>=`.
    Ge,
    /// `<=`.
    Le,
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
}

impl Cmp {
    fn holds<T: PartialOrd>(self, a: T, b: T) -> bool {
        match self {
            Cmp::Gt => a > b,
            Cmp::Lt => a < b,
            Cmp::Ge => a >= b,
            Cmp::Le => a <= b,
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
        }
    }
}

/// `op` on two integers of `format`, given as their bits. Arithmetic wraps modulo
/// 2^n; `/` rounds toward zero and `%` is its remainder; a shift by the width or more
/// shifts every bit out.
fn int_bin(op: Bin, format: Format, a: u64, b: u64) -> Result<u64, Fault> {
    let bits = format.bits().unwrap_or(64);
    let m = mask(bits);
    let (x, y) = (int_value(format, a), int_value(format, b));
    let wrap = |v: i128| (v as u64) & m;
    Ok(match op {
        Bin::Add => a.wrapping_add(b) & m,
        Bin::Sub => a.wrapping_sub(b) & m,
        Bin::Mul => a.wrapping_mul(b) & m,
        Bin::Div | Bin::Rem if y == 0 => return Err(Fault::DivisionByZero),
        Bin::Div => wrap(x / y),
        Bin::Rem => wrap(x % y),
        Bin::And => a & b,
        Bin::Or => a | b,
        Bin::Xor => a ^ b,
        Bin::Shl | Bin::Shr if y < 0 => {
            return Err(Fault::undefined(format!("a shift by {y}")));
        }
        Bin::Shl if y >= i128::from(bits) => 0,
        Bin::Shl => (a << (y as u32)) & m,
        Bin::Shr if y >= i128::from(bits) => {
            if x < 0 {
                m
            } else {
                0
            }
        }
        Bin::Shr => wrap(x >> (y as u32)),
        Bin::Min => wrap(x.min(y)),
        Bin::Max => wrap(x.max(y)),
    })
}

fn f32_bin(op: Bin, a: f32, b: f32) -> Result<f32, Fault> {
    Ok(match op {
        Bin::Add => a + b,
        Bin::Sub => a - b,
        Bin::Mul => a * b,
        Bin::Div => a / b,
        Bin::Min => a.min(b),
        Bin::Max => a.max(b),
        _ => return Err(Fault::undefined(format!("{op:?} on floats"))),
    })
}

fn f64_bin(op: Bin, a: f64, b: f64) -> Result<f64, Fault> {
    Ok(match op {
        Bin::Add => a + b,
        Bin::Sub => a - b,
        Bin::Mul => a * b,
        Bin::Div => a / b,
        Bin::Min => a.min(b),
        Bin::Max => a.max(b),
        _ => return Err(Fault::undefined(format!("{op:?} on floats"))),
    })
}

/// `a op b` on two scalars. Two integers must be of one format; two floats of
/// formats held alike give the left one's format, an `f32`-held float with an `f64`
/// gives an `f64`.
pub fn scalar_binary(op: Bin, a: Scalar, b: Scalar) -> Result<Scalar, Fault> {
    match (a, b) {
        (Scalar::Int(fa, x), Scalar::Int(fb, y)) if fa == fb => {
            Ok(Scalar::Int(fa, int_bin(op, fa, x, y)?))
        }
        (Scalar::F32(f, x), Scalar::F32(_, y)) => Ok(Scalar::F32(f, f32_bin(op, x, y)?)),
        (Scalar::F32(..) | Scalar::F64(_), Scalar::F32(..) | Scalar::F64(_)) => {
            let (x, y) = (a.as_f64().unwrap_or(0.0), b.as_f64().unwrap_or(0.0));
            Ok(Scalar::F64(f64_bin(op, x, y)?))
        }
        _ => Err(Fault::undefined(format!("{op:?} on {a:?} and {b:?}"))),
    }
}

/// `a cmp b` on two scalars: integers by their value, floats by theirs, and two
/// booleans for `==` and `!=`.
pub fn compare(cmp: Cmp, a: Scalar, b: Scalar) -> Result<bool, Fault> {
    match (a, b) {
        (Scalar::Int(..), Scalar::Int(..)) => {
            Ok(cmp.holds(a.as_i128().unwrap_or(0), b.as_i128().unwrap_or(0)))
        }
        (Scalar::F32(_, x), Scalar::F32(_, y)) => Ok(cmp.holds(x, y)),
        (Scalar::F32(..) | Scalar::F64(_), Scalar::F32(..) | Scalar::F64(_)) => {
            Ok(cmp.holds(a.as_f64().unwrap_or(0.0), b.as_f64().unwrap_or(0.0)))
        }
        (Scalar::Bool(x), Scalar::Bool(y)) if matches!(cmp, Cmp::Eq | Cmp::Ne) => {
            Ok(cmp.holds(x, y))
        }
        _ => Err(Fault::undefined(format!("{cmp:?} on {a:?} and {b:?}"))),
    }
}

// ---- Compute on tiles ----

/// A tile whose elements are not what its shape and format say is not defined.
fn whole(t: &Tile) -> Result<(), Fault> {
    if t.is_whole() {
        Ok(())
    } else {
        Err(Fault::undefined(
            "a tile whose elements do not match its shape",
        ))
    }
}

/// Element `i` of a tile.
fn elem(t: &Tile, i: usize) -> Result<Scalar, Fault> {
    t.at(i).ok_or(Fault::OutOfRange {
        index: i,
        len: t.elements.len(),
    })
}

/// An operand of a tile operation: a tile, or a scalar spread over the tile.
#[derive(Clone, Copy, Debug)]
pub enum Operand<'a> {
    /// A tile.
    Tile(&'a Tile),
    /// A scalar, the same at every element.
    Scalar(Scalar),
}

impl Operand<'_> {
    fn tile(&self) -> Option<&Tile> {
        match self {
            Operand::Tile(t) => Some(t),
            Operand::Scalar(_) => None,
        }
    }

    fn held(&self) -> Option<Held> {
        match self {
            Operand::Tile(t) => Some(t.format.held()),
            Operand::Scalar(s) => s.format().map(Format::held),
        }
    }

    /// The operand's elements as `f32`, `n` of them.
    fn f32s(&self, n: usize) -> Vec<f32> {
        match self {
            Operand::Tile(t) => match &t.elements {
                Elements::F32(v) => v.clone(),
                Elements::F64(v) => v.iter().map(|&x| x as f32).collect(),
                Elements::Int(_) => vec![0.0; n],
            },
            Operand::Scalar(s) => vec![s.as_f64().unwrap_or(0.0) as f32; n],
        }
    }

    /// The operand's elements as `f64`, `n` of them.
    fn f64s(&self, n: usize) -> Vec<f64> {
        match self {
            Operand::Tile(t) => t.to_f64().unwrap_or_else(|| vec![0.0; n]),
            Operand::Scalar(s) => vec![s.as_f64().unwrap_or(0.0); n],
        }
    }

    /// The operand's elements as integers of `format`, each wrapped to its width.
    fn ints(&self, format: Format, n: usize) -> Vec<u64> {
        let m = mask(format.bits().unwrap_or(64));
        match self {
            Operand::Tile(t) => match t.to_ints() {
                Some(v) => v.iter().map(|&x| (x as u64) & m).collect(),
                None => vec![0; n],
            },
            Operand::Scalar(s) => vec![(s.as_i128().unwrap_or(0) as u64) & m; n],
        }
    }
}

/// The shape and format of a tile operation's result: the tiles' shape (all the same),
/// and the format of the first tile, or `f64` where an `f32`-held float meets an `f64`.
/// Integers and floats do not mix.
fn result_of(operands: &[Operand<'_>]) -> Result<(Format, TileShape), Fault> {
    let tiles: Vec<&Tile> = operands.iter().filter_map(Operand::tile).collect();
    for tile in &tiles {
        whole(tile)?;
    }
    let Some(first) = tiles.first() else {
        return Err(Fault::undefined("a tile operation with no tile"));
    };
    if tiles.iter().any(|t| t.shape != first.shape) {
        return Err(Fault::undefined("tiles of different shapes"));
    }
    let mut helds = Vec::new();
    for operand in operands {
        match operand.held() {
            Some(held) => helds.push(held),
            None => return Err(Fault::undefined("a boolean operand of a tile operation")),
        }
    }
    let ints = helds.iter().filter(|&&h| h == Held::Int).count();
    if ints != 0 && ints != helds.len() {
        return Err(Fault::undefined(
            "integers and floats in one tile operation",
        ));
    }
    let wide = tiles.iter().any(|t| t.format.held() == Held::F64)
        && tiles.iter().any(|t| t.format.held() == Held::F32);
    let format = if wide { Format::F64 } else { first.format };
    Ok((format, first.shape))
}

/// `a op b` elementwise: `+ - * /`, `minimum` (`Bin::Min`) and `maximum` (`Bin::Max`),
/// on two tiles or a tile and a scalar. Integer tiles wrap; an integer divided by zero
/// is a fault.
pub fn binary(op: Bin, a: Operand<'_>, b: Operand<'_>) -> Result<Tile, Fault> {
    if !matches!(
        op,
        Bin::Add | Bin::Sub | Bin::Mul | Bin::Div | Bin::Min | Bin::Max
    ) {
        return Err(Fault::undefined(format!("{op:?} on tiles")));
    }
    let (format, shape) = result_of(&[a, b])?;
    let n = shape.elements();
    let elements = match format.held() {
        Held::Int => {
            let (x, y) = (a.ints(format, n), b.ints(format, n));
            let v: Result<Vec<u64>, Fault> = x
                .iter()
                .zip(&y)
                .map(|(&p, &q)| int_bin(op, format, p, q))
                .collect();
            Elements::Int(v?)
        }
        Held::F32 => {
            let (x, y) = (a.f32s(n), b.f32s(n));
            let v: Result<Vec<f32>, Fault> =
                x.iter().zip(&y).map(|(&p, &q)| f32_bin(op, p, q)).collect();
            Elements::F32(v?)
        }
        Held::F64 => {
            let (x, y) = (a.f64s(n), b.f64s(n));
            let v: Result<Vec<f64>, Fault> =
                x.iter().zip(&y).map(|(&p, &q)| f64_bin(op, p, q)).collect();
            Elements::F64(v?)
        }
    };
    Ok(Tile {
        format,
        shape,
        elements,
    })
}

/// A function of one tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
    /// `neg`, and the unary `-`.
    Neg,
    /// `abs`.
    Abs,
    /// `relu`: the element, or zero where it is below zero.
    Relu,
    /// `square`.
    Square,
    /// `floor`.
    Floor,
    /// `exp`.
    Exp,
    /// `exp2`.
    Exp2,
    /// `tanh`.
    Tanh,
    /// `erf`.
    Erf,
    /// `log`, the natural logarithm.
    Log,
    /// `sigmoid`: 1 / (1 + exp(-x)).
    Sigmoid,
    /// `silu`: x * sigmoid(x).
    Silu,
    /// `sqrt`.
    Sqrt,
    /// `rsqrt`: 1 / sqrt(x).
    Rsqrt,
    /// `reciprocal`: 1 / x.
    Reciprocal,
}

impl Unary {
    /// Whether the function is defined on integer tiles (the elementwise ones).
    pub fn on_integers(self) -> bool {
        matches!(
            self,
            Unary::Neg | Unary::Abs | Unary::Relu | Unary::Square | Unary::Floor
        )
    }

    fn f64(self, x: f64) -> f64 {
        match self {
            Unary::Neg => -x,
            Unary::Abs => x.abs(),
            Unary::Relu => {
                if x < 0.0 {
                    0.0
                } else {
                    x
                }
            }
            Unary::Square => x * x,
            Unary::Floor => x.floor(),
            Unary::Exp => x.exp(),
            Unary::Exp2 => x.exp2(),
            Unary::Tanh => x.tanh(),
            Unary::Erf => erf(x),
            Unary::Log => x.ln(),
            Unary::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Unary::Silu => x / (1.0 + (-x).exp()),
            Unary::Sqrt => x.sqrt(),
            Unary::Rsqrt => 1.0 / x.sqrt(),
            Unary::Reciprocal => 1.0 / x,
        }
    }

    fn f32(self, x: f32) -> f32 {
        match self {
            Unary::Neg => -x,
            Unary::Abs => x.abs(),
            Unary::Relu => {
                if x < 0.0 {
                    0.0
                } else {
                    x
                }
            }
            Unary::Square => x * x,
            Unary::Floor => x.floor(),
            Unary::Exp => x.exp(),
            Unary::Exp2 => x.exp2(),
            Unary::Tanh => x.tanh(),
            Unary::Erf => erf(f64::from(x)) as f32,
            Unary::Log => x.ln(),
            Unary::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Unary::Silu => x / (1.0 + (-x).exp()),
            Unary::Sqrt => x.sqrt(),
            Unary::Rsqrt => 1.0 / x.sqrt(),
            Unary::Reciprocal => 1.0 / x,
        }
    }
}

/// The error function, to the precision of an `f64`: its series below 2.5, and one
/// minus the continued fraction of `erfc` above.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    let two_over_root_pi = std::f64::consts::FRAC_2_SQRT_PI;
    let value = if a < 2.5 {
        let (mut sum, mut term, mut n) = (0.0f64, a, 0.0f64);
        loop {
            let part = term / (2.0 * n + 1.0);
            sum += part;
            if part.abs() <= 1e-17 * sum.abs() || n > 200.0 {
                break;
            }
            n += 1.0;
            term *= -a * a / n;
        }
        sum * two_over_root_pi
    } else {
        let mut f = a;
        for k in (1..=80).rev() {
            f = a + (f64::from(k) / 2.0) / f;
        }
        1.0 - (-a * a).exp() / (f * std::f64::consts::PI.sqrt())
    };
    value.copysign(x)
}

/// `f(a)` elementwise. On an integer tile only `neg`, `abs`, `relu`, `square` and
/// `floor` are defined, and they wrap.
pub fn unary(f: Unary, a: &Tile) -> Result<Tile, Fault> {
    whole(a)?;
    let elements = match &a.elements {
        Elements::F32(v) => Elements::F32(v.iter().map(|&x| f.f32(x)).collect()),
        Elements::F64(v) => Elements::F64(v.iter().map(|&x| f.f64(x)).collect()),
        Elements::Int(_) if !f.on_integers() => {
            return Err(Fault::undefined(format!("{f:?} on an integer tile")));
        }
        Elements::Int(v) => {
            let m = mask(a.format.bits().unwrap_or(64));
            Elements::Int(
                v.iter()
                    .map(|&bits| {
                        let x = int_value(a.format, bits);
                        let y = match f {
                            Unary::Neg => -x,
                            Unary::Abs => x.abs(),
                            Unary::Relu => x.max(0),
                            Unary::Square => x.wrapping_mul(x),
                            _ => x,
                        };
                        (y as u64) & m
                    })
                    .collect(),
            )
        }
    };
    Ok(Tile {
        format: a.format,
        shape: a.shape,
        elements,
    })
}

/// `where(a cmp b, x, y)`: x where the comparison holds, y elsewhere, element by
/// element. Each of the four is a tile or a scalar; the result has the format of the
/// first tile among x and y, or of a or b when x and y are both scalars.
pub fn select(
    cmp: Cmp,
    a: Operand<'_>,
    b: Operand<'_>,
    x: Operand<'_>,
    y: Operand<'_>,
) -> Result<Tile, Fault> {
    let (_, shape) = result_of(&[a, b])
        .or_else(|_| result_of(&[x, y]))
        .map_err(|_| Fault::undefined("a `where` with no tile"))?;
    for operand in [a, b, x, y] {
        if operand.tile().is_some_and(|t| t.shape != shape) {
            return Err(Fault::undefined("tiles of different shapes"));
        }
    }
    let n = shape.elements();
    let test = compare_lanes(cmp, a, b, n)?;
    let (format, _) = result_of(&[x, y]).or_else(|_| {
        let tile = a.tile().or(b.tile());
        tile.map(|t| (t.format, shape))
            .ok_or_else(|| Fault::undefined("no format"))
    })?;
    let pick = |p: &Operand<'_>| -> Result<Tile, Fault> {
        match p {
            Operand::Tile(t) => convert(t, format),
            Operand::Scalar(_) => binary(Bin::Add, Operand::Tile(&Tile::zeros(format, shape)), *p),
        }
    };
    let (tx, ty) = (pick(&x)?, pick(&y)?);
    let elements = match (tx.elements, ty.elements) {
        (Elements::Int(p), Elements::Int(q)) => Elements::Int(choose(&test, p, q)),
        (Elements::F32(p), Elements::F32(q)) => Elements::F32(choose(&test, p, q)),
        (Elements::F64(p), Elements::F64(q)) => Elements::F64(choose(&test, p, q)),
        _ => return Err(Fault::undefined("integers and floats in one `where`")),
    };
    Ok(Tile {
        format,
        shape,
        elements,
    })
}

fn choose<T>(test: &[bool], p: Vec<T>, q: Vec<T>) -> Vec<T> {
    p.into_iter()
        .zip(q)
        .zip(test)
        .map(|((p, q), &t)| if t { p } else { q })
        .collect()
}

fn compare_lanes(cmp: Cmp, a: Operand<'_>, b: Operand<'_>, n: usize) -> Result<Vec<bool>, Fault> {
    let lanes = |o: &Operand<'_>| -> Result<Vec<Scalar>, Fault> {
        match o {
            Operand::Tile(t) => (0..n).map(|i| elem(t, i)).collect(),
            Operand::Scalar(s) => Ok(vec![*s; n]),
        }
    };
    let (p, q) = (lanes(&a)?, lanes(&b)?);
    p.iter()
        .zip(&q)
        .map(|(&x, &y)| compare(cmp, x, y))
        .collect()
}

/// An axis of a tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// `axis = row`: one value for each row.
    Row,
    /// `axis = col`: one value for each column.
    Col,
}

/// A reduction along an axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduce {
    /// `reduce_sum`.
    Sum,
    /// `reduce_max`.
    Max,
    /// `reduce_min`.
    Min,
}

/// `reduce_sum(a, axis = row)` and its kin: one value for each row (or column), typed
/// as a tile. The value of row r stands in column 0 of row r (of column c, in row 0
/// of column c); every other element is zero. The elements are combined in order,
/// from the first column (row).
pub fn reduce(kind: Reduce, a: &Tile, axis: Axis) -> Result<Tile, Fault> {
    whole(a)?;
    let TileShape { rows, cols } = a.shape;
    let op = match kind {
        Reduce::Sum => Bin::Add,
        Reduce::Max => Bin::Max,
        Reduce::Min => Bin::Min,
    };
    let mut out = Tile::zeros(a.format, a.shape);
    let (outer, inner) = match axis {
        Axis::Row => (rows, cols),
        Axis::Col => (cols, rows),
    };
    for o in 0..outer {
        let index = |i: usize| match axis {
            Axis::Row => o * cols + i,
            Axis::Col => i * cols + o,
        };
        if inner == 0 {
            continue;
        }
        let mut acc = elem(a, index(0))?;
        for i in 1..inner {
            acc = scalar_binary(op, acc, elem(a, index(i))?)?;
        }
        put_at(&mut out, index(0), acc);
    }
    Ok(out)
}

/// Writes `value` at flat position `i` of `tile`, held as the tile holds it.
fn put_at(tile: &mut Tile, i: usize, value: Scalar) {
    match &mut tile.elements {
        Elements::Int(v) => {
            v[i] = (value.as_i128().unwrap_or(0) as u64) & mask(tile.format.bits().unwrap_or(64))
        }
        Elements::F32(v) => v[i] = value.as_f64().unwrap_or(0.0) as f32,
        Elements::F64(v) => v[i] = value.as_f64().unwrap_or(0.0),
    }
}

/// `v.col()`: the value of each row, in column 0, spread along the row.
pub fn spread_col(v: &Tile) -> Result<Tile, Fault> {
    let cols = v.shape.cols;
    spread(v, |i| (i / cols.max(1)) * cols)
}

/// `w.row()`: row 0 spread along the columns, to every row.
pub fn spread_row(w: &Tile) -> Result<Tile, Fault> {
    let cols = w.shape.cols;
    spread(w, |i| i % cols.max(1))
}

fn spread(t: &Tile, from: impl Fn(usize) -> usize) -> Result<Tile, Fault> {
    whole(t)?;
    let n = t.shape.elements();
    let elements = match &t.elements {
        Elements::Int(v) => Elements::Int((0..n).map(|i| v[from(i)]).collect()),
        Elements::F32(v) => Elements::F32((0..n).map(|i| v[from(i)]).collect()),
        Elements::F64(v) => Elements::F64((0..n).map(|i| v[from(i)]).collect()),
    };
    Ok(Tile {
        format: t.format,
        shape: t.shape,
        elements,
    })
}

/// `matmul(a, b)`: the matrix product, each element the sum over k of a[i, k] * b[k, j]
/// taken in order of k. a's columns must be b's rows. The format is a's, or `f64`
/// where an `f32`-held float meets an `f64`.
pub fn matmul(a: &Tile, b: &Tile) -> Result<Tile, Fault> {
    whole(a)?;
    whole(b)?;
    if a.shape.cols != b.shape.rows {
        return Err(Fault::undefined(
            "a product of tiles whose shapes do not meet",
        ));
    }
    let (n, k, m) = (a.shape.rows, a.shape.cols, b.shape.cols);
    let shape = TileShape { rows: n, cols: m };
    let (ha, hb) = (a.format.held(), b.format.held());
    if (ha == Held::Int) != (hb == Held::Int) {
        return Err(Fault::undefined("a product of an integer and a float tile"));
    }
    let elements = if ha == Held::Int {
        let (x, y) = (
            Operand::Tile(a).ints(a.format, n * k),
            Operand::Tile(b).ints(a.format, k * m),
        );
        let mut out = vec![0u64; n * m];
        for i in 0..n {
            for j in 0..m {
                let mut acc = 0u64;
                for p in 0..k {
                    acc = acc.wrapping_add(x[i * k + p].wrapping_mul(y[p * m + j]));
                }
                out[i * m + j] = acc & mask(a.format.bits().unwrap_or(64));
            }
        }
        Elements::Int(out)
    } else if ha == Held::F64 || hb == Held::F64 {
        let (x, y) = (Operand::Tile(a).f64s(n * k), Operand::Tile(b).f64s(k * m));
        let mut out = vec![0.0f64; n * m];
        for i in 0..n {
            for j in 0..m {
                out[i * m + j] = (0..k).fold(0.0, |acc, p| acc + x[i * k + p] * y[p * m + j]);
            }
        }
        Elements::F64(out)
    } else {
        let (x, y) = (Operand::Tile(a).f32s(n * k), Operand::Tile(b).f32s(k * m));
        let mut out = vec![0.0f32; n * m];
        for i in 0..n {
            for j in 0..m {
                out[i * m + j] = (0..k).fold(0.0, |acc, p| acc + x[i * k + p] * y[p * m + j]);
            }
        }
        Elements::F32(out)
    };
    let format = if ha == Held::F32 && hb == Held::F64 {
        Format::F64
    } else {
        a.format
    };
    Ok(Tile {
        format,
        shape,
        elements,
    })
}

/// `a.T`: the tile turned, rows becoming columns.
pub fn transpose(a: &Tile) -> Result<Tile, Fault> {
    whole(a)?;
    let TileShape { rows, cols } = a.shape;
    let from = |i: usize| (i % rows) * cols + i / rows;
    let n = rows * cols;
    let elements = match &a.elements {
        Elements::Int(v) => Elements::Int((0..n).map(|i| v[from(i)]).collect()),
        Elements::F32(v) => Elements::F32((0..n).map(|i| v[from(i)]).collect()),
        Elements::F64(v) => Elements::F64((0..n).map(|i| v[from(i)]).collect()),
    };
    Ok(Tile {
        format: a.format,
        shape: TileShape {
            rows: cols,
            cols: rows,
        },
        elements,
    })
}

/// `zeros`: a tile of zeros.
pub fn zeros(format: Format, shape: TileShape) -> Tile {
    Tile::zeros(format, shape)
}

/// The tile as `format` holds it, for a write to a tensor or a member of another
/// format: the same values, with no rounding to the format's precision. Floats held as
/// `f32` move between their formats unchanged; to and from `f64` they widen, or narrow
/// to the nearest `f32`. Integers wrap to the new width. An integer does not become a
/// float, nor a float an integer.
pub fn convert(t: &Tile, format: Format) -> Result<Tile, Fault> {
    whole(t)?;
    let n = t.shape.elements();
    let elements = match (format.held(), &t.elements) {
        (Held::F32, Elements::F32(v)) => Elements::F32(v.clone()),
        (Held::F32, Elements::F64(v)) => Elements::F32(v.iter().map(|&x| x as f32).collect()),
        (Held::F64, Elements::F32(_) | Elements::F64(_)) => Elements::F64(Operand::Tile(t).f64s(n)),
        (Held::Int, Elements::Int(_)) => Elements::Int(Operand::Tile(t).ints(format, n)),
        _ => {
            return Err(Fault::undefined(format!(
                "a tile of {} written as {format}",
                t.format
            )));
        }
    };
    Ok(Tile {
        format,
        shape: t.shape,
        elements,
    })
}

// ---- Views of a tensor ----

fn view_end<T>() -> Result<T, Fault> {
    Err(Fault::Failure(Failure::ViewEnd))
}

/// `t.tiles(first, n)`: n consecutive tiles. Past the tensor's end: `view_end`.
pub fn tiles(heap: &Heap, t: TensorId, first: usize, n: usize) -> Result<View, Fault> {
    let tensor = heap.get(t)?;
    match first.checked_add(n) {
        Some(end) if end <= tensor.tile_count() => Ok(View {
            tensor: t,
            kind: ViewKind::Tiles(Walk::Run { first, n }),
        }),
        _ => view_end(),
    }
}

/// `t.tiles[r0:r1:s, c0:c1:s]`: a rectangle of the tensor's grid of tiles, walked row
/// by row. A range past the grid: `view_end`. A step of 0, or a range that ends before
/// it starts, is not defined.
pub fn rect(heap: &Heap, t: TensorId, rows: Steps, cols: Steps) -> Result<View, Fault> {
    let tensor = heap.get(t)?;
    for range in [rows, cols] {
        if range.step == 0 || range.end < range.start {
            return Err(Fault::undefined(format!(
                "the range {}:{}:{}",
                range.start, range.end, range.step
            )));
        }
    }
    if rows.end > tensor.grid.0 || cols.end > tensor.grid.1 {
        return view_end();
    }
    Ok(View {
        tensor: t,
        kind: ViewKind::Tiles(Walk::Rect {
            rows,
            cols,
            grid_cols: tensor.grid.1,
            turned: false,
        }),
    })
}

/// `v.T`: a rectangle view turned, walked column by column. The tiles themselves are
/// not turned. Defined on a rectangle only.
pub fn turn(v: &View) -> Result<View, Fault> {
    match &v.kind {
        ViewKind::Tiles(Walk::Rect {
            rows,
            cols,
            grid_cols,
            turned,
        }) => Ok(View {
            tensor: v.tensor,
            kind: ViewKind::Tiles(Walk::Rect {
                rows: *rows,
                cols: *cols,
                grid_cols: *grid_cols,
                turned: !turned,
            }),
        }),
        _ => Err(Fault::undefined("`.T` of a view that is not a rectangle")),
    }
}

/// The bytes of a tensor's tiles, one tile after another.
fn tensor_bytes(tensor: &Tensor) -> Result<usize, Fault> {
    let width = tensor
        .format
        .bytes()
        .ok_or_else(|| Fault::undefined(format!("the bytes of a tensor of {}", tensor.format)))?;
    Ok(width * tensor.tile.elements() * tensor.tile_count())
}

/// `t.items(size = B)` (`range` left out): every whole item of B bytes of the tensor;
/// `t.items(first, n, size = B)`: n items from item `first`, or `view_end` past the
/// last whole item. Item `i` is bytes `i * B` to `(i + 1) * B` of the tensor's tiles
/// laid one after another, each tile's elements row by row, little-endian.
pub fn items(
    heap: &Heap,
    t: TensorId,
    range: Option<(usize, usize)>,
    size: usize,
) -> Result<View, Fault> {
    let tensor = heap.get(t)?;
    if size == 0 {
        return Err(Fault::undefined("items of 0 bytes"));
    }
    let whole = tensor_bytes(tensor)? / size;
    let (first, n) = range.unwrap_or((0, whole));
    match first.checked_add(n) {
        Some(end) if end <= whole => Ok(View {
            tensor: t,
            kind: ViewKind::Items { size, first, n },
        }),
        _ => view_end(),
    }
}

fn tile_walk(v: &View) -> Result<&Walk, Fault> {
    match &v.kind {
        ViewKind::Tiles(walk) => Ok(walk),
        ViewKind::Items { .. } => Err(Fault::undefined("tiles of an items view")),
    }
}

/// `v[i]`: tile i of a view, read.
pub fn read(heap: &Heap, v: &View, i: usize) -> Result<Tile, Fault> {
    let walk = tile_walk(v)?;
    let numbers = walk.tiles();
    let Some(&number) = numbers.get(i) else {
        return Err(Fault::OutOfRange {
            index: i,
            len: numbers.len(),
        });
    };
    let tensor = heap.get(v.tensor)?;
    tensor.tiles.get(number).cloned().map_or_else(view_end, Ok)
}

/// The one tile of a view of one tile.
fn one_tile(heap: &Heap, v: &View, what: &str) -> Result<Tile, Fault> {
    if v.len() != 1 || !matches!(v.kind, ViewKind::Tiles(_)) {
        return Err(Fault::undefined(format!(
            "`{what}` of a view of {} tiles",
            v.len()
        )));
    }
    read(heap, v, 0)
}

/// `v.words()`: the scalars of a view of one tile, row by row. The tensor's format
/// must be a scalar type.
pub fn words(heap: &Heap, v: &View) -> Result<Vec<Scalar>, Fault> {
    let tile = one_tile(heap, v, "words()")?;
    if !tile.format.is_scalar_type() {
        return Err(Fault::undefined(format!(
            "the words of a tile of {}",
            tile.format
        )));
    }
    whole(&tile)?;
    (0..tile.shape.elements()).map(|i| elem(&tile, i)).collect()
}

/// `v.records(R, count = c)`: the first c records of a view of one tile, each its
/// fields in order, read from the tile's bytes with no gap. Records past the tile's
/// bytes: `view_end`.
pub fn records(
    heap: &Heap,
    v: &View,
    record: &RecordType,
    count: usize,
) -> Result<Vec<Vec<Scalar>>, Fault> {
    let tile = one_tile(heap, v, "records()")?;
    let bytes = tile
        .to_bytes()
        .ok_or_else(|| Fault::undefined(format!("the bytes of a tile of {}", tile.format)))?;
    let mut widths = Vec::new();
    for (name, format) in &record.fields {
        match format.bytes() {
            Some(w) if format.is_scalar_type() => widths.push((*format, w)),
            _ => return Err(Fault::undefined(format!("the field {name} of {format}"))),
        }
    }
    let size: usize = widths.iter().map(|(_, w)| w).sum();
    match size.checked_mul(count) {
        Some(total) if total <= bytes.len() => {}
        _ => return view_end(),
    }
    let mut out = Vec::with_capacity(count);
    let mut at = 0;
    for _ in 0..count {
        let mut fields = Vec::with_capacity(widths.len());
        for &(format, width) in &widths {
            let chunk = &bytes[at..at + width];
            at += width;
            let one = TileShape { rows: 1, cols: 1 };
            let field = Tile::from_bytes(format, one, chunk)
                .ok_or_else(|| Fault::undefined("a field's bytes"))?;
            fields.push(elem(&field, 0)?);
        }
        out.push(fields);
    }
    Ok(out)
}

/// The bytes of a member, read: every element must be filled.
fn member_bytes(member: &Member) -> Result<Vec<u8>, Fault> {
    for i in 0..member.len() {
        if !member.is_filled(i) {
            return Err(Fault::Failure(Failure::EmptyState));
        }
    }
    let mut out = Vec::new();
    match &member.data {
        MemberData::Tiles(tiles) => {
            for tile in tiles {
                out.extend(tile.to_bytes().ok_or_else(|| {
                    Fault::undefined(format!("the bytes of a member of {}", member.format))
                })?);
            }
        }
        MemberData::Scalars(scalars) => {
            let one = TileShape { rows: 1, cols: 1 };
            for &s in scalars {
                let tile = scalar_tile(member.format, one, s)?;
                out.extend(tile.to_bytes().ok_or_else(|| {
                    Fault::undefined(format!("the bytes of a member of {}", member.format))
                })?);
            }
        }
    }
    Ok(out)
}

fn scalar_tile(format: Format, shape: TileShape, s: Scalar) -> Result<Tile, Fault> {
    binary(
        Bin::Add,
        Operand::Tile(&Tile::zeros(format, shape)),
        Operand::Scalar(s),
    )
}

/// `items.put(src, at = i)`: writes the whole of the state member `src` as item i of
/// an items view. The member's bytes must be the item's size; reading it needs every
/// element filled.
pub fn put(heap: &mut Heap, v: &View, src: &Member, at: usize) -> Result<(), Fault> {
    let ViewKind::Items { size, first, n } = v.kind else {
        return Err(Fault::undefined("`put` into a view of tiles"));
    };
    if at >= n {
        return Err(Fault::OutOfRange { index: at, len: n });
    }
    let bytes = member_bytes(src)?;
    if bytes.len() != size {
        return Err(Fault::undefined(format!(
            "a member of {} bytes put into an item of {size}",
            bytes.len()
        )));
    }
    let tensor = heap.get_mut(v.tensor)?;
    let per_tile = tensor_bytes(tensor)? / tensor.tile_count().max(1);
    let start = (first + at) * size;
    for (k, &byte) in bytes.iter().enumerate() {
        let place = start + k;
        let (number, offset) = (place / per_tile, place % per_tile);
        let Some(tile) = tensor.tiles.get(number) else {
            return view_end();
        };
        let mut tile_bytes = tile.to_bytes().ok_or_else(|| Fault::undefined("bytes"))?;
        match tile_bytes.get_mut(offset) {
            Some(slot) => *slot = byte,
            None => return view_end(),
        }
        tensor.tiles[number] = Tile::from_bytes(tile.format, tile.shape, &tile_bytes)
            .ok_or_else(|| Fault::undefined("bytes"))?;
    }
    Ok(())
}

/// `partition(view, rank, size)`: this rank's contiguous part of a view. The view's
/// L tiles (or items) are dealt in rank order, and the first `L mod size` ranks get
/// one more: the parts of all ranks cover the view exactly once.
pub fn partition(v: &View, rank: usize, size: usize) -> Result<View, Fault> {
    if size == 0 || rank >= size {
        return Err(Fault::undefined(format!(
            "rank {rank} of a group of {size}"
        )));
    }
    let total = v.len();
    let (base, extra) = (total / size, total % size);
    let first = rank * base + rank.min(extra);
    let n = base + usize::from(rank < extra);
    let kind = match &v.kind {
        ViewKind::Tiles(walk) => ViewKind::Tiles(Walk::Part {
            of: Box::new(walk.clone()),
            first,
            n,
        }),
        ViewKind::Items { size, first: f, .. } => ViewKind::Items {
            size: *size,
            first: f + first,
            n,
        },
    };
    Ok(View {
        tensor: v.tensor,
        kind,
    })
}

/// `stream(view, block = n)`: the view's blocks of n tiles, read in order.
pub fn stream(v: &View, block: usize) -> Result<Stream, Fault> {
    if block == 0 {
        return Err(Fault::undefined("a stream with `block = 0`"));
    }
    tile_walk(v)?;
    Ok(Stream {
        view: v.clone(),
        block,
        next: 0,
    })
}

/// The stream's next block, read from the heap now, or `None` at its end. The last
/// block holds what is left: a ragged end is one shorter block.
pub fn next_block(heap: &Heap, s: &mut Stream) -> Result<Option<Block>, Fault> {
    let numbers = tile_walk(&s.view)?.tiles();
    if s.next >= numbers.len() {
        return Ok(None);
    }
    let end = s.next.saturating_add(s.block).min(numbers.len());
    let tensor = heap.get(s.view.tensor)?;
    let mut tiles = Vec::with_capacity(end - s.next);
    for &number in &numbers[s.next..end] {
        match tensor.tiles.get(number) {
            Some(tile) => tiles.push(tile.clone()),
            None => return view_end(),
        }
    }
    s.next = end;
    Ok(Some(Block { tiles }))
}

/// `zip(s1, s2)`'s next pair of blocks, or `None` when either stream has ended.
pub fn next_pair(
    heap: &Heap,
    a: &mut Stream,
    b: &mut Stream,
) -> Result<Option<(Block, Block)>, Fault> {
    let (sa, sb) = (a.clone(), b.clone());
    match (next_block(heap, a)?, next_block(heap, b)?) {
        (Some(x), Some(y)) => Ok(Some((x, y))),
        _ => {
            *a = sa;
            *b = sb;
            Ok(None)
        }
    }
}

/// `sink(view, block = n)`: a write stream over the view.
pub fn sink(v: &View, block: usize) -> Result<Sink, Fault> {
    if block == 0 {
        return Err(Fault::undefined("a sink with `block = 0`"));
    }
    tile_walk(v)?;
    Ok(Sink {
        view: v.clone(),
        block,
        next: 0,
    })
}

/// `out.push(b)`: writes the next block, each tile as the tensor's format holds it.
/// Only the tiles the view names are written: a block longer than what is left is cut
/// at the view's end. A push when the view is all written is `view_end`.
pub fn push(heap: &mut Heap, s: &mut Sink, block: &Block) -> Result<(), Fault> {
    let numbers = tile_walk(&s.view)?.tiles();
    if s.next >= numbers.len() {
        return view_end();
    }
    if block.tiles.len() > s.block {
        return Err(Fault::undefined(format!(
            "a block of {} tiles pushed into a sink of blocks of {}",
            block.tiles.len(),
            s.block
        )));
    }
    let end = (s.next + block.tiles.len()).min(numbers.len());
    write_tiles(heap, s.view.tensor, &numbers[s.next..end], &block.tiles)?;
    s.next = end;
    Ok(())
}

fn write_tiles(
    heap: &mut Heap,
    t: TensorId,
    numbers: &[usize],
    tiles: &[Tile],
) -> Result<(), Fault> {
    let tensor = heap.get_mut(t)?;
    for (&number, tile) in numbers.iter().zip(tiles) {
        if tile.shape != tensor.tile {
            return Err(Fault::undefined(
                "a tile of another shape than the tensor's",
            ));
        }
        let written = convert(tile, tensor.format)?;
        match tensor.tiles.get_mut(number) {
            Some(slot) => *slot = written,
            None => return view_end(),
        }
    }
    Ok(())
}

/// `store view <- value`: writes a block of as many tiles as the view names (one tile
/// for a view of one tile), each as the tensor's format holds it.
pub fn store(heap: &mut Heap, v: &View, value: &Block) -> Result<(), Fault> {
    let numbers = tile_walk(v)?.tiles();
    if numbers.len() != value.tiles.len() {
        return Err(Fault::undefined(format!(
            "a store of {} tiles into a view of {}",
            value.tiles.len(),
            numbers.len()
        )));
    }
    write_tiles(heap, v.tensor, &numbers, &value.tiles)
}

// ---- State ----

fn index_of(member: &Member, i: usize) -> Result<(), Fault> {
    if i >= member.len() {
        Err(Fault::OutOfRange {
            index: i,
            len: member.len(),
        })
    } else {
        Ok(())
    }
}

fn mark_all(member: &mut Member) {
    member.filled = match &member.filled {
        Filled::Whole(_) => Filled::Whole(true),
        Filled::Element(marks) => Filled::Element(vec![true; marks.len()]),
    };
}

/// `s.fill(stream)`: fills a member of tiles whole from the stream, which must hold
/// exactly as many tiles as the member. The stream is read to its end.
pub fn fill(heap: &Heap, member: &mut Member, s: &mut Stream) -> Result<(), Fault> {
    let MemberData::Tiles(slots) = &member.data else {
        return Err(Fault::undefined("`fill` of a member of scalars"));
    };
    let mut tiles = Vec::with_capacity(slots.len());
    while let Some(block) = next_block(heap, s)? {
        tiles.extend(block.tiles);
    }
    if tiles.len() != slots.len() {
        return Err(Fault::undefined(format!(
            "a fill of {} tiles into a member of {}",
            tiles.len(),
            slots.len()
        )));
    }
    let shape = slots.first().map(|t| t.shape);
    let mut written = Vec::with_capacity(tiles.len());
    for tile in &tiles {
        if Some(tile.shape) != shape {
            return Err(Fault::undefined(
                "a tile of another shape than the member's",
            ));
        }
        written.push(convert(tile, member.format)?);
    }
    member.data = MemberData::Tiles(written);
    mark_all(member);
    Ok(())
}

/// `s.zero()`: clears every element; the member is filled.
pub fn zero(member: &mut Member) {
    match &mut member.data {
        MemberData::Tiles(tiles) => {
            for tile in tiles.iter_mut() {
                *tile = Tile::zeros(member.format, tile.shape);
            }
        }
        MemberData::Scalars(scalars) => {
            for s in scalars.iter_mut() {
                *s = match member.format.held() {
                    Held::Int => Scalar::Int(member.format, 0),
                    Held::F32 => Scalar::F32(member.format, 0.0),
                    Held::F64 => Scalar::F64(0.0),
                };
            }
        }
    }
    mark_all(member);
}

/// A value of a member's element: a tile or a scalar.
#[derive(Clone, Debug, PartialEq)]
pub enum Element {
    /// A tile of a member of tiles.
    Tile(Tile),
    /// A scalar of a member of scalars.
    Scalar(Scalar),
}

/// `s[i] = x`: writes element i. Always allowed. It fills element i of a member marked
/// `@filled(element)`; a member filled whole stays as it was.
pub fn set(member: &mut Member, i: usize, value: Element) -> Result<(), Fault> {
    index_of(member, i)?;
    match (&mut member.data, value) {
        (MemberData::Tiles(tiles), Element::Tile(tile)) => {
            if tile.shape != tiles[i].shape {
                return Err(Fault::undefined(
                    "a tile of another shape than the member's",
                ));
            }
            tiles[i] = convert(&tile, member.format)?;
        }
        (MemberData::Scalars(scalars), Element::Scalar(s)) if s.format() == Some(member.format) => {
            scalars[i] = s;
        }
        (_, value) => {
            return Err(Fault::undefined(format!(
                "{value:?} written to a member of {}",
                member.format
            )));
        }
    }
    if let Filled::Element(marks) = &mut member.filled {
        if let Some(mark) = marks.get_mut(i) {
            *mark = true;
        }
    }
    Ok(())
}

/// `s[i]`: element i, read. Not filled: `empty_state`.
pub fn get(member: &Member, i: usize) -> Result<Element, Fault> {
    index_of(member, i)?;
    if !member.is_filled(i) {
        return Err(Fault::Failure(Failure::EmptyState));
    }
    Ok(match &member.data {
        MemberData::Tiles(tiles) => Element::Tile(tiles[i].clone()),
        MemberData::Scalars(scalars) => Element::Scalar(scalars[i]),
    })
}

/// `s.bump(i)`: reads element i of a member of integers, then adds one, wrapping.
/// Gives the value read.
pub fn bump(member: &mut Member, i: usize) -> Result<Scalar, Fault> {
    let Element::Scalar(old) = get(member, i)? else {
        return Err(Fault::undefined("`bump` of a member of tiles"));
    };
    let Scalar::Int(format, bits) = old else {
        return Err(Fault::undefined("`bump` of a member of floats"));
    };
    let new = Scalar::Int(format, int_bin(Bin::Add, format, bits, 1)?);
    set(member, i, Element::Scalar(new))?;
    Ok(old)
}
