//! The values a program computes with, as the reference interpreter holds them: scalars,
//! tiles, blocks of tiles, tensors on the heap, views of them, streams and sinks over a
//! view, and the members of a worker's state. Plain data; no knowledge of the syntax tree.
//!
//! A format is a notation. Every float format (`f16 f32 bf16 bfp8 bfp4 bfp2`) is held as
//! `f32` elements and `f64` as `f64`; an integer is held in its own width, and its
//! arithmetic wraps modulo 2^n. Nothing is rounded to a format's precision.

use std::fmt;

pub use crate::types::{Axis, Failure, Format, TileShape};

/// How the interpreter holds the elements of a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    /// An integer in its own width, as the low bits of a `u64`.
    Int,
    /// An `f32`: every float format but `f64`.
    F32,
    /// An `f64`.
    F64,
}

/// How the interpreter holds an element of this format.
pub fn held(format: Format) -> Held {
    match format {
        Format::F64 => Held::F64,
        Format::F16 | Format::F32 | Format::Bf16 | Format::Bfp8 | Format::Bfp4 | Format::Bfp2 => {
            Held::F32
        }
        _ => Held::Int,
    }
}

/// A tile shape counted as the interpreter indexes: rows, columns and elements as
/// `usize`. A count past the machine's addresses reads as `usize::MAX`, which no
/// tile's elements reach.
pub trait Extent {
    /// Rows of elements.
    fn height(self) -> usize;
    /// Columns of elements.
    fn width(self) -> usize;
    /// Elements of a tile.
    fn count(self) -> usize;
}

impl Extent for TileShape {
    fn height(self) -> usize {
        usize::try_from(self.rows).unwrap_or(usize::MAX)
    }
    fn width(self) -> usize {
        usize::try_from(self.cols).unwrap_or(usize::MAX)
    }
    fn count(self) -> usize {
        self.height().saturating_mul(self.width())
    }
}

/// The mask of an integer `bits` wide.
pub fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// One scalar value: an integer of a format, a float, or a boolean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    /// An integer: its format and its bits, masked to the format's width.
    Int(Format, u64),
    /// A value of a float format held as `f32`.
    F32(Format, f32),
    /// An `f64`.
    F64(f64),
    /// A boolean: the result of a comparison, a `send` or an `offer`.
    Bool(bool),
}

impl Scalar {
    /// The integer `value` of `format`, wrapped modulo 2^n; `None` if `format` is no
    /// integer format.
    pub fn int(format: Format, value: i128) -> Option<Scalar> {
        let bits = format.bits()?;
        Some(Scalar::Int(format, (value as u64) & mask(bits)))
    }

    /// The float `value` of `format`, held as the format holds it; `None` if `format`
    /// is an integer format.
    pub fn float(format: Format, value: f64) -> Option<Scalar> {
        match held(format) {
            Held::F32 => Some(Scalar::F32(format, value as f32)),
            Held::F64 => Some(Scalar::F64(value)),
            Held::Int => None,
        }
    }

    /// The scalar's format; `None` for a boolean.
    pub fn format(&self) -> Option<Format> {
        match *self {
            Scalar::Int(format, _) | Scalar::F32(format, _) => Some(format),
            Scalar::F64(_) => Some(Format::F64),
            Scalar::Bool(_) => None,
        }
    }

    /// An integer's mathematical value, sign-extended for a signed format.
    pub fn as_i128(&self) -> Option<i128> {
        match *self {
            Scalar::Int(format, bits) => Some(int_value(format, bits)),
            _ => None,
        }
    }

    /// A float's value, widened to `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Scalar::F32(_, v) => Some(f64::from(v)),
            Scalar::F64(v) => Some(v),
            _ => None,
        }
    }
}

/// The mathematical value of an integer element of `format`.
pub fn int_value(format: Format, bits: u64) -> i128 {
    let width = format.bits().unwrap_or(64);
    if format.is_signed() && width < 64 {
        let shift = 64 - width;
        i128::from(((bits << shift) as i64) >> shift)
    } else if format.is_signed() {
        i128::from(bits as i64)
    } else {
        i128::from(bits)
    }
}

/// The elements of a tile, row by row, as the format holds them.
#[derive(Clone, Debug, PartialEq)]
pub enum Elements {
    /// Integers, masked to their width.
    Int(Vec<u64>),
    /// Floats held as `f32`.
    F32(Vec<f32>),
    /// `f64` floats.
    F64(Vec<f64>),
}

impl Elements {
    /// Zeros, `n` of them, held as `format` holds them.
    pub fn zeros(format: Format, n: usize) -> Elements {
        match held(format) {
            Held::Int => Elements::Int(vec![0; n]),
            Held::F32 => Elements::F32(vec![0.0; n]),
            Held::F64 => Elements::F64(vec![0.0; n]),
        }
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        match self {
            Elements::Int(v) => v.len(),
            Elements::F32(v) => v.len(),
            Elements::F64(v) => v.len(),
        }
    }

    /// Whether there is no element.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One tile: a block of rows by columns of elements of one format.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    /// The format of its elements.
    pub format: Format,
    /// Its shape.
    pub shape: TileShape,
    /// Its elements, row by row.
    pub elements: Elements,
}

impl Tile {
    /// A tile of zeros.
    pub fn zeros(format: Format, shape: TileShape) -> Tile {
        Tile {
            format,
            shape,
            elements: Elements::zeros(format, shape.count()),
        }
    }

    /// A tile of a float format from values given row by row; `None` when the format
    /// is an integer format or the count is not the shape's.
    pub fn from_f64(format: Format, shape: TileShape, values: &[f64]) -> Option<Tile> {
        if values.len() != shape.count() {
            return None;
        }
        let elements = match held(format) {
            Held::F32 => Elements::F32(values.iter().map(|&v| v as f32).collect()),
            Held::F64 => Elements::F64(values.to_vec()),
            Held::Int => return None,
        };
        Some(Tile {
            format,
            shape,
            elements,
        })
    }

    /// A tile of an integer format from values given row by row, each wrapped to the
    /// format's width; `None` when the format is a float format or the count is not
    /// the shape's.
    pub fn from_ints(format: Format, shape: TileShape, values: &[i128]) -> Option<Tile> {
        let bits = format.bits()?;
        if values.len() != shape.count() {
            return None;
        }
        let elements = Elements::Int(values.iter().map(|&v| (v as u64) & mask(bits)).collect());
        Some(Tile {
            format,
            shape,
            elements,
        })
    }

    /// The element at row `r`, column `c`, as a scalar; `None` outside the tile.
    pub fn get(&self, r: usize, c: usize) -> Option<Scalar> {
        if r >= self.shape.height() || c >= self.shape.width() {
            return None;
        }
        self.at(r * self.shape.width() + c)
    }

    /// The element at flat position `i` (row by row), as a scalar; `None` past the
    /// elements.
    pub fn at(&self, i: usize) -> Option<Scalar> {
        match &self.elements {
            Elements::Int(v) => v.get(i).map(|&b| Scalar::Int(self.format, b)),
            Elements::F32(v) => v.get(i).map(|&x| Scalar::F32(self.format, x)),
            Elements::F64(v) => v.get(i).map(|&x| Scalar::F64(x)),
        }
    }

    /// Whether the tile holds exactly as many elements as its shape names, each held as
    /// its format holds it.
    pub fn is_whole(&self) -> bool {
        let held = matches!(
            (&self.elements, held(self.format)),
            (Elements::Int(_), Held::Int)
                | (Elements::F32(_), Held::F32)
                | (Elements::F64(_), Held::F64)
        );
        held && self.elements.len() == self.shape.count()
    }

    /// Every element as an `f64`, row by row; `None` for an integer tile.
    pub fn to_f64(&self) -> Option<Vec<f64>> {
        match &self.elements {
            Elements::F32(v) => Some(v.iter().map(|&x| f64::from(x)).collect()),
            Elements::F64(v) => Some(v.clone()),
            Elements::Int(_) => None,
        }
    }

    /// Every element's mathematical value, row by row; `None` for a float tile.
    pub fn to_ints(&self) -> Option<Vec<i128>> {
        match &self.elements {
            Elements::Int(v) => Some(v.iter().map(|&b| int_value(self.format, b)).collect()),
            _ => None,
        }
    }

    /// The tile's bytes, element after element row by row, each little-endian; `None`
    /// for a format whose encoding the language does not give (see `Format::bytes`).
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        let width = self.format.bytes()?;
        let mut out = Vec::with_capacity(width * self.shape.count());
        match &self.elements {
            Elements::Int(v) => {
                for &bits in v {
                    out.extend_from_slice(&bits.to_le_bytes()[..width]);
                }
            }
            Elements::F32(v) => {
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            Elements::F64(v) => {
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        Some(out)
    }

    /// The tile of `format` and `shape` whose bytes are `bytes`; `None` when the
    /// format has no exact layout or the count is not the tile's.
    pub fn from_bytes(format: Format, shape: TileShape, bytes: &[u8]) -> Option<Tile> {
        let width = format.bytes()?;
        if bytes.len() != width * shape.count() {
            return None;
        }
        let chunks = bytes.chunks_exact(width);
        let elements = match held(format) {
            Held::Int => Elements::Int(
                chunks
                    .map(|c| {
                        let mut b = [0u8; 8];
                        b[..width].copy_from_slice(c);
                        u64::from_le_bytes(b)
                    })
                    .collect(),
            ),
            Held::F32 => Elements::F32(
                chunks
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            Held::F64 => Elements::F64(
                chunks
                    .map(|c| {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(c);
                        f64::from_le_bytes(b)
                    })
                    .collect(),
            ),
        };
        Some(Tile {
            format,
            shape,
            elements,
        })
    }
}

/// What a reduction along an axis yields: a value for each row (or column), which is
/// not a tile. Elementwise operations apply to it, with a scalar or another of its
/// kind; `.col()` (for each row) or `.row()` (for each column) makes it a tile. It is
/// held in a tile: the value of row r in column 0 of row r, of column c in row 0 of
/// column c; the other elements are never seen. A block's reduction along a row (R43,
/// R45) holds one such tile for each row of the block's tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduced {
    /// The axis it was reduced along: `Row` gives a value for each row.
    pub axis: Axis,
    /// The tiles that hold it: one for a tile's; one for each row of a block's.
    pub tiles: Vec<Tile>,
    /// Of a block's reduction, the tiles a row of that block: what `.col()` spreads
    /// it back over. None: a tile's.
    pub cols: Option<usize>,
}

/// A block of r rows of c tiles, `tile<F>[r, c]` (R45): what a stream with
/// `block = [r, c]` yields; `block = n` gives one row of n tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// The tiles, row after row, each in the view's order.
    pub tiles: Vec<Tile>,
    /// The rows of tiles; a ragged last block of one row has fewer tiles.
    pub rows: usize,
}

impl Block {
    /// A block of one row of these tiles.
    pub fn row(tiles: Vec<Tile>) -> Block {
        Block { tiles, rows: 1 }
    }

    /// The tiles a row.
    pub fn cols(&self) -> usize {
        self.tiles.len() / self.rows.max(1)
    }
}

/// A tensor's place on the heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

/// A tensor: a named array the execution context holds, addressed in tiles. Its
/// logical shape is in elements; its tiles cover it, padded up to whole tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// The name the execution context gave it.
    pub name: String,
    /// The format of its elements.
    pub format: Format,
    /// The logical shape: rows and columns of elements.
    pub shape: (usize, usize),
    /// The shape of one tile.
    pub tile: TileShape,
    /// The grid of tiles: rows and columns of tiles (the padded shape in tiles).
    pub grid: (usize, usize),
    /// The tiles, row of tiles after row of tiles: tile `i` is at grid row
    /// `i / grid.1`, column `i % grid.1`.
    pub tiles: Vec<Tile>,
}

impl Tensor {
    /// A tensor of zeros of the logical shape `rows` by `cols`, in tiles of `tile`.
    pub fn zeros(name: &str, format: Format, rows: usize, cols: usize, tile: TileShape) -> Tensor {
        let grid = (
            rows.div_ceil(tile.height().max(1)),
            cols.div_ceil(tile.width().max(1)),
        );
        Tensor {
            name: name.to_string(),
            format,
            shape: (rows, cols),
            tile,
            grid,
            tiles: vec![Tile::zeros(format, tile); grid.0 * grid.1],
        }
    }

    /// The number of tiles.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }
}

/// The tensors of one execution context.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Heap {
    /// The tensors, by `TensorId`.
    pub tensors: Vec<Tensor>,
}

impl Heap {
    /// Puts a tensor on the heap and gives its place.
    pub fn add(&mut self, tensor: Tensor) -> TensorId {
        self.tensors.push(tensor);
        TensorId(self.tensors.len() - 1)
    }

    /// The tensor at `id`.
    pub fn get(&self, id: TensorId) -> Result<&Tensor, Fault> {
        self.tensors
            .get(id.0)
            .ok_or_else(|| Fault::Undefined(format!("no tensor {}", id.0)))
    }

    /// The tensor at `id`, to write.
    pub fn get_mut(&mut self, id: TensorId) -> Result<&mut Tensor, Fault> {
        self.tensors
            .get_mut(id.0)
            .ok_or_else(|| Fault::Undefined(format!("no tensor {}", id.0)))
    }
}

/// A range with a step, `start:end:step`: `start`, `start + step`, ... below `end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Steps {
    /// The first value.
    pub start: usize,
    /// The bound, not included.
    pub end: usize,
    /// The step, above 0.
    pub step: usize,
}

impl Steps {
    /// The values of the range, in order.
    pub fn values(self) -> impl Iterator<Item = usize> {
        (self.start..self.end).step_by(self.step.max(1))
    }

    /// How many values the range has.
    pub fn len(self) -> usize {
        if self.end <= self.start {
            0
        } else {
            (self.end - self.start).div_ceil(self.step.max(1))
        }
    }

    /// Whether the range has no value.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// The tiles a view names, in the order it walks them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Walk {
    /// `t.tiles(first, n)`: n consecutive tiles.
    Run {
        /// The first tile.
        first: usize,
        /// How many.
        n: usize,
    },
    /// `t.tiles[r0:r1:s, c0:c1:s]`: a rectangle, walked row by row, or column by column
    /// once turned (`.T`).
    Rect {
        /// The rows of tiles.
        rows: Steps,
        /// The columns of tiles.
        cols: Steps,
        /// The tensor's grid width in tiles, to number a tile.
        grid_cols: usize,
        /// Whether the rectangle is turned: walked column by column.
        turned: bool,
    },
    /// A contiguous part of another walk: what `partition` gives.
    Part {
        /// The walk it is part of.
        of: Box<Walk>,
        /// The position of its first tile in that walk.
        first: usize,
        /// How many tiles.
        n: usize,
    },
}

impl Walk {
    /// The tensor's tile numbers, in the walk's order.
    pub fn tiles(&self) -> Vec<usize> {
        match self {
            Walk::Run { first, n } => (*first..first.saturating_add(*n)).collect(),
            Walk::Rect {
                rows,
                cols,
                grid_cols,
                turned,
            } => {
                let mut out = Vec::with_capacity(rows.len() * cols.len());
                if *turned {
                    for c in cols.values() {
                        out.extend(
                            rows.values()
                                .map(|r| r.saturating_mul(*grid_cols).saturating_add(c)),
                        );
                    }
                } else {
                    for r in rows.values() {
                        out.extend(
                            cols.values()
                                .map(|c| r.saturating_mul(*grid_cols).saturating_add(c)),
                        );
                    }
                }
                out
            }
            Walk::Part { of, first, n } => of.tiles().into_iter().skip(*first).take(*n).collect(),
        }
    }

    /// How many tiles the walk names.
    pub fn len(&self) -> usize {
        match self {
            Walk::Run { n, .. } | Walk::Part { n, .. } => *n,
            Walk::Rect { rows, cols, .. } => rows.len() * cols.len(),
        }
    }

    /// Whether the walk names no tile.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What a view names of its tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewKind {
    /// Tiles, in a walk.
    Tiles(Walk),
    /// `t.items(first, n, size = B)`: n items of B bytes from item `first`, item `i`
    /// being bytes `i * B` to `(i + 1) * B` of the tensor's bytes, tile after tile.
    Items {
        /// The bytes of an item.
        size: usize,
        /// The first item.
        first: usize,
        /// How many.
        n: usize,
    },
}

/// A view: a range or rectangle of a tensor's tiles, or items of a byte size. Made
/// only inside its tensor (a view past the end is the failure `view_end`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    /// The tensor it views.
    pub tensor: TensorId,
    /// What it names.
    pub kind: ViewKind,
}

impl View {
    /// How many tiles, or items, the view names.
    pub fn len(&self) -> usize {
        match &self.kind {
            ViewKind::Tiles(walk) => walk.len(),
            ViewKind::Items { n, .. } => *n,
        }
    }

    /// Whether the view names nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A stream over a view: its blocks of `block` tiles, in order; a ragged end is one
/// shorter block. It reads the heap as it goes.
#[derive(Clone, Debug, PartialEq)]
pub struct Stream {
    /// The view it reads.
    pub view: View,
    /// The tiles of a row of a block.
    pub block: usize,
    /// The rows of a block (R45): more than one, the blocks of a rectangle view.
    pub rows: usize,
    /// The position in the view of the next tile to read.
    pub next: usize,
}

/// Two streams walked together, block by block: `zip(s1, s2)`. Made only of two
/// streams of as many blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct Zip {
    /// The first stream.
    pub a: Stream,
    /// The second stream.
    pub b: Stream,
}

/// A write stream over a view: each push writes the next block. Only the tiles the
/// view names are written.
#[derive(Clone, Debug, PartialEq)]
pub struct Sink {
    /// The view it writes.
    pub view: View,
    /// The tiles of a block.
    pub block: usize,
    /// The position in the view of the next tile to write.
    pub next: usize,
}

/// The type of a record: named fields of scalar types, laid out one after another with
/// no gap, in the order declared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordType {
    /// The record's name.
    pub name: String,
    /// Its fields, in order: a name and a scalar type.
    pub fields: Vec<(String, Format)>,
}

/// What a state member holds, element by element.
#[derive(Clone, Debug, PartialEq)]
pub enum MemberData {
    /// Tiles: `tile<F>[n]`.
    Tiles(Vec<Tile>),
    /// Scalars: `u32[n]` and the like.
    Scalars(Vec<Scalar>),
}

/// Which parts of a member are filled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Filled {
    /// Filled whole, or not at all: the default.
    Whole(bool),
    /// `@filled(element)`: each element on its own.
    Element(Vec<bool>),
}

/// One member of a worker's state. Empty until filled; reading what is not filled is
/// the failure `empty_state`. Writing is always allowed.
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    /// The member's name.
    pub name: String,
    /// The format of its elements.
    pub format: Format,
    /// Its elements.
    pub data: MemberData,
    /// What is filled.
    pub filled: Filled,
}

impl Member {
    /// An empty member of `n` tiles of `format` and `shape`; `by_element` for a member
    /// marked `@filled(element)`.
    pub fn tiles(
        name: &str,
        format: Format,
        shape: TileShape,
        n: usize,
        by_element: bool,
    ) -> Member {
        Member {
            name: name.to_string(),
            format,
            data: MemberData::Tiles(vec![Tile::zeros(format, shape); n]),
            filled: Member::empty(n, by_element),
        }
    }

    /// An empty member of `n` scalars of `format`; `None` when `format` is no scalar
    /// type.
    pub fn scalars(name: &str, format: Format, n: usize, by_element: bool) -> Option<Member> {
        if !format.as_scalar().is_some() {
            return None;
        }
        let zero = match held(format) {
            Held::Int => Scalar::Int(format, 0),
            Held::F32 => Scalar::F32(format, 0.0),
            Held::F64 => Scalar::F64(0.0),
        };
        Some(Member {
            name: name.to_string(),
            format,
            data: MemberData::Scalars(vec![zero; n]),
            filled: Member::empty(n, by_element),
        })
    }

    fn empty(n: usize, by_element: bool) -> Filled {
        if by_element {
            Filled::Element(vec![false; n])
        } else {
            Filled::Whole(false)
        }
    }

    /// How many elements the member has.
    pub fn len(&self) -> usize {
        match &self.data {
            MemberData::Tiles(v) => v.len(),
            MemberData::Scalars(v) => v.len(),
        }
    }

    /// Whether the member has no element.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether element `i` may be read.
    pub fn is_filled(&self, i: usize) -> bool {
        match &self.filled {
            Filled::Whole(filled) => *filled,
            Filled::Element(marks) => marks.get(i).copied().unwrap_or(false),
        }
    }
}

/// Why an operation gave no value. Returned, never panicked.
#[derive(Clone, Debug, PartialEq)]
pub enum Fault {
    /// A named failure of the invocation.
    Failure(Failure),
    /// The operation is not defined on these values. Program compile refuses every
    /// program that would do this; the interpreter reports it rather than guess.
    Undefined(String),
}

impl Fault {
    /// The fault of an operation not defined on its values.
    pub fn undefined(what: impl Into<String>) -> Fault {
        Fault::Undefined(what.into())
    }
}

impl From<Failure> for Fault {
    fn from(failure: Failure) -> Fault {
        Fault::Failure(failure)
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Failure(failure) => f.write_str(failure.name()),
            Fault::Undefined(what) => write!(f, "not defined: {what}"),
        }
    }
}

// ---- the place record: races (B2) ----

/// A place of a tensor (B2): bytes `lo .. hi` of one tile. An access through tiles
/// names the whole tile; an item names its bytes in each tile it lies in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorPlace {
    /// The tensor.
    pub tensor: usize,
    /// The tile.
    pub tile: usize,
    /// The first byte of the tile.
    pub lo: usize,
    /// After the last byte; `usize::MAX` for the whole tile.
    pub hi: usize,
}

impl TensorPlace {
    /// A whole tile.
    pub fn tile(tensor: TensorId, tile: usize) -> TensorPlace {
        TensorPlace {
            tensor: tensor.0,
            tile,
            lo: 0,
            hi: usize::MAX,
        }
    }

    fn meets(&self, other: &TensorPlace) -> bool {
        self.lo < other.hi && other.lo < self.hi
    }
}

/// One access to a place: which invocation (its number in the record), which rank,
/// that rank's clock entry when it acted, whether it wrote, and where.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Act {
    /// The invocation, by its number in the record.
    pub inv: usize,
    /// The rank in its group.
    pub rank: usize,
    /// The rank's own clock entry at the access.
    pub epoch: u64,
    /// A write (else a read).
    pub write: bool,
    /// The body and instruction.
    pub at: (crate::program::BodyRef, crate::program::body::InstId),
}

/// The accesses kept for one tile: the last writes and the reads since, each with its
/// bytes.
#[derive(Clone, Debug, Default, PartialEq)]
struct TileRecord {
    writes: Vec<(TensorPlace, Act)>,
    reads: Vec<(TensorPlace, Act)>,
}

/// The record that finds races. Across invocations, an access is ordered after
/// another when the other's invocation is among its own's predecessors (completion,
/// end, a chain's earlier call, or what the host or a control program saw end
/// before it made this one). Inside an invocation, each rank has a vector clock over
/// the group, joined at a meeting, an arrival of a broadcast and a message; a write
/// on another rank is seen only through a publication of its place, which carries
/// the publisher's clock for it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Races {
    tiles: std::collections::BTreeMap<(usize, usize), TileRecord>,
    /// Each invocation's predecessors, closed under "before".
    before: Vec<std::collections::BTreeSet<usize>>,
    /// Each invocation's line in the run's report.
    lines: Vec<usize>,
    /// Each rank's clock, in the invocation running.
    clocks: Vec<Vec<u64>>,
    /// What each rank sees of each place's writes: a clock, by (rank, tensor, tile).
    seen: std::collections::BTreeMap<(usize, usize, usize), Vec<u64>>,
}

/// A clock made the join of itself and another.
fn join(into: &mut Vec<u64>, other: &[u64]) {
    if into.len() < other.len() {
        into.resize(other.len(), 0);
    }
    for (a, b) in into.iter_mut().zip(other) {
        *a = (*a).max(*b);
    }
}

impl Races {
    /// How many invocations the record has begun.
    pub fn count(&self) -> usize {
        self.before.len()
    }

    /// Begins an invocation on a group of `size` ranks, after `before` (and all
    /// before them), whose line in the report is `line`; gives its number.
    pub fn begin(
        &mut self,
        before: &std::collections::BTreeSet<usize>,
        size: usize,
        line: usize,
    ) -> usize {
        let mut all = before.clone();
        for b in before {
            if let Some(more) = self.before.get(*b) {
                all.extend(more.iter().copied());
            }
        }
        self.before.push(all);
        self.lines.push(line);
        self.clocks = (0..size)
            .map(|r| {
                let mut c = vec![0; size];
                if let Some(own) = c.get_mut(r) {
                    *own = 1;
                }
                c
            })
            .collect();
        self.seen.clear();
        self.before.len() - 1
    }

    /// Every invocation begun so far: what the host has seen end.
    pub fn all(&self) -> std::collections::BTreeSet<usize> {
        (0..self.before.len()).collect()
    }

    /// Rank `rank`'s clock as an edge carries it from here; the rank's own entry
    /// moves on, so what it does next is after the edge.
    pub fn release(&mut self, rank: usize) -> Vec<u64> {
        let Some(clock) = self.clocks.get_mut(rank) else {
            return Vec::new();
        };
        let out = clock.clone();
        if let Some(own) = clock.get_mut(rank) {
            *own += 1;
        }
        out
    }

    /// An edge arrives at `rank` with a clock.
    pub fn acquire(&mut self, rank: usize, clock: &[u64]) {
        if let Some(mine) = self.clocks.get_mut(rank) {
            join(mine, clock);
        }
    }

    /// The publications `rank` makes of the tiles named: for each, what it sees of
    /// the tile's writes, its own up to now included.
    pub fn publish(
        &self,
        rank: usize,
        tiles: &[(usize, usize)],
    ) -> Vec<((usize, usize), Vec<u64>)> {
        let own = self
            .clocks
            .get(rank)
            .and_then(|c| c.get(rank))
            .copied()
            .unwrap_or(0);
        tiles
            .iter()
            .map(|&(t, n)| {
                let mut c = self.seen.get(&(rank, t, n)).cloned().unwrap_or_default();
                if c.len() <= rank {
                    c.resize(rank + 1, 0);
                }
                if let Some(e) = c.get_mut(rank) {
                    *e = (*e).max(own);
                }
                ((t, n), c)
            })
            .collect()
    }

    /// Publications arrive at `rank`.
    pub fn receive(&mut self, rank: usize, publications: &[((usize, usize), Vec<u64>)]) {
        for ((t, n), c) in publications {
            join(self.seen.entry((rank, *t, *n)).or_default(), c);
        }
    }

    /// An access by `rank` of the invocation `inv` to a place: a race with an earlier
    /// access is given as its description, else it is recorded.
    pub fn access(
        &mut self,
        inv: usize,
        rank: usize,
        place: TensorPlace,
        write: bool,
        at: (crate::program::BodyRef, crate::program::body::InstId),
    ) -> Result<(), String> {
        let epoch = self
            .clocks
            .get(rank)
            .and_then(|c| c.get(rank))
            .copied()
            .unwrap_or(0);
        let act = Act {
            inv,
            rank,
            epoch,
            write,
            at,
        };
        let record = self.tiles.entry((place.tensor, place.tile)).or_default();
        let earlier = record
            .writes
            .iter()
            .chain(record.reads.iter().filter(|_| write));
        for (p, a) in earlier {
            if !p.meets(&place) || ordered(&self.before, &self.clocks, &self.seen, a, &act, place) {
                continue;
            }
            return Err(race_text(&self.lines, place, a, &act));
        }
        if write {
            record
                .writes
                .retain(|(p, _)| !(place.lo <= p.lo && p.hi <= place.hi));
            record
                .reads
                .retain(|(p, _)| !(place.lo <= p.lo && p.hi <= place.hi));
            record.writes.push((place, act));
        } else {
            record
                .reads
                .retain(|(p, a)| !(*p == place && a.inv == inv && a.rank == rank));
            record.reads.push((place, act));
        }
        Ok(())
    }
}

/// Whether the earlier act `a` is ordered before `b` on `place`.
fn ordered(
    before: &[std::collections::BTreeSet<usize>],
    clocks: &[Vec<u64>],
    seen: &std::collections::BTreeMap<(usize, usize, usize), Vec<u64>>,
    a: &Act,
    b: &Act,
    place: TensorPlace,
) -> bool {
    if a.inv != b.inv {
        return before.get(b.inv).is_some_and(|s| s.contains(&a.inv));
    }
    if a.rank == b.rank {
        return true;
    }
    let entry = |c: Option<&Vec<u64>>| c.and_then(|c| c.get(a.rank)).copied().unwrap_or(0);
    if a.write {
        // a write on another rank is seen only through a publication of its place
        entry(seen.get(&(b.rank, place.tensor, place.tile))) >= a.epoch
    } else {
        entry(clocks.get(b.rank)) >= a.epoch
    }
}

fn race_text(lines: &[usize], place: TensorPlace, a: &Act, b: &Act) -> String {
    let what = |x: &Act| if x.write { "a write" } else { "a read" };
    let line = |x: &Act| lines.get(x.inv).copied().unwrap_or(0);
    let at = |x: &Act| {
        format!(
            "{} i{}",
            crate::program::print::body_ref(&x.at.0),
            x.at.1 .0
        )
    };
    let edge = if a.inv != b.inv {
        "no completion, end or chain edge orders the two invocations"
    } else if a.write {
        "no publication of the place reaches the second rank"
    } else {
        "no edge orders the two ranks"
    };
    format!(
        "a race on tensor {} tile {}: {} at #{} rank {} ({}) and {} at #{} rank {} ({}): {edge}",
        place.tensor,
        place.tile,
        what(a),
        line(a),
        a.rank,
        at(a),
        what(b),
        line(b),
        b.rank,
        at(b)
    )
}
