//! The checker's working type of a value: a `types::Type` as far as the text shows
//! it, with `Unknown` after an error and the counts the text states, which some
//! rules compare.

use std::fmt;

use crate::types::{Axis, Format, ParamId, ScalarType};

/// How many tiles or elements: a number the text states (a module parameter bound at
/// build included), a module parameter's value, the group's size, or another value the
/// run gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Count {
    /// A number known now.
    Known(u64),
    /// A module parameter bound at load or at the invocation.
    Param(ParamId),
    /// The size of the group: what `gather` yields.
    GroupSize,
    /// Another value: an argument, or an expression.
    Unknown,
}

impl Count {
    /// True when the count is known to be more than one.
    pub fn more_than_one(self) -> bool {
        matches!(self, Count::Known(n) if n > 1)
    }
}

/// The shape of a block: its rows and its columns of tiles (R45).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Rows of tiles.
    pub rows: Count,
    /// Tiles a row.
    pub cols: Count,
}

impl Shape {
    /// A tile's shape: one row of one tile.
    pub const TILE: Shape = Shape::row(Count::Known(1));

    /// One row of n tiles: what `[n]` writes.
    pub const fn row(cols: Count) -> Shape {
        Shape {
            rows: Count::Known(1),
            cols,
        }
    }

    /// Whether the two shapes may be one: equal, or either not known.
    pub fn agrees(self, other: Shape) -> bool {
        let one = |a: Count, b: Count| a == b || a == Count::Unknown || b == Count::Unknown;
        one(self.rows, other.rows) && one(self.cols, other.cols)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = |c: Count| match c {
            Count::Known(n) => n.to_string(),
            Count::GroupSize => "group.size".to_string(),
            _ => "n".to_string(),
        };
        match self.rows {
            Count::Known(1) => write!(f, "[{}]", count(self.cols)),
            rows => write!(f, "[{}, {}]", count(rows), count(self.cols)),
        }
    }
}

/// A number written in the text, before its place gives it a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lit {
    /// A whole number: `u32` where its place gives no type.
    Whole,
    /// A number with a point or an exponent: `f32` where its place gives no type.
    Point,
}

impl Lit {
    /// The type of the number where its place gives none.
    pub fn alone(self) -> ScalarType {
        match self {
            Lit::Whole => ScalarType::U32,
            Lit::Point => ScalarType::F32,
        }
    }
}

/// The type of a value. `Unknown` follows an error, so that one mistake gives one
/// diagnostic: it agrees with every type.
#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    /// Not known, after an error.
    Unknown,
    /// No value: an operation done for its effect.
    Void,
    /// A scalar of a type.
    Scalar(ScalarType),
    /// A number written in the text (or an operation on such numbers only) whose type
    /// its place decides: the other operand's, a parameter's, a field's, a declared
    /// type. Where the place gives none, `u32` for a whole number and `f32` for one
    /// with a point or an exponent.
    Number(Lit),
    /// What some operands ask: an integer of any type. No value has it.
    AnyInt,
    /// What some operands ask: a number of any type. No value has it.
    AnyNumber,
    /// `bool`: the result of a comparison, a `send` or an `offer`.
    Bool,
    /// One tile, of a format, or of any format (`zeros`).
    Tile(Option<Format>),
    /// An elementwise comparison of tiles: the first operand of `where`.
    Mask,
    /// What `reduce_sum(a, axis = ...)` and its kin yield from a tile of a format: one
    /// value for each row or column; of a block, of each of its tiles, the block's
    /// shape kept (a tile's: 1 and 1). Not a tile: `.col()` (`.row()` along columns)
    /// spreads it into one.
    Reduced(Option<Format>, Axis, Shape),
    /// `tile<F>[r, c]`: a block of r rows of c tiles (R45); `tile<F>[n]` is `[1, n]`.
    Block(Option<Format>, Shape),
    /// n scalars, such as a state member `u32[4]`.
    Scalars(Box<Ty>, Count),
    /// `tensor<F>`, with one copy for each island when `replicated`.
    Tensor {
        /// The element format.
        format: Option<Format>,
        /// `@replicated`.
        replicated: bool,
    },
    /// `t.tiles` before its rectangle `[r0:r1, c0:c1]`.
    TileGrid(Option<Format>),
    /// A view of a tensor's tiles, with the number of tiles when the text shows it.
    View {
        /// The element format.
        format: Option<Format>,
        /// How many tiles the view names.
        tiles: Count,
    },
    /// `t.items(...)`: items, how many (where the text shows it), and the bytes of one.
    Items(Count, Count),
    /// `v.words()`: the scalars of one tile, of the scalar type of the tensor's format.
    Words(Option<ScalarType>),
    /// `v.records(R, count = c)`: c records of type R.
    Records(String, Count),
    /// One record of type R.
    Record(String),
    /// A message of type M.
    Message(String),
    /// `m.body`: the body of a message, the run of bytes that belongs to it.
    Body,
    /// `stream(...)`: blocks of the element type, in order, from a view of a number of
    /// tiles.
    Stream(Box<Ty>, Count),
    /// `zip(s1, s2, ...)`: several streams walked together.
    Zip(Vec<Ty>),
    /// `sink(view, block = n)`: a write stream of blocks of n tiles of a format.
    Sink(Option<Format>, Shape),
    /// `broadcast(...)`, by the place of that call: its format is what its casts send.
    Broadcast((usize, usize)),
    /// `b.arrive()` of the broadcast at that place: each block and its word.
    Arrive((usize, usize)),
    /// `range(n)`: the numbers below n, of n's type.
    Range(ScalarType),
    /// The handle of an invocation, with its result type when it has one.
    Handle(Option<Box<Ty>>),
    /// `outcome(h)`, with the invocation's result type when it has one.
    Outcome(Option<Box<Ty>>),
    /// `o.reason`: the name of a failure.
    Reason,
    /// `group(first, count)`: workers an invocation runs on.
    Group,
    /// `group.split(...)`: the group in sub-groups.
    SubGroups,
    /// `group.leaders(...)`: the leaders of the sub-groups.
    Leaders,
    /// A string.
    Str,
    /// `[a, b, ...]`.
    List,
}

impl Ty {
    /// True for a scalar or a number written in the text.
    pub fn is_number(&self) -> bool {
        matches!(self, Ty::Scalar(_) | Ty::Number(_))
    }

    /// True for an integer scalar, or a whole number written in the text.
    pub fn is_integer(&self) -> bool {
        match self {
            Ty::Scalar(s) => !s.is_float(),
            Ty::Number(lit) => *lit == Lit::Whole,
            _ => false,
        }
    }

    /// True for a float scalar, or a number written with a point or an exponent.
    pub fn is_float(&self) -> bool {
        match self {
            Ty::Scalar(s) => s.is_float(),
            Ty::Number(lit) => *lit == Lit::Point,
            _ => false,
        }
    }

    /// True for a value that can be computed on elementwise: a tile or a scalar.
    pub fn is_tile_operand(&self) -> bool {
        matches!(self, Ty::Tile(_)) || self.is_number()
    }

    /// True when a value of this type may stand where `want` is asked. Formats are
    /// notation and are not compared. A scalar stands only where its own type is
    /// asked (nothing is converted silently); a number written in the text stands
    /// where a type that can hold it is asked.
    pub fn fits(&self, want: &Ty) -> bool {
        match (self, want) {
            (Ty::Unknown, _) | (_, Ty::Unknown) => true,
            (Ty::Scalar(a), Ty::Scalar(b)) => a == b,
            (Ty::Number(Lit::Whole), Ty::Scalar(_)) => true,
            (Ty::Number(Lit::Point), Ty::Scalar(b)) => b.is_float(),
            (got, Ty::AnyInt) => got.is_integer(),
            (got, Ty::AnyNumber) => got.is_number(),
            (Ty::Bool, Ty::Bool) => true,
            // a float format may stand for another (a notation); integer formats agree
            (Ty::Tile(Some(a)), Ty::Tile(Some(b))) => a == b || !(a.is_integer() || b.is_integer()),
            (Ty::Tile(_), Ty::Tile(_)) => true,
            (Ty::Block(..), Ty::Block(..)) => true,
            (Ty::Scalars(_, _), Ty::Scalars(_, _)) => true,
            (Ty::Tensor { .. }, Ty::Tensor { .. }) => true,
            (Ty::View { .. }, Ty::View { .. }) => true,
            (Ty::Record(a), Ty::Record(b)) | (Ty::Message(a), Ty::Message(b)) => a == b,
            (a, b) => std::mem::discriminant(a) == std::mem::discriminant(b),
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format = |format: &Option<Format>| match format {
            Some(format) => format.name().to_string(),
            None => "F".to_string(),
        };
        match self {
            Ty::Unknown => f.write_str("an unknown value"),
            Ty::Void => f.write_str("no value"),
            Ty::Scalar(s) => write!(f, "{s}"),
            Ty::Number(Lit::Whole) => f.write_str("a whole number"),
            Ty::Number(Lit::Point) => f.write_str("a number with a point"),
            Ty::AnyInt => f.write_str("an integer"),
            Ty::AnyNumber => f.write_str("a number"),
            Ty::Bool => f.write_str("bool"),
            Ty::Tile(fm) => write!(f, "tile<{}>", format(fm)),
            Ty::Mask => f.write_str("a comparison of tiles"),
            Ty::Reduced(_, axis, shape) => {
                let axis = match axis {
                    Axis::Row => "row",
                    Axis::Col => "column",
                };
                write!(f, "a value for each {axis}")?;
                if *shape != Shape::TILE {
                    write!(f, " of a block {shape}")?;
                }
                Ok(())
            }
            Ty::Block(fm, shape) => write!(f, "tile<{}>{shape}", format(fm)),
            Ty::Scalars(_, _) => f.write_str("scalars"),
            Ty::Tensor { format: fm, .. } => write!(f, "tensor<{}>", format(fm)),
            Ty::TileGrid(_) => f.write_str("the tiles of a tensor"),
            Ty::View { .. } => f.write_str("a view"),
            Ty::Items(..) => f.write_str("items"),
            Ty::Words(_) => f.write_str("the words of a tile"),
            Ty::Records(r, _) => write!(f, "records {r}"),
            Ty::Record(r) => write!(f, "a record {r}"),
            Ty::Message(m) => write!(f, "a message {m}"),
            Ty::Body => f.write_str("the body of a message"),
            Ty::Stream(_, _) => f.write_str("a stream"),
            Ty::Zip(_) => f.write_str("a zip of streams"),
            Ty::Sink(_, _) => f.write_str("a sink"),
            Ty::Broadcast(_) => f.write_str("a broadcast"),
            Ty::Arrive(_) => f.write_str("the arrivals of a broadcast"),
            Ty::Range(_) => f.write_str("a range"),
            Ty::Handle(_) => f.write_str("an invocation's handle"),
            Ty::Outcome(_) => f.write_str("an outcome"),
            Ty::Reason => f.write_str("a failure's name"),
            Ty::Group => f.write_str("a group"),
            Ty::SubGroups => f.write_str("sub-groups"),
            Ty::Leaders => f.write_str("the leaders of sub-groups"),
            Ty::Str => f.write_str("a string"),
            Ty::List => f.write_str("a list"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scalar_is_not_converted_silently() {
        let u32_ = Ty::Scalar(ScalarType::U32);
        assert!(u32_.fits(&u32_));
        assert!(!u32_.fits(&Ty::Scalar(ScalarType::I32)));
        assert!(u32_.fits(&Ty::AnyInt));
        assert!(Ty::Number(Lit::Whole).fits(&Ty::Scalar(ScalarType::F16)));
        assert!(!Ty::Number(Lit::Point).fits(&Ty::Scalar(ScalarType::U8)));
        assert!(!Ty::Number(Lit::Point).fits(&Ty::AnyInt));
    }

    #[test]
    fn unknown_fits_everything() {
        assert!(Ty::Unknown.fits(&Ty::AnyInt));
        assert!(Ty::Tile(None).fits(&Ty::Unknown));
    }

    #[test]
    fn formats_are_not_compared() {
        assert!(Ty::Tile(Some(Format::F32)).fits(&Ty::Tile(Some(Format::Bf16))));
    }

    #[test]
    fn a_tile_is_not_a_scalar() {
        assert!(!Ty::Tile(None).fits(&Ty::AnyInt));
        assert!(!Ty::Block(None, Shape::row(Count::Known(4))).fits(&Ty::Tile(None)));
    }
}
