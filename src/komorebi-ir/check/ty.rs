//! The checker's working type of a value: a `types::Type` as far as the text shows
//! it, with `Unknown` after an error and the counts the text states, which some
//! rules compare.

use std::fmt;

use crate::types::{Axis, Format};

/// How many tiles or elements: a number the text states, or one it leaves to a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Count {
    /// A number written as a literal.
    Known(u64),
    /// A number given by a parameter or an argument.
    Unknown,
}

impl Count {
    /// True when the count is known to be more than one.
    pub fn more_than_one(self) -> bool {
        matches!(self, Count::Known(n) if n > 1)
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
    /// An integer scalar.
    Int,
    /// A float scalar.
    Float,
    /// `bool`: the result of a comparison, a `send` or an `offer`.
    Bool,
    /// One tile, of a format, or of any format (`zeros`).
    Tile(Option<Format>),
    /// An elementwise comparison of tiles: the first operand of `where`.
    Mask,
    /// What `reduce_sum(a, axis = ...)` and its kin yield: one value for each row or
    /// column. Not a tile: `.col()` (`.row()` along columns) spreads it into one.
    Reduced(Axis),
    /// `tile<F>[n]`: a block of n tiles.
    Block(Option<Format>, Count),
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
    /// `t.items(...)`: items of a byte size, with their number when the text shows it.
    Items(Count),
    /// `v.words()`: the scalars of one tile.
    Words,
    /// `v.records(R, count = c)`: c records of type R.
    Records(String, Count),
    /// One record of type R.
    Record(String),
    /// A message of type M.
    Message(String),
    /// `stream(...)`: blocks of the element type, in order, from a view of a number of
    /// tiles.
    Stream(Box<Ty>, Count),
    /// `zip(s1, s2, ...)`: several streams walked together.
    Zip(Vec<Ty>),
    /// `sink(...)`: a write stream.
    Sink,
    /// `broadcast(...)`.
    Broadcast,
    /// `b.arrive()`: each block and its word.
    Arrive,
    /// `range(n)`.
    Range,
    /// The handle of an invocation, with its result type when it has one.
    Handle(Option<Box<Ty>>),
    /// `outcome(h)`.
    Outcome,
    /// `o.reason`: the name of a failure.
    Reason,
    /// `group(first, count)`: workers an invocation runs on.
    Group,
    /// `group.split(...)` or `group.leaders(...)`.
    Parts,
    /// A string.
    Str,
    /// `[a, b, ...]`.
    List,
}

impl Ty {
    /// True for an integer or a float scalar.
    pub fn is_number(&self) -> bool {
        matches!(self, Ty::Int | Ty::Float)
    }

    /// True for a value that can be computed on elementwise: a tile or a scalar.
    pub fn is_tile_operand(&self) -> bool {
        matches!(self, Ty::Tile(_) | Ty::Int | Ty::Float)
    }

    /// True when a value of this type may stand where `want` is asked. Formats are
    /// notation and are not compared; integers of any width agree.
    pub fn fits(&self, want: &Ty) -> bool {
        match (self, want) {
            (Ty::Unknown, _) | (_, Ty::Unknown) => true,
            (Ty::Int, Ty::Int) | (Ty::Int, Ty::Float) | (Ty::Float, Ty::Float) => true,
            (Ty::Bool, Ty::Bool) => true,
            (Ty::Tile(_), Ty::Tile(_)) => true,
            (Ty::Block(_, _), Ty::Block(_, _)) => true,
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
            Ty::Int => f.write_str("an integer"),
            Ty::Float => f.write_str("a float"),
            Ty::Bool => f.write_str("bool"),
            Ty::Tile(fm) => write!(f, "tile<{}>", format(fm)),
            Ty::Mask => f.write_str("a comparison of tiles"),
            Ty::Reduced(Axis::Row) => f.write_str("a value for each row"),
            Ty::Reduced(Axis::Col) => f.write_str("a value for each column"),
            Ty::Block(fm, Count::Known(n)) => write!(f, "tile<{}>[{n}]", format(fm)),
            Ty::Block(fm, Count::Unknown) => write!(f, "tile<{}>[n]", format(fm)),
            Ty::Scalars(_, _) => f.write_str("scalars"),
            Ty::Tensor { format: fm, .. } => write!(f, "tensor<{}>", format(fm)),
            Ty::TileGrid(_) => f.write_str("the tiles of a tensor"),
            Ty::View { .. } => f.write_str("a view"),
            Ty::Items(_) => f.write_str("items"),
            Ty::Words => f.write_str("the words of a tile"),
            Ty::Records(r, _) => write!(f, "records {r}"),
            Ty::Record(r) => write!(f, "a record {r}"),
            Ty::Message(m) => write!(f, "a message {m}"),
            Ty::Stream(_, _) => f.write_str("a stream"),
            Ty::Zip(_) => f.write_str("a zip of streams"),
            Ty::Sink => f.write_str("a sink"),
            Ty::Broadcast => f.write_str("a broadcast"),
            Ty::Arrive => f.write_str("the arrivals of a broadcast"),
            Ty::Range => f.write_str("a range"),
            Ty::Handle(_) => f.write_str("an invocation's handle"),
            Ty::Outcome => f.write_str("an outcome"),
            Ty::Reason => f.write_str("a failure's name"),
            Ty::Group => f.write_str("a group"),
            Ty::Parts => f.write_str("sub-groups"),
            Ty::Str => f.write_str("a string"),
            Ty::List => f.write_str("a list"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_fits_everything() {
        assert!(Ty::Unknown.fits(&Ty::Int));
        assert!(Ty::Tile(None).fits(&Ty::Unknown));
    }

    #[test]
    fn formats_are_not_compared() {
        assert!(Ty::Tile(Some(Format::F32)).fits(&Ty::Tile(Some(Format::Bf16))));
    }

    #[test]
    fn a_tile_is_not_a_scalar() {
        assert!(!Ty::Tile(None).fits(&Ty::Int));
        assert!(!Ty::Block(None, Count::Known(4)).fits(&Ty::Tile(None)));
    }
}
