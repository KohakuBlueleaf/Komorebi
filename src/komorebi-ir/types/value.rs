//! The type of a value: what a name stands for and what an operation yields, with the
//! tile shape, the axis of a reduction, counts, and references to declarations.

use std::fmt;

use super::{Format, MessageId, ParamId, RecordId, ScalarType};

/// The module's tile shape, in elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileShape {
    /// Rows.
    pub rows: u64,
    /// Columns.
    pub cols: u64,
}

impl TileShape {
    /// The elements of one tile.
    pub fn elements(self) -> u64 {
        self.rows * self.cols
    }
}

/// The axis of a reduction: a value for each row, or for each column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// `axis = row`: one value for each row.
    Row,
    /// `axis = col`: one value for each column.
    Col,
}

/// How many tiles (or bytes): a constant, a module parameter's value, or the size of
/// the group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Count {
    /// A number the text writes.
    Const(u64),
    /// The value of a module parameter (`@bind(load)` or `@bind(invoke)`).
    Param(ParamId),
    /// The size of the invocation's group: what `gather` yields.
    GroupSize,
}

/// The type of a value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// A scalar.
    Scalar(ScalarType),
    /// `bool`: the result of a comparison, a `send` or an `offer`.
    Bool,
    /// `tile<F>`.
    Tile(Format),
    /// A value for each row or each column of a tile of this format, or of each tile
    /// of a block of these rows and columns of tiles (a tile's: 1 and 1): what a
    /// reduction yields. Not a tile: `.col()` or `.row()` spreads it into one.
    Reduced(Format, Axis, Count, Count),
    /// `tile<F>[r, c]`: a block of r rows of c tiles (R45); `tile<F>[n]` is `[1, n]`.
    Block(Format, Count, Count),
    /// `tensor<F>`, and whether it is `@replicated`.
    Tensor(Format, bool),
    /// A view of tiles of a tensor of this format; with none, a view of any format: what
    /// a parameter declared `view` takes.
    View(Option<Format>),
    /// A view of items of this many bytes: `t.items(...)`.
    Items(Count),
    /// A read stream of blocks of this format, of these rows and columns of tiles.
    Stream(Format, Count, Count),
    /// Two streams walked together: `zip(s1, s2)`.
    Zip(Box<Type>, Box<Type>),
    /// A write stream of blocks of this format, of these rows and columns of tiles.
    Sink(Format, Count, Count),
    /// The scalars of one tile: `v.words()`.
    Words(ScalarType),
    /// The records of one tile: `v.records(R, count = c)`.
    Records(RecordId),
    /// One record.
    RecordValue(RecordId),
    /// A message.
    Message(MessageId),
    /// A message's body: the run of bytes that belongs to the message that arrived.
    Body,
    /// Sub-groups of the group: `group.split(...)`.
    SubGroups,
    /// Their leaders: `group.leaders(...)`.
    Leaders,
    /// A broadcast: `broadcast(...)`, with the format of the tiles its casts send.
    Broadcast(Format),
    /// A group of workers, for `invoke`'s `on`.
    Group,
    /// An invocation's handle, with its result's type when it has one.
    Handle(Option<ScalarType>),
    /// An invocation's outcome, with its value's type when it has one.
    Outcome(Option<ScalarType>),
    /// A failure's name, compared with `==`.
    Reason,
}

impl Count {
    /// The one row of a block written `[n]`.
    pub const ONE: Count = Count::Const(1);
}

/// The shape of a block as the text writes it: `[n]` for one row, else `[r, c]`.
pub fn shape_text(rows: &Count, cols: &Count) -> String {
    match rows {
        Count::Const(1) => format!("[{cols}]"),
        _ => format!("[{rows}, {cols}]"),
    }
}

impl fmt::Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Count::Const(n) => write!(f, "{n}"),
            Count::Param(p) => write!(f, "parameter {}", p.0),
            Count::GroupSize => f.write_str("group.size"),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Scalar(s) => write!(f, "{s}"),
            Type::Bool => f.write_str("bool"),
            Type::Tile(fm) => write!(f, "tile<{fm}>"),
            Type::Reduced(fm, axis, r, c) => {
                let axis = match axis {
                    Axis::Row => "row",
                    Axis::Col => "column",
                };
                write!(f, "a value of {fm} for each {axis}")?;
                match (r, c) {
                    (Count::Const(1), Count::Const(1)) => Ok(()),
                    _ => write!(f, " of tile<{fm}>{}", shape_text(r, c)),
                }
            }
            Type::Block(fm, r, c) => write!(f, "tile<{fm}>{}", shape_text(r, c)),
            Type::Tensor(fm, false) => write!(f, "tensor<{fm}>"),
            Type::Tensor(fm, true) => write!(f, "tensor<{fm}> @replicated"),
            Type::View(Some(fm)) => write!(f, "a view of tensor<{fm}>"),
            Type::View(None) => f.write_str("a view"),
            Type::Items(n) => write!(f, "items of {n} bytes"),
            Type::Stream(fm, Count::Const(1), n) => {
                write!(f, "a stream of tile<{fm}>, {n} at a time")
            }
            Type::Stream(fm, r, c) => write!(f, "a stream of tile<{fm}>{}", shape_text(r, c)),
            Type::Zip(a, b) => write!(f, "a zip of {a} and {b}"),
            Type::Sink(fm, Count::Const(1), n) => write!(f, "a sink of tile<{fm}>, {n} at a time"),
            Type::Sink(fm, r, c) => write!(f, "a sink of tile<{fm}>{}", shape_text(r, c)),
            Type::Words(s) => write!(f, "the {s} words of a tile"),
            Type::Records(r) => write!(f, "the records of type {}", r.0),
            Type::RecordValue(r) => write!(f, "a record of type {}", r.0),
            Type::Message(m) => write!(f, "a message of type {}", m.0),
            Type::Body => f.write_str("a message's body"),
            Type::SubGroups => f.write_str("sub-groups"),
            Type::Leaders => f.write_str("the leaders of sub-groups"),
            Type::Broadcast(g) => write!(f, "a broadcast of {g} tiles"),
            Type::Group => f.write_str("a group"),
            Type::Handle(_) => f.write_str("an invocation's handle"),
            Type::Outcome(_) => f.write_str("an outcome"),
            Type::Reason => f.write_str("a failure's name"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_forms() {
        assert_eq!(Type::Tile(Format::Bf16).to_string(), "tile<bf16>");
        assert_eq!(
            Type::Block(Format::F32, Count::ONE, Count::Const(4)).to_string(),
            "tile<f32>[4]"
        );
        assert_eq!(
            Type::Block(Format::F32, Count::ONE, Count::GroupSize).to_string(),
            "tile<f32>[group.size]"
        );
        assert_eq!(
            Type::Block(Format::Bf16, Count::Const(4), Count::Const(1)).to_string(),
            "tile<bf16>[4, 1]"
        );
        assert_eq!(Type::Scalar(ScalarType::U32).to_string(), "u32");
    }

    #[test]
    fn a_tile_has_its_elements() {
        assert_eq!(TileShape { rows: 32, cols: 32 }.elements(), 1024);
    }
}
