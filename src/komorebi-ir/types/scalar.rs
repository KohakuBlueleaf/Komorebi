//! The scalar types: eleven names, a type of their own. A block format is never a
//! scalar type.

use std::fmt;

use super::Format;

/// A scalar type: `u8` to `u64`, `i8` to `i64`, `f16`, `f32`, `f64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// `u8`.
    U8,
    /// `u16`.
    U16,
    /// `u32`.
    U32,
    /// `u64`.
    U64,
    /// `i8`.
    I8,
    /// `i16`.
    I16,
    /// `i32`.
    I32,
    /// `i64`.
    I64,
    /// `f16`.
    F16,
    /// `f32`.
    F32,
    /// `f64`.
    F64,
}

impl ScalarType {
    /// Every scalar type, in the order of the enum.
    pub const ALL: [ScalarType; 11] = [
        ScalarType::U8,
        ScalarType::U16,
        ScalarType::U32,
        ScalarType::U64,
        ScalarType::I8,
        ScalarType::I16,
        ScalarType::I32,
        ScalarType::I64,
        ScalarType::F16,
        ScalarType::F32,
        ScalarType::F64,
    ];

    /// The type's name in the text.
    pub fn name(self) -> &'static str {
        self.format().name()
    }

    /// The scalar type of a name, if the name is one.
    pub fn from_name(name: &str) -> Option<ScalarType> {
        ScalarType::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The format of the same name.
    pub fn format(self) -> Format {
        match self {
            ScalarType::U8 => Format::U8,
            ScalarType::U16 => Format::U16,
            ScalarType::U32 => Format::U32,
            ScalarType::U64 => Format::U64,
            ScalarType::I8 => Format::I8,
            ScalarType::I16 => Format::I16,
            ScalarType::I32 => Format::I32,
            ScalarType::I64 => Format::I64,
            ScalarType::F16 => Format::F16,
            ScalarType::F32 => Format::F32,
            ScalarType::F64 => Format::F64,
        }
    }

    /// The width in bits: of an integer, or of a float (16, 32, 64).
    pub fn bits(self) -> u32 {
        match self {
            ScalarType::U8 | ScalarType::I8 => 8,
            ScalarType::U16 | ScalarType::I16 | ScalarType::F16 => 16,
            ScalarType::U32 | ScalarType::I32 | ScalarType::F32 => 32,
            ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => 64,
        }
    }

    /// True for a signed integer type.
    pub fn is_signed(self) -> bool {
        self.format().is_signed()
    }

    /// True for `f16`, `f32` and `f64`.
    pub fn is_float(self) -> bool {
        matches!(self, ScalarType::F16 | ScalarType::F32 | ScalarType::F64)
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for s in ScalarType::ALL {
            assert_eq!(ScalarType::from_name(s.name()), Some(s));
            assert_eq!(s.format().as_scalar(), Some(s));
        }
        assert_eq!(ScalarType::from_name("bf16"), None);
    }

    #[test]
    fn widths_and_kinds() {
        assert_eq!(ScalarType::F16.bits(), 16);
        assert_eq!(ScalarType::I64.bits(), 64);
        assert!(ScalarType::I32.is_signed() && !ScalarType::U32.is_signed());
        assert!(ScalarType::F64.is_float() && !ScalarType::U8.is_float());
    }
}
