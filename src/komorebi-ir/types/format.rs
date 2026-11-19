//! Element formats: the closed set a module may declare and use.

use std::fmt;

/// An element format. The set is closed: `f16 f32 f64`, `bf16`, the integers, `bfp8`,
/// `bfp4` and `bfp2`; another name is not a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
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
    /// `bf16`.
    Bf16,
    /// `bfp8`: a sign and 7 magnitude bits an element, an exponent shared by 16.
    Bfp8,
    /// `bfp4`: the same with 3 magnitude bits.
    Bfp4,
    /// `bfp2`: the same with 1 magnitude bit.
    Bfp2,
}

impl Format {
    /// Every format, in the order of the enum.
    pub const ALL: [Format; 15] = [
        Format::U8,
        Format::U16,
        Format::U32,
        Format::U64,
        Format::I8,
        Format::I16,
        Format::I32,
        Format::I64,
        Format::F16,
        Format::F32,
        Format::F64,
        Format::Bf16,
        Format::Bfp8,
        Format::Bfp4,
        Format::Bfp2,
    ];

    /// The format's name in the text.
    pub fn name(self) -> &'static str {
        match self {
            Format::U8 => "u8",
            Format::U16 => "u16",
            Format::U32 => "u32",
            Format::U64 => "u64",
            Format::I8 => "i8",
            Format::I16 => "i16",
            Format::I32 => "i32",
            Format::I64 => "i64",
            Format::F16 => "f16",
            Format::F32 => "f32",
            Format::F64 => "f64",
            Format::Bf16 => "bf16",
            Format::Bfp8 => "bfp8",
            Format::Bfp4 => "bfp4",
            Format::Bfp2 => "bfp2",
        }
    }

    /// The format of a name, if the name is one.
    pub fn from_name(name: &str) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.name() == name)
    }

    /// True for an integer format.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Format::U8
                | Format::U16
                | Format::U32
                | Format::U64
                | Format::I8
                | Format::I16
                | Format::I32
                | Format::I64
        )
    }

    /// The bytes of one element, for a format whose bytes the language states: an
    /// integer type, `f32` or `f64`. The other formats are notations with no stated
    /// bytes, and `items`, `records` and `words` are not used on them.
    pub fn bytes(self) -> Option<u64> {
        match self {
            Format::U8 | Format::I8 => Some(1),
            Format::U16 | Format::I16 => Some(2),
            Format::U32 | Format::I32 | Format::F32 => Some(4),
            Format::U64 | Format::I64 | Format::F64 => Some(8),
            Format::F16 | Format::Bf16 | Format::Bfp8 | Format::Bfp4 | Format::Bfp2 => None,
        }
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for format in Format::ALL {
            assert_eq!(Format::from_name(format.name()), Some(format));
        }
        assert_eq!(Format::from_name("bf17"), None);
    }

    #[test]
    fn stated_bytes() {
        assert_eq!(Format::U32.bytes(), Some(4));
        assert_eq!(Format::F64.bytes(), Some(8));
        assert_eq!(Format::F16.bytes(), None);
        assert_eq!(Format::Bf16.bytes(), None);
        assert_eq!(Format::Bfp8.bytes(), None);
        assert!(Format::I8.is_integer());
        assert!(!Format::F32.is_integer());
    }
}
