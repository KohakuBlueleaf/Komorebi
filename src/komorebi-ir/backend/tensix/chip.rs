//! What the Tensix backend states, from what Komorebi's code knows of each chip and of
//! its device framework. Each number names where it comes from.

use crate::types::{Format, TileShape};

use super::super::{Limit, Provided, Refusal, Target, NOT_BUILT};

/// The target a compile is for: no chip, since the text is one for both chips
/// (`tile_<f>.hpp` holds both chips' branches) and the key holds none. The chip comes
/// at load (`load::chip`).
pub const ARCH: &str = "tensix";

/// Why a compile's target is refused, if it is: it must be `tensix` with no features;
/// a chip's name is refused by name.
pub fn compile_target(target: &Target) -> Option<Refusal> {
    let what = if Chip::of(&target.arch).is_some() {
        format!(
            "the Tensix backend compiles for the target `{ARCH}`, not a chip (`{}`): the text is one for both chips; the chip is checked at load",
            target.arch
        )
    } else if target.arch != ARCH {
        format!(
            "the Tensix backend compiles for the target `{ARCH}`, not `{}`",
            target.arch
        )
    } else if !target.features.is_empty() {
        format!("the Tensix backend's target `{ARCH}` takes no features")
    } else {
        return None;
    };
    Some(Refusal {
        code: NOT_BUILT,
        what,
        at: None,
    })
}

/// A chip the backend builds for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chip {
    /// `blackhole` (P100, P150, P300: `device/chips.py`, `CARD_TYPES`).
    Blackhole,
    /// `wormhole_b0` (N150, N300, a Galaxy's chips).
    WormholeB0,
}

impl Chip {
    /// The chip a target's architecture names, if one is built.
    pub fn of(arch: &str) -> Option<Chip> {
        match arch {
            "blackhole" => Some(Chip::Blackhole),
            "wormhole_b0" => Some(Chip::WormholeB0),
            _ => None,
        }
    }

    /// The bytes of a core's L1 (`device/chips.py`, `l1_bytes`).
    pub fn l1_bytes(self) -> u64 {
        match self {
            Chip::Blackhole => 1536 * 1024,
            Chip::WormholeB0 => 1464 * 1024,
        }
    }
}

/// The abilities the framework provides on both chips: a grid group, dynamic runs,
/// islands and state between invocations (`docs/reference/classes.md`, `dynamic.md`,
/// `islands.md`). What the emitter does not print yet is refused by name.
pub const ABILITIES: &[Provided] = &[
    Provided::Grid,
    Provided::Dynamic,
    Provided::Islands,
    Provided::LocalState,
];

/// The limits on a program's text.
pub const LIMITS: &[(Limit, u64)] = &[
    // abi/km_abi.h, KM_HANDLER_USER_MAX: "handlers 1..239 belong to the program".
    (Limit::HandlersPerProgram, 239),
    // abi/km_abi.h, km_command_t: ten argument words, a0 to a9.
    (Limit::ArgumentWords, 10),
    // docs/reference/dynamic.md: a message is at most 256 bytes, 64 words; word 0
    // the runtime's, and word 1 too in a program with bodies.
    (Limit::MessageFields, 62),
];

/// The formats a buffer holds (`formats.py`, `DTYPES`), on both chips.
pub const FORMATS: &[Format] = &[
    Format::Bf16,
    Format::F32,
    Format::F16,
    Format::Bfp8,
    Format::Bfp4,
    Format::Bfp2,
    Format::I32,
    Format::U32,
    Format::U16,
    Format::U8,
];

/// The one tile shape: 32 by 32 (`formats.py`, `TILE`).
pub const TILE: TileShape = TileShape { rows: 32, cols: 32 };

/// The argument words of a tensor parameter: its address (`run.submit`); its tile
/// count is not sent, a host checks its bounds (`EntryPoint.bounds`).
pub const TENSOR_WORDS: u64 = 1;

/// The commands of one ring at once (`abi/km_abi.h`, `KM_RING_SLOTS`).
pub const RING_SLOTS: u64 = 64;

/// A load's depth when it names none (`dsl/program.py`, `make`'s `depth`).
pub const DEPTH: u64 = 4;

/// A dynamic run's live messages when it names none (`docs/reference/dynamic.md`,
/// `max_live`).
pub const MAX_LIVE: u64 = 1 << 16;

/// The forward-progress guarantees.
pub const PROGRESS: &[&str] = &[
    "a handler runs to its end on its core: one persistent kernel a program, a switch a command",
    "the commands of one ring run in order",
    "a command has a deadline and gives up after it (KM_GUARD)",
];

/// The bytes of a tile of `format` and `shape` in L1 (`formats.py`, `tile_bytes`): a
/// block-float tile is its 16-element blocks' bytes plus one exponent byte for each,
/// rounded as the chip packs a 32 by 32 tile (1088, 576, 320 bytes).
pub fn tile_bytes(format: Format, shape: TileShape) -> u64 {
    let elements = shape.rows * shape.cols;
    let packed = |bits: u64| elements * bits / 8 + elements / 16;
    match format {
        Format::Bfp8 => packed(8),
        Format::Bfp4 => packed(4),
        Format::Bfp2 => packed(2),
        Format::Bf16 => elements * 2,
        other => elements * other.bytes().map_or(4, |b| b as u64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tile_bytes_are_todays() {
        let at = |f| tile_bytes(f, TILE);
        assert_eq!(
            [Format::Bf16, Format::F32, Format::U8, Format::U16].map(at),
            [2048, 4096, 1024, 2048]
        );
        assert_eq!(
            [Format::Bfp8, Format::Bfp4, Format::Bfp2].map(at),
            [1088, 576, 320]
        );
    }
}
