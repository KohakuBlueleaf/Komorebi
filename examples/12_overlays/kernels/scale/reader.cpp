// Overlay scale, its reader part: fills the core's input tiles with small numbers (1
// to 7, picked by core, tile and position) and pushes them to compute.
#include "komorebi/device/km_worker.hpp"
#include "../overlays.h"

namespace {

    // bfloat16 bits of 1 to 7
    const uint16_t kSmall[kSmallValues] = {
        0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0
    };

}  // namespace

uint32_t km_overlay(km::Reader& reader, const ScaleArgument& argument) {
    for (uint32_t tile = 0; tile < argument.tiles; tile++) {
        if (!reader.reserve(kCbIn, 1)) {
            return 0;
        }
        volatile tt_l1_ptr uint16_t*
            values = (
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(kCbIn))
            );
        for (uint32_t index = 0; index < kTileValues; index++) {
            values[index] = kSmall[(argument.core + tile + index) % kSmallValues];
        }
        cb_push_back(kCbIn, 1);
    }
    return argument.tiles;
}
