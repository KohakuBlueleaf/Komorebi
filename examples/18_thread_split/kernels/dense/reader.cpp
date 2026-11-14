// Overlay dense, its reader part: each unit's input tile (TS_SMALL as bfloat16) to
// compute, which doubles it; fast on units of kind 2, `penalty` cycles more on others.
#include "komorebi/device/km_worker.hpp"
#include "../../split.h"

namespace {

    // bfloat16 bits of 1 to 7
    const uint16_t kSmall[7] = {0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0};

}  // namespace

uint32_t km_overlay(km::Reader& reader, const ts_range& range) {
    for (uint32_t unit = range.first; unit < range.first + range.count; unit++) {
        if (TS_KIND(unit, range.run) != TS_OVERLAY_DENSE - 1) {
            const uint32_t start = km::wall();
            while (km::wall() - start < range.penalty) {
                ;
            }
        }
        if (!reader.reserve(TS_CB_IN, 1)) {
            return 0;
        }
        volatile tt_l1_ptr uint16_t*
            values = (
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(TS_CB_IN))
            );
        for (uint32_t index = 0; index < TS_TILE_VALUES; index++) {
            values[index] = kSmall[TS_SMALL(unit, index) - 1];
        }
        cb_push_back(TS_CB_IN, 1);
    }
    return range.count;
}
