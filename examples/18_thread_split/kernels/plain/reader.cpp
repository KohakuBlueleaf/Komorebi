// Overlay plain, its reader part: each unit's result tile written value by value on
// this core; fast on units of kind 0, `penalty` cycles more on the others.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "../../split.h"

namespace {

    // bfloat16 bits of 2, 4, ..., 14: twice TS_SMALL
    const uint16_t kTwice[7] = {0x4000, 0x4080, 0x40C0, 0x4100, 0x4120, 0x4140, 0x4160};

}  // namespace

// --8<-- [start:entry]
uint32_t km_overlay(km::Reader& reader, const ts_range& range) {
    const InterleavedAddrGen<true> tiles = {
        .bank_base_address = range.tiles, .page_size = TS_TILE_BYTES
    };
    const uint32_t tile = get_write_ptr(TS_CB_TILE);
    volatile tt_l1_ptr uint16_t*
        values = (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile));
    for (uint32_t unit = range.first; unit < range.first + range.count; unit++) {
        if (TS_KIND(unit, range.run) != TS_OVERLAY_PLAIN - 1) {
            const uint32_t start = km::wall();
            while (km::wall() - start < range.penalty) {
                ;
            }
        }
        for (uint32_t index = 0; index < TS_TILE_VALUES; index++) {
            values[index] = kTwice[TS_SMALL(unit, index) - 1];
        }
        noc_async_write(tile, get_noc_addr(range.page + unit, tiles), TS_TILE_BYTES);
        noc_async_write_barrier();
    }
    return range.count;
}

// --8<-- [end:entry]
