// Overlay scale, its writer part: writes the tiles compute made to the core's pages in
// DRAM.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "../overlays.h"

uint32_t km_overlay(km::Writer& writer, const DrainArgument& argument) {
    const InterleavedAddrGen<true> tiles = {
        .bank_base_address = argument.address, .page_size = kTileValues * 2
    };
    for (uint32_t tile = 0; tile < argument.tiles; tile++) {
        if (!writer.wait_front(kCbOut, 1)) {
            return tile;
        }
        const uint32_t page = argument.core * argument.tiles + tile;
        noc_async_write(
            get_read_ptr(kCbOut), get_noc_addr(page, tiles), kTileValues * 2
        );
        noc_async_write_barrier();
        cb_pop_front(kCbOut, 1);
    }
    return argument.tiles;
}
