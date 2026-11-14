// Overlay dense, its writer part: writes the tiles compute made to the range's pages.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "../../split.h"

uint32_t km_overlay(km::Writer& writer, const ts_drain& drain) {
    const InterleavedAddrGen<true> tiles = {
        .bank_base_address = drain.tiles, .page_size = TS_TILE_BYTES
    };
    for (uint32_t unit = drain.first; unit < drain.first + drain.count; unit++) {
        if (!writer.wait_front(TS_CB_OUT, 1)) {
            return unit - drain.first;
        }
        noc_async_write(
            get_read_ptr(TS_CB_OUT),
            get_noc_addr(drain.page + unit, tiles),
            TS_TILE_BYTES
        );
        noc_async_write_barrier();
        cb_pop_front(TS_CB_OUT, 1);
    }
    return drain.count;
}
