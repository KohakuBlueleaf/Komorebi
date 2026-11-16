// Overlay dense, its writer part: writes the tiles compute made to the range's pages.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "../../split.h"

uint32_t km_overlay(km::Writer& writer, const ts_drain& drain) {
    const km::Tensor<0> tiles(drain.tiles);
    for (uint32_t unit = drain.first; unit < drain.first + drain.count; unit++) {
        if (!writer.wait_front(TS_CB_OUT, 1)) {
            return unit - drain.first;
        }
        if (!km::write_pages_at(
            writer,
            drain.slot,
            drain.sequence,
            tiles,
            drain.page + unit,
            1,
            get_read_ptr(TS_CB_OUT)
        )) {
            return unit - drain.first;
        }
        noc_async_write_barrier();
        cb_pop_front(TS_CB_OUT, 1);
    }
    return drain.count;
}
