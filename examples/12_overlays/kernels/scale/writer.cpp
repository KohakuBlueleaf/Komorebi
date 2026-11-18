// Overlay scale, its writer part: writes the tiles compute made to the core's pages of
// the tiles tensor, wherever they live (km::write_pages_at).
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "../overlays.h"

uint32_t km_overlay(km::Writer& writer, const DrainArgument& argument) {
    const km::Tensor<0> tiles(argument.tensor);
    for (uint32_t tile = 0; tile < argument.tiles; tile++) {
        if (!writer.wait_front(kCbOut, 1)) {
            return tile;
        }
        const uint32_t page = argument.core * argument.tiles + tile;
        if (!km::write_pages_at(
            writer,
            argument.slot,
            argument.sequence,
            tiles,
            page,
            1,
            get_read_ptr(kCbOut)
        )) {
            return tile;
        }
        noc_async_write_barrier();
        cb_pop_front(kCbOut, 1);
    }
    return argument.tiles;
}
