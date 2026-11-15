// SCALE's writer part: writes the tiles compute made to the core's pages of the
// tensor the command names.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "../add.h"

uint32_t km_overlay(km::Writer& writer, const km_command_t& command) {
    const km::Tensor<0> tiles(command.a6);
    for (uint32_t tile = 0; tile < command.a3; tile++) {
        if (!writer.wait_front(kCbOut, 1)) {
            return tile;
        }
        const uint32_t page = writer.me() * command.a3 + tile;
        if (!km::write_pages_at(
            writer,
            km::slot(command),
            command.sequence,
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
    return command.a3;
}
