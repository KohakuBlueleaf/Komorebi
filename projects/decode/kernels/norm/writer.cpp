// Norm, writer: this core's tiles of the output (a2), as compute packs them.
#include "api/dataflow/dataflow_api.h"
#include "../ops.hpp"

uint32_t km_overlay(km::Writer& writer, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const km::Tensor<0> output(command.a2);
    const uint32_t first = (writer.me() - km::first(command)) * DC_NORM_SHARE;
    if (!writer.wait_front(DC_CB_NORMED, DC_NORM_SHARE)) {
        return 0;
    }
    if (!km::write_pages(
        writer, command, output, first, DC_NORM_SHARE, get_read_ptr(DC_CB_NORMED)
    ) ||
        !flushed(writer, km::kStWriter, km::kFlushSent)) {
        return 0;
    }
    cb_pop_front(DC_CB_NORMED, DC_NORM_SHARE);
    return 1;
}
