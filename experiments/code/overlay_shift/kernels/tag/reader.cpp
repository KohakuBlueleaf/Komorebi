// Both programs' reader: TAG calls overlay a0 of the program's own numbers (plus its
// KM_OVERLAY_SHIFT) and writes the result, a0, the shift and held to its line.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_overlay.hpp"
#include "komorebi/device/km_program.hpp"
#include "../shift.h"

void km_program(km::Reader& reader, const km_command_t& command) {
    if (km::handler(command) != kHandlerTag) {
        return;
    }
    km::Overlay overlay(reader, command.a0);
    const uint32_t held = overlay.held();
    if (!overlay.load()) {
        return;
    }
    const uint32_t result = overlay.call(TagArgument{command.a1 ^ reader.me()});
    volatile tt_l1_ptr uint32_t* line = km::l1(get_write_ptr(kCbLine));
    line[0] = command.sequence;
    line[1] = result;
    line[2] = command.a0;
    line[3] = KM_OVERLAY_SHIFT;
    line[4] = held;
    const km::Tensor<0> lines(command.a2);
    const uint32_t page = command.a3 + reader.me() - km::first(command);
    if (!km::write_pages(reader, command, lines, page, 1, get_write_ptr(kCbLine))) {
        return;
    }
    noc_async_write_barrier();
}
