// The frame program's reader: the passes' (handlers from 11), the fill's, SHOW, SHADE,
// LIGHT (in the fill's namespace), APPLY with APPLY_WINDOWS, by handler.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/tensors/km_ask.hpp"
#ifdef APPLY_WINDOWS
#include "komorebi/device/movement/km_movement.hpp"
#endif

#include "grid.h"  // before the namespaces: one copy for all of them

#define RASTER_NO_ENTRY

namespace fill_k {
#include "reader.cpp"
#include "show_reader.hpp"
#include "shade_reader.hpp"
#include "light_reader.hpp"
}  // namespace fill_k

namespace pass_k {
#include "setup_reader.cpp"
}  // namespace pass_k

#ifdef APPLY_WINDOWS
namespace apply_k {
    constexpr uint32_t kApplyArgsAt = fill_k::Lists::next_compile_time_args_offset();
#include "apply_reader.hpp"
}  // namespace apply_k
#endif

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == fill_k::kHandlerShow) {
            fill_k::show_reader_command(reader, command);
        } else if (km::handler(command) == fill_k::kHandlerShade) {
            fill_k::shade_reader_command(reader, command);
        } else if (km::handler(command) == fill_k::kHandlerLight) {
            fill_k::light_reader_command(reader, command);
#ifdef APPLY_WINDOWS
        } else if (km::handler(command) == apply_k::kHandlerApply) {
            apply_k::apply_reader_command(reader, command);
#endif
        } else if (km::handler(command) >= pass_k::kHandlerLanes) {
            pass_k::setup_reader_command(reader, command);
        } else {
            fill_k::fill_reader_command(reader, command);
        }
    });
}
