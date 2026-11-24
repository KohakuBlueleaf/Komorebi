// The frame program's writer: the passes' (handlers from 11), the fill's, SHOW, SHADE,
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
#include "writer.cpp"
#include "show_writer.hpp"
#include "shade_writer.hpp"
#include "light_writer.hpp"
}  // namespace fill_k

namespace pass_k {
#include "setup_writer.cpp"
}  // namespace pass_k

#ifdef APPLY_WINDOWS
namespace apply_k {
    constexpr uint32_t kApplyArgsAt = fill_k::Profiles::next_compile_time_args_offset();
#include "apply_writer.hpp"
}  // namespace apply_k
#endif

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == fill_k::kHandlerShow) {
            return fill_k::show_writer_command(writer, command);
        }
        if (km::handler(command) == fill_k::kHandlerShade) {
            return fill_k::shade_writer_command(writer, command);
        }
        if (km::handler(command) == fill_k::kHandlerLight) {
            return fill_k::light_writer_command(writer, command);
        }
#ifdef APPLY_WINDOWS
        if (km::handler(command) == apply_k::kHandlerApply) {
            return apply_k::apply_writer_command(writer, command);
        }
#endif
        if (km::handler(command) >= pass_k::kHandlerLanes) {
            return pass_k::setup_writer_command(writer, command);
        }
        return fill_k::fill_writer_command(writer, command);
    });
}
