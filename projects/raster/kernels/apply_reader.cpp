// APPLY's own program's reader (apply.py): mover A of every command.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/movement/km_movement.hpp"
#include "komorebi/device/tensors/km_ask.hpp"

constexpr uint32_t kApplyArgsAt = 0;  // x, w, y, tables, profile: its only tensors

#include "apply_reader.hpp"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerApply) {
            apply_reader_command(reader, command);
        }
    });
}
