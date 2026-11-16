// The example's data-movement kernel, built for both movers: each moves the blocks the
// command's pattern gives it, of the read stream and (COPY) of the write stream.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/movement/km_movement.hpp"
#include "movement.h"

constexpr km::BufferPair kInput = {kInputMoverA, kInputMoverB};
constexpr km::BufferPair kOutput = {kOutputMoverA, kOutputMoverB};

void kernel_main() {
    using Input = km::Tensor<0>;
    using Output = km::Tensor<Input::next_compile_time_args_offset()>;
    km::Movement movement;
    movement.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler != kHandlerRead && handler != kHandlerCopy) {
            return true;
        }
        const uint32_t rank = movement.me() - km::first(command);
        const uint32_t tiles = command.a2;
        const uint32_t tiles_per_block = command.a3;
        const km::Pattern pattern = km::Pattern::of(command.a4);
        const bool copy = handler == kHandlerCopy;
        const km::Stream read(
            km::kMoveRead,
            Input(command.a0),
            rank * tiles,
            tiles,
            tiles_per_block,
            copy ? 1 : command.a5,
            kInput,
            pattern
        );
        if (!copy) {
            return movement.move(read);
        }
        const km::Stream write(
            km::kMoveWrite,
            Output(command.a1),
            rank * tiles,
            tiles,
            tiles_per_block,
            1,
            kOutput,
            pattern
        );
        return movement.move(read, write);
    });
}
