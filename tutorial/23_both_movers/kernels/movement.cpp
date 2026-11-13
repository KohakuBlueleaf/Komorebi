// The data-movement kernel of both CPUs: each copies the blocks the pattern gives it,
// reading them into its own half of the pair and writing them on from there.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_movement.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto arguments = TensorAccessorArgs<0>();  // source and destination alike
    constexpr km::BufferPair halves = {kHalfA, kHalfB};
    km::Movement movement;
    // --8<-- [start:run]
    movement.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerCopy) {
            const uint32_t source_address = command.a0;
            const uint32_t first_tile = command.a1;
            const uint32_t destination_address = command.a2;
            const uint32_t first_destination_tile = command.a3;
            const uint32_t tile_count = command.a4;
            const km::Pattern pattern = km::Pattern::of(command.a5);
            const km::Stream read(
                km::kMoveRead,
                TensorAccessor(arguments, source_address),
                first_tile,
                tile_count,
                kBlockTiles,
                1,
                halves,
                pattern
            );
            const km::Stream write(
                km::kMoveWrite,
                TensorAccessor(arguments, destination_address),
                first_destination_tile,
                tile_count,
                kBlockTiles,
                1,
                halves,
                pattern
            );
            return movement.move(read, write);
        }
        return true;
    });
    // --8<-- [end:run]
}
