// The example's data-movement kernel, built for both movers: a core's share of a and b
// from DRAM into L1 and of out back to DRAM, each mover taking the blocks of the
// pattern.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gate.h"
#include "komorebi/device/km_movement.hpp"

constexpr km::BufferPair kA = {kAMoverA, kAMoverB};
constexpr km::BufferPair kB = {kBMoverA, kBMoverB};
constexpr km::BufferPair kOut = {kOutMoverA, kOutMoverB};

void kernel_main() {
    // a, b and out share one layout
    constexpr auto tensors = TensorAccessorArgs<0>();
    // stays loaded; runs once per command
    km::Movement movement;
    movement.run([&](const km_command_t& command) {
        // tiles a core, and where this core's share starts
        const uint32_t tiles = command.a3;
        const uint32_t first = (movement.me() - km::first(command)) * tiles;
        // which mover moves which block
        const km::Pattern pattern = km::Pattern::of(command.a4);
        const auto stream = [&](
            uint32_t direction, uint32_t address, km::BufferPair pair
        ) {
            return (
                km::Stream(
                    direction,
                    TensorAccessor(tensors, address),
                    first,
                    tiles,
                    kBlock,
                    1,
                    pair,
                    pattern
                )
            );
        };
        return (
            movement.move(
                stream(km::kMoveRead, command.a0, kA),
                stream(km::kMoveRead, command.a1, kB),
                stream(km::kMoveWrite, command.a2, kOut)
            )
        );
    });
}
