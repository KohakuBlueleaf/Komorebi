// The gate's data movement (../../22_gate/kernels/movement.cpp) as the hand kernel of
// the class's handler `gate`: a core's share of a and b into L1 and of out back.
#pragma once

namespace gate_k {

    // GATE on each mover: the command's tiles a core from rank x tiles, each mover
    // taking its pattern's blocks through the declared pairs of the class's header.
    template <typename A, typename B, typename Out>
    inline bool gate_movement(
        km::Movement& movement,
        const km_command_t& command,
        const A& a,
        const B& b,
        const Out& out,
        uint32_t tiles,
        km::Pattern pattern
    ) {
        const uint32_t first = (movement.me() - km::first(command)) * tiles;
        const auto stream = [&](
            uint32_t direction, const auto& tensor, km::BufferPair pair
        ) {
            return (
                km::Stream(
                    direction, tensor, first, tiles, kConstBlock, 1, pair, pattern
                )
            );
        };
        return (
            movement.move(
                stream(km::kMoveRead, a, kPairAIn),
                stream(km::kMoveRead, b, kPairBIn),
                stream(km::kMoveWrite, out, kPairResult)
            )
        );
    }

}  // namespace gate_k
