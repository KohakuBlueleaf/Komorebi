// The gate's movement as a program of its own: both movers call the class's hand
// function gate_k::gate_movement on each command.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/movement/km_movement.hpp"
#include "hand.h"

constexpr km::BufferPair kPairAIn = {kPairAInMoverA, kPairAInMoverB};
constexpr km::BufferPair kPairBIn = {kPairBInMoverA, kPairBInMoverB};
constexpr km::BufferPair kPairResult = {kPairResultMoverA, kPairResultMoverB};

#include "../gate_movement.hpp"

void kernel_main() {
    using A = km::Tensor<0>;
    using B = km::Tensor<A::next_compile_time_args_offset()>;
    using Out = km::Tensor<B::next_compile_time_args_offset()>;
    km::Movement movement;
    movement.run([&](const km_command_t& command) {
        const A a(command.a0);
        const B b(command.a1);
        const Out out(command.a2);
        return (
            gate_k::gate_movement(
                movement, command, a, b, out, command.a3, km::Pattern::of(command.a4)
            )
        );
    });
}
