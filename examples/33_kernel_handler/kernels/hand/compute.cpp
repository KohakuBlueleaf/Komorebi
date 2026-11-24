// The gate's compute as a program of its own: the class's start-up, then the class's
// hand function gate_k::gate_compute on each command.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/movement/km_movement.hpp"
#include "komorebi/device/km_worker.hpp"
#include "hand.h"

constexpr km::BufferPair kPairAIn = {kPairAInMoverA, kPairAInMoverB};
constexpr km::BufferPair kPairBIn = {kPairBInMoverA, kPairBInMoverB};
constexpr km::BufferPair kPairResult = {kPairResultMoverA, kPairResultMoverB};

#include "../gate_compute.hpp"

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kPairAIn.mover_a, kPairResult.mover_a);
    compute.run([&](const km_command_t& command) {
        gate_k::gate_compute(compute, command, command.a3, km::Pattern::of(command.a4));
    });
}
