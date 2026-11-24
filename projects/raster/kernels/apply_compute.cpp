// APPLY's own program's compute kernel (apply.py): the hardware started on x's and
// y's formats once, then the products of every command.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "komorebi/device/core/km_core.hpp"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/movement/km_movement.hpp"

#include "apply_compute.hpp"

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbApplyX, kCbApplyY);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerApply) {
            apply_compute_command(compute, command);
        }
    });
}
