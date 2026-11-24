// The frame program's compute kernel: the passes', the fill's, SHADE's (and APPLY's)
// file, each in a namespace without its entry, chosen by handler; one start-up.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#ifdef RASTER_SPREAD
#include "api/compute/tile_move_copy.h"
#endif
#include "komorebi/device/km_worker.hpp"
#ifdef APPLY_WINDOWS
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "komorebi/device/core/km_core.hpp"
#include "komorebi/device/movement/km_movement.hpp"
#endif
#ifdef TRISC_PACK
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_dest.h"
#include "ckernel_ops.h"
#include "komorebi/device/km_dst.hpp"
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif
#endif

#include "grid.h"  // before the namespaces: one copy for all of them

#define RASTER_NO_ENTRY

namespace fill_k {
#include "compute.cpp"
}  // namespace fill_k

namespace pass_k {
#include "setup_compute.cpp"
}  // namespace pass_k

namespace shade_k {
#include "shade_compute.hpp"
}  // namespace shade_k

#ifdef APPLY_WINDOWS
namespace apply_k {
#include "apply_compute.hpp"
}  // namespace apply_k
#endif

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(fill_k::kCbIn, fill_k::kCbOut);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) == shade_k::kHandlerShade) {
            shade_k::shade_compute_command(command);
        } else if (km::handler(command) == shade_k::kHandlerLight) {
            shade_k::light_compute_command(command);
#ifdef APPLY_WINDOWS
        } else if (km::handler(command) == apply_k::kHandlerApply) {
            // the units as the last handler left them: APPLY's formats again
            compute_kernel_hw_startup(apply_k::kCbApplyX, apply_k::kCbApplyY);
            apply_k::apply_compute_command(compute, command);
#endif
        } else if (km::handler(command) >= pass_k::kHandlerLanes) {
            pass_k::setup_compute_command(compute, command);
        } else {
            fill_k::fill_compute_command(command);
        }
    });
}
