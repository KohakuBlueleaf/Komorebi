// The frame program's compute kernel: the passes' and the fill's file, each in a
// namespace without its entry, chosen by segment; uint32 buffers both, one start-up.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/km_worker.hpp"
#ifdef TRISC_PACK
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_dest.h"
#include "ckernel_ops.h"
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif
#endif

#define RASTER_NO_ENTRY

namespace fill_k {
#include "compute.cpp"
}  // namespace fill_k

namespace pass_k {
#include "setup_compute.cpp"
}  // namespace pass_k

void kernel_main() {
    km::Compute cp;
    compute_kernel_hw_startup(fill_k::kCbIn, fill_k::kCbOut);
    cp.run([&](const km_record_t& r) {
        if (km::seg(r) >= pass_k::kSegLanes) {
            pass_k::setup_compute_record(cp, r);
        } else {
            fill_k::fill_compute_record(r);
        }
    });
}
