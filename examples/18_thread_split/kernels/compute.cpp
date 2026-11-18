// Compute: in a PASS, takes the reader's jobs; for each dense range it loads the dense
// overlay's compute parts (the meeting with the reader's load) and calls them.
#include <stdint.h>

#include "../split.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/overlays/km_overlay.hpp"

namespace {

    // The next job's overlay and count, the same on unpack, math and pack; 0: the end.
    uint32_t take(const km::Compute& compute, uint32_t& count) {
        if (!compute.wait_front(TS_CB_JOB_COMPUTE, 1, 64)) {
            return 0;
        }
        uint32_t overlay = 0;
        UNPACK(({
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* job = (
                km::l1(ckernel::get_tile_l1_byte_address(TS_CB_JOB_COMPUTE, 0))
            );
            overlay = job[TS_JOB_OVERLAY];
            count = job[TS_JOB_COUNT];
            cb_pop_front(TS_CB_JOB_COMPUTE, 1);
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, overlay);
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, count);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, overlay);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, count);
        }));
        MATH(({
            overlay = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
            count = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        }));
        PACK(({
            overlay = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
            count = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        }));
        return overlay;
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(TS_CB_IN, TS_CB_OUT);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != TS_HANDLER_PASS || command.a7 != 0) {
            return;
        }
        uint32_t count = 0;
        while (take(compute, count) == TS_OVERLAY_DENSE) {
            km::Overlay dense(compute, TS_OVERLAY_DENSE);
            if (!dense.load()) {
                return;
            }
            dense.call(ts_scale{count});
        }
    });
}
