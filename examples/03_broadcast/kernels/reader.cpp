// Broadcast reader: the row broadcast, on NoC0. Each row's sender reads the row's A
// blocks from DRAM and multicasts them along the row (READ: every rank reads its own).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "broadcast.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    km::Reader rd;
    rd.run([&](const km_command_t& r) {
        const uint32_t handler = km::handler(r);
        if (handler != kHandlerBroadcast && handler != kHandlerRead) {
            return;
        }
        km::Broadcast ln(rd, r, kArgBroadcast);
        if (!ln.ok() || !ln.included()) {
            return;
        }
        const auto a = TensorAccessor(a_args, r.a0);
        const uint32_t first = rd.arg(kArgIndex) * r.a7 * ln.pages();
        if (handler == kHandlerRead) {
            read_blocks(rd, km::kStReader, r, ln, kCbRow, a, first, r.a3, r.a7);
            return;
        }
        const bool stalls = rd.me() - km::first(r) == r.a4;
        broadcast_blocks(ln, a, first, r.a3, r.a7, [&](uint32_t k) {
            if (!stalls || k != r.a5) {
                return false;
            }
            rd.wait_stop(r);  // a test: this receiver stops taking blocks
            return true;
        });
    });
}
