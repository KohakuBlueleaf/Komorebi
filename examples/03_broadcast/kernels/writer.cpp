// Broadcast writer: the column broadcast, on NoC1 (W blocks down each column), then the
// core's checksum tile at output page a6 + rank.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "broadcast.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto out_args = (
        TensorAccessorArgs<w_args.next_compile_time_args_offset()>()
    );
    km::Writer wr;
    wr.run([&](const km_command_t& r) {
        const uint32_t handler = km::handler(r);
        if (handler != kHandlerBroadcast && handler != kHandlerRead) {
            return true;
        }
        km::Broadcast ln(wr, r, kArgBroadcast);
        if (!ln.ok()) {
            return false;
        }
        if (!ln.included()) {
            return true;
        }
        const auto w = TensorAccessor(w_args, r.a1);
        const uint32_t first = wr.arg(kArgIndex) * r.a7 * ln.pages();
        const bool got = handler == kHandlerRead
            ? read_blocks(wr, km::kStWriter, r, ln, kCbCol, w, first, r.a3, r.a7)
            : broadcast_blocks(ln, w, first, r.a3, r.a7, [](uint32_t) {
                  return false;
              });
        if (!got || !wr.wait_front(kCbOut, 1)) {
            return false;
        }
        const auto out = TensorAccessor(out_args, r.a2);
        noc_async_write_page(r.a6 + wr.me() - km::first(r), out, get_read_ptr(kCbOut));
        const bool sent = wr.poll(
                              [] {
                                  return ncrisc_noc_nonposted_writes_sent(noc_index);
                              },
                              wr.args().wait_limit,
                              wr.args().wait_limit
                          ) == km::kGaveNone;
        cb_pop_front(kCbOut, 1);
        return sent;
    });
}
