// Broadcast writer: the column lane, on NoC1 (W blocks down each column), then the
// member's checksum tile at output page a6 + member.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "broadcast.h"
#include "komorebi/device/km_lane.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto out_args = (
        TensorAccessorArgs<w_args.next_compile_time_args_offset()>()
    );
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        const uint32_t seg = km::seg(r);
        if (seg != kSegCast && seg != kSegRead) {
            return true;
        }
        km::Lane ln(wr, r, kArgLane);
        if (!ln.ok()) {
            return false;
        }
        if (!ln.member()) {
            return true;
        }
        const auto w = TensorAccessor(w_args, r.a1);
        const uint32_t first = wr.arg(kArgIndex) * r.a3 * ln.pages();
        const bool got = seg == kSegRead
            ? read_blocks(wr, km::kStWriter, r, ln, kCbCol, w, first, r.a3)
            : cast_blocks(ln, w, first, r.a3, [](uint32_t) {
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
