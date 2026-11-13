// K-split writer, NoC1: a member other than its group's root sends its part (fp32) to
// its receiver; the root writes the group's sum (bf16) to Y, tiles outside Y dropped.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "split.h"

void kernel_main() {
    constexpr auto y_args = TensorAccessorArgs<0>();
    km::Writer wr;
    const auto plan = [&](uint32_t i) {
        return wr.arg(i);
    };
    wr.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerSplit || r.a4 != kSplitFull) {
            return true;
        }
        const uint32_t mt = plan(kSpMt), nt = plan(kSpNt);
        const uint32_t oh = plan(kSpOh), ow = plan(kSpOw), p = oh * ow;
        const uint32_t nbw = plan(kSpNbw), s = plan(kSpS);
        const km::Group g = km::digit_group(r, wr.me(), 1, s);
        const km::Arrival arrival = {kCbArrive, plan(kSpArrive), p};
        if (g.rank != 0) {
            return km::reduce_emit(wr, r, arrival, g, r.a3, kCbSum, p);
        }
        if (!km::reduce_result(wr, kCbSplitOut, p)) {
            return false;
        }
        const uint32_t q = (wr.me() - km::first(r)) / s;
        const uint32_t m0 = q / nbw * oh, n0 = q % nbw * ow;
        const auto y = TensorAccessor(y_args, r.a2);
        const uint32_t src = get_read_ptr(kCbSplitOut);
        for (uint32_t i = 0; i < oh && m0 + i < mt; ++i) {
            for (uint32_t j = 0; j < ow && n0 + j < nt; ++j) {
                noc_async_write_page(
                    (m0 + i) * nt + n0 + j, y, src + (i * ow + j) * kTileBytes
                );
            }
        }
        if (!flushed(wr, km::kStWriter, km::kFlushSent)) {
            return false;
        }
        cb_pop_front(kCbSplitOut, p);
        return true;
    });
}
