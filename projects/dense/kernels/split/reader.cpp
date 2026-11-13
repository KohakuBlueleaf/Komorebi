// K-split reader, NoC0: this core's 1/S of K for its group's output block, A and W in
// chunks of kc K tiles; then the parts this core sums (by owner or by tree).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "split.h"

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto w_args = (
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>()
    );
    km::Reader rd;
    const auto plan = [&](uint32_t i) {
        return rd.arg(i);
    };
    rd.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerSplit) {
            return;
        }
        const uint32_t mt = plan(kSpMt), kt = plan(kSpKt), nt = plan(kSpNt);
        const uint32_t oh = plan(kSpOh), ow = plan(kSpOw), kc = plan(kSpKc);
        const uint32_t nbw = plan(kSpNbw), s = plan(kSpS);
        const km::Group g = km::digit_group(r, rd.me(), 1, s);
        const uint32_t q = (rd.me() - km::first(r)) / s;
        const uint32_t m0 = q / nbw * oh, n0 = q % nbw * ow;
        const km::Partition part = km::partition(kt, g.rank, g.size);
        const auto a = TensorAccessor(a_args, r.a0);
        const auto w = TensorAccessor(w_args, r.a1);
        for (uint32_t k0 = 0; k0 < part.len; k0 += kc) {
            const uint32_t n = part.len - k0 < kc ? part.len - k0 : kc;
            const uint32_t k = part.start + k0;
            if (!rd.reserve(kCbSa, oh * kc) || !rd.reserve(kCbSw, kc * ow)) {
                return;
            }
            const uint32_t da = get_write_ptr(kCbSa), dw = get_write_ptr(kCbSw);
            for (uint32_t i = 0; i < oh && m0 + i < mt; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read_page(
                        (m0 + i) * kt + k + j, a, da + (i * kc + j) * kTileBytes
                    );
                }
            }
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t c = 0; c < ow && n0 + c < nt; ++c) {
                    noc_async_read_page(
                        (k + j) * nt + n0 + c, w, dw + (j * ow + c) * kTileBytes
                    );
                }
            }
            if (!flushed(rd, km::kStReader, km::kFlushReads)) {
                return;
            }
            cb_push_back(kCbSa, oh * kc);
            cb_push_back(kCbSw, kc * ow);
        }
        if (r.a4 == kSplitFull) {
            km::reduce_arrive(rd, r, {kCbArrive, plan(kSpArrive), oh * ow}, g, r.a3);
        }
    });
}
