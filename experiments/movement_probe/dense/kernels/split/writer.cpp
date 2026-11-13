// K-split writer, NoC1: first the K chunks CHUNK_PATTERN gives mover B; then a core
// other than its group's root sends its part (fp32) on, the root writes the group's sum
// to Y.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../../../kernels/movement.hpp"
#include "komorebi/device/km_port.hpp"
#include "split.h"

namespace {

    // Timed until `buffer` has room for `pages`. False: gave up (recorded).
    bool room(const km::Writer& writer, uint32_t buffer, uint32_t pages) {
        const uint32_t code = writer.poll(
            [&] {
                return cb_pages_reservable_at_back(buffer, pages);
            },
            writer.args().wait_limit
        );
        if (code != km::kGaveNone) {
            writer.gave_up(km::kStWriter, code, km::kWhereData, 0, 0);
            return false;
        }
        cb_reserve_back(buffer, pages);
        return true;
    }

    // Mover B's K chunks of this core: A and W into its halves, one barrier a chunk.
    template <typename AccessorA, typename AccessorW>
    bool read_chunks(
        const km::Writer& writer,
        const km_command_t& command,
        const AccessorA& a,
        const AccessorW& w
    ) {
        const auto plan = [&](uint32_t index) {
            return writer.arg(index);
        };
        const uint32_t mt = plan(kSpMt), kt = plan(kSpKt);
        const uint32_t nt = plan(kSpNt), oh = plan(kSpOh);
        const uint32_t ow = plan(kSpOw), kc = plan(kSpKc);
        const uint32_t blocks_per_row = plan(kSpNbw), s = plan(kSpS);
        const km::Group group = km::digit_group(command, writer.me(), 1, s);
        const uint32_t block = (writer.me() - km::first(command)) / s;
        const uint32_t m0 = block / blocks_per_row * oh;
        const uint32_t n0 = block % blocks_per_row * ow;
        const km::Partition part = km::partition(kt, group.rank, group.size);
        const movement::Pattern pattern = movement::pattern_of(CHUNK_PATTERN);
        for (uint32_t k0 = 0, chunk = 0; k0 < part.len; k0 += kc, chunk++) {
            if (movement::owner(pattern, chunk) != movement::kMoverB) {
                continue;
            }
            const uint32_t n = part.len - k0 < kc ? part.len - k0 : kc;
            const uint32_t k = part.start + k0;
            if (
                !room(writer, kChunkAMoverB, oh * kc) ||
                !room(writer, kChunkWMoverB, kc * ow)
            ) {
                return false;
            }
            const uint32_t to_a = get_write_ptr(kChunkAMoverB);
            const uint32_t to_w = get_write_ptr(kChunkWMoverB);
            for (uint32_t i = 0; i < oh && m0 + i < mt; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read_page(
                        (m0 + i) * kt + k + j, a, to_a + (i * kc + j) * kTileBytes
                    );
                }
            }
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t c = 0; c < ow && n0 + c < nt; ++c) {
                    noc_async_read_page(
                        (k + j) * nt + n0 + c, w, to_w + (j * ow + c) * kTileBytes
                    );
                }
            }
            if (!movement::settle(writer, km::kStWriter, true, false)) {
                return false;
            }
            cb_push_back(kChunkAMoverB, oh * kc);
            cb_push_back(kChunkWMoverB, kc * ow);
        }
        return true;
    }

}  // namespace

void kernel_main() {
    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto a_args = (
        TensorAccessorArgs<y_args.next_compile_time_args_offset()>()
    );
    constexpr auto w_args = (
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>()
    );
    km::Writer wr;
    const auto plan = [&](uint32_t i) {
        return wr.arg(i);
    };
    wr.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerSplit) {
            return true;
        }
        const auto a = TensorAccessor(a_args, r.a0);
        const auto w = TensorAccessor(w_args, r.a1);
        if (!read_chunks(wr, r, a, w)) {
            return false;
        }
        if (r.a4 != kSplitFull) {
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
