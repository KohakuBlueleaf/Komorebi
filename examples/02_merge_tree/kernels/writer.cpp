// Merge-tree writer: rank 0 writes the total to output pages a3 on (its checksum the
// result word if asked); every other rank sends its part (a5 stalls instead).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_port.hpp"
#include "merge.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    km::Writer wr;
    wr.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerMerge) {
            return true;
        }
        const km::Group g = km::group_of(r, wr.me());
        const km::Arrival arrival = {kCbArrive, kArrivePages, r.a7};
        if (g.rank == r.a5) {  // a test: waits for the stop flag, sends nothing
            const uint32_t code = wr.poll(
                [] {
                    return false;
                },
                wr.args().idle_limit
            );
            wr.gave_up(km::kStWriter, code, km::kWhereStall, km::slot(r), r.seq);
            return false;
        }
        if (g.rank != 0) {
            return km::reduce_emit(wr, r, arrival, g, r.a4, kCbOut, r.a7);
        }
        if (!km::reduce_result(wr, kCbOut, r.a7)) {
            return false;
        }
        // The command's result word, only when it asks for one: the total's 32-bit
        // words summed (a checksum: one more pass over the total, so only if asked).
        if (km::flags(r) & KM_F_RESULT) {
            const volatile tt_l1_ptr uint32_t*
                words = (
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(kCbOut))
                );
            uint32_t sum = 0;
            for (uint32_t i = 0; i < r.a7 * kTileBytes / 4; i++) {
                sum += words[i];
            }
            wr.result(sum);
        }
        const auto out = TensorAccessor(out_args, r.a2);
        for (uint32_t d = 0; d < r.a7; d++) {
            noc_async_write_page(r.a3 + d, out, get_read_ptr(kCbOut) + d * kTileBytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, r.a7);
        return true;
    });
}
