// Work-cluster example, grid side: STEP changes each core's state word (page p is
// core p's); SUM's leader reads the loop's words and gives their sum as its result.
#include <stdint.h>

#include "../decide.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

constexpr uint32_t kCbScratch = 16;

void kernel_main() {
    constexpr auto state_args = TensorAccessorArgs<0>();
    km::Writer wr;
    const uint32_t scratch = get_write_ptr(kCbScratch);
    volatile uint32_t* word = reinterpret_cast<volatile uint32_t*>(scratch);
    // NoC barriers polled with the wait limit; false (no report) if it passed.
    auto reads_done = [&] {
        const bool ok = wr.poll(
            [&] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            wr.args().wait_limit
        ) == km::kGaveNone;
        invalidate_l1_cache();
        return ok;
    };
    auto writes_done = [&] {
        return wr.poll(
            [&] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            wr.args().wait_limit
        ) == km::kGaveNone;
    };
    wr.run([&](const km_command_t& r) {
        const uint32_t handler = km::handler(r);
        const auto x = TensorAccessor(state_args, r.a0);
        if (handler == WC_HANDLER_SUM) {
            if (wr.me() != km::first(r)) {
                return true;
            }
            uint32_t sum = 0;
            for (uint32_t p = km::first(r); p < km::first(r) + r.a1; p++) {
                noc_async_read_page(p, x, scratch);
                if (!reads_done()) {
                    return false;
                }
                sum += word[0];
            }
            wr.result(sum);
            return true;
        }
        if (handler != WC_HANDLER_STEP) {
            return true;
        }
        const uint32_t page = wr.me();
        noc_async_read_page(page, x, scratch);
        if (!reads_done()) {
            return false;
        }
        word[0] = word[0] + r.a1 + page;
        noc_async_write_page(page, x, scratch);
        return writes_done();
    });
}
