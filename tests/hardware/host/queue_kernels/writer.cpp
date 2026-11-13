// Queue order case, grid side, the tensor at a0: PUT's leader writes a2 into page a1;
// SUM's counts the pages below a1 that hold a2; BUSY's reads page 0 a1 times.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

constexpr uint32_t kSegPut = 1;
constexpr uint32_t kSegSum = 2;
constexpr uint32_t kSegBusy = 3;
constexpr uint32_t kCbScratch = 16;

void kernel_main() {
    constexpr auto pages_args = TensorAccessorArgs<0>();
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
    wr.run([&](const km_record_t& r) {
        const uint32_t seg = km::seg(r);
        if (seg < kSegPut || seg > kSegBusy || wr.me() != km::first(r)) {
            return true;
        }
        const auto x = TensorAccessor(pages_args, r.a0);
        if (seg == kSegPut) {
            word[0] = r.a2;
            noc_async_write_page(r.a1, x, scratch);
            return writes_done();
        }
        uint32_t same = 0;
        for (uint32_t p = 0; p < r.a1; p++) {
            noc_async_read_page(seg == kSegSum ? p : 0, x, scratch);
            if (!reads_done()) {
                return false;
            }
            same += word[0] == r.a2;
        }
        wr.result(same);
        return true;
    });
}
