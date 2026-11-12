// Map/collect reader: MAP reads the member's share of the T input tiles in blocks;
// STALL makes member a0 wait for the stop word.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "mapcollect.h"

void kernel_main() {
    constexpr auto in_args = TensorAccessorArgs<0>();
    km::Reader rd;
    uint32_t at = 0;  // write position in kCbIn, in pages: a block never wraps
    rd.run([&](const km_record_t& r) {
        const uint32_t index = rd.me() - km::first(r);
        switch (km::seg(r)) {
            case kSegMap: {
                const km::Share s = km::share(r.a1, index, km::count(r));
                const auto in = TensorAccessor(in_args, r.a0);
                for (uint32_t done = 0; done < s.len;) {
                    uint32_t n = s.len - done < kBlock ? s.len - done : kBlock;
                    n = n < kInPages - at ? n : kInPages - at;
                    if (!rd.reserve(kCbIn, n)) {
                        return;
                    }
                    const uint32_t dst = get_write_ptr(kCbIn);
                    for (uint32_t k = 0; k < n; k++) {
                        noc_async_read_page(
                            r.a6 + s.start + done + k, in, dst + k * kTileBytes
                        );
                    }
                    noc_async_read_barrier();
                    cb_push_back(kCbIn, n);
                    done += n;
                    at = (at + n) % kInPages;
                }
                break;
            }
            case kSegBadItem:  // from page `at`: past the end unless at == 0
                if (index == r.a0 && r.a1 == 1 && !rd.reserve(kCbIn, kInPages)) {
                    return;
                }
                break;
            case kSegStall:
                if (index == r.a0) {
                    rd.wait_stop(r);
                }
                break;
            default:
                break;
        }
    });
}
