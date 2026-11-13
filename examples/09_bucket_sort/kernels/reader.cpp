// Bucket sort, Tensix side, the READER (RISC-V core 0 of each Tensix core, NoC 0): for
// every COUNT and PLACE command it brings this core's partition of the keys to L1.
#include <stdint.h>

#include "../bucket_sort.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

// Two 4 KiB pages of keys: while the writer reads one, the reader fills the other.
constexpr uint32_t kCbKeys = 0;

void kernel_main() {
    // The keys tensor's layout (interleaved over the DRAM banks, 4 KiB pages), fixed
    // when the host built the program.
    constexpr auto keys_args = TensorAccessorArgs<0>();
    km::Reader rd;
    // The runtime's loop: the body runs once per command whose group holds this core.
    rd.run([&](const km_command_t& r) {
        const uint32_t handler = km::handler(r);
        if (handler != BS_HANDLER_COUNT && handler != BS_HANDLER_PLACE) {
            return;
        }
        // Steps 1 and 3: my partition is a run of keys [start, start + len) in input
        // order, from my rank in the group; the writer computes the same partition.
        const uint32_t rank = rd.me() - km::first(r);
        const km::Partition s = km::partition(r.a1, rank, km::count(r));
        if (s.len == 0) {
            return;
        }
        const auto keys = TensorAccessor(keys_args, r.a0);
        const uint32_t first = s.start / BS_KEYS_PER_PAGE;
        const uint32_t last = (s.start + s.len - 1) / BS_KEYS_PER_PAGE;
        for (uint32_t page = first; page <= last; page++) {
            // A free page in the buffer (with a timeout: false means the core is
            // stopping).
            if (!rd.reserve(kCbKeys, 1)) {
                return;
            }
            // DRAM to L1 over NoC 0; the wait for it times out at the wait limit.
            noc_async_read_page(page, keys, get_write_ptr(kCbKeys));
            const uint32_t code = rd.poll(
                [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                },
                rd.args().wait_limit
            );
            if (code != km::kGaveNone) {
                return;
            }
            // The page is whole in L1: hand it to the writer.
            cb_push_back(kCbKeys, 1);
        }
    });
}
