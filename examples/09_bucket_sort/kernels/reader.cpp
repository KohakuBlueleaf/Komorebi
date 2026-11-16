// Bucket sort, Tensix side, the READER role (here on RISC-V core 0 of each Tensix core,
// on NoC 0): for every COUNT and PLACE command it brings this core's keys to L1.
#include <stdint.h>

#include "../bucket_sort.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"

// Two 4 KiB pages of keys: while the writer reads one, the reader fills the other.
constexpr uint32_t kCbKeys = 0;

void kernel_main() {
    // The keys tensor's layout (interleaved over the DRAM banks, 4 KiB pages) and
    // where its pages live, fixed when the host built the program.
    using Keys = km::Tensor<0>;
    km::Reader reader;
    // The runtime's loop: the body runs once per command whose group holds this core.
    reader.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler != BS_HANDLER_COUNT && handler != BS_HANDLER_PLACE) {
            return;
        }
        const uint32_t keys_address = command.a0;
        const uint32_t key_count = command.a1;
        // Steps 1 and 3: my partition is a run of keys [start, start + len) in input
        // order, from my rank in the group; the writer computes the same partition.
        const uint32_t rank = reader.me() - km::first(command);
        const km::Partition partition = (
            km::partition(key_count, rank, km::count(command))
        );
        if (partition.len == 0) {
            return;
        }
        const Keys keys(keys_address);
        const uint32_t first_page = partition.start / BS_KEYS_PER_PAGE;
        const uint32_t last_page = (
            (partition.start + partition.len - 1) / BS_KEYS_PER_PAGE
        );
        for (uint32_t page = first_page; page <= last_page; page++) {
            // A free page in the buffer (with a timeout: false means the core is
            // stopping).
            if (!reader.reserve(kCbKeys, 1)) {
                return;
            }
            // DRAM to L1 over NoC 0 (another island's page: asked for and waited
            // for); the wait for it times out at the wait limit.
            if (!km::read_pages(
                reader, command, keys, page, 1, get_write_ptr(kCbKeys)
            )) {
                return;
            }
            const uint32_t code = reader.poll(
                [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                },
                reader.args().wait_limit
            );
            if (code != km::kGaveNone) {
                return;
            }
            // The page is whole in L1: hand it to the writer.
            cb_push_back(kCbKeys, 1);
        }
    });
}
