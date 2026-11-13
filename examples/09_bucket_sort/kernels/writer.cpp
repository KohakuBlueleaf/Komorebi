// Bucket sort, Tensix side, the WRITER role (here on RISC-V core 1 of each Tensix core,
// on NoC 1). It runs COUNT (step 1) and PLACE (step 3) on the keys the reader brings.
#include <stdint.h>

#include "../bucket_sort.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

constexpr uint32_t kCbKeys = 0;  // filled by the reader, a page of 1,024 keys at a time
constexpr uint32_t kCbScratch = 16;  // own L1: a word per bucket, then the slots

namespace {

    // Block `block`'s table address (of the command's three) and DRAM bank (`banks`
    // holds 8 bits each).
    uint32_t table_of(const uint32_t tables[3], uint32_t block) {
        return block == 0 ? tables[0] : block == 1 ? tables[1] : tables[2];
    }

    uint32_t bank_of(uint32_t banks, uint32_t block) {
        return (banks >> (8 * block)) & 0xFFu;
    }

}  // namespace

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<0>();
    km::Writer writer;
    const uint32_t scratch = get_write_ptr(kCbScratch);
    // One word per bucket: the counts in COUNT, the next place of each bucket in PLACE.
    uint32_t* bucket = reinterpret_cast<uint32_t*>(scratch);
    const uint32_t slots = scratch + 4 * BS_BUCKETS_MAX;

    // Every NoC wait below is polled with the wait limit and the stop flag: false means
    // the core is stopping, and the body then returns false (no report).
    auto reads_done = [&] {
        const bool ok = writer.poll(
            [&] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
        invalidate_l1_cache();  // this RISC's load cache must not hold the old words
        return ok;
    };
    // Sent: the data has left L1, so the slots may be written again.
    auto writes_sent = [&] {
        return writer.poll(
            [&] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
    };
    // Acknowledged: the data is in DRAM, so the command may be reported done.
    auto writes_done = [&] {
        return writer.poll(
            [&] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
    };

    // The runtime's loop: the body runs once for each command whose group holds this
    // core, and returns whether the command is done here (then the runtime reports it).
    writer.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler != BS_HANDLER_COUNT && handler != BS_HANDLER_PLACE) {
            return true;
        }
        const uint32_t key_count = command.a1;
        const uint32_t log2_buckets = command.a2 & 0xFFu, blocks = command.a2 >> 8;
        const uint32_t banks = command.a3;
        const uint32_t tables[3] = {command.a4, command.a5, command.a6};
        const uint32_t output_address = command.a7;
        // The same partition of keys as the reader's, the same blocks as the thread's.
        const uint32_t rank = writer.me() - km::first(command);
        const km::Partition partition = (
            km::partition(key_count, rank, km::count(command))
        );
        const uint32_t buckets = 1u << log2_buckets, shift = 32 - log2_buckets;
        const uint32_t chunks = buckets / BS_CHUNK;

        if (handler == BS_HANDLER_PLACE) {
            // Step 3a: my row of the plan, block by block, from each work cluster's
            // region in its DRAM bank into L1: where my first key of each bucket goes.
            for (uint32_t block = 0; block < blocks; block++) {
                uint32_t first_chunk, end_chunk;
                bs_block(block, blocks, chunks, &first_chunk, &end_chunk);
                const uint32_t offset = (rank * buckets + first_chunk * BS_CHUNK) * 4;
                if (end_chunk > first_chunk) {
                    noc_async_read(
                        get_noc_addr_from_bank_id<true>(
                            bank_of(banks, block), table_of(tables, block) + offset
                        ),
                        scratch + first_chunk * BS_CHUNK * 4,
                        (end_chunk - first_chunk) * BS_CHUNK * 4
                    );
                }
            }
            if (!reads_done()) {
                return false;
            }
        } else {
            // Step 1a: my counters, one per bucket, start at zero.
            for (uint32_t bucket_index = 0; bucket_index < buckets; bucket_index++) {
                bucket[bucket_index] = 0;
            }
        }

        // Steps 1b and 3b: my keys, a page at a time, as the reader brings them.
        const auto output = TensorAccessor(output_args, output_address);
        uint32_t written = 0;
        const uint32_t first_page = partition.start / BS_KEYS_PER_PAGE;
        const uint32_t last_page = (
            (partition.start + partition.len - 1) / BS_KEYS_PER_PAGE
        );
        for (
            uint32_t page = first_page; partition.len != 0 && page <= last_page; page++
        ) {
            if (!writer.wait_front(kCbKeys, 1)) {
                return false;
            }
            invalidate_l1_cache();  // the reader's NoC wrote this page behind the cache
            const uint32_t* keys = reinterpret_cast<uint32_t*>(get_read_ptr(kCbKeys));
            const uint32_t base = page * BS_KEYS_PER_PAGE;
            const uint32_t first_key = (
                partition.start > base ? partition.start - base : 0
            );
            const uint32_t keys_end = partition.start + partition.len - base;
            const uint32_t end_key = (
                keys_end < BS_KEYS_PER_PAGE ? keys_end : BS_KEYS_PER_PAGE
            );
            for (uint32_t key_index = first_key; key_index < end_key; key_index++) {
                // the bucket: the top log2(buckets) bits
                const uint32_t bucket_index = keys[key_index] >> shift;
                if (handler == BS_HANDLER_COUNT) {
                    bucket[bucket_index]++;
                    continue;
                }
                // PLACE: the key goes to its bucket's next place; the place moves on.
                const uint32_t place = bucket[bucket_index]++;
                if (place >= key_count) {
                    continue;  // a wrong plan never writes outside the output
                }
                // Every slot written since the last check must have left L1 first.
                if (written != 0 && written % BS_SLOTS == 0 && !writes_sent()) {
                    return false;
                }
                // A NoC write's two ends share their low 4 address bits: the key sits
                // in its 16-byte slot at the word its place has in DRAM.
                const uint32_t source = (
                    slots + (written % BS_SLOTS) * 16 + (place % 4) * 4
                );
                *reinterpret_cast<volatile uint32_t*>(source) = keys[key_index];
                const uint32_t page_of = place / BS_KEYS_PER_PAGE;
                const uint32_t offset = (place % BS_KEYS_PER_PAGE) * 4;
                noc_async_write(source, output.get_noc_addr(page_of, offset), 4);
                written++;
            }
            cb_pop_front(kCbKeys, 1);
        }

        if (handler == BS_HANDLER_COUNT) {
            // Step 1c: my row of the counts table to each block's work cluster, in its
            // region in its bank, where its thread reads it (README.md, data layout).
            for (uint32_t block = 0; block < blocks; block++) {
                uint32_t first_chunk, end_chunk;
                bs_block(block, blocks, chunks, &first_chunk, &end_chunk);
                const uint32_t offset = (rank * buckets + first_chunk * BS_CHUNK) * 4;
                if (end_chunk > first_chunk) {
                    noc_async_write(
                        scratch + first_chunk * BS_CHUNK * 4,
                        get_noc_addr_from_bank_id<true>(
                            bank_of(banks, block), table_of(tables, block) + offset
                        ),
                        (end_chunk - first_chunk) * BS_CHUNK * 4
                    );
                }
            }
        }
        // Everything this command wrote is in DRAM before the command is reported done:
        // the next reader of it (the thread, or the host) may look at once.
        return writes_done();
    });
}
