// Bucket sort, Tensix side, the WRITER (RISC-V core 1 of each Tensix core, NoC 1). It
// runs COUNT (step 1) and PLACE (step 3) on the keys the reader brings into L1.
#include <stdint.h>

#include "../bucket_sort.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

constexpr uint32_t kCbKeys = 0;  // filled by the reader, a page of 1,024 keys at a time
constexpr uint32_t kCbScratch = 16;  // L1 of this core alone: B words, then the slots

namespace {

    // Block k's table address (command words a4 to a6) and DRAM bank (a3, 8 bits each).
    uint32_t table_of(const km_command_t& r, uint32_t k) {
        return k == 0 ? r.a4 : k == 1 ? r.a5 : r.a6;
    }

    uint32_t bank_of(const km_command_t& r, uint32_t k) {
        return (r.a3 >> (8 * k)) & 0xFFu;
    }

}  // namespace

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    km::Writer wr;
    const uint32_t scratch = get_write_ptr(kCbScratch);
    // One word per bucket: the counts in COUNT, the next place of each bucket in PLACE.
    uint32_t* bucket = reinterpret_cast<uint32_t*>(scratch);
    const uint32_t slots = scratch + 4 * BS_BUCKETS_MAX;

    // Every NoC wait below is polled with the wait limit and the stop flag: false means
    // the core is stopping, and the body then returns false (no report).
    auto reads_done = [&] {
        const bool ok = wr.poll(
                            [&] {
                                return ncrisc_noc_reads_flushed(noc_index);
                            },
                            wr.args().wait_limit
                        ) == km::kGaveNone;
        invalidate_l1_cache();  // this RISC's load cache must not hold the old words
        return ok;
    };
    // Sent: the data has left L1, so the slots may be written again.
    auto writes_sent = [&] {
        return wr.poll(
                   [&] {
                       return ncrisc_noc_nonposted_writes_sent(noc_index);
                   },
                   wr.args().wait_limit
               ) == km::kGaveNone;
    };
    // Acknowledged: the data is in DRAM, so the command may be reported done.
    auto writes_done = [&] {
        return wr.poll(
                   [&] {
                       return ncrisc_noc_nonposted_writes_flushed(noc_index);
                   },
                   wr.args().wait_limit
               ) == km::kGaveNone;
    };

    // The runtime's loop: the body runs once for each command whose group holds this
    // core, and returns whether the command is done here (then the runtime reports it).
    wr.run([&](const km_command_t& r) {
        const uint32_t handler = km::handler(r);
        if (handler != BS_HANDLER_COUNT && handler != BS_HANDLER_PLACE) {
            return true;
        }
        // The same partition of keys as the reader's, the same blocks as the thread's.
        const uint32_t rank = wr.me() - km::first(r);
        const km::Partition s = km::partition(r.a1, rank, km::count(r));
        const uint32_t log2_b = r.a2 & 0xFFu, blocks = r.a2 >> 8;
        const uint32_t buckets = 1u << log2_b, shift = 32 - log2_b;
        const uint32_t chunks = buckets / BS_CHUNK;

        if (handler == BS_HANDLER_PLACE) {
            // Step 3a: my row of the plan, block by block, from each work cluster's
            // region in its DRAM bank into L1: where my first key of each bucket goes.
            for (uint32_t k = 0; k < blocks; k++) {
                uint32_t lo, hi;
                bs_block(k, blocks, chunks, &lo, &hi);
                const uint32_t at = (rank * buckets + lo * BS_CHUNK) * 4;
                if (hi > lo) {
                    noc_async_read(
                        get_noc_addr_from_bank_id<true>(
                            bank_of(r, k), table_of(r, k) + at
                        ),
                        scratch + lo * BS_CHUNK * 4,
                        (hi - lo) * BS_CHUNK * 4
                    );
                }
            }
            if (!reads_done()) {
                return false;
            }
        } else {
            // Step 1a: my B counters start at zero.
            for (uint32_t b = 0; b < buckets; b++) {
                bucket[b] = 0;
            }
        }

        // Steps 1b and 3b: my keys, a page at a time, as the reader brings them.
        const auto out = TensorAccessor(out_args, r.a7);
        uint32_t written = 0;
        const uint32_t first = s.start / BS_KEYS_PER_PAGE;
        const uint32_t last = (s.start + s.len - 1) / BS_KEYS_PER_PAGE;
        for (uint32_t page = first; s.len != 0 && page <= last; page++) {
            if (!wr.wait_front(kCbKeys, 1)) {
                return false;
            }
            invalidate_l1_cache();  // the reader's NoC wrote this page behind the cache
            const uint32_t* keys = reinterpret_cast<uint32_t*>(get_read_ptr(kCbKeys));
            const uint32_t base = page * BS_KEYS_PER_PAGE;
            const uint32_t lo = s.start > base ? s.start - base : 0;
            const uint32_t end = s.start + s.len - base;
            const uint32_t hi = end < BS_KEYS_PER_PAGE ? end : BS_KEYS_PER_PAGE;
            for (uint32_t i = lo; i < hi; i++) {
                const uint32_t b = keys[i] >>
                    shift;  // the bucket: the top log2(B) bits
                if (handler == BS_HANDLER_COUNT) {
                    bucket[b]++;
                    continue;
                }
                // PLACE: the key goes to its bucket's next place; the place moves on.
                const uint32_t to = bucket[b]++;
                if (to >= r.a1) {
                    continue;  // a wrong plan never writes outside the output
                }
                // Every slot written since the last check must have left L1 first.
                if (written != 0 && written % BS_SLOTS == 0 && !writes_sent()) {
                    return false;
                }
                // A NoC write's two ends share their low 4 address bits: the key sits
                // in its 16-byte slot at the word its place has in DRAM.
                const uint32_t src = slots + (written % BS_SLOTS) * 16 + (to % 4) * 4;
                *reinterpret_cast<volatile uint32_t*>(src) = keys[i];
                const uint32_t page_of = to / BS_KEYS_PER_PAGE;
                const uint32_t offset = (to % BS_KEYS_PER_PAGE) * 4;
                noc_async_write(src, out.get_noc_addr(page_of, offset), 4);
                written++;
            }
            cb_pop_front(kCbKeys, 1);
        }

        if (handler == BS_HANDLER_COUNT) {
            // Step 1c: my row of the counts table to each block's work cluster, in its
            // region in its bank, where its thread reads it (README.md, data layout).
            for (uint32_t k = 0; k < blocks; k++) {
                uint32_t lo, hi;
                bs_block(k, blocks, chunks, &lo, &hi);
                const uint32_t at = (rank * buckets + lo * BS_CHUNK) * 4;
                if (hi > lo) {
                    noc_async_write(
                        scratch + lo * BS_CHUNK * 4,
                        get_noc_addr_from_bank_id<true>(
                            bank_of(r, k), table_of(r, k) + at
                        ),
                        (hi - lo) * BS_CHUNK * 4
                    );
                }
            }
        }
        // Everything this command wrote is in DRAM before the command is reported done:
        // the next reader of it (the thread, or the host) may look at once.
        return writes_done();
    });
}
