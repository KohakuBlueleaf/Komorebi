// DRAM reader: `count` pages, first + i * step, `block` reads a barrier, into its own
// L1 shard after the run's cores meet at a counter; the clock every `every` blocks.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/core/km_core.hpp"

namespace {

    constexpr uint32_t kTrace = 64;             // first trace word
    constexpr uint32_t kTraceWords = 960;       // samples at most
    constexpr uint64_t kLimit = 2000000000ull;  // cycles of any wait (about 1.5 s)

    // The barrier on this kernel's reads, with a timeout.
    bool reads_flushed() {
        const uint64_t t0 = km::wall64();
        do {
            if (ncrisc_noc_reads_flushed(noc_index)) {
                return true;
            }
        } while (km::wall64() - t0 < kLimit);
        return false;
    }

}  // namespace

// Args: area, landing, its pages, source, page bytes, first, step, count, block,
// every, leader's NoC x | y << 16, its counter's address, kernels in the run, tag.
void kernel_main() {
    const uint32_t area = get_arg_val<uint32_t>(0), landing = get_arg_val<uint32_t>(1);
    const uint32_t slots = get_arg_val<uint32_t>(2), source = get_arg_val<uint32_t>(3);
    const uint32_t bytes = get_arg_val<uint32_t>(4), first = get_arg_val<uint32_t>(5);
    const uint32_t step = get_arg_val<uint32_t>(6), count = get_arg_val<uint32_t>(7);
    const uint32_t block = get_arg_val<uint32_t>(8), every = get_arg_val<uint32_t>(9);
    const uint32_t leader = get_arg_val<uint32_t>(10);
    const uint32_t counter = get_arg_val<uint32_t>(11);
    const uint32_t kernels = get_arg_val<uint32_t>(12), tag = get_arg_val<uint32_t>(13);
    volatile tt_l1_ptr uint32_t* o = km::l1(area);
    o[0] = tag;
    o[1] = noc_index;
    o[7] = 0;  // 1: the meeting timed out
    // meet: add one to the leader's counter, then read it until every kernel came
    const uint64_t at = get_noc_addr(leader & 0xFFFF, leader >> 16, counter);
    noc_semaphore_inc(at, 1);
    noc_async_atomic_barrier();
    const uint64_t met = km::wall64();
    while (true) {
        noc_async_read(at, area + 4 * 15, 4);
        if (!reads_flushed()) {
            o[7] = 1;
            break;
        }
        invalidate_l1_cache();
        if (o[15] >= kernels) {
            break;
        }
        if (km::wall64() - met > kLimit) {
            o[7] = 1;
            break;
        }
    }
    const uint64_t start = km::wall64();
    uint32_t page = first, done = 0, blocks = 0, samples = 0, slot = 0;
    bool ok = true;
    while (done < count && ok) {
        const uint32_t n = count - done < block ? count - done : block;
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t bank = page % NUM_DRAM_BANKS;
            const uint64_t from = (
                get_noc_addr_from_bank_id<true>(
                    bank, source + (page / NUM_DRAM_BANKS) * bytes
                )
            );
            noc_async_read(from, landing + slot * bytes, bytes);
            page += step;
            slot = slot + 1 == slots ? 0 : slot + 1;
        }
        ok = reads_flushed();
        done += n;
        blocks++;
        if (blocks % every == 0 && samples < kTraceWords) {
            o[kTrace + samples++] = static_cast<uint32_t>(km::wall64() - start);
        }
    }
    const uint64_t end = km::wall64();
    o[2] = static_cast<uint32_t>(start);
    o[3] = static_cast<uint32_t>(start >> 32);
    o[4] = static_cast<uint32_t>(end - start);
    o[5] = done;
    o[6] = samples;
    o[8] = ok ? 1 : 0;
    o[9] = static_cast<uint32_t>(start - met);  // the wait at the meeting
}
