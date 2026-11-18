// DRAM reader: `count` pages, first + i * step, `block` reads a barrier, into its own
// L1 shard after the run's cores meet at a counter; the clock every `every` blocks.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/core/km_core.hpp"

namespace {

    constexpr uint32_t kTrace = 64;             // first trace word
    constexpr uint32_t kTraceWords = 960;       // samples at most
    constexpr uint64_t kLimit = 2000000000ull;  // cycles of any wait (about 1.5 s)
    constexpr uint32_t kBlocksMax = 64;         // blocks a consumer follows
    constexpr uint32_t kNoPart = 1;             // a block's stamp: no part of it here

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

// Args: area, landing, slots, source, page bytes, first, step, count, block, every,
// leader x | y << 16, counter, kernels, tag, pattern length, blocks taken, banks, ...
void kernel_main() {
    const uint32_t area = get_arg_val<uint32_t>(0), landing = get_arg_val<uint32_t>(1);
    const uint32_t slots = get_arg_val<uint32_t>(2), source = get_arg_val<uint32_t>(3);
    const uint32_t bytes = get_arg_val<uint32_t>(4), first = get_arg_val<uint32_t>(5);
    const uint32_t step = get_arg_val<uint32_t>(6), count = get_arg_val<uint32_t>(7);
    const uint32_t block = get_arg_val<uint32_t>(8), every = get_arg_val<uint32_t>(9);
    const uint32_t leader = get_arg_val<uint32_t>(10);
    const uint32_t counter = get_arg_val<uint32_t>(11);
    const uint32_t kernels = get_arg_val<uint32_t>(12), tag = get_arg_val<uint32_t>(13);
    const uint32_t length = get_arg_val<uint32_t>(14);
    const uint32_t takes = get_arg_val<uint32_t>(15);
    const uint32_t banks = get_arg_val<uint32_t>(16);
    // ... rows of `run` pages `pitch` apart (run 0: none), and X read in step with
    // them: source, page bytes, first, pages a block (0: none), pages, landing, pattern
    const uint32_t run = get_arg_val<uint32_t>(17), pitch = get_arg_val<uint32_t>(18);
    const uint32_t x_source = get_arg_val<uint32_t>(19);
    const uint32_t x_bytes = get_arg_val<uint32_t>(20);
    const uint32_t x_first = get_arg_val<uint32_t>(21);
    const uint32_t x_block = get_arg_val<uint32_t>(22);
    const uint32_t x_count = get_arg_val<uint32_t>(23);
    const uint32_t x_landing = get_arg_val<uint32_t>(24);
    const uint32_t x_length = get_arg_val<uint32_t>(25);
    const uint32_t x_takes = get_arg_val<uint32_t>(26);
    // ... a consumer: cycles a block (0: none), the stamps' words, blocks it lets ahead
    const uint32_t consume = get_arg_val<uint32_t>(27);
    const uint32_t stamps = get_arg_val<uint32_t>(28);
    const uint32_t depth = get_arg_val<uint32_t>(29);
    const uint32_t risc = (tag >> 16) & 1;
    volatile tt_l1_ptr uint32_t* own = km::l1(stamps + 4 * kBlocksMax * risc);
    volatile tt_l1_ptr uint32_t* other = km::l1(stamps + 4 * kBlocksMax * (1 - risc));
    if (consume != 0) {
        for (uint32_t j = 0; j < kBlocksMax; j++) {
            own[j] = 0;  // cleared before the meeting: both are clear after it
        }
    }
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
    uint32_t page = first, done = 0, blocks = 0, samples = 0, slot = 0, x_done = 0;
    bool ok = true;
    // The consumer takes blocks in order, each when both kernels' parts landed, for
    // `consume` cycles; its time on the low clock word (both kernels compute the same)
    uint32_t consumed = 0, free_at = 0;
    const auto consumer_took = [&](uint32_t block_index) {
        while (consumed <= block_index && ok) {
            const uint64_t t0 = km::wall64();
            uint32_t mine_at = 0, theirs_at = 0;
            do {
                invalidate_l1_cache();
                mine_at = own[consumed];
                theirs_at = other[consumed];
            } while ((mine_at == 0 || theirs_at == 0) && km::wall64() - t0 < kLimit);
            if (mine_at == 0 || theirs_at == 0) {
                ok = false;
                break;
            }
            uint32_t at = mine_at == kNoPart ? theirs_at : mine_at;
            if (theirs_at != kNoPart && static_cast<int32_t>(theirs_at - at) > 0) {
                at = theirs_at;
            }
            if (consumed == 0 || static_cast<int32_t>(at - free_at) > 0) {
                free_at = at;
            }
            free_at += consume;
            consumed++;
        }
        while (
            static_cast<int32_t>(static_cast<uint32_t>(km::wall64()) - free_at) < 0
        ) {
        }
    };
    // `count` pages of the banks in `banks` from `first`, in blocks; block b is this
    // kernel's when bit b % length of `takes` is set (the other kernel takes the rest)
    for (uint32_t seen = 0, b = 0; seen < count && ok; b++) {
        const uint32_t n = count - seen < block ? count - seen : block;
        const bool mine = (takes >> (b % length)) & 1;
        const uint32_t x_at = b * x_block;
        const bool x_mine = (
            x_block != 0 && x_at < x_count && ((x_takes >> (b % x_length)) & 1)
        );
        const bool counted = consume != 0 && b < kBlocksMax;
        if (counted && (mine || x_mine) && b >= depth) {
            consumer_took(b - depth);  // room in the half: the consumer took b - depth
        }
        // X's block b first, on its own pattern (a zip's pass issues both, one barrier)
        if (x_mine) {
            const uint32_t x_n = x_count - x_at < x_block ? x_count - x_at : x_block;
            for (uint32_t i = 0; i < x_n; i++) {
                const uint32_t p = x_first + x_at + i;
                const uint64_t from = (
                    get_noc_addr_from_bank_id<true>(
                        p % NUM_DRAM_BANKS, x_source + (p / NUM_DRAM_BANKS) * x_bytes
                    )
                );
                noc_async_read(from, x_landing + (i & 1) * x_bytes, x_bytes);
            }
            x_done += x_n;
            if (!mine) {
                ok = reads_flushed();
            }
        }
        for (uint32_t i = 0; i < n; i++) {
            if (run != 0) {  // rows of `run` pages, `pitch` apart
                page = first + (seen + i) / run * pitch + (seen + i) % run;
            }
            while (!((banks >> (page % NUM_DRAM_BANKS)) & 1)) {
                page += step;
            }
            if (mine) {
                const uint64_t from = (
                    get_noc_addr_from_bank_id<true>(
                        page % NUM_DRAM_BANKS, source + (page / NUM_DRAM_BANKS) * bytes
                    )
                );
                noc_async_read(from, landing + slot * bytes, bytes);
                slot = slot + 1 == slots ? 0 : slot + 1;
            }
            page += step;
        }
        seen += n;
        if (mine) {
            ok = ok && reads_flushed();
            done += n;
            blocks++;
            if (blocks % every == 0 && samples < kTraceWords) {
                o[kTrace + samples++] = static_cast<uint32_t>(km::wall64() - start);
            }
        }
        if (counted) {
            own[b] = (
                mine || x_mine ? static_cast<uint32_t>(km::wall64()) | 2u : kNoPart
            );
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
    o[10] = x_done;
}
