// Hop probe, sender: per packet, reserve n positions on the destination's IN_TAIL with
// one non-posted atomic add, then write the n items to its DRAM queue. Bounded waits.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hop.h"

using namespace hop;

namespace {

    uint64_t wall64() {
        const uint32_t lo = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
        );
        const uint32_t hi = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H)
        );
        return static_cast<uint64_t>(hi) << 32 | lo;
    }

    template <typename Ready>
    bool wait_bounded(Ready ready, uint64_t limit) {
        const uint64_t t0 = wall64();
        while (!ready()) {
            if (wall64() - t0 > limit) {
                return false;
            }
        }
        return true;
    }

    void atomic_add(uint64_t dst, uint32_t incr, uint32_t ret) {
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            dst,
            NOC_UNICAST_WRITE_VC,
            incr,
            31,
            false,
            false,
            ret
        );
    }

}  // namespace

// Args: shard, queue base, cap, me, cores, mode (0 pair: core 0 to core 1, 1 random),
// packets, n, salt, then the NoC x | y << 16 of every core.
void kernel_main() {
    const uint32_t sh = get_arg_val<uint32_t>(0), qbase = get_arg_val<uint32_t>(1);
    const uint32_t cap = get_arg_val<uint32_t>(2), me = get_arg_val<uint32_t>(3);
    const uint32_t cores = get_arg_val<uint32_t>(4), mode = get_arg_val<uint32_t>(5);
    const uint32_t packets = get_arg_val<uint32_t>(6), n = get_arg_val<uint32_t>(7);
    const uint32_t salt = get_arg_val<uint32_t>(8);
    constexpr uint32_t kTable = 9;
    volatile uint32_t* st = reinterpret_cast<volatile uint32_t*>(sh + kSStat);
    volatile uint32_t* ret = reinterpret_cast<volatile uint32_t*>(sh + kRet);
    volatile uint32_t* buf = reinterpret_cast<volatile uint32_t*>(sh + kSend);
    for (uint32_t w = 0; w < 16; w++) {
        st[w] = 0;
    }
    const InterleavedAddrGen<true> q = {
        .bank_base_address = qbase, .page_size = kChunkItems * kItemBytes
    };
    auto noc_of = [&](uint32_t core, uint32_t addr) {
        const uint32_t xy = get_arg_val<uint32_t>(kTable + core);
        return get_noc_addr(xy & 0xFFFF, xy >> 16, addr);
    };
    const bool active = mode == 1 || me == 0;

    // Start barrier on core 0 (every core's sender counts in).
    atomic_add(noc_of(0, sh + kReady), 1, sh + kRet);
    if (!wait_bounded(
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            kLimit
        )) {
        st[kSFail] = kFailAtomic;
        st[kSDone] = 1;
        return;
    }
    volatile uint32_t* peek = reinterpret_cast<volatile uint32_t*>(sh + kPeek);
    bool read_hung = false;
    const bool all_in = wait_bounded(
        [&] {
            noc_async_read(noc_of(0, sh + kReady), sh + kPeek, 4);
            if (!wait_bounded(
                    [] {
                        return ncrisc_noc_reads_flushed(noc_index);
                    },
                    kLimit
                )) {
                read_hung = true;
                return true;
            }
            invalidate_l1_cache();
            return *peek >= cores;
        },
        kStartLimit
    );
    if (!all_in || read_hung) {
        st[kSFail] = kFailBarrier;
        st[kSDone] = 1;
        return;
    }
    st[kSStart] = static_cast<uint32_t>(wall64());
    uint32_t x = me * 7919u + 1u, items = 0, amax = 0, asum = 0, fail = kOk;
    for (uint32_t k = 0; active && k < packets; k++) {
        x = lcg(x);
        const uint32_t dest = mode == 0 ? 1 : (x >> 8) % cores;
        const uint64_t a0 = wall64();
        ret[0] = 0xDEADBEEFu;
        atomic_add(noc_of(dest, sh + kTail), n, sh + kRet);
        if (!wait_bounded(
                [] {
                    return ncrisc_noc_nonposted_atomics_flushed(noc_index);
                },
                kLimit
            )) {
            fail = kFailAtomic;
            break;
        }
        const uint32_t ac = static_cast<uint32_t>(wall64() - a0);
        amax = ac > amax ? ac : amax;
        asum += ac;
        invalidate_l1_cache();
        const uint32_t p = ret[0];
        // The previous packet's writes must have left the buffer before it is refilled.
        if (!wait_bounded(
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                kLimit
            )) {
            fail = kFailFlush;
            break;
        }
        for (uint32_t i = 0; i < n; i++) {
            volatile uint32_t* it = buf + i * kItemWords;
            const uint32_t mark = (p + i + 1) ^ salt;
            it[0] = it[4] = it[8] = it[12] = mark;
            it[kWSrc] = me;
            it[kWPacket] = k;
            it[kWIndex] = i;
            it[kWPayload] = payload(me, k, i);
        }
        // Contiguous runs inside one DRAM page of the destination's queue.
        for (uint32_t i = 0; i < n;) {
            const uint32_t pos = (p + i) & (cap - 1);
            const uint32_t in_page = pos % kChunkItems;
            uint32_t run = kChunkItems - in_page;
            run = run < n - i ? run : n - i;
            const uint32_t page = dest * (cap / kChunkItems) + pos / kChunkItems;
            noc_async_write(
                sh + kSend + i * kItemBytes,
                get_noc_addr(page, q, in_page * kItemBytes),
                run * kItemBytes
            );
            i += run;
        }
        items += n;
    }
    if (fail == kOk &&
        !wait_bounded(
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            kLimit
        )) {
        fail = kFailFlush;
    }
    st[kSEnd] = static_cast<uint32_t>(wall64());
    st[kSItems] = items;
    st[kSAtomMax] = amax;
    st[kSAtomSum] = asum;
    st[kSFail] = fail;
    st[kSDone] = 1;
}
