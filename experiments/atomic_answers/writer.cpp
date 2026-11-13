// Probe writer (RISCV_1, NoC1): K reservations by non-posted atomic add on random
// cores, one at a time, each answer a timed wait in a return line of its own.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "stress.h"

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

    uint32_t xorshift(uint32_t& s) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }

    // The add's old value goes to `ret`, a line of this core (as the runtime does).
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

    template <typename F>
    bool timed(uint64_t cycles, F done) {
        const uint64_t end = wall64() + cycles;
        for (uint32_t i = 0;; i++) {
            if (done()) {
                return true;
            }
            if ((i & 15) == 15 && wall64() > end) {
                return false;
            }
        }
    }
}  // namespace

// Args: block address, askers, K, core index, seed, answer timeout (cycles), first
// counter tile and their count (0: every core holds its own tail), fixed return line.
void kernel_main() {
    const uint32_t blk = get_arg_val<uint32_t>(0), cores = get_arg_val<uint32_t>(1);
    const uint32_t K = get_arg_val<uint32_t>(2), me = get_arg_val<uint32_t>(3);
    uint32_t rnd = get_arg_val<uint32_t>(4);
    const uint32_t timeout = get_arg_val<uint32_t>(5);
    const uint32_t cbase = get_arg_val<uint32_t>(6), ccount = get_arg_val<uint32_t>(7);
    const uint32_t fixed = get_arg_val<uint32_t>(8);
    volatile tt_l1_ptr uint32_t* w = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blk)
    );
    volatile tt_l1_ptr uint32_t* st = w + hs::kStat;
    auto noc_of = [&](uint32_t c, uint32_t word) -> uint64_t {
        const uint32_t xy = w[hs::kTable + c];
        return get_noc_addr(xy & 0xFFFF, xy >> 16, blk + 4 * word);
    };
    for (uint32_t i = 0; i < hs::kRingN * 4; i++) {
        w[hs::kRing + i] = hs::kSentinel;
    }

    // Start barrier on core 0: one second for the launch skew.
    constexpr uint64_t kSecond = 1350000000ull;
    atomic_add(noc_of(0, hs::kBar), 1, blk + 4 * hs::kBarRet);
    if (!timed(kSecond, [] {
            return ncrisc_noc_nonposted_atomics_flushed(noc_index);
        })) {
        st[hs::sFail] = 1;
        st[hs::sDone] = 1;
        return;
    }
    const uint64_t b0 = wall64();
    for (;;) {
        noc_async_read(noc_of(0, hs::kBar), blk + 4 * hs::kPeek, 4);
        if (!timed(kSecond, [] {
                return ncrisc_noc_reads_flushed(noc_index);
            })) {
            st[hs::sFail] = 2;
            st[hs::sDone] = 1;
            return;
        }
        invalidate_l1_cache();
        if (w[hs::kPeek] >= cores) {
            break;
        }
        if (wall64() - b0 > kSecond) {
            st[hs::sFail] = 2;
            st[hs::sDone] = 1;
            return;
        }
    }

    uint32_t slot = 0, dead = 0, lost = 0, latmax = 0, k = 0;
    uint64_t latsum = 0;
    const uint64_t t0 = wall64();
    for (; k < K; k++) {
        const uint32_t r = xorshift(rnd);
        const uint32_t d = (r >> 8) % cores;
        const uint32_t n = 1 + (((r >> 24) & 7) == 0);  // 1, sometimes 2
        if (fixed) {
            slot = 0;
        } else {
            if (dead == 0xFFFFFFFFu) {  // every return line holds a lost answer
                st[hs::sFail] = 4;
                break;
            }
            while ((dead >> slot) & 1) {
                slot = (slot + 1) % hs::kRingN;
            }
        }
        volatile tt_l1_ptr uint32_t* ret = w + hs::kRing + 4 * slot;
        const uint64_t a = wall64();
        atomic_add(
            ccount ? noc_of(cbase + d % ccount, hs::kTailArr + 4 * d)
                   : noc_of(d, hs::kTail),
            n,
            blk + 4 * (hs::kRing + 4 * slot)
        );
        if (!timed(timeout, [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            })) {
            st[hs::sFail] = 3;
            break;
        }
        const bool got = timed(timeout, [&] {
            invalidate_l1_cache();
            return ret[0] != hs::kSentinel;
        });
        const uint32_t lat = static_cast<uint32_t>(wall64() - a);
        if (!got) {  // counted as answered, but the old value never landed
            lost++;
            w[hs::kLostN + d] += n;
            if (fixed) {
                break;  // a late answer could land in the one line: stop here
            }
            dead |= 1u << slot;  // never reused: a late answer lands there
            continue;
        }
        latsum += lat;
        latmax = lat > latmax ? lat : latmax;
        const uint32_t p = ret[0];
        ret[0] = hs::kSentinel;
        if (p < w[hs::kLastEnd + d]) {  // an answer older than this core's last one
            st[hs::sViol] = st[hs::sViol] + 1;
        }
        w[hs::kLastEnd + d] = p + n;
        w[hs::kSent + d] += n;
        slot = (slot + 1) % hs::kRingN;
    }
    st[hs::sResv] = k;
    st[hs::sLost] = lost;
    st[hs::sLatMax] = latmax;
    st[hs::sLatLo] = static_cast<uint32_t>(latsum);
    st[hs::sLatHi] = static_cast<uint32_t>(latsum >> 32);
    const uint64_t t1 = wall64() - t0;
    st[hs::sCycLo] = static_cast<uint32_t>(t1);
    st[hs::sCycHi] = static_cast<uint32_t>(t1 >> 32);
    st[hs::sDone] = 1;
}
