// X280 ports experiment, grid side: the reader owns the core's port to its X280 core
// and keeps the stats page of each record (xp.h).
#include <stdint.h>

#include "../xp.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_x280.hpp"
#include "komorebi/device/km_worker.hpp"

constexpr uint32_t kCbScratch = 16;

namespace {

    // Message stats in L1 over a run's rounds, written to the core's stats page after
    // each record.
    struct Stats {
        volatile tt_l1_ptr uint32_t* w;

        // Round 1 starts the stats; later rounds add to them.
        void begin(uint32_t round) {
            if (round <= 1) {
                for (uint32_t i = 0; i < XP_PAGE_BYTES / 4; i++) {
                    w[i] = 0;
                }
                w[XP_ST_MIN] = 0xFFFFFFFFu;
            }
            w[XP_ST_ROUND] = round;
        }

        void time(uint32_t dt) {
            w[XP_ST_COUNT] = w[XP_ST_COUNT] + 1;
            w[XP_ST_MIN] = dt < w[XP_ST_MIN] ? dt : w[XP_ST_MIN];
            w[XP_ST_MAX] = dt > w[XP_ST_MAX] ? dt : w[XP_ST_MAX];
            const uint32_t lo = w[XP_ST_SUM_LO] + dt;
            w[XP_ST_SUM_HI] = w[XP_ST_SUM_HI] + (lo < dt ? 1 : 0);
            w[XP_ST_SUM_LO] = lo;
            uint32_t bin = dt;
            if (dt >= 16) {
                uint32_t e = 4;
                while (e < 31 && (dt >> (e + 1)) != 0) {
                    e++;
                }
                bin = 16 + 8 * (e - 4) + ((dt >> (e - 3)) & 7);
                bin = bin < XP_HIST_BINS - 1 ? bin : XP_HIST_BINS - 1;
            }
            w[XP_ST_HIST + bin] = w[XP_ST_HIST + bin] + 1;
        }
    };

}  // namespace

void kernel_main() {
    constexpr auto stats_args = TensorAccessorArgs<0>();
    km::Reader rd;
    km::X280Port port(rd);
    const uint32_t scratch = get_write_ptr(kCbScratch);
    volatile tt_l1_ptr uint32_t* flags = km::l1(scratch + XP_SCRATCH_FLAGS);
    volatile tt_l1_ptr uint32_t* block = km::l1(scratch + XP_SCRATCH_BLOCK);
    Stats st{km::l1(scratch + XP_SCRATCH_STATS)};
    const uint32_t me = rd.me();

    // A NoC barrier polled with the wait limit; false (recorded) if it passed.
    auto flushed = [&](const km_record_t& r, bool reads) {
        const uint32_t code = rd.poll(
            [&] {
                return (
                    reads ? ncrisc_noc_reads_flushed(noc_index)
                          : ncrisc_noc_nonposted_writes_flushed(noc_index)
                );
            },
            rd.args().wait_limit
        );
        if (code != km::kGaveNone) {
            rd.gave_up(km::kStReader, code, km::kWhereData, km::slot(r), r.seq);
            return false;
        }
        return true;
    };

    // The stats page to DRAM, after the writer's count of reads when it reads.
    auto put_stats = [&](const km_record_t& r, uint32_t stats, bool reads) {
        if (reads) {
            const uint32_t code = rd.poll(
                [&] {
                    return flags[4] == r.seq;
                },
                rd.args().wait_limit
            );
            if (code != km::kGaveNone) {
                rd.gave_up(km::kStReader, code, km::kWhereData, km::slot(r), r.seq);
                return;
            }
            st.w[XP_ST_READS] = flags[3];
        }
        const auto out = TensorAccessor(stats_args, stats);
        noc_async_write_page(me, out, scratch + XP_SCRATCH_STATS);
        flushed(r, false);
    };

    rd.run([&](const km_record_t& r) {
        const uint32_t seg = km::seg(r);
        km::X280Msg m;
        if (seg == XP_SEG_TELL) {
            if (port.wait_told(m)) {
                flags[1] = m.w[0];
                flags[0] = r.seq;
            }
            return;
        }
        if (seg == XP_SEG_SEND) {
            port.tell(r.a0 + me, r.a7, 0);
            return;
        }
        if (seg == XP_SEG_ASK) {
            st.begin(r.a7);
            for (uint32_t k = 1; k <= r.a0; k++) {
                const uint32_t t0 = km::wall();
                if (!port.call(XP_ASK_PLAIN, k, me, m)) {
                    st.w[XP_ST_GAVE] = 1;
                    break;
                }
                st.time(km::wall() - t0);
                if (m.w[0] != XP_ANSWER(k, me)) {
                    st.w[XP_ST_BAD] = st.w[XP_ST_BAD] + 1;
                }
            }
            flags[2] = r.seq;  // the writer's reads may end
            if (st.w[XP_ST_GAVE] == 0 && (r.a2 & XP_F_DONE)) {
                port.tell(XP_DONE, r.a7, st.w[XP_ST_COUNT]);
            }
            if (st.w[XP_ST_GAVE] == 0) {
                put_stats(r, r.a1, (r.a2 & XP_F_READS) != 0);
            }
            return;
        }
        if (seg == XP_SEG_BLOCK) {
            st.begin(r.a7);
            for (uint32_t k = 1; k <= r.a0; k++) {
                for (uint32_t i = 0; i < XP_BLOCK_BYTES / 4; i++) {
                    block[i] = 0;
                }
                const uint32_t t0 = km::wall();
                const bool l1 = r.a1 == XP_BLOCK_L1;
                const uint32_t what = l1 ? XP_ASK_BLOCK : XP_ASK_DBLOCK;
                if (!port.call(what, scratch + XP_SCRATCH_BLOCK, k, m)) {
                    st.w[XP_ST_GAVE] = 1;
                    break;
                }
                if (!l1) {  // the answer: bank address and the bank's tile
                    const uint32_t xy = m.w[1];
                    const uint64_t src = get_noc_addr(xy & 0xFFFFu, xy >> 16, m.w[0]);
                    noc_async_read(src, scratch + XP_SCRATCH_BLOCK, XP_BLOCK_BYTES);
                    if (!flushed(r, true)) {
                        st.w[XP_ST_GAVE] = 1;
                        break;
                    }
                }
                st.time(km::wall() - t0);
                invalidate_l1_cache();
                uint32_t bad = 0;
                for (uint32_t i = 0; i < XP_BLOCK_BYTES / 4; i++) {
                    bad += block[i] != XP_BLOCK_WORD(k, me, i) ? 1 : 0;
                }
                st.w[XP_ST_BAD] = st.w[XP_ST_BAD] + (bad ? 1 : 0);
            }
            if (st.w[XP_ST_GAVE] == 0) {
                port.tell(XP_DONE, r.a7, st.w[XP_ST_COUNT]);
                put_stats(r, r.a2, false);
            }
            return;
        }
        if (seg == XP_SEG_RATE) {
            st.begin(r.a7);
            const uint64_t end = km::wall64() + (static_cast<uint64_t>(r.a0) << 10);
            for (uint32_t k = 1; km::wall64() < end; k++) {
                if (!port.tell(k, me, r.a7)) {
                    st.w[XP_ST_GAVE] = 1;
                    break;
                }
                st.w[XP_ST_SENT] = st.w[XP_ST_SENT] + 1;
                if (port.told(m)) {
                    st.w[XP_ST_SEEN] = st.w[XP_ST_SEEN] + 1;
                }
                if ((k & 7) == 0) {
                    const uint32_t t0 = km::wall();
                    if (!port.call(XP_ASK_PLAIN, k, me, m)) {
                        st.w[XP_ST_GAVE] = 1;
                        break;
                    }
                    st.time(km::wall() - t0);
                    st.w[XP_ST_BAD] += m.w[0] != XP_ANSWER(k, me) ? 1 : 0;
                }
            }
            if (st.w[XP_ST_GAVE] == 0) {
                port.tell(XP_DONE, r.a7, st.w[XP_ST_SENT]);
                put_stats(r, r.a1, false);
            }
        }
    });
}
