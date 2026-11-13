// Command side (README "What runs"): the landing kernel, protocol A only; at its end it
// raises the commands-done word for the broadcast kernel of the same core.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttx_dispatch.hpp"
#include "landing.h"
#include "broadcast.h"

namespace {

    // Timed waits: a poll is a few tens of cycles, so about one to two seconds.
    constexpr uint32_t kSpinLimit = 1u << 26;
    constexpr uint32_t kHealLimit = 1u << 20;

    volatile tt_l1_ptr uint32_t* words(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

    // Index of the first payload word of `rec` that is not command seq's, or 16 if all
    // match. The echo words are not payload.
    uint32_t first_bad(volatile tt_l1_ptr uint32_t* rec, uint32_t seq, uint32_t tag) {
        for (uint32_t w = 0; w < 16; w++) {
            if ((w & 3u) == 0) {
                continue;
            }
            if (rec[w] != landing_word(seq, w, tag)) {
                return w;
            }
        }
        return 16;
    }

}  // namespace

void kernel_main() {
    uint32_t a = 0;
    const uint32_t cnt_dram = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const auto disp = ttx::Args::read(a);
    const uint32_t tag = disp.tag;
    const uint32_t depth = disp.depth;

    // L1 survives processes: clear the command slots before the check-in, so nothing
    // from an earlier run is accepted.
    volatile tt_l1_ptr uint32_t* ctl = words(get_write_ptr(ttx::kCtlCb));
    for (uint32_t w = 0; w < LANDING_SLOTS * LANDING_SLOT_BYTES / 4; w++) {
        ctl[w] = 0;
    }

    uint32_t accepted = 0, torn = 0, early = 0, torn_seq = 0, torn_word = 0;
    uint32_t torn_value = 0, healed = 0, heal_polls = 0, lost = 0, lost_seq = 0;
    uint32_t last_seq = 0;
    uint64_t idle = 0;

    ttx::Worker worker(disp);
    worker.start();

    for (uint32_t seq = 1; seq <= n; seq++) {
        const uint32_t slot = depth > 1 ? seq % depth : 0;
        volatile tt_l1_ptr uint32_t* rec = ctl + slot * (LANDING_SLOT_BYTES / 4);
        const uint32_t echo = landing_echo(seq, tag);
        bool ok = false;
        for (uint32_t spins = 0; spins < kSpinLimit; spins++) {
            invalidate_l1_cache();
            const uint32_t m =
                ((rec[0] == echo) + (rec[4] == echo) + (rec[8] == echo) +
                 (rec[12] == echo));
            if (m == 4) {
                ok = true;
                break;
            }
            if (m != 0) {
                early++;
                continue;
            }
            idle++;
        }
        if (!ok) {
            // Never accepted: count it and complete the last seq at once, so the
            // leaders and the X280 finish instead of waiting for this core.
            lost++;
            lost_seq = seq;
            worker.complete(n);
            break;
        }
        const uint32_t bad = first_bad(rec, seq, tag);
        if (bad < 16) {
            if (torn == 0) {
                torn_seq = seq;
                torn_word = bad;
                torn_value = rec[bad];
            }
            torn++;
            // Diagnosis only (the command stays counted as torn): does it verify later?
            for (uint32_t k = 1; k <= kHealLimit; k++) {
                invalidate_l1_cache();
                if (first_bad(rec, seq, tag) == 16) {
                    healed++;
                    heal_polls = k > heal_polls ? k : heal_polls;
                    break;
                }
            }
        }
        accepted++;
        if (!worker.complete(seq)) {
            lost++;
            lost_seq = seq;
            break;
        }
        last_seq = seq;
    }

    // The broadcast kernel of this core may stop its broadcasts now.
    *words(get_write_ptr(kShareCb)) = broadcast_commands_done(tag);

    volatile tt_l1_ptr uint32_t* c = words(get_write_ptr(kRecCntCb));
    for (uint32_t w = 0; w < LC_WORDS; w++) {
        c[w] = 0;
    }
    c[LC_ACCEPTED] = accepted;
    c[LC_TORN] = torn;
    c[LC_EARLY] = early;
    c[LC_IDLE_LO] = static_cast<uint32_t>(idle);
    c[LC_IDLE_HI] = static_cast<uint32_t>(idle >> 32);
    c[LC_TORN_SEQ] = torn_seq;
    c[LC_TORN_WORD] = torn_word;
    c[LC_TORN_VALUE] = torn_value;
    c[LC_HEALED] = healed;
    c[LC_HEAL_POLLS] = heal_polls;
    c[LC_LOST] = lost;
    c[LC_LOST_SEQ] = lost_seq;
    c[LC_TAG] = tag;
    c[LC_LAST_SEQ] = last_seq;
    c[LC_MAGIC] = LANDING_MAGIC | me;
    const InterleavedAddrGen<true> dst = {
        .bank_base_address = cnt_dram, .page_size = LC_WORDS * 4
    };
    noc_async_write(get_write_ptr(kRecCntCb), get_noc_addr(me, dst), LC_WORDS * 4);
    noc_async_write_barrier();
}
