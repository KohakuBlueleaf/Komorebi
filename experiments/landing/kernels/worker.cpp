// Landing kernel on every core: for sequence = 1..n it waits for command sequence
// (protocol A, or B with PROTO_B), checks its payload, counts, completes it.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttx_dispatch.hpp"
#include "landing.h"

namespace {

    // Timed waits: a poll is a few tens of cycles, so about one to two seconds.
    constexpr uint32_t kSpinLimit = 1u << 26;
    constexpr uint32_t kHealLimit = 1u << 20;
    constexpr uint32_t kCntCb = 9;

    volatile tt_l1_ptr uint32_t* words(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

    inline uint32_t wall() {
        return (
            *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_L
            )
        );
    }

    // Index of the first payload word of `rec` that is not command sequence's, or 16 if
    // all match. Protocol A's echo words are not payload.
    uint32_t first_bad(
        volatile tt_l1_ptr uint32_t* rec, uint32_t sequence, uint32_t tag
    ) {
        for (uint32_t w = 0; w < 16; w++) {
#ifndef PROTO_B
            if ((w & 3u) == 0) {
                continue;
            }
#endif
            if (rec[w] != landing_word(sequence, w, tag)) {
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

    // L1 survives processes: clear the command slots and the sequence word before the
    // check-in, so nothing from an earlier run is accepted.
    volatile tt_l1_ptr uint32_t* control = words(get_write_ptr(ttx::kControlCb));
    for (uint32_t w = 0; w < LANDING_SLOTS * LANDING_SLOT_BYTES / 4; w++) {
        control[w] = 0;
    }
    volatile tt_l1_ptr uint32_t* sequence_word = control + LANDING_SEQUENCE_OFF / 4;
    *sequence_word = 0;

    uint32_t accepted = 0, torn = 0, early = 0, torn_sequence = 0, torn_word = 0;
    uint32_t torn_value = 0, healed = 0, heal_polls = 0, lost = 0, lost_sequence = 0;
    uint32_t check_max = 0, check_min = 0xFFFFFFFFu, checks = 0, last_sequence = 0;
    uint64_t idle = 0, check_sum = 0;
    const uint32_t s0 = wall();
    const uint32_t marker_cost = wall() - s0;

    ttx::Worker worker(disp);
    worker.start();

    for (uint32_t sequence = 1; sequence <= n; sequence++) {
        const uint32_t slot = depth > 1 ? sequence % depth : 0;
        volatile tt_l1_ptr uint32_t* rec = control + slot * (LANDING_SLOT_BYTES / 4);
        [[maybe_unused]] const uint32_t echo = landing_echo(sequence, tag);
        bool ok = false;
        for (uint32_t spins = 0; spins < kSpinLimit; spins++) {
#ifdef TIMESTAMP
            const uint32_t t0 = wall();
#endif
            invalidate_l1_cache();
#ifdef PROTO_B
            const uint32_t v = *sequence_word;
            ok = (
                (v >> 16) == tag &&
                static_cast<int16_t>(
                    static_cast<uint16_t>(v) - static_cast<uint16_t>(sequence)
                ) >= 0
            );
#else
            const uint32_t m = (
                (rec[0] == echo) + (rec[4] == echo) + (rec[8] == echo) +
                (rec[12] == echo)
            );
            ok = m == 4;
#endif
            if (ok) {
#ifdef TIMESTAMP
                const uint32_t dt = wall() - t0;
                check_sum += dt;
                check_max = dt > check_max ? dt : check_max;
                check_min = dt < check_min ? dt : check_min;
                checks++;
#endif
                break;
            }
#ifndef PROTO_B
            if (m != 0) {
                early++;
                continue;
            }
#endif
            idle++;
        }
        if (!ok) {
            // Never accepted: count it and complete the last sequence at once, so the
            // leaders and the X280 finish instead of waiting for this core.
            lost++;
            lost_sequence = sequence;
            worker.complete(n);
            break;
        }
        const uint32_t bad = first_bad(rec, sequence, tag);
        if (bad < 16) {
            if (torn == 0) {
                torn_sequence = sequence;
                torn_word = bad;
                torn_value = rec[bad];
            }
            torn++;
            // Diagnosis only (the command stays counted as torn): does it verify later?
            for (uint32_t k = 1; k <= kHealLimit; k++) {
                invalidate_l1_cache();
                if (first_bad(rec, sequence, tag) == 16) {
                    healed++;
                    heal_polls = k > heal_polls ? k : heal_polls;
                    break;
                }
            }
        }
        accepted++;
        if (!worker.complete(sequence)) {
            lost++;
            lost_sequence = sequence;
            break;
        }
        last_sequence = sequence;
    }

    volatile tt_l1_ptr uint32_t* c = words(get_write_ptr(kCntCb));
    for (uint32_t w = 0; w < LC_WORDS; w++) {
        c[w] = 0;
    }
    c[LC_ACCEPTED] = accepted;
    c[LC_TORN] = torn;
    c[LC_EARLY] = early;
    c[LC_IDLE_LO] = static_cast<uint32_t>(idle);
    c[LC_IDLE_HI] = static_cast<uint32_t>(idle >> 32);
    c[LC_CHECK_LO] = static_cast<uint32_t>(check_sum);
    c[LC_CHECK_HI] = static_cast<uint32_t>(check_sum >> 32);
    c[LC_CHECK_MAX] = check_max;
    c[LC_CHECK_MIN] = checks ? check_min : 0;
    c[LC_CHECKS] = checks;
    c[LC_TORN_SEQUENCE] = torn_sequence;
    c[LC_TORN_WORD] = torn_word;
    c[LC_TORN_VALUE] = torn_value;
    c[LC_HEALED] = healed;
    c[LC_HEAL_POLLS] = heal_polls;
    c[LC_LOST] = lost;
    c[LC_LOST_SEQUENCE] = lost_sequence;
    c[LC_MARKER_COST] = marker_cost;
    c[LC_TAG] = tag;
    c[LC_LAST_SEQUENCE] = last_sequence;
    c[LC_MAGIC] = LANDING_MAGIC | me;
    const InterleavedAddrGen<true> dst = {
        .bank_base_address = cnt_dram, .page_size = LC_WORDS * 4
    };
    noc_async_write(get_write_ptr(kCntCb), get_noc_addr(me, dst), LC_WORDS * 4);
    noc_async_write_barrier();
}
