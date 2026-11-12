// Stand-in Tensix worker for the dispatch kernel tests (one data-movement kernel per
// core): only what the kernel needs. Every wait is bounded; counters go to page `me`.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../../../../src/komorebi_native/include/komorebi/abi/km_abi.h"
#include "dk_test.h"

#ifdef DK_COMPILE_ERROR
#error "DK_COMPILE_ERROR: the compile-error test of the harness"
#endif

namespace {

    constexpr uint32_t kCtlCb = 8;
    constexpr uint32_t kOutCb = 9;
    constexpr uint32_t kPollEvery = 256;

    // Runtime arguments (host module: Plan::worker_args).
    struct Args {
        uint32_t me, cores, tag, l2_x, l2_y, mailbox, ready, ack_base, leader_x,
            leader_y, rank, members, is_leader, is_hello, table, out, limit, depth;
    };

    struct Worker {
        Args a;
        uint32_t ctl;
        uint64_t limit;  // cycles
        volatile tt_l1_ptr uint32_t* state;
        uint32_t cnt[DK_WORDS];

        volatile tt_l1_ptr uint32_t* at(uint32_t off) const {
            return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + off);
        }

        static uint64_t now() {
            const uint32_t lo = (
                *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                    RISCV_DEBUG_REG_WALL_CLOCK_L
                )
            );
            const uint32_t hi = (
                *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                    RISCV_DEBUG_REG_WALL_CLOCK_H
                )
            );
            return (static_cast<uint64_t>(hi) << 32) | lo;
        }

        bool stopped() const {
            return *at(KM_CTL_STOP) != 0;
        }

        // Records why and where a wait gave up.
        void gave_up(uint32_t code, uint32_t where) {
            state[DK_S_GAVE] = code;
            state[DK_S_WHERE] = where;
        }

        // Every kPollEvery spins: false (with the gave-up record) once the stop word is
        // set or `by` has passed.
        bool keep_waiting(uint32_t& spins, uint64_t by, uint32_t where) {
            if (++spins != kPollEvery) {
                return true;
            }
            spins = 0;
            invalidate_l1_cache();
            if (stopped()) {
                gave_up(DK_GAVE_STOP, where);
                return false;
            }
            if (now() > by) {
                gave_up(DK_GAVE_LIMIT, where);
                return false;
            }
            return true;
        }

        // Writes the 4-byte word `v` to NoC address `dst` (16-byte aligned) and waits
        // for it.
        void send_word(uint64_t dst, uint32_t v) {
            *at(KM_CTL_OUT) = v;
            noc_async_write(ctl + KM_CTL_OUT, dst, 4);
            noc_async_write_barrier();
        }

        // Clears the ring, stop word, counters and state, loads the core table, checks
        // in.
        bool start() {
            for (uint32_t off = KM_CTL_RING; off < KM_CTL_RSVD; off += 4) {
                *at(off) = 0;
            }
            *at(KM_CTL_STOP) = 0;
            for (uint32_t j = 0; j < KM_RING_SLOTS; j++) {
                *at(KM_CTL_DONE + 16 * j) = 0;
            }
            for (uint32_t w = 0; w <= DK_S_SEQ; w++) {
                state[w] = 0;
            }
            const InterleavedAddrGen<true> table = {
                .bank_base_address = a.table, .page_size = KM_CORES_MAX * 4
            };
            noc_async_read(
                get_noc_addr(0, table), ctl + KM_CTL_CORES, KM_CORES_MAX * 4
            );
            noc_async_read_barrier();
            if (!a.is_leader) {
                send_word(
                    get_noc_addr(
                        a.leader_x, a.leader_y, ctl + KM_CTL_BOOT + 16 * a.rank
                    ),
                    a.tag << 16
                );
            } else {
                const uint64_t by = now() + limit;
                uint32_t spins = 0;
                for (uint32_t r = 0; r < a.members;) {
                    invalidate_l1_cache();
                    if ((*at(KM_CTL_BOOT + 16 * r) >> 16) == a.tag) {
                        r++;
                    } else if (!keep_waiting(spins, by, DK_WHERE_BOOT)) {
                        return false;
                    }
                }
                send_word(get_noc_addr(a.l2_x, a.l2_y, a.ready), KM_READY);
            }
            if (a.is_hello) {
                *at(KM_CTL_OUT) = 1;
                *at(KM_CTL_OUT + 4) = ctl;
                noc_async_write(
                    ctl + KM_CTL_OUT, get_noc_addr(a.l2_x, a.l2_y, a.mailbox), 8
                );
                noc_async_write_barrier();
            }
            return true;
        }

        static bool landed(volatile tt_l1_ptr km_record_t* r) {
            return KM_LANDED(r);
        }

        bool mine(uint32_t team) const {
            const uint32_t first = team & 0xFFFFu, count = team >> 16;
            return a.me >= first && a.me < first + count;
        }

        // Takes into KM_CTL_JOB the oldest new landed record that is this core's or
        // EXIT (others are consumed). False if the stop word or limit ends the wait.
        bool next(uint32_t& slot) {
            const uint64_t by = now() + limit;
            uint32_t spins = 0;
            for (;;) {
                invalidate_l1_cache();
                uint32_t best = KM_RING_SLOTS, best_seq = 0;
                for (uint32_t j = 0; j < a.depth; j++) {
                    auto* r = (
                        reinterpret_cast<volatile tt_l1_ptr km_record_t*>(
                            ctl + KM_CTL_RING + 64 * j
                        )
                    );
                    const uint32_t s = r->seq;
                    if (static_cast<int32_t>(s - state[j]) <= 0 || !landed(r)) {
                        continue;
                    }
                    if ((r->hdr & 0xFFu) != KM_SEG_EXIT && !mine(r->team)) {
                        state[j] = s;
                        cnt[DK_C_IGNORED]++;
                        continue;
                    }
                    if (best == KM_RING_SLOTS ||
                        static_cast<int32_t>(s - best_seq) < 0) {
                        best = j;
                        best_seq = s;
                    }
                }
                if (best != KM_RING_SLOTS) {
                    auto* r = at(KM_CTL_RING + 64 * best);
                    auto* job = at(KM_CTL_JOB);
                    for (uint32_t w = 0; w < 16; w++) {
                        job[w] = r[w];
                    }
                    auto* copy = reinterpret_cast<volatile tt_l1_ptr km_record_t*>(job);
                    if (landed(copy) && copy->seq == best_seq) {
                        state[best] = best_seq;
                        state[DK_S_SEQ] = best_seq;
                        slot = best;
                        return true;
                    }
                }
                if (stopped()) {
                    gave_up(DK_GAVE_STOP, DK_WHERE_NEXT);
                    return false;
                }
                if (!keep_waiting(spins, by, DK_WHERE_NEXT)) {
                    return false;
                }
            }
        }

        // Waits `cycles` Tensix cycles, or until the stop word.
        bool delay(uint32_t cycles) {
            const uint64_t until = now() + cycles;
            const uint64_t by = until + limit;
            uint32_t spins = 0;
            while (now() < until) {
                if (!keep_waiting(spins, by, DK_WHERE_DELAY)) {
                    return false;
                }
            }
            return true;
        }

        // Waits until the stop word is set (or the limit runs out).
        void stall() {
            const uint64_t by = now() + limit;
            uint32_t spins = 0;
            while (keep_waiting(spins, by, DK_WHERE_STALL)) {
                ;
            }
        }

        // Runs the segment of the record in KM_CTL_JOB. Returns false if a wait ended
        // on the stop word or its limit.
        bool run(const volatile tt_l1_ptr km_record_t* r, uint32_t slot) {
            const uint32_t seg = r->hdr & 0xFFu;
            if (((r->hdr >> 16) & 0xFFu) != slot) {
                cnt[DK_C_HDR_ERRORS]++;
            }
            cnt[DK_C_RUN]++;
            cnt[DK_C_KEY_SUM] += r->a0;
            cnt[DK_C_KEY_MIX] ^= dk_mix(r->a0);
            if (seg < 4) {
                cnt[DK_C_SEG + seg]++;
            }
            const uint32_t args[10] = {
                r->a0, r->a1, r->a2, r->a3, r->a4, r->a5, r->a6, r->a7, r->a8, r->a9
            };
            switch (seg) {
                case KM_SEG_NOP:
                    return true;
                case DK_SEG_CHECK:
                    for (uint32_t w = 2; w < 10; w++) {
                        if (args[w] != dk_arg(args[0], args[1], w)) {
                            cnt[DK_C_ARG_ERRORS]++;
                        }
                    }
                    return true;
                case DK_SEG_DELAY:
                    return delay(args[1]);
                case DK_SEG_STALL:
                    if (args[1] == a.me) {
                        stall();
                        return false;
                    }
                    return true;
                default:
                    cnt[DK_C_HDR_ERRORS]++;
                    return true;
            }
        }

        // Reports completion: members increment the leader's counter of the slot; the
        // leader waits for count - 1 of them, then pushes the ack word to the X280.
        bool complete(const volatile tt_l1_ptr km_record_t* r, uint32_t slot) {
            const uint32_t first = r->team & 0xFFFFu, count = r->team >> 16;
            if (a.me != first) {
                const uint32_t xy = *at(KM_CTL_CORES + 4 * first);
                noc_semaphore_inc(
                    get_noc_addr(xy & 0xFFFFu, xy >> 16, ctl + KM_CTL_DONE + 16 * slot),
                    1
                );
                noc_async_atomic_barrier();
                cnt[DK_C_INCS]++;
                return true;
            }
            volatile tt_l1_ptr uint32_t* done = at(KM_CTL_DONE + 16 * slot);
            const uint64_t by = now() + limit;
            uint32_t spins = 0;
            for (;;) {
                invalidate_l1_cache();
                if (*done == count - 1) {
                    break;
                }
                if (stopped()) {
                    gave_up(DK_GAVE_STOP, DK_WHERE_DONE);
                    return false;
                }
                if (!keep_waiting(spins, by, DK_WHERE_DONE)) {
                    return false;
                }
            }
            *done = 0;
            send_word(
                get_noc_addr(a.l2_x, a.l2_y, a.ack_base + slot * KM_ACK_STRIDE),
                KM_ACK(r->seq)
            );
            cnt[DK_C_ACKS]++;
            return true;
        }

        // Counters and state words to page `me` of the output tensor.
        void flush() {
            cnt[DK_C_MAGIC] = DK_MAGIC | a.me;
            cnt[DK_C_TAG] = a.tag;
            for (uint32_t w = 0; w <= DK_S_SEQ; w++) {
                cnt[DK_C_STATE + w] = state[w];
            }
            volatile tt_l1_ptr uint32_t*
                page = (
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_write_ptr(kOutCb)
                    )
                );
            for (uint32_t w = 0; w < DK_WORDS; w++) {
                page[w] = cnt[w];
            }
            const InterleavedAddrGen<true> out = {
                .bank_base_address = a.out, .page_size = DK_WORDS * 4
            };
            noc_async_write(
                get_write_ptr(kOutCb), get_noc_addr(a.me, out), DK_WORDS * 4
            );
            noc_async_write_barrier();
        }
    };

}  // namespace

void kernel_main() {
    Worker k;
    uint32_t* f = &k.a.me;
    for (uint32_t i = 0; i < sizeof(Args) / 4; i++) {
        f[i] = get_arg_val<uint32_t>(i);
    }
    k.ctl = get_write_ptr(kCtlCb);
    k.limit = static_cast<uint64_t>(k.a.limit) << 20;
    k.state = k.at(KM_CTL_STATE);
    for (uint32_t w = 0; w < DK_WORDS; w++) {
        k.cnt[w] = 0;
    }
    if (k.start()) {
        uint32_t slot = 0;
        while (k.next(slot)) {
            auto* r = (
                reinterpret_cast<volatile tt_l1_ptr km_record_t*>(k.at(KM_CTL_JOB))
            );
            const bool exit = (r->hdr & 0xFFu) == KM_SEG_EXIT;
            if (exit) {
                k.cnt[DK_C_EXIT] = r->seq;
            } else if (!k.run(r, slot)) {
                break;
            }
            if (!k.complete(r, slot)) {
                break;
            }
            if (exit) {
                k.gave_up(0, DK_WHERE_EXIT);
                break;
            }
        }
    }
    k.flush();
}
