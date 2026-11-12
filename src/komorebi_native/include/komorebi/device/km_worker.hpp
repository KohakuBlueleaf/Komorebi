// The resident worker: the runtime owns each kernel's loop, a program supplies the body
// (a switch on the segment). EXIT, or a wait that gave up, ends all three kernels.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_record.hpp"

#if !defined(COMPILE_FOR_TRISC)
#include "km_team.hpp"
#include "km_trace.hpp"

namespace km {

    class Reader : public Core {
    public:
        Reader() : trace_(args()) {
        }

        // Start-up, then body(record) for each record of this core until EXIT or a wait
        // gives up. body may call wait_stop() and the bounded buffer waits below.
        template <typename Body>
        void run(Body body) {
            if (!start()) {
                leave_others();
                put_byte(state(), kStEnds, kEndReader, kEndGaveUp);
                return;
            }
            while (const km_record_t* r = next()) {
                if (seg(*r) == KM_SEG_EXIT) {
                    put_byte(state(), kStEnds, kEndReader, kEndExit);
                    return;
                }
                body(*r);
            }
            put_byte(state(), kStEnds, kEndReader, kEndGaveUp);
        }

        // A wait only the stop word ends (bounded by the idle limit). Returns true on
        // stop.
        bool wait_stop(const km_record_t& r) const {
            const uint32_t code = poll(
                [] {
                    return false;
                },
                args().idle_limit
            );
            gave_up(kStReader, code, kWhereStall, slot(r), r.seq);
            return code == kGaveStop;
        }

        // Bounded cb_reserve_back for a segment's data. False: gave up (recorded).
        bool reserve(uint32_t cb, uint32_t pages) const {
            const uint32_t code = poll(
                [&] {
                    return cb_pages_reservable_at_back(cb, pages);
                },
                args().wait_limit
            );
            if (code != kGaveNone) {
                gave_up(kStReader, code, kWhereData, 0, 0);
                return false;
            }
            cb_reserve_back(cb, pages);
            return true;
        }

    private:
        // Clears ring, counters, state and stop words, loads the core table, checks in
        // (leaders report READY, one core sends the hello). False: a leader gave up.
        bool start() {
            const Args& a = args();
            for (uint32_t w = 0; w < KM_RING_SLOTS * kRecordWords; w++) {
                ctl(KM_CTL_RING)[w] = 0;
            }
            for (uint32_t j = 0; j < KM_RING_SLOTS; j++) {
                ctl(KM_CTL_DONE)[j * 4] = 0;
            }
            for (uint32_t w = 0; w < kStateWords; w++) {
                state()[w] = 0;
            }
            *ctl(KM_CTL_STOP) = 0;
            trace_.clear();
            const InterleavedAddrGen<true> table = {
                .bank_base_address = a.cores, .page_size = KM_CORES_MAX * 4
            };
            noc_async_read(
                get_noc_addr(0, table), a.ctl + KM_CTL_CORES, KM_CORES_MAX * 4
            );
            noc_async_read_barrier();
            invalidate_l1_cache();
            state()[kStMagic] = kStateMagic | a.tag << 8 | a.me;

            if (a.me != a.leader) {
                *ctl(kOutCheckin) = a.tag << 16;
                noc_async_write(
                    a.ctl + kOutCheckin,
                    core_noc(a.leader, a.ctl + KM_CTL_BOOT + 16 * a.rank),
                    4
                );
                noc_async_write_barrier();
            } else {
                volatile tt_l1_ptr uint32_t* boot = ctl(KM_CTL_BOOT);
                const uint32_t code = poll(
                    [&] {
                        for (uint32_t k = 0; k < a.members; k++) {
                            if ((boot[k * 4] >> 16) != a.tag) {
                                return false;
                            }
                        }
                        return true;
                    },
                    a.wait_limit
                );
                if (code != kGaveNone) {
                    gave_up(kStReader, code, kWhereCheckin, 0, 0);
                    return false;
                }
                *ctl(kOutReady) = KM_READY;
                noc_async_write(
                    a.ctl + kOutReady, disp_noc(a.ready_base + a.ready_slot * 16), 4
                );
                noc_async_write_barrier();
            }
            if (a.mailbox != 0) {
                ctl(kOutHello)[0] = 1;
                ctl(kOutHello)[1] = a.ctl;
                noc_async_write(a.ctl + kOutHello, disp_noc(a.mailbox), 8);
                noc_async_write_barrier();
            }
            return true;
        }

        // When compute and writer took the last record: the next one into KM_CTL_JOB
        // and a token to each. nullptr: a wait gave up (they get an EXIT record).
        const km_record_t* next() {
            const Args& a = args();
            const uint32_t code = poll(
                [] {
                    return (
                        cb_pages_reservable_at_back(kCbTokenCompute, 1) &&
                        cb_pages_reservable_at_back(kCbTokenWriter, 1)
                    );
                },
                a.idle_limit,
                a.wait_limit
            );
            if (code != kGaveNone) {
                gave_up(kStReader, code, kWhereToken, 0, 0);
                return nullptr;
            }
            cb_reserve_back(kCbTokenCompute, 1);
            cb_reserve_back(kCbTokenWriter, 1);

            const volatile km_record_t*
                ring = reinterpret_cast<const volatile km_record_t*>(ctl(KM_CTL_RING));
            volatile km_record_t* job = (
                reinterpret_cast<volatile km_record_t*>(ctl(KM_CTL_JOB))
            );
            volatile tt_l1_ptr uint32_t* last = state() + kStLast;
            const uint64_t t0 = wall64();
            const uint64_t end = t0 + (static_cast<uint64_t>(a.idle_limit) << 10);
            uint32_t passes = 0, gave = kGaveNone, seq = 0;
            for (;;) {
                invalidate_l1_cache();
                if (stop()) {
                    gave = kGaveStop;
                    break;
                }
                const Pass p = scan(ring, last, a.depth, a.me);
                passes++;
                if (p.pick >= 0 && !p.partial) {
                    copy_record(&ring[p.pick], job);
                    if (job->seq == p.pick_seq && KM_LANDED(job)) {
                        last[p.pick] = p.pick_seq;
                        seq = p.pick_seq;
                        break;
                    }
                }
                if ((passes & 15) == 0 && wall64() > end) {
                    gave = kGaveLimit;
                    break;
                }
            }
            const uint32_t t1 = wall();
            add_sat(state() + kStPasses, passes);
            add_sat(
                state() + kStNextCycles, static_cast<uint32_t>((wall64() - t0) >> 4)
            );
            if (gave != kGaveNone) {
                gave_up(kStReader, gave, kWhereNext, 0, 0);
                hand_off_exit();
                return nullptr;
            }
            trace_.put(seq, kTraceSeen, t1);
            trace_.put(seq, kTraceExtra, passes);
            add_sat(state() + kStRuns, 1);
            hand_off(seq);
            return const_cast<const km_record_t*>(job);
        }

        // An EXIT record with sequence 0 in KM_CTL_JOB, handed to compute and writer.
        void hand_off_exit() {
            volatile km_record_t* job = (
                reinterpret_cast<volatile km_record_t*>(ctl(KM_CTL_JOB))
            );
            job->seq = job->land1 = job->land2 = job->land3 = 0;
            job->hdr = make_hdr(KM_SEG_EXIT, 0, 0);
            hand_off(0);
        }

        void hand_off(uint32_t seq) {
            l1(get_write_ptr(kCbTokenCompute))[0] = seq;
            l1(get_write_ptr(kCbTokenWriter))[0] = seq;
            cb_push_back(kCbTokenCompute, 1);
            cb_push_back(kCbTokenWriter, 1);
        }

        // After a failed start-up: compute and writer get an EXIT record if they can.
        void leave_others() {
            const uint32_t code = poll(
                [] {
                    return (
                        cb_pages_reservable_at_back(kCbTokenCompute, 1) &&
                        cb_pages_reservable_at_back(kCbTokenWriter, 1)
                    );
                },
                args().wait_limit
            );
            if (code == kGaveNone) {
                cb_reserve_back(kCbTokenCompute, 1);
                cb_reserve_back(kCbTokenWriter, 1);
                hand_off_exit();
            }
        }

        Trace trace_;
    };

    class Writer : public Core {
    public:
        Writer() : trace_(args()) {
        }

        // Per record: body(record) says whether the core reports it; writes flushed,
        // complete() runs. EXIT is completed, then the writer leaves.
        template <typename Body>
        void run(Body body) {
            const Args& a = args();
            uint32_t end = kEndExit;
            for (;;) {
                const uint32_t code = poll(
                    [] {
                        return cb_pages_available_at_front(kCbTokenWriter, 1);
                    },
                    a.idle_limit,
                    a.wait_limit
                );
                if (code != kGaveNone) {
                    gave_up(kStWriter, code, kWhereWriter, 0, 0);
                    end = kEndGaveUp;
                    break;
                }
                invalidate_l1_cache();
                km_record_t r;
                copy_record(ctl(KM_CTL_JOB), &r);
                cb_pop_front(kCbTokenWriter, 1);
                if (seg(r) == KM_SEG_EXIT) {
                    if (r.seq == 0) {  // the reader gave up
                        end = kEndAbort;
                    } else if (complete(*this, r)) {
                        trace_.stamp(r.seq, kTraceSent);
                    }
                    break;
                }
                const bool report = body(r);
                noc_async_write_barrier();
                trace_.stamp(r.seq, kTraceDone);
                if (report && complete(*this, r)) {
                    trace_.stamp(r.seq, kTraceSent);
                }
                add_sat(state() + kStCompleted, 1);
            }
            trace_.flush(a.me);
            put_byte(state(), kStEnds, kEndWriter, end);
        }

        // Bounded cb_wait_front for a segment's data. False: gave up (recorded).
        bool wait_front(uint32_t cb, uint32_t pages) const {
            const uint32_t code = poll(
                [&] {
                    return cb_pages_available_at_front(cb, pages);
                },
                args().wait_limit
            );
            if (code != kGaveNone) {
                gave_up(kStWriter, code, kWhereData, 0, 0);
                return false;
            }
            cb_wait_front(cb, pages);
            return true;
        }

    private:
        Trace trace_;
    };

}  // namespace km

#else
#include "api/compute/cb_api.h"
#include "api/compute/common.h"

namespace km {

    // Compute: unpack waits for the token and passes its sequence to math and pack; all
    // three read KM_CTL_JOB and confirm by mailbox before the token is freed.
    class Compute {
    public:
        Compute() : ctl_(get_arg_val<uint32_t>(0)), me_(get_arg_val<uint32_t>(1)) {
        }

        uint32_t me() const {
            return me_;
        }

        uint32_t arg(uint32_t i) const {
            return get_arg_val<uint32_t>(kComputeArgs + i);
        }

        template <typename Body>
        void run(Body body) {
            volatile tt_l1_ptr uint32_t* st = l1(ctl_ + KM_CTL_STATE);
            uint32_t wrong = 0, end = kEndExit;
            for (;;) {
                cb_wait_front(kCbTokenCompute, 1);
                const uint32_t seq = ckernel::read_tile_value(kCbTokenCompute, 0, 0);
                invalidate_l1_cache();
                km_record_t r;
                copy_record(l1(ctl_ + KM_CTL_JOB), &r);
                wrong += (r.seq != seq || !KM_LANDED(&r)) && wrong < 255;
                MATH((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, seq)));
                PACK((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, seq)));
                UNPACK(((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId)));
                UNPACK(((void)ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)));
                cb_pop_front(kCbTokenCompute, 1);
                UNPACK((add_sat(st + kStSeen, 1)));
                UNPACK((put_byte(st, kStWrong, 0, wrong)));
                MATH((put_byte(st, kStWrong, 1, wrong)));
                PACK((put_byte(st, kStWrong, 2, wrong)));
                if (seg(r) == KM_SEG_EXIT) {
                    end = r.seq == 0 ? kEndAbort : kEndExit;
                    break;
                }
                body(r);
            }
            PACK((put_byte(st, kStEnds, kEndCompute, end)));
        }

    private:
        uint32_t ctl_;
        uint32_t me_;
    };

}  // namespace km
#endif
