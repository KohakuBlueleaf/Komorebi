// The persistent kernel: the runtime owns each kernel's loop, a program supplies the
// body (a switch on the handler). EXIT, or a wait that gave up, ends all three kernels.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_command.hpp"

#if !defined(COMPILE_FOR_TRISC)
#include "km_completion.hpp"
#include "km_trace.hpp"

namespace km {

    // Word of each DONE slot: the last sequence whose reader body returned in that
    // slot (a broadcast's finish() included), read by the writer before it reports.
    constexpr uint32_t kDoneReader = 2;

    class Reader : public Core {
    public:
        Reader() : trace_(args()) {
        }

        // Start-up, then body(command) for each command of this core until EXIT or a
        // wait gives up. body may call wait_stop() and the timed buffer waits below.
        template <typename Body>
        void run(Body body) {
            if (!start()) {
                leave_others();
                put_byte(state(), kStEnds, kEndReader, kEndGaveUp);
                return;
            }
            while (const km_command_t* command = next()) {
                if (handler(*command) == KM_HANDLER_EXIT) {
                    put_byte(state(), kStEnds, kEndReader, kEndExit);
                    return;
                }
                body(*command);
                control(
                    KM_CONTROL_DONE + slot(*command) * 16
                )[kDoneReader] = command->sequence;
            }
            put_byte(state(), kStEnds, kEndReader, kEndGaveUp);
        }

        // A wait only the stop flag ends (the idle limit its timeout). Returns true on
        // stop.
        bool wait_stop(const km_command_t& command) const {
            const uint32_t code = poll(
                [] {
                    return false;
                },
                args().idle_limit
            );
            gave_up(kStReader, code, kWhereStall, slot(command), command.sequence);
            return code == kGaveStop;
        }

        // Timed cb_reserve_back of a contiguous item: the buffer, every push and
        // every pop on it must be multiples of the item. False: gave up (recorded).
        bool reserve(uint32_t cb, uint32_t pages) const {
            if (!item_fits(cb, get_local_cb_interface(cb).fifo_wr_ptr, pages)) {
                gave_up(kStReader, kGaveBadItem, kWhereData, 0, 0);
                return false;
            }
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
        // Clears ring, counters, state and stop flags, loads the core table, checks in
        // (leaders report READY, one core sends the hello). False: a leader gave up.
        bool start() {
            const Args& arguments = args();
            for (uint32_t word = 0; word < KM_RING_SLOTS * kCommandWords; word++) {
                control(KM_CONTROL_RING)[word] = 0;
            }
            for (uint32_t slot_index = 0; slot_index < KM_RING_SLOTS; slot_index++) {
                control(KM_CONTROL_DONE)[slot_index * 4] = 0;
                control(KM_CONTROL_DONE)[slot_index * 4 + kDoneCompute] = 0;
                control(KM_CONTROL_DONE)[slot_index * 4 + kDoneReader] = 0;
            }
            for (uint32_t word = 0; word < kStateWords; word++) {
                state()[word] = 0;
            }
            *control(KM_CONTROL_STOP) = 0;
            for (
                uint32_t word = 0; word < KM_BROADCASTS * KM_BROADCAST_BYTES / 4; word++
            ) {
                control(KM_CONTROL_BROADCAST)[word] = 0;
            }
            for (uint32_t word = 0; word < KM_PORTS * KM_PORT_STRIDE / 4; word++) {
                control(KM_CONTROL_PORT)[word] = 0;
            }
            trace_.clear();
            const InterleavedAddrGen<true> table = {
                .bank_base_address = arguments.cores, .page_size = KM_CORES_MAX * 4
            };
            noc_async_read(
                get_noc_addr(0, table),
                arguments.control + KM_CONTROL_CORES,
                KM_CORES_MAX * 4
            );
            noc_async_read_barrier();
            invalidate_l1_cache();
            state()[kStMagic] = kStateMagic | arguments.tag << 8 | arguments.me;

            if (arguments.me != arguments.leader) {
                *out_slot(arguments.control, kOutCheckin) = arguments.tag << 16;
                small_write<kOutCheckin, 4>(
                    arguments.control,
                    core_noc(arguments.leader, arguments.control + KM_CONTROL_BOOT),
                    arguments.rank
                );
                noc_async_write_barrier();
            } else {
                volatile tt_l1_ptr uint32_t* boot = control(KM_CONTROL_BOOT);
                const uint32_t code = poll(
                    [&] {
                        for (uint32_t peer = 0; peer < arguments.peers; peer++) {
                            if ((boot[peer * 4] >> 16) != arguments.tag) {
                                return false;
                            }
                        }
                        return true;
                    },
                    arguments.wait_limit
                );
                if (code != kGaveNone) {
                    gave_up(kStReader, code, kWhereCheckin, 0, 0);
                    return false;
                }
                *out_slot(arguments.control, kOutReady) = KM_READY;
                small_write<kOutReady, 4>(
                    arguments.control,
                    disp_noc(arguments.ready_base),
                    arguments.ready_slot
                );
                noc_async_write_barrier();
            }
            if (arguments.mailbox != 0) {
                out_slot(arguments.control, kOutHello)[0] = 1;
                out_slot(arguments.control, kOutHello)[1] = arguments.control;
                small_write<kOutHello, 8>(
                    arguments.control, disp_noc(arguments.mailbox), 0
                );
                noc_async_write_barrier();
            }
            return true;
        }

        // When compute and writer took the last command: the next one into
        // KM_CONTROL_JOB and a token to each. nullptr: a wait gave up (they get EXIT).
        const km_command_t* next() {
            const Args& arguments = args();
            const uint32_t code = poll(
                [] {
                    return (
                        cb_pages_reservable_at_back(kCbTokenCompute, 1) &&
                        cb_pages_reservable_at_back(kCbTokenWriter, 1)
                    );
                },
                arguments.idle_limit,
                arguments.wait_limit
            );
            if (code != kGaveNone) {
                gave_up(kStReader, code, kWhereToken, 0, 0);
                leave_others();
                return nullptr;
            }
            cb_reserve_back(kCbTokenCompute, 1);
            cb_reserve_back(kCbTokenWriter, 1);

            const volatile km_command_t*
                ring = (
                    reinterpret_cast<const volatile km_command_t*>(
                        control(KM_CONTROL_RING)
                    )
                );
            volatile km_command_t* job = (
                reinterpret_cast<volatile km_command_t*>(control(KM_CONTROL_JOB))
            );
            volatile tt_l1_ptr uint32_t* last = state() + kStLast;
            const uint64_t started = wall64();
            const uint64_t end = (
                arguments.idle_limit == 0
                    ? ~0ull
                    : started + (static_cast<uint64_t>(arguments.idle_limit) << 10)
            );
            uint32_t passes = 0, gave = kGaveNone, sequence = 0;
            for (;;) {
                invalidate_l1_cache();
                if (stop()) {
                    gave = kGaveStop;
                    break;
                }
                const Pass found = scan(ring, last, arguments.depth, arguments.me);
                passes++;
                if (found.pick >= 0 && !found.partial) {
                    copy_command(&ring[found.pick], job);
                    if (job->sequence == found.pick_sequence && KM_LANDED(job)) {
                        last[found.pick] = found.pick_sequence;
                        const km_command_t& got = *const_cast<const km_command_t*>(job);
                        // A command names its own slot; one found elsewhere is not run.
                        if (slot(got) != static_cast<uint32_t>(found.pick)) {
                            add_sat(state() + kStSlotWrong, 1);
                            continue;
                        }
                        // EXIT completes as a group of every worker, whatever it names.
                        if (handler(got) == KM_HANDLER_EXIT) {
                            job->group = make_group(0, arguments.workers);
                        }
                        sequence = found.pick_sequence;
                        break;
                    }
                }
                if ((passes & 15) == 0 && wall64() > end) {
                    gave = kGaveLimit;
                    break;
                }
            }
            const uint32_t taken = wall();
            add_sat(state() + kStPasses, passes);
            add_sat(
                state() + kStNextCycles,
                static_cast<uint32_t>((wall64() - started) >> 4)
            );
            if (gave != kGaveNone) {
                gave_up(kStReader, gave, kWhereNext, 0, 0);
                hand_off_exit();
                return nullptr;
            }
            trace_.put(sequence, kTraceSeen, taken);
            trace_.put(sequence, kTraceExtra, passes);
            add_sat(state() + kStRuns, 1);
            hand_off(sequence);
            return const_cast<const km_command_t*>(job);
        }

        // An EXIT command with sequence 0 in KM_CONTROL_JOB, handed to compute and
        // writer.
        void hand_off_exit() {
            volatile km_command_t* job = (
                reinterpret_cast<volatile km_command_t*>(control(KM_CONTROL_JOB))
            );
            job->sequence = job->land1 = job->land2 = job->land3 = 0;
            job->hdr = make_hdr(KM_HANDLER_EXIT, 0, 0);
            hand_off(0);
        }

        void hand_off(uint32_t sequence) {
            l1(get_write_ptr(kCbTokenCompute))[0] = sequence;
            l1(get_write_ptr(kCbTokenWriter))[0] = sequence;
            cb_push_back(kCbTokenCompute, 1);
            cb_push_back(kCbTokenWriter, 1);
        }

        // After a give-up: compute and writer get an EXIT command if its token pages
        // free up within the grace of a stop.
        void leave_others() {
            const uint32_t code = poll(
                [] {
                    return (
                        cb_pages_reservable_at_back(kCbTokenCompute, 1) &&
                        cb_pages_reservable_at_back(kCbTokenWriter, 1)
                    );
                },
                args().wait_limit,
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

        // Per command: body(command) says whether to report, writes are flushed,
        // compute and reader end their bodies, complete() runs; EXIT completes, leaves.
        template <typename Body>
        void run(Body body) {
            const Args& arguments = args();
            uint32_t end = kEndExit;
            for (;;) {
                const uint32_t code = poll(
                    [] {
                        return cb_pages_available_at_front(kCbTokenWriter, 1);
                    },
                    arguments.idle_limit,
                    arguments.wait_limit
                );
                if (code != kGaveNone) {
                    gave_up(kStWriter, code, kWhereWriter, 0, 0);
                    end = kEndGaveUp;
                    break;
                }
                invalidate_l1_cache();
                km_command_t command;
                copy_command(control(KM_CONTROL_JOB), &command);
                cb_pop_front(kCbTokenWriter, 1);
                if (handler(command) == KM_HANDLER_EXIT) {
                    if (command.sequence == 0) {  // the reader gave up
                        end = kEndAbort;
                    } else if (complete(*this, command)) {
                        trace_.timestamp(command.sequence, kTraceSent);
                    }
                    break;
                }
                result_ = 0;
                const bool report = body(command);
                noc_async_write_barrier();
                trace_.timestamp(command.sequence, kTraceDone);
                if (
                    report && bodies_done(command) && complete(*this, command, result_)
                ) {
                    trace_.timestamp(command.sequence, kTraceSent);
                }
                add_sat(state() + kStCompleted, 1);
            }
            noc_async_write_barrier();  // the last acknowledgement has landed
            trace_.flush(arguments.me);
            put_byte(state(), kStEnds, kEndWriter, end);
        }

        // The command's result word, sent with the leader's ack when the command has
        // KM_FLAG_RESULT (0 if never called). Elsewhere it is not sent.
        void result(uint32_t value) {
            result_ = value;
        }

        // Timed (`times` wait limits) cb_wait_front of a contiguous item: buffer,
        // pushes and pops are multiples of it. False: gave up (recorded).
        bool wait_front(uint32_t cb, uint32_t pages, uint32_t times = 1) const {
            if (!item_fits(cb, get_local_cb_interface(cb).fifo_rd_ptr, pages)) {
                gave_up(kStWriter, kGaveBadItem, kWhereData, 0, 0);
                return false;
            }
            const uint32_t code = poll(
                [&] {
                    return cb_pages_available_at_front(cb, pages);
                },
                args().wait_limit * times
            );
            if (code != kGaveNone) {
                gave_up(kStWriter, code, kWhereData, 0, 0);
                return false;
            }
            cb_wait_front(cb, pages);
            return true;
        }

    private:
        // Timed wait until compute (its body returned on all three threads) and the
        // reader marked command r's sequence in its slot. False: gave up (recorded).
        bool bodies_done(const km_command_t& command) const {
            const uint32_t slot_index = slot(command);
            volatile tt_l1_ptr uint32_t* mark = (
                control(KM_CONTROL_DONE + slot_index * 16)
            );
            const uint32_t code = poll(
                [&] {
                    return (
                        mark[kDoneCompute] == command.sequence &&
                        mark[kDoneReader] == command.sequence
                    );
                },
                args().wait_limit
            );
            if (code != kGaveNone) {
                gave_up(kStWriter, code, kWhereCDone, slot_index, command.sequence);
                return false;
            }
            return true;
        }

        Trace trace_;
        uint32_t result_ = 0;
    };

}  // namespace km

#else
namespace km {

    // Compute: unpack's timed wait for the token, then it tells math and pack by
    // mailbox; all three copy KM_CONTROL_JOB and confirm by mailbox before it is freed.
    class Compute {
    public:
        Compute() :
            control_(get_arg_val<uint32_t>(kCArgControl)),
            me_(get_arg_val<uint32_t>(kCArgMe)),
            wait_(get_arg_val<uint32_t>(kCArgWait)),
            idle_(get_arg_val<uint32_t>(kCArgIdle)) {
        }

        uint32_t me() const {
            return me_;
        }

        uint32_t arg(uint32_t index) const {
            return get_arg_val<uint32_t>(kComputeArgs + index);
        }

        // Timed cb_wait_front of a contiguous item (buffer, pushes, pops: multiples
        // of it), `times` wait limits. Unpack decides, math and pack get its answer.
        bool wait_front(uint32_t cb, uint32_t pages, uint32_t times = 1) const {
            uint32_t ok = 0;
            UNPACK(({
                volatile tt_l1_ptr uint32_t* got = get_cb_tiles_received_ptr(cb);
                const bool fits = (
                    item_fits(cb, get_local_cb_interface(cb).fifo_rd_ptr, pages)
                );
                const uint32_t code = !fits
                    ? kGaveBadItem
                    : poll_stop(
                        stop(),
                        [&] {
                            const uint16_t available = (
                                static_cast<uint16_t>(
                                    ckernel::reg_read(
                                        static_cast<uint32_t>(
                                            reinterpret_cast<uintptr_t>(got)
                                        )
                                    ) -
                                    get_local_cb_interface(cb).tiles_acked
                                )
                            );
                            return available >= pages;
                        },
                        wait_ * times,
                        0
                    );
                ok = code == kGaveNone;
                if (ok) {
                    cb_wait_front(cb, pages);
                } else {
                    command_gave_up(
                        control_, kStCompute, code, kWhereCInput, 0, sequence_
                    );
                }
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, ok);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, ok);
            }));
            MATH((ok = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
            PACK((ok = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
            return ok != 0;
        }

        // Timed cb_reserve_back of a contiguous item (buffer, pushes, pops: multiples
        // of it), decided by pack. False on pack: gave up (recorded); true elsewhere.
        bool reserve_back(uint32_t cb, uint32_t pages) const {
            bool ok = true;
            PACK(({
                volatile tt_reg_ptr uint32_t* acked = get_cb_tiles_acked_ptr(cb);
                const bool fits = (
                    item_fits(cb, get_local_cb_interface(cb).fifo_wr_ptr, pages)
                );
                const uint32_t code = !fits
                    ? kGaveBadItem
                    : poll_stop(
                        stop(),
                        [&] {
                            const uint16_t used = (
                                static_cast<uint16_t>(
                                    get_local_cb_interface(cb).tiles_received -
                                    ckernel::reg_read(
                                        static_cast<uint32_t>(
                                            reinterpret_cast<uintptr_t>(acked)
                                        )
                                    )
                                )
                            );
                            return (
                                get_local_cb_interface(cb).fifo_num_pages - used >=
                                pages
                            );
                        },
                        wait_,
                        0
                    );
                ok = code == kGaveNone;
                if (ok) {
                    cb_reserve_back(cb, pages);
                } else {
                    command_gave_up(
                        control_, kStCompute, code, kWhereCOutput, 0, sequence_
                    );
                }
            }));
            return ok;
        }

        template <typename Body>
        void run(Body body) {
            volatile tt_l1_ptr uint32_t* state_words = state();
            uint32_t wrong = 0, end = kEndExit;
            for (;;) {
                uint32_t sequence = 0;
                if (!take(sequence)) {
                    end = kEndGaveUp;
                    break;
                }
                sequence_ = sequence;
                invalidate_l1_cache();
                km_command_t command;
                copy_command(l1(control_ + KM_CONTROL_JOB), &command);
                wrong += (
                    (command.sequence != sequence || !KM_LANDED(&command)) &&
                    wrong < 255
                );
                MATH((
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, sequence)
                ));
                PACK((
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, sequence)
                ));
                UNPACK(((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId)));
                UNPACK(((void)ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)));
                cb_pop_front(kCbTokenCompute, 1);
                UNPACK((add_sat(state_words + kStSeen, 1)));
                UNPACK((put_byte(state_words, kStWrong, 0, wrong)));
                MATH((put_byte(state_words, kStWrong, 1, wrong)));
                PACK((put_byte(state_words, kStWrong, 2, wrong)));
                if (handler(command) == KM_HANDLER_EXIT) {
                    end = command.sequence == 0 ? kEndAbort : kEndExit;
                    break;
                }
                body(command);
                // Math and pack tell unpack they left the body; unpack marks it.
                MATH((
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, sequence)
                ));
                PACK((
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, sequence)
                ));
                UNPACK(({
                    (void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);
                    (void)ckernel::mailbox_read(ckernel::ThreadId::PackThreadId);
                    l1(
                        control_ + KM_CONTROL_DONE + slot(command) * 16
                    )[kDoneCompute] = sequence;
                }));
            }
            PACK((put_byte(state_words, kStEnds, kEndCompute, end)));
            (void)end;  // only pack writes the end byte
        }

    private:
        volatile tt_l1_ptr uint32_t* state() const {
            return l1(control_ + KM_CONTROL_STATE);
        }

        volatile tt_l1_ptr uint32_t* stop() const {
            return l1(control_ + KM_CONTROL_STOP);
        }

        // The next token (timed like the reader's wait for a command): unpack polls
        // and sends ok, then the sequence, to math and pack. False: gave up (recorded).
        bool take(uint32_t& sequence) const {
            uint32_t ok = 0;
            UNPACK(({
                volatile tt_l1_ptr uint32_t* got = (
                    get_cb_tiles_received_ptr(kCbTokenCompute)
                );
                const uint32_t code = poll_stop(
                    stop(),
                    [&] {
                        return (
                            static_cast<uint16_t>(
                                ckernel::reg_read(
                                    static_cast<uint32_t>(
                                        reinterpret_cast<uintptr_t>(got)
                                    )
                                ) -
                                get_local_cb_interface(kCbTokenCompute).tiles_acked
                            ) >= 1
                        );
                    },
                    idle_,
                    wait_
                );
                ok = code == kGaveNone;
                if (ok) {
                    cb_wait_front(kCbTokenCompute, 1);
                    sequence = (
                        l1(ckernel::get_tile_l1_byte_address(kCbTokenCompute, 0))[0]
                    );
                } else {
                    command_gave_up(control_, kStCompute, code, kWhereCToken, 0, 0);
                }
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, ok);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, ok);
                if (ok) {
                    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, sequence);
                    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, sequence);
                }
            }));
            MATH(({
                ok = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
                sequence = (
                    ok ? ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId) : 0
                );
            }));
            PACK(({
                ok = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
                sequence = (
                    ok ? ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId) : 0
                );
            }));
            return ok != 0;
        }

        uint32_t control_;
        uint32_t me_;
        uint32_t wait_;
        uint32_t idle_;
        uint32_t sequence_ = 0;  // the command being run, for give-up words
    };

}  // namespace km
#endif
