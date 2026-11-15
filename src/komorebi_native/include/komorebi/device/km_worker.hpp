// The persistent kernel: the runtime owns each kernel's loop, a program the body. EXIT
// or a wait that gave up ends all three kernels; the dispatcher's give-up, one command.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_command.hpp"

namespace km {

    // A command the runtime's service overlay serves (a transfer between DRAM and the
    // host channel, or a copy between DRAM pages), not a body: one range check.
    inline bool served(const km_command_t& command) {
        return (
            static_cast<uint32_t>(handler(command)) - KM_HANDLER_SERVED_FIRST <=
            KM_HANDLER_SERVED_LAST - KM_HANDLER_SERVED_FIRST
        );
    }

    // A handler added to the live run (run.add): one look at a word of the overlay
    // area; enter() loads and calls the overlay of its number (km_overlay.hpp).
    inline bool entered(uint32_t control, const km_command_t& command);
    template <typename Kernel>
    bool enter(Kernel& kernel, const km_command_t& command);

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
#include "km_completion.hpp"
#include "km_trace.hpp"
#if defined(KM_PAGE_LINKS)
#include "km_pages.hpp"
#include "km_relay.hpp"
#endif

namespace km {

    // A PAGES command, served on the link cores of a run built with KM_PAGE_LINKS.
    inline bool pages(const km_command_t& command) {
#if defined(KM_PAGE_LINKS)
        return handler(command) == KM_HANDLER_PAGES;
#else
        return static_cast<void>(command), false;
#endif
    }
#if !defined(KM_PAGE_LINKS)
    template <typename Kernel>
    inline bool pages_serve(const Kernel&, const km_command_t&, uint32_t, uint32_t) {
        return true;
    }

    template <typename Kernel>
    inline void pages_serve(const Kernel&, uint32_t, uint32_t) {
    }

    template <typename Kernel>
    inline void pages_end(const Kernel&) {
    }
#endif

    // A link core's services (the link cores' own kernels, built with
    // KM_LINK_SERVICES); a rank's kernel serves a PAGES command's wait alone.
    template <typename Kernel>
    inline bool link_serve(
        const Kernel& kernel, const km_command_t& command, uint32_t cpu, uint32_t word
    ) {
#if defined(KM_LINK_SERVICES) || !defined(KM_PAGE_LINKS)
        return pages_serve(kernel, command, cpu, word);
#else
        const PagesRequest r = pages_of(command);
        const uint32_t code = (
            r.mode != KM_PAGES_WAIT
                ? kGaveBadTransfer
                : cpu == 0
                    ? pages_wait(kernel, r)
                    : kGaveNone
        );
        if (code != kGaveNone) {
            kernel.gave_up(word, code, kWherePages, slot(command), command.sequence);
            return false;
        }
        return true;
#endif
    }

    // Between commands: a link core's services; nothing on a rank.
    template <typename Kernel>
    inline void link_serve(const Kernel& kernel, uint32_t cpu, uint32_t word) {
#if defined(KM_LINK_SERVICES) || !defined(KM_PAGE_LINKS)
        pages_serve(kernel, cpu, word);
#else
        static_cast<void>(kernel), static_cast<void>(cpu), static_cast<void>(word);
#endif
    }

    // At the end: a link core closes its connections; nothing on a rank.
    template <typename Kernel>
    inline void link_end(const Kernel& kernel) {
#if defined(KM_LINK_SERVICES) || !defined(KM_PAGE_LINKS)
        pages_end(kernel);
#else
        static_cast<void>(kernel);
#endif
    }

    // A command the runtime serves (served()): the reader loads the runtime's service
    // overlay on first use and calls it (km_overlay.hpp).
    template <typename Kernel>
    void serve(Kernel& reader, const km_command_t& command);

    // Word of each DONE slot: the last sequence whose reader body returned in that
    // slot (a broadcast's finish() included), read by the writer before it reports.
    constexpr uint32_t kDoneReader = 2;

    class Reader : public Core {
    public:
        Reader() : Core(kRoleReader), trace_(args()) {
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
                    link_end(*this);
                    put_byte(state(), kStEnds, kEndReader, kEndExit);
                    return;
                }
                begin(*command);
                if (pages(*command)) {
                    link_serve(*this, *command, 0, kStReader);
                } else if (served(*command)) {
                    serve(*this, *command);
                } else if (entered(args().control, *command)) {
                    enter(*this, *command);
                } else {
                    body(*command);
                }
                control(
                    KM_CONTROL_DONE + slot(*command) * 16
                )[kDoneReader] = command->sequence;
                finish(*command, kStReader);  // a failed clear stops the tile
            }
            link_end(*this);
            put_byte(state(), kStEnds, kEndReader, kEndGaveUp);
        }

        // The command's block (run.submit(block=)), in the reader's copy: once per
        // command.
        Block block(const km_command_t& command) const {
            return block_in(command, 0);
        }

        // A wait only the stop flag ends (the idle limit its timeout). Returns true on
        // stop.
        bool wait_stop(const km_command_t& command) const {
            const uint32_t code = poll_stop_at(
                kWhereStall,
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
            produce(cb);
            const uint32_t code = poll_at(
                kWhereData,
                [&] {
                    return cb_pages_reservable_at_back(cb, pages);
                },
                args().wait_limit
            );
            if (code != kGaveNone) {
                gave_up(
                    kStReader, code, kWhereData, running_slot(), running_sequence()
                );
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
            control(KM_CONTROL_BLOCK)[0] = 0;  // no command's block read in yet
            control(KM_CONTROL_BLOCK + KM_BLOCK_AREA_BYTES)[0] = 0;
            for (uint32_t slot_index = 0; slot_index < KM_RING_SLOTS; slot_index++) {
                control(KM_CONTROL_ABORT)[slot_index] = 0;
            }
            for (uint32_t word = kLvTrace; word < kLvWords; word++) {
                control(
                    KM_CONTROL_LIVE
                )[word] = 0;  // the reader's and the shared words
            }
            clear_live(arguments.control, role());
            control(KM_CONTROL_LIVE)[kLvTrace] = trace_.base();
            control(KM_CONTROL_LIVE)[kLvTraceCap] = trace_.base() ? arguments.trace_cap
                                                                  : 0;
            *control(KM_CONTROL_STOP) = 0;
            for (
                uint32_t word = 0; word < KM_BROADCASTS * KM_BROADCAST_BYTES / 4; word++
            ) {
                control(KM_CONTROL_BROADCAST)[word] = 0;
            }
            for (uint32_t word = 0; word < KM_PORT_SLOTS * KM_PORT_STRIDE / 4; word++) {
                control(KM_CONTROL_PORT)[word] = 0;
            }
            trace_.clear();
            const InterleavedAddrGen<true> table = {
                .bank_base_address = arguments.cores, .page_size = KM_CORES_MAX * 4
            };
            noc_async_read(
                table.get_noc_addr(0),
                arguments.control + KM_CONTROL_CORES,
                KM_CORES_MAX * 4
            );
            if (!flushed(kFlushReads, kStReader)) {
                return false;
            }
            state()[kStMagic] = kStateMagic | arguments.tag << 8 | arguments.me;

            if (arguments.me != arguments.leader) {
                *out_slot(arguments.control, kOutCheckin) = arguments.tag << 16;
                small_write<kOutCheckin, 4>(
                    arguments.control,
                    core_noc(arguments.leader, arguments.control + KM_CONTROL_BOOT),
                    arguments.rank
                );
                if (!flushed(kFlushWrites, kStReader)) {
                    return false;
                }
            } else {
                volatile tt_l1_ptr uint32_t* boot = control(KM_CONTROL_BOOT);
                const uint32_t code = poll_at(
                    kWhereCheckin,
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
                if (!flushed(kFlushWrites, kStReader)) {
                    return false;
                }
            }
            if (arguments.mailbox != 0) {
                out_slot(arguments.control, kOutHello)[0] = 1;
                out_slot(arguments.control, kOutHello)[1] = arguments.control;
                small_write<kOutHello, 8>(
                    arguments.control, disp_noc(arguments.mailbox), 0
                );
                return flushed(kFlushWrites, kStReader);
            }
            return true;
        }

        // When compute and writer took the last command: the next one into
        // KM_CONTROL_JOB and a token to each. nullptr: a wait gave up (they get EXIT).
        const km_command_t* next() {
            const Args& arguments = args();
            const uint32_t code = poll_at(
                kWhereToken,
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
            const Islands view = islands_at(arguments.control);
            const uint32_t island_first = view.first();
            const uint32_t link = (
                KM_GROUP_LINK_BASE + (arguments.me - view.base() - view.ranks())
            );
            // a link core is in no group of ranks: its name is its link name
            const bool serves = link_core(view, arguments.me);
            const uint32_t rank = serves ? link : view.offset() + arguments.me;
            control(KM_CONTROL_LIVE)[kLvWhere + kRoleReader] = kWhereNext;
            // a link core relays between commands (km_relay.hpp)
            const bool relays = link_core(view, arguments.me);
            for (;;) {
                if (serves) {  // a link core's services between commands
                    link_serve(*this, 0, kStReader);
                }
                invalidate_l1_cache();
                if (stop()) {
                    gave = kGaveStop;
                    break;
                }
#if defined(KM_LINK_SERVICES)
                if (relays) {
                    Relay::turn(*this);
                }
#else
                static_cast<void>(relays);
#endif
                const Pass found = scan(ring, last, arguments.depth, rank, link);
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
                            job->group = make_group(island_first, arguments.workers);
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
            control(KM_CONTROL_LIVE)[kLvWhere + kRoleReader] = 0;
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
        Writer() : Core(kRoleWriter), trace_(args()) {
            clear_live(args().control, kRoleWriter);
        }

        // Per command: body(command) says whether to report, writes are flushed,
        // compute and reader end their bodies, complete() runs; EXIT completes, leaves.
        template <typename Body>
        void run(Body body) {
            const Args& arguments = args();
            const bool serves = link_core(islands_at(arguments.control), arguments.me);
            uint32_t end = kEndExit;
            for (;;) {
                const uint32_t code = poll_at(
                    kWhereWriter,
                    [&] {
                        if (serves) {  // a link core's services between commands
                            link_serve(*this, 1, kStWriter);
#if defined(KM_LINK_SERVICES) && !defined(KM_WIRE)  // the wire's: link 0's alone
                            Relay::turn(*this);     // its link's relay asks, if any
#endif
                        }
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
                begin(command);
                result_ = 0;
                const bool report = (
                    pages(command)
                        ? link_serve(*this, command, 1, kStWriter)
                        : __builtin_expect(served(command), 0) ||
                            (entered(arguments.control, command) ? enter(*this, command)
                                                                 : body(command))
                );
                const bool written = flushed(kFlushWrites, kStWriter);
                trace_.timestamp(command.sequence, kTraceDone);
                if (
                    written && report && !given_up(command) && bodies_done(command) &&
                    complete(*this, command, result_)
                ) {
                    trace_.timestamp(command.sequence, kTraceSent);
                } else if (given_up(command)) {  // no report: the counts it got go
                    add_sat(control(KM_CONTROL_LIVE) + kLvGivenUp, 1);
                    control(KM_CONTROL_DONE + slot(command) * 16)[0] = 0;
                }
                add_sat(state() + kStCompleted, 1);
                finish(command, kStWriter);
            }
            link_end(*this);
            flushed(kFlushWrites, kStWriter);  // the last acknowledgement has landed
            trace_.flush(arguments.me, *this);
            put_byte(state(), kStEnds, kEndWriter, end);
        }

        // The command's result word, sent with the leader's ack when the command has
        // KM_FLAG_RESULT (0 if never called). Elsewhere it is not sent.
        void result(uint32_t value) {
            result_ = value;
        }

        // The command's block (run.submit(block=)), in the writer's copy: once per
        // command.
        Block block(const km_command_t& command) const {
            return block_in(command, 1);
        }

        // Timed (`times` wait limits) cb_wait_front of a contiguous item: buffer,
        // pushes and pops are multiples of it. False: gave up (recorded).
        bool wait_front(uint32_t cb, uint32_t pages, uint32_t times = 1) const {
            if (!item_fits(cb, get_local_cb_interface(cb).fifo_rd_ptr, pages)) {
                gave_up(kStWriter, kGaveBadItem, kWhereData, 0, 0);
                return false;
            }
            consume(cb);
            const uint32_t code = poll_at(
                kWhereData,
                [&] {
                    return cb_pages_available_at_front(cb, pages);
                },
                args().wait_limit * times
            );
            if (code != kGaveNone) {
                gave_up(
                    kStWriter, code, kWhereData, running_slot(), running_sequence()
                );
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
            const uint32_t code = poll_at(
                kWhereCDone,
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

    // Compute: unpack's timed token wait, told to math and pack by mailbox; all three
    // copy KM_CONTROL_JOB and confirm. Mailbox waits show live, bounded by unpack's.
    class Compute {
    public:
        Compute() :
            control_(get_arg_val<uint32_t>(kCArgControl)),
            me_(islands_at(control_).offset() + get_arg_val<uint32_t>(kCArgMe)),
            wait_(get_arg_val<uint32_t>(kCArgWait)),
            idle_(get_arg_val<uint32_t>(kCArgIdle)) {
            UNPACK((role_ = kRoleUnpack));
            MATH((role_ = kRoleMath));
            PACK((role_ = kRolePack));
            clear_live(control_, role_);
        }

        uint32_t me() const {
            return me_;
        }

        uint32_t arg(uint32_t index) const {
            return get_arg_val<uint32_t>(kComputeArgs + KM_PROGRAM_ARGS_AT + index);
        }

        // The control block's address and the wait limit (1024-cycle units).
        uint32_t control() const {
            return control_;
        }

        uint32_t wait_limit() const {
            return wait_;
        }

        // Timed cb_wait_front of a contiguous item (buffer, pushes, pops: multiples
        // of it), `times` wait limits. Unpack decides, math and pack get its answer.
        bool wait_front(uint32_t cb, uint32_t pages, uint32_t times = 1) const {
            uint32_t ok = 0;
            UNPACK(({
                consumes_ |= cb < kClearBuffers ? 1u << cb : 0;
                volatile tt_l1_ptr uint32_t* got = get_cb_tiles_received_ptr(cb);
                const bool fits = (
                    item_fits(cb, get_local_cb_interface(cb).fifo_rd_ptr, pages)
                );
                const uint32_t code = !fits
                    ? kGaveBadItem
                    : poll_live(
                        control_,
                        role_,
                        kWhereCInput,
                        true,
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
                        control_, kStCompute, code, kWhereCInput, slot_, sequence_
                    );
                }
                ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, ok);
                ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, ok);
            }));
            MATH((ok = from_unpack()));
            PACK((ok = from_unpack()));
            return ok != 0;
        }

        // A word from one compute CPU to another (an inline body's value), by
        // mailbox: `hand` on the sender, `handed` on the receiver (shown live).
        void hand(uint32_t thread, uint32_t word) const {
            ckernel::mailbox_write(thread, word);
        }

        uint32_t handed(uint32_t thread) const {
            return heard(thread, kWhereCMailbox);
        }

        // A give-up the command's words decided (each compute CPU sees the same
        // words and ends the command): recorded once, by unpack.
        void gave_up(uint32_t code, uint32_t where) const {
            UNPACK(
                (command_gave_up(control_, kStCompute, code, where, slot_, sequence_))
            );
        }

        // Timed cb_reserve_back of a contiguous item (buffer, pushes, pops: multiples
        // of it), decided by pack. False on pack: gave up (recorded); true elsewhere.
        bool reserve_back(uint32_t cb, uint32_t pages) const {
            bool ok = true;
            PACK(({
                if (cb < kClearBuffers && !((produces_ >> cb) & 1)) {
                    produces_ |= 1u << cb;
                    l1(
                        control_ + KM_CONTROL_LIVE
                    )[kLvProduces + kProducerPack] = (produces_);
                }
                volatile tt_reg_ptr uint32_t* acked = get_cb_tiles_acked_ptr(cb);
                const bool fits = (
                    item_fits(cb, get_local_cb_interface(cb).fifo_wr_ptr, pages)
                );
                const uint32_t code = !fits
                    ? kGaveBadItem
                    : poll_live(
                        control_,
                        role_,
                        kWhereCOutput,
                        true,
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
                        control_, kStCompute, code, kWhereCOutput, slot_, sequence_
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
                slot_ = slot(command);
                l1(control_ + KM_CONTROL_LIVE)[kLvSlot + role_] = slot_;
                l1(control_ + KM_CONTROL_LIVE)[kLvSequence + role_] = sequence;
                MATH((
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, sequence)
                ));
                PACK((
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, sequence)
                ));
                confirmed();  // math and pack copied the command
                cb_pop_front(kCbTokenCompute, 1);
                UNPACK((add_sat(state_words + kStSeen, 1)));
                UNPACK((put_byte(state_words, kStWrong, 0, wrong)));
                MATH((put_byte(state_words, kStWrong, 1, wrong)));
                PACK((put_byte(state_words, kStWrong, 2, wrong)));
                if (handler(command) == KM_HANDLER_EXIT) {
                    end = command.sequence == 0 ? kEndAbort : kEndExit;
                    break;
                }
                if (__builtin_expect(!served(command), 1)) {
                    if (entered(control_, command)) {
                        enter(*this, command);
                    } else {
                        body(command);
                    }
                }
                // Math and pack tell unpack they left the body; unpack marks it.
                MATH((
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, sequence)
                ));
                PACK((
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, sequence)
                ));
                confirmed();
                UNPACK((
                    l1(control_ + KM_CONTROL_DONE + slot_ * 16)[kDoneCompute] = sequence
                ));
                if (!finish()) {
                    end = kEndGaveUp;
                    break;
                }
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

        // Math and pack: unpack's next word, shown live as kWhereCMailbox.
        uint32_t from_unpack() const {
            return heard(ckernel::ThreadId::UnpackThreadId, kWhereCMailbox);
        }

        // Unpack: math's and pack's words (each sends its sequence).
        void confirmed() const {
            UNPACK(({
                (void)heard(ckernel::ThreadId::MathThreadId, kWhereCConfirm);
                (void)heard(ckernel::ThreadId::PackThreadId, kWhereCConfirm);
            }));
        }

        uint32_t heard(uint32_t thread, uint32_t where) const {
            volatile tt_l1_ptr uint32_t*
                live = (l1(control_ + KM_CONTROL_LIVE) + kLvWhere + role_);
            *live = where;
            const uint32_t word = ckernel::mailbox_read(thread);
            *live = 0;
            return word;
        }

        // The command's end: pack's page counts, the commands left; after a give-up,
        // unpack pops the pages left in its input buffers.
        bool finish() {
            index_++;
            bool ok = true;
            PACK((end_command(
                control_, role_, index_, produces_, kProducerPack, [](uint32_t cb) {
                    return get_local_cb_interface(cb).tiles_received;
                }
            )));
            MATH((end_command(control_, role_, index_, 0, kProducerPack, [](uint32_t) {
                return 0u;
            })));
            UNPACK(({
                end_command(control_, role_, index_, 0, kProducerPack, [](uint32_t) {
                    return 0u;
                });
                if (consumes_ != 0 && given_up(control_, slot_, sequence_)) {
                    const uint32_t code = clear_buffers(
                        control_,
                        role_,
                        index_,
                        consumes_,
                        wait_,
                        [](uint32_t cb) {
                            return (
                                static_cast<uint16_t>(
                                    get_local_cb_interface(cb).tiles_acked
                                )
                            );
                        },
                        [](uint32_t cb) {
                            cb_pop_front(cb, 1);
                        }
                    );
                    if (code != kGaveNone) {
                        command_gave_up(
                            control_, kStCompute, code, kWhereClear, slot_, sequence_
                        );
                        ok = false;
                    }
                }
            }));
            return ok;
        }

        // The next token (timed like the reader's wait for a command): unpack polls
        // and sends ok, then the sequence, to math and pack. False: gave up (recorded).
        bool take(uint32_t& sequence) const {
            uint32_t ok = 0;
            UNPACK(({
                volatile tt_l1_ptr uint32_t* got = (
                    get_cb_tiles_received_ptr(kCbTokenCompute)
                );
                const uint32_t code = poll_live(
                    control_,
                    role_,
                    kWhereCToken,
                    false,
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
            MATH((ok = arrived(sequence)));
            PACK((ok = arrived(sequence)));
            return ok != 0;
        }

        // Math and pack: unpack's ok, then the sequence.
        bool arrived(uint32_t& sequence) const {
            if (from_unpack() == 0) {
                return false;
            }
            sequence = from_unpack();
            return true;
        }

        uint32_t control_;
        uint32_t me_;
        uint32_t wait_;
        uint32_t idle_;
        uint32_t role_ = kRoleUnpack;
        uint32_t index_ = 0;     // commands this thread has ended
        uint32_t sequence_ = 0;  // the command being run, for give-up words
        uint32_t slot_ = 0;
        mutable uint32_t produces_ = 0;  // pack: the buffers it fills
        mutable uint32_t consumes_ = 0;  // unpack: the buffers it drains
    };

}  // namespace km
#endif

// serve() and enter(): the loads of the service overlay and of an added handler
#include "km_overlay.hpp"
