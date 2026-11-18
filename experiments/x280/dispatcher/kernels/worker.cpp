// Stand-in Tensix worker for the dispatcher tests (one data-movement kernel per
// core): only what the kernel needs. Every wait is timed; counters go to page `me`.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dk_test.h"
#include "komorebi/abi/km_abi.h"
#include "komorebi/device/core/km_core.hpp"

#ifdef DK_COMPILE_ERROR
#error "DK_COMPILE_ERROR: the compile-error test of the harness"
#endif

namespace {

    constexpr uint32_t kControlCb = 8;
    constexpr uint32_t kOutCb = 9;
    constexpr uint32_t kPollEvery = 256;

    // A NoC write's source and destination share their low 4 address bits.
    static_assert(
        KM_CONTROL_OUT % 16 == 0 && KM_CONTROL_BOOT % 16 == 0, "16-byte slots"
    );
    static_assert(
        KM_CONTROL_CORES % 16 == 0 && KM_ACK_STRIDE % 16 == 0, "16-byte slots"
    );
    static_assert(noc_mode != DM_DYNAMIC_NOC, "the timed NoC waits use the static NoC");

    // Runtime arguments (host module: Plan::worker_args).
    struct Args {
        uint32_t me, cores, tag, l2_x, l2_y, mailbox, ready, ack_base, leader_x,
            leader_y, rank, peers, is_leader, is_hello, table, out, limit, depth;
    };

    struct Worker {
        Args runtime_args;
        uint32_t control;
        uint32_t result;  // of the command just run
        uint64_t limit;   // cycles
        volatile tt_l1_ptr uint32_t* state;
        uint32_t counters[DK_WORDS];

        volatile tt_l1_ptr uint32_t* l1_word(uint32_t offset) const {
            return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control + offset);
        }

        static uint64_t now() {
            return km::wall64();
        }

        bool stopped() const {
            return *l1_word(KM_CONTROL_STOP) != 0;
        }

        // Commands why and where a wait gave up.
        void gave_up(uint32_t code, uint32_t where) {
            state[DK_S_GAVE] = code;
            state[DK_S_WHERE] = where;
        }

        // Every kPollEvery spins: false (with the gave-up command) once the stop flag
        // is set or `deadline` has passed.
        bool keep_waiting(uint32_t& spins, uint64_t deadline, uint32_t where) {
            if (++spins != kPollEvery) {
                return true;
            }
            spins = 0;
            invalidate_l1_cache();
            if (stopped()) {
                gave_up(DK_GAVE_STOP, where);
                return false;
            }
            if (now() > deadline) {
                gave_up(DK_GAVE_LIMIT, where);
                return false;
            }
            return true;
        }

        // Waits until flushed() holds, at most `limit` cycles; false (with the gave-up
        // command) if it does not.
        template <typename Condition>
        bool settle(Condition flushed, uint32_t where) {
            if (flushed()) {
                invalidate_l1_cache();
                return true;
            }
            const uint64_t deadline = now() + limit;
            while (!flushed()) {
                if (now() > deadline) {
                    gave_up(DK_GAVE_NOC, where);
                    return false;
                }
            }
            invalidate_l1_cache();
            return true;
        }

        bool reads_done() {
            return settle(
                [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                },
                DK_WHERE_NOC_READ
            );
        }

        bool writes_done() {
            return settle(
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                DK_WHERE_NOC_WRITE
            );
        }

        bool atomics_done() {
            return settle(
                [] {
                    return ncrisc_noc_nonposted_atomics_flushed(noc_index);
                },
                DK_WHERE_NOC_ATOMIC
            );
        }

        // Whether every address the host passed is 16-byte aligned (else the
        // gave-up command).
        bool aligned() {
            const uint32_t any = (
                control | runtime_args.mailbox | runtime_args.ready |
                runtime_args.ack_base | runtime_args.table | runtime_args.out |
                get_write_ptr(kOutCb)
            );
            if (any & 15u) {
                gave_up(DK_GAVE_ALIGN, DK_WHERE_BOOT);
                return false;
            }
            return true;
        }

        // Writes the 4-byte word `value` to NoC address `destination` (16-byte aligned)
        // and waits for it, at most `limit` cycles.
        bool send_word(uint64_t destination, uint32_t value) {
            *l1_word(KM_CONTROL_OUT) = value;
            noc_async_write(control + KM_CONTROL_OUT, destination, 4);
            return writes_done();
        }

        // Writes {first, second} as one 8-byte write to NoC address `destination`
        // (16-byte aligned) and waits for it, at most `limit` cycles.
        bool send_pair(uint64_t destination, uint32_t first, uint32_t second) {
            *l1_word(KM_CONTROL_OUT) = first;
            *l1_word(KM_CONTROL_OUT + 4) = second;
            noc_async_write(control + KM_CONTROL_OUT, destination, 8);
            return writes_done();
        }

        // Clears the ring, stop flag, counters and state, loads the core table, checks
        // in.
        bool start() {
            for (
                uint32_t offset = KM_CONTROL_RING; offset < KM_CONTROL_BYTES;
                offset += 4
            ) {
                *l1_word(offset) = 0;
            }
            *l1_word(KM_CONTROL_STOP) = 0;
            for (uint32_t slot = 0; slot < KM_RING_SLOTS; slot++) {
                *l1_word(KM_CONTROL_DONE + 16 * slot) = 0;
            }
            for (uint32_t word = 0; word <= DK_S_SEQUENCE; word++) {
                state[word] = 0;
            }
            const InterleavedAddrGen<true> table = {
                .bank_base_address = runtime_args.table, .page_size = KM_CORES_MAX * 4
            };
            noc_async_read(
                get_noc_addr(0, table), control + KM_CONTROL_CORES, KM_CORES_MAX * 4
            );
            if (!reads_done()) {
                return false;
            }
            if (!runtime_args.is_leader) {
                if (!send_word(
                    get_noc_addr(
                        runtime_args.leader_x,
                        runtime_args.leader_y,
                        control + KM_CONTROL_BOOT + 16 * runtime_args.rank
                    ),
                    runtime_args.tag << 16
                )) {
                    return false;
                }
            } else {
                const uint64_t deadline = now() + limit;
                uint32_t spins = 0;
                for (uint32_t peer = 0; peer < runtime_args.peers;) {
                    invalidate_l1_cache();
                    if (
                        (*l1_word(KM_CONTROL_BOOT + 16 * peer) >> 16) ==
                        runtime_args.tag
                    ) {
                        peer++;
                    } else if (!keep_waiting(spins, deadline, DK_WHERE_BOOT)) {
                        return false;
                    }
                }
                if (!send_word(
                    get_noc_addr(
                        runtime_args.l2_x, runtime_args.l2_y, runtime_args.ready
                    ),
                    KM_READY
                )) {
                    return false;
                }
            }
            if (runtime_args.is_hello) {
                *l1_word(KM_CONTROL_OUT) = 1;
                *l1_word(KM_CONTROL_OUT + 4) = control;
                noc_async_write(
                    control + KM_CONTROL_OUT,
                    get_noc_addr(
                        runtime_args.l2_x, runtime_args.l2_y, runtime_args.mailbox
                    ),
                    8
                );
                return writes_done();
            }
            return true;
        }

        static bool landed(volatile tt_l1_ptr km_command_t* command) {
            return KM_LANDED(command);
        }

        bool mine(uint32_t group) const {
            const uint32_t first = group & 0xFFFFu, count = group >> 16;
            return runtime_args.me >= first && runtime_args.me < first + count;
        }

        // Takes into KM_CONTROL_JOB the oldest new landed command that is this core's
        // or EXIT (others are consumed). False if the stop flag or limit ends the wait.
        bool next(uint32_t& slot) {
            const uint64_t deadline = now() + limit;
            uint32_t spins = 0;
            for (;;) {
                invalidate_l1_cache();
                uint32_t best = KM_RING_SLOTS, best_sequence = 0;
                for (
                    uint32_t ring_slot = 0; ring_slot < runtime_args.depth; ring_slot++
                ) {
                    auto* slot_command = (
                        reinterpret_cast<volatile tt_l1_ptr km_command_t*>(
                            control + KM_CONTROL_RING + 64 * ring_slot
                        )
                    );
                    const uint32_t sequence = slot_command->sequence;
                    if (
                        static_cast<int32_t>(sequence - state[ring_slot]) <= 0 ||
                        !landed(slot_command)
                    ) {
                        continue;
                    }
                    if (
                        (slot_command->hdr & 0xFFu) != KM_HANDLER_EXIT &&
                        !mine(slot_command->group)
                    ) {
                        state[ring_slot] = sequence;
                        counters[DK_C_IGNORED]++;
                        continue;
                    }
                    if (
                        best == KM_RING_SLOTS ||
                        static_cast<int32_t>(sequence - best_sequence) < 0
                    ) {
                        best = ring_slot;
                        best_sequence = sequence;
                    }
                }
                if (best != KM_RING_SLOTS) {
                    auto* slot_words = l1_word(KM_CONTROL_RING + 64 * best);
                    auto* job = l1_word(KM_CONTROL_JOB);
                    for (uint32_t word = 0; word < 16; word++) {
                        job[word] = slot_words[word];
                    }
                    auto* copy = (
                        reinterpret_cast<volatile tt_l1_ptr km_command_t*>(job)
                    );
                    if (landed(copy) && copy->sequence == best_sequence) {
                        state[best] = best_sequence;
                        state[DK_S_SEQUENCE] = best_sequence;
                        slot = best;
                        return true;
                    }
                }
                if (stopped()) {
                    gave_up(DK_GAVE_STOP, DK_WHERE_NEXT);
                    return false;
                }
                if (!keep_waiting(spins, deadline, DK_WHERE_NEXT)) {
                    return false;
                }
            }
        }

        // Waits `cycles` Tensix cycles, or until the stop flag.
        bool delay(uint32_t cycles) {
            const uint64_t until = now() + cycles;
            const uint64_t deadline = until + limit;
            uint32_t spins = 0;
            while (now() < until) {
                if (!keep_waiting(spins, deadline, DK_WHERE_DELAY)) {
                    return false;
                }
            }
            return true;
        }

        // Waits until the stop flag is set (or the limit runs out).
        void stall() {
            const uint64_t deadline = now() + limit;
            uint32_t spins = 0;
            while (keep_waiting(spins, deadline, DK_WHERE_STALL)) {
                ;
            }
        }

        // Runs the handler of the command in KM_CONTROL_JOB. Returns false if a wait
        // ended on the stop flag or its limit.
        bool run(const volatile tt_l1_ptr km_command_t* command, uint32_t slot) {
            const uint32_t handler = command->hdr & 0xFFu;
            if (((command->hdr >> 16) & 0xFFu) != slot) {
                counters[DK_C_HDR_ERRORS]++;
            }
            counters[DK_C_RUN]++;
            const uint32_t key = command->a0;
            counters[DK_C_KEY_SUM] += key;
            counters[DK_C_KEY_MIX] ^= dk_mix(key);
            if (handler < 5) {
                counters[DK_C_HANDLER + handler]++;
            }
            result = 0;
            const uint32_t args[10] = {
                key,
                command->a1,
                command->a2,
                command->a3,
                command->a4,
                command->a5,
                command->a6,
                command->a7,
                command->a8,
                command->a9
            };
            switch (handler) {
                case KM_HANDLER_NOP:
                    return true;
                case DK_HANDLER_CHECK:
                    for (uint32_t word = 2; word < 10; word++) {
                        if (args[word] != dk_arg(args[0], args[1], word)) {
                            counters[DK_C_ARG_ERRORS]++;
                        }
                    }
                    return true;
                case DK_HANDLER_DELAY:
                    return delay(args[1]);
                case DK_HANDLER_RESULT:
                    result = dk_result(args[0], args[1]);
                    return true;
                case DK_HANDLER_SUM:
                    result = args[0] + (command->group >> 16);
                    return true;
                case DK_HANDLER_STALL:
                    if (args[1] == runtime_args.me) {
                        stall();
                        return false;
                    }
                    return true;
                default:
                    counters[DK_C_HDR_ERRORS]++;
                    return true;
            }
        }

        // Reports completion: cores increment the leader's slot counter; the leader
        // waits for count - 1, then sends the ack ({ack, result} with KM_FLAG_RESULT).
        bool complete(const volatile tt_l1_ptr km_command_t* command, uint32_t slot) {
            const uint32_t first = command->group & 0xFFFFu;
            const uint32_t count = command->group >> 16;
            if (runtime_args.me != first) {
                const uint32_t leader_xy = *l1_word(KM_CONTROL_CORES + 4 * first);
                noc_semaphore_inc(
                    get_noc_addr(
                        leader_xy & 0xFFFFu,
                        leader_xy >> 16,
                        control + KM_CONTROL_DONE + 16 * slot
                    ),
                    1
                );
                if (!atomics_done()) {
                    return false;
                }
                counters[DK_C_INCS]++;
                return true;
            }
            volatile tt_l1_ptr uint32_t* done = l1_word(KM_CONTROL_DONE + 16 * slot);
            const uint64_t deadline = now() + limit;
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
                if (!keep_waiting(spins, deadline, DK_WHERE_DONE)) {
                    return false;
                }
            }
            *done = 0;
            const uint64_t ack = (
                get_noc_addr(
                    runtime_args.l2_x,
                    runtime_args.l2_y,
                    runtime_args.ack_base + slot * KM_ACK_STRIDE
                )
            );
            const bool sent = (
                ((command->hdr >> 8) & KM_FLAG_RESULT)
                    ? send_pair(ack, KM_ACK(command->sequence), result)
                    : send_word(ack, KM_ACK(command->sequence))
            );
            if (!sent) {
                return false;
            }
            counters[DK_C_ACKS]++;
            return true;
        }

        // Counters and state words to page `me` of the output tensor.
        void flush() {
            counters[DK_C_MAGIC] = DK_MAGIC | runtime_args.me;
            counters[DK_C_TAG] = runtime_args.tag;
            for (uint32_t word = 0; word <= DK_S_SEQUENCE; word++) {
                counters[DK_C_STATE + word] = state[word];
            }
            volatile tt_l1_ptr uint32_t*
                page = (
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        get_write_ptr(kOutCb)
                    )
                );
            for (uint32_t word = 0; word < DK_WORDS; word++) {
                page[word] = counters[word];
            }
            const InterleavedAddrGen<true> out = {
                .bank_base_address = runtime_args.out, .page_size = DK_WORDS * 4
            };
            noc_async_write(
                get_write_ptr(kOutCb), get_noc_addr(runtime_args.me, out), DK_WORDS * 4
            );
            writes_done();
        }
    };

}  // namespace

void kernel_main() {
    Worker worker;
    uint32_t* fields = &worker.runtime_args.me;
    for (uint32_t i = 0; i < sizeof(Args) / 4; i++) {
        fields[i] = get_arg_val<uint32_t>(i);
    }
    worker.control = get_write_ptr(kControlCb);
    worker.limit = static_cast<uint64_t>(worker.runtime_args.limit) << 20;
    worker.state = worker.l1_word(KM_CONTROL_STATE);
    for (uint32_t word = 0; word < DK_WORDS; word++) {
        worker.counters[word] = 0;
    }
    const bool aligned = worker.aligned();
    if (aligned && worker.start()) {
        uint32_t slot = 0;
        while (worker.next(slot)) {
            auto* command = (
                reinterpret_cast<volatile tt_l1_ptr km_command_t*>(
                    worker.l1_word(KM_CONTROL_JOB)
                )
            );
            const bool exit = (command->hdr & 0xFFu) == KM_HANDLER_EXIT;
            if (exit) {
                worker.counters[DK_C_EXIT] = command->sequence;
            } else if (!worker.run(command, slot)) {
                break;
            }
            if (!worker.complete(command, slot)) {
                break;
            }
            if (exit) {
                worker.gave_up(0, DK_WHERE_EXIT);
                break;
            }
        }
    }
    if (aligned) {
        worker.flush();
    }
}
