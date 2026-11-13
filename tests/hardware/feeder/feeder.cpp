// Test feeder: one Tensix core outside every group plays the dispatcher (commands from
// DRAM, unicast to every worker, acks in its own L1, stop flag on a missed deadline).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_core.hpp"
#include "feeder.h"

namespace {

    uint32_t arg(uint32_t i) {
        return get_arg_val<uint32_t>(i);
    }

}  // namespace

void kernel_main() {
    const uint32_t control = arg(kFdControl);
    const uint32_t workers = arg(kFdWorkers), command_count = arg(kFdCommands);
    uint32_t depth = arg(kFdDepth);
    depth = depth == 0 || depth > KM_RING_SLOTS ? KM_RING_SLOTS : depth;
    const uint64_t deadline = static_cast<uint64_t>(arg(kFdDeadline)) << 10;
    const uint64_t boot_limit = static_cast<uint64_t>(arg(kFdBootLimit)) << 10;
    const uint32_t table_address = get_write_ptr(kFdCbTable);
    const uint32_t results_address = get_write_ptr(kFdCbResults);
    volatile tt_l1_ptr uint32_t* cores = km::l1(control + KM_CONTROL_CORES);
    volatile tt_l1_ptr uint32_t* ack = km::l1(
        control + KM_CONTROL_DONE
    );  // the ack word of a slot: ack[slot * 4]
    volatile tt_l1_ptr uint32_t* results_words = km::l1(results_address);
    auto row = [&](uint32_t row_index) {
        return results_words + row_index * 16;
    };
    auto core = [&](uint32_t worker, uint32_t address) {
        return get_noc_addr(cores[worker] & 0xFFFFu, cores[worker] >> 16, address);
    };

    // Core table and commands into L1; results cleared.
    const InterleavedAddrGen<true> core_table = {
        .bank_base_address = arg(kFdCores), .page_size = KM_CORES_MAX * 4
    };
    noc_async_read(
        get_noc_addr(0, core_table), control + KM_CONTROL_CORES, KM_CORES_MAX * 4
    );
    const InterleavedAddrGen<true> command_table = {
        .bank_base_address = arg(kFdTable), .page_size = 64
    };
    for (uint32_t command_index = 0; command_index < command_count; command_index++) {
        noc_async_read(
            get_noc_addr(command_index, command_table),
            table_address + command_index * 64,
            64
        );
        if ((command_index & 63) == 63) {
            noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t word = 0; word < (command_count + 2) * 16; word++) {
        results_words[word] = 0;
    }
    row(0)[kFrMagic] = kFeederMagic;

    // Ends the run: commands `status` and writes 1 to the stop flag of every worker.
    auto stop_all = [&](uint32_t status) {
        row(0)[kFrStatus] = status;
        *km::out_slot(control, 0) = 1;
        for (uint32_t worker = 0; worker < workers; worker++) {
            km::small_write<0, 4>(control, core(worker, control + KM_CONTROL_STOP), 0);
        }
        noc_async_write_barrier();
    };

    // Start-up: each group's READY (KM_CONTROL_BOOT) and the hello {1, control}
    // (KM_CONTROL_JOB). Ack words are zero from the host, only the grid writes them.
    const uint32_t groups = arg(kFdGroups);
    volatile tt_l1_ptr uint32_t* ready_words = km::l1(control + KM_CONTROL_BOOT);
    volatile tt_l1_ptr uint32_t* mailbox = km::l1(control + KM_CONTROL_JOB);
    const uint64_t boot_started = km::wall64();
    uint32_t ready = 0;
    bool booted = false;
    while (km::wall64() - boot_started < boot_limit) {
        invalidate_l1_cache();
        ready = 0;
        for (uint32_t group_index = 0; group_index < groups; group_index++) {
            ready |= (ready_words[group_index * 4] == KM_READY) << group_index;
        }
        if (ready == (1u << groups) - 1 && mailbox[0] == 1 && mailbox[1] == control) {
            booted = true;
            break;
        }
    }
    row(0)[kFrBoot] = static_cast<uint32_t>((km::wall64() - boot_started) >> 4);
    row(0)[kFrReady] = ready;

    int32_t busy[KM_RING_SLOTS];
    uint32_t busy_sequence[KM_RING_SLOTS];
    uint32_t busy_group[KM_RING_SLOTS];
    bool busy_result[KM_RING_SLOTS];
    uint64_t submitted_at[KM_RING_SLOTS];
    for (uint32_t slot = 0; slot < KM_RING_SLOTS; slot++) {
        busy[slot] = -1;
    }
    uint32_t sequence = 0, next = 0, done = 0, in_flight = 0, max_busy = 0;
    bool exited = false;  // an EXIT command of the table was submitted: nothing after it
    // The dynamic run in flight (one at a time): its table index, sequence and group.
    const uint32_t messages_handler = arg(kFdMessagesHandler);
    int32_t messages_index = -1;
    uint32_t messages_sequence = 0, messages_group = 0;
    bool messages_ended = true;
    auto busy_slot_of = [&](int32_t index) {
        uint32_t slot = 0;
        while (slot < depth && busy[slot] != index) {
            slot++;
        }
        return slot;
    };
    // The two-pass check of the group's status lines: 1 quiet (every line idle and
    // the same in both passes, sent equal to consumed), 0 not, -(1 + core) gave up.
    volatile tt_l1_ptr uint32_t* lines = km::l1(arg(kFdLines));
    volatile tt_l1_ptr uint32_t* first_pass = lines + KM_CORES_MAX * 4;
    auto quiet = [&](volatile tt_l1_ptr uint32_t* messages_record) -> int32_t {
        messages_record
            [kFrRecMessagesChecks] = (messages_record[kFrRecMessagesChecks] + 1);
        const uint32_t first_core = messages_group & 0xFFFFu;
        const uint32_t end = first_core + (messages_group >> 16);
        invalidate_l1_cache();
        bool idle = true;
        for (uint32_t core_index = first_core; core_index < end; core_index++) {
            for (uint32_t word = 0; word < 4; word++) {
                first_pass[core_index * 4 + word] = lines[core_index * 4 + word];
            }
            const uint32_t state = first_pass[core_index * 4 + KM_MESSAGES_LINE_STATE];
            if (
                first_pass[core_index * 4 + KM_MESSAGES_LINE_RUN] != messages_sequence
            ) {
                idle = false;
            } else if (state == KM_MESSAGES_GAVE_UP) {
                return -static_cast<int32_t>(core_index + 1);
            } else if (state != KM_MESSAGES_IDLE) {
                idle = false;
            }
        }
        if (!idle) {
            return 0;
        }
        invalidate_l1_cache();
        uint64_t sent = 0, consumed = 0;
        for (uint32_t core_index = first_core; core_index < end; core_index++) {
            for (uint32_t word = 0; word < 4; word++) {
                if (lines[core_index * 4 + word] != first_pass[core_index * 4 + word]) {
                    return 0;
                }
            }
            sent += first_pass[core_index * 4 + KM_MESSAGES_LINE_SENT];
            consumed += first_pass[core_index * 4 + KM_MESSAGES_LINE_CONSUMED];
        }
        messages_record[kFrRecMessagesSent] = static_cast<uint32_t>(sent);
        return sent == consumed ? 1 : 0;
    };
    uint64_t first_submit_time = 0, last_ack_time = 0, submit_cycles = 0;

    // Submits command `source` (64 bytes in L1) to the given slot of every worker.
    auto submit = [&](uint32_t slot, uint32_t source, bool raw) {
        volatile tt_l1_ptr km_command_t*
            slot_command = (
                reinterpret_cast<volatile tt_l1_ptr km_command_t*>(
                    control + KM_CONTROL_RING + slot * 64
                )
            );
        km::copy_command(
            reinterpret_cast<volatile tt_l1_ptr km_command_t*>(source), slot_command
        );
        sequence++;
        slot_command->sequence = (
            slot_command->land1 = slot_command->land2 = slot_command->land3 = sequence
        );
        if (!raw) {  // a raw command keeps the slot it names (test only)
            slot_command->hdr = (slot_command->hdr & 0xFF00FFFFu) | slot << 16;
        }
        const uint64_t started = km::wall64();
        if (arg(kFdRelease) != 0) {  // one release time for every core (test only)
            slot_command->a9 = static_cast<uint32_t>(started) + arg(kFdRelease);
        }
        for (uint32_t worker = 0; worker < workers; worker++) {
            noc_async_write(
                control + KM_CONTROL_RING + slot * 64,
                core(worker, control + KM_CONTROL_RING + slot * 64),
                64
            );
        }
        noc_async_writes_flushed();
        const uint64_t finished = km::wall64();
        submit_cycles += finished - started;
        return started;
    };

    if (!booted) {
        stop_all(kFsBoot);
    } else {
        row(0)[kFrStatus] = kFsDone;
        bool failed = false;
        while (!failed && (exited || done >= command_count ? in_flight != 0 : true)) {
            invalidate_l1_cache();
            const uint64_t now = km::wall64();
            for (uint32_t slot = 0; slot < depth; slot++) {
                if (busy[slot] < 0) {
                    continue;
                }
                volatile tt_l1_ptr uint32_t* record = row(1 + busy[slot]);
                if (ack[slot * 4] == KM_ACK(busy_sequence[slot])) {
                    if (
                        static_cast<uint32_t>(busy[slot]) == command_count
                    ) {  // a dynamic run's end
                        row(
                            1 + messages_index
                        )[kFrRecMessagesEndAck] = static_cast<uint32_t>(now);
                        busy[slot] = -1;
                        in_flight--;
                        continue;
                    }
                    record[kFrRecAcked] = static_cast<uint32_t>(now);
                    if (busy_result[slot]) {  // one 8-byte write with the ack word
                        record[kFrRecResult] = ack[slot * 4 + 1];
                    }
                    record[kFrRecStatus] = kFsDone;
                    last_ack_time = now;
                    busy[slot] = -1;
                    in_flight--;
                    done++;
                } else if (now - submitted_at[slot] > deadline) {
                    record[kFrRecStatus] = kFsTimeout;
                    row(0)[kFrSlot] = slot;
                    row(0)[kFrSequence] = busy_sequence[slot];
                    row(0)[kFrIndex] = busy[slot];
                    stop_all(kFsTimeout);
                    failed = true;
                    break;
                }
            }
            // A dynamic run in flight: its end command goes out once the lines are
            // quiet.
            if (!failed && messages_index >= 0 && !messages_ended) {
                volatile tt_l1_ptr uint32_t* messages_record = row(1 + messages_index);
                const int32_t quiet_state = quiet(messages_record);
                if (quiet_state < 0) {
                    messages_record[kFrRecMessagesGave] = (
                        static_cast<uint32_t>(-quiet_state)
                    );
                    row(0)[kFrSlot] = busy_slot_of(messages_index);
                    row(0)[kFrSequence] = messages_sequence;
                    row(0)[kFrIndex] = messages_index;
                    stop_all(kFsMessagesGave);
                    failed = true;
                    continue;
                }
                uint32_t slot = 0;
                while (slot < depth && busy[slot] >= 0) {
                    slot++;
                }
                if (quiet_state > 0 && slot < depth) {
                    messages_record[kFrRecMessagesEnd] = km::wall();
                    volatile tt_l1_ptr uint32_t* end_command = (
                        km::l1(table_address + (command_count + 1) * 64)
                    );
                    for (uint32_t word = 0; word < 16; word++) {
                        end_command[word] = 0;
                    }
                    end_command[1] = km::make_hdr(KM_HANDLER_MESSAGES_END, 0, 0);
                    end_command[2] = messages_group;
                    end_command[3] = messages_sequence;  // a0: the run it ends
                    submitted_at[slot] = (
                        submit(slot, table_address + (command_count + 1) * 64, false)
                    );
                    busy[slot] = static_cast<int32_t>(command_count);
                    busy_sequence[slot] = sequence;
                    busy_group[slot] = messages_group;
                    in_flight++;
                    messages_record[kFrRecMessagesEndSequence] = sequence;
                    messages_ended = true;
                }
            }
            if (failed || next >= command_count || exited) {
                continue;
            }
            const uint32_t source = table_address + next * 64;
            volatile tt_l1_ptr uint32_t* record = row(1 + next);
            volatile tt_l1_ptr uint32_t* command_words = km::l1(source);
            const bool is_exit = (command_words[1] & 0xFFu) == KM_HANDLER_EXIT;
            const bool raw = arg(kFdRaw) == next + 1;
            if (is_exit) {  // to every worker, and only when no slot is busy
                command_words[2] = raw ? command_words[2] : KM_GROUP_ALL(workers);
                if (in_flight != 0) {
                    record[kFrRecHeld] = 1;
                    continue;
                }
            }
            if (
                (command_words[1] >> 8) & KM_FLAG_EXCLUSIVE
            ) {  // held while an overlapping group is busy
                bool held = false;
                for (uint32_t busy_index = 0; busy_index < depth; busy_index++) {
                    held |= (
                        busy[busy_index] >= 0 &&
                        km::groups_overlap(busy_group[busy_index], command_words[2])
                    );
                }
                if (held) {
                    record[kFrRecHeld] = 1;
                    continue;
                }
            }
            uint32_t slot = 0;
            while (slot < depth && busy[slot] >= 0) {
                slot++;
            }
            if (slot == depth) {
                continue;
            }
            const uint64_t started = submit(slot, source, raw);
            first_submit_time = next == 0 ? started : first_submit_time;
            record[kFrRecSequence] = sequence;
            record[kFrRecSlot] = slot;
            record[kFrCmdSubmit] = static_cast<uint32_t>(started);
            record[kFrCmdSubmitted] = km::wall();
            busy[slot] = static_cast<int32_t>(next);
            busy_sequence[slot] = sequence;
            busy_group[slot] = command_words[2];
            busy_result[slot] = (command_words[1] >> 8) & KM_FLAG_RESULT;
            submitted_at[slot] = started;
            in_flight++;
            max_busy = in_flight > max_busy ? in_flight : max_busy;
            record[kFrRecBusy] = in_flight;
            if (KM_IS_DYNAMIC(
                messages_handler, command_words[1] & 0xFFu
            )) {  // a dynamic run starts
                messages_index = static_cast<int32_t>(next);
                messages_sequence = sequence;
                messages_group = command_words[2];
                messages_ended = false;
            }
            next++;
            exited = is_exit;
        }
        if (exited && !failed) {
            row(0)[kFrExitAcked] = row(next)[kFrRecStatus] == kFsDone;
        } else if (!failed && arg(kFdExit) != 0) {
            volatile tt_l1_ptr uint32_t* exit_command = km::l1(
                table_address + command_count * 64
            );  // spare command
            for (uint32_t word = 0; word < 16; word++) {
                exit_command[word] = 0;
            }
            exit_command[1] = km::make_hdr(KM_HANDLER_EXIT, 0, 0);
            exit_command[2] = KM_GROUP_ALL(workers);
            const uint64_t started = submit(
                0, table_address + command_count * 64, false
            );          // every slot is free
            for (;;) {  // EXIT completes like any command
                invalidate_l1_cache();
                if (ack[0] == KM_ACK(sequence)) {
                    row(0)[kFrExitAcked] = 1;
                    break;
                }
                if (km::wall64() - started > deadline) {
                    row(0)[kFrSlot] = 0;
                    row(0)[kFrSequence] = sequence;
                    row(0)[kFrIndex] = command_count;
                    stop_all(kFsTimeout);
                    break;
                }
            }
        }
    }

    row(0)[kFrSubmitted] = next;
    row(0)[kFrDone] = done;
    row(0)[kFrFirst] = static_cast<uint32_t>(first_submit_time);
    row(0)[kFrLast] = static_cast<uint32_t>(last_ack_time);
    const uint64_t span = done != 0 ? last_ack_time - first_submit_time
                                    : 0;  // 0: nothing acknowledged
    row(0)[kFrSpanLo] = static_cast<uint32_t>(span);
    row(0)[kFrSpanHi] = static_cast<uint32_t>(span >> 32);
    row(0)[kFrSubmitCycles] = static_cast<uint32_t>(submit_cycles);
    row(0)[kFrMaxBusy] = max_busy;
    const InterleavedAddrGen<true> results_table = {
        .bank_base_address = arg(kFdResults), .page_size = 64
    };
    for (uint32_t row_index = 0; row_index <= command_count + 1; row_index++) {
        noc_async_write(
            results_address + row_index * 64, get_noc_addr(row_index, results_table), 64
        );
    }
    noc_async_write_barrier();
}
