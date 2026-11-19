// The resident engines on TRISC 0 of the dispatcher's core (td_compute.cpp): the X280's
// stream interpreter and graph runs (x280/kernel/engines/engine.h, graph.h) over L1.

// Source 2 of out (td_engines.hpp has out's side): its command ring and its events in
// its area (kTe*). Its state lies in L1 too, the TRISC's local memory holds the stack.
#pragma once

#include <stdint.h>

#include "api/compute/common.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/abi/km_graph.h"
#include "komorebi/abi/km_stream.h"
#include "komorebi/device/core/km_core.hpp"
#include "td_engines.hpp"

// The graph executor as the X280 and the host build it; `template` names a field there.
#define template template_
#include "km_graph_engine.c"
#undef template

namespace td {

    // A stream's state (the X280's KMD_S_*).
    constexpr uint32_t kStreamFree = 0, kStreamRunning = 1;
    constexpr uint32_t kStreamEnding = 2;  // its end or error waits for room in the ring
    // A graph run's state (KMD_GR_*).
    constexpr uint32_t kRunFree = 0, kRunRunning = 1;
    constexpr uint32_t kRunEnding = 2;  // its end or error waits for room in the ring
    // In Sent::stream: the command is graph run (stream & 1)'s, pc its node.
    constexpr uint32_t kSentGraph = 0x80;
    constexpr uint32_t kSentCount = 256;  // a power of two (the X280's KMD_SENT)
    constexpr uint32_t kOrphans = KM_RING_SLOTS;
    // What a micro-op's fields mean (KMD_UOP_*): registers a, b, c, a jump, a template.
    constexpr uint32_t kUopA = 1, kUopB = 2, kUopC = 4, kUopJump = 8, kUopTemplate = 16;
    constexpr uint32_t kUopBad = 0x80;
    // A micro-op's turn: done, wait for a later pass, or its stream ended.
    enum class Uop { kDone, kWait, kEnded };

    static_assert(kSentCount >= kTeRingDepth + KM_RING_SLOTS + KM_QUEUE_JOBS);
    static_assert(KM_GRAPH_MAX_ACTIVE == 2u, "kSentGraph keeps one bit for the run");

    // A stream; it runs in place in the upload region (kmd_stream_t).
    struct Stream {
        uint32_t registers[KM_STREAM_REGISTERS];
        uint32_t state;
        uint32_t request, pc, steps, max_steps, op_count, command_count, offset;
        uint32_t pending;  // its commands and jobs sent and not yet answered
        uint32_t sent;    // the command of the SUBMIT or ENQUEUE at pc, not known taken
        uint32_t seen;    // sent: out took it in an earlier pass
        uint32_t waking;  // rejected busy: sends again after wake number wake_at
        uint32_t wake_at;
        uint32_t jobs;  // jobs it enqueued
        uint32_t drop;  // its queued jobs are to be dropped (KMD_HANDLER_DROP owed)
        uint32_t last_result;
        uint32_t order;  // activation number; turns go in this order
        uint32_t post_status, post_ticks;
        uint32_t t_command;  // each command's deadline, microseconds
    };

    // A command sent and not yet answered, at its number modulo kSentCount.
    struct Sent {
        uint32_t command_number;  // 0: free
        uint32_t request, order;
        uint16_t pc;
        uint8_t stream, result;
    };

    // A graph run (kmd_graph_run_t).
    struct GraphRun {
        uint32_t state;
        uint32_t request, order;
        uint32_t kept;        // a3 of its end: the graph is kept from now on
        uint32_t wait;        // a command was rejected: no step before the next answer
        uint32_t round_owed;  // a KM_EVENT_GRAPH event waits for room
        uint32_t post_status, post_ticks;
        uint32_t jobs;       // queued commands it sent: its end drops those left
        uint32_t t_command;  // microseconds
        kmg_state_t executor;
    };

    // TRISC 0's state, at kTeState of its area (kmd_engine_t and kmd_graphs_t).
    struct Engine {
        const volatile tt_l1_ptr uint8_t* upload;
        uint32_t upload_bytes, cores;
        volatile tt_l1_ptr uint32_t* ring;
        volatile tt_l1_ptr uint32_t* consumed;
        volatile tt_l1_ptr uint32_t* events;
        volatile tt_l1_ptr uint32_t* read;
        volatile tt_l1_ptr uint32_t* out;   // out's words (kTo*)
        volatile tt_l1_ptr uint32_t* own;   // its words (kTn*)
        volatile tt_l1_ptr uint32_t* over;  // in's kTwOver: the tag once it ended
        volatile tt_l1_ptr uint32_t* stop;  // the dispatcher's stop flag
        const volatile tt_l1_ptr GraphEntry* table;
        uint32_t tag;
        uint32_t scale;               // 50 MHz ticks a wall-clock tick, times 2^26
        uint32_t start_microseconds;  // out's deadline at the start
        uint32_t jobs, next, room_to, seen, wakes, told, starts;
        uint32_t start[KM_GRAPH_MAX_ACTIVE];  // graph entries to start, from the loop
        uint32_t deadline;  // microseconds of the last KM_HANDLER_DEADLINE it sent
        Stream streams[KM_STREAM_MAX_ACTIVE];
        uint32_t orders;
        Sent sent[kSentCount];
        uint32_t orphans;
        uint32_t orphan_request[kOrphans], orphan_status[kOrphans];
        uint32_t orphan_ticks[kOrphans];
        uint32_t passes, commands, posts, runs;
        GraphRun graphs[KM_GRAPH_MAX_ACTIVE];
        uint32_t line[16];  // the command being written
    };

    static_assert(sizeof(Engine) <= kTeUpload - kTeState, "TRISC 0's state");

    // The X280's 50 MHz timer from the wall clock, its low word.
    inline uint32_t engine_time(const Engine* engine) {
        const uint64_t wall = km::wall64();
        const uint64_t low = (wall & 0xFFFFFFFFu) * engine->scale;
        const uint32_t high = static_cast<uint32_t>(wall >> 32) * engine->scale;
        return (high << 6) + static_cast<uint32_t>(low >> 26);
    }

    inline uint32_t uop_fields(uint32_t op) {
        switch (op) {
            case KM_OP_SET:
            case KM_OP_RAND:
            case KM_OP_HALT:
            case KM_OP_RESULT:
            case KM_OP_EVENT:
            case KM_OP_TIME:
                return kUopA;
            case KM_OP_MOV:
            case KM_OP_ADDI:
            case KM_OP_SHR:
                return kUopA | kUopB;
            case KM_OP_ADD:
            case KM_OP_SUB:
            case KM_OP_MUL:
            case KM_OP_AND:
            case KM_OP_MOD:
                return kUopA | kUopB | kUopC;
            case KM_OP_JMP:
                return kUopJump;
            case KM_OP_JLT:
            case KM_OP_JEQ:
                return kUopA | kUopB | kUopJump;
            case KM_OP_JNZ:
                return kUopA | kUopJump;
            case KM_OP_SUBMIT:
                return kUopTemplate;
            case KM_OP_ENQUEUE:
                return kUopA | kUopTemplate;
            case KM_OP_WAIT:
                return 0;
            default:
                return kUopBad;
        }
    }

    // Bytes of `items` items of `size`, whole 64-byte lines.
    inline uint32_t stream_lines(uint32_t items, uint32_t size) {
        return (items * size + 63u) & ~63u;
    }

    // Template `index` of the stream at `base` with `op_count` micro-ops: 20 words.
    inline const volatile tt_l1_ptr uint32_t* stream_template(
        const volatile tt_l1_ptr uint8_t* base, uint32_t op_count, uint32_t index
    ) {
        return (
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                base + sizeof(km_stream_t) + stream_lines(op_count, sizeof(km_uop_t)) +
                index * sizeof(km_template_t)
            )
        );
    }

    // Whether template words `words` break the rules.
    inline bool template_bad(const volatile tt_l1_ptr uint32_t* words) {
        const uint32_t header = words[1], handler = header & 0xFFu;
        const uint32_t patch_args = words[16], patch_registers = words[17];
        const uint32_t tail = words[18];
        const uint32_t patch_count = (tail >> 16) & 0xFFu;
        if (
            handler < 1u || handler > KM_HANDLER_USER_MAX ||
            (((header >> 8) & 0xFFu) & ~(KM_FLAG_EXCLUSIVE | KM_FLAG_RESULT)) ||
            patch_count > KM_TEMPLATE_PATCHES
        ) {
            return true;
        }
        for (uint32_t i = 0; i < patch_count; i++) {
            if (
                ((patch_args >> (8u * i)) & 0xFFu) >= KM_TEMPLATE_ARGS ||
                ((patch_registers >> (8u * i)) & 0xFFu) >= KM_STREAM_REGISTERS
            ) {
                return true;
            }
        }
        for (uint32_t i = 0; i < 2; i++) {
            const uint32_t register_number = (tail >> (8u * i)) & 0xFFu;
            if (
                register_number != KM_TEMPLATE_NONE &&
                register_number >= KM_STREAM_REGISTERS
            ) {
                return true;
            }
        }
        return false;
    }

    // The check of the upload of `bytes` at `base`: 0, or reason | pc << 8.
    __attribute__((noinline)) inline uint32_t stream_check(
        const volatile tt_l1_ptr uint8_t* base, uint32_t bytes
    ) {
        const volatile tt_l1_ptr uint32_t*
            header = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(base);
        if (bytes < sizeof(km_stream_t) || (bytes & 63u)) {
            return KM_STREAM_ERROR_HEADER;
        }
        const uint32_t op_count = header[2], command_count = header[3];
        if (
            header[0] != KM_STREAM_MAGIC || header[1] != KM_STREAM_VERSION ||
            header[7] == 0u || op_count == 0u || op_count > KM_STREAM_MAX_OPS ||
            command_count > KM_STREAM_MAX_OPS || header[4] == 0u
        ) {
            return KM_STREAM_ERROR_HEADER;
        }
        const uint64_t size = (
            sizeof(km_stream_t) +
            ((static_cast<uint64_t>(op_count) * sizeof(km_uop_t) + 63u) & ~63ull) +
            (
                (static_cast<uint64_t>(command_count) * sizeof(km_template_t) + 63u) &
                ~63ull
            )
        );
        if (size != bytes) {
            return KM_STREAM_ERROR_HEADER;
        }
        const volatile tt_l1_ptr uint32_t* ops = header + sizeof(km_stream_t) / 4;
        for (uint32_t pc = 0; pc < op_count; pc++) {
            const uint32_t word = ops[4u * pc], immediate = ops[4u * pc + 1u];
            const uint32_t fields = uop_fields(word & 0xFFu);
            if (fields & kUopBad) {
                return KM_STREAM_ERROR_OPCODE | pc << 8;
            }
            for (uint32_t i = 0; i < 3; i++) {
                if ((fields & (1u << i)) && ((word >> (8u + 8u * i)) & 0xFFu) >= 16u) {
                    return KM_STREAM_ERROR_REGISTER | pc << 8;
                }
            }
            if ((fields & kUopJump) && immediate >= op_count) {
                return KM_STREAM_ERROR_JUMP | pc << 8;
            }
            if ((fields & kUopTemplate) && immediate >= command_count) {
                return KM_STREAM_ERROR_TEMPLATE | pc << 8;
            }
        }
        for (uint32_t index = 0; index < command_count; index++) {
            if (!template_bad(stream_template(base, op_count, index))) {
                continue;
            }
            // reported at the template's first SUBMIT or ENQUEUE, 0 if it has none
            for (uint32_t pc = 0; pc < op_count; pc++) {
                const uint32_t op = ops[4u * pc] & 0xFFu;
                if (
                    (op == KM_OP_SUBMIT || op == KM_OP_ENQUEUE) &&
                    ops[4u * pc + 1u] == index
                ) {
                    return KM_STREAM_ERROR_TEMPLATE | pc << 8;
                }
            }
            return KM_STREAM_ERROR_TEMPLATE;
        }
        return 0;
    }

    // Whether commands next .. next + count - 1 fit the ring.
    inline bool eng_room(Engine* engine, uint32_t count) {
        if (static_cast<int32_t>(engine->room_to - (engine->next + count - 1u)) >= 0) {
            return true;
        }
        invalidate_l1_cache();
        engine->room_to = *engine->consumed + kTeRingDepth;
        return static_cast<int32_t>(engine->room_to - (engine->next + count - 1u)) >= 0;
    }

    // Writes engine->line as command engine->next: the words, then its number in the
    // four landing words, which out reads first. The caller checked the room.
    inline void eng_write(Engine* engine) {
        const uint32_t number = engine->next++;
        volatile tt_l1_ptr uint32_t*
            to = (engine->ring + (number & (kTeRingDepth - 1u)) * 16u);
        for (uint32_t word = 0; word < 16; word++) {
            if (word % 4 != 0) {
                to[word] = engine->line[word];
            }
        }
        asm volatile("" ::: "memory");
        for (uint32_t word = 0; word < 16; word += 4) {
            to[word] = number;
        }
    }

    // A command to out with handler `handler` and arguments a0 to a3 (no group).
    inline void eng_command(
        Engine* engine,
        uint32_t handler,
        uint32_t a0,
        uint32_t a1,
        uint32_t a2,
        uint32_t a3 = 0
    ) {
        uint32_t* line = engine->line;
        for (uint32_t word = 0; word < 16; word++) {
            line[word] = 0;
        }
        line[1] = handler;
        line[3] = a0;
        line[5] = a1;
        line[6] = a2;
        line[7] = a3;
        eng_write(engine);
    }

    // Posts the host event {request, status, ticks} through out, with a3 `a3`, if the
    // ring has room. Whether posted.
    inline bool eng_post(
        Engine* engine,
        uint32_t request,
        uint32_t status,
        uint32_t ticks,
        uint32_t a3 = 0
    ) {
        if (!eng_room(engine, 1)) {
            return false;
        }
        eng_command(engine, KMD_HANDLER_POST, request, status, ticks, a3);
        engine->posts++;
        return true;
    }

    // Ends `stream` with KM_EVENT_STREAM_ERROR: `reason` at `pc`.
    inline void stream_fail(Stream* stream, uint32_t reason, uint32_t pc) {
        stream->state = kStreamEnding;
        stream->drop = stream->jobs != 0;
        stream->post_status = KM_EVENT_STREAM_ERROR | (reason | pc << 8) << 8;
        stream->post_ticks = 0;
    }

    // Sends what is owed: ending streams' drops and events, orphans' timeouts.
    __attribute__((noinline)) inline void eng_owed(Engine* engine) {
        for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
            Stream* stream = &engine->streams[i];
            if (stream->state != kStreamEnding) {
                continue;
            }
            if (stream->drop && eng_room(engine, 1)) {
                eng_command(engine, KMD_HANDLER_DROP, stream->order & 0xFFFFu, 0, 0);
                stream->drop = 0;
            }
            if (
                !stream->drop &&
                eng_post(
                    engine, stream->request, stream->post_status, stream->post_ticks
                )
            ) {
                stream->state = kStreamFree;
            }
        }
        while (engine->orphans != 0) {
            const uint32_t orphan = engine->orphans - 1u;
            if (!eng_post(
                engine,
                engine->orphan_request[orphan],
                engine->orphan_status[orphan],
                engine->orphan_ticks[orphan]
            )) {
                return;
            }
            engine->orphans--;
        }
    }

    // A timeout of a command whose stream or graph ended: posted as it is.
    inline void eng_orphan(
        Engine* engine, uint32_t request, uint32_t status, uint32_t ticks
    ) {
        if (engine->orphans < kOrphans) {
            engine->orphan_request[engine->orphans] = request;
            engine->orphan_status[engine->orphans] = status;
            engine->orphan_ticks[engine->orphans] = ticks;
            engine->orphans++;
        }
    }

    // The host's deadline in microseconds now: its last DEADLINE, or the start's.
    inline uint32_t host_microseconds(const Engine* engine) {
        const uint32_t microseconds = engine->out[kToDeadline];
        return microseconds != 0 ? microseconds : engine->start_microseconds;
    }

    // RUN from out: the upload of `bytes` at `offset`, admitted. Starts it, or ends
    // it at once.
    __attribute__((noinline)) inline void stream_start(
        Engine* engine, uint32_t offset, uint32_t bytes
    ) {
        Stream* stream = nullptr;
        for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE && stream == nullptr; i++) {
            stream = engine->streams[i].state == kStreamFree ? &engine->streams[i] : 0;
        }
        if (
            stream == nullptr || offset > engine->upload_bytes ||
            bytes > engine->upload_bytes - offset || bytes < sizeof(km_stream_t)
        ) {
            return;  // out admits neither
        }
        const volatile tt_l1_ptr uint32_t*
            header = (
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                    engine->upload + offset
                )
            );
        stream->request = header[6];
        stream->order = ++engine->orders;
        const uint32_t error = stream_check(engine->upload + offset, bytes);
        if (error != 0) {
            stream->state = kStreamEnding;
            stream->drop = 0;
            stream->post_status = KM_EVENT_STREAM_ERROR | error << 8;
            stream->post_ticks = 0;
            return;
        }
        for (uint32_t i = 0; i < KM_STREAM_REGISTERS; i++) {
            stream->registers[i] = 0;
        }
        stream->registers[KM_STREAM_SEED_REGISTER] = header[7];
        stream->pc = stream->steps = stream->pending = stream->last_result = 0;
        stream->jobs = stream->drop = 0;
        stream->sent = stream->seen = stream->waking = 0;
        stream->op_count = header[2];
        stream->command_count = header[3];
        stream->max_steps = header[4];
        stream->t_command = header[5] ? header[5] : host_microseconds(engine);
        stream->offset = offset;
        stream->state = kStreamRunning;
        engine->runs++;
    }

    // The graphs' events (graph.h's graph_event): whether it was theirs.
    inline bool graph_event(
        Engine* engine, uint32_t request, uint32_t status, uint32_t ticks
    ) {
        const uint32_t kind = status & 0xFFu, why = status >> 8;
        if (kind == kEventGraphRun) {
            // started from the loop, not here: kmg_check's stack is not added to this
            if (engine->starts < KM_GRAPH_MAX_ACTIVE) {
                engine->start[engine->starts++] = request;
            }
            return true;
        }
        if (kind == KMD_EVENT_CANCEL) {
            for (uint32_t i = 0; i < KM_GRAPH_MAX_ACTIVE; i++) {
                GraphRun* run = &engine->graphs[i];
                if (run->state != kRunFree && run->request == ticks) {
                    kmg_stop(
                        &run->executor, KM_STREAM_ERROR_CANCELLED, KM_GRAPH_NO_NODE
                    );
                    return true;
                }
            }
            return false;
        }
        if (kind == KMD_EVENT_WAKE) {
            for (uint32_t i = 0; i < KM_GRAPH_MAX_ACTIVE; i++) {
                engine->graphs[i].wait = 0;
            }
            return false;  // the streams may want it too
        }
        Sent* sent = &engine->sent[request % kSentCount];
        if (sent->command_number != request || !(sent->stream & kSentGraph)) {
            return false;
        }
        sent->command_number = 0;
        GraphRun* run = &engine->graphs[sent->stream & 1u];
        const bool live = run->state == kRunRunning && run->order == sent->order;
        if (live) {
            run->wait = 0;
        }
        if (kind == KM_EVENT_DONE) {
            if (live) {
                kmg_complete(&run->executor, sent->pc, sent->result ? ticks : 0u);
            }
        } else if (kind == KM_EVENT_TIMEOUT) {
            if (live) {
                kmg_stop(&run->executor, KM_STREAM_ERROR_DEADLINE, sent->pc);
            } else {
                eng_orphan(engine, sent->request, status, ticks);
            }
        } else if (kind == KM_EVENT_REJECTED && why == KMD_REJECT_BUSY) {
            if (live) {
                kmg_retry(&run->executor, sent->pc);
                run->wait = 1;
            }
        } else if (live) {
            kmg_stop(
                &run->executor,
                why == KMD_REJECT_GROUP
                    ? KM_STREAM_ERROR_GROUP
                    : KM_STREAM_ERROR_TEMPLATE,
                sent->pc
            );
        }
        return true;
    }

    // One event from out: RUN, CANCEL, or the answer to a command of the engines.
    inline void eng_event(
        Engine* engine, uint32_t request, uint32_t status, uint32_t ticks
    ) {
        const uint32_t kind = status & 0xFFu;
        if (graph_event(engine, request, status, ticks)) {
            return;
        }
        if (kind == KMD_EVENT_RUN) {
            stream_start(engine, request, ticks);
            return;
        }
        if (kind == KMD_EVENT_WAKE) {
            engine->wakes++;
            return;
        }
        if (kind == KMD_EVENT_CANCEL) {
            for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
                Stream* stream = &engine->streams[i];
                if (stream->state == kStreamRunning && stream->request == ticks) {
                    stream_fail(stream, KM_STREAM_ERROR_CANCELLED, stream->pc);
                }
            }
            return;
        }
        Sent* sent = &engine->sent[request % kSentCount];
        if (sent->command_number != request) {
            return;  // not a command: nothing waits for it
        }
        sent->command_number = 0;
        Stream* stream = &engine->streams[sent->stream % KM_STREAM_MAX_ACTIVE];
        const bool live = (
            stream->state == kStreamRunning && stream->order == sent->order
        );
        if (kind == KM_EVENT_DONE) {
            if (live) {
                stream->pending--;
                stream->last_result = sent->result ? ticks : stream->last_result;
            }
        } else if (kind == KM_EVENT_REJECTED && status >> 8 == KMD_REJECT_BUSY) {
            if (live) {  // not taken: sent again after the next wake
                stream->pending--;
                stream->sent = stream->seen = 0;
                stream->waking = 1;
                stream->wake_at = engine->wakes;
            }
        } else if (kind == KM_EVENT_TIMEOUT) {
            if (live) {
                stream_fail(stream, KM_STREAM_ERROR_DEADLINE, sent->pc);
            } else {
                eng_orphan(engine, sent->request, status, ticks);
            }
        } else if (live) {
            const uint32_t why = (
                status >> 8 == KMD_REJECT_GROUP
                    ? KM_STREAM_ERROR_GROUP
                    : KM_STREAM_ERROR_TEMPLATE
            );
            stream_fail(stream, why, sent->pc);
        }
    }

    // Reads every event out wrote (index, fields, index again), then stores the read
    // count. Returns how many.
    __attribute__((noinline)) inline uint32_t eng_events(Engine* engine) {
        uint32_t count = 0;
        for (; count < kTeEventDepth; count++) {
            const uint32_t number = engine->seen + 1u;
            volatile tt_l1_ptr uint32_t*
                words = (engine->events + (number & (kTeEventDepth - 1u)) * 4u);
            invalidate_l1_cache();
            if (words[0] != number) {
                break;
            }
            const uint32_t request = words[1], status = words[2], ticks = words[3];
            invalidate_l1_cache();
            if (words[0] != number) {
                break;
            }
            engine->seen = number;
            eng_event(engine, request, status, ticks);
        }
        if (engine->seen != engine->told) {
            *engine->read = engine->seen;
            engine->told = engine->seen;
        }
        return count;
    }

    // SUBMIT (`job` 0) or ENQUEUE of `job` cores, template `index` at `pc`: checked,
    // then sent (DEADLINE first if changed) if there is room; else it waits.
    inline Uop stream_command(
        Engine* engine, Stream* stream, uint32_t index, uint32_t pc, uint32_t job
    ) {
        if (job != 0 && job > engine->cores) {
            stream_fail(stream, KM_STREAM_ERROR_NEVER_FITS, pc);
            return Uop::kEnded;
        }
        if (index >= stream->command_count) {
            stream_fail(stream, KM_STREAM_ERROR_TEMPLATE, pc);
            return Uop::kEnded;
        }
        const volatile tt_l1_ptr uint32_t* words = (
            stream_template(engine->upload + stream->offset, stream->op_count, index)
        );
        const uint32_t patch_args = words[16], patch_registers = words[17];
        const uint32_t tail = words[18];
        const uint32_t group = words[2];
        const uint32_t first_register = tail & 0xFFu;
        const uint32_t count_register = (tail >> 8) & 0xFFu;
        uint32_t first = 0, count = job;
        if (job == 0) {
            first = (
                first_register == KM_TEMPLATE_NONE
                    ? group & 0xFFFFu
                    : stream->registers[first_register & 15u]
            );
            count = (
                count_register == KM_TEMPLATE_NONE
                    ? group >> 16
                    : stream->registers[count_register & 15u]
            );
            // a work cluster's thread takes no exclusive command
            const uint32_t exclusive = (words[1] >> 8) & KM_FLAG_EXCLUSIVE;
            const uint32_t ranks = engine->cores;
            if (
                KM_GROUP_IS_CLUSTER(first, count)
                    ? exclusive != 0
                    : count == 0 || first > ranks || count > ranks - first
            ) {
                stream_fail(stream, KM_STREAM_ERROR_GROUP, pc);
                return Uop::kEnded;
            }
        } else if (
            engine->out[kToQueued] + (engine->jobs - engine->out[kToTaken]) >=
            KM_QUEUE_JOBS
        ) {
            return Uop::kWait;  // jobs in the queue and on their way to it
        }
        const uint32_t lines = stream->t_command != engine->deadline ? 2u : 1u;
        const uint32_t command_number = engine->next + lines - 1u;
        if (
            !eng_room(engine, lines) ||
            engine->sent[command_number % kSentCount].command_number != 0
        ) {
            return Uop::kWait;
        }
        if (lines == 2u) {
            const uint32_t microseconds = stream->t_command;
            eng_command(
                engine, KM_HANDLER_DEADLINE, microseconds == 0 ? 1u : microseconds, 0, 0
            );
            engine->deadline = stream->t_command;
        }
        uint32_t* line = engine->line;
        for (uint32_t word = 0; word < 16; word++) {
            line[word] = words[word];
        }
        line[2] = first | count << 16;
        if (job != 0) {
            // out's queue: KM_FLAG_QUEUE, the stream's tag in the group's first half
            line[1] = (line[1] & 0xFF00FFFFu) | KM_FLAG_QUEUE << 8;
            line[2] = (stream->order & 0xFFFFu) | count << 16;
            engine->jobs++;
            stream->jobs++;
        }
        // the word of argument a0 to a9 in km_command_t
        static constexpr uint8_t arg_word[KM_TEMPLATE_ARGS] = {
            3, 5, 6, 7, 9, 10, 11, 13, 14, 15
        };
        const uint32_t patch_count = (tail >> 16) & 0xFFu;
        for (uint32_t i = 0; i < patch_count && i < KM_TEMPLATE_PATCHES; i++) {
            const uint32_t arg = (patch_args >> (8u * i)) & 0xFFu;
            const uint32_t register_number = (patch_registers >> (8u * i)) & 15u;
            if (arg < KM_TEMPLATE_ARGS) {
                line[arg_word[arg]] = stream->registers[register_number];
            }
        }
        eng_write(engine);
        Sent* sent = &engine->sent[command_number % kSentCount];
        sent->command_number = command_number;
        sent->request = stream->request;
        sent->order = stream->order;
        sent->pc = static_cast<uint16_t>(pc);
        sent->stream = static_cast<uint8_t>(stream - engine->streams);
        sent->result = ((line[1] >> 8) & KM_FLAG_RESULT) != 0;
        engine->commands++;
        stream->pending++;
        stream->sent = command_number;
        return Uop::kWait;
    }

    // Whether an event waits to be read.
    inline bool eng_unread(const Engine* engine) {
        const uint32_t number = engine->seen + 1u;
        invalidate_l1_cache();
        return engine->events[(number & (kTeEventDepth - 1u)) * 4u] == number;
    }

    // SUBMIT or ENQUEUE at pc: sent, then done once out took it (a busy rejection is
    // written before out counts it taken, so no unread event means taken).
    inline Uop stream_send(
        Engine* engine, Stream* stream, uint32_t index, uint32_t pc, uint32_t job
    ) {
        if (stream->sent == 0) {
            if (stream->waking && engine->wakes == stream->wake_at) {
                return Uop::kWait;
            }
            stream->waking = 0;
            return stream_command(engine, stream, index, pc, job);
        }
        if (!stream->seen) {
            invalidate_l1_cache();
            if (static_cast<int32_t>(*engine->consumed - stream->sent) < 0) {
                return Uop::kWait;
            }
            if (eng_unread(engine)) {
                stream->seen = 1;  // the next pass reads those events first
                return Uop::kWait;
            }
        }
        stream->sent = stream->seen = 0;
        return Uop::kDone;
    }

    // One micro-op of `stream` at its pc; a wait is no step.
    inline Uop stream_op(Engine* engine, Stream* stream) {
        const uint32_t pc = stream->pc;
        if (pc >= stream->op_count) {
            stream_fail(stream, KM_STREAM_ERROR_JUMP, stream->op_count);
            return Uop::kEnded;
        }
        if (stream->steps >= stream->max_steps) {
            stream_fail(stream, KM_STREAM_ERROR_STEPS, pc);
            return Uop::kEnded;
        }
        const volatile tt_l1_ptr uint32_t*
            ops = (
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                    engine->upload + stream->offset + sizeof(km_stream_t)
                )
            );
        const uint32_t word = ops[4u * pc], immediate = ops[4u * pc + 1u];
        const uint32_t op = word & 0xFFu;
        uint32_t* registers = stream->registers;
        uint32_t* register_a = &registers[(word >> 8) & 15u];
        const uint32_t value_b = registers[(word >> 16) & 15u];
        const uint32_t value_c = registers[(word >> 24) & 15u];
        uint32_t next = pc + 1u;
        switch (op) {
            case KM_OP_SET:
                *register_a = immediate;
                break;
            case KM_OP_MOV:
                *register_a = value_b;
                break;
            case KM_OP_ADD:
                *register_a = value_b + value_c;
                break;
            case KM_OP_ADDI:
                *register_a = value_b + immediate;
                break;
            case KM_OP_SUB:
                *register_a = value_b - value_c;
                break;
            case KM_OP_MUL:
                *register_a = value_b * value_c;
                break;
            case KM_OP_AND:
                *register_a = value_b & value_c;
                break;
            case KM_OP_SHR:
                *register_a = value_b >> (immediate & 31u);
                break;
            case KM_OP_MOD:
                if (value_c == 0) {
                    stream_fail(stream, KM_STREAM_ERROR_MOD_ZERO, pc);
                    return Uop::kEnded;
                }
                *register_a = value_b % value_c;
                break;
            case KM_OP_RAND: {
                uint32_t random = registers[KM_STREAM_SEED_REGISTER];
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                registers[KM_STREAM_SEED_REGISTER] = random;
                *register_a = random;
                break;
            }
            case KM_OP_JMP:
                next = immediate;
                break;
            case KM_OP_JLT:
                next = *register_a < value_b ? immediate : next;
                break;
            case KM_OP_JEQ:
                next = *register_a == value_b ? immediate : next;
                break;
            case KM_OP_JNZ:
                next = *register_a != 0 ? immediate : next;
                break;
            case KM_OP_HALT:
                // an implicit WAIT 0, then the end: auxiliary the low 24 bits
                if (stream->pending != 0) {
                    return Uop::kWait;
                }
                stream->state = kStreamEnding;
                stream->drop = 0;
                stream->post_status = (
                    KM_EVENT_STREAM_END | (*register_a & 0xFFFFFFu) << 8
                );
                stream->post_ticks = *register_a;
                return Uop::kEnded;
            case KM_OP_SUBMIT:
            case KM_OP_ENQUEUE: {
                if (op == KM_OP_ENQUEUE && *register_a == 0) {
                    stream_fail(stream, KM_STREAM_ERROR_NEVER_FITS, pc);
                    return Uop::kEnded;
                }
                const uint32_t job = op == KM_OP_ENQUEUE ? *register_a : 0u;
                const Uop outcome = stream_send(engine, stream, immediate, pc, job);
                if (outcome != Uop::kDone) {
                    return outcome;
                }
                break;
            }
            case KM_OP_WAIT:
                if (stream->pending > immediate) {
                    return Uop::kWait;
                }
                break;
            case KM_OP_RESULT:
                *register_a = stream->last_result;
                break;
            case KM_OP_EVENT:
                if (!eng_post(
                    engine,
                    stream->request,
                    KM_EVENT_STREAM | (*register_a & 0xFFFFFFu) << 8,
                    *register_a
                )) {
                    return Uop::kWait;
                }
                break;
            case KM_OP_TIME:
                *register_a = static_cast<uint32_t>(engine_time(engine));
                break;
            default:
                stream_fail(stream, KM_STREAM_ERROR_OPCODE, pc);
                return Uop::kEnded;
        }
        stream->steps++;
        stream->pc = next;
        return Uop::kDone;
    }

    // The streams' part of a pass: each running stream up to KM_STEP_OPS micro-ops, in
    // activation order from stream `passes % running`. Whether one runs.
    __attribute__((noinline)) inline bool eng_step(Engine* engine) {
        Stream* turn[KM_STREAM_MAX_ACTIVE];
        uint32_t running = 0;
        for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
            Stream* stream = &engine->streams[i];
            if (stream->state != kStreamRunning) {
                continue;
            }
            uint32_t position = running++;
            while (position > 0 && turn[position - 1]->order > stream->order) {
                turn[position] = turn[position - 1];
                position--;
            }
            turn[position] = stream;
        }
        if (running == 0) {
            return false;
        }
        const uint32_t start = engine->passes % running;
        for (uint32_t index = 0; index < running; index++) {
            Stream* stream = turn[(start + index) % running];
            for (uint32_t op = 0; op < KM_STEP_OPS; op++) {
                if (stream_op(engine, stream) != Uop::kDone) {
                    break;
                }
            }
        }
        return true;
    }

    // Ends graph run `run` with `status`, `ticks` (posted when the ring has room).
    inline void graph_end(GraphRun* run, uint32_t status, uint32_t ticks) {
        run->state = kRunEnding;
        run->post_status = status;
        run->post_ticks = ticks;
    }

    // kEventGraphRun for table entry `index`, from the loop: the check, the kept table,
    // the bindings, then the graph runs; a fault ends it at once.
    __attribute__((noinline)) inline void graph_start(Engine* engine, uint32_t index) {
        GraphRun* run = nullptr;
        for (uint32_t j = 0; j < KM_GRAPH_MAX_ACTIVE && run == nullptr; j++) {
            run = engine->graphs[j].state == kRunFree ? &engine->graphs[j] : nullptr;
        }
        const volatile tt_l1_ptr GraphEntry*
            entry = &engine->table[index % kGraphEntries];
        invalidate_l1_cache();
        const uint32_t offset = entry->offset, bytes = entry->bytes;
        const uint32_t binding_count = entry->binding_count;
        if (
            run == nullptr || offset > engine->upload_bytes ||
            bytes > engine->upload_bytes - offset
        ) {
            return;  // out admits neither
        }
        run->request = entry->request;
        run->order = ++engine
                           ->orders;  // shared with the streams: a job's tag names one
        run->wait = run->round_owed = run->jobs = run->kept = 0;
        const void* upload = (
            reinterpret_cast<const void*>(
                reinterpret_cast<uintptr_t>(engine->upload) + offset
            )
        );
        // the free run's left[] for the check's counts, not 256 B of TRISC 0's stack
        uint32_t error = kmg_check_in(upload, bytes, run->executor.left);
        const uint32_t keep = (
            error ? 0u : static_cast<const km_graph_t*>(upload)->flags & KM_GRAPH_KEEP
        );
        if (keep && entry->full) {
            error = KM_STREAM_ERROR_KEPT_FULL | KM_GRAPH_NO_NODE << 8;
        }
        if (error == 0 && (!entry->bind_ok || binding_count > KM_GRAPH_BINDINGS)) {
            error = KM_STREAM_ERROR_HEADER | KM_GRAPH_NO_NODE << 8;
        }
        if (error == 0) {
            uint32_t bindings[KM_GRAPH_BINDINGS];
            const volatile tt_l1_ptr uint32_t*
                words = (
                    reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                        engine->upload + entry->binding_offset
                    )
                );
            for (uint32_t j = 0; j < binding_count; j++) {
                bindings[j] = words[j];
            }
            error = (
                kmg_start(
                    &run->executor, upload, bindings, binding_count, engine->cores
                )
            );
            kmg_island(&run->executor, 0, 0);
            error = error ? error | KM_GRAPH_NO_NODE << 8 : 0u;
        }
        if (error != 0) {
            graph_end(run, KM_EVENT_GRAPH_ERROR | error << 8, 0);
            return;
        }
        const uint32_t microseconds = run->executor.hdr->deadline_microseconds;
        run->t_command = microseconds ? microseconds : host_microseconds(engine);
        run->kept = keep != 0;
        run->state = kRunRunning;
    }

    // The submit function's context: the engines and the run.
    struct GraphContext {
        Engine* engine;
        GraphRun* run;
        uint32_t index;
    };

    // kmg_submit_function: the command into the ring (after a KM_HANDLER_DEADLINE if
    // the deadline changed), or KMG_BUSY when the ring or the sent table has no room.
    __attribute__((noinline)) inline uint32_t graph_submit(
        void* context, uint32_t node, const km_command_t* record
    ) {
        GraphContext* submit_context = static_cast<GraphContext*>(context);
        Engine* engine = submit_context->engine;
        GraphRun* run = submit_context->run;
        const uint32_t lines = run->t_command != engine->deadline ? 2u : 1u;
        const uint32_t command_number = engine->next + lines - 1u;
        const volatile uint32_t* source = (
            reinterpret_cast<const volatile uint32_t*>(record)
        );
        const uint32_t job = (source[1] >> 8) & KM_FLAG_QUEUE;
        if (
            !eng_room(engine, lines) ||
            engine->sent[command_number % kSentCount].command_number != 0
        ) {
            return KMG_BUSY;
        }
        if (
            job &&
            engine->out[kToQueued] + (engine->jobs - engine->out[kToTaken]) >=
                KM_QUEUE_JOBS
        ) {
            return KMG_BUSY;  // jobs in the queue and on their way to it
        }
        if (lines == 2u) {
            const uint32_t microseconds = run->t_command;
            eng_command(
                engine, KM_HANDLER_DEADLINE, microseconds == 0 ? 1u : microseconds, 0, 0
            );
            engine->deadline = run->t_command;
        }
        uint32_t* line = engine->line;
        for (uint32_t word = 0; word < 16; word++) {
            line[word] = source[word];
        }
        if (job) {
            // the run's tag in the group's first half, as a stream's job has its own
            line[1] &= 0xFF00FFFFu;
            line[2] = (line[2] & 0xFFFF0000u) | (run->order & 0xFFFFu);
            engine->jobs++;
            run->jobs++;
        }
        eng_write(engine);
        Sent* sent = &engine->sent[command_number % kSentCount];
        sent->command_number = command_number;
        sent->request = run->request;
        sent->order = run->order;
        sent->pc = static_cast<uint16_t>(node);
        sent->stream = static_cast<uint8_t>(kSentGraph | submit_context->index);
        sent->result = ((source[1] >> 8) & KM_FLAG_RESULT) != 0;
        engine->commands++;
        return KMG_SUBMITTED;
    }

    // Posts what the graphs owe: round events, ends and errors.
    __attribute__((noinline)) inline void graph_owed(Engine* engine) {
        for (uint32_t index = 0; index < KM_GRAPH_MAX_ACTIVE; index++) {
            GraphRun* run = &engine->graphs[index];
            if (run->state == kRunRunning && run->executor.phase == KMG_ERROR) {
                // stopped by an event (a timeout, a cancel)
                const uint32_t auxiliary = (
                    run->executor.error_reason | run->executor.error_node << 8
                );
                graph_end(run, KM_EVENT_GRAPH_ERROR | auxiliary << 8, 0);
            }
            if (run->state == kRunRunning && run->round_owed) {
                const uint32_t done = (run->executor.runs - 1u) & 0xFFFFFFu;
                run->round_owed = (
                    !eng_post(engine, run->request, KM_EVENT_GRAPH | done << 8, 0)
                );
            }
            if (run->state != kRunEnding) {
                continue;
            }
            if (run->jobs && eng_room(engine, 1)) {
                eng_command(engine, KMD_HANDLER_DROP, run->order & 0xFFFFu, 0, 0);
                run->jobs = 0;
            }
            if (
                !run->jobs &&
                eng_post(
                    engine, run->request, run->post_status, run->post_ticks, run->kept
                )
            ) {
                run->state = kRunFree;
            }
        }
    }

    // The graphs' part of a pass: what they owe, then one kmg_step per running graph
    // (none while its round event or a rejected command waits).
    __attribute__((noinline)) inline bool graph_step(Engine* engine) {
        graph_owed(engine);
        bool any = false;
        for (uint32_t index = 0; index < KM_GRAPH_MAX_ACTIVE; index++) {
            GraphRun* run = &engine->graphs[index];
            any = any || run->state != kRunFree;
            if (
                run->state != kRunRunning || run->round_owed ||
                (run->wait && run->executor.phase == KMG_RUNNING)
            ) {
                continue;
            }
            GraphContext context = {engine, run, index};
            const uint32_t got = kmg_step(&run->executor, graph_submit, &context);
            if (got == KMG_ROUND) {
                run->round_owed = 1;
            } else if (got == KMG_END) {
                run->jobs = 0;  // every command completed: no job left to drop
                graph_end(
                    run,
                    KM_EVENT_GRAPH_END | (run->executor.registers[0] & 0xFFFFFFu) << 8,
                    run->executor.registers[1]
                );
            } else if (got == KMG_ERROR) {
                const uint32_t auxiliary = (
                    run->executor.error_reason | run->executor.error_node << 8
                );
                graph_end(run, KM_EVENT_GRAPH_ERROR | auxiliary << 8, 0);
            }
        }
        graph_owed(engine);
        return any;
    }

    // Whether the dispatcher ended (in's last words) or the host stopped its core.
    inline bool engine_ended(const Engine* engine) {
        invalidate_l1_cache();
        return engine->over[0] == engine->tag || engine->stop[0] != 0;
    }

    // TRISC 0: waits for out's start (its area zeroed), then the loop: events, owed
    // posts, streams, graphs, until the dispatcher ends.
    inline void engine() {
        const uint32_t block = get_arg_val<uint32_t>(kTdBlock);
        const uint32_t area = block + kTbBytes + kTaBytes;
        const uint32_t tag = get_arg_val<uint32_t>(kTdTag);
        const uint32_t control = get_arg_val<uint32_t>(kTdControl);
        volatile tt_l1_ptr uint32_t* go = (
            km::l1(block + kTbHand + kTxWords + 4 * kTwGo)
        );
        volatile tt_l1_ptr uint32_t* over = (
            km::l1(block + kTbHand + kTxWords + 4 * kTwOver)
        );
        volatile tt_l1_ptr uint32_t* stop = km::l1(control + KM_CONTROL_STOP);
        do {
            invalidate_l1_cache();
            if (over[0] == tag || stop[0] != 0) {
                return;  // the dispatcher ended before its start
            }
        } while (go[0] != tag);
        Engine* engine = reinterpret_cast<Engine*>(area + kTeState);
        uint32_t* words = reinterpret_cast<uint32_t*>(engine);
        for (uint32_t i = 0; i < sizeof(Engine) / 4; i++) {
            words[i] = 0;
        }
        engine->upload = (
            reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(area + kTeUpload)
        );
        engine->upload_bytes = kTeUploadBytes;
        engine->cores = get_arg_val<uint32_t>(kTdRanks);
        engine->ring = km::l1(area + kTeRing);
        engine->consumed = km::l1(area + kTeWork + KMD_WORK_CONSUMED);
        engine->events = km::l1(area + kTeWork + KMD_WORK_EVENTS);
        engine->read = km::l1(area + kTeRead);
        engine->out = km::l1(area + kTeOut);
        engine->own = km::l1(area + kTeOwn);
        engine->over = over;
        engine->stop = stop;
        engine->table = (
            reinterpret_cast<const volatile tt_l1_ptr GraphEntry*>(area + kTeTable)
        );
        engine->tag = tag;
        // the wall clock's MHz; 32-bit divisions only (no 64-bit one on this core)
        const uint32_t mhz = (get_arg_val<uint32_t>(kTdTicksUs) + 32768u) >> 16;
        const uint32_t units = get_arg_val<uint32_t>(kTdDeadline);  // 1024 ticks
        engine->scale = (50u << 26) / mhz;
        engine->start_microseconds = units / mhz * 1024u + units % mhz * 1024u / mhz;
        engine->next = 1;
        engine->room_to = kTeRingDepth;
        engine->own[kTnState] = 1;
        while (!engine_ended(engine)) {
            engine->passes++;
            const uint32_t read = eng_events(engine);
            for (uint32_t i = 0; i < engine->starts; i++) {  // kmg_check's stack alone
                graph_start(engine, engine->start[i]);
            }
            engine->starts = 0;
            eng_owed(engine);
            const bool streams = eng_step(engine);
            const bool graphs = graph_step(engine);
            eng_owed(engine);
            if (read == 0 && !streams && !graphs && engine->orphans == 0) {
                for (uint32_t i = 0; i < kTdPace; i++) {
                    asm volatile("nop");
                }
            }
        }
        engine->own[kTnPasses] = engine->passes;
        engine->own[kTnCommands] = engine->commands;
        engine->own[kTnPosts] = engine->posts;
        engine->own[kTnRuns] = engine->runs;
        engine->own[kTnState] = 2;
    }

}  // namespace td
