/* The stream engine on core 1 (komorebi/abi/km_stream.h): the check of an upload, the
 * interpreter, which runs each stream in place, and the source's side of its ring. */
#pragma once

#include "kernel.h"

/* A stream's state on core 1. */
#define KMD_S_FREE 0u
#define KMD_S_RUNNING 1u
#define KMD_S_ENDING 2u /* its end or error event waits for room in the ring */

/* A stream; it runs in place in the upload region. */
typedef struct {
    uint32_t registers[KM_STREAM_REGISTERS];
    uint32_t state; /* KMD_S_* */
    uint32_t request, pc, steps, max_steps, op_count, command_count, offset;
    uint32_t pending; /* its commands and jobs sent and not yet answered */
    uint32_t sent;    /* command of the SUBMIT or ENQUEUE at pc, not yet known taken */
    uint32_t seen;    /* sent: core 0 consumed it in an earlier pass */
    uint32_t waking;  /* rejected busy: it sends again after wake number wake_at */
    uint32_t wake_at;
    uint32_t jobs; /* jobs it enqueued */
    uint32_t drop; /* its queued jobs are to be dropped (KMD_HANDLER_DROP owed) */
    uint32_t last_result;
    uint32_t order;                   /* activation number; turns go in this order */
    uint32_t post_status, post_ticks; /* KMD_S_ENDING: the event to post */
    uint64_t t_command; /* deadline of each command it submits, in ticks */
} kmd_stream_t;

/* A command the engine sent and the kernel has not answered, at its command number
 * modulo KMD_SENT (a command waits while its entry is taken). */
typedef struct {
    uint32_t command_number; /* 0: free */
    uint32_t request;        /* the stream's request */
    uint32_t order;          /* the stream's activation number */
    uint16_t pc;             /* the SUBMIT */
    uint8_t stream, result;
} kmd_sent_t;

#define KMD_SENT 256u /* a power of two */
#define KMD_ORPHANS KM_RING_SLOTS

/* Core 1's state, in the state area after core 0's. */
typedef struct {
    uint64_t core, run;
    const volatile kmd_t* kernel;   /* core 0's state: the host's command deadline */
    volatile uint32_t* x280_cores;  /* the core block */
    volatile const uint8_t* upload; /* the upload region */
    uint32_t upload_bytes, cores;
    /* its ring, the source's side */
    volatile uint8_t* command_ring;
    volatile const uint32_t* consumed;
    volatile const uint8_t* events;
    volatile uint32_t* read;
    volatile const uint32_t* stop;
    volatile const uint32_t* queued; /* the kernel's queue: jobs waiting */
    volatile const uint32_t* taken;  /* and the engine's jobs it took in */
    uint32_t jobs;                   /* jobs sent */
    uint32_t next;                   /* number of the next command it writes */
    uint32_t room_to;  /* the last command it may write: consumed + depth, as loaded */
    uint32_t seen;     /* events read */
    uint32_t wakes;    /* KMD_EVENT_WAKE events read */
    uint32_t told;     /* the read count last stored */
    uint32_t wrote;    /* commands written since the last fence */
    uint64_t deadline; /* ticks of the last KM_HANDLER_DEADLINE it sent (0: none) */
    kmd_stream_t streams[KM_STREAM_MAX_ACTIVE];
    uint32_t orders;
    kmd_sent_t sent[KMD_SENT];
    uint32_t orphans; /* timeouts of orphans' commands waiting to be posted */
    uint32_t orphan_request[KMD_ORPHANS], orphan_status[KMD_ORPHANS];
    uint32_t orphan_ticks[KMD_ORPHANS];
    uint64_t passes;
    uint32_t commands, posts, events_in, runs;
    uint64_t* line; /* the command being written: 8 words, 64-byte aligned */
} kmd_engine_t;

_Static_assert(
    sizeof(kmd_engine_t) <= KMD_STATE_RING - KMD_STATE_ENGINE,
    "kmd_engine_t outgrows its DRAM area"
);
_Static_assert(
    KMD_SENT >= KMD_ENGINE_DEPTH + KM_RING_SLOTS + KM_QUEUE_JOBS, "sent table too small"
);

/* What a micro-op's fields mean: registers a, b, c in use, immediate a jump or a
 * template. */
#define KMD_UOP_A 1u
#define KMD_UOP_B 2u
#define KMD_UOP_C 4u
#define KMD_UOP_JUMP 8u
#define KMD_UOP_TEMPLATE 16u
#define KMD_UOP_BAD 0x80u

/* A turn's outcome for one micro-op: done, wait for a later pass, or the stream ended.
 */
typedef enum { KMD_UOP_DONE = 0, KMD_UOP_WAIT = 1, KMD_UOP_ENDED = 2 } kmd_uop_t;

static inline uint32_t uop_fields(uint32_t op) {
    switch (op) {
        case KM_OP_SET:
        case KM_OP_RAND:
        case KM_OP_HALT:
        case KM_OP_RESULT:
        case KM_OP_EVENT:
        case KM_OP_TIME:
            return KMD_UOP_A;
        case KM_OP_MOV:
        case KM_OP_ADDI:
        case KM_OP_SHR:
            return KMD_UOP_A | KMD_UOP_B;
        case KM_OP_ADD:
        case KM_OP_SUB:
        case KM_OP_MUL:
        case KM_OP_AND:
        case KM_OP_MOD:
            return KMD_UOP_A | KMD_UOP_B | KMD_UOP_C;
        case KM_OP_JMP:
            return KMD_UOP_JUMP;
        case KM_OP_JLT:
        case KM_OP_JEQ:
            return KMD_UOP_A | KMD_UOP_B | KMD_UOP_JUMP;
        case KM_OP_JNZ:
            return KMD_UOP_A | KMD_UOP_JUMP;
        case KM_OP_SUBMIT:
            return KMD_UOP_TEMPLATE;
        case KM_OP_ENQUEUE:
            return KMD_UOP_A | KMD_UOP_TEMPLATE;
        case KM_OP_WAIT:
            return 0;
        default:
            return KMD_UOP_BAD;
    }
}

/* Bytes of `items` items of `size` rounded up to whole 64-byte lines. */
static inline uint64_t stream_lines(uint64_t items, uint64_t size) {
    return (items * size + 63u) & ~(uint64_t)63u;
}

/* Template `template_index` of stream data at `base` with `op_count` micro-ops (80
 * bytes, 8-aligned). */
static inline const volatile uint64_t* stream_template(
    const volatile uint8_t* base, uint32_t op_count, uint32_t template_index
) {
    return (
        (const volatile uint64_t*)(
            base + sizeof(km_stream_t) + stream_lines(op_count, sizeof(km_uop_t)) +
            (uint64_t)template_index * sizeof(km_template_t)
        )
    );
}

/* Whether template words `words` (km_template_t as ten 64-bit words) break the rules.
 */
static inline int template_bad(const volatile uint64_t* words) {
    const uint32_t header = (uint32_t)(words[0] >> 32), handler = header & 0xFFu;
    const uint64_t patch = words[8], tail = words[9];
    const uint32_t patch_count = (uint32_t)(tail >> 16) & 0xFFu;
    if (
        handler < 1u || handler > KM_HANDLER_USER_MAX ||
        (((header >> 8) & 0xFFu) & ~(KM_FLAG_EXCLUSIVE | KM_FLAG_RESULT)) ||
        patch_count > KM_TEMPLATE_PATCHES
    ) {
        return 1;
    }
    for (uint32_t i = 0; i < patch_count; i++) {
        const uint32_t arg = (uint32_t)(patch >> (8u * i)) & 0xFFu;
        const uint32_t register_number = (uint32_t)(patch >> (32u + 8u * i)) & 0xFFu;
        if (arg >= KM_TEMPLATE_ARGS || register_number >= KM_STREAM_REGISTERS) {
            return 1;
        }
    }
    for (uint32_t i = 0; i < 2; i++) {
        const uint32_t register_number = (uint32_t)(tail >> (8u * i)) & 0xFFu;
        if (
            register_number != KM_TEMPLATE_NONE &&
            register_number >= KM_STREAM_REGISTERS
        ) {
            return 1;
        }
    }
    return 0;
}

/* The check of the upload of `bytes` at `base` (whole lines inside the region): 0, or
 * reason | pc << 8. Header, then micro-ops, then templates; the first fault wins. */
static uint32_t stream_check(const volatile uint8_t* base, uint32_t bytes) {
    const volatile uint32_t* header = (const volatile uint32_t*)base;
    if (bytes < sizeof(km_stream_t) || (bytes & 63u)) {
        return KM_STREAM_ERROR_HEADER;
    }
    x280_fence_ir();
    const uint32_t op_count = header[2], command_count = header[3];
    if (
        header[0] != KM_STREAM_MAGIC || header[1] != KM_STREAM_VERSION ||
        header[7] == 0u || op_count == 0u || op_count > KM_STREAM_MAX_OPS ||
        command_count > KM_STREAM_MAX_OPS || header[4] == 0u
    ) {
        return KM_STREAM_ERROR_HEADER;
    }
    const uint64_t size = (
        sizeof(km_stream_t) + stream_lines(op_count, sizeof(km_uop_t)) +
        stream_lines(command_count, sizeof(km_template_t))
    );
    if (size != bytes) {
        return KM_STREAM_ERROR_HEADER;
    }
    const volatile uint64_t* ops = (
        (const volatile uint64_t*)(base + sizeof(km_stream_t))
    );
    for (uint32_t pc = 0; pc < op_count; pc++) {
        const uint64_t word = ops[2u * pc];
        const uint32_t fields = uop_fields((uint32_t)word & 0xFFu);
        const uint32_t immediate = (uint32_t)(word >> 32);
        if (fields & KMD_UOP_BAD) {
            return KM_STREAM_ERROR_OPCODE | pc << 8;
        }
        for (uint32_t i = 0; i < 3; i++) {
            if (
                (fields & (1u << i)) &&
                ((uint32_t)(word >> (8u + 8u * i)) & 0xFFu) >= 16u
            ) {
                return KM_STREAM_ERROR_REGISTER | pc << 8;
            }
        }
        if ((fields & KMD_UOP_JUMP) && immediate >= op_count) {
            return KM_STREAM_ERROR_JUMP | pc << 8;
        }
        if ((fields & KMD_UOP_TEMPLATE) && immediate >= command_count) {
            return KM_STREAM_ERROR_TEMPLATE | pc << 8;
        }
    }
    for (
        uint32_t template_index = 0; template_index < command_count; template_index++
    ) {
        if (!template_bad(stream_template(base, op_count, template_index))) {
            continue;
        }
        /* Reported at the template's first SUBMIT or ENQUEUE, 0 if it has none. */
        for (uint32_t pc = 0; pc < op_count; pc++) {
            const uint64_t word = ops[2u * pc];
            const uint32_t op = (uint32_t)word & 0xFFu;
            if (
                (op == KM_OP_SUBMIT || op == KM_OP_ENQUEUE) &&
                (uint32_t)(word >> 32) == template_index
            ) {
                return KM_STREAM_ERROR_TEMPLATE | pc << 8;
            }
        }
        return KM_STREAM_ERROR_TEMPLATE;
    }
    return 0;
}

/* Whether commands next .. next + n - 1 fit the ring (the consumed count is loaded
 * only when needed). */
static inline int eng_room(kmd_engine_t* engine, uint32_t count) {
    if ((int32_t)(engine->room_to - (engine->next + count - 1u)) >= 0) {
        return 1;
    }
    x280_fence_ir();
    engine->room_to = *engine->consumed + KMD_ENGINE_DEPTH;
    return (int32_t)(engine->room_to - (engine->next + count - 1u)) >= 0;
}

/* Writes e->line as command e->next: its number in sequence and the three landing
 * words, one 64-byte store. The caller checked the room. */
static inline void eng_write(kmd_engine_t* engine) {
    uint64_t* line = engine->line;
    const uint32_t number = engine->next++;
    line[0] = (line[0] & ~(uint64_t)0xFFFFFFFFu) | number;
    for (uint32_t word = 2; word < 8; word += 2) {
        line[word] = (line[word] & ~(uint64_t)0xFFFFFFFFu) | number;
    }
    kmd_store_line(
        engine->command_ring + (number % KMD_ENGINE_DEPTH) * sizeof(km_command_t), line
    );
    engine->wrote++;
}

/* A command to the kernel with handler `handler` and arguments a0, a1, a2 (no group).
 */
static inline void eng_command(
    kmd_engine_t* engine, uint32_t handler, uint32_t a0, uint32_t a1, uint32_t a2
) {
    uint64_t* line = engine->line;
    line[0] = (uint64_t)handler << 32;
    line[1] = (uint64_t)a0 << 32;
    line[2] = (uint64_t)a1 << 32;
    line[3] = a2;
    for (uint32_t word = 4; word < 8; word++) {
        line[word] = 0;
    }
    eng_write(engine);
}

/* Posts the host event {request, status, ticks} through the kernel, if the ring has
 * room. Returns 1 when posted. */
static inline int eng_post(
    kmd_engine_t* engine, uint32_t request, uint32_t status, uint32_t ticks
) {
    if (!eng_room(engine, 1)) {
        return 0;
    }
    eng_command(engine, KMD_HANDLER_POST, request, status, ticks);
    engine->posts++;
    return 1;
}

/* Ends stream `stream` with KM_EVENT_STREAM_ERROR: `reason` at program counter `pc`.
 * Its commands still in flight become orphans. */
static inline void stream_fail(kmd_stream_t* stream, uint32_t reason, uint32_t pc) {
    stream->state = KMD_S_ENDING;
    stream->drop = stream->jobs != 0;
    stream->post_status = KM_EVENT_STREAM_ERROR | (reason | pc << 8) << 8;
    stream->post_ticks = 0;
}

/* Sends what is owed: ending streams' drops and end or error events, orphans'
 * timeouts. */
static void eng_owed(kmd_engine_t* engine) {
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        kmd_stream_t* stream = &engine->streams[i];
        if (stream->state != KMD_S_ENDING) {
            continue;
        }
        if (stream->drop && eng_room(engine, 1)) {
            eng_command(engine, KMD_HANDLER_DROP, stream->order & 0xFFFFu, 0, 0);
            stream->drop = 0;
        }
        if (
            !stream->drop &&
            eng_post(engine, stream->request, stream->post_status, stream->post_ticks)
        ) {
            stream->state = KMD_S_FREE;
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

/* Whether anything waits to be posted. */
static inline int eng_owes(const kmd_engine_t* engine) {
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        if (engine->streams[i].state == KMD_S_ENDING) {
            return 1;
        }
    }
    return engine->orphans != 0;
}

/* The graph executor beside the streams; it writes the ring with the code above. */
#include "graph.h"

/* RUN from the kernel: the upload of `bytes` at `offset`, admitted (in the region, line
 * aligned, no overlap, a new request). Starts it, or ends it at once. */
static void stream_start(kmd_engine_t* engine, uint32_t offset, uint32_t bytes) {
    kmd_stream_t* stream = 0;
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE && stream == 0; i++) {
        stream = engine->streams[i].state == KMD_S_FREE ? &engine->streams[i] : 0;
    }
    if (
        stream == 0 || offset > engine->upload_bytes ||
        bytes > engine->upload_bytes - offset || bytes < sizeof(km_stream_t)
    ) {
        return; /* the kernel admits neither */
    }
    const volatile uint32_t* header = (
        (const volatile uint32_t*)(engine->upload + offset)
    );
    x280_fence_ir();
    stream->request = header[6];
    stream->order = ++engine->orders;
    const uint32_t error = stream_check(engine->upload + offset, bytes);
    if (error != 0) {
        stream->state = KMD_S_ENDING;
        stream->post_status = KM_EVENT_STREAM_ERROR | error << 8;
        stream->post_ticks = 0;
        return;
    }
    for (
        uint32_t register_number = 0; register_number < KM_STREAM_REGISTERS;
        register_number++
    ) {
        stream->registers[register_number] = 0;
    }
    stream->registers[KM_STREAM_SEED_REGISTER] = header[7];
    stream->pc = (
        stream->steps = stream
                            ->pending = stream
                                            ->last_result = stream
                                                                ->jobs = stream
                                                                             ->drop = 0
    );
    stream->sent = stream->seen = stream->waking = 0;
    stream->op_count = header[2];
    stream->command_count = header[3];
    stream->max_steps = header[4];
    stream->t_command = (
        header[5]
            ? (uint64_t)header[5] * 50u
            : engine->kernel->ring[KMD_SRC_HOST].t_command
    );
    stream->offset = offset;
    stream->state = KMD_S_RUNNING;
    engine->runs++;
}

/* One event from the kernel: RUN, CANCEL, or the answer to a command of the engine. */
static void eng_event(
    kmd_engine_t* engine, uint32_t request, uint32_t status, uint32_t ticks
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
            kmd_stream_t* stream = &engine->streams[i];
            if (stream->state == KMD_S_RUNNING && stream->request == ticks) {
                stream_fail(stream, KM_STREAM_ERROR_CANCELLED, stream->pc);
            }
        }
        return;
    }
    kmd_sent_t* sent_command = &engine->sent[request % KMD_SENT];
    if (sent_command->command_number != request) {
        return; /* not a command: nothing waits for it */
    }
    sent_command->command_number = 0;
    kmd_stream_t* stream = (
        &engine->streams[sent_command->stream % KM_STREAM_MAX_ACTIVE]
    );
    const int live = (
        stream->state == KMD_S_RUNNING && stream->order == sent_command->order
    );
    if (kind == KM_EVENT_DONE) {
        if (live) {
            stream->pending--;
            stream->last_result = sent_command->result ? ticks : stream->last_result;
        }
    } else if (kind == KM_EVENT_REJECTED && status >> 8 == KMD_REJECT_BUSY) {
        /* Not taken: the stream sends it again after the next wake. */
        if (live) {
            stream->pending--;
            stream->sent = stream->seen = 0;
            stream->waking = 1;
            stream->wake_at = engine->wakes;
        }
    } else if (kind == KM_EVENT_TIMEOUT) {
        if (live) {
            stream_fail(stream, KM_STREAM_ERROR_DEADLINE, sent_command->pc);
        } else if (engine->orphans < KMD_ORPHANS) {
            /* An orphan: the ordinary timeout event, with its stream's request. */
            engine->orphan_request[engine->orphans] = sent_command->request;
            engine->orphan_status[engine->orphans] = status;
            engine->orphan_ticks[engine->orphans] = ticks;
            engine->orphans++;
        }
    } else if (live) {
        const uint32_t why = (
            status >> 8 == KMD_REJECT_GROUP
                ? KM_STREAM_ERROR_GROUP
                : KM_STREAM_ERROR_TEMPLATE
        );
        stream_fail(stream, why, sent_command->pc);
    }
}

/* Reads every event the kernel has pushed (index, fields, index again, as the host
 * does), then stores the read count. */
static void eng_events(kmd_engine_t* engine) {
    for (uint32_t events_read = 0; events_read < KMD_ENGINE_EVENTS; events_read++) {
        const uint32_t number = engine->seen + 1u;
        volatile const uint32_t*
            words = (
                (volatile const uint32_t*)(
                    engine->events + (number % KMD_ENGINE_EVENTS) * sizeof(km_event_t)
                )
            );
        x280_fence_ir();
        if (words[0] != number) {
            break;
        }
        x280_fence_ir();
        const uint32_t request = words[1], status = words[2], ticks = words[3];
        x280_fence_ir();
        if (words[0] != number) {
            break;
        }
        engine->seen = number;
        engine->events_in++;
        eng_event(engine, request, status, ticks);
    }
    if (engine->seen != engine->told) {
        *engine->read = engine->seen;
        engine->told = engine->seen;
        engine->wrote++;
    }
}

/* SUBMIT (`job` 0) or ENQUEUE of `job` cores, template `template_index` at `pc`:
 * checked, then sent (DEADLINE first if changed) if there is room; else it waits. */
static kmd_uop_t stream_command(
    kmd_engine_t* engine,
    kmd_stream_t* stream,
    uint32_t template_index,
    uint32_t pc,
    uint32_t job
) {
    if (job != 0 && job > engine->cores) {
        stream_fail(stream, KM_STREAM_ERROR_NEVER_FITS, pc);
        return KMD_UOP_ENDED;
    }
    if (template_index >= stream->command_count) {
        stream_fail(stream, KM_STREAM_ERROR_TEMPLATE, pc);
        return KMD_UOP_ENDED;
    }
    const volatile uint64_t* template_words = (
        stream_template(
            engine->upload + stream->offset, stream->op_count, template_index
        )
    );
    const uint64_t patch = template_words[8], tail = template_words[9];
    const uint32_t group = (uint32_t)template_words[1];
    const uint32_t first_register = (uint32_t)tail & 0xFFu;
    const uint32_t count_register = (uint32_t)(tail >> 8) & 0xFFu;
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
        /* a work cluster's thread takes no exclusive command */
        const uint32_t exclusive = (
            (uint32_t)(template_words[0] >> 40) & KM_FLAG_EXCLUSIVE
        );
        if (
            KM_GROUP_IS_CLUSTER(first, count)
                ? exclusive != 0
                : count == 0 || first > engine->cores || count > engine->cores - first
        ) {
            stream_fail(stream, KM_STREAM_ERROR_GROUP, pc);
            return KMD_UOP_ENDED;
        }
    } else if (*engine->queued + (engine->jobs - *engine->taken) >= KM_QUEUE_JOBS) {
        return KMD_UOP_WAIT; /* jobs in the queue and on their way to it */
    }
    const uint32_t lines = stream->t_command != engine->deadline ? 2u : 1u;
    const uint32_t command_number = engine->next + lines - 1u;
    if (
        !eng_room(engine, lines) ||
        engine->sent[command_number % KMD_SENT].command_number != 0
    ) {
        return KMD_UOP_WAIT;
    }
    if (lines == 2u) {
        const uint64_t microseconds = stream->t_command / 50u;
        eng_command(
            engine,
            KM_HANDLER_DEADLINE,
            microseconds == 0 ? 1u : (uint32_t)microseconds,
            0,
            0
        );
        engine->deadline = stream->t_command;
    }
    uint64_t* line = engine->line;
    for (uint32_t i = 0; i < 8; i++) {
        line[i] = template_words[i];
    }
    uint32_t* words = (uint32_t*)line;
    words[2] = first | count << 16;
    if (job != 0) {
        /* The kernel's queue: KM_FLAG_QUEUE, and the stream's tag in the slot bits. */
        words[1] = (
            (words[1] & 0xFFFFu) | KM_FLAG_QUEUE << 8 | (stream->order & 0xFFFFu) << 16
        );
        engine->jobs++;
        stream->jobs++;
    }
    /* Word of argument a0 to a9 in km_command_t. */
    static const uint8_t arg_word[KM_TEMPLATE_ARGS] = {
        3, 5, 6, 7, 9, 10, 11, 13, 14, 15
    };
    const uint32_t patch_count = (uint32_t)(tail >> 16) & 0xFFu;
    for (uint32_t i = 0; i < patch_count && i < KM_TEMPLATE_PATCHES; i++) {
        const uint32_t arg = (uint32_t)(patch >> (8u * i)) & 0xFFu;
        const uint32_t register_number = (uint32_t)(patch >> (32u + 8u * i)) & 15u;
        if (arg < KM_TEMPLATE_ARGS) {
            words[arg_word[arg]] = stream->registers[register_number];
        }
    }
    eng_write(engine);
    kmd_sent_t* sent_command = &engine->sent[command_number % KMD_SENT];
    sent_command->command_number = command_number;
    sent_command->request = stream->request;
    sent_command->order = stream->order;
    sent_command->pc = (uint16_t)pc;
    sent_command->stream = (uint8_t)(stream - engine->streams);
    sent_command->result = ((words[1] >> 8) & KM_FLAG_RESULT) != 0;
    engine->commands++;
    stream->pending++;
    stream->sent = command_number;
    return KMD_UOP_WAIT;
}

/* Whether an event waits to be read. */
static inline int eng_unread(const kmd_engine_t* engine) {
    const uint32_t number = engine->seen + 1u;
    x280_fence_ir();
    return (
        *(volatile const uint32_t*)(
            engine->events + (number % KMD_ENGINE_EVENTS) * 16u
        ) == number
    );
}

/* SUBMIT or ENQUEUE at pc: sent, then done once core 0 took it. Core 0 pushes a busy
 * rejection before it counts the command consumed, so no unread event means taken. */
static kmd_uop_t stream_send(
    kmd_engine_t* engine,
    kmd_stream_t* stream,
    uint32_t template_index,
    uint32_t pc,
    uint32_t job
) {
    if (stream->sent == 0) {
        if (stream->waking && engine->wakes == stream->wake_at) {
            return KMD_UOP_WAIT;
        }
        stream->waking = 0;
        return stream_command(engine, stream, template_index, pc, job);
    }
    if (!stream->seen) {
        x280_fence_ir();
        if ((int32_t)(*engine->consumed - stream->sent) < 0) {
            return KMD_UOP_WAIT;
        }
        if (eng_unread(engine)) {
            /* The next pass reads those events first. */
            stream->seen = 1;
            return KMD_UOP_WAIT;
        }
    }
    stream->sent = stream->seen = 0;
    return KMD_UOP_DONE;
}

/* One micro-op of stream `stream` at its pc (KM_STREAM_ERROR_JUMP if past the end,
 * KM_STREAM_ERROR_STEPS after max_steps); a wait is no step. */
static kmd_uop_t stream_op(kmd_engine_t* engine, kmd_stream_t* stream) {
    const uint32_t pc = stream->pc;
    if (pc >= stream->op_count) {
        stream_fail(stream, KM_STREAM_ERROR_JUMP, stream->op_count);
        return KMD_UOP_ENDED;
    }
    if (stream->steps >= stream->max_steps) {
        stream_fail(stream, KM_STREAM_ERROR_STEPS, pc);
        return KMD_UOP_ENDED;
    }
    const volatile uint64_t* ops = (
        (const volatile uint64_t*)(
            engine->upload + stream->offset + sizeof(km_stream_t)
        )
    );
    const uint64_t word = ops[2u * pc];
    const uint32_t op = (uint32_t)word & 0xFFu, immediate = (uint32_t)(word >> 32);
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
                return KMD_UOP_ENDED;
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
            /* An implicit WAIT 0, then the end event: auxiliary the low 24 bits, ticks
             * all. */
            if (stream->pending != 0) {
                return KMD_UOP_WAIT;
            }
            stream->state = KMD_S_ENDING;
            stream->post_status = KM_EVENT_STREAM_END | (*register_a & 0xFFFFFFu) << 8;
            stream->post_ticks = *register_a;
            return KMD_UOP_ENDED;
        case KM_OP_SUBMIT:
        case KM_OP_ENQUEUE: {
            if (op == KM_OP_ENQUEUE && *register_a == 0) {
                stream_fail(stream, KM_STREAM_ERROR_NEVER_FITS, pc);
                return KMD_UOP_ENDED;
            }
            const uint32_t job = op == KM_OP_ENQUEUE ? *register_a : 0u;
            const kmd_uop_t outcome = stream_send(engine, stream, immediate, pc, job);
            if (outcome != KMD_UOP_DONE) {
                return outcome;
            }
            break;
        }
        case KM_OP_WAIT:
            if (stream->pending > immediate) {
                return KMD_UOP_WAIT;
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
                return KMD_UOP_WAIT;
            }
            break;
        case KM_OP_TIME:
            *register_a = (uint32_t)x280_mtime();
            break;
        default:
            stream_fail(stream, KM_STREAM_ERROR_OPCODE, pc);
            return KMD_UOP_ENDED;
    }
    stream->steps++;
    stream->pc = next;
    return KMD_UOP_DONE;
}

/* The streams' part of a pass: each running stream runs up to KM_STEP_OPS micro-ops,
 * in activation order starting at stream `pass % running` (submitters take turns). */
static void eng_step(kmd_engine_t* engine) {
    kmd_stream_t* turn[KM_STREAM_MAX_ACTIVE];
    uint32_t running = 0;
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        kmd_stream_t* stream = &engine->streams[i];
        if (stream->state != KMD_S_RUNNING) {
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
        return;
    }
    const uint32_t start = (uint32_t)(engine->passes % running);
    for (uint32_t turn_index = 0; turn_index < running; turn_index++) {
        kmd_stream_t* stream = turn[(start + turn_index) % running];
        for (uint32_t op = 0; op < KM_STEP_OPS; op++) {
            if (stream_op(engine, stream) != KMD_UOP_DONE) {
                break;
            }
        }
    }
}

/* Events, owed posts, streams, one fence per pass; on either stop flag it posts what it
 * still owes for at most half of KMD_ENGINE_STOP_TICKS. */
static void eng_loop(kmd_engine_t* engine) {
    uint32_t spins = 0;
    for (;;) {
        engine->passes++;
        eng_events(engine);
        eng_owed(engine);
        eng_step(engine);
        graph_step(engine);
        eng_owed(engine);
        if (engine->wrote) {
            x280_fence_ow();
            engine->wrote = 0;
        }
        x280_fence_ir();
        if (*engine->stop != 0) {
            break;
        }
        if (++spins == KMD_POLL_EVERY) {
            spins = 0;
            engine->x280_cores[KMD_H_MARKER1] = (uint32_t)(engine->passes >> 10);
            if (x280_hart_stop_requested(engine->core, engine->run)) {
                break;
            }
        }
    }
    const uint64_t deadline = x280_mtime() + KMD_ENGINE_STOP_TICKS / 2u;
    do {
        eng_events(engine);
        eng_owed(engine);
        graph_owed(engine);
        x280_fence_ow();
    } while ((eng_owes(engine) || graph_owes(engine)) && x280_mtime() < deadline);
}

/* Core 1: waits for the kernel's go (timeout: twice t_start, or the runtime's stop),
 * runs the engine's loop, and reports it stopped in the core block. */
static void eng_run(uint64_t core, const volatile kmd_args_t* arguments, uint64_t run) {
    volatile uint8_t* state = (
        (volatile uint8_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + arguments->state_ddr)
    );
    volatile uint8_t* ring = state + KMD_STATE_RING;
    volatile uint32_t* x280_cores = (
        (volatile uint32_t*)((volatile uint8_t*)arguments + KMD_LIM_CORES)
    );
    volatile const uint32_t* go = (volatile const uint32_t*)(ring + KMD_ERING_GO);
    x280_cores[KMD_H_ENGINE] = KMD_ENGINE_WAITING;
    const uint64_t deadline = x280_mtime() + 2u * arguments->t_start;
    uint32_t spins = 0, started = 0;
    for (;;) {
        x280_fence_ir();
        if (*(volatile const uint64_t*)go == run) {
            x280_fence_ir();
            started = go[2] == 1u;
            break;
        }
        if (++spins == KMD_POLL_EVERY) {
            spins = 0;
            if (x280_hart_stop_requested(core, run) || x280_mtime() > deadline) {
                break;
            }
        }
    }
    if (!started) {
        x280_cores[KMD_H_ENGINE] = KMD_ENGINE_UNUSED;
        x280_fence_ow();
        return;
    }
    uint64_t line_storage[16];
    kmd_engine_t* engine = (kmd_engine_t*)(uintptr_t)(state + KMD_STATE_ENGINE);
    engine->core = core;
    engine->run = run;
    engine->kernel = (const volatile kmd_t*)(state + KMD_STATE_KERNEL);
    engine->x280_cores = x280_cores;
    engine->upload = (volatile const uint8_t*)arguments + KMD_LIM_UPLOAD;
    engine->upload_bytes = (uint32_t)arguments->upload_bytes;
    engine->cores = (uint32_t)arguments->cores;
    engine->command_ring = ring + KMD_ERING_CMDS;
    engine->consumed = (volatile const uint32_t*)(ring + KMD_ERING_CONSUMED);
    engine->events = ring + KMD_ERING_EVENTS;
    engine->read = (volatile uint32_t*)(ring + KMD_ERING_READ);
    engine->stop = (volatile const uint32_t*)(ring + KMD_ERING_STOP);
    engine->queued = (volatile const uint32_t*)(ring + KMD_ERING_QUEUED);
    engine->taken = (volatile const uint32_t*)(ring + KMD_ERING_TAKEN);
    engine->next = 1;
    engine->room_to = KMD_ENGINE_DEPTH;
    engine->line = (uint64_t*)(((uintptr_t)line_storage + 63u) & ~(uintptr_t)63u);
    x280_cores[KMD_H_ENGINE] = KMD_ENGINE_RUNNING;
    eng_loop(engine);
    x280_cores[KMD_H_PASSES] = (uint32_t)engine->passes;
    x280_cores[KMD_H_COMMANDS] = engine->commands;
    x280_cores[KMD_H_POSTS] = engine->posts;
    x280_cores[KMD_H_EVENTS] = engine->events_in;
    x280_cores[KMD_H_RUNS] = engine->runs;
    kmd_flush_lines(state + KMD_STATE_ENGINE, KMD_STATE_BYTES - KMD_STATE_ENGINE);
    x280_fence_ow();
    x280_cores[KMD_H_ENGINE] = KMD_ENGINE_STOPPED;
    x280_fence_ow();
}
