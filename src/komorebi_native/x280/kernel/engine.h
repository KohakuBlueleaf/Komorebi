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
    uint32_t regs[KM_STREAM_REGS];
    uint32_t state; /* KMD_S_* */
    uint32_t req, pc, steps, max_steps, n_ops, n_records, off;
    uint32_t pending; /* its records and jobs sent and not yet answered */
    uint32_t jobs;    /* jobs it submitted */
    uint32_t drop;    /* its queued jobs are to be dropped (KMD_SEG_DROP owed) */
    uint32_t last_result;
    uint32_t order;                   /* activation number; turns go in this order */
    uint32_t post_status, post_ticks; /* KMD_S_ENDING: the event to post */
    uint64_t t_record;                /* deadline of each record it issues, in ticks */
} kmd_stream_t;

/* A record the engine sent and the kernel has not answered, at its command number
 * modulo KMD_SENT (a record waits while its entry is taken). */
typedef struct {
    uint32_t cmd;   /* 0: free */
    uint32_t req;   /* the stream's req */
    uint32_t order; /* the stream's activation number */
    uint16_t pc;    /* the ISSUE */
    uint8_t stream, result;
} kmd_sent_t;

#define KMD_SENT 256u /* a power of two */
#define KMD_ORPHANS KM_RING_SLOTS

/* Core 1's state, in the state area after core 0's. */
typedef struct {
    uint64_t hart, run;
    const volatile kmd_t* kernel;   /* core 0's state: the host's record deadline */
    volatile uint32_t* harts;       /* the hart block */
    volatile const uint8_t* upload; /* the upload region */
    uint32_t upload_bytes, cores;
    /* its ring, the source's side */
    volatile uint8_t* cmds;
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
    uint32_t told;     /* the read count last stored */
    uint32_t wrote;    /* commands written since the last fence */
    uint64_t deadline; /* ticks of the last KM_SEG_DEADLINE it sent (0: none) */
    kmd_stream_t streams[KM_STREAM_MAX_ACTIVE];
    uint32_t orders;
    kmd_sent_t sent[KMD_SENT];
    uint32_t orphans; /* timeouts of orphans' records waiting to be posted */
    uint32_t orphan_req[KMD_ORPHANS], orphan_status[KMD_ORPHANS];
    uint32_t orphan_ticks[KMD_ORPHANS];
    uint64_t passes;
    uint32_t records, posts, events_in, runs;
    uint64_t* line; /* the command being written: 8 words, 64-byte aligned */
} kmd_engine_t;

_Static_assert(
    sizeof(kmd_engine_t) <= KMD_STATE_RING - KMD_STATE_ENGINE,
    "kmd_engine_t outgrows its DRAM area"
);
_Static_assert(
    KMD_SENT >= KMD_ENGINE_DEPTH + KM_RING_SLOTS + KM_QUEUE_JOBS, "sent table too small"
);

/* What a micro-op's fields mean: registers a, b, c in use, imm a jump or a template. */
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
        case KM_OP_ISSUE:
            return KMD_UOP_TEMPLATE;
        case KM_OP_SUBMIT:
            return KMD_UOP_A | KMD_UOP_TEMPLATE;
        case KM_OP_WAIT:
            return 0;
        default:
            return KMD_UOP_BAD;
    }
}

/* Bytes of `n` items of `size` rounded up to whole 64-byte lines. */
static inline uint64_t stream_lines(uint64_t n, uint64_t size) {
    return (n * size + 63u) & ~(uint64_t)63u;
}

/* Template `t` of stream data at `base` with `n_ops` micro-ops (80 bytes, 8-aligned).
 */
static inline const volatile uint64_t* stream_template(
    const volatile uint8_t* base, uint32_t n_ops, uint32_t t
) {
    return (
        (const volatile uint64_t*)(base + sizeof(km_stream_t) +
                                              stream_lines(n_ops, sizeof(km_uop_t)) +
        (uint64_t)t * sizeof(km_template_t))
    );
}

/* Whether template words `w` (km_template_t as ten 64-bit words) break the rules. */
static inline int template_bad(const volatile uint64_t* w) {
    const uint32_t hdr = (uint32_t)(w[0] >> 32), seg = hdr & 0xFFu;
    const uint64_t patch = w[8], tail = w[9];
    const uint32_t n = (uint32_t)(tail >> 16) & 0xFFu;
    if (seg < 1u || seg > KM_SEG_USER_MAX ||
        (((hdr >> 8) & 0xFFu) & ~(KM_F_EXCL | KM_F_RESULT)) || n > KM_TPL_PATCHES) {
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t arg = (uint32_t)(patch >> (8u * i)) & 0xFFu;
        const uint32_t reg = (uint32_t)(patch >> (32u + 8u * i)) & 0xFFu;
        if (arg >= KM_TPL_ARGS || reg >= KM_STREAM_REGS) {
            return 1;
        }
    }
    for (uint32_t i = 0; i < 2; i++) {
        const uint32_t r = (uint32_t)(tail >> (8u * i)) & 0xFFu;
        if (r != KM_TPL_NONE && r >= KM_STREAM_REGS) {
            return 1;
        }
    }
    return 0;
}

/* The check of the upload of `bytes` at `base` (whole lines inside the region): 0, or
 * reason | pc << 8. Header, then micro-ops, then templates; the first fault wins. */
static uint32_t stream_check(const volatile uint8_t* base, uint32_t bytes) {
    const volatile uint32_t* h = (const volatile uint32_t*)base;
    if (bytes < sizeof(km_stream_t) || (bytes & 63u)) {
        return KM_SERR_HEADER;
    }
    x280_fence_ir();
    const uint32_t n_ops = h[2], n_records = h[3];
    if (h[0] != KM_STREAM_MAGIC || h[1] != KM_STREAM_VERSION || h[7] == 0u ||
        n_ops == 0u || n_ops > KM_STREAM_MAX_OPS || n_records > KM_STREAM_MAX_OPS ||
        h[4] == 0u) {
        return KM_SERR_HEADER;
    }
    const uint64_t size =
        (sizeof(km_stream_t) + stream_lines(n_ops, sizeof(km_uop_t)) +
         stream_lines(n_records, sizeof(km_template_t)));
    if (size != bytes) {
        return KM_SERR_HEADER;
    }
    const volatile uint64_t* ops = (
        (const volatile uint64_t*)(base + sizeof(km_stream_t))
    );
    for (uint32_t pc = 0; pc < n_ops; pc++) {
        const uint64_t w = ops[2u * pc];
        const uint32_t f = uop_fields((uint32_t)w & 0xFFu), imm = (uint32_t)(w >> 32);
        if (f & KMD_UOP_BAD) {
            return KM_SERR_OPCODE | pc << 8;
        }
        for (uint32_t i = 0; i < 3; i++) {
            if ((f & (1u << i)) && ((uint32_t)(w >> (8u + 8u * i)) & 0xFFu) >= 16u) {
                return KM_SERR_REGISTER | pc << 8;
            }
        }
        if ((f & KMD_UOP_JUMP) && imm >= n_ops) {
            return KM_SERR_JUMP | pc << 8;
        }
        if ((f & KMD_UOP_TEMPLATE) && imm >= n_records) {
            return KM_SERR_TEMPLATE | pc << 8;
        }
    }
    for (uint32_t t = 0; t < n_records; t++) {
        if (!template_bad(stream_template(base, n_ops, t))) {
            continue;
        }
        /* Reported at the template's first ISSUE or SUBMIT, 0 if it has none. */
        for (uint32_t pc = 0; pc < n_ops; pc++) {
            const uint64_t w = ops[2u * pc];
            const uint32_t op = (uint32_t)w & 0xFFu;
            if ((op == KM_OP_ISSUE || op == KM_OP_SUBMIT) && (uint32_t)(w >> 32) == t) {
                return KM_SERR_TEMPLATE | pc << 8;
            }
        }
        return KM_SERR_TEMPLATE;
    }
    return 0;
}

/* Whether commands next .. next + n - 1 fit the ring (the consumed count is loaded
 * only when needed). */
static inline int eng_room(kmd_engine_t* e, uint32_t n) {
    if ((int32_t)(e->room_to - (e->next + n - 1u)) >= 0) {
        return 1;
    }
    x280_fence_ir();
    e->room_to = *e->consumed + KMD_ENGINE_DEPTH;
    return (int32_t)(e->room_to - (e->next + n - 1u)) >= 0;
}

/* Writes e->line as command e->next: its number in seq and the three landing words,
 * one 64-byte store. The caller checked the room. */
static inline void eng_write(kmd_engine_t* e) {
    uint64_t* l = e->line;
    const uint32_t i = e->next++;
    l[0] = (l[0] & ~(uint64_t)0xFFFFFFFFu) | i;
    for (uint32_t j = 2; j < 8; j += 2) {
        l[j] = (l[j] & ~(uint64_t)0xFFFFFFFFu) | i;
    }
    kmd_store_line(e->cmds + (i % KMD_ENGINE_DEPTH) * sizeof(km_record_t), l);
    e->wrote++;
}

/* A command to the kernel with segment `seg` and arguments a0, a1, a2 (no team). */
static inline void eng_command(
    kmd_engine_t* e, uint32_t seg, uint32_t a0, uint32_t a1, uint32_t a2
) {
    uint64_t* l = e->line;
    l[0] = (uint64_t)seg << 32;
    l[1] = (uint64_t)a0 << 32;
    l[2] = (uint64_t)a1 << 32;
    l[3] = a2;
    for (uint32_t j = 4; j < 8; j++) {
        l[j] = 0;
    }
    eng_write(e);
}

/* Posts the host event {req, status, ticks} through the kernel, if the ring has room.
 * Returns 1 when posted. */
static inline int eng_post(
    kmd_engine_t* e, uint32_t req, uint32_t status, uint32_t ticks
) {
    if (!eng_room(e, 1)) {
        return 0;
    }
    eng_command(e, KMD_SEG_POST, req, status, ticks);
    e->posts++;
    return 1;
}

/* Ends stream `s` with KM_EV_STREAM_ERROR: `reason` at program counter `pc`. Its
 * records still in flight become orphans. */
static inline void stream_fail(kmd_stream_t* s, uint32_t reason, uint32_t pc) {
    s->state = KMD_S_ENDING;
    s->drop = s->jobs != 0;
    s->post_status = KM_EV_STREAM_ERROR | (reason | pc << 8) << 8;
    s->post_ticks = 0;
}

/* Sends what is owed: ending streams' drops and end or error events, orphans'
 * timeouts. */
static void eng_owed(kmd_engine_t* e) {
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        kmd_stream_t* s = &e->streams[i];
        if (s->state != KMD_S_ENDING) {
            continue;
        }
        if (s->drop && eng_room(e, 1)) {
            eng_command(e, KMD_SEG_DROP, s->order & 0xFFFFu, 0, 0);
            s->drop = 0;
        }
        if (!s->drop && eng_post(e, s->req, s->post_status, s->post_ticks)) {
            s->state = KMD_S_FREE;
        }
    }
    while (e->orphans != 0) {
        const uint32_t j = e->orphans - 1u;
        if (!eng_post(e, e->orphan_req[j], e->orphan_status[j], e->orphan_ticks[j])) {
            return;
        }
        e->orphans--;
    }
}

/* Whether anything waits to be posted. */
static inline int eng_owes(const kmd_engine_t* e) {
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        if (e->streams[i].state == KMD_S_ENDING) {
            return 1;
        }
    }
    return e->orphans != 0;
}

/* RUN from the kernel: the upload of `bytes` at `off`, which the kernel admitted (in
 * the region, line aligned, no overlap, a new req). Starts it, or ends it at once. */
static void stream_start(kmd_engine_t* e, uint32_t off, uint32_t bytes) {
    kmd_stream_t* s = 0;
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE && s == 0; i++) {
        s = e->streams[i].state == KMD_S_FREE ? &e->streams[i] : 0;
    }
    if (s == 0 || off > e->upload_bytes || bytes > e->upload_bytes - off ||
        bytes < sizeof(km_stream_t)) {
        return; /* the kernel admits neither */
    }
    const volatile uint32_t* h = (const volatile uint32_t*)(e->upload + off);
    x280_fence_ir();
    s->req = h[6];
    s->order = ++e->orders;
    const uint32_t err = stream_check(e->upload + off, bytes);
    if (err != 0) {
        s->state = KMD_S_ENDING;
        s->post_status = KM_EV_STREAM_ERROR | err << 8;
        s->post_ticks = 0;
        return;
    }
    for (uint32_t r = 0; r < KM_STREAM_REGS; r++) {
        s->regs[r] = 0;
    }
    s->regs[KM_STREAM_SEED_REG] = h[7];
    s->pc = s->steps = s->pending = s->last_result = s->jobs = s->drop = 0;
    s->n_ops = h[2];
    s->n_records = h[3];
    s->max_steps = h[4];
    s->t_record = h[5] ? (uint64_t)h[5] * 50u : e->kernel->ring[KMD_SRC_HOST].t_record;
    s->off = off;
    s->state = KMD_S_RUNNING;
    e->runs++;
}

/* One event from the kernel: RUN, CANCEL, or the answer to a command of the engine. */
static void eng_event(kmd_engine_t* e, uint32_t req, uint32_t status, uint32_t ticks) {
    const uint32_t kind = status & 0xFFu;
    if (kind == KMD_EV_RUN) {
        stream_start(e, req, ticks);
        return;
    }
    if (kind == KMD_EV_CANCEL) {
        for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
            kmd_stream_t* s = &e->streams[i];
            if (s->state == KMD_S_RUNNING && s->req == ticks) {
                stream_fail(s, KM_SERR_CANCELLED, s->pc);
            }
        }
        return;
    }
    kmd_sent_t* x = &e->sent[req % KMD_SENT];
    if (x->cmd != req) {
        return; /* not a record: nothing waits for it */
    }
    x->cmd = 0;
    kmd_stream_t* s = &e->streams[x->stream % KM_STREAM_MAX_ACTIVE];
    const int live = s->state == KMD_S_RUNNING && s->order == x->order;
    if (kind == KM_EV_DONE) {
        if (live) {
            s->pending--;
            s->last_result = x->result ? ticks : s->last_result;
        }
    } else if (kind == KM_EV_TIMEOUT) {
        if (live) {
            stream_fail(s, KM_SERR_DEADLINE, x->pc);
        } else if (e->orphans < KMD_ORPHANS) {
            /* An orphan: the ordinary timeout event, with its stream's req. */
            e->orphan_req[e->orphans] = x->req;
            e->orphan_status[e->orphans] = status;
            e->orphan_ticks[e->orphans] = ticks;
            e->orphans++;
        }
    } else if (live) {
        const uint32_t why = (
            status >> 8 == KMD_REFUSE_TEAM ? KM_SERR_TEAM : KM_SERR_TEMPLATE
        );
        stream_fail(s, why, x->pc);
    }
}

/* Reads every event the kernel has pushed (index, fields, index again, as the host
 * does), then stores the read count. */
static void eng_events(kmd_engine_t* e) {
    for (uint32_t n = 0; n < KMD_ENGINE_EVENTS; n++) {
        const uint32_t i = e->seen + 1u;
        volatile const uint32_t*
            p = (
                (volatile const uint32_t*)(e->events +
                                                      (i % KMD_ENGINE_EVENTS) *
                sizeof(km_event_t))
            );
        x280_fence_ir();
        if (p[0] != i) {
            break;
        }
        x280_fence_ir();
        const uint32_t req = p[1], status = p[2], ticks = p[3];
        x280_fence_ir();
        if (p[0] != i) {
            break;
        }
        e->seen = i;
        e->events_in++;
        eng_event(e, req, status, ticks);
    }
    if (e->seen != e->told) {
        *e->read = e->seen;
        e->told = e->seen;
        e->wrote++;
    }
}

/* ISSUE (`job` 0) or SUBMIT of `job` cores, template `t` at `pc`: checked, then sent
 * (DEADLINE first if changed) when the ring and the queue have room, else it waits. */
static kmd_uop_t stream_record(
    kmd_engine_t* e, kmd_stream_t* s, uint32_t t, uint32_t pc, uint32_t job
) {
    if (job != 0 && job > e->cores) {
        stream_fail(s, KM_SERR_NEVER_FITS, pc);
        return KMD_UOP_ENDED;
    }
    if (t >= s->n_records) {
        stream_fail(s, KM_SERR_TEMPLATE, pc);
        return KMD_UOP_ENDED;
    }
    const volatile uint64_t* w = stream_template(e->upload + s->off, s->n_ops, t);
    const uint64_t patch = w[8], tail = w[9];
    const uint32_t team = (uint32_t)w[1], fr = (uint32_t)tail & 0xFFu;
    const uint32_t cr = (uint32_t)(tail >> 8) & 0xFFu;
    uint32_t first = 0, count = job;
    if (job == 0) {
        first = fr == KM_TPL_NONE ? team & 0xFFFFu : s->regs[fr & 15u];
        count = cr == KM_TPL_NONE ? team >> 16 : s->regs[cr & 15u];
        if (count == 0 || first > e->cores || count > e->cores - first) {
            stream_fail(s, KM_SERR_TEAM, pc);
            return KMD_UOP_ENDED;
        }
    } else if (*e->queued + (e->jobs - *e->taken) >= KM_QUEUE_JOBS) {
        return KMD_UOP_WAIT; /* jobs in the queue and on their way to it */
    }
    const uint32_t lines = s->t_record != e->deadline ? 2u : 1u;
    const uint32_t cmd = e->next + lines - 1u;
    if (!eng_room(e, lines) || e->sent[cmd % KMD_SENT].cmd != 0) {
        return KMD_UOP_WAIT;
    }
    if (lines == 2u) {
        const uint64_t us = s->t_record / 50u;
        eng_command(e, KM_SEG_DEADLINE, us == 0 ? 1u : (uint32_t)us, 0, 0);
        e->deadline = s->t_record;
    }
    uint64_t* l = e->line;
    for (uint32_t i = 0; i < 8; i++) {
        l[i] = w[i];
    }
    uint32_t* words = (uint32_t*)l;
    words[2] = first | count << 16;
    if (job != 0) {
        /* The kernel's queue: KM_F_QUEUE, and the stream's tag in the slot bits. */
        words[1] = (words[1] & 0xFFFFu) | KM_F_QUEUE << 8 | (s->order & 0xFFFFu) << 16;
        e->jobs++;
        s->jobs++;
    }
    /* Word of argument a0 to a9 in km_record_t. */
    static const uint8_t arg_word[KM_TPL_ARGS] = {3, 5, 6, 7, 9, 10, 11, 13, 14, 15};
    const uint32_t n = (uint32_t)(tail >> 16) & 0xFFu;
    for (uint32_t i = 0; i < n && i < KM_TPL_PATCHES; i++) {
        const uint32_t arg = (uint32_t)(patch >> (8u * i)) & 0xFFu;
        const uint32_t reg = (uint32_t)(patch >> (32u + 8u * i)) & 15u;
        if (arg < KM_TPL_ARGS) {
            words[arg_word[arg]] = s->regs[reg];
        }
    }
    eng_write(e);
    kmd_sent_t* x = &e->sent[cmd % KMD_SENT];
    x->cmd = cmd;
    x->req = s->req;
    x->order = s->order;
    x->pc = (uint16_t)pc;
    x->stream = (uint8_t)(s - e->streams);
    x->result = ((words[1] >> 8) & KM_F_RESULT) != 0;
    e->records++;
    s->pending++;
    return KMD_UOP_DONE;
}

/* One micro-op of stream `s` at its pc (KM_SERR_JUMP if past the end, KM_SERR_STEPS
 * after max_steps); a micro-op that waits does not count as a step. */
static kmd_uop_t stream_op(kmd_engine_t* e, kmd_stream_t* s) {
    const uint32_t pc = s->pc;
    if (pc >= s->n_ops) {
        stream_fail(s, KM_SERR_JUMP, s->n_ops);
        return KMD_UOP_ENDED;
    }
    if (s->steps >= s->max_steps) {
        stream_fail(s, KM_SERR_STEPS, pc);
        return KMD_UOP_ENDED;
    }
    const volatile uint64_t* ops = (
        (const volatile uint64_t*)(e->upload + s->off + sizeof(km_stream_t))
    );
    const uint64_t w = ops[2u * pc];
    const uint32_t op = (uint32_t)w & 0xFFu, imm = (uint32_t)(w >> 32);
    uint32_t* r = s->regs;
    uint32_t* a = &r[(w >> 8) & 15u];
    const uint32_t b = r[(w >> 16) & 15u], c = r[(w >> 24) & 15u];
    uint32_t next = pc + 1u;
    switch (op) {
        case KM_OP_SET:
            *a = imm;
            break;
        case KM_OP_MOV:
            *a = b;
            break;
        case KM_OP_ADD:
            *a = b + c;
            break;
        case KM_OP_ADDI:
            *a = b + imm;
            break;
        case KM_OP_SUB:
            *a = b - c;
            break;
        case KM_OP_MUL:
            *a = b * c;
            break;
        case KM_OP_AND:
            *a = b & c;
            break;
        case KM_OP_SHR:
            *a = b >> (imm & 31u);
            break;
        case KM_OP_MOD:
            if (c == 0) {
                stream_fail(s, KM_SERR_MOD_ZERO, pc);
                return KMD_UOP_ENDED;
            }
            *a = b % c;
            break;
        case KM_OP_RAND: {
            uint32_t x = r[KM_STREAM_SEED_REG];
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            r[KM_STREAM_SEED_REG] = x;
            *a = x;
            break;
        }
        case KM_OP_JMP:
            next = imm;
            break;
        case KM_OP_JLT:
            next = *a < b ? imm : next;
            break;
        case KM_OP_JEQ:
            next = *a == b ? imm : next;
            break;
        case KM_OP_JNZ:
            next = *a != 0 ? imm : next;
            break;
        case KM_OP_HALT:
            /* An implicit WAIT 0, then the end event: aux the low 24 bits, ticks all.
             */
            if (s->pending != 0) {
                return KMD_UOP_WAIT;
            }
            s->state = KMD_S_ENDING;
            s->post_status = KM_EV_STREAM_END | (*a & 0xFFFFFFu) << 8;
            s->post_ticks = *a;
            return KMD_UOP_ENDED;
        case KM_OP_ISSUE:
        case KM_OP_SUBMIT: {
            if (op == KM_OP_SUBMIT && *a == 0) {
                stream_fail(s, KM_SERR_NEVER_FITS, pc);
                return KMD_UOP_ENDED;
            }
            const uint32_t job = op == KM_OP_SUBMIT ? *a : 0u;
            const kmd_uop_t u = stream_record(e, s, imm, pc, job);
            if (u != KMD_UOP_DONE) {
                return u;
            }
            break;
        }
        case KM_OP_WAIT:
            if (s->pending > imm) {
                return KMD_UOP_WAIT;
            }
            break;
        case KM_OP_RESULT:
            *a = s->last_result;
            break;
        case KM_OP_EVENT:
            if (!eng_post(e, s->req, KM_EV_STREAM | (*a & 0xFFFFFFu) << 8, *a)) {
                return KMD_UOP_WAIT;
            }
            break;
        case KM_OP_TIME:
            *a = (uint32_t)x280_mtime();
            break;
        default:
            stream_fail(s, KM_SERR_OPCODE, pc);
            return KMD_UOP_ENDED;
    }
    s->steps++;
    s->pc = next;
    return KMD_UOP_DONE;
}

/* The streams' share of a pass: each running stream runs up to KM_STEP_OPS micro-ops,
 * in activation order starting at stream `pass % running` (issuers take turns). */
static void eng_step(kmd_engine_t* e) {
    kmd_stream_t* turn[KM_STREAM_MAX_ACTIVE];
    uint32_t n = 0;
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        kmd_stream_t* s = &e->streams[i];
        if (s->state != KMD_S_RUNNING) {
            continue;
        }
        uint32_t at = n++;
        while (at > 0 && turn[at - 1]->order > s->order) {
            turn[at] = turn[at - 1];
            at--;
        }
        turn[at] = s;
    }
    if (n == 0) {
        return;
    }
    const uint32_t start = (uint32_t)(e->passes % n);
    for (uint32_t j = 0; j < n; j++) {
        kmd_stream_t* s = turn[(start + j) % n];
        for (uint32_t op = 0; op < KM_STEP_OPS; op++) {
            if (stream_op(e, s) != KMD_UOP_DONE) {
                break;
            }
        }
    }
}

/* Events, owed posts, streams, one fence per pass; on either stop word it posts what it
 * still owes for at most half of KMD_ENGINE_STOP_TICKS. */
static void eng_loop(kmd_engine_t* e) {
    uint32_t spins = 0;
    for (;;) {
        e->passes++;
        eng_events(e);
        eng_owed(e);
        eng_step(e);
        eng_owed(e);
        if (e->wrote) {
            x280_fence_ow();
            e->wrote = 0;
        }
        x280_fence_ir();
        if (*e->stop != 0) {
            break;
        }
        if (++spins == KMD_POLL_EVERY) {
            spins = 0;
            e->harts[KMD_H_STAMP1] = (uint32_t)(e->passes >> 10);
            if (x280_hart_stop_requested(e->hart, e->run)) {
                break;
            }
        }
    }
    const uint64_t by = x280_mtime() + KMD_ENGINE_STOP_TICKS / 2u;
    do {
        eng_events(e);
        eng_owed(e);
        x280_fence_ow();
    } while (eng_owes(e) && x280_mtime() < by);
}

/* Core 1: waits for the kernel's go (bounded by twice t_start and the runtime's stop),
 * runs the engine's loop, and reports it stopped in the hart block. */
static void eng_run(uint64_t hart, const volatile kmd_args_t* a, uint64_t run) {
    volatile uint8_t* state = (
        (volatile uint8_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->state_ddr)
    );
    volatile uint8_t* ring = state + KMD_STATE_RING;
    volatile uint32_t* harts = (
        (volatile uint32_t*)((volatile uint8_t*)a + KMD_LIM_HARTS)
    );
    volatile const uint32_t* go = (volatile const uint32_t*)(ring + KMD_ERING_GO);
    harts[KMD_H_ENGINE] = KMD_ENGINE_WAITING;
    const uint64_t by = x280_mtime() + 2u * a->t_start;
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
            if (x280_hart_stop_requested(hart, run) || x280_mtime() > by) {
                break;
            }
        }
    }
    if (!started) {
        harts[KMD_H_ENGINE] = KMD_ENGINE_UNUSED;
        x280_fence_ow();
        return;
    }
    uint64_t line_buf[16];
    kmd_engine_t* e = (kmd_engine_t*)(uintptr_t)(state + KMD_STATE_ENGINE);
    e->hart = hart;
    e->run = run;
    e->kernel = (const volatile kmd_t*)(state + KMD_STATE_KERNEL);
    e->harts = harts;
    e->upload = (volatile const uint8_t*)a + KMD_LIM_UPLOAD;
    e->upload_bytes = (uint32_t)a->upload_bytes;
    e->cores = (uint32_t)a->cores;
    e->cmds = ring + KMD_ERING_CMDS;
    e->consumed = (volatile const uint32_t*)(ring + KMD_ERING_CONSUMED);
    e->events = ring + KMD_ERING_EVENTS;
    e->read = (volatile uint32_t*)(ring + KMD_ERING_READ);
    e->stop = (volatile const uint32_t*)(ring + KMD_ERING_STOP);
    e->queued = (volatile const uint32_t*)(ring + KMD_ERING_QUEUED);
    e->taken = (volatile const uint32_t*)(ring + KMD_ERING_TAKEN);
    e->next = 1;
    e->room_to = KMD_ENGINE_DEPTH;
    e->line = (uint64_t*)(((uintptr_t)line_buf + 63u) & ~(uintptr_t)63u);
    harts[KMD_H_ENGINE] = KMD_ENGINE_RUNNING;
    eng_loop(e);
    harts[KMD_H_PASSES] = (uint32_t)e->passes;
    harts[KMD_H_RECORDS] = e->records;
    harts[KMD_H_POSTS] = e->posts;
    harts[KMD_H_EVENTS] = e->events_in;
    harts[KMD_H_RUNS] = e->runs;
    kmd_flush_lines(state + KMD_STATE_ENGINE, KMD_STATE_BYTES - KMD_STATE_ENGINE);
    x280_fence_ow();
    harts[KMD_H_ENGINE] = KMD_ENGINE_STOPPED;
    x280_fence_ow();
}
