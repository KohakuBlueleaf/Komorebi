/* Micro-op streams (komorebi/abi/km_stream.h): the check of an upload, KM_SEG_RUN and
 * KM_SEG_CANCEL, and the interpreter, which runs each stream in place. */
#pragma once

#include "completion.h"
#include "event_ring.h"
#include "kernel.h"

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

/* Ends stream `s` (its end or error event is already pushed). Its records in flight
 * stay in the completion table as orphans, with their deadlines. */
static inline void stream_end(kmd_t* k, kmd_stream_t* s) {
    s->running = 0;
    k->nstreams--;
}

/* Ends stream `s` with a KM_EV_STREAM_ERROR event: `reason` at program counter `pc`. */
static inline void stream_fail(
    kmd_t* k, kmd_stream_t* s, uint32_t reason, uint32_t pc
) {
    events_push(k, s->req, KM_EV_STREAM_ERROR | (reason | pc << 8) << 8, 0);
    k->stream_events++;
    stream_end(k, s);
}

/* KM_SEG_RUN, command `cmd`: starts the upload of `bytes` at `off` in the region, or
 * answers with an error event (the header's req when it lies in the region, else 0). */
static void stream_run(kmd_t* k, uint32_t cmd, uint32_t off, uint32_t bytes) {
    const uint32_t region = k->upload_bytes;
    uint32_t req = 0;
    if (region >= sizeof(km_stream_t) && off <= region - sizeof(km_stream_t) &&
        bytes >= 28u) {
        /* Byte loads: `off` may be unaligned here. */
        const volatile uint8_t* p = k->upload + off + 24u;
        x280_fence_ir();
        req = p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    }
    if (k->exited || k->nstreams == KM_STREAM_MAX_ACTIVE) {
        const uint32_t why = k->exited ? KMD_REFUSE_EXITED : KMD_REFUSE_STREAMS;
        events_push(k, cmd, KM_EV_REFUSED | (why << 8), 0);
        k->refused++;
        return;
    }
    uint32_t err = 0;
    if ((off & 63u) || off > region || bytes > region - off) {
        err = KM_SERR_HEADER;
    }
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE && err == 0; i++) {
        const kmd_stream_t* s = &k->streams[i];
        if (s->running &&
            ((off < s->off + s->bytes && s->off < off + bytes) || s->req == req)) {
            err = KM_SERR_HEADER;
        }
    }
    if (err == 0) {
        err = stream_check(k->upload + off, bytes);
    }
    if (err != 0) {
        events_push(k, req, KM_EV_STREAM_ERROR | (err << 8), 0);
        k->stream_events++;
        return;
    }
    kmd_stream_t* s = &k->streams[0];
    while (s->running) {
        s++;
    }
    const volatile uint32_t* h = (const volatile uint32_t*)(k->upload + off);
    for (uint32_t r = 0; r < KM_STREAM_REGS; r++) {
        s->regs[r] = 0;
    }
    s->regs[KM_STREAM_SEED_REG] = h[7];
    s->req = req;
    s->pc = s->steps = s->pending = s->last_result = 0;
    s->n_ops = h[2];
    s->n_records = h[3];
    s->max_steps = h[4];
    s->t_record = h[5] ? (uint64_t)h[5] * 50u : k->t_record;
    s->off = off;
    s->bytes = bytes;
    s->order = ++k->orders;
    s->running = 1;
    k->nstreams++;
}

/* KM_SEG_CANCEL, command `cmd`: ends the active stream whose req is `req` with reason
 * KM_SERR_CANCELLED at the micro-op it would run next, or refuses the command. */
static void stream_cancel(kmd_t* k, uint32_t cmd, uint32_t req) {
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        kmd_stream_t* s = &k->streams[i];
        if (s->running && s->req == req) {
            stream_fail(k, s, KM_SERR_CANCELLED, s->pc);
            return;
        }
    }
    events_push(k, cmd, KM_EV_REFUSED | (KMD_REFUSE_UNKNOWN << 8), 0);
    k->refused++;
}

/* ISSUE of template `t` at `pc`: the team is checked first, then a free slot, room for
 * the event ring's reservations and, with KM_F_EXCL, no overlap; else it waits. */
static kmd_uop_t stream_issue(
    kmd_t* k, kmd_stream_t* s, uint32_t t, uint32_t pc, uint64_t now
) {
    if (t >= s->n_records) {
        stream_fail(k, s, KM_SERR_TEMPLATE, pc);
        return KMD_UOP_ENDED;
    }
    const volatile uint64_t* w = stream_template(k->upload + s->off, s->n_ops, t);
    const uint64_t patch = w[8], tail = w[9];
    const uint32_t team = (uint32_t)w[1], fr = (uint32_t)tail & 0xFFu;
    const uint32_t cr = (uint32_t)(tail >> 8) & 0xFFu;
    const uint32_t first = fr == KM_TPL_NONE ? team & 0xFFFFu : s->regs[fr & 15u];
    const uint32_t count = cr == KM_TPL_NONE ? team >> 16 : s->regs[cr & 15u];
    if (team_outside(k, first, count)) {
        stream_fail(k, s, KM_SERR_TEAM, pc);
        return KMD_UOP_ENDED;
    }
    const uint32_t slot = slot_free(k);
    if (slot == KM_RING_SLOTS || !events_room(k)) {
        return KMD_UOP_WAIT;
    }
    const uint32_t hdr = (uint32_t)(w[0] >> 32);
    if ((hdr & (KM_F_EXCL << 8)) && slots_overlap(k, first, count)) {
        return KMD_UOP_WAIT;
    }
    uint64_t* l = k->eline;
    for (uint32_t i = 0; i < 8; i++) {
        l[i] = w[i];
    }
    uint32_t* words = (uint32_t*)l;
    words[2] = first | count << 16;
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
    record_issue(k, l, slot, s->req, now, s->t_record);
    k->slot_order[slot] = s->order;
    k->slot_stream[slot] = (uint8_t)(s - k->streams);
    k->slot_pc[slot] = (uint16_t)pc;
    k->stream_records++;
    s->pending++;
    return KMD_UOP_DONE;
}

/* One micro-op of stream `s` at its pc (KM_SERR_JUMP if past the end, KM_SERR_STEPS
 * after max_steps); a micro-op that waits does not count as a step. */
static kmd_uop_t stream_op(kmd_t* k, kmd_stream_t* s, uint64_t now) {
    const uint32_t pc = s->pc;
    if (pc >= s->n_ops) {
        stream_fail(k, s, KM_SERR_JUMP, s->n_ops);
        return KMD_UOP_ENDED;
    }
    if (s->steps >= s->max_steps) {
        stream_fail(k, s, KM_SERR_STEPS, pc);
        return KMD_UOP_ENDED;
    }
    const volatile uint64_t* ops = (
        (const volatile uint64_t*)(k->upload + s->off + sizeof(km_stream_t))
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
                stream_fail(k, s, KM_SERR_MOD_ZERO, pc);
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
            events_push(k, s->req, KM_EV_STREAM_END | (*a & 0xFFFFFFu) << 8, *a);
            k->stream_events++;
            stream_end(k, s);
            return KMD_UOP_ENDED;
        case KM_OP_ISSUE: {
            const kmd_uop_t u = stream_issue(k, s, imm, pc, now);
            if (u != KMD_UOP_DONE) {
                return u;
            }
            break;
        }
        case KM_OP_SUBMIT:
            /* The job queue is not built yet: the count is checked, then it waits. */
            if (*a == 0 || *a > k->cores) {
                stream_fail(k, s, KM_SERR_NEVER_FITS, pc);
                return KMD_UOP_ENDED;
            }
            return KMD_UOP_WAIT;
        case KM_OP_WAIT:
            if (s->pending > imm) {
                return KMD_UOP_WAIT;
            }
            break;
        case KM_OP_RESULT:
            *a = s->last_result;
            break;
        case KM_OP_EVENT:
            if (!events_room(k)) {
                return KMD_UOP_WAIT;
            }
            events_push(k, s->req, KM_EV_STREAM | (*a & 0xFFFFFFu) << 8, *a);
            k->stream_events++;
            break;
        case KM_OP_TIME:
            *a = (uint32_t)x280_mtime();
            break;
        default:
            stream_fail(k, s, KM_SERR_OPCODE, pc);
            return KMD_UOP_ENDED;
    }
    s->steps++;
    s->pc = next;
    return KMD_UOP_DONE;
}

/* The engines' share of a pass: each active stream runs up to KM_STEP_OPS micro-ops,
 * in activation order starting at stream `pass % active` (issuers take turns). */
static void engines_step(kmd_t* k, uint64_t now) {
    kmd_stream_t* turn[KM_STREAM_MAX_ACTIVE];
    uint32_t n = 0;
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        kmd_stream_t* s = &k->streams[i];
        if (!s->running) {
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
    const uint32_t start = (uint32_t)(k->passes % n);
    for (uint32_t j = 0; j < n; j++) {
        kmd_stream_t* s = turn[(start + j) % n];
        for (uint32_t op = 0; op < KM_STEP_OPS; op++) {
            if (stream_op(k, s, now) != KMD_UOP_DONE) {
                break;
            }
        }
    }
}
