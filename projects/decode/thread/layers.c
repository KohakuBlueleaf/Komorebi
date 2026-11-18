/* Decode, the thread: LAYERS runs every layer of every token, each layer's commands in
 * order (decode.h, DC_STEP_*), each waited for before the next unless overlapped. */
#include <stdint.h>

#include "kmx.h"
#include "kmx_chain.h"
#include "chain.h"
#include "decode.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* one command's submit or wait */
#define PROGRESS_END 0xC0000000u        /* the last progress marker: the job's status */
#define KM_SPLIT_FULL 0u
#define KM_SPLIT_PARTS 1u
#define KM_SPLIT_OP 2u
#define KM_SPLIT_SECOND_SET 0x100u /* a3: the sum's upper ports and slots */
#define TOKEN_NONE 0xFFFFFFFFu     /* token_form's last word: no records written */

typedef struct {
    const uint32_t* words;
    const uint32_t* tensors;
    const uint32_t* layer;
    uint64_t step_ticks[DC_STEPS];
    uint32_t step_most[DC_STEPS], step_least[DC_STEPS];
    uint32_t done;
    uint32_t bare;     /* products only: no operation (nor the norm in them) */
    uint32_t parts;    /* and each product's parts only: no sum, nothing written */
    uint32_t apart;    /* q, k and v as three products, not qkv */
    uint32_t overlap;  /* and q, k, v, o each submitted before the one ahead is done */
    uint32_t attend;   /* ATTEND's a3 flags: DC_ATTEND_READ_ONLY, DC_ATTEND_LOCAL */
    uint32_t position; /* the first token's place in the cache */
    uint32_t index; /* the layer's number: its head norms' weights in the qk tensor */
    uint32_t rows;  /* the tokens a step: a3's DC_ROWS field of every command */
    uint64_t span_ticks; /* q's submit to o's event, summed over the layers */
    uint64_t
        thread_ticks[DC_THREADS]; /* the chain's: records, submit, wait (decode.h) */
    uint32_t chain;     /* the layer as one command: the cores go on by themselves */
    uint32_t records;   /* the chain's records: its first steps */
    uint32_t token;     /* and a token as one command: every layer's records, once */
    uint32_t cores;     /* the records' cores: the chain command's */
    uint32_t ahead_off; /* no record's W read in the record before */
    uint32_t data;      /* the records whose input is sent, a bit a step (decode.h) */
    uint32_t data_dram; /* their DRAM copy acknowledged before the marks too */
    uint32_t head;      /* a token's: the head's records after its layers' (parts) */
} job_t;

/* Overlapped, q, k, v and o alternate the sum's two sets: k and o the second. */
static int overlapped(const job_t* job, uint32_t step) {
    return job->overlap && step >= DC_STEP_Q && step <= DC_STEP_O;
}

/* Whether the job runs step `step`: qkv and attention, or q, k and v. */
static int runs(const job_t* job, uint32_t step) {
    if (step == DC_STEP_QKV || step == DC_STEP_ATTEND) {
        return !job->apart;
    }
    if (step >= DC_STEP_Q && step <= DC_STEP_V) {
        return job->apart;
    }
    return step != DC_STEP_CHAIN; /* the chain's: only its own command (run_chain) */
}

/* A SPLIT command on the product's cores: A, W, Y, the plan, the operation and its
 * words (decode.h: add's, or scale's and gate's). */
static void product(
    const job_t* job,
    uint32_t step,
    km_command_t* command,
    uint32_t a,
    uint32_t w,
    uint32_t y,
    uint32_t plan,
    uint32_t operation,
    const uint32_t* other
) {
    const uint32_t* words = job->words;
    kmx_command(command, 0u, words[DC_WORD_SPLIT], 0u, words[plan] >> 8);
    if (!overlapped(job, step)) {
        command->hdr |= words[DC_WORD_EXCLUSIVE] << 8;
    }
    command->a0 = a;
    command->a1 = w;
    command->a2 = y;
    command->a3 = words[DC_WORD_REDUCE] | job->rows << DC_ROWS_SHIFT;
    if (overlapped(job, step) && (step == DC_STEP_K || step == DC_STEP_O)) {
        command->a3 |= KM_SPLIT_SECOND_SET;
    }
    command->a4 = job->parts ? KM_SPLIT_PARTS : KM_SPLIT_FULL;
    command->a5 = words[plan] & 0xFFu;
    command->a6 = operation == 0u || job->bare ? 0u : words[operation];
    command->a7 = other[0];
    command->a8 = other[1];
    command->a9 = other[2];
}

/* The layer's weight word of product `step`. */
static uint32_t weight_of(uint32_t step) {
    switch (step) {
        case DC_STEP_QKV:
            return DC_W_QKV;
        case DC_STEP_Q:
            return DC_W_Q;
        case DC_STEP_K:
            return DC_W_K;
        case DC_STEP_V:
            return DC_W_V;
        case DC_STEP_O:
            return DC_W_O;
        case DC_STEP_GATE_UP:
            return DC_W_GATE_UP;
        default:
            return DC_W_DOWN;
    }
}

/* ATTEND on every KV head's group: q, k, v from qkv, the layer's cache, a. */
static void attend_command(const job_t* job, km_command_t* command) {
    const uint32_t* words = job->words;
    const uint32_t* t = job->tensors;
    const uint32_t plan = words[DC_WORD_PLAN_ATTEND];
    kmx_command(command, 0u, words[DC_WORD_SPLIT], 0u, plan >> 8);
    command->hdr |= words[DC_WORD_EXCLUSIVE] << 8;
    command->a0 = t[DC_TENSOR_QKV];
    command->a1 = job->layer[DC_W_KEYS];
    command->a2 = t[DC_TENSOR_A];
    command->a3 = words[DC_WORD_REDUCE] | job->attend | job->rows << DC_ROWS_SHIFT;
    command->a4 = KM_SPLIT_OP;
    command->a5 = plan & 0xFFu;
    command->a6 = words[DC_WORD_ATTEND];
    command->a7 = job->layer[DC_W_VALUES];
    command->a8 = t[DC_TENSOR_ATTEND];
    command->a9 = job->position | words[DC_WORD_CACHE_TILES] << 16;
}

/* Step `step`'s command of the job's layer. */
static void step_command(const job_t* job, uint32_t step, km_command_t* command) {
    const uint32_t* t = job->tensors;
    const uint32_t* w = job->layer;
    if (step == DC_STEP_ATTEND) {
        attend_command(job, command);
        return;
    }
    const uint32_t epsilon = job->words[DC_WORD_EPSILON];
    uint32_t a = t[DC_TENSOR_X], y = 0u, plan = DC_WORD_PLAN_Q;
    uint32_t operation = DC_WORD_SCALE; /* x's norm1, after the product */
    uint32_t other[3] = {t[DC_TENSOR_SQUARES1], t[DC_TENSOR_CONSTANTS], epsilon};
    switch (step) {
        case DC_STEP_QKV: /* and q's and k's head norms and rotation (DC_QK) */
            y = t[DC_TENSOR_QKV];
            plan = DC_WORD_PLAN_QKV;
            other[1] = t[DC_TENSOR_QK];
            other[2] = job->position | job->index << 16;
            break;
        case DC_STEP_Q: /* into qkv's first columns, as qkv writes it */
            y = t[DC_TENSOR_QKV];
            break;
        case DC_STEP_K:
        case DC_STEP_V:
            y = t[step == DC_STEP_K ? DC_TENSOR_K : DC_TENSOR_V];
            plan = DC_WORD_PLAN_KV;
            break;
        case DC_STEP_O: /* apart (no attention): q, qkv's first columns, stands in */
            a = job->apart ? t[DC_TENSOR_QKV] : t[DC_TENSOR_A];
            y = t[DC_TENSOR_X1];
            operation = DC_WORD_ADD; /* x1 = x + o, and x1's squares */
            other[0] = t[DC_TENSOR_X];
            other[2] = t[DC_TENSOR_SQUARES2];
            break;
        case DC_STEP_GATE_UP: /* bare: the whole sum, into a tensor as wide */
            a = t[DC_TENSOR_X1];
            y = job->bare ? t[DC_TENSOR_SUMS] : t[DC_TENSOR_GATED];
            plan = DC_WORD_PLAN_GATE_UP;
            operation = DC_WORD_GATE; /* x1's norm2, then the gate */
            other[0] = t[DC_TENSOR_SQUARES2];
            break;
        default: /* DC_STEP_DOWN */
            a = t[DC_TENSOR_GATED];
            y = t[DC_TENSOR_X];
            plan = DC_WORD_PLAN_DOWN;
            operation = DC_WORD_ADD; /* x = x1 + down, and x's squares */
            other[0] = t[DC_TENSOR_X1];
            other[2] = t[DC_TENSOR_SQUARES1];
            break;
    }
    product(job, step, command, a, w[weight_of(step)], y, plan, operation, other);
    if (step == DC_STEP_QKV) {
        command->a3 |= DC_QK;
    }
}

/* Submits step `step` (retrying a full ring): KMX_OK once queued, its handle. */
static int submit_step(const job_t* job, uint32_t step, uint32_t* handle) {
    km_command_t command;
    step_command(job, step, &command);
    uint64_t deadline = kmx_time() + WAIT_TICKS;
    int status = kmx_submit(&command, handle);
    while (status == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
        status = kmx_submit(&command, handle);
    }
    return status;
}

/* Waits for step `step`'s command and counts its time: KMX_OK once done. */
static int wait_step(job_t* job, uint32_t step, uint32_t handle) {
    kmx_event event;
    int status = kmx_wait(handle, kmx_time() + WAIT_TICKS, &event);
    if (status != KMX_OK) {
        return status;
    }
    if (event.kind != KM_EVENT_DONE) {
        return KMX_BAD;
    }
    job->done++;
    job->step_ticks[step] += event.ticks;
    job->step_most[step] = (
        event.ticks > job->step_most[step] ? event.ticks : job->step_most[step]
    );
    job->step_least[step] = (
        event.ticks < job->step_least[step] ? event.ticks : job->step_least[step]
    );
    return KMX_OK;
}

/* Step `step`, submitted and waited for. */
static int run_step(job_t* job, uint32_t step) {
    uint32_t handle = 0u;
    const int status = submit_step(job, step, &handle);
    return status != KMX_OK ? status : wait_step(job, step, handle);
}

/* q, k, v, o with two in flight: each submitted before the one ahead of it is waited
 * for (k, v and o do not read what the one ahead writes; o reads q, done by then). */
static int run_attention(job_t* job) {
    static const uint32_t steps[] = {DC_STEP_Q, DC_STEP_K, DC_STEP_V, DC_STEP_O};
    const uint64_t started = kmx_time();
    uint32_t ahead = 0u, handle = 0u;
    int status = submit_step(job, steps[0], &ahead);
    for (uint32_t i = 1u; i < 4u && status == KMX_OK; i++) {
        status = submit_step(job, steps[i], &handle);
        if (status == KMX_OK) {
            status = wait_step(job, steps[i - 1u], ahead);
            ahead = handle;
        }
    }
    if (status == KMX_OK) {
        status = wait_step(job, DC_STEP_O, ahead);
    }
    job->span_ticks += kmx_time() - started;
    return status;
}

/* The fused layer's steps in order, as a chain's records. */
static const uint32_t chain_steps[] = {
    DC_STEP_QKV, DC_STEP_ATTEND, DC_STEP_O, DC_STEP_GATE_UP, DC_STEP_DOWN
};
#define CHAIN_STEPS 5u

/* Record `record` after a token's layers: 0 the final norm (x diag(weight) / rms into
 * xn), then head part record - 1 (each block's largest; a9 part | token columns). */
static void head_command(const job_t* job, uint32_t record, km_command_t* command) {
    const uint32_t* t = job->tensors;
    const uint32_t* words = job->words;
    if (record == 0u) {
        const uint32_t other[3] = {
            t[DC_TENSOR_SQUARES1], t[DC_TENSOR_CONSTANTS], words[DC_WORD_EPSILON]
        };
        product(
            job,
            DC_STEP_Q,
            command,
            t[DC_TENSOR_X],
            t[DC_TENSOR_NORM],
            t[DC_TENSOR_XN],
            DC_WORD_PLAN_Q,
            DC_WORD_SCALE,
            other
        );
        return;
    }
    const uint32_t part = record - 1u, columns = words[DC_WORD_HEAD_COLUMNS];
    const uint32_t left = words[DC_WORD_VOCAB] - part * columns;
    const uint32_t other[3] = {0u, 0u, part | (left < columns ? left : columns) << 16};
    product(
        job,
        DC_STEP_QKV,
        command,
        t[DC_TENSOR_XN],
        t[DC_TENSOR_HEAD + part],
        t[DC_TENSOR_LARGEST],
        DC_WORD_PLAN_HEAD,
        DC_WORD_LARGEST,
        other
    );
}

/* Whether step `step`'s command writes the output the step after it reads. */
static int chain_writes(const job_t* job, uint32_t step) {
    if (step == DC_STEP_ATTEND) {
        return (job->attend & DC_ATTEND_READ_ONLY) == 0u;
    }
    return !job->parts;
}

/* Whether step `step`'s parts go up its group's tree (its receivers send credits). */
static int chain_sends(const job_t* job, uint32_t step) {
    if (step == DC_STEP_ATTEND) {
        return (job->attend & (DC_ATTEND_READ_ONLY | DC_ATTEND_LOCAL)) == 0u;
    }
    return !job->parts;
}

/* The job's form its token records were written for (a4, first, count, rows), then 1
 * once written (after the load none: the thread has no zeroed data), their cores. */
static uint32_t token_form[6] = {0u, 0u, 0u, 0u, TOKEN_NONE, 0u};

/* The records (chain.h) of `count` layers into region 3, each naming the next; `token`:
 * the position from the command's a0, then job->head head parts (the first marked). */
static int chain_write(job_t* job, uint32_t count, int token) {
    const uint32_t* words = job->words;
    const uint32_t steps = job->records, total = steps * count;
    const uint32_t heads = token && job->head ? 1u + job->head : 0u;
    if (total + heads > kChainRecordsMax || heads > 1u + DC_HEAD_PARTS_MOST) {
        return KMX_BAD;
    }
    const uint32_t* layer = job->layer;
    const uint32_t index = job->index, position = job->position;
    if (token) {
        job->position = 0u;
    }
    km_command_t command;
    uint32_t address = 0u;
    job->cores = 0u;
    for (uint32_t g = total + heads; g-- > total;) { /* the head's, last first */
        head_command(job, g - total, &command);
        kmx_chain_call call;
        call.command = &command;
        call.word = call.by_ranges = call.k_base = call.k_tiles = 0u;
        call.ranges[0] = call.ranges[1] = call.ranges[2] = 0u;
        call.size = call.width = call.handler = 0u;
        /* the norm waits for down's x, the first part for xn; the others run on the
         * first's cores, which have read xn */
        call.flags = (g <= total + 1u ? kChainMarked : 0u) | kChainSends;
        call.row = DC_CHAIN_LAST_ROW;
        if (kmx_chain_record(3, g, &call, address, &address) != KMX_OK) {
            return KMX_BAD;
        }
        job->cores = (
            (command.group >> 16) > job->cores ? command.group >> 16 : job->cores
        );
    }
    for (uint32_t g = total; g-- > 0u;) { /* last first: each knows the next */
        const uint32_t i = g % steps, step = chain_steps[i];
        job->layer = layer + DC_LAYER_WORDS * (g / steps);
        job->index = index + g / steps;
        step_command(job, step, &command);
        kmx_chain_call call; /* field by field: no memset */
        call.command = &command;
        call.word = call.by_ranges = call.k_base = call.k_tiles = 0u;
        call.ranges[0] = call.ranges[1] = call.ranges[2] = 0u;
        call.size = call.width = call.row = call.handler = 0u;
        uint32_t flags = 0u;
        if (g > 0u && chain_writes(job, chain_steps[(g - 1u) % steps])) {
            flags |= kChainMarked;
        }
        if (
            (flags & kChainMarked) != 0u && step != DC_STEP_ATTEND &&
            ((job->data >> i) & 1u) != 0u
        ) {
            flags |= kChainDataMark;
        }
        if (chain_sends(job, step)) {
            flags |= kChainSends;
        }
        call.flags = flags;
        if (token && (step == DC_STEP_QKV || step == DC_STEP_ATTEND)) {
            call.word = 15u; /* a9: qkv's (the op's) and attend's */
        }
        call.by_ranges = step == DC_STEP_ATTEND;
        for (uint32_t range = 0u; call.by_ranges && range < kChainRangeCount; range++) {
            call.ranges[range] = words[DC_WORD_ATTEND_NEEDS + range];
        }
        /* the group's size and K range: the plan's (the program reads them) */
        call.width = (
            step == DC_STEP_GATE_UP && !job->bare ? words[DC_WORD_GATE_TILES] : 0u
        );
        /* stamp row: the step's; the token's last record its own (bench.py folds it) */
        call.row = count > 1u && g == total - 1u ? DC_CHAIN_LAST_ROW : i;
        if (kmx_chain_record(3, g, &call, address, &address) != KMX_OK) {
            return KMX_BAD;
        }
        job->cores = (
            (command.group >> 16) > job->cores ? command.group >> 16 : job->cores
        );
    }
    job->layer = layer;
    job->index = index;
    job->position = position;
    kmx_flush(3);
    return KMX_OK;
}

/* One SPLIT in the chain's mode from region 3's first record, a0 the token's word,
 * on the records' cores (job->cores); waited for once. */
static int chain_run(job_t* job, uint32_t a0) {
    const uint32_t* words = job->words;
    km_command_t chain;
    const uint64_t written = kmx_time();
    const uint32_t flags = (
        (job->ahead_off ? kChainAheadOff : 0u) | (job->data_dram ? kChainDataDram : 0u)
    );
    if (kmx_chain_command(
        &chain, 3, words[DC_WORD_SPLIT], job->cores, kChainMode, a0, flags
    ) != KMX_OK) {
        return KMX_BAD;
    }
    chain.hdr |= words[DC_WORD_EXCLUSIVE] << 8;
    uint32_t handle = 0u;
    uint64_t deadline = kmx_time() + WAIT_TICKS;
    int status = kmx_submit(&chain, &handle);
    while (status == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
        status = kmx_submit(&chain, &handle);
    }
    const uint64_t submitted = kmx_time();
    job->thread_ticks[DC_THREAD_SUBMIT] += submitted - written;
    if (status == KMX_OK) {
        status = wait_step(job, DC_STEP_CHAIN, handle);
    }
    job->thread_ticks[DC_THREAD_WAIT] += kmx_time() - submitted;
    return status;
}

/* The layer as one command: its records written, then run (stage 6). */
static int run_chain(job_t* job) {
    const uint64_t started = kmx_time();
    token_form[4] = TOKEN_NONE; /* region 3 holds this layer's records now */
    const int status = chain_write(job, 1u, 0);
    job->thread_ticks[DC_THREAD_RECORDS] += kmx_time() - started;
    return status != KMX_OK ? status : chain_run(job, 0u);
}

/* A token as one command: every layer's records written once a form, then each token
 * one command with its position. */
static int run_token(job_t* job, const km_command_t* command) {
    const uint32_t form[4] = {command->a4, command->a3, command->a1, command->a6};
    int status = KMX_OK;
    uint32_t same = token_form[4] == 1u;
    for (uint32_t word = 0u; word < 4u; word++) {
        same = same && token_form[word] == form[word];
    }
    if (!same) {
        const uint64_t started = kmx_time();
        token_form[4] = TOKEN_NONE;
        status = chain_write(job, command->a1, 1);
        job->thread_ticks[DC_THREAD_RECORDS] += kmx_time() - started;
        for (uint32_t word = 0u; word < 4u && status == KMX_OK; word++) {
            token_form[word] = form[word];
        }
        token_form[4] = status == KMX_OK ? 1u : TOKEN_NONE;
        token_form[5] = job->cores;
    }
    job->cores = token_form[5];
    return status != KMX_OK ? status : chain_run(job, job->position);
}

/* GENERATE's regions (after LAYERS' four): the table's 64 tiles that hold a token's
 * column; x's tiles and its squares' as written; the head's largest pages; `made`. */
enum { REGION_GATHER = 4, REGION_ROW = 5, REGION_LARGEST = 6, REGION_MADE = 7 };

#define TILE_BYTES 2048u
#define COPY_CORES 16u /* the cores a copy of the thread's takes, from core 0 */

/* A tile's element (row, column) in its faces' order, in bf16 values. */
static uint32_t tile_at(uint32_t row, uint32_t column) {
    return (
        ((row >> 4) * 2u + (column >> 4)) * 256u + (row & 15u) * 16u + (column & 15u)
    );
}

static uint16_t bf16_of(float value) { /* round to nearest even */

    union {
        float f;
        uint32_t u;
    } bits;

    bits.f = value;
    return (uint16_t)((bits.u + 0x7FFFu + ((bits.u >> 16) & 1u)) >> 16);
}

/* Token `id`'s embedding: its column of the head's table (64 tiles, 16 fetches at a
 * time) into x's row 0 and its sum of squares into squares1, as the host does. */
static int embed(const job_t* job, uint32_t id, uint64_t deadline) {
    const uint32_t* t = job->tensors;
    const uint32_t columns = job->words[DC_WORD_HEAD_COLUMNS];
    const uint32_t part = id / columns, column = id % columns;
    const uint32_t row_tiles = columns / 32u;
    uint64_t bytes = 0;
    const uint16_t* gathered = (const uint16_t*)kmx_region(REGION_GATHER, &bytes);
    uint16_t* row = (uint16_t*)kmx_region(REGION_ROW, &bytes);
    if (
        gathered == 0 || row == 0 || bytes < (DC_ROW_TILES + DC_PARTS) * TILE_BYTES ||
        part >= job->head
    ) {
        return KMX_BAD;
    }
    for (uint32_t k = 0u; k < DC_ROW_TILES; k += COPY_CORES) {
        uint32_t handles[COPY_CORES];
        for (uint32_t i = 0u; i < COPY_CORES; i++) {
            const kmx_dram from = {
                t[DC_TENSOR_HEAD + part],
                KM_COPY_INTERLEAVED,
                TILE_BYTES,
                (k + i) * row_tiles + column / 32u
            };
            const int status = (
                kmx_fetch_submit(
                    REGION_GATHER,
                    (k + i) * TILE_BYTES,
                    &from,
                    TILE_BYTES,
                    i,
                    1u,
                    &handles[i]
                )
            );
            if (status != KMX_OK) {
                return status;
            }
        }
        for (uint32_t i = 0u; i < COPY_CORES; i++) {
            kmx_event event;
            const int status = kmx_wait(handles[i], deadline, &event);
            if (status != KMX_OK) {
                return status;
            }
        }
    }
    double sum = 0.0;
    for (uint32_t k = 0u; k < DC_ROW_TILES; k++) {
        const uint16_t* tile = gathered + k * (TILE_BYTES / 2u);
        uint16_t* out = row + k * (TILE_BYTES / 2u);
        for (uint32_t r = 0u; r < 32u; r++) {
            const uint16_t value = tile[tile_at(r, column % 32u)];

            union {
                uint32_t u;
                float f;
            } bits;

            bits.u = (uint32_t)value << 16;
            sum += (double)bits.f * (double)bits.f;
            out[tile_at(0u, r)] = value;
        }
    }
    row[DC_ROW_TILES * (TILE_BYTES / 2u)] = bf16_of((float)sum);
    const kmx_dram x = {t[DC_TENSOR_X], KM_COPY_INTERLEAVED, TILE_BYTES, 0u};
    const kmx_dram squares = {
        t[DC_TENSOR_SQUARES1], KM_COPY_INTERLEAVED, TILE_BYTES, 0u
    };
    int status = (
        kmx_store(
            REGION_ROW, 0u, &x, DC_ROW_TILES * TILE_BYTES, 0u, COPY_CORES, deadline, 0
        )
    );
    if (status == KMX_OK) {
        status = (
            kmx_store(
                REGION_ROW,
                DC_ROW_TILES * TILE_BYTES,
                &squares,
                DC_PARTS * TILE_BYTES,
                0u,
                DC_PARTS,
                deadline,
                0
            )
        );
    }
    return status;
}

/* The token the head chose: the largest of the blocks' pages (bf16 bits, the column in
 * its part; all ones past the vocab), the first index on a tie. */
static int choose(const job_t* job, uint32_t* id, uint64_t deadline) {
    const uint32_t blocks = job->words[DC_WORD_HEAD_COLUMNS] / (32u * 8u);
    const uint32_t pages = blocks * job->head;
    const kmx_dram from = {
        job->tensors[DC_TENSOR_LARGEST], KM_COPY_INTERLEAVED, TILE_BYTES, 0u
    };
    uint64_t bytes = 0;
    const uint32_t* words = (const uint32_t*)kmx_region(REGION_LARGEST, &bytes);
    if (words == 0 || bytes < (uint64_t)pages * TILE_BYTES) {
        return KMX_BAD;
    }
    const int status = (
        kmx_fetch(
            REGION_LARGEST,
            0u,
            &from,
            (uint64_t)pages * TILE_BYTES,
            0u,
            COPY_CORES,
            deadline,
            0
        )
    );
    if (status != KMX_OK) {
        return status;
    }
    uint32_t best_key = 0u, best = 0xFFFFFFFFu;
    for (uint32_t page = 0u; page < pages; page++) {
        const uint32_t bits = words[page * (TILE_BYTES / 4u)];
        const uint32_t column = words[page * (TILE_BYTES / 4u) + 1u];
        if (column == 0xFFFFFFFFu) {
            continue;
        }
        const uint32_t key = (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
        const uint32_t index = (
            page / blocks * job->words[DC_WORD_HEAD_COLUMNS] + column
        );
        if (
            best == 0xFFFFFFFFu || key > best_key || (key == best_key && index < best)
        ) {
            best_key = key;
            best = index;
        }
    }
    *id = best;
    return best == 0xFFFFFFFFu ? KMX_BAD : KMX_OK;
}

/* LAYERS: a0 steps, a1 layers, a2 stats slot, a3 first layer, a4 bits, a5 the first
 * position, a6 rows a step (decode.h). Result: the ticks, or 0xFFFFFFF0 | status. */
static uint32_t layers(const km_command_t* command) {
    uint64_t weight_bytes = 0, tensor_bytes = 0, stats_bytes = 0;
    const uint32_t* weights = (const uint32_t*)kmx_region(0, &weight_bytes);
    const uint32_t* tensors = (const uint32_t*)kmx_region(1, &tensor_bytes);
    uint32_t* stats = (uint32_t*)kmx_region(2, &stats_bytes);
    const kmx_args* args = kmx_args_get();
    const uint32_t tokens = command->a0, count = command->a1, first = command->a3;
    if (
        weights == 0 || tensors == 0 || stats == 0 || args->nwords < DC_WORDS ||
        command->a2 >= DC_STATS_SLOTS || command->a6 == 0u || command->a6 > 32u ||
        4ull * DC_STATS_WORDS * DC_STATS_SLOTS > stats_bytes || count == 0 ||
        count > DC_QK_LAYERS || first > DC_QK_LAYERS - count ||
        4ull * DC_LAYER_WORDS * ((uint64_t)first + count) > weight_bytes ||
        4ull * DC_TENSORS > tensor_bytes
    ) {
        return 0xFFFFFFFFu;
    }
    kmx_invalidate(0);
    kmx_invalidate(1);
    job_t job;
    job.words = args->words;
    job.tensors = tensors;
    job.done = 0u;
    job.bare = command->a4 & 1u;
    job.parts = job.bare && (command->a4 & 2u) != 0u;
    job.apart = (command->a4 & 4u) != 0u;
    job.overlap = job.apart && (command->a4 & 8u) != 0u;
    job.chain = !job.apart && (command->a4 & DC_LAYERS_CHAIN) != 0u;
    job.records = (command->a4 & DC_LAYERS_PAIR) != 0u ? 2u : CHAIN_STEPS;
    job.token = job.chain && (command->a4 & DC_LAYERS_TOKEN) != 0u;
    job.ahead_off = (command->a4 & DC_LAYERS_AHEAD_OFF) != 0u;
    job.data = (command->a4 >> DC_LAYERS_DATA_SHIFT) & DC_LAYERS_DATA_MASK;
    job.data_dram = (command->a4 & DC_LAYERS_DATA_DRAM) != 0u;
    job.head = (
        job.token && (command->a4 & DC_LAYERS_HEAD) != 0u ? job.words[DC_WORD_HEAD] : 0u
    );
    job.attend = (
        ((command->a4 & 16u) != 0u ? DC_ATTEND_READ_ONLY : 0u) |
        ((command->a4 & 32u) != 0u ? DC_ATTEND_LOCAL : 0u) |
        ((command->a4 & 64u) != 0u ? DC_ATTEND_EXP_TWICE : 0u)
    );
    job.rows = command->a6;
    job.span_ticks = 0ull;
    for (uint32_t part = 0u; part < DC_THREADS; part++) {
        job.thread_ticks[part] = 0ull;
    }
    for (uint32_t step = 0; step < DC_STEPS; step++) {
        job.step_ticks[step] = 0ull;
        job.step_most[step] = 0u;
        job.step_least[step] = 0xFFFFFFFFu;
    }
    const uint32_t generate = kmx_handler(command) == kHandlerGenerate;
    uint64_t made_bytes = 0, row_bytes = 0;
    uint32_t* made = (uint32_t*)kmx_region(REGION_MADE, &made_bytes);
    uint32_t* row = (uint32_t*)kmx_region(REGION_ROW, &row_bytes);
    uint32_t id = command->a7, made_count = 0u;
    if (generate) {
        if (
            !job.head || made == 0 || row == 0 || tokens > DC_MADE_MOST ||
            made_bytes < 4ull * DC_MADE_WORDS * DC_MADE_MOST
        ) {
            return 0xFFFFFFFFu;
        }
        for (uint64_t word = 0u; word < row_bytes / 4u; word++) {
            row[word] = 0u; /* x's other rows and the squares' other places */
        }
    }
    const uint64_t started = kmx_time();
    int status = KMX_OK;
    for (uint32_t token = 0; token < tokens && status == KMX_OK; token++) {
        if (generate) {
            const uint64_t before = kmx_time();
            status = embed(&job, id, before + WAIT_TICKS);
            job.thread_ticks[DC_THREAD_EMBED] += kmx_time() - before;
        }
        for (uint32_t layer = 0; layer < count && status == KMX_OK; layer++) {
            job.layer = weights + DC_LAYER_WORDS * (first + layer);
            job.index = first + layer;
            job.position = command->a5 + token * job.rows;
            kmx_progress(token << 16 | layer);
            if (job.token) {
                status = run_token(&job, command);
                break;
            }
            if (job.chain) {
                status = run_chain(&job);
                continue;
            }
            for (uint32_t step = 0; step < DC_STEPS && status == KMX_OK; step++) {
                if (job.overlap && step == DC_STEP_Q) {
                    status = run_attention(&job);
                    step = DC_STEP_O;
                } else if (runs(&job, step)) {
                    status = run_step(&job, step);
                }
            }
        }
        if (generate && status == KMX_OK) {
            const uint64_t before = kmx_time();
            status = choose(&job, &id, before + WAIT_TICKS);
            const uint64_t now = kmx_time();
            job.thread_ticks[DC_THREAD_CHOOSE] += now - before;
            uint32_t* entry = made + DC_MADE_WORDS * made_count++;
            entry[0] = id;
            entry[1] = (uint32_t)now;
            entry[2] = (uint32_t)(now >> 32);
            entry[3] = (uint32_t)status;
            if (id == command->a8) {
                break;
            }
        }
    }
    if (generate) {
        kmx_flush(REGION_MADE);
    }
    const uint64_t ticks = kmx_time() - started;
    kmx_progress(PROGRESS_END | (uint32_t)status);
    uint32_t* slot = stats + DC_STATS_WORDS * command->a2;
    slot[DC_ST_STATUS] = (uint32_t)status;
    slot[DC_ST_DONE] = job.done;
    slot[DC_ST_TICKS_LOW] = (uint32_t)ticks;
    slot[DC_ST_TICKS_HIGH] = (uint32_t)(ticks >> 32);
    for (uint32_t step = 0; step < DC_STEPS; step++) {
        uint32_t* words = slot + DC_ST_STEPS + DC_ST_STEP_WORDS * step;
        words[0] = (uint32_t)job.step_ticks[step];
        words[1] = (uint32_t)(job.step_ticks[step] >> 32);
        words[2] = job.step_most[step];
        words[3] = job.step_least[step];
    }
    slot[DC_ST_SPAN_LOW] = (uint32_t)job.span_ticks;
    slot[DC_ST_SPAN_HIGH] = (uint32_t)(job.span_ticks >> 32);
    for (uint32_t part = 0u; part < DC_THREADS; part++) {
        slot[DC_ST_THREAD + 2u * part] = (uint32_t)job.thread_ticks[part];
        slot[DC_ST_THREAD + 2u * part + 1u] = (uint32_t)(job.thread_ticks[part] >> 32);
    }
    kmx_flush(2);
    if (generate) {
        return status == KMX_OK ? made_count : 0xFFFFFFF0u | (uint32_t)status;
    }
    return status == KMX_OK ? (uint32_t)ticks : 0xFFFFFFF0u | (uint32_t)status;
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerLayers:
        case kHandlerGenerate:
            return layers(command);
    }
    return 0u;
}
