/* The two-card decode, a card's thread: its half of every layer as the chain's records,
 * a crossing after each product (two.h), a layer or a token as one command. */
#include <stdint.h>

#include "kmx.h"
#include "kmx_chain.h"
#include "chain.h"
#include "decode.h"
#include "two.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* one command's submit or wait */
#define PROGRESS_END 0xC0000000u        /* the last progress marker: the job's status */
#define KM_SPLIT_FULL 0u
#define KM_SPLIT_OP 2u
#define TOKEN_NONE 0xFFFFFFFFu /* token_form's last word: no records written */

typedef struct {
    const uint32_t* words;
    const uint32_t* tensors;
    const uint32_t* layer;
    uint32_t done;
    uint32_t position; /* the first token's place in the cache */
    uint32_t index; /* the layer's number: its head norms' weights in the qk tensor */
    uint32_t local; /* each crossing marks locally only (DC2_LAYERS_LOCAL) */
    uint32_t ahead_off;
    uint32_t cores; /* the chain command's: every rank of the island */
    uint64_t thread_ticks[DC_THREADS];
    uint64_t chain_ticks;
} job_t;

/* A SPLIT command on `plan`'s cores from this island's first rank on. */
static void split_command(const job_t* job, km_command_t* command, uint32_t plan) {
    const uint32_t* words = job->words;
    const uint32_t word = words[plan];
    kmx_command(command, 0u, words[DC_WORD_SPLIT], words[DC2_WORD_FIRST], word >> 8);
    command->hdr |= (words[DC_WORD_EXCLUSIVE] | KM_FLAG_PART) << 8;
    command->a3 = words[DC_WORD_REDUCE] | 1u << DC_ROWS_SHIFT;
    command->a5 = word & 0xFFu;
}

/* A product: A, W, Y, the plan, the operation and its three words (decode.h). */
static void product(
    const job_t* job,
    km_command_t* command,
    uint32_t a,
    uint32_t w,
    uint32_t y,
    uint32_t plan,
    uint32_t operation,
    uint32_t o0,
    uint32_t o1,
    uint32_t o2
) {
    split_command(job, command, plan);
    command->a0 = a;
    command->a1 = w;
    command->a2 = y;
    command->a4 = KM_SPLIT_FULL;
    command->a6 = job->words[operation];
    command->a7 = o0;
    command->a8 = o1;
    command->a9 = o2;
}

/* A crossing on the leader alone: the activation's table word, this card's first page
 * and pages, the squares' word and pages (0: none); local: the mark alone. */
static void crossing(
    const job_t* job,
    km_command_t* command,
    uint32_t tensor,
    uint32_t pages,
    uint32_t squares
) {
    const uint32_t* words = job->words;
    const uint32_t* t = job->tensors;
    split_command(job, command, DC2_WORD_PLAN_CROSS);
    command->group = (words[DC2_WORD_FIRST] + words[DC2_WORD_LEADER]) | 1u << 16;
    command->a0 = t[tensor];
    command->a1 = words[DC2_WORD_CARD] * pages;
    command->a2 = job->local ? 0u : pages;
    command->a4 = KM_SPLIT_OP;
    command->a6 = words[DC2_WORD_CROSS];
    command->a7 = squares != 0u ? t[squares] : 0u;
    command->a8 = squares != 0u && !job->local ? DC2_SQUARES_HALF : 0u;
    command->a9 = words[DC2_WORD_FAR_LEADER];
}

/* Record `record`'s command of the job's layer (DC2_RECORD_*). */
static void record_command(const job_t* job, uint32_t record, km_command_t* command) {
    const uint32_t* t = job->tensors;
    const uint32_t* w = job->layer;
    const uint32_t* words = job->words;
    const uint32_t epsilon = words[DC_WORD_EPSILON];
    switch (record) {
        case DC2_RECORD_QKV: /* x's norm, then q's and k's head norms and rotation */
            product(
                job,
                command,
                t[DC2_TENSOR_X],
                w[DC_W_QKV],
                t[DC2_TENSOR_QKV],
                DC_WORD_PLAN_QKV,
                DC_WORD_SCALE,
                t[DC2_TENSOR_SQUARES1],
                t[DC2_TENSOR_QK],
                job->position | job->index << 16
            );
            command->a3 |= DC_QK;
            return;
        case DC2_RECORD_ATTEND: /* this card's KV heads, into its half of a */
            split_command(job, command, DC_WORD_PLAN_ATTEND);
            command->a0 = t[DC2_TENSOR_QKV];
            command->a1 = w[DC_W_KEYS];
            command->a2 = t[DC2_TENSOR_A_HALF];
            command->a4 = KM_SPLIT_OP;
            command->a6 = words[DC_WORD_ATTEND];
            command->a7 = w[DC_W_VALUES];
            command->a8 = t[DC2_TENSOR_ATTEND];
            command->a9 = job->position | words[DC_WORD_CACHE_TILES] << 16;
            return;
        case DC2_RECORD_G1:
            crossing(job, command, DC2_WORD_A, DC_HEAD_GROUP * DC_HEAD_TILES * 4u, 0u);
            return;
        case DC2_RECORD_O: /* x1 = x + o, x1's squares: this card's half */
            product(
                job,
                command,
                t[DC2_TENSOR_A],
                w[DC_W_O],
                t[DC2_TENSOR_X1_HALF],
                DC2_WORD_PLAN_O,
                DC_WORD_ADD,
                t[DC2_TENSOR_X_HALF],
                t[DC2_TENSOR_CONSTANTS],
                t[DC2_TENSOR_SQUARES2]
            );
            return;
        case DC2_RECORD_G2:
            crossing(job, command, DC2_WORD_X1, DC_ROW_TILES / 2u, DC2_WORD_SQUARES2);
            return;
        case DC2_RECORD_GATE_UP: /* x1's norm, then the gate */
            product(
                job,
                command,
                t[DC2_TENSOR_X1],
                w[DC_W_GATE_UP],
                t[DC2_TENSOR_GATED_HALF],
                DC_WORD_PLAN_GATE_UP,
                DC_WORD_GATE,
                t[DC2_TENSOR_SQUARES2],
                t[DC2_TENSOR_CONSTANTS],
                epsilon
            );
            return;
        case DC2_RECORD_G3:
            crossing(job, command, DC2_WORD_GATED, words[DC2_WORD_GATE_TILES], 0u);
            return;
        case DC2_RECORD_DOWN: /* x = x1 + down, x's squares: this card's half */
            product(
                job,
                command,
                t[DC2_TENSOR_GATED],
                w[DC_W_DOWN],
                t[DC2_TENSOR_X_HALF],
                DC_WORD_PLAN_DOWN,
                DC_WORD_ADD,
                t[DC2_TENSOR_X1_HALF],
                t[DC2_TENSOR_CONSTANTS],
                t[DC2_TENSOR_SQUARES1]
            );
            return;
        default: /* DC2_RECORD_G4 */
            crossing(job, command, DC2_WORD_X, DC_ROW_TILES / 2u, DC2_WORD_SQUARES1);
            return;
    }
}

static int is_crossing(uint32_t record) {
    return (
        record == DC2_RECORD_G1 || record == DC2_RECORD_G2 || record == DC2_RECORD_G3 ||
        record == DC2_RECORD_G4
    );
}

/* A crossing's width: the tiles of the next record's input it releases. */
static uint32_t crossing_width(const job_t* job, uint32_t record) {
    return (
        record == DC2_RECORD_G3 ? 2u * job->words[DC2_WORD_GATE_TILES] : DC_ROW_TILES
    );
}

/* The stamp row of a record: the products' as on one card, then G1, G2 and G4, G3. */
static uint32_t stamp_row(uint32_t record) {
    static const uint32_t rows[DC2_RECORDS] = {0u, 1u, 5u, 2u, 6u, 3u, 7u, 4u, 6u};
    return rows[record];
}

/* The chain command's cores: every record's, from the island's first rank. */
static uint32_t chain_cores(const uint32_t* words) {
    static const uint32_t plans[] = {
        DC_WORD_PLAN_QKV,
        DC_WORD_PLAN_ATTEND,
        DC2_WORD_PLAN_O,
        DC_WORD_PLAN_GATE_UP,
        DC_WORD_PLAN_DOWN
    };
    uint32_t cores = words[DC2_WORD_LEADER] + 1u;
    for (uint32_t i = 0u; i < sizeof(plans) / sizeof(plans[0]); i++) {
        cores = (words[plans[i]] >> 8) > cores ? words[plans[i]] >> 8 : cores;
    }
    return cores;
}

/* The job's form its token records were written for (a4, layers, then 1 once written),
 * after the load none (the thread has no zeroed data). */
static uint32_t token_form[3] = {0u, 0u, TOKEN_NONE};

/* The records of `count` layers into region 3, each layer's last naming the next
 * layer's first; `token`: the position from the chain command's a0. */
static int chain_write(job_t* job, uint32_t count, int token) {
    const uint32_t* words = job->words;
    const uint32_t total = DC2_RECORDS * count;
    if (total > kChainRecordsMax) {
        return KMX_BAD;
    }
    const uint32_t* layer = job->layer;
    const uint32_t index = job->index, position = job->position;
    if (token) {
        job->position = 0u;
    }
    km_command_t command;
    uint32_t address = 0u;
    for (uint32_t g = total; g-- > 0u;) { /* last first: each knows the next */
        const uint32_t record = g % DC2_RECORDS;
        job->layer = layer + DC_LAYER_WORDS * (g / DC2_RECORDS);
        job->index = index + g / DC2_RECORDS;
        record_command(job, record, &command);
        kmx_chain_call call; /* field by field: no memset */
        call.command = &command;
        call.word = call.by_ranges = call.k_base = call.k_tiles = 0u;
        call.ranges[0] = call.ranges[1] = call.ranges[2] = 0u;
        call.size = call.width = call.row = call.handler = 0u;
        call.flags = g > 0u ? kChainMarked : 0u;
        if (!is_crossing(record)) {
            call.flags |= kChainSends;
        }
        if (token && (record == DC2_RECORD_QKV || record == DC2_RECORD_ATTEND)) {
            call.word = 15u; /* a9: qkv's (the op's) and attend's */
        }
        call.by_ranges = record == DC2_RECORD_ATTEND;
        for (uint32_t range = 0u; call.by_ranges && range < kChainRangeCount; range++) {
            call.ranges[range] = words[DC_WORD_ATTEND_NEEDS + range];
        }
        if (record == DC2_RECORD_GATE_UP) {
            call.width = words[DC_WORD_GATE_TILES];
        } else if (is_crossing(record)) {
            call.width = crossing_width(job, record);
        }
        call.row = stamp_row(record);
        if (kmx_chain_record(3, g, &call, address, &address) != KMX_OK) {
            return KMX_BAD;
        }
    }
    job->layer = layer;
    job->index = index;
    job->position = position;
    kmx_flush(3);
    return KMX_OK;
}

/* One SPLIT in the chain's mode on every rank of the island from region 3's first
 * record, a0 the token's word; waited for once. */
static int chain_run(job_t* job, uint32_t a0) {
    const uint32_t* words = job->words;
    km_command_t chain;
    const uint64_t written = kmx_time();
    if (kmx_chain_command(
        &chain,
        3,
        words[DC_WORD_SPLIT],
        job->cores,
        kChainMode,
        a0,
        job->ahead_off ? kChainAheadOff : 0u
    ) != KMX_OK) {
        return KMX_BAD;
    }
    chain.group = words[DC2_WORD_FIRST] | job->cores << 16;
    chain.hdr |= (words[DC_WORD_EXCLUSIVE] | KM_FLAG_PART) << 8;
    uint32_t handle = 0u;
    uint64_t deadline = kmx_time() + WAIT_TICKS;
    int status = kmx_submit(&chain, &handle);
    while (status == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
        status = kmx_submit(&chain, &handle);
    }
    const uint64_t submitted = kmx_time();
    job->thread_ticks[DC_THREAD_SUBMIT] += submitted - written;
    if (status == KMX_OK) {
        kmx_event event;
        status = kmx_wait(handle, kmx_time() + WAIT_TICKS, &event);
        if (status == KMX_OK && event.kind != KM_EVENT_DONE) {
            status = KMX_BAD;
        }
        if (status == KMX_OK) {
            job->done++;
            job->chain_ticks += event.ticks;
        }
    }
    job->thread_ticks[DC_THREAD_WAIT] += kmx_time() - submitted;
    return status;
}

/* LAYERS2: a0 tokens, a1 layers, a2 stats slot, a3 first layer, a4 bits (decode.h's
 * DC_LAYERS_TOKEN, DC_LAYERS_AHEAD_OFF; DC2_LAYERS_LOCAL), a5 the first position. */
static uint32_t layers(const km_command_t* command) {
    uint64_t weight_bytes = 0, tensor_bytes = 0, stats_bytes = 0;
    const uint32_t* weights = (const uint32_t*)kmx_region(0, &weight_bytes);
    const uint32_t* tensors = (const uint32_t*)kmx_region(1, &tensor_bytes);
    uint32_t* stats = (uint32_t*)kmx_region(2, &stats_bytes);
    const kmx_args* args = kmx_args_get();
    const uint32_t tokens = command->a0, count = command->a1, first = command->a3;
    if (
        weights == 0 || tensors == 0 || stats == 0 || args->nwords < DC2_WORDS ||
        command->a2 >= DC_STATS_SLOTS ||
        4ull * DC_STATS_WORDS * DC_STATS_SLOTS > stats_bytes || count == 0 ||
        count > DC_QK_LAYERS || first > DC_QK_LAYERS - count ||
        4ull * DC_LAYER_WORDS * ((uint64_t)first + count) > weight_bytes ||
        4ull * DC2_TENSORS > tensor_bytes
    ) {
        return 0xFFFFFFFFu;
    }
    kmx_invalidate(0);
    kmx_invalidate(1);
    job_t job;
    job.words = args->words;
    job.tensors = tensors;
    job.done = 0u;
    job.local = (command->a4 & DC2_LAYERS_LOCAL) != 0u;
    job.ahead_off = (command->a4 & DC_LAYERS_AHEAD_OFF) != 0u;
    job.cores = chain_cores(args->words);
    job.chain_ticks = 0ull;
    for (uint32_t part = 0u; part < DC_THREADS; part++) {
        job.thread_ticks[part] = 0ull;
    }
    const int token = (command->a4 & DC_LAYERS_TOKEN) != 0u;
    const uint64_t started = kmx_time();
    int status = KMX_OK;
    for (uint32_t step = 0; step < tokens && status == KMX_OK; step++) {
        job.layer = weights + DC_LAYER_WORDS * first;
        job.index = first;
        job.position = command->a5 + step;
        kmx_progress(step << 16);
        if (token) { /* every layer's records once a form, then a command a token */
            const uint32_t form[2] = {command->a4, count};
            if (
                token_form[2] != 1u || token_form[0] != form[0] ||
                token_form[1] != form[1]
            ) {
                const uint64_t at = kmx_time();
                token_form[2] = TOKEN_NONE;
                status = chain_write(&job, count, 1);
                job.thread_ticks[DC_THREAD_RECORDS] += kmx_time() - at;
                token_form[0] = form[0];
                token_form[1] = form[1];
                token_form[2] = status == KMX_OK ? 1u : TOKEN_NONE;
            }
            if (status == KMX_OK) {
                status = chain_run(&job, job.position);
            }
            continue;
        }
        for (uint32_t layer = 0; layer < count && status == KMX_OK; layer++) {
            job.layer = weights + DC_LAYER_WORDS * (first + layer);
            job.index = first + layer;
            const uint64_t at = kmx_time();
            token_form[2] = TOKEN_NONE; /* region 3 holds this layer's records now */
            status = chain_write(&job, 1u, 0);
            job.thread_ticks[DC_THREAD_RECORDS] += kmx_time() - at;
            if (status == KMX_OK) {
                status = chain_run(&job, 0u);
            }
        }
    }
    const uint64_t ticks = kmx_time() - started;
    kmx_progress(PROGRESS_END | (uint32_t)status);
    uint32_t* slot = stats + DC_STATS_WORDS * command->a2;
    slot[DC_ST_STATUS] = (uint32_t)status;
    slot[DC_ST_DONE] = job.done;
    slot[DC_ST_TICKS_LOW] = (uint32_t)ticks;
    slot[DC_ST_TICKS_HIGH] = (uint32_t)(ticks >> 32);
    uint32_t* chain = slot + DC_ST_STEPS + DC_ST_STEP_WORDS * DC_STEP_CHAIN;
    chain[0] = (uint32_t)job.chain_ticks;
    chain[1] = (uint32_t)(job.chain_ticks >> 32);
    for (uint32_t part = 0u; part < DC_THREADS; part++) {
        slot[DC_ST_THREAD + 2u * part] = (uint32_t)job.thread_ticks[part];
        slot[DC_ST_THREAD + 2u * part + 1u] = (uint32_t)(job.thread_ticks[part] >> 32);
    }
    kmx_flush(2);
    return status == KMX_OK ? (uint32_t)ticks : 0xFFFFFFF0u | (uint32_t)status;
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerLayers:
            return layers(command);
    }
    return 0u;
}
