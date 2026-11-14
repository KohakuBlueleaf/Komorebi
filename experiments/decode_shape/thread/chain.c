/* Decode shape, the thread: CHAIN submits every product of every layer of every token
 * as K-split commands to the grid and waits for them, at most `window` in flight. */
#include <stdint.h>

#include "kmx.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* one command's submit or wait */
#define WINDOW_MAX 8u
#define CHAINS_MAX 2u
#define STATS_WORDS 16u
#define STATS_SLOTS 16u
#define PRODUCTS 7u       /* products a layer */
#define REJECTED_BUSY 11u /* the rejection reason of a command that may come again */
#define REJECTED_MAX 65536u
#define PROGRESS_END 0xC0000000u /* the last progress marker: the chain's status */

/* The thread's words: the grid's SPLIT handler and the reduce form. */
#define WORD_SPLIT 0u
#define WORD_REDUCE 1u

/* A slot's words in region "stats". */
#define ST_STATUS 0u
#define ST_DONE 1u
#define ST_REJECTED 2u
#define ST_NOT_DONE 3u
#define ST_TICKS_LOW 4u
#define ST_TICKS_HIGH 5u
#define ST_EVENT_LOW 6u
#define ST_EVENT_HIGH 7u
#define ST_EVENT_MAX 8u
#define ST_EVENT_MIN 9u

/* One chain's commands in flight, oldest first: handle and product index. */
typedef struct {
    uint32_t handle[WINDOW_MAX];
    uint32_t product[WINDOW_MAX];
    uint32_t first, count;
} inflight;

typedef struct {
    const uint32_t* weights;
    const uint32_t* buffers; /* (a, y) of chain c at 2 * (pair + c) */
    uint32_t layers, mode, flags, cores, pair, chains;
    uint64_t event_ticks;
    uint32_t done, rejected, not_done, event_max, event_min;
} job_t;

/* Product `index` of chain `chain`: layer (index / 7 + chain * layers / 2) % layers,
 * A and Y the chain's two buffers in turn. */
static void product(
    const job_t* job, uint32_t chain, uint32_t index, km_command_t* command
) {
    const uint32_t* words = kmx_args_get()->words;
    const uint32_t layer_shift = chain * (job->layers / 2u);
    const uint32_t layer = (index / PRODUCTS + layer_shift) % job->layers;
    const uint32_t* pair = job->buffers + 2u * (job->pair + chain);
    kmx_command(command, 0u, words[WORD_SPLIT], chain * job->cores, job->cores);
    command->hdr |= job->flags << 8;
    command->a0 = pair[index & 1u];
    command->a1 = job->weights[layer * PRODUCTS + index % PRODUCTS];
    command->a2 = pair[(index + 1u) & 1u];
    command->a3 = words[WORD_REDUCE];
    command->a4 = job->mode;
}

/* Submits product `index` of `chain`, retrying a full ring: KMX_OK and its handle. */
static int submit(const job_t* job, uint32_t chain, uint32_t index, uint32_t* handle) {
    km_command_t command;
    product(job, chain, index, &command);
    const uint64_t deadline = kmx_time() + WAIT_TICKS;
    int status = kmx_submit(&command, handle);
    while (status == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
        status = kmx_submit(&command, handle);
    }
    return status;
}

/* Waits for the oldest command of `queue`; a rejected one is submitted again at the
 * tail. KMX_OK, or the first failure. */
static int retire(job_t* job, uint32_t chain, inflight* queue) {
    const uint32_t at = queue->first % WINDOW_MAX;
    kmx_event event;
    int status = kmx_wait(queue->handle[at], kmx_time() + WAIT_TICKS, &event);
    if (status != KMX_OK) {
        return status;
    }
    const uint32_t index = queue->product[at];
    queue->first++;
    queue->count--;
    if (event.kind == KM_EVENT_REJECTED) {
        /* only a busy dispatcher is worth a retry, and not without end */
        job->rejected++;
        if (event.auxiliary != REJECTED_BUSY || job->rejected > REJECTED_MAX) {
            return KMX_BAD;
        }
        const uint32_t tail = (queue->first + queue->count) % WINDOW_MAX;
        status = submit(job, chain, index, &queue->handle[tail]);
        if (status == KMX_OK) {
            queue->product[tail] = index;
            queue->count++;
        }
        return status;
    }
    if (event.kind != KM_EVENT_DONE) {
        job->not_done++;
        return KMX_LATE;
    }
    job->done++;
    job->event_ticks += event.ticks;
    job->event_max = event.ticks > job->event_max ? event.ticks : job->event_max;
    job->event_min = event.ticks < job->event_min ? event.ticks : job->event_min;
    return KMX_OK;
}

/* CHAIN: a0 tokens, a1 layers, a2 mode, a3 command flags, a4 window, a5 cores a
 * chain, a6 chains, a7 buffer pair, a8 stats slot. The result word: ticks taken. */
static uint32_t chain(const km_command_t* command) {
    uint64_t weight_bytes = 0, buffer_bytes = 0, stats_bytes = 0;
    const uint32_t* weights = (const uint32_t*)kmx_region(0, &weight_bytes);
    const uint32_t* buffers = (const uint32_t*)kmx_region(1, &buffer_bytes);
    uint32_t* stats = (uint32_t*)kmx_region(2, &stats_bytes);
    const uint32_t tokens = command->a0, window = command->a4;
    job_t job = {
        weights,
        buffers,
        command->a1,
        command->a2,
        command->a3,
        command->a5,
        command->a7,
        command->a6,
        0ull,
        0u,
        0u,
        0u,
        0u,
        0xFFFFFFFFu,
    };
    if (
        weights == 0 || buffers == 0 || stats == 0 || command->a8 >= STATS_SLOTS ||
        4ull * STATS_WORDS * STATS_SLOTS > stats_bytes || job.layers == 0 ||
        4ull * PRODUCTS * job.layers > weight_bytes || window == 0 ||
        window > WINDOW_MAX || job.chains == 0 || job.chains > CHAINS_MAX ||
        8ull * (job.pair + job.chains) > buffer_bytes
    ) {
        return 0xFFFFFFFFu;
    }
    kmx_invalidate(0);
    kmx_invalidate(1);
    inflight queues[CHAINS_MAX];
    for (uint32_t c = 0; c < CHAINS_MAX; c++) {
        queues[c].first = queues[c].count = 0u;
    }
    const uint32_t products = tokens * job.layers * PRODUCTS;
    const uint64_t started = kmx_time();
    int status = KMX_OK;
    for (uint32_t index = 0; index < products && status == KMX_OK; index++) {
        for (uint32_t c = 0; c < job.chains && status == KMX_OK; c++) {
            inflight* queue = &queues[c];
            while (queue->count >= window && status == KMX_OK) {
                status = retire(&job, c, queue);
            }
            const uint32_t tail = (queue->first + queue->count) % WINDOW_MAX;
            if (status == KMX_OK) {
                kmx_progress(index + 1u); /* the product being submitted */
                status = submit(&job, c, index, &queue->handle[tail]);
            }
            if (status == KMX_OK) {
                queue->product[tail] = index;
                queue->count++;
            }
        }
    }
    for (uint32_t c = 0; c < job.chains; c++) {
        while (queues[c].count > 0u && status == KMX_OK) {
            status = retire(&job, c, &queues[c]);
        }
    }
    const uint64_t ticks = kmx_time() - started;
    kmx_progress(PROGRESS_END | (uint32_t)status);
    uint32_t* slot = stats + STATS_WORDS * command->a8;
    slot[ST_STATUS] = (uint32_t)status;
    slot[ST_DONE] = job.done;
    slot[ST_REJECTED] = job.rejected;
    slot[ST_NOT_DONE] = job.not_done;
    slot[ST_TICKS_LOW] = (uint32_t)ticks;
    slot[ST_TICKS_HIGH] = (uint32_t)(ticks >> 32);
    slot[ST_EVENT_LOW] = (uint32_t)job.event_ticks;
    slot[ST_EVENT_HIGH] = (uint32_t)(job.event_ticks >> 32);
    slot[ST_EVENT_MAX] = job.event_max;
    slot[ST_EVENT_MIN] = job.event_min;
    kmx_flush(2);
    return status == KMX_OK ? (uint32_t)ticks : 0xFFFFFFF0u | (uint32_t)status;
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerChain:
            return chain(command);
    }
    return 0u;
}
