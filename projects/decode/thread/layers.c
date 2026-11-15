/* Decode, the thread: LAYERS runs every layer of every token, each layer's commands in
 * order (decode.h, DC_STEP_*), each one waited for before the next. */
#include <stdint.h>

#include "kmx.h"
#include "decode.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* one command's submit or wait */
#define PROGRESS_END 0xC0000000u        /* the last progress marker: the job's status */
#define KM_SPLIT_FULL 0u
#define KM_SPLIT_PARTS 1u
#define KM_SPLIT_OP 2u
#define KM_SPLIT_SECOND_SET 0x100u /* a3: the sum's upper ports and slots */

typedef struct {
    const uint32_t* words;
    const uint32_t* tensors;
    const uint32_t* layer;
    uint64_t step_ticks[DC_STEPS];
    uint32_t step_most[DC_STEPS], step_least[DC_STEPS];
    uint32_t done;
    uint32_t bare;       /* products only: no operation, no norm */
    uint32_t parts;      /* and each product's parts only: no sum, nothing written */
    uint32_t overlap;    /* q, k, v, o: each submitted before the one ahead is done */
    uint64_t span_ticks; /* q's submit to o's event, summed over the layers */
} job_t;

/* Overlapped, the attention products alternate the sum's two sets: k and o the second.
 */
static int overlapped(const job_t* job, uint32_t step) {
    return job->overlap && step >= DC_STEP_Q && step <= DC_STEP_O;
}

/* A SPLIT command on the product's cores: A, W, Y, the plan and the operation. */
static void product(
    const job_t* job,
    uint32_t step,
    km_command_t* command,
    uint32_t a,
    uint32_t w,
    uint32_t y,
    uint32_t plan,
    uint32_t operation,
    uint32_t other
) {
    const uint32_t* words = job->words;
    kmx_command(command, 0u, words[DC_WORD_SPLIT], 0u, words[plan] >> 8);
    if (!overlapped(job, step)) {
        command->hdr |= words[DC_WORD_EXCLUSIVE] << 8;
    }
    command->a0 = a;
    command->a1 = w;
    command->a2 = y;
    command->a3 = words[DC_WORD_REDUCE];
    if (overlapped(job, step) && (step == DC_STEP_K || step == DC_STEP_O)) {
        command->a3 |= KM_SPLIT_SECOND_SET;
    }
    command->a4 = job->parts ? KM_SPLIT_PARTS : KM_SPLIT_FULL;
    command->a5 = words[plan] & 0xFFu;
    command->a6 = operation == 0u || job->bare ? 0u : words[operation];
    command->a7 = other;
}

/* A norm command on DC_NORM_CORES cores: x into h with the layer's weight `weight`. */
static void norm(const job_t* job, km_command_t* command, uint32_t weight) {
    const uint32_t* words = job->words;
    const uint32_t* tensors = job->tensors;
    kmx_command(command, 0u, words[DC_WORD_SPLIT], 0u, DC_NORM_CORES);
    command->a1 = words[DC_WORD_EPSILON];
    command->a2 = tensors[DC_TENSOR_H];
    command->a4 = KM_SPLIT_OP;
    command->a6 = words[DC_WORD_NORM];
    command->a7 = tensors[DC_TENSOR_X];
    command->a8 = job->layer[weight];
    command->a9 = tensors[DC_TENSOR_CONSTANTS];
}

/* Step `step`'s command of the job's layer. */
static void step_command(const job_t* job, uint32_t step, km_command_t* command) {
    const uint32_t* t = job->tensors;
    const uint32_t* w = job->layer;
    switch (step) {
        case DC_STEP_NORM1:
            norm(job, command, DC_W_NORM1);
            return;
        case DC_STEP_Q:
            product(
                job,
                step,
                command,
                t[DC_TENSOR_H],
                w[DC_W_Q],
                t[DC_TENSOR_Q],
                DC_WORD_PLAN_Q,
                0u,
                0u
            );
            return;
        case DC_STEP_K:
            product(
                job,
                step,
                command,
                t[DC_TENSOR_H],
                w[DC_W_K],
                t[DC_TENSOR_K],
                DC_WORD_PLAN_KV,
                0u,
                0u
            );
            return;
        case DC_STEP_V:
            product(
                job,
                step,
                command,
                t[DC_TENSOR_H],
                w[DC_W_V],
                t[DC_TENSOR_V],
                DC_WORD_PLAN_KV,
                0u,
                0u
            );
            return;
        case DC_STEP_O: /* attention left out: its output stands in as q */
            product(
                job,
                step,
                command,
                t[DC_TENSOR_Q],
                w[DC_W_O],
                t[DC_TENSOR_X],
                DC_WORD_PLAN_Q,
                DC_WORD_ADD,
                t[DC_TENSOR_X]
            );
            return;
        case DC_STEP_NORM2:
            norm(job, command, DC_W_NORM2);
            return;
        case DC_STEP_UP:
            product(
                job,
                step,
                command,
                t[DC_TENSOR_H],
                w[DC_W_UP],
                t[DC_TENSOR_UP],
                DC_WORD_PLAN_UP,
                0u,
                0u
            );
            return;
        case DC_STEP_GATE:
            product(
                job,
                step,
                command,
                t[DC_TENSOR_H],
                w[DC_W_GATE],
                t[DC_TENSOR_GATED],
                DC_WORD_PLAN_UP,
                DC_WORD_GATE,
                t[DC_TENSOR_UP]
            );
            return;
        default: /* DC_STEP_DOWN */
            product(
                job,
                step,
                command,
                t[DC_TENSOR_GATED],
                w[DC_W_DOWN],
                t[DC_TENSOR_X],
                DC_WORD_PLAN_DOWN,
                DC_WORD_ADD,
                t[DC_TENSOR_X]
            );
            return;
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
    const uint64_t started = kmx_time();
    uint32_t ahead = 0u, handle = 0u;
    int status = submit_step(job, DC_STEP_Q, &ahead);
    for (uint32_t step = DC_STEP_K; step <= DC_STEP_O && status == KMX_OK; step++) {
        status = submit_step(job, step, &handle);
        if (status == KMX_OK) {
            status = wait_step(job, step - 1u, ahead);
            ahead = handle;
        }
    }
    if (status == KMX_OK) {
        status = wait_step(job, DC_STEP_O, ahead);
    }
    job->span_ticks += kmx_time() - started;
    return status;
}

/* LAYERS: a0 tokens, a1 layers, a2 stats slot, a3 first layer, a4 its bits (decode.h).
 * The result word: the ticks taken, or 0xFFFFFFF0 | status. */
static uint32_t layers(const km_command_t* command) {
    uint64_t weight_bytes = 0, tensor_bytes = 0, stats_bytes = 0;
    const uint32_t* weights = (const uint32_t*)kmx_region(0, &weight_bytes);
    const uint32_t* tensors = (const uint32_t*)kmx_region(1, &tensor_bytes);
    uint32_t* stats = (uint32_t*)kmx_region(2, &stats_bytes);
    const kmx_args* args = kmx_args_get();
    const uint32_t tokens = command->a0, count = command->a1, first = command->a3;
    if (
        weights == 0 || tensors == 0 || stats == 0 || args->nwords < DC_WORDS ||
        command->a2 >= DC_STATS_SLOTS ||
        4ull * DC_STATS_WORDS * DC_STATS_SLOTS > stats_bytes || count == 0 ||
        4ull * DC_LAYER_WORDS * (first + count) > weight_bytes ||
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
    job.overlap = (command->a4 & 4u) != 0u;
    job.span_ticks = 0ull;
    for (uint32_t step = 0; step < DC_STEPS; step++) {
        job.step_ticks[step] = 0ull;
        job.step_most[step] = 0u;
        job.step_least[step] = 0xFFFFFFFFu;
    }
    const uint64_t started = kmx_time();
    int status = KMX_OK;
    for (uint32_t token = 0; token < tokens && status == KMX_OK; token++) {
        for (uint32_t layer = 0; layer < count && status == KMX_OK; layer++) {
            job.layer = weights + DC_LAYER_WORDS * (first + layer);
            kmx_progress(token << 16 | layer);
            for (uint32_t step = 0; step < DC_STEPS && status == KMX_OK; step++) {
                if (job.overlap && step == DC_STEP_Q) {
                    status = run_attention(&job);
                    step = DC_STEP_O;
                } else if (
                    !job.bare || (step != DC_STEP_NORM1 && step != DC_STEP_NORM2)
                ) {
                    status = run_step(&job, step);
                }
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
    for (uint32_t step = 0; step < DC_STEPS; step++) {
        uint32_t* words = slot + DC_ST_STEPS + DC_ST_STEP_WORDS * step;
        words[0] = (uint32_t)job.step_ticks[step];
        words[1] = (uint32_t)(job.step_ticks[step] >> 32);
        words[2] = job.step_most[step];
        words[3] = job.step_least[step];
    }
    slot[DC_ST_SPAN_LOW] = (uint32_t)job.span_ticks;
    slot[DC_ST_SPAN_HIGH] = (uint32_t)(job.span_ticks >> 32);
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
