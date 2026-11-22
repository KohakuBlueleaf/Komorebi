/* The frame on the work cluster: each command submitted and waited for; a trace is cut
 * into chunk commands of equal predicted cost, or pulled (RP_MODE_PULL). */
#include <stdint.h>

#include "kmx.h"
#include "plan.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per command, longer than its deadline */

typedef struct {
    uint32_t pred[RP_PACKETS_MAX]; /* each packet's predicted cost */
    uint32_t bound[RP_CORES_MAX * RP_CHUNKS + 1];
} work_t;

typedef struct {
    const uint32_t* words;
    const uint32_t* table;
    uint32_t workers, packets, capacity, mode, bank_plus_one, per, pulled;
    uint32_t directories_address, costs_address; /* bank addresses */
    uint8_t* directories;
    const uint8_t* costs;
    work_t* work;
    uint32_t* log;
    uint32_t used[RP_TRACES]; /* chunk commands of the trace's last submit */
    uint32_t output[RP_OUT_WORDS];
    void* step; /* the plan's parallel step, after work_t in the work region */
} plan_t;

/* One parallel step: the source trace and how it predicts (source_of), the owners'
 * sums, each quarter's sum and what each X280 core found. */
typedef struct {
    plan_t* context;
    uint32_t source, owner, plus, pieces;
    uint64_t owner_sums[4][RP_CORES_MAX];
    uint64_t quarter_sums[4];
    uint32_t bad_quarters[4];
} part_t;

/* part_t's place in the work region: after work_t, 64-byte aligned (uint64 fields) */
#define PART_AT ((sizeof(work_t) + 63u) & ~(uint64_t)63u)
_Static_assert(PART_AT % 64u == 0, "part_t aligned");

/* Fills `step` field by field (a struct initializer may call memset, which does not
 * link) */
static void part_init(part_t* step, plan_t* context) {
    step->context = context;
    step->source = RP_TRACES;
    step->owner = step->plus = step->pieces = 0;
    for (int quarter = 0; quarter < 4; quarter++) {
        step->quarter_sums[quarter] = 0;
        step->bad_quarters[quarter] = 0;
        for (uint32_t core = 0; core < RP_CORES_MAX; core++) {
            step->owner_sums[quarter][core] = 0;
        }
    }
}

static uint32_t start_of(uint32_t items, int xcore) {
    return (uint32_t)((uint64_t)items * (uint32_t)xcore / 4u);
}

static uint32_t end_of(uint32_t items, int xcore) {
    return start_of(items, xcore + 1);
}

static uint32_t block_at(
    const plan_t* context, uint32_t trace, uint32_t chunk, uint32_t core
) {
    return ((trace * RP_CHUNKS + chunk) * context->workers + core) * RP_COST_BYTES;
}

static uint32_t dir_at(
    const plan_t* context, uint32_t trace, uint32_t chunk, uint32_t core
) {
    return ((trace * RP_CHUNKS + chunk) * context->workers + core) * RP_DIR_BYTES;
}

/* The owner core of `packet` (the owners' ranges are in core order) */
static uint32_t owner_of(const plan_t* context, uint32_t packet) {
    const uint32_t* base = context->table + RP_T_BASE;
    uint32_t low = 0, high = context->workers - 1;
    while (low < high) {
        const uint32_t middle = (low + high + 1) / 2;
        if (base[2 * middle] <= packet) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

/* The source trace's cost blocks of a quarter of the cores: each packet's cycles (or
 * hits, owner bit 1) + plus into pred, or with owner bit 0 summed by owner core */
static void gather_part(int xcore, void* arg) {
    part_t* step = (part_t*)arg;
    plan_t* context = step->context;
    const uint32_t* base = context->table + RP_T_BASE;
    const uint32_t values = step->owner & 2 ? RP_COST_HEAD + RP_COST_MAX : RP_COST_HEAD;
    for (
        uint32_t core = start_of(context->workers, xcore);
        core < end_of(context->workers, xcore);
        core++
    ) {
        for (uint32_t chunk = 0; chunk < context->used[step->source]; chunk++) {
            const uint32_t* block = (
                (const uint32_t*)(
                    context->costs + block_at(context, step->source, chunk, core)
                )
            );
            const uint32_t first_packet = block[0], packets = block[1];
            if (
                packets > RP_COST_MAX || first_packet > context->packets ||
                packets > context->packets - first_packet
            ) {
                step->bad_quarters[xcore] = 1;
                continue;
            }
            if ((step->owner & 1) == 0) {
                for (uint32_t packet = 0; packet < packets; packet++) {
                    context->work->pred
                        [first_packet + packet] = (block[values + packet] + step->plus);
                }
                continue;
            }
            uint32_t owner = packets ? owner_of(context, first_packet) : 0;
            for (uint32_t packet = 0; packet < packets; packet++) {
                while (
                    owner + 1 < context->workers &&
                    base[2 * (owner + 1)] <= first_packet + packet
                ) {
                    owner++;
                }
                step->owner_sums[xcore][owner] += block[values + packet];
            }
        }
    }
}

/* Each owner's packets: the owner's mean + plus */
static void spread_part(int xcore, void* arg) {
    part_t* step = (part_t*)arg;
    plan_t* context = step->context;
    const uint32_t* base = context->table + RP_T_BASE;
    for (
        uint32_t core = start_of(context->workers, xcore);
        core < end_of(context->workers, xcore);
        core++
    ) {
        const uint32_t first_packet = base[2 * core], packets = base[2 * core + 1];
        const uint64_t sum = (
            step->owner_sums[0][core] + step->owner_sums[1][core] +
            step->owner_sums[2][core] + step->owner_sums[3][core]
        );
        const uint32_t mean = (packets ? (uint32_t)(sum / packets) : 0) + step->plus;
        for (
            uint32_t packet = first_packet;
            packet < first_packet + packets && packet < context->packets;
            packet++
        ) {
            context->work->pred[packet] = mean;
        }
    }
}

static void equal_part(int xcore, void* arg) {
    part_t* step = (part_t*)arg;
    for (
        uint32_t packet = start_of(step->context->packets, xcore);
        packet < end_of(step->context->packets, xcore);
        packet++
    ) {
        step->context->work->pred[packet] = 1;
    }
}

static void sum_part(int xcore, void* arg) {
    part_t* step = (part_t*)arg;
    const uint32_t* predictions = step->context->work->pred;
    uint64_t sum = 0;
    for (
        uint32_t packet = start_of(step->context->packets, xcore);
        packet < end_of(step->context->packets, xcore);
        packet++
    ) {
        sum += predictions[packet];
    }
    step->quarter_sums[xcore] = sum;
}

/* pred for `trace`: from the source's blocks, by owner, or equal packets */
static int predict(plan_t* context, part_t* step) {
    if (step->source >= RP_TRACES) {
        return kmx_parallel(equal_part, step) == KMX_OK ? 0 : -1;
    }
    kmx_invalidate_at(
        RP_R_COST,
        block_at(context, step->source, 0, 0),
        (uint64_t)context->used[step->source] * context->workers * RP_COST_BYTES
    );
    if (kmx_parallel(gather_part, step) != KMX_OK) {
        return -1;
    }
    if ((step->owner & 1) && kmx_parallel(spread_part, step) != KMX_OK) {
        return -1;
    }
    return (
        (step->bad_quarters[0] | step->bad_quarters[1] | step->bad_quarters[2] |
             step->bad_quarters[3])
            ? -1
            : 0
    );
}

static uint64_t target_of(uint64_t total, uint32_t pieces, uint32_t piece) {
    return total / pieces * piece + total % pieces * piece / pieces;
}

/* End of piece `piece`: the packet boundary nearest to piece / pieces of the total
 * cost, for this X280 core's quarter (one scan from the quarter's start sum) */
static void cut_part(int xcore, void* arg) {
    part_t* step = (part_t*)arg;
    work_t* work = step->context->work;
    const uint32_t packets = step->context->packets, pieces = step->pieces;
    const uint64_t total = (
        step->quarter_sums[0] + step->quarter_sums[1] + step->quarter_sums[2] +
        step->quarter_sums[3]
    );
    if (total == 0) {
        for (
            uint32_t piece = start_of(pieces, xcore) + (xcore == 0);
            piece < end_of(pieces, xcore);
            piece++
        ) {
            work->bound[piece] = (uint32_t)((uint64_t)packets * piece / pieces);
        }
        return;
    }
    uint64_t before = 0;
    for (int quarter = 0; quarter < xcore; quarter++) {
        before += step->quarter_sums[quarter];
    }
    const uint64_t through = before + step->quarter_sums[xcore];
    uint32_t packet = start_of(packets, xcore), end = end_of(packets, xcore);
    uint64_t cost_before = before; /* the cost before `packet` */
    uint32_t piece = (
        xcore == 0 ? 1 : (uint32_t)(((before + 1) * pieces + total - 1) / total)
    );
    for (; piece < pieces; piece++) {
        const uint64_t target = target_of(total, pieces, piece);
        if (xcore < 3 && target > through) {
            break;
        }
        while (packet < end && cost_before + work->pred[packet] < target) {
            cost_before += work->pred[packet];
            packet++;
        }
        /* the boundary before or after `packet`, whichever is nearer the target */
        uint32_t boundary = packet;
        if (
            packet < end &&
            cost_before + work->pred[packet] - target <= target - cost_before
        ) {
            boundary = packet + 1;
        }
        work->bound[piece] = boundary;
    }
}

/* Cuts into workers x chunks pieces, chunks from the least that can hold the packets,
 * until no piece has more than `capacity` (at RP_CHUNKS, capped from the front). */
static uint32_t cut(plan_t* context, part_t* step) {
    work_t* work = context->work;
    const uint32_t round_capacity = context->workers * context->capacity;
    if (kmx_parallel(sum_part, step) != KMX_OK) {
        return 0;
    }
    uint32_t chunks = (context->packets + round_capacity - 1) / round_capacity;
    for (; chunks <= RP_CHUNKS; chunks++) {
        step->pieces = context->workers * chunks;
        work->bound[0] = 0;
        work->bound[step->pieces] = context->packets;
        if (kmx_parallel(cut_part, step) != KMX_OK) {
            return 0;
        }
        uint32_t largest = 0;
        for (uint32_t piece = 0; piece < step->pieces; piece++) {
            const uint32_t size = work->bound[piece + 1] - work->bound[piece];
            largest = size > largest ? size : largest;
        }
        if (largest <= context->capacity) {
            return chunks;
        }
    }
    chunks = RP_CHUNKS;
    const uint32_t pieces = context->workers * chunks;
    for (uint32_t piece = 0; piece < pieces; piece++) {
        const uint32_t left = (pieces - piece - 1) * context->capacity;
        uint32_t end = work->bound[piece + 1];
        end = (
            end > work->bound[piece] + context->capacity
                ? work->bound[piece] + context->capacity
                : end
        );
        end = (
            context->packets > left && end < context->packets - left
                ? context->packets - left
                : end
        );
        work->bound[piece + 1] = end;
    }
    return work->bound[pieces] == context->packets ? chunks : 0;
}

/* The chunks of a trace as the host cuts them: owner `core`'s `chunk`-th run of
 * `capacity` packets */
static uint32_t host_cut(plan_t* context) {
    const uint32_t* base = context->table + RP_T_BASE;
    uint32_t most = 0;
    for (uint32_t core = 0; core < context->workers; core++) {
        most = base[2 * core + 1] > most ? base[2 * core + 1] : most;
    }
    const uint32_t chunks = (
        most ? (most + context->capacity - 1) / context->capacity : 1
    );
    if (chunks > RP_CHUNKS) {
        return 0;
    }
    for (uint32_t core = 0; core < context->workers; core++) {
        for (uint32_t chunk = 0; chunk < chunks; chunk++) {
            const uint32_t owned = base[2 * core + 1];
            const uint32_t skipped = chunk * context->capacity;
            const uint32_t take = (
                owned > skipped
                    ? (owned - skipped < context->capacity ? owned - skipped
                                                             : context->capacity)
                    : 0
            );
            context->work->bound
                [core * chunks + chunk] = (base[2 * core] + (take ? skipped : 0));
            context->work->pred[core * chunks + chunk] = take; /* the counts, kept
                                                                  beside the starts */
        }
    }
    return chunks;
}

/* Which trace predicts `trace` (RP_TRACES: none) and how. The cold frame of
 * RP_MODE_FRAME: the shadow by the primary's; plan() cuts the rest as the host does. */
static void source_of(
    const plan_t* context, uint32_t trace, uint32_t frame, part_t* step
) {
    step->source = RP_TRACES;
    step->owner = 0;
    step->plus = 0;
    if (context->mode == RP_MODE_FRAME) {
        step->source = frame > 0 ? trace : (trace == 1 ? 0 : RP_TRACES);
    } else if (context->mode == RP_MODE_TRACE && trace >= 1) {
        step->source = trace == 3 ? 2 : 0;
        step->owner = trace == 1 ? 0 : 1;
    } else if (context->mode == RP_MODE_COUNTS && trace >= 1) {
        step->source = trace == 3 ? 2 : 0;
        step->owner = trace == 1 ? 2 : 3;
        step->plus = 1;
    }
    if (step->source < RP_TRACES && context->used[step->source] == 0) {
        step->source = RP_TRACES;
    }
}

/* The directory entries of `trace` for its `chunks` chunk commands, from the bounds */
static void write_dirs(plan_t* context, uint32_t trace, uint32_t chunks, int host) {
    for (uint32_t core = 0; core < context->workers; core++) {
        for (uint32_t chunk = 0; chunk < chunks; chunk++) {
            uint32_t* entry = (
                (uint32_t*)(context->directories + dir_at(context, trace, chunk, core))
            );
            const uint32_t piece = core * chunks + chunk;
            for (uint32_t word = 0; word < RP_DIR_BYTES / 4; word++) {
                entry[word] = 0;
            }
            entry[RP_DIR_FIRST] = context->work->bound[piece];
            entry[RP_DIR_COUNT] = (
                host
                    ? context->work->pred[piece]
                    : context->work->bound[piece + 1] - context->work->bound[piece]
            );
            entry[RP_DIR_COST] = (
                context->costs_address + block_at(context, trace, chunk, core)
            );
        }
    }
    kmx_flush_at(
        RP_R_DIRS,
        dir_at(context, trace, 0, 0),
        (uint64_t)chunks * context->workers * RP_DIR_BYTES
    );
}

/* The plan of `trace` for this frame: its chunk commands' count, 0 on failure */
static uint32_t plan(
    plan_t* context, uint32_t trace, uint32_t frame, uint32_t* sub_ticks
) {
    part_t* step = (part_t*)context->step;
    part_init(step, context);
    source_of(context, trace, frame, step);
    /* the frame mode without a source cuts as the host: equal packets lose (WHY-SLOW)
     */
    if (
        context->mode == RP_MODE_HOST || context->mode == RP_MODE_PULL ||
        (context->mode == RP_MODE_FRAME && step->source == RP_TRACES)
    ) {
        const uint32_t chunks = host_cut(context);
        if (chunks) {
            write_dirs(context, trace, chunks, 1);
        }
        return chunks;
    }
    uint64_t started = kmx_time();
    if (predict(context, step) != 0) {
        return 0;
    }
    sub_ticks[0] += (uint32_t)(kmx_time() - started);
    started = kmx_time();
    const uint32_t chunks = cut(context, step);
    sub_ticks[2] += (uint32_t)(kmx_time() - started);
    started = kmx_time();
    if (chunks) {
        write_dirs(context, trace, chunks, 0);
    }
    sub_ticks[3] += (uint32_t)(kmx_time() - started);
    return chunks;
}

/* Submits `command` and waits for its event: 0, or the code to report */
static uint32_t run(plan_t* context, const km_command_t* command, uint32_t* ticks) {
    const uint64_t until = kmx_time() + WAIT_TICKS;
    uint32_t handle;
    int status = kmx_submit(command, &handle);
    while (status == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        status = kmx_submit(command, &handle);
    }
    if (status != KMX_OK) {
        return RP_RING;
    }
    kmx_event event;
    if (kmx_wait(handle, kmx_time() + WAIT_TICKS, &event) != KMX_OK) {
        return RP_RING;
    }
    *ticks = event.ticks;
    context->output[2] = event.kind;
    context->output[3] = event.auxiliary;
    return event.kind == KM_EVENT_DONE ? 0u : RP_COMMAND;
}

/* Submits the pulled `command` with kmx_pull_run (ranges of `per` packets) and serves
 * it until its event: 0, or the code to report. */
static uint32_t run_pulled(
    plan_t* context, const km_command_t* command, uint32_t* ticks
) {
    kmx_pull pull; /* field by field: an initializer may call memset */
    pull.units = context->packets;
    pull.per = context->per;
    pull.mode = KMX_PULL_COUNT;
    pull.log = 0;
    pull.pick = 0;
    pull.arg = 0;
    pull.pick_word = 0;
    const uint64_t until = kmx_time() + WAIT_TICKS;
    kmx_event event;
    int status = kmx_pull_run(command, &pull, &event);
    while (status == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        status = kmx_pull_run(command, &pull, &event);
    }
    if (status != KMX_OK) {
        return RP_RING;
    }
    *ticks = event.ticks;
    context->output[2] = event.kind;
    context->output[3] = event.auxiliary;
    return event.kind == KM_EVENT_DONE ? 0u : RP_COMMAND;
}

/* Fills `command` field by field (a struct assignment may call memset, which does not
 * link) */
static void fill_command(km_command_t* command, const uint32_t* record) {
    command->sequence = command->land1 = command->land2 = command->land3 = 0;
    command->hdr = record[1];
    command->group = record[2];
    command->a0 = record[3];
    command->a1 = record[4];
    command->a2 = record[5];
    command->a3 = record[6];
    command->a4 = record[7];
    command->a5 = record[8];
    command->a6 = record[9];
    command->a7 = record[10];
    command->a8 = record[11];
    command->a9 = record[12];
}

static uint32_t frame_once(plan_t* context, uint32_t frame) {
    uint32_t* log = context->log + frame * RP_LOG_WORDS;
    uint32_t submitted = 0;
    const uint64_t started = kmx_time();
    for (
        uint32_t command_index = 0; command_index < context->words[RP_W_COMMANDS];
        command_index++
    ) {
        const uint32_t* record = context->table + command_index * RP_REC_WORDS;
        const uint32_t kind = record[0] & 0xFFu;
        if ((record[0] & RP_KIND_FIRST) && frame > 0) {
            continue;
        }
        km_command_t command;
        fill_command(&command, record);
        uint32_t chunks = 1;
        const int pulled = (
            kind != 0 && context->mode == RP_MODE_PULL &&
            ((context->pulled >> (kind - 1)) & 1u)
        );
        if (pulled) {
            /* one command, no plan: the cores pull the trace's packets */
            const uint32_t trace = kind - 1;
            log[RP_L_PLAN + trace] = 0;
            log[RP_L_CHUNKS + trace] = 1;
            context->used[trace] = 0; /* no cost blocks */
            command.a3 |= RP_PULLED;
            command.a4 = context->table[RP_T_PROF + trace * RP_CHUNKS];
            uint32_t ticks = 0;
            const uint32_t code = run_pulled(context, &command, &ticks);
            if (submitted < RP_LOG_WORDS - RP_L_COMMAND) {
                log[RP_L_COMMAND + submitted] = ticks;
            }
            submitted++;
            if (code) {
                context->output[1] = command_index;
                return code;
            }
            continue;
        }
        if (kind != 0) {
            const uint32_t trace = kind - 1;
            const uint64_t plan_started = kmx_time();
            chunks = plan(context, trace, frame, log + RP_L_SUB);
            log[RP_L_PLAN + trace] = (uint32_t)(kmx_time() - plan_started);
            log[RP_L_CHUNKS + trace] = chunks;
            if (chunks == 0) {
                context->output[1] = command_index;
                return RP_PLAN;
            }
            context->used[trace] = chunks;
            command.a9 |= context->bank_plus_one << RP_BANK_SHIFT;
        }
        for (uint32_t chunk = 0; chunk < chunks; chunk++) {
            if (kind != 0) {
                const uint32_t trace = kind - 1;
                command.a0 = (
                    context->directories_address + dir_at(context, trace, chunk, 0)
                );
                command.a4 = context->table[RP_T_PROF + trace * RP_CHUNKS + chunk];
            }
            uint32_t ticks = 0;
            const uint32_t code = run(context, &command, &ticks);
            if (submitted < RP_LOG_WORDS - RP_L_COMMAND) {
                log[RP_L_COMMAND + submitted] = ticks;
            }
            submitted++;
            if (code) {
                context->output[1] = command_index;
                return code;
            }
        }
    }
    log[RP_L_SUBMITTED] = submitted;
    log[RP_L_FRAME] = (uint32_t)(kmx_time() - started);
    return 0;
}

static int finish(uint32_t code, plan_t* context) {
    kmx_flush(RP_R_LOG);
    kmx_result(code, context->output, RP_OUT_WORDS);
    return 0;
}

int kmx_main(int xcore) {
    (void)xcore;
    const kmx_args* arguments = kmx_args_get();
    plan_t context;
    for (uint32_t word = 0; word < RP_OUT_WORDS; word++) {
        context.output[word] = 0;
    }
    for (uint32_t trace = 0; trace < RP_TRACES; trace++) {
        context.used[trace] = 0;
    }
    if (arguments->nwords < RP_WORDS || arguments->nregions < RP_REGIONS) {
        return finish(RP_BAD_ARGS, &context);
    }
    context.words = arguments->words;
    const uint64_t needed[RP_REGIONS] = {
        4u * RP_TABLE_WORDS,
        RP_DIRS_BYTES,
        RP_COST_REGION_BYTES,
        PART_AT + sizeof(part_t),
        RP_LOG_BYTES,
    };
    void* regions[RP_REGIONS];
    for (int region = 0; region < (int)RP_REGIONS; region++) {
        uint64_t bytes;
        regions[region] = kmx_region(region, &bytes);
        if (
            regions[region] == 0 || bytes < needed[region] ||
            arguments->region[region].offset >> 32
        ) {
            return finish(RP_BAD_ARGS, &context);
        }
    }
    context.table = (const uint32_t*)regions[RP_R_TABLE];
    context.directories = (uint8_t*)regions[RP_R_DIRS];
    context.costs = (const uint8_t*)regions[RP_R_COST];
    context.work = (work_t*)regions[RP_R_WORK];
    context.step = (uint8_t*)regions[RP_R_WORK] + PART_AT;
    context.log = (uint32_t*)regions[RP_R_LOG];
    context.directories_address = (uint32_t)arguments->region[RP_R_DIRS].offset;
    context.costs_address = (uint32_t)arguments->region[RP_R_COST].offset;
    context.mode = context.words[RP_W_MODE];
    context.workers = context.words[RP_W_WORKERS];
    context.packets = context.words[RP_W_PACKETS];
    context.capacity = context.words[RP_W_CAP];
    context.bank_plus_one = 1 + context.words[RP_W_BANK];
    context.per = context.words[RP_W_PER];
    context.pulled = context.words[RP_W_PULLED];
    const uint32_t frames = context.words[RP_W_FRAMES];
    if (
        context.mode > RP_MODE_PULL ||
        (context.mode == RP_MODE_PULL && context.per == 0) || context.workers == 0 ||
        context.workers > RP_CORES_MAX || context.packets == 0 ||
        context.packets > RP_PACKETS_MAX || context.capacity == 0 ||
        context.capacity > RP_COST_MAX ||
        context.words[RP_W_COMMANDS] > RP_COMMANDS_MAX || frames == 0 ||
        frames > RP_FRAMES_MAX
    ) {
        return finish(RP_BAD_ARGS, &context);
    }
    /* lines an earlier image left dirty would be written over what the cores write */
    kmx_flush(RP_R_COST);
    for (uint32_t frame = 0; frame < frames; frame++) {
        kmx_progress(frame + 1);
        const uint32_t code = frame_once(&context, frame);
        if (code) {
            return finish(code, &context);
        }
        context.output[0] = frame + 1;
    }
    return finish(RP_OK, &context);
}
