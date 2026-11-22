/* The frame on the work cluster: COUNT, SUM, HIST, HSUM, the middle pass (middle.c),
 * PLACE, SETUP, FILL, each submitted and waited for (words, regions: frame.h). */

/* Built with RF_SERVE, kmx_main sets up and the thread serves FRAME commands: a frame
 * and SHOW each, from the camera the host wrote before the command. */
#include <stdint.h>

#include "frame.h"
#include "kmx.h"
#include "middle.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per command, longer than its deadline */

typedef struct {
    rm_scratch scratch;
    rm_item items[RM_MAX_ITEMS];
    uint32_t tile_item[RM_TILES + 1];
    uint32_t heavy[RM_MAX_HEAVY], bands[RM_MAX_HEAVY];
    uint32_t ranges[RF_MAX_CORES + 1]; /* COUNT's and SETUP's blocks of each core */
    uint32_t cuts[RF_MAX_CORES + 1];
} work_t;

typedef struct {
    const uint32_t* words;
    uint32_t bank_plus_one; /* 1 + the regions' bank */
    uint32_t address[RF_REGIONS];
    uint32_t result[RF_OUT_WORDS];
    uint32_t ranged; /* frames run with ranges: the costs of the last are in */
    uint32_t ready;  /* the next frame's ranges are in (set during FILL) */
} frame_t;

/* The middle pass's parts on the four cores: each a quarter of the tiles or items. */
typedef struct {
    const uint32_t* counters;
    uint32_t cores, workers, item_count;
    work_t* work;
} part_t;

static void costs_part(int core, void* arg) {
    const part_t* part = (const part_t*)arg;
    const uint32_t quarter = (uint32_t)core;
    rm_costs(
        part->counters,
        part->cores,
        &part->work->scratch,
        RM_TILES * quarter / 4,
        RM_TILES * (quarter + 1) / 4
    );
}

static void sort_part(int core, void* arg) {
    const part_t* part = (const part_t*)arg;
    const uint32_t quarter = (uint32_t)core;
    rm_sort(
        &part->work->scratch,
        part->item_count * quarter / 4,
        part->item_count * (quarter + 1) / 4
    );
}

/* the quarters in halves on cores 0 and 1 */
static void merge_part(int core, void* arg) {
    const part_t* part = (const part_t*)arg;
    const uint32_t quarter = (uint32_t)core;
    if (quarter < 2) {
        rm_merge(
            &part->work->scratch,
            part->item_count * 2 * quarter / 4,
            part->item_count * (2 * quarter + 1) / 4,
            part->item_count * (2 * quarter + 2) / 4
        );
    }
}

/* Fills `command` field by field (a struct assignment may call memset, which does not
 * link). */
static void new_command(km_command_t* command, uint32_t handler, const frame_t* frame) {
    command->sequence = command->land1 = command->land2 = command->land3 = 0;
    command->hdr = handler;
    command->group = frame->words[RF_W_WORKERS] << 16;
    command->a0 = (
        command->a1 = command->a2 = command->a3 = command->a4 = command
                                                                    ->a6 = command
                                                                               ->a7 = 0
    );
    command->a9 = frame->words[RF_W_SPROF]; /* profile_address */
    command->a5 = frame->words[RF_W_CORES]; /* cores */
    command->a8 = frame->bank_plus_one;     /* bank */
}

/* Submits `command`: 0 with its handle, or the code to report. */
static uint32_t submit(const km_command_t* command, uint32_t* handle) {
    const uint64_t until = kmx_time() + WAIT_TICKS;
    int error = kmx_submit(command, handle);
    while (error == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        error = kmx_submit(command, handle);
    }
    return error == KMX_OK ? 0u : RF_RING;
}

/* Waits for a submitted command's event: 0, or the code to report. */
static uint32_t wait_for(uint32_t handle) {
    kmx_event event;
    if (kmx_wait(handle, kmx_time() + WAIT_TICKS, &event) != KMX_OK) {
        return RF_RING;
    }
    return event.kind == KM_EVENT_DONE ? 0u : RF_COMMAND;
}

/* Submits `command` and waits for its event: 0, or the code to report. */
static uint32_t run(const km_command_t* command) {
    uint32_t handle;
    const uint32_t code = submit(command, &handle);
    return code ? code : wait_for(handle);
}

static int finish(uint32_t code, frame_t* frame) {
    kmx_result(code, frame->result, RF_OUT_WORDS);
    return 0;
}

static uint32_t elapsed(uint64_t* mark) {
    const uint64_t now = kmx_time();
    const uint32_t ticks = (uint32_t)(now - *mark);
    *mark = now;
    return ticks;
}

/* Each core's blocks for COUNT and SETUP, into the camera's region: equal partitions,
 * then equal parts of the last frame's predicted cost (kernels/setup.h kCostKnots). */
static void set_ranges(frame_t* frame, work_t* work, void* const* regions) {
    const uint32_t cores = frame->words[RF_W_CORES], blocks = frame->words[RF_W_BLOCKS];
    uint32_t* block_ranges = work->ranges;
    volatile uint32_t* camera = (volatile uint32_t*)regions[RF_R_CAMERA];
    uint64_t total = 0;
    if (frame->ranged) {
        kmx_invalidate_at(RF_R_CAMERA, RF_COSTS_AT, 128u * cores);
        for (uint32_t core = 0; core < cores; core++) {
            const uint32_t word = (RF_COSTS_AT + 128u * core) / 4;
            if (camera[word] != block_ranges[core + 1] - block_ranges[core]) {
                total = 0; /* not the last frame's ranges: equal partitions */
                break;
            }
            total += camera[word + RF_COST_KNOTS];
        }
    }
    if (total == 0) {
        for (uint32_t core = 0; core <= cores; core++) {
            block_ranges[core] = (uint32_t)((uint64_t)core * blocks / cores);
        }
    } else {
        uint32_t* cut_blocks = work->cuts;
        uint32_t cut_index = 1;
        uint64_t want = total / cores; /* the cost before cut `cut_index` */
        uint64_t base = 0;             /* the cost of the cores before `core` */
        cut_blocks[0] = 0;
        /* each region word read costs about 30 ns: a cut's knots found by halves */
        for (uint32_t core = 0; core < cores && cut_index < cores; core++) {
            const uint32_t word = (RF_COSTS_AT + 128u * core) / 4;
            const uint64_t core_end = base + camera[word + RF_COST_KNOTS];
            const uint32_t first = block_ranges[core];
            const uint32_t count = block_ranges[core + 1] - first;
            while (cut_index < cores && want <= core_end) {
                uint32_t low = 1, high = RF_COST_KNOTS; /* the first knot >= want */
                while (low < high) {
                    const uint32_t middle = (low + high) / 2;
                    if (base + camera[word + middle] >= want) {
                        high = middle;
                    } else {
                        low = middle + 1;
                    }
                }
                const uint64_t cost = base + camera[word + low];
                const uint64_t last_cost = (
                    base + (low > 1 ? camera[word + low - 1] : 0)
                );
                const uint32_t place = first + low * count / RF_COST_KNOTS;
                uint32_t block = first + (low - 1) * count / RF_COST_KNOTS;
                if (want > last_cost) {
                    block += (
                        (uint32_t)(
                            (want - last_cost) * (place - block) / (cost - last_cost)
                        )
                    );
                }
                cut_blocks[cut_index] = (
                    block > cut_blocks[cut_index - 1]
                        ? block
                        : cut_blocks[cut_index - 1]
                );
                cut_index++;
                want = total * cut_index / cores;
            }
            base = core_end;
        }
        for (; cut_index < cores; cut_index++) {
            cut_blocks[cut_index] = blocks;
        }
        cut_blocks[cores] = blocks;
        for (uint32_t cut_index = 0; cut_index <= cores; cut_index++) {
            block_ranges[cut_index] = cut_blocks[cut_index];
        }
    }
    for (uint32_t core = 0; core < cores; core++) {
        camera[(RF_RANGES_AT + 64u * core) / 4] = block_ranges[core];
        camera[(RF_RANGES_AT + 64u * core) / 4 + 1] = block_ranges[core + 1];
    }
    kmx_flush_at(RF_R_CAMERA, RF_RANGES_AT, 64u * cores);
    frame->ranged++;
}

/* One frame: COUNT to FILL (or as far as RF_W_STEPS says); adds each part's ticks to
 * the result words. 0, or the code to report. */
static uint32_t frame_once(frame_t* frame, work_t* work, void* const* regions) {
    const uint32_t cores = frame->words[RF_W_CORES];
    const uint32_t workers = frame->words[RF_W_WORKERS];
    uint64_t mark = kmx_time();

    /* COUNT */
    km_command_t command;
    kmx_progress(1);
    /* fewer blocks than cores: a block each at most, nothing to balance */
    const uint32_t ranges = (
        frame->words[RF_W_BALANCE] && cores <= RF_MAX_CORES &&
            frame->words[RF_W_BLOCKS] >= cores
            ? RF_RANGES
            : 0
    );
    if (ranges && !frame->ready) {
        set_ranges(frame, work, regions);
    }
    frame->ready = 0;
    new_command(&command, RF_HANDLER_COUNT, frame);
    command.a0 = frame->words[RF_W_MESH];            /* mesh_address */
    command.a1 = frame->words[RF_W_COUNTERS];        /* counters_address */
    command.a2 = frame->words[RF_W_BLOCKS] | ranges; /* blocks_word */
    command.a3 = frame->address[RF_R_CAMERA];        /* camera_address */
    uint32_t code = run(&command);
    frame->result[RF_O_COUNT] += elapsed(&mark);
    if (code || frame->words[RF_W_STEPS] < 2) {
        return code;
    }

    /* SUM: the counters' totals, and each core's entries before it per tile */
    kmx_progress(2);
    new_command(&command, RF_HANDLER_SUM, frame);
    command.a1 = frame->words[RF_W_COUNTERS]; /* counters_address */
    command.a2 = frame->words[RF_W_SLOTS];    /* prefixes_address */
    command.a3 = frame->address[RF_R_SUMS];   /* totals_address */
    command.a6 = cores;                       /* counting_cores */
    code = run(&command);
    frame->result[RF_O_SUM] += elapsed(&mark);
    if (code) {
        return code;
    }

    /* the tiles to cut, and their histograms' sums */
    kmx_invalidate(RF_R_SUMS);
    frame->result[RF_O_TAKE] += elapsed(&mark);
    const uint32_t* sums = (const uint32_t*)regions[RF_R_SUMS];
    part_t part;
    part.counters = sums; /* the totals, as one core's counters */
    part.cores = 1;
    part.workers = workers;
    part.work = work;
    if (kmx_parallel(costs_part, &part) != KMX_OK) {
        return RF_PARALLEL;
    }
    const uint32_t heavy_count = (
        rm_pick(&work->scratch, workers, work->heavy, work->bands)
    );
    frame->result[RF_O_HEAVY] = heavy_count;
    if (heavy_count > RF_HIST_MAX) {
        return RF_TOO_MANY_HEAVY;
    }
    frame->result[RF_O_HEAVY_MICROSECONDS] += elapsed(&mark);
    if (heavy_count) {
        volatile uint32_t* heavy_words = (volatile uint32_t*)regions[RF_R_HEAVY];
        heavy_words[0] = heavy_count;
        for (uint32_t heavy_index = 0; heavy_index < heavy_count; heavy_index++) {
            heavy_words[1 + heavy_index] = work->heavy[heavy_index];
        }
        kmx_flush(RF_R_HEAVY);
        new_command(&command, RF_HANDLER_HIST, frame);
        command.a0 = frame->address[RF_R_HEAVY]; /* heavy_address */
        command.a1 = frame->words[RF_W_HIST];    /* histograms_address */
        command.a4 = RF_HIST_BYTES;              /* histogram_page_bytes */
        kmx_progress(3);
        code = run(&command);
        frame->result[RF_O_HIST] += elapsed(&mark);
        if (code) {
            return code;
        }
        new_command(&command, RF_HANDLER_HSUM, frame);
        command.a0 = heavy_count;                /* heavy_tiles */
        command.a1 = frame->words[RF_W_HIST];    /* histograms_address */
        command.a2 = frame->words[RF_W_HPREFIX]; /* prefixes_address */
        command.a3 = frame->address[RF_R_SUMS];  /* totals_address */
        command.a4 = RF_HIST_BYTES;              /* histogram_page_bytes */
        command.a6 = cores;                      /* counting_cores */
        code = run(&command);
        frame->result[RF_O_HSUM] += elapsed(&mark);
        if (code) {
            return code;
        }
        kmx_invalidate(RF_R_SUMS);
        const uint32_t* heavy_sums = sums + RF_SUMS_HEAVY_AT / 4;
        for (uint32_t heavy_index = 0; heavy_index < heavy_count; heavy_index++) {
            for (uint32_t group = 0; group < 2 * RM_GROUPS; group++) {
                work->scratch.histograms
                    [heavy_index]
                    [group] = (heavy_sums[2 * RM_GROUPS * heavy_index + group]);
            }
        }
        frame->result[RF_O_TAKE] += elapsed(&mark);
    }

    /* middle pass: items, cores, lists' layout in words for PLACE, base slots */
    kmx_progress(4);
    uint32_t item_count = 0;
    if (!rm_items(
        work->heavy,
        work->bands,
        heavy_count,
        &work->scratch,
        work->items,
        &item_count,
        work->tile_item
    )) {
        return RF_NO_LAYOUT;
    }
    frame->result[RF_O_CUT] += elapsed(&mark);
    part.item_count = item_count;
    if (
        kmx_parallel(sort_part, &part) != KMX_OK ||
        kmx_parallel(merge_part, &part) != KMX_OK
    ) {
        return RF_PARALLEL;
    }
    rm_merge(&work->scratch, 0, item_count / 2, item_count);
    frame->result[RF_O_SORT] += elapsed(&mark);
    if (!rm_owners(workers, &work->scratch, item_count)) {
        return RF_NO_LAYOUT;
    }
    frame->result[RF_O_ASSIGN] += elapsed(&mark);
    uint32_t* items = (uint32_t*)regions[RF_R_ITEMS];
    const uint32_t slots = (
        rm_compact(
            workers,
            &work->scratch,
            work->items,
            item_count,
            items + RF_ITEM_WORDS_AT / 4,
            items + RF_CORE_WORDS_AT / 4
        )
    );
    uint32_t mark_count = workers;
    for (uint32_t i = 0; i < item_count; i++) {
        mark_count += work->items[i].total != 0;
    }
    frame->result[RF_O_SLOTS] = slots;
    frame->result[RF_O_ITEMS] = item_count;
    frame->result[RF_O_MARKS] = mark_count;
    if (slots == 0 || item_count > RF_SLOTS_BYTES / 4) {
        return RF_NO_LAYOUT;
    }
    if (slots > frame->words[RF_W_LIST_SLOTS] || slots >= 1u << RM_BASE_SLOT_BITS) {
        return RF_LISTS_SMALL;
    }
    rm_base(&work->scratch, work->items, item_count, items + RF_BASE_AT / 4);
    frame->result[RF_O_COMPACT] += elapsed(&mark);
    for (uint32_t i = 0; i < RM_TILES; i++) {
        const uint32_t first = work->tile_item[i];
        items[i] = first | (work->tile_item[i + 1] - first) << 16;
    }
    uint16_t* rows = (uint16_t*)(items + RM_TILES);
    for (uint32_t i = 0; i < item_count; i++) {
        rows[i] = (uint16_t)(work->items[i].first_row | work->items[i].end_row << 8);
    }
    items[RF_HPREFIX_ADDR_AT / 4] = frame->words[RF_W_HPREFIX];
    uint32_t* frame_words = items + RF_FRAME_WORDS_AT / 4;
    frame_words[0] = frame->words[RF_W_FRAME_PAGES];
    for (uint32_t frame_slot = 0; frame_slot < RF_FRAME_MAX; frame_slot++) {
        frame_words[1 + frame_slot] = (
            frame_slot < frame->words[RF_W_FRAMES]
                ? frame->words[RF_W_FRAME + frame_slot]
                : 0
        );
    }
    frame->result[RF_O_PAGES] += elapsed(&mark);
    kmx_flush(RF_R_ITEMS);
    frame->result[RF_O_GIVE] += elapsed(&mark);
    frame->result[RF_O_MIDDLE] = (
        frame->result[RF_O_TAKE] + frame->result[RF_O_HEAVY_MICROSECONDS] +
        frame->result[RF_O_CUT] + frame->result[RF_O_SORT] +
        frame->result[RF_O_ASSIGN] + frame->result[RF_O_COMPACT] +
        frame->result[RF_O_PAGES] + frame->result[RF_O_GIVE]
    );
    if (frame->words[RF_W_STEPS] < 3) {
        return 0;
    }

    /* PLACE, then SETUP */
    kmx_progress(5);
    new_command(&command, RF_HANDLER_PLACE, frame);
    command.a0 = frame->address[RF_R_ITEMS]; /* items_region_address */
    command.a1 = frame->words[RF_W_DIR];     /* directory_address */
    command.a5 = workers; /* every fill core writes its directory page */
    command.a6 = workers; /* fill_cores */
    command.a7 = frame->words[RF_W_LISTS]; /* lists_address */
    code = run(&command);
    frame->result[RF_O_PLACE] += elapsed(&mark);
    if (code) {
        return code;
    }
    kmx_progress(6);
    new_command(&command, RF_HANDLER_SETUP, frame);
    command.a0 = frame->words[RF_W_MESH];            /* mesh_address */
    command.a1 = frame->words[RF_W_LISTS];           /* lists_address */
    command.a2 = frame->words[RF_W_BLOCKS] | ranges; /* blocks_word */
    command.a3 = frame->address[RF_R_CAMERA];        /* camera_address */
    command.a4 = frame->words[RF_W_SLOTS];           /* slots_address */
    command.a6 = slots | RF_SETUP_PREFIX;            /* list_slots */
    command.a7 = frame->address[RF_R_ITEMS];         /* items_address */
    code = run(&command);
    frame->result[RF_O_SETUP] += elapsed(&mark);
    if (code || frame->words[RF_W_STEPS] < 4) {
        return code;
    }

    /* FILL */
    kmx_progress(7);
    new_command(&command, RF_HANDLER_FILL, frame);
    command.a0 = frame->words[RF_W_DIR];   /* directory_address */
    command.a1 = frame->words[RF_W_LISTS]; /* lists_address */
    command.a3 = RF_PAGES_PER_ITEM;        /* pages_per_item */
    command.a4 = frame->words[RF_W_PROF];  /* profile_address */
    uint32_t handle;
    code = submit(&command, &handle);
    if (code == 0) {
        /* the next frame's ranges while the grid fills: COUNT and SETUP are done */
        if (ranges) {
            set_ranges(frame, work, regions);
            frame->ready = 1;
        }
        code = wait_for(handle);
    }
    frame->result[RF_O_FILL] += elapsed(&mark);
    kmx_progress(8);
    return code;
}

/* The thread's state, kept from kmx_main for the FRAME commands (in .data: the image
 * links no .bss). */
static frame_t g_frame __attribute__((section(".data"))) = {0};
static void* g_regions[RF_REGIONS] __attribute__((section(".data"))) = {0};
#ifdef RF_SERVE
static uint32_t g_set_up __attribute__((section(".data"))) = RF_NOT_SERVING;
static uint32_t g_served __attribute__((section(".data"))) = 0;
#endif

int kmx_main(int core) {
    (void)core;
    const kmx_args* arguments = kmx_args_get();
    frame_t* frame = &g_frame;
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        frame->result[i] = 0;
    }
    if (arguments->nwords < RF_WORDS || arguments->nregions < RF_REGIONS) {
        return finish(RF_BAD_ARGS, frame);
    }
    frame->words = arguments->words;
    frame->bank_plus_one = 1 + frame->words[RF_W_BANK];
    const uint32_t cores = frame->words[RF_W_CORES];
    const uint32_t workers = frame->words[RF_W_WORKERS];
    frame->ranged = frame->ready = 0;
    const uint32_t need[RF_REGIONS] = {
        frame->words[RF_W_BALANCE] ? RF_CAMERA_BYTES : 4 * 16,
        RF_HEAVY_BYTES,
        RF_ITEMS_REGION_BYTES,
        sizeof(work_t),
        RF_SUMS_BYTES,
        RF_TIMES_BYTES,
        RF_SHOW_TABLE_BYTES,
        RF_PUSH_BYTES,
    };
    void** regions = g_regions;
    for (int i = 0; i < (int)RF_REGIONS; i++) {
        uint64_t bytes;
        regions[i] = kmx_region(i, &bytes);
        if (regions[i] == 0 || bytes < need[i] || arguments->region[i].offset >> 32) {
            return finish(RF_BAD_ARGS, frame);
        }
        frame->address[i] = (uint32_t)arguments->region[i].offset;
    }
    if (
        cores == 0 || cores > workers || workers > RM_MAX_CORES ||
        frame->words[RF_W_FRAMES] > RF_FRAME_MAX
    ) {
        return finish(RF_BAD_ARGS, frame);
    }
    work_t* work = (work_t*)regions[RF_R_WORK];
    /* lines an earlier image left dirty would be written over what the cores write */
    kmx_flush(RF_R_SUMS);
    kmx_flush(RF_R_CAMERA);
    /* SHOW's thresholds as the host wrote them at the load, not an earlier image's */
    kmx_invalidate(RF_R_SHOW);
    const uint32_t frames = frame->words[RF_W_REPEAT];
#ifdef RF_SERVE
    if (frames == 0) {
        g_set_up = frame->words[RF_W_DISPLAY] ? 0u : RF_BAD_ARGS;
        return finish(g_set_up, frame);
    }
#endif
    for (uint32_t repeat = 0; repeat < (frames ? frames : 1); repeat++) {
        const uint32_t code = frame_once(frame, work, regions);
        if (code) {
            return finish(code, frame);
        }
        frame->result[RF_O_FRAMES] = repeat + 1;
    }
    return finish(0, frame);
}

#ifdef RF_SERVE
/* SHOW: the frame's colour into the display, a page a tile (kernels/show.h). 0, or
 * the code to report. */
static uint32_t show(frame_t* frame) {
    km_command_t command;
    new_command(&command, RF_HANDLER_SHOW, frame);
    command.a0 = frame->words[RF_W_DIR];      /* directory_address */
    command.a1 = frame->words[RF_W_DISPLAY];  /* display_address */
    command.a2 = frame->words[RF_W_SHOWPROF]; /* profile_address */
    command.a3 = RF_DISPLAY_PAGES;            /* display_pages */
    if (kmx_block(&command, RF_R_SHOW, 0, RF_SHOW_TABLE_BYTES) != KMX_OK) {
        return RF_BAD_ARGS;
    }
    return run(&command);
}

_Static_assert(KM_TRANSFER_NONCE == 7u, "a transfer's nonce is its a7");

/* The display's PUSH from the words the host wrote into the push region, its nonce word
 * XORed with `key` (projects/nn's TRANSFER entry). 0, or the code to report. */
static uint32_t push(uint32_t key) {
    kmx_invalidate_at(RF_R_PUSH, 0, RF_PUSH_BYTES);
    const volatile uint32_t* words = (const volatile uint32_t*)g_regions[RF_R_PUSH];
    const uint32_t header = words[RF_PUSH_HEADER], group = words[RF_PUSH_GROUP];
    km_command_t command;
    kmx_command(&command, header >> 24, header & 0xFFu, group & 0xFFFFu, group >> 16);
    command.hdr |= header & 0xFF00u;
    command.a0 = words[RF_PUSH_ARGS + 0];
    command.a1 = words[RF_PUSH_ARGS + 1];
    command.a2 = words[RF_PUSH_ARGS + 2];
    command.a3 = words[RF_PUSH_ARGS + 3];
    command.a4 = words[RF_PUSH_ARGS + 4];
    command.a5 = words[RF_PUSH_ARGS + 5];
    command.a6 = words[RF_PUSH_ARGS + 6];
    command.a7 = words[RF_PUSH_ARGS + 7] ^ key; /* KM_TRANSFER_NONCE */
    command.a8 = words[RF_PUSH_ARGS + 8];
    command.a9 = words[RF_PUSH_ARGS + 9];
    return run(&command);
}

/* FRAME: one frame from the camera at the start of its region, then SHOW, then (a1)
 * the display's PUSH; its parts' ticks go to a row of the times region. */
static uint32_t serve_frame(const km_command_t* command) {
    frame_t* frame = &g_frame;
    const uint64_t started = kmx_time();
    kmx_invalidate_at(RF_R_CAMERA, 0, 64);
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        frame->result[i] = 0;
    }
    uint32_t code = frame_once(frame, (work_t*)g_regions[RF_R_WORK], g_regions);
    uint32_t show_ticks = 0;
    if (code == 0) {
        const uint64_t show_started = kmx_time();
        code = show(frame);
        show_ticks = (uint32_t)(kmx_time() - show_started);
    }
    uint32_t push_ticks = 0;
    if (code == 0 && command->a1) {
        const uint64_t push_started = kmx_time();
        code = push(command->a0);
        push_ticks = (uint32_t)(kmx_time() - push_started);
    }
    frame->result[RF_O_FRAMES] = 1;
    const uint32_t row = (g_served % RF_TIMES_ROWS) * RF_TIMES_ROW_WORDS;
    volatile uint32_t* times = (volatile uint32_t*)g_regions[RF_R_TIMES] + row;
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        times[i] = frame->result[i];
    }
    times[RF_T_SHOW] = show_ticks;
    times[RF_T_PUSH] = push_ticks;
    times[RF_T_NUMBER] = g_served;
    times[RF_T_CODE] = code;
    times[RF_T_FRAME] = (uint32_t)(kmx_time() - started);
    kmx_flush_at(RF_R_TIMES, 4u * row, 4u * RF_TIMES_ROW_WORDS);
    g_served++;
    return code;
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerFrame:
            return g_set_up ? g_set_up : serve_frame(command);
    }
    return 0;
}
#endif
