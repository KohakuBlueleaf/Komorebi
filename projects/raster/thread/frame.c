/* The frame on the work cluster: COUNT, SUM, HIST, HSUM, the middle pass (middle.c),
 * PLACE, SETUP, FILL, each submitted and waited for (words, regions: frame.h). */

/* RF_SERVE: kmx_main sets up, the thread serves FRAME (each layer's frame, SHOW;
 * glass: LIGHT linear then glass) and on a lit frame SUN commands. */
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
    /* COUNT's and SETUP's blocks of each core, per area of the camera's region */
    uint32_t ranges[RF_AREAS][RF_MAX_CORES + 1];
    uint32_t cuts[RF_MAX_CORES + 1];
} work_t;

typedef struct {
    const uint32_t* words;
    uint32_t bank_plus_one; /* 1 + the regions' bank */
    uint32_t address[RF_REGIONS];
    uint32_t result[RF_OUT_WORDS];
    /* per area (frame.h RF_AREAS): frames run with ranges, the costs of the last are
     * in; the next frame's ranges are in (set during FILL) */
    uint32_t ranged[RF_AREAS];
    uint32_t ready[RF_AREAS];
    /* the camera page COUNT, SETUP, SHADE read (a SUN command: the sun region's) */
    uint32_t camera;
    uint32_t area; /* the area of the camera's region this layer's frame uses */
    /* the grid word of the commands (frame.h RF_GRID_*; a SUN command: the sun's), its
     * tiles, and the camera region's bytes (the eye's grid word in it, RF_GRID_AT) */
    uint32_t grid, tiles;
    uint64_t camera_bytes;
} frame_t;

/* The middle pass's parts on the four cores: each a quarter of the tiles or items. */
typedef struct {
    const uint32_t* counters;
    uint32_t cores, workers, item_count, tiles;
    work_t* work;
} part_t;

static void costs_part(int core, void* arg) {
    const part_t* part = (const part_t*)arg;
    const uint32_t quarter = (uint32_t)core;
    rm_costs(
        part->counters,
        part->cores,
        &part->work->scratch,
        part->tiles * quarter / 4,
        part->tiles * (quarter + 1) / 4
    );
}

/* The tiles of a grid word (frame.h RF_GRID_*), or 0 when it is not one. */
static uint32_t grid_tiles(uint32_t grid) {
    const uint32_t across = grid & 0xFFu, down = grid >> 8 & 0xFFu;
    if (grid == 0) {
        return RF_GRID_DEFAULT * RF_GRID_DEFAULT;
    }
    if (
        grid >> 16 || across == 0 || down == 0 || across > RF_GRID_ACROSS_MAX ||
        down > RF_GRID_DOWN_MAX || across * down > RM_TILES
    ) {
        return 0;
    }
    return across * down;
}

/* The commands' grid: the eye's word from the camera's region (when it holds one),
 * else 32 x 32. 0, or the code to report. */
static uint32_t eye_grid(frame_t* frame, void* const* regions) {
    uint32_t grid = 0;
    if (frame->camera_bytes >= RF_GRID_AT + 64u) {
        kmx_invalidate_at(RF_R_CAMERA, RF_GRID_AT, 64);
        grid = ((const volatile uint32_t*)regions[RF_R_CAMERA])[RF_GRID_AT / 4];
    }
    const uint32_t tiles = grid_tiles(grid);
    if (tiles == 0) {
        return RF_BAD_GRID;
    }
    frame->grid = grid;
    frame->tiles = tiles;
    return 0;
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
    command->a9 = frame->words[RF_W_SPROF];                /* profile_address */
    command->a5 = frame->words[RF_W_CORES];                /* cores */
    command->a8 = frame->bank_plus_one | frame->grid << 8; /* bank, grid */
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

/* Each core's blocks for COUNT and SETUP, into the layer's area: equal partitions, then
 * equal parts of the last frame's predicted cost (kernels/setup.h kCostKnots). */
static void set_ranges(frame_t* frame, work_t* work, void* const* regions) {
    const uint32_t cores = frame->words[RF_W_CORES], blocks = frame->words[RF_W_BLOCKS];
    const uint32_t area = frame->area, at = area * RF_AREA_BYTES;
    uint32_t* block_ranges = work->ranges[area];
    volatile uint32_t* camera = (volatile uint32_t*)regions[RF_R_CAMERA] + at / 4;
    uint64_t total = 0;
    if (frame->ranged[area]) {
        kmx_invalidate_at(RF_R_CAMERA, at + RF_COSTS_AT, 128u * cores);
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
    kmx_flush_at(RF_R_CAMERA, at + RF_RANGES_AT, 64u * cores);
    frame->ranged[area]++;
}

/* The camera page COUNT and SETUP read: with ranges the layer's area (but area 0's, a
 * copy of the frame's camera), else the frame's camera. */
static uint32_t area_camera(frame_t* frame, void* const* regions, uint32_t ranges) {
    if (!ranges || frame->area == 0) {
        return frame->camera;
    }
    const uint32_t at = frame->area * RF_AREA_BYTES;
    const uint32_t sun = frame->area >= RF_AREA_SUN && frame->area < RF_AREA_GLASS;
    const uint32_t from = sun ? RF_R_SUN : RF_R_CAMERA;
    /* the host's camera, or the sun's camera of the area's quadrant */
    const uint32_t from_at = (
        sun
            ? RF_SUN_CAMERA_BYTES * ((frame->area - RF_AREA_SUN) / RF_AREAS_A_QUADRANT)
            : 0
    );
    kmx_invalidate_at(from, from_at, 64);
    const volatile uint32_t*
        source = ((const volatile uint32_t*)regions[from] + from_at / 4);
    volatile uint32_t* page = (volatile uint32_t*)regions[RF_R_CAMERA] + at / 4;
    for (uint32_t i = 0; i < 16; i++) {
        page[i] = source[i];
    }
    kmx_flush_at(RF_R_CAMERA, at, 64);
    return frame->address[RF_R_CAMERA] + at;
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
    if (ranges) { /* each layer's area keeps its own next ranges */
        if (!frame->ready[frame->area]) {
            set_ranges(frame, work, regions);
        }
        frame->ready[frame->area] = 0;
    }
    const uint32_t camera = area_camera(frame, regions, ranges);
    new_command(&command, RF_HANDLER_COUNT, frame);
    command.a0 = frame->words[RF_W_MESH];            /* mesh_address */
    command.a1 = frame->words[RF_W_COUNTERS];        /* counters_address */
    command.a2 = frame->words[RF_W_BLOCKS] | ranges; /* blocks_word */
    command.a3 = camera;                             /* camera_address */
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
    part.tiles = frame->tiles;
    part.work = work;
    if (kmx_parallel(costs_part, &part) != KMX_OK) {
        return RF_PARALLEL;
    }
    const uint32_t heavy_count = (
        rm_pick(&work->scratch, workers, work->heavy, work->bands, frame->tiles)
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
        work->tile_item,
        frame->tiles
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
    items[RF_SHADE_ADDR_AT / 4] = frame->words[RF_W_SHADE];
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
    command.a3 = camera;                             /* camera_address */
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
            frame->ready[frame->area] = 1;
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
/* the masked layer's words: the program's, its own in place of the opaque layer's */
static uint32_t g_masked_words[RF_WORDS] __attribute__((section(".data"))) = {0};
/* a SUN command's opaque layer's words: the program's (its ranges and costs in the
 * camera region's sun areas) */
static uint32_t g_sun_words[RF_WORDS] __attribute__((section(".data"))) = {0};
static uint32_t g_suns __attribute__((section(".data"))) = 0;
/* the masked layer's SHADE's profile tensor (0: none) */
static uint32_t g_masked_shade_profile __attribute__((section(".data"))) = 0;
/* the result words before the masked layer's frame: its own parts are the difference */
static uint32_t g_before[RF_OUT_WORDS] __attribute__((section(".data"))) = {0};
/* the glass layer's (RF_G_*: 0 none), its words (as the masked layer's), SHADE's
 * profile, its own parts */
static uint32_t g_glass[RF_G_WORDS] __attribute__((section(".data"))) = {0};
static uint32_t g_glass_words[RF_WORDS] __attribute__((section(".data"))) = {0};
static uint32_t g_glass_shade_profile __attribute__((section(".data"))) = 0;
static uint32_t g_glass_parts[RF_OUT_WORDS] __attribute__((section(".data"))) = {0};

/* Fills g_masked_words: 0, or the code to report. */
static uint32_t masked_words(const uint32_t* words) {
    if (words[RF_W_M_FRAMES] > RF_FRAME_MAX) {
        return RF_BAD_ARGS;
    }
    for (uint32_t i = 0; i < RF_WORDS; i++) {
        g_masked_words[i] = words[i];
    }
    g_masked_words[RF_W_MESH] = words[RF_W_M_MESH];
    g_masked_words[RF_W_BLOCKS] = words[RF_W_M_BLOCKS];
    g_masked_words[RF_W_LISTS] = words[RF_W_M_LISTS];
    g_masked_words[RF_W_LIST_SLOTS] = words[RF_W_M_LIST_SLOTS];
    g_masked_words[RF_W_DIR] = words[RF_W_M_DIR];
    g_masked_words[RF_W_SHADE] = words[RF_W_M_SHADE];
    g_masked_words[RF_W_FRAMES] = words[RF_W_M_FRAMES];
    for (uint32_t frame_slot = 0; frame_slot < RF_FRAME_MAX; frame_slot++) {
        g_masked_words[RF_W_FRAME + frame_slot] = words[RF_W_M_FRAME + frame_slot];
    }
    /* its own profile tensors, when the host gave them */
    kmx_invalidate_at(RF_R_CAMERA, RF_PROFILES_AT, 64);
    const volatile uint32_t* profiles = (
        (const volatile uint32_t*)g_regions[RF_R_CAMERA] + RF_PROFILES_AT / 4
    );
    if (profiles[RF_P_FILL]) {
        g_masked_words[RF_W_PROF] = profiles[RF_P_FILL];
    }
    if (profiles[RF_P_PASSES]) {
        g_masked_words[RF_W_SPROF] = profiles[RF_P_PASSES];
    }
    g_masked_shade_profile = profiles[RF_P_SHADE];
    return 0;
}

/* Fills g_glass and g_glass_words from area 0 (frame.h RF_GLASS_WORDS_AT): 0, or the
 * code to report. */
static uint32_t glass_words(const uint32_t* words) {
    _Static_assert(4 * RF_G_WORDS <= 128, "the glass words: two lines");
    kmx_invalidate_at(
        RF_R_CAMERA, RF_PROFILES_AT, RF_GLASS_WORDS_AT - RF_PROFILES_AT + 128
    );
    const volatile uint32_t* profiles = (
        (const volatile uint32_t*)g_regions[RF_R_CAMERA] + RF_PROFILES_AT / 4
    );
    const volatile uint32_t* glass = (
        (const volatile uint32_t*)g_regions[RF_R_CAMERA] + RF_GLASS_WORDS_AT / 4
    );
    for (uint32_t i = 0; i < RF_G_WORDS; i++) {
        g_glass[i] = glass[i];
    }
    if (g_glass[RF_G_MESH] == 0) {
        return 0;
    }
    if (g_glass[RF_G_FRAMES] > RF_FRAME_MAX) {
        g_glass[RF_G_MESH] = 0;
        return RF_BAD_ARGS;
    }
    for (uint32_t i = 0; i < RF_WORDS; i++) {
        g_glass_words[i] = words[i];
    }
    g_glass_words[RF_W_MESH] = g_glass[RF_G_MESH];
    g_glass_words[RF_W_BLOCKS] = g_glass[RF_G_BLOCKS];
    g_glass_words[RF_W_LISTS] = g_glass[RF_G_LISTS];
    g_glass_words[RF_W_LIST_SLOTS] = g_glass[RF_G_LIST_SLOTS];
    g_glass_words[RF_W_DIR] = g_glass[RF_G_DIR];
    g_glass_words[RF_W_SHADE] = g_glass[RF_G_SHADE];
    g_glass_words[RF_W_FRAMES] = g_glass[RF_G_FRAMES];
    for (uint32_t frame_slot = 0; frame_slot < RF_FRAME_MAX; frame_slot++) {
        g_glass_words[RF_W_FRAME + frame_slot] = g_glass[RF_G_FRAME + frame_slot];
    }
    if (profiles[RF_P_GLASS_FILL]) {
        g_glass_words[RF_W_PROF] = profiles[RF_P_GLASS_FILL];
    }
    if (profiles[RF_P_GLASS_PASSES]) {
        g_glass_words[RF_W_SPROF] = profiles[RF_P_GLASS_PASSES];
    }
    g_glass_shade_profile = profiles[RF_P_GLASS_SHADE];
    return 0;
}

/* Fills g_sun_words. */
static void sun_words(const uint32_t* words) {
    for (uint32_t i = 0; i < RF_WORDS; i++) {
        g_sun_words[i] = words[i];
    }
}
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
    for (uint32_t area = 0; area < RF_AREAS; area++) {
        frame->ranged[area] = frame->ready[area] = 0;
    }
    frame->area = 0;
    const uint32_t need[RF_REGIONS] = {
        frame->words[RF_W_BALANCE] ? RF_CAMERA_BYTES : 4 * 16,
        RF_HEAVY_BYTES,
        RF_ITEMS_REGION_BYTES,
        sizeof(work_t),
        RF_SUMS_BYTES,
        RF_TIMES_BYTES,
        RF_SHOW_TABLE_BYTES,
        RF_PUSH_BYTES,
        RF_SUN_BYTES,
        RF_TIMES_BYTES,
    };
    void** regions = g_regions;
    for (int i = 0; i < (int)RF_REGIONS; i++) {
        uint64_t bytes;
        regions[i] = kmx_region(i, &bytes);
        if (regions[i] == 0 || bytes < need[i] || arguments->region[i].offset >> 32) {
            return finish(RF_BAD_ARGS, frame);
        }
        frame->address[i] = (uint32_t)arguments->region[i].offset;
        if (i == (int)RF_R_CAMERA) {
            frame->camera_bytes = bytes;
        }
    }
    frame->camera = frame->address[RF_R_CAMERA];
    if (
        cores == 0 || cores > workers || workers > RM_MAX_CORES ||
        frame->words[RF_W_FRAMES] > RF_FRAME_MAX
    ) {
        return finish(RF_BAD_ARGS, frame);
    }
    if (eye_grid(frame, regions)) {
        return finish(RF_BAD_GRID, frame);
    }
    work_t* work = (work_t*)regions[RF_R_WORK];
    /* lines an earlier image left dirty would be written over what the cores write */
    kmx_flush(RF_R_SUMS);
    kmx_flush(RF_R_CAMERA);
    /* SHOW's thresholds as the host wrote them at the load, not an earlier image's;
     * likewise the sun region (the thread never reads it, the cores do) */
    kmx_invalidate(RF_R_SHOW);
    kmx_invalidate(RF_R_SUN);
    const uint32_t frames = frame->words[RF_W_REPEAT];
#ifdef RF_SERVE
    if (frames == 0) {
        g_set_up = frame->words[RF_W_DISPLAY] ? 0u : RF_BAD_ARGS;
        if (g_set_up == 0 && frame->words[RF_W_M_MESH]) {
            g_set_up = masked_words(frame->words);
        }
        if (g_set_up == 0 && frame->words[RF_W_SHADE]) {
            g_set_up = glass_words(frame->words);
        }
        sun_words(frame->words);
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
/* SHOW: the frame's colour into the display (kernels/show.h); `raw`: its 1/z into the
 * shadow map of `across` quadrants a side as quadrant j n + i. 0, or the code. */
static uint32_t show(frame_t* frame, uint32_t raw, uint32_t across, uint32_t quadrant) {
    km_command_t command;
    new_command(&command, RF_HANDLER_SHOW, frame);
    command.a0 = frame->words[RF_W_DIR]; /* directory_address */
    /* display_address, profile_address (raw: the shadow map's pages, none) */
    command.a1 = raw ? frame->words[RF_W_SUN_PAGES] : frame->words[RF_W_DISPLAY];
    command.a2 = raw ? 0 : frame->words[RF_W_SHOWPROF];
    /* display_pages: a page a tile; raw: the map's, across x across frames */
    command.a3 = raw ? across * across * RF_DISPLAY_PAGES : RF_DISPLAY_PAGES;
    command.a4 = raw ? RF_SHOW_RAW : 0;
    /* a8 becomes the block's word (kmx_block): SHOW takes no grid (show.h) */
    if (raw) { /* the quadrant's first page, the map's pages a row (show.h kShowRaw) */
        const uint32_t row_pages = across * RF_SUN_TILES_ACROSS;
        command.a5 = (
            quadrant / across * RF_SUN_TILES_ACROSS * row_pages +
            quadrant % across * RF_SUN_TILES_ACROSS
        );
        command.a6 = row_pages;
    }
    if (kmx_block(&command, RF_R_SHOW, 0, RF_SHOW_TABLE_BYTES) != KMX_OK) {
        return RF_BAD_ARGS;
    }
    return run(&command);
}

/* SHADE (kernels/shade.h): `opaque`, the masked or glass layer's into it; lit
 * `surface`; `mode` a9, `profile` its tensor. 0, or the code to report. */
static uint32_t shade(
    frame_t* frame,
    uint32_t directory,
    uint32_t lists,
    uint32_t opaque,
    uint32_t surface,
    uint32_t mode,
    uint32_t profile
) {
    km_command_t command;
    new_command(&command, RF_HANDLER_SHADE, frame);
    command.a0 = directory;                      /* directory_address */
    command.a1 = lists;                          /* lists_address */
    command.a2 = frame->words[RF_W_TEXTURES];    /* textures_address */
    command.a3 = frame->words[RF_W_SHADE_TABLE]; /* table_address */
    command.a4 = profile;                        /* profile_address */
    command.a6 = opaque;                         /* opaque_directory_address */
    command.a9 = mode;
    if (frame->words[RF_W_SURFACE]) {
        command.a5 = frame->camera; /* camera_address (a8 its bank) */
        command.a7 = surface;       /* surface_address */
    }
    return run(&command);
}

/* LIGHT: each tile's lit colour from its surface planes (kernels/shade.h); `mode` its
 * a9 (RF_LIGHT_*): the glass layer's over the linear colour. 0, or the code. */
static uint32_t light(frame_t* frame, uint32_t mode) {
    km_command_t command;
    new_command(&command, RF_HANDLER_LIGHT, frame);
    command.a0 = frame->words[RF_W_DIR]; /* directory_address */
    /* the table: the sun region's, which the host rewrites when the sun moves */
    command.a1 = frame->address[RF_R_SUN] + RF_SUN_TABLE_AT;
    command.a5 = frame->camera;              /* camera_address (a8 its bank) */
    command.a7 = frame->words[RF_W_SURFACE]; /* surface_address */
    command.a9 = mode;
    if (mode == RF_LIGHT_GLASS) {
        command.a2 = frame->words[RF_W_SHADE_TABLE]; /* the materials' tau */
        command.a6 = g_glass[RF_G_DIR];              /* the glass layer's frame */
        command.a7 = g_glass[RF_G_SURFACE];          /* its surface, LIGHT's profile */
    }
    return run(&command);
}

/* Another layer (`words`, `area`): COUNT to FILL, its own result words into
 * `parts`, the frame's summed, the opaque's middle pass kept. 0, or the code. */
static uint32_t layer_frame(
    frame_t* frame, work_t* work, uint32_t* words, uint32_t area, uint32_t* parts
) {
    /* scalars: a local array's initialiser may call memcpy, which does not link */
    const uint32_t slots = frame->result[RF_O_SLOTS];
    const uint32_t items = frame->result[RF_O_ITEMS];
    const uint32_t marks = frame->result[RF_O_MARKS];
    const uint32_t heavy = frame->result[RF_O_HEAVY];
    const uint32_t* own = frame->words;
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        parts[i] = frame->result[i];
    }
    frame->words = words;
    frame->area = area;
    const uint32_t code = frame_once(frame, work, g_regions);
    frame->area = 0;
    frame->words = own;
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        parts[i] = frame->result[i] - parts[i];
    }
    parts[RF_O_SLOTS] = frame->result[RF_O_SLOTS];
    parts[RF_O_ITEMS] = frame->result[RF_O_ITEMS];
    parts[RF_O_HEAVY] = frame->result[RF_O_HEAVY];
    frame->result[RF_O_SLOTS] = slots;
    frame->result[RF_O_ITEMS] = items;
    frame->result[RF_O_MARKS] = marks;
    frame->result[RF_O_HEAVY] = heavy;
    return code;
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

/* FRAME: each layer's frame, SHADE when shaded (each layer's), LIGHT when lit, SHOW,
 * (a1) PUSH; ticks to a times row (parts summed over layers, middle pass: opaque). */
static uint32_t serve_frame(const km_command_t* command) {
    frame_t* frame = &g_frame;
    const uint32_t* words = frame->words;
    const uint64_t started = kmx_time();
    kmx_invalidate_at(RF_R_CAMERA, 0, 64);
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        frame->result[i] = 0;
    }
    work_t* work = (work_t*)g_regions[RF_R_WORK];
    uint32_t code = eye_grid(frame, g_regions); /* a size may change between frames */
    if (code == 0) {
        code = frame_once(frame, work, g_regions);
    }
    const uint64_t opaque_ended = kmx_time();
    uint32_t masked_ticks = 0, glass_ticks = 0;
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        g_before[i] = g_glass_parts[i] = 0;
    }
    if (code == 0 && words[RF_W_M_MESH]) {
        code = layer_frame(frame, work, g_masked_words, RF_AREA_MASKED, g_before);
        masked_ticks = (uint32_t)(kmx_time() - opaque_ended);
    }
    const uint32_t glass = g_glass[RF_G_MESH] != 0;
    if (code == 0 && glass) {
        const uint64_t glass_started = kmx_time();
        code = layer_frame(frame, work, g_glass_words, RF_AREA_GLASS, g_glass_parts);
        glass_ticks = (uint32_t)(kmx_time() - glass_started);
    }
    const uint32_t lit = words[RF_W_SURFACE];
    uint32_t shade_ticks = 0, shade_masked_ticks = 0, shade_glass_ticks = 0;
    if (code == 0 && words[RF_W_SHADE]) {
        const uint64_t shade_started = kmx_time();
        code = (
            shade(
                frame, words[RF_W_DIR], words[RF_W_LISTS], 0, lit, 0, words[RF_W_SHADEPROF]
            )
        );
        if (code == 0 && words[RF_W_M_MESH]) {
            const uint64_t masked_started = kmx_time();
            /* unlit before a glass layer: the merged 1/z too (lit SHADE gives it) */
            code = (
                shade(
                    frame,
                    words[RF_W_M_DIR],
                    words[RF_W_M_LISTS],
                    words[RF_W_DIR],
                    lit,
                    glass ? RF_SHADE_MERGED_DEPTH : 0,
                    g_masked_shade_profile
                )
            );
            shade_masked_ticks = (uint32_t)(kmx_time() - masked_started);
        }
        if (code == 0 && glass) {
            /* lit: the glass layer's own surface; unlit: blended into the frame */
            const uint64_t glass_started = kmx_time();
            code = (
                shade(
                    frame,
                    g_glass[RF_G_DIR],
                    g_glass[RF_G_LISTS],
                    lit ? 0 : words[RF_W_DIR],
                    g_glass[RF_G_SURFACE],
                    lit ? 0 : RF_SHADE_GLASS,
                    g_glass_shade_profile
                )
            );
            shade_glass_ticks = (uint32_t)(kmx_time() - glass_started);
        }
        shade_ticks = (uint32_t)(kmx_time() - shade_started);
    }
    uint32_t light_ticks = 0, light_glass_ticks = 0;
    if (code == 0 && lit) {
        const uint64_t light_started = kmx_time();
        code = light(frame, glass ? RF_LIGHT_LINEAR : 0);
        if (code == 0 && glass) {
            const uint64_t glass_started = kmx_time();
            code = light(frame, RF_LIGHT_GLASS);
            light_glass_ticks = (uint32_t)(kmx_time() - glass_started);
        }
        light_ticks = (uint32_t)(kmx_time() - light_started);
    }
    uint32_t show_ticks = 0;
    if (code == 0) {
        const uint64_t show_started = kmx_time();
        code = show(frame, 0, 1, 0);
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
    times[RF_T_SHADE] = shade_ticks;
    times[RF_T_OPAQUE] = (uint32_t)(opaque_ended - started);
    times[RF_T_MASKED] = masked_ticks;
    times[RF_T_SHADE_MASKED] = shade_masked_ticks;
    times[RF_T_LIGHT] = light_ticks;
    times[RF_T_GLASS] = glass_ticks;
    times[RF_T_SHADE_GLASS] = shade_glass_ticks;
    times[RF_T_LIGHT_GLASS] = light_glass_ticks;
    times[RF_T_PUSH] = push_ticks;
    times[RF_T_NUMBER] = g_served;
    times[RF_T_CODE] = code;
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        times[RF_T_M_PARTS + i] = g_before[i];
        times[RF_T_G_PARTS + i] = g_glass_parts[i];
    }
    times[RF_T_FRAME] = (uint32_t)(kmx_time() - started);
    kmx_flush_at(RF_R_TIMES, 4u * row, 4u * RF_TIMES_ROW_WORDS);
    g_served++;
    return code;
}

/* SUN, per quadrant of the shadow map: the layers' frames, the masked merge, SHOW's raw
 * copy into its pages (README "The sun"); a times row, its parts summed. */
static uint32_t serve_sun(void) {
    frame_t* frame = &g_frame;
    const uint32_t* words = frame->words;
    if (!words[RF_W_SURFACE] || !words[RF_W_SUN_PAGES]) {
        return RF_NOT_LIT;
    }
    const uint64_t started = kmx_time();
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        frame->result[i] = 0;
    }
    kmx_invalidate_at(RF_R_SUN, RF_SUN_ACROSS_AT, 64);
    const uint32_t across = (
        ((const volatile uint32_t*)g_regions[RF_R_SUN])[RF_SUN_ACROSS_AT / 4]
    );
    uint32_t code = across == 0 || across > RF_SUN_ACROSS_MAX ? RF_BAD_ARGS : 0;
    work_t* work = (work_t*)g_regions[RF_R_WORK];
    uint32_t opaque_ticks = 0, masked_ticks = 0, merge_ticks = 0, copy_ticks = 0;
    uint32_t quadrant = 0;
    /* the sun's own grid for its passes, as its camera; the eye's again after */
    const uint32_t eye = frame->grid, eye_tiles = frame->tiles;
    frame->grid = RF_SUN_GRID;
    frame->tiles = grid_tiles(RF_SUN_GRID);
    for (; code == 0 && quadrant < across * across; quadrant++) {
        uint64_t mark = kmx_time();
        frame->camera = frame->address[RF_R_SUN] + RF_SUN_CAMERA_BYTES * quadrant;
        frame->words = g_sun_words;
        frame->area = RF_AREA_SUN + RF_AREAS_A_QUADRANT * quadrant;
        code = frame_once(frame, work, g_regions);
        opaque_ticks += elapsed(&mark);
        if (code == 0 && words[RF_W_M_MESH]) {
            frame->words = g_masked_words;
            frame->area += RF_AREA_MASKED;
            code = frame_once(frame, work, g_regions);
        }
        frame->area = 0;
        frame->words = words;
        masked_ticks += elapsed(&mark);
        if (code == 0 && words[RF_W_M_MESH]) {
            code = (
                shade(
                    frame,
                    words[RF_W_M_DIR],
                    words[RF_W_M_LISTS],
                    words[RF_W_DIR],
                    words[RF_W_SURFACE],
                    0,
                    g_masked_shade_profile
                )
            );
        }
        merge_ticks += elapsed(&mark);
        if (code == 0) {
            code = show(frame, 1, across, quadrant);
        }
        copy_ticks += elapsed(&mark);
    }
    frame->area = 0;
    frame->words = words;
    frame->camera = frame->address[RF_R_CAMERA];
    frame->grid = eye;
    frame->tiles = eye_tiles;
    const uint32_t row = (g_suns % RF_TIMES_ROWS) * RF_TIMES_ROW_WORDS;
    volatile uint32_t* times = (volatile uint32_t*)g_regions[RF_R_SUN_TIMES] + row;
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        times[i] = frame->result[i];
    }
    times[RF_S_OPAQUE] = opaque_ticks;
    times[RF_S_MASKED] = masked_ticks;
    times[RF_S_MERGE] = merge_ticks;
    times[RF_S_COPY] = copy_ticks;
    times[RF_S_QUADRANTS] = quadrant;
    times[RF_S_NUMBER] = g_suns;
    times[RF_S_CODE] = code;
    times[RF_S_FRAMES] = g_served;
    times[RF_S_TOTAL] = (uint32_t)(kmx_time() - started);
    kmx_flush_at(RF_R_SUN_TIMES, 4u * row, 4u * RF_TIMES_ROW_WORDS);
    g_suns++;
    return code;
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerFrame:
            return g_set_up ? g_set_up : serve_frame(command);
        case kHandlerSun:
            return g_set_up ? g_set_up : serve_sun();
    }
    return 0;
}
#endif
