/* The frame on the work cluster: COUNT, SUM, HIST, HSUM, the middle pass (middle.c),
 * PLACE, SETUP, FILL, each command submitted and waited for. Words, regions, codes:
 * frame.h */
#include <stdint.h>

#include "frame.h"
#include "kmx.h"
#include "middle.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per command, longer than its deadline */

typedef struct {
    rm_scratch s;
    rm_item items[RM_MAX_ITEMS];
    uint32_t tile_item[RM_TILES + 1];
    uint32_t heavy[RM_MAX_HEAVY], bands[RM_MAX_HEAVY];
    uint32_t ranges[RF_MAX_CORES + 1]; /* COUNT's and SETUP's blocks of each core */
    uint32_t costs[RF_MAX_CORES], cuts[RF_MAX_CORES + 1];
} work_t;

typedef struct {
    const uint32_t* w;
    uint32_t bank1; /* 1 + the regions' bank */
    uint32_t addr[RF_REGIONS];
    uint32_t out[RF_OUT_WORDS];
    uint32_t ranged; /* frames run with ranges: the costs of the last are in */
} frame_t;

/* The middle pass's parts on the four cores: each a quarter of the tiles or items. */
typedef struct {
    const uint32_t* counters;
    uint32_t cores, workers, ni;
    work_t* wk;
} part_t;

static void costs_part(int core, void* arg) {
    const part_t* p = (const part_t*)arg;
    const uint32_t h = (uint32_t)core;
    rm_costs(
        p->counters, p->cores, &p->wk->s, RM_TILES * h / 4, RM_TILES * (h + 1) / 4
    );
}

static void sort_part(int core, void* arg) {
    const part_t* p = (const part_t*)arg;
    const uint32_t h = (uint32_t)core;
    rm_sort(&p->wk->s, p->ni * h / 4, p->ni * (h + 1) / 4);
}

/* the quarters in halves on cores 0 and 1 */
static void merge_part(int core, void* arg) {
    const part_t* p = (const part_t*)arg;
    const uint32_t h = (uint32_t)core;
    if (h < 2) {
        rm_merge(
            &p->wk->s,
            p->ni * 2 * h / 4,
            p->ni * (2 * h + 1) / 4,
            p->ni * (2 * h + 2) / 4
        );
    }
}

/* Fills r field by field (a struct assignment may call memset, which does not link). */
static void command(km_command_t* r, uint32_t handler, const frame_t* f) {
    r->seq = r->land1 = r->land2 = r->land3 = 0;
    r->hdr = handler;
    r->group = f->w[RF_W_WORKERS] << 16;
    r->a0 = r->a1 = r->a2 = r->a3 = r->a4 = r->a6 = r->a7 = 0;
    r->a9 = f->w[RF_W_SPROF];
    r->a5 = f->w[RF_W_CORES];
    r->a8 = f->bank1;
}

/* Submits r and waits for its event: 0, or the code to report. */
static uint32_t run(const km_command_t* r) {
    const uint64_t until = kmx_time() + WAIT_TICKS;
    uint32_t handle;
    int e = kmx_submit(r, &handle);
    while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        e = kmx_submit(r, &handle);
    }
    if (e != KMX_OK) {
        return RF_RING;
    }
    kmx_event ev;
    if (kmx_wait(handle, kmx_time() + WAIT_TICKS, &ev) != KMX_OK) {
        return RF_RING;
    }
    return ev.kind == KM_EV_DONE ? 0u : RF_COMMAND;
}

static int finish(uint32_t code, frame_t* f) {
    kmx_result(code, f->out, RF_OUT_WORDS);
    return 0;
}

static uint32_t elapsed(uint64_t* t) {
    const uint64_t now = kmx_time();
    const uint32_t d = (uint32_t)(now - *t);
    *t = now;
    return d;
}

/* Each core's blocks for COUNT and SETUP, into the camera's region: equal partitions,
 * then equal parts of the last frame's ticks (spread evenly over each core's blocks).
 */
static void set_ranges(frame_t* f, work_t* wk, void* const* reg) {
    const uint32_t cores = f->w[RF_W_CORES], blocks = f->w[RF_W_BLOCKS];
    uint32_t* r = wk->ranges;
    volatile uint32_t* cam = (volatile uint32_t*)reg[RF_R_CAMERA];
    uint64_t total = 0;
    if (f->ranged) {
        kmx_invalidate_at(RF_R_CAMERA, RF_COSTS_AT, 128u * cores);
        for (uint32_t c = 0; c < cores; c++) {
            const uint32_t at = (RF_COSTS_AT + 128u * c) / 4;
            wk->costs[c] = cam[at] + cam[at + 16];
            total += wk->costs[c];
        }
    }
    if (total == 0) {
        for (uint32_t c = 0; c <= cores; c++) {
            r[c] = (uint32_t)((uint64_t)c * blocks / cores);
        }
    } else {
        uint32_t* cut = wk->cuts;
        uint32_t c = 0;
        uint64_t before = 0; /* ticks of the cores before c */
        cut[0] = 0;
        for (uint32_t k = 1; k < cores; k++) {
            const uint64_t want = total * k / cores;
            while (c < cores && before + wk->costs[c] <= want) {
                before += wk->costs[c];
                c++;
            }
            uint32_t b = blocks;
            if (c < cores) {
                const uint32_t n = r[c + 1] - r[c];
                b = r[c] + (uint32_t)((want - before) * n / wk->costs[c]);
            }
            cut[k] = b > cut[k - 1] ? b : cut[k - 1];
        }
        cut[cores] = blocks;
        for (uint32_t k = 0; k <= cores; k++) {
            r[k] = cut[k];
        }
    }
    for (uint32_t c = 0; c < cores; c++) {
        cam[(RF_RANGES_AT + 64u * c) / 4] = r[c];
        cam[(RF_RANGES_AT + 64u * c) / 4 + 1] = r[c + 1];
    }
    kmx_flush_at(RF_R_CAMERA, RF_RANGES_AT, 64u * cores);
    f->ranged++;
}

/* One frame: COUNT to FILL (or as far as RF_W_STEPS says); adds each part's ticks to
 * the result words. 0, or the code to report. */
static uint32_t frame_once(frame_t* f, work_t* wk, void* const* reg) {
    const uint32_t cores = f->w[RF_W_CORES], workers = f->w[RF_W_WORKERS];
    uint64_t t = kmx_time();

    /* COUNT */
    km_command_t r;
    kmx_progress(1);
    /* fewer blocks than cores: a block each at most, nothing to balance */
    const uint32_t ranges =
        (f->w[RF_W_BALANCE] && cores <= RF_MAX_CORES && f->w[RF_W_BLOCKS] >= cores
             ? RF_RANGES
             : 0);
    if (ranges) {
        set_ranges(f, wk, reg);
    }
    command(&r, RF_HANDLER_COUNT, f);
    r.a0 = f->w[RF_W_MESH];
    r.a1 = f->w[RF_W_COUNTERS];
    r.a2 = f->w[RF_W_BLOCKS] | ranges;
    r.a3 = f->addr[RF_R_CAMERA];
    uint32_t code = run(&r);
    f->out[RF_O_COUNT] += elapsed(&t);
    if (code || f->w[RF_W_STEPS] < 2) {
        return code;
    }

    /* SUM: the counters' totals, and each core's entries before it per tile */
    kmx_progress(2);
    command(&r, RF_HANDLER_SUM, f);
    r.a1 = f->w[RF_W_COUNTERS];
    r.a2 = f->w[RF_W_SLOTS];
    r.a3 = f->addr[RF_R_SUMS];
    r.a6 = cores;
    code = run(&r);
    f->out[RF_O_SUM] += elapsed(&t);
    if (code) {
        return code;
    }

    /* the tiles to cut, and their histograms' sums */
    kmx_invalidate(RF_R_SUMS);
    f->out[RF_O_TAKE] += elapsed(&t);
    const uint32_t* sums = (const uint32_t*)reg[RF_R_SUMS];
    part_t p;
    p.counters = sums; /* the totals, as one core's counters */
    p.cores = 1;
    p.workers = workers;
    p.wk = wk;
    if (kmx_par(costs_part, &p) != KMX_OK) {
        return RF_PAR;
    }
    const uint32_t nh = rm_pick(&wk->s, workers, wk->heavy, wk->bands);
    f->out[RF_O_HEAVY] = nh;
    if (nh > RF_HIST_MAX) {
        return RF_TOO_MANY_HEAVY;
    }
    f->out[RF_O_HEAVY_US] += elapsed(&t);
    if (nh) {
        volatile uint32_t* hv = (volatile uint32_t*)reg[RF_R_HEAVY];
        hv[0] = nh;
        for (uint32_t h = 0; h < nh; h++) {
            hv[1 + h] = wk->heavy[h];
        }
        kmx_flush(RF_R_HEAVY);
        command(&r, RF_HANDLER_HIST, f);
        r.a0 = f->addr[RF_R_HEAVY];
        r.a1 = f->w[RF_W_HIST];
        r.a4 = RF_HIST_BYTES;
        kmx_progress(3);
        code = run(&r);
        f->out[RF_O_HIST] += elapsed(&t);
        if (code) {
            return code;
        }
        command(&r, RF_HANDLER_HSUM, f);
        r.a0 = nh;
        r.a1 = f->w[RF_W_HIST];
        r.a2 = f->w[RF_W_HPREFIX];
        r.a3 = f->addr[RF_R_SUMS];
        r.a4 = RF_HIST_BYTES;
        r.a6 = cores;
        code = run(&r);
        f->out[RF_O_HSUM] += elapsed(&t);
        if (code) {
            return code;
        }
        kmx_invalidate(RF_R_SUMS);
        const uint32_t* hs = sums + RF_SUMS_HEAVY_AT / 4;
        for (uint32_t h = 0; h < nh; h++) {
            for (uint32_t g = 0; g < 2 * RM_GROUPS; g++) {
                wk->s.hist[h][g] = hs[2 * RM_GROUPS * h + g];
            }
        }
        f->out[RF_O_TAKE] += elapsed(&t);
    }

    /* middle pass: items, cores, lists' layout in words for PLACE, base slots */
    kmx_progress(4);
    uint32_t ni = 0;
    if (!rm_items(wk->heavy, wk->bands, nh, &wk->s, wk->items, &ni, wk->tile_item)) {
        return RF_NO_LAYOUT;
    }
    f->out[RF_O_CUT] += elapsed(&t);
    p.ni = ni;
    if (kmx_par(sort_part, &p) != KMX_OK || kmx_par(merge_part, &p) != KMX_OK) {
        return RF_PAR;
    }
    rm_merge(&wk->s, 0, ni / 2, ni);
    f->out[RF_O_SORT] += elapsed(&t);
    if (!rm_owners(workers, &wk->s, ni)) {
        return RF_NO_LAYOUT;
    }
    f->out[RF_O_ASSIGN] += elapsed(&t);
    uint32_t* items = (uint32_t*)reg[RF_R_ITEMS];
    const uint32_t slots = (
        rm_compact(
            workers,
            &wk->s,
            wk->items,
            ni,
            items + RF_ITEM_WORDS_AT / 4,
            items + RF_CORE_WORDS_AT / 4
        )
    );
    uint32_t nm = workers;
    for (uint32_t i = 0; i < ni; i++) {
        nm += wk->items[i].total != 0;
    }
    f->out[RF_O_SLOTS] = slots;
    f->out[RF_O_ITEMS] = ni;
    f->out[RF_O_MARKS] = nm;
    if (slots == 0 || ni > RF_SLOTS_BYTES / 4) {
        return RF_NO_LAYOUT;
    }
    if (slots > f->w[RF_W_LIST_SLOTS] || slots >= 1u << RM_BASE_SLOT_BITS) {
        return RF_LISTS_SMALL;
    }
    rm_base(&wk->s, wk->items, ni, items + RF_BASE_AT / 4);
    f->out[RF_O_COMPACT] += elapsed(&t);
    for (uint32_t i = 0; i < RM_TILES; i++) {
        const uint32_t first = wk->tile_item[i];
        items[i] = first | (wk->tile_item[i + 1] - first) << 16;
    }
    uint16_t* rows = (uint16_t*)(items + RM_TILES);
    for (uint32_t i = 0; i < ni; i++) {
        rows[i] = (uint16_t)(wk->items[i].row0 | wk->items[i].row1 << 8);
    }
    items[RF_HPREFIX_ADDR_AT / 4] = f->w[RF_W_HPREFIX];
    uint32_t* fw = items + RF_FRAME_WORDS_AT / 4;
    fw[0] = f->w[RF_W_FRAME_PAGES];
    for (uint32_t k = 0; k < RF_FRAME_MAX; k++) {
        fw[1 + k] = k < f->w[RF_W_FRAMES] ? f->w[RF_W_FRAME + k] : 0;
    }
    f->out[RF_O_PAGES] += elapsed(&t);
    kmx_flush(RF_R_ITEMS);
    f->out[RF_O_GIVE] += elapsed(&t);
    f->out[RF_O_MIDDLE] =
        (f->out[RF_O_TAKE] + f->out[RF_O_HEAVY_US] + f->out[RF_O_CUT] +
         f->out[RF_O_SORT] + f->out[RF_O_ASSIGN] + f->out[RF_O_COMPACT] +
         f->out[RF_O_PAGES] + f->out[RF_O_GIVE]);
    if (f->w[RF_W_STEPS] < 3) {
        return 0;
    }

    /* PLACE, then SETUP */
    kmx_progress(5);
    command(&r, RF_HANDLER_PLACE, f);
    r.a0 = f->addr[RF_R_ITEMS];
    r.a1 = f->w[RF_W_DIR];
    r.a6 = workers;
    r.a7 = f->w[RF_W_LISTS];
    code = run(&r);
    f->out[RF_O_PLACE] += elapsed(&t);
    if (code) {
        return code;
    }
    kmx_progress(6);
    command(&r, RF_HANDLER_SETUP, f);
    r.a0 = f->w[RF_W_MESH];
    r.a1 = f->w[RF_W_LISTS];
    r.a2 = f->w[RF_W_BLOCKS] | ranges;
    r.a3 = f->addr[RF_R_CAMERA];
    r.a4 = f->w[RF_W_SLOTS];
    r.a6 = slots | RF_SETUP_PREFIX;
    r.a7 = f->addr[RF_R_ITEMS];
    code = run(&r);
    f->out[RF_O_SETUP] += elapsed(&t);
    if (code || f->w[RF_W_STEPS] < 4) {
        return code;
    }

    /* FILL */
    kmx_progress(7);
    command(&r, RF_HANDLER_FILL, f);
    r.a0 = f->w[RF_W_DIR];
    r.a1 = f->w[RF_W_LISTS];
    r.a3 = RF_PAGES_PER_ITEM;
    r.a4 = f->w[RF_W_PROF];
    code = run(&r);
    f->out[RF_O_FILL] += elapsed(&t);
    kmx_progress(8);
    return code;
}

int kmx_main(int core) {
    (void)core;
    const kmx_args* a = kmx_args_get();
    frame_t f;
    for (uint32_t i = 0; i < RF_OUT_WORDS; i++) {
        f.out[i] = 0;
    }
    if (a->nwords < RF_WORDS || a->nregions < RF_REGIONS) {
        return finish(RF_BAD_ARGS, &f);
    }
    f.w = a->words;
    f.bank1 = 1 + f.w[RF_W_BANK];
    const uint32_t cores = f.w[RF_W_CORES], workers = f.w[RF_W_WORKERS];
    f.ranged = 0;
    const uint32_t need[RF_REGIONS] = {
        f.w[RF_W_BALANCE] ? RF_CAMERA_BYTES : 4 * 16,
        RF_HEAVY_BYTES,
        RF_ITEMS_REGION_BYTES,
        sizeof(work_t),
        RF_SUMS_BYTES,
    };
    void* reg[RF_REGIONS];
    for (int i = 0; i < (int)RF_REGIONS; i++) {
        uint64_t bytes;
        reg[i] = kmx_region(i, &bytes);
        if (reg[i] == 0 || bytes < need[i] || a->region[i].offset >> 32) {
            return finish(RF_BAD_ARGS, &f);
        }
        f.addr[i] = (uint32_t)a->region[i].offset;
    }
    if (cores == 0 || cores > workers || workers > RM_MAX_CORES ||
        f.w[RF_W_FRAMES] > RF_FRAME_MAX) {
        return finish(RF_BAD_ARGS, &f);
    }
    work_t* wk = (work_t*)reg[RF_R_WORK];
    /* lines an earlier image left dirty would be written over what the cores write */
    kmx_flush(RF_R_SUMS);
    kmx_flush(RF_R_CAMERA);
    const uint32_t frames = f.w[RF_W_REPEAT] ? f.w[RF_W_REPEAT] : 1;
    for (uint32_t k = 0; k < frames; k++) {
        const uint32_t code = frame_once(&f, wk, reg);
        if (code) {
            return finish(code, &f);
        }
        f.out[RF_O_FRAMES] = k + 1;
    }
    return finish(0, &f);
}
