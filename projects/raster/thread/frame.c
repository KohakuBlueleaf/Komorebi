/* The frame on the work cluster: COUNT, HIST, the step (middle.c), PLACE, SETUP, FILL,
 * each record issued on the ring and waited for. Words, regions, codes: frame.h */
#include <stdint.h>

#include "frame.h"
#include "kmx.h"
#include "middle.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per record, longer than its deadline */

typedef struct {
    rm_scratch s;
    rm_item items[RM_MAX_ITEMS];
    uint32_t tile_item[RM_TILES + 1];
    rm_mark marks[RM_MAX_ITEMS + RM_MAX_CORES];
    uint32_t heavy[RM_MAX_HEAVY], bands[RM_MAX_HEAVY];
} work_t;

typedef struct {
    const uint32_t* w;
    uint32_t bank1; /* 1 + the regions' bank */
    uint32_t addr[RF_REGIONS];
    uint32_t out[RF_OUT_WORDS];
} frame_t;

/* The step's parts on the four harts: each a quarter of the tiles, cores or items. */
typedef struct {
    const uint32_t* counters;
    const uint16_t* hist;
    uint32_t cores, workers, nh, ni;
    work_t* wk;
    uint32_t* dir;
    uint32_t* next;
} part_t;

static void costs_part(int hart, void* arg) {
    const part_t* p = (const part_t*)arg;
    const uint32_t h = (uint32_t)hart;
    rm_costs(
        p->counters, p->cores, &p->wk->s, RM_TILES * h / 4, RM_TILES * (h + 1) / 4
    );
}

static void clear_part(int hart, void* arg) {
    const part_t* p = (const part_t*)arg;
    const uint32_t h = (uint32_t)hart;
    rm_clear(p->dir, p->workers * h / 4, p->workers * (h + 1) / 4);
}

static void next_part(int hart, void* arg) {
    const part_t* p = (const part_t*)arg;
    const uint32_t h = (uint32_t)hart;
    rm_next(
        p->counters,
        p->hist,
        p->cores,
        p->nh,
        &p->wk->s,
        p->wk->items,
        p->ni * h / 4,
        p->ni * (h + 1) / 4,
        p->next,
        RF_SLOTS_BYTES / 4
    );
}

/* Fills r field by field (a struct assignment may call memset, which does not link). */
static void record(km_record_t* r, uint32_t seg, const frame_t* f) {
    r->seq = r->land1 = r->land2 = r->land3 = 0;
    r->hdr = seg;
    r->team = f->w[RF_W_WORKERS] << 16;
    r->a0 = r->a1 = r->a2 = r->a3 = r->a4 = r->a6 = r->a7 = r->a9 = 0;
    r->a5 = f->w[RF_W_CORES];
    r->a8 = f->bank1;
}

/* Issues r and waits for its event: 0, or the code to report. */
static uint32_t run(const km_record_t* r) {
    const uint64_t until = kmx_time() + WAIT_TICKS;
    uint32_t ticket;
    int e = kmx_issue(r, &ticket);
    while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        e = kmx_issue(r, &ticket);
    }
    if (e != KMX_OK) {
        return RF_RING;
    }
    kmx_event ev;
    if (kmx_wait(ticket, kmx_time() + WAIT_TICKS, &ev) != KMX_OK) {
        return RF_RING;
    }
    return ev.kind == KM_EV_DONE ? 0u : RF_RECORD;
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

/* One frame: COUNT to FILL (or as far as RF_W_STEPS says); adds each part's ticks to
 * the result words. 0, or the code to report. */
static uint32_t frame_once(frame_t* f, work_t* wk, void* const* reg) {
    const uint32_t cores = f->w[RF_W_CORES], workers = f->w[RF_W_WORKERS];
    uint64_t t = kmx_time();

    /* COUNT */
    km_record_t r;
    kmx_progress(1);
    record(&r, RF_SEG_COUNT, f);
    r.a0 = f->w[RF_W_MESH];
    r.a1 = f->addr[RF_R_COUNTERS];
    r.a2 = f->w[RF_W_BLOCKS];
    r.a3 = f->addr[RF_R_CAMERA];
    uint32_t code = run(&r);
    f->out[RF_O_COUNT] += elapsed(&t);
    if (code || f->w[RF_W_STEPS] < 2) {
        return code;
    }

    /* the tiles to cut, and their histograms */
    kmx_progress(2);
    kmx_take(RF_R_COUNTERS);
    f->out[RF_O_TAKE] += elapsed(&t);
    part_t p;
    p.counters = (const uint32_t*)reg[RF_R_COUNTERS];
    p.hist = (const uint16_t*)reg[RF_R_HIST];
    p.cores = cores;
    p.workers = workers;
    p.wk = wk;
    p.dir = (uint32_t*)reg[RF_R_DIR];
    p.next = (uint32_t*)reg[RF_R_SLOTS];
    if (kmx_par(costs_part, &p) != KMX_OK) {
        return RF_PAR;
    }
    const uint32_t nh = rm_pick(&wk->s, workers, wk->heavy, wk->bands);
    p.nh = nh;
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
        kmx_give(RF_R_HEAVY);
        record(&r, RF_SEG_HIST, f);
        r.a0 = f->addr[RF_R_HEAVY];
        r.a1 = f->addr[RF_R_HIST];
        r.a4 = 64 * nh;
        kmx_progress(3);
        code = run(&r);
        if (code) {
            return code;
        }
        kmx_take(RF_R_HIST);
    }
    f->out[RF_O_HIST] += elapsed(&t);

    /* the step: items, cores, layout, next slots */
    kmx_progress(4);
    uint32_t ni = 0, nm = 0;
    uint32_t* dir = p.dir;
    if (kmx_par(clear_part, &p) != KMX_OK) {
        return RF_PAR;
    }
    f->out[RF_O_CLEAR] += elapsed(&t);
    const uint32_t slots = (
        rm_layout(
            p.hist,
            cores,
            workers,
            wk->heavy,
            wk->bands,
            nh,
            &wk->s,
            wk->items,
            &ni,
            wk->tile_item,
            dir,
            wk->marks,
            &nm
        )
    );
    p.ni = ni;
    f->out[RF_O_LAYOUT] += elapsed(&t);
    if (slots != 0 && ni <= RF_SLOTS_BYTES / 4 && kmx_par(next_part, &p) != KMX_OK) {
        return RF_PAR;
    }
    f->out[RF_O_NEXT] += elapsed(&t);
    f->out[RF_O_SLOTS] = slots;
    f->out[RF_O_ITEMS] = ni;
    f->out[RF_O_MARKS] = nm;
    if (slots == 0 || ni > RF_SLOTS_BYTES / 4) {
        return RF_NO_LAYOUT;
    }
    if (slots > f->w[RF_W_LIST_SLOTS]) {
        return RF_LISTS_SMALL;
    }
    uint32_t* items = (uint32_t*)reg[RF_R_ITEMS];
    for (uint32_t i = 0; i < RM_TILES; i++) {
        const uint32_t first = wk->tile_item[i];
        items[i] = first | (wk->tile_item[i + 1] - first) << 16;
    }
    uint16_t* rows = (uint16_t*)(items + RM_TILES);
    for (uint32_t i = 0; i < ni; i++) {
        rows[i] = (uint16_t)(wk->items[i].row0 | wk->items[i].row1 << 8);
    }
    for (uint32_t c = 0; c < workers; c++) {
        uint32_t* d = dir + c * RM_DIR_WORDS;
        d[RF_DIR_FRAME] = f->w[RF_W_FRAME_PAGES];
        for (uint32_t k = 0; k < f->w[RF_W_FRAMES]; k++) {
            d[RF_DIR_FRAME + 1 + k] = f->w[RF_W_FRAME + k];
        }
    }
    uint32_t* mk = (uint32_t*)reg[RF_R_MARKS];
    for (uint32_t m = 0; m < nm; m++) {
        uint32_t* s = mk + m * (RF_MARK_BYTES / 4);
        for (uint32_t k = 0; k < RF_MARK_BYTES / 4; k++) {
            s[k] = 0;
        }
        const rm_mark* x = &wk->marks[m];
        s[0] = x->mark;
        if (x->mark == RM_MARK_TILE) {
            s[RF_HDR_TILE] = x->tile;
            s[RF_HDR_COUNT] = x->count;
            s[RF_HDR_ROW0] = x->row0;
            s[RF_HDR_ROW1] = x->row1;
        }
        s[32] = x->slot;
    }
    f->out[RF_O_PAGES] += elapsed(&t);
    kmx_give(RF_R_SLOTS);
    kmx_give(RF_R_ITEMS);
    kmx_give(RF_R_DIR);
    kmx_give(RF_R_MARKS);
    f->out[RF_O_GIVE] += elapsed(&t);
    f->out[RF_O_MIDDLE] =
        (f->out[RF_O_TAKE] + f->out[RF_O_HEAVY_US] + f->out[RF_O_CLEAR] +
         f->out[RF_O_LAYOUT] + f->out[RF_O_NEXT] + f->out[RF_O_PAGES] +
         f->out[RF_O_GIVE]);
    if (f->w[RF_W_STEPS] < 3) {
        return 0;
    }

    /* PLACE, then SETUP */
    kmx_progress(5);
    record(&r, RF_SEG_PLACE, f);
    r.a0 = f->addr[RF_R_DIR];
    r.a1 = f->w[RF_W_DIR];
    r.a2 = f->addr[RF_R_MARKS];
    r.a4 = nm;
    r.a6 = workers;
    r.a7 = f->w[RF_W_LISTS];
    code = run(&r);
    f->out[RF_O_PLACE] += elapsed(&t);
    if (code) {
        return code;
    }
    kmx_progress(6);
    record(&r, RF_SEG_SETUP, f);
    r.a0 = f->w[RF_W_MESH];
    r.a1 = f->w[RF_W_LISTS];
    r.a2 = f->w[RF_W_BLOCKS];
    r.a3 = f->addr[RF_R_CAMERA];
    r.a4 = f->addr[RF_R_SLOTS];
    r.a6 = slots;
    r.a7 = f->addr[RF_R_ITEMS];
    code = run(&r);
    f->out[RF_O_SETUP] += elapsed(&t);
    if (code || f->w[RF_W_STEPS] < 4) {
        return code;
    }

    /* FILL */
    kmx_progress(7);
    record(&r, RF_SEG_FILL, f);
    r.a0 = f->w[RF_W_DIR];
    r.a1 = f->w[RF_W_LISTS];
    r.a3 = RF_PAGES_PER_ITEM;
    r.a4 = f->w[RF_W_PROF];
    code = run(&r);
    f->out[RF_O_FILL] += elapsed(&t);
    kmx_progress(8);
    return code;
}

int kmx_main(int hart) {
    (void)hart;
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
    const uint32_t need[RF_REGIONS] = {
        4 * 16,
        workers * RF_COUNTER_BYTES,
        RF_HEAVY_BYTES,
        workers * RF_HIST_MAX * 64,
        workers * RF_SLOTS_BYTES,
        RF_ITEMS_BYTES,
        workers * RF_DIR_BYTES,
        (RM_MAX_ITEMS + RM_MAX_CORES) * RF_MARK_BYTES,
        sizeof(work_t),
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
