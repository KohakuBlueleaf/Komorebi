/* The frame on the work cluster: each command submitted and waited for; before each
 * trace four X280 cores cut its packets into chunk commands of equal predicted cost. */
#include <stdint.h>

#include "kmx.h"
#include "plan.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per command, longer than its deadline */

typedef struct {
    uint32_t pred[RP_PACKETS_MAX]; /* each packet's predicted cost */
    uint32_t bound[RP_CORES_MAX * RP_CHUNKS + 1];
} work_t;

typedef struct {
    const uint32_t* w;
    const uint32_t* table;
    uint32_t workers, packets, cap, mode, bank1;
    uint32_t dirs_addr, cost_addr; /* bank addresses */
    uint8_t* dirs;
    const uint8_t* cost;
    work_t* wk;
    uint32_t* log;
    uint32_t used[RP_TRACES]; /* chunk commands of the trace's last submit */
    uint32_t out[RP_OUT_WORDS];
    void* part; /* the plan's parallel step, after work_t in the work region */
} plan_t;

/* One parallel step: the source trace and how it predicts (source_of), the owners'
 * sums, each quarter's sum and what each X280 core found. */
typedef struct {
    plan_t* p;
    uint32_t src, owner, plus, pieces;
    uint64_t own[4][RP_CORES_MAX];
    uint64_t part[4];
    uint32_t bad[4];
} part_t;

/* part_t's place in the work region: after work_t, 64-byte aligned (uint64 fields) */
#define PART_AT ((sizeof(work_t) + 63u) & ~(uint64_t)63u)
_Static_assert(PART_AT % 64u == 0, "part_t aligned");

/* Fills a field by field (a struct initializer may call memset, which does not link) */
static void part_init(part_t* a, plan_t* p) {
    a->p = p;
    a->src = RP_TRACES;
    a->owner = a->plus = a->pieces = 0;
    for (int h = 0; h < 4; h++) {
        a->part[h] = 0;
        a->bad[h] = 0;
        for (uint32_t c = 0; c < RP_CORES_MAX; c++) {
            a->own[h][c] = 0;
        }
    }
}

static uint32_t lo_of(uint32_t n, int xcore) {
    return (uint32_t)((uint64_t)n * (uint32_t)xcore / 4u);
}

static uint32_t hi_of(uint32_t n, int xcore) {
    return lo_of(n, xcore + 1);
}

static uint32_t block_at(const plan_t* p, uint32_t t, uint32_t k, uint32_t c) {
    return ((t * RP_CHUNKS + k) * p->workers + c) * RP_COST_BYTES;
}

static uint32_t dir_at(const plan_t* p, uint32_t t, uint32_t k, uint32_t c) {
    return ((t * RP_CHUNKS + k) * p->workers + c) * RP_DIR_BYTES;
}

/* The owner core of packet i (the owners' ranges are in core order) */
static uint32_t owner_of(const plan_t* p, uint32_t i) {
    const uint32_t* base = p->table + RP_T_BASE;
    uint32_t lo = 0, hi = p->workers - 1;
    while (lo < hi) {
        const uint32_t mid = (lo + hi + 1) / 2;
        if (base[2 * mid] <= i) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

/* The source trace's cost blocks of a quarter of the cores: each packet's cycles (or
 * hits, owner bit 1) + plus into pred, or with owner bit 0 summed by owner core */
static void gather_part(int xcore, void* arg) {
    part_t* a = (part_t*)arg;
    plan_t* p = a->p;
    const uint32_t* base = p->table + RP_T_BASE;
    const uint32_t from = a->owner & 2 ? RP_COST_HEAD + RP_COST_MAX : RP_COST_HEAD;
    for (uint32_t c = lo_of(p->workers, xcore); c < hi_of(p->workers, xcore); c++) {
        for (uint32_t k = 0; k < p->used[a->src]; k++) {
            const uint32_t* b = (const uint32_t*)(p->cost + block_at(p, a->src, k, c));
            const uint32_t first = b[0], n = b[1];
            if (n > RP_COST_MAX || first > p->packets || n > p->packets - first) {
                a->bad[xcore] = 1;
                continue;
            }
            if ((a->owner & 1) == 0) {
                for (uint32_t i = 0; i < n; i++) {
                    p->wk->pred[first + i] = b[from + i] + a->plus;
                }
                continue;
            }
            uint32_t o = n ? owner_of(p, first) : 0;
            for (uint32_t i = 0; i < n; i++) {
                while (o + 1 < p->workers && base[2 * (o + 1)] <= first + i) {
                    o++;
                }
                a->own[xcore][o] += b[from + i];
            }
        }
    }
}

/* Each owner's packets: the owner's mean + plus */
static void spread_part(int xcore, void* arg) {
    part_t* a = (part_t*)arg;
    plan_t* p = a->p;
    const uint32_t* base = p->table + RP_T_BASE;
    for (uint32_t c = lo_of(p->workers, xcore); c < hi_of(p->workers, xcore); c++) {
        const uint32_t f = base[2 * c], n = base[2 * c + 1];
        const uint64_t s = a->own[0][c] + a->own[1][c] + a->own[2][c] + a->own[3][c];
        const uint32_t v = (n ? (uint32_t)(s / n) : 0) + a->plus;
        for (uint32_t i = f; i < f + n && i < p->packets; i++) {
            p->wk->pred[i] = v;
        }
    }
}

static void equal_part(int xcore, void* arg) {
    part_t* a = (part_t*)arg;
    for (
        uint32_t i = lo_of(a->p->packets, xcore); i < hi_of(a->p->packets, xcore); i++
    ) {
        a->p->wk->pred[i] = 1;
    }
}

static void sum_part(int xcore, void* arg) {
    part_t* a = (part_t*)arg;
    const uint32_t* v = a->p->wk->pred;
    uint64_t s = 0;
    for (
        uint32_t i = lo_of(a->p->packets, xcore); i < hi_of(a->p->packets, xcore); i++
    ) {
        s += v[i];
    }
    a->part[xcore] = s;
}

/* pred for trace t: from the source's blocks, by owner, or equal packets */
static int predict(plan_t* p, part_t* a) {
    if (a->src >= RP_TRACES) {
        return kmx_par(equal_part, a) == KMX_OK ? 0 : -1;
    }
    kmx_invalidate_at(
        RP_R_COST,
        block_at(p, a->src, 0, 0),
        (uint64_t)p->used[a->src] * p->workers * RP_COST_BYTES
    );
    if (kmx_par(gather_part, a) != KMX_OK) {
        return -1;
    }
    if ((a->owner & 1) && kmx_par(spread_part, a) != KMX_OK) {
        return -1;
    }
    return (a->bad[0] | a->bad[1] | a->bad[2] | a->bad[3]) ? -1 : 0;
}

static uint64_t target_of(uint64_t total, uint32_t pieces, uint32_t j) {
    return total / pieces * j + total % pieces * j / pieces;
}

/* Piece j's end: the packet boundary nearest to j / pieces of the total cost, for the
 * targets in this X280 core's quarter (one scan from the quarter's start sum) */
static void cut_part(int xcore, void* arg) {
    part_t* a = (part_t*)arg;
    work_t* wk = a->p->wk;
    const uint32_t n = a->p->packets, pieces = a->pieces;
    const uint64_t total = a->part[0] + a->part[1] + a->part[2] + a->part[3];
    if (total == 0) {
        for (
            uint32_t j = lo_of(pieces, xcore) + (xcore == 0); j < hi_of(pieces, xcore);
            j++
        ) {
            wk->bound[j] = (uint32_t)((uint64_t)n * j / pieces);
        }
        return;
    }
    uint64_t from = 0;
    for (int h = 0; h < xcore; h++) {
        from += a->part[h];
    }
    const uint64_t to = from + a->part[xcore];
    uint32_t i = lo_of(n, xcore), end = hi_of(n, xcore);
    uint64_t s = from; /* the cost before packet i */
    uint32_t j = xcore == 0 ? 1 : (uint32_t)(((from + 1) * pieces + total - 1) / total);
    for (; j < pieces; j++) {
        const uint64_t target = target_of(total, pieces, j);
        if (xcore < 3 && target > to) {
            break;
        }
        while (i < end && s + wk->pred[i] < target) {
            s += wk->pred[i];
            i++;
        }
        /* the boundary before or after packet i, whichever is nearer the target */
        uint32_t at = i;
        if (i < end && s + wk->pred[i] - target <= target - s) {
            at = i + 1;
        }
        wk->bound[j] = at;
    }
}

/* Cuts into workers x m pieces, m from the least that can hold the packets, until no
 * piece has more than cap; at RP_CHUNKS pieces are capped from the front. */
static uint32_t cut(plan_t* p, part_t* a) {
    work_t* wk = p->wk;
    const uint32_t per = p->workers * p->cap;
    if (kmx_par(sum_part, a) != KMX_OK) {
        return 0;
    }
    uint32_t m = (p->packets + per - 1) / per;
    for (; m <= RP_CHUNKS; m++) {
        a->pieces = p->workers * m;
        wk->bound[0] = 0;
        wk->bound[a->pieces] = p->packets;
        if (kmx_par(cut_part, a) != KMX_OK) {
            return 0;
        }
        uint32_t big = 0;
        for (uint32_t j = 0; j < a->pieces; j++) {
            const uint32_t n = wk->bound[j + 1] - wk->bound[j];
            big = n > big ? n : big;
        }
        if (big <= p->cap) {
            return m;
        }
    }
    m = RP_CHUNKS;
    const uint32_t pieces = p->workers * m;
    for (uint32_t j = 0; j < pieces; j++) {
        const uint32_t left = (pieces - j - 1) * p->cap;
        uint32_t end = wk->bound[j + 1];
        end = end > wk->bound[j] + p->cap ? wk->bound[j] + p->cap : end;
        end = p->packets > left && end < p->packets - left ? p->packets - left : end;
        wk->bound[j + 1] = end;
    }
    return wk->bound[pieces] == p->packets ? m : 0;
}

/* Trace t's chunks as the host cuts them: owner c's k-th run of cap packets */
static uint32_t host_cut(plan_t* p) {
    const uint32_t* base = p->table + RP_T_BASE;
    uint32_t most = 0;
    for (uint32_t c = 0; c < p->workers; c++) {
        most = base[2 * c + 1] > most ? base[2 * c + 1] : most;
    }
    const uint32_t m = most ? (most + p->cap - 1) / p->cap : 1;
    if (m > RP_CHUNKS) {
        return 0;
    }
    for (uint32_t c = 0; c < p->workers; c++) {
        for (uint32_t k = 0; k < m; k++) {
            const uint32_t n = base[2 * c + 1], at = k * p->cap;
            const uint32_t take = n > at ? (n - at < p->cap ? n - at : p->cap) : 0;
            p->wk->bound[c * m + k] = base[2 * c] + (take ? at : 0);
            p->wk->pred[c * m + k] = take; /* the counts, kept beside the starts */
        }
    }
    return m;
}

/* Which trace predicts trace t (RP_TRACES: none) and how. The cold frame of
 * RP_MODE_FRAME: the shadow by the primary's; plan() cuts the rest as the host does. */
static void source_of(const plan_t* p, uint32_t t, uint32_t frame, part_t* a) {
    a->src = RP_TRACES;
    a->owner = 0;
    a->plus = 0;
    if (p->mode == RP_MODE_FRAME) {
        a->src = frame > 0 ? t : (t == 1 ? 0 : RP_TRACES);
    } else if (p->mode == RP_MODE_TRACE && t >= 1) {
        a->src = t == 3 ? 2 : 0;
        a->owner = t == 1 ? 0 : 1;
    } else if (p->mode == RP_MODE_COUNTS && t >= 1) {
        a->src = t == 3 ? 2 : 0;
        a->owner = t == 1 ? 2 : 3;
        a->plus = 1;
    }
    if (a->src < RP_TRACES && p->used[a->src] == 0) {
        a->src = RP_TRACES;
    }
}

/* Trace t's directory entries for its m chunk commands, from the bounds */
static void write_dirs(plan_t* p, uint32_t t, uint32_t m, int host) {
    for (uint32_t c = 0; c < p->workers; c++) {
        for (uint32_t k = 0; k < m; k++) {
            uint32_t* d = (uint32_t*)(p->dirs + dir_at(p, t, k, c));
            const uint32_t j = c * m + k;
            for (uint32_t i = 0; i < RP_DIR_BYTES / 4; i++) {
                d[i] = 0;
            }
            d[RP_DIR_FIRST] = p->wk->bound[j];
            d[RP_DIR_COUNT] =
                (host ? p->wk->pred[j] : p->wk->bound[j + 1] - p->wk->bound[j]);
            d[RP_DIR_COST] = p->cost_addr + block_at(p, t, k, c);
        }
    }
    kmx_flush_at(
        RP_R_DIRS, dir_at(p, t, 0, 0), (uint64_t)m * p->workers * RP_DIR_BYTES
    );
}

/* Trace t's plan for this frame: its chunk commands' count, 0 on failure */
static uint32_t plan(plan_t* p, uint32_t t, uint32_t frame, uint32_t* sub) {
    part_t* a = (part_t*)p->part;
    part_init(a, p);
    source_of(p, t, frame, a);
    /* the frame mode without a source cuts as the host: equal packets lose (WHY-SLOW)
     */
    if (p->mode == RP_MODE_HOST || (p->mode == RP_MODE_FRAME && a->src == RP_TRACES)) {
        const uint32_t m = host_cut(p);
        if (m) {
            write_dirs(p, t, m, 1);
        }
        return m;
    }
    uint64_t at = kmx_time();
    if (predict(p, a) != 0) {
        return 0;
    }
    sub[0] += (uint32_t)(kmx_time() - at);
    at = kmx_time();
    const uint32_t m = cut(p, a);
    sub[2] += (uint32_t)(kmx_time() - at);
    at = kmx_time();
    if (m) {
        write_dirs(p, t, m, 0);
    }
    sub[3] += (uint32_t)(kmx_time() - at);
    return m;
}

/* Submits r and waits for its event: 0, or the code to report */
static uint32_t run(plan_t* p, const km_command_t* r, uint32_t* ticks) {
    const uint64_t until = kmx_time() + WAIT_TICKS;
    uint32_t handle;
    int e = kmx_submit(r, &handle);
    while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        e = kmx_submit(r, &handle);
    }
    if (e != KMX_OK) {
        return RP_RING;
    }
    kmx_event ev;
    if (kmx_wait(handle, kmx_time() + WAIT_TICKS, &ev) != KMX_OK) {
        return RP_RING;
    }
    *ticks = ev.ticks;
    p->out[2] = ev.kind;
    p->out[3] = ev.aux;
    return ev.kind == KM_EV_DONE ? 0u : RP_COMMAND;
}

/* Fills r field by field (a struct assignment may call memset, which does not link) */
static void command(km_command_t* r, const uint32_t* e) {
    r->seq = r->land1 = r->land2 = r->land3 = 0;
    r->hdr = e[1];
    r->group = e[2];
    r->a0 = e[3];
    r->a1 = e[4];
    r->a2 = e[5];
    r->a3 = e[6];
    r->a4 = e[7];
    r->a5 = e[8];
    r->a6 = e[9];
    r->a7 = e[10];
    r->a8 = e[11];
    r->a9 = e[12];
}

static uint32_t frame_once(plan_t* p, uint32_t frame) {
    uint32_t* log = p->log + frame * RP_LOG_WORDS;
    uint32_t submitted = 0;
    const uint64_t t0 = kmx_time();
    for (uint32_t i = 0; i < p->w[RP_W_COMMANDS]; i++) {
        const uint32_t* e = p->table + i * RP_REC_WORDS;
        const uint32_t kind = e[0] & 0xFFu;
        if ((e[0] & RP_KIND_FIRST) && frame > 0) {
            continue;
        }
        km_command_t r;
        command(&r, e);
        uint32_t m = 1;
        if (kind != 0) {
            const uint32_t t = kind - 1;
            const uint64_t a = kmx_time();
            m = plan(p, t, frame, log + RP_L_SUB);
            log[RP_L_PLAN + t] = (uint32_t)(kmx_time() - a);
            log[RP_L_CHUNKS + t] = m;
            if (m == 0) {
                p->out[1] = i;
                return RP_PLAN;
            }
            p->used[t] = m;
            r.a9 |= p->bank1 << RP_BANK_SHIFT;
        }
        for (uint32_t k = 0; k < m; k++) {
            if (kind != 0) {
                const uint32_t t = kind - 1;
                r.a0 = p->dirs_addr + dir_at(p, t, k, 0);
                r.a4 = p->table[RP_T_PROF + t * RP_CHUNKS + k];
            }
            uint32_t ticks = 0;
            const uint32_t code = run(p, &r, &ticks);
            if (submitted < RP_LOG_WORDS - RP_L_COMMAND) {
                log[RP_L_COMMAND + submitted] = ticks;
            }
            submitted++;
            if (code) {
                p->out[1] = i;
                return code;
            }
        }
    }
    log[RP_L_SUBMITTED] = submitted;
    log[RP_L_FRAME] = (uint32_t)(kmx_time() - t0);
    return 0;
}

static int finish(uint32_t code, plan_t* p) {
    kmx_flush(RP_R_LOG);
    kmx_result(code, p->out, RP_OUT_WORDS);
    return 0;
}

int kmx_main(int xcore) {
    (void)xcore;
    const kmx_args* a = kmx_args_get();
    plan_t p;
    for (uint32_t i = 0; i < RP_OUT_WORDS; i++) {
        p.out[i] = 0;
    }
    for (uint32_t t = 0; t < RP_TRACES; t++) {
        p.used[t] = 0;
    }
    if (a->nwords < RP_WORDS || a->nregions < RP_REGIONS) {
        return finish(RP_BAD_ARGS, &p);
    }
    p.w = a->words;
    const uint64_t need[RP_REGIONS] = {
        4u * RP_TABLE_WORDS,
        RP_DIRS_BYTES,
        RP_COST_REGION_BYTES,
        PART_AT + sizeof(part_t),
        RP_LOG_BYTES,
    };
    void* reg[RP_REGIONS];
    for (int i = 0; i < (int)RP_REGIONS; i++) {
        uint64_t bytes;
        reg[i] = kmx_region(i, &bytes);
        if (reg[i] == 0 || bytes < need[i] || a->region[i].offset >> 32) {
            return finish(RP_BAD_ARGS, &p);
        }
    }
    p.table = (const uint32_t*)reg[RP_R_TABLE];
    p.dirs = (uint8_t*)reg[RP_R_DIRS];
    p.cost = (const uint8_t*)reg[RP_R_COST];
    p.wk = (work_t*)reg[RP_R_WORK];
    p.part = (uint8_t*)reg[RP_R_WORK] + PART_AT;
    p.log = (uint32_t*)reg[RP_R_LOG];
    p.dirs_addr = (uint32_t)a->region[RP_R_DIRS].offset;
    p.cost_addr = (uint32_t)a->region[RP_R_COST].offset;
    p.mode = p.w[RP_W_MODE];
    p.workers = p.w[RP_W_WORKERS];
    p.packets = p.w[RP_W_PACKETS];
    p.cap = p.w[RP_W_CAP];
    p.bank1 = 1 + p.w[RP_W_BANK];
    const uint32_t frames = p.w[RP_W_FRAMES];
    if (p.mode > RP_MODE_FRAME || p.workers == 0 || p.workers > RP_CORES_MAX ||
        p.packets == 0 || p.packets > RP_PACKETS_MAX || p.cap == 0 ||
        p.cap > RP_COST_MAX || p.w[RP_W_COMMANDS] > RP_COMMANDS_MAX || frames == 0 ||
        frames > RP_FRAMES_MAX) {
        return finish(RP_BAD_ARGS, &p);
    }
    /* lines an earlier image left dirty would be written over what the cores write */
    kmx_flush(RP_R_COST);
    for (uint32_t f = 0; f < frames; f++) {
        kmx_progress(f + 1);
        const uint32_t code = frame_once(&p, f);
        if (code) {
            return finish(code, &p);
        }
        p.out[0] = f + 1;
    }
    return finish(RP_OK, &p);
}
