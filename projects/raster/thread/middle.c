/* The middle pass between COUNT and SETUP (middle.h), as passes.py: tiles over RM_SPLIT
 * of the mean cut; items by falling cost to the least loaded core (ties: lower first).
 */
#include "middle.h"

/* Stable merge sort of idx by falling key, through tmp. */
static void sort_desc(uint32_t* idx, uint32_t* tmp, const uint64_t* key, uint32_t n) {
    for (uint32_t width = 1; width < n; width *= 2) {
        for (uint32_t lo = 0; lo < n; lo += 2 * width) {
            uint32_t mid = lo + width < n ? lo + width : n;
            uint32_t hi = lo + 2 * width < n ? lo + 2 * width : n;
            uint32_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi) {
                tmp[k++] = key[idx[j]] > key[idx[i]] ? idx[j++] : idx[i++];
            }
            while (i < mid) {
                tmp[k++] = idx[i++];
            }
            while (j < hi) {
                tmp[k++] = idx[j++];
            }
        }
        for (uint32_t k = 0; k < n; k++) {
            idx[k] = tmp[k];
        }
    }
}

/* Restores the heap of keys (load << 8 | core: by load, ties by core) after its
 * first key grew. */
static void sift_down(uint64_t* heap, uint32_t n) {
    uint32_t at = 0;
    const uint64_t x = heap[0];
    for (;;) {
        const uint32_t l = 2 * at + 1, r = l + 1;
        if (l >= n) {
            break;
        }
        const uint32_t m = r < n && heap[r] < heap[l] ? r : l;
        if (heap[m] >= x) {
            break;
        }
        heap[at] = heap[m];
        at = m;
    }
    heap[at] = x;
}

void rm_costs(
    const uint32_t* counters, uint32_t cores, rm_scratch* s, uint32_t t0, uint32_t t1
) {
    for (uint32_t t = t0; t < t1; t++) {
        s->entries[t] = s->rows[t] = 0;
    }
    /* core by core, so that the counters are read in order (a walk across the cores
     * per tile misses the cache at every step) */
    for (uint32_t c = 0; c < cores; c++) {
        const uint32_t* k = counters + c * RM_COUNTERS;
        for (uint32_t t = t0; t < t1; t++) {
            s->entries[t] += k[t];
            s->rows[t] += k[RM_TILES + t];
        }
    }
    for (uint32_t t = t0; t < t1; t++) {
        const uint64_t n = s->entries[t], r = s->rows[t];
        s->tile_cost[t] = (
            n ? RM_COST_TRI * n + RM_COST_ROW * r + RM_COST_TILE : RM_COST_EMPTY
        );
    }
}

uint32_t rm_pick(rm_scratch* s, uint32_t fill_cores, uint32_t* heavy, uint32_t* bands) {
    uint64_t sum = 0, max = 0;
    for (uint32_t t = 0; t < RM_TILES; t++) {
        sum += s->tile_cost[t];
        max = s->tile_cost[t] > max ? s->tile_cost[t] : max;
    }
    const double mean = (double)sum / (double)fill_cores;
    if ((double)max <= mean) {
        return 0;
    }
    const double target = RM_SPLIT * mean;
    uint32_t n = 0;
    for (uint32_t t = 0; t < RM_TILES; t++) {
        const double q = (double)s->tile_cost[t] / target;
        uint32_t k = (uint32_t)q;
        k += (double)k < q;
        k = k < RM_MAX_BANDS ? k : RM_MAX_BANDS;
        if (k >= 2) {
            if (n < RM_MAX_HEAVY) {
                heavy[n] = t;
                bands[n] = k;
            }
            n++;
        }
    }
    return n;
}

uint32_t rm_heavy(
    const uint32_t* counters,
    uint32_t setup_cores,
    uint32_t fill_cores,
    rm_scratch* s,
    uint32_t* heavy,
    uint32_t* bands
) {
    rm_costs(counters, setup_cores, s, 0, RM_TILES);
    return rm_pick(s, fill_cores, heavy, bands);
}

void rm_clear(uint32_t* dir, uint32_t c0, uint32_t c1) {
    for (uint32_t w = c0 * RM_DIR_WORDS; w < c1 * RM_DIR_WORDS; w++) {
        dir[w] = 0;
    }
}

/* Entries of a tile's group counts that reach rows [row0, row1). */
static uint64_t band_count(
    const uint16_t* f, const uint16_t* l, uint32_t row0, uint32_t row1
) {
    uint64_t n = 0;
    for (uint32_t g = 0; g < row1 / 2; g++) {
        n += f[g];
    }
    for (uint32_t g = 0; g < row0 / 2; g++) {
        n -= l[g];
    }
    return n;
}

/* Band bounds (rows) of one heavy tile from its counts summed over the cores. */
static uint32_t cut(
    const uint64_t* f, const uint64_t* l, uint32_t k, uint32_t* bounds
) {
    uint64_t cum[RM_GROUPS], c = 0, cf = 0, cl = 0;
    for (uint32_t g = 0; g < RM_GROUPS; g++) {
        cf += f[g];
        const uint64_t cover = cf - cl; /* starts so far, less ends before g */
        cl += l[g];
        c += 2 * RM_COST_ROW * cover + RM_COST_TRI * f[g];
        cum[g] = c;
    }
    uint32_t n = 0;
    bounds[n++] = 0;
    for (uint32_t j = 1; j < k; j++) {
        uint32_t g = 0;
        while (g < RM_GROUPS && k * cum[g] < cum[RM_GROUPS - 1] * j) {
            g++;
        }
        const uint32_t row = 2 * (g + 1);
        if (row > bounds[n - 1] && row < 2 * RM_GROUPS) {
            bounds[n++] = row;
        }
    }
    bounds[n++] = 2 * RM_GROUPS;
    return n - 1;
}

void rm_hist(
    const uint16_t* hist, uint32_t setup_cores, uint32_t n_heavy, rm_scratch* s
) {
    /* core by core (in memory order) */
    for (uint32_t h = 0; h < n_heavy; h++) {
        for (uint32_t g = 0; g < 2 * RM_GROUPS; g++) {
            s->hist[h][g] = 0;
        }
    }
    for (uint32_t c = 0; c < setup_cores; c++) {
        const uint16_t* hc = hist + c * n_heavy * 2 * RM_GROUPS;
        for (uint32_t h = 0; h < n_heavy; h++) {
            for (uint32_t g = 0; g < 2 * RM_GROUPS; g++) {
                s->hist[h][g] += hc[h * 2 * RM_GROUPS + g];
            }
        }
    }
}

uint32_t rm_items(
    const uint32_t* heavy,
    const uint32_t* bands,
    uint32_t n_heavy,
    rm_scratch* s,
    rm_item* items,
    uint32_t* n_items,
    uint32_t* tile_item
) {
    if (n_heavy > RM_MAX_HEAVY) {
        return 0;
    }
    for (uint32_t t = 0; t < RM_TILES; t++) {
        s->heavy_of[t] = -1;
    }
    for (uint32_t h = 0; h < n_heavy; h++) {
        s->heavy_of[heavy[h]] = (int32_t)h;
    }
    /* items in tile order, with their entries and costs */
    uint32_t ni = 0;
    for (uint32_t t = 0; t < RM_TILES; t++) {
        tile_item[t] = ni;
        const int32_t h = s->heavy_of[t];
        if (h < 0) {
            items[ni] = (rm_item){t, 0, 32, (uint32_t)s->entries[t]};
            s->cost[ni++] = s->tile_cost[t];
            continue;
        }
        const uint64_t* f = s->hist[h];
        const uint64_t* l = f + RM_GROUPS;
        uint32_t bounds[RM_MAX_BANDS + 1];
        const uint32_t nb = cut(f, l, bands[h], bounds);
        if (ni + nb + RM_TILES - 1 - t > RM_MAX_ITEMS) {
            return 0;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const uint32_t r0 = bounds[b], r1 = bounds[b + 1];
            uint64_t n = 0, cover = 0, cf = 0, cl = 0;
            for (uint32_t g = 0; g < r1 / 2; g++) {
                n += f[g]; /* starting before row1, less ending before row0 */
                n -= g < r0 / 2 ? l[g] : 0;
                cf += f[g];
                if (g >= r0 / 2) {
                    cover += cf - cl;
                }
                cl += l[g];
            }
            items[ni] = (rm_item){t, r0, r1, (uint32_t)n};
            s->cost[ni++] = (
                n
                    ? RM_COST_TRI * n + RM_COST_ROW * 2 * cover + RM_COST_TILE
                    : RM_COST_EMPTY
            );
        }
    }
    tile_item[RM_TILES] = ni;
    *n_items = ni;
    return 1;
}

void rm_sort(rm_scratch* s, uint32_t lo, uint32_t hi) {
    for (uint32_t i = lo; i < hi; i++) {
        s->order[i] = i;
    }
    sort_desc(s->order + lo, s->tmp + lo, s->cost, hi - lo);
}

void rm_merge(rm_scratch* s, uint32_t lo, uint32_t mid, uint32_t hi) {
    const uint64_t* key = s->cost;
    uint32_t* idx = s->order;
    uint32_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        s->tmp[k++] = key[idx[j]] > key[idx[i]] ? idx[j++] : idx[i++];
    }
    while (i < mid) {
        s->tmp[k++] = idx[i++];
    }
    while (j < hi) {
        s->tmp[k++] = idx[j++];
    }
    for (k = lo; k < hi; k++) {
        idx[k] = s->tmp[k];
    }
}

uint32_t rm_assign(uint32_t fill_cores, rm_scratch* s, uint32_t ni) {
    rm_sort(s, 0, ni);
    return rm_owners(fill_cores, s, ni);
}

uint32_t rm_owners(uint32_t fill_cores, rm_scratch* s, uint32_t ni) {
    if (fill_cores == 0 || fill_cores > RM_MAX_CORES) {
        return 0;
    }
    /* the least loaded core (ties: the lower) from a heap of the cores' keys; with all
     * loads 0 the cores in order are one */
    uint64_t* heap = s->heap;
    for (uint32_t c = 0; c < fill_cores; c++) {
        heap[c] = c;
    }
    for (uint32_t k = 0; k < ni; k++) {
        const uint32_t i = s->order[k];
        s->owner[i] = (uint32_t)(heap[0] & 0xFF);
        heap[0] += s->cost[i] << 8;
        sift_down(heap, fill_cores);
    }
    for (uint32_t c = 0; c < fill_cores; c++) {
        s->load[heap[c] & 0xFF] = heap[c] >> 8;
    }
    return 1;
}

uint32_t rm_layout(
    uint32_t fill_cores,
    const uint32_t* heavy,
    const uint32_t* bands,
    uint32_t n_heavy,
    rm_scratch* s,
    rm_item* items,
    uint32_t* n_items,
    uint32_t* tile_item,
    uint32_t* dir,
    rm_mark* marks,
    uint32_t* n_marks
) {
    if (
        !rm_items(heavy, bands, n_heavy, s, items, n_items, tile_item) ||
        !rm_assign(fill_cores, s, *n_items)
    ) {
        return 0;
    }
    const uint32_t ni = *n_items;
    /* each core's items in item (tile) order */
    for (uint32_t c = 0; c <= fill_cores; c++) {
        s->first[c] = 0;
    }
    for (uint32_t i = 0; i < ni; i++) {
        s->first[s->owner[i] + 1]++;
    }
    for (uint32_t c = 0; c < fill_cores; c++) {
        s->first[c + 1] += s->first[c];
    }
    for (uint32_t i = 0; i < ni; i++) {
        s->tmp[s->first[s->owner[i]]++] = i;
    }
    for (uint32_t c = fill_cores; c > 0; c--) {
        s->first[c] = s->first[c - 1];
    }
    s->first[0] = 0;
    /* each core's stream: headers, records, the end slot */
    uint32_t page = 0, nm = 0;
    for (uint32_t c = 0; c < fill_cores; c++) {
        uint32_t* d = dir + c * RM_DIR_WORDS;
        uint32_t at = RM_PAGE_SLOTS * page, n = 0;
        for (uint32_t j = s->first[c]; j < s->first[c + 1]; j++) {
            const uint32_t i = s->tmp[j];
            if (n == RM_DIR_MAX_ITEMS) {
                return 0;
            }
            const rm_item* it = &items[i];
            uint32_t id = it->tile;
            if (it->row0 != 0 || it->row1 != 32) {
                id |= it->row0 << RM_ITEM_ROW0_SHIFT | it->row1 << RM_ITEM_ROW1_SHIFT;
            }
            if (it->total == 0) {
                d[RM_DIR_IDS + n++] = id | RM_ITEM_EMPTY;
                continue;
            }
            d[RM_DIR_IDS + n++] = id;
            s->header[i] = at;
            marks[nm++] = (rm_mark){
                at, RM_MARK_TILE, it->tile, it->total, it->row0, it->row1
            };
            at += 1 + it->total;
        }
        marks[nm++] = (rm_mark){at, RM_MARK_END, 0, 0, 0, 0};
        const uint32_t pages = (
            (at + 1 - RM_PAGE_SLOTS * page + RM_PAGE_SLOTS - 1) / RM_PAGE_SLOTS
        );
        d[RM_DIR_FIRST] = page;
        d[RM_DIR_PAGES] = pages;
        d[RM_DIR_ITEMS] = n;
        page += pages;
    }
    *n_marks = nm;
    return RM_PAGE_SLOTS * page;
}

uint32_t rm_compact(
    uint32_t fill_cores,
    rm_scratch* s,
    const rm_item* items,
    uint32_t n_items,
    uint32_t* item_words,
    uint32_t* core_words
) {
    uint32_t* len = s->len;
    uint32_t* count = s->count;
    for (uint32_t c = 0; c < fill_cores; c++) {
        len[c] = count[c] = 0;
    }
    for (uint32_t i = 0; i < n_items; i++) {
        const uint32_t c = s->owner[i];
        count[c]++;
        len[c] += items[i].total ? 1 + items[i].total : 0;
    }
    uint32_t page = 0;
    for (uint32_t c = 0; c < fill_cores; c++) {
        if (count[c] > RM_DIR_MAX_ITEMS) {
            return 0;
        }
        const uint32_t pages = (len[c] + 1 + RM_PAGE_SLOTS - 1) / RM_PAGE_SLOTS;
        uint32_t* w = core_words + RM_CORE_WORDS * c;
        w[0] = page;
        w[1] = pages;
        w[2] = count[c];
        w[3] = RM_PAGE_SLOTS * page + len[c]; /* the end slot */
        len[c] = RM_PAGE_SLOTS * page;        /* from here: the next header */
        page += pages;
    }
    for (uint32_t i = 0; i < n_items; i++) {
        const uint32_t c = s->owner[i], total = items[i].total;
        s->header[i] = total ? len[c] : 0;
        item_words[RM_ITEM_WORDS * i] = c | total << 8;
        item_words[RM_ITEM_WORDS * i + 1] = s->header[i];
        len[c] += total ? 1 + total : 0;
    }
    return RM_PAGE_SLOTS * page;
}

void rm_base(
    const rm_scratch* s, const rm_item* items, uint32_t n_items, uint32_t* base
) {
    for (uint32_t i = 0; i < n_items; i++) {
        const int32_t h = s->heavy_of[items[i].tile];
        base[i] = (
            (items[i].total ? s->header[i] + 1 : 0) |
            (h < 0 ? 0 : (uint32_t)(h + 1) << RM_BASE_SLOT_BITS)
        );
    }
}

void rm_next(
    const uint32_t* counters,
    const uint16_t* hist,
    uint32_t setup_cores,
    uint32_t n_heavy,
    rm_scratch* s,
    const rm_item* items,
    uint32_t i0,
    uint32_t i1,
    uint32_t* next,
    uint32_t next_stride
) {
    for (uint32_t i = i0; i < i1; i++) {
        s->at[i] = items[i].total ? s->header[i] + 1 : 0;
    }
    for (uint32_t c = 0; c < setup_cores; c++) {
        const uint32_t* k = counters + c * RM_COUNTERS;
        uint32_t* nx = next + c * next_stride;
        for (uint32_t i = i0; i < i1; i++) {
            const rm_item* it = &items[i];
            nx[i] = s->at[i];
            if (it->total == 0) {
                continue;
            }
            const int32_t h = s->heavy_of[it->tile];
            if (h < 0) {
                s->at[i] += k[it->tile];
            } else {
                const uint16_t* hc = hist + (c * n_heavy + (uint32_t)h) * 2 * RM_GROUPS;
                s->at[i] += (
                    (uint32_t)band_count(hc, hc + RM_GROUPS, it->row0, it->row1)
                );
            }
        }
    }
}

uint32_t rm_middle(
    const uint32_t* counters,
    const uint16_t* hist,
    uint32_t setup_cores,
    uint32_t fill_cores,
    const uint32_t* heavy,
    const uint32_t* bands,
    uint32_t n_heavy,
    rm_scratch* s,
    rm_item* items,
    uint32_t* n_items,
    uint32_t* tile_item,
    uint32_t* dir,
    rm_mark* marks,
    uint32_t* n_marks,
    uint32_t* next,
    uint32_t next_stride
) {
    if (fill_cores > RM_MAX_CORES) {
        return 0;
    }
    if (n_heavy > RM_MAX_HEAVY) {
        return 0;
    }
    rm_clear(dir, 0, fill_cores);
    rm_hist(hist, setup_cores, n_heavy, s);
    const uint32_t slots = (
        rm_layout(
            fill_cores,
            heavy,
            bands,
            n_heavy,
            s,
            items,
            n_items,
            tile_item,
            dir,
            marks,
            n_marks
        )
    );
    if (slots == 0 || *n_items > next_stride) {
        return 0;
    }
    rm_next(
        counters, hist, setup_cores, n_heavy, s, items, 0, *n_items, next, next_stride
    );
    return slots;
}
