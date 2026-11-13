/* The step between COUNT and SETUP in plain C (X280 thread, host check), as passes.py:
 * costs, bands, items to cores, list layout and directory, each core's next slots. */
#pragma once

#include <stdint.h>

#define RM_TILES 1024u
#define RM_GROUPS 16u /* 2-row groups of a tile */
#define RM_MAX_BANDS 16u
#define RM_MAX_HEAVY 256u /* tiles cut into bands; more: rm_middle refuses */
/* All bands number at most about 2 * fill cores / RM_SPLIT (1,503 at 248 cores)
 * besides the whole tiles. */
#define RM_MAX_ITEMS 2048u
#define RM_COUNTERS                                                  \
    (2 * RM_TILES) /* a core's counters: entries per tile, then rows \
                    */
#define RM_MAX_CORES 248u
#define RM_DIR_WORDS 1024u /* a directory page (raster.h) */
#define RM_DIR_FIRST 0u
#define RM_DIR_PAGES 1u
#define RM_DIR_ITEMS 2u
#define RM_DIR_IDS 3u
#define RM_DIR_MAX_ITEMS 997u
#define RM_ITEM_EMPTY 0x80000000u
#define RM_ITEM_ROW0_SHIFT 16u
#define RM_ITEM_ROW1_SHIFT 22u
#define RM_PAGE_SLOTS 16u
#define RM_MARK_TILE 0xFFFFFFFEu
#define RM_MARK_END 0xFFFFFFFFu

/* Cost model (lists.COST) and the cut (passes.SPLIT): per entry, per row, per tile
 * with entries, per empty tile; a tile above RM_SPLIT of a core's mean load is cut. */
#define RM_COST_TRI 260u
#define RM_COST_ROW 60u
#define RM_COST_TILE 6800u
#define RM_COST_EMPTY 2000u
#define RM_SPLIT 0.33

/* An item: a tile's rows [row0, row1) (0, 32: the whole tile), its entries. */
typedef struct {
    uint32_t tile, row0, row1, total;
} rm_item;

/* A list slot the step writes: a header (RM_MARK_TILE, tile, records, rows) or a
 * core's end slot (RM_MARK_END). */
typedef struct {
    uint32_t slot, mark, tile, count, row0, row1;
} rm_mark;

typedef struct {
    uint64_t entries[RM_TILES], rows[RM_TILES]; /* over the setup cores */
    uint64_t tile_cost[RM_TILES];
    uint64_t cost[RM_MAX_ITEMS];
    uint32_t order[RM_MAX_ITEMS];
    uint32_t tmp[RM_MAX_ITEMS];
    uint32_t owner[RM_MAX_ITEMS];
    uint32_t header[RM_MAX_ITEMS];
    uint64_t load[RM_MAX_CORES];
    uint32_t heap[RM_MAX_CORES];                /* cores by (load, core), least first */
    uint32_t first[RM_MAX_CORES + 1];           /* each core's items in s->tmp */
    uint32_t at[RM_MAX_ITEMS];                  /* next slots while they are written */
    uint64_t hist[RM_MAX_HEAVY][2 * RM_GROUPS]; /* heavy tiles' counts over the cores */
    int32_t heavy_of[RM_TILES];                 /* index among the heavy tiles, or -1 */
} rm_scratch;

/* The step in parts for the four cores, ranges side by side: rm_costs (all tiles),
 * rm_pick, rm_clear (all fill cores), rm_layout, rm_next (all items). */

/* Entries, rows and cost of tiles [t0, t1) over the cores. */
void rm_costs(
    const uint32_t* counters, uint32_t cores, rm_scratch* s, uint32_t t0, uint32_t t1
);

/* The tiles to cut, ascending, and their bands: returns how many; the first
 * RM_MAX_HEAVY of them are written. */
uint32_t rm_pick(rm_scratch* s, uint32_t fill_cores, uint32_t* heavy, uint32_t* bands);

/* Zeroes the directory pages of fill cores [c0, c1). */
void rm_clear(uint32_t* dir, uint32_t c0, uint32_t c1);

/* Returns the lists' slots, or 0 when the counts do not fit (a core over
 * RM_DIR_MAX_ITEMS items, RM_MAX_ITEMS items, too many cores or heavy tiles). */
uint32_t rm_layout(
    const uint16_t* hist, /* (setup cores, n_heavy, 32): starts, then ends per group */
    uint32_t setup_cores,
    uint32_t fill_cores,
    const uint32_t* heavy,
    const uint32_t* bands,
    uint32_t n_heavy,
    rm_scratch* s,
    rm_item* items,
    uint32_t* n_items,
    uint32_t* tile_item, /* RM_TILES + 1: tile t's items from tile_item[t] */
    uint32_t* dir,       /* fill cores x RM_DIR_WORDS, zeroed before */
    rm_mark* marks,
    uint32_t* n_marks
);

/* Setup core c's first slot of item i at next[c * next_stride + i], items [i0, i1). */
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
);

/* The whole step in turn: rm_costs and rm_pick ... */
uint32_t rm_heavy(
    const uint32_t* counters,
    uint32_t setup_cores,
    uint32_t fill_cores,
    rm_scratch* s,
    uint32_t* heavy,
    uint32_t* bands
);

/* ... then rm_clear, rm_layout and rm_next (0 also when the items pass next_stride). */
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
);
