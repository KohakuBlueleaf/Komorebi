/* The frame thread's words, regions, result and codes, shared with card_thread.py,
 * which checks the kernels' numbers here against kernels/setup.h and raster.h. */
#pragma once

/* Handlers (kernels/setup.h) */
#define RF_HANDLER_COUNT 12u
#define RF_HANDLER_SETUP 13u
#define RF_HANDLER_HIST 14u
#define RF_HANDLER_PLACE 15u
#define RF_HANDLER_SUM 16u
#define RF_HANDLER_HSUM 17u
#define RF_HANDLER_FILL 1u   /* kernels/raster.h */
#define RF_PAGES_PER_ITEM 5u /* the frame's pages per tile: the fill's planes */

/* The directory's frame tensors (kernels/raster.h) */
#define RF_FRAME_MAX 15u

/* Page sizes (kernels/setup.h) */
#define RF_COUNTER_BYTES 8192u
#define RF_HEAVY_BYTES 4096u
#define RF_SLOTS_BYTES 8192u
#define RF_ITEMS_BYTES 8192u
#define RF_HIST_MAX 256u /* heavy tiles the histograms hold */
#define RF_HIST_BYTES (64u * RF_HIST_MAX)
#define RF_SETUP_PREFIX 0x80000000u /* SETUP's slots from SUM's and HSUM's prefixes */
#define RF_BASE_AT 8192u            /* in the items region: the base words */
#define RF_HPREFIX_ADDR_AT 16384u   /* then the address of HSUM's pages */
#define RF_HPREFIX_BYTES 32768u     /* an HSUM page, one per counting core */
#define RF_SUMS_BYTES 40960u /* SUM's totals (entries, rows per tile), then HSUM's */
#define RF_SUMS_HEAVY_AT 8192u
#define RF_FRAME_WORDS_AT 16448u /* in the items region: PLACE's frame words, */
#define RF_ITEM_WORDS_AT 16512u  /* the items' words (rm_compact), */
#define RF_CORE_WORDS_AT 32896u  /* the fill cores' words */
#define RF_ITEMS_REGION_BYTES 36864u
/* COUNT's and SETUP's blocks from the camera's region (kernels/setup.h, kRanges) */
#define RF_RANGES 0x80000000u
#define RF_RANGES_AT 64u   /* each core's first and end block, 64 bytes a core */
#define RF_COSTS_AT 16384u /* each core's COUNT and SETUP ticks, 128 bytes a core */
#define RF_CAMERA_BYTES 49152u
#define RF_MAX_CORES 248u

/* Words */
#define RF_W_CORES 0       /* cores that count and set up (the first ones) */
#define RF_W_WORKERS 1     /* the group, and the cores that fill */
#define RF_W_BANK 2        /* the regions' DRAM bank */
#define RF_W_MESH 3        /* the mesh tensor */
#define RF_W_BLOCKS 4      /* its blocks */
#define RF_W_LISTS 5       /* the lists tensor */
#define RF_W_LIST_SLOTS 6  /* its slots */
#define RF_W_DIR 7         /* the directory tensor */
#define RF_W_STEPS 8       /* how far: 1 COUNT, 2 middle pass, 3 PLACE, SETUP, 4 FILL */
#define RF_W_FRAME_PAGES 9 /* pages per frame tensor */
#define RF_W_FRAMES 10     /* frame tensors */
#define RF_W_FRAME 11      /* their addresses, RF_FRAME_MAX words */
#define RF_W_PROF (RF_W_FRAME + RF_FRAME_MAX) /* the fill's profile tensor */
#define RF_W_REPEAT (RF_W_PROF + 1)           /* frames in a row (0: one) */
#define RF_W_SPROF (RF_W_REPEAT + 1) /* the passes' profile tensor (a9), 0: none */
/* Tensors only the cores read and write, a page per worker: */
#define RF_W_COUNTERS (RF_W_SPROF + 1)  /* COUNT's counters */
#define RF_W_HIST (RF_W_COUNTERS + 1)   /* HIST's histograms (RF_HIST_BYTES pages) */
#define RF_W_SLOTS (RF_W_HIST + 1)      /* SUM's entries before each core per tile */
#define RF_W_HPREFIX (RF_W_SLOTS + 1)   /* HSUM's counts before each core */
#define RF_W_BALANCE (RF_W_HPREFIX + 1) /* 1: COUNT's and SETUP's ranges by cost */
#define RF_WORDS (RF_W_BALANCE + 1)

/* Regions, in this order */
#define RF_R_CAMERA 0 /* the camera's 16 numbers */
#define RF_R_HEAVY 1  /* the heavy tiles: their number, then the tiles */
#define RF_R_ITEMS 2  /* SETUP's items page, base words, PLACE's words */
#define RF_R_WORK 3   /* the thread's own memory */
#define RF_R_SUMS 4   /* SUM's and HSUM's totals over the cores */
#define RF_REGIONS 5u

/* Result words: what the middle pass found, and the timer ticks of each part */
#define RF_O_SLOTS 0
#define RF_O_ITEMS 1
#define RF_O_MARKS 2
#define RF_O_HEAVY 3
#define RF_O_COUNT 4  /* COUNT submitted to its event */
#define RF_O_HIST 5   /* HIST, with the heavy tiles' page */
#define RF_O_MIDDLE 6 /* the middle pass: totals to the words SETUP and PLACE read */
#define RF_O_PLACE 7
#define RF_O_SETUP 8
#define RF_O_TAKE 9 /* the middle pass's parts: the totals' invalidates */
#define RF_O_HEAVY_MICROSECONDS 10 /* the tiles to cut */
#define RF_O_ASSIGN 11             /* rm_owners */
#define RF_O_PAGES 12              /* the items page and frame words */
#define RF_O_GIVE 13               /* the flushes to the grid */
#define RF_O_CUT 14                /* rm_items */
#define RF_O_COMPACT 15            /* rm_compact, rm_base */
#define RF_O_FILL 16
#define RF_O_FRAMES 17 /* frames finished; the parts' ticks are their sums */
#define RF_O_SUM 18    /* SUM submitted to its event */
#define RF_O_HSUM 19   /* HSUM */
#define RF_O_SORT 20   /* the items by falling cost (rm_sort, rm_merge) */
#define RF_OUT_WORDS 21u

/* Codes */
#define RF_BAD_ARGS 1u
#define RF_RING 2u    /* a command could not be submitted or its event did not come */
#define RF_COMMAND 3u /* a command did not finish (timeout, rejected) */
#define RF_TOO_MANY_HEAVY 4u
#define RF_NO_LAYOUT 5u   /* the middle pass could not lay the lists out */
#define RF_LISTS_SMALL 6u /* the lists tensor has fewer slots than the frame needs */
#define RF_PARALLEL 7u    /* a part on the four cores did not finish (late, stopped) */
