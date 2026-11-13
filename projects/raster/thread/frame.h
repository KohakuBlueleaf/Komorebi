/* The frame thread's words, regions, result and codes, shared with card_thread.py,
 * which checks the kernels' numbers here against kernels/setup.h and raster.h. */
#pragma once

/* Segments (kernels/setup.h) */
#define RF_SEG_COUNT 12u
#define RF_SEG_SETUP 13u
#define RF_SEG_HIST 14u
#define RF_SEG_PLACE 15u
#define RF_SEG_FILL 1u       /* kernels/raster.h */
#define RF_PAGES_PER_ITEM 5u /* the frame's pages per tile: the fill's planes */

/* List slots and directory (kernels/raster.h) */
#define RF_HDR_TILE 1u
#define RF_HDR_COUNT 2u
#define RF_HDR_ROW0 3u
#define RF_HDR_ROW1 4u
#define RF_DIR_FRAME 1000u
#define RF_FRAME_MAX 15u

/* Page sizes (kernels/setup.h) */
#define RF_COUNTER_BYTES 8192u
#define RF_HEAVY_BYTES 4096u
#define RF_SLOTS_BYTES 8192u
#define RF_ITEMS_BYTES 8192u
#define RF_DIR_BYTES 4096u
#define RF_MARK_BYTES 192u
#define RF_HIST_MAX 256u /* heavy tiles the histogram region holds */

/* Words */
#define RF_W_CORES 0       /* cores that count and set up (the first ones) */
#define RF_W_WORKERS 1     /* the team, and the cores that fill */
#define RF_W_BANK 2        /* the regions' DRAM bank */
#define RF_W_MESH 3        /* the mesh tensor */
#define RF_W_BLOCKS 4      /* its blocks */
#define RF_W_LISTS 5       /* the lists tensor */
#define RF_W_LIST_SLOTS 6  /* its slots */
#define RF_W_DIR 7         /* the directory tensor */
#define RF_W_STEPS 8       /* 1: COUNT; 2: and the step; 3: and PLACE, SETUP; 4: FILL */
#define RF_W_FRAME_PAGES 9 /* pages per frame tensor */
#define RF_W_FRAMES 10     /* frame tensors */
#define RF_W_FRAME 11      /* their addresses, RF_FRAME_MAX words */
#define RF_W_PROF (RF_W_FRAME + RF_FRAME_MAX) /* the fill's profile tensor */
#define RF_W_REPEAT (RF_W_PROF + 1)           /* frames in a row (0: one) */
#define RF_WORDS (RF_W_REPEAT + 1)

/* Regions, in this order */
#define RF_R_CAMERA 0   /* the camera's 16 numbers */
#define RF_R_COUNTERS 1 /* COUNT's counters, a page per worker */
#define RF_R_HEAVY 2    /* the heavy tiles: their number, then the tiles */
#define RF_R_HIST 3     /* HIST's histograms, 64 bytes per heavy tile per worker */
#define RF_R_SLOTS 4    /* SETUP's next slot per item, a page per worker */
#define RF_R_ITEMS 5    /* SETUP's items page */
#define RF_R_DIR 6      /* the directory pages, a page per worker */
#define RF_R_MARKS 7    /* list headers and end slots with their slot numbers */
#define RF_R_WORK 8     /* the thread's own memory */
#define RF_REGIONS 9u

/* Result words: what the step found, and the timer ticks of each part */
#define RF_O_SLOTS 0
#define RF_O_ITEMS 1
#define RF_O_MARKS 2
#define RF_O_HEAVY 3
#define RF_O_COUNT 4  /* COUNT issued to its event */
#define RF_O_HIST 5   /* HIST, with the heavy tiles' page */
#define RF_O_MIDDLE 6 /* the step: counters to the pages SETUP and PLACE read */
#define RF_O_PLACE 7
#define RF_O_SETUP 8
#define RF_O_TAKE 9      /* the step's parts: the counters' hand-over */
#define RF_O_HEAVY_US 10 /* the tiles to cut */
#define RF_O_LAYOUT 11   /* rm_layout (after HIST) */
#define RF_O_PAGES 12    /* the items page, directory frame words, marks */
#define RF_O_GIVE 13     /* the hand-overs to the grid */
#define RF_O_CLEAR 14    /* rm_clear */
#define RF_O_NEXT 15     /* rm_next */
#define RF_O_FILL 16
#define RF_O_FRAMES 17 /* frames finished; the parts' ticks are their sums */
#define RF_OUT_WORDS 18u

/* Codes */
#define RF_BAD_ARGS 1u
#define RF_RING 2u   /* a record could not be issued or its event did not come */
#define RF_RECORD 3u /* a record did not finish (timeout, refused) */
#define RF_TOO_MANY_HEAVY 4u
#define RF_NO_LAYOUT 5u   /* rm_middle could not lay the lists out */
#define RF_LISTS_SMALL 6u /* the lists tensor has fewer slots than the frame needs */
#define RF_PAR 7u         /* a part on the four harts did not finish (late, stopped) */
