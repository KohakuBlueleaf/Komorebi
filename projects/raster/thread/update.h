/* UPDATE's regions, words, times rows and codes (update.c), shared with update.py;
 * the sizes are the scene's, in the words. */
#pragma once

#include <stdint.h>

#include "kmx.h"

/* Regions after frame.h's (RF_REGIONS), in this order */
#define RF_R_UPDATE 10     /* the header: the apply's words 0 to 15, then RF_U_* */
#define RF_R_ELEMENTS 11   /* the elements' arrays and the field's operator by row */
#define RF_R_SHADOW 12     /* the shadow map's pages, fetched from the light pages */
#define RF_R_WORK_U 13     /* sky, L0, the passes' total, x's and y's tiles, the rows */
#define RF_R_FIELDPAGES 14 /* the field's pages as they go to the light pages */
#define RF_UPDATE_REGIONS 15u
#define RF_UPDATE_BYTES 4096u /* the update region */

/* Words of the update region (uint32). 0 to 15 are the apply's: 0 to 9 its command's
 * a0 to a9, 10 its group (first | count << 16). */
#define RF_U_APPLY_A0 0
#define RF_U_APPLY_GROUP 10
#define RF_U_N 16            /* elements, a multiple of 32 (pad rows 0) */
#define RF_U_TILES 17        /* x's and y's tiles: N / 32 */
#define RF_U_ROWS 18         /* the field's rows: voxels x 6 */
#define RF_U_WEIGHTS 19      /* the operator's (column, weight) pairs */
#define RF_U_PAGES 20        /* the field's pages */
#define RF_U_LIGHT 21        /* the light pages tensor's address */
#define RF_U_FIELD_FIRST 22  /* the field's first light page */
#define RF_U_SHADOW_PAGES 23 /* the shadow map's pages (the light pages' first) */
#define RF_U_SIDE 24         /* the shadow map's side */
#define RF_U_X 25            /* x's tensor (bf16 tiles), its page bytes */
#define RF_U_X_PAGE 26
#define RF_U_Y 27 /* y's tensor (float32 tiles), its page bytes */
#define RF_U_Y_PAGE 28
#define RF_U_CORES 29 /* the cores that fetch and store (from core 0) */
/* byte offsets in the elements region: position, normal, base, albedo / pi, transfer;
 * the operator's row starts, (column, weight) pairs, each row's place (update.py) */
#define RF_U_E_POSITION 30
#define RF_U_E_NORMAL 31
#define RF_U_E_BASE 32
#define RF_U_E_ALBEDO 33
#define RF_U_E_TRANSFER 34
#define RF_U_E_STARTS 35
#define RF_U_E_PAIRS 36
#define RF_U_E_PLACES 37
/* byte offsets in the work region: sky, L0, total [N, 3] float32, x's tiles (bf16),
 * y's tiles (float32), the rows' values [rows, 3] float32 */
#define RF_U_W_SKY 38
#define RF_U_W_L0 39
#define RF_U_W_TOTAL 40
#define RF_U_W_X 41
#define RF_U_W_Y 42
#define RF_U_W_VALUES 43
/* the update's inputs, written by the host on each sun change: */
#define RF_U_BOUNCES 44    /* the passes, 0 to RF_U_BOUNCES_MAX */
#define RF_U_TEXEL 45      /* float: the shadow texel, q = position + normal texel */
#define RF_U_BIAS 46       /* float: the shadow bias in 1/z */
#define RF_U_SUN 47        /* float x 3: the sun's direction */
#define RF_U_IRRADIANCE 50 /* float x 3: the sun's irradiance */
#define RF_U_CAMERA 53     /* float x 16: the sun's camera */
#define RF_U_SKY 69        /* float x 25 x 3: the sky's coefficients, term by term */
#define RF_U_WORDS (RF_U_SKY + RF_SKY_TERMS * 3)
#define RF_SKY_TERMS 25
#define RF_U_BOUNCES_MAX 3u
#define RF_PAGE_TEXELS 1024u /* a light page's 32-bit texels */

/* The times rows (update region from word RF_U_TIMES, update k's at row k %
 * RF_U_TIMES_ROWS): ticks of each part of the update, summed over its steps (README) */
#define RF_U_TIMES 256
#define RF_U_TIMES_ROWS 16
#define RF_UT_FETCH 0   /* the shadow map's pages fetched */
#define RF_UT_SOURCES 1 /* sun, sky, L0 and pass 1's x */
/* pass p (0 to 2): x stored, the apply, y fetched, the total and the next x */
#define RF_UT_PASS 2
#define RF_UT_PASS_WORDS 4
#define RF_UT_ROWS 14   /* the field's values by row */
#define RF_UT_PACK 15   /* the scale, the texels into the pages (the first: zeroed) */
#define RF_UT_STORE 16  /* the pages stored into the light pages */
#define RF_UT_TABLE 17  /* LIGHT's unit and scale words in the sun region */
#define RF_UT_TOTAL 18  /* the commands of the update, call to answer, summed */
#define RF_UT_NUMBER 19 /* updates finished before it: its row */
#define RF_UT_CODE 20   /* the last command's answer */
#define RF_UT_STEP 21   /* the last step */
#define RF_UT_PASSES 22 /* passes taken */
#define RF_UT_SCALE 23  /* float: the field's scale */
#define RF_UT_ZERO 24   /* the pages zeroed (the first update only), part of PACK */
#define RF_UT_WORDS 32

/* UPDATE's a0: the step */
#define RF_STEP_ALL 0u     /* the whole update (the applies by apply_pass) */
#define RF_STEP_SOURCES 1u /* the sources, through pass 1's x stored */
#define RF_STEP_PASS 2u    /* y fetched, added; the next x stored while passes remain */
#define RF_STEP_FINISH 3u  /* the field, its pages, the table words */

/* Codes after frame.h's */
#define RF_NO_APPLY 11u /* built without RF_APPLY: the passes need the host's apply */
/* the words or regions do not hold the update, or a step out of turn */
#define RF_BAD_UPDATE 12u
#define RF_COPY 13u /* a fetch or a store did not finish */

/* LIGHT's table words the update writes (kernels/shade.h) */
#define RF_LT_FIELD_UNIT 652u
#define RF_LT_FIELD_SCALE 655u

#ifdef RF_APPLY
#define RF_HANDLER_APPLY 7u /* kernels: APPLY, y = M x */
#endif

/* UPDATE (step `command->a0`): 0, or the code to report. */
uint32_t rf_update(const km_command_t* command);
