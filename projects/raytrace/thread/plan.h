/* The frame thread's words, regions, tables and codes, shared with planner.py, which
 * checks the kernels' numbers here against kernels/raytrace.h. */
#pragma once

/* Kernels (kernels/raytrace.h) */
#define RP_HANDLER_TREELETS 5u
#define RP_BANK_SHIFT 8u
#define RP_DIR_BYTES 64u
#define RP_DIR_FIRST 0u
#define RP_DIR_COUNT 1u
#define RP_DIR_COST 2u
#define RP_COST_HEAD 16u
#define RP_COST_MAX 192u
#define RP_COST_BYTES 1600u /* 4 x (head + 2 x max): first, count; cycles; hits */

/* Sizes */
#define RP_TRACES 4u      /* primary, shadow, bounce, second shadow */
#define RP_CHUNKS 8u      /* chunk records a trace may take */
#define RP_CORES_MAX 128u /* cores of the group */
#define RP_PACKETS_MAX 32768u
#define RP_COMMANDS_MAX 32u /* records of a frame in the table */
#define RP_REC_WORDS 13u    /* kind, hdr, group, a0 to a9 */
#define RP_KIND_FIRST 256u  /* kind flag: the cold frame only (LOAD) */

/* How a trace's packets are cut into its chunk records */
#define RP_MODE_HOST 0u   /* as the host cuts them: full records, then the rest */
#define RP_MODE_EQUAL 1u  /* equal packets */
#define RP_MODE_COUNTS 2u /* equal rays: the previous trace's hits */
#define RP_MODE_TRACE 3u  /* equal cost: the previous trace's cycles */
#define RP_MODE_FRAME 4u  /* the previous frame's cycles (cold frame: below) */

/* Words */
#define RP_W_MODE 0u
#define RP_W_FRAMES 1u
#define RP_W_WORKERS 2u
#define RP_W_BANK 3u
#define RP_W_PACKETS 4u
#define RP_W_CAP 5u      /* packets a core may take in one record */
#define RP_W_COMMANDS 6u /* records of a frame in the table */
#define RP_WORDS 7u

/* Regions */
#define RP_R_TABLE 0u
#define RP_R_DIRS 1u /* (trace, chunk, core) directory entries */
#define RP_R_COST 2u /* (trace, chunk, core) cost blocks */
#define RP_R_WORK 3u
#define RP_R_LOG 4u
#define RP_REGIONS 5u

/* The table: records (kind 0: as given; 1 + t: trace t, its chunks by the plan), the
 * owners' directory (first, count per core), the profile pages per trace and chunk. */
#define RP_T_BASE (RP_COMMANDS_MAX * RP_REC_WORDS)
#define RP_T_PROF (RP_T_BASE + 2u * RP_CORES_MAX)
#define RP_TABLE_WORDS (RP_T_PROF + RP_TRACES * RP_CHUNKS)

/* The log, per frame: frame ticks, then per trace plan ticks and chunks, then the
 * ticks of each record in submit order. */
#define RP_L_FRAME 0u
#define RP_L_PLAN 1u
#define RP_L_CHUNKS 5u
#define RP_L_SUBMITTED 9u
#define RP_L_SUB 10u /* ticks of the plans' gather, prediction, cut, entries */
#define RP_L_COMMAND 16u
#define RP_LOG_WORDS 128u /* per frame */
#define RP_FRAMES_MAX 64u

#define RP_DIRS_BYTES (RP_TRACES * RP_CHUNKS * RP_CORES_MAX * RP_DIR_BYTES)
#define RP_COST_REGION_BYTES (RP_TRACES * RP_CHUNKS * RP_CORES_MAX * RP_COST_BYTES)
#define RP_LOG_BYTES (RP_FRAMES_MAX * RP_LOG_WORDS * 4u)

/* Result codes */
#define RP_OK 0u
#define RP_BAD_ARGS 1u
#define RP_RING 2u
#define RP_COMMAND 3u
#define RP_PLAN 4u
#define RP_OUT_WORDS 4u /* frames done, failing record, its event kind, its aux */
