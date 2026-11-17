/* The layout an X280 thread shares with the host: its block in the work cluster's
 * local memory (LIM), the argument block, the region table and the result block. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "komorebi/abi/km_abi.h"

#if defined(__cplusplus)
#define KMX_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define KMX_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define KMX_MAGIC 0x31786D6Bu /* "kmx1" */
#define KMX_VERSION 5u
#define KMX_REGIONS_MAX 16u
/* The DRAM bank of work cluster c (1 to 3), where its regions live. */
#define KMX_BANK_OF_CLUSTER(c) ((c) == 1u ? 6u : 7u)
#define KMX_WORDS_MAX 64u              /* the program's own argument words */
#define KMX_RESULT_WORDS_MAX 48u       /* words a thread returns */
#define KMX_DEADLINE_MAX 3000000000ull /* 60 s of the 50 MHz timer */

/* LIM layout, in bytes from the cluster's available LIM base. The host writes the
 * argument block and the stop line as whole 64-byte lines; the rest is the thread's. */
#define KMX_LIM_ARGS 0x000u   /* kmx_args (host, before the start) */
#define KMX_LIM_STOP 0x240u   /* line from the host: word 0 = KMX_STOP_WORD */
#define KMX_LIM_RESULT 0x280u /* kmx_result_block (the thread) */
#define KMX_LIM_CORES 0x380u  /* core h's spawn line at + 64 * h (KMX_SPAWN_*) */
/* kmx_ring_descriptor (host, before the start); zero: no ring */
#define KMX_LIM_RING 0x480u
#define KMX_LIM_PEERS 0x4C0u /* kmx_peers (host, before the start) */
#define KMX_LIM_PORTS 0x500u /* kmx_ports (host, before the start) */
#define KMX_LIM_RANKS 0x540u /* kmx_ranks_line (host, before the start) */
#define KMX_LIM_BYTES 0x580u
#define KMX_STOP_WORD 0x53544F50u /* "STOP" */

/* A spawn line's 64-bit words: the call's number, fn and arg (core 0 writes them), the
 * last number core h finished (core h). Core 0's own line: word 0 ends the service. */
#define KMX_SPAWN_GEN 0u
#define KMX_SPAWN_FUNCTION 1u
#define KMX_SPAWN_ARG 2u
#define KMX_SPAWN_DONE 3u
#define KMX_SPAWN_EXIT 0u

/* Argument flags. KEEP_PREFETCH: leave the L2 prefetchers as they are instead of the
 * documented values (they keep either until a card reset). */
#define KMX_FLAG_KEEP_PREFETCH 1u

/* Argument block offsets, in bytes. */
#define KMX_ARGS_REGIONS 0x040u
#define KMX_ARGS_WORDS 0x140u
#define KMX_ARGS_BYTES 0x240u

/* One region of the cluster's DRAM bank: its bank address (a multiple of 64) and
 * size. The thread sees it at X280_LOCAL_DDR_TILE_BASE + offset. */
typedef struct {
    uint64_t offset;
    uint64_t bytes;
} kmx_region_t;

typedef struct {
    uint64_t run_value;   /* this run's value: the block is stale otherwise */
    uint32_t magic;       /* KMX_MAGIC */
    uint32_t version;     /* KMX_VERSION */
    uint64_t deadline;    /* timer ticks the whole thread may take, 1 to the maximum */
    uint32_t nregions;    /* entries of region[] in use */
    uint32_t nwords;      /* entries of words[] in use */
    uint64_t check;       /* bank address of the check line (the bank check) */
    uint64_t check_value; /* what the host wrote at the start of that line */
    uint64_t flags;       /* KMX_FLAG_* */
    uint32_t cluster;     /* the work cluster the thread runs on, 1 to 3 */
    uint32_t island;      /* its island | islands << 16; 0: island 0 of one */
    kmx_region_t region[KMX_REGIONS_MAX];
    uint32_t words[KMX_WORDS_MAX];
} kmx_args;

/* The clusters running the same image in one program (kmx_parallel_all, kmx_push).
 * Tiles are NoC coordinates, x | y << 16, indexed by cluster; lead 0: no others. */
typedef struct {
    uint32_t nonce;   /* this load's value: lines left from another load differ */
    uint32_t lead;    /* the cluster that calls kmx_parallel_all */
    uint32_t mask;    /* bit c: cluster c runs this image */
    uint32_t all;     /* LIM offset of the kmx_parallel_all lines, the same on each */
    uint32_t tile[4]; /* [c]: cluster c's tile */
    uint32_t bank[4]; /* [c]: the DRAM tile of cluster c's bank */
    uint32_t reserved[4];
} kmx_peers;

/* The kmx_parallel_all lines at `all`, outside the host's block: the lead's call
 * (32-bit words KMX_CALL_*), in the lead's LIM cluster c's done line at + 64 * c. */
#define KMX_ALL_CALL 0x00u
#define KMX_ALL_DONE 0x00u
#define KMX_ALL_BYTES 0x100u
#define KMX_CALL_NONCE 0u
#define KMX_CALL_GEN 1u      /* the call's number; done lines: the last one finished */
#define KMX_CALL_FUNCTION 2u /* 64 bits */
#define KMX_CALL_ARG 4u      /* 64 bits */
#define KMX_CALL_EXIT 6u
/* cluster c's LIM through window KMX_PEER_WINDOW + c - 1 */
#define KMX_PEER_WINDOW 17u
#define KMX_PUSH_WINDOW 20u /* core h pushes through window KMX_PUSH_WINDOW + h */

/* X280 ports (at 0: none). The host writes the line, and before any start the area
 * at `at`: each core's tile, X280 core and mailbox line (KM_BOX_*, the core's). */
typedef struct {
    uint32_t at;      /* LIM offset of the area, a multiple of 64 */
    uint32_t control; /* the cores' control block address in L1 */
    uint32_t cores;   /* entries of the tables, 1 to KMX_PORTS_CORES */
    uint32_t flags;   /* KMX_PORTS_* */
    uint32_t state;   /* kmx_port_run's: n << 1 while run n serves, | 1 once it ended */
    uint32_t next;    /* kmx_pull_run: the cluster's counter, zeroed by the lead */
    uint32_t lost;    /* kmx_pull_run: ranges a full log could not keep */
    uint32_t pull[4]; /* kmx_pull_run: units, per, mode | log << 8, pick is pick_word */
    uint64_t pick, arg; /* kmx_pull_run: the pick function (0: the counter), its arg */
} kmx_ports;

#define KMX_PORTS_CORES 128u /* table entries: KM_CORES_MAX */
#define KMX_PORTS_NOC1 1u    /* flags: the windows onto the cores use NoC 1 */
#define KMX_PORTS_TILES 0x0u /* + 4 i: core i's tile, x | y << 16 */
/* + i: cluster << 2 | X280 core serving core i; 0: none */
#define KMX_PORTS_OWNER 0x200u
#define KMX_PORTS_BOX 0x280u /* + 64 i: core i's mailbox line */
#define KMX_PORTS_BYTES (KMX_PORTS_BOX + 64u * KMX_PORTS_CORES)
/* core i's control block through window KMX_PORT_WINDOW + i */
#define KMX_PORT_WINDOW 24u

/* X280 ranks of a dynamic run (at 0: none). The host writes the line, and before any
 * start the area at `at`: each core's tile, the cores' credit lines, inbox places. */
typedef struct {
    uint32_t at;    /* LIM offset of the area, a multiple of 64 */
    uint32_t ranks; /* X280 cores 0 to ranks - 1 are ranks, 1 to KM_X280_RANKS_MAX */
    uint32_t cores; /* entries of the tile table, the grid's cores */
    uint32_t queue; /* bank address of X280 rank 0's queue; rank r's r queues on */
    uint32_t cap;   /* messages a queue, a power of two */
    uint32_t message_bytes; /* 64 to 256, a multiple of 64 */
    uint32_t quota;         /* sends each X280 rank starts a run with */
    uint32_t counters; /* the run group's last `counters` cores are counter tiles */
    uint32_t idle;     /* ticks with nothing to take before its line says idle */
    uint32_t inbox;    /* L1 address of every core's inbox */
    uint32_t inbox_messages; /* messages an inbox holds, a power of two */
    uint32_t lines;          /* the dispatcher's LIM address of core 0's status line */
    uint32_t lines_tile;     /* the dispatcher's tile, x | y << 16 */
    uint32_t wait;           /* ticks a wait for an inbox's room may take */
    uint32_t far;     /* over islands: LIM offset of the far area (KMX_FAR_*); 0 none */
    uint32_t far_noc; /* that area's address as a link core reads it over the NoC */
} kmx_ranks_line;

#define KMX_RANKS_TILES 0x000u   /* + 4 i: core i's tile, x | y << 16 */
#define KMX_RANKS_CREDITS 0x200u /* + 16 i: core i's credit line (KM_X280_CREDIT_*) */
#define KMX_RANKS_PLACES 0xA00u  /* + 4 i: core i's next inbox position, an amoadd */
#define KMX_RANKS_BYTES 0xC00u
/* X280 rank r's window onto the dispatcher's status lines (non-posted) */
#define KMX_RANKS_WINDOW 152u
/* The far area (over islands, host-written): this cluster's tile (x | y << 16), the
 * link area's L1 address, the islands; then the sends' counts and staged messages */
#define KMX_FAR_HEAD 0x000u
/* + 16 i: island i's first flat rank, its ranks, the tile of island 0's link core 0
 * toward it (0: island 0), the ring's L1 address there (the rank's, 0 until read) */
#define KMX_FAR_ISLANDS 0x040u
#define KMX_FAR_SENT 0x140u     /* + 4 r: rank r's ring lines written this load */
#define KMX_FAR_CONFIRMS 0x180u /* + 16 (r * KM_X280_RING_LINES + k): place k's */
#define KMX_FAR_STAGE 0x380u    /* + 256 (r * KM_X280_RING_LINES + k): place k's */
#define KMX_FAR_BYTES 0x2380u
/* island i's window onto island 0's link core toward it (posted) */
#define KMX_FAR_WINDOW 156u

/* The command ring as the thread's cluster addresses it: command slots, the kernel's
 * consumed word, the events, the read-count line. Depths: powers of two. */
#define KMX_RING_DEPTH_MAX 1024u
/* flags: the ring is in cached memory, sweep on each flush or invalidate */
#define KMX_RING_CACHED 1u
/* flags: cmd and read are NoC addresses in the tile reserved[0] names (x | y << 16),
 * inside one 2 MiB window the start-up opens; consumed and events are local. */
#define KMX_RING_REMOTE 2u
#define KMX_RING_WINDOW 16u /* the window slot it uses */
/* Commands the dispatcher hands the thread (kmx_program): KMX_RING_IN_DEPTH lines of
 * 64 bytes this many bytes after the events; line n (from 1) in slot n % depth. */
#define KMX_RING_IN_AFTER(event_depth) (((event_depth) * 16u + 63u) & ~63u)
#define KMX_RING_IN_DEPTH 8u
/* The run line of X280 ranks (KM_X280_RUN_*): this many bytes after those lines. */
#define KMX_RING_RUN_AFTER (KMX_RING_IN_DEPTH * 64u)
/* The thread's answer to line a0, a1 the word kmx_program returned: a command of the
 * dispatcher's own, which sends no event for it. */
#define KMX_HANDLER_ANSWER 243u

typedef struct {
    uint64_t cmd;      /* command_depth 64-byte slots, 64-byte aligned */
    uint64_t consumed; /* a 32-bit word the kernel writes */
    uint64_t events;   /* event_depth 16-byte events (km_event_t) */
    uint64_t read;     /* a 64-byte line: word 0 the number of events the thread took */
    uint32_t command_depth;
    uint32_t event_depth;
    uint32_t flags; /* KMX_RING_* */
    uint32_t reserved[1];
    uint32_t buffer; /* kmx_fetch, kmx_store: the cores' L1 for a copy (0: none) */
    /* kmx_load: field p's record in DRAM at records + p * record_step, p < programs
     * (programs 0: the run's groups do not switch) */
    uint32_t records, record_step, programs;
} kmx_ring_descriptor;

/* Return codes of the calls that can fail. */
#define KMX_OK 0
#define KMX_FULL 1    /* no free command slot, or kmx_wait has set aside all it can */
#define KMX_STOPPED 2 /* a stop was asked for, or the thread has no ring */
#define KMX_LATE 3    /* the deadline passed */
#define KMX_NONE 4    /* kmx_poll: no event yet; kmx_serve: this cluster leads */
#define KMX_BAD 5     /* an argument out of range: a core, cluster, region or size */

/* An event of a command the thread submitted: its handle, kind (KM_EVENT_*), auxiliary
 * and ticks (for KM_FLAG_RESULT commands: the group leader's result word). */
typedef struct {
    uint32_t handle, kind, auxiliary, ticks;
} kmx_event;

/* The kernel's wake (handle 0): room again after a rejection with auxiliary 11.
 * kmx_wait passes it over; kmx_poll returns it. */
#define KMX_EVENT_WAKE 0x12u
/* The rejection's auxiliary for a command that cannot go now (the dispatcher's 11). */
#define KMX_REJECT_BUSY 11u

/* States, in the result block. */
#define KMX_STATE_BOOT 1u    /* start-up, before kmx_main */
#define KMX_STATE_RUNNING 2u /* in kmx_main */
#define KMX_STATE_DONE 3u    /* kmx_main returned, no library error */
#define KMX_STATE_FAILED 4u  /* a library error: see error */

/* Library errors, in the result block (the thread's own code is separate). */
#define KMX_ERROR_NONE 0u
#define KMX_ERROR_ARGS 1u    /* the argument block is not this run's, or out of range */
#define KMX_ERROR_BANK 2u    /* the check line does not hold check_value: wrong bank */
#define KMX_ERROR_LATE 3u    /* a wait reached the run's deadline */
#define KMX_ERROR_STOPPED 4u /* the host or the runtime asked for a stop */
#define KMX_ERROR_PARALLEL 5u /* kmx_parallel called from a core other than 0 */
#define KMX_ERROR_RESULT 6u   /* kmx_result with more than KMX_RESULT_WORDS_MAX words */

typedef struct {
    uint32_t state;   /* KMX_STATE_* */
    uint32_t error;   /* KMX_ERROR_* (the first one) */
    uint32_t code;    /* the thread's code: kmx_result's, else kmx_main's return */
    uint32_t n;       /* result words */
    uint32_t where;   /* the last kmx_progress marker */
    uint32_t t_where; /* timer ticks from the start to that marker */
    uint32_t ticks;   /* timer ticks from the start to the end of kmx_main */
    uint32_t reserved[9];
    uint32_t words[KMX_RESULT_WORDS_MAX];
} kmx_result_block;

KMX_STATIC_ASSERT(offsetof(kmx_args, deadline) == 16, "kmx_args.deadline at 16");
KMX_STATIC_ASSERT(offsetof(kmx_args, check) == 32, "kmx_args.check at 32");
KMX_STATIC_ASSERT(offsetof(kmx_args, check_value) == 40, "check_value at 40");
KMX_STATIC_ASSERT(offsetof(kmx_args, flags) == 48, "kmx_args.flags at 48");
KMX_STATIC_ASSERT(offsetof(kmx_args, cluster) == 56, "kmx_args.cluster at 56");
KMX_STATIC_ASSERT(offsetof(kmx_args, region) == KMX_ARGS_REGIONS, "region table");
KMX_STATIC_ASSERT(offsetof(kmx_args, words) == KMX_ARGS_WORDS, "argument words");
KMX_STATIC_ASSERT(sizeof(kmx_args) == KMX_ARGS_BYTES, "kmx_args size");
KMX_STATIC_ASSERT(KMX_LIM_ARGS + KMX_ARGS_BYTES <= KMX_LIM_STOP, "LIM: args");
KMX_STATIC_ASSERT(sizeof(kmx_result_block) == 256, "kmx_result_block size");
KMX_STATIC_ASSERT(KMX_LIM_RESULT + 256 <= KMX_LIM_CORES, "LIM: result");
KMX_STATIC_ASSERT(KMX_LIM_CORES + 4 * 64 <= KMX_LIM_RING, "LIM: spawn lines");
KMX_STATIC_ASSERT(sizeof(kmx_peers) == 64, "kmx_peers is one line");
KMX_STATIC_ASSERT(KMX_LIM_PEERS + 64 <= KMX_LIM_BYTES, "LIM: peers");
KMX_STATIC_ASSERT(sizeof(kmx_ring_descriptor) == 64, "kmx_ring_descriptor is one line");
KMX_STATIC_ASSERT(KMX_LIM_RING + 64 <= KMX_LIM_BYTES, "LIM: ring");
KMX_STATIC_ASSERT(sizeof(kmx_ports) == 64, "kmx_ports is one line");
KMX_STATIC_ASSERT(KMX_LIM_PORTS + 64 <= KMX_LIM_BYTES, "LIM: ports");
KMX_STATIC_ASSERT(KMX_PORTS_OWNER == 4u * KMX_PORTS_CORES, "ports: tiles, then owners");
KMX_STATIC_ASSERT(KMX_PORTS_BOX >= KMX_PORTS_OWNER + KMX_PORTS_CORES, "ports: owners");
KMX_STATIC_ASSERT(
    KMX_PORT_WINDOW + KMX_PORTS_CORES <= 192u, "ports: below the host's windows"
);
KMX_STATIC_ASSERT(sizeof(kmx_ranks_line) == 64, "kmx_ranks_line is one line");
KMX_STATIC_ASSERT(KMX_LIM_RANKS + 64 <= KMX_LIM_BYTES, "LIM: ranks");
KMX_STATIC_ASSERT(
    KMX_RANKS_CREDITS == 4u * KMX_PORTS_CORES &&
        KMX_RANKS_PLACES ==
            KMX_RANKS_CREDITS + KM_X280_CREDIT_BYTES * KMX_PORTS_CORES &&
        KMX_RANKS_BYTES == KMX_RANKS_PLACES + 4u * KMX_PORTS_CORES,
    "ranks: tiles, credit lines, places"
);
KMX_STATIC_ASSERT(
    KMX_PORT_WINDOW + KMX_PORTS_CORES <= KMX_RANKS_WINDOW &&
        KMX_RANKS_WINDOW + KM_X280_RANKS_MAX <= 192u,
    "ranks: windows after the ports', below the host's"
);
KMX_STATIC_ASSERT(
    KMX_FAR_ISLANDS + 16u * KM_ISLANDS_MAX <= KMX_FAR_SENT &&
        KMX_FAR_SENT + 4u * KM_X280_RANKS_MAX <= KMX_FAR_CONFIRMS &&
        KMX_FAR_CONFIRMS + 16u * KM_X280_RANKS_MAX * KM_X280_RING_LINES <=
            KMX_FAR_STAGE &&
        KMX_FAR_STAGE + 256u * KM_X280_RANKS_MAX * KM_X280_RING_LINES ==
            KMX_FAR_BYTES &&
        KMX_RANKS_WINDOW + KM_X280_RANKS_MAX <= KMX_FAR_WINDOW &&
        KMX_FAR_WINDOW + KM_ISLANDS_MAX <= 192u,
    "far: islands, counts, confirms, staged messages; windows after the ranks'"
);
