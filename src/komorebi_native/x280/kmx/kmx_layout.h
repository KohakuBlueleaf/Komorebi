/* The layout an X280 thread shares with the host: its block in the work cluster's
 * local memory (LIM), the argument block, the region table and the result block. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#define KMX_STATIC_ASSERT(c, msg) static_assert(c, msg)
#else
#define KMX_STATIC_ASSERT(c, msg) _Static_assert(c, msg)
#endif

#define KMX_MAGIC 0x31786D6Bu /* "kmx1" */
#define KMX_VERSION 4u
#define KMX_REGIONS_MAX 16u
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
#define KMX_LIM_BYTES 0x540u
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
    uint32_t reserved;
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
    uint32_t pull[4]; /* kmx_pull_run: units, per pull, mode | log region << 8, 0 */
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

/* The command ring as the thread's cluster addresses it: command slots, the kernel's
 * consumed word, the events, the read-count line. Depths: powers of two. */
#define KMX_RING_DEPTH_MAX 1024u
/* flags: the ring is in cached memory, sweep on each flush or invalidate */
#define KMX_RING_CACHED 1u
/* flags: cmd and read are NoC addresses in the tile reserved[0] names (x | y << 16),
 * inside one 2 MiB window the start-up opens; consumed and events are local. */
#define KMX_RING_REMOTE 2u
#define KMX_RING_WINDOW 16u /* the window slot it uses */

typedef struct {
    uint64_t cmd;      /* command_depth 64-byte slots, 64-byte aligned */
    uint64_t consumed; /* a 32-bit word the kernel writes */
    uint64_t events;   /* event_depth 16-byte events (km_event_t) */
    uint64_t read;     /* a 64-byte line: word 0 the number of events the thread took */
    uint32_t command_depth;
    uint32_t event_depth;
    uint32_t flags; /* KMX_RING_* */
    uint32_t reserved[5];
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
