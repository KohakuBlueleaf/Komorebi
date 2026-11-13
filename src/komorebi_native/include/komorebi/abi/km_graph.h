/* Komorebi ABI: the graph a host uploads to the dispatcher. Plain C, usable from C
   and C++; builds on km_stream.h (micro-ops, command templates, error reasons). */
#ifndef KOMOREBI_ABI_KM_GRAPH_H
#define KOMOREBI_ABI_KM_GRAPH_H

#include <stddef.h>

#include "komorebi/abi/km_stream.h"

#define KM_GRAPH_MAGIC 0x4B4D4752u /* "KMGR" */
#define KM_GRAPH_VERSION 1u
#define KM_GRAPH_MAX_NODES 256u
#define KM_GRAPH_MAX_EDGES 1024u
#define KM_GRAPH_MAX_PREDECESSORS 4u /* predecessors of one node */
#define KM_GRAPH_MAX_TEMPLATES 64u
#define KM_GRAPH_MAX_OPS 4096u   /* micro-ops of all CPU slices together */
#define KM_GRAPH_MAX_STEPS 4096u /* largest step_limit */
#define KM_GRAPH_BINDINGS 16u    /* values copied into registers 0.. at each run */
#define KM_GRAPH_STEP 16u        /* nodes taken from the ready ring per step */
#define KM_GRAPH_STEP_OPS 64u /* a step ends after a CPU node once it ran this many */
#define KM_GRAPH_MAX_ACTIVE 2u
#define KM_GRAPH_MAX_KEPT 8u
/* Header flag: the graph is kept after it ends. It runs in place, so the host does not
   reuse its range of the upload region while it is kept; nothing is evicted. */
#define KM_GRAPH_KEEP 1u
#define KM_GRAPH_NONE 0xFFu      /* no register: skip, result, again register */
#define KM_GRAPH_NO_NODE 0xFFFFu /* error not tied to a node */

/* Node kinds. */
#define KM_NODE_COMMAND 1u /* submit one command template */
#define KM_NODE_CPU 2u     /* run a slice of compute micro-ops in the engine */
#define KM_NODE_JOIN 3u    /* no work */

/* KM_HANDLER_RUN for a graph: a2 = KM_GRAPH_RUN_UPLOAD (a0 offset, a1 bytes) or
   KM_GRAPH_RUN_KEPT (a0 = req of a kept one); a3 = binding offset, a4 = value count. */
#define KM_GRAPH_RUN_UPLOAD 1u
#define KM_GRAPH_RUN_KEPT 2u

/* Events: KM_EVENT_GRAPH: a round ended and another starts, auxiliary = rounds done.
   KM_EVENT_GRAPH_END: auxiliary = register 0 (low 24 bits), ticks = register 1. */
#define KM_EVENT_GRAPH 7u
#define KM_EVENT_GRAPH_END 8u
#define KM_EVENT_GRAPH_ERROR 9u /* auxiliary = reason | node << 8 */

/* Error reasons beyond the stream's KM_STREAM_ERROR_* (1 to 11, same meaning). */
#define KM_STREAM_ERROR_NODE 12u      /* bad kind, flags, unused field or slice */
#define KM_STREAM_ERROR_EDGE 13u      /* bad successor list or predecessor count */
#define KM_STREAM_ERROR_RUNS 14u      /* `again` still set after max_runs rounds */
#define KM_STREAM_ERROR_KEPT_FULL 15u /* the keep bit while 8 graphs are kept */

/* KM_HANDLER_CANCEL with the req of a kept graph: a running one is cancelled (reason
   10) and stays kept; one not running is dropped and its range freed. */

typedef struct {
    uint32_t magic, version;
    uint32_t node_count, edge_count, op_count, command_count;
    uint32_t req; /* request id of the graph's events (word 6, as a stream's) */
    uint32_t again_register;        /* KM_GRAPH_NONE: one round */
    uint32_t max_runs;              /* rounds at most, the first included */
    uint32_t deadline_microseconds; /* for each command; 0 is the kernel's default */
    uint32_t step_limit; /* micro-ops one CPU node may run, 1 to KM_GRAPH_MAX_STEPS */
    uint32_t flags;      /* KM_GRAPH_KEEP; the other bits are 0 */
    uint32_t pad[4];
} km_graph_t;

typedef struct {
    uint8_t kind, flags, predecessor_count, skip_register; /* flags are 0 */
    uint16_t arg;                  /* COMMAND: template; CPU: first micro-op */
    uint16_t op_count;             /* CPU: micro-ops in the slice */
    uint8_t result_register, pad0; /* COMMAND: register for its result word */
    uint16_t successor_offset, successor_count; /* successor list in the edge table */
    uint16_t pad1;
} km_node_t;

KM_STATIC_ASSERT(sizeof(km_graph_t) == 64, "the header is one 64-byte line");
KM_STATIC_ASSERT(sizeof(km_node_t) == 16, "a node is 16 bytes");
KM_STATIC_ASSERT(offsetof(km_graph_t, req) == 24, "req is header word 6");
KM_STATIC_ASSERT(offsetof(km_graph_t, step_limit) == 40, "header words");
KM_STATIC_ASSERT(offsetof(km_graph_t, flags) == 44, "header words");
KM_STATIC_ASSERT(offsetof(km_node_t, result_register) == 8, "node fields");
KM_STATIC_ASSERT(offsetof(km_node_t, successor_offset) == 10, "node fields");
KM_STATIC_ASSERT(KM_GRAPH_MAX_NODES <= KM_GRAPH_NO_NODE, "node numbers fit 16 bits");

#endif /* KOMOREBI_ABI_KM_GRAPH_H */
