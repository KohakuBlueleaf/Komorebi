/* Komorebi ABI: the graph a host uploads to the dispatch kernel. Plain C, usable from C
   and C++; builds on km_stream.h (micro-ops, record templates, error reasons). */
#ifndef KOMOREBI_ABI_KM_GRAPH_H
#define KOMOREBI_ABI_KM_GRAPH_H

#include <stddef.h>

#include "komorebi/abi/km_stream.h"

#define KM_GRAPH_MAGIC 0x4B4D4752u /* "KMGR" */
#define KM_GRAPH_VERSION 1u
#define KM_GRAPH_MAX_NODES 256u
#define KM_GRAPH_MAX_EDGES 1024u
#define KM_GRAPH_MAX_PRED 4u /* predecessors of one node */
#define KM_GRAPH_MAX_TEMPLATES 64u
#define KM_GRAPH_MAX_OPS 4096u   /* micro-ops of all CPU slices together */
#define KM_GRAPH_MAX_STEPS 4096u /* largest step_limit */
#define KM_GRAPH_BINDINGS 16u    /* values copied into registers 0.. at each run */
#define KM_GRAPH_STEP 16u        /* nodes taken from the ready ring per step */
#define KM_GRAPH_STEP_OPS 64u /* a step ends after a CPU node once it ran this many */
#define KM_GRAPH_MAX_ACTIVE 2u
#define KM_GRAPH_MAX_KEPT 8u
#define KM_GRAPH_NONE 0xFFu      /* no register: skip_reg, result_reg, again_reg */
#define KM_GRAPH_NO_NODE 0xFFFFu /* error not tied to a node */

/* Node kinds. */
#define KM_NODE_RECORD 1u /* issue one record template */
#define KM_NODE_CPU 2u    /* run a slice of compute micro-ops in the engine */
#define KM_NODE_JOIN 3u   /* no work */

/* KM_SEG_RUN for a graph: a2 = KM_GRUN_UPLOAD (a0 offset, a1 bytes) or KM_GRUN_KEPT
   (a0 = req of a kept graph); a3 = binding block offset, a4 = its value count. */
#define KM_GRUN_UPLOAD 1u
#define KM_GRUN_KEPT 2u

/* Events; `req` is the header's. KM_EV_GRAPH: a round ended and another starts, aux =
   rounds done. KM_EV_GRAPH_END: aux = register 0 (low 24 bits), ticks = register 1. */
#define KM_EV_GRAPH 7u
#define KM_EV_GRAPH_END 8u
#define KM_EV_GRAPH_ERROR 9u /* aux = reason | node << 8 */

/* Error reasons beyond the stream's KM_SERR_* (1 to 11, same meaning). */
#define KM_SERR_NODE 12u /* bad kind, flags, unused field or slice */
#define KM_SERR_EDGE 13u /* bad successor list or predecessor count */
#define KM_SERR_RUNS 14u /* `again` still set after max_runs rounds */

typedef struct {
    uint32_t magic, version;
    uint32_t n_nodes, n_edges, n_ops, n_records;
    uint32_t req;         /* request id of the graph's events (word 6, as a stream's) */
    uint32_t again_reg;   /* KM_GRAPH_NONE: one round */
    uint32_t max_runs;    /* rounds at most, the first included */
    uint32_t deadline_us; /* for each record; 0 is the kernel's default */
    uint32_t step_limit;  /* micro-ops one CPU node may run, 1 to KM_GRAPH_MAX_STEPS */
    uint32_t pad[5];
} km_graph_t;

typedef struct {
    uint8_t kind, flags, n_pred, skip_reg; /* flags are 0 */
    uint16_t arg;                          /* RECORD: template; CPU: first micro-op */
    uint16_t n_ops;                        /* CPU: micro-ops in the slice */
    uint8_t result_reg, pad0;              /* RECORD: register for its result word */
    uint16_t succ_off, succ_n;             /* successor list in the edge table */
    uint16_t pad1;
} km_node_t;

KM_STATIC_ASSERT(sizeof(km_graph_t) == 64, "the header is one 64-byte line");
KM_STATIC_ASSERT(sizeof(km_node_t) == 16, "a node is 16 bytes");
KM_STATIC_ASSERT(offsetof(km_graph_t, req) == 24, "req is header word 6");
KM_STATIC_ASSERT(offsetof(km_graph_t, step_limit) == 40, "header words");
KM_STATIC_ASSERT(offsetof(km_node_t, result_reg) == 8, "node fields");
KM_STATIC_ASSERT(offsetof(km_node_t, succ_off) == 10, "node fields");
KM_STATIC_ASSERT(KM_GRAPH_MAX_NODES <= KM_GRAPH_NO_NODE, "node numbers fit 16 bits");

#endif /* KOMOREBI_ABI_KM_GRAPH_H */
