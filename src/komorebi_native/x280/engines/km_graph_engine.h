/* The graph executor: plain C with no kernel internals. The same file runs in the
   dispatcher and on the host (CPU tests, the host-fed executor). */
#ifndef KOMOREBI_X280_KM_GRAPH_ENGINE_H
#define KOMOREBI_X280_KM_GRAPH_ENGINE_H

#include <stdint.h>

#include "komorebi/abi/km_graph.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* What a submit function returns. */
#define KMG_SUBMITTED 0u
#define KMG_BUSY 1u /* no slot, an overlapping exclusive group, or a full queue */

/* What kmg_step returns; KMG_RUNNING, KMG_END and KMG_ERROR are also the phase. */
#define KMG_RUNNING 0u
#define KMG_ROUND 1u /* a round ended and the next one started */
#define KMG_END 2u
#define KMG_ERROR 3u

/* Node states. */
#define KMG_WAITING 0u /* predecessors not done */
#define KMG_READY 1u   /* in the ready ring */
#define KMG_FLIGHT 2u  /* a command submitted, its completion not seen */
#define KMG_DONE 3u

/* Submits `command` for `node` (sequence, landing and slot bits are 0; a queued command
   keeps KM_FLAG_QUEUE and first 0). Returns KMG_SUBMITTED or KMG_BUSY. */
typedef uint32_t (*kmg_submit_function)(
    void* context, uint32_t node, const km_command_t* command
);

/* All engine state, in a block the caller gives. The first 20 words are fixed: the
   registers, then runs, phase, error reason and error node. */
typedef struct {
    uint32_t registers[KM_STREAM_REGISTERS];
    uint32_t runs; /* the round being run, from 1 */
    uint32_t phase;
    uint32_t error_reason, error_node;
    uint32_t done, in_flight, head, ready_n, core_count;
    const km_graph_t* hdr;
    const km_node_t* nodes;
    const uint16_t* edges;
    const km_uop_t* ops;
    const km_template_t* template;
    uint8_t left[KM_GRAPH_MAX_NODES];
    uint8_t status[KM_GRAPH_MAX_NODES];
    uint16_t ready[KM_GRAPH_MAX_NODES];
} kmg_state_t;

/* sizeof(kmg_state_t), for hosts that allocate the block. */
uint32_t kmg_state_bytes(void);

/* 0 if the upload is a valid graph, else reason | node << 8 (node KM_GRAPH_NO_NODE
   when the fault is not at a node). The first fault wins. */
uint32_t kmg_check(const void* upload, uint32_t bytes);

/* Starts a checked upload, which stays in place and unchanged while the graph runs.
   Returns 0, or KM_STREAM_ERROR_HEADER for more than KM_GRAPH_BINDINGS bindings. */
uint32_t kmg_start(
    kmg_state_t* state,
    const void* upload,
    const uint32_t* bindings,
    uint32_t n_bindings,
    uint32_t core_count
);

/* One step: at most KM_GRAPH_STEP ready nodes. */
uint32_t kmg_step(kmg_state_t* state, kmg_submit_function submit, void* context);

/* The command of `node` completed with `result`. Returns 0, or 1 if the graph is not
   running or the node has no command in flight (nothing changes). */
uint32_t kmg_complete(kmg_state_t* state, uint32_t node, uint32_t result);

/* The command of `node` was submitted, then rejected for now (no slot, a group busy):
   the node goes back to the ready ring's tail. Returns 0, or 1 as kmg_complete. */
uint32_t kmg_retry(kmg_state_t* state, uint32_t node);

/* Ends a running graph with an error from outside: a deadline, a cancel. */
void kmg_stop(kmg_state_t* state, uint32_t reason, uint32_t node);

#if defined(__cplusplus)
}
#endif

#endif /* KOMOREBI_X280_KM_GRAPH_ENGINE_H */
