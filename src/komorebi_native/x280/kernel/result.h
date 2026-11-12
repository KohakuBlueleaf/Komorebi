/* Result block in LIM: the kernel's state, where it is, and at the finish the error
 * record and the counters. The host reads it after stopping the kernel. */
#pragma once

#include "kernel.h"

static inline void result_word(kmd_t* k, uint32_t word, uint32_t value) {
    k->result[word] = value;
}

static inline void result_wide(kmd_t* k, uint32_t lo, uint64_t value) {
    k->result[lo] = (uint32_t)value;
    k->result[lo + 1] = (uint32_t)(value >> 32);
}

/* Enters a loop: the where code, in the result block. */
static inline void result_where(kmd_t* k, uint32_t where) {
    k->where = where;
    k->result[KMD_R_WHERE] = where;
}

/* Records an error with the slot, sequence, request and value it concerns. */
static inline void result_error(
    kmd_t* k, uint32_t error, uint32_t slot, uint32_t seq, uint32_t req, uint32_t seen
) {
    k->error = error;
    k->result[KMD_R_SLOT] = slot;
    k->result[KMD_R_SEQ] = seq;
    k->result[KMD_R_REQ] = req;
    k->result[KMD_R_SEEN] = seen;
    k->result[KMD_R_ERROR] = error;
}

/* The counters and the final state. */
static inline void result_finish(kmd_t* k, uint64_t now) {
    result_word(k, KMD_R_COMMANDS, k->ring[KMD_SRC_HOST].next - 1u);
    result_word(k, KMD_R_RECORDS, k->records);
    result_word(k, KMD_R_EVENTS, k->ring[KMD_SRC_HOST].pushed);
    result_word(k, KMD_R_BUSY, k->busy);
    result_word(k, KMD_R_MAX_BUSY, k->max_busy);
    result_word(k, KMD_R_REFUSED, k->refused);
    result_word(k, KMD_R_HELD, k->held);
    result_word(k, KMD_R_NO_ROOM, k->no_room);
    result_word(k, KMD_R_CHECKED, k->checked);
    result_word(k, KMD_R_BAD_LINES, k->bad_lines);
    result_word(k, KMD_R_BAD_FIRST, k->bad_first);
    result_word(k, KMD_R_RESULTS, k->results);
    result_word(k, KMD_R_RESULT, k->last_result);
    result_wide(k, KMD_R_TICKS_LO, k->t_first ? now - k->t_first : 0);
    result_wide(k, KMD_R_IDLE_LO, k->idle_passes);
    result_wide(k, KMD_R_FULL_LO, k->full_passes);
    result_wide(k, KMD_R_PASSES_LO, k->passes);
    const uint64_t last = (
        k->t_last_event > k->t_first ? k->t_last_event - k->t_first : 0
    );
    result_wide(k, KMD_R_LAST_LO, last);
    result_word(k, KMD_R_ERROR, k->error);
    x280_fence_ow();
    result_word(k, KMD_R_STATE, k->error ? KMD_STATE_FAILED : KMD_STATE_DONE);
}
