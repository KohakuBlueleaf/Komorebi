/* Result block in LIM: the kernel's state, where it is, and at the finish the error
 * command and the counters. The host reads it after stopping the kernel. */
#pragma once

#include "kernel.h"

static inline void result_word(kmd_t* dispatcher, uint32_t word, uint32_t value) {
    dispatcher->result[word] = value;
}

static inline void result_wide(kmd_t* dispatcher, uint32_t low, uint64_t value) {
    dispatcher->result[low] = (uint32_t)value;
    dispatcher->result[low + 1] = (uint32_t)(value >> 32);
}

/* Enters a loop: the where code, in the result block. */
static inline void result_where(kmd_t* dispatcher, uint32_t where) {
    dispatcher->where = where;
    dispatcher->result[KMD_R_WHERE] = where;
}

/* Commands an error with the slot, sequence, request and value it concerns. */
static inline void result_error(
    kmd_t* dispatcher,
    uint32_t error,
    uint32_t slot,
    uint32_t sequence,
    uint32_t request,
    uint32_t seen
) {
    dispatcher->error = error;
    dispatcher->result[KMD_R_SLOT] = slot;
    dispatcher->result[KMD_R_SEQUENCE] = sequence;
    dispatcher->result[KMD_R_REQ] = request;
    dispatcher->result[KMD_R_SEEN] = seen;
    dispatcher->result[KMD_R_ERROR] = error;
}

/* The counters and the final state. */
static inline void result_finish(kmd_t* dispatcher, uint64_t now) {
    result_word(
        dispatcher, KMD_R_HOST_COMMANDS, dispatcher->ring[KMD_SRC_HOST].next - 1u
    );
    result_word(dispatcher, KMD_R_COMMANDS, dispatcher->commands);
    result_word(dispatcher, KMD_R_EVENTS, dispatcher->ring[KMD_SRC_HOST].pushed);
    result_word(dispatcher, KMD_R_BUSY, dispatcher->busy);
    result_word(dispatcher, KMD_R_MAX_BUSY, dispatcher->max_busy);
    result_word(dispatcher, KMD_R_REJECTED, dispatcher->rejected);
    result_word(dispatcher, KMD_R_HELD, dispatcher->held);
    result_word(dispatcher, KMD_R_NO_ROOM, dispatcher->no_room);
    result_word(dispatcher, KMD_R_CHECKED, dispatcher->checked);
    result_word(dispatcher, KMD_R_BAD_LINES, dispatcher->bad_lines);
    result_word(dispatcher, KMD_R_BAD_FIRST, dispatcher->bad_first);
    result_word(dispatcher, KMD_R_RESULTS, dispatcher->results);
    result_word(dispatcher, KMD_R_RESULT, dispatcher->last_result);
    result_wide(
        dispatcher,
        KMD_R_TICKS_LO,
        dispatcher->first_command_mtime ? now - dispatcher->first_command_mtime : 0
    );
    result_wide(dispatcher, KMD_R_IDLE_LO, dispatcher->idle_passes);
    result_wide(dispatcher, KMD_R_FULL_LO, dispatcher->full_passes);
    result_wide(dispatcher, KMD_R_PASSES_LO, dispatcher->passes);
    const uint64_t last = (
        dispatcher->last_event_mtime > dispatcher->first_command_mtime
            ? dispatcher->last_event_mtime - dispatcher->first_command_mtime
            : 0
    );
    result_wide(dispatcher, KMD_R_LAST_LO, last);
    result_word(dispatcher, KMD_R_ERROR, dispatcher->error);
    x280_fence_ow();
    result_word(
        dispatcher, KMD_R_STATE, dispatcher->error ? KMD_STATE_FAILED : KMD_STATE_DONE
    );
}
