/* The thread of example 25, on work cluster 1 of each island. Island 0's sends
 * commands whose groups or threads are on island 1 and times them; island 1's
 * answers FOLD. kmx_program serves the host's commands on both. */
#include <stdint.h>

#include "kmx.h"

#define WAIT_TICKS (KMX_TICKS_PER_SECOND / 2u) /* a command's wait */
#define NO_RANK 0xFFFFFFFFu
#define BAD 0xFFFFFFFFu          /* a command went wrong */
#define WRONG_ISLAND 0xFFFFFFFEu /* kmx_island_of named another island */
#define FOLD_STEP 0x9E3779B1u

/* Submits `command` (again while the ring is full) and waits for its event. */
static int run(const km_command_t* command, kmx_event* event) {
    const uint64_t deadline = kmx_time() + WAIT_TICKS;
    uint32_t handle = 0;
    int status;
    do {
        status = kmx_submit(command, &handle);
    } while (status == KMX_FULL && kmx_time() < deadline);
    return status != KMX_OK ? status : kmx_wait(handle, deadline, event);
}

/* MAP (example 01's handler kMap) on flat ranks [first, first + count): rank r adds
 * input tiles 2 (first + r) and the next, stores output page first + r. */
static void map_command(
    km_command_t* command, uint32_t first, uint32_t count, uint32_t source, uint32_t out
) {
    kmx_command(command, 0, kMap, first, count);
    command->a0 = source;
    command->a1 = 2u * count; /* tiles: two a rank */
    command->a2 = out;
    command->a3 = first; /* first_page */
    command->a4 = 0;     /* slow_cycles */
    command->a5 = NO_RANK;
    command->a6 = 2u * first; /* first_tile */
}

/* MAPS: a0 first, a1 count, a2 rounds, a3 the input, a4 the output; one MAP after the
 * other. Its word: the ticks of all rounds, or BAD. */
static uint32_t maps(const km_command_t* command) {
    km_command_t map;
    kmx_event event;
    map_command(&map, command->a0, command->a1, command->a3, command->a4);
    const uint64_t started = kmx_time();
    for (uint32_t round = 0; round < command->a2; round++) {
        if (run(&map, &event) != KMX_OK || event.kind != KM_EVENT_DONE) {
            return BAD;
        }
    }
    return (uint32_t)(kmx_time() - started);
}

/* What island 1's FOLD answers for x: names the island that served it. */
static uint32_t fold(uint32_t x) {
    return x * FOLD_STEP + (uint32_t)kmx_island();
}

/* ASK: a0 a thread's number, a1 rounds, a2 the first x; FOLD to that thread for x,
 * x + 1, ... with the answer as the result. Its word: the ticks of all, or BAD. */
static uint32_t ask(const km_command_t* command) {
    const uint32_t number = command->a0;
    if (kmx_island_of(number) != 1) {
        return WRONG_ISLAND;
    }
    km_command_t question;
    kmx_event event;
    kmx_command(&question, 0, kHandlerFold, KM_GROUP_CLUSTER_BASE + number, 1);
    question.hdr |= KM_FLAG_RESULT << 8;
    const uint64_t started = kmx_time();
    for (uint32_t round = 0; round < command->a1; round++) {
        const uint32_t x = command->a2 + round;
        question.a0 = x;
        if (
            run(&question, &event) != KMX_OK || event.kind != KM_EVENT_DONE ||
            event.ticks != x * FOLD_STEP + 1u /* island 1 served it */
        ) {
            return BAD;
        }
    }
    return (uint32_t)(kmx_time() - started);
}

/* This thread's commands from now on have `microseconds` (KM_HANDLER_DEADLINE: no
 * event comes for it). */
static int set_deadline(uint32_t microseconds) {
    km_command_t command;
    kmx_command(&command, 0, KM_HANDLER_DEADLINE, 0, 0);
    command.a0 = microseconds;
    uint32_t handle = 0;
    return kmx_submit(&command, &handle);
}

/* HANG: a0 first, a1 count, a2 the hung rank of the group, a3 its deadline and a4 the
 * one after it (microseconds); example 01's HANG, its reader waiting for an arrival
 * that never comes. Its word: the event's kind << 8 | its auxiliary's low byte (a
 * part on another island that timed out: KM_EVENT_ISLAND + that island). */
static uint32_t hang(const km_command_t* command) {
    km_command_t hung;
    kmx_event event;
    kmx_command(&hung, 0, kHang, command->a0, command->a1);
    hung.a0 = command->a2;
    hung.a1 = 1u; /* kHangPort */
    uint32_t word = BAD;
    if (set_deadline(command->a3) == KMX_OK && run(&hung, &event) == KMX_OK) {
        word = event.kind << 8 | (event.auxiliary & 0xFFu);
    }
    return set_deadline(command->a4) == KMX_OK ? word : BAD;
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerMaps:
            return maps(command);
        case kHandlerAsk:
            return ask(command);
        case kHandlerFold:
            return fold(command->a0);
        case kHandlerHang:
            return hang(command);
    }
    return BAD;
}
