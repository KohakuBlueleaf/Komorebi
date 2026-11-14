/* Inspect: core 0 reads cores' state, live words and last trace entries from their
 * control blocks through one window, a core a loop pass, into the inspect block. */
#pragma once

#include "completion.h"
#include "kernel.h"
#include "ring.h"

static inline volatile uint32_t* inspect_header(const kmd_t* dispatcher) {
    return (volatile uint32_t*)(dispatcher->lim + KMD_LIM_INSPECT);
}

/* Starts reading cores [first, first + count) (the group lies inside the grid): a
 * host's inspect answers `request`; a timeout's names slot `slot`, `sequence`. */
static inline void inspect_begin(
    kmd_t* dispatcher,
    uint32_t kind,
    uint32_t request,
    uint32_t first,
    uint32_t count,
    uint32_t slot,
    uint32_t sequence,
    uint64_t now
) {
    volatile uint32_t* header = inspect_header(dispatcher);
    header[KMD_I_MARK] = 0;
    x280_fence_ow(); /* the block reads as incomplete before any word changes */
    header[KMD_I_KIND] = kind;
    header[KMD_I_REQUEST] = request;
    header[KMD_I_FIRST] = first;
    header[KMD_I_COUNT] = count;
    header[KMD_I_DONE] = 0;
    header[KMD_I_SLOT] = slot;
    header[KMD_I_SEQUENCE] = sequence;
    header[KMD_I_START] = (uint32_t)now;
    header[KMD_I_TICKS] = 0;
    kmd_inspect_t* inspect = &dispatcher->inspect;
    inspect->active = 1;
    inspect->kind = kind;
    inspect->request = request;
    inspect->first = first;
    inspect->count = count;
    inspect->next = 0;
    inspect->leaving = 0;
    inspect->slot = slot;
    inspect->sequence = sequence;
    inspect->start = now;
}

/* `bytes` (a multiple of 8) by aligned 64-bit loads from the window into LIM. */
static inline void inspect_copy(
    volatile uint8_t* to, volatile const uint8_t* from, uint32_t bytes
) {
    volatile uint64_t* destination = (volatile uint64_t*)to;
    volatile const uint64_t* source = (volatile const uint64_t*)from;
    for (uint32_t word = 0; word < bytes / 8u; word++) {
        destination[word] = source[word];
    }
}

/* Whether [address, address + bytes) lies in a core's L1 above the firmware's part. */
static inline int inspect_in_l1(uint32_t address, uint32_t bytes) {
    return (
        address >= KMD_CONTROL_FLOOR && address <= KMD_L1_BYTES &&
        bytes <= KMD_L1_BYTES - address
    );
}

/* The inspect window aimed at core `core`'s L1, at the host's core table coordinates;
 * 0 for a core without them, a control block outside L1, or a window not programmed. */
static volatile const uint8_t* inspect_aim(kmd_t* dispatcher, uint32_t core) {
    const uint32_t coordinates = (
        ((volatile const uint32_t*)(dispatcher->lim + KMD_LIM_COORDS))[core]
    );
    if (
        coordinates == 0 ||
        !inspect_in_l1(dispatcher->control_address, KM_CONTROL_BYTES)
    ) {
        return 0;
    }
    noc_tlb_2m_t programmed;
    volatile const uint8_t* window = (
        (volatile const uint8_t*)noc_configure_tlb_2m_checked(
            KMD_WIN_INSPECT,
            coordinates & 0xFFFFu,
            coordinates >> 16,
            0,
            NOC_ORDER_DEFAULT,
            &programmed
        )
    );
    if (window == 0) {
        return 0;
    }
    dispatcher->opened |= 1u << 6;
    return noc_verify_tlb_2m(KMD_WIN_INSPECT, &programmed) != 0 ? 0 : window;
}

/* Record `index` of the block: core `core`'s state words and live words, then the
 * trace entries of the inspect's sequence and the three before it, where held. */
static void inspect_core(kmd_t* dispatcher, uint32_t index, uint32_t core) {
    volatile uint8_t* record = (
        dispatcher->lim + KMD_LIM_INSPECT + KMD_INSPECT_HEAD +
        index * KMD_INSPECT_RECORD
    );
    volatile uint32_t* line = (volatile uint32_t*)record;
    const uint32_t coordinates = (
        ((volatile const uint32_t*)(dispatcher->lim + KMD_LIM_COORDS))[core]
    );
    const uint32_t control = dispatcher->control_address;
    uint32_t flags = 0;
    volatile const uint8_t* window = inspect_aim(dispatcher, core);
    for (uint32_t entry = 0; entry < KMD_INSPECT_ENTRIES; entry++) {
        const uint32_t sequence = dispatcher->inspect.sequence - entry;
        line[KMD_IR_SEQUENCES + entry] = sequence;
    }
    if (window != 0) {
        inspect_copy(
            record + KMD_INSPECT_STATE,
            window + control + KM_CONTROL_STATE,
            4u * KM_STATE_WORDS
        );
        inspect_copy(
            record + KMD_INSPECT_LIVE, window + control + KM_CONTROL_LIVE, KM_LIVE_READ
        );
        flags |= KMD_INSPECT_READ;
        const volatile uint32_t* live = (
            (const volatile uint32_t*)(record + KMD_INSPECT_LIVE)
        );
        const uint32_t trace = live[KM_LIVE_TRACE], cap = live[KM_LIVE_TRACE_CAP];
        if (
            trace != 0 && trace % KM_TRACE_ENTRY == 0 && cap <= KMD_L1_BYTES &&
            inspect_in_l1(trace, cap * KM_TRACE_ENTRY)
        ) {
            for (uint32_t entry = 0; entry < KMD_INSPECT_ENTRIES; entry++) {
                const uint32_t sequence = line[KMD_IR_SEQUENCES + entry];
                volatile uint8_t*
                    to = (record + KMD_INSPECT_TRACE + entry * KM_TRACE_ENTRY);
                if (sequence != 0 && sequence - 1u < cap) {
                    inspect_copy(
                        to,
                        window + trace + (sequence - 1u) * KM_TRACE_ENTRY,
                        KM_TRACE_ENTRY
                    );
                } else {
                    ((volatile uint64_t*)to)[0] = 0;
                    ((volatile uint64_t*)to)[1] = 0;
                }
            }
            flags |= KMD_INSPECT_TRACED;
        }
    }
    line[KMD_IR_CORE] = core;
    line[KMD_IR_COORDINATES] = coordinates;
    line[KMD_IR_FLAGS] = flags;
    line[KMD_IR_MTIME] = (uint32_t)x280_mtime();
}

/* Whether core `core` runs no part of command `sequence` any more: none of its kernels'
 * live sequence words (16 to 40 bytes into its live words) holds it; 0 unreachable. */
static int inspect_left(kmd_t* dispatcher, uint32_t core, uint32_t sequence) {
    volatile const uint8_t* window = inspect_aim(dispatcher, core);
    if (window == 0) {
        return 0;
    }
    volatile const uint64_t*
        words = (
            (volatile const uint64_t*)(
                window + dispatcher->control_address + KM_CONTROL_LIVE + 16u
            )
        );
    uint32_t live[6];
    for (uint32_t word = 0; word < 3u; word++) {
        const uint64_t pair = words[word];
        live[2u * word] = (uint32_t)pair;
        live[2u * word + 1u] = (uint32_t)(pair >> 32);
    }
    for (uint32_t role = 0; role < KM_LIVE_ROLES; role++) {
        if (live[KM_LIVE_SEQUENCE - 4u + role] == sequence) {
            return 0;
        }
    }
    return 1;
}

/* A timeout's abort: the slot's abort word on every core (each gives up the command
 * it runs only if it is this one); its cores are then checked (inspect_leave). */
static void inspect_abort(kmd_t* dispatcher, uint32_t slot) {
    const uint32_t sequence = dispatcher->slot_sequence[slot];
    for (uint32_t panel = 0; panel < dispatcher->panels; panel++) {
        *(volatile uint32_t*)(
            dispatcher->control[panel] + KM_CONTROL_ABORT + 4u * slot
        ) = sequence;
    }
    x280_fence_ow();
}

/* A timeout's end once every core of the group left the command: the slot freed, the
 * give-up counted, KM_EVENT_TIMEOUT to its source. */
static void inspect_give_up(kmd_t* dispatcher, uint32_t slot, uint64_t now) {
    const uint32_t sequence = dispatcher->slot_sequence[slot];
    const uint32_t took = (uint32_t)(now - dispatcher->slot_submitted[slot]);
    dispatcher->busy &= ~(1ull << slot);
    dispatcher->result_busy &= ~(1ull << slot);
    dispatcher->busy_count--;
    trace(dispatcher, KMD_TRACE_FREE, slot, dispatcher->slot_request[slot]);
    kmd_ring_t* ring = &dispatcher->ring[dispatcher->slot_source[slot]];
    ring->busy_count--;
    const uint32_t auxiliary = (
        slot | (sequence & KM_EVENT_SEQUENCE_MASK) << KM_EVENT_SEQUENCE_SHIFT
    );
    ring_push(
        ring, dispatcher->slot_request[slot], KM_EVENT_TIMEOUT | auxiliary << 8, took
    );
    dispatcher->x280_cores[KMD_H_GIVEN_UP] = ++dispatcher->given_up;
    dispatcher->last_event_mtime = now;
    if (dispatcher->waits) {
        sources_wake(dispatcher);
    }
}

/* A timeout's cores after the abort, one a pass: the give-up once none runs the
 * command; at KMD_GIVE_UP_TICKS, the slot kept for the next poll's fatal deadline. */
static void inspect_leave(kmd_t* dispatcher) {
    kmd_inspect_t* inspect = &dispatcher->inspect;
    const uint64_t now = x280_mtime();
    if (
        inspect->next < inspect->count &&
        inspect_left(dispatcher, inspect->first + inspect->next, inspect->sequence)
    ) {
        inspect->next++;
    }
    if (inspect->next == inspect->count) {
        inspect->active = 0;
        inspect_give_up(dispatcher, inspect->slot, now);
    } else if (now > inspect->until) {
        inspect->active = 0;
        dispatcher->slot_handler[inspect->slot] = KMD_HANDLER_KEPT;
        dispatcher->slot_deadline[inspect->slot] = 0;
    }
}

/* One core of the inspect; after the last, the mark, then the host's PONG or the
 * timeout's abort. A timeout's command that completed meanwhile ends the inspect. */
static void inspect_turn(kmd_t* dispatcher) {
    kmd_inspect_t* inspect = &dispatcher->inspect;
    if (
        inspect->kind == KMD_INSPECT_TIMEOUT &&
        (
            !(dispatcher->busy & (1ull << inspect->slot)) ||
            dispatcher->slot_sequence[inspect->slot] != inspect->sequence
        )
    ) {
        inspect->active = 0;
        return;
    }
    if (inspect->leaving) {
        inspect_leave(dispatcher);
        return;
    }
    inspect_core(dispatcher, inspect->next, inspect->first + inspect->next);
    volatile uint32_t* header = inspect_header(dispatcher);
    header[KMD_I_DONE] = ++inspect->next;
    if (inspect->next < inspect->count) {
        return;
    }
    const uint64_t now = x280_mtime();
    header[KMD_I_TICKS] = (uint32_t)(now - inspect->start);
    x280_fence_ow(); /* every record and word lands before the mark */
    header[KMD_I_MARK] = KMD_INSPECT_MARK;
    x280_fence_ow();
    if (inspect->kind == KMD_INSPECT_HOST) {
        inspect->active = 0;
        kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_HOST];
        ring->owed--;
        ring_push(
            ring, inspect->request, KM_EVENT_PONG, (uint32_t)(now - inspect->start)
        );
        dispatcher->last_event_mtime = now;
    } else {
        inspect_abort(dispatcher, inspect->slot);
        inspect->leaving = 1;
        inspect->next = 0;
        inspect->until = now + KMD_GIVE_UP_TICKS;
    }
}

/* Whether the command in busy slot `slot` may be given up at its deadline, the run
 * going on: a program's handler from the host, not a dynamic run's, not one kept. */
static inline int inspect_may_give_up(const kmd_t* dispatcher, uint32_t slot) {
    const uint32_t handler = dispatcher->slot_handler[slot];
    return (
        dispatcher->slot_source[slot] == KMD_SRC_HOST &&
        handler <= KM_HANDLER_USER_MAX &&
        !KM_IS_DYNAMIC(dispatcher->messages.handler, handler) &&
        dispatcher->messages.end != slot + 1u
    );
}

/* At a command's deadline: an inspect of its group, then its give-up (the slot's
 * deadline is put out of reach meanwhile). */
static inline void inspect_timeout(kmd_t* dispatcher, uint32_t slot, uint64_t now) {
    const uint32_t group = dispatcher->slot_group[slot];
    dispatcher->slot_deadline[slot] = ~0ull;
    inspect_begin(
        dispatcher,
        KMD_INSPECT_TIMEOUT,
        dispatcher->slot_request[slot],
        group & 0xFFFFu,
        group >> 16,
        slot,
        dispatcher->slot_sequence[slot],
        now
    );
}

/* Clears the inspect window if it was programmed. */
static inline void inspect_close(kmd_t* dispatcher) {
    if (dispatcher->opened & (1u << 6)) {
        noc_clear_tlb_2m(KMD_WIN_INSPECT);
        dispatcher->opened &= ~(1u << 6);
    }
}
