/* Commands across islands: a command of this island's sources whose group (or thread)
 * has a part on the island at the link's other end crosses as a packet; that island's
 * dispatcher runs it as a command of its own source KMD_SRC_LINK and sends DONE back.
 * Packets go out through link core 0 of the island (a PAGES LINE to the CPU that holds
 * link 0) and land in the far dispatcher's link inbox, one writer an inbox part
 * (link_inbox): the lines, then the count word. */
#pragma once

#include "completion.h"
#include "kernel.h"
#include "ring.h"

/* A packet: the command line (its group word in flat ranks), then the crossing line. */
typedef struct {
    uint64_t line[16];
} kmd_packet_t;

/* A command of this island's sources with a part on another island. */
typedef struct {
    uint32_t request, source, parts, generation;
    uint32_t
        sequence;    /* the number its events carry (the local part's, or one taken) */
    uint32_t first;  /* the local part's first core (its DONE's auxiliary) */
    uint32_t status; /* KM_EVENT_DONE, or the first other end: its event status */
    uint32_t word;   /* the result word, from the part that holds the group's first */
    uint32_t result; /* it has KM_FLAG_RESULT: 1, 2 once a part gave the word */
    uint32_t late;   /* its TIMEOUT went out; the rest of its parts send none */
    uint32_t far;    /* the island of its (first) part across */
    uint32_t holder; /* the island that holds the group's first rank: its result word */
    uint64_t submitted, deadline;
} kmd_crossing_t;

/* A command from the other island, taken into the link ring: whose it is. */
typedef struct {
    uint32_t number; /* the link ring's number (its events' request); 0: free */
    uint32_t island, ticket;
} kmd_origin_t;

#define KMD_CROSSINGS 8u     /* crossings in flight a dispatcher starts */
#define KMD_LINK_DEPTH 16u   /* the link ring's command lines */
#define KMD_LINK_EVENTS 32u  /* its events */
#define KMD_LINK_OUT 16u     /* packets waiting for the link core */
#define KMD_LINK_ORIGINS 16u /* commands from the other island not answered yet */

typedef struct {
    /* the link ring (source KMD_SRC_LINK): lines, events, its words */
    uint64_t commands[KMD_LINK_DEPTH][8];
    uint8_t events[KMD_LINK_EVENTS * 16u] __attribute__((aligned(16)));
    uint32_t consumed, told, read, pad;
    uint64_t line[8]; /* the line being taken */
    uint64_t send[8]; /* a LINK command being built */
    kmd_packet_t out[KMD_LINK_OUT];
    uint32_t left[KMD_LINK_DEPTH]; /* each line's ticks left of its deadline */
    kmd_origin_t origin[KMD_LINK_ORIGINS];
    kmd_crossing_t crossing[KMD_CROSSINGS];
    uint32_t ranks[KM_ISLANDS_MAX];       /* each island's ranks: first | count << 16 */
    uint32_t threads[KM_ISLANDS_MAX];     /* its thread numbers: first | count << 16 */
    uint8_t slot_crossing[KM_RING_SLOTS]; /* 1 + the crossing of a local part */
    uint8_t slot_stage[KM_RING_SLOTS];    /* 1 + the stage of a LINK command */
    uint8_t out_to[KMD_LINK_OUT];         /* the island each queued packet goes to */
    uint32_t far_tiles[KMD_L_TILES_MAX];  /* each island's dispatcher tile */
    uint32_t far_blocks[KMD_L_TILES_MAX]; /* and the NoC address of its link block */
    uint32_t sent_to[KMD_L_TILES_MAX];    /* packets sent to each island */
    uint32_t taken_from[KMD_L_TILES_MAX]; /* packets read from its inbox part */
    uint32_t crossings;                   /* crossings in flight at most */
    volatile uint8_t* window;             /* the link core's area */
    volatile const uint8_t* inbox;
    uint32_t island, islands, core, far, far_tile, far_block, area;
    uint32_t stored; /* lines put into the link ring */
    uint32_t taken;  /* packets read from the inbox */
    uint32_t sent;   /* packets handed to the link core */
    uint32_t out_head, out_tail;
    uint32_t stages;      /* bit s: stage s holds a packet in flight */
    uint32_t busy;        /* bit e: crossing e in flight */
    uint32_t events_read; /* link ring events turned into DONE packets */
    /* counters */
    uint32_t crossed, received, answered, gave_up;
} kmd_link_t;

_Static_assert(
    KMD_STATE_LINK + sizeof(kmd_link_t) <= KMD_STATE_BYTES,
    "the link outgrows its state"
);
_Static_assert(KMD_STATE_QUEUE + sizeof(kmd_queue_t) <= KMD_STATE_LINK, "queue, link");
_Static_assert(KMD_CROSSINGS < 256u && KMD_LINK_DEPTH >= KMD_CROSSINGS, "tickets");

/* 32-bit word `word` of the 64-bit words at `line`, and its store. */
static inline uint32_t link_word(const uint64_t* line, uint32_t word) {
    return (uint32_t)(line[word / 2u] >> 32u * (word % 2u));
}

static inline void link_set(uint64_t* line, uint32_t word, uint32_t value) {
    const uint32_t shift = 32u * (word % 2u);
    line[word / 2u] = (
        (line[word / 2u] & ~((uint64_t)0xFFFFFFFFu << shift)) | (uint64_t)value << shift
    );
}

static inline kmd_link_t* link_state(kmd_t* dispatcher) {
    return (kmd_link_t*)((uint8_t*)dispatcher + KMD_STATE_LINK);
}

/* The bits set in `bits` (no libgcc here). */
static inline uint32_t link_bits(uint32_t bits) {
    uint32_t count = 0;
    for (; bits != 0; bits &= bits - 1u) {
        count++;
    }
    return count;
}

/* Island `from`'s part of island `to`'s inbox (offsets in the link block): its count
 * line, its first slot, its slots; more than two islands: 16 / (islands - 1) each. */
static inline void link_inbox(
    const kmd_link_t* link,
    uint32_t to,
    uint32_t from,
    uint32_t* count,
    uint32_t* slots_at,
    uint32_t* slots
) {
    if (link->islands == 2u) {
        *count = KMD_LINK_COUNT;
        *slots_at = KMD_LINK_SLOTS_AT;
        *slots = KMD_LINK_SLOTS;
        return;
    }
    const uint32_t part = KMD_LINK_SLOTS / (link->islands - 1u);
    const uint32_t index = from < to ? from : from - 1u;
    *count = KMD_LINK_SLOTS_AT + index * part * KM_LINK_PACKET_BYTES;
    *slots_at = *count + KM_LINK_PACKET_BYTES;
    *slots = part - 1u;
}

/* Reads the host's link line; with its mark, the window onto the link core's area and
 * the link ring (the link core: link core 0, the card's core after the island's ranks).
 * KMD_ERROR_NONE also with no link; KMD_ERROR_WINDOW. */
static inline uint32_t link_open(kmd_t* dispatcher) {
    volatile const uint32_t* words = (
        (volatile const uint32_t*)(dispatcher->lim + KMD_LIM_LINK)
    );
    x280_fence_ir();
    if (words[KMD_L_MARK] != KMD_LINK_MARK || dispatcher->island_count == 0) {
        return KMD_ERROR_NONE;
    }
    kmd_link_t* link = link_state(dispatcher);
    link->island = words[KMD_L_ISLAND];
    link->islands = words[KMD_L_ISLANDS];
    link->core = words[KMD_L_CORE];
    link->far = words[KMD_L_FAR];
    link->far_tile = words[KMD_L_FAR_TILE];
    link->far_block = words[KMD_L_FAR_BLOCK];
    const uint32_t area = words[KMD_L_AREA];
    link->area = area;
    if (
        link->islands < 2u || link->islands > KMD_L_TILES_MAX ||
        link->island >= link->islands || link->far >= link->islands ||
        link->far == link->island || link->core >= dispatcher->cores ||
        link->core != dispatcher->island_base + dispatcher->island_count ||
        (area & 63u) || area < KMD_CONTROL_FLOOR ||
        area > KMD_L1_BYTES - KM_LINK_AREA_BYTES || (link->far_block & 63u)
    ) {
        return KMD_ERROR_ARGS;
    }
    const uint32_t tiles = link->islands == 2u ? 0u : link->islands;
    for (uint32_t island = 0; island < tiles && island < KMD_L_TILES_MAX; island++) {
        link->far_tiles[island] = words[KMD_L_TILES + 2u * island];
        link->far_blocks[island] = words[KMD_L_TILES + 2u * island + 1u];
        if (link->far_blocks[island] & 63u) {
            return KMD_ERROR_ARGS;
        }
    }
    if (link->islands == 2u && link->far < KMD_L_TILES_MAX) {
        link->far_tiles[link->far] = link->far_tile;
        link->far_blocks[link->far] = link->far_block;
    }
    /* each pair's unread packets: a crossing's command and a DONE of the far one's */
    uint32_t count = 0, slots_at = 0, slots = 0;
    link_inbox(link, link->island, link->far, &count, &slots_at, &slots);
    link->crossings = slots / 2u < KMD_CROSSINGS ? slots / 2u : KMD_CROSSINGS;
    if (link->crossings == 0) {
        return KMD_ERROR_ARGS;
    }
    volatile const uint32_t* table = words + KMD_LINK_TABLE / 4u;
    for (uint32_t island = 0; island < link->islands; island++) {
        link->ranks[island] = table[2u * island];
        link->threads[island] = table[2u * island + 1u];
    }
    const uint32_t coordinates = (
        ((volatile const uint32_t*)(dispatcher->lim + KMD_LIM_COORDS))[link->core]
    );
    noc_tlb_2m_t programmed;
    volatile uint8_t* window = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            KMD_WIN_LINK,
            coordinates & 0xFFFFu,
            coordinates >> 16,
            area,
            NOC_ORDER_POSTED,
            &programmed
        )
    );
    if (window == 0) {
        return KMD_ERROR_WINDOW;
    }
    dispatcher->opened |= 1u << 7;
    if (noc_verify_tlb_2m(KMD_WIN_LINK, &programmed) != 0) {
        return KMD_ERROR_WINDOW;
    }
    link->window = window;
    link->inbox = dispatcher->lim + KMD_LIM_LINK;
    kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_LINK];
    ring->commands = (volatile const uint8_t*)link->commands;
    ring->consumed = &link->consumed;
    ring->events = link->events;
    ring->told = &link->told;
    ring->read = &link->read;
    ring->depth = KMD_LINK_DEPTH;
    ring->event_depth = KMD_LINK_EVENTS;
    ring->next = 1;
    ring->line = link->line;
    ring->t_command = dispatcher->ring[KMD_SRC_HOST].t_command;
    dispatcher->thread_first = link->threads[link->island] & 0xFFFFu;
    dispatcher->link_on = 1;
    return KMD_ERROR_NONE;
}

static inline void link_close(kmd_t* dispatcher) {
    if (dispatcher->opened & (1u << 7)) {
        x280_fence_ow();
        noc_clear_tlb_2m(KMD_WIN_LINK);
        dispatcher->opened &= ~(1u << 7);
    }
}

/* The island that holds rank `rank`, or KM_ISLANDS_MAX for none. */
static inline uint32_t link_island_of(const kmd_link_t* link, uint32_t rank) {
    for (uint32_t island = 0; island < link->islands; island++) {
        const uint32_t first = link->ranks[island] & 0xFFFFu;
        if (rank - first < (link->ranks[island] >> 16)) {
            return island;
        }
    }
    return KM_ISLANDS_MAX;
}

/* The island whose thread has number `number` (a group's first less the base), or
 * KM_ISLANDS_MAX. */
static inline uint32_t link_thread_island(const kmd_link_t* link, uint32_t number) {
    for (uint32_t island = 0; island < link->islands; island++) {
        const uint32_t first = link->threads[island] & 0xFFFFu;
        if (number - first < (link->threads[island] >> 16)) {
            return island;
        }
    }
    return KM_ISLANDS_MAX;
}

/* Whether a command's group word reaches outside this island: ranks beyond its own, or
 * a thread number another island serves. Only with a link. Link cores never do. */
KMD_HOT int link_reaches_out(const kmd_t* dispatcher, uint32_t group) {
    const uint32_t first = group & 0xFFFFu, count = group >> 16;
    if (KM_GROUP_IS_LINK(first)) {
        return 0;
    }
    if (first >= KM_GROUP_CLUSTER_BASE) {
        const uint32_t number = first - KM_GROUP_CLUSTER_BASE;
        return count == 1u && number - dispatcher->thread_first >= KMD_WORK_CLUSTERS;
    }
    const uint32_t low = dispatcher->island_first;
    return first < low || first + count > low + dispatcher->island_count;
}

static inline uint32_t link_out_room(const kmd_link_t* link) {
    return KMD_LINK_OUT - (link->out_tail - link->out_head);
}

/* Queues a packet for island `to`: `command` (8 words, or 0 for a DONE) and the
 * crossing line's words. */
static inline void link_queue(
    kmd_link_t* link,
    uint32_t to,
    const uint64_t* command,
    uint32_t kind,
    uint32_t ticket,
    uint32_t left,
    uint32_t ticks
) {
    kmd_packet_t* packet = &link->out[link->out_tail % KMD_LINK_OUT];
    for (uint32_t word = 0; word < 16; word++) {
        packet->line[word] = word < 8 && command != 0 ? command[word] : 0;
    }
    uint64_t* crossing = &packet->line[8];
    link_set(crossing, KMD_X_KIND, kind | link->island << 8);
    link_set(crossing, KMD_X_TICKET, ticket);
    link_set(crossing, KMD_X_LEFT, left);
    link_set(crossing, KMD_X_TICKS, ticks);
    link->out_to[link->out_tail % KMD_LINK_OUT] = (uint8_t)to;
    link->out_tail++;
}

/* A TIMEOUT that gives a command up goes to `source`: the host's count of give-ups
 * (KMD_H_GIVEN_UP) grows by one first, so that the host goes on past exactly the
 * TIMEOUTs it is sent of commands given up. */
static inline void link_gave(kmd_t* dispatcher, uint32_t source) {
    if (source == KMD_SRC_HOST) {
        dispatcher->x280_cores[KMD_H_GIVEN_UP] = ++dispatcher->given_up;
    }
}

/* A crossing's part came back (or the local part ended): its status and ticks; the
 * last part sends the crossing's event unless its TIMEOUT went out. */
static void link_part(
    kmd_t* dispatcher,
    uint32_t index,
    uint32_t island,
    uint32_t status,
    uint32_t ticks,
    uint64_t now
) {
    kmd_link_t* link = link_state(dispatcher);
    kmd_crossing_t* crossing = &link->crossing[index];
    const uint32_t kind = status & 0xFFu;
    if (kind != KM_EVENT_DONE && crossing->status == KM_EVENT_DONE) {
        crossing->status = (
            kind == KM_EVENT_TIMEOUT
                ? KM_EVENT_TIMEOUT | (KM_EVENT_ISLAND + island) << 8
                : status
        );
    }
    if (kind == KM_EVENT_DONE && crossing->result == 1u && island == crossing->holder) {
        crossing->word = ticks; /* as on one island: the part of the group's first */
        crossing->result = 2u;
    }
    if (--crossing->parts != 0) {
        return;
    }
    link->busy &= ~(1u << index);
    if (crossing->late) {
        return;
    }
    kmd_ring_t* ring = &dispatcher->ring[crossing->source];
    ring->busy_count--;
    uint32_t out = crossing->status;
    const uint32_t sequence = (
        (crossing->sequence & KM_EVENT_SEQUENCE_MASK) << KM_EVENT_SEQUENCE_SHIFT
    );
    uint32_t took = (uint32_t)(now - crossing->submitted);
    if ((out & 0xFFu) == KM_EVENT_DONE) {
        out = KM_EVENT_DONE | (crossing->first | sequence) << 8;
        if (crossing->result) {
            dispatcher->last_result = took = crossing->word;
            dispatcher->results++;
        }
    } else if ((out & 0xFFu) == KM_EVENT_TIMEOUT) {
        out = (out & 0xFFFFu) | sequence << 8;
        link_gave(dispatcher, crossing->source);
    }
    ring_push(ring, crossing->request, out, took);
    dispatcher->last_event_mtime = now;
    if (dispatcher->waits) {
        sources_wake(dispatcher);
    }
}

/* A LINE of stage `stage` lost on hung link core `k`: its command packet's crossing,
 * if still in flight, is named lost and ends at link_late's next look. */
static void link_lost(kmd_t* dispatcher, uint32_t stage, uint32_t k, uint32_t ticks) {
    kmd_link_t* link = link_state(dispatcher);
    volatile const uint32_t*
        crossing_line = (
            (volatile const uint32_t*)(
                link->window + KM_LINK_STAGE + stage * KM_LINK_STAGE_BYTES + 64u
            )
        );
    x280_fence_ir();
    const uint32_t kind = crossing_line[KMD_X_KIND] & 0xFFu;
    const uint32_t ticket = crossing_line[KMD_X_TICKET];
    const uint32_t index = ticket & 0xFFu;
    if (kind != KMD_CROSS_COMMAND || index >= KMD_CROSSINGS) {
        return; /* a DONE for the other island: its crossing ends at its deadline */
    }
    kmd_crossing_t* crossing = &link->crossing[index];
    if (
        !(link->busy & (1u << index)) || crossing->late ||
        crossing->generation != ticket >> 8
    ) {
        return;
    }
    link_event(
        dispatcher,
        crossing->source == KMD_SRC_HOST ? crossing->request : 0u,
        k,
        KMD_LINK_LOST,
        ticks
    );
    crossing->deadline = 0;
}

/* A busy slot of the link freed (its ack, or a give-up): a crossing's local part, or a
 * LINK command whose stage is free again. Returns 1 if the slot was the link's. */
static int link_freed(
    kmd_t* dispatcher, uint32_t slot, uint32_t status, uint32_t ticks, uint64_t now
) {
    kmd_link_t* link = link_state(dispatcher);
    dispatcher->link_slots &= ~(1ull << slot);
    if (link->slot_stage[slot] != 0) {
        const uint32_t stage = link->slot_stage[slot] - 1u;
        link->stages &= ~(1u << stage);
        link->slot_stage[slot] = 0;
        if (dispatcher->slot_lost[slot] != 0 && (status & 0xFFu) != KM_EVENT_DONE) {
            link_lost(dispatcher, stage, dispatcher->slot_lost[slot] - 1u, ticks);
        }
        return 1;
    }
    if (link->slot_crossing[slot] != 0) {
        const uint32_t index = link->slot_crossing[slot] - 1u;
        link->slot_crossing[slot] = 0;
        link_part(dispatcher, index, link->island, status, ticks, now);
        return 1;
    }
    return 0;
}

/* take()'s turn for a command whose group reaches out: rejected, held, or its local
 * part submitted and one packet queued for the far island; one event at the end. */
static __attribute__((noinline)) kmd_take_t
link_cross(kmd_t* dispatcher, uint32_t source, uint64_t now) {
    kmd_link_t* link = link_state(dispatcher);
    kmd_ring_t* ring = &dispatcher->ring[source];
    uint64_t* line = ring->line;
    const uint32_t request = ring->next, header = (uint32_t)(line[0] >> 32);
    const uint32_t handler = header & 0xFFu, flags = (header >> 8) & 0xFFu;
    const uint32_t group = (uint32_t)line[1];
    const uint32_t first = group & 0xFFFFu, count = group >> 16;
    const uint32_t thread = first >= KM_GROUP_CLUSTER_BASE;
    uint32_t part = 0, reject = 0, fars = 0, far = KM_ISLANDS_MAX;
    if (handler > KM_HANDLER_USER_MAX) {
        reject = KMD_REJECT_HANDLER;
    } else if (flags & (1u | KM_FLAG_EXCLUSIVE | KM_FLAG_QUEUE)) {
        reject = KMD_REJECT_FLAGS;
    } else if (dispatcher->exited) {
        reject = KMD_REJECT_EXITED;
    } else if (!thread && KM_IS_DYNAMIC(dispatcher->messages.handler, handler)) {
        reject = KMD_REJECT_MESSAGES;
    } else if (thread) {
        far = link_thread_island(link, first - KM_GROUP_CLUSTER_BASE);
        fars = 1u << far;
        if (far == KM_ISLANDS_MAX || far == link->island) {
            reject = KMD_REJECT_GROUP;
        }
    } else {
        /* every rank is an island's (the islands' ranks follow each other from 0), and
         * another island has some: a part to each of them */
        const uint32_t end = first + count;
        const uint32_t last = link->ranks[link->islands - 1u];
        if (count == 0 || first > end || end > (last & 0xFFFFu) + (last >> 16)) {
            reject = KMD_REJECT_GROUP;
        }
        for (uint32_t island = link->islands; island-- > 0;) {
            const uint32_t low = link->ranks[island] & 0xFFFFu;
            const uint32_t high = low + (link->ranks[island] >> 16);
            if (island != link->island && first < high && end > low) {
                fars |= 1u << island;
                far = island;
            }
        }
        if (fars == 0) {
            reject = KMD_REJECT_GROUP;
        }
        part = island_part(dispatcher, group);
        if (
            reject == 0 && part != 0 && dispatcher->programs &&
            !program_held(dispatcher, header >> 24, part & 0xFFFFu, part >> 16)
        ) {
            reject = KMD_REJECT_PROGRAM;
        }
    }
    if (reject != 0) {
        ring_consumed(ring);
        ring_push(ring, request, KM_EVENT_REJECTED | (reject << 8), 0);
        dispatcher->last_event_mtime = now;
        dispatcher->rejected++;
        return KMD_TAKEN;
    }
    if (
        link_bits(link->busy) >= link->crossings ||
        link_out_room(link) < link_bits(fars)
    ) {
        return KMD_HELD;
    }
    uint32_t slot = KM_RING_SLOTS;
    if (part != 0) {
        slot = slot_free(dispatcher);
        if (slot == KM_RING_SLOTS || messages_held(dispatcher, handler)) {
            return KMD_HELD;
        }
    }
    uint32_t index = 0;
    while (link->busy & (1u << index)) {
        index++;
    }
    kmd_crossing_t* crossing = &link->crossing[index];
    crossing->generation = (crossing->generation + 1u) & 0xFFFFFFu;
    const uint32_t ticket = index | crossing->generation << 8;
    for (uint32_t island = 0; island < link->islands; island++) {
        if (fars & (1u << island)) {
            link_queue(
                link,
                island,
                line,
                KMD_CROSS_COMMAND,
                ticket,
                (uint32_t)ring->t_command,
                0
            );
        }
    }
    crossing->request = request;
    crossing->source = source;
    crossing->parts = (part != 0 ? 1u : 0u) + link_bits(fars);
    crossing->status = KM_EVENT_DONE;
    crossing->result = (flags & KM_FLAG_RESULT) != 0;
    crossing->word = 0;
    crossing->late = 0;
    crossing->far = far;
    crossing->holder = thread ? far : link_island_of(link, first);
    crossing->first = part & 0xFFFFu;
    crossing->submitted = now;
    crossing->deadline = now + ring->t_command;
    link->busy |= 1u << index;
    link->crossed++;
    ring->busy_count++; /* the crossing's event */
    if (part != 0) {
        command_submit(
            dispatcher,
            line,
            slot,
            request,
            source,
            now,
            ring->t_command,
            KMD_TRACE_SUBMIT
        );
        crossing->sequence = dispatcher->slot_sequence[slot];
        link->slot_crossing[slot] = (uint8_t)(index + 1u);
        dispatcher->link_slots |= 1ull << slot;
    } else {
        uint32_t sequence = ++dispatcher->sequence;
        if (sequence == 0) {
            sequence = ++dispatcher->sequence;
        }
        crossing->sequence = sequence;
    }
    ring_consumed(ring);
    return KMD_TAKEN;
}

/* Hands queued packets to link core 0: each staged in a free stage of its area (three
 * lines through the window), then a PAGES LINE command to that core, named outside the
 * ranks (KM_GROUP_LINK_BASE). */
static void link_send(kmd_t* dispatcher, uint64_t now) {
    kmd_link_t* link = link_state(dispatcher);
    while (link->out_head != link->out_tail) {
        uint32_t stage = 0;
        while (stage < KM_LINK_STAGES && (link->stages & (1u << stage))) {
            stage++;
        }
        const uint32_t slot = slot_free(dispatcher);
        if (stage == KM_LINK_STAGES || slot == KM_RING_SLOTS) {
            return;
        }
        kmd_packet_t* packet = &link->out[link->out_head % KMD_LINK_OUT];
        const uint32_t to = link->out_to[link->out_head % KMD_LINK_OUT];
        const uint32_t number = ++link->sent_to[to];
        link->sent++;
        /* the command line's number and landing words, and the crossing line's: the
         * far dispatcher takes the packet when all show it */
        for (uint32_t word = 0; word < 8; word += 2) {
            packet->line[word] = (packet->line[word] & ~(uint64_t)0xFFFFFFFFu) | number;
        }
        link_set(&packet->line[8], KMD_X_NUMBER, number);
        uint64_t count[8] __attribute__((aligned(64)));
        count[0] = (uint64_t)number | (uint64_t)number << 32;
        for (uint32_t word = 1; word < 8; word++) {
            count[word] = 0;
        }
        volatile uint8_t*
            at = (link->window + KM_LINK_STAGE + stage * KM_LINK_STAGE_BYTES);
        kmd_store_line(at, &packet->line[0]);
        kmd_store_line(at + 64u, &packet->line[8]);
        kmd_store_line(at + 128u, count);
        x280_fence_ow();
        link->out_head++;
        const uint32_t far = link->far_blocks[to];
        uint32_t count_at = 0, slots_at = 0, slots = 0;
        link_inbox(link, to, link->island, &count_at, &slots_at, &slots);
        uint64_t* send = link->send;
        send[0] = (uint64_t)KM_HANDLER_PAGES << 32;
        send[1] = (uint64_t)(KM_GROUP_LINK_BASE | 1u << 16) | (uint64_t)stage << 32;
        send[2] = (uint64_t)number << 32;
        send[3] = (
            (uint64_t)link->far_tiles[to] |
            (uint64_t)(far + slots_at + (number % slots) * KM_LINK_PACKET_BYTES) << 32
        );
        send[4] = (uint64_t)(far + count_at) << 32;
        send[5] = link->area; /* a5; a7: one hop, the LINE mode */
        send[6] = (uint64_t)(1u << 16 | KM_PAGES_LINE << 24) << 32;
        send[7] = 0;
        command_submit(
            dispatcher,
            send,
            slot,
            0,
            KMD_SRC_LINK,
            now,
            dispatcher->ring[KMD_SRC_HOST].t_command,
            KMD_TRACE_SUBMIT
        );
        link->stages |= 1u << stage;
        link->slot_stage[slot] = (uint8_t)(stage + 1u);
        dispatcher->link_slots |= 1ull << slot;
    }
}

/* Reads island `from`'s inbox part: commands into the link ring (whose, time left),
 * DONEs to their crossings; a packet not yet whole, or no room, waits. */
static void link_receive_from(kmd_t* dispatcher, uint32_t from, uint64_t now) {
    kmd_link_t* link = link_state(dispatcher);
    uint32_t count_at = 0, slots_at = 0, slots = 0;
    link_inbox(link, link->island, from, &count_at, &slots_at, &slots);
    x280_fence_ir();
    const uint32_t landed = *(volatile const uint32_t*)(link->inbox + count_at);
    while (link->taken_from[from] != landed) {
        const uint32_t number = link->taken_from[from] + 1u;
        volatile const uint64_t*
            words = (
                (volatile const uint64_t*)(
                    link->inbox + slots_at + (number % slots) * KM_LINK_PACKET_BYTES
                )
            );
        uint64_t packet[16];
        x280_fence_ir();
        for (uint32_t word = 0; word < 16; word++) {
            packet[word] = words[word];
        }
        const uint64_t* crossing = &packet[8];
        if (
            (uint32_t)packet[0] != number || (uint32_t)packet[2] != number ||
            (uint32_t)packet[4] != number || (uint32_t)packet[6] != number ||
            link_word(crossing, KMD_X_NUMBER) != number
        ) {
            return; /* not whole yet */
        }
        const uint32_t kind = link_word(crossing, KMD_X_KIND) & 0xFFu;
        const uint32_t island = link_word(crossing, KMD_X_KIND) >> 8 & 0xFFu;
        if (kind == KMD_CROSS_COMMAND) {
            kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_LINK];
            uint32_t free_origin = KMD_LINK_ORIGINS;
            for (uint32_t k = 0; k < KMD_LINK_ORIGINS; k++) {
                if (link->origin[k].number == 0) {
                    free_origin = k;
                    break;
                }
            }
            if (
                link->stored + 1u - ring->next >= KMD_LINK_DEPTH ||
                free_origin == KMD_LINK_ORIGINS
            ) {
                return; /* waits for room */
            }
            const uint32_t at = ++link->stored;
            uint64_t* stored = link->commands[at % KMD_LINK_DEPTH];
            for (uint32_t word = 0; word < 8; word++) {
                stored[word] = (
                    word % 2 == 0
                        ? (packet[word] & ~(uint64_t)0xFFFFFFFFu) | at
                        : packet[word]
                );
            }
            link->left[at % KMD_LINK_DEPTH] = link_word(crossing, KMD_X_LEFT);
            link->origin[free_origin].number = at;
            link->origin[free_origin].island = island;
            link->origin[free_origin].ticket = link_word(crossing, KMD_X_TICKET);
            link->received++;
        } else if (kind == KMD_CROSS_DONE) {
            const uint32_t ticket = link_word(crossing, KMD_X_TICKET);
            const uint32_t index = ticket & 0xFFu;
            if (
                index < KMD_CROSSINGS && (link->busy & (1u << index)) &&
                link->crossing[index].generation == ticket >> 8
            ) {
                link_part(
                    dispatcher,
                    index,
                    island,
                    link_word(crossing, KMD_X_LEFT),
                    link_word(crossing, KMD_X_TICKS),
                    now
                );
            }
        }
        link->taken_from[from] = number;
        link->taken++;
    }
}

/* Reads every other island's part of the inbox. */
static void link_receive(kmd_t* dispatcher, uint64_t now) {
    kmd_link_t* link = link_state(dispatcher);
    for (uint32_t from = 0; from < link->islands; from++) {
        if (from != link->island) {
            link_receive_from(dispatcher, from, now);
        }
    }
}

/* The link ring's events, each a DONE packet to the island its command came from. */
static void link_answer(kmd_t* dispatcher) {
    kmd_link_t* link = link_state(dispatcher);
    kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_LINK];
    while (link->events_read != ring->pushed && link_out_room(link) != 0) {
        const uint32_t index = link->events_read + 1u;
        const uint64_t* event = (
            (const uint64_t*)(link->events + (index & (KMD_LINK_EVENTS - 1u)) * 16u)
        );
        const uint32_t request = link_word(event, 1);
        for (uint32_t k = 0; request != 0 && k < KMD_LINK_ORIGINS; k++) {
            if (link->origin[k].number == request) {
                link_queue(
                    link,
                    link->origin[k].island,
                    0,
                    KMD_CROSS_DONE,
                    link->origin[k].ticket,
                    link_word(event, 2),
                    link_word(event, 3)
                );
                link->origin[k].number = 0;
                link->answered++;
                break;
            }
        }
        link->events_read = index;
        link->read = index;
    }
}

/* One pass of the link: the inbox, the answers, the sends; before the link ring's turn
 * the deadline of its next line is the time it has left. */
static inline void link_turn(kmd_t* dispatcher, uint64_t now) {
    kmd_link_t* link = link_state(dispatcher);
    link_receive(dispatcher, now);
    kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_LINK];
    if (ring->line_state == 0) {
        ring->t_command = link->left[ring->next % KMD_LINK_DEPTH];
    }
    link_answer(dispatcher);
    link_send(dispatcher, now);
}

/* Every poll: crossings past their deadline get their TIMEOUT naming the far island;
 * the run goes on, the parts that come back later send nothing. */
static void link_late(kmd_t* dispatcher, uint64_t now) {
    kmd_link_t* link = link_state(dispatcher);
    for (uint32_t index = 0; index < KMD_CROSSINGS; index++) {
        kmd_crossing_t* crossing = &link->crossing[index];
        if (
            !(link->busy & (1u << index)) || crossing->late || now <= crossing->deadline
        ) {
            continue;
        }
        crossing->late = 1;
        link->gave_up++;
        link_gave(dispatcher, crossing->source);
        kmd_ring_t* ring = &dispatcher->ring[crossing->source];
        ring->busy_count--;
        const uint32_t auxiliary = (
            (KM_EVENT_ISLAND + crossing->far) |
            (crossing->sequence & KM_EVENT_SEQUENCE_MASK) << KM_EVENT_SEQUENCE_SHIFT
        );
        ring_push(
            ring,
            crossing->request,
            KM_EVENT_TIMEOUT | auxiliary << 8,
            (uint32_t)(now - crossing->submitted)
        );
        dispatcher->last_event_mtime = now;
    }
}

/* Whether the link has work: a crossing in flight, or packets not yet sent. */
static inline int link_active(kmd_t* dispatcher) {
    const kmd_link_t* link = link_state(dispatcher);
    return link->busy != 0 || link->out_head != link->out_tail || link->stages != 0;
}

/* The link's counters into the core block. */
static inline void link_count(kmd_t* dispatcher) {
    const kmd_link_t* link = link_state(dispatcher);
    dispatcher->x280_cores[KMD_H_LINK_SENT] = link->sent;
    dispatcher->x280_cores[KMD_H_LINK_TAKEN] = link->taken;
    dispatcher->x280_cores[KMD_H_LINK_CROSSED] = link->crossed;
    dispatcher->x280_cores[KMD_H_LINK_GAVE_UP] = link->gave_up;
}
