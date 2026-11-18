/* A dynamic run across islands whose messages cross as deliveries (KM_DELIVER): the
 * leader's waves decide its end; each island's run entry opens its deliveries. */
#pragma once

#include "dynamic.h"
#include "../kernel.h"
#include "link.h"
#include "x280_counts.h"

/* The islands of the run's group, in rank order; its leader holds the first rank. */
static void messages_islands(kmd_t* dispatcher) {
    kmd_messages_t* run = &dispatcher->messages;
    const kmd_link_t* link = link_state(dispatcher);
    const uint32_t first = run->group & 0xFFFFu, count = run->group >> 16;
    run->low = link_find(link, 0, first);
    run->high = link_find(link, 0, first + count - 1u);
    run->leader = run->low == link->island;
}

/* A run over islands: the X280 ranks' counts (kmd_x280_counts); else 0. */
static uint32_t messages_x280_counts(kmd_t* dispatcher) {
    const kmd_messages_t* run = &dispatcher->messages;
    if (!messages_spans(dispatcher)) {
        return 0;
    }
    const kmd_link_t* link = link_state(dispatcher);
    return (
        kmd_x280_counts(
            link->entries, run->low, run->high, link->island, run->group, run->counters
        )
    );
}

/* This island's run entry on link core 0 (KM_DELIVER_RUN): open with the run's id, or
 * closed (0); the link core delivers a line here only while open with the line's id. */
static void messages_entry(kmd_t* dispatcher, uint32_t id) {
    const kmd_messages_t* run = &dispatcher->messages;
    const uint32_t words[4]
        __attribute__((aligned(16))) = {id, run->sequence, run->group, ~id};
    kmd_store_chunk(link_state(dispatcher)->window + KM_DELIVER_RUN, words);
    x280_fence_ow();
}

/* Queues wave packet `kind` with `command` to every other island of the run. */
static inline void messages_wave_out(
    kmd_t* dispatcher, uint32_t kind, const uint64_t* command
) {
    const kmd_messages_t* run = &dispatcher->messages;
    kmd_link_t* link = link_state(dispatcher);
    for (uint32_t island = run->low; island <= run->high; island++) {
        if (island != link->island) {
            link_queue(link, island, command, kind, 0, 0, 0);
        }
    }
}

/* Island `island`'s bit in a round's answers (bit i: island low + i), 0 for none. */
static inline uint64_t messages_bit(const kmd_messages_t* run, uint32_t island) {
    const uint32_t i = island - run->low;
    return island >= run->low && island <= run->high && i < 64u ? 1ull << i : 0;
}

/* The bits of every island of the run but `here`: a whole round's answers. */
static inline uint64_t messages_others(const kmd_messages_t* run, uint32_t here) {
    uint64_t all = 0;
    for (uint32_t island = run->low; island <= run->high; island++) {
        all |= island != here ? messages_bit(run, island) : 0;
    }
    return all;
}

/* A wave packet from island `island`: a START kept for its part, a ROUND to answer, a
 * COUNT into the leader's round, the END. */
static void messages_wave(kmd_t* dispatcher, uint32_t island, const uint64_t* packet) {
    kmd_messages_t* run = &dispatcher->messages;
    const uint32_t kind = link_word(&packet[8], KMD_X_KIND) & 0xFFu;
    const uint32_t id = (uint32_t)packet[5], word = (uint32_t)(packet[5] >> 32);
    if (kind == KMD_CROSS_START) {
        run->told_id = id;
        run->told_group = word;
    } else if (kind == KMD_CROSS_ROUND) {
        run->asked = word;
        run->asked_id = id;
    } else if (kind == KMD_CROSS_COUNT) {
        const uint64_t bit = messages_bit(run, island);
        if (run->leader && run->id == id && run->round == word && bit != 0 &&
            (run->answered & bit) == 0) { /* an island's first answer of the round */
            run->answers++;
            run->answered |= bit;
            run->quiet &= (uint32_t)packet[7];
            run->sent += packet[1];
            run->consumed += packet[3];
        }
    } else if (kind == KMD_CROSS_END && run->id == id && id != 0) {
        run->ended = 1;
    }
}

/* At the run's deadline, the island whose wave did not come: the leader's, the lowest
 * not answering its round; another island's, the leader (no START, ROUND or END). */
static uint32_t messages_waited(kmd_t* dispatcher) {
    const kmd_messages_t* run = &dispatcher->messages;
    const uint32_t here = link_state(dispatcher)->island;
    if (!run->active || !messages_spans(dispatcher) || run->ended) {
        return KMD_ISLAND_NONE;
    }
    if (!run->leader) {
        return run->low;
    }
    if (run->round == 0 || run->answered == messages_others(run, here)) {
        return KMD_ISLAND_NONE; /* no round out: this island's own cores are late */
    }
    for (uint32_t island = run->low; island <= run->high; island++) {
        const uint32_t i = island - run->low;
        if (island != here && (i >= 64u || !(run->answered >> i & 1u))) {
            return island;
        }
    }
    return KMD_ISLAND_NONE;
}

/* Ticks between two rounds of the leader (10 us). */
#define KMD_WAVE_TICKS 500u

/* The waves, a step a pass, 1 once ended: the leader gives the run its id, then asks
 * rounds; two in a row all quiet, with equal sums and sent == consumed, end it. */
static int messages_waves(kmd_t* dispatcher) {
    kmd_messages_t* run = &dispatcher->messages;
    kmd_link_t* link = link_state(dispatcher);
    const uint32_t others = run->high - run->low;
    uint64_t command[8] = {0};
    uint64_t sent = 0, consumed = 0;
    if (run->ended) {
        return 1;
    }
    if (run->id == 0) {
        if (run->leader) {
            if (link_out_room(link) < others) {
                return 0;
            }
            run->id = run->sequence << 8 | link->island;
            run->id = run->id != 0 ? run->id : 0x100u;
            command[5] = run->id | (uint64_t)run->group << 32;
            messages_wave_out(dispatcher, KMD_CROSS_START, command);
        } else if (run->told_id != 0 && run->told_group == run->group) {
            run->id = run->told_id;
            run->told_id = 0;
        } else {
            return 0;
        }
        messages_entry(dispatcher, run->id);
    }
    if (!run->leader) {
        if (run->asked != 0 && link_out_room(link) != 0) {
            const int quiet = messages_sample(dispatcher, &sent, &consumed);
            command[1] = sent;
            command[3] = consumed;
            command[5] = run->asked_id | (uint64_t)run->asked << 32;
            command[7] = (uint64_t)(quiet && run->asked_id == run->id);
            link_queue(link, run->low, command, KMD_CROSS_COUNT, 0, 0, 0);
            run->asked = 0;
        }
        return 0;
    }
    if (run->round != 0 && run->answered != messages_others(run, link->island)) {
        return 0;
    }
    if (run->round != 0) { /* a whole round */
        if (
            run->quiet && run->have_last && run->sent == run->last_sent &&
            run->consumed == run->last_consumed && run->sent == run->consumed
        ) {
            if (link_out_room(link) < others) {
                return 0;
            }
            command[5] = run->id;
            messages_wave_out(dispatcher, KMD_CROSS_END, command);
            run->ended = 1;
            return 1;
        }
        run->have_last = run->quiet;
        run->last_sent = run->sent;
        run->last_consumed = run->consumed;
        run->round = 0;
    }
    const uint64_t now = x280_mtime();
    if (now < run->next_round || link_out_room(link) < others) {
        return 0;
    }
    run->quiet = (uint32_t)messages_sample(dispatcher, &sent, &consumed);
    if (!run->quiet) {
        run->have_last = 0;
        return 0; /* this island is not quiet: no round yet */
    }
    run->sent = sent;
    run->consumed = consumed;
    run->round = run->round_number = run->round_number + 1u;
    run->answers = 0;
    run->answered = 0;
    run->next_round = now + KMD_WAVE_TICKS;
    command[5] = run->id | (uint64_t)run->round << 32;
    messages_wave_out(dispatcher, KMD_CROSS_ROUND, command);
    return 0;
}
