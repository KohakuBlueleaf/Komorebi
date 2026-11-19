/* Core 0's side of the work clusters' rings (sources 2 to 4), and the commands any
 * source sends to a cluster's thread, with their answers and deadlines. */
#pragma once

#include "completion.h"
#include "../kernel.h"
#include "ring.h"

static inline int work_pow2(uint64_t value) {
    return value != 0 && (value & (value - 1u)) == 0;
}

/* Whether the work rings are usable: none (work_cmds 0), or all lines after the upload
 * region inside LIM and each remote area in use inside one 2 MiB window. */
static inline int work_args_valid(const volatile kmd_args_t* arguments) {
    if (arguments->work_cmds == 0) {
        return 1;
    }
    const uint64_t depth = arguments->work_depth;
    const uint64_t event_depth = arguments->work_event_depth;
    const uint64_t remote = KMD_WORK_RUN(event_depth) + KMD_WORK_RUN_BYTES;
    int ok = (
        work_pow2(depth) && depth <= KMD_WORK_DEPTH_MAX && work_pow2(event_depth) &&
        event_depth <= KMD_WORK_EVENTS_MAX && (arguments->work_cmds & 63u) == 0 &&
        arguments->work_cmds >= KMD_LIM_UPLOAD + arguments->upload_bytes &&
        (uintptr_t)arguments + arguments->work_cmds +
                KMD_WORK_CLUSTERS * KMD_WORK_RING_BYTES(depth) <=
            X280_LIM_END
    );
    for (uint32_t ring_index = 0; ring_index < KMD_WORK_CLUSTERS; ring_index++) {
        const uint64_t noc = arguments->work_noc[ring_index];
        const uint64_t address = arguments->work_noc_addr[ring_index];
        ok = (
            ok &&
            (
                address == 0 ||
                (
                    (address & 63u) == 0 &&
                    (address & (NOC_2M_WINDOW_STRIDE - 1u)) + remote <=
                        NOC_2M_WINDOW_STRIDE &&
                    (noc & 0xFFFFu) <= NOC_MAX_COORD && (noc >> 16) <= NOC_MAX_COORD
                )
            )
        );
    }
    return ok;
}

/* Programs ring w's window onto its cluster's area and points source KMD_SRC_WORK + w
 * at both parts. Returns KMD_ERROR_NONE (also with no ring) or KMD_ERROR_WINDOW. */
static inline uint32_t work_open_one(
    kmd_t* dispatcher, uint32_t ring_index, uint64_t* line
) {
    const volatile kmd_args_t* arguments = dispatcher->args;
    if (arguments->work_cmds == 0 || arguments->work_noc_addr[ring_index] == 0) {
        return KMD_ERROR_NONE;
    }
    noc_tlb_2m_t programmed;
    volatile uint8_t* window = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            KMD_WIN_WORK + ring_index,
            (uint32_t)(arguments->work_noc[ring_index] & 0xFFFFu),
            (uint32_t)(arguments->work_noc[ring_index] >> 16),
            arguments->work_noc_addr[ring_index],
            NOC_ORDER_POSTED,
            &programmed
        )
    );
    if (window == 0) {
        return KMD_ERROR_WINDOW;
    }
    dispatcher->opened |= 1u << (3u + ring_index);
    if (noc_verify_tlb_2m(KMD_WIN_WORK + ring_index, &programmed) != 0) {
        return KMD_ERROR_WINDOW;
    }
    kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_WORK + ring_index];
    ring->depth = (uint32_t)arguments->work_depth;
    ring->event_depth = (uint32_t)arguments->work_event_depth;
    ring->commands = (
        dispatcher->lim + arguments->work_cmds +
        ring_index * KMD_WORK_RING_BYTES(ring->depth)
    );
    ring->read = (volatile const uint32_t*)(ring->commands + ring->depth * 64u);
    ring->consumed = (volatile uint32_t*)(window + KMD_WORK_CONSUMED);
    ring->told = (volatile uint32_t*)(window + KMD_WORK_TOLD);
    ring->events = window + KMD_WORK_EVENTS;
    ring->next = 1;
    ring->t_command = dispatcher->ring[KMD_SRC_HOST].t_command;
    ring->line = line;
    dispatcher->thread[ring_index].lines = window + KMD_WORK_IN(ring->event_depth);
    dispatcher->thread[ring_index].next = 1;
    return KMD_ERROR_NONE;
}

/* Every ring in use; `lines` holds one 64-byte aligned command line per ring. */
static inline uint32_t work_open(kmd_t* dispatcher, uint64_t* lines) {
    for (uint32_t ring_index = 0; ring_index < KMD_WORK_CLUSTERS; ring_index++) {
        if (
            work_open_one(dispatcher, ring_index, lines + 8u * ring_index) !=
            KMD_ERROR_NONE
        ) {
            return KMD_ERROR_WINDOW;
        }
    }
    return KMD_ERROR_NONE;
}

/* Finish: each ring's last event count goes out, then its window is cleared. */
static inline void work_close(kmd_t* dispatcher) {
    for (uint32_t ring_index = 0; ring_index < KMD_WORK_CLUSTERS; ring_index++) {
        if (dispatcher->opened & (1u << (3u + ring_index))) {
            ring_flush(&dispatcher->ring[KMD_SRC_WORK + ring_index]);
            x280_fence_ow();
            noc_clear_tlb_2m(KMD_WIN_WORK + ring_index);
            dispatcher->opened &= ~(1u << (3u + ring_index));
            dispatcher->ring[KMD_SRC_WORK + ring_index].events = 0;
        }
    }
}

/* Stores slot `slot`'s line into the thread's LIM: four 16-byte stores, as the events
 * (no 64-byte store into that LIM is relied on), then a fence. */
static inline void work_store(kmd_thread_t* thread, uint32_t slot) {
    for (uint32_t chunk = 0; chunk < 4; chunk++) {
        kmd_store_chunk(
            thread->lines + slot * 64u + 16u * chunk,
            (const uint32_t*)&thread->line[slot][2u * chunk]
        );
    }
    x280_fence_ow();
}

/* A command of `source` to a work cluster: the thread's next line (stored if it serves)
 * or rejected; with every line unanswered, busy (the host's: KMD_REJECT_THREAD). */
static __attribute__((noinline)) kmd_take_t
work_command(kmd_t* dispatcher, uint32_t source, uint64_t now) {
    kmd_ring_t* ring = &dispatcher->ring[source];
    uint64_t* line = ring->line;
    const uint32_t request = ring->next, header = (uint32_t)(line[0] >> 32);
    const uint32_t flags = (header >> 8) & 0xFFu, group = (uint32_t)line[1];
    /* on an island, thread numbers are flat: this island's first is its cluster 1 */
    const uint32_t cluster = (
        (group & 0xFFFFu) - KM_GROUP_CLUSTER_BASE -
        (dispatcher->thread_first ? dispatcher->thread_first - 1u : 0u)
    );
    uint32_t reject = 0;
    if (flags & (1u | KM_FLAG_EXCLUSIVE | KM_FLAG_QUEUE)) {
        reject = KMD_REJECT_FLAGS;
    } else if (dispatcher->exited) {
        reject = KMD_REJECT_EXITED;
    } else if (
        cluster < 1 || cluster > KMD_WORK_CLUSTERS || (group >> 16) != 1 ||
        dispatcher->thread[cluster - 1].lines == 0
    ) {
        reject = KMD_REJECT_GROUP;
    }
    kmd_thread_t* thread = &dispatcher->thread[reject ? 0 : cluster - 1];
    const uint32_t number = thread->next, slot = number & (KMD_WORK_IN_DEPTH - 1u);
    if (reject == 0 && (thread->busy & (1u << slot))) {
        if (source != KMD_SRC_HOST) {
            return KMD_HELD; /* rejected busy with the wake owed (source_busy) */
        }
        reject = KMD_REJECT_THREAD;
    }
    ring_consumed(ring);
    dispatcher->last_event_mtime = now;
    if (reject != 0) {
        ring_push(ring, request, KM_EVENT_REJECTED | (reject << 8), 0);
        dispatcher->rejected++;
        return KMD_TAKEN;
    }
    if (dispatcher->first_command_mtime == 0) { /* or at a thread's: command_submit */
        dispatcher->first_command_mtime = now;
    }
    uint32_t sequence = ++dispatcher->sequence;
    if (sequence == 0) {
        sequence = ++dispatcher->sequence;
    }
    uint64_t* kept = thread->line[slot];
    kept[0] = number | (uint64_t)(header & 0xFF00FFFFu) << 32;
    for (uint32_t word = 1; word < 8; word++) {
        kept[word] = (
            word % 2 == 0 ? (line[word] & ~(uint64_t)0xFFFFFFFFu) | number : line[word]
        );
    }
    if (thread->serving) {
        work_store(thread, slot);
    } else {
        thread->unstored |= 1u << slot;
    }
    thread->next = number + 1u;
    thread->number[slot] = number;
    thread->request[slot] = request;
    thread->sequence[slot] = sequence;
    thread->source[slot] = (uint8_t)source;
    thread->submitted[slot] = now;
    thread->deadline[slot] = now + ring->t_command;
    thread->busy |= 1u << slot;
    thread->result = (
        (thread->result & ~(1u << slot)) | ((flags & KM_FLAG_RESULT) ? 1u << slot : 0u)
    );
    ring->busy_count++;
    dispatcher->thread_busy++;
    return KMD_TAKEN;
}

/* Cluster `ring_index + 1`'s answer to line `number`: the command's DONE event (ticks:
 * the word with KM_FLAG_RESULT, else the time) unless its timeout went out. */
KMD_COLD __attribute__((noinline)) void work_answer(
    kmd_t* dispatcher, uint32_t ring_index, uint32_t number, uint32_t word, uint64_t now
) {
    kmd_thread_t* thread = &dispatcher->thread[ring_index];
    if (number == 0 && !thread->serving) {
        /* the thread serves: the lines taken before, oldest first */
        thread->serving = 1;
        for (uint32_t back = KMD_WORK_IN_DEPTH; back > 0; back--) {
            const uint32_t slot = (thread->next - back) & (KMD_WORK_IN_DEPTH - 1u);
            if (thread->unstored & (1u << slot)) {
                work_store(thread, slot);
            }
        }
        thread->unstored = 0;
        if (thread->run_unstored) {
            messages_run_store(thread);
        }
        return;
    }
    const uint32_t slot = number & (KMD_WORK_IN_DEPTH - 1u), bit = 1u << slot;
    if (!(thread->busy & bit) || thread->number[slot] != number) {
        return;
    }
    thread->busy &= ~bit;
    dispatcher->thread_busy--;
    if (thread->late & bit) {
        thread->late &= ~bit;
    } else {
        kmd_ring_t* ring = &dispatcher->ring[thread->source[slot]];
        uint32_t ticks = (uint32_t)(now - thread->submitted[slot]);
        if (thread->result & bit) {
            dispatcher->last_result = ticks = word;
            dispatcher->results++;
        }
        const uint32_t auxiliary = (
            (ring_index + 1u) |
            (thread->sequence[slot] & KM_EVENT_SEQUENCE_MASK) << KM_EVENT_SEQUENCE_SHIFT
        );
        ring->busy_count--;
        ring_push(ring, thread->request[slot], KM_EVENT_DONE | auxiliary << 8, ticks);
        dispatcher->last_event_mtime = now;
    }
    if (dispatcher->waits) {
        sources_wake(dispatcher);
    }
}

/* Every unanswered line whose deadline passed at `now` gets its KM_EVENT_TIMEOUT
 * (auxiliary: the cluster); the kernel goes on, the line stays taken until answered. */
KMD_COLD __attribute__((noinline)) void work_late(kmd_t* dispatcher, uint64_t now) {
    for (uint32_t ring_index = 0; ring_index < KMD_WORK_CLUSTERS; ring_index++) {
        kmd_thread_t* thread = &dispatcher->thread[ring_index];
        for (uint32_t slot = 0; slot < KMD_WORK_IN_DEPTH; slot++) {
            const uint32_t bit = 1u << slot;
            if (
                (thread->busy & ~thread->late & bit) == 0 ||
                now <= thread->deadline[slot]
            ) {
                continue;
            }
            thread->late |= bit;
            kmd_ring_t* ring = &dispatcher->ring[thread->source[slot]];
            const uint32_t auxiliary = (
                (ring_index + 1u) |
                (thread->sequence[slot] & KM_EVENT_SEQUENCE_MASK)
                    << KM_EVENT_SEQUENCE_SHIFT
            );
            ring->busy_count--;
            ring_push(
                ring,
                thread->request[slot],
                KM_EVENT_TIMEOUT | auxiliary << 8,
                (uint32_t)(now - thread->submitted[slot])
            );
            dispatcher->last_event_mtime = now;
        }
    }
}

/* Lines handed to threads that neither got an answer nor a timeout: no event. */
KMD_COLD uint32_t work_abandoned(const kmd_t* dispatcher) {
    uint32_t abandoned = 0;
    for (uint32_t ring_index = 0; ring_index < KMD_WORK_CLUSTERS; ring_index++) {
        const kmd_thread_t* thread = &dispatcher->thread[ring_index];
        for (uint32_t slot = 0; slot < KMD_WORK_IN_DEPTH; slot++) {
            abandoned += (thread->busy & ~thread->late) >> slot & 1u;
        }
    }
    return abandoned;
}
