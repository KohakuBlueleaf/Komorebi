/* Komorebi-nn, the executor: one X280 thread that walks a list of entries (nn.h), each
 * one command submitted and waited for, or a run of records as one chain command. */
#include <stdint.h>

#include "kmx.h"
#include "nn.h"

_Static_assert(KM_TRANSFER_NONCE == 7u, "a transfer's nonce is its a7");

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* one command's submit or wait */

typedef struct {
    const volatile uint32_t* list; /* the list region; the host writes it */
    uint64_t bytes;
    uint32_t base; /* the list's bank address: an entry's address less it, its offset */
    uint32_t* slot;
    uint32_t slot_index;
    uint64_t terms[NN_TERMS];
    uint32_t entries, polls, stale, sequence;
    uint32_t nonce; /* RUN's a3: a transfer entry's nonce word is XORed with it */
} walk_t;

/* The walk of a run: its regions, the stats slot `slot`; KMX_BAD for a bound. */
static int walk_start(walk_t* walk, uint32_t slot, uint32_t sequence) {
    uint64_t stats_bytes = 0;
    km_command_t block;
    walk->list = (const volatile uint32_t*)kmx_region(NN_REGION_LIST, &walk->bytes);
    uint32_t* stats = (uint32_t*)kmx_region(NN_REGION_STATS, &stats_bytes);
    if (
        walk->list == 0 || stats == 0 || slot >= NN_SLOTS ||
        4ull * NN_SLOT_WORDS * NN_SLOTS > stats_bytes || walk->bytes < NN_LIST_BYTES ||
        kmx_block(&block, NN_REGION_LIST, 0u, 64u) != KMX_OK
    ) {
        return KMX_BAD;
    }
    walk->bytes = NN_LIST_BYTES; /* an entry's index is below NN_ENTRIES_MAX */
    walk->base = block.a7;
    walk->slot = stats + NN_SLOT_WORDS * slot;
    walk->slot_index = slot;
    for (uint32_t term = 0u; term < NN_TERMS; term++) {
        walk->terms[term] = 0ull;
    }
    walk->entries = walk->polls = walk->stale = 0u;
    walk->sequence = sequence;
    walk->nonce = 0u;
    return KMX_OK;
}

/* The offset in the list of the entry at bank address `address`: KMX_BAD outside. */
static int entry_at(const walk_t* walk, uint32_t address, uint32_t* offset) {
    const uint32_t at = address - walk->base;
    if (
        address < walk->base || at % NN_ENTRY_BYTES != 0u ||
        (uint64_t)at + NN_ENTRY_BYTES > walk->bytes
    ) {
        return KMX_BAD;
    }
    *offset = at;
    return KMX_OK;
}

/* Whether the entry at `offset`, its lines read again, is the open list's new one: its
 * sequence and its check (nn.h); the record words of a chain checked with it. */
static int entry_new(const walk_t* walk, uint32_t offset) {
    kmx_invalidate_at(NN_REGION_LIST, offset, NN_ENTRY_BYTES);
    const volatile uint32_t* entry = walk->list + offset / 4u;
    uint32_t sum = 0u;
    for (uint32_t word = 0u; word < NN_ENTRY_WORDS; word++) {
        if (word != NN_ENTRY_NEXT && word != NN_ENTRY_CHECK) {
            sum += entry[word] * NN_CHECK_FACTOR(word);
        }
    }
    return entry[NN_ENTRY_SEQUENCE] == walk->sequence && entry[NN_ENTRY_CHECK] == sum;
}

/* Waits, from `started` for at most `bound` ticks, until the entry at `offset` and, a
 * chain, every record after it are new: KMX_OK, KMX_LATE, or KMX_BAD for a bound. */
static int entries_new(
    walk_t* walk, uint32_t offset, uint64_t started, uint32_t bound
) {
    uint32_t at = offset;
    for (uint32_t record = 0u; record < NN_ENTRIES_MAX; record++) {
        while (!entry_new(walk, at)) {
            walk->stale++;
            if (kmx_time() - started > bound || kmx_stop_requested()) {
                return KMX_LATE;
            }
        }
        const volatile uint32_t* entry = walk->list + at / 4u;
        const uint32_t next = entry[kChainNext];
        if (
            (
                at == offset && walk->list[offset / 4u + NN_ENTRY_KIND] != NN_KIND_CHAIN
            ) ||
            next == 0u
        ) {
            return KMX_OK;
        }
        if (entry_at(walk, next, &at) != KMX_OK) {
            return KMX_BAD;
        }
    }
    return KMX_BAD;
}

/* The command of the entry at `offset` (a COMMAND or a CHAIN): KMX_BAD for a kind
 * that runs nothing. */
static int entry_command(
    const walk_t* walk, uint32_t offset, uint32_t kind, km_command_t* command
) {
    const volatile uint32_t* entry = walk->list + offset / 4u;
    const uint32_t header = entry[NN_ENTRY_HEADER];
    const uint32_t group = (
        entry[kind == NN_KIND_CHAIN ? NN_ENTRY_CHAIN_GROUP : kChainGroup]
    );
    kmx_command(command, header >> 24, header & 0xFFu, group & 0xFFFFu, group >> 16);
    command->hdr |= header & 0xFF00u;
    if (kind == NN_KIND_CHAIN) {
        command->a0 = entry[NN_ENTRY_CHAIN_TOKEN];
        command->a1 = entry[NN_ENTRY_CHAIN_AHEAD];
        command->a4 = entry[NN_ENTRY_CHAIN_MODE];
        return kmx_block(command, NN_REGION_LIST, offset, 64u);
    }
    if (kind != NN_KIND_COMMAND && kind != NN_KIND_TRANSFER) {
        return KMX_BAD;
    }
    command->a0 = entry[kChainArguments + 0];
    command->a1 = entry[kChainArguments + 1];
    command->a2 = entry[kChainArguments + 2];
    command->a3 = entry[kChainArguments + 3];
    command->a4 = entry[kChainArguments + 4];
    command->a5 = entry[kChainArguments + 5];
    command->a6 = entry[kChainArguments + 6];
    command->a7 = entry[kChainArguments + 7];
    command->a8 = entry[kChainArguments + 8];
    command->a9 = entry[kChainArguments + 9];
    if (kind == NN_KIND_TRANSFER) {
        command->a7 ^= walk->nonce; /* KM_TRANSFER_NONCE */
    }
    return KMX_OK;
}

/* Runs the entry at `offset` (fetch begun at `started`): its command submitted, a full
 * ring retried, and waited for. KMX_OK once done; an END entry runs nothing. */
static int run_entry(
    walk_t* walk, uint32_t offset, uint64_t started, uint32_t* kind, uint32_t* next
) {
    const volatile uint32_t* entry = walk->list + offset / 4u;
    *kind = entry[NN_ENTRY_KIND];
    *next = entry[NN_ENTRY_NEXT];
    if (*kind == NN_KIND_END) {
        walk->terms[NN_TERM_FETCH] += kmx_time() - started;
        return KMX_OK;
    }
    km_command_t command;
    if (entry_command(walk, offset, *kind, &command) != KMX_OK) {
        return KMX_BAD;
    }
    const uint64_t built = kmx_time();
    uint32_t handle = 0u;
    const uint64_t deadline = built + WAIT_TICKS;
    int status = kmx_submit(&command, &handle);
    while (status == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
        status = kmx_submit(&command, &handle);
    }
    if (status != KMX_OK) {
        return status;
    }
    const uint64_t submitted = kmx_time();
    kmx_event event;
    status = kmx_wait(handle, submitted + WAIT_TICKS, &event);
    const uint64_t done = kmx_time();
    if (status != KMX_OK) {
        return status;
    }
    const uint32_t index = offset / NN_ENTRY_BYTES;
    if (event.kind != KM_EVENT_DONE) { /* its kind and auxiliary where its ticks go */
        walk->slot[NN_ST_ENTRY_TICKS + index] = event.kind | event.auxiliary << 8;
        return KMX_BAD;
    }
    walk->terms[NN_TERM_FETCH] += built - started;
    walk->terms[NN_TERM_SUBMIT] += submitted - built;
    walk->terms[NN_TERM_WAIT] += done - submitted;
    walk->terms[NN_TERM_COMMAND] += event.ticks;
    walk->slot[NN_ST_ENTRY_TICKS + index] = event.ticks;
    walk->entries++;
    return KMX_OK;
}

/* The run's stats into its slot (flushed): its result word. */
static uint32_t walk_end(walk_t* walk, int status, uint64_t ticks) {
    uint32_t* slot = walk->slot;
    walk->terms[NN_TERM_TOTAL] = ticks;
    slot[NN_ST_STATUS] = (uint32_t)status;
    slot[NN_ST_ENTRIES] = walk->entries;
    slot[NN_ST_POLLS] = walk->polls;
    slot[NN_ST_STALE] = walk->stale;
    for (uint32_t term = 0u; term < NN_TERMS; term++) {
        slot[NN_ST_TERMS + 2u * term] = (uint32_t)walk->terms[term];
        slot[NN_ST_TERMS + 2u * term + 1u] = (uint32_t)(walk->terms[term] >> 32);
    }
    const uint64_t bytes = 4ull * NN_SLOT_WORDS;
    kmx_flush_at(NN_REGION_STATS, bytes * walk->slot_index, bytes);
    return status == KMX_OK ? (uint32_t)ticks : NN_FAILED | (uint32_t)status;
}

/* RUN: the closed list from entry a0 to its end, a1 times; a3 its transfers' nonce. */
static uint32_t run_list(const km_command_t* command) {
    walk_t walk;
    if (walk_start(&walk, command->a2, 0u) != KMX_OK) {
        return NN_FAILED | KMX_BAD;
    }
    walk.nonce = command->a3;
    kmx_invalidate(NN_REGION_LIST); /* the host wrote it, then a barrier */
    const uint64_t started = kmx_time();
    int status = KMX_OK;
    for (uint32_t time = 0u; time < command->a1 && status == KMX_OK; time++) {
        uint32_t address = command->a0, count = 0u;
        while (address != 0u && status == KMX_OK) {
            uint32_t offset = 0u, kind = 0u;
            status = (
                count++ < NN_ENTRIES_MAX ? entry_at(&walk, address, &offset) : KMX_BAD
            );
            if (status == KMX_OK) {
                status = run_entry(&walk, offset, kmx_time(), &kind, &address);
            }
            if (status == KMX_OK && kind == NN_KIND_END) {
                status = KMX_BAD; /* a closed list ends at a next of 0 */
            }
        }
        kmx_progress(time);
    }
    return walk_end(&walk, status, kmx_time() - started);
}

/* DRAIN: the open list after the anchor a0, each entry once its link is written and it
 * is new, until an END entry; a2 the idle ticks waited, a3 its number (nn.h). */
static uint32_t drain(const km_command_t* command) {
    walk_t walk;
    if (
        walk_start(&walk, command->a1, command->a3) != KMX_OK || command->a2 == 0u ||
        command->a3 == 0u
    ) {
        return NN_FAILED | KMX_BAD;
    }
    const uint64_t started = kmx_time();
    uint32_t offset = 0u, kind = NN_KIND_NONE;
    int status = entry_at(&walk, command->a0, &offset);
    while (status == KMX_OK && kind != NN_KIND_END) {
        const volatile uint32_t* entry = walk.list + offset / 4u;
        const uint64_t polled = kmx_time();
        uint32_t next = 0u;
        for (;;) {
            kmx_invalidate_at(NN_REGION_LIST, offset + 64u, 64u);
            next = entry[NN_ENTRY_NEXT];
            walk.polls++;
            if (
                next != 0u || kmx_time() - polled > command->a2 || kmx_stop_requested()
            ) {
                break;
            }
        }
        const uint64_t linked = kmx_time();
        walk.terms[NN_TERM_IDLE] += linked - polled;
        status = next != 0u ? entry_at(&walk, next, &offset) : KMX_LATE;
        if (status == KMX_OK) {
            status = entries_new(&walk, offset, linked, command->a2);
        }
        const uint64_t seen = kmx_time();
        walk.terms[NN_TERM_CHECK] += seen - linked;
        if (status == KMX_OK) {
            status = run_entry(&walk, offset, seen, &kind, &next);
        }
        kmx_progress(walk.entries);
    }
    return walk_end(&walk, status, kmx_time() - started);
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerRun:
            return run_list(command);
        case kHandlerDrain:
            return drain(command);
    }
    return 0u;
}
