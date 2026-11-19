// The dispatcher core's compute kernel, its TRISC 1 part (td_compute.cpp): a thread's
// kmx calls over its area in L1 (kTa*) with KM_TD_THREAD, its sources after this.
#pragma once

#if !defined(KM_TD_THREAD)
namespace td {

    inline void thread() {
    }  // no thread on this load

}  // namespace td
#else

#include <stdint.h>

#include "api/compute/common.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "kmx.h"
#include "komorebi/device/core/km_core.hpp"

// The X280's ring calls (kmx_ring.h) over L1: no cache to sweep, Blackhole's
// invalidated.
#define KMX_RING_FENCE_W() asm volatile("" ::: "memory")
#define KMX_RING_FENCE_R() invalidate_l1_cache()
#define KMX_RING_SWEEP(address, bytes) ((void)(address), (void)(bytes))
#define KMX_RING_NOW() kmx_time()
#define KMX_RING_END() (kmx_stop_requested() ? KMX_STOPPED : KMX_OK)
#include "kmx_ring.h"

static_assert(kTaLim + KMX_LIM_BYTES == kTaRing, "the thread's area");
static_assert(sizeof(kmx_ring_t) <= kTaBytes - kTaState, "the thread's ring state");
static_assert(
    KMX_RING_IN_AFTER(kTaEventDepth) + KMD_WORK_EVENTS == KMD_WORK_IN(kTaEventDepth)
);

// A call a body may not make here: its build fails naming the call and why.
#define KMX_TD_NOT(why) __attribute__((error("not on the dispatcher's core: " why)))
#define KMX_TD_ONE KMX_TD_NOT("one TRISC, not four X280 cores")
#define KMX_TD_NOC KMX_TD_NOT("no NoC of its own, this core's L1 only")
#define KMX_TD_RING \
    KMX_TD_NOT("its command ring serves kmx_submit, kmx_wait and kmx_poll alone")
#define KMX_TD_REGION \
    KMX_TD_NOT("regions in the dispatcher core's L1 are not built yet")

extern "C" {
void* kmx_region(int id, uint64_t* bytes) KMX_TD_REGION;
void kmx_invalidate(int id) KMX_TD_REGION;
void kmx_flush(int id) KMX_TD_REGION;
void kmx_invalidate_at(int id, uint64_t offset, uint64_t bytes) KMX_TD_REGION;
void kmx_flush_at(int id, uint64_t offset, uint64_t bytes) KMX_TD_REGION;
void kmx_evict(void) KMX_TD_NOC;
int kmx_block(
    km_command_t* command, int id, uint64_t offset, uint32_t bytes
) KMX_TD_REGION;
int kmx_load(
    uint32_t program,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
) KMX_TD_RING;
int kmx_spawn(int core, void (*function)(int core, void* arg), void* arg) KMX_TD_ONE;
int kmx_join(int core) KMX_TD_ONE;
int kmx_parallel(void (*function)(int core, void* arg), void* arg) KMX_TD_ONE;
uint32_t kmx_clusters(void) KMX_TD_ONE;
int kmx_lead(void) KMX_TD_ONE;
int kmx_parallel_all(
    void (*function)(int cluster, int core, uint64_t arg), uint64_t arg
) KMX_TD_ONE;
int kmx_serve(void) KMX_TD_ONE;
int kmx_push(
    int cluster, int id, uint64_t offset, const void* source, uint64_t bytes
) KMX_TD_NOC;
int kmx_fetch(
    int id,
    uint64_t offset,
    const kmx_dram* from,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
) KMX_TD_NOC;
int kmx_fetch_clean(
    int id,
    uint64_t offset,
    const kmx_dram* from,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
) KMX_TD_NOC;
int kmx_store(
    int id,
    uint64_t offset,
    const kmx_dram* to,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
) KMX_TD_NOC;
int kmx_fetch_submit(
    int id,
    uint64_t offset,
    const kmx_dram* from,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint32_t* handle
) KMX_TD_NOC;
int kmx_store_submit(
    int id,
    uint64_t offset,
    const kmx_dram* to,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint32_t* handle
) KMX_TD_NOC;
int kmx_port_cores(void) KMX_TD_NOC;
int kmx_port_owner(int core) KMX_TD_NOC;
int kmx_port_tell(int core, uint32_t first, uint32_t second, uint32_t third) KMX_TD_NOC;
int kmx_port_asked(int core, kmx_message* message) KMX_TD_NOC;
int kmx_port_answer(
    int core, uint32_t sequence, uint32_t first, uint32_t second, uint32_t third
) KMX_TD_NOC;
int kmx_port_told(int core, kmx_message* message) KMX_TD_NOC;
int kmx_port_run(
    const km_command_t* command,
    void (*function)(int cluster, int x280_core, uint64_t arg),
    uint64_t arg,
    kmx_event* event
) KMX_TD_NOC;
int kmx_port_open(void) KMX_TD_NOC;
int kmx_pull_run(
    const km_command_t* command, const kmx_pull* settings, kmx_event* event
) KMX_TD_NOC;
int kmx_ranks(void) KMX_TD_RING;
int kmx_messages_run(
    void (*function)(int x280_core, kmx_messages* messages, void* arg),
    void* arg,
    uint64_t deadline
) KMX_TD_RING;
const volatile uint32_t* kmx_messages_next(kmx_messages* messages) KMX_TD_RING;
int kmx_messages_forward(
    kmx_messages* messages, uint32_t rank, const uint32_t* message, uint32_t handler
) KMX_TD_RING;
int kmx_messages_send(
    kmx_messages* messages, uint32_t rank, const uint32_t* message, uint32_t handler
) KMX_TD_RING;
void kmx_messages_consume(kmx_messages* messages) KMX_TD_RING;
}

namespace td {

    // The thread's area and the words it watches, from the dispatcher's arguments.
    struct Thread {
        uint32_t area = 0;
        uint32_t scale = 0;  // the X280's ticks a wall-clock tick, times 2^32
        uint32_t over = 0;   // in's kTwOver: the load's tag once the dispatcher ended
        uint32_t tag = 0;

        bool ended() const {
            invalidate_l1_cache();
            return km::l1(over)[0] == tag;
        }

        volatile tt_l1_ptr uint32_t* start() const {
            return km::l1(area + kTaStart);
        }

        volatile tt_l1_ptr uint32_t* lim(uint32_t offset) const {
            return km::l1(area + kTaLim + offset);
        }

        // The thread's side of its ring, in L1 after the work area (not its local
        // memory).
        kmx_ring_t* ring() const {
            return reinterpret_cast<kmx_ring_t*>(area + kTaState);
        }

        volatile tt_l1_ptr kmx_result_block* result() const {
            return (
                reinterpret_cast<volatile tt_l1_ptr kmx_result_block*>(
                    lim(KMX_LIM_RESULT)
                )
            );
        }

        // The X280's 50 MHz timer from the wall clock, counted from kmx_main.
        uint64_t time() const {
            const uint64_t started = (
                static_cast<uint64_t>(start()[kTaStarted + 1]) << 32 |
                start()[kTaStarted]
            );
            const uint64_t wall = km::wall64() - started;
            const uint64_t low = (wall & 0xFFFFFFFFu) * scale;
            return (wall >> 32) * scale + (low >> 32);
        }
    };

    inline Thread& here() {
        static Thread one;
        return one;
    }

}  // namespace td

extern "C" {

const kmx_args* kmx_args_get(void) {
    return (
        const_cast<const kmx_args*>(
            reinterpret_cast<volatile tt_l1_ptr kmx_args*>(td::here().lim(KMX_LIM_ARGS))
        )
    );
}

int kmx_cluster(void) {
    return 1;
}

int kmx_island(void) {
    return 0;
}

int kmx_islands(void) {
    return 1;
}

int kmx_island_of(uint32_t number) {
    return number >= 1 && number <= 3 ? 0 : -1;
}

uint64_t kmx_time(void) {
    return td::here().time();
}

int kmx_stop_requested(void) {
    const td::Thread& thread = td::here();
    invalidate_l1_cache();
    return thread.lim(KMX_LIM_STOP)[0] == KMX_STOP_WORD || thread.ended();
}

void kmx_progress(uint32_t where) {
    volatile tt_l1_ptr kmx_result_block* block = td::here().result();
    block->where = where;
    block->t_where = static_cast<uint32_t>(kmx_time());
}

void kmx_result(uint32_t code, const void* words, uint32_t word_count) {
    const td::Thread& thread = td::here();
    volatile tt_l1_ptr kmx_result_block* block = thread.result();
    if (word_count > KMX_RESULT_WORDS_MAX) {
        block->error = KMX_ERROR_RESULT;
        word_count = KMX_RESULT_WORDS_MAX;
    }
    const uint32_t* source = static_cast<const uint32_t*>(words);
    for (uint32_t word = 0; word < word_count; word++) {
        block->words[word] = source[word];
    }
    block->n = word_count;
    block->code = code;
    thread.start()[kTaCodeSet] = 1;
}

int kmx_submit(const km_command_t* command, uint32_t* handle) {
    return kmx_ring_submit(td::here().ring(), command, handle);
}

int kmx_wait(uint32_t handle, uint64_t deadline, kmx_event* event) {
    kmx_ring_t* ring = td::here().ring();
    const int status = kmx_ring_wait(ring, handle, deadline, event);
    volatile tt_l1_ptr kmx_result_block* block = td::here().result();
    if (block->error == KMX_ERROR_NONE && status == KMX_LATE) {
        block->error = KMX_ERROR_LATE;  // the X280's: a wait that ended early
    } else if (block->error == KMX_ERROR_NONE && status == KMX_STOPPED) {
        block->error = KMX_ERROR_STOPPED;
    }
    return status;
}

int kmx_poll(kmx_event* event) {
    return kmx_ring_poll(td::here().ring(), event);
}

void kmx_command(
    km_command_t* command,
    uint32_t program,
    uint32_t handler,
    uint32_t first,
    uint32_t count
) {
    command->sequence = command->land1 = command->land2 = command->land3 = 0;
    command->hdr = (handler & 0xFFu) | program << 24;
    command->group = (first & 0xFFFFu) | count << 16;
    command->a0 = command->a1 = command->a2 = command->a3 = command->a4 = 0;
    command->a5 = command->a6 = command->a7 = command->a8 = command->a9 = 0;
}

#pragma weak kmx_main
}

namespace td {

    // TRISC 1: waits for the host's start line (or the dispatcher's end), checks the
    // argument block, runs kmx_main and writes the result block.
    inline void thread() {
        Thread& thread = td::here();
        const uint32_t base = get_arg_val<uint32_t>(
            kTdBlock
        );  // the dispatcher's block
        thread.area = base + kTbBytes;
        thread.tag = get_arg_val<uint32_t>(kTdTag);
        thread.scale = get_arg_val<uint32_t>(kTdThreadScale);
        thread.over = base + kTbHand + kTxWords + 4 * kTwOver;
        while (thread.start()[kTaStart] != thread.tag) {
            if (thread.ended()) {
                return;  // the dispatcher ended: no thread was started
            }
        }
        invalidate_l1_cache();  // the host's argument block, written before the start
        volatile tt_l1_ptr kmx_result_block* block = thread.result();
        const volatile tt_l1_ptr kmx_args*
            arguments = (
                reinterpret_cast<volatile tt_l1_ptr kmx_args*>(thread.lim(KMX_LIM_ARGS))
            );
        const uint64_t started = km::wall64();
        thread.start()[kTaStarted] = static_cast<uint32_t>(started);
        thread.start()[kTaStarted + 1] = static_cast<uint32_t>(started >> 32);
        thread.start()[kTaCodeSet] = 0;
        if (
            arguments->magic != KMX_MAGIC || arguments->version != KMX_VERSION ||
            arguments->nwords > KMX_WORDS_MAX || arguments->cluster != 1
        ) {
            block->error = KMX_ERROR_ARGS;
            block->state = KMX_STATE_FAILED;
            return;
        }
        // its command ring: out's source 1 once out started (its lines zeroed)
        while (km::l1(base + kTbHand + kTxWords + 4 * kTwGo)[0] != thread.tag) {
            if (thread.ended()) {
                block->error = KMX_ERROR_STOPPED;
                block->state = KMX_STATE_FAILED;
                return;
            }
        }
        kmx_ring_descriptor ring = {};
        ring.cmd = thread.area + kTaRing;
        ring.consumed = thread.area + kTaWork + KMD_WORK_CONSUMED;
        ring.events = thread.area + kTaWork + KMD_WORK_EVENTS;
        ring.read = thread.area + kTaRead;
        ring.command_depth = kTaRingDepth;
        ring.event_depth = kTaEventDepth;
        kmx_ring_open(thread.ring(), &ring);
        block->state = KMX_STATE_RUNNING;
        const int code = kmx_main != 0 ? kmx_main(0) : 0;
        if (thread.start()[kTaCodeSet] == 0) {
            block->code = static_cast<uint32_t>(code);
        }
        block->ticks = static_cast<uint32_t>(kmx_time());
        block->state = (
            block->error != KMX_ERROR_NONE ? KMX_STATE_FAILED : KMX_STATE_DONE
        );
    }

}  // namespace td
#endif
