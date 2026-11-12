/* The thread's own files on the CPU: kmx_ring.h against a fake kernel that runs STEP
 * and SUM on a model of the grid. Arguments: the words; output: see README.md. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "decide.h"
#include "kmx.h"

static int fake_kernel(void);
static uint64_t now;
#define KMX_RING_FENCE_W() __sync_synchronize()
#define KMX_RING_FENCE_R() __sync_synchronize()
#define KMX_RING_SWEEP(p, n) ((void)(p), (void)(n))
#define KMX_RING_NOW() (++now)
#define KMX_RING_END() fake_kernel()
#include "kmx_ring.h"

#define CMD_DEPTH 4u
#define EVENT_DEPTH 8u

static _Alignas(64) uint8_t cmd[CMD_DEPTH * 64], events[EVENT_DEPTH * 16];
static _Alignas(64) uint64_t read_line[8];
static uint32_t consumed, sent;
static uint32_t x[KM_CORES_MAX]; /* page p of the state tensor: core p's word */
static kmx_args args;
static uint32_t teams[4096]; /* region 0 */
static kmx_ring_t ring;

/* Takes every landed command in order, runs it on the model, posts its event. */
static int fake_kernel(void) {
    for (;;) {
        const uint32_t i = consumed + 1;
        km_record_t r;
        memcpy(&r, cmd + (i % CMD_DEPTH) * 64, 64);
        if (r.seq != i || !KM_LANDED(&r)) {
            return KMX_OK;
        }
        if (sent - (uint32_t)read_line[0] >= EVENT_DEPTH) {
            return KMX_OK; /* the thread has not taken enough events yet */
        }
        consumed = i;
        const uint32_t seg = r.hdr & 0xFFu, flags = r.hdr >> 8 & 0xFFu;
        const uint32_t first = r.team & 0xFFFFu, count = r.team >> 16;
        uint32_t result = 0;
        if (seg == WC_SEG_STEP) {
            for (uint32_t p = first; p < first + count; p++) {
                x[p] += r.a1 + p;
            }
        } else if (seg == WC_SEG_SUM) {
            for (uint32_t p = 0; p < r.a1; p++) {
                result += x[p];
            }
        }
        printf("record %u %u %u %u %u %u\n", seg, flags, first, count, r.a0, r.a1);
        const uint32_t e[4] = {++sent, i, KM_EV_DONE, flags & KM_F_RESULT ? result : 1};
        memcpy(events + (sent % EVENT_DEPTH) * 16, e, 16);
    }
}

const kmx_args* kmx_args_get(void) {
    return &args;
}

void* kmx_region(int id, uint64_t* bytes) {
    *bytes = id == 0 ? sizeof teams : 0;
    return id == 0 ? teams : 0;
}

void kmx_give(int id) {
    (void)id;
}

int kmx_issue(const km_record_t* r, uint32_t* ticket) {
    return kmx_ring_issue(&ring, r, ticket);
}

int kmx_wait(uint32_t ticket, uint64_t deadline, kmx_event* ev) {
    return kmx_ring_wait(&ring, ticket, deadline, ev);
}

uint64_t kmx_time(void) {
    fake_kernel();
    return ++now;
}

int kmx_stop_requested(void) {
    return 0;
}

void kmx_progress(uint32_t where) {
    (void)where;
}

void kmx_result(uint32_t code, const void* words, uint32_t n) {
    const uint32_t* w = words;
    printf("result %u", code);
    for (uint32_t i = 0; i < n; i++) {
        printf(" %u", w[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    kmx_ring_desc d = {0};
    d.cmd = (uintptr_t)cmd;
    d.consumed = (uintptr_t)&consumed;
    d.events = (uintptr_t)events;
    d.read = (uintptr_t)read_line;
    d.cmd_depth = CMD_DEPTH;
    d.event_depth = EVENT_DEPTH;
    if (!kmx_ring_open(&ring, &d) || argc - 1 > KMX_WORDS_MAX) {
        return 2;
    }
    args.nwords = (uint32_t)(argc - 1);
    for (int i = 1; i < argc; i++) {
        args.words[i - 1] = (uint32_t)strtoul(argv[i], 0, 0);
    }
    for (uint32_t p = 0; p < KM_CORES_MAX; p++) {
        x[p] = 7u * p; /* the host's initial state */
    }
    const int code = kmx_main(0);
    if (code != 0) {
        printf("result %d\n", code);
    }
    const uint32_t rounds = (
        args.words[WC_MAX_ROUNDS] < 4096 ? args.words[WC_MAX_ROUNDS] : 4096
    );
    printf("teams");
    for (uint32_t i = 0; i < rounds; i++) {
        printf(" %u", teams[i]);
    }
    printf("\n");
    return 0;
}
