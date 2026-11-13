/* The kmx ring calls against a fake kernel in plain memory; prints "ok" or the first
 * failed check. Built and run by test_thread.py. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kmx_layout.h"

static uint64_t now;
static int end_code;
#define KMX_RING_FENCE_W() __sync_synchronize()
#define KMX_RING_FENCE_R() __sync_synchronize()
#define KMX_RING_SWEEP(p, n) ((void)(p), (void)(n))
#define KMX_RING_NOW() (now += 1000)
#define KMX_RING_END() end_code
#include "kmx_ring.h"

#define CHECK(c)                               \
    if (!(c)) {                                \
        printf("line %d: %s\n", __LINE__, #c); \
        return 1;                              \
    }

static _Alignas(64) uint8_t cmd[4 * 64], events[8 * 16];
static _Alignas(64) uint64_t read_line[8];
static uint32_t consumed, sent;

/* The fake kernel takes command `i` if its four number words agree. */
static int take_command(uint32_t i, km_command_t* out) {
    memcpy(out, cmd + (i % 4) * 64, 64);
    if (out->sequence != i || !KM_LANDED(out)) {
        return 0;
    }
    consumed = i;
    return 1;
}

static void send_event(uint32_t req, uint32_t status, uint32_t ticks) {
    const uint32_t e[4] = {++sent, req, status, ticks};
    memcpy(events + (sent % 8) * 16, e, 16);
}

int main(void) {
    kmx_ring_t r;
    kmx_ring_descriptor d = {0};
    kmx_event event;
    km_command_t rec = {0}, got;
    uint32_t t[5];
    CHECK(!kmx_ring_open(&r, &d) && kmx_ring_submit(&r, &rec, t) == KMX_STOPPED);
    d.cmd = (uintptr_t)cmd;
    d.consumed = (uintptr_t)&consumed;
    d.events = (uintptr_t)events;
    d.read = (uintptr_t)read_line;
    d.command_depth = 4;
    d.event_depth = 8;
    CHECK(kmx_ring_open(&r, &d));
    /* four commands fill the ring; the fifth waits for the kernel to take one */
    for (uint32_t i = 0; i < 4; i++) {
        rec.a0 = 100 + i;
        CHECK(kmx_ring_submit(&r, &rec, &t[i]) == KMX_OK && t[i] == i + 1);
    }
    CHECK(kmx_ring_submit(&r, &rec, &t[4]) == KMX_FULL);
    CHECK(take_command(1, &got) && got.a0 == 100 && take_command(2, &got));
    CHECK(kmx_ring_submit(&r, &rec, &t[4]) == KMX_OK && t[4] == 5);
    /* events in another order: waiting for 2 keeps 1 for poll */
    CHECK(kmx_ring_poll(&r, &event) == KMX_NONE);
    send_event(1, KM_EVENT_DONE, 7);
    send_event(2, KM_EVENT_REJECTED | 3u << 8, 0);
    CHECK(kmx_ring_wait(&r, 2, ~0ull, &event) == KMX_OK);
    CHECK(event.handle == 2 && event.kind == KM_EVENT_REJECTED && event.auxiliary == 3);
    CHECK(read_line[0] == 2);
    CHECK(kmx_ring_poll(&r, &event) == KMX_OK && event.handle == 1 && event.ticks == 7);
    CHECK(kmx_ring_poll(&r, &event) == KMX_NONE);
    /* a wake: wait passes it over without keeping it, poll returns it */
    send_event(0, KMX_EVENT_WAKE, 0);
    send_event(3, KM_EVENT_DONE, 4);
    CHECK(kmx_ring_wait(&r, 3, ~0ull, &event) == KMX_OK && event.handle == 3);
    CHECK(r.stashed == 0);
    send_event(0, KMX_EVENT_WAKE, 0);
    CHECK(
        kmx_ring_poll(&r, &event) == KMX_OK && event.kind == KMX_EVENT_WAKE &&
        event.handle == 0
    );
    /* a torn event (index not yet written) is not taken */
    memcpy(events + 6 * 16 + 4, (uint32_t[]){3, 0, 9}, 12);
    CHECK(kmx_ring_poll(&r, &event) == KMX_NONE);
    /* look never waits: another handle's event is kept, a wake passed over */
    CHECK(kmx_ring_look(&r, 5, &event) == KMX_NONE);
    send_event(4, KM_EVENT_DONE, 1);
    send_event(0, KMX_EVENT_WAKE, 0);
    CHECK(kmx_ring_look(&r, 5, &event) == KMX_NONE && r.stashed == 1);
    send_event(5, KM_EVENT_TIMEOUT, 2);
    CHECK(kmx_ring_look(&r, 5, &event) == KMX_OK && event.kind == KM_EVENT_TIMEOUT);
    CHECK(kmx_ring_look(&r, 4, &event) == KMX_OK && event.ticks == 1 && r.stashed == 0);
    /* no event by the deadline, then a stop */
    CHECK(kmx_ring_wait(&r, 3, now + 5000, &event) == KMX_LATE);
    end_code = KMX_STOPPED;
    CHECK(kmx_ring_wait(&r, 3, ~0ull, &event) == KMX_STOPPED);
    printf("ok\n");
    return 0;
}
