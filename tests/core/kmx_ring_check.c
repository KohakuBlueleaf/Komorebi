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
static int take_command(uint32_t i, km_record_t* out) {
    memcpy(out, cmd + (i % 4) * 64, 64);
    if (out->seq != i || !KM_LANDED(out)) {
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
    kmx_ring_desc d = {0};
    kmx_event ev;
    km_record_t rec = {0}, got;
    uint32_t t[5];
    CHECK(!kmx_ring_open(&r, &d) && kmx_ring_issue(&r, &rec, t) == KMX_STOPPED);
    d.cmd = (uintptr_t)cmd;
    d.consumed = (uintptr_t)&consumed;
    d.events = (uintptr_t)events;
    d.read = (uintptr_t)read_line;
    d.cmd_depth = 4;
    d.event_depth = 8;
    CHECK(kmx_ring_open(&r, &d));
    /* four commands fill the ring; the fifth waits for the kernel to take one */
    for (uint32_t i = 0; i < 4; i++) {
        rec.a0 = 100 + i;
        CHECK(kmx_ring_issue(&r, &rec, &t[i]) == KMX_OK && t[i] == i + 1);
    }
    CHECK(kmx_ring_issue(&r, &rec, &t[4]) == KMX_FULL);
    CHECK(take_command(1, &got) && got.a0 == 100 && take_command(2, &got));
    CHECK(kmx_ring_issue(&r, &rec, &t[4]) == KMX_OK && t[4] == 5);
    /* events in another order: waiting for 2 keeps 1 for poll */
    CHECK(kmx_ring_poll(&r, &ev) == KMX_NONE);
    send_event(1, KM_EV_DONE, 7);
    send_event(2, KM_EV_REFUSED | 3u << 8, 0);
    CHECK(kmx_ring_wait(&r, 2, ~0ull, &ev) == KMX_OK);
    CHECK(ev.ticket == 2 && ev.kind == KM_EV_REFUSED && ev.aux == 3);
    CHECK(read_line[0] == 2);
    CHECK(kmx_ring_poll(&r, &ev) == KMX_OK && ev.ticket == 1 && ev.ticks == 7);
    CHECK(kmx_ring_poll(&r, &ev) == KMX_NONE);
    /* a wake: wait passes it over without keeping it, poll returns it */
    send_event(0, KMX_EV_WAKE, 0);
    send_event(3, KM_EV_DONE, 4);
    CHECK(kmx_ring_wait(&r, 3, ~0ull, &ev) == KMX_OK && ev.ticket == 3);
    CHECK(r.stashed == 0);
    send_event(0, KMX_EV_WAKE, 0);
    CHECK(kmx_ring_poll(&r, &ev) == KMX_OK && ev.kind == KMX_EV_WAKE && ev.ticket == 0);
    /* a torn event (index not yet written) is not taken */
    memcpy(events + 6 * 16 + 4, (uint32_t[]){3, 0, 9}, 12);
    CHECK(kmx_ring_poll(&r, &ev) == KMX_NONE);
    /* no event by the deadline, then a stop */
    CHECK(kmx_ring_wait(&r, 3, now + 5000, &ev) == KMX_LATE);
    end_code = KMX_STOPPED;
    CHECK(kmx_ring_wait(&r, 3, ~0ull, &ev) == KMX_STOPPED);
    printf("ok\n");
    return 0;
}
