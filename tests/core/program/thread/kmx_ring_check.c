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
#define KMX_RING_SWEEP(address, bytes) ((void)(address), (void)(bytes))
#define KMX_RING_NOW() (now += 1000)
#define KMX_RING_END() end_code
#include "kmx_ring.h"

#define CHECK(condition)                               \
    if (!(condition)) {                                \
        printf("line %d: %s\n", __LINE__, #condition); \
        return 1;                                      \
    }

/* the events, then the lines the dispatcher hands the thread */
static _Alignas(64) uint8_t command_slots[4 * 64], events[8 * 16 + 8 * 64];
static _Alignas(64) uint64_t read_line[8];
static uint32_t consumed, sent;

/* The fake kernel takes command `i` if its four number words agree. */
static int take_command(uint32_t i, km_command_t* out) {
    memcpy(out, command_slots + (i % 4) * 64, 64);
    if (out->sequence != i || !KM_LANDED(out)) {
        return 0;
    }
    consumed = i;
    return 1;
}

static void send_event(uint32_t req, uint32_t status, uint32_t ticks) {
    const uint32_t event_words[4] = {++sent, req, status, ticks};
    memcpy(events + (sent % 8) * 16, event_words, 16);
}

int main(void) {
    kmx_ring_t ring;
    kmx_ring_descriptor descriptor = {0};
    kmx_event event;
    km_command_t command = {0}, got;
    uint32_t handles[5];
    CHECK(
        !kmx_ring_open(&ring, &descriptor) &&
        kmx_ring_submit(&ring, &command, handles) == KMX_STOPPED
    );
    descriptor.cmd = (uintptr_t)command_slots;
    descriptor.consumed = (uintptr_t)&consumed;
    descriptor.events = (uintptr_t)events;
    descriptor.read = (uintptr_t)read_line;
    descriptor.command_depth = 4;
    descriptor.event_depth = 8;
    CHECK(kmx_ring_open(&ring, &descriptor));
    /* four commands fill the ring; the fifth waits for the kernel to take one */
    for (uint32_t i = 0; i < 4; i++) {
        command.a0 = 100 + i;
        CHECK(
            kmx_ring_submit(&ring, &command, &handles[i]) == KMX_OK &&
            handles[i] == i + 1
        );
    }
    CHECK(kmx_ring_submit(&ring, &command, &handles[4]) == KMX_FULL);
    CHECK(take_command(1, &got) && got.a0 == 100 && take_command(2, &got));
    CHECK(kmx_ring_submit(&ring, &command, &handles[4]) == KMX_OK && handles[4] == 5);
    /* events in another order: waiting for 2 keeps 1 for poll */
    CHECK(kmx_ring_poll(&ring, &event) == KMX_NONE);
    send_event(1, KM_EVENT_DONE, 7);
    send_event(2, KM_EVENT_REJECTED | 3u << 8, 0);
    CHECK(kmx_ring_wait(&ring, 2, ~0ull, &event) == KMX_OK);
    CHECK(event.handle == 2 && event.kind == KM_EVENT_REJECTED && event.auxiliary == 3);
    CHECK(read_line[0] == 2);
    CHECK(
        kmx_ring_poll(&ring, &event) == KMX_OK && event.handle == 1 && event.ticks == 7
    );
    CHECK(kmx_ring_poll(&ring, &event) == KMX_NONE);
    /* a wake: wait passes it over without keeping it, poll returns it */
    send_event(0, KMX_EVENT_WAKE, 0);
    send_event(3, KM_EVENT_DONE, 4);
    CHECK(kmx_ring_wait(&ring, 3, ~0ull, &event) == KMX_OK && event.handle == 3);
    CHECK(ring.stashed == 0);
    send_event(0, KMX_EVENT_WAKE, 0);
    CHECK(
        kmx_ring_poll(&ring, &event) == KMX_OK && event.kind == KMX_EVENT_WAKE &&
        event.handle == 0
    );
    /* a torn event (index not yet written) is not taken */
    memcpy(events + 6 * 16 + 4, (uint32_t[]){3, 0, 9}, 12);
    CHECK(kmx_ring_poll(&ring, &event) == KMX_NONE);
    /* look never waits: another handle's event is kept, a wake passed over */
    CHECK(kmx_ring_look(&ring, 5, &event) == KMX_NONE);
    send_event(4, KM_EVENT_DONE, 1);
    send_event(0, KMX_EVENT_WAKE, 0);
    CHECK(kmx_ring_look(&ring, 5, &event) == KMX_NONE && ring.stashed == 1);
    send_event(5, KM_EVENT_TIMEOUT, 2);
    CHECK(kmx_ring_look(&ring, 5, &event) == KMX_OK && event.kind == KM_EVENT_TIMEOUT);
    CHECK(
        kmx_ring_look(&ring, 4, &event) == KMX_OK && event.ticks == 1 &&
        ring.stashed == 0
    );
    /* the dispatcher's lines are taken whole and in order; an answer is a command */
    uint8_t* lines = events + KMX_RING_IN_AFTER(8);
    km_command_t line = {.sequence = 1, .hdr = 10, .a0 = 5, .land1 = 1, .land2 = 1};
    CHECK(!kmx_ring_command(&ring, &got));
    memcpy(lines + 64, &line, 64);
    CHECK(!kmx_ring_command(&ring, &got) && ring.in_next == 1);
    line.land3 = 1;
    memcpy(lines + 64, &line, 64);
    CHECK(kmx_ring_command(&ring, &got) && got.hdr == 10 && got.a0 == 5);
    CHECK(ring.in_next == 2 && !kmx_ring_command(&ring, &got));
    CHECK(kmx_ring_answer(&ring, 1, 77) == KMX_OK && take_command(6, &got));
    CHECK(got.hdr == KMX_HANDLER_ANSWER && got.a0 == 1 && got.a1 == 77);
    /* no event by the deadline, then a stop */
    CHECK(kmx_ring_wait(&ring, 3, now + 5000, &event) == KMX_LATE);
    end_code = KMX_STOPPED;
    CHECK(kmx_ring_wait(&ring, 3, ~0ull, &event) == KMX_STOPPED);
    printf("ok\n");
    return 0;
}
