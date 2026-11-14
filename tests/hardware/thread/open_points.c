/* A thread that serves commands for the open points of commands to a thread: a held
 * command, kmx_parallel inside kmx_program, commands submitted and never waited for. */
#include <stdint.h>

#include "kmx.h"

/* the sum of 0 to count - 1, a quarter on each X280 core */
typedef struct {
    uint32_t count;
    uint64_t sums[4];
} sum_t;

static void sum_part(int core, void* arg) {
    sum_t* work = (sum_t*)arg;
    const uint32_t quarter = (work->count + 3u) / 4u;
    uint64_t sum = 0;
    for (
        uint32_t value = (uint32_t)core * quarter;
        value < work->count && value < ((uint32_t)core + 1u) * quarter;
        value++
    ) {
        sum += value;
    }
    work->sums[core] = sum;
}

/* AFFINE (handler 1) to grid core `core`: a command whose event nobody waits for */
static void affine(km_command_t* command, uint32_t core, uint32_t value) {
    uint32_t* words = (uint32_t*)command;
    for (uint32_t index = 0; index < 16; index++) {
        words[index] = 0; /* a loop, not an initialiser: no memset in the image */
    }
    command->hdr = 1u;
    command->group = core | 1u << 16;
    command->a0 = value;
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerHold: { /* busy for a0 microseconds, then a1 + 1 */
            const uint64_t end = (
                kmx_time() + (uint64_t)command->a0 * KMX_TICKS_PER_SECOND / 1000000u
            );
            while (kmx_time() < end) {
                ;
            }
            return command->a1 + 1u;
        }
        case kHandlerEcho:
            return command->a0 + 1u;
        case kHandlerSum: { /* low 32 bits of the sum, or 0xFFFFFFFF if refused */
            sum_t work = {command->a0, {0, 0, 0, 0}};
            if (kmx_parallel(sum_part, &work) != KMX_OK) {
                return 0xFFFFFFFFu;
            }
            return (
                (uint32_t)(work.sums[0] + work.sums[1] + work.sums[2] + work.sums[3])
            );
        }
        case kHandlerFire: { /* a0 commands, never waited for: submitted | status << 16
                              */
            uint32_t submitted = 0;
            int status = KMX_OK;
            for (; submitted < command->a0; submitted++) {
                km_command_t grid;
                uint32_t handle;
                affine(&grid, submitted % 8u, submitted);
                status = kmx_submit(&grid, &handle);
                if (status != KMX_OK) {
                    break;
                }
            }
            return submitted | (uint32_t)status << 16;
        }
        case kHandlerPoll: { /* events taken: done | rejected << 8 | timeout << 16 */
            uint32_t done = 0, rejected = 0, other = 0;
            kmx_event event;
            while (kmx_poll(&event) == KMX_OK) {
                done += event.kind == KM_EVENT_DONE;
                rejected += event.kind == KM_EVENT_REJECTED;
                other += event.kind != KM_EVENT_DONE && event.kind != KM_EVENT_REJECTED;
            }
            return done | rejected << 8 | other << 16;
        }
    }
    return 0;
}
