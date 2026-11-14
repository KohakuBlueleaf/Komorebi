/* Thread programs example, the thread: ROUNDS runs every round, splitting the cores
 * between spin and scale (from the last round's times, or fixed), submitting both. */
#include <stdint.h>

#include "programs.h"
#include "kmx.h"

/* a command's wait, and a full ring's retry */
#define WAIT_TICKS KMX_TICKS_PER_SECOND
#define SEARCH 4u /* the splits tried on each side of the work's share */

/* Submits `command`, retrying a full ring: KMX_OK and its handle, or why it failed. */
static int submit(const km_command_t* command, uint32_t* handle) {
    const uint64_t deadline = kmx_time() + WAIT_TICKS;
    int status;
    do {
        status = kmx_submit(command, handle);
    } while (status == KMX_FULL && kmx_time() < deadline);
    return status;
}

/* --8<-- [start:split] */
/* Spin's cores: near the work's share, the k whose slower command is fastest. */
static uint32_t split(
    uint32_t cores,
    uint32_t spin_units,
    uint64_t spin_rate,
    uint32_t tiles,
    uint64_t scale_rate
) {
    const uint64_t spin_work = spin_units * spin_rate;
    const uint64_t total = spin_work + tiles * scale_rate;
    const uint32_t share = (uint32_t)((cores * spin_work + total / 2) / total);
    uint32_t best = share < 1 ? 1 : share > cores - 1 ? cores - 1 : share;
    uint64_t best_time = ~0ull;
    const uint32_t low = best > SEARCH ? best - SEARCH : 1;
    const uint32_t high = best + SEARCH < cores ? best + SEARCH : cores - 1;
    for (uint32_t k = low; k <= high; k++) {
        const uint64_t spin = (spin_units + k - 1) / k * spin_rate;
        const uint64_t scale = (tiles + cores - k - 1) / (cores - k) * scale_rate;
        const uint64_t slower = spin > scale ? spin : scale;
        if (slower < best_time) {
            best = k;
            best_time = slower;
        }
    }
    return best;
}

/* --8<-- [end:split] */

/* The ticks of a command with no work on one core (submit to completion): what every
 * command costs whatever its size. 0 if it did not finish. */
static uint32_t empty_ticks(uint32_t program, uint32_t handler) {
    km_command_t command;
    kmx_command(&command, program, handler, 0, 1);
    uint32_t handle = 0;
    kmx_event event;
    if (
        submit(&command, &handle) != KMX_OK ||
        kmx_wait(handle, kmx_time() + WAIT_TICKS, &event) != KMX_OK ||
        event.kind != KM_EVENT_DONE
    ) {
        return 0;
    }
    return event.ticks;
}

/* Ticks a unit, x 1024: a command's ticks less the fixed part, over the most units
 * one of its cores had. */
static uint64_t rate(uint32_t ticks, uint32_t fixed, uint32_t cores, uint32_t units) {
    const uint64_t working = ticks > fixed ? ticks - fixed : 1u;
    const uint64_t out = 1024ull * working / ((units + cores - 1) / cores);
    return out ? out : 1u;
}

/* One job of `rounds` rounds; each round's log line in region 1. */
static uint32_t rounds(const km_command_t* job) {
    const uint32_t* words = kmx_args_get()->words;
    uint64_t round_bytes = 0, log_bytes = 0;
    const uint32_t* plan = (const uint32_t*)kmx_region(0, &round_bytes);
    uint32_t* log = (uint32_t*)kmx_region(1, &log_bytes);
    const uint32_t cores = job->a3, count = job->a2;
    if (
        cores < 2 || 4ull * TP_ROUND * count > round_bytes ||
        4ull * TP_LOG * count > log_bytes
    ) {
        return KMX_BAD;
    }
    uint64_t spin_rate = 1, scale_rate = 1; /* ticks a unit, x 1024 */
    const uint32_t fixed = (
        job->a0 ? 0 : empty_ticks(kProgramSpin, kProgramSpinHandlerSpin)
    );
    int status = KMX_OK;
    for (uint32_t round = 0; round < count && status == KMX_OK; round++) {
        const uint64_t started = kmx_time();
        const uint32_t* units = plan + TP_ROUND * round;
        const uint32_t spin_units = units[TP_ROUND_SPIN];
        const uint32_t tiles = units[TP_ROUND_TILES];
        const uint32_t spin_cores = (
            job->a0 ? job->a1 : split(cores, spin_units, spin_rate, tiles, scale_rate)
        );
        /* --8<-- [start:submit] */
        km_command_t spin, scale;
        kmx_command(&spin, kProgramSpin, kProgramSpinHandlerSpin, 0, spin_cores);
        spin.a0 = units[TP_ROUND_FIRST_UNIT];
        spin.a1 = spin_units;
        spin.a2 = words[TP_WORD_COSTS];
        spin.a3 = words[TP_WORD_OUTS];
        spin.a4 = job->a4;
        kmx_command(
            &scale,
            kProgramScale,
            kProgramScaleHandlerScale,
            spin_cores,
            cores - spin_cores
        );
        scale.a0 = words[TP_WORD_INPUT];
        scale.a1 = tiles;
        scale.a2 = words[TP_WORD_OUTPUT];
        scale.a3 = round * words[TP_WORD_MAX_TILES];
        scale.a4 = job->a5; /* the scalar, float32 bits */
        scale.a6 = units[TP_ROUND_FIRST_TILE];
        uint32_t spin_handle = 0, scale_handle = 0;
        kmx_event spin_event, scale_event;
        status = submit(&spin, &spin_handle);
        if (status == KMX_OK) {
            status = submit(&scale, &scale_handle);
        }
        if (status == KMX_OK) {
            status = kmx_wait(spin_handle, kmx_time() + WAIT_TICKS, &spin_event);
        }
        if (status == KMX_OK) {
            status = kmx_wait(scale_handle, kmx_time() + WAIT_TICKS, &scale_event);
        }
        /* --8<-- [end:submit] */
        uint32_t* line = log + TP_LOG * round;
        line[TP_LOG_SPLIT] = spin_cores;
        line[TP_LOG_SPIN_TICKS] = status == KMX_OK ? spin_event.ticks : 0;
        line[TP_LOG_SCALE_TICKS] = status == KMX_OK ? scale_event.ticks : 0;
        line[TP_LOG_ROUND_TICKS] = (uint32_t)(kmx_time() - started);
        line[TP_LOG_STATUS] = (uint32_t)status;
        if (
            status == KMX_OK &&
            (spin_event.kind != KM_EVENT_DONE || scale_event.kind != KM_EVENT_DONE)
        ) {
            status = KMX_LATE;
            line[TP_LOG_STATUS] = (uint32_t)status;
        }
        if (status == KMX_OK && spin_units != 0 && tiles != 0) {
            spin_rate = rate(spin_event.ticks, fixed, spin_cores, spin_units);
            scale_rate = rate(scale_event.ticks, fixed, cores - spin_cores, tiles);
        }
    }
    kmx_flush(1);
    return (uint32_t)status;
}

uint32_t kmx_program(const km_command_t* command) {
    return kmx_handler(command) == kHandlerRounds ? rounds(command) : KMX_BAD;
}
