/* Two-image probe: every hart counts loop passes until `ticks` of mtime have passed or
 * the runtime asks for a stop, then stores its count. Result: the count, or ~error. */
#include <stdint.h>

#include "probe.h"

#define BEAT_TICKS_MAX 500000000ull /* 10 s */

typedef struct {
    uint64_t run_value;
    uint64_t ticks; /* how long every hart counts */
} beat_args_t;

int main(uint64_t hartid) {
    pr_t p;
    if (!pr_open(&p, hartid, sizeof(beat_args_t))) {
        x280_hart_set_result(hartid, ~(uint64_t)PR_ERR_ARGS);
        return 0;
    }
    const volatile beat_args_t* a = (const volatile beat_args_t*)p.lim;
    if (a->ticks > BEAT_TICKS_MAX) {
        x280_hart_set_result(hartid, ~(uint64_t)PR_ERR_ARGS);
        return 0;
    }
    pr_stamp(&p, 1);
    const uint64_t t0 = x280_mtime();
    p.deadline = t0 + a->ticks;
    uint64_t passes = 0;
    while (x280_mtime() < p.deadline) {
        passes++;
        if ((passes & 1023u) == 0 && x280_hart_stop_requested(hartid, p.run)) {
            break;
        }
    }
    volatile uint64_t* out = (volatile uint64_t*)(p.lim + PR_RESULT);
    out[hartid] = passes;
    out[4 + hartid] = x280_mtime() - t0;
    pr_stamp(&p, 2);
    x280_hart_set_result(hartid, passes);
    return 0;
}
