/* Two-image probe: every core counts loop passes until `ticks` of mtime have passed or
 * the runtime asks for a stop, then stores its count. Result: the count, or ~error. */
#include <stdint.h>

#include "probe.h"

#define BEAT_TICKS_MAX 500000000ull /* 10 s */

typedef struct {
    uint64_t run_value;
    uint64_t ticks; /* how long every core counts */
} beat_args_t;

int main(uint64_t core_id) {
    pr_t p;
    if (!pr_open(&p, core_id, sizeof(beat_args_t))) {
        x280_hart_set_result(core_id, ~(uint64_t)PR_ERR_ARGS);
        return 0;
    }
    const volatile beat_args_t* a = (const volatile beat_args_t*)p.lim;
    if (a->ticks > BEAT_TICKS_MAX) {
        x280_hart_set_result(core_id, ~(uint64_t)PR_ERR_ARGS);
        return 0;
    }
    pr_marker(&p, 1);
    const uint64_t t0 = x280_mtime();
    p.deadline = t0 + a->ticks;
    uint64_t passes = 0;
    while (x280_mtime() < p.deadline) {
        passes++;
        if ((passes & 1023u) == 0 && x280_hart_stop_requested(core_id, p.run)) {
            break;
        }
    }
    volatile uint64_t* out = (volatile uint64_t*)(p.lim + PR_RESULT);
    out[core_id] = passes;
    out[4 + core_id] = x280_mtime() - t0;
    pr_marker(&p, 2);
    x280_hart_set_result(core_id, passes);
    return 0;
}
