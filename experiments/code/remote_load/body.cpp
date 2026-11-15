// A body (body.h): generated chain steps on the argument, with OV_MARK plus a 32-bit
// constant loaded by one lui / addi pair (found by its value, rewritten), then mix().
#define RL_BODY 1
#include "body.h"
#include "overlay_gen.h"

namespace {

    uint32_t calls;  // .bss: 0 at every entry, so 1 here

    inline uint32_t rotl(uint32_t x, uint32_t r) {
        return (x << r) | (x >> (32 - r));
    }

}  // namespace

// One step; the empty asm keeps each step as written (no folding across steps).
#define OV_STEP(k, r, k2)          \
    x = rotl(x ^ (k), (r)) + (k2); \
    asm volatile("" : "+r"(x));

extern "C" uint32_t ov_body(rl::Args* a) {
    calls = calls + 1;
    uint32_t x = a->arg ^ calls;
    OV_STEPS(OV_STEP)
#if defined(OV_MARK)
    uint32_t k;
    asm volatile("lui %0, %%hi(%1)\n\taddi %0, %0, %%lo(%1)" : "=r"(k) : "i"(OV_MARK));
    x = x + k;
#endif
    return a->mix(x);
}
