// A part of a program for any of the five cores: generated straight-line xor, rotate
// and add steps on the argument (overlay_gen.h), then the persistent kernel's mix().
#include "costs.h"
#include "entry.h"
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

extern "C" uint32_t ov_body(const cs::Call* call) {
    calls = calls + 1;
    *call->calls = *call->calls + 1;
    uint32_t x = call->arg ^ calls;
    OV_STEPS(OV_STEP)
    return call->mix(x);
}
