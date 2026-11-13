// A data-movement overlay: generated straight-line xor, rotate and add steps on the
// argument (overlay_gen.h), then the loader's mix(); a .bss counter checks the entry.
#include "overlay_gen.h"
#include "overlay_entry.h"

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

extern "C" uint32_t ov_body(ov::Args* a) {
    calls = calls + 1;
    uint32_t x = a->arg ^ calls;
    OV_STEPS(OV_STEP)
    return a->mix(x);
}
