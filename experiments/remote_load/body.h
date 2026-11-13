// A body naming no symbol of its persistent kernel (so it runs under any): the entry
// at the slot's first word zeroes .bss, counts the call through a pointer, runs it.
#pragma once

#include <stdint.h>

namespace rl {

    struct Args {
        uint32_t arg;               // the message's argument
        uint32_t (*mix)(uint32_t);  // a function of the persistent kernel
        volatile uint32_t* calls;   // its call counter
    };

    using Entry = uint32_t (*)(Args*);

    // The persistent kernel's mix(), mirrored by the host's reference.
    inline uint32_t mix(uint32_t x) {
        return ((x << 7) | (x >> 25)) ^ 0x5A5A5A5Au;
    }

}  // namespace rl

#if defined(RL_BODY)
extern "C" uint32_t ov_body(rl::Args* a);
extern "C" uint32_t __ov_bss_start[], __ov_bss_end[];  // the builder's linker script

extern "C" __attribute__((section(".ov_entry"), used, noinline)) uint32_t
ov_entry(rl::Args* a) {
    for (uint32_t* p = __ov_bss_start; p < __ov_bss_end; p++) {
        *p = 0;
    }
    *a->calls = *a->calls + 1;
    return ov_body(a);
}
#endif
