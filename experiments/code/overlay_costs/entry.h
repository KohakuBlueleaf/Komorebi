// The entry of an overlay at its slot's first word: zeroes the overlay's .bss, then
// runs the body. Hidden symbols keep a position-independent build free of a GOT.
#pragma once

#include "costs.h"

extern "C" uint32_t ov_body(const cs::Call* call) __attribute__((visibility("hidden")));
extern "C" uint32_t __ov_bss_start[] __attribute__((visibility("hidden")));
extern "C" uint32_t __ov_bss_end[] __attribute__((visibility("hidden")));

extern "C" __attribute__((section(".ov_entry"), used, noinline)) uint32_t
ov_entry(const cs::Call* call) {
    for (uint32_t* p = __ov_bss_start; p < __ov_bss_end; p++) {
        *p = 0;
    }
    return ov_body(call);
}
