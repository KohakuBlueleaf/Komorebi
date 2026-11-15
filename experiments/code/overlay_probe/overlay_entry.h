// The entry of an overlay at its slot's first word: zeroes the overlay's .bss, counts
// the call in the loader's counter (data-movement overlays), runs the body.
#pragma once

#include "overlay.h"

extern "C" uint32_t ov_body(ov::Args* a);
extern "C" uint32_t __ov_bss_start[], __ov_bss_end[];  // the overlay's linker script

extern "C" __attribute__((section(".ov_entry"), used, noinline)) uint32_t
ov_entry(ov::Args* a) {
    for (uint32_t* p = __ov_bss_start; p < __ov_bss_end; p++) {
        *p = 0;
    }
#if !defined(COMPILE_FOR_TRISC)  // the counter is the data-movement loader's
    ov_calls = ov_calls + 1;
#endif
    return ov_body(a);
}
