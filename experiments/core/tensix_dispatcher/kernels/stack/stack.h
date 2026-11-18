// A data-movement kernel's free stack, painted first and measured after its body;
// words at STACK_OUT + 16 * role: free bytes, the stack's base, sp at the start.
#pragma once

#include <stdint.h>

extern uint32_t __stack_base[];

namespace stack {

    constexpr uint32_t kPattern = 0xBABABABA;

    // Paints [__stack_base, sp): nothing below sp is in use (RISC-V has no red zone).
    inline uint32_t paint() {
        uint32_t* sp;
        asm volatile("mv %0, sp" : "=r"(sp));
        for (volatile uint32_t* word = __stack_base; word < sp; word++) {
            *word = kPattern;
        }
        return reinterpret_cast<uintptr_t>(sp);
    }

    // The bytes never written since paint(), from the base up, with the base and sp.
    inline void record(uint32_t role, uint32_t sp) {
        const volatile uint32_t* word = __stack_base;
        while (*word == kPattern) {
            word++;
        }
        volatile uint32_t* out = (
            reinterpret_cast<volatile uint32_t*>(STACK_OUT + 16 * role)
        );
        out[0] = (
            reinterpret_cast<uintptr_t>(word) -
            reinterpret_cast<uintptr_t>(__stack_base)
        );
        out[1] = reinterpret_cast<uintptr_t>(__stack_base);
        out[2] = sp;
    }

}  // namespace stack
