// The pack CPU's bodies of example 34's handlers: items into items, word by word.
#pragma once

#include <stdint.h>

namespace shared_k {

    // total = a + b + c and mixed = a ^ b ^ c over `words` words (each an item's L1
    // address); an output of 0 is a give-up of compute: nothing is written.
    inline void mix(
        uint32_t a,
        uint32_t b,
        uint32_t c,
        uint32_t total,
        uint32_t mixed,
        uint32_t words
    ) {
        if (total == 0 || mixed == 0) {
            return;
        }
        const volatile uint32_t* in_a = reinterpret_cast<const volatile uint32_t*>(a);
        const volatile uint32_t* in_b = reinterpret_cast<const volatile uint32_t*>(b);
        const volatile uint32_t* in_c = reinterpret_cast<const volatile uint32_t*>(c);
        volatile uint32_t* out_total = reinterpret_cast<volatile uint32_t*>(total);
        volatile uint32_t* out_mixed = reinterpret_cast<volatile uint32_t*>(mixed);
        for (uint32_t i = 0; i < words; i++) {
            const uint32_t x = in_a[i], y = in_b[i], z = in_c[i];
            out_total[i] = x + y + z;
            out_mixed[i] = x ^ y ^ z;
        }
    }

    // out = in * 3 + key over `words` words (each an item's L1 address); an output of
    // 0 is a give-up of compute: nothing is written.
    inline void scale(uint32_t in, uint32_t out, uint32_t words, uint32_t key) {
        if (out == 0) {
            return;
        }
        const volatile uint32_t* from = reinterpret_cast<const volatile uint32_t*>(in);
        volatile uint32_t* to = reinterpret_cast<volatile uint32_t*>(out);
        for (uint32_t i = 0; i < words; i++) {
            to[i] = from[i] * 3u + key;
        }
    }

}  // namespace shared_k
