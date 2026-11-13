/* DRAM visibility probe: the pattern both sides write and check, the Tensix kernel's
 * modes and its report line. Plain C, included by the kernel and the X280 image. */
#pragma once

#include <stdint.h>

#define DP_WRITE 1u /* the Tensix kernel writes the pattern into the region */
#define DP_COPY 2u  /* it copies the region into the output */
#define DP_DONE 0xD0D0D0D0u

/* Report line of the Tensix kernel (16 words), after the last page it wrote. */
#define DP_REP_NOC_LO 0u /* the NoC address of page 0 of the region */
#define DP_REP_NOC_HI 1u
#define DP_REP_BANK_OFFSET 2u /* bank_to_dram_offset[bank] */
#define DP_REP_FAILED 3u      /* a timed NoC wait gave up */
#define DP_REP_DONE 4u        /* DP_DONE */

/* Word w of page j of a pattern; a different seed gives a different pattern. */
static inline uint32_t dp_word(uint32_t seed, uint32_t page, uint32_t w) {
    uint32_t x = seed * 0x9E3779B1u ^ (page * 4096u + w + 1u) * 0x85EBCA77u;
    x ^= x >> 15;
    return x * 0x2C1B3C6Du;
}
