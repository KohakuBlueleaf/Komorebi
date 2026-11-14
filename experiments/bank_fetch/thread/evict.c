/* The cluster cache's shape, read only (its configuration, ways, way masks, the Zero
 * Device past the pinned span), or the time of kmx_evict against a pad in DRAM. */
#include <stdint.h>

#include "kmx.h"

#define W_MODE 0u
#define W_REPEATS 1u
#define MODE_SHAPE 0u
#define MODE_TIME 1u

#define CCACHE_CONFIG 0x02010000ull    /* banks, ways, log2 sets, log2 line bytes */
#define CCACHE_WAYENABLE 0x02010008ull /* cache ways - 1 */
#define CCACHE_WAYMASK 0x02010800ull   /* 38 masters, 64 bits each */
#define MASTERS 38u
#define MASTER_OF_CORE 34u /* hart 0's L2; hart n's is 34 + n */
#define PL2_CONFIG(core) (0x10104000ull + 0x4000ull * (core))
#define ZERO_DEVICE 0x0A000000ull
#define ZERO_DEVICE_BYTES 0x200000ull /* 0x0A00_0000 to 0x0A1F_FFFF */
#define PINNED_SPAN 0x10C0ull

/* shape words */
#define S_CONFIG 0u
#define S_WAYENABLE 1u
#define S_MASK 2u          /* + master: the mask's low 16 bits */
#define S_HIGH 40u         /* the OR of every mask's high 48 bits, as 32 bits */
#define S_CORE 41u         /* this core's number */
#define S_PL2 42u          /* this core's L2 configuration */
#define S_ZERO_BYTES 43u   /* bytes of the Zero Device read past the pinned span */
#define S_ZERO_NONZERO 44u /* how many of its 64-bit words were not zero */
#define S_WORDS 45u

/* time words: per pad (the Zero Device, DRAM) and per state of the cache before it (a
 * read of region 1, a write of it, an eviction) the best and worst ticks */
#define T_STATES 3u
#define T_WORDS 12u

static uint32_t read32(uint64_t address) {
    return *(volatile const uint32_t*)(uintptr_t)address;
}

static uint64_t read64(uint64_t address) {
    return *(volatile const uint64_t*)(uintptr_t)address;
}

static void shape(uint32_t* out) {
    uint64_t core;
    __asm__ volatile("csrr %0, mhartid" : "=r"(core));
    out[S_CONFIG] = read32(CCACHE_CONFIG);
    out[S_WAYENABLE] = read32(CCACHE_WAYENABLE);
    uint64_t high = 0;
    for (uint32_t master = 0; master < MASTERS; master++) {
        const uint64_t mask = read64(CCACHE_WAYMASK + 8ull * master);
        out[S_MASK + master] = (uint32_t)(mask & 0xFFFFu);
        high |= mask >> 16;
    }
    out[S_HIGH] = (uint32_t)(high | high >> 32);
    out[S_CORE] = (uint32_t)core;
    out[S_PL2] = read32(PL2_CONFIG(core));
    uint32_t nonzero = 0;
    const uint64_t first = ZERO_DEVICE + PINNED_SPAN;
    for (
        uint64_t address = first; address < ZERO_DEVICE + ZERO_DEVICE_BYTES;
        address += 8
    ) {
        nonzero += read64(address) != 0;
    }
    out[S_ZERO_BYTES] = (uint32_t)(ZERO_DEVICE + ZERO_DEVICE_BYTES - first);
    out[S_ZERO_NONZERO] = nonzero;
}

/* The same reads as kmx_evict over region 0 in DRAM. */
static void evict_dram(void) {
    uint64_t bytes = 0;
    volatile const uint64_t* pad = (volatile const uint64_t*)kmx_region(0, &bytes);
    for (uint64_t word = 0; word < bytes / 8; word += 8) {
        (void)pad[word];
    }
}

/* The cache filled with region 1's lines (state 0 read, 1 written) or just evicted. */
static void fill(uint32_t state) {
    uint64_t bytes = 0;
    volatile uint64_t* region = (volatile uint64_t*)kmx_region(1, &bytes);
    for (uint64_t word = 0; state < 2 && word < bytes / 8; word += 8) {
        if (state == 0) {
            (void)region[word];
        } else {
            region[word] = word;
        }
    }
}

static void time_pads(uint32_t repeats, uint32_t* out) {
    for (uint32_t pad = 0; pad < 2; pad++) {
        for (uint32_t state = 0; state < T_STATES; state++) {
            uint32_t* best = &out[2 * (T_STATES * pad + state)];
            best[0] = 0xFFFFFFFFu;
            for (uint32_t r = 0; r < repeats; r++) {
                pad == 0 ? kmx_evict() : evict_dram();
                fill(state);
                const uint64_t start = kmx_time();
                pad == 0 ? kmx_evict() : evict_dram();
                const uint32_t ticks = (uint32_t)(kmx_time() - start);
                best[0] = ticks < best[0] ? ticks : best[0];
                best[1] = ticks > best[1] ? ticks : best[1];
            }
        }
    }
}

int kmx_main(int core) {
    (void)core;
    const uint32_t* words = kmx_args_get()->words;
    uint32_t result[KMX_RESULT_WORDS_MAX] __attribute__((aligned(8)));
    for (uint32_t i = 0; i < KMX_RESULT_WORDS_MAX; i++) {
        result[i] = 0;
    }
    kmx_progress(1 + words[W_MODE]);
    if (words[W_MODE] == MODE_SHAPE) {
        shape(result);
        kmx_result(0, result, S_WORDS);
    } else if (words[W_MODE] == MODE_TIME) {
        time_pads(words[W_REPEATS], result);
        kmx_result(0, result, T_WORDS);
    } else {
        return 1;
    }
    return 0;
}
