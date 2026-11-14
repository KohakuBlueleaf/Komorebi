/* The cluster cache's shape, read only (its configuration, ways, way masks, the Zero
 * Device past the pinned span); kmx_evict's time against a pad in DRAM; its proof. */
#include <stdint.h>

#include "kmx.h"

#define W_MODE 0u
#define W_REPEATS 1u
#define MODE_SHAPE 0u
#define MODE_TIME 1u
#define MODE_PROOF 2u
#define W_BYTES 2u    /* proof: the range's bytes, at LEAD in region 0 */
#define W_VARIANTS 3u /* proof: bit v runs variant v */
#define W_FIRST 4u    /* proof: the first round's number */

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

/* proof: region 0 is LEAD bytes then the range, region 1 the source, region 2 what the
 * other three cores read. Variants: */
#define V_PLAIN 0u /* read, write, evict, copy, evict, check */
#define V_LEAD 1u  /* and region 0's lead read up to the range just before the evict */
#define V_READERS 2u /* and cores 1 to 3 read region 2 the whole time */
#define V_ONE 3u     /* the lead read, no second evict: kmx_fetch's own order */
#define V_NONE 4u    /* the lead read, no evict at all: the check must see bad lines */
#define VARIANTS 5u
#define LEAD 0x40000ull
#define COPY_CORES 4u
/* proof words per variant: rounds, bad lines, then the first bad line's round, line,
 * word read (2), word expected (2), and the variant's ticks */
#define P_ROUNDS 0u
#define P_BAD 1u
#define P_ROUND 2u
#define P_LINE 3u
#define P_GOT 4u
#define P_WANT 6u
#define P_TICKS 8u
#define P_STRIDE 9u
#define P_READ (P_STRIDE * VARIANTS) /* MiB cores 1 to 3 read */
#define P_STATUS (P_READ + 1u)       /* a failed call: status | variant << 8 */
#define P_WORDS (P_STATUS + 1u)

static uint32_t g_stop __attribute__((section(".data"))) = 0;
static uint32_t g_read[4] __attribute__((section(".data"))) = {0, 0, 0, 0};

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

/* Round `round`'s word `index`: different for every round and word. */
static uint64_t pattern(uint64_t round, uint64_t index) {
    uint64_t x = (round << 40 ^ index) * 0x9E3779B97F4A7C15ull;
    x ^= x >> 29;
    return x * 0xBF58476D1CE4E5B9ull ^ x >> 32;
}

/* Cores 1 to 3: line loads over region 2 until g_stop, the stop or the deadline; the
 * MiB read into g_read[core]. */
static void reader(int core, void* arg) {
    uint64_t bytes = 0;
    volatile const uint64_t* noise = (volatile const uint64_t*)kmx_region(2, &bytes);
    uint64_t lines = 0;
    while (!*(volatile uint32_t*)&g_stop && !kmx_stop_requested()) {
        for (uint64_t word = 0; word < bytes / 8; word += 8) {
            (void)noise[word];
        }
        lines += bytes / 64;
    }
    (void)arg;
    g_read[core] = (uint32_t)(lines >> 14);
}

/* One round of `variant`: kmx_fetch_clean status, out[P_BAD] and the first bad line
 * counted. */
static int round_of(uint32_t variant, uint64_t round, uint64_t bytes, uint32_t* out) {
    uint64_t region_bytes = 0;
    volatile uint64_t* source = (volatile uint64_t*)kmx_region(1, &region_bytes);
    volatile uint64_t* lead = (volatile uint64_t*)kmx_region(0, &region_bytes);
    volatile uint64_t* range = lead + LEAD / 8;
    const uint64_t words = bytes / 8;
    for (uint64_t word = 0; word < words; word++) {
        source[word] = pattern(round, word);
    }
    kmx_flush_at(1, 0, bytes);
    uint64_t sum = 0;
    for (uint64_t word = 0; word < words; word++) {
        sum += range[word];
    }
    for (uint64_t line = round % 5; line < words / 8; line += 5) {
        range[8 * line] = 0xDEAD000000000000ull ^ sum ^ line;
    }
    if (variant != V_PLAIN) {
        for (uint64_t word = 0; word < LEAD / 8; word++) {
            sum += lead[word];
        }
    }
    const kmx_dram from = {
        (uint32_t)kmx_args_get()->region[1].offset,
        KMX_BANK_OF_CLUSTER((uint32_t)kmx_cluster()),
        0,
        0
    };
    if (variant != V_NONE) {
        kmx_evict();
    }
    kmx_event event = {0, 0, 0, 0};
    const uint64_t deadline = kmx_time() + KMX_TICKS_PER_SECOND;
    const int status = (
        kmx_fetch_clean(0, LEAD, &from, bytes, 0, COPY_CORES, deadline, &event)
    );
    if (status != KMX_OK) {
        return status;
    }
    if (variant != V_ONE && variant != V_NONE) {
        kmx_evict();
    }
    for (uint64_t line = 0; line < words / 8; line++) {
        uint32_t bad = 0;
        for (uint64_t word = 8 * line; word < 8 * line + 8; word++) {
            const uint64_t got = range[word], want = pattern(round, word);
            if (got != want && bad++ == 0 && out[P_BAD] == 0) {
                out[P_ROUND] = (uint32_t)round;
                out[P_LINE] = (uint32_t)line;
                out[P_GOT] = (uint32_t)got;
                out[P_GOT + 1] = (uint32_t)(got >> 32);
                out[P_WANT] = (uint32_t)want;
                out[P_WANT + 1] = (uint32_t)(want >> 32);
            }
        }
        out[P_BAD] += bad != 0;
    }
    return KMX_OK;
}

static void proof(const uint32_t* words, uint32_t* out) {
    uint64_t round = words[W_FIRST];
    for (uint32_t variant = 0; variant < VARIANTS; variant++) {
        uint32_t* mine = &out[P_STRIDE * variant];
        if (!(words[W_VARIANTS] >> variant & 1u) || out[P_STATUS] != 0) {
            continue;
        }
        if (variant == V_READERS) {
            *(volatile uint32_t*)&g_stop = 0;
            for (int core = 1; core < 4; core++) {
                kmx_spawn(core, reader, 0);
            }
        }
        const uint64_t start = kmx_time();
        for (uint32_t r = 0; r < words[W_REPEATS] && !kmx_stop_requested(); r++) {
            kmx_progress(1000u * (variant + 1) + r);
            const int status = round_of(variant, round++, words[W_BYTES], mine);
            if (status != KMX_OK) {
                out[P_STATUS] = (uint32_t)status | variant << 8 | 1u << 16;
                break;
            }
            mine[P_ROUNDS]++;
        }
        mine[P_TICKS] = (uint32_t)(kmx_time() - start);
        if (variant == V_READERS) {
            *(volatile uint32_t*)&g_stop = 1;
            for (int core = 1; core < 4; core++) {
                kmx_join(core);
                out[P_READ] += ((volatile uint32_t*)g_read)[core];
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
    } else if (words[W_MODE] == MODE_PROOF) {
        proof(words, result);
        kmx_result(0, result, P_WORDS);
    } else {
        return 1;
    }
    return 0;
}
