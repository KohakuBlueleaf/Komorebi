/* The thread of example 13: kmx_main reads its table once, then kmx_program answers
 * each command to this cluster. kHandlerFold and kHandlerServed are its handlers. */
#include <stdint.h>

#include "kmx.h"

#define TABLE_WORDS 256u
#define FOLD_READS 16u
#define FOLD_STRIDE 17u

/* in .data: the image links no .bss */
static const volatile uint32_t* g_table __attribute__((section(".data"))) = 0;
static uint32_t g_served __attribute__((section(".data"))) = 0;

int kmx_main(int core) {
    (void)core;
    uint64_t bytes = 0;
    kmx_invalidate(
        0
    ); /* the table the host wrote at load, not an earlier run's lines */
    const void* table = kmx_region(0, &bytes);
    g_table = bytes >= 4u * TABLE_WORDS ? (const volatile uint32_t*)table : 0;
    return g_table != 0 ? 0 : 1;
}

/* x xor the sum of 16 table words picked by x (thread_commands.py's fold). */
static uint32_t fold(uint32_t x) {
    if (g_table == 0) {
        return 0xFFFFFFFFu;
    }
    uint32_t sum = 0;
    for (uint32_t k = 0; k < FOLD_READS; k++) {
        sum += g_table[(x + k * FOLD_STRIDE) % TABLE_WORDS];
    }
    return x ^ sum;
}

uint32_t kmx_program(const km_command_t* command) {
    g_served++;
    switch (kmx_handler(command)) {
        case kHandlerFold:
            return fold(command->a0);
        case kHandlerServed:
            return g_served;
    }
    return 0;
}
