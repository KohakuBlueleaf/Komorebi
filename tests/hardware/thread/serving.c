/* A thread with no kmx_main: PARALLEL runs one kmx_parallel and answers how many cores
 * marked their line of region "r". */
#include <stdint.h>

#include "kmx.h"

static void mark(int core, void* arg) {
    ((volatile uint32_t*)arg)[16u * (uint32_t)core] = 1u;
}

uint32_t kmx_program(const km_command_t* command) {
    (void)command;
    uint64_t bytes = 0;
    volatile uint32_t* marks = (volatile uint32_t*)kmx_region(0, &bytes);
    if (kmx_parallel(mark, (void*)marks) != KMX_OK) {
        return 0xFFFFFFFFu;
    }
    uint32_t cores = 0;
    for (uint32_t core = 0; core < 4; core++) {
        cores += marks[16u * core];
    }
    return cores;
}
