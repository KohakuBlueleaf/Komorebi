/* Example 29's thread on work cluster 1: CHAIN sends one chain command over every rank;
 * each island's cores walk the records the host put at the same address there. */
#include <stdint.h>

#include "kmx.h"

#define BAD 0xFFFFFFFFu /* the chain went wrong */

/* CHAIN: a0 records' address, a1 handler, a2 first rank, a3 ranks, a4 mode, a5 word,
 * a6 its a1, a7 1 (one piece). Its word: ticks from submit to event, or BAD. */
static uint32_t chain(const km_command_t* given) {
    km_command_t command;
    kmx_event event;
    uint32_t handle = 0;
    kmx_command(&command, 0, given->a1, given->a2, given->a3);
    command.a0 = given->a5;
    command.a1 = given->a6;
    command.a4 = given->a4;
    command.a7 = given->a0;
    command.a8 = 64u | 1u << KM_BLOCK_BANK_SHIFT; /* record 0 in bank 0 */
    const uint64_t started = kmx_time();
    if (
        kmx_submit(&command, &handle) != KMX_OK ||
        kmx_wait(handle, started + KMX_TICKS_PER_SECOND, &event) != KMX_OK ||
        event.kind != KM_EVENT_DONE
    ) {
        return BAD;
    }
    return (uint32_t)(kmx_time() - started);
}

uint32_t kmx_program(const km_command_t* command) {
    return kmx_handler(command) == kHandlerChain ? chain(command) : BAD;
}
