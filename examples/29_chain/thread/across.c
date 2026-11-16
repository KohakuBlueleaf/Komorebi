/* Example 29's thread over islands, on work cluster 1: CHAIN sends one chain command
 * whose group is every rank; the dispatcher sends each island's part to its island,
 * whose cores walk the records the host put at the same address there. */
#include <stdint.h>

#include "kmx.h"

#define BAD 0xFFFFFFFFu /* the chain went wrong */

/* CHAIN: a0 the records' bank address, a1 the chain's handler, a2 its first rank, a3
 * its ranks, a4 its mode, a5 the per-submit word, a6 its a1, a7 its sequence numbers.
 * Its word: the ticks from the submit to the chain's event, or BAD. */
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
    if (given->a7 > 1u) {
        command.hdr |= KM_FLAG_SPAN << 8;
        command.a9 = given->a7 - 1u;
    }
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
