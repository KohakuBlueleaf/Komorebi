/* Example 24, the thread: X280 cores 0 to kmx_ranks() - 1 are ranks of the cores' run;
 * a message is stepped and goes on to a core, or ends here once its hops are used up. */
#include <stdint.h>

#include "kmx.h"
#include "rank.h"

#define RUN_WAIT_SECONDS 20u /* the RUN may come this long after the start */

static void rank(int x280_core, kmx_messages* messages, void* arg) {
    uint32_t* totals = (uint32_t*)arg + kX280Words * (uint32_t)x280_core;
    uint64_t sum = 0;
    uint32_t message[kMessageWords];
    const volatile uint32_t* incoming;
    while ((incoming = kmx_messages_next(messages)) != 0) {
        for (uint32_t word = 1; word < kMessageWords; word++) {
            message[word] = incoming[word];
        }
        message[kMsgValue] = x280_step(message[kMsgValue], messages->rank);
        if (message[kMsgHops] == 0) {
            totals[kX280Consumed]++;
            sum += message[kMsgValue];
            totals[kX280Xor] ^= end_mix(message[kMsgId], message[kMsgValue]);
            kmx_messages_consume(messages);
            continue;
        }
        message[kMsgHops]--;
        const uint32_t to = message[kMsgValue] % messages->count;
        if (kmx_messages_forward(messages, to, message, kHandlerGrid) != KMX_OK) {
            break;
        }
    }
    totals[kX280SumLo] = (uint32_t)sum;
    totals[kX280SumHi] = (uint32_t)(sum >> 32);
    totals[kX280Takes] = messages->consumed;
    totals[kX280Sent] = messages->sent;
    totals[kX280Gave] = messages->gave;
}

/* One run: the totals of rank r at region 0's word r * kX280Words (the library flushes
 * the regions at the end). The code: 0, or 1 + kmx_messages_run's status. */
int kmx_main(int core) {
    (void)core;
    uint64_t bytes = 0;
    uint32_t* totals = (uint32_t*)kmx_region(0, &bytes);
    if (totals == 0 || bytes < 4u * kX280Words * KM_X280_RANKS_MAX) {
        return 1 + KMX_BAD;
    }
    const uint64_t deadline = kmx_time() + RUN_WAIT_SECONDS * KMX_TICKS_PER_SECOND;
    const int status = kmx_messages_run(rank, totals, deadline);
    return status == KMX_OK ? 0 : 1 + status;
}
