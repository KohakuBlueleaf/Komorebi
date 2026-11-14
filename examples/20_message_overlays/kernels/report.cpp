// The report handler, an overlay part: at the thread's origin, count its sum and its
// probes into the core's results, and consume it.
#include <stdint.h>

#include "komorebi/device/km_messages.hpp"
#include "handlers.h"

void km_message(km::Messages& messages, km::Packet& packet) {
    volatile tt_l1_ptr uint32_t* results = km::l1(messages.arg(0));
    uint32_t wrong = 0, probes = 0;
    uint64_t sum = 0;
    for (uint32_t index = 0; index < packet.count; index++) {
        const volatile tt_l1_ptr uint32_t* incoming = packet[index];
        wrong += incoming[kMsgOrigin] != messages.rank();
        sum += incoming[kMsgSum];
        probes += incoming[kMsgProbes];
        messages.consume();
    }
    UNPACK(({
        sum += results[kResSumLo] | static_cast<uint64_t>(results[kResSumHi]) << 32;
        results[kResSumLo] = static_cast<uint32_t>(sum);
        results[kResSumHi] = static_cast<uint32_t>(sum >> 32);
        results[kResReports] += packet.count;
        results[kResProbes] += probes;
        results[kResWrong] += wrong;
    }));
}
