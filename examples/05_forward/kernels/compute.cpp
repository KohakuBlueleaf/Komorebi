// Forward example, compute: each visit advances the message's generator and adds the
// state to its sum; forwarded until none are left, then consumed into core totals.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "forward.h"

void kernel_main() {
    km::Compute cp;
    const uint32_t results = cp.arg(km::kMessagesCArgs);
    cp.run([&](const km_command_t& r) {
        if (km::handler(r) != kHandlerForward) {
            return;
        }
        km::Messages dy(cp, r);
        const uint32_t rank = dy.rank(), count = dy.count();  // counter tiles: none
        uint32_t consumed = 0, visits = 0, nulls = 0, sent = 0, rejected = 0, last = 0;
        uint64_t total = 0;
        uint32_t message[kMessageWords] = {};
        if (dy.seeding()) {
            const km::Partition s = km::partition(r.a0, rank, count);
            for (uint32_t k = 0; k < s.len; k++) {
                message[kMsgId] = s.start + k;
                message[kMsgForwards] = r.a1;
                message[kMsgState] = forward_seed(s.start + k);
                message[kMsgSum] = 0;
                rejected += !dy.send(rank, message);
            }
            if (r.a5 == rank + 1) {  // send until the quota is gone
                message[kMsgId] = kNullId;
                message[kMsgForwards] = 0;
                while (dy.send(rank, message)) {
                    sent++;
                }
                rejected++;
            }
            dy.seeded();
        }
        bool stall = r.a4 == rank + 1;
        while (km::Packet pk = dy.next()) {
            if (stall) {  // test: this core never finishes its first packet
                dy.wait_stop();
                stall = false;
            }
            for (uint32_t i = 0; i < pk.count; i++) {
                const volatile tt_l1_ptr uint32_t* in = pk[i];
                for (uint32_t w = 1; w < kMessageWords; w++) {
                    message[w] = in[w];
                }
                visits++;
                if (message[kMsgId] == kNullId) {
                    nulls++;
                    dy.consume();
                    continue;
                }
                message[kMsgState] = forward_next(message[kMsgState]);
                message[kMsgSum] += message[kMsgState];
                if (message[kMsgForwards] == 0) {
                    consumed++;
                    total += message[kMsgSum];
                    last = km::wall();
                    dy.consume();
                    continue;
                }
                message[kMsgForwards]--;
                dy.forward(
                    r.a2 == kModeOneCore ? r.a3 : message[kMsgState] % count, message
                );
            }
            dy.done(pk);
        }
        UNPACK(({
            volatile tt_l1_ptr uint32_t* o = km::l1(results);
            o[kResConsumed] = consumed;
            o[kResTotalLo] = static_cast<uint32_t>(total);
            o[kResTotalHi] = static_cast<uint32_t>(total >> 32);
            o[kResVisits] = visits;
            o[kResLastConsume] = last;
            o[kResNull] = nulls;
            o[kResSent] = sent;
            o[kResRejected] = rejected;
            o[kResQuota] = dy.quota();
            o[kResRun] = r.sequence;
        }));
    });
}
