// Hop example, compute: every visit advances the item's generator and adds the state to
// its sum; it hops on until no hops are left, then retires into this core's totals.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/km_dyn.hpp"
#include "komorebi/device/km_worker.hpp"
#include "hop.h"

void kernel_main() {
    km::Compute cp;
    const uint32_t results = cp.arg(km::kDynCArgs);
    cp.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegHop) {
            return;
        }
        km::Dyn dy(cp, r);
        const uint32_t member = dy.member(), count = dy.count();  // counter tiles: none
        uint32_t retired = 0, visits = 0, nulls = 0, spawned = 0, refused = 0, last = 0;
        uint64_t total = 0;
        uint32_t item[kItemWords] = {};
        if (dy.seeding()) {
            const km::Share s = km::share(r.a0, member, count);
            for (uint32_t k = 0; k < s.len; k++) {
                item[kItId] = s.start + k;
                item[kItHops] = r.a1;
                item[kItState] = hop_seed(s.start + k);
                item[kItSum] = 0;
                refused += !dy.spawn(member, item);
            }
            if (r.a5 == member + 1) {  // spawn until the quota is gone
                item[kItId] = kNullId;
                item[kItHops] = 0;
                while (dy.spawn(member, item)) {
                    spawned++;
                }
                refused++;
            }
            dy.seeded();
        }
        bool stall = r.a4 == member + 1;
        while (km::Packet pk = dy.next()) {
            if (stall) {  // test: this core never finishes its first packet
                dy.wait_stop();
                stall = false;
            }
            for (uint32_t i = 0; i < pk.n; i++) {
                const volatile tt_l1_ptr uint32_t* in = pk[i];
                for (uint32_t w = 1; w < kItemWords; w++) {
                    item[w] = in[w];
                }
                visits++;
                if (item[kItId] == kNullId) {
                    nulls++;
                    dy.retire();
                    continue;
                }
                item[kItState] = hop_next(item[kItState]);
                item[kItSum] += item[kItState];
                if (item[kItHops] == 0) {
                    retired++;
                    total += item[kItSum];
                    last = km::wall();
                    dy.retire();
                    continue;
                }
                item[kItHops]--;
                dy.hop(r.a2 == kModeOneCore ? r.a3 : item[kItState] % count, item);
            }
            dy.done(pk);
        }
        UNPACK(({
            volatile tt_l1_ptr uint32_t* o = km::l1(results);
            o[kResRetired] = retired;
            o[kResTotalLo] = static_cast<uint32_t>(total);
            o[kResTotalHi] = static_cast<uint32_t>(total >> 32);
            o[kResVisits] = visits;
            o[kResLastRetire] = last;
            o[kResNull] = nulls;
            o[kResSpawned] = spawned;
            o[kResRefused] = refused;
            o[kResQuota] = dy.quota();
            o[kResRun] = r.seq;
        }));
    });
}
