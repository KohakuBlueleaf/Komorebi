// Clock probe: core 0 stamps a ping to each peer, the peer stamps its arrival back;
// two rounds `spin` cycles apart. Words at `out`, in each core's own L1 shard.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_core.hpp"

namespace {

    // Waits until *w != 0 (want == 0) or *w == want, at most `limit` cycles.
    bool wait_for(volatile tt_l1_ptr uint32_t* w, uint32_t want, uint64_t limit) {
        const uint64_t t0 = km::wall64();
        do {
            invalidate_l1_cache();
            if (want == 0 ? *w != 0 : *w == want) {
                return true;
            }
        } while (km::wall64() - t0 < limit);
        return false;
    }

    // Waits until this core's non-posted writes are acknowledged, at most `limit`
    // cycles (the barrier, bounded).
    bool flushed(uint64_t limit) {
        const uint64_t t0 = km::wall64();
        do {
            if (ncrisc_noc_nonposted_writes_flushed(noc_index)) {
                return true;
            }
        } while (km::wall64() - t0 < limit);
        return false;
    }

}  // namespace

void kernel_main() {
    const uint32_t out = get_arg_val<uint32_t>(0), index = get_arg_val<uint32_t>(1);
    const uint32_t peers = get_arg_val<uint32_t>(2), spin = get_arg_val<uint32_t>(3);
    const uint64_t limit = static_cast<uint64_t>(spin) + 2000000000ull;
    volatile tt_l1_ptr uint32_t* o = km::l1(out);
    const uint64_t start = km::wall64();
    o[6] = 0xC10C0000u | index;  // which core this shard belongs to
    // A NoC write keeps the low 4 address bits: source and target words share them.
    if (index == 0) {
        // o[8 + 4 * (p * 2 + r) + {0, 1, 2}] = sent, the peer's arrival, reply.
        for (uint32_t r = 0; r < 2; r++) {
            if (r == 1) {
                while (km::wall64() - start < spin) {
                    ;
                }
            }
            for (uint32_t p = 0; p < peers; p++) {
                const uint32_t xy = get_arg_val<uint32_t>(4 + p);
                volatile tt_l1_ptr uint32_t* w = o + 8 + 4 * (p * 2 + r);
                o[0] = r + 1;  // to the peer's word 0
                const uint32_t sent = km::wall();
                noc_async_write(out, get_noc_addr(xy & 0xFFFF, xy >> 16, out), 4);
                o[7] += !flushed(limit);  // o[7]: waits that ran out (failures)
                o[7] += !wait_for(w + 1, 0, limit);
                const uint32_t reply = km::wall();
                w[0] = sent;
                w[2] = reply;
            }
        }
        const uint64_t end = km::wall64();
        o[4] = static_cast<uint32_t>(end - start);
        o[5] = static_cast<uint32_t>((end - start) >> 32);
    } else {
        // Peer: on ping r + 1 in o[0], stamps its clock (bit 0 set) into the master's
        // word at the same address in both shards.
        const uint32_t xy = get_arg_val<uint32_t>(4);
        for (uint32_t r = 0; r < 2; r++) {
            if (!wait_for(o, r + 1, limit)) {
                o[7] += 1;
                break;
            }
            const uint32_t at = out + 4 * (8 + 4 * ((index - 1) * 2 + r) + 1);
            *km::l1(at) = km::wall() | 1u;
            noc_async_write(at, get_noc_addr(xy & 0xFFFF, xy >> 16, at), 4);
            o[7] += !flushed(limit);
        }
    }
}
