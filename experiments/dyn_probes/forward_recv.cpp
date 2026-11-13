// Forward probe, receiver: reads the reserved positions [head, tail) of this core's
// DRAM queue and consumes the prefix whose marks are right, up to `expected` messages.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "forward.h"

using namespace forward;

namespace {

    uint64_t wall64() {
        const uint32_t lo = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
        );
        const uint32_t hi = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H)
        );
        return static_cast<uint64_t>(hi) << 32 | lo;
    }

    bool reads_flushed_timeout() {
        const uint64_t t0 = wall64();
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            if (wall64() - t0 > kLimit) {
                return false;
            }
        }
        return true;
    }

}  // namespace

// Args: shard, queue base, cap, me, expected messages, salt.
void kernel_main() {
    const uint32_t sh = get_arg_val<uint32_t>(0), qbase = get_arg_val<uint32_t>(1);
    const uint32_t cap = get_arg_val<uint32_t>(2), me = get_arg_val<uint32_t>(3);
    const uint32_t expected = get_arg_val<uint32_t>(4), salt = get_arg_val<uint32_t>(5);
    volatile uint32_t* st = reinterpret_cast<volatile uint32_t*>(sh + kRStat);
    volatile uint32_t* tail = reinterpret_cast<volatile uint32_t*>(sh + kTail);
    volatile uint32_t* buf = reinterpret_cast<volatile uint32_t*>(sh + kRecv);
    for (uint32_t w = 0; w < 16; w++) {
        st[w] = 0;
    }
    const InterleavedAddrGen<true> q = {
        .bank_base_address = qbase, .page_size = kChunkMessages * kMessageBytes
    };
    uint32_t head = 0, torn = 0, shorts = 0, reads = 0, bad = 0, first = 0, last = 0;
    uint32_t fail = kOk;
    uint64_t sum = 0, since = wall64();
    while (head < expected) {
        invalidate_l1_cache();
        const uint32_t t = *tail;
        if (t - head > cap) {
            fail = kFailOverflow;
            break;
        }
        if (wall64() - since > (head == 0 ? kStartLimit : kLimit)) {  // nothing taken
            fail = kFailGap;
            st[kRGapPos] = head;
            break;
        }
        if (t == head) {
            continue;
        }
        uint32_t k = t - head;
        k = k < kMaxPacket ? k : kMaxPacket;
        for (uint32_t i = 0; i < k;) {
            const uint32_t pos = (head + i) & (cap - 1);
            const uint32_t in_page = pos % kChunkMessages;
            uint32_t run = kChunkMessages - in_page;
            run = run < k - i ? run : k - i;
            const uint32_t page = me * (cap / kChunkMessages) + pos / kChunkMessages;
            noc_async_read(
                get_noc_addr(page, q, in_page * kMessageBytes),
                sh + kRecv + i * kMessageBytes,
                run * kMessageBytes
            );
            i += run;
        }
        reads++;
        if (!reads_flushed_timeout()) {
            fail = kFailFlush;
            break;
        }
        invalidate_l1_cache();
        uint32_t m = 0;
        for (; m < k; m++) {
            volatile uint32_t* it = buf + m * kMessageWords;
            const uint32_t mark = (head + m + 1) ^ salt;
            if (it[0] != mark) {
                break;
            }
            if (it[4] != mark || it[8] != mark || it[12] != mark) {
                torn++;  // word 0 landed, another chunk of the message not yet
                break;
            }
            if (it[kWPayload] != payload(it[kWSrc], it[kWPacket], it[kWIndex])) {
                bad++;
            }
            sum += it[kWPayload];
        }
        if (m < k) {
            shorts++;
        }
        if (m > 0) {
            last = static_cast<uint32_t>(wall64());
            first = head == 0 ? last : first;
            head += m;
            since = wall64();
        }
    }
    st[kRGot] = head;
    st[kRSumLo] = static_cast<uint32_t>(sum);
    st[kRSumHi] = static_cast<uint32_t>(sum >> 32);
    st[kRTorn] = torn;
    st[kRShort] = shorts;
    st[kRFirst] = first;
    st[kRLast] = last;
    st[kRReads] = reads;
    st[kRBadSrc] = bad;
    st[kRFail] = fail;
    st[kRDone] = 1;
}
