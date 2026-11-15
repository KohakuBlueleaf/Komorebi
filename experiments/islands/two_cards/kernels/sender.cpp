// Sender: sends `packets` packets over its fabric link, packet p from slot p % window
// into the receiver's same slot, each counted at the receiver; every wait is bounded.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "link.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric;

constexpr uint32_t kHeaders = 4;  // headers in flight before the first is rewritten
constexpr uint32_t kArgs = 16;    // before the connection's; the host's: compile arg 0

inline uint32_t wall_lo() {
    return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

// A 64-bit cycle count kept from the low word, read often enough not to wrap.
struct Clock {
    uint32_t last = wall_lo();
    uint64_t total = 0;

    uint64_t now() {
        const uint32_t t = wall_lo();
        total += t - last;
        last = t;
        return total;
    }
};

// Polls ready() until it holds or `limit` cycles have passed since `clock`'s `start`.
template <typename Ready>
inline bool poll(Clock& clock, uint64_t start, uint64_t limit, Ready ready) {
    for (;;) {
        invalidate_l1_cache();
        if (ready()) {
            return true;
        }
        if (clock.now() - start > limit) {
            return false;
        }
    }
}

void kernel_main() {
    size_t i = 0;
    const uint32_t status_addr = get_arg_val<uint32_t>(i++);
    const uint32_t status_page = get_arg_val<uint32_t>(i++);
    const uint32_t page_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t status_l1 = get_arg_val<uint32_t>(i++);
    const uint32_t slots_l1 = get_arg_val<uint32_t>(i++);
    const uint32_t counter_l1 = get_arg_val<uint32_t>(i++);
    const uint32_t rx_x = get_arg_val<uint32_t>(i++);
    const uint32_t rx_y = get_arg_val<uint32_t>(i++);
    const uint32_t packets = get_arg_val<uint32_t>(i++);
    const uint32_t payload = get_arg_val<uint32_t>(i++);
    const uint32_t window = get_arg_val<uint32_t>(i++);
    const uint32_t seed = get_arg_val<uint32_t>(i++);
    const uint32_t sender_id = get_arg_val<uint32_t>(i++);
    const uint64_t limit = static_cast<uint64_t>(get_arg_val<uint32_t>(i++)) << 20;
    const uint32_t hops = get_arg_val<uint32_t>(
        i++
    );  // chips along the line to the peer
    const uint32_t delay = get_arg_val<uint32_t>(i++);  // cycles before the first send

    volatile tt_l1_ptr uint32_t*
        status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_l1);
    for (uint32_t w = 0; w < kStatusBytes / 4; w++) {
        status[w] = 0;
    }
    status[kWordMagic] = kMagic;
    uint32_t code = kCodeOk;
    uint32_t sent = 0;
    Clock clock;
    uint64_t first = 0, last = 0;
    if (
        i != kArgs || get_compile_time_arg_val(0) != kArgs || hops == 0 || hops > 255 ||
        payload == 0 || payload % 16 != 0 || window == 0 || slots_l1 % 16 != 0 ||
        counter_l1 % 16 != 0
    ) {
        code = kCodeArgs;
    } else {
        volatile tt_l1_ptr uint32_t*
            slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots_l1);
        for (uint32_t s = 0; s < window; s++) {
            for (uint32_t w = 0; w < payload / 4; w++) {
                slots[s * (payload / 4) + w] = pattern(seed, sender_id, s, w);
            }
        }
        auto sender = (
            WorkerToFabricEdmSender::build_from_args< ProgrammableCoreType::TENSIX>(i)
        );
        volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[kHeaders];
        for (uint32_t h = 0; h < kHeaders; h++) {
            headers[h] = PacketHeaderPool::allocate_header();
            volatile tt_l1_ptr uint32_t*
                words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(headers[h]);
            for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; w++) {
                words[w] = 0;  // no field left from an earlier use of the pool
            }
            headers[h]->to_chip_unicast(static_cast<uint8_t>(hops));
        }
        const uint64_t counter_noc = (
            safe_get_noc_addr(rx_x, rx_y, counter_l1, edm_to_local_chip_noc)
        );
        sender.open();
        // The wait lets the receiver, launched at about the same time, start first.
        const uint64_t opened = clock.now();
        while (clock.now() - opened < delay) {
            ;
        }
        first = clock.now();
        auto sent_out = [] {
            return (
                ncrisc_noc_nonposted_writes_sent(noc_index) &&
                ncrisc_noc_nonposted_writes_sent(get_fabric_worker_noc())
            );
        };
        for (uint32_t p = 0; p < packets; p++) {
            const uint32_t h = p % kHeaders;
            if (h == 0 && !poll(clock, clock.now(), limit, sent_out)) {
                code = kCodeFlushTimeout;  // a header is rewritten once its sends left
                break;
            }
            const uint32_t slot = p % window;
            const uint32_t at = slot * payload;
            headers[h]->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader(
                    safe_get_noc_addr(rx_x, rx_y, slots_l1 + at, edm_to_local_chip_noc),
                    counter_noc,
                    1,
                    true
                ),
                payload
            );
            if (!poll(clock, clock.now(), limit, [&] {
                return sender.edm_has_space_for_packet<1>();
            })) {
                code = kCodeSlotTimeout;
                break;
            }
            sender.send_payload_without_header_non_blocking_from_address(
                slots_l1 + at, payload
            );
            sender.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(headers[h]), sizeof(PACKET_HEADER_TYPE)
            );
            sent++;
        }
        if (code == kCodeOk && !poll(clock, clock.now(), limit, sent_out)) {
            code = kCodeFlushTimeout;
        }
        last = clock.now();
        // The teardown: close_start, then the flushes and the router's answer, bounded.
        sender.close_start();
        auto flushed = [] {
            return (
                ncrisc_noc_nonposted_writes_flushed(noc_index) &&
                ncrisc_noc_nonposted_writes_flushed(get_fabric_worker_noc())
            );
        };
        if (!poll(clock, clock.now(), limit, flushed) ||
            !poll(clock, clock.now(), limit, [&] {
                return *sender.worker_teardown_addr == 1;
            })) {
            code = code == kCodeOk ? kCodeCloseTimeout : code;
        } else {
            *sender.worker_teardown_addr = 0;
        }
    }
    const uint64_t cycles = last - first;
    status[kWordCode] = code;
    status[kWordPackets] = sent;
    status[kWordCyclesLo] = static_cast<uint32_t>(cycles);
    status[kWordCyclesHi] = static_cast<uint32_t>(cycles >> 32);
    const InterleavedAddrGen<true> out = {
        .bank_base_address = status_addr, .page_size = page_bytes
    };
    noc_async_write(status_l1, get_noc_addr(status_page, out), kStatusBytes);
    poll(clock, clock.now(), limit, [] {
        return ncrisc_noc_nonposted_writes_flushed(noc_index);
    });
}
