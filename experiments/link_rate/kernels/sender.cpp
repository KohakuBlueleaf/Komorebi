// Sender: streams packets over its fabric link from slot p % window into the
// destination's same slot; headers written once, every wait bounded. `send` picks how
// a packet leaves: 0 the payload and then its header (two writes and a credit each),
// 1 the same two writes on command buffers set up once, 2 header and payload one
// after the other in L1 and one write on set-up command buffers, 3 as 2 with posted
// writes; 1 to 3 check the router's free slots only when the count read last is used
// up.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "link.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

using namespace tt::tt_fabric;

constexpr uint32_t kArgs = 25;  // before the connection's; the host's: compile arg 0

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

// Polls ready() until it holds or `limit` cycles have passed since `start`.
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
    const uint32_t headers_l1 = get_arg_val<uint32_t>(i++);
    const uint32_t slots_l1 = get_arg_val<uint32_t>(i++);
    const uint32_t dst_x = get_arg_val<uint32_t>(i++);
    const uint32_t dst_y = get_arg_val<uint32_t>(i++);
    const uint32_t dst_addr = get_arg_val<uint32_t>(i++);
    const uint32_t cnt_x = get_arg_val<uint32_t>(i++);
    const uint32_t cnt_y = get_arg_val<uint32_t>(i++);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(i++);
    const uint32_t packets = get_arg_val<uint32_t>(i++);
    const uint32_t payload = get_arg_val<uint32_t>(i++);
    const uint32_t window = get_arg_val<uint32_t>(i++);
    const uint32_t seed = get_arg_val<uint32_t>(i++);
    const uint32_t sender_id = get_arg_val<uint32_t>(i++);
    const uint64_t limit = static_cast<uint64_t>(get_arg_val<uint32_t>(i++)) << 20;
    const uint32_t hops = get_arg_val<uint32_t>(i++);
    const uint32_t delay = get_arg_val<uint32_t>(i++);  // cycles before the first send
    const uint32_t fused = get_arg_val<uint32_t>(i++);  // 1: every packet is counted
    const uint32_t batch = get_arg_val<uint32_t>(i++);  // plain mode: packets a count
    const uint32_t wait_every = get_arg_val<uint32_t>(i++);  // 0: never wait for sends
    const uint32_t space_every = get_arg_val<uint32_t>(
        i++
    );  // 1 or 8: packets a space check
    const uint32_t send = get_arg_val<uint32_t>(i++);  // 0 to 3, above

    volatile tt_l1_ptr uint32_t*
        status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_l1);
    for (uint32_t w = 0; w < kStatusBytes / 4; w++) {
        status[w] = 0;
    }
    status[kWordMagic] = kMagic;
    uint32_t code = kCodeOk;
    uint32_t sent = 0;
    Clock clock;
    uint64_t first = 0, last = 0, waited = 0;
    uint32_t slots_in_router = 0;
    if (
        i != kArgs || get_compile_time_arg_val(0) != kArgs || hops == 0 || hops > 255 ||
        payload == 0 || payload % 16 != 0 || window == 0 || slots_l1 % 16 != 0 ||
        headers_l1 % 16 != 0 || dst_addr % 16 != 0 || cnt_addr % 16 != 0 ||
        batch == 0 || send > 3 || sizeof(PACKET_HEADER_TYPE) % 16 != 0
    ) {
        code = kCodeArgs;
    } else {
        // send 2 and 3: packet h (2 * window of them, the plain then the fused) at
        // headers_l1 + h * (header + payload), its payload right after its header (a
        // header of 16-byte words)
        const bool joined = send >= 2;
        const uint32_t packet_bytes = sizeof(PACKET_HEADER_TYPE) + payload;
        const auto header_at = [&](uint32_t h) {
            return headers_l1 + h * (joined ? packet_bytes : kHeaderStride);
        };
        for (uint32_t h = 0; h < (joined ? 2 * window : window); h++) {
            volatile tt_l1_ptr uint32_t* slots = (
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    joined ? header_at(h) + sizeof(PACKET_HEADER_TYPE)
                           : slots_l1 + h * payload
                )
            );
            for (uint32_t w = 0; w < payload / 4; w++) {
                slots[w] = pattern(seed, sender_id, h % window, w);
            }
        }
        auto sender = (
            WorkerToFabricEdmSender::build_from_args< ProgrammableCoreType::TENSIX>(i)
        );
        // Headers 0 to window - 1: a plain write into the slot; window to 2 * window
        // - 1: the same write fused with an increment of the counter.
        const uint64_t counter_noc = (
            safe_get_noc_addr(cnt_x, cnt_y, cnt_addr, edm_to_local_chip_noc)
        );
        for (uint32_t h = 0; h < 2 * window; h++) {
            volatile tt_l1_ptr PACKET_HEADER_TYPE*
                header = (
                    reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                        header_at(h)
                    )
                );
            volatile tt_l1_ptr uint32_t*
                words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(header);
            for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; w++) {
                words[w] = 0;
            }
            header->to_chip_unicast(static_cast<uint8_t>(hops));
            const uint64_t at = (
                safe_get_noc_addr(
                    dst_x, dst_y, dst_addr + (h % window) * payload, edm_to_local_chip_noc
                )
            );
            if (h < window) {
                header->to_noc_unicast_write(NocUnicastCommandHeader{at}, payload);
            } else {
                header->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader(at, counter_noc, 1, true),
                    payload
                );
            }
        }
        slots_in_router = sender.num_buffers_per_channel;
        sender.open();
        if (send == 3) {
            sender.setup_stateful_send_cmd_bufs<true>(noc_index);
        } else if (send != 0) {
            sender.setup_stateful_send_cmd_bufs<false>(noc_index);
        }
        uint32_t free_slots = 0;  // send 1 to 3: the router's free slots read last
        // The wait lets the receiver, launched at about the same time, start first.
        const uint64_t opened = clock.now();
        while (clock.now() - opened < delay) {
            ;
        }
        first = clock.now();
        auto sent_out = [] {
            return (
                ncrisc_noc_nonposted_writes_sent(noc_index) &&
                ncrisc_noc_nonposted_writes_sent(get_fabric_worker_noc()) &&
                ncrisc_noc_posted_writes_sent(noc_index)
            );
        };
        for (uint32_t p = 0; p < packets; p++) {
            if (
                wait_every && p % wait_every == 0 &&
                !poll(clock, clock.now(), limit, sent_out)
            ) {
                code = kCodeFlushTimeout;
                break;
            }
            const uint32_t slot = p % window;
            const bool counted = fused || (p + 1) % batch == 0 || p + 1 == packets;
            const uint32_t h = (counted ? window : 0) + slot;
            if (send != 0) {
                if (free_slots == 0) {
                    free_slots = sender.get_num_free_write_slots();
                }
                if (free_slots == 0) {
                    const uint64_t waiting = clock.now();
                    if (!poll(clock, waiting, limit, [&] {
                        free_slots = sender.get_num_free_write_slots();
                        return free_slots != 0;
                    })) {
                        code = kCodeSlotTimeout;
                        break;
                    }
                    waited += clock.now() - waiting;
                }
                if (send == 1) {
                    sender.send_current_slot_stateful_non_blocking<false>(
                        slots_l1 + slot * payload, payload, header_at(h), noc_index
                    );
                } else if (send == 2) {
                    sender.send_current_slot_stateful_non_blocking_from_address<false>(
                        header_at(h), packet_bytes, noc_index
                    );
                } else {
                    sender.send_current_slot_stateful_non_blocking_from_address<true>(
                        header_at(h), packet_bytes, noc_index
                    );
                }
                free_slots--;
                sent++;
                continue;
            }
            // One check for the next eight slots covers eight packets.
            if (space_every == 8 ? p % 8 == 0 : true) {
                const bool free = (
                    space_every == 8
                        ? sender.edm_has_space_for_packet<8>()
                        : sender.edm_has_space_for_packet<1>()
                );
                if (!free) {
                    const uint64_t waiting = clock.now();
                    if (!poll(clock, waiting, limit, [&] {
                        return (
                            space_every == 8
                                ? sender.edm_has_space_for_packet<8>()
                                : sender.edm_has_space_for_packet<1>()
                        );
                    })) {
                        code = kCodeSlotTimeout;
                        break;
                    }
                    waited += clock.now() - waiting;
                }
            }
            sender.send_payload_without_header_non_blocking_from_address(
                slots_l1 + slot * payload, payload
            );
            sender.send_payload_flush_non_blocking_from_address(
                header_at(h), sizeof(PACKET_HEADER_TYPE)
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
    status[kWordWaitLo] = static_cast<uint32_t>(waited);
    status[kWordWaitHi] = static_cast<uint32_t>(waited >> 32);
    status[kWordSlots] = slots_in_router;
    const InterleavedAddrGen<true> out = {
        .bank_base_address = status_addr, .page_size = page_bytes
    };
    noc_async_write(status_l1, get_noc_addr(status_page, out), kStatusBytes);
    poll(clock, clock.now(), limit, [] {
        return ncrisc_noc_nonposted_writes_flushed(noc_index);
    });
}
