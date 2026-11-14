// Pages across a link (KM_HANDLER_PAGES): each data-movement CPU of a link core sends
// its share of DRAM pages over its own link, a count a batch; or a count is waited for.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "km_command.hpp"
#include "km_core.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

namespace km {

    // A CPU's L1 (its chunk): the connection's two words, a packet header a slot,
    // then the slots the pages pass through.
    constexpr uint32_t kPagesTeardown = 0;
    constexpr uint32_t kPagesCursor = 16;
    constexpr uint32_t kPagesHeaders = 64;
    constexpr uint32_t kPagesHeaderStride = 64;
    constexpr uint32_t kPagesSlotsMax = 15;
    constexpr uint32_t kPagesSlots = (
        kPagesHeaders + kPagesSlotsMax * kPagesHeaderStride
    );
    constexpr uint32_t kPagesBatch = 64;  // packets a count

    struct PagesRequest {
        uint32_t from;
        uint32_t page;
        uint32_t first;
        uint32_t pages;
        uint32_t to;
        uint32_t to_core;
        uint32_t chunk;
        uint32_t count;
        uint32_t counter;
        uint32_t hops;
        uint32_t mode;
        uint32_t links;
        uint32_t buffer;
    };

    inline PagesRequest pages_of(const km_command_t& command) {
        const uint32_t a[10] = {
            command.a0,
            command.a1,
            command.a2,
            command.a3,
            command.a4,
            command.a5,
            command.a6,
            command.a7,
            command.a8,
            command.a9
        };
        const uint32_t core = a[KM_PAGES_TO_CORE];
        const uint32_t counter = a[KM_PAGES_COUNTER];
        return {
            a[KM_PAGES_FROM],
            a[KM_PAGES_PAGE],
            a[KM_PAGES_FIRST],
            a[KM_PAGES_PAGES],
            a[KM_PAGES_TO],
            core & 0xFFFFu,
            (core >> 16) * 64,
            a[KM_PAGES_COUNT],
            counter & 0xFFFFu,
            (counter >> 16) & 0xFFu,
            counter >> 24,
            a[KM_PAGES_LINKS],
            a[KM_PAGES_BUFFER]
        };
    }

    // The connection of this core to the router of Ethernet channel `channel`: the
    // router's words from the table the fabric's start-up wrote, ours at `area`.
    inline tt::tt_fabric::WorkerToFabricEdmSender pages_connection(
        uint32_t channel, uint32_t area
    ) {
        auto* table = (
            reinterpret_cast<
                tt_l1_ptr tt::tt_fabric::tensix_fabric_connections_l1_info_t*>(
                    FABRIC_CONNECTIONS_BASE
                )
        );
        const auto* conn = &table->read_only[channel];
        auto* flow = &table->read_write[channel];
        return tt::tt_fabric::WorkerToFabricEdmSender(
            true,
            conn->edm_noc_x,
            conn->edm_noc_y,
            conn->edm_buffer_base_addr,
            conn->num_buffers_per_channel,
            conn->edm_connection_handshake_addr,
            conn->edm_worker_location_info_addr,
            conn->buffer_size_bytes,
            conn->buffer_index_semaphore_id,
            reinterpret_cast<volatile uint32_t*>(
                reinterpret_cast<uintptr_t>(&flow->worker_flow_control_semaphore)
            ),
            reinterpret_cast<volatile uint32_t* const>(area + kPagesTeardown),
            area + kPagesCursor,
            static_cast<uint32_t>(conn->worker_free_slots_stream_id),
            StreamId{0xFFFFFFFFu},
            write_reg_cmd_buf,
            write_at_cmd_buf
        );
    }

    // Whether the NoC counter at `reg` has reached `mark` (counts wrap).
    inline bool noc_reached(uint32_t reg, uint32_t mark) {
        return static_cast<int32_t>(NOC_STATUS_READ_REG(noc_index, reg) - mark) >= 0;
    }

    // The far NoC address of target page `p` (`k` pages after the request's first).
    inline uint64_t pages_target(
        const PagesRequest& r,
        const InterleavedAddrGen<true>& to,
        uint32_t p,
        uint32_t k
    ) {
        if (r.to_core == KM_PAGES_DRAM) {
            const uint64_t at = to.get_noc_addr(p);
            return (
                safe_get_noc_addr(
                    NOC_UNICAST_ADDR_X(at),
                    NOC_UNICAST_ADDR_Y(at),
                    static_cast<uint32_t>(at),
                    tt::tt_fabric::edm_to_local_chip_noc
                )
            );
        }
        return (
            safe_get_noc_addr(
                r.to_core & 0xFFu,
                r.to_core >> 8,
                r.to + k * r.page,
                tt::tt_fabric::edm_to_local_chip_noc
            )
        );
    }

    // CPU `cpu` of rank `rank` sends its share on link 2 rank + cpu, reads run ahead
    // in the slots; each batch's last packet adds its pages to the count. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_send(
        const Kernel& kernel, const PagesRequest& r, uint32_t rank, uint32_t cpu
    ) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            return kGaveBadTransfer;  // the counters below are the dedicated mode's
        }
        uint32_t cpus = 0;
        while (cpus < 4 && ((r.links >> 8 * cpus) & 0xFFu) != KM_PAGES_NO_LINK) {
            cpus++;
        }
        const uint32_t link = 2 * rank + cpu;
        if (link >= cpus) {
            return kGaveNone;  // no link of this CPU: no share
        }
        const uint32_t area = r.buffer + cpu * r.chunk;
        const uint32_t slots_fit = (
            r.chunk > kPagesSlots && r.page > 0 ? (r.chunk - kPagesSlots) / r.page : 0
        );
        const uint32_t slots = slots_fit < kPagesSlotsMax ? slots_fit : kPagesSlotsMax;
        if (
            slots == 0 || r.page % 16 != 0 || r.page > KM_PAGES_PACKET_MAX ||
            area % 16 != 0 || r.chunk % 64 != 0 || r.hops == 0 ||
            static_cast<uint64_t>(r.buffer) + 2ull * r.chunk > MEM_L1_SIZE
        ) {
            return kGaveBadTransfer;
        }
        const Partition share = partition(r.pages, link, cpus);
        if (share.len == 0) {
            return kGaveNone;
        }
        const uint32_t limit = kernel.args().wait_limit;
        uint32_t code = kGaveNone;
        const auto wait = [&](auto ready) {
            code = kernel.poll_at(
                kWherePages,
                [&] {
                    invalidate_l1_cache();
                    return ready();
                },
                limit
            );
            return code == kGaveNone;
        };
        const InterleavedAddrGen<true> from = {
            .bank_base_address = r.from, .page_size = r.page
        };
        const InterleavedAddrGen<true> to = {
            .bank_base_address = r.to, .page_size = r.page
        };
        const uint64_t counter = (
            safe_get_noc_addr(
                r.counter & 0xFFu,
                r.counter >> 8,
                r.count,
                tt::tt_fabric::edm_to_local_chip_noc
            )
        );
        for (uint32_t s = 0; s < slots; s++) {
            volatile tt_l1_ptr uint32_t* words = (
                l1(area + kPagesHeaders + s * kPagesHeaderStride)
            );
            for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; w++) {
                words[w] = 0;
            }
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(words)
                ->to_chip_unicast(static_cast<uint8_t>(r.hops));
        }
        l1(area + kPagesTeardown)[0] = 0;
        auto sender = pages_connection((r.links >> 8 * link) & 0xFFu, area);
        sender.open();
        uint32_t read_mark[kPagesSlotsMax];
        uint32_t sent_mark[kPagesSlotsMax];
        const auto read = [&](uint32_t k) {
            const uint32_t s = k % slots;
            noc_async_read(
                from.get_noc_addr(r.first + share.start + k),
                area + kPagesSlots + s * r.page,
                r.page
            );
            read_mark[s] = noc_reads_num_issued[noc_index];
        };
        uint32_t ahead = share.len < slots ? share.len : slots;
        for (uint32_t k = 0; k < ahead; k++) {
            read(k);
        }
        uint32_t batch = 0;
        for (uint32_t k = 0; k < share.len; k++) {
            const uint32_t s = k % slots;
            const uint32_t slot = area + kPagesSlots + s * r.page;
            if (!wait([&] {
                return noc_reached(NIU_MST_RD_RESP_RECEIVED, read_mark[s]);
            })) {
                break;
            }
            const uint32_t p = r.first + share.start + k;
            auto* header = (
                reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                    area + kPagesHeaders + s * kPagesHeaderStride
                )
            );
            const uint64_t at = pages_target(r, to, p, share.start + k);
            batch++;
            if (batch == kPagesBatch || k + 1 == share.len) {
                header->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                        at, counter, static_cast<uint16_t>(batch), true
                    ),
                    r.page
                );
                batch = 0;
            } else {
                header->to_noc_unicast_write(
                    tt::tt_fabric::NocUnicastCommandHeader{at}, r.page
                );
            }
            if (!sender.edm_has_space_for_packet<1>() && !wait([&] {
                return sender.edm_has_space_for_packet<1>();
            })) {
                break;
            }
            sender.send_payload_without_header_non_blocking_from_address(slot, r.page);
            sender.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE)
            );
            sent_mark[s] = noc_nonposted_writes_num_issued[noc_index];
            if (ahead < share.len) {  // the slot again, once its packet left L1
                if (!wait([&] {
                    return noc_reached(NIU_MST_NONPOSTED_WR_REQ_SENT, sent_mark[s]);
                })) {
                    break;
                }
                read(ahead++);
            }
        }
        const uint32_t sending = code;
        // The teardown, also after a failed bound: the router must not keep this core.
        wait([] {
            return ncrisc_noc_nonposted_writes_sent(noc_index);
        });
        sender.close_start();
        const bool closed = wait([] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        });
        if (closed && wait([&] {
            return l1(area + kPagesTeardown)[0] == 1;
        })) {
            l1(area + kPagesTeardown)[0] = 0;
        }
        if (!wait([] {
            return ncrisc_noc_reads_flushed(noc_index);
        })) {
            return code;
        }
        return sending != kGaveNone ? sending : code;
    }

    // The reader waits until the count at r.count reaches r.pages, then takes that
    // much off it (one request a count at a time). kGave* code.
    template <typename Kernel>
    inline uint32_t pages_wait(const Kernel& kernel, const PagesRequest& r) {
        volatile tt_l1_ptr uint32_t* count = l1(r.count);
        const uint32_t code = kernel.poll_at(
            kWherePages,
            [&] {
                invalidate_l1_cache();
                return static_cast<int32_t>(count[0] - r.pages) >= 0;
            },
            kernel.args().wait_limit
        );
        if (code == kGaveNone) {
            count[0] -= r.pages;
        }
        return code;
    }

    // The part of CPU `cpu` (0 the reader, 1 the writer) of a PAGES command; a failure
    // gives the command up (recorded). False: given up.
    template <typename Kernel>
    __attribute__((noinline)) bool pages_serve(
        const Kernel& kernel, const km_command_t& command, uint32_t cpu, uint32_t word
    ) {
        const PagesRequest r = pages_of(command);
        uint32_t code = kGaveNone;
        if (r.mode == KM_PAGES_WAIT) {
            code = cpu == 0 ? pages_wait(kernel, r) : kGaveNone;
        } else if (r.mode == KM_PAGES_SEND) {
            code = pages_send(kernel, r, kernel.me() - first(command), cpu);
        } else {
            code = kGaveBadTransfer;
        }
        if (code != kGaveNone) {
            kernel.gave_up(word, code, kWherePages, slot(command), command.sequence);
            return false;
        }
        return true;
    }

}  // namespace km
