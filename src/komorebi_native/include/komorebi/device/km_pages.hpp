// Pages across a link (KM_HANDLER_PAGES): each data-movement CPU of a link core holds
// a connection to its link, from its open to its kernel's end, and sends its share of
// DRAM pages over it, a count a batch; or a count is waited for.
#pragma once

#include <stdint.h>

#include <new>

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
    constexpr uint32_t kPagesHelloEvery = 4096;  // hello polls between two hellos

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

    // This CPU's held connection: open from its PAGES open to its kernel's end (a
    // kernel's .bss starts zero), its L1 at `area` (`chunk` bytes), `hops` to the far
    // island, its link of the `links` its open named. One sender a connection.
    struct PagesHeld {
        uint32_t open;
        uint32_t area;
        uint32_t chunk;
        uint32_t hops;
        uint32_t link;
        uint32_t links;
    };
    inline PagesHeld pages_held;
    alignas(8) inline uint8_t
        pages_sender_bytes[sizeof(tt::tt_fabric::WorkerToFabricEdmSender)];

    inline tt::tt_fabric::WorkerToFabricEdmSender& pages_sender() {
        return *reinterpret_cast<tt::tt_fabric::WorkerToFabricEdmSender*>(
            pages_sender_bytes
        );
    }

    // The links a request names (bytes before the first KM_PAGES_NO_LINK, at most 4).
    inline uint32_t pages_links(const PagesRequest& r) {
        uint32_t links = 0;
        while (links < 4 && ((r.links >> 8 * links) & 0xFFu) != KM_PAGES_NO_LINK) {
            links++;
        }
        return links;
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

    // CPU `cpu` sends `share` over its held connection, reads run ahead in the slots;
    // each batch's last packet adds its pages to the count. A range: `share` is ranks,
    // each rank's pages into its core's L1, each rank's last packet adding to its
    // count. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_push(
        const Kernel& kernel, const PagesRequest& r, uint32_t cpu, Partition share
    ) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            return kGaveBadTransfer;  // the counters below are the dedicated mode's
        }
        const uint32_t area = r.buffer + cpu * r.chunk;
        const uint32_t slots_fit = (
            r.chunk > kPagesSlots && r.page > 0 ? (r.chunk - kPagesSlots) / r.page : 0
        );
        const uint32_t slots = slots_fit < kPagesSlotsMax ? slots_fit : kPagesSlotsMax;
        if (
            slots == 0 || r.page % 16 != 0 || r.page > KM_PAGES_PACKET_MAX ||
            area % 16 != 0 || r.chunk % 64 != 0 || r.hops == 0 ||
            static_cast<uint64_t>(r.buffer) + 2ull * r.chunk > MEM_L1_SIZE ||
            pages_held.open == 0 || pages_held.area != area
        ) {
            return kGaveBadTransfer;
        }
        const bool range = r.mode == KM_PAGES_RANGE;
        // its pages: those of its ranks (a range), else its share of the pages
        const uint32_t base = range ? share.start * r.pages : share.start;
        const uint32_t total = range ? share.len * r.pages : share.len;
        if (total == 0) {
            return kGaveNone;
        }
        const volatile tt_l1_ptr uint32_t* cores = (
            l1(kernel.args().control + KM_CONTROL_CORES)
        );
        // range: the far core of the share's rank q (the core tables of both islands
        // are the same, checked by the session) and address `at` there
        const auto far = [&](uint32_t q, uint32_t at) {
            const uint32_t xy = cores[r.to_core + q];
            return (
                safe_get_noc_addr(
                    xy & 0xFFFFu, xy >> 16, at, tt::tt_fabric::edm_to_local_chip_noc
                )
            );
        };
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
        uint64_t counter = (
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
        auto& sender = pages_sender();
        uint32_t read_mark[kPagesSlotsMax];
        uint32_t sent_mark[kPagesSlotsMax];
        const auto read = [&](uint32_t k) {
            const uint32_t s = k % slots;
            noc_async_read(
                from.get_noc_addr(r.first + base + k),
                area + kPagesSlots + s * r.page,
                r.page
            );
            read_mark[s] = noc_reads_num_issued[noc_index];
        };
        uint32_t ahead = total < slots ? total : slots;
        for (uint32_t k = 0; k < ahead; k++) {
            read(k);
        }
        uint32_t batch = 0;
        for (uint32_t k = 0; k < total; k++) {
            const uint32_t s = k % slots;
            const uint32_t slot = area + kPagesSlots + s * r.page;
            if (!wait([&] {
                return noc_reached(NIU_MST_RD_RESP_RECEIVED, read_mark[s]);
            })) {
                break;
            }
            const uint32_t p = r.first + base + k;
            auto* header = (
                reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                    area + kPagesHeaders + s * kPagesHeaderStride
                )
            );
            // a range: page j of rank q of the share
            const uint32_t q = range ? share.start + k / r.pages : 0;
            const uint32_t j = range ? k % r.pages : 0;
            const uint64_t at = (
                range ? far(q, r.to + j * r.page) : pages_target(r, to, p, base + k)
            );
            if (range) {
                counter = far(q, r.count);
            }
            batch++;
            if (
                batch == kPagesBatch || k + 1 == total ||
                (range && j + 1 == r.pages)
            ) {
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
            if (ahead < total) {  // the slot again, once its packet left L1
                if (!wait([&] {
                    return noc_reached(NIU_MST_NONPOSTED_WR_REQ_SENT, sent_mark[s]);
                })) {
                    break;
                }
                read(ahead++);
            }
        }
        const uint32_t sending = code;
        // The slots and headers are free for the next command once sent and read.
        if (!wait([] {
            return ncrisc_noc_nonposted_writes_sent(noc_index);
        }) || !wait([] {
            return ncrisc_noc_reads_flushed(noc_index);
        })) {
            return code;
        }
        return sending;
    }

    // CPU `cpu` of link core `rank` sends its share on link 2 rank + cpu (pages_push):
    // link 0 carries the lines; with more links, pages go on links 1 on. kGave* code.
    template <typename Kernel>
    inline uint32_t pages_send(
        const Kernel& kernel, const PagesRequest& r, uint32_t rank, uint32_t cpu
    ) {
        const uint32_t cpus = pages_links(r);
        const uint32_t link = 2 * rank + cpu;
        const uint32_t first_link = cpus > 1 ? 1 : 0;
        if (link >= cpus || link < first_link) {
            return kGaveNone;  // no link of this CPU, or the lines link: no share
        }
        const uint32_t ranks = r.counter & 0xFFFFu;  // a range's
        return pages_push(
            kernel,
            r,
            cpu,
            partition(
                r.mode == KM_PAGES_RANGE ? ranks : r.pages,
                link - first_link,
                cpus - first_link
            )
        );
    }

    // CPU `cpu` of link core `rank` opens the connection of link 2 rank + cpu and
    // holds it, then says hello to the same CPU of the far link core, bounded: its
    // word there 1 (up) until the far word here is not 0, then 2; done when both say
    // 2, so that no request goes out before the far side runs. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_open(
        const Kernel& kernel, const PagesRequest& r, uint32_t rank, uint32_t cpu
    ) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            return kGaveBadTransfer;
        }
        const uint32_t link = 2 * rank + cpu;
        if (link >= pages_links(r)) {
            return kGaveNone;  // no link of this CPU
        }
        const uint32_t area = r.buffer + cpu * r.chunk;
        if (
            pages_held.open != 0 || area % 16 != 0 || r.chunk % 64 != 0 ||
            r.chunk <= kPagesSlots || r.hops == 0 ||
            static_cast<uint64_t>(r.buffer) + 2ull * r.chunk > MEM_L1_SIZE
        ) {
            return kGaveBadTransfer;  // one open a kernel: one sender a connection
        }
        l1(area + kPagesTeardown)[0] = 0;
        auto& sender = *new (pages_sender_bytes) tt::tt_fabric::WorkerToFabricEdmSender(
            pages_connection((r.links >> 8 * link) & 0xFFu, area)
        );
        sender.open();
        pages_held = {1, area, r.chunk, r.hops, link, pages_links(r)};
        auto* header = (
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                area + kPagesHeaders
            )
        );
        for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; w++) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(header)[w] = 0;
        }
        header->to_chip_unicast(static_cast<uint8_t>(r.hops));
        const uint64_t far = (
            safe_get_noc_addr(
                r.counter & 0xFFu,
                r.counter >> 8,
                r.count + cpu * KM_PORT_STRIDE,
                tt::tt_fabric::edm_to_local_chip_noc
            )
        );
        volatile tt_l1_ptr uint32_t* here = l1(
            kernel.args().control + KM_CONTROL_PORT +
            (KM_PAGES_HELLO_PORT + cpu) * KM_PORT_STRIDE
        );
        uint32_t said = 0, polls = 0;
        uint32_t code = kernel.poll_at(
            kWherePages,
            [&] {
                invalidate_l1_cache();
                const uint32_t heard = here[0];
                const uint32_t say = heard != 0 ? 2 : 1;
                if (
                    (say != said || ++polls % kPagesHelloEvery == 0) &&
                    ncrisc_noc_nonposted_writes_sent(noc_index) &&
                    sender.edm_has_space_for_packet<1>()
                ) {
                    header->to_noc_unicast_inline_write(
                        tt::tt_fabric::NocUnicastInlineWriteCommandHeader{far, say}
                    );
                    sender.send_payload_flush_non_blocking_from_address(
                        reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE)
                    );
                    said = say;
                }
                return heard == 2 && said == 2;
            },
            kernel.args().wait_limit
        );
        const uint32_t sent = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            kernel.args().wait_limit
        );
        return code != kGaveNone ? code : sent;
    }

    // LINE (link core 0's link 0 CPU): the packet the dispatcher staged in the link
    // area, then its count line, one hop into the other island's dispatcher, over the
    // held connection; once both left the core. Every wait bounded. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_line(
        const Kernel& kernel, const km_command_t& command
    ) {
        const uint32_t a[6] = {
            command.a0, command.a1, command.a2, command.a3, command.a4, command.a5
        };
        const uint32_t stage = a[KM_LINE_STAGE], number = a[KM_LINE_NUMBER];
        const uint32_t x = a[KM_LINE_TILE] & 0xFFFFu, y = a[KM_LINE_TILE] >> 16;
        const uint32_t area = a[KM_LINE_AREA];
        if (
            pages_held.open == 0 || stage >= KM_LINK_STAGES ||
            (a[KM_LINE_PACKET] | a[KM_LINE_COUNT] | area) % 16 != 0 ||
            static_cast<uint64_t>(area) + KM_LINK_AREA_BYTES > MEM_L1_SIZE
        ) {
            return kGaveBadTransfer;
        }
        uint32_t code = kGaveNone;
        const auto wait = [&](auto ready) {
            code = kernel.poll_at(
                kWherePages,
                [&] {
                    invalidate_l1_cache();
                    return ready();
                },
                kernel.args().wait_limit
            );
            return code == kGaveNone;
        };
        const uint32_t at = area + KM_LINK_STAGE + stage * KM_LINK_STAGE_BYTES;
        if (!wait([&] {  // the dispatcher's stores landed
            return (
                l1(at)[0] == number && l1(at + 64)[0] == number &&
                l1(at + KM_LINK_PACKET_BYTES)[1] == number
            );
        })) {
            return code;
        }
        auto& sender = pages_sender();
        const auto packet = [&](uint32_t which, uint32_t to, uint32_t from, uint32_t n) {
            const uint32_t header = area + KM_LINK_HEADERS + which * 64;
            for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; w++) {
                l1(header)[w] = 0;
            }
            auto* made = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header);
            made->to_chip_unicast(1);
            made->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{
                    safe_get_noc_addr(x, y, to, tt::tt_fabric::edm_to_local_chip_noc)
                },
                n
            );
            if (!wait([&] {
                return sender.edm_has_space_for_packet<1>();
            })) {
                return false;
            }
            sender.send_payload_without_header_non_blocking_from_address(from, n);
            sender.send_payload_flush_non_blocking_from_address(
                header, sizeof(PACKET_HEADER_TYPE)
            );
            return true;
        };
        if (
            packet(0, a[KM_LINE_PACKET], at, KM_LINK_PACKET_BYTES) &&
            packet(1, a[KM_LINE_COUNT], at + KM_LINK_PACKET_BYTES, 64)
        ) {
            wait([] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            });
        }
        return code;
    }

    // At the kernel's end: the held connection's teardown, bounded (the router must
    // not keep this core).
    template <typename Kernel>
    __attribute__((noinline)) void pages_end(const Kernel& kernel) {
        if (pages_held.open == 0) {
            return;
        }
        const uint32_t area = pages_held.area;
        pages_held.open = 0;
        const auto wait = [&](auto ready) {
            return (
                kernel.poll_at(
                    kWherePages,
                    [&] {
                        invalidate_l1_cache();
                        return ready();
                    },
                    kernel.args().wait_limit
                ) == kGaveNone
            );
        };
        wait([] {
            return ncrisc_noc_nonposted_writes_sent(noc_index);
        });
        pages_sender().close_start();
        const bool closed = wait([] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        });
        if (closed && wait([&] {
            return l1(area + kPagesTeardown)[0] == 1;
        })) {
            l1(area + kPagesTeardown)[0] = 0;
        }
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

    // The lines link's CPU (link 0): the next ask line its island's cores wrote into
    // the link area at `area`, once landed and while the far island has room for it,
    // into the far link core 0's asks (the same core and area there: both islands'
    // tables are the same); then its line is free again. One line a call. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_forward(
        const Kernel& kernel, uint32_t area
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        invalidate_l1_cache();
        const uint32_t taken = words[KM_ASK_TAKEN / 4];
        const uint32_t line = area + KM_ASK_LINES + taken % KM_ASK_SLOTS * 64;
        if (
            l1(line)[KM_ASK_NUMBER] != taken + 1 || l1(line)[KM_ASK_END] != taken + 1 ||
            taken - words[KM_ASK_SERVED / 4] >= KM_ASK_SLOTS
        ) {
            return kGaveNone;  // no line yet, or the far asks are full
        }
        const uint32_t xy = kernel.control(KM_CONTROL_CORES)[kernel.args().me];
        const uint32_t header = area + KM_ASK_HEADERS;
        for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; w++) {
            l1(header)[w] = 0;
        }
        auto* made = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header);
        made->to_chip_unicast(static_cast<uint8_t>(pages_held.hops));
        made->to_noc_unicast_write(
            tt::tt_fabric::NocUnicastCommandHeader{
                safe_get_noc_addr(
                    xy & 0xFFFFu,
                    xy >> 16,
                    area + KM_ASK_INBOX + taken % KM_ASK_SLOTS * 64,
                    tt::tt_fabric::edm_to_local_chip_noc
                )
            },
            64
        );
        auto& sender = pages_sender();
        const uint32_t limit = kernel.args().wait_limit;
        uint32_t code = kernel.poll_at(
            kWherePages,
            [&] {
                return sender.edm_has_space_for_packet<1>();
            },
            limit
        );
        if (code != kGaveNone) {
            return code;
        }
        sender.send_payload_without_header_non_blocking_from_address(line, 64);
        sender.send_payload_flush_non_blocking_from_address(
            header, sizeof(PACKET_HEADER_TYPE)
        );
        code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            limit
        );
        if (code == kGaveNone) {
            words[KM_ASK_TAKEN / 4] = taken + 1;
        }
        return code;
    }

    // The page CPU (link 1, or link 0 alone): the far island's next ask in the link
    // area at `area`, once landed: its pages from this island's DRAM into the asker's
    // L1, counted at its port (pages_push), then the count of asks answered into the
    // far link core 0, which may then send the next. An ask this island cannot answer
    // is counted bad and passed (the asker's wait names this island). kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_answer(
        const Kernel& kernel, uint32_t cpu, uint32_t area
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        invalidate_l1_cache();
        const uint32_t answered = words[KM_ASK_ANSWERED / 4];
        const volatile tt_l1_ptr uint32_t* line = (
            l1(area + KM_ASK_INBOX + answered % KM_ASK_SLOTS * 64)
        );
        if (line[KM_ASK_NUMBER] != answered + 1 || line[KM_ASK_END] != answered + 1) {
            return kGaveNone;
        }
        const uint32_t asker = line[KM_ASK_CORE];
        const uint32_t place = (asker & 0xFFu) | (asker >> 16 & 0xFFu) << 8;
        const PagesRequest r = {
            line[KM_ASK_TENSOR],
            line[KM_ASK_PAGE],
            line[KM_ASK_FIRST],
            line[KM_ASK_PAGES],
            line[KM_ASK_TO],
            place,
            pages_held.chunk,
            line[KM_ASK_COUNT],
            place,
            pages_held.hops,
            KM_PAGES_SEND,
            0,
            pages_held.area - cpu * pages_held.chunk
        };
        const bool good = (
            line[KM_ASK_HOLDER] == islands_at(kernel.args().control).here() &&
            r.pages != 0 && r.page != 0 && r.page % 16 == 0 &&
            r.page <= KM_PAGES_PACKET_MAX && r.to % 16 == 0 && r.count % 16 == 0
        );
        uint32_t code = (
            good ? pages_push(kernel, r, cpu, Partition{0, r.pages}) : kGaveNone
        );
        if (code != kGaveNone) {
            return code;
        }
        if (!good) {
            words[KM_ASK_BAD / 4] += 1;
        }
        const uint32_t xy = kernel.control(KM_CONTROL_CORES)[kernel.args().me];
        auto* header = (
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                area + KM_ASK_HEADERS + 64
            )
        );
        for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; w++) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(header)[w] = 0;
        }
        header->to_chip_unicast(static_cast<uint8_t>(pages_held.hops));
        header->to_noc_unicast_inline_write(
            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                safe_get_noc_addr(
                    xy & 0xFFFFu,
                    xy >> 16,
                    area + KM_ASK_SERVED,
                    tt::tt_fabric::edm_to_local_chip_noc
                ),
                answered + 1
            }
        );
        auto& sender = pages_sender();
        const uint32_t limit = kernel.args().wait_limit;
        code = kernel.poll_at(
            kWherePages,
            [&] {
                return sender.edm_has_space_for_packet<1>();
            },
            limit
        );
        if (code != kGaveNone) {
            return code;
        }
        sender.send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE)
        );
        code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            limit
        );
        if (code == kGaveNone) {
            words[KM_ASK_ANSWERED / 4] = answered + 1;
        }
        return code;
    }

    // The link core's services, in one place: the part of CPU `cpu` (0 the reader, 1
    // the writer) of a PAGES command (open, send, range, wait, line); a failure gives
    // the command up (recorded). False: given up.
    template <typename Kernel>
    __attribute__((noinline)) bool pages_serve(
        const Kernel& kernel, const km_command_t& command, uint32_t cpu, uint32_t word
    ) {
        const PagesRequest r = pages_of(command);
        // this link core's index: its name in a group of link cores, less the base
        const uint32_t index = kernel.name(command) - KM_GROUP_LINK_BASE;
        const bool links = KM_GROUP_IS_LINK(first(command));
        uint32_t code = kGaveNone;
        if (r.mode == KM_PAGES_WAIT) {
            code = cpu == 0 ? pages_wait(kernel, r) : kGaveNone;
        } else if ((r.mode == KM_PAGES_SEND || r.mode == KM_PAGES_RANGE) && links) {
            code = pages_send(kernel, r, index, cpu);
        } else if (r.mode == KM_PAGES_LINE && links) {
            // link core 0's link 0 CPU, which holds its connection
            code = cpu == 0 && index == 0 ? pages_line(kernel, command) : kGaveNone;
        } else if (r.mode == KM_PAGES_OPEN && links) {
            code = pages_open(kernel, r, index, cpu);
#if defined(KM_LINKS) && defined(COMPILE_FOR_BRISC)  // the reader: link 0's CPU
        } else if (r.mode == KM_PAGES_MESSAGES && links) {
            code = index == 0 && cpu == 0 ? link_messages(kernel, command) : kGaveNone;
#elif defined(KM_LINKS)
        } else if (r.mode == KM_PAGES_MESSAGES && links) {
            code = kGaveNone;
#endif
        } else {
            code = kGaveBadTransfer;
        }
        if (code != kGaveNone) {
            kernel.gave_up(word, code, kWherePages, slot(command), command.sequence);
            return false;
        }
        return true;
    }

    // The link core's services between commands, in one place, a step of each a call
    // (the reader's wait for its next command and the writer's for its token call
    // it): the lines link's CPU sends its island's next ask, the page CPU answers the
    // far island's next. A failure stops the core (recorded in state word `word`).
    template <typename Kernel>
    __attribute__((noinline)) void pages_serve(
        const Kernel& kernel, uint32_t cpu, uint32_t word
    ) {
        const uint32_t area = islands_at(kernel.args().control).link_area();
        if (pages_held.open == 0 || area == 0) {
            return;
        }
        // no command runs: no command's abort ends these waits
        volatile tt_l1_ptr uint32_t* live = kernel.control(KM_CONTROL_LIVE);
        const uint32_t sequence = live[kLvSequence + kernel.role()];
        live[kLvSequence + kernel.role()] = 0;
        uint32_t code = kGaveNone;
        if (pages_held.link == 0) {
            code = pages_forward(kernel, area);
        }
        if (code == kGaveNone && pages_held.link == (pages_held.links > 1 ? 1 : 0)) {
            code = pages_answer(kernel, cpu, area);
        }
        live[kLvSequence + kernel.role()] = sequence;
        if (code != kGaveNone) {
            kernel.gave_up(word, code, kWherePages, 0, 0);
        }
    }

}  // namespace km
