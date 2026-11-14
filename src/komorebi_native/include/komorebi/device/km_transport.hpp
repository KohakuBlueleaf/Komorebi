// A link's transport, the one place a link core's packets leave by: a fabric connection
// to an Ethernet router (two cards), or with KM_WIRE (islands of one card) NoC writes
// straight to the far tile. The services above it are the same on both.
#pragma once

#include <stdint.h>

#include <new>

#include "api/dataflow/dataflow_api.h"
#include "km_core.hpp"

#if !defined(KM_WIRE)
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#endif

// Wall-clock ticks a packet waits on the wire before it leaves (a slow link imitated).
#if !defined(KM_WIRE_DELAY)
#define KM_WIRE_DELAY 0
#endif

namespace km {

#if defined(KM_WIRE)
    // A packet's header on the wire, in its L1 slot as a fabric header is: what
    // LinkSender::send does with the payload.
    struct LinkHeader {
        uint32_t kind;             // kWire*
        uint32_t add;              // the increment
        uint32_t to_low, to_high;  // the write's, the word's or the increment's address
        uint32_t counter_low, counter_high;  // a counted write: its count's address
        uint32_t word;                       // an inline write's word
        uint32_t pad;
    };

    constexpr uint32_t kWireWrite = 1;
    constexpr uint32_t kWireCounted = 2;
    constexpr uint32_t kWireInline = 3;
    constexpr uint32_t kWireIncrement = 4;
    constexpr uint32_t kWireFlushPolls = 1u << 26;  // a counted write's wait, bounded
#else
    using LinkHeader = PACKET_HEADER_TYPE;
#endif
    static_assert(sizeof(LinkHeader) <= 64, "a header slot is 64 bytes");

    using LinkHeaderAt = volatile tt_l1_ptr LinkHeader*;

    // Header slot `at` zeroed and routed `hops` chips on; the caller sets its command.
    inline LinkHeaderAt link_header(uint32_t at, uint32_t hops) {
        volatile tt_l1_ptr uint32_t*
            words = (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(at));
        for (uint32_t word = 0; word < sizeof(LinkHeader) / 4; word++) {
            words[word] = 0;
        }
        LinkHeaderAt header = reinterpret_cast<LinkHeaderAt>(at);
#if !defined(KM_WIRE)
        header->to_chip_unicast(static_cast<uint8_t>(hops));
#else
        (void)hops;
#endif
        return header;
    }

    // The NoC address of `address` on tile (x, y) of the far island.
    inline uint64_t link_noc(uint32_t x, uint32_t y, uint32_t address) {
#if defined(KM_WIRE)
        return get_noc_addr(x, y, address);
#else
        return safe_get_noc_addr(x, y, address, tt::tt_fabric::edm_to_local_chip_noc);
#endif
    }

#if defined(KM_WIRE)
    inline void wire_set(LinkHeaderAt header, uint32_t kind, uint64_t to) {
        header->kind = kind;
        header->to_low = static_cast<uint32_t>(to);
        header->to_high = static_cast<uint32_t>(to >> 32);
    }
#endif

    // The header's command: the payload of `bytes` into `to`.
    inline void link_write(LinkHeaderAt header, uint64_t to, uint32_t bytes) {
#if defined(KM_WIRE)
        (void)bytes;
        wire_set(header, kWireWrite, to);
#else
        header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{to}, bytes);
#endif
    }

    // The payload into `to`, then `add` on the word at `counter` once it has landed.
    inline void link_counted_write(
        LinkHeaderAt header, uint64_t to, uint64_t counter, uint16_t add, uint32_t bytes
    ) {
#if defined(KM_WIRE)
        (void)bytes;
        wire_set(header, kWireCounted, to);
        header->counter_low = static_cast<uint32_t>(counter);
        header->counter_high = static_cast<uint32_t>(counter >> 32);
        header->add = add;
#else
        header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader(
                to, counter, add, true
            ),
            bytes
        );
#endif
    }

    // The word `word` into `to`, no payload.
    inline void link_inline(LinkHeaderAt header, uint64_t to, uint32_t word) {
#if defined(KM_WIRE)
        wire_set(header, kWireInline, to);
        header->word = word;
#else
        header->to_noc_unicast_inline_write(
            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{to, word}
        );
#endif
    }

    // `add` on the word at `to`, no payload.
    inline void link_increment(LinkHeaderAt header, uint64_t to, uint32_t add) {
#if defined(KM_WIRE)
        wire_set(header, kWireIncrement, to);
        header->add = add;
#else
        header->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(to, add, true)
        );
#endif
    }

    // This CPU's held link: open from its PAGES open to its kernel's end (a kernel's
    // .bss starts zero). One sender a link.
    class LinkSender {
    public:
#if defined(KM_WIRE)
        // The wire needs no router: nothing to take or give back.
        void connect(uint32_t channel, uint32_t area) {
            (void)channel;
            (void)area;
        }

        void open() {
        }

        void close_start() {
        }

        bool closed(uint32_t area) const {
            (void)area;
            return true;
        }

        bool space() const {
            return true;
        }

        // The packet: the payload of `bytes` from L1 `from` (0: none) as the header
        // says; a count or a word only once the writes before it landed (bounded).
        bool send(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
            if constexpr (KM_WIRE_DELAY != 0) {
                const uint64_t until = wall64() + KM_WIRE_DELAY;
                while (static_cast<int64_t>(wall64() - until) < 0) {
                    ;
                }
            }
            const uint32_t kind = header->kind;
            const uint64_t to = (
                static_cast<uint64_t>(header->to_high) << 32 | header->to_low
            );
            if (kind == kWireWrite || kind == kWireCounted) {
                noc_async_write(from, to, bytes);
            }
            if (kind == kWireWrite) {
                return true;
            }
            // a count or a word after the writes before it, as a router keeps them
            uint32_t polls = 0;
            while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
                if (++polls > kWireFlushPolls) {
                    return false;
                }
            }
            if (kind == kWireCounted) {
                const uint64_t counter = (
                    static_cast<uint64_t>(header->counter_high) << 32 |
                    header->counter_low
                );
                noc_semaphore_inc(counter, header->add);
            } else if (kind == kWireInline) {
                noc_inline_dw_write(to, header->word);
            } else if (kind == kWireIncrement) {
                noc_semaphore_inc(to, header->add);
            }
            return true;
        }
#else
        // The connection to the router of Ethernet channel `channel`: the router's
        // words from the table the fabric's start-up wrote, ours at `area`.
        void connect(uint32_t channel, uint32_t area) {
            auto* table = (
                reinterpret_cast<
                    tt_l1_ptr tt::tt_fabric::tensix_fabric_connections_l1_info_t*>(
                        FABRIC_CONNECTIONS_BASE
                    )
            );
            const auto* conn = &table->read_only[channel];
            auto* flow = &table->read_write[channel];
            l1_word(area + kTeardown) = 0;
            new (bytes_) tt::tt_fabric::WorkerToFabricEdmSender(
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
                reinterpret_cast<volatile uint32_t* const>(area + kTeardown),
                area + kCursor,
                static_cast<uint32_t>(conn->worker_free_slots_stream_id),
                StreamId{0xFFFFFFFFu},
                write_reg_cmd_buf,
                write_at_cmd_buf
            );
        }

        void open() {
            edm().open();
        }

        void close_start() {
            edm().close_start();
        }

        // The router has let this core go (its teardown word), once; then cleared.
        bool closed(uint32_t area) const {
            if (l1_word(area + kTeardown) != 1) {
                return false;
            }
            l1_word(area + kTeardown) = 0;
            return true;
        }

        bool space() {
            return edm().edm_has_space_for_packet<1>();
        }

        // The packet: the payload of `bytes` from L1 `from` (0: none), then the
        // header, both handed to the router (call when space() is true).
        bool send(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
            if (bytes != 0) {
                edm().send_payload_without_header_non_blocking_from_address(
                    from, bytes
                );
            }
            edm().send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(header), sizeof(LinkHeader)
            );
            return true;
        }
#endif

        // The connection's words in the CPU's L1 (its chunk): teardown, cursor.
        static constexpr uint32_t kTeardown = 0;
        static constexpr uint32_t kCursor = 16;

    private:
        static volatile tt_l1_ptr uint32_t& l1_word(uint32_t address) {
            return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
        }
#if !defined(KM_WIRE)
        tt::tt_fabric::WorkerToFabricEdmSender& edm() {
            return *reinterpret_cast<tt::tt_fabric::WorkerToFabricEdmSender*>(bytes_);
        }

        alignas(8) uint8_t bytes_[sizeof(tt::tt_fabric::WorkerToFabricEdmSender)];
#endif
    };

    inline LinkSender link_sender_held;

    inline LinkSender& link_sender() {
        return link_sender_held;
    }

    // This CPU's writes have left it, on its NoC and the one the fabric's sender uses.
    inline bool link_sent() {
#if defined(KM_WIRE)
        return ncrisc_noc_nonposted_writes_sent(noc_index);
#else
        return (
            ncrisc_noc_nonposted_writes_sent(noc_index) &&
            ncrisc_noc_nonposted_writes_sent(tt::tt_fabric::get_fabric_worker_noc())
        );
#endif
    }

    // ... and have been acknowledged.
    inline bool link_flushed() {
#if defined(KM_WIRE)
        return (
            ncrisc_noc_nonposted_writes_flushed(noc_index) &&
            ncrisc_noc_nonposted_atomics_flushed(noc_index)
        );
#else
        return (
            ncrisc_noc_nonposted_writes_flushed(noc_index) &&
            ncrisc_noc_nonposted_writes_flushed(tt::tt_fabric::get_fabric_worker_noc())
        );
#endif
    }

}  // namespace km
