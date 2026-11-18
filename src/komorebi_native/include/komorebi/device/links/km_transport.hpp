// A link's transport, where a link core's packets leave: a fabric connection to an
// Ethernet router (two cards), or with KM_WIRE (one card) NoC writes to the far tile.
#pragma once

#include <stdint.h>

#include <new>

#include "api/dataflow/dataflow_api.h"
#include "../core/km_core.hpp"

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

    // The NoC address of `address` on tile (x, y) of the far island (a wire: for the
    // NoC `noc` of the CPU that writes it).
    inline uint64_t link_noc(
        uint32_t x, uint32_t y, uint32_t address, uint8_t noc = noc_index
    ) {
#if defined(KM_WIRE)
        return get_noc_addr(x, y, address, noc);
#else
        (void)noc;
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

#if !defined(KM_WIRE)
    // Pages a packet carries to as many addresses (a scatter write), at most; a counted
    // packet at most kCountedPages (the fabric's fused scatter and increment).
    constexpr uint32_t kPacketPages = NOC_SCATTER_WRITE_MAX_CHUNKS;
    constexpr uint32_t kCountedPages = NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS;

    // The header's command: `count` pages of `page` bytes, one after another in the
    // payload, page i into to[i] (2 to kPacketPages pages).
    inline void link_scatter_write(
        LinkHeaderAt header, const uint64_t* to, uint32_t count, uint32_t page
    ) {
        const uint16_t sizes[kPacketPages - 1] = {
            static_cast<uint16_t>(page),
            static_cast<uint16_t>(page),
            static_cast<uint16_t>(page)
        };
        header->to_noc_unicast_scatter_write(
            tt::tt_fabric::NocUnicastScatterCommandHeader(
                to, sizes, static_cast<uint8_t>(count)
            ),
            count * page
        );
    }

    // Two pages of `page` bytes into to[0] and to[1], then `add` on the word at
    // `counter` once they have landed.
    inline void link_scatter_counted(
        LinkHeaderAt header,
        const uint64_t* to,
        uint64_t counter,
        uint16_t add,
        uint32_t page
    ) {
        header->to_noc_fused_unicast_scatter_write_atomic_inc(
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader(
                {to[0], to[1]}, counter, {static_cast<uint16_t>(page)}, add, true
            ),
            2 * page
        );
    }
#else
    // The wire writes a packet to one address: a page a packet where they differ.
    constexpr uint32_t kPacketPages = 1;
    constexpr uint32_t kCountedPages = 1;
#endif

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

    // Link kForwardLink's CPU makes a forwarded packet's last write (forward_turn);
    // the pusher builds its DRAM targets for that CPU's NoC (forward_noc).
    constexpr uint32_t kForwardLink = 0;
    constexpr uint32_t kForwardCpu = kForwardLink % 2;  // link l: CPU l % 2

    // The NoC of the CPU that writes a forwarded packet, seen from CPU `cpu`.
    inline uint8_t forward_noc(uint32_t cpu) {
        return cpu == kForwardCpu ? noc_index : 1 - noc_index;
    }

    // Forwarding over a line: a packet for an island past the one across goes by the
    // table's next link; link cores between send it on (KM_FORWARD_*). Set at OPEN.
    struct ForwardHeld {
        uint32_t area;     // the forward area (0: none)
        uint32_t control;  // this core's control block
        uint32_t me;       // its core of the card
        uint32_t row;      // its row of what crossed (0: not counted)
        uint32_t limit;    // the wait limit
        uint32_t route;    // 1 + the island the packets sent now go to; 0: across
        uint32_t field;    // the KM_CROSS_* field they count in on each link
        uint32_t turning;  // a turn runs: its waits do not turn again
        uint32_t cpu;      // its CPU among those that send on (KM_FORWARD_CPUS)
        uint32_t turns;    // 1: this CPU takes the rooms' packets on (kForwardLink's)
    };

    inline ForwardHeld forward_held;

    // Whether a packet sent now leaves by another link than this core's own across.
    template <int = 0>
    bool forward_needed();

    // Sends a packet sent now on toward its island; bounded. False: gave up.
    template <int = 0>
    bool forward_send(LinkHeaderAt header, uint32_t from, uint32_t bytes);

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

        void ready() {
        }

        bool send_across_stateful(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
            return send_across(header, from, bytes);
        }

        // The packet across: `bytes` from L1 `from` (0: none) as the header says; a
        // count or a word only once the writes before it landed (bounded).
        bool send_across(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
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
            // a count or a word after the writes before it, as a router keeps them;
            // once the links are open the wait is the core's (limit, stop flag)
            const auto flushed = [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            };
            if (forward_held.control != 0) {
                if (poll_stop(
                    l1(forward_held.control + KM_CONTROL_STOP),
                    flushed,
                    forward_held.limit,
                    0
                ) != kGaveNone) {
                    return false;
                }
            } else {
                uint32_t polls = 0;
                while (!flushed()) {
                    if (++polls > kWireFlushPolls) {
                        return false;
                    }
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

        // The packet across: the payload of `bytes` from L1 `from` (0: none), then
        // the header, both handed to the router (call when space() is true).
        bool send_across(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
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

        // NoC command buffers set once for send_stateful's packets: call before a run
        // of them and after any other write or atomic of this CPU.
        void ready() {
            edm().template setup_stateful_send_cmd_bufs<false>();
        }

        // As send_across, on the buffers ready() set: writes set only addresses (4352
        // bytes a packet over one link: 139 ns against 150).
        bool send_across_stateful(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
            edm().template send_current_slot_stateful_non_blocking<false>(
                from, bytes, reinterpret_cast<uint32_t>(header)
            );
            return true;
        }
#endif

        // The packet: across, or on toward the island a route names (forward_send).
        bool send(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
#if defined(KM_FORWARD)  // the host's, over more than two islands
            if (forward_held.route != 0 && forward_needed()) {
                return forward_send(header, from, bytes);
            }
#endif
            return send_across(header, from, bytes);
        }

        // As send, the packets across on the buffers ready() set; a forwarded one as
        // send does, the buffers then set again.
        bool send_stateful(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
#if defined(KM_FORWARD)
            if (forward_held.route != 0 && forward_needed()) {
                const bool sent = forward_send(header, from, bytes);
                ready();
                return sent;
            }
#endif
            return send_across_stateful(header, from, bytes);
        }

        // One packet among this CPU's other writes: ready(), then send_stateful; a
        // header alone by send (no stateful payload write of 0 bytes).
        bool send_one(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
            if (bytes == 0) {
                return send(header, from, bytes);
            }
            ready();
            return send_stateful(header, from, bytes);
        }

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

    // What a forward step did: sent, no room there yet (the packet waits), or not
    // sendable (dropped, counted failed).
    constexpr uint32_t kForwardSent = 0;
    constexpr uint32_t kForwardWait = 1;
    constexpr uint32_t kForwardBad = 2;

    // Packets sent now go to island `island`, counted in KM_CROSS_* field `field` on
    // each link they cross after this one; forward_route_off(): across again.
    inline void forward_route(uint32_t island, uint32_t field) {
        forward_held.route = island + 1;
        forward_held.field = field;
    }

    inline void forward_route_off() {
        forward_held.route = 0;
    }

    // This link core's index on its island (its link cores after the ranks).
    inline uint32_t forward_index(const Islands& view) {
        return forward_held.me - view.base() - view.ranks();
    }

    // The part link core `owner` keeps for its island's link core `writer`.
    constexpr uint32_t forward_part(uint32_t owner, uint32_t writer) {
        return 1 + (writer < owner ? writer : writer - 1);
    }

    // The room of writer core `part` (0: the far one), its CPU `cpu`, at hop level
    // `level`: its slots, landed and taken counts, and the writer's sent and credit.
    constexpr uint32_t forward_room(uint32_t part, uint32_t cpu, uint32_t level) {
        return (part * KM_FORWARD_CPUS + cpu) * KM_FORWARD_LEVELS + level;
    }

    // The NoC address of `address` on core `core` of this card.
    inline uint64_t forward_core(uint32_t core, uint32_t address) {
        const uint32_t xy = (
            core_entry(
                l1(forward_held.control + KM_CONTROL_CORES),
                core,
                forward_held.control,
                kRoles
            )
        );
        return get_noc_addr(xy & 0xFFFFu, xy >> 16, address);
    }

    // The NoC address of `address` on the far link core across (its tile: the island
    // area's KM_ISLAND_RELAY_TILE word).
    inline uint64_t forward_far(uint32_t address) {
        const uint32_t tile = island_area()[KM_ISLAND_RELAY_TILE];
        return link_noc(tile & 0xFFFFu, tile >> 16, address);
    }

    inline volatile tt_l1_ptr uint32_t* forward_state() {
        return l1(forward_held.area + KM_FORWARD_STATE);
    }

    // ready() until it holds, bounded by the wait limit and the stop flag.
    template <typename Ready>
    inline bool forward_wait(Ready ready) {
        return (
            poll_stop(
                l1(forward_held.control + KM_CONTROL_STOP), ready, forward_held.limit, 0
            ) == kGaveNone
        );
    }

    // The packet across once the sender has room, then until its writes left L1.
    inline bool forward_across(LinkHeaderAt header, uint32_t from, uint32_t bytes) {
        auto& sender = link_sender();
        if (!forward_wait([&] {
            return sender.space();
        })) {
            return false;
        }
        if (bytes != 0) {
            sender.ready();  // pages' send, across: forward_send's route is not taken
        }
        return (
            bytes != 0
                ? sender.send_across_stateful(header, from, bytes)
                : sender.send_across(header, from, bytes)
        ) &&
            forward_wait([] {
                return link_sent();
            });
    }

    // (Two islands: every link core's link goes to the other; more: a link core a
    // neighbour, link core i the table's next link i.)
    template <int>
    __attribute__((noinline)) bool forward_needed() {
        const Islands view(island_area());
        const uint32_t to = forward_held.route - 1;
        return (
            view.count() > 2 &&
            (view.link(to) != forward_index(view) || view.hops(to) > 1)
        );
    }

    // The packet at `head` one step on: to the sibling toward it, into the far part 0,
    // or (last link) by its own header; `counted`: one of this core's parts.
    template <int = 0>
    __attribute__((noinline)) uint32_t
    forward_on(const Islands& view, uint32_t head, uint32_t payload, bool counted) {
        ForwardHeld& held = forward_held;
        volatile tt_l1_ptr uint32_t* words = l1(head);
        const uint32_t to = words[0], bytes = words[1], field = words[2];
        const uint32_t k = forward_index(view);
        const uint32_t j = to < view.count() ? view.link(to) : KM_ISLAND_NO_LINK;
        if (j == KM_ISLAND_NO_LINK || to == view.here() || bytes > KM_FORWARD_PAYLOAD) {
            return kForwardBad;
        }
        const auto count = [&] {
            if (!counted || held.row == 0 || field >= KM_CROSS_BYTES) {
                return;
            }
            volatile tt_l1_ptr uint32_t* row = l1(held.row);
            row[field] += 1;
            const uint32_t low = row[KM_CROSS_BYTES];
            row[KM_CROSS_BYTES] = low + bytes;
            row[KM_CROSS_BYTES + 1] += low + bytes < low ? 1u : 0u;
        };
        if (j == k && view.hops(to) == 1) {  // the last link: the packet itself
            if (!forward_across(
                reinterpret_cast<LinkHeaderAt>(head + 64), payload, bytes
            )) {
                return kForwardBad;
            }
            count();
            return kForwardSent;
        }
        // the room by hop level: across, the links it will have crossed less one; to
        // a sibling, those it crossed less one (a level only grows along a path)
        const uint32_t crossed = words[3] + (j == k ? 1 : 0);
        const uint32_t level = crossed > 0 ? crossed - 1 : 0;
        if (level >= KM_FORWARD_LEVELS) {
            return kForwardBad;
        }
        const uint32_t d = (
            forward_room(j == k ? 0 : forward_part(k, j), held.cpu, level)
        );
        volatile tt_l1_ptr uint32_t* state = forward_state();
        const uint32_t sent = state[KM_FORWARD_ROOMS + d];
        if (sent - l1(held.area + KM_FORWARD_CREDIT + 16 * d)[0] >= KM_FORWARD_SLOTS) {
            return kForwardWait;
        }
        const uint32_t room = (
            forward_room(j == k ? 0 : forward_part(j, k), held.cpu, level)
        );
        const uint32_t at = (
            held.area + KM_FORWARD_SLOT +
            (room * KM_FORWARD_SLOTS + sent % KM_FORWARD_SLOTS) * KM_FORWARD_SLOT_BYTES
        );
        const uint32_t landed = held.area + KM_FORWARD_LANDED + 16 * room;
        words[3] = crossed;
        bool ok = false;
        if (j != k) {  // the island's link core toward it, over this card's NoC
            const uint32_t core = view.base() + view.ranks() + j;
            noc_async_write(head, forward_core(core, at), 128);
            if (bytes != 0) {
                noc_async_write(payload, forward_core(core, at + 128), bytes);
            }
            ok = forward_wait([] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            });
            if (ok) {
                noc_semaphore_inc(forward_core(core, landed), 1);
                ok = forward_wait([] {
                    return ncrisc_noc_nonposted_atomics_flushed(noc_index);
                });
            }
        } else if (payload == head + 128) {  // across, whole: one counted write
            const LinkHeaderAt made = (
                link_header(held.area + KM_FORWARD_HEADERS + 128 * held.cpu, 1)
            );
            link_counted_write(
                made, forward_far(at), forward_far(landed), 1, 128 + bytes
            );
            ok = forward_across(made, head, 128 + bytes);
        } else {  // across, the head then the payload counted
            LinkHeaderAt made = (
                link_header(held.area + KM_FORWARD_HEADERS + 128 * held.cpu, 1)
            );
            link_write(made, forward_far(at), 128);
            ok = forward_across(made, head, 128);
            made = link_header(held.area + KM_FORWARD_HEADERS + 128 * held.cpu, 1);
            if (bytes != 0) {
                link_counted_write(
                    made, forward_far(at + 128), forward_far(landed), 1, bytes
                );
            } else {
                link_increment(made, forward_far(landed), 1);
            }
            ok = ok && forward_across(made, payload, bytes);
        }
        if (!ok) {
            return kForwardBad;
        }
        state[KM_FORWARD_ROOMS + d] = sent + 1;
        if (j == k) {
            count();
        }
        return kForwardSent;
    }

    // Room `room`'s count taken, back to its writer: the far link core across (its
    // destination 0), or the island's link core that writes it; its CPU and level.
    template <int = 0>
    __attribute__((noinline)) void forward_credit(
        const Islands& view, uint32_t room, uint32_t taken
    ) {
        ForwardHeld& held = forward_held;
        const uint32_t part = room / (KM_FORWARD_CPUS * KM_FORWARD_LEVELS);
        const uint32_t cpu = room / KM_FORWARD_LEVELS % KM_FORWARD_CPUS;
        const uint32_t level = room % KM_FORWARD_LEVELS;
        if (part == 0) {
            const LinkHeaderAt made = (
                link_header(held.area + KM_FORWARD_HEADERS + 128 * held.cpu + 64, 1)
            );
            link_inline(
                made,
                forward_far(
                    held.area + KM_FORWARD_CREDIT + 16 * forward_room(0, cpu, level)
                ),
                taken
            );
            forward_across(made, 0, 0);
            return;
        }
        const uint32_t k = forward_index(view);
        const uint32_t writer = part - 1 < k ? part - 1 : part;
        noc_inline_dw_write(
            forward_core(
                view.base() + view.ranks() + writer,
                held.area + KM_FORWARD_CREDIT +
                    16 * forward_room(forward_part(writer, k), cpu, level)
            ),
            taken
        );
        forward_wait([] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        });
    }

    // A step: each part's next landed packet on when there is room (never a wait),
    // its slot given back; one that cannot go on is dropped, counted failed.
    template <int = 0>
    __attribute__((noinline)) void forward_turn() {
        ForwardHeld& held = forward_held;
        if (held.area == 0 || held.turning != 0 || held.turns == 0) {
            return;
        }
        held.turning = 1;
        const uint32_t route = held.route;
        held.route = 0;
        const Islands view(island_area());
        volatile tt_l1_ptr uint32_t* state = forward_state();
        for (uint32_t room = 0; room < KM_FORWARD_ROOMS; room++) {
            invalidate_l1_cache();
            const uint32_t taken = state[room];
            if (l1(held.area + KM_FORWARD_LANDED + 16 * room)[0] == taken) {
                continue;
            }
            const uint32_t slot = (
                held.area + KM_FORWARD_SLOT +
                (room * KM_FORWARD_SLOTS + taken % KM_FORWARD_SLOTS) *
                    KM_FORWARD_SLOT_BYTES
            );
            const uint32_t done = forward_on(view, slot, slot + 128, true);
            if (done == kForwardWait) {
                continue;
            }
            if (done == kForwardBad) {
                state[KM_FORWARD_FAILED] += 1;
            }
            state[room] = taken + 1;
            forward_credit(view, room, taken + 1);
        }
        held.route = route;
        held.turning = 0;
    }

    template <int>
    __attribute__((noinline)) bool forward_send(
        LinkHeaderAt header, uint32_t from, uint32_t bytes
    ) {
        ForwardHeld& held = forward_held;
        if (held.area == 0) {
            return false;
        }
        const Islands view(island_area());
        const uint32_t head = held.area + KM_FORWARD_STAGE + 128 * held.cpu;
        volatile tt_l1_ptr uint32_t* words = l1(head);
        words[0] = held.route - 1;
        words[1] = bytes;
        words[2] = held.field;
        words[3] = 0;  // links crossed
        const volatile tt_l1_ptr uint32_t*
            own = (reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(header));
        for (uint32_t w = 0; w < sizeof(LinkHeader) / 4; w++) {
            words[16 + w] = own[w];
        }
        uint32_t done = kForwardWait;
        const bool ready = forward_wait([&] {
            done = forward_on(view, head, from, false);
            if (done == kForwardWait) {
                forward_turn();  // what waits here may be what frees the room
            }
            return done != kForwardWait;
        });
        if (ready && done == kForwardBad) {
            forward_state()[KM_FORWARD_FAILED] += 1;
        }
        return ready && done == kForwardSent;
    }

}  // namespace km
