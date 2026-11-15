// The relay: a core's pages into L1 of a rank of another island and 1 on its port, by
// its link core toward that island to the far link core across, which puts them there.
#pragma once

#include <stdint.h>

#include "../abi/km_abi.h"
#include "km_core.hpp"
#include "km_island.hpp"

#if !defined(COMPILE_FOR_TRISC)
#if defined(KM_PAGE_LINKS)
#include "km_pages.hpp"
#endif

namespace km {

    // An ask's words (64 bytes, the asking core's line for its NoC): the first 48
    // bytes, then its valid word on a line of its own, written last.
    constexpr uint32_t kAskSource = 0;   // the asking core, local
    constexpr uint32_t kAskAddress = 1;  // its pages' L1 address
    constexpr uint32_t kAskBytes = 2;    // at most KM_RELAY_MAX
    constexpr uint32_t kAskRank = 3;     // the flat rank they go to
    constexpr uint32_t kAskTo = 4;       // the L1 address there
    constexpr uint32_t kAskPort = 5;     // its port, counted there (kRelayNoPort: none)
    constexpr uint32_t kAskSlot = 6;     // the asking CPU's own block: its done word
    constexpr uint32_t kAskCpu = 7;      // the link core's CPU that serves it (0, 1)
    constexpr uint32_t kAskValid = 12;   // the token: sequence << 1 | 1
    constexpr uint32_t kRelayServed = 0;  // KM_RELAY_STATE words
    constexpr uint32_t kRelaySent = 1;
    constexpr uint32_t kRelayTaken = 2;
    constexpr uint32_t kRelayFailed = 3;
    constexpr uint32_t kRelayNext = 4;    // the ask line the next scan starts at
    constexpr uint32_t kRelayOnward = 5;  // the token of the ask sent on (0: none)
    constexpr uint32_t kRelayNoPort = 0xFFFFFFFFu;
#if !defined(KM_WIRE)
    // Two cards: link core 0 writes an ask's bytes straight into the far rank's L1, a
    // packet a slot (stage: kRelaySlots of kRelayStageSlot bytes, header each).
    constexpr uint32_t kRelaySlots = 4;
    constexpr uint32_t kRelaySlotAt = KM_RELAY_STAGE + 512;
    constexpr uint32_t kRelayStageSlot = (
        (KM_RELAY_SECOND - kRelaySlotAt) / kRelaySlots / 64 * 64
    );
    static_assert(
        kRelaySlots * 64 <= 512 && kRelayStageSlot >= 4096,
        "the relay's slots fit its stage and inbox"
    );
    constexpr uint32_t kRelayTurnAsks = 16;  // asks served a turn at most
#endif

    // The relay area of this island (0: none).
    inline uint32_t relay_area() {
        return island_area()[KM_ISLAND_RELAY];
    }

    // An asking core's done word of its CPU block `block` (0 the reader's, 1 the
    // writer's), in its own words (the same address on every core of the island).
    inline uint32_t relay_done(uint32_t block) {
        return (
            island_area()[KM_ISLAND_OWN / 4] + block * KM_OWN_CPU_BYTES + KM_OWN_DONE
        );
    }

    // What CPU `cpu` of a link core adds to KM_RELAY_ASKED, _STATE and _LINES for its
    // own relay words (KM_RELAY_SECOND: the second CPU's).
    constexpr uint32_t relay_words(uint32_t cpu) {
        return cpu == 0 ? 0 : KM_RELAY_SECOND - KM_RELAY_ASKED;
    }

    // The rule: rank `source`'s asks take link source % KM_ISLAND_LINKS between two
    // cards (CPU k % 2 of link core k / 2, so a rank's stay in order), else the next.
    struct RelayLink {
        uint32_t core;
        uint32_t cpu;
    };

    inline RelayLink relay_link(const Islands& view, uint32_t source, uint32_t island) {
        const uint32_t first = view.base() + view.ranks() + view.link(island);
#if defined(KM_WIRE)
        static_cast<void>(source);
        return {first, 0};
#else
        const uint32_t links = island_area()[KM_ISLAND_LINKS];
        if (view.count() != 2 || links <= 1) {
            return {first, 0};
        }
        const uint32_t link = source % links;
        return {first + link / 2, link % 2};
#endif
    }

    // `bytes` at L1 `from` into L1 `to` of rank `rank` of another island, then 1 on
    // its port (kRelayNoPort: none), in asks of KM_RELAY_MAX at most; kGave*.
    template <typename Writer>
    uint32_t relay_put(
        const Writer& writer,
        uint32_t sequence,
        uint32_t from,
        uint32_t bytes,
        uint32_t rank,
        uint32_t to,
        uint32_t port
    ) {
        const Islands view(island_area());
        const uint32_t area = relay_area();
        const uint32_t mine = view.own_words();
        const uint32_t line = writer.args().me * 2 + noc_index;
        if (
            area == 0 || mine == 0 || bytes == 0 || view.ranks() == 0 ||
            line >= KM_RELAY_ASKS
        ) {
            return kGaveBadItem;
        }
        const uint32_t token = sequence << 1 | 1;
        // staged in its CPU's own block, written to its line at the link core
        const uint32_t block = writer.role() == kRoleWriter ? 1 : 0;
        const uint32_t own = mine + block * KM_OWN_CPU_BYTES + KM_OWN_LINE;
        const uint32_t ask = area + KM_RELAY_ASK + line * 64;
        volatile tt_l1_ptr uint32_t* done = l1(relay_done(block));
        volatile tt_l1_ptr uint32_t* words = l1(own);
        // the link core toward the rank's island and its CPU, by this rank (relay_link)
        const RelayLink toward = (
            relay_link(view, view.offset() + writer.args().me, view.island_of(rank))
        );
        const uint32_t link = toward.core;
#if defined(KM_WIRE)
        const uint32_t most = KM_RELAY_MAX;  // the far inbox's
#else
        const uint32_t most = bytes;  // streamed from this L1: one ask
#endif
        for (uint32_t off = 0; off < bytes; off += most) {
            const uint32_t n = bytes - off < most ? bytes - off : most;
            words[kAskSource] = writer.args().me;
            words[kAskAddress] = from + off;
            words[kAskBytes] = n;
            words[kAskRank] = rank;
            words[kAskTo] = to + off;
            words[kAskPort] = off + n == bytes ? port : kRelayNoPort;
            words[kAskSlot] = block;
            words[kAskCpu] = toward.cpu;
            words[kAskValid] = token;
            noc_async_write(own, writer.core_noc(link, ask), 48);
            noc_async_write_barrier();
            noc_async_write(own + 48, writer.core_noc(link, ask + 48), 16);
            noc_async_write_barrier();
            noc_semaphore_inc(
                writer.core_noc(link, area + relay_words(toward.cpu) + KM_RELAY_ASKED),
                1
            );
            noc_async_atomic_barrier();
            const uint32_t code = writer.poll_at(
                kWhereEmit,
                [&] {
                    invalidate_l1_cache();
                    return done[0] == token;
                },
                writer.args().wait_limit
            );
            done[0] = 0;
            if (code != kGaveNone) {
                return code;
            }
        }
        return kGaveNone;
    }

    template <typename Writer>
    uint32_t relay_put(
        const Writer& writer,
        const km_command_t& command,
        uint32_t from,
        uint32_t bytes,
        uint32_t rank,
        uint32_t to,
        uint32_t port
    ) {
        return relay_put(writer, command.sequence, from, bytes, rank, to, port);
    }

}  // namespace km

#if defined(KM_PAGE_LINKS)
namespace km {

    // A link CPU between commands: the asks relay_link gives its link (on the wire link
    // 0's alone: one ask, then one landed from the far island). Every wait bounded.
    class Relay {
    public:
        template <typename Kernel>
        static void turn(const Kernel& kernel) {
            const uint32_t area = relay_area();
            if (area == 0 || pages_held.open == 0) {
                return;
            }
#if defined(KM_WIRE)
            const uint32_t links = 1;
#else
            const uint32_t links = island_area()[KM_ISLAND_LINKS];
#endif
            if (pages_held.link != 0 && pages_held.link >= links) {
                return;  // no source rank's link
            }
            Relay relay(area, pages_held.link % 2);
            invalidate_l1_cache();
#if defined(KM_WIRE)
            if (l1(area + KM_RELAY_ASKED)[0] != relay.state(kRelayServed)) {
                relay.send(kernel);
            }
            if (l1(area + KM_RELAY_LANDED)[0] != relay.state(kRelayTaken)) {
                relay.place(kernel);
            } else if (l1(area + KM_RELAY_LINES)[0] != relay.state(kRelayTaken)) {
                relay.credit(kernel);  // owed since its packet failed
            }
#else
            uint32_t served = 0;
            while (
                served < kRelayTurnAsks &&
                l1(relay.words_ + KM_RELAY_ASKED)[0] != relay.state(kRelayServed) &&
                relay.stream(kernel)
            ) {
                served++;
                invalidate_l1_cache();
            }
#endif
        }

    private:
        Relay(uint32_t area, uint32_t cpu) :
            area_(area), words_(area + relay_words(cpu)), cpu_(cpu) {
        }

        volatile tt_l1_ptr uint32_t& state(uint32_t word) const {
            return l1(words_ + KM_RELAY_STATE)[word];
        }

        uint64_t far(uint32_t offset) const {
            const uint32_t tile = island_area()[KM_ISLAND_RELAY_TILE];
            return (
                link_noc(
                    tile & 0xFFFFu, tile >> 16, island_area()[KM_ISLAND_RELAY_FAR] + offset
                )
            );
        }

        template <typename Kernel, typename Ready>
        static bool wait(const Kernel& kernel, Ready ready) {
            return kernel.poll_stop_at(
                kWhereRelay,
                [&] {
                    invalidate_l1_cache();
                    return ready();
                },
                kernel.args().wait_limit
            ) == kGaveNone;
        }

        // `ready` at once, else waited for as wait() does.
        template <typename Kernel, typename Ready>
        static bool soon(const Kernel& kernel, Ready ready) {
            return ready() || wait(kernel, ready);
        }

        // Header `which`, routed one chip on: a write of `bytes` to `to`, or (bytes 0)
        // an increment of the word at `to`.
        LinkHeaderAt header(uint32_t which, uint64_t to, uint32_t bytes) const {
            const LinkHeaderAt made = (
                link_header(area_ + KM_RELAY_HEADERS + which * 64, 1)
            );
            if (bytes != 0) {
                link_write(made, to, bytes);
            } else {
                link_increment(made, to, 1);
            }
            return made;
        }

        // One packet over the held connection: space, payload, header, sent.
        template <typename Kernel>
        static bool packet(
            const Kernel& kernel, LinkHeaderAt header, uint32_t from, uint32_t bytes
        ) {
            auto& sender = link_sender();
            if (!wait(kernel, [&] {
                return sender.space();
            })) {
                return false;
            }
            if (!sender.send_one(header, from, bytes)) {
                return false;
            }
            return wait(kernel, [] {
                return link_sent();
            });
        }

        // The next valid ask line after the last served, round robin; 0: none landed.
        uint32_t next_ask() const {
            // a core's two lines at 2 me: its core of the card (a split card's island
            // starts at its base), the island's link cores' too (an ask sent on)
            const Islands view(island_area());
            const uint32_t cores = view.base() + view.ranks() + KM_FORWARD_PARTS;
            const uint32_t lines = (
                cores * 2 < KM_RELAY_ASKS ? cores * 2 : KM_RELAY_ASKS
            );
            const uint32_t start = state(kRelayNext);
            uint32_t ask = 0;
            for (uint32_t n = 0; n < lines && ask == 0; n++) {
                const uint32_t line = (start + n) % lines;
                const uint32_t at = area_ + KM_RELAY_ASK + line * 64;
                ask = (l1(at)[kAskValid] & 1u) != 0 && l1(at)[kAskCpu] == cpu_ ? at : 0;
                state(kRelayNext) = line + 1;
            }
            return ask;
        }

        // The asking core's done word: its ask's bytes have left its L1.
        template <typename Kernel>
        void done(const Kernel& kernel, volatile tt_l1_ptr uint32_t* a) {
            const uint32_t token = a[kAskValid], source = a[kAskSource];
            const uint32_t at = relay_done(a[kAskSlot] & 1u);
            a[kAskValid] = 0;
            volatile tt_l1_ptr uint32_t* line = l1(words_ + KM_RELAY_LINES + 16);
            line[0] = token;
            noc_async_write(
                words_ + KM_RELAY_LINES + 16, kernel.core_noc(source, at), 4
            );
            noc_async_write_barrier();
        }

#if !defined(KM_WIRE)
        // The next ask's bytes into the far rank's L1, a count once all landed (its
        // port, else a spare word), so its next ask lands after; false: none.
        template <typename Kernel>
        bool stream(const Kernel& kernel) {
            const uint32_t ask = next_ask();
            if (ask == 0) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* a = l1(ask);
            const Islands view(island_area());
            const uint32_t bytes = a[kAskBytes], from = a[kAskAddress];
            const uint32_t source = a[kAskSource], rank = a[kAskRank];
            const uint32_t to = a[kAskTo], port = a[kAskPort];
            const uint32_t island = view.island_of(rank);
            const uint32_t core = rank -
                view.entry_first(island);  // both cards' tables
            bool ok = (
                island != view.here() && core < view.entry_count(island) &&
                core < KM_CORES_MAX && (port < KM_PORTS || port == kRelayNoPort)
            );
            const uint32_t xy = l1(kernel.args().control + KM_CONTROL_CORES)[core];
            const uint64_t counter = (
                port == kRelayNoPort
                    ? far(KM_RELAY_CREDIT)
                    : link_noc(
                        xy & 0xFFFFu,
                        xy >> 16,
                        kernel.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE
                    )
            );
            auto& sender = link_sender();
            // the slots: this CPU's page chunk before its ring (free between its PAGES
            // commands), else the relay's stage
            const uint32_t packet = pages_packet();
            const uint32_t room = pages_room(pages_held.chunk, packet);
            const uint32_t fit = (
                room > kPagesSlots && packet != 0 ? (room - kPagesSlots) / packet : 0
            );
            const bool paged = fit >= kRelaySlots;
            ok = ok && (paged || cpu_ == 0);  // the stage is the first CPU's
            // a packet's bytes: the island's, or the stage's slot where it is less
            const uint32_t size = (
                paged || packet < kRelayStageSlot ? packet : kRelayStageSlot
            );
            ok = ok && size != 0;
            const uint32_t slots = (
                paged ? (fit < kPagesSlotsMax ? fit : kPagesSlotsMax) : kRelaySlots
            );
            const uint32_t slot_at = (
                paged ? pages_held.area + kPagesSlots : area_ + kRelaySlotAt
            );
            const uint32_t header_at = (
                paged ? pages_held.area + kPagesHeaders : area_ + KM_RELAY_STAGE
            );
            uint32_t sent_mark[kPagesSlotsMax] = {};
            const uint32_t chunks = size != 0 ? (bytes + size - 1) / size : 0;
            const uint32_t tail = bytes - (chunks - 1) * size;
            // Slot r's read carries transaction id 1 + r: reads run ahead in the slots.
            uint32_t ahead = 0, r = 0;
            const auto read = [&] {
                noc_async_read_set_trid(1 + r);
                noc_async_read(
                    kernel.core_noc(source, from + ahead * size),
                    slot_at + r * size,
                    ahead + 1 == chunks ? tail : size
                );
                ahead++;
                r = r + 1 == slots ? 0 : r + 1;
            };
            while (ok && ahead < chunks && ahead < slots) {
                read();
            }
            uint32_t s = 0, last = slots - 1;
            sender.ready();  // pages' send: nothing but reads until the last packet
            for (uint32_t k = 0; ok && k < chunks; k++) {
                const uint32_t n = k + 1 == chunks ? tail : size;
                const uint32_t slot = slot_at + s * size;
                if (k != 0 && ahead < chunks) {  // the last packet's slot, once it left
                    ok = soon(kernel, [&] {
                        return (
                            noc_reached(NIU_MST_NONPOSTED_WR_REQ_SENT, sent_mark[last])
                        );
                    });
                    if (ok) {
                        read();
                    }
                }
                ok = ok && soon(kernel, [&] {
                    return (
                        ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + s)
                    );
                });
                if (!ok) {
                    break;
                }
                // a slot's header made once an ask; then its command alone
                const LinkHeaderAt made = (
                    k < slots
                        ? link_header(header_at + s * kPagesHeaderStride, 1)
                        : reinterpret_cast<LinkHeaderAt>(
                            header_at + s * kPagesHeaderStride
                        )
                );
                const uint64_t at = (link_noc(xy & 0xFFFFu, xy >> 16, to + k * size));
                if (k + 1 == chunks) {
                    link_counted_write(made, at, counter, 1, n);
                } else {
                    link_write(made, at, n);
                }
                ok = soon(
                    kernel,
                    [&] {
                        return sender.space();
                    }
                ) &&
                    sender.send_stateful(made, slot, n);
                sent_mark[s] = noc_nonposted_writes_num_issued[noc_index];
                last = s;
                s = s + 1 == slots ? 0 : s + 1;
            }
            noc_async_read_set_trid(0);
            ok = ok && wait(kernel, [] {
                return ncrisc_noc_reads_flushed(noc_index);
            });
            ok = ok && wait(kernel, [] {
                return link_sent();
            });  // the slots and headers free for the next ask
            state(kRelayServed) = state(kRelayServed) + 1;
            if (!ok) {
                a[kAskValid] = 0;
                state(kRelayFailed) = state(kRelayFailed) + 1;
                return true;  // the asking core's wait gives its command up
            }
            state(kRelaySent) = state(kRelaySent) + 1;
            cross_count(view.link_area(), KM_CROSS_PARTS, 1, bytes);
            done(kernel, a);
            return true;
        }
#endif

        // The next valid ask after the last: its pages chunk by chunk into the far
        // inbox, the ask, one count, the done word; it places what lands meanwhile.
        template <typename Kernel>
        void send(const Kernel& kernel) {
            const uint32_t ask = next_ask();
            if (ask == 0) {
                return;  // its valid word has not landed yet
            }
            volatile tt_l1_ptr uint32_t* a = l1(ask);
            const uint32_t bytes = a[kAskBytes], token = a[kAskValid];
            const uint32_t from = a[kAskAddress], source = a[kAskSource];
            bool ok =
                bytes <= KM_RELAY_MAX && wait(kernel, [&] {
                    if (l1(area_ + KM_RELAY_LANDED)[0] != state(kRelayTaken)) {
                        place(kernel);
                    } else if (l1(area_ + KM_RELAY_LINES)[0] != state(kRelayTaken)) {
                        credit(kernel);  // owed since its packet failed
                    }
#if defined(KM_FORWARD)
                    forward_turn();
#endif
                    return l1(area_ + KM_RELAY_CREDIT)[0] == state(kRelaySent);
                });
            for (uint32_t off = 0; ok && off < bytes; off += KM_RELAY_CHUNK) {
                const uint32_t n = (
                    bytes - off < KM_RELAY_CHUNK ? bytes - off : KM_RELAY_CHUNK
                );
                noc_async_read(
                    kernel.core_noc(source, from + off), area_ + KM_RELAY_STAGE, n
                );
                noc_async_read_barrier();
                ok = (
                    packet(
                        kernel,
                        header(0, far(KM_RELAY_INBOX + 64 + off), n),
                        area_ + KM_RELAY_STAGE,
                        n
                    )
                );
            }
            for (uint32_t w = 0; w < 16; w++) {
                l1(area_ + KM_RELAY_DESC)[w] = a[w];
            }
            ok = (
                ok &&
                packet(
                    kernel,
                    header(1, far(KM_RELAY_INBOX), 64),
                    area_ + KM_RELAY_DESC,
                    64
                )
            );
            ok = ok && packet(kernel, header(2, far(KM_RELAY_LANDED), 0), 0, 0);
            a[kAskValid] = 0;
            state(kRelayServed) = state(kRelayServed) + 1;
            if (!ok) {
                state(kRelayFailed) = state(kRelayFailed) + 1;
                return;  // the leader's wait gives its command up
            }
            state(kRelaySent) = state(kRelaySent) + 1;
            cross_count(
                Islands(island_area()).link_area(), KM_CROSS_PARTS, 1, bytes + 64
            );
            volatile tt_l1_ptr uint32_t* line = l1(area_ + KM_RELAY_LINES + 16);
            line[0] = token;
            const uint32_t done = relay_done(a[kAskSlot] & 1u);
            noc_async_write(
                area_ + KM_RELAY_LINES + 16, kernel.core_noc(source, done), 4
            );
            noc_async_write_barrier();
        }

        // The landed ask for a rank further on, asked of the link core toward it (its
        // relay sends it on, then its done word here). False: none to ask.
        template <typename Kernel>
        bool onward(const Kernel& kernel, const Islands& view, uint32_t island) {
            const uint32_t me = kernel.args().me;
            const uint32_t toward = view.link(island);
            const uint32_t line = me * 2;
            if (
                view.count() <= 2 || toward == KM_ISLAND_NO_LINK ||
                toward == me - view.base() - view.ranks() || line >= KM_RELAY_ASKS
            ) {
                return false;
            }
            const uint32_t core = view.base() + view.ranks() + toward;
            const volatile tt_l1_ptr uint32_t* a = l1(area_ + KM_RELAY_INBOX);
            volatile tt_l1_ptr uint32_t* d = l1(area_ + KM_RELAY_DESC);
            const uint32_t token = (state(kRelayTaken) + 1) << 1 | 1;
            for (uint32_t w = 0; w < 16; w++) {
                d[w] = 0;
            }
            d[kAskSource] = me;
            d[kAskAddress] = area_ + KM_RELAY_INBOX + 64;
            d[kAskBytes] = a[kAskBytes];
            d[kAskRank] = a[kAskRank];
            d[kAskTo] = a[kAskTo];
            d[kAskPort] = a[kAskPort];
            d[kAskSlot] = 0;
            d[kAskValid] = token;
            l1(relay_done(0))[0] = 0;
            const uint32_t ask = area_ + KM_RELAY_ASK + line * 64;
            noc_async_write(area_ + KM_RELAY_DESC, kernel.core_noc(core, ask), 48);
            noc_async_write_barrier();
            noc_async_write(
                area_ + KM_RELAY_DESC + 48, kernel.core_noc(core, ask + 48), 16
            );
            noc_async_write_barrier();
            noc_semaphore_inc(kernel.core_noc(core, area_ + KM_RELAY_ASKED), 1);
            noc_async_atomic_barrier();
            state(kRelayOnward) = token;
            return true;
        }

        // The landed ask: its pages into the rank's L1, its port bumped, the credit
        // back; a rank further on: sent on, the credit back once that is done.
        template <typename Kernel>
        void place(const Kernel& kernel) {
            const Islands view(island_area());
            volatile tt_l1_ptr uint32_t* a = l1(area_ + KM_RELAY_INBOX);
            const uint32_t core = a[kAskRank] - view.offset();  // its core of the card
            const uint32_t bytes = a[kAskBytes];
            const uint32_t port_index = a[kAskPort];
            const bool here = a[kAskRank] - view.first() < view.ranks();
            const bool fits = (
                here && bytes <= KM_RELAY_MAX &&
                (port_index < KM_PORTS || port_index == kRelayNoPort)
            );
            bool sent_on = false;  // a rank further on: sent on and taken there
#if defined(KM_FORWARD)
            if (state(kRelayOnward) != 0) {
                if (l1(relay_done(0))[0] != state(kRelayOnward)) {
                    return;  // not taken there yet
                }
                state(kRelayOnward) = 0;
                sent_on = true;
            } else if (
                !here && bytes <= KM_RELAY_MAX &&
                onward(kernel, view, view.island_of(a[kAskRank]))
            ) {
                return;
            }
#endif
            if (sent_on) {
                // its credit back below, nothing placed here
            } else if (fits) {
                noc_async_write(
                    area_ + KM_RELAY_INBOX + 64, kernel.core_noc(core, a[kAskTo]), bytes
                );
                noc_async_write_barrier();
                if (port_index != kRelayNoPort) {
                    const uint32_t port = (
                        kernel.args().control + KM_CONTROL_PORT +
                        port_index * KM_PORT_STRIDE
                    );
                    noc_semaphore_inc(kernel.core_noc(core, port), 1);
                    noc_async_atomic_barrier();
                }
            } else {
                state(kRelayFailed) = state(kRelayFailed) + 1;
            }
            state(kRelayTaken) = state(kRelayTaken) + 1;
            credit(kernel);
        }

        // The credit (Taken) back to the far side. A failed packet leaves it owed
        // (KM_RELAY_LINES not Taken): the next turn sends it again.
        template <typename Kernel>
        void credit(const Kernel& kernel) {
            l1(area_ + KM_RELAY_LINES)[0] = state(kRelayTaken);
            if (!packet(
                kernel,
                header(3, far(KM_RELAY_CREDIT), 16),
                area_ + KM_RELAY_LINES,
                16
            )) {
                state(kRelayFailed) = state(kRelayFailed) + 1;
                l1(area_ + KM_RELAY_LINES)[0] = state(kRelayTaken) - 1;
            }
        }

        uint32_t area_;
        uint32_t words_;  // area_, or the second CPU's words less KM_RELAY_ASKED
        uint32_t cpu_;
    };

}  // namespace km
#endif
#endif
