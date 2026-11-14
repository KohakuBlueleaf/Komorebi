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

    constexpr uint32_t kWhereRelay = 61;  // link core 0: an ask it could not serve

    // An ask's words (64 bytes, the asking core's line for its NoC): the first 48
    // bytes, then its valid word on a line of its own, written last.
    constexpr uint32_t kAskSource = 0;   // the asking core, local
    constexpr uint32_t kAskAddress = 1;  // its pages' L1 address
    constexpr uint32_t kAskBytes = 2;    // at most KM_RELAY_MAX
    constexpr uint32_t kAskRank = 3;     // the flat rank they go to
    constexpr uint32_t kAskTo = 4;       // the L1 address there
    constexpr uint32_t kAskPort = 5;     // its port, counted there (kRelayNoPort: none)
    constexpr uint32_t kAskSlot = 6;     // the asking kernel's NoC: its done word
    constexpr uint32_t kAskValid = 12;   // the token: sequence << 1 | 1
    constexpr uint32_t kRelayServed = 0;  // KM_RELAY_STATE words
    constexpr uint32_t kRelaySent = 1;
    constexpr uint32_t kRelayTaken = 2;
    constexpr uint32_t kRelayFailed = 3;
    constexpr uint32_t kRelayNext = 4;  // the ask line the next scan starts at
    constexpr uint32_t kRelayNoPort = 0xFFFFFFFFu;

    // The relay area of this island (0: none).
    inline uint32_t relay_area() {
        return island_area()[KM_ISLAND_RELAY];
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
        const uint32_t line = writer.args().me * 2 + noc_index;
        if (area == 0 || bytes == 0 || view.ranks() == 0 || line >= KM_RELAY_ASKS) {
            return kGaveBadItem;
        }
        const uint32_t token = sequence << 1 | 1;
        const uint32_t ask = area + KM_RELAY_ASK + line * 64;
        volatile tt_l1_ptr uint32_t* done = l1(area + KM_RELAY_DONE + noc_index * 16);
        volatile tt_l1_ptr uint32_t* words = l1(ask);
        // the link core toward the rank's island: the table's next link after the ranks
        const uint32_t link = (
            view.base() + view.ranks() + view.link(view.island_of(rank))
        );
        for (uint32_t off = 0; off < bytes; off += KM_RELAY_MAX) {
            const uint32_t n = bytes - off < KM_RELAY_MAX ? bytes - off : KM_RELAY_MAX;
            words[kAskSource] = writer.args().me;
            words[kAskAddress] = from + off;
            words[kAskBytes] = n;
            words[kAskRank] = rank;
            words[kAskTo] = to + off;
            words[kAskPort] = off + n == bytes ? port : kRelayNoPort;
            words[kAskSlot] = noc_index;
            words[kAskValid] = token;
            noc_async_write(ask, writer.core_noc(link, ask), 48);
            noc_async_write_barrier();
            noc_async_write(ask + 48, writer.core_noc(link, ask + 48), 16);
            noc_async_write_barrier();
            noc_semaphore_inc(writer.core_noc(link, area + KM_RELAY_ASKED), 1);
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

    // Link core 0's link 0 CPU, between commands: serves one ask of its island's
    // cores, then one landed from the far island. Every wait bounded.
    class Relay {
    public:
        template <typename Kernel>
        static void turn(const Kernel& kernel) {
            const uint32_t area = relay_area();
            if (area == 0 || pages_held.open == 0 || pages_held.link != 0) {
                return;  // the CPU that holds its pair's first link relays
            }
            Relay relay(area);
            invalidate_l1_cache();
            if (l1(area + KM_RELAY_ASKED)[0] != relay.state(kRelayServed)) {
                relay.send(kernel);
            }
            if (l1(area + KM_RELAY_LANDED)[0] != relay.state(kRelayTaken)) {
                relay.place(kernel);
            }
        }

    private:
        explicit Relay(uint32_t area) : area_(area) {
        }

        volatile tt_l1_ptr uint32_t& state(uint32_t word) const {
            return l1(area_ + KM_RELAY_STATE)[word];
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
            if (!sender.send(header, from, bytes)) {
                return false;
            }
            return wait(kernel, [] {
                return link_sent();
            });
        }

        // The next valid ask after the last: its pages chunk by chunk into the far
        // inbox, the ask, one count, the done word; it places what lands meanwhile.
        template <typename Kernel>
        void send(const Kernel& kernel) {
            // a core's two lines at 2 me: its core of the card (a split card's island
            // starts at its base)
            const Islands view(island_area());
            const uint32_t lines = (view.base() + view.ranks()) * 2;
            const uint32_t start = state(kRelayNext);
            uint32_t ask = 0;
            for (uint32_t n = 0; n < lines && ask == 0; n++) {
                const uint32_t line = (start + n) % lines;
                const uint32_t at = area_ + KM_RELAY_ASK + line * 64;
                ask = (l1(at)[kAskValid] & 1u) != 0 ? at : 0;
                state(kRelayNext) = line + 1;
            }
            if (ask == 0) {
                return;  // its valid word has not landed yet
            }
            volatile tt_l1_ptr uint32_t* a = l1(ask);
            const uint32_t bytes = a[kAskBytes], token = a[kAskValid];
            const uint32_t from = a[kAskAddress], source = a[kAskSource];
            bool ok = bytes <= KM_RELAY_MAX && wait(kernel, [&] {
                if (l1(area_ + KM_RELAY_LANDED)[0] != state(kRelayTaken)) {
                    place(kernel);
                }
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
            const uint32_t done = area_ + KM_RELAY_DONE + (a[kAskSlot] & 1u) * 16;
            noc_async_write(
                area_ + KM_RELAY_LINES + 16, kernel.core_noc(source, done), 4
            );
            noc_async_write_barrier();
        }

        // The landed ask: its pages into the rank's L1, its port bumped, the credit
        // (this side's taken count) back across.
        template <typename Kernel>
        void place(const Kernel& kernel) {
            const Islands view(island_area());
            volatile tt_l1_ptr uint32_t* a = l1(area_ + KM_RELAY_INBOX);
            const uint32_t core = a[kAskRank] - view.offset();  // its core of the card
            const uint32_t bytes = a[kAskBytes];
            const uint32_t port_index = a[kAskPort];
            const bool fits = (
                a[kAskRank] - view.first() < view.ranks() && bytes <= KM_RELAY_MAX &&
                (port_index < KM_PORTS || port_index == kRelayNoPort)
            );
            if (fits) {
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
            l1(area_ + KM_RELAY_LINES)[0] = state(kRelayTaken);
            if (!packet(
                kernel,
                header(3, far(KM_RELAY_CREDIT), 16),
                area_ + KM_RELAY_LINES,
                16
            )) {
                state(kRelayFailed) = state(kRelayFailed) + 1;
            }
        }

        uint32_t area_;
    };

}  // namespace km
#endif
#endif
