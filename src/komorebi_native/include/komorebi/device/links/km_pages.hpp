// Pages across a link (KM_HANDLER_PAGES): each data-movement CPU of a link core holds
// its link, sends its share of pages, a count a batch, or waits for a count
#pragma once

#include <stdint.h>

#include <new>

#include "api/dataflow/dataflow_api.h"
#include "../core/km_command.hpp"
#include "../core/km_core.hpp"
#include "km_transport.hpp"

namespace km {

    // A CPU's L1 (its chunk): the connection's two words (LinkSender), a packet header
    // a slot, then the slots the pages pass through.
    constexpr uint32_t kPagesHeaders = 64;
    constexpr uint32_t kPagesHeaderStride = 64;
    constexpr uint32_t kPagesSlotsMax = 60;
    constexpr uint32_t kPagesSlotsDefault = 15;
    constexpr uint32_t kPagesReadIds = NOC_MAX_TRANSACTION_ID;  // ids 1 .. 15
    constexpr uint32_t kPagesSlots = (
        kPagesHeaders + kPagesSlotsMax * kPagesHeaderStride
    );
    constexpr uint32_t kPagesBatch = 64;  // packets a count
    // A line of this many pages or more is a bulk one: on link 0 too where it is a
    // page CPU.
    constexpr uint32_t kPagesBulk = 64;
    // From this many pages a share reads several a transaction (a block of banks); a
    // smaller share a page a read (four a read measured a fifth slower).
    constexpr uint32_t kPagesBulkShare = 32;
    constexpr uint32_t kPagesHelloEvery = 4096;  // hello polls between two hellos

    // A page CPU's put ring at its chunk's top (kRingChunkMin bytes or more): header
    // slots, then payload slots, reserved by the putting core by atomic.
    constexpr uint32_t kRingChunkMin = 128 * 1024;
    constexpr uint32_t kRingSlotsMax = 32;
    constexpr uint32_t kRingSlotsMin = 8;
    constexpr uint32_t kRingHeader = 128;
    constexpr uint32_t kRingPiece = 16;  // pages a putting core reserves at once
    constexpr uint32_t kRingMark = 16;
    constexpr uint32_t kRingBytes = 17;
    constexpr uint32_t kRingNoAnswer = 0xFFFFFFFFu;  // the tail's atomic not yet in
    constexpr uint32_t kRingBursts = 4;  // bursts a turn of the ring at most
    // a core waiting for ring room rereads the ring after the slots ahead could have
    // sent (a packet's cycles each: 8 KiB at 40 GB/s), kRingPollMost at most
    constexpr uint32_t kRingPollCycles = 270;
    constexpr uint32_t kRingPollMost = 64;
    static_assert(
        sizeof(LinkHeader) <= kRingMark * 4 &&
            kRingPiece * kRingHeader <= KM_PUT_BUILD_BYTES,
        "a ring header slot: the header, then its mark; payload slots whole lines"
    );

    // The payload bytes of a packet on this island's links (the session's; 0: none).
    inline uint32_t pages_packet() {
        return Islands(island_area()).packet();
    }

    // The put queue's L1 address on a link core (its own L1, KM_LINK_CB; the same on
    // every link core and island), 0: none.
#if defined(KM_LINK_CHUNK) && !defined(KM_LINK_QUEUE)
#error "a link core's own chunks follow its put queue: KM_LINK_QUEUE"
#endif
    inline uint32_t link_queue() {
#if defined(KM_LINK_QUEUE_CB)  // the chip's index where it is not KM_LINK_CB (Wormhole)
        return get_write_ptr(KM_LINK_QUEUE_CB);
#elif defined(KM_LINK_QUEUE)
        return get_write_ptr(KM_LINK_CB);
#else
        return 0;
#endif
    }

    // The places for puts between islands past the next, after the put queue (bytes).
#if defined(KM_FAR_PUTS)
    constexpr uint32_t kFarBytes = KM_FAR_BYTES(KM_ISLANDS_MAX);
#else
    constexpr uint32_t kFarBytes = 0;
#endif
#if defined(KM_DELIVER)  // then the deliveries' places (km_deliver.hpp)
    constexpr uint32_t kDeliverPlaceBytes = (
        KM_DELIVER_BYTES(KM_DELIVER_ISLANDS, KM_DELIVER_SLOTS, KM_DELIVER_SLOT_BYTES)
#if defined(KM_X280_FAR)  // and island 0's X280 ranks' rings
        + KM_X280_RINGS_BYTES
#endif
    );
    template <typename Kernel>
    uint32_t deliver_put(const Kernel&, uint32_t, uint32_t, uint32_t, uint32_t);
    template <typename Kernel>
    uint32_t deliver_confirm(const Kernel&, uint32_t);
#if defined(KM_X280_FAR)
    template <typename Kernel>
    uint32_t deliver_x280(const Kernel&, uint32_t, uint32_t);
#endif
    inline void deliver_zero(uint32_t area);
    template <typename Kernel>
    void deliver_gave(const Kernel&, uint32_t);
#else
    constexpr uint32_t kDeliverPlaceBytes = 0;
#endif

    // The forward rooms (KM_FORWARD), after the far places: a link core's alone (a
    // rank's build beside the link cores' own kernels has no services: none).
#if defined(KM_FORWARD) && defined(KM_LINK_SERVICES)
#if !defined(KM_LINK_QUEUE)
#error "the forward rooms follow the put queue: KM_LINK_QUEUE"
#endif
    constexpr uint32_t kForwardBytes = KM_FORWARD_BYTES;
#else
    constexpr uint32_t kForwardBytes = 0;
#endif

    // The slots of the ring of a chunk of `chunk` bytes, slots of `packet` bytes (0:
    // none).
    constexpr uint32_t ring_slots(uint32_t chunk, uint32_t packet) {
        if (chunk < kRingChunkMin || packet == 0) {
            return 0;
        }
        const uint32_t slots = chunk / 2 / (kRingHeader + packet);
        return (
            slots < kRingSlotsMin
                ? 0
                : slots < kRingSlotsMax
                    ? slots
                    : kRingSlotsMax
        );
    }

    // A chunk's bytes before its ring: the connection's words, then the headers and
    // slots of the CPU's own sends.
    constexpr uint32_t pages_room(uint32_t chunk, uint32_t packet) {
        return chunk - ring_slots(chunk, packet) * (kRingHeader + packet);
    }

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
        uint32_t from_core;  // KM_PAGES_DRAM: from DRAM; else x | y << 16, from its L1
        uint32_t shape;      // KM_PAGES_SHAPE_*: 0, the loop's own
        uint32_t to_first;   // KM_PAGES_DRAM: the target's page for the share's first
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
            a[KM_PAGES_BUFFER] & ((1u << KM_PAGES_SHAPE_SHIFT) - 1u),
            KM_PAGES_DRAM,
            a[KM_PAGES_BUFFER] >> KM_PAGES_SHAPE_SHIFT,
            a[KM_PAGES_FIRST]
        };
    }

    // This CPU's connection, held from its PAGES open to its kernel's end: L1 at `area`
    // (`chunk` bytes), `hops` to the far island, its link of `links`.
    struct PagesHeld {
        uint32_t open;
        uint32_t area;
        uint32_t chunk;
        uint32_t hops;
        uint32_t link;
        uint32_t links;
    };

    inline PagesHeld pages_held;

    // This CPU's ring as sent: at L1 `at` (`slots` headers, then payloads), the slots
    // handed to the link, the NoC's issued-write count after the last.
    struct PagesRing {
        uint32_t at;
        uint32_t slots;
        uint32_t issued;
        uint32_t mark;  // the writes issued after the last burst
        // the oldest burst not yet counted sent: the slots issued by it, its mark
        uint32_t held;
        uint32_t held_mark;
        bool holding;
        uint32_t left;  // the wall clock at its last turn's end
    };

    inline PagesRing pages_ring_held;

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

#if defined(KM_LINK_THIN_SECOND)
    // A link core's second link (KM_LINK_THIN_SECOND, Wormhole), toward the same
    // neighbour: the writer's, which sends what the reader's push hands it.
    static_assert(
        kLinksACore == 1, "a second link: the writer holds no link of its own"
    );
#if defined(KM_WIRE)
#error "a second link is the fabric's (cards), not a split card's wire"
#endif
    // Its L1, the chunk after the reader's: the sender's words, its hello's header,
    // the reader's word (handed), the writer's (their writes left), the records.
    constexpr uint32_t kSideHeader = 64;
    constexpr uint32_t kSidePut = 128;
    constexpr uint32_t kSideDone = 192;
    constexpr uint32_t kSideRecords = 256;  // kSideSlots of 16 bytes: header, from, bytes
    constexpr uint32_t kSideSlots = 16;

    // Both CPUs': the second link there (the reader: to hand to; the writer: held), its
    // L1, this CPU's count (the reader's handed, the writer's left).
    struct SideHeld {
        uint32_t open;
        uint32_t area;
        uint32_t count;
        uint32_t batches;  // the reader's counted batches (the odd ones on the second)
        uint32_t row;      // the writer's row of what crossed (the link area's second)
    };

    inline SideHeld side_held;
    inline uint32_t side_taken;              // the writer: records sent
    inline uint32_t side_marks[kSideSlots];  // the writer: its issued writes after each

    inline volatile tt_l1_ptr uint32_t& side_word(uint32_t offset) {
        return l1(side_held.area + offset)[0];
    }

    // The second link's L1: the chunk after the reader's (pages_open's buffer, chunk).
    inline uint32_t side_area_of(const PagesRequest& r) {
#if defined(KM_LINK_CHUNK)
        return (
            link_queue() + KM_QUEUE_BYTES + kFarBytes + kForwardBytes +
            kDeliverPlaceBytes + KM_LINK_CHUNK
        );
#else
        return r.buffer + r.chunk;
#endif
    }

    // The second link's channel for link core `rank` of its pair (OPEN's word).
    inline uint32_t side_channel(const PagesRequest& r, uint32_t rank) {
        return rank < 4 ? (r.to >> 8 * rank) & 0xFFu : KM_PAGES_NO_LINK;
    }

    // The reader at its OPEN: whether its core has a second link, both words zero (the
    // writer's too: it is advanced only after a record this reader hands).
    inline void side_reader_open(const PagesRequest& r, uint32_t rank) {
        side_held = {side_channel(r, rank) != KM_PAGES_NO_LINK ? 1u : 0u, 0, 0, 0, 0};
        if (side_held.open != 0) {
            side_held.area = side_area_of(r);
            side_word(kSidePut) = 0;
            side_word(kSideDone) = 0;
        }
    }

    // The reader: the packet at `header` (its payload `bytes` at `from`) to the writer
    // once a record is free; `wait` bounds the wait. False: given up.
    template <typename Wait>
    inline bool side_hand(
        LinkHeaderAt header, uint32_t from, uint32_t bytes, Wait wait
    ) {
        const uint32_t n = side_held.count;
        if (!wait([n] {
            return n - side_word(kSideDone) < kSideSlots;
        })) {
            return false;
        }
        volatile tt_l1_ptr uint32_t* record = (
            l1(side_held.area + kSideRecords + n % kSideSlots * 16)
        );
        record[0] = reinterpret_cast<uint32_t>(header);
        record[1] = from;
        record[2] = bytes;
        asm volatile("" ::: "memory");
        side_held.count = n + 1;
        side_word(kSidePut) = n + 1;  // the record is the writer's from here
        return true;
    }

    // The reader: whether the writes of every record up to `n` have left (counts wrap).
    inline bool side_left(uint32_t n) {
        return static_cast<int32_t>(side_word(kSideDone) - n) >= 0;
    }

    // The writer, a step: the records whose writes left given back, then the handed
    // ones sent while the router has room.
    inline void side_step() {
        if (side_held.open == 0) {
            return;
        }
        invalidate_l1_cache();
        uint32_t done = side_held.count;
        while (
            done != side_taken &&
            noc_reached(NIU_MST_NONPOSTED_WR_REQ_SENT, side_marks[done % kSideSlots])
        ) {
            done++;
        }
        if (done != side_held.count) {
            side_held.count = done;
            side_word(kSideDone) = done;
        }
        const uint32_t put = side_word(kSidePut);
        auto& sender = link_sender();
        while (side_taken != put && sender.space()) {
            const volatile tt_l1_ptr uint32_t* record = (
                l1(side_held.area + kSideRecords + side_taken % kSideSlots * 16)
            );
            const LinkHeaderAt header = reinterpret_cast<LinkHeaderAt>(record[0]);
            const uint32_t bytes = record[2];
            sender.ready();  // across, never forwarded: the far island is a neighbour
            sender.send_across_stateful(header, record[1], bytes);
            const uint32_t issued = noc_nonposted_writes_num_issued[noc_index];
            side_marks[side_taken % kSideSlots] = issued;
            side_taken++;
            if (side_held.row != 0) {  // packets as its pages, and their bytes
                volatile tt_l1_ptr uint32_t* row = l1(side_held.row);
                row[KM_CROSS_PAGES] += 1;
                const uint32_t low = row[KM_CROSS_BYTES];
                row[KM_CROSS_BYTES] = low + bytes;
                row[KM_CROSS_BYTES + 1] += low + bytes < low ? 1u : 0u;
            }
        }
    }

    // The writer at the kernel's end: its connection closed (bounded waits).
    template <typename Kernel>
    void side_end(const Kernel& kernel) {
        if (side_held.open == 0) {
            return;
        }
        side_held.open = 0;
        const auto wait = [&](auto ready) {
            return kernel.poll_at(
                kWherePages,
                [&] {
                    invalidate_l1_cache();
                    return ready();
                },
                kernel.args().wait_limit
            ) == kGaveNone;
        };
        wait([] {
            return ncrisc_noc_nonposted_writes_sent(noc_index);
        });
        link_sender().close_start();
        const bool closed = wait([] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        });
        if (closed) {
            const uint32_t area = side_held.area;
            wait([area] {
                return link_sender().closed(area);
            });
        }
    }

    // The writer's OPEN (link core `rank` of its pair): its second link's connection
    // held, hello with the far writer (port KM_PAGES_HELLO_PORT + 1). kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t
    side_open(const Kernel& kernel, const PagesRequest& r, uint32_t rank) {
        const uint32_t channel = side_channel(r, rank);
        if (channel == KM_PAGES_NO_LINK) {
            return kGaveNone;
        }
        const uint32_t area = side_area_of(r);
        if (
            side_held.open != 0 || area % 16 != 0 || r.hops == 0 ||
            static_cast<uint64_t>(area) + kSideRecords + 16 * kSideSlots > MEM_L1_SIZE
        ) {
            return kGaveBadTransfer;
        }
        auto& sender = link_sender();
        sender.connect(channel, area);
        sender.open();
        const uint32_t links_area = islands_at(kernel.args().control).link_area();
        side_held = {1, area, 0, 0, links_area != 0 ? links_area + KM_LINK_CROSS : 0};
        if (side_held.row != 0) {
            side_held.row += KM_CROSS_ROW;  // the writer's own row, zero from its open
            for (uint32_t w = 0; w < KM_CROSS_ROW / 4; w++) {
                l1(side_held.row)[w] = 0;
            }
        }
        side_taken = 0;
        const LinkHeaderAt header = link_header(area + kSideHeader, r.hops);
        const uint64_t far = (
            link_noc(r.counter & 0xFFu, r.counter >> 8, r.count + KM_PORT_STRIDE)
        );
        volatile tt_l1_ptr uint32_t* here = (
            l1(
                kernel.args().control + KM_CONTROL_PORT +
                (KM_PAGES_HELLO_PORT + 1) * KM_PORT_STRIDE
            )
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
                    ncrisc_noc_nonposted_writes_sent(noc_index) && sender.space()
                ) {
                    link_inline(header, far, say);
                    sender.send_across(header, 0, 0);
                    said = say;
                }
                return heard == 2 && said == 2;
            },
            kernel.args().wait_limit
        );
        if (code != kGaveNone) {  // no hello: the connection closed
            side_end(kernel);
            return code;
        }
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            kernel.args().wait_limit
        );
    }

    // DONE's word the reader sets when its body returned (km_worker.hpp's kDoneReader).
    constexpr uint32_t kSideReaderDone = 2;

    // The writer's part of a PAGES command: OPEN its second link; SEND and RANGE the
    // records handed until the reader's body returned and all left. False: given up.
    template <typename Kernel>
    __attribute__((noinline)) bool side_serve(
        const Kernel& kernel, const km_command_t& command, uint32_t word
    ) {
        const PagesRequest r = pages_of(command);
        const bool links = KM_GROUP_IS_LINK(first(command));
        uint32_t code = kGaveNone;
        if (r.mode == KM_PAGES_OPEN && links) {
            // its index in its pair's link cores (an OPEN's first: the pair's first)
            const uint32_t index = kernel.name(command) - KM_GROUP_LINK_BASE;
            code = side_open(kernel, r, index - r.first);
        } else if (
            (r.mode == KM_PAGES_SEND || r.mode == KM_PAGES_RANGE) && links &&
            side_held.open != 0
        ) {
            volatile tt_l1_ptr uint32_t* mark = (
                kernel.control(KM_CONTROL_DONE + slot(command) * 16)
            );
            code = kernel.poll_at(
                kWherePages,
                [&] {
                    side_step();
                    return (
                        mark[kSideReaderDone] == command.sequence &&
                        side_taken == side_word(kSidePut) &&
                        side_held.count == side_taken
                    );
                },
                kernel.args().wait_limit
            );
        }
        if (code != kGaveNone) {
            kernel.gave_up(word, code, kWherePages, slot(command), command.sequence);
            return false;
        }
        return true;
    }
#endif

    // The words a line's last word checks: every word between its number and its end,
    // so a copy read as the line lands fails when any word in it is the old line's.
    constexpr uint32_t kLineFirst = KM_ASK_NUMBER + 1, kLineLast = KM_ASK_END;

    // A line's last word (KM_ASK_END): its number folded with its other words, each
    // rotated; one word changed always changes it. A torn copy fails it, is read again.
    template <typename Words>
    inline uint32_t line_end(const Words& line, uint32_t number) {
        uint32_t words[kLineLast];
#pragma GCC unroll 14
        for (uint32_t w = kLineFirst; w < kLineLast; w++) {
            words[w] = line[w];
        }
        uint32_t check = number;
#pragma GCC unroll 14
        for (uint32_t w = kLineFirst; w < kLineLast; w++) {
            check = (check << 5 | check >> 27) ^ words[w];
        }
        return check;
    }

    // Numbers the line `number` and gives it its last word; written after every word.
    inline void line_seal(volatile tt_l1_ptr uint32_t* line, uint32_t number) {
        line[KM_ASK_NUMBER] = number;
        line[KM_ASK_END] = line_end(line, number);
    }

    // Line `from` into `to` numbered `number`, its last word folded from the words as
    // they pass (no second read of L1).
    template <typename Words>
    inline void line_copy(
        volatile tt_l1_ptr uint32_t* to, const Words& from, uint32_t number
    ) {
        uint32_t words[16];
        for (uint32_t w = 0; w < 16; w++) {
            words[w] = from[w];
        }
        words[KM_ASK_NUMBER] = number;
        words[KM_ASK_END] = line_end(words, number);
        for (uint32_t w = 0; w < 16; w++) {
            to[w] = words[w];
        }
    }

    // Whether the line holds number `number` whole (its last word checks).
    inline bool line_whole(const volatile tt_l1_ptr uint32_t* line, uint32_t number) {
        return (
            line[KM_ASK_NUMBER] == number && line[KM_ASK_END] == line_end(line, number)
        );
    }

    // Line `number` at `line` copied once into `copy` and checked there (its number,
    // then its last word): true when whole. The reader then works from the copy alone.
    inline bool line_take(
        const volatile tt_l1_ptr uint32_t* line, uint32_t number, uint32_t (&copy)[16]
    ) {
        if (line[KM_ASK_NUMBER] != number) {
            return false;  // not this line yet: one word read
        }
#pragma GCC unroll 16
        for (uint32_t w = 0; w < 16; w++) {
            copy[w] = line[w];
        }
        return (
            copy[KM_ASK_NUMBER] == number && copy[KM_ASK_END] == line_end(copy, number)
        );
    }

    // What this CPU sent across, counted in its row of the link area at `area` (0: not
    // counted): `n` more of word `field`, `bytes` more bytes.
    inline void cross_count(uint32_t area, uint32_t field, uint32_t n, uint32_t bytes) {
        if (area == 0) {
            return;
        }
        volatile tt_l1_ptr uint32_t* row = (
            l1(area + KM_LINK_CROSS + pages_held.link % kLinksACore * KM_CROSS_ROW)
        );
        row[field] += n;
        const uint32_t low = row[KM_CROSS_BYTES];
        row[KM_CROSS_BYTES] = low + bytes;
        row[KM_CROSS_BYTES + 1] += low + bytes < low ? 1u : 0u;
    }

    // Whether a side's core word names DRAM: KM_PAGES_DRAM or KM_PAGES_BANK | b.
    inline bool pages_in_dram(uint32_t where) {
        return where == KM_PAGES_DRAM || (where & 0xFFFFFF00u) == KM_PAGES_BANK;
    }

    // This chip's NoC address on NoC `noc` of page `index` of a DRAM side at bank
    // address `address` (`page` bytes): interleaved (KM_PAGES_DRAM) or bank b's range.
    inline uint64_t pages_dram(
        uint32_t where,
        uint32_t address,
        uint32_t page,
        uint32_t index,
        uint8_t noc = noc_index
    ) {
        if (where == KM_PAGES_DRAM) {
            const InterleavedAddrGen<true> tensor = {
                .bank_base_address = address, .page_size = page
            };
            return tensor.get_noc_addr(index, 0, noc);
        }
        return (
            get_noc_addr_from_bank_id<true>(where & 0xFFu, address + index * page, noc)
        );
    }

    // The far island's NoC address of page `index` of its DRAM side `where` at bank
    // address `address` (`page` bytes), for NoC `noc`: its banks are where ours are.
    inline uint64_t pages_far_dram(
        uint32_t where,
        uint32_t address,
        uint32_t page,
        uint32_t index,
        uint8_t noc = noc_index
    ) {
        const uint64_t at = pages_dram(where, address, page, index, noc);
        return (
            link_noc(
                NOC_UNICAST_ADDR_X(at),
                NOC_UNICAST_ADDR_Y(at),
                static_cast<uint32_t>(at),
                noc
            )
        );
    }

    // The far NoC address of the target of the page `k` pages after the request's
    // first: page to_first + k of the DRAM side (for NoC `noc`), or L1 k pages on.
    inline uint64_t pages_target(const PagesRequest& r, uint32_t k, uint8_t noc) {
        if (pages_in_dram(r.to_core)) {
            return pages_far_dram(r.to_core, r.to, r.page, r.to_first + k, noc);
        }
        return link_noc(r.to_core & 0xFFu, r.to_core >> 8, r.to + k * r.page);
    }

    template <typename Kernel>
    uint32_t pages_ring_open(
        const Kernel& kernel, uint32_t area, uint32_t chunk, uint32_t cpu
    );  // below

    // CPU `cpu` sends `share` over its connection, reading ahead; each batch's last
    // packet counts its pages. A range: ranks. Shapes: KM_PAGES_SHAPE_*. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_push(
        const Kernel& kernel, const PagesRequest& r, uint32_t cpu, Partition share
    ) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            return kGaveBadTransfer;  // the counters below are the dedicated mode's
        }
        // the CPU's chunk: the one its open took (a link core's own, or the run's)
        const uint32_t area = pages_held.area;
        const uint32_t chunk = pages_held.chunk;
        const bool range = r.mode == KM_PAGES_RANGE;
        const bool dram = r.from_core == KM_PAGES_DRAM;  // interleaved: by banks
        const bool in_dram = pages_in_dram(r.from_core);
        const uint32_t shape = range ? 0 : r.shape & KM_PAGES_SHAPE_MEASURE;
        const bool no_read = (shape & KM_PAGES_SHAPE_NO_READ) != 0;
        const bool no_wait = no_read && (shape & KM_PAGES_SHAPE_NO_WAIT) != 0;
#if defined(KM_LINK_THIN_SECOND)
        // both links: each read's packets a counted batch, every other one on the
        // second link (its count after its own pages); a forwarded push on its route
#if defined(KM_FORWARD)
        const bool forwarded = forward_held.route != 0 && forward_needed();
#else
        const bool forwarded = false;
#endif
        const bool both = cpu == 0 && side_held.open != 0 && !forwarded;
#endif
#if defined(KM_WIRE) && defined(KM_FORWARD)
        // a forwarded write is made at its last hop by the turning CPU: its DRAM
        // targets for that CPU's NoC
        const bool routed = forward_held.route != 0 && forward_needed();
        const uint8_t noc = routed ? forward_noc(cpu) : noc_index;
#else
        const uint8_t noc = noc_index;
#endif
        const uint32_t packet = pages_packet();
        const uint32_t room = pages_room(chunk, packet);  // the ring: putting cores'
        const uint32_t slots_fit = (
            room > kPagesSlots && r.page > 0 ? (room - kPagesSlots) / r.page : 0
        );
        // pages a read and slots: the shape's, else the loop's own (measured: 16 pages
        // a read in three groups or more, else 4 in groups as many as fit)
        uint32_t per_read = 1u << 2 * (shape & 3u);
        uint32_t asked = (
            (shape >> 2 & 3u) != 0
                ? kPagesSlotsDefault << ((shape >> 2 & 3u) - 1)
                : kPagesSlotsDefault
        );
        const uint32_t mine = range ? share.len * r.pages : share.len;
        if (shape == 0 && mine >= kPagesBulkShare) {
            // a read's block (every bank's pages from DRAM) within the share, else
            // its pages would go a page a read in a few groups; slots as many as fit
            const uint32_t spread = r.from_core == KM_PAGES_DRAM ? NUM_DRAM_BANKS : 1;
            per_read = (
                slots_fit >= 3 * 16 && mine >= spread * 16
                    ? 16
                    : slots_fit >= 2 * 4 && mine >= spread * 4
                        ? 4
                        : 1
            );
            asked = kPagesSlotsMax;
        }
#if defined(KM_LINK_THIN_SECOND)
        if (both && shape == 0 && per_read > 4) {
            per_read = 4;  // groups enough to read ahead of both links
        }
#endif
        // from DRAM only where a bank's pages lie one after another; a range's pages
        // in order (each rank's last page counts the rank's)
        if (range || (dram && r.page % 64 != 0)) {
            per_read = 1;
            asked = asked < kPagesSlotsDefault ? asked : kPagesSlotsDefault;
        }
        uint32_t slots = slots_fit < asked ? slots_fit : asked;
        slots = slots < kPagesSlotsMax ? slots : kPagesSlotsMax;
        const uint32_t groups = slots / per_read;  // a read's slots each
        // pages a packet: as many of a read's as the packet holds, at most kPacketPages
        const uint32_t fits = r.page != 0 ? packet / r.page : 0;
        const uint32_t per_packet = fits < kPacketPages ? fits : kPacketPages;
        if (
            groups == 0 || per_packet == 0 || r.page % 16 != 0 || area % 16 != 0 ||
            chunk % 64 != 0 || r.hops == 0 || pages_held.open == 0
        ) {
            return kGaveBadTransfer;
        }
        // its pages: those of its ranks (a range), else its share of the pages
        const uint32_t base = range ? share.start * r.pages : share.start;
        const uint32_t total = range ? share.len * r.pages : share.len;
        if (total == 0) {
            return kGaveNone;
        }
        // read g's pages: from DRAM per_read pages of one bank for each bank of a
        // block, then the rest one a read; from L1 per_read pages in a row
        const uint32_t banks = dram ? NUM_DRAM_BANKS : 1;
        const uint32_t block = banks * per_read;
        const uint32_t whole = per_read == 1 ? 0 : total / block * banks;
        const uint32_t reads = (
            per_read == 1
                ? total
                : dram
                    ? whole + (total - whole * per_read)
                    : (total + per_read - 1) / per_read
        );

        struct Read {
            uint32_t k;       // its first page, after the share's first
            uint32_t n;       // pages
            uint32_t stride;  // pages from one of its pages to the next
        };

        const auto read_of = [&](uint32_t g) -> Read {
            if (per_read == 1) {
                return {g, 1, 1};
            }
            if (!dram) {
                const uint32_t k = g * per_read;
                return {k, total - k < per_read ? total - k : per_read, 1};
            }
            if (g < whole) {
                return {g / banks * block + g % banks, per_read, banks};
            }
            return {whole * per_read + (g - whole), 1, 1};
        };
        const volatile tt_l1_ptr uint32_t* cores = (
            l1(kernel.args().control + KM_CONTROL_CORES)
        );
        // range: the far core of the share's rank q (the core tables of both islands
        // are the same, checked by the session) and address `at` there
        const auto far = [&](uint32_t q, uint32_t at) {
            const uint32_t xy = (
                core_entry(cores, r.to_core + q, kernel.args().control, kernel.role())
            );
            return link_noc(xy & 0xFFFFu, xy >> 16, at);
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
        uint64_t counter = link_noc(r.counter & 0xFFu, r.counter >> 8, r.count);
        for (uint32_t s = 0; s < groups * per_read; s++) {
            link_header(area + kPagesHeaders + s * kPagesHeaderStride, r.hops);
        }
        auto& sender = link_sender();
        sender.ready();
        // Read g goes into group g % groups with id 1 + its group (modulo the ids):
        // bank reads land out of order; groups sharing an id are waited for together.
        const auto id_of = [](uint32_t group) {
            return 1 + group % kPagesReadIds;
        };
        const auto read = [&](uint32_t g) {
            if (no_read) {
                return;
            }
            const Read one = read_of(g);
            const uint32_t group = g % groups;
            noc_async_read_set_trid(id_of(group));
            const uint32_t index = r.first + base + one.k;
            noc_async_read(
                in_dram
                    ? pages_dram(r.from_core, r.from, r.page, index)
                    : get_noc_addr(
                        r.from_core & 0xFFFFu,
                        r.from_core >> 16,
                        r.from + (base + one.k) * r.page
                    ),
                area + kPagesSlots + group * per_read * r.page,
                one.n * r.page
            );
        };
        uint32_t ahead = reads < groups ? reads : groups;
        for (uint32_t g = 0; g < ahead; g++) {
            read(g);
        }
        uint32_t batch = 0, sent = 0;
        // the issued count after the group before's last packet: that group is read
        // again once its packets left L1, a group later, so no send waits for it
        uint32_t before = 0;
#if defined(KM_LINK_THIN_SECOND)
        uint32_t side_before = side_held.count;
        bool on_side = false;
#endif
        for (uint32_t g = 0; g < reads && code == kGaveNone; g++) {
            const uint32_t group = g % groups;
            if (!no_read && !wait([&] {
                return (
                    ncrisc_noc_read_with_transaction_id_flushed(noc_index, id_of(group))
                );
            })) {
                break;
            }
            const Read one = read_of(g);
            // a packet: up to per_packet of the read's pages (one after another in its
            // slots), each to its own target (a scatter write) unless they follow
            for (uint32_t m = 0; m < one.n;) {
                const uint32_t s = group * per_read + m;
                const uint32_t slot = area + kPagesSlots + s * r.page;
                const LinkHeaderAt header = (
                    reinterpret_cast<LinkHeaderAt>(
                        area + kPagesHeaders + s * kPagesHeaderStride
                    )
                );
                uint32_t c = one.n - m < per_packet ? one.n - m : per_packet;
                const auto counted = [&] {
                    // a range: page j of rank q of the share, a page a packet
                    const uint32_t k = one.k + m * one.stride;
#if defined(KM_LINK_THIN_SECOND)
                    const bool read_end = both && m + c == one.n;  // a read a batch
#else
                    constexpr bool read_end = false;
#endif
                    return (
                        batch + c >= kPagesBatch || sent + c == total ||
                        (range && k % r.pages + 1 == r.pages) || read_end
                    );
                };
                bool count = counted();
                if (count && c > kCountedPages) {
                    c = kCountedPages;  // the fused count's pages at most
                    count = counted();
                }
                uint64_t to[kPacketPages];
                bool follow = true;
                for (uint32_t i = 0; i < c; i++) {
                    const uint32_t k = one.k + (m + i) * one.stride;
                    const uint32_t q = range ? share.start + k / r.pages : 0;
                    to[i] = (
                        range
                            ? far(q, r.to + k % r.pages * r.page)
                            : pages_target(r, base + k, noc)
                    );
                    if (range) {
                        counter = far(q, r.count);
                    }
                    follow = follow && to[i] == to[0] + i * r.page;
                }
                const uint32_t bytes = c * r.page;
#if defined(KM_LINK_THIN_SECOND)
                if (batch == 0) {
                    on_side = both && (side_held.batches++ & 1u) != 0;
                }
#endif
                batch += c;
                if (count) {
                    if (follow) {
                        link_counted_write(
                            header, to[0], counter, static_cast<uint16_t>(batch), bytes
                        );
                    } else {
#if !defined(KM_WIRE)
                        link_scatter_counted(
                            header, to, counter, static_cast<uint16_t>(batch), r.page
                        );
#endif
                    }
                    batch = 0;
                } else if (follow) {
                    link_write(header, to[0], bytes);
                } else {
#if !defined(KM_WIRE)
                    link_scatter_write(header, to, c, r.page);
#endif
                }
#if defined(KM_LINK_THIN_SECOND)
                if (on_side) {
                    if (!side_hand(header, slot, bytes, wait)) {
                        break;
                    }
                    sent += c;
                    m += c;
                    continue;
                }
#endif
                if (!sender.space() && !wait([&] {
                    return sender.space();
                })) {
                    break;
                }
                if (!sender.send_stateful(header, slot, bytes)) {
                    code = kGaveLimit;
                    break;
                }
                sent += c;
                m += c;
            }
            if (code != kGaveNone) {
                break;
            }
            // the group before again (this one's, with one group), once its packets
            // left L1
            const uint32_t mark = noc_nonposted_writes_num_issued[noc_index];
            const bool alone = groups == 1;
#if defined(KM_LINK_THIN_SECOND)
            const uint32_t side_mark = side_held.count;
#endif
            if ((alone || g > 0) && ahead < reads) {
                const uint32_t left = alone ? mark : before;
#if defined(KM_LINK_THIN_SECOND)
                const uint32_t side_left_at = alone ? side_mark : side_before;
                if (!no_wait && !wait([&] {
                    return side_left(side_left_at);
                })) {
                    break;
                }
#endif
                if (!no_wait && !wait([&] {
                    return noc_reached(NIU_MST_NONPOSTED_WR_REQ_SENT, left);
                })) {
                    break;
                }
                read(ahead++);
            }
            before = mark;
#if defined(KM_LINK_THIN_SECOND)
            side_before = side_mark;
#endif
        }
        const uint32_t sending = code;
        noc_async_read_set_trid(0);
        cross_count(
            islands_at(kernel.args().control).link_area(),
            KM_CROSS_PAGES,
            sent,
            sent * r.page
        );
        // The slots and headers are free for the next command once sent and read.
#if defined(KM_LINK_THIN_SECOND)
        const uint32_t side_all = side_held.count;  // and the writer's records left
        if (!wait([side_all] {
            return side_left(side_all);
        })) {
            return code;
        }
#endif
        if (!wait([] {
            return ncrisc_noc_nonposted_writes_sent(noc_index);
        }) ||
            !wait([] {
                return ncrisc_noc_reads_flushed(noc_index);
            })) {
            return code;
        }
        return sending;
    }

    // CPU `cpu` of link core `rank` sends its share on its link (link_at; pages_push);
    // link 0 carries the lines, and pages only with KM_PAGES_SHAPE_LINK0. kGave* code.
    template <typename Kernel>
    inline uint32_t pages_send(
        const Kernel& kernel, const PagesRequest& r, uint32_t rank, uint32_t cpu
    ) {
        const uint32_t cpus = pages_links(r);
        const uint32_t link = link_at(rank, cpu);
        const bool all = (r.shape & KM_PAGES_SHAPE_LINK0) != 0;
        const uint32_t first_link = cpus > 1 && !all ? 1 : 0;
        if (link >= cpus || link < first_link) {
            return kGaveNone;  // no link of this CPU, or the lines link: no share
        }
        const uint32_t ranks = r.counter & 0xFFFFu;  // a range's
        return (
            pages_push(
                kernel,
                r,
                cpu,
                partition(
                    r.mode == KM_PAGES_RANGE ? ranks : r.pages,
                    link - first_link,
                    cpus - first_link
                )
            )
        );
    }

    // At the kernel's end, or after an OPEN whose hello never came (defined below).
    template <typename Kernel>
    void pages_end(const Kernel& kernel);

    // CPU `cpu` of link core `rank` opens its link's (link_at) connection, holds it,
    // greets the far CPU (bounded; done when both say 2), then its ring. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_open(
        const Kernel& kernel, const PagesRequest& r, uint32_t rank, uint32_t cpu
    ) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            return kGaveBadTransfer;
        }
        const uint32_t link = link_at(rank, cpu);
        if (link >= pages_links(r)) {
            return kGaveNone;  // no link of this CPU
        }
#if defined(KM_LINK_CHUNK)
        // a link core's own chunks (the session's buffer KM_LINK_CB, on link cores
        // alone, after the put queue), not the run's transfer buffer every core has
        const uint32_t chunk = KM_LINK_CHUNK;
        const uint32_t buffer = (
            link_queue() + KM_QUEUE_BYTES + kFarBytes + kForwardBytes +
            kDeliverPlaceBytes
        );
#else
        const uint32_t chunk = r.chunk;
        const uint32_t buffer = r.buffer;
#endif
        const uint32_t area = buffer + cpu * chunk;
        if (
            pages_held.open != 0 || area % 16 != 0 || chunk % 64 != 0 ||
            chunk <= kPagesSlots || r.hops == 0 ||
            static_cast<uint64_t>(buffer) + 2ull * chunk > MEM_L1_SIZE
        ) {
            return kGaveBadTransfer;  // one open a kernel: one sender a connection
        }
        auto& sender = link_sender();
        sender.connect((r.links >> 8 * link) & 0xFFu, area);
        sender.open();
        pages_held = {1, area, chunk, r.hops, link, pages_links(r)};
#if defined(KM_LINK_THIN_SECOND)
        side_reader_open(r, rank);
#endif
        const uint32_t queue = link_queue();
        if (link == 0 && queue != 0) {
            // link core 0's put queue empty (no line in either ring, no page landed,
            // none pending) before the hello: no far line comes before it
            for (uint32_t n = 0; n < 2 * KM_QUEUE_LINES; n++) {
                l1(queue + KM_QUEUE_PUTS + n * 64)[KM_ASK_NUMBER] = 0;
            }
            for (uint32_t n = 0; n < KM_QUEUE_LINES; n++) {
                l1(queue + KM_QUEUE_ARRIVED)[n] = 0;
            }
            l1(queue + KM_QUEUE_WORDS)[KM_QUEUE_PENDING_COUNT] = 0;
            l1(queue + KM_QUEUE_WORDS)[KM_QUEUE_LAST] = 0;
#if defined(KM_FAR_PUTS)
            constexpr uint32_t n = KM_ISLANDS_MAX;
            for (uint32_t w = 0; w < KM_FAR_LINE(n) / 4; w++) {
                l1(queue + KM_QUEUE_BYTES)[w] = 0;
            }
            for (uint32_t i = 0; i < n * KM_FAR_LINES; i++) {
                l1(queue + KM_QUEUE_BYTES + KM_FAR_LINE(n) + 64 * i)[KM_ASK_NUMBER] = 0;
            }
#endif
#if defined(KM_FORWARD)
            for (uint32_t w = 0; w < KM_FORWARD_HEADERS / 4; w++) {
                l1(queue + KM_QUEUE_BYTES + kFarBytes)[w] = 0;  // no packet in a room
            }
#endif
#if defined(KM_DELIVER)
            deliver_zero(islands_at(kernel.args().control).link_area());
#endif
        }
        const uint32_t links_area = islands_at(kernel.args().control).link_area();
        const uint32_t row = (
            links_area + KM_LINK_CROSS + link % kLinksACore * KM_CROSS_ROW
        );
        forward_held = {
            kForwardBytes != 0 && queue != 0 ? queue + KM_QUEUE_BYTES + kFarBytes : 0,
            kernel.args().control,
            kernel.args().me,
            links_area != 0 ? row : 0,
            kernel.args().wait_limit,
            0,
            KM_FORWARD_NO_FIELD,
            0,
            KM_FORWARD_CPUS > 1 ? link % kLinksACore : 0,
            link == kForwardLink ? 1u : 0
        };
        const LinkHeaderAt header = link_header(area + kPagesHeaders, r.hops);
        const uint64_t far = (
            link_noc(r.counter & 0xFFu, r.counter >> 8, r.count + cpu * KM_PORT_STRIDE)
        );
        volatile tt_l1_ptr uint32_t* here = (
            l1(
                kernel.args().control + KM_CONTROL_PORT +
                (KM_PAGES_HELLO_PORT + cpu) * KM_PORT_STRIDE
            )
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
                    ncrisc_noc_nonposted_writes_sent(noc_index) && sender.space()
                ) {
                    link_inline(header, far, say);
                    sender.send(header, 0, 0);
                    said = say;
                }
                return heard == 2 && said == 2;
            },
            kernel.args().wait_limit
        );
        if (code != kGaveNone) {  // no hello: the connection closed, a later OPEN may
            pages_end(kernel);
            return code;
        }
        code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            kernel.args().wait_limit
        );
        if (code != kGaveNone) {
            return code;
        }
        return pages_ring_open(kernel, area, chunk, cpu);
    }

    // LINE (link core 0's link 0 CPU): the staged packet, then its count line, into
    // island a6's dispatcher (forwarded past the one across); bounded. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t
    pages_line(const Kernel& kernel, const km_command_t& command) {
        const uint32_t a[7] = {
            command.a0,
            command.a1,
            command.a2,
            command.a3,
            command.a4,
            command.a5,
            command.a6
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
        if (!wait([&] {  // the dispatcher's stores landed: every row's number
            return (
                l1(at)[0] == number && l1(at)[4] == number && l1(at)[8] == number &&
                l1(at)[12] == number && l1(at + 64)[0] == number &&
                l1(at + 64)[5] == number && l1(at + KM_LINK_PACKET_BYTES)[1] == number
            );
        })) {
            return code;
        }
        auto& sender = link_sender();
        const auto packet = [&](
            uint32_t which, uint32_t to, uint32_t from, uint32_t n
        ) {
            const LinkHeaderAt made = (
                link_header(area + KM_LINK_HEADERS + which * 64, 1)
            );
            link_write(made, link_noc(x, y, to), n);
            if (!wait([&] {
                return sender.space();
            })) {
                return false;
            }
            return sender.send_one(made, from, n);
        };
        // counted here when this core's link is the first it crosses, else where it
        // crosses (forward_on)
        const Islands view = islands_at(kernel.args().control);
        const uint32_t to = a[KM_LINE_ISLAND];
        const bool mine = to >= view.count() || view.link(to) == forward_index(view);
        forward_route(to, KM_CROSS_LINES);
        const bool sent = (
            packet(0, a[KM_LINE_PACKET], at, KM_LINK_PACKET_BYTES) &&
            packet(1, a[KM_LINE_COUNT], at + KM_LINK_PACKET_BYTES, 64)
        );
        forward_route_off();
        if (sent) {
            if (mine) {
                cross_count(area, KM_CROSS_LINES, 1, KM_LINK_PACKET_BYTES + 64);
            }
            wait([] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            });
        } else if (code == kGaveNone) {
            code = kGaveLimit;  // send_one refused it (as pages_push): the line is lost
        }
        return code;
    }

    // COUNTS: CPU `cpu` of link core `rank` writes its row of what crossed into page
    // link_at(rank, cpu) of the DRAM tensor at r.to, then waits for the write.
    template <typename Kernel>
    inline uint32_t pages_counts(
        const Kernel& kernel, const PagesRequest& r, uint32_t rank, uint32_t cpu
    ) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            return kGaveBadTransfer;
        }
        const uint32_t area = islands_at(kernel.args().control).link_area();
        const uint32_t page = link_at(rank, cpu);
        if (kLinksACore < 2 && page == KM_PAGES_NO_LINK) {
            return kGaveNone;  // no link of this CPU: no row
        }
        if (area == 0 || r.page < KM_CROSS_ROW || r.page % 64 != 0) {
            return kGaveBadTransfer;
        }
        const InterleavedAddrGen<true> to = {
            .bank_base_address = r.to, .page_size = r.page
        };
        noc_async_write(
            area + KM_LINK_CROSS + cpu * KM_CROSS_ROW,
            to.get_noc_addr(page),
            KM_CROSS_ROW
        );
        if (r.page >= KM_CROSS_ROW + 32) {  // then its ring's words and idle cycles
            noc_async_write(
                area + KM_PUT_RING_WORDS + 16 * cpu,
                to.get_noc_addr(page) + KM_CROSS_ROW,
                16
            );
            noc_async_write(
                area + KM_PUT_RING_IDLE + 16 * cpu,
                to.get_noc_addr(page) + KM_CROSS_ROW + 16,
                16
            );
        }
        if (r.page >= KM_COUNTS_PUT_PAGE && link_queue() != 0) {  // then its put words
            const uint64_t row = to.get_noc_addr(page);
            noc_async_write(area + KM_PUT_TAIL, row + KM_COUNTS_PUT, 48);
            noc_async_write(link_queue(), row + KM_COUNTS_QUEUE, 16);
        }
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
    }

#if defined(KM_PAGE_LINKS)  // else km_worker.hpp's stubs (one card: no link services)
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
            return kernel.poll_at(
                kWherePages,
                [&] {
                    invalidate_l1_cache();
                    return ready();
                },
                kernel.args().wait_limit
            ) == kGaveNone;
        };
        wait([] {
            return ncrisc_noc_nonposted_writes_sent(noc_index);
        });
        link_sender().close_start();
        const bool closed = wait([] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        });
        if (closed) {
            wait([&] {
                return link_sender().closed(area);
            });
        }
    }
#endif

    // The reader waits until the count at r.count reaches r.pages, then takes that
    // much off it by an atomic (one request a count at a time). kGave* code.
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
        if (code != kGaveNone) {
            return code;
        }
        // taken off by an atomic: a count landing meanwhile is not lost
        noc_semaphore_inc(
            get_noc_addr(reinterpret_cast<uint32_t>(count)), 0u - r.pages
        );
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
    }

    // The page CPUs (KM_PUT_CPUS at `area`): from link 1 when the pair has more links
    // (link 0 its lines), so link core 0 holds one; page CPU j holds link first + j.
    inline uint32_t page_cpus(uint32_t area) {
        const uint32_t cpus = l1(area)[KM_PUT_CPUS / 4];
        const uint32_t links = pages_held.links;
        if (cpus != 0 && cpus <= links) {
            return cpus;
        }
        return links > 1 ? links - 1 : 1;
    }

    inline uint32_t page_first(uint32_t area) {
        const uint32_t cpus = page_cpus(area);
        return pages_held.links > cpus ? 1 : 0;
    }

    inline bool page_cpu(uint32_t area) {
        const uint32_t first = page_first(area);
        return pages_held.link >= first && pages_held.link < first + page_cpus(area);
    }

    inline uint32_t page_index(uint32_t area) {
        return pages_held.link - page_first(area);
    }

    // The page CPU that sends a put line across, confirms and counts the bad ones: the
    // one holding link 1 (link 0 when alone).
    inline uint32_t page_lead(uint32_t area) {
        if constexpr (kLinksACore == 1) {  // link core 0's own CPU (its put queue's)
            return 0;
        }
        return page_first(area) == 0 && pages_held.links > 1 ? 1 : 0;
    }

    // Page CPU j's share of a line of `pages` pages: link 0's CPU only in a bulk line,
    // a smaller one on the others (its CPU kept for its lines).
    inline Partition page_share(uint32_t area, uint32_t pages, uint32_t j) {
        const uint32_t cpus = page_cpus(area);
        if (page_lead(area) == 0 || pages >= kPagesBulk) {
            return partition(pages, j, cpus);
        }
        return j == 0 ? Partition{0, 0} : partition(pages, j - 1, cpus - 1);
    }

    // The first link core of this core's pair (the island's link cores toward one
    // island): where the lines are.
    template <typename Kernel>
    inline uint32_t link_core0(const Kernel& kernel) {
        return kernel.args().me - pages_held.link / kLinksACore;
    }

    // The far island's link core 0 (x | y << 16): link core 0's relay tile, read once a
    // kernel by a page CPU on another link core (0 until then).
    inline uint32_t pages_far0;

    // Sets pages_far0 if it is not yet, with L1 `copy` (16 bytes) for the read across.
    // kGave* code.
    template <typename Kernel>
    inline uint32_t pages_far_core0(const Kernel& kernel, uint32_t copy) {
        if (pages_far0 != 0) {
            return kGaveNone;
        }
        const uint32_t home = link_core0(kernel);
        if (kernel.args().me == home) {
            pages_far0 = island_area()[KM_ISLAND_RELAY_TILE];
            return kGaveNone;
        }
        const uint32_t at = (
            reinterpret_cast<uint32_t>(&island_area()[KM_ISLAND_RELAY_TILE])
        );
        noc_async_read(kernel.core_noc(home, at & ~15u), copy, 16);
        const uint32_t code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
        invalidate_l1_cache();
        if (code == kGaveNone) {
            pages_far0 = l1(copy)[(at & 15u) / 4];
        }
        return code;
    }

    // The line at `line` of link core 0's area: itself on link core 0, else read over
    // the NoC into `copy` (64 bytes, bounded). 0: gave up (`code`).
    template <typename Kernel>
    inline uint32_t pages_line_at(
        const Kernel& kernel, uint32_t line, uint32_t copy, uint32_t& code
    ) {
        const uint32_t home = link_core0(kernel);
        if (kernel.args().me == home) {
            return line;
        }
        noc_async_read(kernel.core_noc(home, line), copy, 64);
        code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
        invalidate_l1_cache();
        return code == kGaveNone ? copy : 0;
    }

    // CPU `cpu`'s ring at L1 `area` (`chunk` bytes), at its open: marks and words
    // cleared; a page CPU's ring goes into link core 0's put block. kGave* code.
    template <typename Kernel>
    uint32_t pages_ring_open(
        const Kernel& kernel, uint32_t area, uint32_t chunk, uint32_t cpu
    ) {
        const uint32_t slots = ring_slots(chunk, pages_packet());
        const uint32_t at = area + chunk - slots * (kRingHeader + pages_packet());
        pages_ring_held = {at, slots, 0, 0};
        for (uint32_t s = 0; s < slots; s++) {
            l1(at + s * kRingHeader)[kRingMark] = 0;
        }
        const uint32_t links = islands_at(kernel.args().control).link_area();
        if (links == 0) {
            return kGaveNone;
        }
        volatile tt_l1_ptr uint32_t* words = l1(links + KM_PUT_RING_WORDS + 16 * cpu);
        words[KM_PUT_RING_TAIL / 4] = 0;
        words[KM_PUT_RING_SENT / 4] = 0;
        words[KM_PUT_RING_SENDING / 4] = 0;
        words[KM_PUT_RING_ROUTER / 4] = 0;
        for (uint32_t w = 0; w < 4; w++) {
            l1(links + KM_PUT_RING_IDLE + 16 * cpu)[w] = 0;
        }
        pages_ring_held.left = wall();
        const uint32_t home = link_core0(kernel);
        const uint32_t queue = link_queue();
        if (kernel.args().me == home && cpu == 0 && queue != 0) {
            // the put queue (emptied before the hello), where the putting cores read
            l1(links)[KM_PUT_QUEUE / 4] = queue;
        }
        if (!page_cpu(links)) {
            return kGaveNone;
        }
        const uint32_t j = page_index(links);
        if (kernel.args().me == home) {
            l1(links)[KM_PUT_RING_FAR / 4] = island_area()[KM_ISLAND_RELAY_TILE];
        }
        const uint32_t said[3][2] = {
            {KM_PUT_RING + 4 * j, slots != 0 ? at : 0},
            {KM_PUT_RING_CORE + 4 * j, kernel.args().me | cpu << 16 | noc_index << 24},
            {KM_PUT_RING_SLOTS, slots | 1u << 31}
        };
        for (const auto& one : said) {  // the far word above first: a fence
            invalidate_l1_cache();
            noc_inline_dw_write(kernel.core_noc(home, links + one[0]), one[1]);
        }
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
    }

    // Page CPU `cpu` of its core: its ring's ready slots, in order, over its link in
    // bursts; a burst is counted sent once its packets left L1. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t
    pages_ring(const Kernel& kernel, uint32_t cpu, uint32_t area) {
        PagesRing& ring = pages_ring_held;
        if (ring.slots == 0) {
            return kGaveNone;
        }
        volatile tt_l1_ptr uint32_t* words = l1(area + KM_PUT_RING_WORDS + 16 * cpu);
        // the slots of the held burst counted sent once its packets left L1, so the
        // putting cores may fill them while later bursts go (no wait for a burst)
        const auto publish = [&] {
            if (
                ring.holding &&
                noc_reached(NIU_MST_NONPOSTED_WR_REQ_SENT, ring.held_mark)
            ) {
                words[KM_PUT_RING_SENT / 4] = ring.held;
                ring.holding = false;
            }
        };
        publish();
        if (
            !ring.holding && words[KM_PUT_RING_SENT / 4] != ring.issued &&
            noc_reached(NIU_MST_NONPOSTED_WR_REQ_SENT, ring.mark)
        ) {
            words[KM_PUT_RING_SENT / 4] = ring.issued;
        }
        // the cycles since its last turn, by what its first slot is now
        invalidate_l1_cache();
        const uint32_t started = wall();
        volatile tt_l1_ptr uint32_t* idle = l1(area + KM_PUT_RING_IDLE + 16 * cpu);
        uint32_t since = KM_PUT_RING_LATE;
        if (
            l1(ring.at + ring.issued % ring.slots * kRingHeader)[kRingMark] !=
            ring.issued + 1
        ) {
            since = (
                words[KM_PUT_RING_TAIL / 4] == ring.issued
                    ? KM_PUT_RING_EMPTY
                    : KM_PUT_RING_FILLING
            );
        }
        idle[since / 4] += started - ring.left;
        auto& sender = link_sender();
        const uint32_t payloads = ring.at + ring.slots * kRingHeader;
        const uint32_t packet = pages_packet();
        const uint32_t limit = kernel.args().wait_limit;
        uint32_t code = kGaveNone;
        uint32_t router = 0;  // cycles waiting for the router's slots
        bool sending = false;
        for (uint32_t bursts = 0; bursts < kRingBursts; bursts++) {
            uint32_t packets = 0, bytes = 0;
            invalidate_l1_cache();
            while (packets < ring.slots) {
                const uint32_t s = ring.issued % ring.slots;
                const uint32_t header = ring.at + s * kRingHeader;
                if (l1(header)[kRingMark] != ring.issued + 1) {
                    break;  // not ready yet (a wait here keeps the CPU's other
                            // services, the put lines' confirmations, late)
                }
                const uint32_t size = l1(header)[kRingBytes];
                if (!sender.space()) {
                    const uint32_t waiting = wall();
                    code = kernel.poll_at(
                        kWherePages,
                        [&] {
                            publish();
                            return sender.space();
                        },
                        limit
                    );
                    router += wall() - waiting;
                    if (code != kGaveNone) {
                        return code;
                    }
                }
                if (packets == 0) {
                    sender.ready();
                }
                if (!sender.send_stateful(
                    reinterpret_cast<LinkHeaderAt>(header),
                    payloads + s * packet,
                    size
                )) {
                    return kGaveLimit;
                }
                ring.issued++;
                packets++;
                bytes += size;
            }
            if (packets == 0) {
                break;  // nothing ready: the CPU's other services
            }
            sending = true;
            ring.mark = noc_nonposted_writes_num_issued[noc_index];
            cross_count(area, KM_CROSS_PAGES, packets, bytes);
            publish();
            if (!ring.holding) {  // this burst held until its packets left L1
                ring.held = ring.issued;
                ring.held_mark = ring.mark;
                ring.holding = true;
            }
        }
        publish();
        if (sending) {  // the turn's cycles sending, and of them waiting for the router
            words[KM_PUT_RING_SENDING / 4] += wall() - started;
            words[KM_PUT_RING_ROUTER / 4] += router;
            idle[KM_PUT_RING_TURNS / 4] += 1;
        }
        ring.left = wall();
        return code;
    }

    // A header in L1 `at` for a packet over the held link, all its words cleared first.
    inline LinkHeaderAt pages_header(uint32_t at) {
        return link_header(at, pages_held.hops);
    }

    // Sends `header` (with `bytes` of payload from L1 `payload`, 0: none) over the held
    // link once it has room, then waits until it left. kGave* code.
    template <typename Kernel>
    inline uint32_t pages_packet(
        const Kernel& kernel, LinkHeaderAt header, uint32_t payload, uint32_t bytes
    ) {
        auto& sender = link_sender();
        const uint32_t limit = kernel.args().wait_limit;
        const uint32_t code = kernel.poll_at(
            kWherePages,
            [&] {
                return sender.space();
            },
            limit
        );
        if (code != kGaveNone) {
            return code;
        }
        if (!sender.send(header, payload, bytes)) {
            return kGaveLimit;
        }
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            limit
        );
    }

    // A get of an island further on: queued again at link core `core` toward it
    // (pages_onward: the line reserved; `later`: not free yet). kGave* code.
    inline uint32_t pages_onward;  // 1 + the line number reserved; 0: none

    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_requeue(
        const Kernel& kernel,
        const volatile tt_l1_ptr uint32_t* line,
        uint32_t area,
        uint32_t core,
        bool& later
    ) {
        const uint32_t limit = kernel.args().wait_limit;
        volatile tt_l1_ptr uint32_t* returned = l1(area + KM_ASK_RETURNED);
        uint32_t code = kGaveNone;
        later = false;
        if (pages_onward == 0) {
            returned[0] = 0xFFFFFFFFu;  // the atomic's answer not yet in
            invalidate_l1_cache();
            noc_fast_atomic_increment<noc_mode, true>(
                noc_index,
                write_at_cmd_buf,
                kernel.core_noc(core, area + KM_PUT_TAIL),
                NOC_UNICAST_WRITE_VC,
                1,
                31,
                false,
                false,
                area + KM_ASK_RETURNED
            );
            atomic_return_reset(noc_index, write_at_cmd_buf);
            code = kernel.poll_at(
                kWherePages,
                [&] {
                    invalidate_l1_cache();
                    return (
                        ncrisc_noc_nonposted_atomics_flushed(noc_index) &&
                        returned[0] != 0xFFFFFFFFu
                    );
                },
                limit
            );
            if (code != kGaveNone) {
                return code;
            }
            pages_onward = returned[0] + 1;
        }
        const uint32_t number = pages_onward - 1;
        // its place free once each page CPU there took line KM_QUEUE_LINES before
        noc_async_read(
            kernel.core_noc(core, area + KM_PUT_TAKEN), area + KM_ASK_READ, 16
        );
        code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            limit
        );
        invalidate_l1_cache();
        for (uint32_t j = 0; code == kGaveNone && j < page_cpus(area); j++) {
            if (number - l1(area + KM_ASK_READ)[j] >= KM_QUEUE_LINES) {
                later = true;
            }
        }
        if (code != kGaveNone || later) {
            return code;
        }
        line_copy(l1(area + KM_ASK_OWN), line, number + 1);
        noc_async_write(
            area + KM_ASK_OWN,
            kernel.core_noc(
                core, link_queue() + KM_QUEUE_PUTS + number % KM_QUEUE_LINES * 64
            ),
            64
        );
        code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        if (code == kGaveNone) {
            pages_onward = 0;
        }
        return code;
    }

    // Page CPU j, the far ask `answered` + 1 at `at` for part of a page: one CPU in

    // A get line refused here: KM_PORT_REFUSED onto its asker's count (`asker` x | y <<
    // 16, `count` its L1 address), so its wait ends and names the refusal. kGave* code.
    template <typename Kernel>
    inline uint32_t pages_ask_refused(
        const Kernel& kernel,
        uint32_t header_at,
        uint32_t from,
        uint32_t asker,
        uint32_t count
    ) {
        forward_route(from, KM_CROSS_PAGES);  // back to the asker
        auto* header = pages_header(header_at);
        link_increment(
            header, link_noc(asker & 0xFFFFu, asker >> 16, count), KM_PORT_REFUSED
        );
        const uint32_t code = pages_packet(kernel, header, 0, 0);
        forward_route_off();
        return code;
    }

    // turn sends the range's words, then its 16-byte lines counted; then answered.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_answer_part(
        const Kernel& kernel,
        uint32_t cpu,
        uint32_t area,
        uint32_t at,
        uint32_t answered
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        const volatile tt_l1_ptr uint32_t* line = l1(at);
        const uint32_t asker = line[KM_ASK_CORE];
        const uint32_t place = (asker & 0xFFu) | (asker >> 16 & 0xFFu) << 8;
        const uint32_t page = line[KM_ASK_PAGE];
        const uint32_t to = line[KM_ASK_TO];
        const uint32_t count = line[KM_ASK_COUNT];
        const uint32_t offset = line[KM_ASK_GET_SPAN] & 0xFFFFu;
        const uint32_t bytes = line[KM_ASK_GET_SPAN] >> 16;
        const uint64_t from = (
            pages_dram(KM_PAGES_DRAM, line[KM_ASK_TENSOR], page, line[KM_ASK_FIRST]) +
            offset
        );
        // the range read at a 64-byte line of the slots plus its DRAM address's low 6
        // bits (the DRAM read's rule), so its 16-byte lines are the asker's too
        const uint32_t slot = (
            ((pages_held.area + kPagesSlots + 63) & ~63u) +
            static_cast<uint32_t>(from) % 64
        );
        const bool good = (
            line[KM_ASK_HOLDER] == islands_at(kernel.args().control).here() &&
            line[KM_ASK_PAGES] == 1 && page != 0 && page % 16 == 0 &&
            bytes <= pages_packet() && bytes != 0 && (offset | bytes) % 4 == 0 &&
            offset + bytes <= page && (static_cast<uint32_t>(from) ^ to) % 16 == 0 &&
            count % 16 == 0 && pages_held.open != 0 &&
            slot + bytes <=
                pages_held.area + pages_room(pages_held.chunk, pages_packet())
        );
        // the asks in turn over the page CPUs (but link 0's, which keeps its lines)
        const uint32_t j = page_index(area);
        const uint32_t spare = page_lead(area), cpus = page_cpus(area);
        const uint32_t server = cpus > spare ? spare + answered % (cpus - spare) : 0;
        uint32_t code = kGaveNone;
        if (good && j == server) {
            noc_async_read(from, slot, bytes);
            code = kernel.poll_at(
                kWherePages,
                [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                },
                kernel.args().wait_limit
            );
            invalidate_l1_cache();
            const uint64_t into = link_noc(place & 0xFFu, place >> 8, to);
            const uint64_t counter = link_noc(place & 0xFFu, place >> 8, count);
            const uint32_t head = (
                (16 - to % 16) % 16 < bytes ? (16 - to % 16) % 16 : bytes
            );
            const uint32_t lines = (bytes - head) / 16 * 16;
            const uint32_t header_at = pages_held.area + kPagesHeaders;
            forward_route(line[KM_ASK_FROM], KM_CROSS_PAGES);  // back to the asker
            for (uint32_t w = 0; w < bytes && code == kGaveNone; w += 4) {
                if (w == head) {
                    w += lines;  // the lines go last, with the count
                    if (w == bytes) {
                        break;
                    }
                }
                auto* header = pages_header(header_at);
                link_inline(header, into + w, l1(slot + w)[0]);
                code = pages_packet(kernel, header, 0, 0);
            }
            if (code == kGaveNone) {
                auto* header = pages_header(header_at);
                if (lines != 0) {
                    link_counted_write(header, into + head, counter, 1, lines);
                } else {
                    link_increment(header, counter, 1);
                }
                code = (
                    pages_packet(kernel, header, lines != 0 ? slot + head : 0, lines)
                );
            }
            forward_route_off();
            cross_count(
                islands_at(kernel.args().control).link_area(), KM_CROSS_PAGES, 1, bytes
            );
        }
        if (code != kGaveNone) {
            return code;
        }
        if (!good && j == page_lead(area)) {
            words[KM_ASK_BAD / 4] += 1;
            code = (
                pages_ask_refused(
                    kernel, pages_held.area + kPagesHeaders, line[KM_ASK_FROM], asker, count
                )
            );
            if (code != kGaveNone) {
                return code;
            }
        }
        const uint32_t xy = pages_far0;  // the far link core 0
        const uint32_t own = area + KM_PAGE_CPU + cpu * KM_PAGE_CPU_BYTES;
        auto* header = pages_header(own + KM_PAGE_CPU_HEADER);
        link_inline(
            header,
            link_noc(xy & 0xFFFFu, xy >> 16, area + KM_ASK_SERVED + 4 * j),
            answered + 1
        );
        code = pages_packet(kernel, header, 0, 0);
        if (code == kGaveNone) {
            words[KM_ASK_ANSWERED / 4 + cpu] = answered + 1;
        }
        return code;
    }

    // Page CPU j answers the far island's next landed ask: share j of its pages from
    // DRAM into the asker's L1 (pages_push), counted at the far link core 0.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t
    pages_answer(const Kernel& kernel, uint32_t cpu, uint32_t area) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        invalidate_l1_cache();
        const uint32_t own = area + KM_PAGE_CPU + cpu * KM_PAGE_CPU_BYTES;
        const uint32_t answered = words[KM_ASK_ANSWERED / 4 + cpu];
        uint32_t code = kGaveNone;
        const uint32_t at = (
            pages_line_at(
                kernel,
                area + KM_ASK_INBOX + answered % KM_ASK_SLOTS * 64,
                own + KM_PAGE_CPU_ASK,
                code
            )
        );
        if (at == 0) {
            return code;
        }
        const volatile tt_l1_ptr uint32_t* line = l1(at);  // forwarded as it is
        uint32_t copy[16];  // the line as checked: every field read from here
        if (!line_take(line, answered + 1, copy)) {
            return kGaveNone;
        }
        const bool part = (copy[KM_ASK_RING] & KM_ASK_FORM) == KM_ASK_GET_BYTES;
#if !defined(KM_FORWARD)
        if (part) {  // part of a page: a link apart, held here
            return pages_answer_part(kernel, cpu, area, at, answered);
        }
#endif
        const uint32_t asker = copy[KM_ASK_CORE];
        const uint32_t place = (asker & 0xFFu) | (asker >> 16 & 0xFFu) << 8;
        // into the asker's DRAM (KM_ASK_DRAM: 1 interleaved, 2 + b bank b)
        const uint32_t dram = copy[KM_ASK_DRAM];
        const uint32_t into = dram == 1 ? KM_PAGES_DRAM : KM_PAGES_BANK | (dram - 2);
        const PagesRequest r = {
            copy[KM_ASK_TENSOR],
            copy[KM_ASK_PAGE],
            copy[KM_ASK_FIRST],
            copy[KM_ASK_PAGES],
            copy[KM_ASK_TO],
            dram != 0 ? into : place,
            pages_held.chunk,
            copy[KM_ASK_COUNT],
            place,
            pages_held.hops,
            KM_PAGES_SEND,
            0,
            pages_held.area - cpu * pages_held.chunk,
            KM_PAGES_DRAM,
            0,
            copy[KM_ASK_DRAM_FIRST]
        };
        const Islands view = islands_at(kernel.args().control);
        const uint32_t holder = copy[KM_ASK_HOLDER];
        const bool fits = (
            r.pages != 0 && r.page != 0 && r.page % 16 == 0 &&
            r.page <= pages_packet() && r.to % 16 == 0 && r.count % 16 == 0
        );
        // an island further on holds it: on by this island's link toward it (queued
        // there once, by the lead page CPU)
        const uint32_t toward = holder < view.count() ? view.link(holder) : 0xFFu;
#if defined(KM_FORWARD)
        const bool onward = (
            holder != view.here() && (fits || part) && toward != KM_ISLAND_NO_LINK &&
            toward != forward_index(view)
        );
        if (part && !onward) {  // part of a page, held here: its own answer
            return pages_answer_part(kernel, cpu, area, at, answered);
        }
#else
        const bool onward = false;
#endif
        const bool good = holder == view.here() && fits;
        const uint32_t j = page_index(area);
        const Partition share = page_share(area, r.pages, j);
        if (onward && j == page_lead(area)) {
#if defined(KM_FORWARD)
            bool later = false;
            const uint32_t core = view.base() + view.ranks() + toward;
            code = pages_requeue(kernel, line, area, core, later);
            if (later) {
                return kGaveNone;  // its line there is not free yet
            }
#endif
        } else if (good && share.len != 0) {
            forward_route(copy[KM_ASK_FROM], KM_CROSS_PAGES);  // back to the asker
            code = pages_push(kernel, r, cpu, share);
            forward_route_off();
        }
        if (code != kGaveNone) {
            return code;
        }
        if (!good && !onward && j == page_lead(area)) {
            words[KM_ASK_BAD / 4] += 1;
            code = (
                pages_ask_refused(
                    kernel, own + KM_PAGE_CPU_HEADER, copy[KM_ASK_FROM], asker, r.count
                )
            );
            if (code != kGaveNone) {
                return code;
            }
        }
        const uint32_t xy = pages_far0;  // the far link core 0
        auto* header = pages_header(own + KM_PAGE_CPU_HEADER);
        link_inline(
            header,
            link_noc(xy & 0xFFFFu, xy >> 16, area + KM_ASK_SERVED + 4 * j),
            answered + 1
        );
        code = pages_packet(kernel, header, 0, 0);
        if (code == kGaveNone) {
            words[KM_ASK_ANSWERED / 4 + cpu] = answered + 1;
        }
        return code;
    }

    // The lead page CPU's time of a put line's step `step` (KM_STAMP_*), in its link
    // area at `area` (link core 0's): what a collective's latency is made of.
    inline void put_stamp(uint32_t area, uint32_t step) {
        l1(area + KM_PUT_STAMPS)[step] = wall();
    }

    // The lead page CPU: get line `at` on into the far link core 0's asks (through L1
    // `copy`, numbered there) while every far page CPU has room; `sent`. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_get_across(
        const Kernel& kernel, uint32_t area, uint32_t at, uint32_t copy, bool& sent
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        const uint32_t gets = words[KM_ASK_TAKEN / 4];  // get lines sent across
        sent = false;
        for (uint32_t j = 0; j < page_cpus(area); j++) {
            if (gets - words[KM_ASK_SERVED / 4 + j] >= KM_ASK_SLOTS) {
                return kGaveNone;  // the far asks are full
            }
        }
        line_copy(l1(copy), l1(at), gets + 1);
        const uint32_t xy = pages_far0;  // the far link core 0
        auto* header = pages_header(area + KM_ASK_HEADERS);
        link_write(
            header,
            link_noc(
                xy & 0xFFFFu, xy >> 16, area + KM_ASK_INBOX + gets % KM_ASK_SLOTS * 64
            ),
            64
        );
        const uint32_t code = pages_packet(kernel, header, copy, 64);
        if (code == kGaveNone) {
            words[KM_ASK_TAKEN / 4] = gets + 1;
            cross_count(area, KM_CROSS_ASKS, 1, 64);
            sent = true;
        }
        return code;
    }

    // Page CPU j has taken line `taken` + 1: its count, where the asking cores read it
    // (link core 0's area). kGave* code.
    template <typename Kernel>
    inline uint32_t pages_taken(
        const Kernel& kernel, uint32_t area, uint32_t j, uint32_t taken
    ) {
        l1(area)[KM_PUT_TAKEN / 4 + j] = taken + 1;
        const uint32_t home = link_core0(kernel);
        if (kernel.args().me == home) {
            return kGaveNone;
        }
        noc_inline_dw_write(
            kernel.core_noc(home, area + KM_PUT_TAKEN + 4 * j), taken + 1
        );
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
    }

    // A batch's landing page in its holder's DRAM: one a holder, source island and put
    // line owner, then one a holder and far place (source s: s * KM_FAR_LINES + k).
    constexpr uint32_t batch_page(
        uint32_t islands, uint32_t holder, uint32_t source, uint32_t owner
    ) {
        return (holder * islands + source) * KM_QUEUE_LINES + owner;
    }

    constexpr uint32_t batch_far_page(
        uint32_t islands, uint32_t holder, uint32_t place
    ) {
        return (
            islands * islands * KM_QUEUE_LINES + holder * islands * KM_FAR_LINES + place
        );
    }

#if defined(KM_FAR_PUTS)
    // This page CPU's far put lines toward each island past the next (the lead's own
    // count is KM_FAR_SENT): the place of its next one there.
    inline uint32_t far_mine[KM_ISLANDS_MAX];

    // A page CPU but the lead: whether the lead sent line `count` toward `holder` (its
    // KM_FAR_SENT on link core 0, read into `copy`, 16 bytes). kGave* code.
    template <typename Kernel>
    inline uint32_t far_lead_sent(
        const Kernel& kernel, uint32_t holder, uint32_t count, uint32_t copy, bool& sent
    ) {
        constexpr uint32_t n = KM_ISLANDS_MAX;
        const uint32_t at = link_queue() + KM_QUEUE_BYTES + KM_FAR_SENT(n) + 4 * holder;
        const uint32_t home = link_core0(kernel);
        if (kernel.args().me == home) {  // the lead's other CPU: its words are here
            invalidate_l1_cache();
            sent = static_cast<int32_t>(l1(at)[0] - count) > 0;
            return kGaveNone;
        }
        noc_async_read(kernel.core_noc(home, at & ~15u), copy, 16);
        const uint32_t code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
        invalidate_l1_cache();
        sent = static_cast<int32_t>(l1(copy)[(at & 15u) / 4] - count) > 0;
        return code;
    }

    // Page CPU j's share of a put line's pages for an island past the next, pushed once
    // the lead took the place there and sent the line (the holder counts pages first).
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_put_far(
        const Kernel& kernel,
        uint32_t cpu,
        uint32_t area,
        uint32_t at,
        uint32_t j,
        uint32_t copy,
        bool& sent
    ) {
        constexpr uint32_t n = KM_ISLANDS_MAX;
        const Islands view = islands_at(kernel.args().control);
        volatile tt_l1_ptr uint32_t* line = l1(at);
        const uint32_t far = link_queue() + KM_QUEUE_BYTES;
        volatile tt_l1_ptr uint32_t* words = l1(far);
        const uint32_t holder = line[KM_ASK_HOLDER];
        const uint32_t tile = holder < n ? view.far_tile(holder, false) : 0;
        if (tile == 0) {  // no place to send it to: given up, its place not taken
            return kGaveBadItem;
        }
        const bool lead = j == page_lead(area);
        invalidate_l1_cache();
        const uint32_t count = (
            lead ? words[KM_FAR_SENT(n) / 4 + holder] : far_mine[holder]
        );
        if (lead) {
            sent = count - words[KM_FAR_CREDIT(n) / 4 + holder] < KM_FAR_LINES;
        } else {
            const uint32_t code = far_lead_sent(kernel, holder, count, copy, sent);
            if (code != kGaveNone) {
                return code;
            }
        }
        if (!sent) {
            return kGaveNone;
        }
        const uint32_t i = view.here() * KM_FAR_LINES + count % KM_FAR_LINES;
        const uint32_t dram = line[KM_ASK_DRAM];
        const uint32_t out = dram == 1 ? KM_PAGES_DRAM : KM_PAGES_BANK | (dram - 2);
        const uint32_t form = line[KM_ASK_RING] & KM_ASK_FORM;
        const bool batch = form == KM_ASK_RING_BATCH;  // onto its page for this place
        const uint32_t to_first = (
            batch ? batch_far_page(view.count(), holder, i) : line[KM_ASK_FIRST]
        );
        const PagesRequest r = {
            line[KM_ASK_TO],
            line[KM_ASK_PAGE],
            dram != 0 ? line[KM_ASK_DRAM_FIRST] : line[KM_ASK_FIRST],
            line[KM_ASK_PAGES],
            line[KM_ASK_TENSOR],
            KM_PAGES_DRAM,
            pages_held.chunk,
            far + KM_FAR_COUNT(n) + 4 * i,
            (tile & 0xFFu) | (tile >> 16 & 0xFFu) << 8,
            pages_held.hops,
            KM_PAGES_SEND,
            0,
            pages_held.area - cpu * pages_held.chunk,
            dram != 0 ? out : line[KM_ASK_CORE],
            0,
            to_first
        };
        const bool good = (
            holder < view.count() && r.pages != 0 && r.page != 0 && r.page % 16 == 0 &&
            r.page <= pages_packet() && r.from % 16 == 0 &&
            (form == 0 || (batch && dram == 0 && r.pages == 1))
        );
        const Partition share = partition(r.pages, j, page_cpus(area));  // all CPUs
        uint32_t code = kGaveNone;
        if (good && share.len != 0) {  // this CPU's share, into the place's count
            forward_route(holder, KM_CROSS_PAGES);
            code = pages_push(kernel, r, cpu, share);
            forward_route_off();
        }
        if (!lead) {
            if (code == kGaveNone) {
                far_mine[holder] = count + 1;
            }
            return code;
        }
        if (!good) {
            line[KM_ASK_PAGES] = 0;  // confirmed with no page
            line_seal(line, line[KM_ASK_NUMBER]);
            l1(area)[KM_PUT_BAD / 4] += 1;
        }
        if (code == kGaveNone) {
            forward_route(holder, KM_CROSS_LINES);
            auto* header = pages_header(area + KM_PUT_HEADERS);
            link_write(
                header,
                link_noc(tile & 0xFFFFu, tile >> 16, far + KM_FAR_LINE(n) + 64 * i),
                64
            );
            code = pages_packet(kernel, header, at, 64);
        }
        forward_route_off();
        if (code == kGaveNone) {
            words[KM_FAR_SENT(n) / 4 + holder] = count + 1;
            cross_count(area, KM_CROSS_PUTS, 1, 64);
        }
        return code;
    }

#endif

    // Page CPU j: the queue's next line. A put: share j of its pages into the holder's
    // DRAM, the lead sends the line across; a get: the lead sends it on. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t
    pages_put(const Kernel& kernel, uint32_t cpu, uint32_t area) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        invalidate_l1_cache();
        const uint32_t j = page_index(area);
        const uint32_t own = area + KM_PAGE_CPU + cpu * KM_PAGE_CPU_BYTES;
        const uint32_t taken = words[KM_PUT_TAKEN / 4 + j];
        const uint32_t queue = link_queue();
        const uint32_t place_at = taken % KM_QUEUE_LINES * 64;
        const uint32_t slot = queue + KM_QUEUE_PUTS + place_at;
        const bool lead = j == page_lead(area);
        if (queue == 0) {
            return kGaveNone;  // no put queue: no put line
        }
        uint32_t code = kGaveNone;
        const uint32_t at = pages_line_at(kernel, slot, own + KM_PAGE_CPU_PUT, code);
        if (at == 0) {
            return code;
        }
        volatile tt_l1_ptr uint32_t* line = l1(at);
        uint32_t copy[16];  // the line as checked: every field read from here
        if (!line_take(line, taken + 1, copy)) {
            return kGaveNone;  // no line yet
        }
        if (copy[KM_ASK_RING] >> KM_ASK_OP_SHIFT == KM_ASK_OP_READ) {
            if (lead) {  // the other page CPUs only take it
                bool sent = true;
                if (copy[KM_ASK_HOLDER] != islands_at(kernel.args().control).here()) {
                    code = (
                        pages_get_across(kernel, area, at, own + KM_PAGE_CPU_ASK, sent)
                    );
                } else {
                    words[KM_ASK_BAD / 4] += 1;
                }
                if (code != kGaveNone || !sent) {
                    return code;  // not sent yet: taken on a later pass
                }
            }
            return pages_taken(kernel, area, j, taken);
        }
        const uint32_t op = copy[KM_ASK_RING] >> KM_ASK_OP_SHIFT;
        if (op == KM_ASK_OP_L1 || op == KM_ASK_OP_ADD) {  // km_far_serve.hpp
            return far_put(kernel, cpu, area, at, taken);
        }
#if defined(KM_DELIVER)
        if (op == KM_ASK_OP_DELIVER) {  // km_deliver.hpp
            return deliver_put(kernel, cpu, area, at, taken);
        }
#endif
#if defined(KM_FAR_PUTS)
        if (islands_at(kernel.args().control).hops(copy[KM_ASK_HOLDER]) > 1) {
            bool sent = true;  // each page CPU its share, the lead the line
            code = pages_put_far(kernel, cpu, area, at, j, own + KM_PAGE_CPU_ASK, sent);
            if (code != kGaveNone || !sent) {
                return code;  // not sent yet: taken on a later pass
            }
            return pages_taken(kernel, area, j, taken);
        }
#endif
        const uint32_t xy = pages_far0;  // the far link core 0
        const uint32_t place = (xy & 0xFFu) | (xy >> 16 & 0xFFu) << 8;
        // from the asker's DRAM (KM_ASK_DRAM: 1 interleaved, 2 + b bank b)
        const uint32_t dram = copy[KM_ASK_DRAM];
        const uint32_t out = dram == 1 ? KM_PAGES_DRAM : KM_PAGES_BANK | (dram - 2);
        const PagesRequest r = {
            copy[KM_ASK_TO],
            copy[KM_ASK_PAGE],
            dram != 0 ? copy[KM_ASK_DRAM_FIRST] : copy[KM_ASK_FIRST],
            copy[KM_ASK_PAGES],
            copy[KM_ASK_TENSOR],
            KM_PAGES_DRAM,
            pages_held.chunk,
            queue + KM_QUEUE_ARRIVED + copy[KM_ASK_OWNER] % KM_QUEUE_LINES * 4,
            place,
            pages_held.hops,
            KM_PAGES_SEND,
            0,
            pages_held.area - cpu * pages_held.chunk,
            dram != 0 ? out : copy[KM_ASK_CORE],
            0,
            copy[KM_ASK_FIRST]
        };
        const bool good = (
            copy[KM_ASK_HOLDER] != islands_at(kernel.args().control).here() &&
            r.pages != 0 && r.page != 0 && r.page % 16 == 0 &&
            r.page <= pages_packet() && r.from % 16 == 0
        );
        if (lead) {  // on link core 0: the line is its own
            put_stamp(area, KM_STAMP_TAKEN);
            if (!good) {
                line[KM_ASK_PAGES] = 0;
                line_seal(line, copy[KM_ASK_NUMBER]);
                words[KM_PUT_BAD / 4] += 1;
            }
            // the next arrival place: gets take none (KM_QUEUE_LAST)
            const uint32_t sent = words[KM_PUT_DONE / 4];
            auto* header = pages_header(area + KM_PUT_HEADERS);
            link_write(
                header,
                link_noc(
                    xy & 0xFFFFu,
                    xy >> 16,
                    queue + KM_QUEUE_ARRIVALS + sent % KM_QUEUE_LINES * 64
                ),
                64
            );
            code = pages_packet(kernel, header, at, 64);
            if (code == kGaveNone) {
                words[KM_PUT_DONE / 4] = sent + 1;
                cross_count(area, KM_CROSS_PUTS, 1, 64);
                put_stamp(area, KM_STAMP_ACROSS);
            }
        }
        const Partition share = page_share(area, r.pages, j);
        const bool ring = (copy[KM_ASK_RING] & KM_ASK_FORM) != 0;
        if (code == kGaveNone && good && !ring && share.len != 0) {
            code = pages_push(kernel, r, cpu, share);
        }
        if (lead && code == kGaveNone) {
            put_stamp(area, KM_STAMP_PUSHED);
        }
        if (code != kGaveNone) {
            return code;
        }
        return pages_taken(kernel, area, j, taken);
    }

    // CPU `cpu`'s batch of byte records in this core's own words `own` (KM_OWN_BATCH),
    // 0: none.
    inline uint32_t put_batch(uint32_t own, uint32_t cpu) {
        const uint32_t base = own + KM_OWN_BATCH;
        return own != 0 ? (base + 63) / 64 * 64 + cpu * KM_PUT_BATCH_BYTES : 0;
    }

    // `code`, else kGaveBadItem when a confirmed line of this CPU's (own words `own`)
    // was refused by its holder (KM_OWN_REFUSED), the mark then cleared: named once.
    inline uint32_t put_refusal(volatile tt_l1_ptr uint32_t* own, uint32_t code) {
        invalidate_l1_cache();
        for (uint32_t p = 0; p < 2 && code == kGaveNone; p++) {
            const uint32_t word = own[KM_OWN_PUT_CONFIRMED / 4 + p];
            if (
                (word & KM_OWN_REFUSED) != 0 &&
                KM_OWN_CONFIRMED(word) == own[KM_OWN_PUT_NUMBER / 4 + p]
            ) {  // that parity's line done: nothing else writes the word
                               // now
                own[KM_OWN_PUT_CONFIRMED / 4 + p] = KM_OWN_CONFIRMED(word);
                code = kGaveBadItem;
            }
        }
        return code;
    }

    // A pending batch line (KM_QUEUE_E_PAGES): its pages, its source island at bit 16,
    // and this mark until its records are written (the confirming loop passes it).
    constexpr uint32_t kPutBatchMark = 1u << 30;
    constexpr uint32_t kPutBatchPages = 0xFFFFu;
    constexpr uint32_t kPutBatchSource = 16;
    // A pending entry's far word (KM_QUEUE_E_FAR) for a line the holder refused: a far
    // op with no far word, one spoiled on its way, a batch with a bad record.
    constexpr uint32_t kFarRefused = 0xFFFFFFFFu;

    // The lead page CPU: the batch on landing page `page` (batch_page), each record one
    // NoC write in order, waited for; a bad record ends it, counted, `refused` set.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_batch_apply(
        const Kernel& kernel, uint32_t area, uint32_t page, bool& refused
    ) {
        const Islands view = islands_at(kernel.args().control);
        const uint32_t room = put_batch(view.own_words(), 0);  // a link core puts none
        invalidate_l1_cache();
        const uint32_t landing = l1(area)[KM_PUT_LANDED / 4];
        if (room == 0 || landing == 0) {
            l1(area)[KM_PUT_BAD / 4] += 1;
            refused = true;
            return kGaveNone;
        }
        const uint32_t limit = kernel.args().wait_limit;
        noc_async_read(
            pages_dram(KM_PAGES_DRAM, landing, KM_PUT_BATCH_BYTES, page),
            room,
            KM_PUT_BATCH_BYTES
        );
        uint32_t code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            limit
        );
        if (code != kGaveNone) {
            return code;
        }
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* head = l1(room);
        const uint32_t used = head[KM_BATCH_USED];
        bool good = used >= KM_BATCH_HEAD && used <= KM_PUT_BATCH_BYTES;
        for (uint32_t at = KM_BATCH_HEAD; good && at < used;) {
            const volatile tt_l1_ptr uint32_t* record = l1(room + at);
            const uint32_t page_bytes = record[2];
            const uint32_t offset = record[3] & ((1u << KM_BATCH_SPAN_SHIFT) - 1);
            const uint32_t bytes = record[3] >> KM_BATCH_SPAN_SHIFT;
            const uint32_t size = KM_BATCH_HEAD + (offset % 16 + bytes + 15) / 16 * 16;
            good = (
                bytes != 0 && (offset | bytes) % 4 == 0 && page_bytes % 16 == 0 &&
                offset + bytes <= page_bytes && size <= used - at
            );
            if (good) {  // its bytes at the offset's low 4 bits (the NoC rule)
                const uint64_t to = (
                    pages_dram(KM_PAGES_DRAM, record[0], page_bytes, record[1])
                );
                noc_async_write(
                    room + at + KM_BATCH_HEAD + offset % 16, to + offset, bytes
                );
                at += size;
            }
        }
        if (!good) {
            l1(area)[KM_PUT_BAD / 4] += 1;
            refused = true;
        }
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
    }

    // The lead page CPU: a gather's `pages`, written here, onto its keepers' ports
    // (share k onto keeper k's), a local atomic each; bad keepers counted. kGave* code.
    template <typename Kernel>
    inline uint32_t pages_keeper(
        const Kernel& kernel, uint32_t area, uint32_t keeper, uint32_t pages
    ) {
        if (keeper == 0 || pages == 0) {
            return kGaveNone;
        }
        const Islands view = islands_at(kernel.args().control);
        const uint32_t rank = (keeper & 0xFFu) - 1;
        const uint32_t ranks = (keeper >> KM_KEEPER_RANKS_SHIFT & 0xFFu) + 1;
        const uint32_t port = keeper >> KM_KEEPER_PORT_SHIFT;
        // a port from KM_PORT_SLOTS on: a word of the keeper's counts table
        const uint32_t word = port - KM_PORT_SLOTS;
        if (
            rank + ranks > view.ranks() ||
            (port >= KM_PORT_SLOTS && (word >= view.count() || view.counts() == 0))
        ) {
            l1(area)[KM_PUT_BAD / 4] += 1;
            return kGaveNone;
        }
        const uint32_t count = (
            port >= KM_PORT_SLOTS
                ? view.counts() + 4 * word
                : kernel.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE +
                    KM_PORT_COUNT * 4
        );
        for (uint32_t k = 0; k < ranks; k++) {
            const uint32_t share = partition(pages, k, ranks).len;
            if (share != 0) {
                noc_semaphore_inc(
                    kernel.core_noc(view.base() + rank + k, count), share
                );
            }
        }
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
    }

#if defined(KM_FAR_PUTS)
    // The lead: each island's next line in its places here once its pages landed,
    // confirmed to its asker and the place given back to the island it came from.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t
    pages_confirm_far(const Kernel& kernel, uint32_t area) {
        constexpr uint32_t n = KM_ISLANDS_MAX;
        const Islands view = islands_at(kernel.args().control);
        const uint32_t far = link_queue() + KM_QUEUE_BYTES;
        volatile tt_l1_ptr uint32_t* words = l1(far);
        invalidate_l1_cache();
        for (uint32_t s = 0; s < view.count() && s < n; s++) {
            const uint32_t taken = words[KM_FAR_TAKEN(n) / 4 + s];
            const uint32_t i = s * KM_FAR_LINES + taken % KM_FAR_LINES;
            volatile tt_l1_ptr uint32_t* line = l1(far + KM_FAR_LINE(n) + 64 * i);
            const uint32_t number = line[KM_ASK_NUMBER];
            uint32_t copy[16];  // the line as checked: every field read from here
            if (number == 0 || !line_take(line, number, copy)) {
                continue;  // none, or not whole yet
            }
            const uint32_t pages = copy[KM_ASK_PAGES];
            if (static_cast<int32_t>(words[KM_FAR_COUNT(n) / 4 + i] - pages) < 0) {
                continue;  // its pages not all landed
            }
            const uint32_t asker = copy[KM_ASK_CORE];
            const uint32_t owner = copy[KM_ASK_OWNER] % KM_QUEUE_LINES;
            const bool batch = (copy[KM_ASK_RING] & KM_ASK_FORM) == KM_ASK_RING_BATCH;
            bool refused = false;  // an L1 write's or an add's, refused there
            uint32_t code = far_land_line(kernel, line, refused);
            if (code == kGaveNone && batch && pages != 0) {  // written, then confirmed
                code = (
                    pages_batch_apply(
                        kernel, area, batch_far_page(view.count(), view.here(), i), refused
                    )
                );
            }
            forward_route(copy[KM_ASK_FROM], KM_CROSS_PAGES);  // back to the asker
            if (code == kGaveNone && pages != 0 && !batch) {   // a batch counts no page
                auto* header = pages_header(area + KM_PUT_HEADERS + 64);
                link_increment(
                    header,
                    link_noc(asker & 0xFFFFu, asker >> 16, copy[KM_ASK_COUNT]),
                    pages
                );
                code = pages_packet(kernel, header, 0, 0);
            }
            if (code == kGaveNone) {
                const uint32_t own = (
                    view.own_words() + (owner / 2 % 2 == 1 ? KM_OWN_CPU_BYTES : 0)
                );
                auto* done = pages_header(area + KM_PUT_HEADERS + 128);
                link_inline(
                    done,
                    link_noc(
                        asker & 0xFFFFu,
                        asker >> 16,
                        own + KM_OWN_PUT_CONFIRMED + owner % 2 * 4
                    ),
                    KM_OWN_CONFIRM(number - 1u, view.here()) |  // the holder's
                        (refused ? KM_OWN_REFUSED : 0u)
                );
                code = pages_packet(kernel, done, 0, 0);
            }
            forward_route_off();
            if (code == kGaveNone) {
                const uint32_t dram = copy[KM_ASK_DRAM];
                code = (
                    pages_keeper(
                        kernel, area, dram != 0 ? copy[KM_ASK_KEEPER] : 0, pages
                    )
                );
            }
            if (code != kGaveNone) {
                return code;
            }
            words[KM_FAR_COUNT(n) / 4 + i] -= pages;
            line[KM_ASK_NUMBER] = 0;
            words[KM_FAR_TAKEN(n) / 4 + s] = taken + 1;
            const uint32_t tile = view.far_tile(s, true);  // its place back
            forward_route(s, KM_CROSS_LINES);
            auto* back = pages_header(area + KM_PUT_HEADERS + 128);
            link_inline(
                back,
                link_noc(
                    tile & 0xFFFFu, tile >> 16, far + KM_FAR_CREDIT(n) + 4 * view.here()
                ),
                taken + 1
            );
            code = pages_packet(kernel, back, 0, 0);
            forward_route_off();
            if (code != kGaveNone) {
                return code;
            }
            cross_count(area, KM_CROSS_PUTS, 1, pages != 0 ? 12 : 8);
        }
        return kGaveNone;
    }
#endif

    // The lead page CPU: far put lines arrive as pending; each once its pages landed
    // (KM_QUEUE_ARRIVED) is confirmed to the asker's port and parity word. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t
    pages_confirm(const Kernel& kernel, uint32_t area) {
        const uint32_t queue = link_queue();
        if (queue == 0) {
            return kGaveNone;
        }
        volatile tt_l1_ptr uint32_t* words = l1(area);
#if defined(KM_FAR_PUTS)
        const uint32_t far = pages_confirm_far(kernel, area);
        if (far != kGaveNone) {
            return far;
        }
#endif
#if defined(KM_DELIVER)
        const uint32_t delivered = deliver_confirm(kernel, area);  // km_deliver.hpp
        if (delivered != kGaveNone) {
            return delivered;
        }
#endif
        volatile tt_l1_ptr uint32_t*
            count = (l1(queue + KM_QUEUE_WORDS) + KM_QUEUE_PENDING_COUNT);
        invalidate_l1_cache();
        // the arrivals, in order (KM_PUT_CONFIRMED: the far lines taken); never more
        // pending than places: a CPU's two lines at most
        for (uint32_t n = 0; n < KM_ASK_SLOTS && count[0] < KM_QUEUE_LINES; n++) {
            const uint32_t taken = words[KM_PUT_CONFIRMED / 4];
            const volatile tt_l1_ptr uint32_t* line = (
                l1(queue + KM_QUEUE_ARRIVALS + taken % KM_QUEUE_LINES * 64)
            );
            // a new line: whole (its last word checks), past the last (gets skip
            // numbers)
            const uint32_t number = line[KM_ASK_NUMBER];
            const uint32_t last = l1(queue + KM_QUEUE_WORDS)[KM_QUEUE_LAST];
            uint32_t copy[16];  // the line as checked: every field read from here
            if (
                static_cast<int32_t>(number - last) <= 0 ||
                !line_take(line, number, copy)
            ) {
                break;
            }
            volatile tt_l1_ptr uint32_t* entry = (
                l1(queue + KM_QUEUE_PENDING + count[0] * KM_QUEUE_ENTRY)
            );
            entry[KM_QUEUE_E_OWNER] = copy[KM_ASK_OWNER] % KM_QUEUE_LINES;
            entry[KM_QUEUE_E_PAGES] = copy[KM_ASK_PAGES];
            entry[KM_QUEUE_E_CORE] = copy[KM_ASK_CORE];
            entry[KM_QUEUE_E_COUNT] = copy[KM_ASK_COUNT];
            entry[KM_QUEUE_E_NUMBER] = number;
            // a gather's keeper (a line from the asker's DRAM; word 14 is a get's span)
            entry[KM_QUEUE_E_KEEPER] = copy[KM_ASK_DRAM] != 0 ? copy[KM_ASK_KEEPER] : 0;
            far_arrived(kernel, entry, line);  // an L1 write's or an add's words
            put_stamp(area, KM_STAMP_ARRIVED);
            const uint32_t form = copy[KM_ASK_RING] & KM_ASK_FORM;
            if (form == KM_ASK_RING_BATCH && copy[KM_ASK_PAGES] != 0) {
                entry[KM_QUEUE_E_PAGES] = (
                    (copy[KM_ASK_PAGES] & kPutBatchPages) | kPutBatchMark |
                    (copy[KM_ASK_FROM] & 0xFFu) << kPutBatchSource
                );
            }
            count[0] = count[0] + 1;
            l1(queue + KM_QUEUE_WORDS)[KM_QUEUE_LAST] = number;
            words[KM_PUT_CONFIRMED / 4] = taken + 1;
        }
        // each pending batch whose page has landed: its records written, then it
        // counts no page (its count taken off here), so it is confirmed below
        for (uint32_t i = 0; i < count[0]; i++) {
            volatile tt_l1_ptr uint32_t* entry = (
                l1(queue + KM_QUEUE_PENDING + i * KM_QUEUE_ENTRY)
            );
            const uint32_t pages = entry[KM_QUEUE_E_PAGES];
            volatile tt_l1_ptr uint32_t*
                arrived = (l1(queue + KM_QUEUE_ARRIVED) + entry[KM_QUEUE_E_OWNER]);
            if (
                (pages & kPutBatchMark) == 0 ||
                static_cast<int32_t>(arrived[0] - (pages & kPutBatchPages)) < 0
            ) {
                continue;
            }
            bool refused = false;
            const uint32_t code = (
                pages_batch_apply(
                    kernel,
                    area,
                    batch_page(
                        islands_at(kernel.args().control).count(),
                        islands_at(kernel.args().control).here(),
                        pages >> kPutBatchSource & 0xFFu,
                        entry[KM_QUEUE_E_OWNER]
                    ),
                    refused
                )
            );
            if (code != kGaveNone) {
                return code;
            }
            arrived[0] -= pages & kPutBatchPages;
            entry[KM_QUEUE_E_PAGES] = 0;
            if (refused) {  // confirmed below as refused
                entry[KM_QUEUE_E_FAR] = kFarRefused;
            }
        }
        // each pending line whose pages have all landed, the last moved into its place
        for (uint32_t i = 0; i < count[0];) {
            volatile tt_l1_ptr uint32_t* entry = (
                l1(queue + KM_QUEUE_PENDING + i * KM_QUEUE_ENTRY)
            );
            const uint32_t owner = entry[KM_QUEUE_E_OWNER];
            const uint32_t pages = entry[KM_QUEUE_E_PAGES];
            volatile tt_l1_ptr uint32_t*
                arrived = (l1(queue + KM_QUEUE_ARRIVED) + owner);
            if (
                static_cast<int32_t>(arrived[0] - pages) < 0 ||
                far_behind(kernel, queue, i)
            ) {
                i++;
                continue;  // not all written yet, or after its CPU's earlier line
            }
            const uint32_t asker = entry[KM_QUEUE_E_CORE];
            const bool refused = entry[KM_QUEUE_E_FAR] == kFarRefused;
            uint32_t code = far_land(kernel, area, entry);  // an L1 write's, an add's
            if (code == kGaveNone && pages != 0) {
                auto* header = pages_header(area + KM_PUT_HEADERS + 64);
                link_increment(
                    header,
                    link_noc(asker & 0xFFFFu, asker >> 16, entry[KM_QUEUE_E_COUNT]),
                    pages
                );
                code = pages_packet(kernel, header, 0, 0);
            }
            if (code != kGaveNone) {
                return code;
            }
            // the owner's count back to 0 before its next line on this parity: that
            // line's pages add by atomics, which a later store would lose
            arrived[0] -= pages;
            // the asker's CPU (owner / 2 % 2: 1 the writer) and parity (owner % 2), in
            // its core's own words (one address on every island, the session's)
            const uint32_t own = (
                islands_at(kernel.args().control).own_words() +
                (owner / 2 % 2 == 1 ? KM_OWN_CPU_BYTES : 0)
            );
            auto* done = pages_header(area + KM_PUT_HEADERS + 128);
            link_inline(
                done,
                link_noc(
                    asker & 0xFFFFu,
                    asker >> 16,
                    own + KM_OWN_PUT_CONFIRMED + owner % 2 * 4
                ),
                KM_OWN_CONFIRM(
                    entry[KM_QUEUE_E_NUMBER] - 1u,
                    islands_at(kernel.args().control).here()  // the holder's
                ) | (refused ? KM_OWN_REFUSED : 0u)
            );
            code = pages_packet(kernel, done, 0, 0);
            if (code != kGaveNone) {
                return code;
            }
            code = pages_keeper(kernel, area, entry[KM_QUEUE_E_KEEPER], pages);
            if (code != kGaveNone) {
                return code;
            }
            put_stamp(area, KM_STAMP_CONFIRMED);
            cross_count(area, KM_CROSS_PUTS, 1, pages != 0 ? 8 : 4);
            count[0] = count[0] - 1;
            // the later ones moved down, in arrival order (far_behind keeps a CPU's
            // lines in order)
            for (
                uint32_t w = i * KM_QUEUE_ENTRY / 4; w < count[0] * KM_QUEUE_ENTRY / 4;
                w++
            ) {
                l1(
                    queue + KM_QUEUE_PENDING
                )[w] = (l1(queue + KM_QUEUE_PENDING)[w + KM_QUEUE_ENTRY / 4]);
            }
        }
        return kGaveNone;
    }

    // A putting core's view of the island's rings (link core 0's put block, read a
    // put): page CPUs, slots a ring, far link core 0, put queue, rings and cores.
    struct PutRings {
        uint32_t read;
        uint32_t cpus;
        uint32_t slots;
        uint32_t far;
        uint32_t queue;
        uint32_t ring[KM_PAGE_CPUS];
        uint32_t core[KM_PAGE_CPUS];
    };

    // `rings` all zero, a field at a time: a block clear (`PutRings rings{}`) may
    // compile to a memset call, which an overlay part cannot make (put_run_clear).
    inline void put_rings_clear(PutRings& rings) {
        rings.read = rings.cpus = rings.slots = rings.far = rings.queue = 0;
        for (uint32_t j = 0; j < KM_PAGE_CPUS; ++j) {
            rings.ring[j] = rings.core[j] = 0;
        }
    }

    constexpr uint32_t kPutNoLine = 0xFFFFFFFFu;

    // Reads `rings` from link core `link0`'s `area` into L1 `room` until every page CPU
    // said its ring and (with `number`) took line KM_QUEUE_LINES before it (bounded).
    template <typename Kernel>
    __attribute__((noinline)) uint32_t put_rings_read(
        const Kernel& core,
        uint32_t link0,
        uint32_t area,
        uint32_t room,
        PutRings& rings,
        uint32_t number = kPutNoLine
    ) {
        const volatile tt_l1_ptr uint32_t* words = l1(room);
        const auto at = [&](uint32_t offset) {
            return words[(offset - KM_PUT_TAKEN) / 4];
        };
        bool reading = false;
        const uint32_t code = core.poll_at(
            kWhereAskPlace,
            [&] {
                if (reading) {
                    if (!ncrisc_noc_reads_flushed(noc_index)) {
                        return false;
                    }
                    reading = false;
                    invalidate_l1_cache();
                    const uint32_t cpus = at(KM_PUT_CPUS);
                    const uint32_t slots = at(KM_PUT_RING_SLOTS);
                    bool said = (
                        slots != 0 && cpus != 0 && cpus <= KM_PAGE_CPUS &&
                        at(KM_PUT_QUEUE) != 0
                    );
                    for (uint32_t j = 0; said && j < cpus; j++) {
                        said = at(KM_PUT_RING_CORE + 4 * j) != 0;
                        said = (
                            said &&
                            (
                                number == kPutNoLine ||
                                number - at(KM_PUT_TAKEN + 4 * j) < KM_QUEUE_LINES
                            )
                        );
                    }
                    if (said) {
                        return true;
                    }
                }
                noc_async_read(
                    core.core_noc(link0, area + KM_PUT_TAKEN), room, KM_PUT_RING_READ
                );
                reading = true;
                return false;
            },
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            return code;
        }
        rings = {
            1,
            at(KM_PUT_CPUS),
            at(KM_PUT_RING_SLOTS) & ~(1u << 31),
            at(KM_PUT_RING_FAR),
            at(KM_PUT_QUEUE)
        };
        for (uint32_t j = 0; j < rings.cpus; j++) {
            rings.ring[j] = at(KM_PUT_RING + 4 * j);
            rings.core[j] = at(KM_PUT_RING_CORE + 4 * j);
            rings.slots = rings.ring[j] != 0 ? rings.slots : 0;
        }
        return kGaveNone;
    }

    // A putting core's run of put line `number` into the rings: pages first .. first +
    // pages - 1 at bank `address` of the island `hops` away; L1 `build`, `answer`.
    struct PutRun {
        PutRings rings;  // slots 0: no rings, a line a piece (km_ask.hpp put_piece)
        uint32_t holder;
        uint32_t waited;  // pages put_piece waited for itself (no rings)
        uint32_t area;
        uint32_t build;
        uint32_t answer;
        uint32_t number;
        uint32_t hops;
        uint32_t address;
        uint32_t bytes;
        uint32_t first;
        uint32_t pages;
        uint32_t done;
        uint32_t piece;
        uint32_t owner;  // its line's KM_ASK_OWNER: where the far island counts it
        uint32_t turn;   // cycles its pieces waited for their slots (atomic, room)
        uint32_t write;  // cycles they wrote pages and headers into the slots
    };

    // `run` all zero, a field at a time: a block clear of the struct (`run = {}`, an
    // aggregate's) may compile to a memset call, which an overlay part cannot make.
    inline void put_run_clear(PutRun& run) {
        run.rings.read = run.rings.cpus = run.rings.slots = 0;
        run.rings.far = run.rings.queue = 0;
        for (uint32_t j = 0; j < KM_PAGE_CPUS; ++j) {
            run.rings.ring[j] = run.rings.core[j] = 0;
        }
        run.holder = run.waited = run.area = run.build = run.answer = 0;
        run.number = run.hops = run.address = run.bytes = run.first = 0;
        run.pages = run.done = run.piece = run.owner = run.turn = run.write = 0;
    }

    // The next `count` pages of `run` from L1 `from` into the rings, a piece at a time,
    // in turn from page CPU number % cpus on; `from` is free on return. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t
    put_ring_pieces(const Kernel& core, PutRun& run, uint32_t from, uint32_t count) {
        const PutRings& rings = run.rings;
        const uint32_t limit = core.args().wait_limit;
        const uint32_t packet = pages_packet();
        const uint32_t bytes = run.bytes;
        const uint32_t fits = bytes != 0 ? packet / bytes : 0;
        const uint32_t per = fits < kPacketPages ? fits : kPacketPages;
        const uint32_t counted = per < kCountedPages ? per : kCountedPages;
        // packets a piece: half a ring, the headers' room at most
        const uint32_t half = rings.slots / 2;
        const uint32_t most = half < kRingPiece ? half : kRingPiece;
        if (per == 0 || most == 0 || count > run.pages - run.done || rings.cpus == 0) {
            return kGaveBadItem;
        }
        const uint32_t largest = (most - 1) * per + counted;  // pages a piece
        const uint64_t counter = (
            link_noc(
                rings.far & 0xFFFFu,
                rings.far >> 16,
                rings.queue + KM_QUEUE_ARRIVED + run.owner * 4
            )
        );
        volatile tt_l1_ptr uint32_t* words = l1(run.answer);
        const auto written = [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        };
        uint32_t code = kGaveNone, done = 0;
        for (; done < count; run.piece++) {
            const uint32_t n = count - done < largest ? count - done : largest;
            // its packets: the uncounted ones of `per` pages, then the counted one
            const uint32_t last = n < counted ? n : counted;
            const uint32_t packets = (n - last + per - 1) / per + 1;
            // packet k's pages, `at` of the piece's before it: `per`, the last
            // uncounted one what is left, the counted one `last`
            const auto packet_pages = [&](uint32_t k, uint32_t at) {
                const uint32_t left = n - last - at;
                return k + 1 == packets ? last : left < per ? left : per;
            };
            const uint32_t j = (run.number + run.piece) % rings.cpus;
            const uint32_t ring = rings.ring[j];
            const uint32_t server = rings.core[j] & 0xFFFFu;  // its core
            const uint32_t tail = (
                run.area + KM_PUT_RING_WORDS + 16 * (rings.core[j] >> 16 & 0xFFu)
            );
#if defined(KM_WIRE)
            // the ring's CPU makes the writes: their DRAM addresses for its NoC
            const uint8_t noc = rings.core[j] >> 24 & 1u;
#else
            const uint8_t noc = noc_index;
#endif
            const uint32_t asked = wall();
            words[0] = kRingNoAnswer;
            invalidate_l1_cache();  // a fence: the store lands before the answer can
            noc_fast_atomic_increment<noc_mode, true>(
                noc_index,
                write_at_cmd_buf,
                core.core_noc(server, tail + KM_PUT_RING_TAIL),
                NOC_UNICAST_WRITE_VC,
                packets,
                31,
                false,
                false,
                run.answer
            );
            atomic_return_reset(noc_index, write_at_cmd_buf);
            // the atomic's answer (the piece's first slot, a sequence), then the
            // ring's words read until it has room
            uint32_t slot = 0, until = 0;
            bool reserved = false, pausing = false;
            code = core.poll_at(
                kWhereAskSlot,
                [&] {
                    if (pausing) {  // the ring sends the slots before ours meanwhile
                        if (static_cast<int32_t>(wall() - until) < 0) {
                            return false;
                        }
                        pausing = false;
                        noc_async_read(core.core_noc(server, tail), run.answer, 16);
                        return false;
                    }
                    invalidate_l1_cache();
                    if (!reserved) {
                        if (
                            !ncrisc_noc_nonposted_atomics_flushed(noc_index) ||
                            words[0] == kRingNoAnswer
                        ) {
                            return false;
                        }
                        reserved = true;
                        slot = words[0];
                        // the ring's words read at once
                        noc_async_read(core.core_noc(server, tail), run.answer, 16);
                        return false;
                    }
                    if (!ncrisc_noc_reads_flushed(noc_index)) {
                        return false;
                    }
                    const uint32_t ahead = (
                        slot + packets - words[KM_PUT_RING_SENT / 4] - rings.slots
                    );
                    if (static_cast<int32_t>(ahead) <= 0) {
                        return true;
                    }
                    // reread after the slots ahead could be sent (kRingPollCycles
                    // each): every core reading at once would hold the link core's NoC
                    const uint32_t wait = ahead < kRingPollMost ? ahead : kRingPollMost;
                    until = wall() + wait * kRingPollCycles;
                    pausing = true;
                    return false;
                },
                limit
            );
            if (code != kGaveNone) {
                break;
            }
            const uint32_t roomy = wall();
            run.turn += roomy - asked;
            const uint32_t payloads = ring + rings.slots * kRingHeader;
            for (uint32_t k = 0, at = 0; k < packets; k++) {
                const uint32_t has = packet_pages(k, at);
                noc_async_write(
                    from + (done + at) * bytes,
                    core.core_noc(server, payloads + (slot + k) % rings.slots * packet),
                    has * bytes
                );
                at += has;
            }
            code = core.poll_at(kWhereAskSent, written, limit);  // pages, then marks
            if (code != kGaveNone) {
                break;
            }
            for (uint32_t k = 0, at = 0; k < packets; k++) {
                const uint32_t has = packet_pages(k, at);
                const uint32_t h = run.build + k * kRingHeader;
                const LinkHeaderAt header = link_header(h, run.hops);
                uint64_t to[kPacketPages];
                for (uint32_t i = 0; i < has; i++) {
                    to[i] = (
                        pages_far_dram(
                            KM_PAGES_DRAM,
                            run.address,
                            bytes,
                            run.first + run.done + done + at + i,
                            noc
                        )
                    );
                }
                if (k + 1 == packets) {
                    if (has == 1) {
                        link_counted_write(
                            header, to[0], counter, static_cast<uint16_t>(n), bytes
                        );
                    } else {
#if !defined(KM_WIRE)
                        link_scatter_counted(
                            header, to, counter, static_cast<uint16_t>(n), bytes
                        );
#endif
                    }
                } else if (has == 1) {
                    link_write(header, to[0], bytes);
                } else {
#if !defined(KM_WIRE)
                    link_scatter_write(header, to, has, bytes);
#endif
                }
                l1(h)[kRingMark] = slot + k + 1;
                l1(h)[kRingBytes] = has * bytes;
                at += has;
            }
            for (uint32_t k = 0; k < packets;) {  // the header slots: one or two runs
                const uint32_t s = (slot + k) % rings.slots;
                const uint32_t span = (
                    packets - k < rings.slots - s ? packets - k : rings.slots - s
                );
                noc_async_write(
                    run.build + k * kRingHeader,
                    core.core_noc(server, ring + s * kRingHeader),
                    span * kRingHeader
                );
                k += span;
            }
            run.write += wall() - roomy;
            done += n;
        }
        run.done += done;
        const uint32_t flushing = wall();
        const uint32_t sent = core.poll_at(kWhereAskSent, written, limit);
        run.write += wall() - flushing;
        return code != kGaveNone ? code : sent;
    }

#if defined(KM_PAGE_LINKS)  // else km_worker.hpp's stubs (one card: no link services)
    // The link core's services in one place: CPU `cpu`'s part (0 reader, 1 writer) of a
    // PAGES command (open, send, range, wait, line). False: given up (recorded).
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
            // its index in the command's link cores (those toward one island)
            code = pages_send(kernel, r, kernel.name(command) - first(command), cpu);
        } else if (r.mode == KM_PAGES_LINE && links) {
            // the CPU that holds its pair's link 0
            code = (
                cpu == 0 && pages_held.link == 0
                    ? pages_line(kernel, command)
                    : kGaveNone
            );
        } else if (r.mode == KM_PAGES_OPEN && links) {
            // its index in its pair's link cores (an OPEN's first: the pair's first)
            code = pages_open(kernel, r, index - r.first, cpu);
        } else if (r.mode == KM_PAGES_COUNTS && links) {
            code = pages_counts(kernel, r, index, cpu);
#if defined(KM_LINKS) && (defined(COMPILE_FOR_BRISC) || defined(KM_MESSAGE_SERVICES))
        } else if (r.mode == KM_PAGES_MESSAGES && links) {
            // link 0's CPU; with a service a link, each link's below the services
            code = (
                link_serves(kernel, cpu) ? link_messages(kernel, command) : kGaveNone
            );
#elif defined(KM_LINKS)  // the writer, link 0's service alone: none here
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

    // The link core's services between commands, a step each a call: answers, the
    // queue's lines, confirmations, ring sends. A failure stops the core (`word`).
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
#if defined(KM_FORWARD)
        if (pages_held.link == 0) {
            forward_turn();
        }
#endif
        if (page_cpu(area)) {
            if (code == kGaveNone) {
                const uint32_t own = area + KM_PAGE_CPU + cpu * KM_PAGE_CPU_BYTES;
                code = pages_far_core0(kernel, own + KM_PAGE_CPU_HEADER);
            }
            if (code == kGaveNone) {
                code = pages_answer(kernel, cpu, area);
            }
            if (code == kGaveNone) {
                code = pages_put(kernel, cpu, area);
            }
#if defined(KM_X280_FAR)
            if (code == kGaveNone && page_index(area) == page_lead(area)) {
                code = deliver_x280(kernel, cpu, area);  // km_deliver.hpp
            }
#endif
            if (code == kGaveNone && page_index(area) == page_lead(area)) {
                l1(area + KM_PUT_STAMPS)[KM_STAMP_PASSES] += 1;
                code = pages_confirm(kernel, area);
            }
            if (code == kGaveNone) {
                code = pages_ring(kernel, cpu, area);
            }
        }
        live[kLvSequence + kernel.role()] = sequence;
        if (code != kGaveNone) {
            kernel.gave_up(word, code, kWherePages, 0, 0);
#if defined(KM_DELIVER)
            deliver_gave(kernel, code);  // a run's status line (km_deliver.hpp)
#endif
        }
    }
#endif

}  // namespace km
