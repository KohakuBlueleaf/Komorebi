// Pages across a link (KM_HANDLER_PAGES): each data-movement CPU of a link core holds
// a connection to its link, from its open to its kernel's end, and sends its share of
// DRAM pages over it, a count a batch; or a count is waited for.
#pragma once

#include <stdint.h>

#include <new>

#include "api/dataflow/dataflow_api.h"
#include "km_command.hpp"
#include "km_core.hpp"
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
    // A CPU's share of this many pages or more reads several a transaction (a block
    // of the banks); a smaller one a page a read (measured: example 01's one page a
    // line crossed a fifth slower four pages a read).
    constexpr uint32_t kPagesBulkShare = 32;
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
        uint32_t from_core;  // KM_PAGES_DRAM: from DRAM; else x | y << 16, from its L1
        uint32_t shape;  // KM_PAGES_SHAPE_*: 0, the loop's own
        uint32_t to_first;  // KM_PAGES_DRAM: the target's page for the share's first
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

    // What this CPU sent across, counted in its row of the link area at `area` (0: not
    // counted): `n` more of word `field`, `bytes` more bytes.
    inline void cross_count(uint32_t area, uint32_t field, uint32_t n, uint32_t bytes) {
        if (area == 0) {
            return;
        }
        volatile tt_l1_ptr uint32_t* row = (
            l1(area + KM_LINK_CROSS + (pages_held.link & 1u) * KM_CROSS_ROW)
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

    // This chip's NoC address of page `index` of a DRAM side at bank address
    // `address` (`page` bytes): interleaved (KM_PAGES_DRAM) or bank b's range.
    inline uint64_t pages_dram(
        uint32_t where, uint32_t address, uint32_t page, uint32_t index
    ) {
        if (where == KM_PAGES_DRAM) {
            const InterleavedAddrGen<true> tensor = {
                .bank_base_address = address, .page_size = page
            };
            return tensor.get_noc_addr(index);
        }
        return get_noc_addr_from_bank_id<true>(where & 0xFFu, address + index * page);
    }

    // The far NoC address of the target of the page `k` pages after the request's
    // first: page to_first + k of the DRAM side, or L1 at k pages after `to`.
    inline uint64_t pages_target(const PagesRequest& r, uint32_t k) {
        if (pages_in_dram(r.to_core)) {
            const uint64_t at = pages_dram(r.to_core, r.to, r.page, r.to_first + k);
            return (
                link_noc(
                    NOC_UNICAST_ADDR_X(at),
                    NOC_UNICAST_ADDR_Y(at),
                    static_cast<uint32_t>(at)
                )
            );
        }
        return link_noc(r.to_core & 0xFFu, r.to_core >> 8, r.to + k * r.page);
    }

    template <typename Kernel>
    uint32_t pages_forward(const Kernel& kernel, uint32_t area);  // below

    // CPU `cpu` sends `share` over its held connection, reads run ahead in the slots;
    // each batch's last packet adds its pages to the count. A range: `share` is ranks,
    // each rank's pages into its core's L1, each rank's last packet adding to its
    // count. A send's shape (KM_PAGES_SHAPE_*) may read several pages a transaction:
    // pages of one DRAM bank (p, p + banks, ..), or pages one after another in L1,
    // a group of slots a read. The lines link's CPU (link 0) sends its island's next
    // ask line, if one waits, after each read's packets. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_push(
        const Kernel& kernel, const PagesRequest& r, uint32_t cpu, Partition share
    ) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            return kGaveBadTransfer;  // the counters below are the dedicated mode's
        }
        const uint32_t area = r.buffer + cpu * r.chunk;
        const bool range = r.mode == KM_PAGES_RANGE;
        const bool dram = r.from_core == KM_PAGES_DRAM;  // interleaved: by banks
        const bool in_dram = pages_in_dram(r.from_core);
        const uint32_t shape = range ? 0 : r.shape & KM_PAGES_SHAPE_MEASURE;
        // the lines link's CPU: its island's lines first, between two reads
        const uint32_t lines = (
            pages_held.link == 0 && pages_held.links > 1
                ? islands_at(kernel.args().control).link_area()
                : 0
        );
        const bool no_read = (shape & KM_PAGES_SHAPE_NO_READ) != 0;
        const bool no_wait = no_read && (shape & KM_PAGES_SHAPE_NO_WAIT) != 0;
        const uint32_t slots_fit = (
            r.chunk > kPagesSlots && r.page > 0 ? (r.chunk - kPagesSlots) / r.page : 0
        );
        // pages a read and slots: the shape's, else the loop's own (measured: 16 pages
        // a read in three groups or more, else 4 in groups as many as fit)
        uint32_t per_read = 1u << 2 * (shape & 3u);
        uint32_t asked = (
            (shape >> 2 & 3u) != 0 ? kPagesSlotsDefault << ((shape >> 2 & 3u) - 1)
                                   : kPagesSlotsDefault
        );
        const uint32_t mine = range ? share.len * r.pages : share.len;
        if (shape == 0 && mine >= kPagesBulkShare) {
            per_read = slots_fit >= 3 * 16 ? 16 : slots_fit >= 2 * 4 ? 4 : 1;
            asked = per_read == 1 ? kPagesSlotsDefault : kPagesSlotsMax;
        }
        // from DRAM only where a bank's pages lie one after another; a range's pages
        // in order (each rank's last page counts the rank's)
        if (range || (dram && r.page % 64 != 0)) {
            per_read = 1;
            asked = asked < kPagesSlotsDefault ? asked : kPagesSlotsDefault;
        }
        uint32_t slots = slots_fit < asked ? slots_fit : asked;
        slots = slots < kPagesSlotsMax ? slots : kPagesSlotsMax;
        const uint32_t groups = slots / per_read;  // a read's slots each
        if (
            groups == 0 || r.page % 16 != 0 || r.page > KM_PAGES_PACKET_MAX ||
            area % 16 != 0 || r.chunk % 64 != 0 || r.hops == 0 ||
            static_cast<uint64_t>(r.buffer) + 2ull * r.chunk > MEM_L1_SIZE ||
            pages_held.open == 0 || pages_held.area != area
        ) {
            return kGaveBadTransfer;
        }
        // its pages: those of its ranks (a range), else its share of the pages
        const uint32_t base = range ? share.start * r.pages : share.start;
        const uint32_t total = range ? share.len * r.pages : share.len;
        if (total == 0) {
            return kGaveNone;
        }
        // read g's pages: from DRAM, per_read pages of one bank for each bank in a
        // block of banks * per_read pages, then the pages after the last whole block
        // one a read; from L1, per_read pages one after another
        const uint32_t banks = dram ? NUM_DRAM_BANKS : 1;
        const uint32_t block = banks * per_read;
        const uint32_t whole = per_read == 1 ? 0 : total / block * banks;
        const uint32_t reads = (
            per_read == 1 ? total
            : dram        ? whole + (total - whole * per_read)
                          : (total + per_read - 1) / per_read
        );
        struct Read {
            uint32_t k;  // its first page, after the share's first
            uint32_t n;  // pages
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
            const uint32_t xy = cores[r.to_core + q];
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
        // Read g goes into group g % groups and carries transaction id 1 + its group
        // (modulo the ids): reads from several banks land out of order, so a count of
        // responses does not say which group is full. More groups than ids share an
        // id, and a wait on it waits for both.
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
                in_dram ? pages_dram(r.from_core, r.from, r.page, index)
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
            for (uint32_t m = 0; m < one.n; m++) {
                const uint32_t k = one.k + m * one.stride;
                const uint32_t s = group * per_read + m;
                const uint32_t slot = area + kPagesSlots + s * r.page;
                const uint32_t p = r.first + base + k;
                const LinkHeaderAt header = (
                    reinterpret_cast<LinkHeaderAt>(
                        area + kPagesHeaders + s * kPagesHeaderStride
                    )
                );
                // a range: page j of rank q of the share
                const uint32_t q = range ? share.start + k / r.pages : 0;
                const uint32_t j = range ? k % r.pages : 0;
                const uint64_t at = (
                    range ? far(q, r.to + j * r.page) : pages_target(r, base + k)
                );
                if (range) {
                    counter = far(q, r.count);
                }
                batch++;
                if (
                    batch == kPagesBatch || sent + 1 == total ||
                    (range && j + 1 == r.pages)
                ) {
                    link_counted_write(
                        header, at, counter, static_cast<uint16_t>(batch), r.page
                    );
                    batch = 0;
                } else {
                    link_write(header, at, r.page);
                }
                if (!sender.space() && !wait([&] {
                    return sender.space();
                })) {
                    break;
                }
                if (!sender.send(header, slot, r.page)) {
                    code = kGaveLimit;
                    break;
                }
                sent++;
            }
            if (code != kGaveNone) {
                break;
            }
            // the group before again (this one's, with one group), once its packets
            // left L1
            const uint32_t mark = noc_nonposted_writes_num_issued[noc_index];
            const bool alone = groups == 1;
            if ((alone || g > 0) && ahead < reads) {
                const uint32_t left = alone ? mark : before;
                if (!no_wait && !wait([&] {
                    return noc_reached(NIU_MST_NONPOSTED_WR_REQ_SENT, left);
                })) {
                    break;
                }
                read(ahead++);
            }
            before = mark;
            if (lines != 0) {
                code = pages_forward(kernel, lines);
            }
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
    // link 0 carries the lines; with more links, pages go on links 1 on, and on link 0
    // too when the request says so (KM_PAGES_SHAPE_LINK0). kGave* code.
    template <typename Kernel>
    inline uint32_t pages_send(
        const Kernel& kernel, const PagesRequest& r, uint32_t rank, uint32_t cpu
    ) {
        const uint32_t cpus = pages_links(r);
        const uint32_t link = 2 * rank + cpu;
        const bool all = (r.shape & KM_PAGES_SHAPE_LINK0) != 0;
        const uint32_t first_link = cpus > 1 && !all ? 1 : 0;
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
        auto& sender = link_sender();
        sender.connect((r.links >> 8 * link) & 0xFFu, area);
        sender.open();
        pages_held = {1, area, r.chunk, r.hops, link, pages_links(r)};
        const uint32_t links_area = islands_at(kernel.args().control).link_area();
        const uint32_t row = links_area + KM_LINK_CROSS + (link & 1u) * KM_CROSS_ROW;
        forward_held = {
            island_area()[KM_ISLAND_FORWARD / 4],
            kernel.args().control,
            kernel.args().me,
            links_area != 0 ? row : 0,
            kernel.args().wait_limit,
            0,
            KM_FORWARD_NO_FIELD,
            0
        };
        const LinkHeaderAt header = link_header(area + kPagesHeaders, r.hops);
        const uint64_t far = (
            link_noc(r.counter & 0xFFu, r.counter >> 8, r.count + cpu * KM_PORT_STRIDE)
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
        const uint32_t sent = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            kernel.args().wait_limit
        );
        return code != kGaveNone ? code : sent;
    }

    // LINE (link core 0's link 0 CPU): the staged packet, then its count line, into
    // island a6's dispatcher (forwarded past the one across); bounded. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_line(
        const Kernel& kernel, const km_command_t& command
    ) {
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
        if (!wait([&] {  // the dispatcher's stores landed
            return (
                l1(at)[0] == number && l1(at + 64)[0] == number &&
                l1(at + KM_LINK_PACKET_BYTES)[1] == number
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
            return sender.send(made, from, n);
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
        }
        return code;
    }

    // COUNTS: CPU `cpu` of link core `rank` writes its row of what crossed into page
    // 2 rank + cpu of the DRAM tensor at r.to, then waits for the write. kGave* code.
    template <typename Kernel>
    inline uint32_t pages_counts(
        const Kernel& kernel, const PagesRequest& r, uint32_t rank, uint32_t cpu
    ) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            return kGaveBadTransfer;
        }
        const uint32_t area = islands_at(kernel.args().control).link_area();
        if (area == 0 || r.page < KM_CROSS_ROW || r.page % 64 != 0) {
            return kGaveBadTransfer;
        }
        const InterleavedAddrGen<true> to = {
            .bank_base_address = r.to, .page_size = r.page
        };
        noc_async_write(
            area + KM_LINK_CROSS + cpu * KM_CROSS_ROW,
            to.get_noc_addr(2 * rank + cpu),
            KM_CROSS_ROW
        );
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
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

    // The island's page CPUs (KM_PUT_CPUS in the link area at `area`, the session's):
    // those holding the last links, every link when no message service holds link 0,
    // else links 1 on (link 0 when it is the only link). Page CPU j holds link
    // first + j and serves share j of every ask and put (partition(pages, j, cpus)).
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
        return pages_held.links > cpus ? pages_held.links - cpus : 0;
    }
    inline bool page_cpu(uint32_t area) {
        return pages_held.link >= page_first(area);
    }
    inline uint32_t page_index(uint32_t area) {
        return pages_held.link - page_first(area);
    }

    // The page CPU that sends a put line across, confirms and counts the bad ones: the
    // one holding link 1 (link 0 when alone).
    inline uint32_t page_lead(uint32_t area) {
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
        return kernel.args().me - pages_held.link / 2;
    }

    // The far island's link core 0 (x | y << 16): the relay tile of this island's link
    // core 0 (each link core's is the far link core across from it), read from there
    // once a kernel by a page CPU on another link core (0 until then).
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

    // The line at `line` of link core 0's area as this CPU sees it: the line itself
    // on link core 0, else read across the NoC into `copy` (64 bytes, bounded). 0: the
    // read gave up (`code` says why).
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

    // The lines link's CPU (link 0): the next ask line its island's cores wrote into
    // the link area at `area`, once landed and while every far page CPU has room for
    // it, into the far link core 0's asks (its tile in the island area, the same area
    // address there); then its line is free again. One line a call. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_forward(
        const Kernel& kernel, uint32_t area
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        invalidate_l1_cache();
        const uint32_t taken = words[KM_ASK_TAKEN / 4];
        const uint32_t line = area + KM_ASK_LINES + taken % KM_ASK_SLOTS * 64;
        if (l1(line)[KM_ASK_NUMBER] != taken + 1 || l1(line)[KM_ASK_END] != taken + 1) {
            return kGaveNone;  // no line yet
        }
        for (uint32_t j = 0; j < page_cpus(area); j++) {
            if (taken - words[KM_ASK_SERVED / 4 + j] >= KM_ASK_SLOTS) {
                return kGaveNone;  // the far asks are full
            }
        }
        const uint32_t xy = island_area()[KM_ISLAND_RELAY_TILE];
        const LinkHeaderAt made = link_header(area + KM_ASK_HEADERS, pages_held.hops);
        link_write(
            made,
            link_noc(
                xy & 0xFFFFu, xy >> 16, area + KM_ASK_INBOX + taken % KM_ASK_SLOTS * 64
            ),
            64
        );
        auto& sender = link_sender();
        const uint32_t limit = kernel.args().wait_limit;
        uint32_t code = kernel.poll_at(
            kWherePages,
            [&] {
                return sender.space();
            },
            limit
        );
        if (code != kGaveNone) {
            return code;
        }
        if (!sender.send(made, line, 64)) {
            return kGaveLimit;
        }
        code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_writes_sent(noc_index);
            },
            limit
        );
        if (code == kGaveNone) {
            words[KM_ASK_TAKEN / 4] = taken + 1;
            cross_count(area, KM_CROSS_ASKS, 1, 64);
        }
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

    // An ask for pages an island further on holds: requeued at link core `core` toward
    // it (pages_onward reserves the line; `later`: not free yet). kGave* code.
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
                kernel.core_noc(core, area + KM_ASK_TAIL),
                NOC_UNICAST_WRITE_VC,
                1,
                31,
                false,
                false,
                area + KM_ASK_RETURNED
            );
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
        noc_async_read(
            kernel.core_noc(core, area + KM_ASK_TAKEN), area + KM_ASK_READ, 16
        );
        code = kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            limit
        );
        invalidate_l1_cache();
        if (code != kGaveNone || number - l1(area + KM_ASK_READ)[0] >= KM_ASK_SLOTS) {
            later = code == kGaveNone;
            return code;
        }
        volatile tt_l1_ptr uint32_t* own = l1(area + KM_ASK_OWN);
        for (uint32_t w = 0; w < 16; w++) {
            own[w] = line[w];
        }
        own[KM_ASK_NUMBER] = number + 1;
        own[KM_ASK_END] = number + 1;
        noc_async_write(
            area + KM_ASK_OWN,
            kernel.core_noc(core, area + KM_ASK_LINES + number % KM_ASK_SLOTS * 64),
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

    // Page CPU j (CPU `cpu` of its core): the far island's next ask (in link core 0's
    // link area at `area`), once landed: share j of its pages from this island's DRAM
    // into the asker's L1, each batch counted at the asker's port once landed
    // (pages_push), then this CPU's count of asks answered into the far link core 0,
    // which sends a line again once every page CPU answered the one before it. An ask
    // this island cannot answer is passed (counted bad by page CPU 0; the asker's
    // wait names this island). kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_answer(
        const Kernel& kernel, uint32_t cpu, uint32_t area
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        invalidate_l1_cache();
        const uint32_t own = area + KM_PAGE_CPU + cpu * KM_PAGE_CPU_BYTES;
        const uint32_t answered = words[KM_ASK_ANSWERED / 4 + cpu];
        uint32_t code = kGaveNone;
        const uint32_t at = pages_line_at(
            kernel,
            area + KM_ASK_INBOX + answered % KM_ASK_SLOTS * 64,
            own + KM_PAGE_CPU_ASK,
            code
        );
        if (at == 0) {
            return code;
        }
        const volatile tt_l1_ptr uint32_t* line = l1(at);
        if (line[KM_ASK_NUMBER] != answered + 1 || line[KM_ASK_END] != answered + 1) {
            return kGaveNone;
        }
        const uint32_t asker = line[KM_ASK_CORE];
        const uint32_t place = (asker & 0xFFu) | (asker >> 16 & 0xFFu) << 8;
        // into the asker's DRAM (KM_ASK_DRAM: 1 interleaved, 2 + b bank b)
        const uint32_t dram = line[KM_ASK_DRAM];
        const uint32_t into = dram == 1 ? KM_PAGES_DRAM : KM_PAGES_BANK | (dram - 2);
        const PagesRequest r = {
            line[KM_ASK_TENSOR],
            line[KM_ASK_PAGE],
            line[KM_ASK_FIRST],
            line[KM_ASK_PAGES],
            line[KM_ASK_TO],
            dram != 0 ? into : place,
            pages_held.chunk,
            line[KM_ASK_COUNT],
            place,
            pages_held.hops,
            KM_PAGES_SEND,
            0,
            pages_held.area - cpu * pages_held.chunk,
            KM_PAGES_DRAM,
            0,
            line[KM_ASK_DRAM_FIRST]
        };
        const Islands view = islands_at(kernel.args().control);
        const uint32_t holder = line[KM_ASK_HOLDER];
        const bool fits = (
            r.pages != 0 && r.page != 0 && r.page % 16 == 0 &&
            r.page <= KM_PAGES_PACKET_MAX && r.to % 16 == 0 && r.count % 16 == 0
        );
        // an island further on holds it: on by this island's link toward it (one
        // page CPU a link)
        const uint32_t toward = holder < view.count() ? view.link(holder) : 0xFFu;
        const bool onward = (
            holder != view.here() && fits && page_cpus(area) == 1 &&
            toward != KM_ISLAND_NO_LINK && toward != forward_index(view)
        );
        const bool good = holder == view.here() && fits;
        const uint32_t j = page_index(area);
        const Partition share = page_share(area, r.pages, j);
        if (onward) {
            bool later = false;
            const uint32_t core = view.base() + view.ranks() + toward;
            code = pages_requeue(kernel, line, area, core, later);
            if (later) {
                return kGaveNone;  // its line there is not free yet
            }
        } else if (good && share.len != 0) {
            forward_route(line[KM_ASK_FROM], KM_CROSS_PAGES);  // back to the asker
            code = pages_push(kernel, r, cpu, share);
            forward_route_off();
        }
        if (code != kGaveNone) {
            return code;
        }
        if (!good && !onward && j == page_lead(area)) {
            words[KM_ASK_BAD / 4] += 1;
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

    // Page CPU j (CPU `cpu` of its core): its island's next put line (in link core 0's
    // link area at `area`), once landed. Page CPU 0, while the far island has room for
    // it, sends the line into the far link core 0's puts; every page CPU then sends
    // share j of its pages from the asking core's L1 into the holder's DRAM
    // (pages_push), each batch's last packet adding its pages, once all it carried are
    // written, to the far link core 0's count of pages landed from page CPU j; then
    // its count of lines taken (on link core 0: the line is free again once every page
    // CPU took it). A put this island cannot send goes across with no pages (the
    // asker's wait names the island). kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_put(
        const Kernel& kernel, uint32_t cpu, uint32_t area
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        invalidate_l1_cache();
        const uint32_t j = page_index(area);
        const uint32_t own = area + KM_PAGE_CPU + cpu * KM_PAGE_CPU_BYTES;
        const uint32_t taken = words[KM_PUT_TAKEN / 4 + j];
        const uint32_t slot = area + KM_PUT_LINES + taken % KM_ASK_SLOTS * 64;
        const bool lead = j == page_lead(area);
        if (lead && taken - words[KM_PUT_DONE / 4] >= KM_ASK_SLOTS) {
            return kGaveNone;  // the far puts are full
        }
        uint32_t code = kGaveNone;
        const uint32_t at = pages_line_at(kernel, slot, own + KM_PAGE_CPU_PUT, code);
        if (at == 0) {
            return code;
        }
        volatile tt_l1_ptr uint32_t* line = l1(at);
        if (line[KM_ASK_NUMBER] != taken + 1 || line[KM_ASK_END] != taken + 1) {
            return kGaveNone;  // no line yet
        }
        const uint32_t home = link_core0(kernel);
        const uint32_t xy = pages_far0;  // the far link core 0
        const uint32_t place = (xy & 0xFFu) | (xy >> 16 & 0xFFu) << 8;
        // from the asker's DRAM (KM_ASK_DRAM: 1 interleaved, 2 + b bank b)
        const uint32_t dram = line[KM_ASK_DRAM];
        const uint32_t out = dram == 1 ? KM_PAGES_DRAM : KM_PAGES_BANK | (dram - 2);
        const PagesRequest r = {
            line[KM_ASK_TO],
            line[KM_ASK_PAGE],
            dram != 0 ? line[KM_ASK_DRAM_FIRST] : line[KM_ASK_FIRST],
            line[KM_ASK_PAGES],
            line[KM_ASK_TENSOR],
            KM_PAGES_DRAM,
            pages_held.chunk,
            area + KM_PUT_LANDED + 4 * j,
            place,
            pages_held.hops,
            KM_PAGES_SEND,
            0,
            pages_held.area - cpu * pages_held.chunk,
            dram != 0 ? out : line[KM_ASK_CORE],
            0,
            line[KM_ASK_FIRST]
        };
        const bool good = (
            line[KM_ASK_HOLDER] != islands_at(kernel.args().control).here() &&
            r.pages != 0 && r.page != 0 && r.page % 16 == 0 &&
            r.page <= KM_PAGES_PACKET_MAX && r.from % 16 == 0
        );
        if (lead) {  // on link core 0: the line is its own
            if (!good) {
                line[KM_ASK_PAGES] = 0;
                words[KM_PUT_BAD / 4] += 1;
            }
            auto* header = pages_header(area + KM_PUT_HEADERS);
            link_write(
                header,
                link_noc(
                    xy & 0xFFFFu,
                    xy >> 16,
                    area + KM_PUT_INBOX + taken % KM_ASK_SLOTS * 64
                ),
                64
            );
            code = pages_packet(kernel, header, at, 64);
            if (code == kGaveNone) {
                cross_count(area, KM_CROSS_PUTS, 1, 64);
            }
        }
        const Partition share = page_share(area, r.pages, j);
        if (code == kGaveNone && good && share.len != 0) {
            code = pages_push(kernel, r, cpu, share);
        }
        if (code != kGaveNone) {
            return code;
        }
        words[KM_PUT_TAKEN / 4 + j] = taken + 1;
        if (kernel.args().me != home) {  // its count where the asking cores read it
            noc_inline_dw_write(
                kernel.core_noc(home, area + KM_PUT_TAKEN + 4 * j), taken + 1
            );
            code = kernel.poll_at(
                kWherePages,
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                kernel.args().wait_limit
            );
        }
        return code;
    }

    // Page CPU 0: the far island's next put, once its line and every page CPU's share
    // of its pages have landed here (written): its pages onto the asker's port, then
    // the count of puts confirmed into the far link core 0, which may then send the
    // next. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t pages_confirm(
        const Kernel& kernel, uint32_t area
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        invalidate_l1_cache();
        const uint32_t confirmed = words[KM_PUT_CONFIRMED / 4];
        const volatile tt_l1_ptr uint32_t* line = (
            l1(area + KM_PUT_INBOX + confirmed % KM_ASK_SLOTS * 64)
        );
        const uint32_t pages = line[KM_ASK_PAGES];
        if (line[KM_ASK_NUMBER] != confirmed + 1 || line[KM_ASK_END] != confirmed + 1) {
            return kGaveNone;
        }
        const uint32_t cpus = page_cpus(area);
        for (uint32_t j = 0; j < cpus; j++) {
            const uint32_t share = page_share(area, pages, j).len;
            if (
                static_cast<int32_t>(
                    words[KM_PUT_LANDED / 4 + j] - words[KM_PUT_USED / 4 + j] - share
                ) < 0
            ) {
                return kGaveNone;  // a share not all written yet
            }
        }
        uint32_t code = kGaveNone;
        if (pages != 0) {
            const uint32_t asker = line[KM_ASK_CORE];
            auto* header = pages_header(area + KM_PUT_HEADERS + 64);
            link_increment(
                header,
                link_noc(asker & 0xFFFFu, asker >> 16, line[KM_ASK_COUNT]),
                pages
            );
            code = pages_packet(kernel, header, 0, 0);
        }
        if (code != kGaveNone) {
            return code;
        }
        const uint32_t xy = island_area()[KM_ISLAND_RELAY_TILE];  // the far link core
        auto* done = pages_header(area + KM_PUT_HEADERS + 128);
        link_inline(
            done, link_noc(xy & 0xFFFFu, xy >> 16, area + KM_PUT_DONE), confirmed + 1
        );
        code = pages_packet(kernel, done, 0, 0);
        if (code == kGaveNone) {
            for (uint32_t j = 0; j < cpus; j++) {
                words[KM_PUT_USED / 4 + j] += page_share(area, pages, j).len;
            }
            words[KM_PUT_CONFIRMED / 4] = confirmed + 1;
            cross_count(area, KM_CROSS_PUTS, 1, pages != 0 ? 8 : 4);
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
#if defined(KM_LINKS) && defined(COMPILE_FOR_BRISC)  // the reader: link 0's CPU
        } else if (r.mode == KM_PAGES_MESSAGES && links) {
            code = (
                cpu == 0 && pages_held.link == 0
                    ? link_messages(kernel, command)
                    : kGaveNone
            );
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
    // it): the lines link's CPU sends its island's next ask; each page CPU answers its
    // share of the far island's next, sends its share of its island's next put, and
    // page CPU 0 confirms the far island's next.
    // A failure stops the core (recorded in state word `word`).
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
            forward_turn();
            code = pages_forward(kernel, area);
        }
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
            if (code == kGaveNone && page_index(area) == page_lead(area)) {
                code = pages_confirm(kernel, area);
            }
        }
        live[kLvSequence + kernel.role()] = sequence;
        if (code != kGaveNone) {
            kernel.gave_up(word, code, kWherePages, 0, 0);
        }
    }

}  // namespace km
