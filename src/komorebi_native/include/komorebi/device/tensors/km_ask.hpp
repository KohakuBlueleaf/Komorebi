// A kernel's read and write of a tensor's pages wherever they live: this island's
// locally, others by a line into the link service's queue. Takes a km::Tensor.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_extent.hpp"
#include "../links/km_pages.hpp"
#include "km_placement.hpp"
#include "../movement/km_port.hpp"
#include "km_transfer.hpp"

namespace km {

    constexpr uint32_t kAskNoAnswer = 0xFFFFFFFFu;  // the atomic's answer not yet in

    // The link service's queue on link core 0: reserve count, counts taken (one a page
    // CPU), lines; the asking CPU's own line and answer words (KM_OWN_*); a put or not.
    struct AskQueue {
        uint32_t tail;
        uint32_t taken;
        uint32_t lines;
        uint32_t own;
        uint32_t returned;
        uint32_t read;
        bool cursors;
    };

    // The queue for the reader's gets, and for CPU `role`'s puts: one queue, one line
    // format (functions, not objects: an overlay keeps no copy of the runtime's data)
    constexpr AskQueue ask_gets() {
        return {
            KM_PUT_TAIL,
            KM_PUT_TAKEN,
            KM_PUT_LINES,
            KM_OWN_LINE,
            KM_OWN_RETURNED,
            KM_OWN_READ,
            false
        };
    }

    constexpr AskQueue ask_puts(uint32_t role) {
        const uint32_t own = role == kRoleWriter ? KM_OWN_CPU_BYTES : 0;
        return {
            KM_PUT_TAIL,
            KM_PUT_TAKEN,
            KM_PUT_LINES,
            own + KM_OWN_LINE,
            own + KM_OWN_RETURNED,
            own + KM_OWN_READ,
            true
        };
    }

    // The get queue of CPU `role`: the writer's own block after the reader's.
    constexpr AskQueue ask_gets(uint32_t role) {
        if (role != kRoleWriter) {
            return ask_gets();
        }
        constexpr uint32_t own = KM_OWN_CPU_BYTES;
        return {
            KM_PUT_TAIL,
            KM_PUT_TAKEN,
            KM_PUT_LINES,
            own + KM_OWN_LINE,
            own + KM_OWN_RETURNED,
            own + KM_OWN_READ,
            false
        };
    }

    // A tensor as the page calls take it: its accessor's compile arguments from `Base`
    // on (the host's tensor_args()) and its reference word (`tensor.address`).
    template <uint32_t Base>
    struct Tensor {
        static constexpr auto kArgs = TensorAccessorArgs<Base>();
        using Accessor = decltype(TensorAccessor(kArgs, 0u));
        // Built for one island (no tensor table): the word is this island's bank
        // address and every page is here.
#if defined(KM_TENSOR_TABLE)
        static constexpr bool kLocal = false;
#else
        static constexpr bool kLocal = true;
#endif

        Accessor accessor;
        uint32_t word;

        explicit Tensor(uint32_t word) :
            accessor(TensorAccessor(kArgs, word)), word(word) {
        }

        static constexpr uint32_t next_compile_time_args_offset() {
            return kArgs.next_compile_time_args_offset();
        }

        uint32_t page_bytes() const {
            return accessor.get_aligned_page_size();
        }
    };

    // Whether `T` is a Tensor (the page calls' tensor) or a plain accessor; the
    // accessor of either.
    template <typename T>
    constexpr bool kIsTensor = false;
    template <uint32_t Base>
    constexpr bool kIsTensor<Tensor<Base>> = true;

    // Whether some page of `T` may be another island's: a Tensor not known local.
    template <typename T>
    constexpr bool kMayCross = false;
    template <uint32_t Base>
    constexpr bool kMayCross<Tensor<Base>> = !Tensor<Base>::kLocal;

    template <typename T>
    inline const auto& accessor_of(const T& tensor) {
        if constexpr (kIsTensor<T>) {
            return tensor.accessor;
        } else {
            return tensor;
        }
    }

    // The port a core's page calls count at: its reader's or its writer's, past the
    // ports a kernel names, so no reduce, arrive or emit counts there too.
    template <typename Kernel>
    inline uint32_t pages_port(const Kernel& core) {
        return KM_PORT_PAGES + (core.role() == kRoleWriter ? 1 : 0);
    }

#if KM_GUARD && defined(KM_PAGE_LINKS)
    // A give-up that took line `number` of the link core toward island `holder` and
    // never wrote it: its queue waits for that line; named in the CPU's lost words.
    template <typename Kernel>
    inline void ask_abandoned(
        const Kernel& core, uint32_t sequence, uint32_t holder, uint32_t number
    ) {
        const uint32_t cpu = core.role() == kRoleWriter ? 1u : 0u;
        volatile tt_l1_ptr uint32_t* live = core.control(KM_CONTROL_LIVE);
        live[kLvLost + cpu] = sequence;
        live[kLvLostAt + cpu] = number;
        live[kLvLostPages] = (
            (live[kLvLostPages] & ~(0xFFFFu << 16 * cpu)) |
            (1u << 15 | (holder & 0xFFu) << 7) << 16 * cpu
        );
    }
#endif

    // One line into `queue`: pages first .. first + pages - 1 at `address` of `holder`,
    // into L1 `at` (or `dram`), at `port`, `keeper`, `form` (a batch). False: gave up.
    template <typename Kernel>
    inline bool ask_line_at(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const AskQueue& queue,
        uint32_t holder,
        uint32_t address,
        uint32_t bytes,
        uint32_t first,
        uint32_t pages,
        uint32_t at,
        uint32_t port,
        uint32_t dram = 0,
        uint32_t dram_first = 0,
        PutRun* run = nullptr,
        uint32_t keeper = 0,
        uint32_t form = 0
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t area = view.link_area();
        const uint32_t mine = view.own_words();
        const uint32_t named = slot | holder << 8;
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
#if KM_GUARD && defined(KM_PAGE_LINKS)
        uint32_t held = 0;  // 1 + the line number taken while its line is unwritten
#endif
        const auto give_up = [&](uint32_t code) {
#if KM_GUARD && defined(KM_PAGE_LINKS)
            if (held != 0) {
                ask_abandoned(core, sequence, holder, held - 1);
            }
#endif
            core.gave_up(word, code, kWhereAsk, named, sequence);
            return false;
        };
        const uint32_t hops = view.hops(holder);
#if defined(KM_FAR_PUTS)
        const bool far = queue.cursors && hops > 1 && run == nullptr;  // pages_put_far
#else
        const bool far = false;
#endif
        if (
            area == 0 || mine == 0 || hops == 0 ||
            (queue.cursors && hops != 1 && !far) || port >= KM_PORT_SLOTS ||
            bytes == 0 || bytes % 16 != 0 || bytes > view.packet() || at % 16 != 0
        ) {
            // another island (a put: a link apart, not forwarded), whole packets
#if KM_GUARD && defined(KM_PAGE_LINKS)
            return give_up(kGaveRefused);  // nothing held yet: this command alone
#else
            return give_up(kGaveBadItem);
#endif
        }
        // the link core toward the holder: the table's next link after the ranks
        const uint32_t link = view.base() + view.ranks() + view.link(holder);
        const uint32_t limit = core.args().wait_limit;
        // a put: this CPU's own line on the parity after its last, once the far island
        // confirmed the line last sent on it (its block of the core's own words)
        const uint32_t block = (
            mine + (core.role() == kRoleWriter ? KM_OWN_CPU_BYTES : 0)
        );
        volatile tt_l1_ptr uint32_t* own = l1(block);
        uint32_t owner = 0, parity = 0;
        if (queue.cursors) {
            const uint32_t index = core.args().me - view.base();
            if (index >= KM_CORES_MAX) {
                return give_up(kGaveBadItem);
            }
            invalidate_l1_cache();
            parity = own[KM_OWN_PUT_SENT / 4] & 1u;
            owner = (index * 2 + (core.role() == kRoleWriter ? 1 : 0)) * 2 + parity;
            const uint32_t code = core.poll_at(
                kWhereAskConfirm,
                [&] {
                    invalidate_l1_cache();
                    return (
                        own[KM_OWN_PUT_NUMBER / 4 + parity] ==
                        own[KM_OWN_PUT_CONFIRMED / 4 + parity]
                    );
                },
                limit
            );
            if (code != kGaveNone) {
                return give_up(code);
            }
        }
        volatile tt_l1_ptr uint32_t* returned = l1(mine + queue.returned);
        returned[0] = kAskNoAnswer;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            core.core_noc(link, area + queue.tail),
            NOC_UNICAST_WRITE_VC,
            1,
            31,
            false,
            false,
            mine + queue.returned
        );
        uint32_t code = core.poll_at(
            kWhereAskNumber,
            [&] {
                invalidate_l1_cache();
                return (
                    ncrisc_noc_nonposted_atomics_flushed(noc_index) &&
                    returned[0] != kAskNoAnswer
                );
            },
            limit
        );
        if (code != kGaveNone) {
            return give_up(code);
        }
        const uint32_t number = returned[0];
#if KM_GUARD && defined(KM_PAGE_LINKS)
        held = number + 1;
#endif
        // its place in the queue free: every page CPU took line KM_QUEUE_LINES before
        const uint32_t build = block + KM_OWN_BUILD;
        PutRings rings;
        put_rings_clear(rings);
        code = put_rings_read(core, link, area, build, rings, number);
        if (code != kGaveNone) {
            return give_up(code);
        }
        const bool ring = queue.cursors && dram == 0 && rings.slots != 0 && hops == 1;
        const uint32_t to_line = (
            rings.queue + KM_QUEUE_PUTS + number % KM_QUEUE_LINES * 64
        );
        if (run != nullptr && !ring) {
            return give_up(kGaveBadItem);  // a run's bytes follow only into rings
        }
        volatile tt_l1_ptr uint32_t* line = l1(mine + queue.own);
        for (uint32_t w = 0; w < 16; w++) {
            line[w] = 0;
        }
        line[KM_ASK_TENSOR] = address;
        line[KM_ASK_PAGE] = bytes;
        line[KM_ASK_FIRST] = first;
        line[KM_ASK_PAGES] = pages;
        line[KM_ASK_CORE] = core.control(KM_CONTROL_CORES)[core.args().me];
        line[KM_ASK_TO] = at;
        line[KM_ASK_COUNT] = (
            core.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE +
            KM_PORT_COUNT * 4
        );
        line[KM_ASK_HOLDER] = holder;
        line[KM_ASK_FROM] = view.here();
        line[KM_ASK_DRAM] = dram;
        line[KM_ASK_DRAM_FIRST] = dram_first;
        line[KM_ASK_RING] = (
            (queue.cursors ? KM_ASK_OP_WRITE : KM_ASK_OP_READ) << KM_ASK_OP_SHIFT |
            (ring ? 1 : form)
        );
        line[KM_ASK_OWNER] = owner;
        if (queue.cursors && dram != 0) {
            line[KM_ASK_KEEPER] = keeper;
        }
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        noc_async_write(mine + queue.own, core.core_noc(link, to_line), 64);
#if KM_GUARD && defined(KM_PAGE_LINKS)
        held = 0;
        if (
            dram == 0 && form == 0 && port == pages_port(core)
        ) {  // settled after a give-up
            port_words(core, port)[KM_PORT_FLIGHT] += pages;
        }
#endif
        code = core.poll_at(
            kWhereAskSent,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        if (queue.cursors) {  // its line on this parity, until the far island's word
            own[KM_OWN_PUT_SENT / 4] = own[KM_OWN_PUT_SENT / 4] + 1;
            own[KM_OWN_PUT_NUMBER / 4 + parity] = KM_OWN_CONFIRM(number, holder);
        }
        if (code == kGaveNone && ring) {
            PutRun local;
            PutRun& one = run != nullptr ? *run : local;  // run: put_piece sends
            put_run_clear(one);  // not an aggregate: its block clear calls memset
            one.rings = rings;
            one.holder = holder;
            one.area = area;
            one.build = build;
            one.answer = mine + queue.returned;
            one.number = number;
            one.hops = view.hops(holder);
            one.address = address;
            one.bytes = bytes;
            one.first = first;
            one.pages = pages;
            one.owner = owner;
            if (run == nullptr) {
                code = put_ring_pieces(core, one, at, pages);
            }
        }
        return code == kGaveNone || give_up(code);
    }

    // ask_line_at for `command`.
    template <typename Kernel>
    inline bool ask_line(
        const Kernel& core,
        const km_command_t& command,
        const AskQueue& queue,
        uint32_t holder,
        uint32_t address,
        uint32_t bytes,
        uint32_t first,
        uint32_t pages,
        uint32_t at,
        uint32_t port
    ) {
        return (
            ask_line_at(
                core,
                slot(command),
                command.sequence,
                queue,
                holder,
                address,
                bytes,
                first,
                pages,
                at,
                port
            )
        );
    }

    // Waits until port `port` has counted `wanted` pages asked for or put, then clears
    // it, for command (`slot`, `sequence`); a give-up names the island. False: gave up.
    template <typename Kernel>
    inline bool ask_wait(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        uint32_t port,
        uint32_t wanted,
        uint32_t holder
    ) {
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        if (port >= KM_PORT_SLOTS) {
            core.gave_up(word, kGaveBadItem, kWhereAsk, slot | holder << 8, sequence);
            return false;
        }
        volatile tt_l1_ptr uint32_t* words = port_words(core, port);
        words[KM_PORT_WANT] = wanted;
        const uint32_t code = core.poll_at(
            kWhereAskCount,
            [&] {
                return static_cast<int32_t>(words[KM_PORT_COUNT] - wanted) >= 0;
            },
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            core.gave_up(word, code, kWhereAsk, slot | holder << 8, sequence);
            return false;
        }
        words[KM_PORT_COUNT] = 0;
        words[KM_PORT_WANT] = 0;
#if KM_GUARD && defined(KM_PAGE_LINKS)
        if (port == pages_port(core)) {
            const uint32_t flight = words[KM_PORT_FLIGHT];
            words[KM_PORT_FLIGHT] = flight > wanted ? flight - wanted : 0;
        }
#endif
        return true;
    }

    // Reader: one get line (the far island pushes the pages into L1 `to`).
    inline bool ask_pages(
        const Reader& reader,
        const km_command_t& command,
        uint32_t holder,
        uint32_t address,
        uint32_t bytes,
        uint32_t first,
        uint32_t pages,
        uint32_t to
    ) {
        return (
            ask_line(
                reader,
                command,
                ask_gets(),
                holder,
                address,
                bytes,
                first,
                pages,
                to,
                pages_port(reader)
            )
        );
    }

    // Reader or writer: one put line, returned once sent (island `holder` writes pages
    // first .. at bank address `address`); L1 `from` kept until put_wait has counted.
    template <typename Kernel>
    inline bool put_pages(
        const Kernel& core,
        const km_command_t& command,
        uint32_t holder,
        uint32_t address,
        uint32_t bytes,
        uint32_t first,
        uint32_t pages,
        uint32_t from
    ) {
        return (
            ask_line(
                core,
                command,
                ask_puts(core.role()),
                holder,
                address,
                bytes,
                first,
                pages,
                from,
                pages_port(core)
            )
        );
    }

    // Names a run of puts (`pages` from `first`, bank `address` of `holder`) in one
    // line; put_piece sends the bytes, put_wait(run) counts. False: gave up.
    template <typename Kernel>
    inline bool put_pages(
        const Kernel& core,
        const km_command_t& command,
        uint32_t holder,
        uint32_t address,
        uint32_t bytes,
        uint32_t first,
        uint32_t pages,
        PutRun& run
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t area = view.link_area();
        put_run_clear(run);
        run.holder = holder;
        run.address = address;
        run.bytes = bytes;
        run.first = first;
        run.pages = pages;
        const uint32_t mine = view.own_words();
        const uint32_t hops = view.hops(holder);
#if defined(KM_FAR_PUTS)
        const bool far = hops > 1;  // a line a piece (put_piece), sent on by the lead
#else
        const bool far = false;
#endif
        if (area == 0 || mine == 0 || hops == 0 || (hops != 1 && !far)) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                kGaveBadItem,
                kWhereAsk,
                slot(command) | holder << 8,
                command.sequence
            );
            return false;
        }
        if (far) {
            return true;  // no rings: put_piece sends and waits for a line a piece
        }
        const AskQueue& queue = ask_puts(core.role());
        const uint32_t build =
            (  // its CPU's block of the own words
                mine + (core.role() == kRoleWriter ? KM_OWN_CPU_BYTES : 0) +
                KM_OWN_BUILD
            );
        const uint32_t link = view.base() + view.ranks() + view.link(holder);
        const uint32_t code = put_rings_read(core, link, area, build, run.rings);
        if (code != kGaveNone) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                code,
                kWhereAsk,
                slot(command) | holder << 8,
                command.sequence
            );
            return false;
        }
        if (run.rings.slots == 0) {
            return true;  // a line a piece
        }
        PutRun* named = &run;
        return (
            ask_line_at(
                core,
                slot(command),
                command.sequence,
                queue,
                holder,
                address,
                bytes,
                first,
                pages,
                0,
                pages_port(core),
                0,
                0,
                named
            )
        );
    }

    // Reader or writer: the next `count` pages of `run` (put_pages) from L1 `from`,
    // one after another; `from` free on return. False: gave up (recorded).
    template <typename Kernel>
    inline bool put_piece(
        const Kernel& core,
        const km_command_t& command,
        PutRun& run,
        uint32_t from,
        uint32_t count
    ) {
        if (run.rings.slots == 0) {  // no rings: its own line, waited for
            const uint32_t first = run.first + run.done;
            if (!ask_line(
                core,
                command,
                ask_puts(core.role()),
                run.holder,
                run.address,
                run.bytes,
                first,
                count,
                from,
                pages_port(core)
            ) ||
                !ask_wait(
                    core,
                    slot(command),
                    command.sequence,
                    pages_port(core),
                    count,
                    run.holder
                )) {
                return false;
            }
            run.done += count;
            run.waited += count;
            return true;
        }
        const uint32_t code = put_ring_pieces(core, run, from, count);
        if (code != kGaveNone) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                code,
                kWhereAsk,
                slot(command) | run.holder << 8,
                command.sequence
            );
            return false;
        }
        return true;
    }

    // Reader or writer: waits until its page port has counted the pages of `run` not
    // yet waited for (put_piece), clears it.
    template <typename Kernel>
    inline bool put_wait(const Kernel& core, const km_command_t& command, PutRun& run) {
        return (
            ask_wait(
                core,
                slot(command),
                command.sequence,
                pages_port(core),
                run.pages - run.waited,
                run.holder
            )
        );
    }

    // Reader or writer: waits until its page port has counted all `pages` of this
    // CPU's puts sent since the last wait (the last to island `holder`), clears it.
    template <typename Kernel>
    inline bool put_wait(
        const Kernel& core, const km_command_t& command, uint32_t pages, uint32_t holder
    ) {
        return (
            ask_wait(
                core, slot(command), command.sequence, pages_port(core), pages, holder
            )
        );
    }

    // Reader or writer: pages first .. of tensor `tensor` (its word; `args`: its
    // accessor's) into L1 `to`; another island's runs asked for. One card: a read.
    template <typename Kernel, typename Args>
    inline bool read_pages_at(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const Args& args,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t to
    ) {
        if (!pages_inside(core, slot, sequence, tensor, first, pages, args)) {
            return false;  // KM_GUARD
        }
        const uint32_t here = island();
        const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
        const uint32_t port = pages_port(core);
        invalidate_l1_cache();  // the table may have grown since the last read
        uint32_t asked = 0, holder = here;
        for (uint32_t k = 0; k < pages;) {
            const PageAt at = page_at(tensor, first + k);
            if (at.island == here) {
                noc_async_read_page(
                    at.index, TensorAccessor(args, at.address), to + k * bytes
                );
                k++;
                continue;
            }
            const uint32_t run = page_run(tensor, first + k, pages - k, at);
#if !defined(KM_ASKS_BY_HOST)
            if (!ask_line_at(
                core,
                slot,
                sequence,
                ask_gets(core.role()),
                at.island,
                at.address,
                bytes,
                at.index,
                run,
                to + k * bytes,
                port
            )) {
                return false;
            }
#endif
            asked += run;
            holder = at.island;
            k += run;
        }
        // asks by the host: the host's send brings them to `to` and counts them here
        return asked == 0 || ask_wait(core, slot, sequence, port, asked, holder);
    }

    // read_pages_at for `command`.
    template <typename Kernel, typename Args>
    inline bool read_pages(
        const Kernel& core,
        const km_command_t& command,
        const Args& args,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t to
    ) {
        return (
            read_pages_at(
                core, slot(command), command.sequence, args, tensor, first, pages, to
            )
        );
    }

    // Reader or writer: pages first .. of tensor `tensor` (its word) from L1 `from`;
    // each other island's run one put line. `from` free on return.
    template <typename Kernel, typename Args>
    inline bool write_pages_at(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const Args& args,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t from KM_SOURCE_LINE
    ) {
        if (!pages_inside(core, slot, sequence, tensor, first, pages, args)) {
            return false;  // KM_GUARD
        }
        const uint32_t here = island();
        const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
        if (!page_source_ok(core, from, bytes KM_LINE)) {
            return false;  // KM_GUARD
        }
        const uint32_t port = pages_port(core);
        invalidate_l1_cache();  // the table may have grown since the last read
        uint32_t put = 0, holder = here;
        for (uint32_t k = 0; k < pages;) {
            const PageAt at = page_at(tensor, first + k);
            if (at.island == here) {
                noc_async_write_page(
                    at.index, TensorAccessor(args, at.address), from + k * bytes
                );
                k++;
                continue;
            }
            const uint32_t run = page_run(tensor, first + k, pages - k, at);
            if (!ask_line_at(
                core,
                slot,
                sequence,
                ask_puts(core.role()),
                at.island,
                at.address,
                bytes,
                at.index,
                run,
                from + k * bytes,
                port
            )) {
                return false;
            }
            put += run;
            holder = at.island;
            k += run;
        }
        return put == 0 || ask_wait(core, slot, sequence, port, put, holder);
    }

    // write_pages_at for `command`.
    template <typename Kernel, typename Args>
    inline bool write_pages(
        const Kernel& core,
        const km_command_t& command,
        const Args& args,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t from KM_SOURCE_LINE
    ) {
        return (
            write_pages_at(
                core,
                slot(command),
                command.sequence,
                args,
                tensor,
                first,
                pages,
                from KM_LINE
            )
        );
    }

    // The read of a Tensor's pages: built for one island, the local reads alone;
    // else read_pages_at by its word.
    template <typename Kernel, uint32_t Base>
    inline bool read_pages_at(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const Tensor<Base>& tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t to
    ) {
        if constexpr (Tensor<Base>::kLocal) {
            if (!pages_inside(core, slot, sequence, tensor, first, pages)) {
                return false;  // KM_GUARD
            }
            const uint32_t bytes = tensor.page_bytes();
            for (uint32_t k = 0; k < pages; k++) {
                noc_async_read_page(first + k, tensor.accessor, to + k * bytes);
            }
            return true;
        } else {
            return (
                read_pages_at(
                    core, slot, sequence, Tensor<Base>::kArgs, tensor.word, first, pages, to
                )
            );
        }
    }

    // read_pages_at of a Tensor for `command`.
    template <typename Kernel, uint32_t Base>
    inline bool read_pages(
        const Kernel& core,
        const km_command_t& command,
        const Tensor<Base>& tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t to
    ) {
        return (
            read_pages_at(
                core, slot(command), command.sequence, tensor, first, pages, to
            )
        );
    }

    // A get line (ask_line_at's) of `bytes` at `offset` of page `index` (bank address
    // `address`, `page` bytes) on island `holder`, into L1 `into`. False: gave up.
    template <typename Kernel>
    inline bool ask_bytes_line(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        uint32_t holder,
        uint32_t address,
        uint32_t page,
        uint32_t index,
        uint32_t offset,
        uint32_t bytes,
        uint32_t into
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t area = view.link_area();
        const uint32_t mine = view.own_words();
        const AskQueue queue = ask_gets(core.role());
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        const auto give_up = [&](uint32_t code) {
            core.gave_up(word, code, kWhereAsk, slot | holder << 8, sequence);
            return false;
        };
        if (
            area == 0 || mine == 0 || view.hops(holder) != 1 || page == 0 ||
            page % 16 != 0 || page > view.packet() || into % 4 != 0
        ) {
            return give_up(kGaveBadItem);  // a link apart, a page a packet holds
        }
        const uint32_t link = view.base() + view.ranks() + view.link(holder);
        const uint32_t limit = core.args().wait_limit;
        volatile tt_l1_ptr uint32_t* returned = l1(mine + queue.returned);
        returned[0] = kAskNoAnswer;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            core.core_noc(link, area + queue.tail),
            NOC_UNICAST_WRITE_VC,
            1,
            31,
            false,
            false,
            mine + queue.returned
        );
        uint32_t code = core.poll_at(
            kWhereAskNumber,
            [&] {
                invalidate_l1_cache();
                return (
                    ncrisc_noc_nonposted_atomics_flushed(noc_index) &&
                    returned[0] != kAskNoAnswer
                );
            },
            limit
        );
        if (code != kGaveNone) {
            return give_up(code);
        }
        const uint32_t number = returned[0];
        // its place in the queue free: every page CPU took line KM_QUEUE_LINES before
        PutRings rings;
        put_rings_clear(rings);
        const uint32_t build = (
            mine + (core.role() == kRoleWriter ? KM_OWN_CPU_BYTES : 0) + KM_OWN_BUILD
        );
        code = put_rings_read(core, link, area, build, rings, number);
        if (code != kGaveNone) {
            return give_up(code);
        }
        volatile tt_l1_ptr uint32_t* line = l1(mine + queue.own);
        for (uint32_t w = 0; w < 16; w++) {
            line[w] = 0;
        }
        line[KM_ASK_TENSOR] = address;
        line[KM_ASK_PAGE] = page;
        line[KM_ASK_FIRST] = index;
        line[KM_ASK_PAGES] = 1;
        line[KM_ASK_CORE] = core.control(KM_CONTROL_CORES)[core.args().me];
        line[KM_ASK_TO] = into;
        line[KM_ASK_COUNT] = (
            core.args().control + KM_CONTROL_PORT + pages_port(core) * KM_PORT_STRIDE +
            KM_PORT_COUNT * 4
        );
        line[KM_ASK_HOLDER] = holder;
        line[KM_ASK_FROM] = view.here();
        line[KM_ASK_RING] = KM_ASK_OP_READ << KM_ASK_OP_SHIFT | KM_ASK_GET_BYTES;
        line[KM_ASK_GET_SPAN] = offset | bytes << 16;
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        noc_async_write(
            mine + queue.own,
            core.core_noc(
                link, rings.queue + KM_QUEUE_PUTS + number % KM_QUEUE_LINES * 64
            ),
            64
        );
        code = core.poll_at(
            kWhereAskSent,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        return code == kGaveNone || give_up(code);
    }

    // Whether `bytes` at `offset` of a page of `page_bytes` may land at L1 `into`: 4s
    // in the page, `into` the offset mod 64 (a DRAM read's rule). Else gives up.
    template <typename Kernel>
    inline bool bytes_fit(
        const Kernel& core,
        const km_command_t& command,
        uint32_t page_bytes,
        uint32_t offset,
        uint32_t bytes,
        uint32_t into
    ) {
        if (
            bytes == 0 || (offset | bytes) % 4 != 0 || offset >= page_bytes ||
            bytes > page_bytes - offset || (into ^ offset) % 64 != 0
        ) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                kGaveBadTransfer,
                kWhereAsk,
                slot(command),
                command.sequence
            );
            return false;
        }
        return true;
    }

    // `bytes` at `offset` of page `page` of `tensor` (its word, `args`) into L1 `into`,
    // not waited for: here a NoC read, else a get line, `asked` + 1. False: gave up.
    template <typename Kernel, typename Args>
    inline bool ask_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Args& args,
        uint32_t tensor,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t into,
        uint32_t& asked
    ) {
        const uint32_t page_bytes = TensorAccessor(args, 0).get_aligned_page_size();
        if (!bytes_fit(core, command, page_bytes, offset, bytes, into)) {
            return false;
        }
        invalidate_l1_cache();  // the table may have grown since the last read
        const PageAt at = page_at(tensor, page);
        if (at.island == island()) {
            noc_async_read(
                TensorAccessor(args, at.address).get_noc_addr(at.index, offset),
                into,
                bytes
            );
            return true;
        }
        if (!ask_bytes_line(
            core,
            slot(command),
            command.sequence,
            at.island,
            at.address,
            page_bytes,
            at.index,
            offset,
            bytes,
            into
        )) {
            return false;
        }
        asked++;
        return true;
    }

    // ask_bytes of a Tensor: built for one island, the local read alone.
    template <typename Kernel, uint32_t Base>
    inline bool ask_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Tensor<Base>& tensor,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t into,
        uint32_t& asked
    ) {
        if constexpr (Tensor<Base>::kLocal) {
            if (!bytes_fit(core, command, tensor.page_bytes(), offset, bytes, into)) {
                return false;
            }
            noc_async_read(tensor.accessor.get_noc_addr(page, offset), into, bytes);
            return true;
        } else {
            return (
                ask_bytes(
                    core,
                    command,
                    Tensor<Base>::kArgs,
                    tensor.word,
                    page,
                    offset,
                    bytes,
                    into,
                    asked
                )
            );
        }
    }

    // Waits until this CPU's ask_bytes ranges landed: its reads, and `asked` counted
    // at its page port (cleared; `asked` 0 again). False: gave up (recorded).
    template <typename Kernel>
    inline bool bytes_wait(
        const Kernel& core, const km_command_t& command, uint32_t& asked
    ) {
        const uint32_t code = core.poll_at(
            kWhereAskCount,
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                code,
                kWhereAsk,
                slot(command),
                command.sequence
            );
            return false;
        }
        const uint32_t wanted = asked;
        asked = 0;
        return (
            wanted == 0 ||
            ask_wait(
                core,
                slot(command),
                command.sequence,
                pages_port(core),
                wanted,
                KM_PAGES_NO_LINK
            )
        );
    }

    // Reader or writer: the `bytes` bytes at byte `offset` of page `page` of `tensor`
    // in L1 `into` on return (ask_bytes, then bytes_wait). False: gave up.
    template <typename Kernel, uint32_t Base>
    inline bool read_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Tensor<Base>& tensor,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t into
    ) {
        uint32_t asked = 0;
        return (
            ask_bytes(core, command, tensor, page, offset, bytes, into, asked) &&
            bytes_wait(core, command, asked)
        );
    }

    // read_bytes of tensor `tensor` by its word (`args`: its accessor's).
    template <typename Kernel, typename Args>
    inline bool read_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Args& args,
        uint32_t tensor,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t into
    ) {
        uint32_t asked = 0;
        return (
            ask_bytes(core, command, args, tensor, page, offset, bytes, into, asked) &&
            bytes_wait(core, command, asked)
        );
    }

    // The write of a Tensor's pages: built for one island, the local writes alone;
    // else write_pages_at by its word.
    template <typename Kernel, uint32_t Base>
    inline bool write_pages_at(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const Tensor<Base>& tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t from KM_SOURCE_LINE
    ) {
        if constexpr (Tensor<Base>::kLocal) {
            if (!pages_inside(core, slot, sequence, tensor, first, pages)) {
                return false;  // KM_GUARD
            }
            const uint32_t bytes = tensor.page_bytes();
            if (!page_source_ok(core, from, bytes KM_LINE)) {
                return false;  // KM_GUARD
            }
            for (uint32_t k = 0; k < pages; k++) {
                noc_async_write_page(first + k, tensor.accessor, from + k * bytes);
            }
            return true;
        } else {
            return (
                write_pages_at(
                    core,
                    slot,
                    sequence,
                    Tensor<Base>::kArgs,
                    tensor.word,
                    first,
                    pages,
                    from KM_LINE
                )
            );
        }
    }

    // write_pages_at of a Tensor for `command`.
    template <typename Kernel, uint32_t Base>
    inline bool write_pages(
        const Kernel& core,
        const km_command_t& command,
        const Tensor<Base>& tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t from KM_SOURCE_LINE
    ) {
        return (
            write_pages_at(
                core, slot(command), command.sequence, tensor, first, pages, from KM_LINE
            )
        );
    }

    // This CPU's batch at L1 `batch` sent after its last was confirmed: a line and the
    // batch through the rings, or a far line past the next island. False: gave up.
    template <typename Kernel>
    inline bool put_batch_send(
        const Kernel& core, uint32_t slot, uint32_t sequence, uint32_t batch
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t area = view.link_area();
        const uint32_t mine = view.own_words();
        const AskQueue queue = ask_puts(core.role());
        volatile tt_l1_ptr uint32_t* head = l1(batch);
        const uint32_t holder = head[KM_BATCH_HOLDER];
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        const auto give_up = [&](uint32_t code) {
            core.gave_up(word, code, kWhereAsk, slot | holder << 8, sequence);
            return false;
        };
        const uint32_t me = core.args().me - view.base();
        const uint32_t hops = view.hops(holder);
#if defined(KM_FAR_PUTS)
        const bool far = hops > 1;  // pages_put_far: onto its page for this island
#else
        const bool far = false;
#endif
        if (
            area == 0 || mine == 0 || hops == 0 || (hops != 1 && !far) ||
            me >= KM_CORES_MAX || KM_PUT_BATCH_BYTES > view.packet()
        ) {
            return give_up(kGaveBadItem);  // another island, a batch a packet holds
        }
        const uint32_t link = view.base() + view.ranks() + view.link(holder);
        const uint32_t limit = core.args().wait_limit;
        const uint32_t block = (
            mine + (core.role() == kRoleWriter ? KM_OWN_CPU_BYTES : 0)
        );
        const uint32_t build = block + KM_OWN_BUILD;
        // the rings and the landing (read with them), before a line is reserved
        PutRings rings;
        put_rings_clear(rings);
        uint32_t code = put_rings_read(core, link, area, build, rings);
        if (code != kGaveNone) {
            return give_up(code);
        }
        const uint32_t landing = l1(build)[(KM_PUT_LANDING - KM_PUT_TAKEN) / 4];
        if (rings.slots == 0 || landing == 0) {
            return give_up(kGaveBadItem);  // a batch follows only into rings
        }
        volatile tt_l1_ptr uint32_t* own = l1(block);
        code = core.poll_at(
            kWhereAskConfirm,
            [&] {
                invalidate_l1_cache();
                return (
                    own[KM_OWN_PUT_NUMBER / 4] == own[KM_OWN_PUT_CONFIRMED / 4] &&
                    own[KM_OWN_PUT_NUMBER / 4 + 1] == own[KM_OWN_PUT_CONFIRMED / 4 + 1]
                );
            },
            limit
        );
        if (code != kGaveNone) {
            return give_up(code);
        }
        if (far) {  // the lead reads the batch from here: kept until confirmed
            const bool sent = (
                ask_line_at(
                    core,
                    slot,
                    sequence,
                    queue,
                    holder,
                    landing,
                    KM_PUT_BATCH_BYTES,
                    0,
                    1,
                    batch,
                    pages_port(core),
                    0,
                    0,
                    nullptr,
                    0,
                    KM_ASK_RING_BATCH
                )
            );
            if (!sent) {
                return false;
            }
            code = core.poll_at(
                kWhereAskConfirm,
                [&] {
                    invalidate_l1_cache();
                    return (
                        own[KM_OWN_PUT_NUMBER / 4] == own[KM_OWN_PUT_CONFIRMED / 4] &&
                        own[KM_OWN_PUT_NUMBER / 4 + 1] ==
                            own[KM_OWN_PUT_CONFIRMED / 4 + 1]
                    );
                },
                limit
            );
            head[KM_BATCH_USED] = KM_BATCH_HEAD;
            head[KM_BATCH_RECORDS] = 0;
            return code == kGaveNone || give_up(code);
        }
        const uint32_t parity = own[KM_OWN_PUT_SENT / 4] & 1u;
        const uint32_t owner = (
            (me * 2 + (core.role() == kRoleWriter ? 1 : 0)) * 2 + parity
        );
        volatile tt_l1_ptr uint32_t* returned = l1(mine + queue.returned);
        returned[0] = kAskNoAnswer;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            core.core_noc(link, area + queue.tail),
            NOC_UNICAST_WRITE_VC,
            1,
            31,
            false,
            false,
            mine + queue.returned
        );
        code = core.poll_at(
            kWhereAskNumber,
            [&] {
                invalidate_l1_cache();
                return (
                    ncrisc_noc_nonposted_atomics_flushed(noc_index) &&
                    returned[0] != kAskNoAnswer
                );
            },
            limit
        );
        if (code != kGaveNone) {
            return give_up(code);
        }
        const uint32_t number = returned[0];
        code = put_rings_read(core, link, area, build, rings, number);
        if (code != kGaveNone) {
            return give_up(code);
        }
        const uint32_t first = batch_page(view.count(), holder, view.here(), owner);
        volatile tt_l1_ptr uint32_t* line = l1(mine + queue.own);
        for (uint32_t w = 0; w < 16; w++) {
            line[w] = 0;
        }
        line[KM_ASK_TENSOR] = landing;
        line[KM_ASK_PAGE] = KM_PUT_BATCH_BYTES;
        line[KM_ASK_FIRST] = first;
        line[KM_ASK_PAGES] = 1;
        line[KM_ASK_CORE] = core.control(KM_CONTROL_CORES)[core.args().me];
        line[KM_ASK_COUNT] = (
            core.args().control + KM_CONTROL_PORT + pages_port(core) * KM_PORT_STRIDE +
            KM_PORT_COUNT * 4
        );
        line[KM_ASK_HOLDER] = holder;
        line[KM_ASK_FROM] = view.here();
        line[KM_ASK_RING] = KM_ASK_OP_WRITE << KM_ASK_OP_SHIFT | KM_ASK_RING_BATCH;
        line[KM_ASK_OWNER] = owner;
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        const uint32_t to_line = (
            rings.queue + KM_QUEUE_PUTS + number % KM_QUEUE_LINES * 64
        );
        noc_async_write(mine + queue.own, core.core_noc(link, to_line), 64);
        code = core.poll_at(
            kWhereAskSent,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        own[KM_OWN_PUT_SENT / 4] = own[KM_OWN_PUT_SENT / 4] + 1;
        own[KM_OWN_PUT_NUMBER / 4 + parity] = KM_OWN_CONFIRM(number, holder);
        if (code == kGaveNone) {
            PutRun run;
            put_run_clear(run);
            run.rings = rings;
            run.holder = holder;
            run.area = area;
            run.build = build;
            run.answer = mine + queue.returned;
            run.number = number;
            run.hops = 1;
            run.address = landing;
            run.bytes = KM_PUT_BATCH_BYTES;
            run.first = first;
            run.pages = 1;
            run.owner = owner;
            code = put_ring_pieces(core, run, batch, 1);
        }
        head[KM_BATCH_USED] = KM_BATCH_HEAD;
        head[KM_BATCH_RECORDS] = 0;
        return code == kGaveNone || give_up(code);
    }

    // Bytes `offset` on of page `page` become L1 `source`'s, unwaited (put_wait): here
    // a NoC write (`source` kept), else a record in this CPU's batch. False: gave up.
    template <typename Kernel, uint32_t Base>
    inline bool put_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Tensor<Base>& tensor,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t source
    ) {
        // multiples of 4 inside the page, `source` with the offset's low 4 bits
        const uint32_t page_bytes = tensor.page_bytes();
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        if (
            bytes == 0 || (offset | bytes) % 4 != 0 || offset >= page_bytes ||
            bytes > page_bytes - offset || (source ^ offset) % 16 != 0
        ) {
            core.gave_up(
                word, kGaveBadTransfer, kWhereAsk, slot(command), command.sequence
            );
            return false;
        }
        if constexpr (Tensor<Base>::kLocal) {
            noc_async_write(source, tensor.accessor.get_noc_addr(page, offset), bytes);
            return true;
        } else {
            const PageAt at = page_at(tensor.word, page);
            if (at.island == island()) {
                noc_async_write(
                    source,
                    TensorAccessor(Tensor<Base>::kArgs, at.address)
                        .get_noc_addr(at.index, offset),
                    bytes
                );
                return true;
            }
            const uint32_t own = islands_at(core.args().control).own_words();
            const uint32_t batch = put_batch(own, core.role() == kRoleWriter ? 1 : 0);
            if (batch == 0 || offset + bytes > (1u << KM_BATCH_SPAN_SHIFT)) {
                core.gave_up(
                    word,
                    kGaveBadItem,
                    kWhereAsk,
                    slot(command) | at.island << 8,
                    command.sequence
                );
                return false;
            }
            volatile tt_l1_ptr uint32_t* head = l1(batch);
            if (  // an earlier command's, or none yet
                head[KM_BATCH_SEQUENCE] != command.sequence ||
                head[KM_BATCH_USED] < KM_BATCH_HEAD
            ) {
                head[KM_BATCH_USED] = KM_BATCH_HEAD;
                head[KM_BATCH_RECORDS] = 0;
                head[KM_BATCH_SEQUENCE] = command.sequence;
            }
            // a record's bytes at most: a batch's, less its header, the record's and
            // the offset's low 4 bits
            constexpr uint32_t most = KM_PUT_BATCH_BYTES - 3 * KM_BATCH_HEAD;
            for (uint32_t done = 0; done < bytes;) {
                const uint32_t to = offset + done;
                const uint32_t n = bytes - done < most ? bytes - done : most;
                const uint32_t size = KM_BATCH_HEAD + (to % 16 + n + 15) / 16 * 16;
                if (
                    head[KM_BATCH_RECORDS] != 0 &&
                    (
                        head[KM_BATCH_HOLDER] != at.island ||
                        head[KM_BATCH_USED] + size > KM_PUT_BATCH_BYTES
                    ) &&
                    !put_batch_send(core, slot(command), command.sequence, batch)
                ) {
                    return false;
                }
                head[KM_BATCH_HOLDER] = at.island;
                const uint32_t record = batch + head[KM_BATCH_USED];
                l1(record)[0] = at.address;
                l1(record)[1] = at.index;
                l1(record)[2] = page_bytes;
                l1(record)[3] = to | n << KM_BATCH_SPAN_SHIFT;
                volatile tt_l1_ptr uint32_t* into = (
                    l1(record + KM_BATCH_HEAD + to % 16)
                );
                const volatile tt_l1_ptr uint32_t* from = l1(source + done);
                for (uint32_t w = 0; w < n / 4; w++) {
                    into[w] = from[w];
                }
                head[KM_BATCH_USED] = head[KM_BATCH_USED] + size;
                head[KM_BATCH_RECORDS] = head[KM_BATCH_RECORDS] + 1;
                done += n;
            }
            return true;
        }
    }

    // This CPU's batch sent, then waits until its batches were confirmed (written
    // there) and its writes here are written. False: gave up.
    template <typename Kernel>
    inline bool put_wait(const Kernel& core, const km_command_t& command) {
        const uint32_t mine = islands_at(core.args().control).own_words();
#if defined(KM_TENSOR_TABLE)  // else every page is here: no batch
        const uint32_t batch = put_batch(mine, core.role() == kRoleWriter ? 1 : 0);
        if (
            batch != 0 && l1(batch)[KM_BATCH_SEQUENCE] == command.sequence &&
            l1(batch)[KM_BATCH_RECORDS] != 0 &&
            !put_batch_send(core, slot(command), command.sequence, batch)
        ) {
            return false;
        }
#endif
        const volatile tt_l1_ptr uint32_t* own = (
            l1(mine + (core.role() == kRoleWriter ? KM_OWN_CPU_BYTES : 0))
        );
        const uint32_t code = core.poll_at(
            kWhereAskConfirm,
            [&] {
                invalidate_l1_cache();
                return (
                    ncrisc_noc_nonposted_writes_flushed(noc_index) &&
                    (
                        mine == 0 ||
                        (
                            own[KM_OWN_PUT_NUMBER / 4] ==
                                own[KM_OWN_PUT_CONFIRMED / 4] &&
                            own[KM_OWN_PUT_NUMBER / 4 + 1] ==
                                own[KM_OWN_PUT_CONFIRMED / 4 + 1]
                        )
                    )
                );
            },
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                code,
                kWhereAsk,
                slot(command) | island() << 8,
                command.sequence
            );
            return false;
        }
        return true;
    }

    // put_bytes, then put_wait: the bytes written on return.
    template <typename Kernel, uint32_t Base>
    inline bool write_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Tensor<Base>& tensor,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t source
    ) {
        return (
            put_bytes(core, command, tensor, page, offset, bytes, source) &&
            put_wait(core, command)
        );
    }

    // Whether a copy's side names, by its reference word, a tensor of which another
    // island holds pages (its pages then go page by page by their holder).
    inline bool copy_placed(const km_copy_side_t& side) {
        return side.bank == KM_COPY_INTERLEAVED && held_elsewhere(side.address);
    }

    // The copy of two tensors no other island holds pages of, by their bank addresses
    // here; a side placed over islands keeps its word (copy_across).
    inline km_copy_t copy_resolved(km_copy_t c) {
        invalidate_l1_cache();  // the table may have grown since the last read
        const auto resolve = [](km_copy_side_t& side) {
            if (
                side.bank == KM_COPY_INTERLEAVED && side.address < KM_TENSORS_MAX &&
                !held_elsewhere(side.address)
            ) {
                side.address = page_at(side.address, side.first).address;
            }
        };
        resolve(c.from);
        resolve(c.to);
        return c;
    }

    // Whether a copy has a side with pages another island holds.
    inline bool copy_crosses(const km_copy_t& c) {
        invalidate_l1_cache();  // the table may have grown since the last read
        return copy_placed(c.from) || copy_placed(c.to);
    }

    // Reader: rank `rank` of `count` copies by chunks; each run a placed side holds on
    // another island is one get or put line, waited per chunk. False: gave up.
    inline bool copy_across(
        const Reader& reader,
        const km_command_t& command,
        const km_copy_t& c,
        uint32_t rank,
        uint32_t count
    ) {
        uint32_t code = kGaveNone;
        const auto give_up = [&](uint32_t why) {
            reader.gave_up(kStReader, why, kWhereCopy, slot(command), command.sequence);
            return false;
        };
        if (rank >= count || !km_copy_fits(&c, count, MEM_L1_SIZE, NUM_DRAM_BANKS)) {
            return give_up(kGaveBadTransfer);
        }
        const auto wait = [&](bool (*ready)()) {
            code = reader.poll_at(kWhereCopy, ready, reader.args().wait_limit);
            return code == kGaveNone;
        };
        const auto read_done = [] {
            return ncrisc_noc_reads_flushed(noc_index);
        };
        const auto write_done = [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        };
        // the lines' pages on the port; a miss names the island
        const auto copy_wait = [&](uint32_t holder, uint32_t pages) {
            return (
                ask_wait(
                    reader,
                    slot(command),
                    command.sequence,
                    pages_port(reader),
                    pages,
                    holder
                )
            );
        };
        const uint32_t here = island();
        const bool from = copy_placed(c.from);
        const bool to = copy_placed(c.to);
        const Partition share = partition(c.pages, rank, count);
        const uint32_t chunk_pages = c.chunk / c.page;
        if (from != to) {
            // the placed side's runs: elsewhere one line each, DRAM to DRAM; here
            // through the chunk
            const km_copy_side_t& placed = from ? c.from : c.to;
            const km_copy_side_t& mine = from ? c.to : c.from;
            const AskQueue queue = from ? ask_gets() : ask_puts(kRoleReader);
            uint32_t lined = 0, holder = here;
            for (uint32_t k = 0; k < share.len;) {
                const uint32_t page = placed.first + share.start + k;
                const PageAt where = page_at(placed.address, page);
                const uint32_t run = (
                    page_run(placed.address, page, share.len - k, where)
                );
                if (where.island != here) {
                    if (!ask_line_at(
                        reader,
                        slot(command),
                        command.sequence,
                        queue,
                        where.island,
                        where.address,
                        c.page,
                        where.index,
                        run,
                        mine.address,
                        pages_port(reader),
                        mine.bank == KM_COPY_INTERLEAVED ? 1 : 2 + mine.bank,
                        mine.first + share.start + k
                    )) {
                        return false;
                    }
                    lined += run;
                    holder = where.island;
                    k += run;
                    continue;
                }
                const uint32_t n = run < chunk_pages ? run : chunk_pages;
                km_copy_side_t local = placed;
                local.address = where.address;
                for (uint32_t m = 0; m < n; m++) {
                    const uint64_t at = (
                        from
                            ? copy_noc(local, c.page, where.index + m)
                            : copy_noc(mine, c.page, mine.first + share.start + k + m)
                    );
                    noc_async_read(at, c.buffer + m * c.page, c.page);
                }
                if (!wait(read_done)) {
                    return give_up(code);
                }
                for (uint32_t m = 0; m < n; m++) {
                    const uint64_t at = (
                        from
                            ? copy_noc(mine, c.page, mine.first + share.start + k + m)
                            : copy_noc(local, c.page, where.index + m)
                    );
                    noc_async_write(c.buffer + m * c.page, at, c.page);
                }
                if (!wait(write_done)) {
                    return give_up(code);
                }
                k += n;
            }
            return lined == 0 || copy_wait(holder, lined);
        }
        // One side of `n` pages from `start`: local ones by `local(noc, k)`, runs held
        // elsewhere one line each from L1 `at`. Returns pages done, last run's island.
        const auto side = [&](
            const km_copy_side_t& s,
            bool placed,
            const AskQueue& queue,
            uint32_t start,
            uint32_t n,
            uint32_t at,
            auto local,
            uint32_t& holder
        ) -> uint32_t {
            uint32_t lined = 0;
            for (uint32_t k = 0; k < n;) {
                const uint32_t page = s.first + start + k;
                const PageAt where = placed ? page_at(s.address, page)
                                            : PageAt{here, s.address, page};
                if (where.island == here) {
                    km_copy_side_t mine = s;
                    mine.address = where.address;
                    local(copy_noc(mine, c.page, where.index), k);
                    k++;
                    continue;
                }
                const uint32_t run = page_run(s.address, page, n - k, where);
                if (!ask_line(
                    reader,
                    command,
                    queue,
                    where.island,
                    where.address,
                    c.page,
                    where.index,
                    run,
                    at + k * c.page,
                    pages_port(reader)
                )) {
                    return 0xFFFFFFFFu;
                }
                lined += run;
                holder = where.island;
                k += run;
            }
            return lined;
        };
        for (uint32_t done = 0; done < share.len; done += chunk_pages) {
            const uint32_t n = (
                share.len - done < chunk_pages ? share.len - done : chunk_pages
            );
            const uint32_t start = share.start + done;
            uint32_t holder = here;
            const uint32_t got = side(
                c.from,
                from,
                ask_gets(),
                start,
                n,
                c.buffer,
                [&](uint64_t noc, uint32_t k) {
                    noc_async_read(noc, c.buffer + k * c.page, c.page);
                },
                holder
            );
            if (got == 0xFFFFFFFFu) {
                return false;
            }
            if (!wait(read_done)) {
                return give_up(code);
            }
            if (got != 0 && !copy_wait(holder, got)) {
                return false;
            }
            const uint32_t put = side(
                c.to,
                to,
                ask_puts(kRoleReader),
                start,
                n,
                c.buffer,
                [&](uint64_t noc, uint32_t k) {
                    noc_async_write(c.buffer + k * c.page, noc, c.page);
                },
                holder
            );
            if (put == 0xFFFFFFFFu) {
                return false;
            }
            if (!wait(write_done)) {
                return give_up(code);
            }
            if (put != 0 && !copy_wait(holder, put)) {
                return false;
            }
        }
        return true;
    }

    // Records in this CPU's batch for island `holder`: bytes `offset` on of page
    // `index` (`page` bytes) at bank address `address`, L1 `source`'s. False: gave up.
    template <typename Kernel>
    inline bool put_records(
        const Kernel& core,
        const km_command_t& command,
        uint32_t holder,
        uint32_t address,
        uint32_t index,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t source
    ) {
        const uint32_t own = islands_at(core.args().control).own_words();
        const uint32_t batch = put_batch(own, core.role() == kRoleWriter ? 1 : 0);
        if (batch == 0 || offset + bytes > (1u << KM_BATCH_SPAN_SHIFT)) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                kGaveBadItem,
                kWhereAsk,
                slot(command) | holder << 8,
                command.sequence
            );
            return false;
        }
        volatile tt_l1_ptr uint32_t* head = l1(batch);
        if (  // an earlier command's, or none yet
            head[KM_BATCH_SEQUENCE] != command.sequence ||
            head[KM_BATCH_USED] < KM_BATCH_HEAD
        ) {
            head[KM_BATCH_USED] = KM_BATCH_HEAD;
            head[KM_BATCH_RECORDS] = 0;
            head[KM_BATCH_SEQUENCE] = command.sequence;
        }
        // a record's bytes at most: a batch's, less its header, the record's and the
        // offset's low 4 bits
        constexpr uint32_t most = KM_PUT_BATCH_BYTES - 3 * KM_BATCH_HEAD;
        for (uint32_t done = 0; done < bytes;) {
            const uint32_t to = offset + done;
            const uint32_t n = bytes - done < most ? bytes - done : most;
            const uint32_t size = KM_BATCH_HEAD + (to % 16 + n + 15) / 16 * 16;
            if (
                head[KM_BATCH_RECORDS] != 0 &&
                (
                    head[KM_BATCH_HOLDER] != holder ||
                    head[KM_BATCH_USED] + size > KM_PUT_BATCH_BYTES
                ) &&
                !put_batch_send(core, slot(command), command.sequence, batch)
            ) {
                return false;
            }
            head[KM_BATCH_HOLDER] = holder;
            const uint32_t record = batch + head[KM_BATCH_USED];
            l1(record)[0] = address;
            l1(record)[1] = index;
            l1(record)[2] = page;
            l1(record)[3] = to | n << KM_BATCH_SPAN_SHIFT;
            volatile tt_l1_ptr uint32_t* into = l1(record + KM_BATCH_HEAD + to % 16);
            const volatile tt_l1_ptr uint32_t* from = l1(source + done);
            for (uint32_t w = 0; w < n / 4; w++) {
                into[w] = from[w];
            }
            head[KM_BATCH_USED] = head[KM_BATCH_USED] + size;
            head[KM_BATCH_RECORDS] = head[KM_BATCH_RECORDS] + 1;
            done += n;
        }
        return true;
    }

    // Built for one island (no tensor table): SizedTensor's and BankRange's words are
    // bank addresses and every page is here.
#if defined(KM_TENSOR_TABLE)
    constexpr bool kDramLocal = false;
#else
    constexpr bool kDramLocal = true;
#endif

    // An interleaved DRAM tensor (word `word`: its reference word) read and written in
    // pages of `page_bytes` (64s), as long as its entry's bytes in each bank allow.
    struct SizedTensor {
        using DramForm = void;
        uint32_t word;
        uint32_t bytes;

        SizedTensor(uint32_t word, uint32_t page_bytes) :
            word(word), bytes(page_bytes) {
        }

        uint32_t page_bytes() const {
            return bytes;
        }

        // Built for one island: page `page` plus `offset`, as a kernel names it.
        uint64_t local(uint32_t page, uint32_t offset) const {
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = word, .page_size = bytes
            };
            return pages.get_noc_addr(page, offset);
        }

        DramRun place(uint32_t page, uint32_t most) const {
            return sized_page_at(word, bytes, page, most);
        }
    };

    // A range of DRAM bank `bank` (word `word`: its entry, or a bank address there)
    // in pages of `page_bytes` (64s); an entry's bank and bank address are its own.
    struct BankRange {
        using DramForm = void;
        uint32_t word;
        uint32_t bank;
        uint32_t bytes;

        BankRange(uint32_t word, uint32_t bank, uint32_t page_bytes) :
            word(word), bank(bank), bytes(page_bytes) {
        }

        uint32_t page_bytes() const {
            return bytes;
        }

        // Built for one island: page `page` plus `offset` in the bank.
        uint64_t local(uint32_t page, uint32_t offset) const {
            return get_noc_addr_from_bank_id<true>(bank, word + page * bytes + offset);
        }

        DramRun place(uint32_t page, uint32_t) const {
            return bank_page_at(word, bank, bytes, page);
        }
    };

    // A page the entry of a SizedTensor or BankRange refuses: recorded, false.
    template <typename Kernel>
    inline bool dram_refused(const Kernel& core, uint32_t slot, uint32_t sequence) {
        core.gave_up(
            core.role() == kRoleWriter ? kStWriter : kStReader,
            kGaveBadItem,
            kWhereAsk,
            slot,
            sequence
        );
        return false;
    }

    // The bank and bank address of byte `offset` of index `index` of the DRAM pages
    // of `page` bytes at bank address `address` (DramRun's).
    struct DramSpot {
        uint32_t bank;
        uint32_t address;
    };

    inline DramSpot dram_spot(
        const DramRun& at, uint32_t index, uint32_t page, uint32_t offset
    ) {
        return {
            index % NUM_DRAM_BANKS, at.address + index / NUM_DRAM_BANKS * page + offset
        };
    }

    // Pages first .. of `form` into L1 `to`: here a read a page, another island's
    // runs a get line each, waited for. One island: the reads alone. False: gave up.
    template <typename Kernel, typename Form, typename = typename Form::DramForm>
    inline bool read_pages_at(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const Form& form,
        uint32_t first,
        uint32_t pages,
        uint32_t to
    ) {
        if (!pages_inside(core, slot, sequence, form, first, pages)) {
            return false;  // KM_GUARD
        }
        const uint32_t bytes = form.page_bytes();
        if constexpr (kDramLocal) {
            for (uint32_t k = 0; k < pages; k++) {
                noc_async_read(form.local(first + k, 0), to + k * bytes, bytes);
            }
            return true;
        } else {
            const uint32_t here = island();
            const uint32_t port = pages_port(core);
            invalidate_l1_cache();  // the table may have grown since the last read
            uint32_t asked = 0, holder = here;
            for (uint32_t k = 0; k < pages;) {
                const DramRun at = form.place(first + k, pages - k);
                if (at.pages == 0) {
                    return dram_refused(core, slot, sequence);
                }
                if (at.island == here) {
                    const InterleavedAddrGen<true> place = {
                        .bank_base_address = at.address, .page_size = bytes
                    };
                    for (uint32_t m = 0; m < at.pages; m++) {
                        noc_async_read(
                            place.get_noc_addr(at.index + m),
                            to + (k + m) * bytes,
                            bytes
                        );
                    }
                } else {
                    if (!ask_line_at(
                        core,
                        slot,
                        sequence,
                        ask_gets(core.role()),
                        at.island,
                        at.address,
                        bytes,
                        at.index,
                        at.pages,
                        to + k * bytes,
                        port
                    )) {
                        return false;
                    }
                    asked += at.pages;
                    holder = at.island;
                }
                k += at.pages;
            }
            return asked == 0 || ask_wait(core, slot, sequence, port, asked, holder);
        }
    }

    // read_pages_at of `form` for `command`.
    template <typename Kernel, typename Form, typename = typename Form::DramForm>
    inline bool read_pages(
        const Kernel& core,
        const km_command_t& command,
        const Form& form,
        uint32_t first,
        uint32_t pages,
        uint32_t to
    ) {
        return (
            read_pages_at(core, slot(command), command.sequence, form, first, pages, to)
        );
    }

    // Pages first .. of `form` from L1 `from`: here a write a page, another island's
    // runs a put line each, waited for. `from` free on return. False: gave up.
    template <typename Kernel, typename Form, typename = typename Form::DramForm>
    inline bool write_pages_at(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const Form& form,
        uint32_t first,
        uint32_t pages,
        uint32_t from KM_SOURCE_LINE
    ) {
        if (!pages_inside(core, slot, sequence, form, first, pages)) {
            return false;  // KM_GUARD
        }
        const uint32_t bytes = form.page_bytes();
        if (!page_source_ok(core, from, bytes KM_LINE)) {
            return false;  // KM_GUARD
        }
        if constexpr (kDramLocal) {
            for (uint32_t k = 0; k < pages; k++) {
                noc_async_write(from + k * bytes, form.local(first + k, 0), bytes);
            }
            return true;
        } else {
            const uint32_t here = island();
            const uint32_t port = pages_port(core);
            invalidate_l1_cache();  // the table may have grown since the last read
            uint32_t put = 0, holder = here;
            for (uint32_t k = 0; k < pages;) {
                const DramRun at = form.place(first + k, pages - k);
                if (at.pages == 0) {
                    return dram_refused(core, slot, sequence);
                }
                if (at.island == here) {
                    const InterleavedAddrGen<true> place = {
                        .bank_base_address = at.address, .page_size = bytes
                    };
                    for (uint32_t m = 0; m < at.pages; m++) {
                        noc_async_write(
                            from + (k + m) * bytes,
                            place.get_noc_addr(at.index + m),
                            bytes
                        );
                    }
                } else {
                    if (!ask_line_at(
                        core,
                        slot,
                        sequence,
                        ask_puts(core.role()),
                        at.island,
                        at.address,
                        bytes,
                        at.index,
                        at.pages,
                        from + k * bytes,
                        port
                    )) {
                        return false;
                    }
                    put += at.pages;
                    holder = at.island;
                }
                k += at.pages;
            }
            return put == 0 || ask_wait(core, slot, sequence, port, put, holder);
        }
    }

    // write_pages_at of `form` for `command`.
    template <typename Kernel, typename Form, typename = typename Form::DramForm>
    inline bool write_pages(
        const Kernel& core,
        const km_command_t& command,
        const Form& form,
        uint32_t first,
        uint32_t pages,
        uint32_t from KM_SOURCE_LINE
    ) {
        return (
            write_pages_at(
                core, slot(command), command.sequence, form, first, pages, from KM_LINE
            )
        );
    }

    // `bytes` at `offset` of page `page` of `form` into L1 `into`, not waited for
    // (bytes_wait): here a NoC read, else a get line, `asked` + 1. False: gave up.
    template <typename Kernel, typename Form, typename = typename Form::DramForm>
    inline bool ask_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Form& form,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t into,
        uint32_t& asked
    ) {
        if constexpr (kDramLocal) {
            noc_async_read(form.local(page, offset), into, bytes);
            return true;
        } else {
            if (!bytes_fit(core, command, form.page_bytes(), offset, bytes, into)) {
                return false;
            }
            invalidate_l1_cache();  // the table may have grown since the last read
            const DramRun at = form.place(page, 1);
            if (at.pages == 0) {
                return dram_refused(core, slot(command), command.sequence);
            }
            const DramSpot spot = dram_spot(at, at.index, form.page_bytes(), offset);
            if (at.island == island()) {
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(spot.bank, spot.address),
                    into,
                    bytes
                );
                return true;
            }
            // the range as a page of its own from its 64-byte line, index its bank
            const uint32_t low = spot.address % 64;
            if (!ask_bytes_line(
                core,
                slot(command),
                command.sequence,
                at.island,
                spot.address - low,
                (low + bytes + 15) / 16 * 16,
                spot.bank,
                low,
                bytes,
                into
            )) {
                return false;
            }
            asked++;
            return true;
        }
    }

    // ask_bytes of `form`, then bytes_wait: the bytes in L1 `into` on return.
    template <typename Kernel, typename Form, typename = typename Form::DramForm>
    inline bool read_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Form& form,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t into
    ) {
        uint32_t asked = 0;
        return (
            ask_bytes(core, command, form, page, offset, bytes, into, asked) &&
            bytes_wait(core, command, asked)
        );
    }

    // Bytes `offset` on of page `page` of `form` become L1 `source`'s, unwaited
    // (put_wait): here a NoC write, else records in this CPU's batch. False: gave up.
    template <typename Kernel, typename Form, typename = typename Form::DramForm>
    inline bool put_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Form& form,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t source
    ) {
        if constexpr (kDramLocal) {
            noc_async_write(source, form.local(page, offset), bytes);
            return true;
        } else {
            // multiples of 4 inside the page, `source` with the offset's low 4 bits
            const uint32_t page_bytes = form.page_bytes();
            if (
                bytes == 0 || (offset | bytes) % 4 != 0 || offset >= page_bytes ||
                bytes > page_bytes - offset || (source ^ offset) % 16 != 0
            ) {
                core.gave_up(
                    core.role() == kRoleWriter ? kStWriter : kStReader,
                    kGaveBadTransfer,
                    kWhereAsk,
                    slot(command),
                    command.sequence
                );
                return false;
            }
            invalidate_l1_cache();  // the table may have grown since the last read
            const DramRun at = form.place(page, 1);
            if (at.pages == 0) {
                return dram_refused(core, slot(command), command.sequence);
            }
            const DramSpot spot = dram_spot(at, at.index, page_bytes, 0);
            if (at.island == island()) {
                noc_async_write(
                    source,
                    get_noc_addr_from_bank_id<true>(spot.bank, spot.address + offset),
                    bytes
                );
                return true;
            }
            // the page's start in its bank as index `bank` of pages at that address
            return (
                put_records(
                    core,
                    command,
                    at.island,
                    spot.address,
                    spot.bank,
                    page_bytes,
                    offset,
                    bytes,
                    source
                )
            );
        }
    }

    // put_bytes of `form`, then put_wait: the bytes written on return.
    template <typename Kernel, typename Form, typename = typename Form::DramForm>
    inline bool write_bytes(
        const Kernel& core,
        const km_command_t& command,
        const Form& form,
        uint32_t page,
        uint32_t offset,
        uint32_t bytes,
        uint32_t source
    ) {
        return (
            put_bytes(core, command, form, page, offset, bytes, source) &&
            put_wait(core, command)
        );
    }

    // Collectives across islands: a rank's pages in its island's DRAM go to a neighbour
    // island's tensor by the page CPUs, a put line a run; waited for once at the end.

    // Pages posted by km::gather_pages and not yet waited for, the last island posted
    // to.
    struct Gather {
        uint32_t pages;
        uint32_t island;
    };

    // Keepers: ranks rank .. rank + ranks - 1 of the far island and their port; once a
    // gather's pages are all written there, keeper k counts share k of them.
    constexpr uint32_t keeper(uint32_t rank, uint32_t port, uint32_t ranks = 1) {
        return (
            (rank + 1) | (ranks - 1) << KM_KEEPER_RANKS_SHIFT |
            port << KM_KEEPER_PORT_SHIFT
        );
    }

    // Page `page` of `tensor` on island `island`: a copy's there, a split tensor's
    // where page_at says (another island's: address 0).
    inline PageAt gather_place(uint32_t tensor, uint32_t page, uint32_t island) {
        const Islands view(island_area());
        const volatile uint32_t* entry = tensor_entry(view, tensor);
        if (entry == nullptr || island >= view.count()) {
            return {island, 0, page};
        }
        if (entry[3] & 1u) {
            const volatile uint32_t* parts = entry + KM_TENSOR_HEADER_BYTES / 4;
            return {island, parts[island * kPartWords], page};
        }
        const PageAt at = page_at(tensor, page);
        return at.island == island ? at : PageAt{island, 0, page};
    }

    // gather_pages (below), page first + k of `tensor` into page `into_first` + k of
    // `into`.
    template <typename Kernel>
    __attribute__((noinline)) bool gather_run(
        const Kernel& core,
        const km_command_t& command,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t to,
        Gather& gather,
        uint32_t keeper,
        uint32_t into,
        uint32_t into_first
    ) {
        if (pages == 0) {
            return true;
        }
        const uint32_t target = into != 0 ? into : tensor;
        const Islands view = islands_at(core.args().control);
        const uint32_t here = view.here();
        const volatile uint32_t* entry = tensor_entry(view, tensor);
        const volatile uint32_t* other = tensor_entry(view, target);
        const auto give_up = [&](uint32_t code) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                code,
                kWhereAsk,
                slot(command) | to << 8,
                command.sequence
            );
            return false;
        };
        // tensors of the table, pages of one size, to a neighbour
        if (
            entry == nullptr || other == nullptr || other[0] != entry[0] ||
            to == here || view.hops(to) != 1
        ) {
            return give_up(kGaveBadItem);
        }
        const uint32_t bytes = entry[0];
        for (uint32_t k = 0; k < pages;) {
            const uint32_t p = first + k;
            const uint32_t q = into_first + k;
            const PageAt from = gather_place(tensor, p, here);
            const PageAt at = gather_place(target, q, to);
            if (from.address == 0 || at.address == 0) {
                return give_up(kGaveBadItem);  // not here, or not there
            }
            uint32_t run = page_run(tensor, p, pages - k, from);
            run = page_run(target, q, run, at);
            if (!ask_line_at(
                core,
                slot(command),
                command.sequence,
                ask_puts(core.role()),
                to,
                at.address,
                bytes,
                at.index,
                run,
                from.address,
                pages_port(core),
                1,
                from.index,
                nullptr,
                keeper
            )) {
                return false;
            }
            gather.pages += run;
            gather.island = to;
            k += run;
        }
        return true;
    }

    // Pages first .. + pages - 1 of `tensor` here become those of `into` (0: `tensor`)
    // on neighbour `to`: a put line a run from DRAM, `keeper` counts. False: gave up.
    template <typename Kernel>
    inline bool gather_pages(
        const Kernel& core,
        const km_command_t& command,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t to,
        Gather& gather,
        uint32_t keeper = 0,
        uint32_t into = 0
    ) {
        return (
            gather_run(
                core, command, tensor, first, pages, to, gather, keeper, into, first
            )
        );
    }

    // Reader or writer: waits until every page `gather` posted was written on its
    // island (counted at its page port), then `gather` is empty. False: gave up.
    template <typename Kernel>
    inline bool gather_wait(
        const Kernel& core, const km_command_t& command, Gather& gather
    ) {
        if (gather.pages == 0) {
            return true;
        }
        const uint32_t pages = gather.pages;
        gather.pages = 0;
        return (
            ask_wait(
                core,
                slot(command),
                command.sequence,
                pages_port(core),
                pages,
                gather.island
            )
        );
    }

    // A give-up of the collectives' (kWhereAsk), recorded; false.
    template <typename Kernel>
    __attribute__((noinline)) bool coll_gave_up(
        const Kernel& core, const km_command_t& command, uint32_t code
    ) {
        core.gave_up(
            core.role() == kRoleWriter ? kStWriter : kStReader,
            code,
            kWhereAsk,
            slot(command),
            command.sequence
        );
        return false;
    }

    // This CPU's atomics flushed (bounded). False: gave up.
    template <typename Kernel>
    __attribute__((noinline)) bool coll_flushed(
        const Kernel& core, const km_command_t& command
    ) {
        const uint32_t code = core.poll_at(
            kWhereAsk,
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            core.args().wait_limit
        );
        return code == kGaveNone || coll_gave_up(core, command, code);
    }

    // Waits (bounded) until this core's L1 word `word` has counted `wanted`, then takes
    // them off by an atomic (a count landing meanwhile is not lost). False: gave up.
    template <typename Kernel>
    __attribute__((noinline)) bool taken_off(
        const Kernel& core,
        const km_command_t& command,
        volatile tt_l1_ptr uint32_t* word,
        uint32_t wanted
    ) {
        const uint32_t code = core.poll_at(
            kWhereAsk,
            [&] {
                invalidate_l1_cache();
                return static_cast<int32_t>(*word - wanted) >= 0;
            },
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            return coll_gave_up(core, command, code);
        }
        noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(word)), 0u - wanted);
        return coll_flushed(core, command);
    }

    // Reader or writer, a keeper: waits until its port `port` has counted `pages` pages
    // gathered into this island (km::keeper), then takes them off it. False: gave up.
    template <typename Kernel>
    inline bool gathered(
        const Kernel& core, const km_command_t& command, uint32_t port, uint32_t pages
    ) {
        if (port >= KM_PORT_SLOTS) {
            return coll_gave_up(core, command, kGaveBadItem);
        }
        return (
            pages == 0 ||
            taken_off(core, command, &port_words(core, port)[KM_PORT_COUNT], pages)
        );
    }

    // km::all_gather and km::all_reduce: every rank of the command calls one; the first
    // of an island's ranks sends its shares over the islands' tree, all share a fold.

    // The collectives' wall clocks (KM_COLL_STAMPS: a copy's reference word, a 128-byte
    // page a flat rank, written at the return); compiled out by default.
    enum CollStamp : uint32_t {
        kCsEntry,
        kCsMet,
        kCsArrived = 3,
        kCsConfirmed,
        kCsReleased,
        kCsReadCycles = 20,  // sums over this call's folds: waits for the reads,
        kCsAddCycles,        // the adds, waits for the writes, the words added
        kCsWriteCycles,
        kCsAddWords,
        kCsStamps = 32
    };
#if defined(KM_COLL_STAMPS)
#if defined(KM_OVERLAY_PART)
    // an overlay's statics are linked into its slot, in L1
    inline volatile tt_l1_ptr uint32_t* coll_words() {
        alignas(16) static uint32_t part[kCsStamps];
        return part;
    }

    inline void coll_stamps_in(uint32_t) {
    }
#else
    // a kernel's statics are the core's local memory, which the NoC cannot read: its
    // stamps are kept in 4 * kCsStamps bytes of L1 the program gives (coll_stamps_in)
    inline volatile tt_l1_ptr uint32_t*& coll_place() {
        static volatile tt_l1_ptr uint32_t* place = nullptr;
        return place;
    }

    inline volatile tt_l1_ptr uint32_t* coll_words() {
        return coll_place();
    }

    // The kernel's stamps at L1 `l1` (16-byte aligned, below the end of L1); else none.
    inline void coll_stamps_in(uint32_t l1) {
        const bool fits = l1 != 0 && l1 % 16 == 0 && l1 + 4 * kCsStamps <= MEM_L1_SIZE;
        coll_place() = fits ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1)
                            : nullptr;
    }
#endif

    inline void coll_stamp(uint32_t step) {
        if (coll_words() != nullptr) {
            coll_words()[step] = wall();
        }
    }

    inline void coll_count(uint32_t step, uint32_t amount) {
        if (coll_words() != nullptr) {
            coll_words()[step] += amount;
        }
    }

    inline void coll_begin() {
        for (uint32_t w = 0; coll_words() != nullptr && w < kCsStamps; w++) {
            coll_words()[w] = 0;
        }
        coll_stamp(kCsEntry);
    }

    // This rank's stamps into its page of KM_COLL_STAMPS on its island, waited for.
    template <typename Kernel>
    inline void coll_stamps_out(const Kernel& core) {
        const uint32_t here = islands_at(core.args().control).here();
        const PageAt at = gather_place(KM_COLL_STAMPS, core.me(), here);
        const uint32_t words = reinterpret_cast<uint32_t>(coll_words());
        // a NoC write's source is L1 only (a kernel's statics are its local memory)
        if (at.address == 0 || words == 0 || words + 4 * kCsStamps > MEM_L1_SIZE) {
            return;
        }
        noc_async_write(
            words,
            pages_dram(KM_PAGES_DRAM, at.address, 4 * kCsStamps, at.index),
            4 * kCsStamps
        );
        core.poll_at(
            kWhereAsk,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            core.args().wait_limit
        );
    }
#else
    inline void coll_stamp(uint32_t) {
    }

    inline void coll_count(uint32_t, uint32_t) {
    }

    inline void coll_begin() {
    }

    inline void coll_stamps_in(uint32_t) {
    }

    template <typename Kernel>
    inline void coll_stamps_out(const Kernel&) {
    }
#endif

    // This island's ranks of `command`: the leader's flat rank, how many, this core's
    // place among them (from its flat rank `me`).
    struct Team {
        uint32_t leader;
        uint32_t size;
        uint32_t index;
    };

    // The releases so far in this command: every call's markers follow the last call's,
    // so a second collective in one command never meets the first's marker.
    struct TeamGoes {
        uint32_t sequence;
        uint32_t goes;
    };

#if defined(KM_OVERLAY_PART)
    // a part's own: its .bss is zeroed at each call, and a command is one call
    inline TeamGoes& team_goes_now() {
        static TeamGoes own = {};
        return own;
    }
#else
    inline TeamGoes team_goes = {};

    inline TeamGoes& team_goes_now() {
        return team_goes;
    }
#endif

    inline Team team_of(const Islands& view, const km_command_t& command, uint32_t me) {
        const uint32_t here = view.here();
        const uint32_t e0 = view.entry_first(here), e1 = e0 + view.entry_count(here);
        const uint32_t f0 = first(command), f1 = f0 + count(command);
        const uint32_t lo = f0 > e0 ? f0 : e0, hi = f1 < e1 ? f1 : e1;
        return {lo, hi > lo ? hi - lo : 0, me - lo};
    }

    // The leader of `command`'s ranks on island `island`: its index there.
    inline uint32_t team_leader(
        const Islands& view, const km_command_t& command, uint32_t island
    ) {
        const uint32_t e0 = view.entry_first(island), f0 = first(command);
        return (f0 > e0 ? f0 : e0) - e0;
    }

    // One more meeting on flat rank `rank` of this island (word KM_PORT_MET of its
    // KM_PORT_LEFT slot); `flush`: then this CPU's atomics flushed. False: gave up.
    template <typename Kernel>
    inline bool team_add(
        const Kernel& core,
        const km_command_t& command,
        uint32_t rank,
        bool flush = true
    ) {
        noc_semaphore_inc(
            core.core_noc(
                core.core_of(rank),
                core.args().control + KM_CONTROL_PORT + KM_PORT_LEFT * KM_PORT_STRIDE +
                    4 * KM_PORT_MET
            ),
            1
        );
        return !flush || coll_flushed(core, command);
    }

    // Waits (bounded) for `meetings` more on this core's meeting word, then takes them
    // off. False: gave up.
    template <typename Kernel>
    inline bool team_met(
        const Kernel& core, const km_command_t& command, uint32_t meetings
    ) {
        return (
            meetings == 0 ||
            taken_off(
                core, command, &port_words(core, KM_PORT_LEFT)[KM_PORT_MET], meetings
            )
        );
    }

    // The team meets: each rank tells the leader, the leader waits for all. False: gave
    // up.
    template <typename Kernel>
    inline bool team_meet(
        const Kernel& core, const km_command_t& command, const Team& t
    ) {
        if (t.index != 0) {
            return team_add(core, command, t.leader);
        }
        return team_met(core, command, t.size - 1);
    }

    // Word KM_PORT_GO of this core's KM_PORT_LEFT slot (L1, the control block).
    template <typename Kernel>
    inline uint32_t team_go_word(const Kernel& core) {
        return (
            core.args().control + KM_CONTROL_PORT + KM_PORT_LEFT * KM_PORT_STRIDE +
            4 * KM_PORT_GO
        );
    }

    // The leader's release of a whole island: `marker` into every other core's go word,
    // a multicast a rectangle of `layout` (KM_ISLAND_GO), the writes waited for.
    template <typename Kernel>
    __attribute__((noinline)) bool team_cast(
        const Kernel& core,
        const km_command_t& command,
        uint32_t layout,
        uint32_t marker
    ) {
        const uint32_t go = team_go_word(core);
        const uint32_t columns = layout & 0xFFu;
        const uint32_t rows = layout >> KM_ISLAND_GO_ROWS & 0xFFu;
        const uint32_t split = layout >> KM_ISLAND_GO_SPLIT & 0xFFu;
        const uint32_t end = split != 0 ? split : columns;  // panel 0's columns
        const uint32_t base = islands_at(core.args().control).base();
        const volatile tt_l1_ptr uint32_t* table = core.control(KM_CONTROL_CORES);
        // columns x0 .. x1, rows y0 .. y1: the leader's row after it, panel 0's other
        // rows, panel 1; none holds the leader (core base, column 0 of row 0)
        const uint32_t rects[3][4] = {
            {1, end - 1, 0, 0},
            {0, end - 1, 1, rows - 1},
            {end, columns - 1, 0, rows - 1}
        };
        *l1(go) = marker;
        for (uint32_t r = 0; r < 3; r++) {
            const uint32_t x0 = rects[r][0], x1 = rects[r][1];
            const uint32_t y0 = rects[r][2], y1 = rects[r][3];
            if (x0 > x1 || y0 > y1 || x1 >= columns || y1 >= rows) {
                continue;  // empty
            }
            const uint32_t control = core.args().control;
            const uint32_t low = (
                core_entry(table, base + y0 * columns + x0, control, core.role())
            );
            const uint32_t high = (
                core_entry(table, base + y1 * columns + x1, control, core.role())
            );
            const uint32_t from = noc_index == 0 ? low
                                                 : high;  // NoC1: from the far end
            const uint32_t to = noc_index == 0 ? high : low;
            noc_async_write_multicast(
                go,
                get_noc_multicast_addr(
                    from & 0xFFFFu, from >> 16, to & 0xFFFFu, to >> 16, go
                ),
                4,
                (x1 - x0 + 1) * (y1 - y0 + 1)
            );
        }
        const uint32_t code = core.poll_at(
            kWhereAsk,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            core.args().wait_limit
        );
        return code == kGaveNone || coll_gave_up(core, command, code);
    }

    // The leader lets the team go: a whole island led from its first core, a marker
    // multicast a rectangle (KM_ISLAND_GO) each rank waits for; else an atomic a rank.
    template <typename Kernel>
    inline bool team_go(const Kernel& core, const km_command_t& command, Team& t) {
        const Islands view = islands_at(core.args().control);
        const uint32_t layout = view.go();
        TeamGoes& now = team_goes_now();
        if (now.sequence != command.sequence) {
            now = {command.sequence, 0};
        }
        const uint32_t goes = now.goes++ % 7 + 1;  // 1 to 7: two in a row differ
        if (
            (layout & KM_ISLAND_GO_ON) != 0 && t.size > 1 && t.size == view.ranks() &&
            core.core_of(t.leader) == view.base()
        ) {
            const uint32_t marker = command.sequence << 3 | goes;
            if (t.index == 0) {
                return team_cast(core, command, layout, marker);
            }
            const volatile tt_l1_ptr uint32_t* word = l1(team_go_word(core));
            const uint32_t code = core.poll_at(
                kWhereAsk,
                [&] {
                    invalidate_l1_cache();
                    return *word == marker;
                },
                core.args().wait_limit
            );
            return code == kGaveNone || coll_gave_up(core, command, code);
        }
        if (t.index != 0) {
            return team_met(core, command, 1);
        }
        for (uint32_t k = 1; k < t.size; k++) {
            team_add(core, command, t.leader + k, false);
        }
        return coll_flushed(core, command);
    }

    // What km::all_reduce adds: 32-bit integers (modulo 2^32), float32, or two
    // bfloat16 a word (each added in float32, rounded to nearest even).
    enum class Sum : uint32_t { Int32 = 0, Float32 = 1, Bfloat16 = 2 };

    // float32 a + b, rounded to nearest even (IEEE 754, subnormals kept), in integer
    // arithmetic: an overlay cannot call the soft-float routines the kernels hold.
    __attribute__((noinline)) inline uint32_t sum_float32(uint32_t a, uint32_t b) {
        if ((a & 0x7FFFFFFFu) < (b & 0x7FFFFFFFu)) {  // |a| >= |b|
            const uint32_t t = a;
            a = b, b = t;
        }
        uint32_t ea = a >> 23 & 0xFFu, eb = b >> 23 & 0xFFu;
        const uint32_t sign = a & 0x80000000u;
        if (ea == 0xFFu) {  // a NaN stays one (quiet); inf - inf is one
            if ((a & 0x7FFFFFu) != 0) {
                return a | 0x400000u;
            }
            return (
                eb == 0xFFu && (b & 0x7FFFFFu) != 0
                    ? b | 0x400000u
                    : eb == 0xFFu && (a ^ b) >> 31
                        ? 0x7FC00000u
                        : a
            );
        }
        // the significands with three bits below them (guard, round, sticky)
        uint32_t ma = (a & 0x7FFFFFu) | (ea != 0 ? 0x800000u : 0);
        uint32_t mb = (b & 0x7FFFFFu) | (eb != 0 ? 0x800000u : 0);
        ea = ea != 0 ? ea : 1, eb = eb != 0 ? eb : 1;
        ma <<= 3, mb <<= 3;
        const uint32_t d = ea - eb;
        if (d >= 27) {
            mb = mb != 0;
        } else if (d != 0) {
            mb = mb >> d | ((mb & ((1u << d) - 1)) != 0);
        }
        uint32_t m;
        if ((a ^ b) >> 31 == 0) {
            m = ma + mb;
            if (m >= 1u << 27) {
                m = m >> 1 | (m & 1), ea++;
            }
        } else {
            m = ma - mb;
            if (m == 0) {
                return 0;  // x - x: +0
            }
            while (m < 1u << 26 && ea > 1) {
                m <<= 1, ea--;
            }
        }
        const uint32_t low = m & 7;
        m >>= 3;
        if (low > 4 || (low == 4 && (m & 1))) {
            m++;
            if (m == 1u << 24) {
                m >>= 1, ea++;
            }
        }
        if (ea >= 0xFFu) {
            return sign | 0x7F800000u;  // infinity
        }
        return sign | (m >= 0x800000u ? ea : 0) << 23 | (m & 0x7FFFFFu);
    }

    inline uint32_t sum_bfloat16(uint32_t a, uint32_t b) {
        const uint32_t u = sum_float32(a << 16, b << 16);
        if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu) != 0) {
            return (u >> 16) | 0x40u;  // a NaN stays one
        }
        return (u + 0x7FFFu + (u >> 16 & 1u)) >> 16;
    }

    inline uint32_t sum_word(Sum kind, uint32_t a, uint32_t b) {
        if (kind == Sum::Float32) {
            return sum_float32(a, b);
        }
        if (kind == Sum::Bfloat16) {
            return (
                sum_bfloat16(a & 0xFFFFu, b & 0xFFFFu) |
                sum_bfloat16(a >> 16, b >> 16) << 16
            );
        }
        return a + b;
    }

    // The NoC address of page `page` of the copy `tensor` (`bytes` a page) on this
    // island `here`.
    __attribute__((noinline)) inline uint64_t fold_page(
        uint32_t tensor, uint32_t page, uint32_t here, uint32_t bytes
    ) {
        const PageAt place = gather_place(tensor, page, here);
        return pages_dram(KM_PAGES_DRAM, place.address, bytes, place.index);
    }

    // The team's fold: page to_first + q of `to` = a_first + q of `a` + b_first + q of
    // `b`, its share of q < pages, through L1 `buffer` (`room`: two pages or more).
    template <typename Kernel>
    __attribute__((noinline)) bool team_fold(
        const Kernel& core,
        const km_command_t& command,
        const Team& t,
        Sum kind,
        uint32_t to,
        uint32_t to_first,
        uint32_t a,
        uint32_t a_first,
        uint32_t b,
        uint32_t b_first,
        uint32_t pages,
        uint32_t buffer,
        uint32_t room
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t here = view.here();
        const volatile uint32_t* entry = tensor_entry(view, to);
        const uint32_t bytes = entry != nullptr ? entry[0] : 0;
        const uint32_t step = bytes != 0 ? room / (2 * bytes) : 0;
        const uint32_t limit = core.args().wait_limit;
        if (step == 0 || buffer % 16 != 0) {
            return coll_gave_up(core, command, kGaveBadItem);
        }
        const auto at = [&](uint32_t tensor, uint32_t page) {
            return fold_page(tensor, page, here, bytes);
        };
        const Partition share = partition(pages, t.index, t.size);
        for (uint32_t k = 0; k < share.len;) {
            const uint32_t n = share.len - k < step ? share.len - k : step;
            const uint32_t q = share.start + k;
            for (uint32_t i = 0; i < n; i++) {
                noc_async_read(at(a, a_first + q + i), buffer + i * bytes, bytes);
                noc_async_read(at(b, b_first + q + i), buffer + (n + i) * bytes, bytes);
            }
            uint32_t clock = wall();
            uint32_t code = core.poll_at(
                kWhereAsk,
                [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                },
                limit
            );
            if (code != kGaveNone) {
                return coll_gave_up(core, command, code);
            }
            coll_count(kCsReadCycles, wall() - clock);
            clock = wall();
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* x = l1(buffer);
            const volatile tt_l1_ptr uint32_t* y = l1(buffer + n * bytes);
            for (uint32_t w = 0; w < n * bytes / 4; w++) {
                x[w] = sum_word(kind, x[w], y[w]);
            }
            coll_count(kCsAddCycles, wall() - clock);
            coll_count(kCsAddWords, n * bytes / 4);
            for (uint32_t i = 0; i < n; i++) {
                noc_async_write(buffer + i * bytes, at(to, to_first + q + i), bytes);
            }
            clock = wall();
            code = core.poll_at(
                kWhereAsk,
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                limit
            );
            if (code != kGaveNone) {
                return coll_gave_up(core, command, code);
            }
            coll_count(kCsWriteCycles, wall() - clock);
            k += n;
        }
        return true;
    }

    // A collective's tree over the islands: the routes' tree of each root (tree());
    // the user's: island i's parent in 4 bits at 4 i of low, high (the root its own).
    struct Tree {
        uint32_t low;
        uint32_t high;
        bool given;
    };

    constexpr Tree tree() {
        return {0, 0, false};
    }

    constexpr Tree tree(uint32_t low, uint32_t high) {
        return {low, high, true};
    }

    inline uint32_t tree_parent(const Tree& tree, uint32_t island) {
        return (
            (island < 8 ? tree.low >> 4 * island : tree.high >> 4 * (island - 8)) & 15u
        );
    }

    // The root of a tree of the user's: its one island that is its own parent (a cycle
    // the host refuses); KM_ISLANDS_MAX: none, or more than 16 islands.
    inline uint32_t tree_root(const Islands& view, const Tree& tree) {
        const uint32_t n = view.count();
        uint32_t root = KM_ISLANDS_MAX, roots = 0;
        bool bad = n > 16;
        for (uint32_t i = 0; i < n && !bad; i++) {
            const uint32_t parent = tree_parent(tree, i);
            roots += parent == i ? 1 : 0;
            root = parent == i ? i : root;
            bad = parent >= n;
        }
        return !bad && roots == 1 ? root : KM_ISLANDS_MAX;
    }

    // In the routes' tree of `root`: this island's parent (this island at the root)
    // and its port there (`back`), from the next link toward the root.
    struct TreeUp {
        uint32_t parent;
        uint32_t back;
    };

    inline TreeUp tree_up(const Islands& view, uint32_t root) {
        const uint32_t here = view.here();
        uint32_t link = view.link(root);
        for (uint32_t port = 0; root != here && port < KM_ISLAND_PORTS_MAX; port++) {
            const uint32_t word = view.port_link(port, 0, 0);
            const uint32_t links = word >> 24;
            if (link < links) {
                return {word >> 16 & 0xFFu, view.port_link(port, 0, 1) >> 24};
            }
            link -= links;
        }
        return {here, 0};
    }

    // The island on port `port` of this one.
    inline uint32_t tree_far(const Islands& view, uint32_t port) {
        return view.port_link(port, 0, 0) >> 16 & 0xFFu;
    }

    // This island in share c's tree: the ports of its children, its parent (itself at
    // the root) and its port there; `ok` false: a user's parent not one link away.
    struct TreeAt {
        uint32_t kids;
        TreeUp up;
        bool ok;
    };

    inline TreeAt tree_at(const Islands& view, const Tree& tree, uint32_t c) {
        if (!tree.given) {
            return {view.children(c), tree_up(view, c), true};
        }
        const uint32_t here = view.here(), parent = tree_parent(tree, here);
        TreeAt at = {0, {here, 0}, parent == here};
        for (uint32_t port = 0; port < KM_ISLAND_PORTS_MAX; port++) {
            const uint32_t word = view.port_link(port, 0, 0);
            const uint32_t far = word >> 16 & 0xFFu;
            if (word >> 24 == 0) {
                continue;  // no neighbour on this port
            }
            at.kids |= (tree_parent(tree, far) == here ? 1u : 0u) << port;
            if (far == parent && parent != here) {
                at.up = {far, view.port_link(port, 0, 1) >> 24};
                at.ok = true;
            }
        }
        return at;
    }

    // The pages of share `c` of a collective on `pages` pages over `n` islands: all
    // of them at the root `root` alone (root < n), else partition(pages, c, n).
    inline Partition tree_share(uint32_t pages, uint32_t c, uint32_t n, uint32_t root) {
        if (root < n) {
            return c == root ? Partition{0, pages} : Partition{0, 0};
        }
        return partition(pages, c, n);
    }

    // The leader waits (bounded) until word `c` of its counts table counted `pages`
    // pages, then takes them off. False: gave up.
    template <typename Kernel>
    inline bool tree_counted(
        const Kernel& core, const km_command_t& command, uint32_t c, uint32_t pages
    ) {
        const uint32_t counts = islands_at(core.args().control).counts();
        if (counts == 0) {
            return coll_gave_up(core, command, kGaveBadItem);
        }
        return pages == 0 || taken_off(core, command, l1(counts + 4 * c), pages);
    }

    // The leader's down sweep: each share, nearest root first, from its parent (counted
    // in its counts table) on to its children in its root's tree, DRAM to DRAM.
    template <typename Kernel>
    __attribute__((noinline)) bool tree_down(
        const Kernel& core,
        const km_command_t& command,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t root,
        const Tree& tree,
        Gather& posted
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t n = view.count(), here = view.here();
        for (uint32_t h = 0, done = 0; done < n; h++) {
            for (uint32_t c = 0; c < n; c++) {
                if (view.hops(c) != h) {
                    continue;
                }
                done++;
                const Partition part = tree_share(pages, c, n, root);
                if (part.len == 0) {
                    continue;
                }
                const TreeAt at = tree_at(view, tree, c);
                if (!at.ok) {
                    return coll_gave_up(core, command, kGaveBadItem);
                }
                if (c != here && !tree_counted(core, command, c, part.len)) {
                    return false;
                }
                for (uint32_t kids = at.kids, port = 0; kids != 0; port++) {
                    if ((kids >> port & 1u) == 0) {
                        continue;
                    }
                    kids &= ~(1u << port);
                    const uint32_t to = tree_far(view, port);
                    if (!gather_run(
                        core,
                        command,
                        tensor,
                        first + part.start,
                        part.len,
                        to,
                        posted,
                        keeper(team_leader(view, command, to), KM_PORT_SLOTS + c),
                        0,
                        first + part.start
                    )) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // The team's up sweep, each share farthest root first: its children's partials (in
    // scratch at port x pages) added into the copy, then put into its parent's scratch.
    template <typename Kernel>
    __attribute__((noinline)) bool tree_reduce(
        const Kernel& core,
        const km_command_t& command,
        Team& t,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t scratch,
        Sum kind,
        uint32_t buffer,
        uint32_t room,
        uint32_t root,
        const Tree& tree,
        Gather& posted
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t n = view.count(), here = view.here();
        const bool lead = t.index == 0;
        uint32_t deepest = 0;
        for (uint32_t c = 0; c < n; c++) {
            deepest = view.hops(c) > deepest ? view.hops(c) : deepest;
        }
        for (uint32_t h = deepest + 1; h-- > 0;) {
            for (uint32_t c = 0; c < n; c++) {
                const Partition part = tree_share(pages, c, n, root);
                if (view.hops(c) != h || part.len == 0) {
                    continue;
                }
                const TreeAt at = tree_at(view, tree, c);
                const uint32_t kids = at.kids;
                if (!at.ok) {
                    return coll_gave_up(core, command, kGaveBadItem);
                }
                if (kids != 0) {
                    const uint32_t partials = __builtin_popcount(kids) * part.len;
                    if (
                        (lead && !tree_counted(core, command, c, partials)) ||
                        !team_go(core, command, t)
                    ) {
                        return false;
                    }
                    for (uint32_t port = 0; port < KM_ISLAND_PORTS_MAX; port++) {
                        if (
                            (kids >> port & 1u) != 0 &&
                            !team_fold(
                                core,
                                command,
                                t,
                                kind,
                                tensor,
                                first + part.start,
                                tensor,
                                first + part.start,
                                scratch,
                                port * pages + part.start,
                                part.len,
                                buffer,
                                room
                            )
                        ) {
                            return false;
                        }
                    }
                    if (!team_meet(core, command, t)) {
                        return false;
                    }
                }
                if (c == here || !lead) {
                    continue;
                }
                const TreeUp up = at.up;
                if (!gather_run(
                    core,
                    command,
                    tensor,
                    first + part.start,
                    part.len,
                    up.parent,
                    posted,
                    keeper(
                        team_leader(view, command, up.parent), KM_PORT_SLOTS + c
                    ),
                    scratch,
                    up.back * pages + part.start
                )) {
                    return false;
                }
            }
        }
        return true;
    }

    // The leader's posts confirmed, then the team let go. False: gave up.
    template <typename Kernel>
    inline bool tree_end(
        const Kernel& core, const km_command_t& command, Team& t, Gather& posted
    ) {
        if (t.index == 0) {
            coll_stamp(kCsArrived);
            if (!gather_wait(core, command, posted)) {
                return false;
            }
            coll_stamp(kCsConfirmed);
        }
        if (!team_go(core, command, t)) {
            return false;
        }
        coll_stamp(kCsReleased);
        coll_stamps_out(core);
        return true;
    }

    // All-gather of pages first .. + pages - 1 of the copy `tensor`, island c's share
    // partition(pages, c, islands), each down its island's tree: every copy all of it.
    template <typename Kernel>
    __attribute__((noinline)) bool all_gather(
        const Kernel& core,
        const km_command_t& command,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        Tree tree = km::tree()
    ) {
#if defined(KM_TENSOR_TABLE)
        const Islands view = islands_at(core.args().control);
        if (view.count() < 2) {
            return true;
        }
        if (tree.given) {  // each share from its own island: the routes' trees
            return coll_gave_up(core, command, kGaveBadItem);
        }
        coll_begin();
        Team t = team_of(view, command, core.me());
        Gather posted = {};
        if (!team_meet(core, command, t)) {
            return false;
        }
        coll_stamp(kCsMet);
        if (
            t.index == 0 &&
            !tree_down(core, command, tensor, first, pages, view.count(), tree, posted)
        ) {
            return false;
        }
        return tree_end(core, command, t, posted);
#else
        static_cast<void>(core), static_cast<void>(command), static_cast<void>(tensor);
        static_cast<void>(first), static_cast<void>(pages), static_cast<void>(tree);
        return true;  // one island: it holds the whole tensor
#endif
    }

    // The tree reduces of km::all_reduce (root: islands) and km::reduce_to, then (down)
    // the sums down the same trees. False: gave up.
    template <typename Kernel>
    __attribute__((noinline)) bool tree_collective(
        const Kernel& core,
        const km_command_t& command,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t scratch,
        Sum kind,
        uint32_t buffer,
        uint32_t room,
        uint32_t root,
        bool down,
        Tree tree
    ) {
        const Islands view = islands_at(core.args().control);
        if (tree.given) {  // a user's tree: every page up (and down) it, from its root
            const uint32_t at = tree_root(view, tree);
            if (at >= view.count() || (root < view.count() && root != at)) {
                return coll_gave_up(core, command, kGaveBadItem);
            }
            root = at;
        }
        coll_begin();
        Team t = team_of(view, command, core.me());
        Gather posted = {};
        if (!team_meet(core, command, t)) {
            return false;
        }
        coll_stamp(kCsMet);
        if (!tree_reduce(
            core,
            command,
            t,
            tensor,
            first,
            pages,
            scratch,
            kind,
            buffer,
            room,
            root,
            tree,
            posted
        ) ||
            (
                down && t.index == 0 &&
                !tree_down(core, command, tensor, first, pages, root, tree, posted)
            )) {
            return false;
        }
        return tree_end(core, command, t, posted);
    }

    // All-reduce of pages first .. + pages - 1 of the copy `tensor` (`kind`): share c
    // summed up island c's tree, then down it; `scratch` ports x pages, `buffer` L1.
    template <typename Kernel>
    __attribute__((noinline)) bool all_reduce(
        const Kernel& core,
        const km_command_t& command,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t scratch,
        Sum kind,
        uint32_t buffer,
        uint32_t room,
        Tree tree = km::tree()
    ) {
#if defined(KM_TENSOR_TABLE)
        const uint32_t n = islands_at(core.args().control).count();
        return (
            n < 2 ||
            tree_collective(
                core,
                command,
                tensor,
                first,
                pages,
                scratch,
                kind,
                buffer,
                room,
                n,
                true,
                tree
            )
        );
#else
        static_cast<void>(core), static_cast<void>(command), static_cast<void>(tensor);
        static_cast<void>(first), static_cast<void>(pages), static_cast<void>(scratch);
        static_cast<void>(kind), static_cast<void>(buffer), static_cast<void>(room);
        static_cast<void>(tree);
        return true;  // one island: its partial is the sum
#endif
    }

    // Reduce of the copies' pages first .. + pages - 1 into island `root`'s copy, up
    // its tree (the other copies are left partial); `scratch` ports x pages.
    template <typename Kernel>
    __attribute__((noinline)) bool reduce_to(
        const Kernel& core,
        const km_command_t& command,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t scratch,
        Sum kind,
        uint32_t buffer,
        uint32_t room,
        uint32_t root,
        Tree tree = km::tree()
    ) {
#if defined(KM_TENSOR_TABLE)
        const Islands view = islands_at(core.args().control);
        if (view.count() < 2) {
            return true;
        }
        if (root >= view.count()) {
            return coll_gave_up(core, command, kGaveBadItem);
        }
        return (
            tree_collective(
                core,
                command,
                tensor,
                first,
                pages,
                scratch,
                kind,
                buffer,
                room,
                root,
                false,
                tree
            )
        );
#else
        static_cast<void>(core), static_cast<void>(command), static_cast<void>(tensor);
        static_cast<void>(first), static_cast<void>(pages), static_cast<void>(scratch);
        static_cast<void>(kind), static_cast<void>(buffer), static_cast<void>(room);
        static_cast<void>(root), static_cast<void>(tree);
        return true;  // one island: its copy is the sum
#endif
    }

    // Broadcast of island `root`'s copy's pages first .. + pages - 1 into every other
    // island's copy, down its tree.
    template <typename Kernel>
    __attribute__((noinline)) bool broadcast_from(
        const Kernel& core,
        const km_command_t& command,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t root,
        Tree tree = km::tree()
    ) {
#if defined(KM_TENSOR_TABLE)
        const Islands view = islands_at(core.args().control);
        if (view.count() < 2) {
            return true;
        }
        if (root >= view.count() || (tree.given && tree_root(view, tree) != root)) {
            return coll_gave_up(core, command, kGaveBadItem);
        }
        coll_begin();
        Team t = team_of(view, command, core.me());
        Gather posted = {};
        if (!team_meet(core, command, t)) {
            return false;
        }
        coll_stamp(kCsMet);
        if (
            t.index == 0 &&
            !tree_down(core, command, tensor, first, pages, root, tree, posted)
        ) {
            return false;
        }
        return tree_end(core, command, t, posted);
#else
        static_cast<void>(core), static_cast<void>(command), static_cast<void>(tensor);
        static_cast<void>(first), static_cast<void>(pages), static_cast<void>(root);
        static_cast<void>(tree);
        return true;  // one island: its copy is the root's
#endif
    }

}  // namespace km
