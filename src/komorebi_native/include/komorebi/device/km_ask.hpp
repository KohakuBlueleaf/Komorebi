// A kernel's read and write of a tensor's pages wherever they live: this island's
// locally, others by a line into link core 0 (get or put). Takes a km::Tensor.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_pages.hpp"
#include "km_placement.hpp"
#include "km_port.hpp"
#include "km_transfer.hpp"

namespace km {

    constexpr uint32_t kAskNoAnswer = 0xFFFFFFFFu;  // the atomic's answer not yet in

    // A queue of lines in link core 0's link area: reserve count, count taken (one per
    // page CPU), lines; and the asking CPU's own line and answer words (KM_OWN_*).
    struct AskQueue {
        uint32_t tail;
        uint32_t taken;
        uint32_t lines;
        uint32_t own;
        uint32_t returned;
        uint32_t read;
        bool cursors;
    };

    // The get queue (the reader's) and the put queue of CPU `role`
    // (functions, not objects: an overlay keeps no copy of the runtime's data)
    constexpr AskQueue ask_gets() {
        return {
            KM_ASK_TAIL,
            KM_ASK_TAKEN,
            KM_ASK_LINES,
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
            KM_ASK_TAIL,
            KM_ASK_TAKEN,
            KM_ASK_LINES,
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

    // One line into `queue` for pages first .. first + pages - 1 at bank `address` of
    // island `holder`, into L1 `at` (or `dram`), counted at `port`; a put from DRAM may
    // name a keeper on the holder (KM_ASK_KEEPER, km::keeper). False: gave up.
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
        uint32_t keeper = 0
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t area = view.link_area();
        const uint32_t mine = view.own_words();
        const uint32_t named = slot | holder << 8;
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        const auto give_up = [&](uint32_t code) {
            core.gave_up(word, code, kWhereAsk, named, sequence);
            return false;
        };
        const uint32_t hops = view.hops(holder);
        if (
            area == 0 || mine == 0 || hops == 0 || (queue.cursors && hops != 1) ||
            port >= KM_PORT_SLOTS || bytes == 0 || bytes % 16 != 0 ||
            bytes > view.packet() || at % 16 != 0
        ) {
            // another island (a put: a link apart, not forwarded), whole packets
            return give_up(kGaveBadItem);
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
                kWhereAsk,
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
            kWhereAsk,
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
        // a put: its put queue place free (every page CPU took line KM_QUEUE_LINES
        // before); a get: free once link core 0 took line KM_ASK_SLOTS before
        const uint32_t build = block + KM_OWN_BUILD;
        bool ring = false;
        PutRings rings{};
        uint32_t to_line = area + queue.lines + number % KM_ASK_SLOTS * 64;
        if (queue.cursors) {
            code = put_rings_read(core, link, area, build, rings, number);
            if (code != kGaveNone) {
                return give_up(code);
            }
            ring = dram == 0 && rings.slots != 0;
            to_line = rings.queue + KM_QUEUE_PUTS + number % KM_QUEUE_LINES * 64;
        } else {
            volatile tt_l1_ptr uint32_t* taken = l1(mine + queue.read);
            bool reading = false;
            code = core.poll_at(
                kWhereAsk,
                [&] {
                    if (reading) {
                        if (!ncrisc_noc_reads_flushed(noc_index)) {
                            return false;
                        }
                        reading = false;
                        invalidate_l1_cache();
                        if (number - taken[0] < KM_ASK_SLOTS) {
                            return true;
                        }
                    }
                    noc_async_read(
                        core.core_noc(link, area + queue.taken), mine + queue.read, 16
                    );
                    reading = true;
                    return false;
                },
                limit
            );
            if (code != kGaveNone) {
                return give_up(code);
            }
        }
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
        line[KM_ASK_RING] = ring ? 1 : 0;
        line[KM_ASK_OWNER] = owner;
        if (queue.cursors && dram != 0) {
            line[KM_ASK_KEEPER] = keeper;
        }
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        noc_async_write(mine + queue.own, core.core_noc(link, to_line), 64);
        code = core.poll_at(
            kWhereAsk,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        if (queue.cursors) {  // its line on this parity, until the far island's word
            own[KM_OWN_PUT_SENT / 4] = own[KM_OWN_PUT_SENT / 4] + 1;
            own[KM_OWN_PUT_NUMBER / 4 + parity] = number + 1;
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
            kWhereAsk,
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
        if (area == 0 || mine == 0 || view.hops(holder) != 1) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                kGaveBadItem,
                kWhereAsk,
                slot(command) | holder << 8,
                command.sequence
            );
            return false;
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
        uint32_t from
    ) {
        const uint32_t here = island();
        const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
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
        uint32_t from
    ) {
        return (
            write_pages_at(
                core, slot(command), command.sequence, args, tensor, first, pages, from
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
            kWhereAsk,
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
        // the line free once link core 0 took the one KM_ASK_SLOTS before
        volatile tt_l1_ptr uint32_t* taken = l1(mine + queue.read);
        bool reading = false;
        code = core.poll_at(
            kWhereAsk,
            [&] {
                if (reading) {
                    if (!ncrisc_noc_reads_flushed(noc_index)) {
                        return false;
                    }
                    reading = false;
                    invalidate_l1_cache();
                    if (number - taken[0] < KM_ASK_SLOTS) {
                        return true;
                    }
                }
                noc_async_read(
                    core.core_noc(link, area + queue.taken), mine + queue.read, 16
                );
                reading = true;
                return false;
            },
            limit
        );
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
        line[KM_ASK_RING] = KM_ASK_GET_BYTES;
        line[KM_ASK_GET_SPAN] = offset | bytes << 16;
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        noc_async_write(
            mine + queue.own,
            core.core_noc(link, area + queue.lines + number % KM_ASK_SLOTS * 64),
            64
        );
        code = core.poll_at(
            kWhereAsk,
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
            kWhereAsk,
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
        uint32_t from
    ) {
        if constexpr (Tensor<Base>::kLocal) {
            const uint32_t bytes = tensor.page_bytes();
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
                    from
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
        uint32_t from
    ) {
        return (
            write_pages_at(
                core, slot(command), command.sequence, tensor, first, pages, from
            )
        );
    }

    // This CPU's batch at L1 `batch` sent once its last was confirmed (so in order): a
    // line, then the batch through the rings onto its landing page. False: gave up.
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
        if (
            area == 0 || mine == 0 || view.hops(holder) != 1 || me >= KM_CORES_MAX ||
            KM_PUT_BATCH_BYTES > view.packet()
        ) {
            return give_up(kGaveBadItem);  // a link apart, a batch a packet holds
        }
        const uint32_t link = view.base() + view.ranks() + view.link(holder);
        const uint32_t limit = core.args().wait_limit;
        const uint32_t block = (
            mine + (core.role() == kRoleWriter ? KM_OWN_CPU_BYTES : 0)
        );
        const uint32_t build = block + KM_OWN_BUILD;
        // the rings and the landing (read with them), before a line is reserved
        PutRings rings{};
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
            kWhereAsk,
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
            kWhereAsk,
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
        const uint32_t first = holder * KM_QUEUE_LINES + owner;  // its landing page
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
        line[KM_ASK_RING] = KM_ASK_RING_BATCH;
        line[KM_ASK_OWNER] = owner;
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        const uint32_t to_line = (
            rings.queue + KM_QUEUE_PUTS + number % KM_QUEUE_LINES * 64
        );
        noc_async_write(mine + queue.own, core.core_noc(link, to_line), 64);
        code = core.poll_at(
            kWhereAsk,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        own[KM_OWN_PUT_SENT / 4] = own[KM_OWN_PUT_SENT / 4] + 1;
        own[KM_OWN_PUT_NUMBER / 4 + parity] = number + 1;
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
            kWhereAsk,
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
        uint32_t from
    ) {
        const uint32_t bytes = form.page_bytes();
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
        uint32_t from
    ) {
        return (
            write_pages_at(
                core, slot(command), command.sequence, form, first, pages, from
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

    // Collectives across islands: pages a rank names in its island's DRAM become pages
    // of a neighbour island's tensor, DRAM to DRAM by the link cores' page CPUs; the
    // rank posts a line a run and goes on (km::gather_pages), waits once at the end
    // (km::gather_wait); a keeper on the far island waits for them (km::gathered).

    // Pages posted by km::gather_pages and not yet waited for, the last island posted to.
    struct Gather {
        uint32_t pages;
        uint32_t island;
    };

    // A keeper: rank `rank` of the far island (its index there) and its port `port`
    // (0 .. KM_PORTS - 1), which counts a gather's pages once all are written there.
    constexpr uint32_t keeper(uint32_t rank, uint32_t port) {
        return (rank + 1) | port << KM_KEEPER_PORT_SHIFT;
    }

    // Where page `page` of tensor `tensor` (its reference word) is on island `island`:
    // a copy's there, a split tensor's where page_at says (another island's: address 0).
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

    // Reader or writer: pages first .. first + pages - 1 of tensor `tensor` (its
    // reference word; pages this island holds, its part or its copy) become the same
    // pages of tensor `into` (0: `tensor`, a copy) on island `to`, a neighbour (hops 1):
    // a put line a run from this island's DRAM (KM_ASK_DRAM), which the page CPUs read
    // and send; returns once the lines are written (a CPU's third line in flight waits
    // for its first's confirmation). `keeper` (km::keeper, 0: none) counts them at its
    // port once written there. False: gave up (recorded).
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
            const PageAt from = gather_place(tensor, p, here);
            const PageAt at = gather_place(target, p, to);
            if (from.address == 0 || at.address == 0) {
                return give_up(kGaveBadItem);  // not here, or not there
            }
            uint32_t run = page_run(tensor, p, pages - k, from);
            run = page_run(target, p, run, at);
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

    // Reader or writer, a keeper: waits until its port `port` has counted `pages` pages
    // gathered into this island (km::keeper), then takes them off it. False: gave up.
    template <typename Kernel>
    inline bool gathered(
        const Kernel& core, const km_command_t& command, uint32_t port, uint32_t pages
    ) {
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        if (port >= KM_PORTS) {
            core.gave_up(word, kGaveBadItem, kWhereAsk, slot(command), command.sequence);
            return false;
        }
        volatile tt_l1_ptr uint32_t* words = port_words(core, port);
        const uint32_t code = core.poll_at(
            kWhereAsk,
            [&] {
                invalidate_l1_cache();
                return static_cast<int32_t>(words[KM_PORT_COUNT] - pages) >= 0;
            },
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            core.gave_up(word, code, kWhereAsk, slot(command), command.sequence);
            return false;
        }
        words[KM_PORT_COUNT] -= pages;
        return true;
    }

}  // namespace km
