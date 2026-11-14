// A kernel's read and write of a tensor's pages wherever they live: the local read or
// write for the pages its island holds; for pages another island holds, a line into
// its island's link core 0. A get: its link 0 CPU sends it to the holder's link core
// 0, whose page CPU pushes the pages into this core's L1 and counts them at its port.
// A put: its page CPU reads the pages from this core into the holder's DRAM, and the
// holder's link core 0 counts them at this core's page port (KM_PORT_PAGES, no kernel
// names it) once written. On one island: the local read or write. Takes a km::Tensor.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_pages.hpp"
#include "km_placement.hpp"
#include "km_port.hpp"
#include "km_transfer.hpp"

namespace km {

    constexpr uint32_t kAskNoAnswer = 0xFFFFFFFFu;  // the atomic's answer not yet in

    // A queue of lines in link core 0's link area (offsets): the count its island's
    // cores reserve lines with, the count it took (`cursors`: a count each page CPU
    // took, the page CPUs in the eighth word), its lines; and the asking CPU's own
    // line and its two answer words, in its own core's own words (KM_OWN_*).
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

    // One line into `queue` for pages first .. first + pages - 1 of the tensor at bank
    // address `address` (`bytes` aligned page bytes), held by island `holder`, and L1
    // `at` of this core (`dram`, KM_ASK_DRAM's: bank address `at` of this island's
    // DRAM, from its page `dram_first`), counted at port `port`, for command (`slot`,
    // `sequence`): a line reserved on link core 0 with a local atomic. A put from L1,
    // when the page CPUs have rings: its pages written into them here after the line
    // (pages_into_rings; `at` free on return). False: gave up (recorded).
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
        PutRun* run = nullptr
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
        // a put: this CPU's own line (its owner: core, CPU, parity) on the parity after
        // its last, once the far island confirmed the line it last sent on it (its
        // block of the core's own words)
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
        // a put: its place in the put queue free (every page CPU took the line
        // KM_QUEUE_LINES before), read with the rings (put_rings_read); a get: the
        // line free once link core 0 took the one KM_ASK_SLOTS before
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

    // Reader or writer: names a run of puts, pages first .. first + pages - 1 at bank
    // address `address` of island `holder` (`bytes` aligned page bytes), with one put
    // line, and returns; its bytes follow by put_piece, a piece at a time, as they are
    // made; put_wait(run) counts it. Where the page CPUs have rings (the link cores'
    // chunk large enough), the line names the whole run and each piece goes into the
    // rings; else nothing is sent here and each piece is a put line of its own,
    // waited for. False: gave up (recorded).
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
        const uint32_t build = (  // its CPU's block of the own words
            mine + (core.role() == kRoleWriter ? KM_OWN_CPU_BYTES : 0) + KM_OWN_BUILD
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
            if (
                !ask_line(
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
                )
            ) {
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

    // Whether a copy's side names, by its reference word, a tensor of which another
    // island holds pages (its pages then go page by page by their holder).
    inline bool copy_placed(const km_copy_side_t& side) {
        return side.bank == KM_COPY_INTERLEAVED && held_elsewhere(side.address);
    }

    // The copy with each side a tensor's reference word names, and of which no other
    // island holds pages, at its bank address on this island (its pages at their own
    // indices there); a side placed over islands keeps its word (copy_across).
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

    // Reader: rank `rank` of `count` copies its pages a chunk at a time, a side whose
    // tensor is placed over islands page by page by its holder: this island's pages
    // read or written as copy_pages does, each run another island holds one get
    // (into the chunk) or put (from it) line, counted at the reader's page port and
    // waited for before the chunk goes on. When one side is placed and the other is
    // not, each run another island holds is one line of the whole run straight between
    // the two DRAMs, all in flight, waited for once at the end. False: gave up
    // (recorded).
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
        // One side of `n` pages from index `start` of the share: local pages by
        // `local(noc, k)`, runs held elsewhere by one line each from L1 `at`. The
        // pages asked for or put, and the island of the last run.
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

}  // namespace km
