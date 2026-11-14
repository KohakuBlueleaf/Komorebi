// A kernel's read and write of a tensor's pages wherever they live: the local read or
// write for the pages its island holds; for pages another island holds, a line into
// its island's link core 0. A get: its link 0 CPU sends it to the holder's link core
// 0, whose page CPU pushes the pages into this core's L1 and counts them at its port.
// A put: its page CPU reads the pages from this core into the holder's DRAM, and the
// holder's link core 0 counts them at this core's port once written. On one island:
// exactly the local read or write. A kernel takes a tensor as a km::Tensor.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_placement.hpp"
#include "km_port.hpp"
#include "km_transfer.hpp"

namespace km {

    constexpr uint32_t kAskNoAnswer = 0xFFFFFFFFu;  // the atomic's answer not yet in

    // A queue of lines in link core 0's link area (offsets): the count its island's
    // cores reserve lines with, the count it took, its lines; and the asking CPU's own
    // line and its two answer words, on its own core.
    struct AskQueue {
        uint32_t tail;
        uint32_t taken;
        uint32_t lines;
        uint32_t own;
        uint32_t returned;
        uint32_t read;
    };

    // The get queue (the reader's) and the put queue of CPU `role`.
    constexpr AskQueue kAskGets = {
        KM_ASK_TAIL,
        KM_ASK_TAKEN,
        KM_ASK_LINES,
        KM_ASK_OWN,
        KM_ASK_RETURNED,
        KM_ASK_READ
    };

    constexpr AskQueue ask_puts(uint32_t role) {
        const uint32_t own = KM_PUT_OWN + (role == kRoleWriter ? 0 : KM_PUT_OWN_BYTES);
        return {KM_PUT_TAIL, KM_PUT_TAKEN, KM_PUT_LINES, own, own + 0x40, own + 0x50};
    }

    // The get queue of CPU `role`: the writer's own line and answer words at its
    // area's ask lines, which only link core 0 holds lines in.
    constexpr AskQueue ask_gets(uint32_t role) {
        if (role != kRoleWriter) {
            return kAskGets;
        }
        constexpr uint32_t own = KM_ASK_LINES;
        return {KM_ASK_TAIL, KM_ASK_TAKEN, KM_ASK_LINES, own, own + 0x40, own + 0x50};
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

    // One line into `queue` for pages first .. first + pages - 1 of the tensor at bank
    // address `address` (`bytes` aligned page bytes), held by island `holder`, and L1
    // `at` of this core, counted at port `port`, for command (`slot`, `sequence`): a
    // line reserved on link core 0 with a local atomic. False: gave up (recorded).
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
        uint32_t port
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t area = view.link_area();
        const uint32_t named = slot | holder << 8;
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        const auto give_up = [&](uint32_t code) {
            core.gave_up(word, code, kWhereAsk, named, sequence);
            return false;
        };
        if (
            area == 0 || view.hops(holder) != 1 || port >= KM_PORTS || bytes == 0 ||
            bytes % 16 != 0 || bytes > KM_PAGES_PACKET_MAX || at % 16 != 0
        ) {
            return give_up(kGaveBadItem);  // two islands a link apart, whole packets
        }
        // link core 0: the core after the island's ranks
        const uint32_t link = view.base() + view.ranks();
        const uint32_t limit = core.args().wait_limit;
        volatile tt_l1_ptr uint32_t* returned = l1(area + queue.returned);
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
            area + queue.returned
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
        // the line is free once link core 0 took the one KM_ASK_SLOTS before
        volatile tt_l1_ptr uint32_t* taken = l1(area + queue.read);
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
                    core.core_noc(link, area + queue.taken), area + queue.read, 16
                );
                reading = true;
                return false;
            },
            limit
        );
        if (code != kGaveNone) {
            return give_up(code);
        }
        volatile tt_l1_ptr uint32_t* line = l1(area + queue.own);
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
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        noc_async_write(
            area + queue.own,
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
        if (port >= KM_PORTS) {
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
        uint32_t to,
        uint32_t port
    ) {
        return (
            ask_line(
                reader, command, kAskGets, holder, address, bytes, first, pages, to, port
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
        uint32_t to,
        uint32_t port
    ) {
        const uint32_t here = island();
        const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
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
        uint32_t to,
        uint32_t port
    ) {
        return (
            read_pages_at(
                core, slot(command), command.sequence, args, tensor, first, pages, to, port
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
        uint32_t from,
        uint32_t port
    ) {
        const uint32_t here = island();
        const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
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
        uint32_t from,
        uint32_t port
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
                from,
                port
            )
        );
    }

    // The read of a Tensor's pages: built for one island, the local reads alone
    // (`port` unused); else read_pages_at by its word.
    template <typename Kernel, uint32_t Base>
    inline bool read_pages_at(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const Tensor<Base>& tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t to,
        uint32_t port
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
                    core,
                    slot,
                    sequence,
                    Tensor<Base>::kArgs,
                    tensor.word,
                    first,
                    pages,
                    to,
                    port
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
        uint32_t to,
        uint32_t port
    ) {
        return (
            read_pages_at(
                core, slot(command), command.sequence, tensor, first, pages, to, port
            )
        );
    }

    // The write of a Tensor's pages: built for one island, the local writes alone
    // (`port` unused); else write_pages_at by its word.
    template <typename Kernel, uint32_t Base>
    inline bool write_pages_at(
        const Kernel& core,
        uint32_t slot,
        uint32_t sequence,
        const Tensor<Base>& tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t from,
        uint32_t port
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
                    from,
                    port
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
        uint32_t from,
        uint32_t port
    ) {
        return (
            write_pages_at(
                core, slot(command), command.sequence, tensor, first, pages, from, port
            )
        );
    }

    // A copy's port for the pages it asks for or puts (the last port).
    constexpr uint32_t kCopyPort = KM_PORTS - 1;

    // Whether a copy's side names, by its reference word, a tensor of which another
    // island holds pages (its pages then go page by page by their holder).
    inline bool copy_placed(const km_copy_side_t& side) {
        return side.bank == KM_COPY_INTERLEAVED && held_elsewhere(side.address);
    }

    // Whether a copy has a side with pages another island holds.
    inline bool copy_crosses(const km_copy_t& c) {
        invalidate_l1_cache();  // the table may have grown since the last read
        return copy_placed(c.from) || copy_placed(c.to);
    }

    // Reader: rank `rank` of `count` copies its pages a chunk at a time, a side whose
    // tensor is placed over islands page by page by its holder: this island's pages
    // read or written as copy_pages does, each run another island holds one get
    // (into the chunk) or put (from it) line, counted at kCopyPort and waited for
    // before the chunk goes on. False: gave up (recorded).
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
            const uint32_t named = slot(command) | holder << 8;
            return port_wait_at(reader, command, kCopyPort, pages, kWhereAsk, named);
        };
        const uint32_t here = island();
        const bool from = copy_placed(c.from);
        const bool to = copy_placed(c.to);
        const Partition share = partition(c.pages, rank, count);
        const uint32_t chunk_pages = c.chunk / c.page;
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
                    kCopyPort
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
                kAskGets,
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
