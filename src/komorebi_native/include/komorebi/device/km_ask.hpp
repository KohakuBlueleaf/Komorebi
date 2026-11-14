// A kernel's read and write of a tensor's pages wherever they live: another island's
// pages are asked for through link core 0 (a write of one ends the command).
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_placement.hpp"
#include "km_port.hpp"

namespace km {

    constexpr uint32_t kAskNoAnswer = 0xFFFFFFFFu;  // the atomic's answer not yet in

    // A tensor as the page calls take it: accessor, placement and bank address, from
    // compile argument `Base` on (the host's tensor_args()).
    template <uint32_t Base>
    struct Tensor {
        static constexpr auto kArgs = TensorAccessorArgs<Base>();
        using Where = Placement<kArgs.next_compile_time_args_offset()>;
        using Accessor = decltype(TensorAccessor(kArgs, 0u));
        // Every page on this core's island, known when the kernel is built.
        static constexpr bool kLocal = Where::kIslands == 1 || Where::kCopy;

        Accessor accessor;
        uint32_t address;

        explicit Tensor(uint32_t address) :
            accessor(TensorAccessor(kArgs, address)), address(address) {
        }

        static constexpr uint32_t next_compile_time_args_offset() {
            return Where::next_compile_time_args_offset();
        }

        uint32_t page_bytes() const {
            return accessor.get_aligned_page_size();
        }

        uint32_t island_of(uint32_t page) const {
            return km::island_of(Where(), page);
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

    // The gave-up word of this kernel's CPU: the reader's or the writer's.
    inline uint32_t gave_word(const Core& core) {
        return core.role() == kRoleReader ? kStReader : kStWriter;
    }

    // The asking CPU's line and answer words in its link area: the reader's at
    // KM_ASK_OWN, the writer's at the ask lines (only link core 0 holds lines).
    struct AskWords {
        uint32_t own;
        uint32_t returned;
        uint32_t read;
    };

    inline AskWords ask_words(const Core& core, uint32_t area) {
        if (core.role() == kRoleReader) {
            return {area + KM_ASK_OWN, area + KM_ASK_RETURNED, area + KM_ASK_READ};
        }
        return {
            area + KM_ASK_LINES, area + KM_ASK_LINES + 64, area + KM_ASK_LINES + 80
        };
    }

    // One ask line to island `holder` for `pages` pages from `first` into L1 `to`,
    // counted at `port`, for command (slot, sequence). False: gave up (recorded).
    inline bool ask_pages_at(
        const Core& core,
        uint32_t slot,
        uint32_t sequence,
        uint32_t holder,
        uint32_t address,
        uint32_t bytes,
        uint32_t first,
        uint32_t pages,
        uint32_t to,
        uint32_t port
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t area = view.link_area();
        const uint32_t named = slot | holder << 8;
        const auto give_up = [&](uint32_t code) {
            core.gave_up(gave_word(core), code, kWhereAsk, named, sequence);
            return false;
        };
        if (
            area == 0 || view.hops(holder) != 1 || port >= KM_PORTS || bytes == 0 ||
            bytes % 16 != 0 || bytes > KM_PAGES_PACKET_MAX || to % 16 != 0
        ) {
            return give_up(kGaveBadItem);  // two islands a link apart, whole packets
        }
        const AskWords words = ask_words(core, area);
        const uint32_t link = view.ranks();  // link core 0: the core after the ranks
        const uint32_t limit = core.args().wait_limit;
        volatile tt_l1_ptr uint32_t* returned = l1(words.returned);
        returned[0] = kAskNoAnswer;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            core.core_noc(link, area + KM_ASK_TAIL),
            NOC_UNICAST_WRITE_VC,
            1,
            31,
            false,
            false,
            words.returned
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
        volatile tt_l1_ptr uint32_t* taken = l1(words.read);
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
                    core.core_noc(link, area + KM_ASK_TAKEN), words.read, 16
                );
                reading = true;
                return false;
            },
            limit
        );
        if (code != kGaveNone) {
            return give_up(code);
        }
        volatile tt_l1_ptr uint32_t* line = l1(words.own);
        for (uint32_t w = 0; w < 16; w++) {
            line[w] = 0;
        }
        line[KM_ASK_TENSOR] = address;
        line[KM_ASK_PAGE] = bytes;
        line[KM_ASK_FIRST] = first;
        line[KM_ASK_PAGES] = pages;
        line[KM_ASK_CORE] = core.control(KM_CONTROL_CORES)[core.args().me];
        line[KM_ASK_TO] = to;
        line[KM_ASK_COUNT] = (
            core.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE +
            KM_PORT_COUNT * 4
        );
        line[KM_ASK_HOLDER] = holder;
        line[KM_ASK_FROM] = view.here();
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        noc_async_write(
            words.own,
            core.core_noc(link, area + KM_ASK_LINES + number % KM_ASK_SLOTS * 64),
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

    // Reader: ask_pages_at for `command`.
    inline bool ask_pages(
        const Core& core,
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
            ask_pages_at(
                core,
                slot(command),
                command.sequence,
                holder,
                address,
                bytes,
                first,
                pages,
                to,
                port
            )
        );
    }

    // Waits until port `port` has counted `wanted` asked pages, then clears it; a
    // give-up names the island asked. False: gave up (recorded).
    inline bool ask_wait(
        const Core& core,
        uint32_t slot,
        uint32_t sequence,
        uint32_t port,
        uint32_t wanted,
        uint32_t holder
    ) {
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
            core.gave_up(
                gave_word(core), code, kWhereAsk, slot | holder << 8, sequence
            );
            return false;
        }
        words[KM_PORT_COUNT] = 0;
        words[KM_PORT_WANT] = 0;
        return true;
    }

    // Pages first .. into L1 `to` on: this island's read (the caller's barrier waits),
    // each run another island holds asked for and waited for at `port`. False: gave up.
    template <uint32_t Base, typename Accessor>
    inline bool read_pages_at(
        const Core& core,
        uint32_t slot,
        uint32_t sequence,
        const Placement<Base>& placement,
        const Accessor& tensor,
        uint32_t address,
        uint32_t first,
        uint32_t pages,
        uint32_t to,
        uint32_t port
    ) {
        const uint32_t here = island();
        const uint32_t bytes = tensor.get_aligned_page_size();
        uint32_t asked = 0, holder = here;
        for (uint32_t k = 0; k < pages;) {
            const uint32_t at = island_of(placement, first + k);
            if (at == here) {
                noc_async_read_page(first + k, tensor, to + k * bytes);
                k++;
                continue;
            }
            uint32_t run = 1;
            while (k + run < pages && island_of(placement, first + k + run) == at) {
                run++;
            }
#if !defined(KM_ASKS_BY_HOST)
            if (!ask_pages_at(
                core,
                slot,
                sequence,
                at,
                address,
                bytes,
                first + k,
                run,
                to + k * bytes,
                port
            )) {
                return false;
            }
#endif
            asked += run;
            holder = at;
            k += run;
        }
        // asks by the host: the host's send brings them to `to` and counts them here
        return asked == 0 || ask_wait(core, slot, sequence, port, asked, holder);
    }

    // read_pages_at for `command`.
    template <uint32_t Base, typename Accessor>
    inline bool read_pages(
        const Core& core,
        const km_command_t& command,
        const Placement<Base>& placement,
        const Accessor& tensor,
        uint32_t address,
        uint32_t first,
        uint32_t pages,
        uint32_t to,
        uint32_t port
    ) {
        return (
            read_pages_at(
                core,
                slot(command),
                command.sequence,
                placement,
                tensor,
                address,
                first,
                pages,
                to,
                port
            )
        );
    }

    // read_pages_at of a Tensor; a tensor whose island holds every page: the local
    // reads alone (`port` unused).
    template <uint32_t Base>
    inline bool read_pages_at(
        const Core& core,
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
                    typename Tensor<Base>::Where(),
                    tensor.accessor,
                    tensor.address,
                    first,
                    pages,
                    to,
                    port
                )
            );
        }
    }

    // The read of a Tensor's pages for `command`: read_pages_at.
    template <uint32_t Base>
    inline bool read_pages(
        const Core& core,
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

    // Pages first .. of a Tensor from L1 `from` on, written locally (`port`: for the
    // write across); a page another island holds: kGaveNotAcross, nothing written.
    template <uint32_t Base>
    inline bool write_pages_at(
        const Core& core,
        uint32_t slot,
        uint32_t sequence,
        const Tensor<Base>& tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t from,
        uint32_t port
    ) {
        if constexpr (!Tensor<Base>::kLocal) {
            const uint32_t here = island();
            for (uint32_t k = 0; k < pages; k++) {
                const uint32_t at = tensor.island_of(first + k);
                if (at != here) {
                    core.gave_up(
                        gave_word(core),
                        kGaveNotAcross,
                        kWhereAsk,
                        slot | at << 8,
                        sequence
                    );
                    return false;
                }
            }
        }
        (void)port;
        const uint32_t bytes = tensor.page_bytes();
        for (uint32_t k = 0; k < pages; k++) {
            noc_async_write_page(first + k, tensor.accessor, from + k * bytes);
        }
        return true;
    }

    // The write of a Tensor's pages for `command`: write_pages_at.
    template <uint32_t Base>
    inline bool write_pages(
        const Core& core,
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

}  // namespace km
