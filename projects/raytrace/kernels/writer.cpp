// Ray tracing writer: each output page to its place in DRAM, then the profile page
// (compute's counters, then the reader's and its own) to a4.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "timed.hpp"
#include "dir_noc.hpp"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/km_ask.hpp"
#include "raytrace.h"

// The directory's pages, then the profile pages (km::Tensor): compile arguments from 0.
using Directories = km::Tensor<0>;
using Profiles = km::Tensor<Directories::next_compile_time_args_offset()>;

namespace {

    // Hit pages: a nearest hit's with the attributes; the probe's as any hit's.
    uint32_t output_page_bytes(const km_command_t& command, uint32_t handler) {
        const uint32_t mode = command.a3;
        if (handler == kHandlerTrace || handler == kHandlerTreelets) {
            return mode == kModeAny ? kHitAnyBytes : kHitBytes;
        }
        if (handler == kHandlerProbe) {
            return kHitAnyBytes;
        }
        if (handler == kHandlerMath) {
            return kMaOutVectors * kVec * 4;
        }
        return handler == kHandlerArith ? kArOutVectors * kVec * 4 : kRecipPerPage * 4;
    }

    uint32_t output_packets_per_page(uint32_t handler) {
        const bool hits = (
            handler == kHandlerTrace || handler == kHandlerTreelets ||
            handler == kHandlerProbe
        );
        return hits ? kPagePackets : 1;
    }

    // The front output page to page g of a stream, popped; the wait for it is added to
    // `wait`.
    bool write_page(
        km::Writer& writer,
        uint32_t base_address,
        uint32_t bytes,
        uint32_t packets_per_page,
        uint32_t packet,
        uint32_t& wait
    ) {
        const uint32_t started = km::wall();
        if (!writer.wait_front(kCbOut, 1)) {
            return false;
        }
        wait += km::wall() - started;
        const InterleavedAddrGen<true> pages = {
            .bank_base_address = base_address, .page_size = bytes * packets_per_page
        };
        noc_async_write(
            get_read_ptr(kCbOut),
            get_noc_addr(
                packet / packets_per_page, pages, (packet % packets_per_page) * bytes
            ),
            bytes
        );
        if (!timed::writes_sent(writer, km::kStWriter)) {
            return false;
        }
        cb_pop_front(kCbOut, 1);
        return true;
    }

}  // namespace

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        const uint32_t output_address = command.a2;
        const uint32_t profile_address = command.a4, policy_word = command.a9;
        if (
            handler != kHandlerTrace && handler != kHandlerArith &&
            handler != kHandlerRecip && handler != kHandlerTreelets &&
            handler != kHandlerProbe && handler != kHandlerMath && !is_stage(handler)
        ) {
            return true;
        }
        const uint32_t started = km::wall();
        const uint32_t directory_space = get_write_ptr(kCbDirWriter);
        volatile tt_l1_ptr uint32_t* directory = km::l1(directory_space);
        if (
            !read_directory<Directories>(writer, command, handler, directory_space) ||
            !timed::reads_done(writer, km::kStWriter)
        ) {
            return true;
        }
        const uint32_t first = directory[kDirFirst], count = directory[kDirCount];
        uint32_t wait = 0, write = 0;
        if (is_stage(handler)) {
            // Each packet's page of every output stream, in stream order.
            const Stage stage = stage_of(handler);
            for (uint32_t packet = 0; packet < count; ++packet) {
                for (
                    uint32_t stream = stage.inputs;
                    stream < stage.inputs + stage.outputs;
                    ++stream
                ) {
                    const uint32_t page_started = km::wall();
                    const uint32_t bytes = stage.vectors[stream] * kVec * 4;
                    if (!write_page(
                        writer,
                        stream_arg(command, stream),
                        bytes,
                        kPagePackets,
                        first + packet,
                        wait
                    )) {
                        return true;
                    }
                    write += km::wall() - page_started;
                }
            }
        } else {
            const uint32_t bytes = output_page_bytes(command, handler);
            const uint32_t packets_per_page = output_packets_per_page(handler);
            for (uint32_t packet = 0; packet < count; ++packet) {
                const uint32_t page_started = km::wall();
                if (!write_page(
                    writer,
                    output_address,
                    bytes,
                    packets_per_page,
                    first + packet,
                    wait
                )) {
                    return true;
                }
                write += km::wall() - page_started;
            }
        }
        write -= wait;
        // The profile page: compute filled words [0, 32); reader and writer follow.
        if (!writer.wait_front(kCbOut, 1)) {
            return true;
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* profile = km::l1(get_read_ptr(kCbOut));
        volatile tt_l1_ptr uint32_t* reader_words = km::l1(get_write_ptr(kCbDirReader));
        for (uint32_t counter = 0; counter < kRdCounters; ++counter) {
            profile[kProfReader + counter] = reader_words[kRdWords + counter];
        }
        profile[kProfWriter + 0] = wait;
        profile[kProfWriter + 1] = write;
        profile[kProfWriter + 2] = km::wall() - started;
        profile[kProfWriter + 3] = count;
        if (!km::write_pages(
            writer,
            command,
            Profiles(profile_address),
            writer.me(),
            1,
            get_read_ptr(kCbOut)
        )) {
            return true;
        }
        const uint32_t bank = (
            handler == kHandlerTreelets ? policy_word >> kBankShift : 0
        );
        if (bank != 0 && directory[kDirCost] != 0) {
            noc_async_write(
                get_read_ptr(kCbOut) + 4 * kProfCost,
                get_noc_addr_from_bank_id<true>(bank - 1, directory[kDirCost]),
                kCostBytes
            );
        }
        if (!timed::writes_sent(writer, km::kStWriter)) {
            return true;
        }
        cb_pop_front(kCbOut, 1);
        return true;
    });
}
