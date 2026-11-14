// A tensor read in pages of the command's size, and a range of one DRAM bank, across
// islands, reader: PAGES and BYTES (km::SizedTensor), RANGE (km::BankRange).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "forms.h"

// What a handler counted: words checked and wrong, a call given up, cycles in calls.
struct Counts {
    uint32_t checked = 0;
    uint32_t wrong = 0;
    uint32_t failed = 0;
    uint32_t cycles = 0;
};

// The buffer's first 64-byte line: pages and ranges from there, then the report.
uint32_t forms_buffer() {
    return (get_write_ptr(kCbForms) + 63) & ~63u;
}

// The words of `bytes` at L1 `at` xor kFormsMark, in place.
void mark(uint32_t at, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* words = km::l1(at);
    for (uint32_t w = 0; w < bytes / 4; w++) {
        words[w] = words[w] ^ kFormsMark;
    }
}

// PAGES (forms.h): the rank's pages read, checked, marked and written.
void pages(km::Reader& reader, const km_command_t& command, uint32_t rank, Counts& n) {
    const km::SizedTensor tensor(command.a0, command.a1);
    const uint32_t bytes = command.a1, count = command.a3;
    const uint32_t first = rank * count, data = forms_buffer();
    if (bytes * count > kFormsData) {
        n.failed = 1;
        return;
    }
    uint32_t started = km::wall();
    n.failed = !km::read_pages(reader, command, tensor, first, count, data);
    noc_async_read_barrier();
    n.cycles += km::wall() - started;
    if (n.failed) {
        return;
    }
    invalidate_l1_cache();
    for (uint32_t k = 0; k < count; k++) {
        const uint32_t page = first + k;
        const volatile tt_l1_ptr uint32_t* got = km::l1(data + k * bytes);
        for (uint32_t w = 0; w < bytes / 4; w++) {
            const uint32_t at = page / kFormsBanks * bytes + 4 * w;
            n.wrong += got[w] != forms_word(page % kFormsBanks, at) ? 1 : 0;
        }
        n.checked += bytes / 4;
    }
    mark(data, bytes * count);
    started = km::wall();
    n.failed = (
        !km::write_pages(reader, command, tensor, command.a2 + first, count, data)
    );
    noc_async_write_barrier();
    n.cycles += km::wall() - started;
}

// BYTES (forms.h): the rank's items read, checked, marked and put.
void bytes(km::Reader& reader, const km_command_t& command, uint32_t rank, Counts& n) {
    const km::SizedTensor tensor(command.a0, command.a1);
    const uint32_t page = command.a1, sources = command.a2, count = command.a3;
    const uint32_t items = page / kFormsItem, data = forms_buffer();
    if (count * kFormsItem > kFormsData || sources == 0 || items == 0) {
        n.failed = 1;
        return;
    }
    uint32_t started = km::wall();
    uint32_t asked = 0;
    for (uint32_t k = 0; k < count && !n.failed; k++) {
        const uint32_t s = (rank * count + k) % sources;
        n.failed = (
            !km::ask_bytes(
                reader,
                command,
                tensor,
                s / items,
                s % items * kFormsItem,
                kFormsItem,
                data + k * kFormsItem,
                asked
            )
        );
    }
    n.failed = n.failed || !km::bytes_wait(reader, command, asked);
    n.cycles += km::wall() - started;
    if (n.failed) {
        return;
    }
    invalidate_l1_cache();
    for (uint32_t k = 0; k < count; k++) {
        const uint32_t s = (rank * count + k) % sources;
        const uint32_t p = s / items;
        const volatile tt_l1_ptr uint32_t* got = km::l1(data + k * kFormsItem);
        for (uint32_t w = 0; w < kFormsItem / 4; w++) {
            const uint32_t at = p / kFormsBanks * page + s % items * kFormsItem + 4 * w;
            n.wrong += got[w] != forms_word(p % kFormsBanks, at) ? 1 : 0;
        }
        n.checked += kFormsItem / 4;
    }
    mark(data, count * kFormsItem);
    started = km::wall();
    for (uint32_t k = 0; k < count && !n.failed; k++) {
        const uint32_t g = rank * count + k;
        n.failed = (
            !km::put_bytes(
                reader,
                command,
                tensor,
                command.a6 + g / items,
                g % items * kFormsItem,
                kFormsItem,
                data + k * kFormsItem
            )
        );
    }
    n.failed = n.failed || !km::put_wait(reader, command);
    n.cycles += km::wall() - started;
}

// RANGE (forms.h): the rank's two pages and one byte range read, checked, marked and
// written back.
void range(km::Reader& reader, const km_command_t& command, uint32_t rank, Counts& n) {
    const km::BankRange region(command.a0, command.a1, command.a2);
    const uint32_t bytes = command.a2, first = 2 * rank;
    const uint32_t data = forms_buffer();
    const uint32_t part = data + 2 * bytes + kFormsRangeOffset;  // offset mod 64
    if (2 * bytes + 64 + kFormsRangeBytes > kFormsData) {
        n.failed = 1;
        return;
    }
    uint32_t started = km::wall();
    n.failed = (
        !km::read_pages(reader, command, region, first, 2, data) ||
        !km::read_bytes(
            reader, command, region, first, kFormsRangeOffset, kFormsRangeBytes, part
        )
    );
    noc_async_read_barrier();
    n.cycles += km::wall() - started;
    if (n.failed) {
        return;
    }
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint32_t* got = km::l1(data);
    for (uint32_t w = 0; w < 2 * bytes / 4; w++) {
        n.wrong += got[w] != forms_range_word(first * bytes + 4 * w) ? 1 : 0;
    }
    const volatile tt_l1_ptr uint32_t* piece = km::l1(part);
    for (uint32_t w = 0; w < kFormsRangeBytes / 4; w++) {
        const uint32_t at = first * bytes + kFormsRangeOffset + 4 * w;
        n.wrong += piece[w] != forms_range_word(at) ? 1 : 0;
    }
    n.checked += 2 * bytes / 4 + kFormsRangeBytes / 4;
    mark(data, 2 * bytes);
    started = km::wall();
    n.failed = (
        !km::write_pages(reader, command, region, first, 1, data) ||
        !km::write_bytes(reader, command, region, first + 1, 0, bytes, data + bytes)
    );
    noc_async_write_barrier();
    n.cycles += km::wall() - started;
}

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();  // the reports'
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        const uint32_t rank = reader.me() - km::first(command);
        Counts n;
        if (handler == kHandlerPages) {
            pages(reader, command, rank, n);
        } else if (handler == kHandlerBytes) {
            bytes(reader, command, rank, n);
        } else if (handler == kHandlerRange) {
            range(reader, command, rank, n);
        } else {
            return;
        }
        const uint32_t out = forms_buffer() + kFormsData;
        volatile tt_l1_ptr uint32_t* words = km::l1(out);
        const uint32_t head[8] = {
            kFormsMagic,
            rank,
            handler,
            n.checked,
            n.wrong,
            n.failed,
            n.cycles,
            km::island()
        };
        for (uint32_t w = 0; w < 8; w++) {
            words[w] = head[w];
        }
        km::write_pages(reader, command, args, command.a4, command.a5 + rank, 1, out);
        noc_async_write_barrier();
    });
}
