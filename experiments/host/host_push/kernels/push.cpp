// Pushes bytes from a core into the host's memory channel with the PCIe-aware NoC
// calls only (mode 0 a pattern, mode 1 pages of a DRAM tensor); checked before a write.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "push.h"

namespace {

    inline uint64_t wall() {
        const uint32_t lo = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
        );
        const uint32_t hi = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H)
        );
        return static_cast<uint64_t>(hi) << 32 | lo;
    }

    // Polls ready() for at most kLimit cycles; false on a timeout.
    template <typename Ready>
    inline bool poll(Ready ready) {
        const uint64_t t0 = wall();
        for (;;) {
            if (ready()) {
                return true;
            }
            if (wall() - t0 > kLimit) {
                return false;
            }
        }
    }

    inline bool writes_flushed() {
        return poll([] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        });
    }

    inline bool reads_flushed() {
        return poll([] {
            return ncrisc_noc_reads_flushed(noc_index);
        });
    }

}  // namespace

void kernel_main() {
#ifdef KM_PUSH_DRAM
    constexpr auto src_args = TensorAccessorArgs<0>();
#endif
    volatile uint32_t* r = (
        reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(0))
    );
    const uint32_t mode = get_arg_val<uint32_t>(1);
    const uint32_t src = get_arg_val<uint32_t>(2);
    const uint32_t first_page = get_arg_val<uint32_t>(3);
    const uint32_t bytes = get_arg_val<uint32_t>(4);
    const uint32_t page_bytes = get_arg_val<uint32_t>(5);
    const uint32_t offset = get_arg_val<uint32_t>(6);
    const uint32_t chunk = get_arg_val<uint32_t>(7);
    const uint32_t nonce = get_arg_val<uint32_t>(8);
    const uint32_t seq_offset = get_arg_val<uint32_t>(9);
    const uint32_t core = get_arg_val<uint32_t>(10);
    const uint32_t buffer = get_arg_val<uint32_t>(11);
    for (uint32_t i = 0; i < 16; ++i) {
        r[i] = 0;
    }

    // The bounds: nothing is issued unless all of them hold.
    const uint64_t data_end = static_cast<uint64_t>(offset) + bytes;
    const uint64_t seq_end = static_cast<uint64_t>(seq_offset) + 16;
    bool ok = bytes > 0 && chunk > 0 && (bytes % 16) == 0 && (chunk % 16) == 0;
    ok = ok && (offset % 16) == 0 && (seq_offset % 16) == 0;
    ok = ok && data_end <= kChannelLimit && seq_end <= kChannelLimit;
    ok = ok && (seq_end <= offset || seq_offset >= data_end);
    ok = ok && 2ull * chunk + 16 <= buffer;
#ifdef KM_PUSH_DRAM
    ok = ok && mode == 1 && page_bytes > 0 && (page_bytes % 16) == 0;
    ok = ok && (bytes % page_bytes) == 0 && (chunk % page_bytes) == 0;
#else
    ok = ok && mode == 0;
#endif
    if (!ok) {
        r[kStatus] = kBadArguments;
        return;
    }

    const uint32_t base = get_write_ptr(kCbBuffer);
    const uint32_t seq_slot = base + 2 * chunk;
    const uint64_t pcie = (
        static_cast<uint64_t>(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y))
    );
    const uint32_t chunks = (bytes + chunk - 1) / chunk;
    uint32_t status = kDone;
    uint32_t pushed = 0;
    const uint64_t t0 = wall();

#ifdef KM_PUSH_DRAM
    const auto in = TensorAccessor(src_args, src);
    const uint32_t chunk_pages = chunk / page_bytes;
    const uint32_t pages = bytes / page_bytes;
    // Reads chunk k into its buffer; false on a timeout.
    auto read_chunk = [&](uint32_t k) {
        const uint32_t at = base + (k & 1) * chunk;
        const uint32_t from = k * chunk_pages;
        const uint32_t n = (pages - from < chunk_pages) ? pages - from : chunk_pages;
        for (uint32_t p = 0; p < n; ++p) {
            noc_async_read_page(first_page + from + p, in, at + p * page_bytes);
        }
        return reads_flushed();
    };
    if (!read_chunk(0)) {
        status = kReadTimeout;
    }
#endif

    if (status == kDone) {
        noc_async_write_set_pcie_state(pcie | offset);
        for (uint32_t k = 0; k < chunks; ++k) {
            const uint32_t at = base + (k & 1) * chunk;
            const uint32_t n = (bytes - k * chunk < chunk) ? bytes - k * chunk : chunk;
#ifndef KM_PUSH_DRAM
            volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(at);
            const uint32_t first_word = (offset + k * chunk) / 4;
            for (uint32_t i = 0; i < n / 4; ++i) {
                w[i] = pattern_word(first_word + i, nonce);
            }
#endif
            noc_async_write_with_state(at, offset + k * chunk, n);
            pushed += n;
#ifdef KM_PUSH_DRAM
            if (k + 1 < chunks && !read_chunk(k + 1)) {
                status = kReadTimeout;
            }
#endif
            // The buffer may be refilled only after its write is acknowledged.
            if (!writes_flushed()) {
                status = kWriteTimeout;
            }
            if (status != kDone) {
                break;
            }
        }
        // After the batch, before anything is reported.
        noc_async_write_clear_pcie_state();
    }
    const uint64_t t1 = wall();

    // The sequence word, a write of its own after the data was flushed.
    if (status == kDone) {
        volatile uint32_t* s = reinterpret_cast<volatile uint32_t*>(seq_slot);
        s[0] = nonce;
        s[1] = bytes;
        s[2] = core;
        s[3] = kMagic ^ nonce;
        noc_async_write_pcie(seq_slot, pcie | seq_offset, 16);
        if (!writes_flushed()) {
            status = kSequenceTimeout;
        }
    }
    const uint64_t t2 = wall();
    if (status == kDone && !noc_cmd_bufs_mid_clear()) {
        status = kRoutingLeft;
    }
    r[kPushed] = pushed;
    r[kChunks] = chunks;
    r[kStartLow] = static_cast<uint32_t>(t0);
    r[kStartHigh] = static_cast<uint32_t>(t0 >> 32);
    r[kDataLow] = static_cast<uint32_t>(t1);
    r[kDataHigh] = static_cast<uint32_t>(t1 >> 32);
    r[kEndLow] = static_cast<uint32_t>(t2);
    r[kEndHigh] = static_cast<uint32_t>(t2 >> 32);
    r[kMidClear] = noc_cmd_bufs_mid_clear() ? 1 : 0;
    r[kStatus] = status;
}
