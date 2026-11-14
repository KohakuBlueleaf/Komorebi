// Transfers between DRAM and the session's region of the host channel, one rank's
// slice through two L1 chunks, with tt-metal's PCIe-aware calls only (data movement).
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_command.hpp"
#include "km_core.hpp"

namespace km {

    // What a transfer ends with; the runtime turns the last four into a give-up.
    constexpr uint32_t kTransferDone = 0;
    constexpr uint32_t kTransferRefused = 1;  // a bound does not hold: nothing issued
    constexpr uint32_t kTransferRead = 2;     // reads did not land in time
    constexpr uint32_t kTransferWrite = 3;    // writes were not acknowledged in time
    constexpr uint32_t kTransferRouting = 4;  // a command buffer kept its PCIe routing

    // The arguments a0 to a9 of a PUSH or PULL command (KM_TRANSFER_*).
    struct Transfer {
        uint32_t address;  // the tensor's bank address (interleaved over the banks)
        uint32_t page;     // its aligned page bytes
        uint32_t first;    // first page
        uint32_t pages;    // pages of all ranks
        uint32_t offset;   // channel offset of the first page
        uint32_t words;    // channel offset of rank 0's sequence word (PUSH)
        uint32_t region;   // the region: channel offsets [0, region)
        uint32_t nonce;
        uint32_t buffer;  // L1: two chunks, then a word slot
        uint32_t chunk;   // bytes of a chunk, a multiple of the page
    };

    inline Transfer transfer_of(const km_command_t& command) {
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
        return {
            a[KM_TRANSFER_ADDRESS],
            a[KM_TRANSFER_PAGE],
            a[KM_TRANSFER_FIRST],
            a[KM_TRANSFER_PAGES],
            a[KM_TRANSFER_OFFSET],
            a[KM_TRANSFER_WORDS],
            a[KM_TRANSFER_REGION],
            a[KM_TRANSFER_NONCE],
            a[KM_TRANSFER_BUFFER],
            a[KM_TRANSFER_CHUNK]
        };
    }

    // Every bound of a transfer by `count` ranks, checked before anything is issued:
    // the data and the words inside the region apart, the region inside the channel.
    inline bool transfer_fits(const Transfer& t, uint32_t count) {
        const uint64_t end = static_cast<uint64_t>(t.offset) +
            uint64_t{t.pages} * t.page;
        const uint64_t words_end = static_cast<uint64_t>(t.words) +
            uint64_t{count} * KM_TRANSFER_WORD_BYTES;
        bool ok = count >= 1 && count <= KM_CORES_MAX && t.pages >= 1;
        ok = ok && t.page >= 16 && t.page % 16 == 0;
        ok = ok && t.chunk >= t.page && t.chunk % t.page == 0;
        ok = ok && t.offset % 16 == 0 && t.region % 16 == 0 && t.buffer % 16 == 0;
        ok = ok && t.region <= KM_CHANNEL_BYTES && end <= t.region;
        ok = ok &&
            uint64_t{t.buffer} + 2ull * t.chunk + KM_TRANSFER_WORD_BYTES <= MEM_L1_SIZE;
        ok = ok && t.words % 16 == 0 && words_end <= t.region;
        return ok && (words_end <= t.offset || t.words >= end);
    }

    inline uint64_t pcie_noc() {
        return static_cast<uint64_t>(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y));
    }

    inline bool writes_acked() {
        return ncrisc_noc_nonposted_writes_flushed(noc_index);
    }

    // The rank's sequence word (nonce, bytes, rank, the mark ^ nonce, the wall clock at
    // its start and after its data), written after its data writes were acknowledged.
    template <typename Wait>
    inline uint32_t transfer_word(
        const Transfer& t, uint32_t rank, uint32_t bytes, uint64_t start, Wait wait
    ) {
        const uint64_t end = wall64();
        const uint32_t from = t.buffer + 2 * t.chunk;
        volatile tt_l1_ptr uint32_t* word = l1(from);
        word[0] = t.nonce;
        word[1] = bytes;
        word[2] = rank;
        word[3] = KM_TRANSFER_MARK ^ t.nonce;
        word[4] = static_cast<uint32_t>(start);
        word[5] = static_cast<uint32_t>(start >> 32);
        word[6] = static_cast<uint32_t>(end);
        word[7] = static_cast<uint32_t>(end >> 32);
        noc_async_write_pcie(
            from, pcie_noc() | (t.words + rank * KM_TRANSFER_WORD_BYTES), 32
        );
        return wait(writes_acked) ? kTransferDone : kTransferWrite;
    }

    // Rank `rank` of `count` pushes its pages to the channel (set, the writes, flush,
    // clear), then its sequence word. wait(ready) polls with a bound: false on timeout.
    template <typename Wait>
    inline uint32_t push_to_host(
        const Transfer& t, uint32_t rank, uint32_t count, Wait wait
    ) {
        static_assert(sizeof(Wait) > 0 && NOC_INDEX == 0, "PCIe coordinates of NoC 0");
        if (rank >= count || !transfer_fits(t, count)) {
            return kTransferRefused;
        }
        const uint64_t start = wall64();
        const auto reads_landed = [] {
            return ncrisc_noc_reads_flushed(noc_index);
        };
        const Partition share = partition(t.pages, rank, count);
        const uint32_t bytes = share.len * t.page;
        const uint32_t at = t.offset + share.start * t.page;
        const uint32_t chunk_pages = t.chunk / t.page;
        const uint32_t chunks = (share.len + chunk_pages - 1) / chunk_pages;
        const InterleavedAddrGen<true> in = {
            .bank_base_address = t.address, .page_size = t.page
        };
        // Reads chunk k into its buffer (no wait).
        const auto read_chunk = [&](uint32_t k) {
            const uint32_t into = t.buffer + (k & 1) * t.chunk;
            const uint32_t from = k * chunk_pages;
            const uint32_t n = (
                share.len - from < chunk_pages ? share.len - from : chunk_pages
            );
            for (uint32_t p = 0; p < n; ++p) {
                noc_async_read(
                    in.get_noc_addr(t.first + share.start + from + p),
                    into + p * t.page,
                    t.page
                );
            }
        };
        uint32_t status = kTransferDone;
        if (chunks > 0) {
            read_chunk(0);
            if (!wait(reads_landed)) {
                return kTransferRead;
            }
            noc_async_write_set_pcie_state(pcie_noc() | at);
            for (uint32_t k = 0; k < chunks; ++k) {
                const uint32_t left = bytes - k * t.chunk;
                noc_async_write_with_state(
                    t.buffer + (k & 1) * t.chunk,
                    at + k * t.chunk,
                    left < t.chunk ? left : t.chunk
                );
                if (k + 1 < chunks) {
                    read_chunk(k + 1);
                    if (!wait(reads_landed)) {
                        status = kTransferRead;
                    }
                }
                // A buffer is refilled only after its write was acknowledged.
                if (!wait(writes_acked)) {
                    status = kTransferWrite;
                }
                if (status != kTransferDone) {
                    break;
                }
            }
            noc_async_write_clear_pcie_state();
        }
        if (status == kTransferDone) {
            status = transfer_word(t, rank, bytes, start, wait);
        }
        if (!noc_cmd_bufs_mid_clear()) {
            status = kTransferRouting;
        }
        return status;
    }

    // Rank `rank` of `count` pulls its pages from the channel (set, the reads, clear)
    // into DRAM, then once its DRAM writes were acknowledged writes its sequence word.
    template <typename Wait>
    inline uint32_t pull_from_host(
        const Transfer& t, uint32_t rank, uint32_t count, Wait wait
    ) {
        static_assert(sizeof(Wait) > 0 && NOC_INDEX == 0, "PCIe coordinates of NoC 0");
        if (rank >= count || !transfer_fits(t, count)) {
            return kTransferRefused;
        }
        const uint64_t start = wall64();
        const auto reads_landed = [] {
            return ncrisc_noc_reads_flushed(noc_index);
        };
        const Partition share = partition(t.pages, rank, count);
        const uint32_t bytes = share.len * t.page;
        const uint32_t at = t.offset + share.start * t.page;
        const uint32_t chunk_pages = t.chunk / t.page;
        const uint32_t chunks = (share.len + chunk_pages - 1) / chunk_pages;
        const InterleavedAddrGen<true> out = {
            .bank_base_address = t.address, .page_size = t.page
        };
        // Reads chunk k of the channel into its buffer (no wait).
        const auto read_chunk = [&](uint32_t k) {
            const uint32_t left = bytes - k * t.chunk;
            noc_async_read_with_state(
                at + k * t.chunk,
                t.buffer + (k & 1) * t.chunk,
                left < t.chunk ? left : t.chunk
            );
        };
        uint32_t status = kTransferDone;
        if (chunks > 0) {
            noc_async_read_set_pcie_state(pcie_noc() | at);
            read_chunk(0);
            if (!wait(reads_landed)) {
                status = kTransferRead;
            }
            for (uint32_t k = 0; status == kTransferDone && k < chunks; ++k) {
                if (k + 1 < chunks) {
                    read_chunk(k + 1);
                }
                const uint32_t from = k * chunk_pages;
                const uint32_t n = (
                    share.len - from < chunk_pages ? share.len - from : chunk_pages
                );
                for (uint32_t p = 0; p < n; ++p) {
                    noc_async_write(
                        t.buffer + (k & 1) * t.chunk + p * t.page,
                        out.get_noc_addr(t.first + share.start + from + p),
                        t.page
                    );
                }
                if (!wait(reads_landed)) {
                    status = kTransferRead;
                }
                // A buffer is refilled only after its writes were acknowledged.
                if (!wait(writes_acked)) {
                    status = kTransferWrite;
                }
            }
            noc_async_read_clear_pcie_state();
        }
        if (status == kTransferDone) {
            status = transfer_word(t, rank, bytes, start, wait);
        }
        if (!noc_cmd_bufs_mid_clear()) {
            status = kTransferRouting;
        }
        return status;
    }

}  // namespace km
