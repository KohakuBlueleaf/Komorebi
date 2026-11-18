// Transfers between the session's region of the host channel and DRAM (through two L1
// chunks) or an L1 tensor (a rank's own page); copies between DRAM pages.
#pragma once

#include <stdint.h>

#include "../../abi/km_copy.h"
#include "api/dataflow/dataflow_api.h"
#include "../core/km_command.hpp"
#include "../core/km_core.hpp"

namespace km {

    // What a transfer ends with; the runtime turns the last four into a give-up.
    constexpr uint32_t kTransferDone = 0;
    constexpr uint32_t kTransferRefused = 1;  // a bound does not hold: nothing issued
    constexpr uint32_t kTransferRead = 2;     // reads did not land in time
    constexpr uint32_t kTransferWrite = 3;    // writes were not acknowledged in time
    constexpr uint32_t kTransferRouting = 4;  // a command buffer kept its PCIe routing

    // The arguments a0 to a9 of a PUSH or PULL command (KM_TRANSFER_*).
    struct Transfer {
        uint32_t address;  // the tensor's bank address, or its address in every L1
        uint32_t page;     // its aligned page bytes
        uint32_t first;    // first page
        uint32_t pages;    // pages of all ranks
        uint32_t offset;   // channel offset of the first page
        uint32_t words;    // channel offset of rank 0's sequence word (PUSH)
        uint32_t region;   // the region: channel offsets [0, region)
        uint32_t nonce;
        uint32_t buffer;  // L1: two chunks, then a word slot
        uint32_t chunk;   // bytes of a chunk, a multiple of the page; KM_TRANSFER_L1
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

    // Every bound of a transfer by `count` ranks before anything is issued (the host's
    // channel.bound names each): DRAM in whole lines in its banks, L1 past the slot.
    inline __attribute__((noinline)) bool transfer_fits(
        const Transfer& t, uint32_t count
    ) {
        const uint64_t end = static_cast<uint64_t>(t.offset) +
            uint64_t{t.pages} * t.page;
        const uint64_t words_end = static_cast<uint64_t>(t.words) +
            uint64_t{count} * KM_TRANSFER_WORD_BYTES;
        const uint64_t page_end = uint64_t{t.address} + t.page;
        const uint64_t slot_end = uint64_t{t.buffer} + KM_TRANSFER_WORD_BYTES;
        bool ok = count >= 1 && count <= KM_CORES_MAX && t.pages >= 1;
        ok = ok && t.page >= 16 && t.page % 16 == 0;
        if (t.chunk == KM_TRANSFER_L1) {
            ok = ok && t.pages == count && t.address % 16 == 0;
            ok = ok && t.address >= MEM_MAP_END && page_end <= MEM_L1_SIZE;
            ok = ok && (page_end <= t.buffer || slot_end <= t.address);
        } else {
            const uint64_t rows = (uint64_t{t.first} + t.pages + NUM_DRAM_BANKS - 1) /
                NUM_DRAM_BANKS;
            ok = ok && t.chunk >= t.page && t.chunk % t.page == 0;
            ok = ok && ((t.address | t.page | t.buffer) % KM_COPY_ALIGN) == 0;
            ok = ok && t.address + rows * t.page <= KM_COPY_BANK_END;
        }
        ok = ok && t.offset % 16 == 0 && t.region % 16 == 0 && t.buffer % 16 == 0;
        ok = ok && t.region <= KM_CHANNEL_BYTES && end <= t.region;
        ok = ok && slot_end + 2ull * t.chunk <= MEM_L1_SIZE;
        ok = ok && t.words % 16 == 0 && words_end <= t.region;
        return ok && (words_end <= t.offset || t.words >= end);
    }

#if defined(ARCH_BLACKHOLE)
    // PCIE_NOC_X / Y are translated, (19, 24): the same PCIe tile on either NoC.
    constexpr bool kPcieEitherNoc = true;
#else
    constexpr bool kPcieEitherNoc = false;
#endif

    inline uint64_t pcie_noc() {
        return static_cast<uint64_t>(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y));
    }

    inline bool writes_acked() {
        return ncrisc_noc_nonposted_writes_flushed(noc_index);
    }

    inline bool reads_landed() {
        return ncrisc_noc_reads_flushed(noc_index);
    }

    // The rank's sequence word (nonce, bytes, rank, the mark ^ nonce, the wall clock at
    // its start and after its data), written after its data writes were acknowledged.
    template <typename Wait>
    __attribute__((noinline)) uint32_t transfer_word(
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

    // Rank `rank` moves its L1 page to (push) or from the channel in one PCIe-aware
    // write or read, then writes its sequence word.
    template <typename Wait>
    __attribute__((noinline)) uint32_t
    move_l1(const Transfer& t, uint32_t rank, bool push, Wait wait) {
        const uint64_t start = wall64();
        const uint32_t at = t.offset + rank * t.page;
        uint32_t status = kTransferDone;
        if (push) {
            noc_async_write_set_pcie_state(pcie_noc() | at);
            noc_async_write_with_state(t.address, at, t.page);
            if (!wait(writes_acked)) {
                status = kTransferWrite;
            }
            noc_async_write_clear_pcie_state();
        } else {
            noc_async_read_set_pcie_state(pcie_noc() | at);
            noc_async_read_with_state(at, t.address, t.page);
            if (!wait(reads_landed)) {
                status = kTransferRead;
            }
            noc_async_read_clear_pcie_state();
        }
        if (status == kTransferDone) {
            status = transfer_word(t, rank, t.page, start, wait);
        }
        if (!noc_cmd_bufs_mid_clear()) {
            status = kTransferRouting;
        }
        return status;
    }

    // Rank `rank` of `count` pushes its pages to the channel (set, the writes, flush,
    // clear), then its sequence word. wait(ready) polls with a bound: false on timeout.
    template <typename Wait>
    __attribute__((noinline)) uint32_t
    push_to_host(const Transfer& t, uint32_t rank, uint32_t count, Wait wait) {
        static_assert(
            sizeof(Wait) > 0 && (NOC_INDEX == 0 || kPcieEitherNoc),
            "PCIe coordinates of NoC 0"
        );
        if (rank >= count || !transfer_fits(t, count)) {
            return kTransferRefused;
        }
        if (t.chunk == KM_TRANSFER_L1) {
            return move_l1(t, rank, true, wait);
        }
        const uint64_t start = wall64();
        const Partition share = partition(t.pages, rank, count);
        const uint32_t bytes = share.len * t.page;
        const uint32_t at = t.offset + share.start * t.page;
        const uint32_t chunk_pages = t.chunk / t.page;
        const uint32_t chunks = (share.len + chunk_pages - 1) / chunk_pages;
        const InterleavedAddrGen<true> in = {
            .bank_base_address = t.address, .page_size = t.page
        };
        // Reads chunk k into its buffer (no wait).
        const auto read_chunk = [&](uint32_t k) __attribute__((noinline)) {
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
    __attribute__((noinline)) uint32_t
    pull_from_host(const Transfer& t, uint32_t rank, uint32_t count, Wait wait) {
        static_assert(
            sizeof(Wait) > 0 && (NOC_INDEX == 0 || kPcieEitherNoc),
            "PCIe coordinates of NoC 0"
        );
        if (rank >= count || !transfer_fits(t, count)) {
            return kTransferRefused;
        }
        if (t.chunk == KM_TRANSFER_L1) {
            return move_l1(t, rank, false, wait);
        }
        const uint64_t start = wall64();
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

    // The NoC address of page `index` of a copy's side (a bank range or a tensor).
    inline uint64_t copy_noc(
        const km_copy_side_t& side, uint32_t page, uint32_t index
    ) {
        if (side.bank == KM_COPY_INTERLEAVED) {
            const InterleavedAddrGen<true> tensor = {
                .bank_base_address = side.address, .page_size = page
            };
            return tensor.get_noc_addr(index);
        }
        uint32_t bank = 0;
        const uint64_t address = km_copy_at(&side, page, index, NUM_DRAM_BANKS, &bank);
        return get_noc_addr_from_bank_id<true>(bank, static_cast<uint32_t>(address));
    }

    // Rank `rank` of `count` copies its pages through the two chunks: chunk k is read
    // while chunk k - 1 is written. wait(ready) polls with a bound: false on timeout.
    template <typename Wait>
    inline uint32_t copy_pages(
        const km_copy_t& c, uint32_t rank, uint32_t count, Wait wait
    ) {
        if (rank >= count || !km_copy_fits(&c, count, MEM_L1_SIZE, NUM_DRAM_BANKS)) {
            return kTransferRefused;
        }
        const auto reads_landed = [] {
            return ncrisc_noc_reads_flushed(noc_index);
        };
        const Partition share = partition(c.pages, rank, count);
        const uint32_t chunk_pages = c.chunk / c.page;
        const uint32_t chunks = (share.len + chunk_pages - 1) / chunk_pages;
        // Pages of chunk k, and its reads into its buffer (no wait).
        const auto pages_of = [&](uint32_t k) {
            const uint32_t left = share.len - k * chunk_pages;
            return left < chunk_pages ? left : chunk_pages;
        };
        const auto read_chunk = [&](uint32_t k) {
            const uint32_t into = c.buffer + (k & 1) * c.chunk;
            const uint32_t from = share.start + k * chunk_pages;
            for (uint32_t p = 0; p < pages_of(k); ++p) {
                noc_async_read(
                    copy_noc(c.from, c.page, c.from.first + from + p),
                    into + p * c.page,
                    c.page
                );
            }
        };
        uint32_t status = kTransferDone;
        if (chunks > 0) {
            read_chunk(0);
            if (!wait(reads_landed)) {
                return kTransferRead;
            }
        }
        for (uint32_t k = 0; k < chunks; ++k) {
            const uint32_t at = c.buffer + (k & 1) * c.chunk;
            const uint32_t to = share.start + k * chunk_pages;
            for (uint32_t p = 0; p < pages_of(k); ++p) {
                noc_async_write(
                    at + p * c.page, copy_noc(c.to, c.page, c.to.first + to + p), c.page
                );
            }
            if (k + 1 < chunks) {
                read_chunk(k + 1);
                if (!wait(reads_landed)) {
                    status = kTransferRead;
                }
            }
            // A buffer is refilled only after its writes were acknowledged.
            if (!wait(writes_acked)) {
                status = kTransferWrite;
            }
            if (status != kTransferDone) {
                break;
            }
        }
        return status;
    }

}  // namespace km
