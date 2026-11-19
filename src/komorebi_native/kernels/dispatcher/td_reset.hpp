// The Tensix dispatcher's reset of one core (the X280's grid/reset.h), for "out": its
// RISC CPUs held in reset over the NoC, a kept list's writes into its L1, released.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"

namespace td {

#if defined(ARCH_WORMHOLE)
    // Core `worker`'s INSTRN_BUF_STATUS through L1 `line` (16 B): KMD_UNIT_READY when
    // its compute unit is ready; on Wormhole only a card reset clears one that is not.
    template <typename Core>
    uint32_t unit_status(uint32_t worker, Core core, uint32_t line) {
        constexpr uint32_t kAt = KMD_UNIT_STATUS_ADDR % 16;
        noc_async_read(core(worker, KMD_UNIT_STATUS_ADDR), line + kAt, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        return km::l1(line)[kAt / 4] & 0xFFu;
    }
#endif

    // The kept lists, in DRAM pages of kTdStageBytes from `base` (bytes 0: none).
    struct Resets {
        uint32_t base = 0, bytes = 0, block = 0;
        uint64_t wait_ticks = 0;                      // a WAIT's bound, 1 s
        volatile tt_l1_ptr uint32_t* beat = nullptr;  // out's beat (the watch's)

        // `count` bytes from list byte `offset`, read whole 64-byte lines into L1 `to`
        // (64-aligned); the bytes wanted start at `to` + offset % 64.
        void read(uint32_t offset, uint32_t count, uint32_t to) const {
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = base, .page_size = kTdStageBytes
            };
            uint32_t at = offset & ~63u;
            const uint32_t end = (offset + count + 63) & ~63u;
            while (at < end) {
                const uint32_t in = at % kTdStageBytes;
                const uint32_t take = (
                    end - at < kTdStageBytes - in ? end - at : kTdStageBytes - in
                );
                noc_async_read(pages.get_noc_addr(at / kTdStageBytes, in), to, take);
                to += take;
                at += take;
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
        }

        // Word `index` (of 4) of the 16-byte entry at list byte `offset`.
        uint32_t entry_word(uint32_t offset, uint32_t index) const {
            read(offset, 16, block + kTbScratch);
            return km::l1(block + kTbScratch)[(offset % 64) / 4 + index];
        }

        // Keeps the list: `count` bytes from host channel offset `from` (`pcie` its NoC
        // address) into the DRAM pages at `address`.
        uint32_t keep(uint32_t address, uint32_t count, uint64_t pcie, uint32_t from) {
            if (address % 64 != 0 || count % 16 != 0 || from % 64 != 0) {
                return KMD_RESET_BAD_LIST;
            }
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = address, .page_size = kTdStageBytes
            };
            const uint32_t stage = block + kTbStage;
            for (uint32_t at = 0; at < count; at += kTdStageBytes) {
                const uint32_t take = (
                    count - at < kTdStageBytes ? count - at : kTdStageBytes
                );
                noc_async_read_pcie(pcie | (from + at), stage, take);
                noc_async_read_barrier();
                noc_async_write(stage, pages.get_noc_addr(at / kTdStageBytes), take);
                noc_async_write_barrier();
                beat[0]++;
            }
            base = address;
            bytes = count;
            return KMD_RESET_OK;
        }

        // The kept list of core `worker` (a directory's entry, or the one list): true,
        // or false for none.
        bool pick(uint32_t worker, uint32_t& offset, uint32_t& count) const {
            if (bytes == 0 || worker >= KM_CORES_MAX) {
                return false;
            }
            if (bytes < KMD_RESET_ENTRY || entry_word(0, 0) != KMD_RESET_DIRECTORY) {
                offset = 0;
                count = bytes;
                return true;
            }
            const uint32_t lists = entry_word(0, 1);
            const uint32_t byte_at = KMD_RESET_ENTRY + worker;
            const uint32_t list = (
                (
                    entry_word(byte_at & ~15u, (byte_at % 16) / 4) >>
                    (8 * (byte_at % 4))
                ) &
                0xFFu
            );
            const uint32_t entry = KMD_RESET_DIRECTORY_ENTRIES + 16 * list;
            if (list >= lists || entry + 16 > bytes) {
                return false;
            }
            offset = entry_word(entry, 0);
            count = entry_word(entry, 1);
            return true;
        }

        // `size` bytes (whole words) of the list at `offset` into the core's L1 at
        // `address`, through the stage, each piece moved to the address's alignment.
        template <typename Core>
        void copy(
            uint32_t worker, Core core, uint32_t offset, uint32_t address, uint32_t size
        ) const {
            constexpr uint32_t kPiece = kTdStageBytes / 2 - 128;
            const uint32_t from = block + kTbStage, to = from + kTdStageBytes / 2;
            for (uint32_t done = 0; done < size; done += kPiece) {
                const uint32_t take = size - done < kPiece ? size - done : kPiece;
                read(offset + done, take, from);
                const uint32_t start = from + (offset + done) % 64;
                const uint32_t shift = (address + done) % 16;
                for (uint32_t word = 0; word < take / 4; word++) {
                    km::l1(to + shift)[word] = km::l1(start)[word];
                }
                noc_async_write(to + shift, core(worker, address + done), take);
                noc_async_write_barrier();
            }
        }

        // Runs the list of `count` bytes at `offset` on core `worker`: KMD_RESET_*;
        // every wait is bounded.
        template <typename Core>
        uint32_t run(
            uint32_t worker, Core core, uint32_t offset, uint32_t count
        ) const {
            if (
                worker >= KM_CORES_MAX || offset % 16 != 0 || count % 16 != 0 ||
                offset > bytes || count > bytes - offset
            ) {
                return KMD_RESET_BAD_LIST;
            }
            const uint64_t reg = core(worker, KMD_SOFT_RESET_ADDR);
            uint32_t at = 0;
            while (at + KMD_RESET_ENTRY <= count) {
                read(offset + at, 16, block + kTbScratch);
                volatile tt_l1_ptr uint32_t*
                    entry = (km::l1(block + kTbScratch) + ((offset + at) % 64) / 4);
                const uint32_t kind = entry[0], address = entry[1];
                const uint32_t size = entry[2], value = entry[3];
                at += KMD_RESET_ENTRY;
                if (kind == KMD_RESET_END) {
                    break;
                }
                const uint32_t padded = (size + 15) & ~15u;
                const bool in_l1 = (
                    address % 4 == 0 && size % 4 == 0 && address <= MEM_L1_SIZE &&
                    size <= MEM_L1_SIZE - address
                );
                if (kind == KMD_RESET_ASSERT || kind == KMD_RESET_RELEASE) {
                    noc_inline_dw_write(
                        reg,
                        kind == KMD_RESET_ASSERT
                            ? KMD_SOFT_RESET_RISCS
                            : KMD_SOFT_RESET_RELEASE
                    );
                    noc_async_write_barrier();
                } else if (kind == KMD_RESET_WRITE) {
                    if (!in_l1 || padded > count - at) {
                        return KMD_RESET_BAD_LIST;
                    }
                    copy(worker, core, offset + at, address, size);
                    at += padded;
                } else if (kind == KMD_RESET_EACH) {
                    if (
                        !in_l1 || worker >= value || value > KM_CORES_MAX ||
                        static_cast<uint64_t>(padded) * value > count - at
                    ) {
                        return KMD_RESET_BAD_LIST;
                    }
                    copy(worker, core, offset + at + worker * padded, address, size);
                    at += value * padded;
                } else if (kind == KMD_RESET_WAIT) {
                    if (address % 4 != 0 || address > MEM_L1_SIZE - 4) {
                        return KMD_RESET_BAD_LIST;
                    }
                    const uint32_t line = block + kTbScratch;
                    const uint64_t until = km::wall64() + wait_ticks;
                    for (;;) {
                        noc_async_read(core(worker, address & ~15u), line, 16);
                        noc_async_read_barrier();
                        invalidate_l1_cache();
                        if ((km::l1(line)[(address % 16) / 4] & size) == value) {
                            break;
                        }
                        if (km::wall64() > until) {
                            return KMD_RESET_NO_FIRMWARE;
                        }
                        beat[0]++;  // a long wait, not a stall
                    }
                } else {
                    return KMD_RESET_BAD_LIST;
                }
            }
#if defined(ARCH_WORMHOLE)
            const uint32_t unit = unit_status(worker, core, block + kTbScratch);
            if (unit != KMD_UNIT_READY) {
                return KMD_RESET_UNIT | unit << 8;
            }
#endif
            return KMD_RESET_OK;
        }
    };

}  // namespace td
