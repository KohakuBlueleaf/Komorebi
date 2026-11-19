// A pack thread's 32-bit Dst words from and to L1 in the window's order (runtime.md,
// Dst words); KM_DST_WINDOW, the chips table's dst_window, picks the form.
#pragma once

#include <stdint.h>

#include "ckernel.h"

#ifndef KM_DST_WINDOW
#error "KM_DST_WINDOW: the chips table's dst_window (komorebi.device.chips)"
#endif

#if !KM_DST_WINDOW && defined(TRISC_UNPACK)
#include "llk_unpack_A.h"
#endif
#if !KM_DST_WINDOW && defined(TRISC_PACK)
#include "llk_pack.h"
#endif

namespace km::dst {

    constexpr uint32_t kTileWords = 1024;  // a 32 x 32 tile of 32-bit words

#if KM_DST_WINDOW
    constexpr uint32_t kWindow = 0xFFBD8000;  // the pack core's view of Dst

    inline volatile uint32_t* window() {
        return reinterpret_cast<volatile uint32_t*>(kWindow);
    }

    // `words` words (a multiple of 8), the loads grouped ahead of the stores.
    inline void copy(
        volatile uint32_t* destination, const volatile uint32_t* source, uint32_t words
    ) {
        for (uint32_t i = 0; i < words; i += 8) {
            uint32_t loaded[8];
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                loaded[j] = source[i + j];
            }
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                destination[i + j] = loaded[j];
            }
        }
    }

    // Dst words [first, first + words) from `source`; written() before a read of them.
    inline void write(const volatile uint32_t* source, uint32_t first, uint32_t words) {
        copy(window() + first, source, words);
    }

    // Waits until the window's stores up to Dst word `last` are done (a load, used).
    inline void written(uint32_t last) {
        uint32_t last_word = window()[last];
        asm volatile("and x0, x0, %0" : : "r"(last_word) : "memory");
    }

    // Dst words [first, first + words) into `destination`.
    inline void read(volatile uint32_t* destination, uint32_t first, uint32_t words) {
        copy(destination, window() + first, words);
    }

    inline void serve_init(uint32_t) {
    }

    inline bool is_request(uint32_t) {
        return false;
    }

    inline void serve(uint32_t, uint32_t) {
    }
#else
    // A request: bit 31, the tile's 16-byte aligned L1 address, the Dst tile in bits
    // 0 to 3; the pack thread's other mails leave bit 31 clear.
    constexpr uint32_t kRequest = 0x80000000u;

    inline bool is_request(uint32_t mail) {
        return (mail & kRequest) != 0;
    }

#ifdef TRISC_UNPACK
    // The unpacker set up for whole 32-bit tiles into Dst in buffer `cb`'s format.
    inline void serve_init(uint32_t cb) {
        _llk_unpack_A_init_<
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            true>(
                0,
                0,
                ckernel::DEFAULT_TENSOR_SHAPE,
                unpack_src_format[cb],
                unpack_dst_format[cb]
            );
        ckernel::t6_semaphore_init(ckernel::semaphore::UNPACK_TO_DEST, 0, 1);
    }

    // tt-llk's _llk_unpack_A_ to Dst, the row from the request (not the math thread's
    // mail); UNPACK_TO_DEST posted when the unpacker is done. `cb`: serve_init's.
    inline void serve(uint32_t mail, uint32_t cb) {
        using namespace ckernel;
        const uint32_t destination = unpack_dst_format[cb];
        const uint32_t sixteens = (mail & ~kRequest) >> 4;
        const uint32_t address = sixteens - 1;   // as the runtime's buffers give it
        const uint32_t row = 64 * (mail & 0xF);  // a tile's 64 rows of 16 words
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
        wait_for_next_context(2);
        cfg[
            unp_cfg_context == 0
                ? THCON_SEC0_REG3_Base_address_ADDR32
                : THCON_SEC0_REG3_Base_cntx1_address_ADDR32
        ] = address;
        TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0);  // no address bit swizzle
        TT_SETDMAREG(
            0,
            LOWER_HALFWORD(
                canonical_unpA_z_stride(destination)
                << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT
            ),
            0,
            LO_16(p_gpr_unpack::TMP_LO)
        );
        TTI_WRCFG(
            p_gpr_unpack::TMP_LO,
            p_cfg::WRCFG_32b,
            UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32
        );
        if (unp_cfg_context == 0) {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(16 * (4 + row));
        } else {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(16 * (4 + row));
        }
        wait_for_dest_available();
        semaphore_post(semaphore::UNPACK_SYNC);
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        ckernel_template::run();
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        unpack_to_dest_tile_done(unp_cfg_context, destination);
        switch_config_context(unp_cfg_context);
    }
#endif

#ifdef TRISC_PACK
    // Dst words [first, first + words), whole tiles, from `source` (16-byte aligned):
    // a request a tile, each waited for, so `source` is free again on return.
    inline void write(const volatile uint32_t* source, uint32_t first, uint32_t words) {
        using namespace ckernel;
        for (uint32_t i = 0; i < words; i += kTileWords) {
            const uint32_t address = reinterpret_cast<uint32_t>(source + i);
            mailbox_write(
                ThreadId::UnpackThreadId,
                kRequest | address | ((first + i) / kTileWords)
            );
            while (semaphore_read(semaphore::UNPACK_TO_DEST) == 0) {
                ;
            }
            semaphore_get(semaphore::UNPACK_TO_DEST);
        }
    }

    // Nothing: write returns once the unpacker is done.
    inline void written(uint32_t) {
    }

    // Dst words [first, first + words), whole tiles, into `destination` (16-byte
    // aligned) by the packer in its configured output format; done on return.
    inline void read(volatile uint32_t* destination, uint32_t first, uint32_t words) {
        for (uint32_t i = 0; i < words; i += kTileWords) {
            const uint32_t address = reinterpret_cast<uint32_t>(destination + i);
            _llk_pack_<DstSync::SyncFull, true>(
                (first + i) / kTileWords, (address >> 4) - 1
            );
        }
        ckernel::tensix_sync();
    }
#endif
#endif

}  // namespace km::dst
