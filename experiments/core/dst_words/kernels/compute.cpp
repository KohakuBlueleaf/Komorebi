// Dst words probe compute: each input tile into Dst tile 0 by km::dst and back; the
// vector unit's reordering into tile 1 and back; the spans to the info page.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#ifdef ARCH_BLACKHOLE
#include "ckernel_dest.h"
#endif
#include "komorebi/device/km_dst.hpp"
#include "layout.hpp"

namespace {

    using namespace dst_words;

#ifdef TRISC_PACK
    constexpr uint32_t kInt32 = 4;  // SFPLOAD / SFPSTORE: 32 bits as they are
#ifdef ARCH_WORMHOLE
    constexpr uint32_t kStay = ADDR_MOD_3;  // mode 7 with ADDR_MOD_SET_Base set
#else
    constexpr uint32_t kStay = ADDR_MOD_7;
#endif

    // The pack thread's vector unit on Dst row 0: no address step, offset 0.
    void setup() {
        using namespace ckernel;
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(
            ADDR_MOD_7
        );
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
#ifdef ARCH_BLACKHOLE
        configure_dest_access<ThreadId::PackThreadId>(DataFormat::Float32, true);
#endif
        TTI_SFPENCC(3, 0, 0, 10);
        tensix_sync();
    }

    constexpr uint32_t row_of(uint32_t vector) {
        return 4 * (vector >> 1) + 2 * (vector & 1);
    }

    // Dst tile 0's vector v to tile 1's vector 5 v + 3 (mod 32).
    void reorder() {
#ifdef ARCH_WORMHOLE
        TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1);
#endif
        for (uint32_t v = 0; v < 32; ++v) {
            TT_SFPLOAD(0, kInt32, kStay, row_of(v));
            TT_SFPSTORE(0, kInt32, kStay, 64 + row_of((5 * v + 3) % 32));
        }
#ifdef ARCH_WORMHOLE
        TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 0);  // the packer's modes are 0 to 3
#endif
        ckernel::tensix_sync();
    }
#endif

}  // namespace

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(kCbIn, kCbOut);
#ifdef TRISC_UNPACK
    km::dst::serve_init(kCbIn);
    for (uint32_t t = 0; t < tiles; ++t) {
        cb_wait_front(kCbIn, 1);
        ckernel::mailbox_write(
            ckernel::ThreadId::PackThreadId, get_tile_l1_byte_address(kCbIn, 0)
        );
        for (;;) {  // the pack thread's requests, then its word that the tile is done
            const uint32_t mail = (
                ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)
            );
            if (!km::dst::is_request(mail)) {
                break;
            }
            km::dst::serve(mail, kCbIn);
        }
        cb_pop_front(kCbIn, 1);
    }
#endif
#ifdef TRISC_PACK
    setup();
    cb_reserve_back(kCbInfo, 1);
    volatile uint32_t* info = (
        reinterpret_cast<volatile uint32_t*>(
            get_local_cb_interface(kCbInfo).fifo_wr_ptr << 4
        )
    );
    for (uint32_t t = 0; t < tiles; ++t) {
        const auto* source = (
            reinterpret_cast<const volatile uint32_t*>(
                ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
            )
        );
        const uint32_t start = wall();
        km::dst::write(source, 0, km::dst::kTileWords);
        km::dst::written(km::dst::kTileWords - 1);
        const uint32_t written = wall();
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);  // tile done
        reorder();
        cb_reserve_back(kCbOut, 2);
        auto* destination = (
            reinterpret_cast<volatile uint32_t*>(
                get_local_cb_interface(kCbOut).fifo_wr_ptr << 4
            )
        );
        const uint32_t before = wall();
        km::dst::read(destination, 0, km::dst::kTileWords);
        const uint32_t read = wall();
        km::dst::read(
            destination + km::dst::kTileWords, km::dst::kTileWords, km::dst::kTileWords
        );
        asm volatile("fence" ::: "memory");
        cb_push_back(kCbOut, 2);
        if (t < kMaxTiles) {
            info[kWrite + t] = written - start;
            info[kRead + t] = read - before;
        }
    }
    info[kTiles] = tiles;
    asm volatile("fence" ::: "memory");
    info[kNonce] = nonce;
    asm volatile("fence" ::: "memory");
    cb_push_back(kCbInfo, 1);
#endif
}
