// The texture lookup probe's compute kernel: LOOP and the integer COLOUR on unpack or
// pack, and the colour on the vector unit (unpack, multiply, pack), timed by pack.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "lookup.h"
#include "table.hpp"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel::sfpu {

    // Dst tile `out` = the texels of tile `texel` (RGBA8) with r, g and b times the
    // light term of tile `light` (float32), each rounded to the nearest integer.
    inline void texel_light(uint32_t texel, uint32_t light, uint32_t out) {
        constexpr uint32_t kRows = 32;  // a tile's rows in the vector unit's address
        constexpr sfpi::DataLayout kRaw = sfpi::DataLayout::I32;
        for (int row = 0; row < 8; row++) {
            const sfpi::vInt bits = sfpi::dst_reg[texel * kRows].mode<kRaw>();
            const sfpi::vInt light_bits = sfpi::dst_reg[light * kRows].mode<kRaw>();
            const sfpi::vFloat term = sfpi::as<sfpi::vFloat>(light_bits);
            const sfpi::vInt byte = 0xFF;
            const sfpi::vInt two23 = 0x4B000000;  // 2^23 as float32: its mantissa
            sfpi::vInt colour = bits & sfpi::vInt(static_cast<int>(0xFF000000u));
            for (int channel = 0; channel < 3; channel++) {
                const sfpi::vInt value = (
                    sfpi::as<sfpi::vInt>(
                        sfpi::shft(sfpi::as<sfpi::vUInt>(bits), -8 * channel)
                    ) &
                    byte
                );
                // 2^23 + value, less 2^23: the value as a float, exact.
                const sfpi::vFloat exact = (
                    sfpi::as<sfpi::vFloat>(value | two23) - 8388608.0f
                );
                // One multiply-add, one rounding: 2^23 + the nearest integer.
                const sfpi::vFloat lit = exact * term + 8388608.0f;
                const sfpi::vInt rounded = sfpi::as<sfpi::vInt>(lit) & byte;
                colour = (
                    colour |
                    sfpi::as<sfpi::vInt>(
                        sfpi::shft(sfpi::as<sfpi::vUInt>(rounded), 8 * channel)
                    )
                );
            }
            sfpi::dst_reg[out * kRows].mode<kRaw>() = colour;
            sfpi::dst_reg++;
        }
    }

}  // namespace ckernel::sfpu
#endif

namespace {

    inline void texel_light_init() {
        MATH((SFPU_BINARY_INIT(unused)));
    }

    inline void texel_light_tile(uint32_t texel, uint32_t light, uint32_t out) {
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            texel_light,
            texel,
            light,
            out,
            VectorMode::RC
        )));
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbColour, kCbPacked);
    const texture::Tensors at = {
        compute.arg(kArgPages),
        compute.arg(kArgSlots),
        compute.arg(kArgStreams),
        compute.arg(kArgPlanes),
        compute.arg(kArgOut),
        compute.arg(kArgResults),
        compute.arg(kArgReplay),
    };
    compute.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler == kHandlerLoop) {
            UNPACK((command.a0 == kCpuUnpack ? texture::loop(at, command) : (void)0));
            PACK((command.a0 == kCpuPack ? texture::loop(at, command) : (void)0));
            return;
        }
        if (handler != kHandlerColour) {
            return;
        }
        if (command.a0 == kColourInteger) {
            UNPACK((command.a1 == kCpuUnpack ? texture::colour(at, command) : (void)0));
            PACK((command.a1 == kCpuPack ? texture::colour(at, command) : (void)0));
            return;
        }
        const bool multiply = command.a0 == kColourVector;
        const uint32_t reps = command.a2 < kRepsMax ? command.a2 : kRepsMax;
        if (!compute.wait_front(kCbColour, 2)) {
            return;
        }
        // Pack decides; when it gave up, math and pack still meet on every tile.
        const bool room = compute.reserve_back(kCbPacked, 1);
        uint32_t start = 0;
        PACK((start = km::wall()));
        for (uint32_t rep = 0; rep < reps; rep++) {
            tile_regs_acquire();
            copy_init(kCbColour);
            copy_tile(kCbColour, 0, 0);
            copy_tile(kCbColour, 1, 1);
            if (multiply) {
                texel_light_init();
                texel_light_tile(0, 1, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            if (room) {
                pack_tile(0, kCbPacked);
            }
            tile_regs_release();
        }
        PACK(({
            ckernel::tensix_sync();
            uint32_t* results = texture::words(at.results);
            results[kResultReps] = km::wall() - start;
            results[kResultError] = room ? 0 : kErrorWait;
            results[kResultRun] = command.a9;
            texture::fence();
        }));
        cb_pop_front(kCbColour, 2);
        if (room) {
            cb_push_back(kCbPacked, 1);
        }
    });
}
