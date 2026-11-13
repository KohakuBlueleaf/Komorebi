// The probe's persistent compute kernel: configures the units once, parks in a wait
// loop outside every slot; on each go runs the compute loop, its part or a setup.
#include <stdint.h>

#include "api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ckernel.h"
#include "costs.h"
#include "internal/circular_buffer_init.h"

// The compute state the overlays share with this kernel (the runtime's compute build
// defines it): kept and exported, so that an overlay links to these, not to copies.
extern uint32_t unp_cfg_context __attribute__((externally_visible));
extern uint32_t pack_sync_tile_dst_ptr __attribute__((externally_visible));
extern uint32_t math_sync_tile_dst_index __attribute__((externally_visible));
extern uint32_t gl_alu_format_spec_reg __attribute__((externally_visible));

namespace ckernel {
    extern volatile tt_reg_ptr uint* pc_buf_base __attribute__((externally_visible));
    extern volatile tt_reg_ptr uint* regfile __attribute__((externally_visible));
    extern volatile std::uint32_t tt_reg_ptr* mailbox_base[4]
        __attribute__((externally_visible));
}  // namespace ckernel

__attribute__((used)) volatile void* const ov_exports[] = {
    &unp_cfg_context,
    &pack_sync_tile_dst_ptr,
    &math_sync_tile_dst_index,
    &gl_alu_format_spec_reg,
    &ckernel::pc_buf_base,
    &ckernel::regfile,
    &ckernel::mailbox_base,
};

namespace {

    volatile uint32_t part_calls = 0;

    inline uint32_t word(volatile tt_l1_ptr uint32_t* w) {
        return ckernel::reg_read(reinterpret_cast<uint32_t>(w));
    }

    inline uint64_t wall64() {
        for (;;) {
            const uint32_t h = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_1);
            const uint32_t l = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            if (ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_1) == h) {
                return static_cast<uint64_t>(h) << 32 | l;
            }
        }
    }

    __attribute__((noinline)) uint32_t mix(uint32_t x) {
        return cs::mix(x);
    }

    // The table of the runtime's calls a compute overlay makes (the table form).
    __attribute__((noinline)) bool api_wait_front(
        uint32_t cb, uint32_t pages, volatile uint32_t* stop
    ) {
        return cs::Inline::wait_front(cb, pages, stop);
    }

    __attribute__((noinline)) bool api_reserve_back(
        uint32_t cb, uint32_t pages, volatile uint32_t* stop
    ) {
        return cs::Inline::reserve_back(cb, pages, stop);
    }

    __attribute__((noinline)) void api_copy_init(uint32_t cb) {
        cs::Inline::copy_init(cb);
    }

    __attribute__((noinline)) void api_scalar_init() {
        cs::Inline::scalar_init();
    }

    __attribute__((noinline)) void api_acquire() {
        cs::Inline::acquire();
    }

    __attribute__((noinline)) void api_copy(uint32_t cb, uint32_t tile, uint32_t dst) {
        cs::Inline::copy(cb, tile, dst);
    }

    __attribute__((noinline)) void api_mul(uint32_t dst, uint32_t scalar) {
        cs::Inline::mul(dst, scalar);
    }

    __attribute__((noinline)) void api_commit() {
        cs::Inline::commit();
    }

    __attribute__((noinline)) void api_wait() {
        cs::Inline::wait();
    }

    __attribute__((noinline)) void api_pack(uint32_t dst, uint32_t cb) {
        cs::Inline::pack(dst, cb);
    }

    __attribute__((noinline)) void api_release() {
        cs::Inline::release();
    }

    __attribute__((noinline)) void api_pop(uint32_t cb, uint32_t pages) {
        cs::Inline::pop(cb, pages);
    }

    __attribute__((noinline)) void api_push(uint32_t cb, uint32_t pages) {
        cs::Inline::push(cb, pages);
    }

}  // namespace

void kernel_main() {
    const uint32_t control = get_arg_val<uint32_t>(0);
    const uint32_t slots = get_arg_val<uint32_t>(1);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(2);
    const uint64_t idle = static_cast<uint64_t>(get_arg_val<uint32_t>(3)) << 20;
    uint32_t k = 0;
    UNPACK((k = 0));
    MATH((k = 1));
    PACK((k = 2));
    compute_kernel_hw_startup(cs::kCbIn, cs::kCbOut);
    volatile tt_l1_ptr uint32_t* c = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control)
    );
    volatile tt_l1_ptr uint32_t* out = c + cs::kOut + 8 * (1 + k);
    const cs::Entry entry = reinterpret_cast<cs::Entry>(slots + (2 + k) * slot_bytes);
    cs::ComputeApi api{};
    api.wait_front = api_wait_front;
    api.reserve_back = api_reserve_back;
    api.copy_init = api_copy_init;
    api.scalar_init = api_scalar_init;
    api.acquire = api_acquire;
    api.copy = api_copy;
    api.mul = api_mul;
    api.commit = api_commit;
    api.wait = api_wait;
    api.pack = api_pack;
    api.release = api_release;
    api.pop = api_pop;
    api.push = api_push;
    cs::Call call{};
    call.stop = c + cs::kStop;
    call.api = &api;
    call.mix = mix;
    call.calls = &part_calls;
    uint32_t done = 0;
    uint64_t seen_at = wall64();
    for (;;) {
        const uint32_t go = word(c + cs::kGo);
        if (go == cs::kLeave) {
            return;
        }
        if (go == done) {
            if (wall64() - seen_at > idle) {
                return;
            }
            continue;
        }
        const uint64_t t0 = wall64();
        seen_at = t0;
        out[cs::kOutStart] = static_cast<uint32_t>(t0);
        const uint32_t op = word(c + cs::kParkedOp);
        uint32_t status = cs::kOk, result = 0;
        call.arg = word(c + cs::kArg);
        if (op == cs::kParkedCompute || op == cs::kParkedCheck) {
            // this kernel's cached flag value: an overlay may have changed the state
            MATH((ckernel::math::_invalidate_src_zero_flag_state_()));
            call.count = word(c + (op == cs::kParkedCheck ? cs::kTiles : cs::kCount));
            status = cs::compute_loop(call, cs::Inline{});
        } else if (op == cs::kParkedComputeSlot) {
            call.count = word(c + cs::kCount);
            status = entry(&call);
        } else if (op == cs::kParkedPart) {
            result = entry(&call);
        } else if (op == cs::kParkedSetup) {
            uint32_t tt_l1_ptr* layout = (
                reinterpret_cast<uint32_t tt_l1_ptr*>(word(c + cs::kParkedLayout))
            );
            const uint32_t mask = word(c + cs::kParkedMask);
            UNPACK((setup_local_cb_read_write_interfaces<true, false, false>(
                layout, 0, mask
            )));
            PACK((
                setup_local_cb_read_write_interfaces<false, true, true>(layout, 0, mask)
            ));
        } else if (op != cs::kParkedNone && op != cs::kParkedDrain) {
            status = cs::kBadOp;
        }
        out[cs::kOutStatus] = status;
        out[cs::kOutResult] = result;
        out[cs::kOutCycles] = static_cast<uint32_t>(wall64() - t0);
        done = go;
        c[cs::kDone + 1 + k] = done;
    }
}
