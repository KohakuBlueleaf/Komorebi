// The entry of an overlay part at its slot's first byte, after the part's source: it
// zeroes the part's .bss and calls KM_OVERLAY_FUNCTION(kernel, argument).
#pragma once

#include <stdint.h>

#if !defined(KM_OVERLAY_KERNEL) || !defined(KM_OVERLAY_FUNCTION)
#error "a part is built by the overlay builder (KM_OVERLAY_KERNEL and _FUNCTION)"
#endif

namespace km {

    // The part's function with a result word, or without one (0).
    template <typename Kernel, typename Argument>
    inline uint32_t overlay_invoke(
        uint32_t (*function)(Kernel&, const Argument&),
        void* kernel,
        const void* argument
    ) {
        return (
            function(
                *static_cast<Kernel*>(kernel), *static_cast<const Argument*>(argument)
            )
        );
    }

    template <typename Kernel, typename Argument>
    inline uint32_t overlay_invoke(
        void (*function)(Kernel&, const Argument&), void* kernel, const void* argument
    ) {
        function(
            *static_cast<Kernel*>(kernel), *static_cast<const Argument*>(argument)
        );
        return 0;
    }

}  // namespace km

// The builder's linker script: the part's .bss, hidden so that a position-independent
// build reaches them without a table entry.
extern "C" uint32_t __km_overlay_bss_start[] __attribute__((visibility("hidden")));
extern "C" uint32_t __km_overlay_bss_end[] __attribute__((visibility("hidden")));

extern "C" __attribute__((section(".km_overlay_entry"), used, noinline)) uint32_t
km_overlay_entry(void* kernel, const void* argument) {
    for (uint32_t* word = __km_overlay_bss_start; word < __km_overlay_bss_end; word++) {
        *word = 0;
    }
#if defined(TRISC_MATH)
    // this part's copy of the math library's cached flag value: unknown at entry
    ckernel::math::_invalidate_src_zero_flag_state_();
#endif
    return km::overlay_invoke<KM_OVERLAY_KERNEL>(KM_OVERLAY_FUNCTION, kernel, argument);
}
