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

#if defined(KM_PROGRAM_PART)
#if defined(COMPILE_FOR_TRISC) && defined(KM_PROGRAM_SETUP_IN)
#include "api/compute/compute_kernel_hw_startup.h"
#endif

namespace km {

    // A program's part: 1 completes the command (the writer's), 0 does not.
    template <typename Kernel>
    inline uint32_t program_result(
        void (*function)(Kernel&, const km_command_t&),
        Kernel& kernel,
        const km_command_t& command
    ) {
        function(kernel, command);
        return 1;
    }

    template <typename Kernel>
    inline uint32_t program_result(
        bool (*function)(Kernel&, const km_command_t&),
        Kernel& kernel,
        const km_command_t& command
    ) {
        return function(kernel, command) ? 1u : 0u;
    }

}  // namespace km

// The function the entry calls (KM_OVERLAY_FUNCTION): the program's km_program; for a
// load of the program, a compute part's hardware start-up on the program's formats.
static uint32_t km_program_part(
    KM_OVERLAY_KERNEL& kernel, const km_command_t& command
) {
    if (km::handler(command) == KM_HANDLER_LOAD) {
#if defined(COMPILE_FOR_TRISC) && defined(KM_PROGRAM_SETUP_IN)
        compute_kernel_hw_startup(KM_PROGRAM_SETUP_IN, KM_PROGRAM_SETUP_OUT);
#endif
        return 1;
    }
    return km::program_result<KM_OVERLAY_KERNEL>(km_program, kernel, command);
}
#endif

#if defined(KM_MESSAGE_PART)
// The function the entry calls (KM_OVERLAY_FUNCTION): the handler's km_message.
static void km_message_part(KM_OVERLAY_KERNEL&, const km::MessageCall& call) {
    km_message(*call.messages, *call.packet);
}
#endif

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
