// Runs a fixed loop of machine code from a chosen L1 address and times it. The loop is
// written in assembly, copied from the kernel's own text to the address, run with the
// instruction cache invalidated, and timed with the wall clock; address 0 runs it in
// place.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// A loop of a0 passes over `body` (assembly text); the first and last labels bound the
// bytes that are copied.
#define KM_LOOP(name, body) \
    ".globl km_loop_" #name \
    "_begin\n"              \
    ".balign 16\n"          \
    "km_loop_" #name        \
    "_begin:\n"             \
    "1:\n" body             \
    "addi a0, a0, -1\n"     \
    "bnez a0, 1b\n"         \
    "ret\n"                 \
    ".globl km_loop_" #name \
    "_end\n"                \
    "km_loop_" #name "_end:\n"

// Repeats of one line of assembly.
#define KM_REPEAT(count, line) ".rept " #count "\n" line ".endr\n"

#define KM_ADDS(count) KM_REPEAT(count, "addi t1, t1, 1\n")
// A taken jump to the next instruction, a branch that is never taken, a jump over 2
// KiB.
#define KM_JUMPS(count) KM_REPEAT(count, "j 3f\n3:\n")
#define KM_FAR "j 5f\n" KM_REPEAT(512, "nop\n") "5:\n"
// An indirect jump (target in a register) to the next instruction.
#define KM_INDIRECT(count) KM_REPEAT(count, "lla t0, 6f\njalr x0, t0, 0\n6:\n")
#define KM_NOT_TAKEN(count) KM_REPEAT(count, "bne zero, zero, 4f\n")
// Loads and stores in the buffer a1 points to.
#define KM_MEMORY(count) \
    KM_REPEAT(count, "lw t1, 0(a1)\naddi t1, t1, 1\nsw t1, 4(a1)\n")

asm(
    ".pushsection .text.km_loops, \"ax\"\n" KM_LOOP(tiny, "")
        KM_LOOP(small, KM_ADDS(12)) KM_LOOP(medium, KM_ADDS(60))
            KM_LOOP(large, KM_ADDS(600)) KM_LOOP(memory, KM_MEMORY(8))
                KM_LOOP(jumps, KM_JUMPS(16))
                    KM_LOOP(straight, KM_ADDS(16) "bne zero, zero, 4f\n4:\n")
                        KM_LOOP(untaken, KM_NOT_TAKEN(16) "4:\n") KM_LOOP(far, KM_FAR)
                            KM_LOOP(indirect, KM_INDIRECT(8)) ".popsection\n"
);

#define KM_BLOB_LIST(X) \
    X(tiny)             \
    X(small)            \
    X(medium) X(large) X(memory) X(jumps) X(straight) X(untaken) X(far) X(indirect)

#define KM_DECLARE(name) \
    extern "C" char km_loop_##name##_begin[], km_loop_##name##_end[];
KM_BLOB_LIST(KM_DECLARE)

namespace {

    constexpr uint32_t kRegionBytes = 16384;         // filled with nops from 4 KiB below the loop
    constexpr uint32_t kReport = 16;  // report words before the timings

    inline uint32_t wall_low() {
        return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    }

    // The address of a label by the PC-relative form, whatever address the text runs
    // at.
#define KM_ADDRESS(label, out) asm volatile("lla %0, " #label : "=r"(out))

    // Invalidates this core's own instruction cache (RISC-V 0 is bit 0, RISC-V 1 bit
    // 4).
    inline void invalidate_own_icache() {
#if defined(COMPILE_FOR_NCRISC)
        constexpr uint32_t mask = 0x10;
#else
        constexpr uint32_t mask = 0x1;
#endif
        volatile tt_reg_ptr uint32_t*
            config = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(TENSIX_CFG_BASE);
        config[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = mask;
        (void)config[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32];
    }

    // The text of blob `id` (the order of KM_BLOB_LIST): begin and end addresses.
    inline void blob(uint32_t id, uintptr_t& begin, uintptr_t& end) {
        uint32_t next = 0;
#define KM_CASE(name)                              \
    if (id == next++) {                            \
        KM_ADDRESS(km_loop_##name##_begin, begin); \
        KM_ADDRESS(km_loop_##name##_end, end);     \
        return;                                    \
    }
        KM_BLOB_LIST(KM_CASE)
        begin = end = 0;
    }

}  // namespace

// Arguments: 0 report address; 1 runs; 2 repeats a run; then per run: blob, the address
// to run it from (0: in place), iterations, the data buffer for the memory loop, the
// fill word. Report words: 0 status (1 done, 2 refused), 1 runs done, 2 text address of
// the first blob, 3 its size; then runs * repeats cycle counts from word kReport.
void kernel_main() {
    volatile uint32_t* report = (
        reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(0))
    );
    const uint32_t runs = get_arg_val<uint32_t>(1);
    const uint32_t repeats = get_arg_val<uint32_t>(2);
    uintptr_t first_begin, first_end;
    blob(0, first_begin, first_end);
    report[0] = 0;
    report[2] = first_begin;
    report[3] = first_end - first_begin;
    for (uint32_t run = 0; run < runs; ++run) {
        const uint32_t id = get_arg_val<uint32_t>(3 + 5 * run);
        const uint32_t where = get_arg_val<uint32_t>(4 + 5 * run);
        const uint32_t iterations = get_arg_val<uint32_t>(5 + 5 * run);
        const uint32_t data = get_arg_val<uint32_t>(6 + 5 * run);
        const uint32_t fill = get_arg_val<uint32_t>(7 + 5 * run);
        uintptr_t begin, end;
        blob(id, begin, end);
        const uint32_t bytes = end - begin;
        uintptr_t entry = begin;
        if (where != 0) {
            const uint32_t base = (where - 4096) & ~4095u;
            if ((where & 3) != 0 || where - base + bytes > kRegionBytes) {
                report[0] = 2;
                report[4] = where;
                report[5] = base;
                report[6] = bytes;
                return;
            }
            // The rest of the region holds the fill word (nops by default), so no run
            // sees an earlier run's code.
            volatile uint32_t* region = reinterpret_cast<volatile uint32_t*>(base);
            for (uint32_t i = 0; i < kRegionBytes / 4; ++i) {
                region[i] = fill;
            }
            const uint32_t* from = reinterpret_cast<const uint32_t*>(begin);
            volatile uint32_t* to = reinterpret_cast<volatile uint32_t*>(where);
            for (uint32_t i = 0; i < bytes / 4; ++i) {
                to[i] = from[i];
            }
            entry = where;
        }
        for (uint32_t rep = 0; rep < repeats; ++rep) {
            if (rep == 0) {
                invalidate_own_icache();  // the first repeat starts cold, the rest warm
            }
            const uint32_t t0 = wall_low();
            reinterpret_cast<void (*)(uint32_t, uint32_t)>(entry)(iterations, data);
            report[kReport + run * repeats + rep] = wall_low() - t0;
        }
        report[1] = run + 1;
    }
    report[0] = 1;
}
