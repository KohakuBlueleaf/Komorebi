/* Measuring image (core 0): the cost of loads, stores and fences in each place a
 * dispatcher can keep data. Result word i: picoseconds per operation of test i. */
#include <stdint.h>

#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_address_map.h"

#define MC_OPS 4096u
#define MC_PS_PER_TICK 20000u /* mtime runs at 50 MHz */
#define MC_RESULT 0x140u      /* KMD_LIM_RESULT */

/* Tests, in result word order. */
enum {
    MC_LOOP = 0,        /* the loop alone, a register add per pass */
    MC_STACK_LOAD = 1,  /* independent loads from the stack (LIM) */
    MC_STACK_STORE = 2, /* stores to the stack */
    MC_STACK_CHAIN = 3, /* dependent loads from the stack */
    MC_DDR_LOAD = 4,    /* independent loads, cached DRAM (Memory Port) */
    MC_DDR_STORE = 5,
    MC_DDR_CHAIN = 6,
    MC_LIM_POLL = 7, /* fence ir then a load of available LIM, as a poll does */
    MC_FENCE_IR = 8,
    MC_FENCE_OW = 9, /* with no store outstanding */
    MC_MTIME = 10,
    MC_STORE_FENCE = 11,  /* a stack store then fence ow */
    MC_DDR_FENCE_IR = 12, /* fence ir then a cached DRAM load */
    MC_TESTS = 13
};

typedef struct {
    uint64_t run_value, ddr_offset;
} mc_args_t;

static uint32_t per_op(uint64_t t0) {
    return (uint32_t)((x280_mtime() - t0) * MC_PS_PER_TICK / MC_OPS);
}

static __attribute__((noinline)) void measure(volatile uint32_t* out, uint64_t ddr) {
    volatile uint64_t stack[64];
    volatile uint64_t*
        dram = ((volatile uint64_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + ddr));
    volatile const uint64_t* lim = (volatile const uint64_t*)out;
    for (uint32_t i = 0; i < 64; i++) {
        stack[i] = (i + 1) & 63u;
        dram[i] = (i + 1) & 63u;
    }
    uint64_t acc = 0, t;
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        acc += i;
        __asm__ volatile("" : "+r"(acc));
    }
    out[MC_LOOP] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        acc += stack[i & 63u];
    }
    out[MC_STACK_LOAD] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        stack[i & 63u] = i;
    }
    out[MC_STACK_STORE] = per_op(t);
    for (uint32_t i = 0; i < 64; i++) {
        stack[i] = (i + 1) & 63u;
    }
    uint64_t x = 0;
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        x = stack[x];
    }
    out[MC_STACK_CHAIN] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        acc += dram[i & 63u];
    }
    out[MC_DDR_LOAD] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        dram[i & 63u] = i;
    }
    out[MC_DDR_STORE] = per_op(t);
    for (uint32_t i = 0; i < 64; i++) {
        dram[i] = (i + 1) & 63u;
    }
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        x = dram[x];
    }
    out[MC_DDR_CHAIN] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        x280_fence_ir();
        acc += lim[i & 7u];
    }
    out[MC_LIM_POLL] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        x280_fence_ir();
    }
    out[MC_FENCE_IR] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        x280_fence_ow();
    }
    out[MC_FENCE_OW] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        acc += x280_mtime();
    }
    out[MC_MTIME] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        stack[i & 63u] = i;
        x280_fence_ow();
    }
    out[MC_STORE_FENCE] = per_op(t);
    t = x280_mtime();
    for (uint32_t i = 0; i < MC_OPS; i++) {
        x280_fence_ir();
        acc += dram[i & 63u];
    }
    out[MC_DDR_FENCE_IR] = per_op(t);
    out[MC_TESTS] = (uint32_t)(acc + x);
}

int main(uint64_t core_id) {
    __asm__ volatile("csrs mstatus, %0" ::"r"((3ul << 9) | (3ul << 13)));
    if (core_id != 0) {
        return 0;
    }
    const volatile mc_args_t* a = (const volatile mc_args_t*)x280_hart_args(core_id);
    if (
        a == 0 || a->run_value != x280_hart_control(core_id)->run_value ||
        (a->ddr_offset & 63u) != 0
    ) {
        x280_hart_set_result(core_id, 0xE0);
        return 0;
    }
    measure((volatile uint32_t*)((uintptr_t)a + MC_RESULT), a->ddr_offset);
    x280_fence_ow();
    x280_hart_set_result(core_id, 0);
    return 0;
}
