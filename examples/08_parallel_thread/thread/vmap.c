/* f() over n words with the vector unit. The build's -march has no V: the function
 * turns it on for itself, in a file of its own (see README.md, "Vector"). */
#include <riscv_vector.h>
#include <stdint.h>

#include "vmap.h"

__attribute__((target("arch=+v"))) void vmap(
    uint32_t* d, const uint32_t* s, uint64_t n
) {
    while (n > 0) {
        const size_t vl = __riscv_vsetvl_e32m8(n);
        vuint32m8_t x = __riscv_vle32_v_u32m8(s, vl);
        x = __riscv_vxor_vv_u32m8(x, __riscv_vsrl_vx_u32m8(x, 15, vl), vl);
        x = __riscv_vmul_vx_u32m8(x, 2246822519u, vl);
        x = __riscv_vadd_vx_u32m8(x, 1u, vl);
        __riscv_vse32_v_u32m8(d, x, vl);
        s += vl;
        d += vl;
        n -= vl;
    }
}
