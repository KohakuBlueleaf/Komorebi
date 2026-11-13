/* mix() over `words` words with the vector unit. The build's -march has no V: the
 * function turns it on for itself, in a file of its own (see README.md, "Vector"). */
#include <riscv_vector.h>
#include <stdint.h>

#include "vmap.h"

__attribute__((target("arch=+v"))) void vmap(
    uint32_t* destination, const uint32_t* source, uint64_t words
) {
    while (words > 0) {
        const size_t vector_length = __riscv_vsetvl_e32m8(words);
        vuint32m8_t x = __riscv_vle32_v_u32m8(source, vector_length);
        x = (
            __riscv_vxor_vv_u32m8(
                x, __riscv_vsrl_vx_u32m8(x, 15, vector_length), vector_length
            )
        );
        x = __riscv_vmul_vx_u32m8(x, 2246822519u, vector_length);
        x = __riscv_vadd_vx_u32m8(x, 1u, vector_length);
        __riscv_vse32_v_u32m8(destination, x, vector_length);
        source += vector_length;
        destination += vector_length;
        words -= vector_length;
    }
}
