/* A thread that runs class Other's SCALE on a tensor of thread_tensors=
 * (../test_class_thread_with.py): scale_with names it by its load word's place. */
#include "kmx.h"
#include "other_thread.h"

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    uint32_t handle;
    if (
        other_scale_with(&command, 0u, 0u, 4u, 8u, OTHER_WORD_A, OTHER_WORD_Z) != KMX_OK
    ) {
        return 1;
    }
    if (
        other_scale_with(&command, 0u, 0u, 4u, 8u, OTHER_WORD_A, OTHER_WORD_WIDE) ==
        KMX_OK
    ) {
        return 2;
    }
    return kmx_submit(&command, &handle) == KMX_OK ? 0 : 3;
}
