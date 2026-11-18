// The map/collect example's reader with its free stack painted first and measured last.
#include "api/dataflow/dataflow_api.h"
#include "../../../../../examples/01_mapcollect/kernels/mapcollect.h"
#include "stack.h"

// not inlined: its frame lies below the painted sp, so it counts as used
void example_kernel_main() __attribute__((noinline));
#define kernel_main example_kernel_main
#include "../../../../../examples/01_mapcollect/kernels/reader.cpp"
#undef kernel_main

void kernel_main() {
    const uint32_t sp = stack::paint();
    example_kernel_main();
    stack::record(0, sp);
}
