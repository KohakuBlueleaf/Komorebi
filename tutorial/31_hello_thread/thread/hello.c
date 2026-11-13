/* The smallest X280 thread: adds the two numbers Python gave it and returns the sum. */
#include "kmx.h"

int kmx_main(int core) {
    (void)core;
    const kmx_args* arguments = kmx_args_get();
    const uint32_t sum = arguments->words[0] + arguments->words[1];
    kmx_result(0, &sum, 1);
    return 0;
}
