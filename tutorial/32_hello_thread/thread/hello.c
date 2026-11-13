/* The smallest X280 thread: it adds the two numbers Python gave it. */
#include "kmx.h"

int kmx_main(int core) {
    (void)core;
    const kmx_args* arguments = kmx_args_get();
    const uint32_t sum = arguments->words[0] + arguments->words[1];
    kmx_result(0, &sum, 1);
    return 0;
}
