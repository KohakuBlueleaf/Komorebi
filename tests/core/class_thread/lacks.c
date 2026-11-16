/* A thread of class Totals (../test_class_thread_programs.py) naming a handler Totals
 * lacks beside another program's: refused at the make. */
#include "kmx.h"
#include "totals_thread.h"

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    plain_scale(&command, 0u, 0u, 4u, 1u);
    totals_nothing(&command, 0u, 0u, 4u);
    return 0;
}
