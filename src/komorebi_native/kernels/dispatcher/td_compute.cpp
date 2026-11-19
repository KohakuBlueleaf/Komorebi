// The dispatcher core's one compute kernel, by role: TRISC 0 nothing, TRISC 1 the
// thread (td_thread.hpp), TRISC 2 the watch on out and in (td_watch.hpp).
#include <stdint.h>

#include "td_thread.hpp"
#include "td_watch.hpp"

void kernel_main() {
#if defined(TRISC_MATH)
    td::thread();
#elif defined(TRISC_PACK)
    td::watch();
#endif
}
