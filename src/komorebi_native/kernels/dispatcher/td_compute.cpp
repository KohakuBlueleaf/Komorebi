// The dispatcher core's one compute kernel, by role: TRISC 0 the engines, TRISC 1 the
// thread (td_thread.hpp), TRISC 2 the watch (td_watch.hpp).
#include <stdint.h>

#if defined(TRISC_UNPACK)
#include "td_engine.hpp"
#endif
#include "td_thread.hpp"
#include "td_watch.hpp"

void kernel_main() {
#if defined(TRISC_UNPACK)
    td::engine();
#elif defined(TRISC_MATH)
    td::thread();
#elif defined(TRISC_PACK)
    td::watch();
#endif
}
