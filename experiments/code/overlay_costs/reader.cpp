// A reader overlay: api.h's reader loop, its runtime calls through the table
// (OV_TABLE) or compiled in (the shared state by the persistent kernel's symbols).
#include "mover_part.h"
// clang-format off: the part header comes first
#include "api.h"
#include "entry.h"
#include "overlay_gen.h"

extern "C" uint32_t ov_body(const cs::Call* call) {
#if OV_TABLE
    return cs::reader_loop(*call, *static_cast<const cs::MoverApi*>(call->api));
#else
    return cs::reader_loop(*call, cs::InlineMover{});
#endif
}
