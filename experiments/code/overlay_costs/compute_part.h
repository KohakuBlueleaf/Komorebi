// What the runtime's compute build puts before a compute kernel (chlkc_list.h and
// its generated per-core file), for an overlay part compiled with the same command.
#pragma once

#include "ckernel.h"
#include "ckernel_gpr_map.h"
#include "internal/debug/fw_debug.h"
#include "llk_param_structs.h"

using namespace ckernel;

#include "chlkc_descriptors.h"
#if defined(UCK_CHLKC_UNPACK)
#define TRISC_UNPACK
#elif defined(UCK_CHLKC_MATH)
#define TRISC_MATH
#elif defined(UCK_CHLKC_PACK)
#define TRISC_PACK
#endif
#include "defines_generated.h"
