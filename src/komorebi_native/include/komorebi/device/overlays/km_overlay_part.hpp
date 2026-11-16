// What the runtime's build puts before a kernel (brisck.cc, trisck.cc and the compute
// core's generated files), for an overlay part built with the same commands.
#pragma once

#if defined(COMPILE_FOR_TRISC)
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
#include "api/compute/common.h"
#else
#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
#endif
