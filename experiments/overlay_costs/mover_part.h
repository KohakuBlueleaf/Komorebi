// What the runtime's data-movement build puts before a kernel (brisck.cc), for an
// overlay compiled with the same command.
#pragma once

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
