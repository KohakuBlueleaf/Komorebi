// The ray tracer's walk headers, as they are, for the pack CPU alone: a body's #include
// lines go to the top of the kernel of every compute CPU (temporary, see README.md).
#pragma once

#ifdef TRISC_PACK
#include "../../../projects/raytrace/kernels/walk.hpp"
#endif
