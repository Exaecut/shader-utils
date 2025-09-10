#pragma once
// Select backend: define SUPPORT_CUDA for CUDA, otherwise Metal.
#if defined(SUPPORT_CUDA)
    #include "dsl_cuda.h"
#else
    #include "dsl_metal.h"
#endif
