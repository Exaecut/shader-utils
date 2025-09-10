#pragma once

#ifdef USE_HALF_PRECISION
    typedef half4 pixel_format;
#else
    typedef float4 pixel_format;
#endif