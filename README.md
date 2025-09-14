# exaecut/shader-utils

Metal-first shader utility pack with a tiny DSL that makes Metal-style kernels
compile on both Apple Metal and the CUDA runtime. Write once in a Metal-like
dialect, choose backend with a single macro, and keep your math, image, and SDF
helpers identical across platforms.

> License: Apache-2.0

---

## Why this exists

- **One DSL, two backends** - `dsl.h` switches between Metal and CUDA with
  `SUPPORT_CUDA`.
- **Familiar Metal syntax** - Keep `float2/3/4`, `thread_position_in_grid`, and
  address spaces while running on CUDA.
- **Image helpers** - `image_2d<T, Layout>` for reads, writes, and bilinear
  sampling with repeat or mirror.
- **Math + SDF toolbox** - Easing, trig, transforms, signed distance shapes, and
  operators.
- **Filters** - Practical blur building blocks.

### TLDR features ⚡️

- Metal-like code that also compiles under CUDA ✅
- Vector constructors and `mix`, `smoothstep`, `fract`, `dot`, `length` ✅
- Address space and parameter shims like `param_dev_ro`, `param_dev_rw`,
  `param_dev_cbuf` ✅
- `thread_pos_param` and `thread_pos_init` for grid indices ✅
- Image layout policies: `layout_rgba`, `layout_bgra` ✅

---

## Directory

```dir
shader-utils/
├─ common.metal           # Aggregates DSL + math + image
├─ dsl.h                  # Backend switch
├─ dsl_cuda.h             # CUDA shims
├─ dsl_metal.h            # Metal shims
├─ types.metal            # Pixel format typedefs
├─ filters/
│  └─ blur.metal          # Gaussian, radial, directional blur
├─ image/
│  ├─ 2d.metal            # image_2d helpers
│  ├─ coords.metal        # uv helpers
│  └─ tonemapping.metal   # sRGB, ACES, exposure
├─ maths/
│  ├─ easing.metal
│  ├─ transform.metal
│  └─ trigonometry.metal
└─ sdf/
   ├─ operators.metal
   ├─ sdf.metal
   └─ shapes.metal
```

---

## Add as a submodule

```bash
git submodule add <YOUR_ORIGIN>/exaecut/shader-utils.gthird_party/shader-utilsThen include headers from `third_party/shader-utils
```

Choose backend

- **CUDA build**: define `SUPPORT_CUDA`.
- **Metal build**: do not define `SUPPORT_CUDA`.

### CMake sketch

```cmake
add_library(exaecut_shader_utils INTERFACE)
target_include_directories(exaecut_shader_utils INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/shader-utils)

# Your CUDA target
add_library(my_cuda_kernels OBJECT tint_kernels.cu)
target_compile_definitions(my_cuda_kernels PRIVATE SUPPORT_CUDA)
target_link_libraries(my_cuda_kernels PRIVATE exaecut_shader_utils)

# Your Metal target (Xcode toolchain compiles .metal files)
# add_custom_command(... metal ...)
````

### CUDA compile line example

```bash
nvcc -DSUPPORT_CUDA \
     -I third_party/shader-utils \
     -arch=sm_86 -x cu -std=c++17 \
     -c tint_kernels.cu -o tint_kernels.cu.o
```

---

## Quick start example

Below is a minimal tint kernel that:

- reads a `float4*` source image via `image_2d<const float4, layout_bgra>`
- writes to a `float4*` destination
- consumes a constant buffer with `tint_color` and `intensity`

> Works on Metal or CUDA by toggling `SUPPORT_CUDA`.

```cpp
// file: tint.metal or tint_kernels.cu
#include "common.metal"   // pulls in dsl.h, math, and image helpers

// minimal image params for pitch and size
struct image_params {
    uint src_pitch;
    uint dst_pitch;
    uint width;
    uint height;
};

// user parameters passed as a constant buffer
struct tint_params {
    float3 tint_color;  // 0..1
    float  intensity;   // 0..1, 0 = no tint, 1 = full tint
};

kernel void tint_image(param_dev_ro(float4, src_buf, 0),
                       param_dev_rw(float4, dst_buf, 1),
                       param_dev_cbuf(image_params, img, 2),
                       param_dev_cbuf(tint_params,  tint, 3),
                       thread_pos_param(gid))
{
    thread_pos_init(gid);
    if (gid.x >= img.width || gid.y >= img.height) return;

    image_2d<const float4, layout_bgra> src { src_buf, img.src_pitch, uint2(img.width, img.height) };
    image_2d<float4,       layout_bgra> dst { dst_buf, img.dst_pitch, uint2(img.width, img.height) };

    float4 c = src.read(gid);

    // apply tint - lerp the rgb toward rgb * tint_color by intensity
    float3 tinted = mix(c.rgb, c.rgb * tint.tint_color, clamp(tint.intensity, 0.0f, 1.0f));
    float4 outc = float4(tinted, c.a);

    dst.write(gid, outc);
}
```

### Host side notes

- **Metal**: bind buffers at slots 0..3 matching the signature and dispatch a
  grid sized `(width, height)`.
- **CUDA**: pass device pointers for `src_buf` and `dst_buf`, pass
  `image_params` and `tint_params` by value, launch with a 2D grid. The DSL maps
  `thread_pos_param` and `thread_pos_init` to `blockIdx` and `threadIdx` under
  CUDA.

```cpp
// CUDA launch sketch (not exhaustive)
dim3 block(16,16);
dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
image_params img{src_pitch_px, dst_pitch_px, width, height};
tint_params  tp{{1.0f, 0.2f, 0.2f}, 0.75f};

tint_image<<<grid, block>>>(src_dev, dst_dev, img, tp);
```

---

## DSL reference in one screen

- `#include "dsl.h"` switches the shims:

  - `dsl_metal.h` includes `<metal_stdlib>` and maps parameters to
    `[[buffer(N)]]`
  - `dsl_cuda.h` includes CUDA headers and provides minimal `floatN` ops and
    math helpers
- Qualifiers:

  - `param_dev_ro(T,name,slot)` - device const T\*
  - `param_dev_rw(T,name,slot)` - device T\*
  - `param_dev_wo(T,name,slot)` - device T\*
  - `param_dev_cbuf(T,name,slot)` - constant T&
  - `thread_pos_param(gid)` - kernel thread id param
  - `thread_pos_init(gid)` - backend specific init
- Barriers:

  - `threadgroup_barrier_all()`

---

## Tips

- Prefer `image_2d<pixel_format, layout_bgra>` if you want the storage to follow
  `types.metal` and compile with half or float based on `USE_HALF_PRECISION`.
- Use helpers in `image/coords.metal` for uv math like `tex_coord(gid, size)`.
- Keep parameters POD and small for best cross-ABI results.

### What POD means here

POD stands for Plain Old Data. In practice for kernel parameter structs this
means:

- **Standard layout and trivially copyable** - no constructors, destructors,
  methods, virtual functions, or inheritance.
- **No references or pointers inside the struct** - use only scalars like
  `uint`, `float`, and vector types like `float2/3/4`.
- **Fixed size and alignment** - add explicit padding if you mix 64-bit and
  32-bit fields, and keep host and device definitions identical.
- **No STL or dynamic fields** - keep it POD so it can be passed by value to
  both Metal and CUDA safely.

---

## License

Apache-2.0. See `LICENSE`.
