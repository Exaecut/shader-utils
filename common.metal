#ifndef UTILS_COMMON_METAL
#define UTILS_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

/// Compute normalized texture coordinates (UV) from thread grid position.
inline float2 tex_coord(uint2 gid, uint2 size_px) {
  return (float2(gid) + 0.5f) / float2(size_px);
}

/// Convert normalized UV into pixel coordinates.
inline float2 pixel_coord(float2 uv, uint2 size_px) {
  return uv * float2(size_px) - 0.5f;
}

/// Compute linear index for a 2D coordinate given a pitch (stride) in pixels.
inline uint index_of(uint2 xy, uint pitch_px) { return xy.y * pitch_px + xy.x; }

/// Clamp pixel coordinates to the valid [0..size_px-1] range.
inline uint2 clamp_xy(uint2 xy, uint2 size_px) {
  int x = clamp(int(xy.x), 0, int(size_px.x) - 1);
  int y = clamp(int(xy.y), 0, int(size_px.y) - 1);
  return uint2(x, y);
}

/// Read a float4 pixel from memory (float4 storage).
inline float4 image_read(device const float4 *data, uint pitch_px, uint2 size_px,
                         uint2 xy) {
  xy = clamp_xy(xy, size_px);
  return data[index_of(xy, pitch_px)];
}

/// Read a float4 pixel from memory (half4 storage).
inline float4 image_read(device const half4 *data, uint pitch_px, uint2 size_px,
                         uint2 xy) {
  xy = clamp_xy(xy, size_px);
  return float4(data[index_of(xy, pitch_px)]);
}

/// Write a float4 pixel to memory (float4 storage).
inline void image_write(device float4 *data, uint pitch_px, uint2 size_px,
                        uint2 xy, float4 c) {
  xy = clamp_xy(xy, size_px);
  data[index_of(xy, pitch_px)] = c;
}

/// Write a float4 pixel to memory (half4 storage).
inline void image_write(device half4 *data, uint pitch_px, uint2 size_px,
                        uint2 xy, float4 c) {
  xy = clamp_xy(xy, size_px);
  data[index_of(xy, pitch_px)] = half4(c);
}

/// Read bilinear-interpolated float4 pixel from memory (float4 storage).
inline float4 image_read_linear(device const float4 *data, uint pitch_px,
                                uint2 size_px, float2 uv) {
  float2 p = pixel_coord(uv, size_px);
  float2 pf = floor(p);
  float2 f = clamp(p - pf, 0.0f, 1.0f);

  uint2 xy00 = clamp_xy(uint2(pf), size_px);
  uint2 xy10 = clamp_xy(uint2(pf.x + 1.0f, pf.y), size_px);
  uint2 xy01 = clamp_xy(uint2(pf.x, pf.y + 1.0f), size_px);
  uint2 xy11 = clamp_xy(uint2(pf + 1.0f), size_px);

  float4 c00 = image_read(data, pitch_px, size_px, xy00);
  float4 c10 = image_read(data, pitch_px, size_px, xy10);
  float4 c01 = image_read(data, pitch_px, size_px, xy01);
  float4 c11 = image_read(data, pitch_px, size_px, xy11);

  float4 cx0 = mix(c00, c10, f.x);
  float4 cx1 = mix(c01, c11, f.x);
  return mix(cx0, cx1, f.y);
}

/// Read bilinear-interpolated float4 pixel from memory (half4 storage).
inline float4 image_read_linear(device const half4 *data, uint pitch_px,
                                uint2 size_px, float2 uv) {
  float2 p = pixel_coord(uv, size_px);
  float2 pf = floor(p);
  float2 f = clamp(p - pf, 0.0f, 1.0f);

  uint2 xy00 = clamp_xy(uint2(pf), size_px);
  uint2 xy10 = clamp_xy(uint2(pf.x + 1.0f, pf.y), size_px);
  uint2 xy01 = clamp_xy(uint2(pf.x, pf.y + 1.0f), size_px);
  uint2 xy11 = clamp_xy(uint2(pf + 1.0f), size_px);

  float4 c00 = image_read(data, pitch_px, size_px, xy00);
  float4 c10 = image_read(data, pitch_px, size_px, xy10);
  float4 c01 = image_read(data, pitch_px, size_px, xy01);
  float4 c11 = image_read(data, pitch_px, size_px, xy11);

  float4 cx0 = mix(c00, c10, f.x);
  float4 cx1 = mix(c01, c11, f.x);
  return mix(cx0, cx1, f.y);
}

/// Weighted mix that correctly handles alpha (like Porter–Duff "over").
inline float4 weighted_mix(float4 a, float4 b, float t) {
  float ow = a.w * (1.0 - t);
  float iw = b.w * t;
  float new_a = ow + iw;
  float recip = new_a != 0.0 ? 1.0 / new_a : 0.0;

  float3 rgb = (a.xyz * ow + b.xyz * iw) * recip;
  return float4(rgb, new_a);
}

/// Image2D wrapper type providing GLSL-like texture sampling.
///
/// Example:
/// ```metal
/// Image2D<float4> tex { incoming, p.inPitch, uint2(p.width, p.height) };
/// float4 c = tex.sampleLinear(uv);
/// ```
template <typename T> struct image_2d {
  device T *data;
  uint pitch_px;
  uint2 size_px;

  /// Read at integer coords.
  inline float4 read(uint2 xy) const {
    return image_read(data, pitch_px, size_px, xy);
  }

  /// Write at integer coords.
  inline void write(uint2 xy, float4 c) {
    image_write(data, pitch_px, size_px, xy, c);
  }

  /// Sample nearest at normalized UV.
  inline float4 sample_nearest(float2 uv) const {
    uint2 xy = clamp_xy(uint2(pixel_coord(uv, size_px) + 0.5f), size_px);
    return image_read(data, pitch_px, size_px, xy);
  }

  /// Sample bilinear at normalized UV.
  inline float4 sample_linear(float2 uv) const {
    return image_read_linear(data, pitch_px, size_px, uv);
  }

  /// Read at integer coords without bounds check (unsafe).
  inline float4 read_unchecked(uint2 xy) const {
    return data[index_of(xy, pitch_px)];
  }

  /// Write at integer coords without bounds check (unsafe).
  inline void write_unchecked(uint2 xy, float4 c) {
    data[index_of(xy, pitch_px)] = T(c);
  }
};

#endif // UTILS_COMMON_METAL
