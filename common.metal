#include <metal_stdlib>
using namespace metal;

inline float2 uv(uint2 gid, uint2 wh) {
  return (float2(gid) + 0.5f) / float2(wh);
}

inline float2 pix_coords(float2 uv, uint2 wh) { return uv * float2(wh) - 0.5f; }
inline uint index_of(uint2 xy, uint pitch_px) { return xy.y * pitch_px + xy.x; }

inline uint2 clamp_xy(uint2 xy, uint2 wh) {
  int x = clamp(int(xy.x), 0, int(wh.x) - 1);
  int y = clamp(int(xy.y), 0, int(wh.y) - 1);
  return uint2(x, y);
}

// imageRead / imageWrite (float4)
inline float4 imageRead(device const float4 *data, uint pitch_px, uint2 wh,
                        uint2 xy) {
  xy = clamp_xy(xy, wh);
  return data[index_of(xy, pitch_px)];
}

inline void imageWrite(device float4 *data, uint pitch_px, uint2 wh, uint2 xy,
                       float4 c) {
  xy = clamp_xy(xy, wh);
  data[index_of(xy, pitch_px)] = c;
}

// imageRead / imageWrite (half4)
inline float4 imageRead(device const half4 *data, uint pitch_px, uint2 wh,
                        uint2 xy) {
  xy = clamp_xy(xy, wh);
  return float4(data[index_of(xy, pitch_px)]);
}

inline void imageWrite(device half4 *data, uint pitch_px, uint2 wh, uint2 xy,
                       float4 c) {
  xy = clamp_xy(xy, wh);
  data[index_of(xy, pitch_px)] = half4(c);
}

// imageReadLinear (bilinear) : float4
inline float4 imageReadLinear(device const float4 *data, uint pitch_px,
                              uint2 wh, float2 uv_tr) {
  float2 p = pix_coords(uv_tr, wh);
  float2 pf = floor(p);
  float2 f = clamp(p - pf, 0.0f, 1.0f);

  uint2 xy00 = clamp_xy(uint2(pf), wh);
  uint2 xy10 = clamp_xy(uint2(pf.x + 1.0f, pf.y), wh);
  uint2 xy01 = clamp_xy(uint2(pf.x, pf.y + 1.0f), wh);
  uint2 xy11 = clamp_xy(uint2(pf + 1.0f), wh);

  float4 c00 = imageRead(data, pitch_px, wh, xy00);
  float4 c10 = imageRead(data, pitch_px, wh, xy10);
  float4 c01 = imageRead(data, pitch_px, wh, xy01);
  float4 c11 = imageRead(data, pitch_px, wh, xy11);

  float4 cx0 = mix(c00, c10, f.x);
  float4 cx1 = mix(c01, c11, f.x);
  return mix(cx0, cx1, f.y);
}

// imageReadLinear (bilinear) : half4
inline float4 imageReadLinear(device const half4 *data, uint pitch_px, uint2 wh,
                              float2 uv_tr) {
  float2 p = pix_coords(uv_tr, wh);
  float2 pf = floor(p);
  float2 f = clamp(p - pf, 0.0f, 1.0f);

  uint2 xy00 = clamp_xy(uint2(pf), wh);
  uint2 xy10 = clamp_xy(uint2(pf.x + 1.0f, pf.y), wh);
  uint2 xy01 = clamp_xy(uint2(pf.x, pf.y + 1.0f), wh);
  uint2 xy11 = clamp_xy(uint2(pf + 1.0f), wh);

  float4 c00 = imageRead(data, pitch_px, wh, xy00);
  float4 c10 = imageRead(data, pitch_px, wh, xy10);
  float4 c01 = imageRead(data, pitch_px, wh, xy01);
  float4 c11 = imageRead(data, pitch_px, wh, xy11);

  float4 cx0 = mix(c00, c10, f.x);
  float4 cx1 = mix(c01, c11, f.x);
  return mix(cx0, cx1, f.y);
}
