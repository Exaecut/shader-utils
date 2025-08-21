/// Compute normalized texture coordinates (UV) from thread grid position.
inline float2 tex_coord(uint2 gid, uint2 size_px) {
  return (float2(gid) + 0.5f) / float2(size_px);
}

/// Convert normalized UV into pixel coordinates.
inline float2 pixel_coord(float2 uv, uint2 size_px) {
  return uv * float2(size_px) - 0.5f;
}