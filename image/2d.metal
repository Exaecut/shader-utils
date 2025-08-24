/// Compute linear index for a 2D coordinate given a pitch (stride) in pixels.
inline uint index_of(uint2 xy, uint pitch_px) { return xy.y * pitch_px + xy.x; }

/// Clamp pixel coordinates to the valid [0..size_px-1] range.
inline uint2 clamp_xy(uint2 xy, uint2 size_px)
{
	int x = clamp(int(xy.x), 0, int(size_px.x) - 1);
	int y = clamp(int(xy.y), 0, int(size_px.y) - 1);
	return uint2(x, y);
}

/// Read a float4 pixel from memory (float4 storage).
inline float4 image_read(device const float4 *data, uint pitch_px,
						 uint2 size_px, uint2 xy)
{
	xy = clamp_xy(xy, size_px);
	return data[index_of(xy, pitch_px)];
}

/// Read a float4 pixel from memory (half4 storage).
inline float4 image_read(device const half4 *data, uint pitch_px, uint2 size_px,
						 uint2 xy)
{
	xy = clamp_xy(xy, size_px);
	return float4(data[index_of(xy, pitch_px)]);
}

/// Write a float4 pixel to memory (float4 storage).
inline void image_write(device float4 *data, uint pitch_px, uint2 size_px,
						uint2 xy, float4 c)
{
	xy = clamp_xy(xy, size_px);
	data[index_of(xy, pitch_px)] = c;
}

/// Write a float4 pixel to memory (half4 storage).
inline void image_write(device half4 *data, uint pitch_px, uint2 size_px,
						uint2 xy, float4 c)
{
	xy = clamp_xy(xy, size_px);
	data[index_of(xy, pitch_px)] = half4(c);
}

/// Read bilinear-interpolated float4 pixel from memory (float4 storage).
inline float4 image_read_linear(device const float4 *data, uint pitch_px,
								uint2 size_px, float2 uv)
{
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
								uint2 size_px, float2 uv)
{
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
inline float4 weighted_mix(float4 a, float4 b, float t)
{
	float ow = a.w * (1.0 - t);
	float iw = b.w * t;
	float new_a = ow + iw;
	float recip = new_a != 0.0 ? 1.0 / new_a : 0.0;

	float3 rgb = (a.xyz * ow + b.xyz * iw) * recip;
	return float4(rgb, new_a);
}
/// Helpers to convert between float4 and storage types
inline float4 to_float4(float4 v) { return v; }
inline float4 to_float4(half4 v) { return float4(v); }

template <typename T>
inline T from_float4(float4 v);

template <>
inline float4 from_float4<float4>(float4 v) { return v; }

template <>
inline half4 from_float4<half4>(float4 v) { return half4(v); }

/// Layout policies
struct layout_rgba
{
	inline float4 to_rgba(float4 c) const { return c; }
	inline float4 from_rgba(float4 c) const { return c; }
};

struct layout_bgra
{
	inline float4 to_rgba(float4 c) const { return float4(c.b, c.g, c.r, c.a); }
	inline float4 from_rgba(float4 c) const { return float4(c.z, c.y, c.x, c.w); }
};

/// Image2D wrapper type providing texture sampling and channel layout
/// abstraction. Works for float4 or half4 storage transparently.
///
/// Example:
/// ```metal
/// Image2D<float4> tex { incoming, p.inPitch, uint2(p.width, p.height) };
/// float4 c = tex.sampleLinear(uv);
/// ```
template <typename Storage, typename Layout = layout_rgba>
struct image_2d
{
	device Storage *data;
	uint pitch_px;
	uint2 size_px;
	Layout layout;

	/// Read at integer coords.
	inline float4 read(uint2 xy) const
	{
		xy = clamp_xy(xy, size_px);
		return layout.to_rgba((float4)data[index_of(xy, pitch_px)]);
	}

	/// Write at integer coords.
	inline void write(uint2 xy, float4 c)
	{
		xy = clamp_xy(xy, size_px);
		data[index_of(xy, pitch_px)] = (Storage)layout.from_rgba(c);
	}

	/// Sample nearest at normalized UV.
	inline float4 sample_nearest(float2 uv) const
	{
		uint2 xy = clamp_xy(uint2(pixel_coord(uv, size_px) + 0.5f), size_px);
		return layout.to_rgba((float4)data[index_of(xy, pitch_px)]);
	}

	/// Sample bilinear at normalized UV.
	inline float4 sample_linear(float2 uv) const
	{
		return layout.to_rgba(
			image_read_linear(data, pitch_px, size_px, uv));
	}

	    /// Sample bilinear with repeat addressing.
    inline float4 sample_linear_repeat(float2 uv) const
    {
        float2 uv_wrapped = fract(uv); // wrap into [0,1)
        return layout.to_rgba(
            image_read_linear(data, pitch_px, size_px, uv_wrapped));
    }

    /// Sample nearest with repeat addressing.
    inline float4 sample_nearest_repeat(float2 uv) const
    {
        float2 uv_wrapped = fract(uv); // wrap into [0,1)
        uint2 xy = clamp_xy(uint2(pixel_coord(uv_wrapped, size_px) + 0.5f), size_px);
        return layout.to_rgba((float4)data[index_of(xy, pitch_px)]);
    }

    /// Sample bilinear with mirror addressing.
    inline float4 sample_linear_mirror(float2 uv) const
    {
        float2 uv_mirrored = abs(fract(uv * 0.5f) * 2.0f - 1.0f); // mirror repeat
		uv_mirrored.x = 1.0 - uv_mirrored.x;
		uv_mirrored.y = 1.0 - uv_mirrored.y;
        return layout.to_rgba(
            image_read_linear(data, pitch_px, size_px, float2(uv_mirrored.x, uv_mirrored.y)));
    }

    /// Sample nearest with mirror addressing.
    inline float4 sample_nearest_mirror(float2 uv) const
    {
        float2 uv_mirrored = abs(fract(uv * 0.5f) * 2.0f - 1.0f); // mirror repeat
		uv_mirrored.y = 1.0 - uv_mirrored.y;
        uint2 xy = clamp_xy(uint2(pixel_coord(uv_mirrored, size_px) + 0.5f), size_px);
        return layout.to_rgba((float4)data[index_of(xy, pitch_px)]);
    }


	/// Read without bounds check.
	inline float4 read_unchecked(uint2 xy) const
	{
		return layout.to_rgba((float4)data[index_of(xy, pitch_px)]);
	}

	/// Write without bounds check.
	inline void write_unchecked(uint2 xy, float4 c)
	{
		data[index_of(xy, pitch_px)] = (Storage)layout.from_rgba(c);
	}
};
