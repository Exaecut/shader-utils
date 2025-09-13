#pragma once

namespace tonemapping
{
	inline float srgb_to_linear_1(float x)
	{
		return (x <= 0.04045f) ? (x * (1.0f / 12.92f)) : pow((x + 0.055f) * (1.0f / 1.055f), 2.4f);
	}
	inline float3 srgb_to_linear(float3 c)
	{
		return float3(srgb_to_linear_1(c.r),
					  srgb_to_linear_1(c.g),
					  srgb_to_linear_1(c.b));
	}
	inline float linear_to_srgb_1(float x)
	{
		return (x <= 0.0031308f) ? (x * 12.92f) : (1.055f * pow(x, 1.0f / 2.4f) - 0.055f);
	}
	inline float3 linear_to_srgb(float3 c)
	{
		return float3(linear_to_srgb_1(c.r),
					  linear_to_srgb_1(c.g),
					  linear_to_srgb_1(c.b));
	}

	// Exposure - multiply scene linear by 2^EV
	inline float3 apply_exposure_linear(float3 rgb_linear, float EV)
	{
		return rgb_linear * exp2(EV); // exp2(EV) == 2^EV
	}

	// Filmic tonemap (ACES fitted) - good default for LDR targets
	inline float3 aces_filmic(float3 x)
	{
		const float a = 2.51f;
		const float b = 0.03f;
		const float c = 2.43f;
		const float d = 0.59f;
		const float e = 0.14f;
		return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
	}

	// Full path for an sRGB source to sRGB output
	inline float4 exposure_srgb_pipeline(float4 src_srgba, float EV, bool premultiplied)
	{
		float3 rgb = src_srgba.rgb;
		float a = src_srgba.a;

		if (premultiplied && a > 0.0f)
		{
			rgb /= a; // un-premultiply for correct color ops
		}

		// sRGB -> linear
		float3 lin = srgb_to_linear(rgb);

		// exposure in linear
		float3 lin_exposed = apply_exposure_linear(lin, EV);

		// optional tone map for LDR output
		float3 lin_mapped = aces_filmic(lin_exposed);

		// linear -> sRGB
		float3 out_rgb = linear_to_srgb(lin_mapped);

		if (premultiplied)
		{
			out_rgb *= a; // re-premultiply
		}

		return float4(out_rgb, a);
	}

}