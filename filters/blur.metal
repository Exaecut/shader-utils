#include <metal_stdlib>
using namespace metal;

/// Compute 2D Gaussian weight at integer offset (x,y).
inline float gaussian_weight(int x, int y, float sigma)
{
	float s2 = sigma * sigma;
	return exp(-(x * x + y * y) / (2.0f * s2));
}

/// One-pass separable Gaussian blur (horizontal or vertical)
/// - tex: source image (must have .size_px and .sample_linear(uv))
/// - uv: normalized coordinates
/// - sigma: standard deviation
/// - radius: kernel half width (≈ 3 * sigma)
/// - vertical: true = vertical blur, false = horizontal
template <typename Image>
inline float4 gaussian_1d(Image tex, float2 uv, float sigma, int radius, bool vertical)
{
	float2 texel_size = 1.0 / float2(tex.size_px);
	float2 dir = vertical ? float2(0.0, 1.0) : float2(1.0, 0.0);

	float4 sum = float4(0.0);
	float weight_sum = 0.0;

	// Center sample
	float wc = gaussian_weight_1d(0, sigma);
	sum += tex.sample_linear(uv) * wc;
	weight_sum += wc;

	// Symmetric taps
	for (int i = 1; i <= radius; ++i)
	{
		float w = gaussian_weight_1d(i, sigma);
		float2 offset = dir * (float(i) * texel_size);

		float4 c1 = tex.sample_linear(uv + offset);
		float4 c2 = tex.sample_linear(uv - offset);

		sum += (c1 + c2) * w;
		weight_sum += 2.0 * w;
	}

	return sum / max(weight_sum, 1e-8);
}

/// One-pass 2D Gaussian blur
/// - tex: source image
/// - uv: normalized coordinates
/// - sigma: standard deviation of Gaussian
/// - radius: how many pixels around to sample (commonly ~3*sigma)
template <typename Image>
inline float4 gaussian_2d(Image tex, float2 uv, float sigma, int radius)
{
	float4 sum = float4(0.0);
	float weight_sum = 0.0;

	for (int y = -radius; y <= radius; ++y)
	{
		for (int x = -radius; x <= radius; ++x)
		{
			float w = gaussian_weight(x, y, sigma);
			float2 offset = float2(x, y) / float2(tex.size_px); // pixel offset to UV
			float4 c = tex.sample_linear(uv + offset);

			sum += c * w;
			weight_sum += w;
		}
	}

	return sum / weight_sum;
}

/// Linear radial (zoom) blur sampled along the ray from `center` to `uv`.
/// - `strength` in [0..1]: how far we march toward the center (0 = none, 1 = up to center).
/// - `center`: blur origin in normalized UV.
/// - `influence` in [0..1]: radius falloff; effective blur amount is
///    smoothstep(1 - influence, 1, r), where r is distance from center (aspect-corrected).
///    Examples:
///      influence=1.0 -> full blur everywhere.
///      influence=0.5 -> no blur at center, reaches full blur at ~50% radius.
///      influence=0.25 -> clear until ~75% radius, then ramps to full.
/// - `aspect`: width/height to keep the influence field circular on non-square frames.
/// - Implementation details:
///   * Uses N taps with linearly spaced positions between uv and center.
///   * Triangle weights (heavier near the original sample) to reduce banding.
template <typename Storage, typename Layout = layout_rgba>
inline float4 radial_blur_linear(image_2d<const Storage, Layout> src,
								 float2 uv,
								 float strength,
								 float2 center,
								 float influence,
								 uint2 gid)
{
	float2 direction = (uv - center) * strength;
	int samples = 16;

	if (influence <= 0.0001f)
	{
		return src.sample_linear(uv);
	}

	float weight_accum = 0.0;
	float4 color_accum = float4(0.0);

	float influence_mask = smoothstep(1.0f - influence, 1.0f, length(direction));

	float rand = fract(sin(dot(float2(gid), float2(12.9898f, 78.233f))) * 43758.5453f);
	for (int i = 0; i < samples; ++i)
	{
		float progression = ((float(i) + 0.5f) + (rand - 0.5f)) / float(samples);
		float2 offset = uv + direction * influence_mask * progression;

		float weight = (1.0f - progression);
		weight *= weight;

		color_accum += src.sample_linear_mirror(offset) * weight;
		weight_accum += weight;
	}

	return color_accum / max(weight_accum, 1e-6f);
}

/// Directional blur along an angle.
/// - `angle_deg`: direction in degrees (0 = +X, 90 = +Y).
/// - `strength`: length of blur (normalized, ~0..1).
/// - `spread`: controls noise/jitter spread of taps (0 = uniform, >0 = noisier).
template <typename Storage, typename Layout = layout_rgba>
inline float4 directional_blur_linear(image_2d<const Storage, Layout> src,
									  float2 uv,
									  float angle_deg,
									  float strength,
									  float spread,
									  uint2 gid)
{
	// Convert angle to radians
	float angle = radians(angle_deg);

	// Direction vector
	float2 dir = float2(cos(angle), sin(angle)) * strength;

	int samples = 32;

	float weight_accum = 0.0;
	float4 color_accum = float4(0.0);

	// Stable per-pixel random jitter
	float rand = fract(sin(dot(float2(gid), float2(12.9898f, 78.233f))) * 43758.5453f);

	for (int i = 0; i < samples; ++i)
	{
		float progression = (float(i) + 0.5f) / float(samples);

		// Add jitter based on spread
		float jitter = (rand - 0.5f) * spread / float(samples);

		float2 offset = uv + dir * (progression + jitter);

		// Triangle weighting (stronger near center sample)
		// float weight = (1.0f - progression);
		// weight *= weight;
		float weight = exp(-progression * progression * 4.0);

		color_accum += src.sample_linear_mirror(offset) * weight;
		weight_accum += weight;
	}

	return color_accum / max(weight_accum, 1e-6f);
}
