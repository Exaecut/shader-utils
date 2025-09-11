#pragma once

namespace operators
{
	// d: signed distance to shape (0 at edge)
	// glowColor: neon color
	// thickness: how wide the glow band should be
	inline float3 opGlowOutline(float d, float3 glowColor, float thickness)
	{
		float dist = abs(d);
		float x = clamp(1.0 - dist / thickness, 0.0, 1.0);
		float intensity = pow(x, 2.0);

		return glowColor * intensity;
	}

	// d: signed distance to shape (0 at edge)
	// color: neon base color
	// thickness: width of the solid stroke
	// glowRadius: controls the size of the glow halo
	inline float3 opNeonGlow(float d, float3 color, float thickness, float glowRadius)
	{
		float dist = abs(d);

		// --- Core stroke (solid inside band) ---
		float core = smoothstep(thickness, 0.0, dist);

		// --- Middle glow (soft falloff) ---
		float mid = exp(-3.0 * dist / glowRadius);

		// --- Outer halo (wider, softer) ---
		float outer = exp(-1.0 * dist / (2.0 * glowRadius));

		float3 result =
			color * core * 1.0	   // bright core
			+ color * mid * 0.6	   // middle glow
			+ color * outer * 0.3; // wide halo

		return result;
	}

}