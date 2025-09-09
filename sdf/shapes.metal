#include "maths/trigonometry.metal"

namespace shapes
{
	inline float circle(float2 uv, float2 center, float radius)
	{
		return length(uv - center) - radius;
	}

	inline float box(float2 uv, float2 center, float2 size)
	{
		float2 half_size = size * 0.5;
		float2 d = abs(uv - center) - half_size;
		return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
	}

	inline float rounded_box(float2 uv, float2 center, float2 size, float angle, float radius)
	{
		float max_r = 0.5 * min(size.x, size.y);
		radius = clamp(radius, 0.0, max_r);

		float2 p = uv - center;
		float c = cos(angle);
		float s = sin(angle);

		float2 local = float2(
			c * p.x + s * p.y,
			-s * p.x + c * p.y);

		float2 half_size = size * 0.5 - radius;
		float2 d = abs(local) - half_size;
		return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - radius;
	}

	inline float ngon(float2 uv, float2 center, float radius, float nsides, float angle)
	{
		float2 p = uv - center;

		float c = cos(angle);
		float s = sin(angle);
		float2 local = float2(c * p.x + s * p.y,
							  -s * p.x + c * p.y);

		float ang = atan2(local.y, local.x) + HALF_PI;
		float split = TAU / nsides;
		return length(local) * cos(split * floor(0.5 + ang / split) - ang) - radius;
	}

	inline float star(float2 uv, float2 center, float radius, int branches, float innerFactor, float angle)
	{
		float2 p = uv - center;

		float c = cos(angle);
		float s = sin(angle);
		p = float2(c * p.x - s * p.y, s * p.x + c * p.y);

		float angleStep = M_PI_F / float(branches);
		float innerAngle = M_PI_F / innerFactor;

		float2 acs = float2(cos(angleStep), sin(angleStep));
		float2 ecs = float2(cos(innerAngle), sin(innerAngle));

		float localAngle = fmod(atan2(p.x, p.y), 2.0 * angleStep) - angleStep;
		p = length(p) * float2(cos(localAngle), fabs(sin(localAngle)));

		p -= radius * acs;
		p += ecs * clamp(-dot(p, ecs), 0.0, radius * acs.y / ecs.y);

		return length(p) * sign(p.x);
	}

	inline float moon(float2 uv, float2 center, float d, float ra, float rb)
	{
		float2 p = uv - center;
		p.y = fabs(p.y);
		float a = (ra * ra - rb * rb + d * d) / (2.0 * d);
		float b = sqrt(max(ra * ra - a * a, 0.0));
		if (d * (p.x * b - p.y * a) > d * d * max(b - p.y, 0.0))
		{
			return length(p - float2(a, b));
		}
		return max(length(p) - ra, -(length(p - float2(d, 0)) - rb));
	}

	inline float oriented_vesica(float2 uv, float2 center, float2 size, float angle)
	{
		float2 p = uv - center;

		float c = cos(angle);
		float s = sin(angle);
		float2 local = float2(c * p.x + s * p.y,
							  -s * p.x + c * p.y);

		float ra = size.x * 0.5;
		float rb = size.y * 0.5;
		float d = size.x;

		float a = (ra * ra - rb * rb + d * d) / (2.0 * d);
		float b = sqrt(max(ra * ra - a * a, 0.0));

		if (d * (local.x * b - local.y * a) > d * d * max(b - local.y, 0.0))
		{
			return length(local - float2(a, b));
		}
		return max(length(local) - ra,
				   -(length(local - float2(d, 0.0)) - rb));
	}

	inline float heart(float2 uv, float2 center, float radius)
	{
		float2 p = (uv - center) / radius;
		p.x = fabs(p.x);

		if (p.y + p.x > 1.0)
		{
			return (length(p - float2(0.25, 0.75)) - sqrt(2.0) / 4.0) * radius;
		}
		float2 a = p - float2(0.0, 1.0);
		float2 b = p - 0.5 * max(p.x + p.y, 0.0);
		return (min(length(a), length(b)) * sign(p.x - p.y)) * radius;
	}

	inline float rounded_x(float2 uv, float2 center, float width, float radius)
	{
		float2 p = abs(uv - center);
		return length(p - min(p.x + p.y, width) * 0.5) - radius;
	}
}
