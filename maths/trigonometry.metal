#pragma once

#ifndef TRIGONOMETRY_H
#define TRIGONOMETRY_H

constant float PI = 3.14159265358979323846f;
constant float TAU = 6.28318530717958647692f;	  // 2 * PI
constant float HALF_PI = 1.57079632679489661923f; // PI / 2
constant float INV_PI = 0.31830988618379067154f;  // 1 / PI

/// Convert degrees to radians
inline float radians(float deg)
{
	return deg * (M_PI_F / 180.0f);
}

/// Convert radians to degrees
inline float degrees(float rad)
{
	return rad * (180.0f / M_PI_F);
}

#endif // TRIGONOMETRY_H