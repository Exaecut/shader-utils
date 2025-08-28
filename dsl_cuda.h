#pragma once
// CUDA backend: emulate key Metal patterns while staying idiomatic CUDA.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// ---------- basic aliases to mirror Metal ----------
typedef unsigned int uint;
typedef ::uint2 uint2;
typedef ::uint3 uint3;
typedef ::uint4 uint4;

// vector_types.h provides float2/3/4, uint2/3/4.
// We add light operators so Metal-style math compiles cleanly.

__host__ __device__ static inline float2 make_float2(float x, float y)
{
	float2 v;
	v.x = x;
	v.y = y;
	return v;
}
__host__ __device__ static inline float3 make_float3(float x, float y, float z)
{
	float3 v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}
__host__ __device__ static inline float4 make_float4(float x, float y, float z, float w)
{
	float4 v;
	v.x = x;
	v.y = y;
	v.z = z;
	v.w = w;
	return v;
}

__host__ __device__ static inline uint2 make_uint2(uint x, uint y)
{
	uint2 v;
	v.x = x;
	v.y = y;
	return v;
}
__host__ __device__ static inline uint3 make_uint3(uint x, uint y, uint z)
{
	uint3 v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}
__host__ __device__ static inline uint4 make_uint4(uint x, uint y, uint z, uint w)
{
	uint4 v;
	v.x = x;
	v.y = y;
	v.z = z;
	v.w = w;
	return v;
}

#define float2(x, y) make_float2((x), (y))
#define float3(x, y, z) make_float3((x), (y), (z))
#define float4(x, y, z, w) make_float4((x), (y), (z), (w))

#define uint2(x, y) make_uint2((x), (y))
#define uint3(x, y, z) make_uint3((x), (y), (z))
#define uint4(x, y, z, w) make_uint4((x), (y), (z), (w))

// ---------- minimal floatN ops ----------
__host__ __device__ static inline float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__host__ __device__ static inline float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__host__ __device__ static inline float2 operator*(float2 a, float s) { return make_float2(a.x * s, a.y * s); }
__host__ __device__ static inline float2 operator*(float s, float2 a) { return a * s; }
__host__ __device__ static inline float2 operator/(float2 a, float s) { return make_float2(a.x / s, a.y / s); }

__host__ __device__ static inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ static inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ static inline float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ static inline float3 operator*(float s, float3 a) { return a * s; }
__host__ __device__ static inline float3 operator/(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }

__host__ __device__ static inline float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__host__ __device__ static inline float4 operator-(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__host__ __device__ static inline float4 operator*(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
__host__ __device__ static inline float4 operator*(float s, float4 a) { return a * s; }
__host__ __device__ static inline float4 operator/(float4 a, float s) { return make_float4(a.x / s, a.y / s, a.z / s, a.w / s); }

// ---------- math helpers to mirror Metal intrinsics ----------
template <typename T>
__host__ __device__ static inline T min_t(T a, T b) { return a < b ? a : b; }
template <typename T>
__host__ __device__ static inline T max_t(T a, T b) { return a > b ? a : b; }
template <typename T>
__host__ __device__ static inline T clamp_t(T x, T a, T b) { return min_t(max_t(x, a), b); }

__host__ __device__ static inline float mix(float a, float b, float t) { return a + (b - a) * t; }
__host__ __device__ static inline float2 mix(float2 a, float2 b, float t) { return a + (b - a) * t; }
__host__ __device__ static inline float3 mix(float3 a, float3 b, float t) { return a + (b - a) * t; }
__host__ __device__ static inline float4 mix(float4 a, float4 b, float t) { return a + (b - a) * t; }

__host__ __device__ static inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
__host__ __device__ static inline float smoothstep(float a, float b, float x)
{
	float t = clamp_t((x - a) / (b - a), 0.0f, 1.0f);
	return t * t * (3.0f - 2.0f * t);
}
__host__ __device__ static inline float fract(float x) { return x - floorf(x); }

// dot and length for float2/3
__host__ __device__ static inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
__host__ __device__ static inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ static inline float length(float2 v) { return sqrtf(dot(v, v)); }
__host__ __device__ static inline float length(float3 v) { return sqrtf(dot(v, v)); }

// ---------- qualifiers and address spaces ----------
#define kernel __global__
#define device __device__ __forceinline__
#define restrict_ptr __restrict__

// Map common Metal address-space keywords when user code uses them directly.
#define constant const
#define threadgroup_mem __shared__
#define thread_local /* local */

// ---------- parameter helpers ----------
#define param_dev_ro(T, name, slot) const T *name
#define param_dev_rw(T, name, slot) T *name
#define param_dev_wo(T, name, slot) T *name
#define param_dev_cbuf(T, name, slot) const T &name

// ---------- thread id bridging ----------
#define thread_pos_param(name) /* elided on CUDA */
#define thread_pos_init(name)                                      \
	uint2 name = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, \
							blockIdx.y * blockDim.y + threadIdx.y)

// ---------- barriers ----------
#define threadgroup_barrier_all() __syncthreads()