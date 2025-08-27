#pragma once

#define stdlib <metal_stdlib>; \
using namespace metal

#include stdlib;

// ---------- address spaces and qualifiers ----------
#define restrict                       __restrict

// explicit address-space helpers for symmetry with CUDA side
#define device_ptr(T)                   device T*
#define device_cptr(T)                  device const T*
#define constant_ref(T)                 constant T&
#define threadgroup_mem                 threadgroup
#define thread_local                    thread

// ---------- parameter helpers ----------
#define param_dev_ro(T, name, slot)     device_cptr(T) name [[buffer(slot)]]
#define param_dev_rw(T, name, slot)     device_ptr(T) name [[buffer(slot)]]
#define param_dev_wo(T, name, slot)     device_ptr(T) name [[buffer(slot)]]
#define param_dev_cbuf(T, name, slot)       constant_ref(T) name [[buffer(slot)]]

// ---------- thread id bridging ----------
#define thread_pos_param(name)          uint2 name [[thread_position_in_grid]]
#define thread_pos_init(name)           /* no-op on Metal */

// ---------- barriers ----------
#define threadgroup_barrier_all()       threadgroup_barrier(mem_flags::mem_threadgroup)

// ---------- constructors to match CUDA naming on Metal ----------
static inline float2 make_float2(float x, float y)                { return float2(x, y); }
static inline float3 make_float3(float x, float y, float z)       { return float3(x, y, z); }
static inline float4 make_float4(float x, float y, float z, float w){ return float4(x, y, z, w); }

static inline uint2  make_uint2(uint x, uint y)                   { return uint2(x, y); }
static inline uint3  make_uint3(uint x, uint y, uint z)           { return uint3(x, y, z); }
static inline uint4  make_uint4(uint x, uint y, uint z, uint w)   { return uint4(x, y, z, w); }