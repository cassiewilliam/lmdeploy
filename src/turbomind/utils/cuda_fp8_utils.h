/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef ENABLE_FP8
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

inline __device__ float half_to_float(uint16_t h) {
  float f;
#ifndef _MSC_VER
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
#endif
  return f;
}

inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef _MSC_VER
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#endif
  return tmp.u16[0];
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout
vec_conversion(const Tin &x)
{
  return x;
}

// fp8 -> half
template <>
__inline__ __device__ half
vec_conversion<half, __nv_fp8_e4m3>(const __nv_fp8_e4m3 &a)
{
  __nv_fp8_storage_t fp8_val = *reinterpret_cast<const __nv_fp8_storage_t*>(&a);
  __half_raw res = __nv_cvt_fp8_to_halfraw(fp8_val, __NV_E4M3);
  return *reinterpret_cast<const half*>(&(res.x));
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t>(const uint16_t &a)
{
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, __NV_E4M3);
  tmp.u16[0] = res.x;
  tmp.u16[1] = res.y;
  return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(const uint32_t &a)
{
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] = vec_conversion<uint32_t, uint16_t>((uint16_t)a);
  tmp.u32[1] = vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U));
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4 vec_conversion<uint4, uint2>(const uint2 &a) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = vec_conversion<uint2, uint32_t>(a.x);
  tmp.u64[1] = vec_conversion<uint2, uint32_t>(a.y);
  return tmp.u64x2;
}

/**
* Scaled and vectorized conversions, for data exchange between high and low
* precision domains Convention of the scale in API, e.g: FP8_data =
* Quantization( High_Precision_data / scale ) s.t. Quantize(HP / scale) => FP8
* Dequant(FP8) * scale =>  HP
*/

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(const Tin& x, const float& scale)
{
  return x;
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(const uint16_t &a, const float& scale)
{
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, __NV_E4M3);
  tmp.u16[0] = float_to_half(half_to_float(res.x) * scale);
  tmp.u16[1] = float_to_half(half_to_float(res.y) * scale);
  return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(const uint32_t &a, const float& scale)
{
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale);
  tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U), scale);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4 scaled_vec_conversion<uint4, uint2>(const uint2 &a, const float& scale) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale);
  return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
vec_conversion<__nv_bfloat16, __nv_fp8_e4m3>(const __nv_fp8_e4m3 &a)
{
  // Note there is no direct convert function from fp8 to bf16.
  // fp8 -> half
  __nv_fp8_storage_t fp8_val = *reinterpret_cast<const __nv_fp8_storage_t*>(&a);
  __half_raw tmp = __nv_cvt_fp8_to_halfraw(fp8_val, __NV_E4M3);
  // half -> float -> bf16
  return __float2bfloat16(half_to_float(tmp.x));
}

// half -> fp8
template <>
__inline__ __device__ __nv_fp8_e4m3
vec_conversion<__nv_fp8_e4m3, half>(const half &a)
{
  __half_raw tmp;
  tmp.x = *reinterpret_cast<const uint16_t*>(&a);
  __nv_fp8_storage_t res = __nv_cvt_halfraw_to_fp8(tmp, __NV_SATFINITE, __NV_E4M3);
  return *reinterpret_cast<const __nv_fp8_e4m3*>(&res);
}

// __nv_bfloat16 -> fp8
template <>
__inline__ __device__ __nv_fp8_e4m3
vec_conversion<__nv_fp8_e4m3, __nv_bfloat16>(const __nv_bfloat16 &a)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    assert(false);
#else
    __nv_fp8_storage_t res = __nv_cvt_bfloat16raw_to_fp8(
        __nv_bfloat16_raw(a), __NV_SATFINITE, __NV_E4M3);
    return *reinterpret_cast<const __nv_fp8_e4m3*>(&res);
#endif
#ifndef _MSC_VER
    __builtin_unreachable();  // Suppress missing return statement warning
#endif
}


// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, __nv_fp8_e4m3>(const __nv_fp8_e4m3& a,
                                                    const float& scale)
{
  // Note there is no direct convert function from fp8 to bf16.
  // fp8 -> half
  __half_raw res = __nv_cvt_fp8_to_halfraw(*reinterpret_cast<const uint8_t*>(&a), __NV_E4M3);
  // half -> float -> bf16
  float tmp = half_to_float(res.x);
  return __float2bfloat16(tmp * scale);
}

// fp8 -> half
template <>
__inline__ __device__ __half
scaled_vec_conversion<__half, __nv_fp8_e4m3>(const __nv_fp8_e4m3& a,
                                            const float& scale)
{
    __nv_fp8_storage_t fp8_val = *reinterpret_cast<const __nv_fp8_storage_t*>(&a);
    __half_raw tmp = __nv_cvt_fp8_to_halfraw(fp8_val, __NV_E4M3);
    return __float2half(half_to_float(tmp.x) * scale);
}

// half -> fp8
template <>
__inline__ __device__ __nv_fp8_e4m3
scaled_vec_conversion<__nv_fp8_e4m3, half>(const half& a,
                                          const float& scale)
{
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(__half2float(a) * scale, __NV_SATFINITE, __NV_E4M3);
  return *reinterpret_cast<const __nv_fp8_e4m3*>(&res);
}

// __nv_bfloat16 -> fp8
template <>
__inline__ __device__ __nv_fp8_e4m3
scaled_vec_conversion<__nv_fp8_e4m3, __nv_bfloat16>(const __nv_bfloat16& a,
                                                    const float& scale)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    assert(false);
#else
    __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) * scale,
                                                    __NV_SATFINITE, __NV_E4M3);
    return *reinterpret_cast<const __nv_fp8_e4m3*>(&res);
#endif
#ifndef _MSC_VER
  __builtin_unreachable();  // Suppress missing return statement warning
#endif
}

// #define FP8_MHA
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#define FUSE_GEMM_ACT
#endif
#define FP8_GEMM_OUTPUT_QUANT_DISABLE

#ifdef FUSE_GEMM_ACT
#define USE_QGMMA
#endif

namespace turbomind {

// NOTE(Alan): fp8 min max match with torch
static constexpr float FP8_E4M3_MAX = 448.0f;

enum QUANTIZE_MODE
{
    PER_CHANNEL,
    PER_TENSOR,
    PER_CHANNEL_WEIGHT_PER_TENSOR_ACT
};

// Packed Data Type
typedef struct __CUDA_ALIGN__(32) {
    float array[8];
} float8;

typedef struct __CUDA_ALIGN__(16) {
    half array[8];
} half8;

#ifdef ENABLE_BF16
typedef struct __CUDA_ALIGN__(4) {
    __nv_bfloat16 array[2];
} __nv_bfloat16_2;

typedef struct __CUDA_ALIGN__(8) {
    __nv_bfloat162 x, y;
} __nv_bfloat162_2_xy;

typedef struct __CUDA_ALIGN__(8) {
    __nv_bfloat16 array[4];
} __nv_bfloat164;

typedef struct __CUDA_ALIGN__(8) {
    __nv_bfloat162 array[2];
} __nv_bfloat162_2;

typedef struct __CUDA_ALIGN__(16) {
    __nv_bfloat16 array[8];
} __nv_bfloat168;

typedef struct __CUDA_ALIGN__(16) {
    __nv_bfloat162 array[4];
} __nv_bfloat162_4;

typedef struct __CUDA_ALIGN__(32) {
    __nv_bfloat16 array[16];
} __nv_bfloat1616;
#endif

#ifdef ENABLE_FP8
typedef struct __CUDA_ALIGN__(2) {
    __nv_fp8_e4m3 array[2];
} __nv_fp8_2_e4m3;

typedef struct __CUDA_ALIGN__(4) {
    __nv_fp8_e4m3 array[4];
} __nv_fp8_4_e4m3;

typedef struct __CUDA_ALIGN__(4) {
    __nv_fp8x2_e4m3 array[2];
} __nv_fp8x2_x2_e4m3;

typedef struct __CUDA_ALIGN__(8) {
    __nv_fp8_e4m3 array[8];
} __nv_fp8_8_e4m3;

typedef struct __CUDA_ALIGN__(8) {
    __nv_fp8x2_e4m3 array[4];
} __nv_fp8x2_x4_e4m3;

typedef struct __CUDA_ALIGN__(16) {
    __nv_fp8_e4m3 array[16];
} __nv_fp8x16_e4m3;
#endif

// only BF16 and FP8
template<typename T, int PACK_SIZE>
struct PackType {
    using type = float;
};

#ifdef ENABLE_BF16
template<>
struct PackType<__nv_bfloat16, 2> {
    using type = __nv_bfloat16_2;
};

template<>
struct PackType<__nv_bfloat16, 4> {
    using type = __nv_bfloat164;
};

template<>
struct PackType<__nv_bfloat16, 8> {
    using type = __nv_bfloat168;
};
#endif

#ifdef ENABLE_FP8
template<>
struct PackType<__nv_fp8_e4m3, 2> {
    using type = __nv_fp8_2_e4m3;
};

template<>
struct PackType<__nv_fp8_e4m3, 4> {
    using type = __nv_fp8_4_e4m3;
};

template<>
struct PackType<__nv_fp8_e4m3, 8> {
    using type = __nv_fp8_8_e4m3;
};
#endif

__inline__ __device__ void fp8x4_e4m3_to_bfloat2(__nv_bfloat162* out1, __nv_bfloat162* out2, const __nv_fp8x4_e4m3* in)
{
    const char4 tmp_val = reinterpret_cast<const char4*>(in)[0];
    *out1               = __nv_bfloat162((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.x)[0],
                           (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.y)[0]);
    *out2               = __nv_bfloat162((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.z)[0],
                           (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.w)[0]);
}

__inline__ __device__ __nv_bfloat162 fp8x2_e4m3_to_bfloat2(const __nv_fp8x2_e4m3* in)
{
    const char2    tmp_val = reinterpret_cast<const char2*>(in)[0];
    __nv_bfloat162 out     = __nv_bfloat162((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.x)[0],
                                        (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.y)[0]);
    return out;
}

__inline__ __device__ void fp8x4_e4m3_to_half2(half2* out1, half2* out2, const __nv_fp8x4_e4m3* in)
{
    const char4 tmp_val = reinterpret_cast<const char4*>(in)[0];
    *out1               = half2((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.x)[0],
                  (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.y)[0]);
    *out2               = half2((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.z)[0],
                  (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.w)[0]);
}

__inline__ __device__ half2 fp8x2_e4m3_to_half2(const __nv_fp8x2_e4m3* in)
{
    const char2 tmp_val = reinterpret_cast<const char2*>(in)[0];
    half2       out     = half2((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.x)[0],
                      (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.y)[0]);
    return out;
}

template <bool is_scale_inverted>
__device__ __forceinline__ __nv_fp8_e4m3 scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<__nv_fp8_e4m3>(r);
}

// void invokePrintFP8Data(__nv_fp8_e4m3* data, const int size, cudaStream_t stream);

void invokeHalf2Fp8(__nv_fp8_e4m3* output, const half* input, int num_element);

template<typename T_IN, QUANTIZE_MODE quantize_mode>
void invokeScaleFP8QuantMatrix(
    __nv_fp8_e4m3* output, float const* input_scale_inv, T_IN const* input, uint32_t size, uint32_t n, cudaStream_t stream);

template<typename T_OUT, typename T_IN, QUANTIZE_MODE quantize_mode>
void invokeQuantizeMatrix(
    T_OUT* output, float const* input_qua_amax_ptr, T_IN const* input, uint32_t size, uint32_t n, cudaStream_t stream);

template<typename T_OUT, typename T_IN, typename T_FAKE>
void invokeFakeQuantize(T_OUT* dst, const T_IN* src, const int size, cudaStream_t stream);

template<typename T_W>
void invokeComputeFP8QuantizeScale(float* quant_ptr, const T_W* weights, const int k, const int n, cudaStream_t stream);

}  // namespace turbomind
#endif  // ENABLE_FP8
