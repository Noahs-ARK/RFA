#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>
#include <vector>
#include <stdio.h>

typedef torch::Tensor Tensor;
#define FULL_MASK 0xffffffff
const int NUM_THREADS_PER_BLOCK = 1024;
const int DIM_PER_THREAD = 16;
const int HALF2_PER_THREAD = DIM_PER_THREAD / 2;
const int INT4_PER_THREAD = DIM_PER_THREAD / 8;
const float EPS = 1.f;


__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}


__forceinline__ __device__ __half clamp_eps(__half x) {
    __half one = __float2half(EPS);
    return __hgt(x, one) ? x : one;
}


__forceinline__ __device__ __half select_eps(__half x, __half a) {
    const __half eps = __float2half(EPS);
    return __hgt(a, eps) ? x : __float2half(0.f);
}


__forceinline__ __device__
void read_int4(
    const __half * __restrict__ x_local,
    __half2 *x,
    int count) {
    #pragma unroll
    for (int j = 0; j < count; ++ j) {
        *((int4 *) x + j) = *((int4*) x_local + j);
    }
}


__forceinline__ __device__ void
read_int4(int4 in0, __half2 val[4]) {
    val[0] = *((__half2 *) &in0.x);
    val[1] = *((__half2 *) &in0.y);
    val[2] = *((__half2 *) &in0.z);
    val[3] = *((__half2 *) &in0.w);
}


__forceinline__ __device__
void read_sz(
    const __half * __restrict__ s1_local,
    const __half * __restrict__ s2_local,
    const __half * __restrict__ z_local,
    __half2 s1[4][HALF2_PER_THREAD], 
    __half2 s2[4][HALF2_PER_THREAD],
    __half2 z[HALF2_PER_THREAD]) {
    
    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        const __half * __restrict__ s1_ptr \
                = s1_local + DIM_PER_THREAD * i;
        #pragma unroll
        for (int j = 0; j < INT4_PER_THREAD; ++ j) {
            *((int4 *) *(s1 + i) + j) = *((int4*) s1_ptr + j);
        }
    }

    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        const __half * __restrict__ s2_ptr \
                = s2_local + DIM_PER_THREAD * i;
        #pragma unroll
        for (int j = 0; j < INT4_PER_THREAD; ++ j) {
            *((int4 *) *(s2 + i) + j) = *((int4*) s2_ptr + j);
        }
    }

    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) z + j) = *((int4*) z_local + j);
    }
}


__forceinline__ __device__
void write_sz(
    __half * __restrict__ s1_local,
    __half * __restrict__ s2_local,
    __half * __restrict__ z_local,
    __half2 s1[4][HALF2_PER_THREAD], 
    __half2 s2[4][HALF2_PER_THREAD],
    __half2 z[HALF2_PER_THREAD]) {
    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        __half * __restrict__ s1_ptr \
                = s1_local + DIM_PER_THREAD * i;
        #pragma unroll
        for (int j = 0; j < INT4_PER_THREAD; ++ j) {
            *((int4 *) s1_ptr + j) = *((int4 *) *(s1 + i) + j);
        }
    }

    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        __half * __restrict__ s2_ptr \
                = s2_local + DIM_PER_THREAD * i;
        #pragma unroll
        for (int j = 0; j < INT4_PER_THREAD; ++ j) {
            *((int4 *) s2_ptr + j) = *((int4 *) *(s2 + i) + j);
        }
    }

    if (threadIdx.y == 0) {
        #pragma unroll
        for (int j = 0; j < INT4_PER_THREAD; ++ j) {
            *((int4 *) z_local + j) = *((int4*) z + j);
        }
    }
}
