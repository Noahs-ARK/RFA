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
#include "utils.cu"

__device__
void cross_rfa_step4(
        __half * __restrict__ attn_local,
        __half2 q_val[HALF2_PER_THREAD],
        __half2 s1_val[4][HALF2_PER_THREAD], 
        __half2 s2_val[4][HALF2_PER_THREAD],
        __half2 z_val[HALF2_PER_THREAD],
        int num_threads_per_head_dim) {
    __half2 qs[4] = {__float2half2_rn(0.f)};
    __half2 qz = __float2half2_rn(0.f);

    #pragma unroll 
    for (int i = 0;i < 4; ++ i) {
        qs[i] = __float2half2_rn(0.f);
        __half2 qs_swapped = __float2half2_rn(0.f);
        #pragma unroll 
        for (int j = 0;j < HALF2_PER_THREAD; ++ j) {
            qs[i] = __hfma2(s1_val[i][j], q_val[j], qs[i]);
            qs_swapped = __hfma2(s2_val[i][j], q_val[j], qs_swapped);
        }
        qs[i] = __hadd2(qs[i], __lowhigh2highlow(qs_swapped));
    }
    #pragma unroll 
    for (int j = 0; j < HALF2_PER_THREAD; ++ j) {
        qz = __hfma2(z_val[j], q_val[j], qz);
    }
    #pragma unroll 
    for (int offset = num_threads_per_head_dim >> 1;
         offset > 0; 
         offset >>= 1) {
        qz =  __hadd2(qz, __shfl_down_sync(FULL_MASK, qz, offset));
        #pragma unroll 
        for (int i = 0; i < 4; ++ i) {
            qs[i] =  __hadd2(qs[i], __shfl_down_sync(FULL_MASK, qs[i], offset));
        }
    }
    __half qz_half = __hadd(qz.x, qz.y);
    qz_half = clamp_eps(qz_half);
    qz = __half2half2(qz_half);
    
    if (threadIdx.x == 0) {
        #pragma unroll 
        for (int i = 0; i < 4; ++ i) {
            qs[i] =  __h2div(qs[i], qz);
        }
        *((int4 *) attn_local) = *((int4 *) qs);
    }
}


__global__ 
void cross_rfa4(
        const __half * __restrict__ q,
        const __half * __restrict__ s1,
        const __half * __restrict__ s2,
        const __half * __restrict__ z,
        __half * __restrict__ attn,
        int head_dim, 
        int proj_dim,
        int num_threads_per_head_dim) {
    /*
    Args:
        q: [tgt_len, bsz, proj_dim]
        s1: [bsz, proj_dim * head_dim / 2]
        s2: [bsz, proj_dim * head_dim / 2]
        z: [bsz, proj_dim]
        attn: [tgt_len, bsz, head_dim]
    */
    int bid = blockIdx.x;
    int head_dim_offset = threadIdx.y << 3;
    int proj_dim_offset = threadIdx.x * DIM_PER_THREAD;
    int tid = threadIdx.y * num_threads_per_head_dim + threadIdx.x;
    int s_offset = tid * 4 * DIM_PER_THREAD;

    const __half * __restrict__ q_local \
        = q + bid * proj_dim + proj_dim_offset;

    const __half * __restrict__ s1_local \
        = s1 + bid * proj_dim * head_dim / 2 + s_offset;
    const __half * __restrict__ s2_local \
        = s2 + bid * proj_dim * head_dim / 2 + s_offset;
    const __half * __restrict__ z_local \
        = z + bid * proj_dim + proj_dim_offset;
    
    __half * __restrict__ attn_local \
        = attn + bid * head_dim + head_dim_offset;
    
    __half2 q_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 s1_val[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 s2_val[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 z_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    
    read_sz(s1_local, s2_local, z_local,
        s1_val, s2_val, z_val);
    read_q(q_local, q_val);
    cross_rfa_step4(
        attn_local,
        q_val,
        s1_val, s2_val,
        z_val,
        num_threads_per_head_dim);
}

Tensor CrossRFA4(
        Tensor const& q,
        Tensor const& s1,
        Tensor const& s2,
        Tensor const& z) {
    /*
    Args:
        q: [tgt_len, bsz, proj_dim]
        s1: [bsz, proj_dim * head_dim / 2]
        s2: [bsz, proj_dim * head_dim / 2]
        z: [bsz, proj_dim]
        
    Return:
        attn: [tgt_len, bsz, head_dim]
    */
    // column major
    const int bsz = q.size(1);
    const int proj_dim = q.size(2);
    const int head_dim = s1.size(1) * 2 / proj_dim;

    auto act_options  = q.options().requires_grad(false);
    Tensor attn = torch::zeros({1, bsz, head_dim}, act_options);
    
    // num threads per head_dim;
    int num_threads_per_head_dim = proj_dim / DIM_PER_THREAD;
    dim3 dim_grid(bsz);
    dim3 dim_block(num_threads_per_head_dim, head_dim / 8);
    cross_rfa4 <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (q.data_ptr()),
            static_cast<const __half *> (s1.data_ptr()), 
            static_cast<const __half *> (s2.data_ptr()),
            static_cast<const __half *> (z.data_ptr()), 
            static_cast<__half *> (attn.data_ptr()), 
            head_dim, 
            proj_dim,
            num_threads_per_head_dim
    );
    return attn;
}
