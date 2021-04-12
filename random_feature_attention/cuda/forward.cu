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
void rfa_forward_step(
        const __half * __restrict__ q_local,
        const __half * __restrict__ k_local,
        const __half * __restrict__ v_local,
        __half * __restrict__ attn_local,
        __half2 s1[4][HALF2_PER_THREAD], 
        __half2 s2[4][HALF2_PER_THREAD],
        __half2 z[HALF2_PER_THREAD],
        int num_threads_per_head_dim) {
    __half2 qs1[4] = {__float2half2_rn(0.f)};
    __half2 qz = __float2half2_rn(0.f);
    __half2 q[HALF2_PER_THREAD] = { __float2half2_rn(0.f)};
    __half2 k[HALF2_PER_THREAD] = { __float2half2_rn(0.f)};
    __half2 v1[4] = { __float2half2_rn(0.f)};
    read_int4(q_local, q, INT4_PER_THREAD);
    read_int4(k_local, k, INT4_PER_THREAD);
    read_int4(v_local, v1, 1);
    __half2 v2[4] = {
        __lowhigh2highlow(v1[0]),
        __lowhigh2highlow(v1[1]),
        __lowhigh2highlow(v1[2]),
        __lowhigh2highlow(v1[3]),
    };

    #pragma unroll 
    for (int i = 0;i < 4; ++ i) {
        qs1[i] = __float2half2_rn(0.f);
        __half2 qs2 = __float2half2_rn(0.f);
        #pragma unroll 
        for (int j = 0;j < HALF2_PER_THREAD; ++ j) {
            s1[i][j] = __hfma2(v1[i], k[j], s1[i][j]);
            s2[i][j] = __hfma2(v2[i], k[j], s2[i][j]);
            qs1[i] = __hfma2(s1[i][j], q[j], qs1[i]);
            qs2 = __hfma2(s2[i][j], q[j], qs2);
            
        }
        qs1[i] = __hadd2(qs1[i], __lowhigh2highlow(qs2));
    }
    #pragma unroll 
    for (int j = 0; j < HALF2_PER_THREAD; ++ j) {
        z[j] = __hadd2(k[j], z[j]);
        qz = __hfma2(z[j], q[j], qz);
    }
    #pragma unroll 
    for (int offset = num_threads_per_head_dim >> 1;
         offset > 0; 
         offset >>= 1) {
        qz =  __hadd2(qz, __shfl_down_sync(FULL_MASK, qz, offset));
        #pragma unroll 
        for (int i = 0; i < 4; ++ i) {
            qs1[i] =  __hadd2(
                qs1[i], 
                __shfl_down_sync(FULL_MASK, qs1[i], offset));
        }
    }
    __half qz_half = __hadd(qz.x, qz.y);
    qz_half = clamp_eps(qz_half);
    qz = __half2half2(qz_half);
    
    if (threadIdx.x == 0) {
        #pragma unroll 
        for (int i = 0; i < 4; ++ i) {
            qs1[i] =  __h2div(qs1[i], qz);
        }
        *((int4 *) attn_local) = *((int4 *) qs1);
    }
}


__global__ 
void rfa_forward(
        const __half * __restrict__ q,
        const __half * __restrict__ k,
        const __half * __restrict__ v,
        __half * __restrict__ attn,
        int tgt_len,
        int head_dim, 
        int proj_dim,
        int num_threads_per_head_dim,
        int qk_inc_t,
        int v_inc_t) {
    const int bid = blockIdx.x;
    const int head_dim_offset = threadIdx.y << 3;
    const int proj_dim_offset = threadIdx.x * DIM_PER_THREAD;

    const __half * __restrict__ q_local = q + bid * proj_dim + proj_dim_offset;
    const __half * __restrict__ k_local = k + bid * proj_dim  + proj_dim_offset;
    const __half * __restrict__ v_local = v + bid * head_dim + head_dim_offset;

    __half * __restrict__ attn_local = attn + bid * head_dim + head_dim_offset;

    __half2 s1[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 s2[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 z[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    
    for (int t = 0; t < tgt_len; ++ t) {
        rfa_forward_step(
            q_local, k_local, v_local,
            attn_local,
            s1, s2,
            z,
            num_threads_per_head_dim
        );
        
        q_local += qk_inc_t;
        k_local += qk_inc_t;
        v_local += v_inc_t;
        attn_local += v_inc_t;
    }
}

Tensor RFAForward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v) {
    /*
    Args:
        q: [tgt_len, bsz, proj_dim]
        k: [tgt_len, bsz, proj_dim]
        v: [tgt_len, bsz, head_dim]
        
    Return:
        attn: [tgt_len, bsz, head_dim]
    */
    // column major
    const int tgt_len = q.size(0);
    const int bsz = q.size(1);
    const int proj_dim = q.size(2);
    const int head_dim = v.size(2);
    const int qk_inc_t = bsz * proj_dim;
    const int v_inc_t = bsz * head_dim;

    auto act_options  = q.options().requires_grad(false);
    Tensor attn = torch::zeros({tgt_len, bsz, head_dim}, act_options);
    
    int num_threads_per_head_dim = proj_dim / DIM_PER_THREAD;
    dim3 dim_grid(bsz);
    dim3 dim_block(num_threads_per_head_dim, head_dim >> 3);
    rfa_forward <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (q.data_ptr()), 
            static_cast<const __half *> (k.data_ptr()), 
            static_cast<const __half *> (v.data_ptr()), 
            static_cast<__half *> (attn.data_ptr()), 
            tgt_len,
            head_dim, 
            proj_dim,
            num_threads_per_head_dim,
            qk_inc_t,
            v_inc_t
    );
 
    return attn;
}
