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


__forceinline__ __device__
void read_sz1(
    const __half * __restrict__ s_local,
    const __half * __restrict__ z_local,
    __half2 s_val[HALF2_PER_THREAD],
    __half2 z_val[HALF2_PER_THREAD]) {
    
    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) s_val + j) = *((int4*) s_local + j);
    }

    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) z_val + j) = *((int4*) z_local + j);
    }
}


__device__
void cross_rfa_step(
        const __half * __restrict__ q_local,
        __half * __restrict__ attn_local,
        int num_threads_per_head_dim,
        __half2 s_val[HALF2_PER_THREAD], 
        __half2 z_val[HALF2_PER_THREAD]) {
    
    __half2 q_val[HALF2_PER_THREAD] = { __float2half2_rn(0.f)};
    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) q_val + j) = *((int4*) q_local + j);
    } 

    __half2 qs = __float2half2_rn(0.f);
    __half2 qz = __float2half2_rn(0.f);
    #pragma unroll 
    for (int i = 0;i < HALF2_PER_THREAD; ++ i) {
        qs = __hfma2(s_val[i], q_val[i], qs);
        qz = __hfma2(z_val[i], q_val[i], qz);
    }
    #pragma unroll 
    for (int offset = num_threads_per_head_dim >> 1;
         offset > 0; 
         offset >>= 1) {
        qz =  __hadd2(qz, __shfl_down_sync(FULL_MASK, qz, offset));
        qs =  __hadd2(qs, __shfl_down_sync(FULL_MASK, qs, offset));
    }
    __half qs_half = __hadd(qs.x, qs.y);
    __half qz_half = __hadd(qz.x, qz.y);
    qz_half = clamp_eps(qz_half);
    
    if (threadIdx.x == 0) {
        *attn_local = __hdiv(qs_half, qz_half);
    }
}


__global__ 
void cross_rfa(
        const __half * __restrict__ q,
        const __half * __restrict__ s,
        const __half * __restrict__ z,
        __half * __restrict__ attn,
        int head_dim, 
        int proj_dim,
        int num_threads_per_head_dim,
        int num_head_dim_per_block,
        int num_blocks_per_batch) {
    /*
    Args:
        q: [tgt_len, bsz, proj_dim]
        s: [bsz, head_dim, proj_dim]
        z: [bsz, proj_dim]
        attn: [tgt_len, bsz, head_dim]
    */
    const int batch_id = blockIdx.x / num_blocks_per_batch;
    const int proj_dim_offset = threadIdx.x * DIM_PER_THREAD;
    const int head_dim_id \
        = (blockIdx.x % num_blocks_per_batch) * num_head_dim_per_block + threadIdx.y;

    const __half * __restrict__ q_local = q + batch_id * proj_dim + proj_dim_offset;

    const __half * __restrict__ s_local \
        = s + batch_id * head_dim * proj_dim \
            + head_dim_id * proj_dim + proj_dim_offset;
    const __half * __restrict__ z_local = z + batch_id * proj_dim + proj_dim_offset;
    __half * __restrict__ attn_local = attn + batch_id * head_dim + head_dim_id;
    
    __half2 s_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 z_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    
    read_sz1(s_local, z_local, s_val, z_val);
    cross_rfa_step(
        q_local, attn_local,
        num_threads_per_head_dim, 
        s_val, z_val
    );
}

Tensor CrossRFA(
        Tensor const& q,
        Tensor const& s,
        Tensor const& z) {
    /*
    Args:
        q: [tgt_len, bsz, proj_dim]
        s: [bsz, head_dim, proj_dim]
        z: [bsz, proj_dim]
        
    Return:
        attn: [tgt_len, bsz, head_dim]
    */
    // column major
    const int bsz = q.size(1);
    const int proj_dim = q.size(2);
    const int head_dim = s.size(1);

    auto act_options  = q.options().requires_grad(false);
    Tensor attn = torch::zeros({1, bsz, head_dim}, act_options);
    
    // num threads per head_dim;
    int num_threads_per_head_dim = proj_dim / DIM_PER_THREAD;
    const int num_head_dim_per_block = min(
        head_dim, NUM_THREADS_PER_BLOCK / num_threads_per_head_dim); 
    const int num_blocks_per_batch = max(1, head_dim / num_head_dim_per_block);

    dim3 dim_grid(bsz * num_blocks_per_batch);
    // [x, y]
    dim3 dim_block(num_threads_per_head_dim, num_head_dim_per_block);
    cross_rfa <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (q.data_ptr()),
            static_cast<const __half *> (s.data_ptr()),
            static_cast<const __half *> (z.data_ptr()), 
            static_cast<__half *> (attn.data_ptr()), 
            head_dim, 
            proj_dim,
            num_threads_per_head_dim,
            num_head_dim_per_block,
            num_blocks_per_batch
    );
 
    return attn;
}