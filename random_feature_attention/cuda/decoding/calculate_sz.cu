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
void write_sz4(
    const __half * __restrict__ s_local,
    const __half * __restrict__ z_local,
    __half2 s_val[HALF2_PER_THREAD],
    __half2 z_val[HALF2_PER_THREAD]) {
    
    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) s_local + j) = *((int4*) s_val + j);
    }

    if (threadIdx.y == 0) {
        #pragma unroll
        for (int j = 0; j < INT4_PER_THREAD; ++ j) {
            *((int4 *) z_local + j) = *((int4*) z_val + j);
        }
    }
}


__device__
void calculate_sz_step(
        const __half * __restrict__ k_local,
        const __half * __restrict__ v_local,
        int num_threads_per_head_dim,
        __half2 s_val[HALF2_PER_THREAD], 
        __half2 z_val[HALF2_PER_THREAD]) {
    __half2 k_val[HALF2_PER_THREAD] = { __float2half2_rn(0.f)};
    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) k_val + j) = *((int4*) k_local + j);
    }
    __half v_half = *v_local;
    __half2 v_val = __half2half2(v_half);    

    #pragma unroll 
    for (int i = 0;i < HALF2_PER_THREAD; ++ i) {
        s_val[i] = __hfma2(v_val, k_val[i], s_val[i]);
    }
    if (threadIdx.y == 0) {
        #pragma unroll 
        for (int i = 0; i < HALF2_PER_THREAD; ++ i) {
            z_val[i] = __hadd2(k_val[i], z_val[i]);
        }
    }
}


__global__ 
void calculate_sz(
        const __half * __restrict__ k,
        const __half * __restrict__ v,
        __half * __restrict__ s,
        __half * __restrict__ z,
        int src_len,
        int head_dim, 
        int proj_dim,
        int num_threads_per_head_dim,
        int num_head_dim_per_block,
        int num_blocks_per_batch,
        int k_inc_t,
        int v_inc_t) {
    /*
    Args:
        k: [src_len, bsz, proj_dim]
        v: [src_len, bsz, head_dim]
        s: [bsz, head_dim, proj_dim]
        z: [bsz, proj_dim]
    */
    const int batch_id = blockIdx.x / num_blocks_per_batch;
    const int proj_dim_offset = threadIdx.x * DIM_PER_THREAD;
    const int head_dim_id \
        = (blockIdx.x % num_blocks_per_batch) * num_head_dim_per_block + threadIdx.y;

    const __half * __restrict__ k_local = k + batch_id * proj_dim + proj_dim_offset;
    const __half * __restrict__ v_local = v + batch_id * head_dim + head_dim_id;
    
    __half * __restrict__ s_local \
            = s + batch_id * head_dim * proj_dim \
                + head_dim_id * proj_dim + proj_dim_offset;
    __half * __restrict__ z_local = z + batch_id * proj_dim + proj_dim_offset;

    __half2 s_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 z_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    
    num_threads_per_head_dim >>= 1;
    for (int t = 0; t < src_len; ++ t) {
        calculate_sz_step(
            k_local, v_local,
            num_threads_per_head_dim, 
            s_val, z_val
        );
        k_local += k_inc_t;
        v_local += v_inc_t;
    }
    write_sz4(s_local, z_local, s_val, z_val);
}


std::vector<Tensor> CalculateSZ(
        Tensor const& k,
        Tensor const& v) {
    /*
    Args:
        k: [src_len, bsz, proj_dim]
        v: [src_len, bsz, head_dim]
        
    Return:
        s: [bsz, head_dim, proj_dim]
        z: [bsz, proj_dim]
    */
    // column major
    const int src_len = k.size(0);
    const int bsz = k.size(1);
    const int proj_dim = k.size(2);
    const int head_dim = v.size(2);
    
    const int k_inc_t = bsz * proj_dim;
    const int v_inc_t = bsz * head_dim;

    auto act_options  = k.options().requires_grad(false);
    Tensor s = torch::zeros({bsz, head_dim, proj_dim}, act_options);
    Tensor z = torch::zeros({bsz, proj_dim}, act_options);
    
    // num threads per head_dim;
    int num_threads_per_head_dim = proj_dim / DIM_PER_THREAD;
    const int num_head_dim_per_block = min(
        head_dim, NUM_THREADS_PER_BLOCK / num_threads_per_head_dim); 
    const int num_blocks_per_batch = max(1, head_dim / num_head_dim_per_block);

    dim3 dim_grid(bsz * num_blocks_per_batch);
    // [x, y]
    dim3 dim_block(num_threads_per_head_dim, num_head_dim_per_block);
    calculate_sz <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (k.data_ptr()), 
            static_cast<const __half *> (v.data_ptr()), 
            static_cast<__half *> (s.data_ptr()), 
            static_cast<__half *> (z.data_ptr()), 
            src_len,
            head_dim, 
            proj_dim,
            num_threads_per_head_dim,
            num_head_dim_per_block,
            num_blocks_per_batch,
            k_inc_t,
            v_inc_t
    );
 
    return {s, z};
}
