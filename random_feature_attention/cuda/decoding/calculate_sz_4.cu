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
void calculate_sz_step4(
        __half2 k_val[HALF2_PER_THREAD],
        __half2 v1_val[4],
        __half2 s1_val[4][HALF2_PER_THREAD],
        __half2 s2_val[4][HALF2_PER_THREAD],
        __half2 z_val[HALF2_PER_THREAD],
        int num_threads_per_head_dim) {
    __half2 v2_val[4] = {
        __lowhigh2highlow(v1_val[0]),
        __lowhigh2highlow(v1_val[1]),
        __lowhigh2highlow(v1_val[2]),
        __lowhigh2highlow(v1_val[3]),
    };

    #pragma unroll 
    for (int i = 0;i < 4; ++ i) {
        #pragma unroll 
        for (int j = 0;j < HALF2_PER_THREAD; ++ j) {
            s1_val[i][j] = __hfma2(v1_val[i], k_val[j], s1_val[i][j]);
            s2_val[i][j] = __hfma2(v2_val[i], k_val[j], s2_val[i][j]);
        }
    }
    if (threadIdx.y == 0) {
        #pragma unroll 
        for (int j = 0; j < HALF2_PER_THREAD; ++ j) {
            z_val[j] = __hadd2(k_val[j], z_val[j]);
        }
    }
}


__global__ 
void calculate_sz4(
        const __half * __restrict__ k,
        const __half * __restrict__ v,
        __half * __restrict__ s1,
        __half * __restrict__ s2,
        __half * __restrict__ z,
        int src_len,
        int head_dim, 
        int proj_dim,
        int num_threads_per_head_dim,
        int k_inc_t,
        int v_inc_t) {
    /*
    Args:
        k: [src_len, bsz, proj_dim]
        v: [src_len, bsz, head_dim]
        s1: [bsz, proj_dim * head_dim / 2]
        s2: [bsz, proj_dim * head_dim / 2]
        z: [bsz, proj_dim]
    */
    int bid = blockIdx.x;
    int head_dim_offset = threadIdx.y << 3;
    int proj_dim_offset = threadIdx.x * DIM_PER_THREAD;
    int tid = threadIdx.y * num_threads_per_head_dim + threadIdx.x;
    int s_offset = tid * 4 * DIM_PER_THREAD;

    const __half * __restrict__ k_local \
        = k + bid * proj_dim  + proj_dim_offset;
    const __half * __restrict__ v_local \
        = v + bid * head_dim + head_dim_offset;
    
    __half * __restrict__ s1_local \
        = s1 + bid * proj_dim * head_dim / 2 + s_offset;
    __half * __restrict__ s2_local \
        = s2 + bid * proj_dim * head_dim / 2 + s_offset;
    __half * __restrict__ z_local \
        = z + bid * proj_dim + proj_dim_offset;

    __half2 k_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 v_val[4] = {__float2half2_rn(0.f)};

    __half2 s1_val[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 s2_val[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 z_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    
    for (int t = 0; t < src_len; ++ t) {
        read_kv(k_local, v_local, k_val, v_val);
        calculate_sz_step4(
            k_val, v_val,
            s1_val, s2_val, z_val,
            num_threads_per_head_dim
        );
        k_local += k_inc_t;
        v_local += v_inc_t;
    }
    write_sz(s1_local, s2_local, z_local,
        s1_val, s2_val, z_val);
}


std::vector<Tensor> CalculateSZ4(
        Tensor const& k,
        Tensor const& v) {
    /*
    Args:
        k: [src_len, bsz, proj_dim]
        v: [src_len, bsz, head_dim]
        
    Return:
        s1: [bsz, proj_dim * head_dim / 2]
        s2: [bsz, proj_dim * head_dim / 2]
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
    Tensor s1 = torch::zeros({bsz, proj_dim * head_dim / 2}, act_options);
    Tensor s2 = torch::zeros({bsz, proj_dim * head_dim / 2}, act_options);
    Tensor z = torch::zeros({bsz, proj_dim}, act_options);
    
    int num_threads_per_head_dim = proj_dim / DIM_PER_THREAD;
    dim3 dim_grid(bsz);
    dim3 dim_block(num_threads_per_head_dim, head_dim / 8);
    calculate_sz4 <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (k.data_ptr()), 
            static_cast<const __half *> (v.data_ptr()), 
            static_cast<__half *> (s1.data_ptr()), 
            static_cast<__half *> (s2.data_ptr()), 
            static_cast<__half *> (z.data_ptr()), 
            src_len,
            head_dim, 
            proj_dim,
            num_threads_per_head_dim,
            k_inc_t,
            v_inc_t
    );
 
    return {s1, s2, z};
}
