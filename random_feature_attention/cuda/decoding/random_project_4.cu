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
void random_project_step4(
        __half2 q_val[HALF2_PER_THREAD], 
        __half2 k_val[HALF2_PER_THREAD],
        __half2 w_val[8][HALF2_PER_THREAD],
        __half2 b_val[4],
        __half * __restrict__ phi_q,
        __half * __restrict__ phi_k,
        int num_threads_per_proj_dim) {

    __half2 wq[8] = {__float2half2_rn(0.f)};
    __half2 wk[8] = {__float2half2_rn(0.f)};

    #pragma unroll 
    for (int i = 0;i < 8; ++ i) {
        #pragma unroll 
        for (int j = 0;j < HALF2_PER_THREAD; ++ j) {
            wq[i] = __hfma2(w_val[i][j], q_val[j], wq[i]);
            wk[i] = __hfma2(w_val[i][j], k_val[j], wk[i]);
        }
    }

    #pragma unroll 
    for (int offset = num_threads_per_proj_dim >> 1;
         offset > 0; 
         offset >>= 1) {
        #pragma unroll 
        for (int i = 0; i < 8; ++ i) { 
            wq[i] =  __hadd2(
                wq[i], __shfl_down_sync(FULL_MASK, wq[i], offset));
            wk[i] =  __hadd2(
                wk[i], __shfl_down_sync(FULL_MASK, wk[i], offset));
        }
    }
    if ((threadIdx.x % num_threads_per_proj_dim) == 0) {
        __half wq_half[8] = {
            __hadd(__hadd(wq[0].x, wq[0].y), b_val[0].x),
            __hadd(__hadd(wq[1].x, wq[1].y), b_val[0].y),
            __hadd(__hadd(wq[2].x, wq[2].y), b_val[1].x),
            __hadd(__hadd(wq[3].x, wq[3].y), b_val[1].y),
            __hadd(__hadd(wq[4].x, wq[4].y), b_val[2].x),
            __hadd(__hadd(wq[5].x, wq[5].y), b_val[2].y),
            __hadd(__hadd(wq[6].x, wq[6].y), b_val[3].x),
            __hadd(__hadd(wq[7].x, wq[7].y), b_val[3].y) 
        };
        __half wk_half[8] = {
            __hadd(__hadd(wk[0].x, wk[0].y), b_val[0].x),
            __hadd(__hadd(wk[1].x, wk[1].y), b_val[0].y),
            __hadd(__hadd(wk[2].x, wk[2].y), b_val[1].x),
            __hadd(__hadd(wk[3].x, wk[3].y), b_val[1].y),
            __hadd(__hadd(wk[4].x, wk[4].y), b_val[2].x),
            __hadd(__hadd(wk[5].x, wk[5].y), b_val[2].y),
            __hadd(__hadd(wk[6].x, wk[6].y), b_val[3].x),
            __hadd(__hadd(wk[7].x, wk[7].y), b_val[3].y) 
        };
        *((int4 *) phi_q) = ((int4 *) wq_half)[0];
        *((int4 *) phi_k) = ((int4 *) wk_half)[0];
    }
}


__global__ 
void random_project4(
        const __half * __restrict__ q,
        const __half * __restrict__ k,
        const __half * __restrict__ w,
        const __half * __restrict__ b,
        __half * __restrict__ phi_q,
        __half * __restrict__ phi_k,
        int bsz, 
        int num_heads,
        int head_dim, 
        int proj_dim,
        int num_threads_per_proj_dim,
        int num_threads_per_head_dim,
        int num_proj_dim_per_block,
        int num_blocks_per_head,
        int num_blocks_per_batch) {
    /*
    Args:
        q: [1, bsz, num_heads, head_dim]
        k: [1, bsz, num_heads, head_dim]
        w: [num_heads, proj_dim, head_dim]
        b: [num_heads, proj_dim]
        
    Return:
        phi_q: [1, bsz, num_jeads, proj_dim]
        phi_k: [1, bsz, num_heads, proj_dim]
    */
    const int batch_id \
        = blockIdx.x / num_blocks_per_batch;
    const int head_id \
        = (blockIdx.x % num_blocks_per_batch) / num_blocks_per_head;

    const int head_dim_offset \
        = (threadIdx.x % num_threads_per_proj_dim) * DIM_PER_THREAD;
    const int proj_dim_id \
        = (blockIdx.x % num_blocks_per_head) * num_proj_dim_per_block + threadIdx.x / num_threads_per_proj_dim * 8;

    const int proj_dim_offset \
        = (threadIdx.x % num_threads_per_head_dim) * DIM_PER_THREAD;
    const int head_dim_id \
        = (blockIdx.x % num_blocks_per_head) * num_proj_dim_per_block + threadIdx.x / num_threads_per_head_dim * 8;

    const int qk_offset \
        = batch_id * num_heads * head_dim + head_id * head_dim;

    const __half * __restrict__ q_local \
        = q + qk_offset + head_dim_offset; 
    const __half * __restrict__ k_local \
        = k + qk_offset + head_dim_offset;
 
    const __half * __restrict__ w_local \
        = w + head_id * proj_dim * head_dim + proj_dim_id * head_dim + head_dim_offset;
    const __half * __restrict__ b_local \
        = b + head_id * proj_dim + proj_dim_id;

    const int phi_offset \
        = batch_id * num_heads * proj_dim + head_id * proj_dim;
    __half * __restrict__ phi_q_local \
        = phi_q + phi_offset + proj_dim_id; 
    __half * __restrict__ phi_k_local \
        = phi_k + phi_offset + proj_dim_id;
    
    __half2 q_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 k_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 w_val[8][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 b_val[4] = {__float2half2_rn(0.f)};

    read_wb(w_local, b_local, w_val, b_val, head_dim);
    // write_wb(w1_local, b1_local, w_val, b_val, head_dim);
    read_qk(q_local, k_local, q_val, k_val);
    random_project_step4(
        q_val, k_val, 
        w_val, b_val,
        phi_q_local, phi_k_local,
        num_threads_per_proj_dim
    );
}

std::vector<Tensor> 
RandomProject4(
        Tensor const& q,
        Tensor const& k,
        Tensor const& w,
        Tensor const& b) {
    /*
    Args:
        q: [tgt_len, bsz, num_heads, head_dim]
        k: [tgt_len, bsz, num_heads, head_dim]
        w: [num_heads, proj_dim, head_dim]
        b: [num_heads, proj_dim]
        
    Return:
        phi_q: [tgt_len, bsz, num_jeads, proj_dim]
        phi_k: [tgt_len, bsz, num_heads, proj_dim]
    */
    // column major
    const int tgt_len = q.size(0);
    const int bsz = q.size(1);
    const int num_heads = q.size(2);
    const int head_dim = q.size(3);
    const int proj_dim = w.size(1);

    auto act_options  = q.options().requires_grad(false);
    Tensor phi_q = torch::zeros({tgt_len, bsz, num_heads, proj_dim}, act_options);
    Tensor phi_k = torch::zeros({tgt_len, bsz, num_heads, proj_dim}, act_options);
    // num threads per head_dim;
    const int num_threads_per_proj_dim = head_dim / DIM_PER_THREAD;
    const int num_threads_per_head_dim = proj_dim / DIM_PER_THREAD;
    const int num_proj_dim_per_block = min(
        proj_dim, NUM_THREADS_PER_BLOCK / num_threads_per_proj_dim); 
    const int num_blocks_per_head = max(1, proj_dim / num_proj_dim_per_block);
    const int num_blocks_per_batch = num_heads * num_blocks_per_head;
    
    dim3 dim_grid(bsz * num_blocks_per_batch);
    // printf("%d %d\n", num_threads_per_proj_dim, num_proj_dim_per_block);
    // [x, y]
    dim3 dim_block(num_threads_per_proj_dim 
        * num_proj_dim_per_block / 8);
    random_project4 <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (q.data_ptr()), 
            static_cast<const __half *> (k.data_ptr()), 
            static_cast<const __half *> (w.data_ptr()), 
            static_cast<const __half *> (b.data_ptr()),
            static_cast<__half *> (phi_q.data_ptr()), 
            static_cast<__half *> (phi_k.data_ptr()),
            bsz,
            num_heads,
            head_dim, 
            proj_dim,
            num_threads_per_proj_dim,
            num_threads_per_head_dim,
            num_proj_dim_per_block,
            num_blocks_per_head,
            num_blocks_per_batch
    );
 
    return {phi_q, phi_k};
}
