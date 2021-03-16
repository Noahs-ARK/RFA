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
void random_project_step(
        __half2 q_val[HALF2_PER_THREAD], 
        __half2 k_val[HALF2_PER_THREAD],
        __half2 w_val[8][HALF2_PER_THREAD],
        __half2 b_val[4],
        __half2 phi_q_val[HALF2_PER_THREAD], 
        __half2 phi_k_val[HALF2_PER_THREAD],
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
        *((int4 *) phi_q_val) = ((int4 *) wq_half)[0];
        *((int4 *) phi_k_val) = ((int4 *) wk_half)[0];
    }

    #pragma unroll
    for (int i = 0; i < HALF2_PER_THREAD; ++ i) {
        int lane_id = 0;
        int offset = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++ j) {
            q_val[offset + j] = __shfl_sync(FULL_MASK, phi_q_val[j], lane_id);
            k_val[offset + j] = __shfl_sync(FULL_MASK, phi_k_val[j], lane_id);
        }
    }
}


__device__
void rfa_step(
        __half * __restrict__ attn_local,
        __half2 q_val[HALF2_PER_THREAD], 
        __half2 k_val[HALF2_PER_THREAD], 
        __half2 v_val[4], 
        __half2 s1[4][HALF2_PER_THREAD],
        __half2 s2[4][HALF2_PER_THREAD],
        __half2 z[HALF2_PER_THREAD],
        int num_threads_per_head_dim) {
    __half2 qs[4] = {__float2half2_rn(0.f)};
    __half2 qz = __float2half2_rn(0.f);
    
    __half2 v_val_swapped[4] = {
        __lowhigh2highlow(v_val[0]),
        __lowhigh2highlow(v_val[1]),
        __lowhigh2highlow(v_val[2]),
        __lowhigh2highlow(v_val[3]),
    };
    #pragma unroll 
    for (int i = 0;i < 4; ++ i) {
        qs[i] = __float2half2_rn(0.f);
        __half2 qs_swapped = __float2half2_rn(0.f);
        #pragma unroll 
        for (int j = 0;j < HALF2_PER_THREAD; ++ j) {
            s1[i][j] = __hfma2(v_val[i], k_val[j], s1[i][j]);
            s2[i][j] = __hfma2(v_val_swapped[i], k_val[j], s2[i][j]);
            
            qs[i] = __hfma2(s1[i][j], q_val[j], qs[i]);
            qs_swapped = __hfma2(s2[i][j], q_val[j], qs_swapped);
        }
        qs[i] = __hadd2(qs[i], __lowhigh2highlow(qs_swapped));
    }

    #pragma unroll 
    for (int j = 0; j < HALF2_PER_THREAD; ++ j) {
        z[j] = __hadd2(k_val[j], z[j]);
        qz = __hfma2(z[j], q_val[j], qz);
    }

    #pragma unroll 
    for (int offset = num_threads_per_head_dim >> 1; 
        offset > 0; 
        offset >>= 1) {
        qz =  __hadd2(qz, __shfl_down_sync(FULL_MASK, qz, offset));
        #pragma unroll 
        for (int i = 0; i < 4; ++ i) {
            qs[i] =  __hadd2(
                qs[i], __shfl_down_sync(FULL_MASK, qs[i], offset));
        }
    }
    __half qz_half = __hadd(qz.x, qz.y);
    qz_half = clamp_one(qz_half);
    qz = __half2half2(qz_half);
    if ((threadIdx.x % num_threads_per_head_dim) == 0) {
        #pragma unroll 
        for (int i = 0; i < 4; ++ i) {
            qs[i] =  __h2div(qs[i], qz);
        }
        *((int4 *) attn_local) = ((int4 *) qs)[0];
    }
}


__global__ 
void rfa_eval(
        const __half * __restrict__ q,
        const __half * __restrict__ k,
        const __half * __restrict__ v,
        const __half * __restrict__ w,
        const __half * __restrict__ b,
        __half * __restrict__ attn,
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
        q: [tgt_len, bsz, num_heads, head_dim]
        k: [tgt_len, bsz, num_heads, head_dim]
        v: [tgt_len, bsz, num_heads, head_dim]
        w: [num_heads, proj_dim, head_dim]
        b: [num_heads, proj_dim]
        
    Return:
        attn: [tgt_len, bsz, num_heads, head_dim]
    */
    const int batch_id = blockIdx.x / num_blocks_per_batch;
    const int head_id = (blockIdx.x % num_blocks_per_batch) / num_blocks_per_head;

    const int head_dim_offset = (threadIdx.x % num_threads_per_proj_dim) * DIM_PER_THREAD;
    const int proj_dim_id \
        = (blockIdx.x % num_blocks_per_head) * num_proj_dim_per_block + threadIdx.x / num_threads_per_proj_dim * 8;

    const int proj_dim_offset = (threadIdx.x % num_threads_per_head_dim) * DIM_PER_THREAD;
    const int head_dim_id \
        = (blockIdx.x % num_blocks_per_head) * num_proj_dim_per_block + threadIdx.x / num_threads_per_head_dim * 8;

    const int qkv_offset = batch_id * num_heads * head_dim + head_id * head_dim;

    const __half * __restrict__ q_local = q + qkv_offset + head_dim_offset; 
    const __half * __restrict__ k_local = k + qkv_offset + head_dim_offset; 
    const __half * __restrict__ v_local = v + qkv_offset + head_dim_id;

    const __half * __restrict__ w_local \
        = w + head_id * proj_dim * head_dim + proj_dim_id * head_dim + head_dim_offset;
    
    const __half * __restrict__ b_local \
        = b + head_id * proj_dim + proj_dim_id;

    __half * __restrict__ attn_local = attn + qkv_offset + head_dim_id;

    const int q1k1_offset = batch_id * num_heads * proj_dim + head_id * proj_dim;
    __half * __restrict__ q1_local \
        = q1 + q1k1_offset + proj_dim_id; 
    __half * __restrict__ k1_local \
        = k1 + q1k1_offset + proj_dim_id;

    __half2 q_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 k_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 v_val[4] = {__float2half2_rn(0.f)};
    __half2 w_val[8][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 b_val[4] = {__float2half2_rn(0.f)};


    __half2 s1[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 s2[4][HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 z[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};

    read_wb(w_local, b_local, w_val, b_val, head_dim);
    read_qkv(
        q_local, k_local, v_local,
        q_val, k_val, v_val);

    random_project_step(q_val, k_val, w_val, b_val,
        phi_q_val, phi_k_val,
        num_threads_per_proj_dim);
    read_sz(s1_local, s2_local, z_local,
        s1_val, s2_val, z_val);
    // read_qkv(
    //     q_local, k_local, v_local,
    //     q_val, k_val, v_val);
    rfa_step(
        attn_local,
        q_val, k_val, v_val,
        s1_val, s2_val,
        z_val,
        num_threads_per_head_dim
    );
    write_sz(s1_local, s2_local, z_local,
        s1_val, s2_val, z_val);
}

std::vector<Tensor> 
// Tensor
RFAEval(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v,
        Tensor const& w,
        Tensor const& b) {
    /*
    Args:
        q: [tgt_len, bsz, num_jeads, head_dim]
        k: [tgt_len, bsz, num_heads, head_dim]
        v: [tgt_len, bsz, num_heads, head_dim]
        w: [num_heads, proj_dim, head_dim]
        b: [num_heads, proj_dim]
        
    Return:
        attn: [tgt_len, bsz, num_heads, head_dim]
    */
    // column major
    const int bsz = q.size(1);
    const int num_heads = q.size(2);
    const int head_dim = v.size(3);
    const int proj_dim = w.size(1);

    auto act_options  = q.options().requires_grad(false);
    Tensor attn = torch::zeros({
        1, bsz, num_heads, head_dim}, act_options);
    
    // num threads per head_dim;
    const int num_threads_per_proj_dim = head_dim / DIM_PER_THREAD;
    const int num_threads_per_head_dim = proj_dim / DIM_PER_THREAD;
    const int num_proj_dim_per_block = min(
        proj_dim, NUM_THREADS_PER_BLOCK / num_threads_per_proj_dim);                      
    const int num_blocks_per_head = max(1, proj_dim / num_proj_dim_per_block);
    const int num_blocks_per_batch = num_heads * num_blocks_per_head;
    TORCH_CHECK(num_threads_per_head_dim == 2, "2 threads per dim");
    TORCH_CHECK(num_threads_per_proj_dim == 2, "2 threads per dim");
    dim3 dim_grid(bsz * num_blocks_per_batch);
    // [x, y]
    dim3 dim_block(num_threads_per_proj_dim 
        * num_proj_dim_per_block / 8);
    rfa_eval <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (q.data_ptr()), 
            static_cast<const __half *> (k.data_ptr()), 
            static_cast<const __half *> (v.data_ptr()), 
            static_cast<const __half *> (w.data_ptr()), 
            static_cast<const __half *> (b.data_ptr()),
            static_cast<__half *> (attn.data_ptr()),
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
 
    return {attn, q1, k1};
}
