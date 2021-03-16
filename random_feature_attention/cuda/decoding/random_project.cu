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
        int num_threads_per_out_dim,
        __half2 x_val[HALF2_PER_THREAD], 
        __half2 w_val[HALF2_PER_THREAD],
        __half b_val[1],
        __half *phi_x_local) {
    __half2 wx = __float2half2_rn(0.f);

    #pragma unroll 
    for (int i = 0;i < HALF2_PER_THREAD; ++ i) {
        wx = __hfma2(w_val[i], x_val[i], wx);
    }
    #pragma unroll 
    for (int offset = num_threads_per_out_dim >> 1;
         offset > 0; 
         offset >>= 1) {
        wx =  __hadd2(wx, __shfl_down_sync(FULL_MASK, wx, offset));
    }
    __half wx_half = __hadd(wx.x, wx.y);
    if (threadIdx.x == 0) {
        // *phi_x_local = relu(__hadd(wx_half, b_val[0]));
        *phi_x_local = __hadd(wx_half, b_val[0]);
    }
}


__device__
void read(
    const __half * __restrict__ x_local,
    const __half * __restrict__ w_local,
    const __half * __restrict__ b_local,
    __half2 x_val[HALF2_PER_THREAD], 
    __half2 w_val[HALF2_PER_THREAD],
    __half b_val[1]) {

    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) x_val + j) = *((int4*) x_local + j);
    }
    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) w_val + j) = *((int4*) w_local + j);
    }

    b_val[0] = *b_local;
}


__device__
void write(
    __half * __restrict__ x1_local,
    __half * __restrict__ w1_local,
    __half * __restrict__ b1_local,
    __half2 x_val[HALF2_PER_THREAD], 
    __half2 w_val[HALF2_PER_THREAD],
    __half b_val[1]) {
    
    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) x1_local + j) = *((int4*) x_val + j);
    }

    #pragma unroll
    for (int j = 0; j < INT4_PER_THREAD; ++ j) {
        *((int4 *) w1_local + j) = *((int4*) w_val + j);
    }
    
    *b1_local = b_val[0];
}

__global__ 
void random_project(
        const __half * __restrict__ x,
        const __half * __restrict__ w,
        const __half * __restrict__ b,
        __half * __restrict__ phi_x,
        int num_heads,
        int in_dim, 
        int out_dim,
        int num_threads_per_out_dim,
        int num_out_dim_per_block,
        int num_blocks_per_head,
        int num_blocks_per_batch) {
    /*
    Args:
        x: [1, bsz, num_heads, in_dim]
        w: [num_heads, out_dim, in_dim]
        b: [num_heads, out_dim]
        
    Return:
        phi_x: [1, bsz, num_heads, out_dim]
    */

    const int batch_id = blockIdx.x / num_blocks_per_batch;
    const int head_id = (blockIdx.x % num_blocks_per_batch) / num_blocks_per_head;

    const int in_dim_offset = threadIdx.x * DIM_PER_THREAD;
    const int out_dim_id \
        = (blockIdx.x % num_blocks_per_head) * num_out_dim_per_block + threadIdx.y;
    
    const __half * __restrict__ x_local \
        = x + batch_id * num_heads * in_dim \
        + head_id * in_dim + in_dim_offset;
    const __half * __restrict__ w_local \
        = w + head_id * out_dim * in_dim \
        + out_dim_id * in_dim + in_dim_offset;
    const __half * __restrict__ b_local = b + head_id * out_dim + out_dim_id;

    __half * __restrict__ phi_x_local \
        = phi_x + batch_id * num_heads * out_dim \
        + head_id * out_dim + out_dim_id;
    
    __half2 x_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half2 w_val[HALF2_PER_THREAD] = {__float2half2_rn(0.f)};
    __half b_val[1] = {__float2half_rn(0.f)};
    read(x_local, w_local, b_local, x_val, w_val, b_val);
    random_project_step(
        num_threads_per_out_dim, 
        x_val, w_val, b_val,
        phi_x_local
    );
}


Tensor RandomProject(
        Tensor const& x,
        Tensor const& w,
        Tensor const& b) {
    /*
    Args:
        x: [1, bsz, num_heads, in_dim]
        w: [num_heads, out_dim, in_dim]
        b: [num_heads, out_dim]
        
    Return:
        phi_x: [1, bsz, num_heads, out_dim]
    */
    // column major
    const int bsz = x.size(1);
    const int num_heads = x.size(2);
    const int in_dim = x.size(3);
    const int out_dim = w.size(1);

    auto act_options  = x.options().requires_grad(false);
    Tensor phi_x = torch::zeros({1, bsz, num_heads, out_dim}, act_options);

    // num threads per head_dim;
    const int num_threads_per_out_dim = in_dim / DIM_PER_THREAD;
    const int num_out_dim_per_block = min(
        out_dim, NUM_THREADS_PER_BLOCK / num_threads_per_out_dim); 
    const int num_blocks_per_head = max(1, out_dim / num_out_dim_per_block);
    const int num_blocks_per_batch = num_heads * num_blocks_per_head;
    dim3 dim_grid(bsz * num_blocks_per_batch);
    // [x, y]
    dim3 dim_block(num_threads_per_out_dim, num_out_dim_per_block);
    random_project <<<dim_grid, dim_block>>>(
            static_cast<const __half *> (x.data_ptr()),
            static_cast<const __half *> (w.data_ptr()), 
            static_cast<const __half *> (b.data_ptr()),
            static_cast<__half *> (phi_x.data_ptr()), 
            num_heads,
            in_dim, 
            out_dim,
            num_threads_per_out_dim,
            num_out_dim_per_block,
            num_blocks_per_head,
            num_blocks_per_batch
    );
    return phi_x;
}
