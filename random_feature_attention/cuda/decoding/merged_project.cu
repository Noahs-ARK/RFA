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


__forceinline__ __device__ __half relu(__half x) {
    __half zero = __float2half(0.f);
    return __hgt(x, zero) ? x : zero;
}


__device__
void random_project_step(
        int num_threads_per_out_dim,
        __half2 x_val[16], 
        __half2 w_val[16],
        __half b_val[1],
        __half *phi_x_local) {
    __half2 wx = __float2half2_rn(0.f);

    #pragma unroll 
    for (int i = 0;i < 16; ++ i) {
        wx = __hfma2(w_val[i], x_val[i], wx);
    }
    #pragma unroll 
    for (int offset = num_threads_per_out_dim;
         offset > 0; 
         offset >>= 1) {
        wx =  __hadd2(wx, __shfl_down_sync(FULL_MASK, wx, offset));
    }
    __half wx_half = __hadd(wx.x, wx.y);
    if (threadIdx.x == 0) {
        *phi_x_local = relu(__hadd(wx_half, b_val[0]));
    }
}



__device__
void read(
    const __half * __restrict__ x_local,
    const __half * __restrict__ w_local,
    const __half * __restrict__ b_local,
    __half2 x_val[16], 
    __half2 w_val[16],
    __half b_val[1]) {
    int4 in1 = ((int4*) x_local)[0];
    int4 in2 = ((int4*) x_local)[1];
    int4 in3 = ((int4*) x_local)[2];
    int4 in4 = ((int4*) x_local)[3];
    
    x_val[0] = *((__half2 *) &in1.x);
    x_val[1] = *((__half2 *) &in1.y);
    x_val[2] = *((__half2 *) &in1.z);
    x_val[3] = *((__half2 *) &in1.w);
    x_val[4] = *((__half2 *) &in2.x);
    x_val[5] = *((__half2 *) &in2.y);
    x_val[6] = *((__half2 *) &in2.z);
    x_val[7] = *((__half2 *) &in2.w);
    x_val[8] = *((__half2 *) &in3.x);
    x_val[9] = *((__half2 *) &in3.y);
    x_val[10] = *((__half2 *) &in3.z);
    x_val[11] = *((__half2 *) &in3.w);
    x_val[12] = *((__half2 *) &in4.x);
    x_val[13] = *((__half2 *) &in4.y);
    x_val[14] = *((__half2 *) &in4.z);
    x_val[15] = *((__half2 *) &in4.w);

    int4 in5 = ((int4*) w_local)[0];
    int4 in6 = ((int4*) w_local)[1];
    int4 in7 = ((int4*) w_local)[2];
    int4 in8 = ((int4*) w_local)[3];
    
    w_val[0] = *((__half2 *) &in5.x);
    w_val[1] = *((__half2 *) &in5.y);
    w_val[2] = *((__half2 *) &in5.z);
    w_val[3] = *((__half2 *) &in5.w);
    w_val[4] = *((__half2 *) &in6.x);
    w_val[5] = *((__half2 *) &in6.y);
    w_val[6] = *((__half2 *) &in6.z);
    w_val[7] = *((__half2 *) &in6.w);
    w_val[8] = *((__half2 *) &in7.x);
    w_val[9] = *((__half2 *) &in7.y);
    w_val[10] = *((__half2 *) &in7.z);
    w_val[11] = *((__half2 *) &in7.w);
    w_val[12] = *((__half2 *) &in8.x);
    w_val[13] = *((__half2 *) &in8.y);
    w_val[14] = *((__half2 *) &in8.z);
    w_val[15] = *((__half2 *) &in8.w);

    b_val[0] = *b_local;
    return;
}


__forceinline__ __device__
void write(
    __half * __restrict__ x1_local,
    __half * __restrict__ w1_local,
    __half * __restrict__ b1_local,
    __half2 x_val[16], 
    __half2 w_val[16],
    __half b_val[1]) {
    *((int4 *) x1_local) = *((int4 *) x_val);
    *((int4 *) x1_local + 1) = *((int4 *) x_val + 1);
    *((int4 *) x1_local + 2) = *((int4 *) x_val + 2);
    *((int4 *) x1_local + 3) = *((int4 *) x_val + 3);

    *((int4 *) w1_local) = *((int4 *) w_val);
    *((int4 *) w1_local + 1) = *((int4 *) w_val + 1);
    *((int4 *) w1_local + 2) = *((int4 *) w_val + 2);
    *((int4 *) w1_local + 3) = *((int4 *) w_val + 3);
    
    *b1_local = b_val[0];
}

__global__ 
void random_project(
        const real * __restrict__ x,
        const real * __restrict__ w,
        const real * __restrict__ b,
        // real * __restrict__ x1,
        // real * __restrict__ w1,
        // real * __restrict__ b1,
        real * __restrict__ phi_x,
        int in_dim, 
        int out_dim,
        int num_threads_per_out_dim,
        int num_out_dim_per_block,
        int num_blocks_per_batch) {
    /*
    Args:
        x: [1, bsz, in_dim]
        w: [out_dim, in_dim]
        b: [out_dim]
        
    Return:
        phi_x: [1, bsz, out_dim]
    */

    const int batch_id = blockIdx.x / num_blocks_per_batch;

    const int in_dim_offset = threadIdx.x << 5;
    const int out_dim_id \
        = (blockIdx.x % num_blocks_per_batch) * num_out_dim_per_block + threadIdx.y;
    
    
    const __half * __restrict__ x_local \
        = x + batch_id * in_dim + in_dim_offset;
    const __half * __restrict__ w_local \
        = w + out_dim_id * in_dim + in_dim_offset;
    const __half * __restrict__ b_local = b + out_dim_id;

    __half * __restrict__ phi_x_local = phi_x + batch_id * out_dim + out_dim_id;

    // __half * __restrict__ x1_local \
    //     = x1 + batch_id * in_dim + in_dim_offset;
    // __half * __restrict__ w1_local \
    //     = w1 + out_dim_id * in_dim + in_dim_offset;
    // __half * __restrict__ b1_local = b1 + out_dim_id;
    
    __half2 x_val[16] = {__float2half2_rn(0.f)};
    __half2 w_val[16] = {__float2half2_rn(0.f)};
    __half b_val[1] = {__float2half_rn(0.f)};
    read(x_local, w_local, b_local, x_val, w_val, b_val);
    // write(x1_local, w1_local, b1_local, x_val, w_val, b_val);
    num_threads_per_out_dim >>= 1;
    random_project_step(
        num_threads_per_out_dim, 
        x_val, w_val, b_val,
        phi_x_local
    );
    
    // if (threadIdx.x == 0) {
    //     // 0 thread write attn
    //     *((int4 *) attn_local) = ((int4 *) qs_val)[0];
    // }
}


// std::vector<Tensor> 
Tensor
RandomProject(
        Tensor const& x,
        Tensor const& w,
        Tensor const& b) {
    /*
    Args:
        x: [1, bsz, in_dim]
        w: [out_dim, in_dim]
        b: [out_dim]
        
    Return:
        phi_x: [1, bsz, out_dim]
    */
    // column major
    const int bsz = x.size(1);
    const int in_dim = x.size(2);
    const int out_dim = w.size(0);

    auto act_options  = x.options().requires_grad(false);
    Tensor phi_x = torch::zeros({1, bsz, out_dim}, act_options);

    // Tensor x1 = torch::zeros({1, bsz, in_dim}, act_options);
    // Tensor w1 = torch::zeros({out_dim, in_dim}, act_options);
    // Tensor b1 = torch::zeros({out_dim}, act_options);
    
    // num threads per head_dim;
    const int num_threads_per_out_dim = in_dim / 32;
    const int num_out_dim_per_block = min(
        out_dim, NUM_THREADS_PER_BLOCK / num_threads_per_out_dim); 
    const int num_blocks_per_batch = max(1, out_dim / num_out_dim_per_block);
    dim3 dim_grid(bsz * num_blocks_per_batch);
    // [x, y]
    dim3 dim_block(num_threads_per_out_dim, num_out_dim_per_block);
    random_project <<<dim_grid, dim_block>>>(
            static_cast<const real *> (x.data_ptr()),
            static_cast<const real *> (w.data_ptr()), 
            static_cast<const real *> (b.data_ptr()),
            // static_cast<real *> (x1.data_ptr()),
            // static_cast<real *> (w1.data_ptr()), 
            // static_cast<real *> (b1.data_ptr()),
            static_cast<real *> (phi_x.data_ptr()), 
            in_dim, 
            out_dim,
            num_threads_per_out_dim,
            num_out_dim_per_block,
            num_blocks_per_batch
    );
    // return {x1, w1, b1, phi_x};
    return phi_x;
}
