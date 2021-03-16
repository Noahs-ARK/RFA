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
void q_backward_step(
        const __half* __restrict__ q_local,
        const __half* __restrict__ k_local,
        const __half* __restrict__ v_local,
        __half                      &qz_half,
        const __half* __restrict__ ga_local,
        __half2 gq1_val[4],
        __half  &gqz_half,
        __half2 s1_val[4][4],
        __half2 s2_val[4][4],
        __half2 z_val[4],
        __half  shared_mem_half[NUM_THREADS_PER_BLOCK],
        int4    shared_mem_int4[NUM_THREADS_PER_BLOCK],
        int num_threads_per_proj_dim,
        int num_threads_per_head_dim) {
   __half2 q_val[4], k_val[4], v1_val[4], qs1_val[4], gqs1_val[4];

   read_int4(q_local, q_val, 1);
   read_int4(k_local, k_val, 1);
   read_int4(v_local, v1_val, 1);
   read_int4(ga_local, gqs1_val, 1);

   __half2 v2_val[4] = {
        __lowhigh2highlow(v1_val[0]),
        __lowhigh2highlow(v1_val[1]),
        __lowhigh2highlow(v1_val[2]),
        __lowhigh2highlow(v1_val[3]),
    };

    __half2 qz_val = __float2half2_rn(0.f);
    __half2 gqz_val = __float2half2_rn(0.f);

    /* qs qz starts */
    #pragma unroll
    for (int j = 0;j < 4; ++ j) {
        qs1_val[j] = __float2half2_rn(0.f);
        __half2 qs2_val = __float2half2_rn(0.f);
        #pragma unroll
        for (int i = 0;i < 4; ++ i) {
            // s
            s1_val[i][j] = __hfma2(v1_val[j], k_val[i], s1_val[i][j]);
            s2_val[i][j] = __hfma2(v2_val[j], k_val[i], s2_val[i][j]);

            // qs
            qs1_val[j] = __hfma2(s1_val[i][j], q_val[i], qs1_val[j]);
            qs2_val = __hfma2(s2_val[i][j], q_val[i], qs2_val);

        }
        qs1_val[j] = __hadd2(qs1_val[j], __lowhigh2highlow(qs2_val));
        z_val[j] = __hadd2(z_val[j], k_val[j]);
        qz_val = __hfma2(z_val[j], q_val[j], qz_val);
    }


    // 128 x 256 case:
    // sum through 256 / 8 = 32 proj_dim threads:
    // thread_idx:
    // 0 + 1 + 2 + ... + 31
    // 32 + 33 + 34 + ... + 63
    // ...
    #pragma unroll
    for (int offset = num_threads_per_head_dim >> 1; 
         offset > 0; 
         offset >>= 1) {
        qz_val =  __hadd2(qz_val, __shfl_down_sync(FULL_MASK, qz_val, offset));
        #pragma unroll
        for (int j = 0; j < 4; ++ j) {
            qs1_val[j] =  __hadd2(
                qs1_val[j], 
                __shfl_down_sync(FULL_MASK, qs1_val[j], offset));
        }
    }
    qz_half = __hadd(qz_val.x, qz_val.y);
    qz_half = clamp_eps(qz_half);

    int remain = threadIdx.x % num_threads_per_head_dim;
    if (remain == 0) {
        shared_mem_half[threadIdx.x] = qz_half;
        shared_mem_int4[threadIdx.x] = ((int4*) qs1_val)[0];
    }
    __syncthreads();
    if (remain > 0) {
        qz_half = shared_mem_half[threadIdx.x - remain];
        ((int4*) qs1_val)[0] = shared_mem_int4[threadIdx.x - remain];
    }
    __syncthreads();
    qz_val = __half2half2(qz_half);

    /* qs qz done */

    /* gqz and gqs */
    #pragma unroll
    for (int j = 0;j < 4; ++ j) {
        // here it is still g_attn
        gqz_val = __hfma2(gqs1_val[j], qs1_val[j], gqz_val);

        // from now on it is gqs
        gqs1_val[j] = __h2div(gqs1_val[j], qz_val);
    }

    __half2 gqs2_val[4] = {
        __lowhigh2highlow(gqs1_val[0]),
        __lowhigh2highlow(gqs1_val[1]),
        __lowhigh2highlow(gqs1_val[2]),
        __lowhigh2highlow(gqs1_val[3]),
    };

    // 128 x 256 case:
    // sum through 128 / 8 = 16 head_dim threads:
    // thread_idx:
    // 0 + 32 + 64 + ... + 480
    // 1 + 33 + 65 + ... + 481
    // ...
    // 31 + 63 + ... + 511

    gqz_half = __hadd(gqz_val.x, gqz_val.y);
    shared_mem_half[threadIdx.x] = gqz_half;
    __syncthreads();
    if (threadIdx.x < num_threads_per_head_dim) {
        for (int i = num_threads_per_head_dim;
             i < num_threads_per_head_dim * num_threads_per_proj_dim; 
             i += num_threads_per_head_dim) {
            gqz_half = __hadd(gqz_half, shared_mem_half[threadIdx.x + i]);
        }
    }
    __syncthreads();
    gqz_half = __hdiv(__hneg(gqz_half),
                      __hmul(qz_half, qz_half));
    gqz_val = __half2half2(gqz_half);
    /* gqz and gqs done done */
    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        gq1_val[i] = __float2half2_rn(0.f);
        __half2 gq2_val = __float2half2_rn(0.f);
        #pragma unroll
        for (int j = 0;j < 4; ++ j) {
            gq1_val[i] = __hfma2(s1_val[i][j], gqs1_val[j], gq1_val[i]);
            gq2_val = __hfma2(s2_val[i][j], gqs2_val[j], gq2_val);
        }
        gq1_val[i] = __hadd2(gq1_val[i], gq2_val);
    }

    // 128 x 256 case:
    // sum through 128 / 8 = 16 head_dim threads:
    // thread_idx:
    // 0 + 32 + 64 + ... + 480
    // 1 + 33 + 65 + ... + 481
    // ...
    // 31 + 63 + ... + 511
    shared_mem_int4[threadIdx.x] = ((int4*) gq1_val)[0];
    __syncthreads();
    if (threadIdx.x < num_threads_per_head_dim) {
        #pragma unroll
        for (int j = num_threads_per_head_dim;
             j < num_threads_per_proj_dim * num_threads_per_head_dim; 
             j += num_threads_per_head_dim) {
            __half2 tmp[4];
            read_int4(shared_mem_int4[threadIdx.x + j], tmp);
            #pragma unroll
            for (int i = 0;i < 4; ++ i) {
                gq1_val[i] = __hadd2(gq1_val[i], tmp[i]);
            }
        }
        #pragma unroll
        for (int i = 0;i < 4; ++ i) {
            gq1_val[i] = __hfma2(z_val[i], gqz_val,  gq1_val[i]);
        }
    }
}

__device__
void kv_backward_step(
        const __half* __restrict__ q_local,
        const __half* __restrict__ k_local,
        const __half* __restrict__ v_local,
        const __half   qz_half,
        const __half* __restrict__ ga_local,
        __half2 gk1_val[4],
        __half2 gv1_val[4],
        __half  gqz_half,
        __half2 s1_val[4][4],
        __half2 s2_val[4][4],
        __half2 t_val[4],
        int4    shared_mem_int4[NUM_THREADS_PER_BLOCK],
        int num_threads_per_proj_dim,
        int num_threads_per_head_dim) {
    __half2 q1_val[4], k1_val[4], v_val[4], gqs_val[4];
    read_int4(q_local, q1_val, 1);
    read_int4(k_local, k1_val, 1);
    read_int4(v_local, v_val, 1);
    read_int4(ga_local, gqs_val, 1);

    __half2 qz_val = __half2half2(qz_half);
    __half2 gqz_val = __half2half2(gqz_half);
    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
        gqs_val[i] = __h2div(gqs_val[i], qz_val);
    }
    __half2 q2_val[4] = {
        __lowhigh2highlow(q1_val[0]),
        __lowhigh2highlow(q1_val[1]),
        __lowhigh2highlow(q1_val[2]),
        __lowhigh2highlow(q1_val[3]),
    };

    __half2 k2_val[4] = {
        __lowhigh2highlow(k1_val[0]),
        __lowhigh2highlow(k1_val[1]),
        __lowhigh2highlow(k1_val[2]),
        __lowhigh2highlow(k1_val[3]),
    };

    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        gk1_val[i] = __float2half2_rn(0.f);
        __half2 gk2_val = __float2half2_rn(0.f);
        t_val[i] = __hfma2(gqz_val, q1_val[i], t_val[i]);
        #pragma unroll
        for (int j = 0;j < 4; ++ j) {
            s1_val[i][j] = __hfma2(q1_val[i], gqs_val[j], s1_val[i][j]);
            s2_val[i][j] = __hfma2(q2_val[i], gqs_val[j], s2_val[i][j]);

            gk1_val[i] = __hfma2(s1_val[i][j], v_val[j], gk1_val[i]);
            gk2_val = __hfma2(s2_val[i][j], v_val[j], gk2_val);
        }
        gk1_val[i] = __hadd2(gk1_val[i], __lowhigh2highlow(gk2_val));
    }
    #pragma unroll
    for (int j = 0;j < 4; ++ j) {
        gv1_val[j] = __float2half2_rn(0.f);
        __half2 gv2_val = __float2half2_rn(0.f);
        #pragma unroll
        for (int i = 0;i < 4; ++ i) {
            gv1_val[j] = __hfma2(s1_val[i][j], k1_val[i], gv1_val[j]);
            gv2_val = __hfma2(s2_val[i][j], k2_val[i], gv2_val);
        }
        gv1_val[j] = __hadd2(gv1_val[j], gv2_val);
    }

    // 128 x 256 case:
    // sum through 128 / 8 = 16 head_dim threads:
    // thread_idx:
    // 0 + 32 + 64 + ... + 480
    // 1 + 33 + 65 + ... + 481
    // ...
    // 31 + 63 + ... + 511

    shared_mem_int4[threadIdx.x] = ((int4*) gk1_val)[0];
    __syncthreads();
    if (threadIdx.x < num_threads_per_head_dim) {
        #pragma unroll
        for (int j = num_threads_per_head_dim;
             j < num_threads_per_head_dim * num_threads_per_proj_dim; 
             j += num_threads_per_head_dim) {
            __half2 tmp[4];
            read_int4(shared_mem_int4[threadIdx.x + j], tmp);
            #pragma unroll
            for (int i = 0;i < 4; ++ i) {
                gk1_val[i] = __hadd2(gk1_val[i], tmp[i]);
            }
        }
    }
    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        gk1_val[i] = __hadd2(gk1_val[i], t_val[i]);
    }

    // 128 x 256 case:
    // sum through 256 / 8 = 32 proj_dim threads:
    // thread_idx:
    // 0 + 1 + 2 + ... + 31
    // 32 + 33 + 34 + ... + 63
    // ...
    #pragma unroll
    for (int offset = num_threads_per_head_dim >> 1; 
         offset > 0; 
         offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < 4; ++ j) {
            gv1_val[j] =  __hadd2(
                gv1_val[j], 
                __shfl_down_sync(FULL_MASK, gv1_val[j], offset));
        }
    }
}

__device__
void q_backward(
        const __half * __restrict__ q,
        const __half * __restrict__ k,
        const __half * __restrict__ v,
        __half       * __restrict__ qz,
        const __half * __restrict__ grad_attn,
        __half       * __restrict__ grad_qz,
        __half       * __restrict__ grad_q,
        __half        shared_mem_half[NUM_THREADS_PER_BLOCK],
        int4          shared_mem_int4[NUM_THREADS_PER_BLOCK],
        int tgt_len,
        int num_threads_per_proj_dim,
        int num_threads_per_head_dim,
        int qk_inc_t,
        int v_inc_t) {
    /*
        q:         [tgt_len, proj_dim]
        v:         [tgt_len, head_dim]
        qz:        [tgt_len].         shared memory
        grad_attn: [tgt_len, head_dim]

    return:
        grad_q:    [tgt_len, proj_dim]
        grad_qz:   [tgt_len]. shared memory
    */
    int head_dim_offset = (threadIdx.x / num_threads_per_head_dim) << 3;
    int proj_dim_offset = (threadIdx.x % num_threads_per_head_dim) << 3;
    const __half* __restrict__ q_local = q + proj_dim_offset;
    const __half* __restrict__ k_local = k + proj_dim_offset;
    const __half* __restrict__ v_local = v + head_dim_offset;
    const __half* __restrict__ ga_local = grad_attn + head_dim_offset;

    __half* __restrict__ gq_local = grad_q + proj_dim_offset;

    __half2 gq_val[4];
    __half qz_half, gqz_half;

    __half2 s1_val[4][4] = {__float2half2_rn(0.f)};
    __half2 s2_val[4][4] = {__float2half2_rn(0.f)};
    __half2 z_val[4] = {__float2half2_rn(0.f)};
    for (int t = 0; t < tgt_len; ++ t) {
        q_backward_step(
            q_local, k_local, v_local,
            qz_half, ga_local,
            gq_val, gqz_half,
            s1_val, s2_val,
            z_val,
            shared_mem_half,
            shared_mem_int4,
            num_threads_per_proj_dim,
            num_threads_per_head_dim
        );
        if (head_dim_offset == 0) {
            ((int4 *) gq_local)[0] = ((int4 *) gq_val)[0];
        }

        if (threadIdx.x == 0) {
            qz[t] = qz_half;
            grad_qz[t] = gqz_half;
        }
        __syncthreads();
        q_local += qk_inc_t;
        k_local += qk_inc_t;
        v_local += v_inc_t;
        ga_local += v_inc_t;
        gq_local += qk_inc_t;
    }
}

__device__
void kv_backward(
        const __half * __restrict__ q,
        const __half * __restrict__ k,
        const __half * __restrict__ v,
        const __half * __restrict__ qz,
        const __half * __restrict__ grad_attn,
        const __half  * __restrict__ grad_qz,
        __half       * __restrict__ grad_k,
        __half       * __restrict__ grad_v,
        int4          shared_mem_int4[NUM_THREADS_PER_BLOCK],
        int tgt_len,
        int num_threads_per_proj_dim,
        int num_threads_per_head_dim,
        int qk_inc_t,
        int v_inc_t) {
    /*
        q:         [tgt_len, proj_dim]
        v:         [tgt_len, head_dim]
        qz:        [tgt_len].         shared memory
        grad_attn: [tgt_len, head_dim]
        grad_qz:   [tgt_len].         shared memory

    return:
        grad_k:    [tgt_len, proj_dim]
    */
    int head_dim_offset = (threadIdx.x / num_threads_per_head_dim) << 3;
    int proj_dim_offset = (threadIdx.x % num_threads_per_head_dim) << 3;

    const __half* __restrict__ q_local = q + proj_dim_offset;
    const __half* __restrict__ k_local = k + proj_dim_offset;
    const __half* __restrict__ v_local = v + head_dim_offset;
    const __half* __restrict__ ga_local = grad_attn + head_dim_offset;

    __half* __restrict__ gk_local = grad_k + proj_dim_offset;
    __half* __restrict__ gv_local = grad_v + head_dim_offset;

    __half2 s1_val[4][4] = {__float2half2_rn(0.f)};
    __half2 s2_val[4][4] = {__float2half2_rn(0.f)};
    __half2 t_val[4] = {__float2half2_rn(0.f)};
    __half2 gk_val[4], gv_val[4];
    __half qz_half, gqz_half;

    int offset = tgt_len - 1;
    q_local += qk_inc_t * offset;
    k_local += qk_inc_t * offset;
    v_local += v_inc_t * offset;
    ga_local += v_inc_t * offset;

    gk_local += qk_inc_t * offset;
    gv_local += v_inc_t * offset;
    for (int t = 0; t < tgt_len; ++ t) {
        qz_half = qz[tgt_len - t - 1];
        gqz_half = grad_qz[tgt_len - t - 1];
        kv_backward_step(
            q_local, k_local, v_local, qz_half, ga_local,
            gk_val, gv_val, gqz_half,
            s1_val, s2_val,
            t_val,
            shared_mem_int4,
            num_threads_per_proj_dim,
            num_threads_per_head_dim
        );

        if (proj_dim_offset == 0) {
            ((int4 *) gv_local)[0] = ((int4 *) gv_val)[0];
        }
        if (head_dim_offset == 0) {
            ((int4 *) gk_local)[0] = ((int4 *) gk_val)[0];
        }
        __syncthreads();
        q_local -= qk_inc_t;
        k_local -= qk_inc_t;
        v_local -= v_inc_t;
        ga_local -= v_inc_t;

        gk_local -= qk_inc_t;
        gv_local -= v_inc_t;
    }
}

__global__
void rfa_backward(
        const __half * __restrict__ q,
        const __half * __restrict__ k,
        const __half * __restrict__ v,
        const __half * __restrict__ grad_attn,
        __half * __restrict__ grad_q,
        __half * __restrict__ grad_k,
        __half * __restrict__ grad_v,
        const int tgt_len,
        const int head_dim,
        const int proj_dim,
        const int num_threads_per_proj_dim,
        const int num_threads_per_head_dim,
        const int qk_inc_t,
        const int v_inc_t) {
    /*
        q:         [tgt_len, bsz, proj_dim]
        k:         [tgt_len, bsz, proj_dim]
        v:         [tgt_len, bsz, head_dim]
        grad_attn: [tgt_len, bsz, head_dim]

    return:
        grad_q:    [tgt_len, bsz, proj_dim]
        grad_k:    [tgt_len, bsz, proj_dim]
        grad_v:    [tgt_len, bsz, head_dim]
    */
    int bid = blockIdx.x;
    const __half * __restrict__ q_local =  q + bid * proj_dim;
    const __half * __restrict__ k_local =  k + bid * proj_dim;
    const __half * __restrict__ v_local =  v + bid * head_dim;
    const __half * __restrict__ ga_local = grad_attn + bid * head_dim;

    __half * __restrict__ grad_q_local = grad_q + bid * proj_dim;
    __half * __restrict__ grad_k_local = grad_k + bid * proj_dim;
    __half * __restrict__ grad_v_local = grad_v + bid * head_dim;

    extern __shared__ __half qz_shared[];
    __shared__ __half shared_mem_half[NUM_THREADS_PER_BLOCK];
    __shared__ int4 shared_mem_int4[NUM_THREADS_PER_BLOCK];
    __half * __restrict__ gqz_shared = qz_shared + tgt_len;

    q_backward(
        q_local,
        k_local,
        v_local,
        qz_shared,
        ga_local,
        gqz_shared,
        grad_q_local,
        shared_mem_half,
        shared_mem_int4,
        tgt_len,
        num_threads_per_proj_dim,
        num_threads_per_head_dim,
        qk_inc_t,
        v_inc_t);
    __syncthreads();

    kv_backward(
        q_local,
        k_local,
        v_local,
        qz_shared,
        ga_local,
        gqz_shared,
        grad_k_local,
        grad_v_local,
        shared_mem_int4,
        tgt_len,
        num_threads_per_proj_dim,
        num_threads_per_head_dim,
        qk_inc_t,
        v_inc_t);
}


std::vector<Tensor> RFABackward(
        Tensor const& q,
        Tensor const& k,
        Tensor const& v,
        Tensor const& grad_attn) {
    /*
    Args:
        q: [tgt_len, bsz, proj_dim]
        k: [tgt_len, bsz, proj_dim]
        v: [tgt_len, bsz, head_dim]
        grad_attn: [tgt_len, bsz, head_dim]

    Return:
        grad_q: [tgt_len, bsz, proj_dim]
        grad_k: [tgt_len, bsz, proj_dim]
        grad_v: [tgt_len, bsz, head_dim]
    */
    // column major
    const int tgt_len = q.size(0);
    const int bsz = q.size(1);
    const int proj_dim = q.size(2);
    const int head_dim = v.size(2);
    const int qk_inc_t = bsz * proj_dim;
    const int v_inc_t = bsz * head_dim;

    auto act_options  = q.options().requires_grad(false);
    Tensor grad_q = torch::zeros({tgt_len, bsz, proj_dim}, act_options);
    Tensor grad_k = torch::zeros({tgt_len, bsz, proj_dim}, act_options);
    Tensor grad_v = torch::zeros({tgt_len, bsz, head_dim}, act_options);

    const int block_size = proj_dim / 8 * head_dim / 8;
    const int num_threads_per_proj_dim = head_dim / 8;
    const int num_threads_per_head_dim = proj_dim / 8;
    // grad_q: 4 threads per proj_dim
    // 2 blocks per batch
    dim3 dim_grid(bsz);
    dim3 dim_block(block_size);
    rfa_backward <<<dim_grid, dim_block, 2 * sizeof(__half) * tgt_len>>>(
            static_cast<const __half *> (q.data_ptr()),
            static_cast<const __half *> (k.data_ptr()),
            static_cast<const __half *> (v.data_ptr()),
            static_cast<const __half *> (grad_attn.data_ptr()),
            static_cast<__half *> (grad_q.data_ptr()),
            static_cast<__half *> (grad_k.data_ptr()),
            static_cast<__half *> (grad_v.data_ptr()),
            tgt_len,
            head_dim,
            proj_dim,
            num_threads_per_proj_dim,
            num_threads_per_head_dim,
            qk_inc_t,
            v_inc_t
    );

    return {grad_q, grad_k, grad_v};
}

