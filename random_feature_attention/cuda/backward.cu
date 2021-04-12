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
        const __half* __restrict__ g_a_local,
        __half2 g_q1[4],
        __half  &g_qz_half,
        __half2 s1[4][4],
        __half2 s2[4][4],
        __half2 z[4],
        __half  shared_mem_half[NUM_THREADS_PER_BLOCK],
        int4    shared_mem_int4[NUM_THREADS_PER_BLOCK],
        int num_threads_per_proj_dim,
        int num_threads_per_head_dim) {
   __half2 q[4], k[4], v1[4], qs1[4], g_qs1[4];

   read_int4(q_local, q, 1);
   read_int4(k_local, k, 1);
   read_int4(v_local, v1, 1);
   read_int4(g_a_local, g_qs1, 1);

   __half2 v2[4] = {
        __lowhigh2highlow(v1[0]),
        __lowhigh2highlow(v1[1]),
        __lowhigh2highlow(v1[2]),
        __lowhigh2highlow(v1[3]),
    };

    __half2 qz = __float2half2_rn(0.f);
    __half2 g_qz = __float2half2_rn(0.f);

    /* qs qz starts */
    #pragma unroll
    for (int j = 0;j < 4; ++ j) {
        qs1[j] = __float2half2_rn(0.f);
        __half2 qs2 = __float2half2_rn(0.f);
        #pragma unroll
        for (int i = 0;i < 4; ++ i) {
            // s
            s1[i][j] = __hfma2(v1[j], k[i], s1[i][j]);
            s2[i][j] = __hfma2(v2[j], k[i], s2[i][j]);

            // qs
            qs1[j] = __hfma2(s1[i][j], q[i], qs1[j]);
            qs2 = __hfma2(s2[i][j], q[i], qs2);

        }
        qs1[j] = __hadd2(qs1[j], __lowhigh2highlow(qs2));
        z[j] = __hadd2(z[j], k[j]);
        qz = __hfma2(z[j], q[j], qz);
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
        qz =  __hadd2(qz, __shfl_down_sync(FULL_MASK, qz, offset));
        #pragma unroll
        for (int j = 0; j < 4; ++ j) {
            qs1[j] =  __hadd2(
                qs1[j], 
                __shfl_down_sync(FULL_MASK, qs1[j], offset));
        }
    }
    qz_half = __hadd(qz.x, qz.y);
    qz_half = clamp_eps(qz_half);

    int remain = threadIdx.x % num_threads_per_head_dim;
    if (remain == 0) {
        shared_mem_half[threadIdx.x] = qz_half;
        shared_mem_int4[threadIdx.x] = ((int4*) qs1)[0];
    }
    __syncthreads();
    if (remain > 0) {
        qz_half = shared_mem_half[threadIdx.x - remain];
        ((int4*) qs1)[0] = shared_mem_int4[threadIdx.x - remain];
    }
    __syncthreads();
    qz = __half2half2(qz_half);

    /* qs qz done */

    /* g_qz and g_qs */
    #pragma unroll
    for (int j = 0;j < 4; ++ j) {
        // here it is still g_attn
        g_qz = __hfma2(g_qs1[j], qs1[j], g_qz);

        // from now on it is g_qs
        g_qs1[j] = __h2div(g_qs1[j], qz);
    }

    __half2 g_qs2[4] = {
        __lowhigh2highlow(g_qs1[0]),
        __lowhigh2highlow(g_qs1[1]),
        __lowhigh2highlow(g_qs1[2]),
        __lowhigh2highlow(g_qs1[3]),
    };

    // 128 x 256 case:
    // sum through 128 / 8 = 16 head_dim threads:
    // thread_idx:
    // 0 + 32 + 64 + ... + 480
    // 1 + 33 + 65 + ... + 481
    // ...
    // 31 + 63 + ... + 511

    g_qz_half = __hadd(g_qz.x, g_qz.y);
    shared_mem_half[threadIdx.x] = g_qz_half;
    __syncthreads();
    if (threadIdx.x < num_threads_per_head_dim) {
        for (int i = num_threads_per_head_dim;
             i < num_threads_per_head_dim * num_threads_per_proj_dim; 
             i += num_threads_per_head_dim) {
            g_qz_half = __hadd(g_qz_half, shared_mem_half[threadIdx.x + i]);
        }
    }
    __syncthreads();
    g_qz_half = __hdiv(__hneg(g_qz_half),
                       __hmul(qz_half, qz_half));
    g_qz_half = select_eps(g_qz_half, qz_half);
    g_qz = __half2half2(g_qz_half);
    /* g_qz and g_qs done done */
    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        g_q1[i] = __float2half2_rn(0.f);
        __half2 g_q2 = __float2half2_rn(0.f);
        #pragma unroll
        for (int j = 0;j < 4; ++ j) {
            g_q1[i] = __hfma2(s1[i][j], g_qs1[j], g_q1[i]);
            g_q2 = __hfma2(s2[i][j], g_qs2[j], g_q2);
        }
        g_q1[i] = __hadd2(g_q1[i], g_q2);
    }

    // 128 x 256 case:
    // sum through 128 / 8 = 16 head_dim threads:
    // thread_idx:
    // 0 + 32 + 64 + ... + 480
    // 1 + 33 + 65 + ... + 481
    // ...
    // 31 + 63 + ... + 511
    shared_mem_int4[threadIdx.x] = ((int4*) g_q1)[0];
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
                g_q1[i] = __hadd2(g_q1[i], tmp[i]);
            }
        }
        #pragma unroll
        for (int i = 0;i < 4; ++ i) {
            g_q1[i] = __hfma2(z[i], g_qz,  g_q1[i]);
        }
    }
}

__device__
void kv_backward_step(
        const __half* __restrict__ q_local,
        const __half* __restrict__ k_local,
        const __half* __restrict__ v_local,
        const __half   qz_half,
        const __half* __restrict__ g_a_local,
        __half2 g_k1[4],
        __half2 g_v1[4],
        __half  g_qz_half,
        __half2 s1[4][4],
        __half2 s2[4][4],
        __half2 t[4],
        int4    shared_mem_int4[NUM_THREADS_PER_BLOCK],
        int num_threads_per_proj_dim,
        int num_threads_per_head_dim) {
    __half2 q1[4], k1[4], v[4], g_qs[4];
    read_int4(q_local, q1, 1);
    read_int4(k_local, k1, 1);
    read_int4(v_local, v, 1);
    read_int4(g_a_local, g_qs, 1);

    __half2 qz = __half2half2(qz_half);
    __half2 g_qz = __half2half2(g_qz_half);
    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
        g_qs[i] = __h2div(g_qs[i], qz);
    }
    __half2 q2[4] = {
        __lowhigh2highlow(q1[0]),
        __lowhigh2highlow(q1[1]),
        __lowhigh2highlow(q1[2]),
        __lowhigh2highlow(q1[3]),
    };

    __half2 k2[4] = {
        __lowhigh2highlow(k1[0]),
        __lowhigh2highlow(k1[1]),
        __lowhigh2highlow(k1[2]),
        __lowhigh2highlow(k1[3]),
    };

    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        g_k1[i] = __float2half2_rn(0.f);
        __half2 g_k2 = __float2half2_rn(0.f);
        t[i] = __hfma2(g_qz, q1[i], t[i]);
        #pragma unroll
        for (int j = 0;j < 4; ++ j) {
            s1[i][j] = __hfma2(q1[i], g_qs[j], s1[i][j]);
            s2[i][j] = __hfma2(q2[i], g_qs[j], s2[i][j]);

            g_k1[i] = __hfma2(s1[i][j], v[j], g_k1[i]);
            g_k2 = __hfma2(s2[i][j], v[j], g_k2);
        }
        g_k1[i] = __hadd2(g_k1[i], __lowhigh2highlow(g_k2));
    }
    #pragma unroll
    for (int j = 0;j < 4; ++ j) {
        g_v1[j] = __float2half2_rn(0.f);
        __half2 g_v2 = __float2half2_rn(0.f);
        #pragma unroll
        for (int i = 0;i < 4; ++ i) {
            g_v1[j] = __hfma2(s1[i][j], k1[i], g_v1[j]);
            g_v2 = __hfma2(s2[i][j], k2[i], g_v2);
        }
        g_v1[j] = __hadd2(g_v1[j], g_v2);
    }

    // 128 x 256 case:
    // sum through 128 / 8 = 16 head_dim threads:
    // thread_idx:
    // 0 + 32 + 64 + ... + 480
    // 1 + 33 + 65 + ... + 481
    // ...
    // 31 + 63 + ... + 511

    shared_mem_int4[threadIdx.x] = ((int4*) g_k1)[0];
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
                g_k1[i] = __hadd2(g_k1[i], tmp[i]);
            }
        }
    }
    #pragma unroll
    for (int i = 0;i < 4; ++ i) {
        g_k1[i] = __hadd2(g_k1[i], t[i]);
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
            g_v1[j] =  __hadd2(
                g_v1[j], 
                __shfl_down_sync(FULL_MASK, g_v1[j], offset));
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
    const __half* __restrict__ g_a_local = grad_attn + head_dim_offset;

    __half* __restrict__ g_q_local = grad_q + proj_dim_offset;

    __half2 g_q[4];
    __half qz_half, g_qz_half;

    __half2 s1[4][4] = {__float2half2_rn(0.f)};
    __half2 s2[4][4] = {__float2half2_rn(0.f)};
    __half2 z[4] = {__float2half2_rn(0.f)};
    for (int t = 0; t < tgt_len; ++ t) {
        q_backward_step(
            q_local, k_local, v_local,
            qz_half, g_a_local,
            g_q, g_qz_half,
            s1, s2,
            z,
            shared_mem_half,
            shared_mem_int4,
            num_threads_per_proj_dim,
            num_threads_per_head_dim
        );
        if (head_dim_offset == 0) {
            ((int4 *) g_q_local)[0] = ((int4 *) g_q)[0];
        }

        if (threadIdx.x == 0) {
            qz[t] = qz_half;
            grad_qz[t] = g_qz_half;
        }
        __syncthreads();
        q_local += qk_inc_t;
        k_local += qk_inc_t;
        v_local += v_inc_t;
        g_a_local += v_inc_t;
        g_q_local += qk_inc_t;
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
    const __half* __restrict__ g_a_local = grad_attn + head_dim_offset;

    __half* __restrict__ g_k_local = grad_k + proj_dim_offset;
    __half* __restrict__ g_v_local = grad_v + head_dim_offset;

    __half2 s1[4][4] = {__float2half2_rn(0.f)};
    __half2 s2[4][4] = {__float2half2_rn(0.f)};
    __half2 t[4] = {__float2half2_rn(0.f)};
    __half2 g_k[4], g_v[4];
    __half qz_half, g_qz_half;

    int offset = tgt_len - 1;
    q_local += qk_inc_t * offset;
    k_local += qk_inc_t * offset;
    v_local += v_inc_t * offset;
    g_a_local += v_inc_t * offset;

    g_k_local += qk_inc_t * offset;
    g_v_local += v_inc_t * offset;
    for (int i = 0; i < tgt_len; ++ i) {
        qz_half = qz[tgt_len - i - 1];
        g_qz_half = grad_qz[tgt_len - i - 1];
        kv_backward_step(
            q_local, k_local, v_local, qz_half, g_a_local,
            g_k, g_v, g_qz_half,
            s1, s2, t,
            shared_mem_int4,
            num_threads_per_proj_dim,
            num_threads_per_head_dim
        );

        if (proj_dim_offset == 0) {
            ((int4 *) g_v_local)[0] = ((int4 *) g_v)[0];
        }
        if (head_dim_offset == 0) {
            ((int4 *) g_k_local)[0] = ((int4 *) g_k)[0];
        }
        __syncthreads();
        q_local -= qk_inc_t;
        k_local -= qk_inc_t;
        v_local -= v_inc_t;
        g_a_local -= v_inc_t;

        g_k_local -= qk_inc_t;
        g_v_local -= v_inc_t;
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
    const __half * __restrict__ g_a_local = grad_attn + bid * head_dim;

    __half * __restrict__ grad_q_local = grad_q + bid * proj_dim;
    __half * __restrict__ grad_k_local = grad_k + bid * proj_dim;
    __half * __restrict__ grad_v_local = grad_v + bid * head_dim;

    extern __shared__ __half qz_shared[];
    __shared__ __half shared_mem_half[NUM_THREADS_PER_BLOCK];
    __shared__ int4 shared_mem_int4[NUM_THREADS_PER_BLOCK];
    __half * __restrict__ g_qz_shared = qz_shared + tgt_len;

    q_backward(
        q_local,
        k_local,
        v_local,
        qz_shared,
        g_a_local,
        g_qz_shared,
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
        g_a_local,
        g_qz_shared,
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

