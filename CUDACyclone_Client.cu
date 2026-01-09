// ============================================================================
// CUDACyclone Distributed Mode - Client Implementation
// ============================================================================
// GPU worker client that connects to server for work distribution
// Reuses the kernel from CUDACyclone_MultiGPU for proven correctness
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <mutex>
#include <csignal>

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"
#include "CUDACyclone_Protocol.h"
#include "CUDACyclone_Network.h"

// ============================================================================
// KXE PERMUTATION (Host-side for computing block ranges)
// ============================================================================

namespace kxe_host {

// Feistel round function (host-side)
inline uint64_t feistel_round(uint64_t half, uint32_t round_key) {
    uint64_t x = half ^ (uint64_t)round_key;
    x = x * 0x517cc1b727220a95ULL;
    x ^= (x >> 33);
    x = x * 0x9e3779b97f4a7c15ULL;
    x ^= (x >> 33);
    return x;
}

// 4-round Feistel permutation (host-side)
inline uint64_t feistel_permute(uint64_t input, uint64_t domain_size, uint64_t seed) {
    int half_bits = 32;
    uint64_t mask = 0xFFFFFFFFULL;

    // Adjust for smaller domains
    if (domain_size <= (1ULL << 16)) {
        half_bits = 8;
        mask = 0xFFULL;
    } else if (domain_size <= (1ULL << 32)) {
        half_bits = 16;
        mask = 0xFFFFULL;
    }

    uint64_t L = input >> half_bits;
    uint64_t R = input & mask;

    // Derive round keys from seed
    uint32_t round_keys[4];
    uint64_t key_state = seed;
    for (int i = 0; i < 4; ++i) {
        key_state = key_state * 0x5851f42d4c957f2dULL + 0x14057b7ef767814fULL;
        round_keys[i] = (uint32_t)(key_state >> 32);
    }

    // 4 Feistel rounds
    for (int r = 0; r < 4; ++r) {
        uint64_t F = feistel_round(R, round_keys[r]) & mask;
        uint64_t new_L = R;
        uint64_t new_R = L ^ F;
        L = new_L;
        R = new_R;
    }

    return (L << half_bits) | R;
}

// Permute block index with cycle-walking for non-power-of-2 domains
inline uint64_t permute_block(uint64_t block_index, uint64_t total_blocks, uint64_t seed) {
    if (total_blocks <= 1) return 0;

    // Find smallest power of 2 >= total_blocks
    uint64_t domain_size = 1;
    while (domain_size < total_blocks) domain_size <<= 1;

    // Cycle-walk until we get a valid block index
    uint64_t permuted = block_index;
    do {
        permuted = feistel_permute(permuted, domain_size, seed);
    } while (permuted >= total_blocks);

    return permuted;
}

} // namespace kxe_host

// ============================================================================
// CONSTANTS
// ============================================================================

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ============================================================================
// CLIENT CONFIGURATION
// ============================================================================

struct ClientConfig {
    std::string server_host;
    uint16_t server_port;
    std::vector<int> gpu_ids;
    uint32_t batch_size;
    uint32_t batches_per_sm;
    uint32_t slices_per_launch;
    uint32_t heartbeat_interval_sec;
    uint32_t progress_interval_sec;
    bool enable_pincer;
    uint32_t reconnect_delay_sec;

    ClientConfig() {
        server_host = "127.0.0.1";
        server_port = DEFAULT_SERVER_PORT;
        batch_size = 128;
        batches_per_sm = 128;
        slices_per_launch = 16;
        heartbeat_interval_sec = DEFAULT_HEARTBEAT_INTERVAL_SEC;
        progress_interval_sec = DEFAULT_PROGRESS_INTERVAL_SEC;
        enable_pincer = true;
        reconnect_delay_sec = 5;
    }
};

// ============================================================================
// GPU CONTEXT
// ============================================================================

// Constant memory symbols
__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];

struct GPUContext {
    int deviceId;
    cudaDeviceProp prop;
    cudaStream_t stream;

    // Device memory
    uint64_t* d_start_scalars;
    uint64_t* d_Px;
    uint64_t* d_Py;
    uint64_t* d_Rx;
    uint64_t* d_Ry;
    uint64_t* d_counts256;
    int* d_found_flag;
    FoundResult* d_found_result;
    unsigned long long* d_hashes_accum;
    unsigned int* d_any_left;

    // Host pinned memory
    uint64_t* h_counts256;
    uint64_t* h_start_scalars;

    // GPU-specific range
    uint64_t range_start[4];
    uint64_t range_len[4];
    uint64_t per_thread_cnt[4];

    // Execution parameters
    uint64_t threadsTotal;
    int blocks;
    int threadsPerBlock;

    // Runtime state
    std::atomic<bool> completed{false};
    std::atomic<bool> found_match{false};
    unsigned long long last_hashes{0};

    GPUContext() : deviceId(-1), stream(nullptr),
                   d_start_scalars(nullptr), d_Px(nullptr), d_Py(nullptr),
                   d_Rx(nullptr), d_Ry(nullptr), d_counts256(nullptr),
                   d_found_flag(nullptr), d_found_result(nullptr),
                   d_hashes_accum(nullptr), d_any_left(nullptr),
                   h_counts256(nullptr), h_start_scalars(nullptr),
                   threadsTotal(0), blocks(0), threadsPerBlock(256) {
        memset(range_start, 0, sizeof(range_start));
        memset(range_len, 0, sizeof(range_len));
        memset(per_thread_cnt, 0, sizeof(per_thread_cnt));
    }
};

// ============================================================================
// GLOBAL STATE
// ============================================================================

static std::atomic<bool> g_running{true};
static std::atomic<bool> g_found_global{false};
static std::atomic<int> g_found_by_gpu{-1};
static std::mutex g_result_mutex;
static FoundResult g_global_result;
static std::mutex g_socket_mutex;
static socket_t g_server_socket = INVALID_SOCKET_VALUE;
static uint32_t g_client_id = 0;

// KXE mode state (set during registration)
static bool g_kxe_mode = false;
static uint64_t g_kxe_seed = 0;
static uint64_t g_total_blocks = 0;

static void signal_handler(int) {
    g_running.store(false);
}

// ============================================================================
// KERNEL HELPER FUNCTIONS
// ============================================================================

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}

__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask, unsigned lane) {
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

// ============================================================================
// KERNEL (proven working version from CUDACyclone_MultiGPU)
// ============================================================================

__launch_bounds__(256, 2)
__global__ void kernel_search(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], S[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i]  = start_scalars[idx];
    }
    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // Check initial point (S)
        {
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            bool pref = hash160_prefix_equals(h20, target_prefix);
            if (__any_sync(full_mask, pref)) {
                if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=S[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=x1[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y1[k];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }
        }

        // Batch inversion setup
        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

#pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Jx[j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
#pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
#pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
#pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        uint64_t sy_neg[4], sx_neg[4];
        ModNeg256(sy_neg, y1);
        ModNeg256(sx_neg, x1);

        // Process points 0 to half-2
        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            // Positive branch: P + (i+1)*G
            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = hash160_prefix_equals(h20, target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t addv=(uint64_t)(i+1);
                            for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];

                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            // Negative branch: P - (i+1)*G
            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                ModNeg256(py_i, py_i);

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = hash160_prefix_equals(h20, target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t sub=(uint64_t)(i+1);
                            for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            // Update inverse for next iteration
            uint64_t gxmi[4];
#pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        // Process last point (half-1) - only negative branch needed
        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t px3[4], s[4], lam[4];
            uint64_t px_i[4], py_i[4];
#pragma unroll
            for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
            ModNeg256(py_i, py_i);

            ModSub256(s, py_i, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256(px3, px3, x1);
            ModSub256(px3, px3, px_i);

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, y1, &odd);

            uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            bool pref = hash160_prefix_equals(h20, target_prefix);
            if (__any_sync(full_mask, pref)) {
                if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                        uint64_t sub=(uint64_t)half;
                        for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                        uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }

            uint64_t last_dx[4];
#pragma unroll
            for (int j=0;j<4;++j) last_dx[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(last_dx, last_dx, x1);
            _ModMult(inverse, inverse, last_dx);
        }

        // Jump to next batch
        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_Jy[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=c_Jx[j];
            ModSub256(x3, x3, Jx_local);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

#pragma unroll
            for (int j=0;j<4;++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
        }

        // Update scalar and remaining count
        {
            uint64_t addv=(uint64_t)B;
            for (int k=0;k<4 && addv;++k){ uint64_t old=S[k]; S[k]=old+addv; addv=(S[k]<old)?1ull:0ull; }
            sub256_u64_inplace(rem, (uint64_t)B);
        }
        ++batches_done;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

// ============================================================================
// GPU INITIALIZATION
// ============================================================================

bool init_gpu_context(GPUContext& ctx, uint32_t batch_size, uint32_t batches_per_sm) {
    CUDA_CHECK(cudaSetDevice(ctx.deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&ctx.prop, ctx.deviceId));
    CUDA_CHECK(cudaStreamCreate(&ctx.stream));

    ctx.threadsPerBlock = 256;
    int smCount = ctx.prop.multiProcessorCount;
    ctx.blocks = smCount * batches_per_sm;
    ctx.threadsTotal = (uint64_t)ctx.blocks * ctx.threadsPerBlock;

    size_t sz4 = ctx.threadsTotal * 4 * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc(&ctx.d_start_scalars, sz4));
    CUDA_CHECK(cudaMalloc(&ctx.d_Px, sz4));
    CUDA_CHECK(cudaMalloc(&ctx.d_Py, sz4));
    CUDA_CHECK(cudaMalloc(&ctx.d_Rx, sz4));
    CUDA_CHECK(cudaMalloc(&ctx.d_Ry, sz4));
    CUDA_CHECK(cudaMalloc(&ctx.d_counts256, sz4));
    CUDA_CHECK(cudaMalloc(&ctx.d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_found_result, sizeof(FoundResult)));
    CUDA_CHECK(cudaMalloc(&ctx.d_hashes_accum, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx.d_any_left, sizeof(unsigned int)));

    CUDA_CHECK(cudaMallocHost(&ctx.h_counts256, sz4));
    CUDA_CHECK(cudaMallocHost(&ctx.h_start_scalars, sz4));

    int zero_flag = FOUND_NONE;
    CUDA_CHECK(cudaMemcpy(ctx.d_found_flag, &zero_flag, sizeof(int), cudaMemcpyHostToDevice));
    unsigned long long zero_ull = 0;
    CUDA_CHECK(cudaMemcpy(ctx.d_hashes_accum, &zero_ull, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    return true;
}

void cleanup_gpu_context(GPUContext& ctx) {
    if (ctx.deviceId < 0) return;
    cudaSetDevice(ctx.deviceId);

    if (ctx.stream) cudaStreamDestroy(ctx.stream);
    if (ctx.d_start_scalars) cudaFree(ctx.d_start_scalars);
    if (ctx.d_Px) cudaFree(ctx.d_Px);
    if (ctx.d_Py) cudaFree(ctx.d_Py);
    if (ctx.d_Rx) cudaFree(ctx.d_Rx);
    if (ctx.d_Ry) cudaFree(ctx.d_Ry);
    if (ctx.d_counts256) cudaFree(ctx.d_counts256);
    if (ctx.d_found_flag) cudaFree(ctx.d_found_flag);
    if (ctx.d_found_result) cudaFree(ctx.d_found_result);
    if (ctx.d_hashes_accum) cudaFree(ctx.d_hashes_accum);
    if (ctx.d_any_left) cudaFree(ctx.d_any_left);
    if (ctx.h_counts256) cudaFreeHost(ctx.h_counts256);
    if (ctx.h_start_scalars) cudaFreeHost(ctx.h_start_scalars);

    ctx.deviceId = -1;
}

// ============================================================================
// NETWORK COMMUNICATION
// ============================================================================

bool send_to_server(MessageType type, const void* payload = nullptr, uint16_t size = 0) {
    std::lock_guard<std::mutex> lock(g_socket_mutex);
    if (g_server_socket == INVALID_SOCKET_VALUE) return false;
    return net::send_message(g_server_socket, type, payload, size);
}

bool connect_to_server(const ClientConfig& config) {
    std::cout << "[Client] Connecting to " << config.server_host << ":"
              << config.server_port << "...\n";

    g_server_socket = net::connect_to_server(config.server_host, config.server_port, 10000);
    if (g_server_socket == INVALID_SOCKET_VALUE) {
        std::cerr << "[Client] Failed to connect: " << net::get_last_error_string() << "\n";
        return false;
    }

    std::cout << "[Client] Connected!\n";
    return true;
}

bool register_with_server(const ClientConfig& config, const std::vector<GPUContext>& contexts) {
    std::string hostname = net::get_hostname();
    size_t msg_size = sizeof(RegisterRequestMsg) + hostname.size() +
                      contexts.size() * sizeof(GPUInfoMsg);
    std::vector<uint8_t> buffer(msg_size);

    auto* req = reinterpret_cast<RegisterRequestMsg*>(buffer.data());
    req->protocol_version = PROTOCOL_VERSION;
    req->gpu_count = static_cast<uint32_t>(contexts.size());
    req->batch_size = config.batch_size;
    req->slices = config.slices_per_launch;
    req->supports_pincer = config.enable_pincer ? 1 : 0;
    req->hostname_len = static_cast<uint8_t>(std::min(hostname.size(), (size_t)255));

    size_t offset = sizeof(RegisterRequestMsg);
    memcpy(buffer.data() + offset, hostname.c_str(), req->hostname_len);
    offset += req->hostname_len;

    for (const auto& ctx : contexts) {
        auto* info = reinterpret_cast<GPUInfoMsg*>(buffer.data() + offset);
        memset(info, 0, sizeof(GPUInfoMsg));
        strncpy(info->name, ctx.prop.name, sizeof(info->name) - 1);
        info->compute_cap_major = ctx.prop.major;
        info->compute_cap_minor = ctx.prop.minor;
        info->memory_bytes = ctx.prop.totalGlobalMem;
        offset += sizeof(GPUInfoMsg);
    }

    if (!send_to_server(MessageType::REGISTER_REQUEST, buffer.data(),
                        static_cast<uint16_t>(buffer.size()))) {
        std::cerr << "[Client] Failed to send registration\n";
        return false;
    }

    MessageHeader header;
    if (!net::recv_message_header(g_server_socket, header, 10000)) {
        std::cerr << "[Client] Failed to receive registration response\n";
        return false;
    }

    if (header.type() != MessageType::REGISTER_RESPONSE) {
        std::cerr << "[Client] Unexpected response type\n";
        return false;
    }

    RegisterResponseMsg resp;
    if (!net::recv_message_payload(g_server_socket, &resp, sizeof(resp))) {
        std::cerr << "[Client] Failed to receive response payload\n";
        return false;
    }

    if (resp.status != static_cast<uint32_t>(ErrorCode::OK)) {
        std::cerr << "[Client] Registration rejected: "
                  << error_code_name(static_cast<ErrorCode>(resp.status)) << "\n";
        return false;
    }

    g_client_id = resp.client_id;

    // Capture KXE mode settings from server
    g_kxe_mode = (resp.scan_mode == static_cast<uint8_t>(ScanMode::KXE));
    g_kxe_seed = resp.kxe_seed;
    g_total_blocks = resp.total_blocks;

    std::cout << "[Client] Registered as client #" << g_client_id << "\n";
    if (g_kxe_mode) {
        std::cout << "[Client] KXE mode enabled (seed: " << g_kxe_seed
                  << ", blocks: " << g_total_blocks << ")\n";
    }
    return true;
}

bool request_work(WorkAssignmentMsg& work) {
    WorkRequestMsg req;
    req.client_id = g_client_id;
    req.units_requested = 1;

    if (!send_to_server(MessageType::WORK_REQUEST, &req, sizeof(req))) {
        return false;
    }

    MessageHeader header;
    if (!net::recv_message_header(g_server_socket, header, 30000)) {
        return false;
    }

    if (header.type() == MessageType::WORK_ASSIGNMENT) {
        return net::recv_message_payload(g_server_socket, &work, sizeof(work));
    }
    else if (header.type() == MessageType::NO_WORK_AVAILABLE) {
        NoWorkAvailableMsg msg;
        net::recv_message_payload(g_server_socket, &msg, sizeof(msg));
        if (msg.reason == 1) {
            std::cout << "[Client] Search complete\n";
            g_running.store(false);
        }
        return false;
    }
    else if (header.type() == MessageType::KEY_FOUND) {
        KeyFoundMsg msg;
        net::recv_message_payload(g_server_socket, &msg, sizeof(msg));
        std::cout << "\n[Client] KEY FOUND by client #" << msg.finder_client_id << "!\n";
        g_running.store(false);
        return false;
    }
    else if (header.type() == MessageType::SERVER_SHUTDOWN) {
        std::cout << "[Client] Server shutting down\n";
        g_running.store(false);
        return false;
    }

    return false;
}

void send_heartbeat(double speed) {
    HeartbeatMsg msg;
    msg.client_id = g_client_id;
    msg.active_unit_count = 1;
    msg.current_speed_gkeys = speed;
    send_to_server(MessageType::HEARTBEAT, &msg, sizeof(msg));
}

void send_progress(uint32_t unit_id, uint64_t keys_processed, double speed) {
    ProgressReportMsg msg;
    msg.client_id = g_client_id;
    msg.unit_id = unit_id;
    msg.keys_processed = keys_processed;
    msg.speed_gkeys = speed;
    msg.status = 0;
    send_to_server(MessageType::PROGRESS_REPORT, &msg, sizeof(msg));
}

void send_unit_complete(uint32_t unit_id, uint64_t keys_processed, double avg_speed) {
    UnitCompleteMsg msg;
    msg.client_id = g_client_id;
    msg.unit_id = unit_id;
    msg.keys_processed = keys_processed;
    msg.avg_speed_gkeys = avg_speed;
    send_to_server(MessageType::UNIT_COMPLETE, &msg, sizeof(msg));
}

void send_found_result(uint32_t unit_id, const FoundResult& result, const uint8_t target_hash160[20]) {
    FoundResultMsg msg;
    msg.client_id = g_client_id;
    msg.unit_id = unit_id;
    memcpy(msg.scalar, result.scalar, sizeof(msg.scalar));
    memcpy(msg.pubkey_x, result.Rx, sizeof(msg.pubkey_x));
    memcpy(msg.pubkey_y, result.Ry, sizeof(msg.pubkey_y));
    // Copy the target hash160 - we know the found result matches it
    memcpy(msg.hash160, target_hash160, 20);
    send_to_server(MessageType::FOUND_RESULT, &msg, sizeof(msg));
}

// ============================================================================
// WORK EXECUTION
// ============================================================================

void execute_work_unit(std::vector<GPUContext>& contexts,
                       const WorkAssignmentMsg& work,
                       const ClientConfig& config) {
    int num_gpus = static_cast<int>(contexts.size());
    uint32_t batch_size = work.batch_size > 0 ? work.batch_size : config.batch_size;
    uint32_t slices = work.slices > 0 ? work.slices : config.slices_per_launch;

    // Determine actual range to search
    uint64_t actual_range_start[4];
    uint64_t actual_range_end[4];
    uint64_t range_len[4];

    bool is_kxe = (work.scan_mode == static_cast<uint8_t>(ScanMode::KXE));

    if (is_kxe && work.keys_per_block > 0) {
        // KXE mode: compute actual range from permuted block index
        uint64_t block_index = work.kxe_block_index;
        uint64_t total_blocks = g_total_blocks;
        uint64_t seed = work.kxe_seed;

        // Permute the block index
        uint64_t permuted_block = kxe_host::permute_block(block_index, total_blocks, seed);

        // Calculate block start: global_range_start + permuted_block * keys_per_block
        memcpy(actual_range_start, work.range_start, sizeof(actual_range_start));
        uint64_t block_offset[4] = {0, 0, 0, 0};
        __uint128_t prod = (__uint128_t)permuted_block * work.keys_per_block;
        block_offset[0] = (uint64_t)prod;
        block_offset[1] = (uint64_t)(prod >> 64);
        add256(actual_range_start, block_offset, actual_range_start);

        // Block end: block_start + keys_per_block - 1
        memcpy(actual_range_end, actual_range_start, sizeof(actual_range_end));
        add256_u64(actual_range_end, work.keys_per_block - 1, actual_range_end);

        // Range length = keys_per_block
        range_len[0] = work.keys_per_block;
        range_len[1] = range_len[2] = range_len[3] = 0;

        std::cout << "[Client] Executing KXE block #" << work.unit_id
                  << " (permuted: " << permuted_block << ")\n";
        std::cout << "[Client] Block range: " << formatHex256(actual_range_start)
                  << " : " << formatHex256(actual_range_end) << "\n";
    } else {
        // Sequential mode: use range directly
        memcpy(actual_range_start, work.range_start, sizeof(actual_range_start));
        memcpy(actual_range_end, work.range_end, sizeof(actual_range_end));

        // Calculate range size
        sub256(work.range_end, work.range_start, range_len);
        add256_u64(range_len, 1, range_len);

        std::cout << "[Client] Executing work unit #" << work.unit_id << "\n";
        std::cout << "[Client] Range: " << formatHex256(actual_range_start) << " : "
                  << formatHex256(actual_range_end) << "\n";
    }

    // Setup each GPU
    for (int i = 0; i < num_gpus; ++i) {
        GPUContext& ctx = contexts[i];
        CUDA_CHECK(cudaSetDevice(ctx.deviceId));

        // Divide range among GPUs
        uint64_t per_gpu[4], remainder;
        divmod_256_by_u64(range_len, num_gpus, per_gpu, remainder);

        // Calculate offset for this GPU
        uint64_t offset[4] = {0, 0, 0, 0};
        __uint128_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            __uint128_t prod = (__uint128_t)per_gpu[j] * i + carry;
            offset[j] = (uint64_t)prod;
            carry = prod >> 64;
        }

        add256(actual_range_start, offset, ctx.range_start);
        memcpy(ctx.range_len, per_gpu, sizeof(per_gpu));

        // Calculate per-thread work
        uint64_t total_keys_this_gpu[4];
        memcpy(total_keys_this_gpu, per_gpu, sizeof(per_gpu));

        uint64_t batches[4];
        divmod_256_by_u64(total_keys_this_gpu, batch_size, batches, remainder);

        divmod_256_by_u64(batches, ctx.threadsTotal, ctx.per_thread_cnt, remainder);

        // Initialize thread scalars
        for (uint64_t t = 0; t < ctx.threadsTotal; ++t) {
            uint64_t thread_batch_offset = t * ctx.per_thread_cnt[0] * batch_size;
            uint64_t scalar[4];
            memcpy(scalar, ctx.range_start, sizeof(scalar));
            add256_u64(scalar, thread_batch_offset + batch_size / 2, scalar);

            for (int j = 0; j < 4; ++j) {
                ctx.h_start_scalars[t * 4 + j] = scalar[j];
                ctx.h_counts256[t * 4 + j] = (j == 0) ? ctx.per_thread_cnt[0] * batch_size : 0;
            }
        }

        // Copy to device
        size_t sz4 = ctx.threadsTotal * 4 * sizeof(uint64_t);
        CUDA_CHECK(cudaMemcpy(ctx.d_start_scalars, ctx.h_start_scalars, sz4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx.d_counts256, ctx.h_counts256, sz4, cudaMemcpyHostToDevice));

        // Reset flags
        int zero = FOUND_NONE;
        CUDA_CHECK(cudaMemcpy(ctx.d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice));
        unsigned long long zero_ull = 0;
        CUDA_CHECK(cudaMemcpy(ctx.d_hashes_accum, &zero_ull, sizeof(unsigned long long), cudaMemcpyHostToDevice));

        // Set target
        CUDA_CHECK(cudaMemcpyToSymbol(c_target_hash160, work.target_hash160, 20));
        uint32_t prefix = work.target_hash160[0] |
                          (work.target_hash160[1] << 8) |
                          (work.target_hash160[2] << 16) |
                          (work.target_hash160[3] << 24);
        CUDA_CHECK(cudaMemcpyToSymbol(c_target_prefix, &prefix, sizeof(uint32_t)));

        // Compute initial points
        scalarMulKernelBase<<<ctx.blocks, ctx.threadsPerBlock>>>(
            ctx.d_start_scalars, ctx.d_Px, ctx.d_Py, ctx.threadsTotal);
        CUDA_CHECK(cudaDeviceSynchronize());

        ctx.completed.store(false);
        ctx.found_match.store(false);
        ctx.last_hashes = 0;
    }

    // Precompute batch points on GPU 0
    {
        GPUContext& ctx = contexts[0];
        CUDA_CHECK(cudaSetDevice(ctx.deviceId));

        std::vector<uint64_t> h_scalars256((batch_size / 2) * 4, 0);
        for (uint32_t i = 0; i < batch_size / 2; ++i) {
            h_scalars256[i * 4] = i + 1;
        }

        uint64_t* d_scalars256;
        uint64_t* d_temp_x;
        uint64_t* d_temp_y;
        size_t batch_sz = (batch_size / 2) * 4 * sizeof(uint64_t);

        CUDA_CHECK(cudaMalloc(&d_scalars256, batch_sz));
        CUDA_CHECK(cudaMalloc(&d_temp_x, batch_sz));
        CUDA_CHECK(cudaMalloc(&d_temp_y, batch_sz));
        CUDA_CHECK(cudaMemcpy(d_scalars256, h_scalars256.data(), batch_sz, cudaMemcpyHostToDevice));

        scalarMulKernelBase<<<(batch_size / 2 + 255) / 256, 256>>>(
            d_scalars256, d_temp_x, d_temp_y, batch_size / 2);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint64_t> h_Gx(batch_sz / sizeof(uint64_t));
        std::vector<uint64_t> h_Gy(batch_sz / sizeof(uint64_t));
        CUDA_CHECK(cudaMemcpy(h_Gx.data(), d_temp_x, batch_sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Gy.data(), d_temp_y, batch_sz, cudaMemcpyDeviceToHost));

        // Jump point
        std::vector<uint64_t> h_J256(4, 0);
        h_J256[0] = batch_size;
        uint64_t* d_J256;
        uint64_t* d_Jx;
        uint64_t* d_Jy;
        CUDA_CHECK(cudaMalloc(&d_J256, 32));
        CUDA_CHECK(cudaMalloc(&d_Jx, 32));
        CUDA_CHECK(cudaMalloc(&d_Jy, 32));
        CUDA_CHECK(cudaMemcpy(d_J256, h_J256.data(), 32, cudaMemcpyHostToDevice));

        scalarMulKernelBase<<<1, 1>>>(d_J256, d_Jx, d_Jy, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint64_t> h_Jx(4), h_Jy(4);
        CUDA_CHECK(cudaMemcpy(h_Jx.data(), d_Jx, 32, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Jy.data(), d_Jy, 32, cudaMemcpyDeviceToHost));

        // Copy to all GPUs
        for (auto& gpu : contexts) {
            CUDA_CHECK(cudaSetDevice(gpu.deviceId));
            CUDA_CHECK(cudaMemcpyToSymbol(c_Gx, h_Gx.data(), batch_sz));
            CUDA_CHECK(cudaMemcpyToSymbol(c_Gy, h_Gy.data(), batch_sz));
            CUDA_CHECK(cudaMemcpyToSymbol(c_Jx, h_Jx.data(), 32));
            CUDA_CHECK(cudaMemcpyToSymbol(c_Jy, h_Jy.data(), 32));
        }

        cudaFree(d_scalars256);
        cudaFree(d_temp_x);
        cudaFree(d_temp_y);
        cudaFree(d_J256);
        cudaFree(d_Jx);
        cudaFree(d_Jy);
    }

    // Launch workers
    std::vector<std::thread> workers;
    auto start_time = std::chrono::steady_clock::now();
    std::atomic<unsigned long long> total_hashes{0};

    for (int i = 0; i < num_gpus; ++i) {
        workers.emplace_back([&, i]() {
            GPUContext& ctx = contexts[i];
            CUDA_CHECK(cudaSetDevice(ctx.deviceId));

            while (!ctx.completed.load() && g_running.load() && !g_found_global.load()) {
                unsigned int zero = 0;
                CUDA_CHECK(cudaMemcpy(ctx.d_any_left, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));

                kernel_search<<<ctx.blocks, ctx.threadsPerBlock, 0, ctx.stream>>>(
                    ctx.d_Px, ctx.d_Py, ctx.d_Rx, ctx.d_Ry,
                    ctx.d_start_scalars, ctx.d_counts256,
                    ctx.threadsTotal, batch_size, slices,
                    ctx.d_found_flag, ctx.d_found_result,
                    ctx.d_hashes_accum, ctx.d_any_left
                );
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

                // Check found
                int host_found = 0;
                CUDA_CHECK(cudaMemcpy(&host_found, ctx.d_found_flag, sizeof(int), cudaMemcpyDeviceToHost));
                if (host_found == FOUND_READY) {
                    int expected = -1;
                    if (g_found_by_gpu.compare_exchange_strong(expected, ctx.deviceId)) {
                        g_found_global.store(true);
                        FoundResult result;
                        CUDA_CHECK(cudaMemcpy(&result, ctx.d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost));
                        std::lock_guard<std::mutex> lock(g_result_mutex);
                        g_global_result = result;
                        ctx.found_match.store(true);
                    }
                    break;
                }

                // Check work remaining
                unsigned int any_left = 0;
                CUDA_CHECK(cudaMemcpy(&any_left, ctx.d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost));
                if (!any_left) {
                    ctx.completed.store(true);
                    break;
                }

                std::swap(ctx.d_Px, ctx.d_Rx);
                std::swap(ctx.d_Py, ctx.d_Ry);

                unsigned long long h;
                CUDA_CHECK(cudaMemcpy(&h, ctx.d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                total_hashes.fetch_add(h - ctx.last_hashes);
                ctx.last_hashes = h;
            }
        });
    }

    // Monitor and report
    auto last_heartbeat = std::chrono::steady_clock::now();
    auto last_progress = std::chrono::steady_clock::now();

    while (g_running.load() && !g_found_global.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        bool all_done = true;
        for (auto& ctx : contexts) {
            if (!ctx.completed.load()) all_done = false;
        }
        if (all_done) break;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double speed = (elapsed > 0) ? (total_hashes.load() / elapsed / 1e9) : 0.0;

        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_progress).count() >= 10) {
            send_progress(work.unit_id, total_hashes.load(), speed);
            last_progress = now;
        }

        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat).count() >= 30) {
            send_heartbeat(speed);
            last_heartbeat = now;
        }

        std::cout << "\r[Unit #" << work.unit_id << "] "
                  << std::fixed << std::setprecision(2) << speed << " Gkeys/s | "
                  << (total_hashes.load() / 1e9) << " G keys     " << std::flush;
    }

    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }

    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    double avg_speed = (total_time > 0) ? (total_hashes.load() / total_time / 1e9) : 0.0;

    std::cout << "\n";

    if (g_found_global.load()) {
        std::cout << "[Client] KEY FOUND!\n";
        std::cout << "[Client] Private key: " << formatHex256(g_global_result.scalar) << "\n";
        send_found_result(work.unit_id, g_global_result, work.target_hash160);
    } else {
        std::cout << "[Client] Work unit #" << work.unit_id << " completed\n";
        send_unit_complete(work.unit_id, total_hashes.load(), avg_speed);
    }
}

// ============================================================================
// COMMAND-LINE PARSING
// ============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  --server <host:port>  Server address (default: 127.0.0.1:17403)\n\n";
    std::cout << "Options:\n";
    std::cout << "  --gpus <ids>          Comma-separated GPU IDs (default: all)\n";
    std::cout << "  --grid <A,B>          A=batch size, B=batches per SM\n";
    std::cout << "  --slices <N>          Slices per kernel launch\n";
    std::cout << "  --pincer              Enable local pincer mode\n";
    std::cout << "  -h, --help            Show this help\n";
}

bool parse_args(int argc, char* argv[], ClientConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
        else if (arg == "--server" && i + 1 < argc) {
            std::string addr = argv[++i];
            if (!net::parse_address(addr, config.server_host, config.server_port)) {
                std::cerr << "Invalid server address\n";
                return false;
            }
        }
        else if (arg == "--gpus" && i + 1 < argc) {
            std::string gpus = argv[++i];
            std::istringstream iss(gpus);
            std::string token;
            while (std::getline(iss, token, ',')) {
                config.gpu_ids.push_back(std::stoi(token));
            }
        }
        else if (arg == "--grid" && i + 1 < argc) {
            std::string grid = argv[++i];
            size_t comma = grid.find(',');
            if (comma != std::string::npos) {
                config.batch_size = std::stoi(grid.substr(0, comma));
                config.batches_per_sm = std::stoi(grid.substr(comma + 1));
            }
        }
        else if (arg == "--slices" && i + 1 < argc) {
            config.slices_per_launch = std::stoi(argv[++i]);
        }
        else if (arg == "--pincer") {
            config.enable_pincer = true;
        }
    }

    if (config.gpu_ids.empty()) {
        int count;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) config.gpu_ids.push_back(i);
    }

    return true;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    ClientConfig config;
    if (!parse_args(argc, argv, config)) {
        print_usage(argv[0]);
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (!net::initialize()) {
        std::cerr << "[Client] Network init failed\n";
        return 1;
    }

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  CUDACyclone Distributed Client v1.0\n";
    std::cout << "========================================\n\n";

    std::vector<GPUContext> contexts(config.gpu_ids.size());
    for (size_t i = 0; i < config.gpu_ids.size(); ++i) {
        contexts[i].deviceId = config.gpu_ids[i];
        if (!init_gpu_context(contexts[i], config.batch_size, config.batches_per_sm)) {
            std::cerr << "[Client] GPU " << config.gpu_ids[i] << " init failed\n";
            return 1;
        }
        std::cout << "[Client] GPU " << contexts[i].deviceId << ": " << contexts[i].prop.name << "\n";
    }
    std::cout << "\n";

    while (g_running.load()) {
        if (!connect_to_server(config)) {
            std::cout << "[Client] Retry in " << config.reconnect_delay_sec << "s...\n";
            std::this_thread::sleep_for(std::chrono::seconds(config.reconnect_delay_sec));
            continue;
        }

        if (!register_with_server(config, contexts)) {
            net::close_socket(g_server_socket);
            g_server_socket = INVALID_SOCKET_VALUE;
            std::this_thread::sleep_for(std::chrono::seconds(config.reconnect_delay_sec));
            continue;
        }

        while (g_running.load()) {
            WorkAssignmentMsg work;
            if (!request_work(work)) {
                if (!g_running.load()) break;
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }

            g_found_global.store(false);
            g_found_by_gpu.store(-1);

            execute_work_unit(contexts, work, config);

            if (g_found_global.load()) {
                g_running.store(false);
                break;
            }
        }

        DisconnectMsg disc;
        disc.client_id = g_client_id;
        send_to_server(MessageType::DISCONNECT, &disc, sizeof(disc));
        net::close_socket(g_server_socket);
        g_server_socket = INVALID_SOCKET_VALUE;

        if (!g_running.load()) break;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for (auto& ctx : contexts) cleanup_gpu_context(ctx);
    net::cleanup();

    std::cout << "[Client] Shutdown complete\n";
    return 0;
}
