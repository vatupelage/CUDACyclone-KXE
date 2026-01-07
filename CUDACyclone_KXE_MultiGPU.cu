// ============================================================================
// CUDACyclone KXE Mode - Multi-GPU Implementation
// ============================================================================
// Extends single-GPU KXE with multi-GPU support:
// - Each GPU gets a unique stream_id (disjoint by construction)
// - Near-linear scaling with multiple GPUs
// - Checkpoint stores (stream_id, block_counter) per GPU
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
#include <fstream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <csignal>
#include <vector>
#include <mutex>

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"
#include "kxe/KXEPermutation.cuh"

// ============================================================================
// CONFIGURATION
// ============================================================================

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#define MAX_GPUS 8

// ============================================================================
// CONSTANT MEMORY (per GPU)
// ============================================================================

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];
__constant__ uint64_t c_range_start[4];
__constant__ uint64_t c_range_width[4];

// ============================================================================
// GLOBAL STATE
// ============================================================================

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

static std::atomic<bool> g_found_global{false};
static std::atomic<int> g_found_by_gpu{-1};
static std::mutex g_result_mutex;
static FoundResult g_global_result;

// ============================================================================
// GPU CONTEXT
// ============================================================================

struct GPUContext {
    int deviceId;
    cudaDeviceProp prop;
    cudaStream_t stream;

    // Device memory
    uint64_t* d_scalars;
    uint64_t* d_Px;
    uint64_t* d_Py;
    uint64_t* d_Rx;
    uint64_t* d_Ry;
    uint64_t* d_counts;
    int* d_found_flag;
    FoundResult* d_found_result;
    unsigned long long* d_hashes_accum;
    unsigned int* d_any_left;

    // Execution parameters
    uint64_t threadsTotal;
    int blocks;
    int threadsPerBlock;

    // KXE state
    uint32_t stream_id;
    uint64_t block_counter;

    // Runtime state
    std::atomic<bool> completed{false};
    std::atomic<bool> found_match{false};
    unsigned long long last_hashes{0};

    GPUContext() : deviceId(-1), stream(nullptr),
                   d_scalars(nullptr), d_Px(nullptr), d_Py(nullptr),
                   d_Rx(nullptr), d_Ry(nullptr), d_counts(nullptr),
                   d_found_flag(nullptr), d_found_result(nullptr),
                   d_hashes_accum(nullptr), d_any_left(nullptr),
                   threadsTotal(0), blocks(0), threadsPerBlock(256),
                   stream_id(0), block_counter(0) {}
};

// ============================================================================
// DEVICE HELPER FUNCTIONS
// ============================================================================

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}

__device__ __forceinline__ bool warp_found_ready(
    const int* __restrict__ d_found_flag,
    unsigned full_mask, unsigned lane)
{
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

// ============================================================================
// SCALAR INITIALIZATION KERNEL
// ============================================================================

__global__ void kernel_kxe_init_scalars(
    uint64_t* __restrict__ scalars,
    uint64_t* __restrict__ counts,
    uint32_t stream_id,
    uint64_t block_index,
    uint64_t keys_per_block,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint64_t batches_per_thread,
    uint32_t num_streams
)
{
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    uint64_t thread_work_size = batches_per_thread * batch_size;
    uint64_t thread_offset_in_block = gid * thread_work_size;
    uint64_t absolute_offset = block_index * keys_per_block + thread_offset_in_block;

    uint64_t range_width_64 = c_range_width[0];
    bool use_64bit = (c_range_width[1] == 0 && c_range_width[2] == 0 && c_range_width[3] == 0);

    if (use_64bit && absolute_offset >= range_width_64) {
        counts[gid] = 0;
        for (int i = 0; i < 4; ++i) scalars[gid*4+i] = 0;
        return;
    }

    uint64_t scalar[4];
    scalar[0] = c_range_start[0];
    scalar[1] = c_range_start[1];
    scalar[2] = c_range_start[2];
    scalar[3] = c_range_start[3];

    uint64_t offset = absolute_offset + batch_size / 2;
    uint64_t carry = offset;
    for (int k = 0; k < 4 && carry; ++k) {
        uint64_t old = scalar[k];
        scalar[k] = old + carry;
        carry = (scalar[k] < old) ? 1 : 0;
    }

    uint64_t actual_batches = batches_per_thread;
    if (use_64bit) {
        uint64_t keys_this_thread = actual_batches * batch_size;
        if (absolute_offset + keys_this_thread > range_width_64) {
            uint64_t keys_remaining = range_width_64 - absolute_offset;
            actual_batches = keys_remaining / batch_size;
            if (actual_batches == 0) actual_batches = 1;
        }
    }

    for (int i = 0; i < 4; ++i) {
        scalars[gid*4+i] = scalar[i];
    }
    counts[gid] = actual_batches;
}

// ============================================================================
// SEARCH KERNEL
// ============================================================================

__launch_bounds__(256, 2)
__global__ void kernel_kxe_search(
    uint64_t* __restrict__ Px,
    uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ scalars,
    uint64_t* __restrict__ counts,
    uint32_t stream_id,
    uint64_t base_counter,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint32_t num_batches_per_thread,
    uint32_t num_streams,
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

    const unsigned lane = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
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
        S[i] = scalars[idx];
    }

    uint64_t remaining_batches = counts[gid];
    if (remaining_batches == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES();
        return;
    }

    uint32_t batches_done = 0;

    while (batches_done < num_batches_per_thread && remaining_batches > 0) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // Check initial point
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
                        d_found_result->iter = 0;
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

        // Batch processing (same as single-GPU version)
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

        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            // Positive and negative branches
            for (int sign = 0; sign < 2; ++sign) {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
                #pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                if (sign == 1) ModNeg256(py_i, py_i);

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
                            uint64_t delta = (uint64_t)(i+1);
                            if (sign == 0) {
                                for (int k=0;k<4 && delta;++k){ uint64_t old=fs[k]; fs[k]=old+delta; delta=(fs[k]<old)?1ull:0ull; }
                            } else {
                                for (int k=0;k<4 && delta;++k){ uint64_t old=fs[k]; fs[k]=old-delta; delta=(old<delta)?1ull:0ull; }
                            }
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            uint64_t gxmi[4];
            #pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        // Last point (half-1) - negative only
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

        // Update scalar
        {
            uint64_t addv = (uint64_t)B;
            for (int k=0;k<4 && addv;++k){ uint64_t old=S[k]; S[k]=old+addv; addv=(S[k]<old)?1ull:0ull; }
        }

        --remaining_batches;
        ++batches_done;
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        scalars[gid*4+i] = S[i];
    }
    counts[gid] = remaining_batches;

    if (remaining_batches > 0) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

// ============================================================================
// EXTERNAL DECLARATIONS
// ============================================================================

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern bool decode_p2pkh_address(const std::string& addr, uint8_t out20[20]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

// ============================================================================
// GPU INITIALIZATION
// ============================================================================

bool init_gpu(GPUContext& ctx, int deviceId, uint32_t batch_size, uint32_t batches_per_sm) {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        std::cerr << "GPU " << deviceId << " set failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    ctx.deviceId = deviceId;
    cudaGetDeviceProperties(&ctx.prop, deviceId);
    cudaStreamCreate(&ctx.stream);

    ctx.threadsPerBlock = 256;
    ctx.blocks = ctx.prop.multiProcessorCount * batches_per_sm;
    ctx.threadsTotal = (uint64_t)ctx.blocks * ctx.threadsPerBlock;

    size_t sz4 = ctx.threadsTotal * 4 * sizeof(uint64_t);

    cudaMalloc(&ctx.d_scalars, sz4);
    cudaMalloc(&ctx.d_Px, sz4);
    cudaMalloc(&ctx.d_Py, sz4);
    cudaMalloc(&ctx.d_Rx, sz4);
    cudaMalloc(&ctx.d_Ry, sz4);
    cudaMalloc(&ctx.d_counts, ctx.threadsTotal * sizeof(uint64_t));
    cudaMalloc(&ctx.d_found_flag, sizeof(int));
    cudaMalloc(&ctx.d_found_result, sizeof(FoundResult));
    cudaMalloc(&ctx.d_hashes_accum, sizeof(unsigned long long));
    cudaMalloc(&ctx.d_any_left, sizeof(unsigned int));

    int zero = FOUND_NONE;
    unsigned long long zero64 = 0;
    cudaMemcpy(ctx.d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    return true;
}

void cleanup_gpu(GPUContext& ctx) {
    if (ctx.deviceId < 0) return;
    cudaSetDevice(ctx.deviceId);
    if (ctx.stream) cudaStreamDestroy(ctx.stream);
    if (ctx.d_scalars) cudaFree(ctx.d_scalars);
    if (ctx.d_Px) cudaFree(ctx.d_Px);
    if (ctx.d_Py) cudaFree(ctx.d_Py);
    if (ctx.d_Rx) cudaFree(ctx.d_Rx);
    if (ctx.d_Ry) cudaFree(ctx.d_Ry);
    if (ctx.d_counts) cudaFree(ctx.d_counts);
    if (ctx.d_found_flag) cudaFree(ctx.d_found_flag);
    if (ctx.d_found_result) cudaFree(ctx.d_found_result);
    if (ctx.d_hashes_accum) cudaFree(ctx.d_hashes_accum);
    if (ctx.d_any_left) cudaFree(ctx.d_any_left);
    ctx.deviceId = -1;
}

// ============================================================================
// GPU WORKER THREAD
// ============================================================================

void gpu_worker(GPUContext& ctx, uint64_t keys_per_block, uint64_t total_blocks,
                uint32_t batch_size, uint64_t batches_per_thread, uint32_t slices,
                uint32_t num_streams, uint8_t* target_hash160,
                const uint64_t* range_start, const uint64_t* range_width)
{
    cudaSetDevice(ctx.deviceId);

    // Set target hash
    uint32_t prefix_le = (uint32_t)target_hash160[0] | ((uint32_t)target_hash160[1] << 8) |
                         ((uint32_t)target_hash160[2] << 16) | ((uint32_t)target_hash160[3] << 24);
    cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
    cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    cudaMemcpyToSymbol(c_range_start, range_start, 32);
    cudaMemcpyToSymbol(c_range_width, range_width, 32);

    // Precompute batch points (on this GPU)
    {
        uint32_t half = batch_size / 2;
        std::vector<uint64_t> h_scalars(half * 4, 0);
        for (uint32_t k = 0; k < half; ++k) h_scalars[k*4] = k + 1;

        uint64_t *d_s, *d_gx, *d_gy;
        cudaMalloc(&d_s, half * 4 * sizeof(uint64_t));
        cudaMalloc(&d_gx, half * 4 * sizeof(uint64_t));
        cudaMalloc(&d_gy, half * 4 * sizeof(uint64_t));
        cudaMemcpy(d_s, h_scalars.data(), half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

        scalarMulKernelBase<<<(half + 255) / 256, 256>>>(d_s, d_gx, d_gy, half);
        cudaDeviceSynchronize();

        std::vector<uint64_t> h_Gx(half * 4), h_Gy(half * 4);
        cudaMemcpy(h_Gx.data(), d_gx, half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Gy.data(), d_gy, half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpyToSymbol(c_Gx, h_Gx.data(), half * 4 * sizeof(uint64_t));
        cudaMemcpyToSymbol(c_Gy, h_Gy.data(), half * 4 * sizeof(uint64_t));

        cudaFree(d_s); cudaFree(d_gx); cudaFree(d_gy);
    }

    // Precompute jump point
    {
        uint64_t h_scalarB[4] = {batch_size, 0, 0, 0};
        uint64_t *d_sB, *d_jx, *d_jy;
        cudaMalloc(&d_sB, 32); cudaMalloc(&d_jx, 32); cudaMalloc(&d_jy, 32);
        cudaMemcpy(d_sB, h_scalarB, 32, cudaMemcpyHostToDevice);
        scalarMulKernelBase<<<1, 1>>>(d_sB, d_jx, d_jy, 1);
        cudaDeviceSynchronize();
        uint64_t hJx[4], hJy[4];
        cudaMemcpy(hJx, d_jx, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(hJy, d_jy, 32, cudaMemcpyDeviceToHost);
        cudaMemcpyToSymbol(c_Jx, hJx, 32);
        cudaMemcpyToSymbol(c_Jy, hJy, 32);
        cudaFree(d_sB); cudaFree(d_jx); cudaFree(d_jy);
    }

    // Main loop - process blocks assigned to this GPU's stream
    while (!g_sigint && !g_found_global.load() && ctx.block_counter < total_blocks) {
        // Compute permuted block for this stream
        uint64_t permuted_block = kxe_permute_in_range_64(ctx.block_counter, ctx.stream_id, total_blocks);

        // Initialize scalars
        kernel_kxe_init_scalars<<<ctx.blocks, ctx.threadsPerBlock, 0, ctx.stream>>>(
            ctx.d_scalars, ctx.d_counts, ctx.stream_id, permuted_block,
            keys_per_block, ctx.threadsTotal, batch_size, batches_per_thread, num_streams
        );
        cudaStreamSynchronize(ctx.stream);

        // Compute initial points
        scalarMulKernelBase<<<ctx.blocks, ctx.threadsPerBlock, 0, ctx.stream>>>(
            ctx.d_scalars, ctx.d_Px, ctx.d_Py, ctx.threadsTotal
        );
        cudaStreamSynchronize(ctx.stream);

        // Process block
        bool work_remaining = true;
        while (work_remaining && !g_sigint && !g_found_global.load()) {
            unsigned int zeroU = 0;
            cudaMemcpyAsync(ctx.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, ctx.stream);

            kernel_kxe_search<<<ctx.blocks, ctx.threadsPerBlock, 0, ctx.stream>>>(
                ctx.d_Px, ctx.d_Py, ctx.d_Rx, ctx.d_Ry, ctx.d_scalars, ctx.d_counts,
                ctx.stream_id, ctx.block_counter, ctx.threadsTotal, batch_size, slices,
                num_streams, ctx.d_found_flag, ctx.d_found_result,
                ctx.d_hashes_accum, ctx.d_any_left
            );
            cudaStreamSynchronize(ctx.stream);

            // Check for match
            int host_found = 0;
            cudaMemcpy(&host_found, ctx.d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
            if (host_found == FOUND_READY) {
                int expected = -1;
                if (g_found_by_gpu.compare_exchange_strong(expected, ctx.deviceId)) {
                    g_found_global.store(true);
                    FoundResult result;
                    cudaMemcpy(&result, ctx.d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
                    std::lock_guard<std::mutex> lock(g_result_mutex);
                    g_global_result = result;
                    ctx.found_match.store(true);
                }
                break;
            }

            // Check work remaining
            unsigned int any = 0;
            cudaMemcpy(&any, ctx.d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            work_remaining = (any > 0);

            std::swap(ctx.d_Px, ctx.d_Rx);
            std::swap(ctx.d_Py, ctx.d_Ry);
        }

        // Advance to next block for this stream
        ctx.block_counter += num_streams;
    }

    ctx.completed.store(true);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string target_hash_hex, range_hex, address_b58;
    uint32_t batch_size = 128;
    uint32_t batches_per_sm = 128;
    uint32_t slices = 64;
    std::vector<int> gpu_ids;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--target-hash160" && i + 1 < argc) target_hash_hex = argv[++i];
        else if (arg == "--address" && i + 1 < argc) address_b58 = argv[++i];
        else if (arg == "--range" && i + 1 < argc) range_hex = argv[++i];
        else if (arg == "--grid" && i + 1 < argc) {
            std::string grid = argv[++i];
            size_t comma = grid.find(',');
            if (comma != std::string::npos) {
                batch_size = std::stoul(grid.substr(0, comma));
                batches_per_sm = std::stoul(grid.substr(comma + 1));
            }
        }
        else if (arg == "--slices" && i + 1 < argc) slices = std::stoul(argv[++i]);
        else if (arg == "--gpus" && i + 1 < argc) {
            std::string gpus_str = argv[++i];
            std::istringstream iss(gpus_str);
            std::string token;
            while (std::getline(iss, token, ',')) {
                gpu_ids.push_back(std::stoi(token));
            }
        }
    }

    if (gpu_ids.empty()) {
        int count;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count && i < MAX_GPUS; ++i) gpu_ids.push_back(i);
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start:end> (--address <P2PKH> | --target-hash160 <hex>)\n"
                  << "  [--grid A,B] [--slices N] [--gpus 0,1,2,...]\n";
        return EXIT_FAILURE;
    }

    // Parse range
    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) {
        std::cerr << "Error: range format must be start:end\n";
        return EXIT_FAILURE;
    }

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(range_hex.substr(0, colon_pos), range_start) ||
        !hexToLE64(range_hex.substr(colon_pos + 1), range_end)) {
        std::cerr << "Error: invalid range hex\n";
        return EXIT_FAILURE;
    }

    uint8_t target_hash160[20];
    if (!address_b58.empty()) {
        if (!decode_p2pkh_address(address_b58, target_hash160)) {
            std::cerr << "Error: invalid P2PKH address\n";
            return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: invalid target hash160 hex\n";
            return EXIT_FAILURE;
        }
    }

    uint64_t range_width[4];
    sub256(range_end, range_start, range_width);
    add256_u64(range_width, 1, range_width);

    int num_gpus = (int)gpu_ids.size();

    std::cout << "\n";
    std::cout << "======== CUDACyclone KXE Multi-GPU Mode ===============\n";
    std::cout << "GPUs: " << num_gpus << "\n";
    std::cout << "Range: " << formatHex256(range_start) << " : " << formatHex256(range_end) << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "Slices: " << slices << "\n";
    std::cout << "-------------------------------------------------------\n\n";

    // Initialize GPUs
    std::vector<GPUContext> contexts(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
        if (!init_gpu(contexts[i], gpu_ids[i], batch_size, batches_per_sm)) {
            std::cerr << "Failed to init GPU " << gpu_ids[i] << "\n";
            return EXIT_FAILURE;
        }
        contexts[i].stream_id = i;
        contexts[i].block_counter = i;  // Staggered start
        std::cout << "GPU " << i << ": " << contexts[i].prop.name << " (stream " << i << ")\n";
    }

    // Calculate block parameters
    uint64_t batches_per_thread = 10;
    uint64_t keys_per_block = contexts[0].threadsTotal * batches_per_thread * batch_size;
    uint64_t range_width_64 = range_width[0];
    uint64_t total_blocks = (range_width_64 + keys_per_block - 1) / keys_per_block;

    std::cout << "\nKeys per block: " << keys_per_block << "\n";
    std::cout << "Total blocks: " << total_blocks << "\n";
    std::cout << "Blocks per GPU: ~" << (total_blocks + num_gpus - 1) / num_gpus << "\n\n";

    // Launch workers
    std::vector<std::thread> workers;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_gpus; ++i) {
        workers.emplace_back([&, i]() {
            gpu_worker(contexts[i], keys_per_block, total_blocks,
                       batch_size, batches_per_thread, slices, num_gpus,
                       target_hash160, range_start, range_width);
        });
    }

    // Monitor progress
    unsigned long long lastTotal = 0;
    auto tLast = t0;
    while (!g_sigint && !g_found_global.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        bool all_done = true;
        unsigned long long total = 0;
        uint64_t min_block = total_blocks;

        for (auto& ctx : contexts) {
            if (!ctx.completed.load()) all_done = false;
            cudaSetDevice(ctx.deviceId);
            unsigned long long h = 0;
            cudaMemcpy(&h, ctx.d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            total += h;
            if (ctx.block_counter < min_block) min_block = ctx.block_counter;
        }

        if (all_done) break;

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        double speed = (dt > 0) ? ((total - lastTotal) / dt / 1e9) : 0;
        double progress = 100.0 * min_block / total_blocks;

        std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                  << "s | Speed: " << std::setprecision(2) << speed
                  << " Gkeys/s | Progress: " << std::setprecision(1) << progress << "%   ";
        std::cout.flush();

        lastTotal = total;
        tLast = now;
    }

    for (auto& t : workers) t.join();

    std::cout << "\n\n";

    if (g_found_global.load()) {
        std::cout << "======== FOUND MATCH! =================================\n";
        std::cout << "Private Key: " << formatHex256(g_global_result.scalar) << "\n";
        std::cout << "Public Key: " << formatCompressedPubHex(g_global_result.Rx, g_global_result.Ry) << "\n";
        std::cout << "Found by GPU: " << g_found_by_gpu.load() << "\n";
    } else if (g_sigint) {
        std::cout << "======== INTERRUPTED =================================\n";
    } else {
        std::cout << "======== SEARCH COMPLETE ==============================\n";
        std::cout << "Key NOT found in range.\n";
    }

    for (auto& ctx : contexts) cleanup_gpu(ctx);
    return g_found_global.load() ? EXIT_SUCCESS : EXIT_FAILURE;
}
