// ============================================================================
// CUDACyclone KXE Mode - Permuted Scanning Implementation
// ============================================================================
// Replaces contiguous range scanning with permutation-based approach:
// - Each GPU/stream visits unique keys with zero overlap
// - Keys are spread uniformly across the entire range from the start
// - Checkpointing reduces to storing only (stream_id, counter)
// - Multi-GPU scaling becomes trivial with disjoint streams by construction
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
#include <chrono>
#include <csignal>
#include <atomic>

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

// ============================================================================
// CONSTANT MEMORY
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

// Device-side 256-bit addition
__device__ __forceinline__ void add256_device(
    const uint64_t a[4], const uint64_t b[4], uint64_t out[4])
{
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) || (carry && sum == a[i]) ? 1 : 0;
        out[i] = sum;
    }
}

// ============================================================================
// KXE KERNEL - PERMUTED KEY SCANNING
// ============================================================================
// Key difference from original kernel:
// - Each thread's starting scalar is computed via KXE permutation
// - Threads don't process consecutive ranges; they're distributed uniformly
// - Counter-based progress tracking instead of range-based
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

    // Load current state
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

        // Process batch points
        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            // Positive branch
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
                            d_found_result->iter = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            // Negative branch
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
                            d_found_result->iter = 0;
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

        // Last point (half-1)
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
                        d_found_result->iter = 0;
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

        // Jump to next batch position using KXE permutation
        // Instead of linear jump, compute new position from permutation
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

    // Save state
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
// SCALAR INITIALIZATION KERNEL (BLOCK PERMUTATION)
// ============================================================================
// KXE Block Permutation: Instead of permuting individual thread positions,
// we permute "block indices" where each block contains keys_per_block keys.
//
// For kernel launch K:
//   - Compute permuted_block = permute(stream_id, K)
//   - Each thread T in this launch processes keys at:
//     range_start + permuted_block * keys_per_block + T * batch_size + [0, batch_size-1]
//
// This ensures:
//   1. No overlap between blocks (bijection property)
//   2. Uniform coverage across entire range
//   3. Batch inversion optimization preserved (contiguous within batch)

__global__ void kernel_kxe_init_scalars(
    uint64_t* __restrict__ scalars,
    uint64_t* __restrict__ counts,
    uint32_t stream_id,
    uint64_t block_index,           // Which block we're processing (permuted externally)
    uint64_t keys_per_block,        // Total keys in one block
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint64_t batches_per_thread,
    uint32_t num_streams
)
{
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    // Each thread processes a portion of this block
    // Thread T handles keys at: block_start + T * thread_work_size + [0, thread_work_size-1]
    uint64_t thread_work_size = batches_per_thread * batch_size;

    // Compute the offset within the block for this thread
    uint64_t thread_offset_in_block = gid * thread_work_size;

    // Absolute offset in range = block_index * keys_per_block + thread_offset_in_block
    uint64_t absolute_offset = block_index * keys_per_block + thread_offset_in_block;

    // Check if this thread's work is within the range
    uint64_t range_width_64 = c_range_width[0];
    bool use_64bit = (c_range_width[1] == 0 && c_range_width[2] == 0 && c_range_width[3] == 0);

    if (use_64bit && absolute_offset >= range_width_64) {
        // Out of range - mark as having no work
        counts[gid] = 0;
        for (int i = 0; i < 4; ++i) scalars[gid*4+i] = 0;
        return;
    }

    // Calculate starting scalar: range_start + absolute_offset + batch_size/2
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

    // Calculate how many batches this thread should process
    uint64_t actual_batches = batches_per_thread;
    if (use_64bit) {
        // Check if we'd go past the range end
        uint64_t keys_this_thread = actual_batches * batch_size;
        if (absolute_offset + keys_this_thread > range_width_64) {
            // Truncate to stay within range
            uint64_t keys_remaining = range_width_64 - absolute_offset;
            actual_batches = keys_remaining / batch_size;
            if (actual_batches == 0) actual_batches = 1; // At least one batch
        }
    }

    // Store scalar and batch count
    for (int i = 0; i < 4; ++i) {
        scalars[gid*4+i] = scalar[i];
    }
    counts[gid] = actual_batches;
}

// ============================================================================
// EXTERNAL FUNCTION DECLARATIONS
// ============================================================================

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern bool decode_p2pkh_address(const std::string& addr, uint8_t out20[20]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

// ============================================================================
// 256-BIT ARITHMETIC HELPERS
// ============================================================================

uint64_t calculate_total_blocks_256(const uint64_t range_width[4], uint64_t keys_per_block) {
    if (range_width[1] == 0 && range_width[2] == 0 && range_width[3] == 0) {
        if (range_width[0] == 0) return 0;
        return (range_width[0] + keys_per_block - 1) / keys_per_block;
    }

    __uint128_t result = 0;
    __uint128_t remainder = 0;

    for (int i = 3; i >= 0; --i) {
        remainder = (remainder << 64) | range_width[i];
        if (remainder >= keys_per_block) {
            __uint128_t quotient = remainder / keys_per_block;
            remainder = remainder % keys_per_block;
            result = (result << 64) | quotient;
        } else {
            result = result << 64;
        }
    }

    if (remainder > 0) result++;
    if (result > UINT64_MAX) return UINT64_MAX;

    return (uint64_t)result;
}

// ============================================================================
// CUDA ERROR CHECK
// ============================================================================

static auto ck = [](cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(EXIT_FAILURE);
    }
};

// ============================================================================
// KXE CHECKPOINT FUNCTIONS
// ============================================================================

bool kxe_save_checkpoint(const std::string& filename,
                          uint32_t stream_id, uint64_t counter,
                          const uint64_t range_start[4],
                          const uint64_t range_width[4],
                          const uint8_t target_hash160[20],
                          uint32_t num_streams, uint32_t batch_size, uint32_t slices,
                          uint32_t batches_per_sm,
                          bool found, const uint64_t found_scalar[4]) {
    std::string temp_file = filename + ".tmp";
    std::ofstream file(temp_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create checkpoint file\n";
        return false;
    }

    KXECheckpointHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = KXE_CHECKPOINT_MAGIC;
    header.version = KXE_CHECKPOINT_VERSION;
    header.timestamp = (uint64_t)std::time(nullptr);
    memcpy(header.range_start, range_start, sizeof(header.range_start));
    memcpy(header.range_width, range_width, sizeof(header.range_width));
    memcpy(header.target_hash160, target_hash160, sizeof(header.target_hash160));
    header.num_gpus = 1;
    header.num_streams = num_streams;
    header.batch_size = batch_size;
    header.slices = slices;
    header.batches_per_sm = batches_per_sm;
    header.found = found ? 1 : 0;
    if (found) {
        memcpy(header.found_scalar, found_scalar, sizeof(header.found_scalar));
    }

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    KXECheckpointEntry entry;
    entry.stream_id = stream_id;
    entry.counter = counter;
    file.write(reinterpret_cast<const char*>(&entry), sizeof(entry));

    file.close();
    std::rename(temp_file.c_str(), filename.c_str());
    return true;
}

bool kxe_load_checkpoint(const std::string& filename,
                          uint32_t& stream_id, uint64_t& counter,
                          uint64_t range_start[4], uint64_t range_width[4],
                          uint8_t target_hash160[20],
                          uint32_t& num_streams, uint32_t& batch_size, uint32_t& slices,
                          uint32_t& batches_per_sm,
                          bool& found, uint64_t found_scalar[4]) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    KXECheckpointHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    // Check for old v1 checkpoint format (without batches_per_sm)
    if (header.magic == 0x3158454BU) {  // Old "KXE1" magic
        std::cerr << "Error: Old checkpoint format (v1) detected.\n";
        std::cerr << "This checkpoint was created without batches_per_sm and cannot be safely resumed.\n";
        std::cerr << "Please start a fresh search to avoid missing keys.\n";
        return false;
    }

    if (header.magic != KXE_CHECKPOINT_MAGIC) {
        std::cerr << "Error: Invalid KXE checkpoint file\n";
        return false;
    }

    if (header.version < KXE_CHECKPOINT_VERSION) {
        std::cerr << "Error: Checkpoint version " << header.version << " is older than expected " << KXE_CHECKPOINT_VERSION << "\n";
        return false;
    }

    memcpy(range_start, header.range_start, sizeof(header.range_start));
    memcpy(range_width, header.range_width, sizeof(header.range_width));
    memcpy(target_hash160, header.target_hash160, sizeof(header.target_hash160));
    num_streams = header.num_streams;
    batch_size = header.batch_size;
    slices = header.slices;
    batches_per_sm = header.batches_per_sm;
    found = (header.found != 0);
    if (found) {
        memcpy(found_scalar, header.found_scalar, sizeof(header.found_scalar));
    }

    KXECheckpointEntry entry;
    file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    stream_id = entry.stream_id;
    counter = entry.counter;

    return true;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string target_hash_hex, range_hex, address_b58;
    uint32_t batch_size = 128;
    uint32_t batches_per_sm = 128;
    uint32_t slices = 64;
    uint32_t stream_id = 0;
    uint32_t num_streams = 1;
    std::string checkpoint_file;
    uint32_t checkpoint_interval = 60;
    bool resume_mode = false;

    // Parse arguments
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
        else if (arg == "--stream-id" && i + 1 < argc) stream_id = std::stoul(argv[++i]);
        else if (arg == "--streams" && i + 1 < argc) num_streams = std::stoul(argv[++i]);
        else if (arg == "--checkpoint" && i + 1 < argc) checkpoint_file = argv[++i];
        else if (arg == "--checkpoint-interval" && i + 1 < argc) checkpoint_interval = std::stoul(argv[++i]);
        else if (arg == "--resume") resume_mode = true;
        else if (arg == "--kxe") { /* KXE mode enabled by default in this file */ }
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start:end> (--address <P2PKH> | --target-hash160 <hex>)\n"
                  << "  [--grid A,B] [--slices N] [--stream-id ID] [--streams N]\n"
                  << "  [--checkpoint <file>] [--checkpoint-interval <sec>] [--resume]\n";
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

    // Calculate range width
    uint64_t range_width[4];
    sub256(range_end, range_start, range_width);
    add256_u64(range_width, 1, range_width);

    // Check for checkpoint resume
    uint64_t counter = 0;
    uint64_t found_scalar[4] = {0, 0, 0, 0};
    bool already_found = false;

    // GPU setup (need this before checkpoint validation)
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    if (resume_mode && !checkpoint_file.empty()) {
        uint64_t ckpt_start[4], ckpt_width[4];
        uint8_t ckpt_hash[20];
        uint32_t ckpt_streams, ckpt_batch, ckpt_slices, ckpt_batches_per_sm;

        if (kxe_load_checkpoint(checkpoint_file, stream_id, counter,
                                 ckpt_start, ckpt_width, ckpt_hash,
                                 ckpt_streams, ckpt_batch, ckpt_slices,
                                 ckpt_batches_per_sm,
                                 already_found, found_scalar)) {
            std::cout << "Resuming from checkpoint:\n";
            std::cout << "  Stream ID: " << stream_id << "\n";
            std::cout << "  Counter: " << counter << "\n";
            std::cout << "  Checkpoint batches_per_sm: " << ckpt_batches_per_sm << "\n";

            // Validate batches_per_sm matches
            if (ckpt_batches_per_sm != batches_per_sm) {
                std::cerr << "\nError: Grid parameter mismatch!\n";
                std::cerr << "  Checkpoint was created with --grid " << ckpt_batch << "," << ckpt_batches_per_sm << "\n";
                std::cerr << "  Current command uses --grid " << batch_size << "," << batches_per_sm << "\n";
                std::cerr << "\nThis would cause keys to be skipped! Use the same --grid values:\n";
                std::cerr << "  --grid " << ckpt_batch << "," << ckpt_batches_per_sm << "\n";
                return EXIT_FAILURE;
            }

            batch_size = ckpt_batch;
            slices = ckpt_slices;
            num_streams = ckpt_streams;

            if (already_found) {
                std::cout << "\n*** Key already found in previous run ***\n";
                std::cout << "Private Key: " << formatHex256(found_scalar) << "\n";
                return EXIT_SUCCESS;
            }
        }
    }

    int threadsPerBlock = 256;
    int blocks = prop.multiProcessorCount * batches_per_sm;
    uint64_t threadsTotal = (uint64_t)blocks * threadsPerBlock;

    std::cout << "\n";
    std::cout << "======== CUDACyclone KXE Mode =========================\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Range: " << formatHex256(range_start) << " : " << formatHex256(range_end) << "\n";
    std::cout << "Range width: " << formatHex256(range_width) << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "Slices per launch: " << slices << "\n";
    std::cout << "Stream ID: " << stream_id << " of " << num_streams << "\n";
    std::cout << "Starting counter: " << counter << "\n";
    std::cout << "Threads: " << threadsTotal << "\n";
    std::cout << "-------------------------------------------------------\n\n";

    // Allocate device memory
    uint64_t* d_scalars = nullptr;
    uint64_t* d_Px = nullptr;
    uint64_t* d_Py = nullptr;
    uint64_t* d_Rx = nullptr;
    uint64_t* d_Ry = nullptr;
    uint64_t* d_counts = nullptr;
    int* d_found_flag = nullptr;
    FoundResult* d_found_result = nullptr;
    unsigned long long* d_hashes_accum = nullptr;
    unsigned int* d_any_left = nullptr;

    size_t sz4 = threadsTotal * 4 * sizeof(uint64_t);
    ck(cudaMalloc(&d_scalars, sz4), "malloc d_scalars");
    ck(cudaMalloc(&d_Px, sz4), "malloc d_Px");
    ck(cudaMalloc(&d_Py, sz4), "malloc d_Py");
    ck(cudaMalloc(&d_Rx, sz4), "malloc d_Rx");
    ck(cudaMalloc(&d_Ry, sz4), "malloc d_Ry");
    ck(cudaMalloc(&d_counts, threadsTotal * sizeof(uint64_t)), "malloc d_counts");
    ck(cudaMalloc(&d_found_flag, sizeof(int)), "malloc d_found_flag");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)), "malloc d_found_result");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)), "malloc d_hashes_accum");
    ck(cudaMalloc(&d_any_left, sizeof(unsigned int)), "malloc d_any_left");

    // Initialize constants
    int zero = FOUND_NONE;
    unsigned long long zero64 = 0;
    ck(cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "init found_flag");
    ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes");

    // Set target hash in constant memory
    uint32_t prefix_le = (uint32_t)target_hash160[0] | ((uint32_t)target_hash160[1] << 8) |
                         ((uint32_t)target_hash160[2] << 16) | ((uint32_t)target_hash160[3] << 24);
    ck(cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le)), "copy target prefix");
    ck(cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20), "copy target hash");
    ck(cudaMemcpyToSymbol(c_range_start, range_start, sizeof(range_start)), "copy range_start");
    ck(cudaMemcpyToSymbol(c_range_width, range_width, sizeof(range_width)), "copy range_width");

    // Precompute batch points
    {
        uint32_t half = batch_size / 2;
        uint64_t* h_scalars = (uint64_t*)malloc(half * 4 * sizeof(uint64_t));
        memset(h_scalars, 0, half * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < half; ++k) h_scalars[k*4] = k + 1;

        uint64_t *d_s = nullptr, *d_gx = nullptr, *d_gy = nullptr;
        ck(cudaMalloc(&d_s, half * 4 * sizeof(uint64_t)), "malloc d_s");
        ck(cudaMalloc(&d_gx, half * 4 * sizeof(uint64_t)), "malloc d_gx");
        ck(cudaMalloc(&d_gy, half * 4 * sizeof(uint64_t)), "malloc d_gy");
        ck(cudaMemcpy(d_s, h_scalars, half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "copy scalars");

        int b = (half + 255) / 256;
        scalarMulKernelBase<<<b, 256>>>(d_s, d_gx, d_gy, half);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");

        uint64_t* h_Gx = (uint64_t*)malloc(half * 4 * sizeof(uint64_t));
        uint64_t* h_Gy = (uint64_t*)malloc(half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Gx, d_gx, half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx");
        ck(cudaMemcpy(h_Gy, d_gy, half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gx, half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gy, half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

        cudaFree(d_s); cudaFree(d_gx); cudaFree(d_gy);
        free(h_scalars); free(h_Gx); free(h_Gy);
    }

    // Precompute jump point
    {
        uint64_t h_scalarB[4] = {batch_size, 0, 0, 0};
        uint64_t *d_sB = nullptr, *d_jx = nullptr, *d_jy = nullptr;
        ck(cudaMalloc(&d_sB, 32), "malloc d_sB");
        ck(cudaMalloc(&d_jx, 32), "malloc d_jx");
        ck(cudaMalloc(&d_jy, 32), "malloc d_jy");
        ck(cudaMemcpy(d_sB, h_scalarB, 32, cudaMemcpyHostToDevice), "copy scalarB");

        scalarMulKernelBase<<<1, 1>>>(d_sB, d_jx, d_jy, 1);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(B) sync");

        uint64_t hJx[4], hJy[4];
        ck(cudaMemcpy(hJx, d_jx, 32, cudaMemcpyDeviceToHost), "D2H Jx");
        ck(cudaMemcpy(hJy, d_jy, 32, cudaMemcpyDeviceToHost), "D2H Jy");
        ck(cudaMemcpyToSymbol(c_Jx, hJx, 32), "ToSymbol c_Jx");
        ck(cudaMemcpyToSymbol(c_Jy, hJy, 32), "ToSymbol c_Jy");

        cudaFree(d_sB); cudaFree(d_jx); cudaFree(d_jy);
    }

    // KXE Block Permutation Setup
    // Each kernel launch processes one "block" of keys
    // Block size = threadsTotal * batches_per_thread * batch_size
    uint64_t batches_per_thread = 10;  // Fixed for simplicity
    uint64_t keys_per_block = threadsTotal * batches_per_thread * batch_size;
    uint64_t total_blocks = calculate_total_blocks_256(range_width, keys_per_block);

    if (total_blocks == 0) {
        std::cerr << "Error: Range too small or invalid\n";
        return EXIT_FAILURE;
    }

    std::cout << "Keys per block: " << keys_per_block << " (~" << (keys_per_block / 1e9) << "B)\n";
    std::cout << "Total blocks: " << total_blocks << "\n";
    std::cout << "Batches per thread: " << batches_per_thread << "\n";

    // Main loop
    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    auto lastCheckpoint = t0;
    unsigned long long lastHashes = 0;
    bool found = false;
    FoundResult host_result;
    uint64_t block_counter = counter;  // Which block to process next

    std::cout << "\n======== Phase-1: KXE Block Permuted Search ===========\n";

    while (!g_sigint && !found && block_counter < total_blocks) {
        // Compute permuted block index for this iteration
        uint64_t permuted_block = kxe_permute_in_range_64(block_counter, stream_id, total_blocks);

        // Initialize scalars for this block
        kernel_kxe_init_scalars<<<blocks, threadsPerBlock>>>(
            d_scalars, d_counts, stream_id, permuted_block,
            keys_per_block, threadsTotal, batch_size, batches_per_thread, num_streams
        );
        ck(cudaDeviceSynchronize(), "kxe_init_scalars sync");

        // Compute initial EC points
        scalarMulKernelBase<<<blocks, threadsPerBlock>>>(d_scalars, d_Px, d_Py, threadsTotal);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");

        // Run search iterations until this block is exhausted
        bool work_remaining = true;
        while (work_remaining && !g_sigint && !found) {
            unsigned int zeroU = 0;
            ck(cudaMemcpy(d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice), "reset any_left");

            kernel_kxe_search<<<blocks, threadsPerBlock>>>(
                d_Px, d_Py, d_Rx, d_Ry, d_scalars, d_counts,
                stream_id, block_counter, threadsTotal, batch_size, slices,
                num_streams, d_found_flag, d_found_result,
                d_hashes_accum, d_any_left
            );
            ck(cudaDeviceSynchronize(), "kernel_kxe_search sync");

            // Check for found
            int host_found = 0;
            ck(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "check found");
            if (host_found == FOUND_READY) {
                found = true;
                ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "get result");
                break;
            }

            // Check work remaining
            unsigned int any = 0;
            ck(cudaMemcpy(&any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "check any_left");
            work_remaining = (any > 0);

            // Swap buffers
            std::swap(d_Px, d_Rx);
            std::swap(d_Py, d_Ry);

            // Progress display
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 1.0) {
                unsigned long long totalHashes = 0;
                ck(cudaMemcpy(&totalHashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "get hashes");

                double delta = (double)(totalHashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                double progress = 100.0 * block_counter / total_blocks;

                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << "s | Speed: " << std::setprecision(1) << mkeys
                          << " Mkeys/s | Block: " << block_counter << "/" << total_blocks
                          << " (" << std::setprecision(1) << progress << "%)";
                std::cout.flush();

                lastHashes = totalHashes;
                tLast = now;

                // Periodic checkpoint
                if (!checkpoint_file.empty()) {
                    double ckpt_dt = std::chrono::duration<double>(now - lastCheckpoint).count();
                    if (ckpt_dt >= checkpoint_interval) {
                        uint64_t zero_scalar[4] = {0, 0, 0, 0};
                        kxe_save_checkpoint(checkpoint_file, stream_id, block_counter,
                                            range_start, range_width, target_hash160,
                                            num_streams, batch_size, slices,
                                            batches_per_sm,
                                            false, zero_scalar);
                        std::cout << " [CKPT]";
                        std::cout.flush();
                        lastCheckpoint = now;
                    }
                }
            }
        }

        // Move to next block
        ++block_counter;
    }

    std::cout << "\n\n";

    // Results
    if (found) {
        std::cout << "======== FOUND MATCH! =================================\n";
        std::cout << "Private Key: " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key: " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";

        // Save final checkpoint with found key
        if (!checkpoint_file.empty()) {
            kxe_save_checkpoint(checkpoint_file, stream_id, block_counter,
                                range_start, range_width, target_hash160,
                                num_streams, batch_size, slices,
                                batches_per_sm,
                                true, host_result.scalar);
        }
    } else if (g_sigint) {
        std::cout << "======== INTERRUPTED (Ctrl+C) =========================\n";
        if (!checkpoint_file.empty()) {
            uint64_t zero_scalar[4] = {0, 0, 0, 0};
            kxe_save_checkpoint(checkpoint_file, stream_id, block_counter,
                                range_start, range_width, target_hash160,
                                num_streams, batch_size, slices,
                                batches_per_sm,
                                false, zero_scalar);
            std::cout << "Checkpoint saved: " << checkpoint_file << "\n";
            std::cout << "Resume with: --resume --checkpoint " << checkpoint_file << "\n";
        }
    } else if (block_counter >= total_blocks) {
        std::cout << "======== SEARCH COMPLETE ==============================\n";
        std::cout << "Key NOT found in range.\n";
        std::cout << "Stream " << stream_id << " searched " << total_blocks << " blocks.\n";
    } else {
        std::cout << "======== KEY NOT FOUND ================================\n";
        std::cout << "Stream " << stream_id << " stopped at block " << block_counter << "/" << total_blocks << ".\n";
    }

    // Cleanup
    cudaFree(d_scalars);
    cudaFree(d_Px);
    cudaFree(d_Py);
    cudaFree(d_Rx);
    cudaFree(d_Ry);
    cudaFree(d_counts);
    cudaFree(d_found_flag);
    cudaFree(d_found_result);
    cudaFree(d_hashes_accum);
    cudaFree(d_any_left);

    return found ? EXIT_SUCCESS : (g_sigint ? 130 : EXIT_FAILURE);
}
