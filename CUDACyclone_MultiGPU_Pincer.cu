
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
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

// ============================================================================
// BIDIRECTIONAL (PINCER) SCANNING MODE
// ============================================================================
//
// Pincer mode pairs GPUs to scan each partition from both ends simultaneously.
// This provides a 2x speedup on average because the key is found when either
// GPU reaches it.
//
// Example with 4 GPUs:
//
// Partition 0:                    Partition 1:
// START ──────────── MID          MID ──────────── END
//    │                │             │                │
//   GPU 0 ─────►  ◄── GPU 1       GPU 2 ─────►  ◄── GPU 3
//   (FWD)         (BWD)           (FWD)         (BWD)
//
// ============================================================================

// ============================================================================
// SCAN DIRECTION
// ============================================================================

enum ScanDirection {
    SCAN_FORWARD  = 0,   // Scan from start toward end (normal mode)
    SCAN_BACKWARD = 1    // Scan from end toward start (pincer partner)
};

// ============================================================================
// CHECKPOINT SYSTEM
// ============================================================================

#define CHECKPOINT_VERSION 2  // Incremented for pincer support
#define CHECKPOINT_MAGIC 0x43594B4C43484B50ULL  // "CLKCHKPT" in hex

struct CheckpointHeader {
    uint64_t magic;                  // Magic number for validation
    uint32_t version;                // Checkpoint format version
    uint32_t num_gpus;               // Number of GPUs in checkpoint
    uint64_t timestamp;              // Unix timestamp when saved
    uint64_t total_keys_processed;   // Total keys processed across all GPUs
    uint64_t range_start[4];         // Original range start
    uint64_t range_end[4];           // Original range end
    uint8_t target_hash160[20];      // Target hash160
    uint32_t batch_size;             // Batch size used
    uint32_t slices_per_launch;      // Slices per launch
    uint32_t pincer_mode;            // 1 if pincer mode enabled
    uint32_t num_partitions;         // Number of partitions
    uint32_t padding[1];             // Alignment padding
};

struct GPUCheckpointData {
    int32_t device_id;               // GPU device ID
    uint32_t threads_total;          // Number of threads on this GPU
    uint64_t current_position[4];    // Current scalar position (per-thread starting point)
    uint64_t keys_processed;         // Keys processed by this GPU
    uint64_t range_start[4];         // This GPU's range start
    uint64_t range_len[4];           // This GPU's range length
    uint64_t per_thread_cnt[4];      // Keys per thread
    int32_t  direction;              // ScanDirection: 0=forward, 1=backward
    int32_t  partition_id;           // Which partition this GPU belongs to
    int32_t  partner_gpu_idx;        // Index of partner GPU in same partition
    int32_t  padding;                // Alignment padding
};

// Global checkpoint state
static std::string g_checkpoint_file = "";
static std::atomic<bool> g_checkpoint_requested{false};
static std::chrono::steady_clock::time_point g_last_checkpoint_time;

// Forward declarations
struct GPUContext;
bool save_checkpoint(const std::string& filename,
                     const std::vector<GPUContext>& contexts,
                     const uint64_t range_start[4],
                     const uint64_t range_end[4],
                     const uint8_t target_hash160[20],
                     uint32_t batch_size,
                     uint32_t slices_per_launch,
                     bool pincer_mode,
                     int num_partitions,
                     unsigned long long total_keys_processed);

bool load_checkpoint(const std::string& filename,
                     CheckpointHeader& header,
                     std::vector<GPUCheckpointData>& gpu_data);

bool validate_checkpoint(const CheckpointHeader& header,
                         const uint64_t range_start[4],
                         const uint64_t range_end[4],
                         const uint8_t target_hash160[20],
                         uint32_t batch_size,
                         bool pincer_mode);

// ============================================================================
// GLOBAL COORDINATION
// ============================================================================

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

static std::atomic<bool> g_found_global{false};
static std::atomic<int>  g_found_by_gpu{-1};
static std::mutex g_result_mutex;
static FoundResult g_global_result;

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}
__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Constant memory symbols - PER-DEVICE
__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];     // Forward jump point (+B*G)
__constant__ uint64_t c_Jy[4];     // Forward jump point Y
__constant__ uint64_t c_Jy_neg[4]; // Backward jump point Y (negated for -B*G)

// ============================================================================
// GPU CONTEXT STRUCTURE (Extended for Pincer Mode)
// ============================================================================

struct GPUContext {
    int deviceId;
    cudaDeviceProp prop;
    cudaStream_t stream;

    // Device memory allocations
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
    uint64_t range_start[4];      // This GPU's starting scalar
    uint64_t range_len[4];        // This GPU's range length
    uint64_t per_thread_cnt[4];   // Keys per thread on this GPU

    // Execution parameters
    uint64_t threadsTotal;
    int blocks;
    int threadsPerBlock;

    // Pincer mode parameters
    ScanDirection direction;      // SCAN_FORWARD or SCAN_BACKWARD
    int partition_id;             // Which partition this GPU belongs to
    int partner_gpu_idx;          // Index of partner GPU in same partition

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
                   threadsTotal(0), blocks(0), threadsPerBlock(256),
                   direction(SCAN_FORWARD), partition_id(0), partner_gpu_idx(-1) {
        memset(range_start, 0, sizeof(range_start));
        memset(range_len, 0, sizeof(range_len));
        memset(per_thread_cnt, 0, sizeof(per_thread_cnt));
    }
};

// ============================================================================
// BIDIRECTIONAL KERNEL (Supports both forward and backward scanning)
// ============================================================================
//
// Key difference from original kernel:
// - Forward: Uses c_Jy for jump, adds B to scalar
// - Backward: Uses c_Jy_neg for jump (negated Y), subtracts B from scalar
//
// ============================================================================

__launch_bounds__(256, 2)
__global__ void kernel_pincer_scan(
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
    unsigned int* __restrict__ d_any_left,
    int scan_direction  // 0 = forward, 1 = backward
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

    // Select jump point Y based on direction
    uint64_t Jy_local[4];
    if (scan_direction == SCAN_BACKWARD) {
        // Backward: use negated Y for -B*G jump
#pragma unroll
        for (int j = 0; j < 4; ++j) Jy_local[j] = c_Jy_neg[j];
    } else {
        // Forward: use normal Y for +B*G jump
#pragma unroll
        for (int j = 0; j < 4; ++j) Jy_local[j] = c_Jy[j];
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // Check starting point
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

        // Process batch points (both +kG and -kG from center)
        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            // Check P + (i+1)G
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

            // Check P - (i+1)G
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

            uint64_t gxmi[4];
#pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        // Handle last point (half - 1)
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

        // Jump to next batch center point
        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = Jy_local[j];  // Use direction-specific Y
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

        // Update scalar and remaining count based on direction
        if (scan_direction == SCAN_BACKWARD) {
            // Backward: S -= B
            uint64_t sub = (uint64_t)B;
            for (int k=0;k<4 && sub;++k) {
                uint64_t old = S[k];
                S[k] = old - sub;
                sub = (old < sub) ? 1ull : 0ull;
            }
        } else {
            // Forward: S += B
            uint64_t addv = (uint64_t)B;
            for (int k=0;k<4 && addv;++k) {
                uint64_t old = S[k];
                S[k] = old + addv;
                addv = (S[k] < old) ? 1ull : 0ull;
            }
        }
        sub256_u64_inplace(rem, (uint64_t)B);
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
// EXTERNAL DECLARATIONS
// ============================================================================

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern bool decode_p2pkh_address(const std::string& addr, uint8_t out20[20]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static auto ck = [](cudaError_t e, const char* msg){
    if (e != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(EXIT_FAILURE);
    }
};

void divide_range_by_n(const uint64_t range_len[4], int n, uint64_t per_gpu[4]) {
    uint64_t r[4] = {range_len[0], range_len[1], range_len[2], range_len[3]};
    uint64_t remainder = 0;

    for (int i = 3; i >= 0; --i) {
        __uint128_t cur = ((__uint128_t)remainder << 64) | r[i];
        per_gpu[i] = (uint64_t)(cur / n);
        remainder = (uint64_t)(cur % n);
    }
}

void add_offset_256(const uint64_t base[4], const uint64_t offset[4], uint64_t result[4]) {
    __uint128_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t s = (__uint128_t)base[i] + offset[i] + carry;
        result[i] = (uint64_t)s;
        carry = s >> 64;
    }
}

void sub_256(const uint64_t a[4], const uint64_t b[4], uint64_t result[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t bi = b[i] + borrow;
        if (a[i] < bi) {
            result[i] = (uint64_t)(((__uint128_t(1) << 64) + a[i]) - bi);
            borrow = 1;
        } else {
            result[i] = a[i] - bi;
            borrow = 0;
        }
    }
}

void mult_256_by_u64(const uint64_t a[4], uint64_t b, uint64_t result[4]) {
    __uint128_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t prod = (__uint128_t)a[i] * b + carry;
        result[i] = (uint64_t)prod;
        carry = prod >> 64;
    }
}

// Compute field negation (P - Y) for negative jump point
void compute_field_neg_host(const uint64_t y[4], uint64_t neg_y[4]) {
    // P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    uint64_t P[4] = {
        0xFFFFFFFEFFFFFC2FULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    sub_256(P, y, neg_y);
}

// ============================================================================
// GPU INITIALIZATION FOR PINCER MODE
// ============================================================================

bool initialize_gpu_context_pincer(GPUContext& ctx, int deviceId,
                                   const uint64_t partition_start[4],
                                   const uint64_t partition_len[4],
                                   ScanDirection direction,
                                   int partition_id,
                                   int partner_gpu_idx,
                                   const uint8_t target_hash160[20],
                                   uint32_t runtime_points_batch_size,
                                   uint32_t runtime_batches_per_sm) {

    ctx.deviceId = deviceId;
    ctx.direction = direction;
    ctx.partition_id = partition_id;
    ctx.partner_gpu_idx = partner_gpu_idx;

    ck(cudaSetDevice(deviceId), "cudaSetDevice");
    ck(cudaGetDeviceProperties(&ctx.prop, deviceId), "cudaGetDeviceProperties");

    // Copy range length
    for (int i = 0; i < 4; ++i) ctx.range_len[i] = partition_len[i];

    // Calculate starting position based on direction
    const uint32_t B = runtime_points_batch_size;
    const uint32_t half = B >> 1;

    if (direction == SCAN_FORWARD) {
        // Forward: start at partition_start
        for (int i = 0; i < 4; ++i) ctx.range_start[i] = partition_start[i];
    } else {
        // Backward: start at partition_start + partition_len - 1 (partition end)
        // We need to compute the ending point of the partition
        uint64_t partition_end[4];
        add_offset_256(partition_start, partition_len, partition_end);
        // Subtract 1 to get the last key in partition
        uint64_t one[4] = {1, 0, 0, 0};
        sub_256(partition_end, one, ctx.range_start);
    }

    // Execution parameters
    ctx.threadsPerBlock = 256;
    if (ctx.threadsPerBlock > ctx.prop.maxThreadsPerBlock)
        ctx.threadsPerBlock = ctx.prop.maxThreadsPerBlock;

    const uint64_t bytesPerThread = 2ull * 4ull * sizeof(uint64_t);
    size_t totalGlobalMem = ctx.prop.totalGlobalMem;
    const uint64_t reserveBytes = 64ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    // Calculate total batches
    uint64_t q_div_batch[4], r_div_batch = 0ull;
    divmod_256_by_u64(ctx.range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);

    bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
    if (!q_fits_u64) {
        std::cerr << "GPU " << deviceId << ": total batches too large\n";
        return false;
    }
    uint64_t total_batches_u64 = q_div_batch[0];

    uint64_t userUpper = (uint64_t)ctx.prop.multiProcessorCount *
                         (uint64_t)runtime_batches_per_sm *
                         (uint64_t)ctx.threadsPerBlock;

    auto pick_threads_total = [&](uint64_t upper) -> uint64_t {
        if (upper < (uint64_t)ctx.threadsPerBlock) return 0ull;
        uint64_t t = upper - (upper % (uint64_t)ctx.threadsPerBlock);
        uint64_t q = total_batches_u64;
        while (t >= (uint64_t)ctx.threadsPerBlock) {
            if ((q % t) == 0ull) return t;
            t -= (uint64_t)ctx.threadsPerBlock;
        }
        return 0ull;
    };

    uint64_t upper = maxThreadsByMem;
    if (total_batches_u64 < upper) upper = total_batches_u64;
    if (userUpper < upper) upper = userUpper;

    ctx.threadsTotal = pick_threads_total(upper);
    if (ctx.threadsTotal == 0ull) {
        std::cerr << "GPU " << deviceId << ": failed to pick threadsTotal\n";
        return false;
    }
    ctx.blocks = (int)(ctx.threadsTotal / (uint64_t)ctx.threadsPerBlock);

    // Per-thread count
    uint64_t r_u64 = 0ull;
    divmod_256_by_u64(ctx.range_len, ctx.threadsTotal, ctx.per_thread_cnt, r_u64);

    // Create stream
    ck(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking), "cudaStreamCreate");

    // Allocate device memory
    ck(cudaMalloc(&ctx.d_start_scalars, ctx.threadsTotal * 4 * sizeof(uint64_t)), "malloc d_start_scalars");
    ck(cudaMalloc(&ctx.d_Px, ctx.threadsTotal * 4 * sizeof(uint64_t)), "malloc d_Px");
    ck(cudaMalloc(&ctx.d_Py, ctx.threadsTotal * 4 * sizeof(uint64_t)), "malloc d_Py");
    ck(cudaMalloc(&ctx.d_Rx, ctx.threadsTotal * 4 * sizeof(uint64_t)), "malloc d_Rx");
    ck(cudaMalloc(&ctx.d_Ry, ctx.threadsTotal * 4 * sizeof(uint64_t)), "malloc d_Ry");
    ck(cudaMalloc(&ctx.d_counts256, ctx.threadsTotal * 4 * sizeof(uint64_t)), "malloc d_counts256");
    ck(cudaMalloc(&ctx.d_found_flag, sizeof(int)), "malloc d_found_flag");
    ck(cudaMalloc(&ctx.d_found_result, sizeof(FoundResult)), "malloc d_found_result");
    ck(cudaMalloc(&ctx.d_hashes_accum, sizeof(unsigned long long)), "malloc d_hashes_accum");
    ck(cudaMalloc(&ctx.d_any_left, sizeof(unsigned int)), "malloc d_any_left");

    // Allocate host pinned memory
    ck(cudaHostAlloc(&ctx.h_counts256, ctx.threadsTotal * 4 * sizeof(uint64_t),
                     cudaHostAllocWriteCombined | cudaHostAllocMapped), "hostalloc counts");
    ck(cudaHostAlloc(&ctx.h_start_scalars, ctx.threadsTotal * 4 * sizeof(uint64_t),
                     cudaHostAllocWriteCombined | cudaHostAllocMapped), "hostalloc scalars");

    // Initialize counts and scalars
    for (uint64_t i = 0; i < ctx.threadsTotal; ++i) {
        ctx.h_counts256[i*4+0] = ctx.per_thread_cnt[0];
        ctx.h_counts256[i*4+1] = ctx.per_thread_cnt[1];
        ctx.h_counts256[i*4+2] = ctx.per_thread_cnt[2];
        ctx.h_counts256[i*4+3] = ctx.per_thread_cnt[3];
    }

    // Calculate starting scalars based on direction
    if (direction == SCAN_FORWARD) {
        // Forward: each thread starts at partition_start + thread_id * per_thread_cnt + half
        uint64_t cur[4] = {partition_start[0], partition_start[1], partition_start[2], partition_start[3]};
        for (uint64_t i = 0; i < ctx.threadsTotal; ++i) {
            uint64_t Sc[4];
            add256_u64(cur, (uint64_t)half, Sc);
            ctx.h_start_scalars[i*4+0] = Sc[0];
            ctx.h_start_scalars[i*4+1] = Sc[1];
            ctx.h_start_scalars[i*4+2] = Sc[2];
            ctx.h_start_scalars[i*4+3] = Sc[3];

            uint64_t next[4];
            add256(cur, ctx.per_thread_cnt, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
    } else {
        // Backward: each thread starts at partition_end - thread_id * per_thread_cnt - half
        // partition_end = partition_start + partition_len
        uint64_t partition_end[4];
        add_offset_256(partition_start, partition_len, partition_end);

        // Start from end - half (center of first batch from end)
        uint64_t cur[4];
        uint64_t half_arr[4] = {half, 0, 0, 0};
        sub_256(partition_end, half_arr, cur);

        for (uint64_t i = 0; i < ctx.threadsTotal; ++i) {
            ctx.h_start_scalars[i*4+0] = cur[0];
            ctx.h_start_scalars[i*4+1] = cur[1];
            ctx.h_start_scalars[i*4+2] = cur[2];
            ctx.h_start_scalars[i*4+3] = cur[3];

            // Move backward by per_thread_cnt
            uint64_t next[4];
            sub_256(cur, ctx.per_thread_cnt, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
    }

    // Upload target hash
    uint32_t prefix_le = (uint32_t)target_hash160[0]
                       | ((uint32_t)target_hash160[1] << 8)
                       | ((uint32_t)target_hash160[2] << 16)
                       | ((uint32_t)target_hash160[3] << 24);
    ck(cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le)), "copy target prefix");
    ck(cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20), "copy target hash");

    // Copy initial data to device
    ck(cudaMemcpy(ctx.d_start_scalars, ctx.h_start_scalars,
                  ctx.threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "copy scalars");
    ck(cudaMemcpy(ctx.d_counts256, ctx.h_counts256,
                  ctx.threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "copy counts");

    int zero = FOUND_NONE;
    unsigned long long zero64 = 0ull;
    ck(cudaMemcpy(ctx.d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "init found_flag");
    ck(cudaMemcpy(ctx.d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes");

    // Compute initial points
    {
        int blocks_scal = (int)((ctx.threadsTotal + ctx.threadsPerBlock - 1) / ctx.threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, ctx.threadsPerBlock, 0, ctx.stream>>>(
            ctx.d_start_scalars, ctx.d_Px, ctx.d_Py, (int)ctx.threadsTotal);
        ck(cudaStreamSynchronize(ctx.stream), "scalarMulKernelBase sync");
    }

    // Precompute batch points G, 2G, ..., (half)G
    {
        uint64_t* h_scalars_half = nullptr;
        ck(cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t),
                         cudaHostAllocWriteCombined | cudaHostAllocMapped), "hostalloc half scalars");
        memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4 + 0] = (uint64_t)(k + 1);

        uint64_t *d_scalars_half=nullptr, *d_Gx_half=nullptr, *d_Gy_half=nullptr;
        ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "malloc d_scalars_half");
        ck(cudaMalloc(&d_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "malloc d_Gx_half");
        ck(cudaMalloc(&d_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "malloc d_Gy_half");
        ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t),
                      cudaMemcpyHostToDevice), "copy half scalars");

        int blocks_scal = (int)((half + ctx.threadsPerBlock - 1) / ctx.threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, ctx.threadsPerBlock, 0, ctx.stream>>>(
            d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
        ck(cudaStreamSynchronize(ctx.stream), "scalarMulKernelBase(half) sync");

        uint64_t* h_Gx_half = (uint64_t*)malloc((size_t)half * 4 * sizeof(uint64_t));
        uint64_t* h_Gy_half = (uint64_t*)malloc((size_t)half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx");
        ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

        cudaFree(d_scalars_half); cudaFree(d_Gx_half); cudaFree(d_Gy_half);
        cudaFreeHost(h_scalars_half);
        free(h_Gx_half); free(h_Gy_half);
    }

    // Precompute jump point B*G and -B*G (for backward scanning)
    {
        uint64_t* h_scalarB = nullptr;
        ck(cudaHostAlloc(&h_scalarB, 4 * sizeof(uint64_t),
                         cudaHostAllocWriteCombined | cudaHostAllocMapped), "hostalloc scalarB");
        memset(h_scalarB, 0, 4 * sizeof(uint64_t));
        h_scalarB[0] = (uint64_t)B;

        uint64_t *d_scalarB=nullptr, *d_Jx=nullptr, *d_Jy=nullptr;
        ck(cudaMalloc(&d_scalarB, 4 * sizeof(uint64_t)), "malloc d_scalarB");
        ck(cudaMalloc(&d_Jx, 4 * sizeof(uint64_t)), "malloc d_Jx");
        ck(cudaMalloc(&d_Jy, 4 * sizeof(uint64_t)), "malloc d_Jy");
        ck(cudaMemcpy(d_scalarB, h_scalarB, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "copy scalarB");

        scalarMulKernelBase<<<1, 1, 0, ctx.stream>>>(d_scalarB, d_Jx, d_Jy, 1);
        ck(cudaStreamSynchronize(ctx.stream), "scalarMulKernelBase(B) sync");

        uint64_t hJx[4], hJy[4], hJy_neg[4];
        ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
        ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");

        // Compute -B*G by negating Y coordinate (P - Y)
        compute_field_neg_host(hJy, hJy_neg);

        ck(cudaMemcpyToSymbol(c_Jx, hJx, 4 * sizeof(uint64_t)), "ToSymbol c_Jx");
        ck(cudaMemcpyToSymbol(c_Jy, hJy, 4 * sizeof(uint64_t)), "ToSymbol c_Jy");
        ck(cudaMemcpyToSymbol(c_Jy_neg, hJy_neg, 4 * sizeof(uint64_t)), "ToSymbol c_Jy_neg");

        cudaFree(d_scalarB); cudaFree(d_Jx); cudaFree(d_Jy);
        cudaFreeHost(h_scalarB);
    }

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

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
}

// ============================================================================
// CHECKPOINT FUNCTIONS
// ============================================================================

bool save_checkpoint(const std::string& filename,
                     const std::vector<GPUContext>& contexts,
                     const uint64_t range_start[4],
                     const uint64_t range_end[4],
                     const uint8_t target_hash160[20],
                     uint32_t batch_size,
                     uint32_t slices_per_launch,
                     bool pincer_mode,
                     int num_partitions,
                     unsigned long long total_keys_processed) {

    std::string temp_file = filename + ".tmp";
    std::ofstream file(temp_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create checkpoint file: " << temp_file << "\n";
        return false;
    }

    CheckpointHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = CHECKPOINT_MAGIC;
    header.version = CHECKPOINT_VERSION;
    header.num_gpus = (uint32_t)contexts.size();
    header.timestamp = (uint64_t)std::time(nullptr);
    header.total_keys_processed = total_keys_processed;
    memcpy(header.range_start, range_start, sizeof(header.range_start));
    memcpy(header.range_end, range_end, sizeof(header.range_end));
    memcpy(header.target_hash160, target_hash160, sizeof(header.target_hash160));
    header.batch_size = batch_size;
    header.slices_per_launch = slices_per_launch;
    header.pincer_mode = pincer_mode ? 1 : 0;
    header.num_partitions = (uint32_t)num_partitions;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file.good()) {
        std::cerr << "Error: Failed to write checkpoint header\n";
        file.close();
        std::remove(temp_file.c_str());
        return false;
    }

    for (const auto& ctx : contexts) {
        GPUCheckpointData gpu_data;
        memset(&gpu_data, 0, sizeof(gpu_data));

        gpu_data.device_id = ctx.deviceId;
        gpu_data.threads_total = (uint32_t)ctx.threadsTotal;
        memcpy(gpu_data.range_start, ctx.range_start, sizeof(gpu_data.range_start));
        memcpy(gpu_data.range_len, ctx.range_len, sizeof(gpu_data.range_len));
        memcpy(gpu_data.per_thread_cnt, ctx.per_thread_cnt, sizeof(gpu_data.per_thread_cnt));
        gpu_data.direction = (int32_t)ctx.direction;
        gpu_data.partition_id = ctx.partition_id;
        gpu_data.partner_gpu_idx = ctx.partner_gpu_idx;

        cudaSetDevice(ctx.deviceId);
        unsigned long long h = 0;
        cudaMemcpy(&h, ctx.d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        gpu_data.keys_processed = h;

        uint64_t current_scalar[4];
        cudaMemcpy(current_scalar, ctx.d_start_scalars, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        memcpy(gpu_data.current_position, current_scalar, sizeof(gpu_data.current_position));

        file.write(reinterpret_cast<const char*>(&gpu_data), sizeof(gpu_data));
        if (!file.good()) {
            std::cerr << "Error: Failed to write GPU " << ctx.deviceId << " checkpoint data\n";
            file.close();
            std::remove(temp_file.c_str());
            return false;
        }
    }

    file.close();

    if (std::rename(temp_file.c_str(), filename.c_str()) != 0) {
        std::cerr << "Error: Failed to rename checkpoint file\n";
        std::remove(temp_file.c_str());
        return false;
    }

    return true;
}

bool load_checkpoint(const std::string& filename,
                     CheckpointHeader& header,
                     std::vector<GPUCheckpointData>& gpu_data) {

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file.good()) {
        std::cerr << "Error: Failed to read checkpoint header\n";
        return false;
    }

    if (header.magic != CHECKPOINT_MAGIC) {
        std::cerr << "Error: Invalid checkpoint file (bad magic number)\n";
        return false;
    }

    if (header.version > CHECKPOINT_VERSION) {
        std::cerr << "Error: Checkpoint file version " << header.version
                  << " is newer than supported version " << CHECKPOINT_VERSION << "\n";
        return false;
    }

    gpu_data.clear();
    gpu_data.resize(header.num_gpus);

    for (uint32_t i = 0; i < header.num_gpus; ++i) {
        file.read(reinterpret_cast<char*>(&gpu_data[i]), sizeof(GPUCheckpointData));
        if (!file.good()) {
            std::cerr << "Error: Failed to read GPU " << i << " checkpoint data\n";
            return false;
        }
    }

    return true;
}

bool validate_checkpoint(const CheckpointHeader& header,
                         const uint64_t range_start[4],
                         const uint64_t range_end[4],
                         const uint8_t target_hash160[20],
                         uint32_t batch_size,
                         bool pincer_mode) {

    if (memcmp(header.range_start, range_start, sizeof(header.range_start)) != 0 ||
        memcmp(header.range_end, range_end, sizeof(header.range_end)) != 0) {
        std::cerr << "Warning: Checkpoint range does not match current range\n";
        return false;
    }

    if (memcmp(header.target_hash160, target_hash160, sizeof(header.target_hash160)) != 0) {
        std::cerr << "Warning: Checkpoint target does not match current target\n";
        return false;
    }

    if (header.batch_size != batch_size) {
        std::cerr << "Warning: Checkpoint batch size (" << header.batch_size
                  << ") does not match current batch size (" << batch_size << ")\n";
        return false;
    }

    bool checkpoint_pincer = (header.pincer_mode != 0);
    if (checkpoint_pincer != pincer_mode) {
        std::cerr << "Warning: Checkpoint pincer mode (" << (checkpoint_pincer ? "enabled" : "disabled")
                  << ") does not match current mode (" << (pincer_mode ? "enabled" : "disabled") << ")\n";
        return false;
    }

    return true;
}

bool resume_gpu_from_checkpoint(GPUContext& ctx, const GPUCheckpointData& checkpoint_data,
                                 const uint8_t target_hash160[20],
                                 uint32_t runtime_points_batch_size) {

    cudaSetDevice(ctx.deviceId);

    // Restore direction and partition info
    ctx.direction = (ScanDirection)checkpoint_data.direction;
    ctx.partition_id = checkpoint_data.partition_id;
    ctx.partner_gpu_idx = checkpoint_data.partner_gpu_idx;

    unsigned long long h = checkpoint_data.keys_processed;
    ck(cudaMemcpy(ctx.d_hashes_accum, &h, sizeof(unsigned long long), cudaMemcpyHostToDevice),
       "restore hashes_accum");

    uint64_t keys_per_thread = checkpoint_data.keys_processed / ctx.threadsTotal;

    for (uint64_t i = 0; i < ctx.threadsTotal; ++i) {
        uint64_t orig_count[4] = {ctx.per_thread_cnt[0], ctx.per_thread_cnt[1],
                                   ctx.per_thread_cnt[2], ctx.per_thread_cnt[3]};

        uint64_t borrow = keys_per_thread;
        for (int k = 0; k < 4 && borrow; ++k) {
            uint64_t old = orig_count[k];
            orig_count[k] = old - borrow;
            borrow = (old < borrow) ? 1ull : 0ull;
        }

        ctx.h_counts256[i*4+0] = orig_count[0];
        ctx.h_counts256[i*4+1] = orig_count[1];
        ctx.h_counts256[i*4+2] = orig_count[2];
        ctx.h_counts256[i*4+3] = orig_count[3];

        uint64_t new_scalar[4] = {ctx.h_start_scalars[i*4+0], ctx.h_start_scalars[i*4+1],
                                   ctx.h_start_scalars[i*4+2], ctx.h_start_scalars[i*4+3]};

        if (ctx.direction == SCAN_FORWARD) {
            // Forward: add processed keys to scalar
            uint64_t carry = keys_per_thread;
            for (int k = 0; k < 4 && carry; ++k) {
                uint64_t old = new_scalar[k];
                new_scalar[k] = old + carry;
                carry = (new_scalar[k] < old) ? 1ull : 0ull;
            }
        } else {
            // Backward: subtract processed keys from scalar
            uint64_t borrow_s = keys_per_thread;
            for (int k = 0; k < 4 && borrow_s; ++k) {
                uint64_t old = new_scalar[k];
                new_scalar[k] = old - borrow_s;
                borrow_s = (old < borrow_s) ? 1ull : 0ull;
            }
        }

        ctx.h_start_scalars[i*4+0] = new_scalar[0];
        ctx.h_start_scalars[i*4+1] = new_scalar[1];
        ctx.h_start_scalars[i*4+2] = new_scalar[2];
        ctx.h_start_scalars[i*4+3] = new_scalar[3];
    }

    ck(cudaMemcpy(ctx.d_counts256, ctx.h_counts256,
                  ctx.threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "upload resumed counts");
    ck(cudaMemcpy(ctx.d_start_scalars, ctx.h_start_scalars,
                  ctx.threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "upload resumed scalars");

    int blocks_scal = (int)((ctx.threadsTotal + ctx.threadsPerBlock - 1) / ctx.threadsPerBlock);
    scalarMulKernelBase<<<blocks_scal, ctx.threadsPerBlock, 0, ctx.stream>>>(
        ctx.d_start_scalars, ctx.d_Px, ctx.d_Py, (int)ctx.threadsTotal);
    ck(cudaStreamSynchronize(ctx.stream), "scalarMulKernelBase resume sync");

    return true;
}

std::string format_timestamp(uint64_t timestamp) {
    std::time_t t = static_cast<std::time_t>(timestamp);
    std::tm* tm = std::localtime(&t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    return std::string(buf);
}

// ============================================================================
// GPU WORKER THREAD (With direction support)
// ============================================================================

void gpu_worker_thread(GPUContext* ctx, uint32_t batch_size, uint32_t slices_per_launch) {
    cudaSetDevice(ctx->deviceId);

    bool stop = false;

    while (!stop) {
        if (g_sigint || g_found_global.load(std::memory_order_relaxed)) {
            stop = true;
            break;
        }

        unsigned int zeroU = 0u;
        cudaMemcpyAsync(ctx->d_any_left, &zeroU, sizeof(unsigned int),
                        cudaMemcpyHostToDevice, ctx->stream);

        // Launch pincer kernel with direction
        kernel_pincer_scan<<<ctx->blocks, ctx->threadsPerBlock, 0, ctx->stream>>>(
            ctx->d_Px, ctx->d_Py, ctx->d_Rx, ctx->d_Ry,
            ctx->d_start_scalars, ctx->d_counts256,
            ctx->threadsTotal,
            batch_size,
            slices_per_launch,
            ctx->d_found_flag, ctx->d_found_result,
            ctx->d_hashes_accum,
            ctx->d_any_left,
            (int)ctx->direction  // Pass scan direction to kernel
        );

        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "\nGPU " << ctx->deviceId << " kernel launch error: "
                      << cudaGetErrorString(launchErr) << "\n";
            stop = true;
            break;
        }

        while (!stop) {
            if (g_found_global.load(std::memory_order_relaxed)) {
                int found_ready = FOUND_READY;
                cudaMemcpy(ctx->d_found_flag, &found_ready, sizeof(int), cudaMemcpyHostToDevice);
                stop = true;
                break;
            }

            int host_found = 0;
            cudaMemcpy(&host_found, ctx->d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
            if (host_found == FOUND_READY) {
                int expected = -1;
                if (g_found_by_gpu.compare_exchange_strong(expected, ctx->deviceId)) {
                    g_found_global.store(true, std::memory_order_release);

                    FoundResult local_result{};
                    cudaMemcpy(&local_result, ctx->d_found_result, sizeof(FoundResult),
                               cudaMemcpyDeviceToHost);

                    std::lock_guard<std::mutex> lock(g_result_mutex);
                    g_global_result = local_result;
                }
                ctx->found_match.store(true);
                stop = true;
                break;
            }

            cudaError_t qs = cudaStreamQuery(ctx->stream);
            if (qs == cudaSuccess) break;
            else if (qs != cudaErrorNotReady) {
                cudaGetLastError();
                stop = true;
                break;
            }

            if (g_sigint) {
                stop = true;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        cudaStreamSynchronize(ctx->stream);
        if (stop) break;

        unsigned int h_any = 0u;
        cudaMemcpy(&h_any, ctx->d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        std::swap(ctx->d_Px, ctx->d_Rx);
        std::swap(ctx->d_Py, ctx->d_Ry);

        if (h_any == 0u) {
            break;
        }
    }

    ctx->completed.store(true);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string target_hash_hex, range_hex, address_b58;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         = 64;
    std::vector<int> gpu_ids;
    std::string checkpoint_file = "";
    uint32_t checkpoint_interval = 60;
    bool resume_mode = false;
    bool pincer_mode = false;

    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        char* endp=nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10); if (*endp) return false;
        endp=nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10); if (*endp) return false;
        if (aa == 0ul || bb == 0ul) return false;
        if (aa > (1ul<<20) || bb > (1ul<<20)) return false;
        a_out=(uint32_t)aa; b_out=(uint32_t)bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--target-hash160" && i + 1 < argc) target_hash_hex = argv[++i];
        else if (arg == "--address"        && i + 1 < argc) address_b58     = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\" (positive integers).\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm    = b;
        }
        else if (arg == "--slices" && i + 1 < argc) {
            char* endp=nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul<<20)) {
                std::cerr << "Error: --slices must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = (uint32_t)v;
        }
        else if (arg == "--gpus" && i + 1 < argc) {
            std::string gpustr = argv[++i];
            std::stringstream ss(gpustr);
            std::string item;
            while (std::getline(ss, item, ',')) {
                gpu_ids.push_back(std::stoi(item));
            }
        }
        else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpoint_file = argv[++i];
        }
        else if (arg == "--checkpoint-interval" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0 || v > 3600) {
                std::cerr << "Error: --checkpoint-interval must be 1-3600 seconds\n";
                return EXIT_FAILURE;
            }
            checkpoint_interval = (uint32_t)v;
        }
        else if (arg == "--resume") {
            resume_mode = true;
        }
        else if (arg == "--pincer") {
            pincer_mode = true;
        }
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>)\n"
                  << "       [--grid A,B] [--slices N] [--gpus 0,1,2,...] [--pincer]\n"
                  << "       [--checkpoint <file>] [--checkpoint-interval <seconds>] [--resume]\n\n"
                  << "Pincer Mode (--pincer):\n"
                  << "  Bidirectional scanning pairs GPUs to search each partition from both ends.\n"
                  << "  Requires even number of GPUs. Provides ~2x average speedup to find key.\n";
        return EXIT_FAILURE;
    }
    if (!target_hash_hex.empty() && !address_b58.empty()) {
        std::cerr << "Error: provide either --address or --target-hash160, not both.\n";
        return EXIT_FAILURE;
    }

    // Parse range
    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }

    uint8_t target_hash160[20];
    if (!address_b58.empty()) {
        if (!decode_p2pkh_address(address_b58, target_hash160)) {
            std::cerr << "Error: invalid P2PKH address\n"; return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: invalid target hash160 hex\n"; return EXIT_FAILURE;
        }
    }

    // Validate batch size
    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << " (kernel limit).\n";
        return EXIT_FAILURE;
    }

    // Calculate range length
    uint64_t range_len[4];
    sub256(range_end, range_start, range_len);
    add256_u64(range_len, 1ull, range_len);

    // Validate power of two
    auto is_zero_256 = [](const uint64_t a[4])->bool { return (a[0]|a[1]|a[2]|a[3]) == 0ull; };
    auto is_power_of_two_256 = [&](const uint64_t a[4])->bool {
        if (is_zero_256(a)) return false;
        uint64_t am1[4]; uint64_t borrow = 1ull;
        for (int i=0;i<4;++i) {
            uint64_t v = a[i] - borrow; borrow = (a[i] < borrow) ? 1ull : 0ull; am1[i] = v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) am1[k] = a[k]; break; }
        }
        uint64_t and0=a[0]&am1[0], and1=a[1]&am1[1], and2=a[2]&am1[2], and3=a[3]&am1[3];
        return (and0|and1|and2|and3)==0ull;
    };
    if (!is_power_of_two_256(range_len)) {
        std::cerr << "Error: range length (end - start + 1) must be a power of two.\n";
        return EXIT_FAILURE;
    }

    // Detect GPUs
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "Error: no CUDA devices found\n";
        return EXIT_FAILURE;
    }

    if (gpu_ids.empty()) {
        for (int i = 0; i < device_count; ++i) {
            gpu_ids.push_back(i);
        }
    }

    int num_gpus = (int)gpu_ids.size();

    // Pincer mode validation
    if (pincer_mode) {
        if (num_gpus < 2 || (num_gpus & 1) != 0) {
            std::cerr << "Error: Pincer mode requires an even number of GPUs (at least 2).\n";
            std::cerr << "       You have " << num_gpus << " GPU(s) selected.\n";
            return EXIT_FAILURE;
        }
    }

    int num_partitions = pincer_mode ? (num_gpus / 2) : num_gpus;

    // Calculate per-partition range
    uint64_t per_partition_range[4];
    divide_range_by_n(range_len, num_partitions, per_partition_range);

    g_checkpoint_file = checkpoint_file;

    // Check for checkpoint resume
    CheckpointHeader checkpoint_header;
    std::vector<GPUCheckpointData> checkpoint_gpu_data;
    bool resuming = false;
    unsigned long long resumed_keys = 0;

    if (resume_mode && !checkpoint_file.empty()) {
        if (load_checkpoint(checkpoint_file, checkpoint_header, checkpoint_gpu_data)) {
            if (validate_checkpoint(checkpoint_header, range_start, range_end,
                                    target_hash160, runtime_points_batch_size, pincer_mode)) {
                resuming = true;
                resumed_keys = checkpoint_header.total_keys_processed;
                std::cout << "======== Resuming from Checkpoint =====================\n";
                std::cout << "Checkpoint file     : " << checkpoint_file << "\n";
                std::cout << "Saved at            : " << format_timestamp(checkpoint_header.timestamp) << "\n";
                std::cout << "Keys processed      : " << resumed_keys << "\n";
                std::cout << "GPUs in checkpoint  : " << checkpoint_header.num_gpus << "\n";
                std::cout << "Pincer mode         : " << (checkpoint_header.pincer_mode ? "enabled" : "disabled") << "\n";
                std::cout << "-------------------------------------------------------\n\n";
            } else {
                std::cerr << "Warning: Checkpoint validation failed, starting fresh\n";
            }
        } else if (resume_mode) {
            std::cout << "No checkpoint file found, starting fresh search\n\n";
        }
    }

    // Print configuration
    std::cout << "======== Multi-GPU Configuration";
    if (pincer_mode) std::cout << " (PINCER MODE)";
    std::cout << " =====\n";
    std::cout << "Number of GPUs      : " << num_gpus << "\n";
    std::cout << "GPU IDs             : ";
    for (int id : gpu_ids) std::cout << id << " ";
    std::cout << "\n";
    std::cout << "Total range length  : " << formatHex256(range_len) << "\n";
    if (pincer_mode) {
        std::cout << "Partitions          : " << num_partitions << "\n";
        std::cout << "Per-partition range : " << formatHex256(per_partition_range) << "\n";
        std::cout << "Scan mode           : Bidirectional (Pincer)\n";
    } else {
        std::cout << "Per-GPU range       : " << formatHex256(per_partition_range) << "\n";
    }
    std::cout << "Batch size          : " << runtime_points_batch_size << "\n";
    std::cout << "Slices per launch   : " << slices_per_launch << "\n";
    if (!checkpoint_file.empty()) {
        std::cout << "Checkpoint file     : " << checkpoint_file << "\n";
        std::cout << "Checkpoint interval : " << checkpoint_interval << " seconds\n";
    }
    std::cout << "-------------------------------------------------------\n\n";

    // Initialize GPU contexts
    std::vector<GPUContext> contexts(num_gpus);

    std::cout << "======== GPU Initialization ===========================\n";

    if (pincer_mode) {
        // Pincer mode: pair GPUs for each partition
        for (int partition = 0; partition < num_partitions; ++partition) {
            int fwd_gpu_idx = partition * 2;
            int bwd_gpu_idx = partition * 2 + 1;

            // Calculate partition start
            uint64_t partition_start[4];
            uint64_t offset[4];
            mult_256_by_u64(per_partition_range, (uint64_t)partition, offset);
            add_offset_256(range_start, offset, partition_start);

            // Each GPU in the pair scans half the partition
            uint64_t half_partition[4];
            divide_range_by_n(per_partition_range, 2, half_partition);

            // Forward GPU: starts at partition_start, scans forward
            uint64_t fwd_start[4];
            for (int i = 0; i < 4; ++i) fwd_start[i] = partition_start[i];

            if (!initialize_gpu_context_pincer(contexts[fwd_gpu_idx], gpu_ids[fwd_gpu_idx],
                                                fwd_start, half_partition,
                                                SCAN_FORWARD, partition, bwd_gpu_idx,
                                                target_hash160, runtime_points_batch_size, runtime_batches_per_sm)) {
                std::cerr << "Failed to initialize GPU " << gpu_ids[fwd_gpu_idx] << " (forward)\n";
                return EXIT_FAILURE;
            }

            // Backward GPU: starts at partition_start + half_partition, scans backward
            uint64_t bwd_start[4];
            add_offset_256(partition_start, half_partition, bwd_start);

            if (!initialize_gpu_context_pincer(contexts[bwd_gpu_idx], gpu_ids[bwd_gpu_idx],
                                                bwd_start, half_partition,
                                                SCAN_BACKWARD, partition, fwd_gpu_idx,
                                                target_hash160, runtime_points_batch_size, runtime_batches_per_sm)) {
                std::cerr << "Failed to initialize GPU " << gpu_ids[bwd_gpu_idx] << " (backward)\n";
                return EXIT_FAILURE;
            }

            // Print partition info
            size_t freeB=0, totalB=0;

            cudaSetDevice(gpu_ids[fwd_gpu_idx]);
            cudaMemGetInfo(&freeB, &totalB);
            size_t usedB = totalB - freeB;
            double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;
            std::cout << "Partition " << partition << " [" << formatHex256(partition_start).substr(48) << "]:\n";
            std::cout << "  GPU " << gpu_ids[fwd_gpu_idx] << " (" << contexts[fwd_gpu_idx].prop.name << ") FORWARD:  "
                      << contexts[fwd_gpu_idx].threadsTotal << " threads, "
                      << std::fixed << std::setprecision(1) << util << "% VRAM\n";

            cudaSetDevice(gpu_ids[bwd_gpu_idx]);
            cudaMemGetInfo(&freeB, &totalB);
            usedB = totalB - freeB;
            util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;
            std::cout << "  GPU " << gpu_ids[bwd_gpu_idx] << " (" << contexts[bwd_gpu_idx].prop.name << ") BACKWARD: "
                      << contexts[bwd_gpu_idx].threadsTotal << " threads, "
                      << std::fixed << std::setprecision(1) << util << "% VRAM\n";
        }
    } else {
        // Standard mode: each GPU scans its own partition forward
        for (int i = 0; i < num_gpus; ++i) {
            uint64_t partition_start[4];
            uint64_t offset[4];
            mult_256_by_u64(per_partition_range, (uint64_t)i, offset);
            add_offset_256(range_start, offset, partition_start);

            if (!initialize_gpu_context_pincer(contexts[i], gpu_ids[i],
                                                partition_start, per_partition_range,
                                                SCAN_FORWARD, i, -1,
                                                target_hash160, runtime_points_batch_size, runtime_batches_per_sm)) {
                std::cerr << "Failed to initialize GPU " << gpu_ids[i] << "\n";
                return EXIT_FAILURE;
            }

            size_t freeB=0, totalB=0;
            cudaSetDevice(gpu_ids[i]);
            cudaMemGetInfo(&freeB, &totalB);
            size_t usedB = totalB - freeB;
            double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;

            std::cout << "GPU " << gpu_ids[i] << " (" << contexts[i].prop.name << "): "
                      << contexts[i].threadsTotal << " threads, "
                      << contexts[i].blocks << " blocks, "
                      << std::fixed << std::setprecision(1) << util << "% VRAM\n";
            std::cout << "  Range: " << formatHex256(contexts[i].range_start) << " + "
                      << formatHex256(contexts[i].range_len) << "\n";
        }
    }
    std::cout << "-------------------------------------------------------\n\n";

    // Apply checkpoint resume
    if (resuming && checkpoint_gpu_data.size() == (size_t)num_gpus) {
        std::cout << "======== Applying Checkpoint State ====================\n";
        for (int i = 0; i < num_gpus; ++i) {
            const GPUCheckpointData* gpu_ckpt = nullptr;
            for (const auto& ckpt : checkpoint_gpu_data) {
                if (ckpt.device_id == contexts[i].deviceId) {
                    gpu_ckpt = &ckpt;
                    break;
                }
            }
            if (!gpu_ckpt && i < (int)checkpoint_gpu_data.size()) {
                gpu_ckpt = &checkpoint_gpu_data[i];
            }

            if (gpu_ckpt) {
                resume_gpu_from_checkpoint(contexts[i], *gpu_ckpt,
                                           target_hash160, runtime_points_batch_size);
                std::cout << "GPU " << contexts[i].deviceId << ": Resumed with "
                          << gpu_ckpt->keys_processed << " keys already processed";
                if (pincer_mode) {
                    std::cout << " (" << (contexts[i].direction == SCAN_FORWARD ? "FWD" : "BWD") << ")";
                }
                std::cout << "\n";
            }
        }
        std::cout << "-------------------------------------------------------\n\n";
    }

    g_last_checkpoint_time = std::chrono::steady_clock::now();

    // Launch workers
    std::cout << "======== Phase-1: Multi-GPU BruteForce";
    if (pincer_mode) std::cout << " (PINCER)";
    std::cout << " ========\n";

    std::vector<std::thread> workers;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_gpus; ++i) {
        workers.emplace_back(gpu_worker_thread, &contexts[i],
                            runtime_points_batch_size, slices_per_launch);
    }

    // Progress monitoring
    auto tLast = t0;
    unsigned long long lastTotalHashes = 0ull;

    while (true) {
        bool all_done = true;
        for (int i = 0; i < num_gpus; ++i) {
            if (!contexts[i].completed.load()) {
                all_done = false;
                break;
            }
        }
        if (all_done || g_found_global.load() || g_sigint) break;

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();

        if (dt >= 1.0) {
            unsigned long long totalHashes = 0ull;
            for (int i = 0; i < num_gpus; ++i) {
                cudaSetDevice(gpu_ids[i]);
                unsigned long long h = 0ull;
                cudaMemcpy(&h, contexts[i].d_hashes_accum, sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost);
                totalHashes += h;
            }

            double delta = (double)(totalHashes - lastTotalHashes);
            double mkeys = delta / (dt * 1e6);
            double elapsed = std::chrono::duration<double>(now - t0).count();
            long double total_keys_ld = ld_from_u256(range_len);
            long double prog = total_keys_ld > 0.0L ? ((long double)totalHashes / total_keys_ld) * 100.0L : 0.0L;
            if (prog > 100.0L) prog = 100.0L;

            std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                      << " s | Speed: " << std::fixed << std::setprecision(1) << mkeys
                      << " Mkeys/s | Count: " << totalHashes
                      << " | Progress: " << std::fixed << std::setprecision(2) << (double)prog << " %"
                      << " | GPUs: " << num_gpus;
            if (pincer_mode) std::cout << " (P)";
            std::cout.flush();

            lastTotalHashes = totalHashes;
            tLast = now;

            // Checkpoint
            if (!checkpoint_file.empty()) {
                auto checkpoint_now = std::chrono::steady_clock::now();
                double checkpoint_dt = std::chrono::duration<double>(checkpoint_now - g_last_checkpoint_time).count();

                if (checkpoint_dt >= (double)checkpoint_interval || g_checkpoint_requested.load()) {
                    if (save_checkpoint(checkpoint_file, contexts, range_start, range_end,
                                        target_hash160, runtime_points_batch_size,
                                        slices_per_launch, pincer_mode, num_partitions, totalHashes)) {
                        std::cout << " [CKPT]";
                        std::cout.flush();
                    }
                    g_last_checkpoint_time = checkpoint_now;
                    g_checkpoint_requested.store(false);
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (auto& t : workers) {
        t.join();
    }

    std::cout << "\n\n";

    // Results
    int exit_code = EXIT_SUCCESS;

    if (g_found_global.load()) {
        std::lock_guard<std::mutex> lock(g_result_mutex);
        std::cout << "======== FOUND MATCH! =================================\n";
        std::cout << "Found by GPU        : " << g_found_by_gpu.load();
        if (pincer_mode) {
            int found_idx = -1;
            int found_device = g_found_by_gpu.load();
            for (int i = 0; i < num_gpus; ++i) {
                if (contexts[i].deviceId == found_device) {
                    found_idx = i;
                    break;
                }
            }
            if (found_idx >= 0) {
                std::cout << " (" << (contexts[found_idx].direction == SCAN_FORWARD ? "FORWARD" : "BACKWARD")
                          << ", Partition " << contexts[found_idx].partition_id << ")";
            }
        }
        std::cout << "\n";
        std::cout << "Private Key         : " << formatHex256(g_global_result.scalar) << "\n";
        std::cout << "Public Key          : " << formatCompressedPubHex(g_global_result.Rx, g_global_result.Ry) << "\n";
    } else if (g_sigint) {
        std::cout << "======== INTERRUPTED (Ctrl+C) =========================\n";
        std::cout << "Search was interrupted by user.\n";

        if (!checkpoint_file.empty()) {
            unsigned long long finalHashes = 0;
            for (int i = 0; i < num_gpus; ++i) {
                cudaSetDevice(gpu_ids[i]);
                unsigned long long h = 0;
                cudaMemcpy(&h, contexts[i].d_hashes_accum, sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost);
                finalHashes += h;
            }

            if (save_checkpoint(checkpoint_file, contexts, range_start, range_end,
                                target_hash160, runtime_points_batch_size,
                                slices_per_launch, pincer_mode, num_partitions, finalHashes)) {
                std::cout << "Checkpoint saved    : " << checkpoint_file << "\n";
                std::cout << "Keys processed      : " << finalHashes << "\n";
                std::cout << "To resume, run with : --resume --checkpoint " << checkpoint_file;
                if (pincer_mode) std::cout << " --pincer";
                std::cout << "\n";
            }
        }
        exit_code = 130;
    } else {
        std::cout << "======== KEY NOT FOUND (exhaustive) ===================\n";
        std::cout << "Target hash160 was not found within the specified range.\n";
    }

    // Cleanup
    for (int i = 0; i < num_gpus; ++i) {
        cleanup_gpu_context(contexts[i]);
    }

    return exit_code;
}
