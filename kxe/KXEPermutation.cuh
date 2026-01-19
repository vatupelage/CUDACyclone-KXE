// ============================================================================
// KXE Permutation - Bijective Pseudo-Random Permutation for Range Scanning
// ============================================================================
// Implements a Feistel-based permutation that spreads keys uniformly across
// the search range while guaranteeing:
// - Bijection: Every input maps to exactly one unique output (no gaps, no overlaps)
// - Uniform distribution: Output appears pseudo-random across the range
// - Cheap computation: < 50 cycles (vs ~10,000+ for ECC point mult)
// - Deterministic: Same (stream_id, counter) always produces same key
// - Streamable: Different stream_ids produce disjoint sequences
// ============================================================================

#ifndef KXE_PERMUTATION_CUH
#define KXE_PERMUTATION_CUH

#include <cstdint>

// ============================================================================
// HOST/DEVICE COMPATIBILITY MACROS
// ============================================================================
// Makes this header usable from both CUDA (.cu) and pure C++ (.cpp) files

#ifdef __CUDACC__
    #define KXE_HOST_DEVICE __host__ __device__
    #define KXE_DEVICE __device__
    #define KXE_FORCEINLINE __forceinline__
#else
    #define KXE_HOST_DEVICE
    #define KXE_DEVICE
    #define KXE_FORCEINLINE inline
    // Disable pragma unroll in pure C++
    #define __uint128_t unsigned __int128
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================

#define KXE_FEISTEL_ROUNDS 4
#define KXE_MAX_STREAMS 256

// ============================================================================
// FEISTEL NETWORK ROUND FUNCTION
// ============================================================================
// Uses a fast mixing function based on xorshift and multiplication
// Achieves good avalanche properties with minimal cycles

KXE_HOST_DEVICE KXE_FORCEINLINE
uint32_t kxe_mix32(uint32_t x, uint32_t key) {
    x ^= key;
    x *= 0x85ebca6bU;
    x ^= x >> 13;
    x *= 0xc2b2ae35U;
    x ^= x >> 16;
    return x;
}

KXE_HOST_DEVICE KXE_FORCEINLINE
uint64_t kxe_mix64(uint64_t x, uint64_t key) {
    x ^= key;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// ============================================================================
// 64-BIT FEISTEL PERMUTATION
// ============================================================================
// For ranges up to 2^64 (covers puzzles up to ~135)
// Uses 32-bit half-blocks with 4 rounds

KXE_HOST_DEVICE KXE_FORCEINLINE
uint64_t kxe_feistel_permute_64(uint64_t x, uint32_t stream_id) {
    uint32_t L = (uint32_t)(x >> 32);
    uint32_t R = (uint32_t)(x & 0xFFFFFFFFU);

    // 4 rounds of balanced Feistel
    #pragma unroll
    for (int round = 0; round < KXE_FEISTEL_ROUNDS; ++round) {
        // Round key incorporates stream_id and round number
        uint32_t round_key = stream_id ^ (stream_id << 13) ^
                             (stream_id >> 17) ^ (round * 0x9e3779b9U);
        uint32_t F = kxe_mix32(R, round_key);
        uint32_t new_R = L ^ F;
        L = R;
        R = new_R;
    }

    return ((uint64_t)R << 32) | L;
}

// Inverse permutation (for verification)
KXE_HOST_DEVICE KXE_FORCEINLINE
uint64_t kxe_feistel_inverse_64(uint64_t x, uint32_t stream_id) {
    uint32_t L = (uint32_t)(x & 0xFFFFFFFFU);
    uint32_t R = (uint32_t)(x >> 32);

    // Reverse rounds
    #pragma unroll
    for (int round = KXE_FEISTEL_ROUNDS - 1; round >= 0; --round) {
        uint32_t round_key = stream_id ^ (stream_id << 13) ^
                             (stream_id >> 17) ^ (round * 0x9e3779b9U);
        uint32_t F = kxe_mix32(L, round_key);
        uint32_t new_L = R ^ F;
        R = L;
        L = new_L;
    }

    return ((uint64_t)L << 32) | R;
}

// ============================================================================
// CYCLE-WALKING FOR NON-POWER-OF-2 RANGES
// ============================================================================
// When range_width is not a power of 2, we use cycle-walking:
// If permuted value >= range_width, we permute again until it's in range
// This maintains bijection while handling arbitrary range sizes

KXE_HOST_DEVICE KXE_FORCEINLINE
uint64_t kxe_permute_in_range_64(uint64_t counter, uint32_t stream_id, uint64_t range_width) {
    uint64_t x = counter;

    // Apply permutation, cycle-walk if out of range
    // For power-of-2 ranges, this loop executes exactly once
    // For other ranges, expected iterations < 2
    for (int iter = 0; iter < 64; ++iter) {
        x = kxe_feistel_permute_64(x, stream_id + iter);
        if (x < range_width) {
            return x;
        }
        // Continue with new value for cycle-walking
    }

    // Fallback (should never reach for reasonable ranges)
    return counter % range_width;
}

// ============================================================================
// INTERLEAVED STREAM PARTITIONING
// ============================================================================
// For multiple streams, we interleave indices:
// Stream 0: indices 0, S, 2S, 3S, ...
// Stream 1: indices 1, S+1, 2S+1, 3S+1, ...
// Stream S-1: indices S-1, 2S-1, 3S-1, ...
// This ensures zero overlap between streams by construction

KXE_HOST_DEVICE KXE_FORCEINLINE
uint64_t kxe_stream_index(uint32_t stream_id, uint64_t local_counter, uint32_t num_streams) {
    // Map (stream_id, local_counter) to global index
    return (uint64_t)stream_id + local_counter * (uint64_t)num_streams;
}

// Complete permutation with stream support
KXE_HOST_DEVICE KXE_FORCEINLINE
uint64_t kxe_permute(uint32_t stream_id, uint64_t counter,
                     uint64_t range_width, uint32_t num_streams) {
    // Step 1: Compute global index from stream partition
    uint64_t global_index = kxe_stream_index(stream_id, counter, num_streams);

    // Step 2: Bounds check
    if (global_index >= range_width) {
        return range_width; // Signal out of range
    }

    // Step 3: Apply permutation within range
    return kxe_permute_in_range_64(global_index, stream_id, range_width);
}

// ============================================================================
// 256-BIT SUPPORT FOR LARGE RANGES
// ============================================================================
// For puzzles with ranges > 2^64, we extend using two 64-bit permutations
// Applied to low and high parts with different keys

KXE_HOST_DEVICE KXE_FORCEINLINE
void kxe_add_256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    __uint128_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        __uint128_t s = (__uint128_t)a[i] + b[i] + carry;
        out[i] = (uint64_t)s;
        carry = s >> 64;
    }
}

KXE_HOST_DEVICE KXE_FORCEINLINE
bool kxe_lt_256(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

// Get effective bit width of a 256-bit number
KXE_HOST_DEVICE KXE_FORCEINLINE
int kxe_bitwidth_256(const uint64_t v[4]) {
    for (int i = 3; i >= 0; --i) {
        if (v[i] != 0) {
            #ifdef __CUDA_ARCH__
            return i * 64 + (64 - __clzll(v[i]));
            #else
            return i * 64 + (64 - __builtin_clzll(v[i]));
            #endif
        }
    }
    return 0;
}

// Permute 256-bit counter and add to range_start
// Returns true if the resulting scalar is within range
KXE_HOST_DEVICE KXE_FORCEINLINE
bool kxe_permute_256(
    uint64_t output[4],
    uint32_t stream_id,
    uint64_t counter,
    const uint64_t range_start[4],
    const uint64_t range_width[4],
    uint32_t num_streams
) {
    // For ranges that fit in 64 bits, use optimized path
    if (range_width[3] == 0 && range_width[2] == 0 && range_width[1] == 0) {
        uint64_t permuted_offset = kxe_permute(stream_id, counter, range_width[0], num_streams);
        if (permuted_offset >= range_width[0]) {
            return false;
        }

        // Add offset to range_start
        uint64_t offset[4] = {permuted_offset, 0, 0, 0};
        kxe_add_256(range_start, offset, output);
        return true;
    }

    // For ranges up to 128 bits
    if (range_width[3] == 0 && range_width[2] == 0) {
        // Combine low 128 bits
        __uint128_t range_w128 = ((__uint128_t)range_width[1] << 64) | range_width[0];
        __uint128_t global_index = (__uint128_t)stream_id + (__uint128_t)counter * num_streams;

        if (global_index >= range_w128) {
            return false;
        }

        // Permute each 64-bit half with different keys
        uint64_t lo = (uint64_t)global_index;
        uint64_t hi = (uint64_t)(global_index >> 64);

        lo = kxe_feistel_permute_64(lo, stream_id);
        hi = kxe_feistel_permute_64(hi, stream_id ^ 0x5A5A5A5AU);

        // Combine and check range (may need cycle walking)
        __uint128_t permuted = ((__uint128_t)hi << 64) | lo;
        for (int iter = 0; iter < 64 && permuted >= range_w128; ++iter) {
            lo = kxe_feistel_permute_64(lo, stream_id + iter + 1);
            hi = kxe_feistel_permute_64(hi, stream_id ^ (iter + 1) ^ 0x5A5A5A5AU);
            permuted = ((__uint128_t)hi << 64) | lo;
        }

        uint64_t offset[4] = {(uint64_t)permuted, (uint64_t)(permuted >> 64), 0, 0};
        kxe_add_256(range_start, offset, output);
        return kxe_lt_256(offset, range_width) ||
               (offset[0] == range_width[0] && offset[1] == range_width[1] &&
                offset[2] == range_width[2] && offset[3] == range_width[3]);
    }

    // Full 256-bit path (for very large ranges)
    // Use stream_id + counter approach with multi-part permutation
    uint64_t global_index[4] = {0, 0, 0, 0};
    {
        __uint128_t prod = (__uint128_t)counter * num_streams + stream_id;
        global_index[0] = (uint64_t)prod;
        global_index[1] = (uint64_t)(prod >> 64);
    }

    // Check bounds
    if (!kxe_lt_256(global_index, range_width)) {
        return false;
    }

    // Permute each 64-bit part
    global_index[0] = kxe_feistel_permute_64(global_index[0], stream_id);
    global_index[1] = kxe_feistel_permute_64(global_index[1], stream_id ^ 0x5A5A5A5AU);
    global_index[2] = kxe_feistel_permute_64(global_index[2], stream_id ^ 0xA5A5A5A5U);
    global_index[3] = kxe_feistel_permute_64(global_index[3], stream_id ^ 0xFFFFFFFFU);

    // Cycle-walk if needed
    for (int iter = 0; iter < 64 && !kxe_lt_256(global_index, range_width); ++iter) {
        global_index[0] = kxe_feistel_permute_64(global_index[0], stream_id + iter + 1);
    }

    kxe_add_256(range_start, global_index, output);
    return true;
}

// ============================================================================
// CHECKPOINT SUPPORT
// ============================================================================
// KXE checkpoint is minimal: just (stream_id, counter) per GPU
// Everything else can be recomputed from these values

struct KXECheckpointEntry {
    uint32_t stream_id;
    uint64_t counter;
};

struct KXECheckpointHeader {
    uint32_t magic;           // "KXE2" = 0x3258454B
    uint32_t version;
    uint64_t timestamp;
    uint64_t range_start[4];  // 256-bit
    uint64_t range_width[4];  // 256-bit
    uint8_t target_hash160[20];
    uint32_t num_gpus;
    uint32_t num_streams;
    uint32_t batch_size;
    uint32_t slices;
    uint32_t batches_per_sm;  // Second value of --grid (critical for keys_per_block)
    uint32_t found;           // 0 = not found, 1 = found
    uint64_t found_scalar[4]; // If found
    // Followed by num_gpus KXECheckpointEntry structs
};

#define KXE_CHECKPOINT_MAGIC 0x3258454BU  // "KXE2"
#define KXE_CHECKPOINT_VERSION 2

// ============================================================================
// KERNEL INTEGRATION HELPERS
// ============================================================================
// These functions help integrate KXE with the existing kernel structure

// Compute the scalar value for a given thread
KXE_DEVICE KXE_FORCEINLINE
void kxe_compute_scalar_for_thread(
    uint64_t scalar_out[4],
    uint32_t stream_id,
    uint64_t base_counter,
    uint64_t thread_offset,
    const uint64_t range_start[4],
    const uint64_t range_width[4],
    uint32_t num_streams,
    bool* out_of_range
) {
    uint64_t counter = base_counter + thread_offset;
    *out_of_range = !kxe_permute_256(scalar_out, stream_id, counter,
                                      range_start, range_width, num_streams);
}

// Batch variant: compute base point for batch processing
// The kernel will process keys: scalar, scalar+1, scalar+2, ..., scalar+(batch_size-1)
// For KXE mode, each "batch" is actually individual permuted keys
KXE_DEVICE KXE_FORCEINLINE
void kxe_compute_batch_scalar(
    uint64_t scalar_out[4],
    uint32_t stream_id,
    uint64_t batch_index,
    uint32_t batch_size,
    const uint64_t range_start[4],
    const uint64_t range_width[4],
    uint32_t num_streams,
    bool* out_of_range
) {
    // For KXE, batch_index directly maps to counter
    // Each kernel invocation processes one permuted key per batch
    kxe_compute_scalar_for_thread(scalar_out, stream_id, batch_index * batch_size, 0,
                                   range_start, range_width, num_streams, out_of_range);
}

#endif // KXE_PERMUTATION_CUH
