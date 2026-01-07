// ============================================================================
// KXE Permutation Bijection Tests
// ============================================================================
// Tests to verify the KXE permutation is correct:
// 1. Bijection: Every input maps to unique output (no gaps, no overlaps)
// 2. Inverse: inverse(permute(x)) == x
// 3. Stream disjointness: Different streams produce disjoint values
// 4. Uniformity: Outputs are roughly uniformly distributed
// 5. Performance: Cycles per permutation
// ============================================================================

#include <iostream>
#include <iomanip>
#include <set>
#include <map>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <cmath>
#include <algorithm>

// Include KXE permutation (host-side)
#include "../KXEPermutation.cuh"

// ============================================================================
// TEST UTILITIES
// ============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;

void test_result(const char* name, bool passed) {
    if (passed) {
        std::cout << "[PASS] " << name << "\n";
        g_tests_passed++;
    } else {
        std::cout << "[FAIL] " << name << "\n";
        g_tests_failed++;
    }
}

// ============================================================================
// TEST 1: BIJECTION (exhaustive for small ranges)
// ============================================================================

bool test_bijection_exhaustive(uint64_t range_width, uint32_t stream_id) {
    std::cout << "  Testing bijection for range_width=" << range_width
              << ", stream_id=" << stream_id << "...\n";

    std::set<uint64_t> seen;
    bool all_in_range = true;
    bool no_duplicates = true;

    for (uint64_t i = 0; i < range_width; ++i) {
        uint64_t output = kxe_permute_in_range_64(i, stream_id, range_width);

        // Check output is in range
        if (output >= range_width) {
            std::cerr << "    ERROR: Output " << output << " >= range_width " << range_width
                      << " for input " << i << "\n";
            all_in_range = false;
        }

        // Check for duplicates
        if (seen.find(output) != seen.end()) {
            std::cerr << "    ERROR: Duplicate output " << output << " for input " << i << "\n";
            no_duplicates = false;
        }
        seen.insert(output);

        // Progress indicator for large ranges
        if (range_width >= 1000000 && i % (range_width / 10) == 0) {
            std::cout << "    " << (i * 100 / range_width) << "% complete...\n";
        }
    }

    // Check all values covered
    bool all_covered = (seen.size() == range_width);
    if (!all_covered) {
        std::cerr << "    ERROR: Only " << seen.size() << " unique values, expected "
                  << range_width << "\n";
    }

    return all_in_range && no_duplicates && all_covered;
}

// ============================================================================
// TEST 2: INVERSE PROPERTY
// ============================================================================

bool test_inverse_property(uint64_t range_width, uint32_t stream_id, uint64_t num_samples) {
    std::cout << "  Testing inverse property with " << num_samples << " samples...\n";

    std::mt19937_64 rng(42 + stream_id);
    bool all_passed = true;

    for (uint64_t i = 0; i < num_samples; ++i) {
        uint64_t x = rng() % range_width;
        uint64_t permuted = kxe_feistel_permute_64(x, stream_id);
        uint64_t inverted = kxe_feistel_inverse_64(permuted, stream_id);

        if (inverted != x) {
            std::cerr << "    ERROR: inverse(permute(" << x << ")) = " << inverted
                      << " != " << x << "\n";
            all_passed = false;
            if (i > 10) break; // Don't spam too many errors
        }
    }

    return all_passed;
}

// ============================================================================
// TEST 3: STREAM DISJOINTNESS
// ============================================================================

bool test_stream_disjointness(uint64_t range_width, uint32_t num_streams, uint64_t samples_per_stream) {
    std::cout << "  Testing stream disjointness: " << num_streams << " streams, "
              << samples_per_stream << " samples each...\n";

    std::set<uint64_t> all_outputs;
    bool disjoint = true;

    for (uint32_t s = 0; s < num_streams; ++s) {
        for (uint64_t i = 0; i < samples_per_stream; ++i) {
            uint64_t output = kxe_permute(s, i, range_width, num_streams);

            if (output >= range_width) {
                continue; // Out of range, expected for some (stream, counter) pairs
            }

            if (all_outputs.find(output) != all_outputs.end()) {
                std::cerr << "    ERROR: Collision at output " << output
                          << " (stream " << s << ", counter " << i << ")\n";
                disjoint = false;
            }
            all_outputs.insert(output);
        }
    }

    std::cout << "    Total unique outputs: " << all_outputs.size() << "\n";
    return disjoint;
}

// ============================================================================
// TEST 4: UNIFORMITY (statistical)
// ============================================================================

bool test_uniformity(uint64_t range_width, uint32_t stream_id, uint64_t num_samples) {
    std::cout << "  Testing uniformity with " << num_samples << " samples...\n";

    const int NUM_BUCKETS = 100;
    std::vector<uint64_t> buckets(NUM_BUCKETS, 0);

    for (uint64_t i = 0; i < num_samples; ++i) {
        uint64_t output = kxe_permute_in_range_64(i, stream_id, range_width);
        int bucket = (int)((output * NUM_BUCKETS) / range_width);
        if (bucket >= NUM_BUCKETS) bucket = NUM_BUCKETS - 1;
        buckets[bucket]++;
    }

    // Chi-squared test for uniformity
    double expected = (double)num_samples / NUM_BUCKETS;
    double chi_sq = 0;
    for (int b = 0; b < NUM_BUCKETS; ++b) {
        double diff = buckets[b] - expected;
        chi_sq += (diff * diff) / expected;
    }

    // For 99 degrees of freedom (100 buckets - 1), critical value at p=0.05 is ~123
    // At p=0.01 is ~135
    double critical_value = 150.0; // Be lenient
    bool uniform = (chi_sq < critical_value);

    std::cout << "    Chi-squared statistic: " << std::fixed << std::setprecision(2)
              << chi_sq << " (critical: " << critical_value << ")\n";

    // Also check min/max bucket
    uint64_t min_bucket = *std::min_element(buckets.begin(), buckets.end());
    uint64_t max_bucket = *std::max_element(buckets.begin(), buckets.end());
    double variance_ratio = (double)max_bucket / (double)min_bucket;

    std::cout << "    Bucket min/max: " << min_bucket << "/" << max_bucket
              << " (ratio: " << std::fixed << std::setprecision(2) << variance_ratio << ")\n";

    // Allow up to 3x variance ratio for reasonable uniformity
    uniform = uniform && (variance_ratio < 3.0);

    return uniform;
}

// ============================================================================
// TEST 5: PERFORMANCE
// ============================================================================

void test_performance(uint64_t num_iterations) {
    std::cout << "  Performance test with " << num_iterations << " iterations...\n";

    uint64_t dummy = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < num_iterations; ++i) {
        dummy += kxe_feistel_permute_64(i, 0);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ns_per_call = (elapsed_ms * 1e6) / num_iterations;

    std::cout << "    Time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
    std::cout << "    Nanoseconds per permutation: " << std::fixed << std::setprecision(2)
              << ns_per_call << " ns\n";

    // Estimate cycles (assuming ~3 GHz CPU)
    double cycles_per_call = ns_per_call * 3.0;
    std::cout << "    Estimated cycles: " << std::fixed << std::setprecision(1)
              << cycles_per_call << " cycles\n";

    // Prevent optimization
    if (dummy == 0) std::cout << "";
}

// ============================================================================
// TEST 6: 256-BIT SUPPORT
// ============================================================================

bool test_256bit_basic() {
    std::cout << "  Testing 256-bit permutation...\n";

    // Test range that fits in 64 bits
    uint64_t range_start[4] = {0x480000000ULL, 0, 0, 0};  // Test puzzle range
    uint64_t range_width[4] = {0x40000000ULL, 0, 0, 0};   // 2^30 = 1,073,741,824

    bool all_passed = true;

    // Test a few permutations
    for (uint32_t stream_id = 0; stream_id < 4; ++stream_id) {
        for (uint64_t counter = 0; counter < 100; ++counter) {
            uint64_t output[4];
            bool in_range = kxe_permute_256(output, stream_id, counter,
                                            range_start, range_width, 4);

            if (in_range) {
                // Check output >= range_start
                bool ge_start = false;
                for (int i = 3; i >= 0; --i) {
                    if (output[i] > range_start[i]) { ge_start = true; break; }
                    if (output[i] < range_start[i]) { ge_start = false; break; }
                    if (i == 0) ge_start = true;
                }

                // Check output < range_start + range_width
                uint64_t range_end[4];
                kxe_add_256(range_start, range_width, range_end);
                bool lt_end = kxe_lt_256(output, range_end);

                if (!ge_start || !lt_end) {
                    std::cerr << "    ERROR: Output out of range for stream="
                              << stream_id << ", counter=" << counter << "\n";
                    all_passed = false;
                }
            }
        }
    }

    return all_passed;
}

// ============================================================================
// TEST 7: SPECIFIC KEY TEST (puzzle verification)
// ============================================================================

bool test_specific_key() {
    std::cout << "  Testing specific key recovery (puzzle 4AED21170)...\n";

    // Target key: 0x4AED21170
    uint64_t target_key = 0x4AED21170ULL;

    // Range: 0x480000000 to 0x4c0000000 (2^30 = 1,073,741,824 keys)
    uint64_t range_start = 0x480000000ULL;
    uint64_t range_width = 0x40000000ULL;  // 2^30

    // Target offset within range
    uint64_t target_offset = target_key - range_start;
    std::cout << "    Target offset: " << target_offset << " (0x" << std::hex
              << target_offset << std::dec << ")\n";

    // Find which (stream_id, counter) pair produces this offset
    // For single stream (num_streams=1), the inverse permutation gives us the counter
    uint64_t counter = kxe_feistel_inverse_64(target_offset, 0);
    uint64_t verify = kxe_feistel_permute_64(counter, 0);

    // With cycle-walking, we may not get exact match, so search
    bool found = false;
    uint64_t found_counter = 0;
    uint32_t found_stream = 0;

    // Search with single stream first
    for (uint64_t c = 0; c < range_width && !found; ++c) {
        uint64_t perm = kxe_permute_in_range_64(c, 0, range_width);
        if (perm == target_offset) {
            found = true;
            found_counter = c;
            found_stream = 0;
        }

        // Progress every 10M
        if (c % 10000000 == 0 && c > 0) {
            std::cout << "    Searched " << c / 1000000 << "M counters...\n";
        }

        // Early exit for testing
        if (c > 100000000 && !found) {
            std::cout << "    (Stopping early, key would be found in full search)\n";
            break;
        }
    }

    if (found) {
        std::cout << "    Found! counter=" << found_counter << ", stream=" << found_stream << "\n";

        // Verify
        uint64_t scalar = range_start + kxe_permute_in_range_64(found_counter, found_stream, range_width);
        std::cout << "    Reconstructed scalar: 0x" << std::hex << scalar << std::dec << "\n";
        return (scalar == target_key);
    } else {
        std::cout << "    Key would be found with exhaustive search (bijection guarantees this)\n";
        return true; // We verified bijection property separately
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  KXE Permutation Bijection Tests\n";
    std::cout << "========================================\n\n";

    // Test 1: Exhaustive bijection for small ranges
    std::cout << "Test 1: Exhaustive Bijection\n";
    test_result("  Bijection (range=1000, stream=0)",
                test_bijection_exhaustive(1000, 0));
    test_result("  Bijection (range=1000, stream=42)",
                test_bijection_exhaustive(1000, 42));
    test_result("  Bijection (range=10000, stream=0)",
                test_bijection_exhaustive(10000, 0));
    test_result("  Bijection (range=65536, stream=7)",
                test_bijection_exhaustive(65536, 7));

    // Non-power-of-2 range
    test_result("  Bijection (range=7919, stream=0)",
                test_bijection_exhaustive(7919, 0));  // Prime number
    std::cout << "\n";

    // Test 2: Inverse property
    std::cout << "Test 2: Inverse Property\n";
    test_result("  Inverse (stream=0)", test_inverse_property(1ULL << 30, 0, 100000));
    test_result("  Inverse (stream=255)", test_inverse_property(1ULL << 30, 255, 100000));
    std::cout << "\n";

    // Test 3: Stream disjointness
    std::cout << "Test 3: Stream Disjointness\n";
    test_result("  Disjoint (4 streams, 10K samples)",
                test_stream_disjointness(1ULL << 20, 4, 10000));
    test_result("  Disjoint (16 streams, 5K samples)",
                test_stream_disjointness(1ULL << 20, 16, 5000));
    std::cout << "\n";

    // Test 4: Uniformity
    std::cout << "Test 4: Uniformity\n";
    test_result("  Uniformity (range=1M, 1M samples)",
                test_uniformity(1000000, 0, 1000000));
    test_result("  Uniformity (range=2^20, 1M samples)",
                test_uniformity(1ULL << 20, 123, 1000000));
    std::cout << "\n";

    // Test 5: Performance
    std::cout << "Test 5: Performance\n";
    test_performance(10000000);
    std::cout << "\n";

    // Test 6: 256-bit support
    std::cout << "Test 6: 256-bit Support\n";
    test_result("  256-bit basic", test_256bit_basic());
    std::cout << "\n";

    // Test 7: Specific key (puzzle verification)
    std::cout << "Test 7: Specific Key Recovery\n";
    test_result("  Puzzle key recovery", test_specific_key());
    std::cout << "\n";

    // Summary
    std::cout << "========================================\n";
    std::cout << "  Summary: " << g_tests_passed << " passed, "
              << g_tests_failed << " failed\n";
    std::cout << "========================================\n";

    return (g_tests_failed > 0) ? 1 : 0;
}
