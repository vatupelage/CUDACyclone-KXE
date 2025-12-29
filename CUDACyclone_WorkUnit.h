// ============================================================================
// CUDACyclone Distributed Mode - Work Unit Management
// ============================================================================
// Utilities for managing power-of-2 work units in distributed mode
// ============================================================================

#ifndef CUDACYCLONE_WORKUNIT_H
#define CUDACYCLONE_WORKUNIT_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

#include "CUDACyclone_Protocol.h"

// ============================================================================
// 256-BIT ARITHMETIC UTILITIES (Host-side only)
// ============================================================================

namespace arith256 {

// Check if a 256-bit value is zero
inline bool is_zero(const uint64_t v[4]) {
    return (v[0] | v[1] | v[2] | v[3]) == 0;
}

// Compare two 256-bit values: returns -1 if a<b, 0 if a==b, 1 if a>b
inline int compare(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// a >= b
inline bool gte(const uint64_t a[4], const uint64_t b[4]) {
    return compare(a, b) >= 0;
}

// a <= b
inline bool lte(const uint64_t a[4], const uint64_t b[4]) {
    return compare(a, b) <= 0;
}

// Copy 256-bit value
inline void copy(const uint64_t src[4], uint64_t dst[4]) {
    memcpy(dst, src, 32);
}

// Set to zero
inline void zero(uint64_t v[4]) {
    memset(v, 0, 32);
}

// Set from u64
inline void set_u64(uint64_t v[4], uint64_t val) {
    v[0] = val;
    v[1] = v[2] = v[3] = 0;
}

// Add two 256-bit values
inline void add(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    __uint128_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t s = (__uint128_t)a[i] + b[i] + carry;
        out[i] = (uint64_t)s;
        carry = s >> 64;
    }
}

// Add 64-bit value to 256-bit
inline void add_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
    __uint128_t sum = (__uint128_t)a[0] + b;
    out[0] = (uint64_t)sum;
    uint64_t carry = (uint64_t)(sum >> 64);
    for (int i = 1; i < 4; ++i) {
        sum = (__uint128_t)a[i] + carry;
        out[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
}

// Subtract: out = a - b (assumes a >= b)
inline void sub(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t bi = b[i] + borrow;
        if (a[i] < bi) {
            out[i] = (uint64_t)(((__uint128_t)1 << 64) + a[i]) - bi;
            borrow = 1;
        } else {
            out[i] = a[i] - bi;
            borrow = 0;
        }
    }
}

// Subtract 64-bit from 256-bit
inline void sub_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
    uint64_t borrow = (a[0] < b) ? 1 : 0;
    out[0] = a[0] - b;
    for (int i = 1; i < 4; ++i) {
        uint64_t ai = a[i];
        out[i] = ai - borrow;
        borrow = (ai < borrow) ? 1 : 0;
    }
}

// Increment by 1
inline void inc(uint64_t v[4]) {
    for (int i = 0; i < 4; ++i) {
        if (++v[i] != 0) break;
    }
}

// Decrement by 1
inline void dec(uint64_t v[4]) {
    for (int i = 0; i < 4; ++i) {
        if (v[i]-- != 0) break;
    }
}

// Left shift by n bits (n must be 0-255)
inline void shl(const uint64_t a[4], int n, uint64_t out[4]) {
    if (n == 0) {
        copy(a, out);
        return;
    }
    if (n >= 256) {
        zero(out);
        return;
    }

    int word_shift = n / 64;
    int bit_shift = n % 64;

    zero(out);

    if (bit_shift == 0) {
        for (int i = word_shift; i < 4; ++i) {
            out[i] = a[i - word_shift];
        }
    } else {
        for (int i = word_shift; i < 4; ++i) {
            out[i] = a[i - word_shift] << bit_shift;
            if (i > word_shift) {
                out[i] |= a[i - word_shift - 1] >> (64 - bit_shift);
            }
        }
    }
}

// Right shift by n bits
inline void shr(const uint64_t a[4], int n, uint64_t out[4]) {
    if (n == 0) {
        copy(a, out);
        return;
    }
    if (n >= 256) {
        zero(out);
        return;
    }

    int word_shift = n / 64;
    int bit_shift = n % 64;

    zero(out);

    if (bit_shift == 0) {
        for (int i = 0; i < 4 - word_shift; ++i) {
            out[i] = a[i + word_shift];
        }
    } else {
        for (int i = 0; i < 4 - word_shift; ++i) {
            out[i] = a[i + word_shift] >> bit_shift;
            if (i + word_shift + 1 < 4) {
                out[i] |= a[i + word_shift + 1] << (64 - bit_shift);
            }
        }
    }
}

// Multiply by power of 2 (same as left shift)
inline void mul_pow2(const uint64_t a[4], int bits, uint64_t out[4]) {
    shl(a, bits, out);
}

// Divide by power of 2 (same as right shift)
inline void div_pow2(const uint64_t a[4], int bits, uint64_t out[4]) {
    shr(a, bits, out);
}

// Count leading zeros
inline int clz(const uint64_t v[4]) {
    for (int i = 3; i >= 0; --i) {
        if (v[i] != 0) {
            return (3 - i) * 64 + __builtin_clzll(v[i]);
        }
    }
    return 256;
}

// Get bit width (position of highest set bit + 1)
inline int bit_width(const uint64_t v[4]) {
    return 256 - clz(v);
}

// Check if value is a power of 2
inline bool is_power_of_2(const uint64_t v[4]) {
    // Count set bits - should be exactly 1
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        count += __builtin_popcountll(v[i]);
        if (count > 1) return false;
    }
    return count == 1;
}

// Get log2 of value (assumes it's a power of 2)
inline int log2_pow2(const uint64_t v[4]) {
    return bit_width(v) - 1;
}

// Divide and get quotient (assumes divisor fits in 64 bits and is non-zero)
inline void div_u64(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4]) {
    uint64_t remainder = 0;
    for (int i = 3; i >= 0; --i) {
        __uint128_t cur = ((__uint128_t)remainder << 64) | value[i];
        quotient[i] = (uint64_t)(cur / divisor);
        remainder = (uint64_t)(cur % divisor);
    }
}

// Multiply 256-bit by 64-bit
inline void mul_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
    __uint128_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t prod = (__uint128_t)a[i] * b + carry;
        out[i] = (uint64_t)prod;
        carry = prod >> 64;
    }
    // Note: overflow is discarded
}

// Convert to hex string
inline std::string to_hex(const uint64_t v[4]) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (int i = 3; i >= 0; --i) {
        oss << std::setw(16) << v[i];
    }
    // Remove leading zeros but keep at least one digit
    std::string s = oss.str();
    size_t pos = s.find_first_not_of('0');
    return (pos == std::string::npos) ? "0" : s.substr(pos);
}

// Parse from hex string
inline bool from_hex(const std::string& hex, uint64_t out[4]) {
    std::string h = hex;
    if (h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) {
        h = h.substr(2);
    }
    if (h.size() > 64) return false;

    // Pad to 64 characters
    while (h.size() < 64) h = "0" + h;

    for (int i = 0; i < 4; ++i) {
        std::string part = h.substr(i * 16, 16);
        out[3 - i] = std::stoull(part, nullptr, 16);
    }
    return true;
}

// Convert to long double for approximate calculations
inline long double to_ld(const uint64_t v[4]) {
    return std::ldexp((long double)v[3], 192) +
           std::ldexp((long double)v[2], 128) +
           std::ldexp((long double)v[1], 64) +
           (long double)v[0];
}

} // namespace arith256

// ============================================================================
// WORK UNIT STRUCTURE
// ============================================================================

struct WorkUnit {
    uint32_t unit_id;               // Sequential ID
    uint64_t range_start[4];        // 256-bit start (inclusive)
    uint64_t range_end[4];          // 256-bit end (inclusive)
    WorkUnitState state;            // Current state
    uint32_t assigned_client_id;    // Which client is processing
    uint64_t assigned_at;           // Unix timestamp when assigned
    uint64_t completed_at;          // Unix timestamp when completed
    uint64_t keys_processed;        // Progress within unit
    uint32_t reassign_count;        // Number of times reassigned
    ScanDirection direction;        // For pincer mode
    uint32_t pincer_partner_id;     // Partner unit ID (if pincer)

    WorkUnit() {
        unit_id = 0;
        arith256::zero(range_start);
        arith256::zero(range_end);
        state = WorkUnitState::AVAILABLE;
        assigned_client_id = 0;
        assigned_at = 0;
        completed_at = 0;
        keys_processed = 0;
        reassign_count = 0;
        direction = ScanDirection::FORWARD;
        pincer_partner_id = 0;
    }

    // Get size of this unit (end - start + 1)
    void get_size(uint64_t out[4]) const {
        arith256::sub(range_end, range_start, out);
        arith256::inc(out);
    }

    // Check if this unit is complete (all keys processed)
    bool is_fully_processed() const {
        uint64_t size[4];
        get_size(size);
        // Compare keys_processed with size
        uint64_t processed[4] = {keys_processed, 0, 0, 0};
        return arith256::gte(processed, size);
    }

    // Get completion percentage
    double get_progress_percent() const {
        uint64_t size[4];
        get_size(size);
        long double total = arith256::to_ld(size);
        return (total > 0) ? (100.0 * keys_processed / total) : 100.0;
    }
};

// ============================================================================
// WORK UNIT MANAGER
// ============================================================================

class WorkUnitManager {
public:
    WorkUnitManager() : unit_bits_(DEFAULT_WORK_UNIT_BITS), total_units_(0) {
        arith256::zero(range_start_);
        arith256::zero(range_end_);
    }

    // Initialize work units for a range
    // Returns false if range is invalid or not properly aligned
    bool initialize(const uint64_t range_start[4],
                   const uint64_t range_end[4],
                   uint32_t unit_bits) {
        // Validate unit_bits
        if (unit_bits < MIN_WORK_UNIT_BITS || unit_bits > MAX_WORK_UNIT_BITS) {
            return false;
        }

        // Calculate range length
        uint64_t range_len[4];
        arith256::sub(range_end, range_start, range_len);
        arith256::inc(range_len);  // Inclusive

        // Check if range_len is power of 2
        if (!arith256::is_power_of_2(range_len)) {
            return false;
        }

        // Check if range length is at least one work unit
        int range_bits = arith256::log2_pow2(range_len);
        if (range_bits < (int)unit_bits) {
            // Range is smaller than one unit - use range as single unit
            unit_bits = range_bits;
        }

        // Calculate number of work units
        uint32_t num_units = 1U << (range_bits - unit_bits);

        // Store configuration
        arith256::copy(range_start, range_start_);
        arith256::copy(range_end, range_end_);
        unit_bits_ = unit_bits;
        total_units_ = num_units;

        // Generate work units
        units_.clear();
        units_.reserve(num_units);

        uint64_t unit_size[4];
        arith256::set_u64(unit_size, 1);
        arith256::shl(unit_size, unit_bits, unit_size);

        uint64_t current[4];
        arith256::copy(range_start, current);

        for (uint32_t i = 0; i < num_units; ++i) {
            WorkUnit wu;
            wu.unit_id = i;
            arith256::copy(current, wu.range_start);

            // end = start + unit_size - 1
            arith256::add(current, unit_size, wu.range_end);
            arith256::dec(wu.range_end);

            wu.state = WorkUnitState::AVAILABLE;

            // Set up pincer pairing (alternate forward/backward)
            wu.direction = (i % 2 == 0) ? ScanDirection::FORWARD : ScanDirection::BACKWARD;
            wu.pincer_partner_id = (i % 2 == 0) ? (i + 1) : (i - 1);
            if (wu.pincer_partner_id >= num_units) {
                wu.pincer_partner_id = i;  // No partner if at end
            }

            units_.push_back(wu);

            // Move to next unit
            arith256::add(current, unit_size, current);
        }

        return true;
    }

    // Get number of work units
    uint32_t get_total_units() const { return total_units_; }

    // Get unit bits (log2 of unit size)
    uint32_t get_unit_bits() const { return unit_bits_; }

    // Get a work unit by ID
    WorkUnit* get_unit(uint32_t id) {
        if (id >= units_.size()) return nullptr;
        return &units_[id];
    }

    const WorkUnit* get_unit(uint32_t id) const {
        if (id >= units_.size()) return nullptr;
        return &units_[id];
    }

    // Get next available work unit
    WorkUnit* get_next_available() {
        for (auto& wu : units_) {
            if (wu.state == WorkUnitState::AVAILABLE) {
                return &wu;
            }
        }
        return nullptr;
    }

    // Get next available work unit, preferring pincer pairs
    WorkUnit* get_next_available_pincer_aware(bool prefer_partner_assigned) {
        // First, look for units whose partners are already assigned
        if (prefer_partner_assigned) {
            for (auto& wu : units_) {
                if (wu.state == WorkUnitState::AVAILABLE) {
                    WorkUnit* partner = get_unit(wu.pincer_partner_id);
                    if (partner && partner->state == WorkUnitState::ASSIGNED) {
                        return &wu;
                    }
                }
            }
        }

        // Otherwise, return first available
        return get_next_available();
    }

    // Count units by state
    uint32_t count_by_state(WorkUnitState state) const {
        uint32_t count = 0;
        for (const auto& wu : units_) {
            if (wu.state == state) ++count;
        }
        return count;
    }

    // Get total keys processed across all units
    uint64_t get_total_keys_processed() const {
        uint64_t total = 0;
        for (const auto& wu : units_) {
            total += wu.keys_processed;
        }
        return total;
    }

    // Check if all units are completed
    bool is_all_completed() const {
        for (const auto& wu : units_) {
            if (wu.state != WorkUnitState::COMPLETED &&
                wu.state != WorkUnitState::VERIFIED) {
                return false;
            }
        }
        return true;
    }

    // Expire stale assignments
    void expire_stale(uint64_t timeout_sec, uint64_t current_time) {
        for (auto& wu : units_) {
            if (wu.state == WorkUnitState::ASSIGNED) {
                if (current_time - wu.assigned_at > timeout_sec) {
                    wu.state = WorkUnitState::EXPIRED;
                    wu.reassign_count++;
                }
            }
        }
    }

    // Reset expired units to available (if not too many reassigns)
    void reset_expired(uint32_t max_reassigns) {
        for (auto& wu : units_) {
            if (wu.state == WorkUnitState::EXPIRED) {
                if (wu.reassign_count < max_reassigns) {
                    wu.state = WorkUnitState::AVAILABLE;
                }
            }
        }
    }

    // Get all units assigned to a client
    std::vector<uint32_t> get_units_for_client(uint32_t client_id) const {
        std::vector<uint32_t> result;
        for (const auto& wu : units_) {
            if (wu.assigned_client_id == client_id &&
                wu.state == WorkUnitState::ASSIGNED) {
                result.push_back(wu.unit_id);
            }
        }
        return result;
    }

    // Release all units assigned to a client (e.g., on disconnect)
    void release_client_units(uint32_t client_id) {
        for (auto& wu : units_) {
            if (wu.assigned_client_id == client_id &&
                wu.state == WorkUnitState::ASSIGNED) {
                wu.state = WorkUnitState::AVAILABLE;
                wu.assigned_client_id = 0;
                wu.assigned_at = 0;
                // Keep keys_processed for partial progress
            }
        }
    }

    // Access to units vector
    const std::vector<WorkUnit>& get_units() const { return units_; }
    std::vector<WorkUnit>& get_units() { return units_; }

    // Get range info
    const uint64_t* get_range_start() const { return range_start_; }
    const uint64_t* get_range_end() const { return range_end_; }

private:
    uint64_t range_start_[4];
    uint64_t range_end_[4];
    uint32_t unit_bits_;
    uint32_t total_units_;
    std::vector<WorkUnit> units_;
};

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

// Validate range is power-of-2 aligned
inline bool validate_range(const uint64_t start[4], const uint64_t end[4]) {
    // end must be >= start
    if (arith256::compare(end, start) < 0) return false;

    // Calculate length
    uint64_t len[4];
    arith256::sub(end, start, len);
    arith256::inc(len);

    // Must be power of 2
    if (!arith256::is_power_of_2(len)) return false;

    // Start must be aligned to length
    // This means: start % len == 0
    // For power-of-2 len, this is: start & (len-1) == 0
    uint64_t len_minus_1[4];
    arith256::copy(len, len_minus_1);
    arith256::dec(len_minus_1);

    uint64_t masked[4];
    for (int i = 0; i < 4; ++i) {
        masked[i] = start[i] & len_minus_1[i];
    }

    return arith256::is_zero(masked);
}

// Calculate optimal unit bits based on expected total time
inline uint32_t calc_optimal_unit_bits(const uint64_t range_len[4],
                                       double target_unit_time_sec,
                                       double expected_speed_gkeys) {
    // Calculate range bits
    int range_bits = arith256::bit_width(range_len);

    // Target keys per unit = speed * time
    double target_keys = expected_speed_gkeys * 1e9 * target_unit_time_sec;

    // Calculate bits
    int unit_bits = (int)std::log2(target_keys);

    // Clamp to valid range
    unit_bits = std::max((int)MIN_WORK_UNIT_BITS, unit_bits);
    unit_bits = std::min((int)MAX_WORK_UNIT_BITS, unit_bits);
    unit_bits = std::min(range_bits, unit_bits);

    return (uint32_t)unit_bits;
}

#endif // CUDACYCLONE_WORKUNIT_H
