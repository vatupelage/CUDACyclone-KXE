# Understanding --unit-bits in CUDACyclone Distributed Mode

## What is --unit-bits?

The `--unit-bits` parameter controls the size of each work unit in distributed mode. A work unit is a chunk of the keyspace that gets assigned to a single client for processing.

```
Work Unit Size = 2^(unit-bits) keys
```

For example:
- `--unit-bits 36` = 2^36 = 68.7 billion keys per work unit
- `--unit-bits 40` = 2^40 = 1.1 trillion keys per work unit
- `--unit-bits 44` = 2^44 = 17.6 trillion keys per work unit

## How Work Units Are Created

The server divides the total search range into work units:

```
Total Work Units = Range Size / Work Unit Size
                 = 2^(range_bits) / 2^(unit_bits)
                 = 2^(range_bits - unit_bits)
```

**Example:** Puzzle 66 has a range of 2^66 keys
- With `--unit-bits 36`: 2^(66-36) = 2^30 = 1,073,741,824 work units (too many!)
- With `--unit-bits 43`: 2^(66-43) = 2^23 = 8,388,608 work units (good)
- With `--unit-bits 50`: 2^(66-50) = 2^16 = 65,536 work units (fewer, larger units)

## Constraints

### Maximum Work Units
The server has a limit of **10 million work units** to prevent excessive memory usage. Each work unit requires ~120 bytes of memory:

| Work Units | Server Memory |
|------------|---------------|
| 1 million  | ~120 MB       |
| 10 million | ~1.2 GB       |

### Valid Range
- **Minimum:** 28 (268 million keys per unit)
- **Maximum:** 48 (281 trillion keys per unit)

## Choosing the Right --unit-bits

### Step 1: Determine Your Range Size

| Puzzle | Range Start | Range End | Range Bits |
|--------|-------------|-----------|------------|
| 40     | 0x8000000000 | 0xFFFFFFFFFF | 40 |
| 50     | 0x20000000000000 | 0x3FFFFFFFFFFFFF | 50 |
| 66     | 0x20000000000000000 | 0x3FFFFFFFFFFFFFFFF | 66 |
| 67     | 0x40000000000000000 | 0x7FFFFFFFFFFFFFFFF | 67 |
| 68     | 0x80000000000000000 | 0xFFFFFFFFFFFFFFFF | 68 |
| 70     | 0x200000000000000000 | 0x3FFFFFFFFFFFFFFFFF | 70 |
| 71     | 0x400000000000000000 | 0x7FFFFFFFFFFFFFFFFF | 71 |

### Step 2: Calculate Minimum --unit-bits

To stay under 10 million work units:
```
unit_bits >= range_bits - 23
```

| Range Bits | Minimum --unit-bits | Work Units |
|------------|---------------------|------------|
| 40         | 28 (minimum)        | 4,096      |
| 50         | 28                  | 4.2M       |
| 60         | 37                  | 8.4M       |
| 66         | 43                  | 8.4M       |
| 67         | 44                  | 8.4M       |
| 68         | 45                  | 8.4M       |
| 70         | 47                  | 8.4M       |
| 71         | 48                  | 8.4M       |

### Step 3: Consider Work Unit Duration

Each work unit should take a reasonable time to complete. Too short = high overhead. Too long = poor load balancing.

**Recommended work unit duration: 1-10 minutes per GPU**

| GPU Speed | --unit-bits 40 | --unit-bits 44 | --unit-bits 48 |
|-----------|----------------|----------------|----------------|
| 1 Gkeys/s | 18 min         | 4.9 hours      | 78 hours       |
| 5 Gkeys/s | 3.6 min        | 58 min         | 15.6 hours     |
| 10 Gkeys/s| 1.8 min        | 29 min         | 7.8 hours      |
| 20 Gkeys/s| 55 sec         | 14.5 min       | 3.9 hours      |

## Recommended Settings by Puzzle

### Small Puzzles (40-50 bits)
```bash
# Puzzle 40-45: Use minimum
./CUDACyclone_Server --range ... --unit-bits 28

# Puzzle 46-50:
./CUDACyclone_Server --range ... --unit-bits 30
```

### Medium Puzzles (51-60 bits)
```bash
# Puzzle 51-55
./CUDACyclone_Server --range ... --unit-bits 34

# Puzzle 56-60
./CUDACyclone_Server --range ... --unit-bits 38
```

### Large Puzzles (61-70+ bits)
```bash
# Puzzle 61-65
./CUDACyclone_Server --range ... --unit-bits 42

# Puzzle 66
./CUDACyclone_Server --range 20000000000000000:3ffffffffffffffff \
    --target-hash160 ... --unit-bits 44 --kxe

# Puzzle 67
./CUDACyclone_Server --range 40000000000000000:7ffffffffffffffff \
    --target-hash160 ... --unit-bits 45 --kxe

# Puzzle 68
./CUDACyclone_Server --range 80000000000000000:fffffffffffffffff \
    --target-hash160 ... --unit-bits 46 --kxe

# Puzzle 70-71
./CUDACyclone_Server --range 400000000000000000:7fffffffffffffffff \
    --target-hash160 ... --unit-bits 48 --kxe
```

## Trade-offs

### Smaller --unit-bits (More Work Units)
**Pros:**
- Better load balancing across clients
- Finer progress tracking
- Lower impact if a client disconnects

**Cons:**
- Higher server memory usage
- More network overhead
- More checkpoint data

### Larger --unit-bits (Fewer Work Units)
**Pros:**
- Lower server memory usage
- Less network overhead
- Smaller checkpoint files

**Cons:**
- Poorer load balancing
- Coarser progress tracking
- More work lost if client disconnects

## Quick Reference Formula

```
Recommended --unit-bits = max(28, range_bits - 23)
```

This ensures:
- Work units are at least 268M keys (minimum)
- Total work units stay under 10 million
- Good balance between granularity and overhead

## Example Calculation

**Puzzle 66 (range = 2^66 keys):**

1. Range bits = 66
2. Minimum unit-bits = 66 - 23 = 43
3. With --unit-bits 43:
   - Work units = 2^(66-43) = 2^23 = 8,388,608 units
   - Keys per unit = 2^43 = 8.8 trillion keys
   - At 10 Gkeys/s: ~14.6 minutes per unit

4. With --unit-bits 46:
   - Work units = 2^(66-46) = 2^20 = 1,048,576 units
   - Keys per unit = 2^46 = 70.4 trillion keys
   - At 10 Gkeys/s: ~1.95 hours per unit

**Recommendation for Puzzle 66:** Use `--unit-bits 44` or `--unit-bits 45` for a good balance.

## Error Messages

If you see this error:
```
[WorkUnitManager] Error: Range would create 2^34 work units (max 2^32).
Increase --unit-bits to at least 38.
```

Follow the suggestion and increase --unit-bits to the recommended value.

If you see this error:
```
[WorkUnitManager] Error: Range would create 67108864 work units (max 10000000).
Increase --unit-bits to at least 43 (for ~8M units).
```

Increase --unit-bits to reduce the number of work units below 10 million.
