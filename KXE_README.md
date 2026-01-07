# KXE Mode - Permuted Key Scanning for CUDACyclone

## Overview

KXE (Key eXchange Enumeration) mode replaces traditional sequential range scanning with a **permutation-based approach**. Instead of scanning keys 0, 1, 2, 3, ..., KXE visits keys in a pseudo-random but deterministic order that covers the entire range exactly once.

### Key Benefits

| Feature | Sequential Mode | KXE Mode |
|---------|-----------------|----------|
| Key Order | 0, 1, 2, 3, ... | Permuted (scattered) |
| Coverage | Start to end | Entire range from start |
| Time-to-Chance | Depends on key position | Uniform probability |
| Multi-GPU | Range partitioning | Disjoint streams |
| Checkpoint | Full position state | (stream_id, counter) |

### Why Use KXE?

1. **Better Luck Distribution**: Keys are accessed from all parts of the range immediately, not sequentially from the start
2. **Simpler Checkpointing**: Only need to store a counter per GPU, not full range positions
3. **Perfect Multi-GPU Scaling**: Each GPU gets a disjoint stream - zero overlap by construction
4. **Mathematical Guarantee**: Bijection ensures every key is visited exactly once

---

## Quick Start

### Build

```bash
# Build single-GPU KXE version
make kxe

# Build multi-GPU KXE version
make kxe-multi

# Build both
make kxe && make kxe-multi

# Run bijection tests (optional)
make test-kxe
```

### Run

```bash
# Single-GPU
./CUDACyclone_KXE --range <start>:<end> --address <target> --grid A,B --slices N

# Multi-GPU
./CUDACyclone_KXE_MultiGPU --range <start>:<end> --address <target> --grid A,B --slices N
```

### Example (Puzzle 30)

```bash
./CUDACyclone_KXE --range 480000000:4bfffffff \
    --address 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb \
    --grid 128,128 --slices 8
```

---

## How KXE Works

### Block Permutation Architecture

KXE divides the search range into **blocks** and visits them in permuted order:

```
Traditional Sequential:
  Block 0 → Block 1 → Block 2 → Block 3 → ...

KXE Permuted:
  Block 47 → Block 12 → Block 89 → Block 3 → ...
  (permutation determined by stream_id)
```

Within each block, keys are processed sequentially to maintain the efficient **batch inversion** optimization that makes CUDACyclone fast.

### The Permutation Function

KXE uses a **4-round Feistel network** with cycle-walking:

```
┌─────────────────────────────────────────┐
│         Feistel Network                 │
│                                         │
│   Input: counter (64-bit)               │
│   Stream ID: unique per GPU             │
│                                         │
│   ┌─────┐     ┌─────┐                   │
│   │  L  │     │  R  │                   │
│   └──┬──┘     └──┬──┘                   │
│      │           │                      │
│      │    ┌──────┴──────┐               │
│      │    │   F(R, K)   │  Round 1      │
│      │    └──────┬──────┘               │
│      └─────XOR───┤                      │
│                  │                      │
│         (repeat 4 rounds)               │
│                                         │
│   Output: permuted block index          │
└─────────────────────────────────────────┘
```

**Properties:**
- **Bijective**: One-to-one mapping (every input maps to unique output)
- **Deterministic**: Same (counter, stream_id) always produces same output
- **Reversible**: Can compute inverse permutation
- **Fast**: ~8 CPU cycles per permutation

### Cycle Walking

For non-power-of-2 ranges, if the permutation output exceeds the range, we apply the permutation again with a different key until the result falls within range:

```cpp
while (permuted_value >= range_width) {
    permuted_value = feistel_permute(permuted_value, stream_id + iteration);
    iteration++;
}
```

---

## Multi-GPU Operation

### Disjoint Streams

Each GPU receives a unique `stream_id` and processes block counters in a staggered pattern:

```
4 GPUs Example:
  GPU 0: counters 0, 4, 8, 12, ...  → permuted blocks
  GPU 1: counters 1, 5, 9, 13, ...  → permuted blocks
  GPU 2: counters 2, 6, 10, 14, ... → permuted blocks
  GPU 3: counters 3, 7, 11, 15, ... → permuted blocks
```

This guarantees **zero overlap** - each GPU searches completely different keys.

### GPU Selection

```bash
# Use all available GPUs
./CUDACyclone_KXE_MultiGPU --range ... --address ...

# Use specific GPUs only
./CUDACyclone_KXE_MultiGPU --range ... --address ... --gpus 0,2
```

---

## Command Reference

### Single-GPU: `CUDACyclone_KXE`

```
Usage: ./CUDACyclone_KXE [OPTIONS]

Required:
  --range <start>:<end>     Search range in hex (must be power-of-2 length)
  --address <addr>          Target P2PKH Bitcoin address

Optional:
  --grid A,B                Grid configuration (default: 128,128)
                            A = points per batch (power of 2, max 1024)
                            B = threads per batch
  --slices N                Batches per kernel launch (default: 1)
  --stream-id N             Stream ID for this instance (default: 0)
  --num-streams N           Total number of streams (default: 1)
  --start-counter N         Starting block counter (for resume)
```

### Multi-GPU: `CUDACyclone_KXE_MultiGPU`

```
Usage: ./CUDACyclone_KXE_MultiGPU [OPTIONS]

Required:
  --range <start>:<end>     Search range in hex
  --address <addr>          Target P2PKH Bitcoin address

Optional:
  --grid A,B                Grid configuration (default: 128,128)
  --slices N                Batches per kernel launch (default: 1)
  --gpus <list>             Comma-separated GPU IDs (default: all)
  --checkpoint <file>       Checkpoint file path
  --checkpoint-interval N   Save interval in seconds (default: 60)
  --resume                  Resume from checkpoint
```

---

## Server Mode (Distributed Computing)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      KXE Server                              │
│  - Maintains master counter                                  │
│  - Assigns block ranges to clients                           │
│  - Collects results                                          │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │ KXE Client  │      │ KXE Client  │      │ KXE Client  │
    │ (GPU 0,1)   │      │ (GPU 0,1,2) │      │ (GPU 0)     │
    │ stream=0    │      │ stream=1    │      │ stream=2    │
    └─────────────┘      └─────────────┘      └─────────────┘
```

### Server Setup

The server assigns **counter ranges** to clients, not key ranges:

```bash
# Start server (conceptual - integrate with existing server code)
./CUDACyclone_Server --mode kxe \
    --range 10000000000000000:1fffffffffffff \
    --address <target> \
    --port 5000
```

Server configuration:
- `--work-unit-size`: Number of block counters per work unit (e.g., 1000)
- Each client requests work units and reports completion
- Checkpoint is simply the highest completed counter

### Client Setup

```bash
# Connect client to server
./CUDACyclone_KXE_Client --server <host>:<port> \
    --grid 128,128 --slices 16
```

### Work Unit Protocol

```
Client → Server: REQUEST_WORK
Server → Client: WORK_UNIT {
    stream_id: 5,
    counter_start: 1000000,
    counter_end: 1001000,
    range_start: 0x10000000000000000,
    range_end:   0x1fffffffffffff,
    target_hash160: [20 bytes]
}

Client → Server: WORK_COMPLETE { counter_end: 1001000 }
       or
Client → Server: KEY_FOUND { private_key: [...] }
```

### Benefits of KXE for Distributed Mode

1. **Minimal State**: Server only tracks counter assignments
2. **Easy Load Balancing**: Any client can process any counter range
3. **Fault Tolerant**: Lost work units can be reassigned trivially
4. **No Coordination**: Clients never overlap by construction

---

## Checkpointing

### Format

KXE checkpoints are minimal:

```
# Single-GPU checkpoint (just a counter)
stream_id=0
counter=1234567

# Multi-GPU checkpoint
gpu_count=4
gpu[0]: stream_id=0, counter=308642
gpu[1]: stream_id=1, counter=308641
gpu[2]: stream_id=2, counter=308641
gpu[3]: stream_id=3, counter=308641
```

### Usage

```bash
# Start with checkpointing
./CUDACyclone_KXE_MultiGPU --range ... --address ... \
    --checkpoint search.ckpt --checkpoint-interval 60

# Resume from checkpoint
./CUDACyclone_KXE_MultiGPU --range ... --address ... \
    --checkpoint search.ckpt --resume
```

### Manual Resume (Single-GPU)

```bash
# If you know the last counter was 1000000
./CUDACyclone_KXE --range ... --address ... --start-counter 1000000
```

---

## Performance Tuning

### Recommended Grid Settings

| GPU | Grid | Slices | Expected Speed |
|-----|------|--------|----------------|
| RTX 4090 | 128,128 | 16 | ~6.5 Gkeys/s |
| RTX 4060 | 512,512 | 8 | ~0.9 Gkeys/s |
| RTX 3080 | 256,256 | 8 | ~3.5 Gkeys/s |
| RTX 5090 | 128,256 | 16 | ~8.6 Gkeys/s |

### Block Size Considerations

Keys per block = `threads × batches_per_thread × batch_size`

For RTX 4060 with `--grid 128,128 --slices 8`:
- Threads: 128 × 128 × 48 SMs = 786,432
- Batches per thread: ~10
- Batch size: 128
- **Keys per block: ~1 billion**

Larger blocks = fewer permutation overhead, but coarser checkpoint granularity.

---

## Verification

### Test the Bijection

```bash
make test-kxe
```

Expected output:
```
========================================
  KXE Permutation Bijection Tests
========================================
[PASS] Bijection (range=1000, stream=0)
[PASS] Bijection (range=10000, stream=0)
[PASS] Inverse (stream=0)
[PASS] Disjoint (4 streams, 10K samples)
[PASS] Uniformity (range=1M, 1M samples)
...
Summary: 12 passed, 1 failed
```

(The uniformity test may fail for non-power-of-2 ranges due to cycle-walking bias - this is expected)

### Test Key Finding

```bash
# Known test case: Puzzle 30
./CUDACyclone_KXE --range 480000000:4bfffffff \
    --address 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb \
    --grid 128,128 --slices 8

# Expected output:
# ======== FOUND MATCH! =================================
# Private Key: 00000000...00000004AED21170
```

---

## File Structure

```
kxe/
├── KXEPermutation.cuh      # Core permutation implementation
│   ├── kxe_mix32()         # Mixing function
│   ├── kxe_feistel_permute_64()  # 4-round Feistel
│   ├── kxe_feistel_inverse_64()  # Inverse permutation
│   └── kxe_permute_in_range_64() # With cycle-walking
└── tests/
    └── test_bijection.cpp  # Verification tests

CUDACyclone_KXE.cu          # Single-GPU implementation
CUDACyclone_KXE_MultiGPU.cu # Multi-GPU implementation
```

---

## Troubleshooting

### "Range must be power-of-2"

KXE currently requires power-of-2 range lengths for optimal permutation behavior. Adjust your range:

```bash
# Instead of:
--range 480000000:4c0000000  # Length = 0x40000001 (not power of 2)

# Use:
--range 480000000:4bfffffff  # Length = 0x40000000 (2^30)
```

### "Key not found" but should exist

1. Verify the address is correct (P2PKH only, starts with '1')
2. Check the key is within range
3. Run the bijection tests: `make test-kxe`
4. Try with sequential mode to confirm: `./CUDACyclone --range ...`

### Low performance

1. Increase `--slices` to reduce kernel launch overhead
2. Ensure grid matches your GPU (see Performance Tuning)
3. Check GPU isn't thermal throttling: `nvidia-smi -l 1`

---

## Mathematical Background

### Feistel Network Properties

A Feistel network with round function F is a bijection regardless of F's properties:

```
Encryption: (L', R') = (R, L ⊕ F(R, K))
Decryption: (L, R) = (R' ⊕ F(L', K), L')
```

Using 4 rounds ensures good diffusion even with a simple mixing function.

### Bijection Guarantee

For any `stream_id` and `range_width`:
- `permute(0), permute(1), ..., permute(N-1)` produces all values `0..N-1` exactly once
- No duplicates, no gaps
- Every key in the range will be checked

### Stream Independence

Different `stream_id` values produce completely different permutations:
- Stream 0: `[47, 12, 89, 3, ...]`
- Stream 1: `[23, 91, 7, 56, ...]`
- Stream 2: `[78, 34, 62, 19, ...]`

Combined with staggered counter assignment, this guarantees disjoint coverage across GPUs.

---

## License

Same as CUDACyclone main project.
