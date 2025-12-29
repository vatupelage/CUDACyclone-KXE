# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUDACyclone is a GPU-accelerated Bitcoin puzzle solver targeting NVIDIA GPUs. It searches for private keys within a specified range by computing secp256k1 elliptic curve points, hashing them, and comparing against a target Bitcoin address. The implementation achieves 6.5+ Gkeys/s on RTX 4090 and 8.6+ Gkeys/s on RTX 5090.

**Key characteristics:**
- Minimalist CUDA design focused on clarity
- Warp-level parallelism with batch EC operations
- Extremely low VRAM usage (~5% typical)
- Based on VanitySearch cryptographic primitives
- Fixed a critical key-skipping bug in v1.3
- **Multi-GPU support** with near-linear scaling (98.6% efficiency)

## Build System

```bash
# Build single-GPU version
make

# Build multi-GPU version
make multi

# Build both versions
make both

# Clean all
make clean
```

**Build targets:**
| Target | Binary | Description |
|--------|--------|-------------|
| `make` | `CUDACyclone` | Single-GPU version |
| `make multi` | `CUDACyclone_MultiGPU` | Multi-GPU version |
| `make both` | Both binaries | Build everything |

**Makefile architecture:**
- Auto-detects GPU compute capability via `nvidia-smi`
- Compiles for SM architectures: 75, 86, 89, and detected GPU
- Uses `-rdc=true` for separate compilation with device linking
- Optimization flags: `-O3 -use_fast_math --ptxas-options=-O3`
- Multi-GPU version links with `-lpthread` for host threading

## Running the Solver

**Basic usage:**
```bash
./CUDACyclone --range <start_hex>:<end_hex> --address <P2PKH_address> [--grid A,B] [--slices N]
```

**Critical parameters:**
- `--range`: Must be power-of-two length, start must be aligned to range length
- `--grid A,B`: A = points per batch (must be even power of 2, max 1024), B = threads per batch
- `--slices N`: Batches per thread per kernel launch (reduces thermal throttling)

**Optimal configurations:**
- RTX 4090: `--grid 128,128 --slices 16` (prevents speed degradation)
- RTX 4060: `--grid 512,512`
- RTX 5090: `--grid 128,256`

## Multi-GPU Support

**See `MULTI_GPU_README.md` for comprehensive documentation.**

**Quick start:**
```bash
# Build multi-GPU version
make multi

# Build pincer (bidirectional) version
make pincer

# Run on all available GPUs
./CUDACyclone_MultiGPU --range 8000000000:ffffffffff \
    --address 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv \
    --grid 128,128 --slices 16

# Run on specific GPUs only
./CUDACyclone_MultiGPU --range 8000000000:ffffffffff \
    --address 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv \
    --grid 128,128 --slices 16 --gpus 0,2
```

**Architecture highlights:**
- Each GPU searches independent range partition (embarrassingly parallel)
- Zero inter-GPU communication during kernel execution
- Host-side coordination via atomic flags for result detection
- Near-linear scaling: 2x GPUs = 1.98x speedup (98.6% efficiency)

**Tested performance (2x RTX 4090):**
| Configuration | Speed | Scaling |
|--------------|-------|---------|
| Single GPU | 5.76 Gkeys/s | 1.0x |
| Dual GPU | 11.4 Gkeys/s | 1.98x |

## Bidirectional Pincer Mode

**Pincer mode pairs GPUs to scan each partition from both ends simultaneously, providing a statistical 2x speedup.**

```bash
# Build pincer version
make pincer

# Run with pincer mode (requires even number of GPUs: 2, 4, 6, or 8)
./CUDACyclone_Pincer --range 100000000000:1fffffffffff \
    --address 1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk \
    --grid 128,128 --slices 16 --pincer

# With checkpoint
./CUDACyclone_Pincer --range 100000000000:1fffffffffff \
    --address 1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk \
    --grid 128,128 --slices 16 --pincer \
    --checkpoint search.ckpt --checkpoint-interval 60
```

**How pincer mode works (4 GPUs example):**
```
Partition 0:                    Partition 1:
START ──────────── MID          MID ──────────── END
   │                │             │                │
  GPU 0 ─────►  ◄── GPU 1       GPU 2 ─────►  ◄── GPU 3
  (FWD)         (BWD)           (FWD)         (BWD)
```

**Key features:**
- Forward GPU: starts at partition start, jumps +B each batch
- Backward GPU: starts at partition end, jumps -B each batch (uses negated jump point)
- Statistical 2x speedup: on average, key is found in 25% of range vs 50%
- Direction-aware checkpointing for correct resume

**Tested performance (4x RTX 4090):**
| Mode | Speed | Effective Coverage |
|------|-------|-------------------|
| Standard | 23.2 Gkeys/s | 23.2 Gkeys/s |
| Pincer | 23.2 Gkeys/s | **46.4 Gkeys/s** (2x effective) |

## Checkpoint/Resume System

The multi-GPU version supports checkpoint/resume for long-running searches:

```bash
# Start search with checkpointing (saves every 60 seconds)
./CUDACyclone_MultiGPU --range 8000000000:ffffffffff \
    --address 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv \
    --grid 128,128 --slices 16 \
    --checkpoint search.ckpt --checkpoint-interval 60

# Resume from checkpoint
./CUDACyclone_MultiGPU --range 8000000000:ffffffffff \
    --address 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv \
    --grid 128,128 --slices 16 \
    --checkpoint search.ckpt --resume
```

**Checkpoint features:**
- Automatic periodic saves at configurable intervals
- Checkpoint saved on Ctrl+C interrupt
- Atomic file writes (prevents corruption)
- Validates parameters on resume
- Binary format (~400-500 bytes for 2 GPUs)

## Code Architecture

### File Structure

**Core CUDA files:**
- `CUDACyclone.cu`: Single-GPU main entry point, host-side orchestration, kernel launch
- `CUDACyclone_MultiGPU.cu`: Multi-GPU implementation with GPUContext and worker threads
- `CUDAHash.cu`: SHA-256, RIPEMD-160, Hash160 device implementations
- `CUDAMath.h`: Secp256k1 field arithmetic (modular operations)
- `CUDAStructures.h`: Device constants, result structures, flags
- `CUDAUtils.h`: Host/device utilities for 256-bit arithmetic, formatting
- `CUDAHash.cuh`: Hash function headers

**Support files:**
- `Makefile`: Build configuration with multi-architecture support
- `proof.py`: Verification script that tests for key skipping
- `sha256.h`: Standard SHA-256 constants
- `MULTI_GPU_README.md`: Comprehensive multi-GPU documentation

### Kernel Architecture: `kernel_point_add_and_check_oneinv`

This is the single performance-critical kernel. Understanding it is essential for modifications.

**Execution model:**
- Each thread processes 4×64-bit scalar values (256-bit range subdivision)
- Threads process batches of EC points using batch inversion optimization
- Supports "sliced" execution: multiple batches per kernel launch (controlled by `--slices`)

**Batch inversion algorithm:**
1. Precomputes G, 2G, 3G, ..., (batch_size/2)G in constant memory
2. For each batch of size B:
   - Compute denominator products: (G.x - P.x), (2G.x - P.x), ...
   - Perform single modular inversion on accumulated product
   - Back-substitute to get all individual inverses
   - Compute batch_size EC additions using stored precomputed points
3. Checks Hash160 after each point computation (early exit on match)

**Critical data flow:**
- Input: `d_start_scalars[]`, `d_Px[]`, `d_Py[]`, `d_counts256[]` (per-thread remaining work)
- Output: `d_Rx[]`, `d_Ry[]` (updated point coordinates), `d_found_result` (if match)
- Shared: `d_found_flag` (atomic match signaling), `d_hashes_accum` (performance counter)

**Performance optimizations:**
- Warp-level synchronization for early exit (`__any_sync`, `__shfl_sync`)
- Warp-level hash counter reduction (minimizes atomic operations)
- Constant memory for precomputed points (`c_Gx`, `c_Gy`, `c_Jx`, `c_Jy`)
- In-register batch processing (minimal global memory traffic)

### Secp256k1 Implementation Details

**Coordinate system:** Affine coordinates (X, Y) with modular inversion

**Field arithmetic (in CUDAMath.h):**
- Prime field P = 2^256 - 2^32 - 977 (secp256k1 field)
- 5-limb representation: 4×64-bit + 1×64-bit extended (for carry chains)
- Assembly-level operations: `UADDO`, `UADDC`, `UMULLO`, `MADDO` (PTX instructions)
- `_ModMult`: 256×256→512-bit multiply with Barrett-like reduction
- `_ModSqr`: Optimized squaring (avoids redundant multiplications)
- `_ModInv`: Binary Extended GCD using half-delta reduction (62-bit steps)

**Critical modular operations:**
- `ModSub256`: Conditional addition of P on borrow
- `ModNeg256`: Compute P - x
- `ModSub256isOdd`: Returns parity without full computation (for compressed pubkey prefix)

### Hash Pipeline

**Hash160 = RIPEMD160(SHA256(compressed_pubkey))**

**Optimized path in kernel:**
1. `getHash160_33_from_limbs()`: Takes 4×64-bit X coordinate + prefix byte
2. `SHA256_33_from_limbs()`: Packs 33-byte pubkey directly into SHA-256 message schedule
3. `RIPEMD160_from_SHA256_state()`: Feeds SHA state to RIPEMD without intermediate buffer
4. Prefix match check: `hash160_prefix_equals()` (fast 4-byte LE comparison)
5. Full match check: `hash160_matches_prefix_then_full()` (remaining 16 bytes)

**Why this matters:** The hash computation is called 2×(batch_size/2) times per batch, making it the second-most-critical code path after modular multiplication.

### Memory Layout and Constant Memory Usage

**Constant memory (per-SM, read-only, cached):**
- `c_Gx[MAX_BATCH_SIZE/2 * 4]`: X-coordinates of G, 2G, ..., (batch_size/2)G
- `c_Gy[MAX_BATCH_SIZE/2 * 4]`: Y-coordinates
- `c_Jx[4]`, `c_Jy[4]`: Jump point = batch_size × G
- `c_target_hash160[20]`: Target address hash
- `c_target_prefix`: First 4 bytes as uint32 (LE) for fast filtering

**Why constant memory:**
- Prevents thermal throttling from VRAM bandwidth saturation
- Broadcast to all threads in warp simultaneously
- Critical for achieving 6+ Gkeys/s sustained performance

### Initialization Phase (Host Code)

Before kernel launch, main() performs:

1. **Precomputation of batch points:**
   - Calls `scalarMulKernelBase()` to compute G×1, G×2, ..., G×(batch_size/2)
   - Uploads results to constant memory (`c_Gx`, `c_Gy`)

2. **Per-thread starting points:**
   - Divides range into `threadsTotal` equal subdivisions
   - For each thread, computes starting scalar S = range_start + thread_id × per_thread_count + batch_size/2
   - Calls `scalarMulKernelBase()` to compute S×G for all threads
   - Starting at "half" avoids edge cases in batch subtraction operations

3. **Range validation:**
   - Enforces power-of-two range length
   - Enforces alignment: `range_start % range_length == 0`
   - Enforces divisibility by batch size and thread count

## Key Skipping Bug (FIXED in v1.3)

**Historical issue:** Earlier versions skipped keys due to incorrect arithmetic in the batch inversion loop or improper handling of subtraction operations.

**Verification:** Run `proof.py` to test coverage:
```bash
python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512
```

**What it tests:**
- Range start/end with both parities (even/odd offsets)
- Full modulo-batch_size residue coverage
- Random keys across all quartiles of range
- Should report 0 failures on working implementation

## Common Development Tasks

### Changing Batch Size Maximum

Edit `CUDACyclone.cu:41`:
```cuda
#define MAX_BATCH_SIZE 1024
```
Recompile. Note: Increasing beyond 1024 increases constant memory usage and may hurt occupancy.

### Modifying Field Arithmetic

**Location:** `CUDAMath.h`

**Important:** Changes to `_ModMult`, `_ModSqr`, or `_ModInv` affect correctness. Always:
1. Test with proof.py after changes
2. Verify against known test vectors (e.g., secp256k1 generator point operations)
3. Check that reduction maintains canonical form (result < P)

### Adding Support for Other Address Types

Currently supports P2PKH only. To add P2WPKH or P2SH:

1. Modify `decode_p2pkh_address()` in `CUDAUtils.h` to handle different prefixes
2. Adjust hash computation if needed (P2WPKH uses same Hash160)
3. Update command-line parsing to accept address type flag

### Performance Profiling

**Monitor kernel execution:**
- Use `nvprof ./CUDACyclone ...` (legacy) or Nsight Compute
- Key metrics: achieved occupancy, SM utilization, L1 cache hit rate
- Watch for memory throttling (indicates constant memory overflow)

**In-application metrics:**
- Speed (Mkeys/s): Total hashes computed per second
- Progress: Percentage of range completed
- Memory utilization: Percentage of GPU VRAM used

### Debugging Match Detection

**Device-side debugging is limited.** If matches aren't detected:

1. Check `d_found_flag` state machine:
   - FOUND_NONE (0) → FOUND_LOCK (1) → FOUND_READY (2)
   - Only first thread to atomicCAS(FOUND_NONE→FOUND_LOCK) wins

2. Verify hash computation with known inputs:
   - Temporarily modify kernel to print intermediate SHA/RIPEMD states
   - Compare against Python ecdsa library outputs

3. Use proof.py to inject known keys into range

### Scalar Multiplication Implementation

**For initial point computation only** (not in hot path):

`scalarMulBaseAffine()` in `CUDAMath.h` uses double-and-add:
- MSB-first binary decomposition
- Affine point doubling and addition
- Slow but correct for one-time precomputation

**Not used in kernel:** Kernel uses batch addition only, avoiding point multiplication entirely.

## Important Constraints

1. **Range must be power-of-two length** - Algorithm assumes subdivision by powers of 2
2. **Batch size must be even** - Algorithm computes positive/negative points symmetrically
3. **Range start must be aligned** - Ensures per-thread ranges don't overlap
4. **Thread count divisibility** - `total_batches % threadsTotal == 0` required
5. **No backwards compatibility layers** - Code assumes CUDA compute capability ≥ 7.5

## Architectural Decisions

**Why affine coordinates?**
- Batch inversion amortizes cost: 1 ModInv per batch_size additions
- Simpler than Jacobian coordinates for memory-constrained kernels

**Why constant memory for precomputed points?**
- Moved from global memory in v1.1 due to thermal throttling
- Broadcasting reduces memory transactions by 32× (warp size)

**Why sliced kernel execution?**
- Single large kernel launch causes GPU boost clock drops
- Multiple smaller slices maintain higher clock speeds
- Trade-off: slight launch overhead vs. sustained throughput

**Why big-endian storage for elliptic curve coordinates?**
- Matches Bitcoin's serialization format
- Simplifies public key formatting
- Minimal conversion overhead in hash functions
