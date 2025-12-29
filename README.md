# CUDACyclone: GPU Satoshi Puzzle Solver

A high-performance CUDA-based Bitcoin puzzle solver for NVIDIA GPUs. Features **Multi-GPU support**, **Bidirectional Pincer Scanning**, and **Checkpoint/Resume** functionality.

Leveraging **CUDA**, **warp-level parallelism**, and **batch EC operations**, CUDACyclone pushes the limits of cryptographic key search.

Secp256k1 math is based on the excellent work from [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch) and [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul) with major CUDA-specific modifications.

---

## Performance

| Configuration | Speed | Notes |
|--------------|-------|-------|
| 1x RTX 4090 | 6.5 Gkeys/s | Single GPU |
| 1x RTX 5090 | 8.6 Gkeys/s | Single GPU |
| 2x RTX 4090 | 11.4 Gkeys/s | Multi-GPU (98.6% scaling) |
| 4x RTX 4090 | 23.1 Gkeys/s | Pincer Mode |

---

## Key Features

- **Multi-GPU Support**: Near-linear scaling across multiple GPUs
- **Bidirectional Pincer Scanning**: GPUs scan from both ends of partitions for 2x average speedup
- **Checkpoint/Resume**: Save progress and resume interrupted searches
- **Massive Parallelism**: Tens of thousands of threads computing EC points and Hash160 simultaneously
- **Extremely Low VRAM**: ~5% VRAM usage typical
- **Cross Architecture**: Auto-compiles for SM 75, 86, 89, and detected GPU

---

## Build

```bash
# Build single-GPU version
make

# Build multi-GPU version
make multi

# Build pincer (bidirectional) version
make pincer

# Build all versions
make everything

# Clean
make clean
```

**Build Targets:**

| Command | Binary | Description |
|---------|--------|-------------|
| `make` | `CUDACyclone` | Single-GPU version |
| `make multi` | `CUDACyclone_MultiGPU` | Multi-GPU with checkpointing |
| `make pincer` | `CUDACyclone_Pincer` | Bidirectional pincer scanning |
| `make everything` | All binaries | Build all versions |

---

## Usage

### Single GPU

```bash
./CUDACyclone --range <start_hex>:<end_hex> --address <P2PKH_address> [--grid A,B] [--slices N]
```

### Multi-GPU

```bash
./CUDACyclone_MultiGPU --range <start_hex>:<end_hex> --address <P2PKH_address> \
    [--grid A,B] [--slices N] [--gpus 0,1,2,...] \
    [--checkpoint <file>] [--checkpoint-interval <seconds>] [--resume]
```

### Pincer Mode (Bidirectional)

```bash
./CUDACyclone_Pincer --range <start_hex>:<end_hex> --address <P2PKH_address> \
    [--grid A,B] [--slices N] [--gpus 0,1,2,...] --pincer \
    [--checkpoint <file>] [--checkpoint-interval <seconds>] [--resume]
```

---

## Options

| Option | Description |
|--------|-------------|
| `--range` | Search range in hex (must be power of two length) |
| `--address` | Target P2PKH Bitcoin address |
| `--target-hash160` | Target as Hash160 hex (alternative to address) |
| `--grid A,B` | A = points per batch, B = threads per batch |
| `--slices N` | Batches per thread per kernel launch |
| `--gpus 0,1,...` | Specific GPU IDs to use (multi-GPU only) |
| `--pincer` | Enable bidirectional scanning (pincer binary only) |
| `--checkpoint <file>` | Checkpoint file path |
| `--checkpoint-interval N` | Save interval in seconds (default: 60) |
| `--resume` | Resume from checkpoint |

---

## Optimal Grid Settings

| GPU | Recommended Settings |
|-----|---------------------|
| RTX 4090 | `--grid 128,128 --slices 16` |
| RTX 5090 | `--grid 128,256` |
| RTX 4060 | `--grid 512,512` |
| RTX 4070 Ti | `--grid 512,1024` |

---

## Bidirectional Pincer Mode

Pincer mode pairs GPUs to scan each partition from both ends simultaneously:

```
Partition 0:
  GPU 0 (FORWARD)  ────────►  ◄──────── GPU 1 (BACKWARD)

Partition 1:
  GPU 2 (FORWARD)  ────────►  ◄──────── GPU 3 (BACKWARD)
```

**Benefits:**
- Statistical 2x speedup: Keys found after searching 25% of range on average (vs 50%)
- Requires even number of GPUs (2, 4, 6, etc.)
- Each partition is fully covered by its GPU pair

**Example (4x RTX 4090):**

```bash
./CUDACyclone_Pincer --range 100000000000:1fffffffffff \
    --address 1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk \
    --grid 128,128 --slices 16 --pincer
```

Output:
```
======== Multi-GPU Configuration (PINCER MODE) =====
Number of GPUs      : 4
Partitions          : 2
Scan mode           : Bidirectional (Pincer)

======== GPU Initialization ===========================
Partition 0 [0000100000000000]:
  GPU 0 (NVIDIA GeForce RTX 4090) FORWARD:  4194304 threads, 4.8% VRAM
  GPU 1 (NVIDIA GeForce RTX 4090) BACKWARD: 4194304 threads, 4.8% VRAM
Partition 1 [0000180000000000]:
  GPU 2 (NVIDIA GeForce RTX 4090) FORWARD:  4194304 threads, 4.8% VRAM
  GPU 3 (NVIDIA GeForce RTX 4090) BACKWARD: 4194304 threads, 4.8% VRAM

======== Phase-1: Multi-GPU BruteForce (PINCER) ========
Time: 201.2 s | Speed: 23106.4 Mkeys/s | Count: 4659984652416 | Progress: 26.49 % | GPUs: 4 (P)

======== FOUND MATCH! =================================
Found by GPU        : 0 (FORWARD, Partition 0)
Private Key         : 0000000000000000000000000000000000000000000000000000122FCA143C05
```

---

## Checkpoint/Resume

Save progress for long-running searches:

```bash
# Start with checkpointing (saves every 60 seconds)
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

**Features:**
- Automatic periodic saves
- Checkpoint saved on Ctrl+C
- Atomic writes (prevents corruption)
- Parameter validation on resume
- Works with both multi-GPU and pincer modes

---

## Example Output

**Single GPU (RTX 4090):**
```bash
./CUDACyclone --range 200000000000:3fffffffffff --address 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP --grid 128,128 --slices 16

======== PrePhase: GPU Information ====================
Device               : NVIDIA GeForce RTX 4090 (compute 8.9)
SM                   : 128
Memory utilization   : 4.8% (1.14 GB / 23.6 GB)
Total threads        : 4194304

======== Phase-1: BruteForce (sliced) =================
Time: 393.7 s | Speed: 6127.4 Mkeys/s | Count: 2421341587872 | Progress: 6.88 %

======== FOUND MATCH! =================================
Private Key   : 00000000000000000000000000000000000000000000000000002EC18388D544
Public Key    : 03FD5487722D2576CB6D7081426B66A3E2986C1CE8358D479063FB5F2BB6DD5849
```

**Multi-GPU (2x RTX 4090):**
```bash
./CUDACyclone_MultiGPU --range 8000000000:ffffffffff --address 1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv --grid 128,128 --slices 16

======== Multi-GPU Configuration =====
Number of GPUs      : 2
GPU IDs             : 0 1

======== Phase-1: Multi-GPU BruteForce ========
Time: 10.0 s | Speed: 11423.5 Mkeys/s | Count: 114235000000 | Progress: 21.3 %
```

---

## Community Benchmarks

| GPU | Grid | Speed | Notes |
|-----|------|-------|-------|
| RTX 4090 | 128,128 --slices 16 | 6.5 Gkeys/s | Optimal settings |
| RTX 5090 | 128,256 | 8.6 Gkeys/s | Latest gen |
| RTX 4060 | 512,512 | 1.2 Gkeys/s | Budget option |
| RTX 4070 Ti Super | 512,1024 | 3.2 Gkeys/s | Mid-range |
| RTX 3070 Mobile | 256,256 | 1.1 Gkeys/s | Laptop |
| 2x RTX 4090 | 128,128 --slices 16 | 11.4 Gkeys/s | 98.6% scaling |
| 4x RTX 4090 (Pincer) | 128,128 --slices 16 | 23.1 Gkeys/s | Bidirectional |

---

## Installation

**Prerequisites:**
- NVIDIA GPU (Compute Capability 7.5+)
- CUDA Toolkit
- GCC/G++
- Make

**Quick Install (Ubuntu/Debian):**
```bash
apt update
apt install -y build-essential gcc make
apt install -y cuda-toolkit
git clone https://github.com/Dookoo2/CUDACyclone.git
cd CUDACyclone
make everything
```

---

## Verification

Run the proof script to verify key coverage:

```bash
python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512
```

Expected output:
```
================ Summary by blocks ================
Range start A (start+2k)           : total= 128  success= 128  fail=   0
Range start B (start+1+2k)         : total= 128  success= 128  fail=   0
Range end A (end-2k)               : total= 128  success= 128  fail=   0
Range end B (end-1-2k)             : total= 128  success= 128  fail=   0
Full mod 512 residue coverage      : total= 256  success= 256  fail=   0
Random Q1-Q4                       : total=  80  success=  80  fail=   0

Done. Successes=848 Failures=0
```

---

## File Structure

| File | Description |
|------|-------------|
| `CUDACyclone.cu` | Single-GPU implementation |
| `CUDACyclone_MultiGPU.cu` | Multi-GPU with checkpointing |
| `CUDACyclone_MultiGPU_Pincer.cu` | Bidirectional pincer scanning |
| `CUDAHash.cu` | SHA-256, RIPEMD-160, Hash160 |
| `CUDAMath.h` | Secp256k1 field arithmetic |
| `CUDAStructures.h` | Device constants, structures |
| `CUDAUtils.h` | Host/device utilities |
| `Makefile` | Build configuration |
| `proof.py` | Key coverage verification |

---

## Version History

| Version | Changes |
|---------|---------|
| **v1.4** | Added Bidirectional Pincer Mode, Multi-GPU checkpoint/resume |
| **v1.3** | Fixed key-skipping bug, full kernel rewrite |
| **v1.2** | Full CUDA kernel rewrite |
| **v1.1** | Switched to constant memory (thermal throttling fix) |
| **v1.0** | Initial release |

---

## Tips

BTC: bc1qtq4y9l9ajeyxq05ynq09z8p52xdmk4hqky9c8n

---

## License

This project is for educational and research purposes.

## Credits

- [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch) - Secp256k1 foundations
- [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul) - Additional optimizations
