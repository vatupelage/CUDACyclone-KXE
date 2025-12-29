# CUDACyclone Distributed Mode

This document describes the distributed server/client architecture for CUDACyclone, enabling any number of machines with any number of GPUs to collaborate on searching a key range.

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Building](#building)
4. [Server Setup & Usage](#server-setup--usage)
5. [Client Setup & Usage](#client-setup--usage)
6. [What Happens When a Client Joins](#what-happens-when-a-client-joins)
7. [Work Distribution System](#work-distribution-system)
8. [Bidirectional (Pincer) Mode](#bidirectional-pincer-mode)
9. [Checkpoint System](#checkpoint-system)
10. [Network Protocol Details](#network-protocol-details)
11. [Performance Tuning](#performance-tuning)
12. [Troubleshooting](#troubleshooting)
13. [Examples](#examples)

---

## Overview

The distributed mode enables large-scale collaborative key searching across multiple machines. It consists of two components:

| Component | Binary | Description | CUDA Required |
|-----------|--------|-------------|---------------|
| **Server** | `CUDACyclone_Server` | Coordinator that manages work distribution, tracks progress, handles client connections | No |
| **Client** | `CUDACyclone_Client` | GPU worker that connects to server, receives work, executes search kernel, reports results | Yes |

**Key Features:**
- Unlimited clients with unlimited GPUs per client
- Power-of-2 work unit partitioning (required by the algorithm)
- Automatic work reassignment on client failure
- Checkpoint/resume for fault tolerance
- Bidirectional (pincer) mode for 2x statistical speedup
- Near-linear scaling across machines

---

## How It Works

### High-Level Architecture

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              CUDACyclone Server (:17403)                │
                        │                                                         │
                        │  ┌───────────────────────────────────────────────────┐  │
                        │  │                  Work Unit Manager                 │  │
                        │  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬────┐ │  │
                        │  │  │ U0  │ U1  │ U2  │ U3  │ U4  │ U5  │ ... │ UN │ │  │
                        │  │  │DONE │ WIP │AVAIL│AVAIL│ WIP │DONE │     │    │ │  │
                        │  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴────┘ │  │
                        │  └───────────────────────────────────────────────────┘  │
                        │                                                         │
                        │  ┌─────────────────┐  ┌─────────────────────────────┐  │
                        │  │ Client Registry │  │     Checkpoint System       │  │
                        │  │ - Client A: 2GPU│  │  - Periodic saves (5 min)   │  │
                        │  │ - Client B: 4GPU│  │  - Save on Ctrl+C           │  │
                        │  │ - Client C: 1GPU│  │  - Resume on restart        │  │
                        │  └─────────────────┘  └─────────────────────────────┘  │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    │ TCP Port 17403
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    │                               │                               │
                    ▼                               ▼                               ▼
        ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
        │      Client A         │     │      Client B         │     │      Client C         │
        │    (Machine 1)        │     │    (Machine 2)        │     │    (Machine 3)        │
        │  ┌─────────────────┐  │     │  ┌─────────────────┐  │     │  ┌─────────────────┐  │
        │  │ GPU 0   GPU 1   │  │     │  │GPU0 GPU1 GPU2 3 │  │     │  │      GPU 0      │  │
        │  │ ┌───┐   ┌───┐   │  │     │  │┌──┐┌──┐┌──┐┌──┐ │  │     │  │     ┌───┐       │  │
        │  │ │ ▶ │   │ ◀ │   │  │     │  ││▶ ││◀ ││▶ ││◀ │ │  │     │  │     │ ▶ │       │  │
        │  │ └───┘   └───┘   │  │     │  │└──┘└──┘└──┘└──┘ │  │     │  │     └───┘       │  │
        │  │  (pincer pair)  │  │     │  │ (2 pincer pairs)│  │     │  │   (forward)     │  │
        │  └─────────────────┘  │     │  └─────────────────┘  │     │  └─────────────────┘  │
        │  Working on: Unit 1   │     │  Working on: U4, U7   │     │  Working on: Unit 9   │
        └───────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

### The Search Process

1. **Server starts** and divides the total search range into work units (each a power-of-2 size)
2. **Clients connect** and register their GPU capabilities with the server
3. **Server assigns work** units to clients based on availability
4. **Clients execute** GPU kernels to search their assigned range
5. **Clients report** progress periodically and completion when done
6. **Server reassigns** work from failed/slow clients automatically
7. **When found**, the server broadcasts to all clients and saves the result

---

## Building

### Build Commands

```bash
# Build both server and client
make distributed

# Build only server (no CUDA required - can build on any Linux machine)
make server

# Build only client (requires CUDA toolkit and NVIDIA GPU)
make client

# Build everything (standalone + distributed)
make everything

# Clean build
make clean && make distributed
```

### Build Requirements

| Component | Requirements |
|-----------|-------------|
| Server | GCC with C++17 support, pthreads |
| Client | CUDA Toolkit 11.0+, NVIDIA GPU (SM 7.5+), GCC with C++17 |

---

## Server Setup & Usage

### Starting the Server

The server is the coordinator that manages work distribution. Start it first before any clients.

```bash
./CUDACyclone_Server \
    --range <start_hex>:<end_hex> \
    --target-hash160 <40_hex_chars> \
    [options]
```

### Server Command-Line Options

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--range <start>:<end>` | Search range in hexadecimal. **Must be power-of-2 length and properly aligned.** |
| `--target-hash160 <hex>` | Target Bitcoin address hash160 (40 hex characters) |
| OR `--address <P2PKH>` | Target Bitcoin P2PKH address (starts with '1') |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--port <N>` | 17403 | TCP port to listen on |
| `--unit-bits <N>` | 36 | Work unit size as 2^N keys (see table below) |
| `--batch-size <N>` | 128 | Recommended batch size sent to clients |
| `--slices <N>` | 16 | Recommended slices per kernel launch |
| `--pincer` | disabled | Enable server-coordinated bidirectional mode |
| `--checkpoint <file>` | none | Path to checkpoint file for save/resume |
| `--checkpoint-interval <N>` | 300 | Seconds between automatic checkpoints |
| `--heartbeat-timeout <N>` | 90 | Seconds before considering a client dead |
| `--max-clients <N>` | 256 | Maximum simultaneous client connections |

### Choosing Work Unit Size (`--unit-bits`)

The work unit size determines the granularity of work distribution:

| Unit Bits | Keys per Unit | Time @ 6.5 Gkeys/s | Best For |
|-----------|---------------|---------------------|----------|
| 32 | 4.3 billion | ~11 minutes | Small ranges, many clients, testing |
| 34 | 17.2 billion | ~44 minutes | Medium ranges, good balance |
| 36 | 68.7 billion | ~3 hours | **Recommended default** |
| 38 | 274.9 billion | ~12 hours | Large ranges |
| 40 | 1.1 trillion | ~47 hours | Very large ranges, few clients |
| 44 | 17.6 trillion | ~31 days | Massive ranges, persistent clients |

**Guidelines:**
- **Smaller units** = Better load balancing, faster failure recovery, more network messages
- **Larger units** = Less overhead, but more work lost if a client fails
- **Rule of thumb**: Target 1-4 hours per unit for optimal balance

### Server Output

When running, the server displays real-time status:

```
[00:15:32] Units: 45/1024 (4.4%) | Active: 8 | Pending: 971 | Speed: 52.3 Gkeys/s | Keys: 2.84T | Clients: 3
```

| Field | Meaning |
|-------|---------|
| Time | Elapsed time since start |
| Units | Completed/Total work units and percentage |
| Active | Work units currently being processed |
| Pending | Work units waiting to be assigned |
| Speed | Aggregate speed across all clients |
| Keys | Total keys checked so far |
| Clients | Number of connected clients |

---

## Client Setup & Usage

### Starting a Client

Clients connect to the server and perform the actual GPU computation.

```bash
./CUDACyclone_Client \
    --server <host>:<port> \
    [options]
```

### Client Command-Line Options

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--server <host:port>` | Server address (e.g., `192.168.1.100:17403` or `server.local:17403`) |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpus <ids>` | all | Comma-separated GPU IDs to use (e.g., `0,1,2,3`) |
| `--grid <A,B>` | from server | A=batch size, B=batches per SM |
| `--slices <N>` | from server | Slices per kernel launch |
| `--pincer` | disabled | Enable local bidirectional mode (pairs GPUs) |
| `--reconnect-delay <N>` | 5 | Seconds to wait before reconnecting after disconnect |

### GPU Selection Examples

```bash
# Use all available GPUs
./CUDACyclone_Client --server 192.168.1.100:17403

# Use specific GPUs only
./CUDACyclone_Client --server 192.168.1.100:17403 --gpus 0,2,3

# Use single GPU
./CUDACyclone_Client --server 192.168.1.100:17403 --gpus 1

# Use GPUs with local pincer mode (pairs GPUs 0-1 and 2-3)
./CUDACyclone_Client --server 192.168.1.100:17403 --gpus 0,1,2,3 --pincer
```

### Client Output

```
[Client] Connected to server 192.168.1.100:17403
[Client] Registered as client #3 with 4 GPUs
[Client] Received work unit 47: range 0x180000000000000-0x1BFFFFFFFFFFFFF
[GPU 0] Searching forward from 0x180000000000000 @ 6.52 Gkeys/s
[GPU 1] Searching backward from 0x1BFFFFFFFFFFFFF @ 6.48 Gkeys/s
[GPU 2] Searching forward from 0x1A0000000000000 @ 6.51 Gkeys/s
[GPU 3] Searching backward from 0x1DFFFFFFFFFFFFF @ 6.49 Gkeys/s
[Client] Progress: 12.5% | Speed: 26.0 Gkeys/s | Checked: 1.87T keys
```

---

## What Happens When a Client Joins

This section explains the detailed sequence of events when a new client connects to the server.

### Connection Sequence Diagram

```
    CLIENT                                                    SERVER
       │                                                         │
   ┌───┴───┐                                                     │
   │ Start │                                                     │
   └───┬───┘                                                     │
       │                                                         │
       │ ──────────── TCP Connect to port 17403 ───────────────► │
       │                                                         │
       │ ◄─────────── Connection Accepted ─────────────────────  │
       │                                                         │
       │                                                         │
   ════╪═════════════════ REGISTRATION PHASE ════════════════════╪════
       │                                                         │
       │     ┌─────────────────────────────────────────┐         │
       │     │         REGISTER_REQUEST                │         │
       │     │  - Protocol version: 1                  │         │
       │     │  - GPU count: 4                         │         │
       │     │  - GPU info: [RTX4090, RTX4090, ...]    │         │
       │     │  - Hostname: "worker-01"                │         │
       │     │  - Supports pincer: true                │         │
       │     └─────────────────────────────────────────┘         │
       │ ─────────────────────────────────────────────────────►  │
       │                                                         │
       │                                      ┌──────────────────┴──────────────────┐
       │                                      │ Server processes registration:      │
       │                                      │ 1. Validate protocol version        │
       │                                      │ 2. Check client limit not exceeded  │
       │                                      │ 3. Allocate unique client ID        │
       │                                      │ 4. Store client info in registry    │
       │                                      │ 5. Log: "Client #5 connected from   │
       │                                      │    worker-01 with 4 GPUs"           │
       │                                      └──────────────────┬──────────────────┘
       │                                                         │
       │     ┌─────────────────────────────────────────┐         │
       │     │         REGISTER_RESPONSE               │         │
       │     │  - Status: SUCCESS                      │         │
       │     │  - Client ID: 5                         │         │
       │     │  - Target hash160: [20 bytes]           │         │
       │     │  - Recommended batch_size: 128          │         │
       │     │  - Recommended slices: 16               │         │
       │     │  - Pincer mode: enabled/disabled        │         │
       │     └─────────────────────────────────────────┘         │
       │ ◄─────────────────────────────────────────────────────  │
       │                                                         │
       │                                                         │
   ════╪═════════════════ WORK ASSIGNMENT PHASE ═════════════════╪════
       │                                                         │
       │     ┌─────────────────────────────────────────┐         │
       │     │         WORK_REQUEST                    │         │
       │     │  - Client ID: 5                         │         │
       │     │  - GPU slots available: 4               │         │
       │     └─────────────────────────────────────────┘         │
       │ ─────────────────────────────────────────────────────►  │
       │                                                         │
       │                                      ┌──────────────────┴──────────────────┐
       │                                      │ Server assigns work:                │
       │                                      │ 1. Find AVAILABLE work units        │
       │                                      │ 2. Mark units as ASSIGNED           │
       │                                      │ 3. Record assignment to client #5   │
       │                                      │ 4. Set assignment timestamp         │
       │                                      └──────────────────┬──────────────────┘
       │                                                         │
       │     ┌─────────────────────────────────────────┐         │
       │     │         WORK_ASSIGNMENT                 │         │
       │     │  - Unit ID: 47                          │         │
       │     │  - Range start: 0x180000000000000       │         │
       │     │  - Range end:   0x1BFFFFFFFFFFFFF       │         │
       │     │  - Direction: FORWARD (or BACKWARD)     │         │
       │     │  - Keys to check: 68,719,476,736        │         │
       │     └─────────────────────────────────────────┘         │
       │ ◄─────────────────────────────────────────────────────  │
       │                                                         │
       │                                                         │
   ┌───┴───────────────────────────────────────┐                 │
   │ Client initializes GPUs:                  │                 │
   │ 1. Allocate device memory                 │                 │
   │ 2. Copy target hash160 to constant mem    │                 │
   │ 3. Precompute EC points (G, 2G, ... nG)   │                 │
   │ 4. Initialize starting points per thread  │                 │
   │ 5. Launch search kernels on all GPUs      │                 │
   └───┬───────────────────────────────────────┘                 │
       │                                                         │
       │                                                         │
   ════╪═════════════════ ACTIVE WORK PHASE ═════════════════════╪════
       │                                                         │
       │     ┌─────────────────────────────────────────┐         │
       │     │         HEARTBEAT (every 30s)           │         │
       │     │  - Client ID: 5                         │         │
       │     │  - Status: WORKING                      │         │
       │     │  - Active units: [47]                   │         │
       │     └─────────────────────────────────────────┘         │
       │ ─────────────────────────────────────────────────────►  │
       │                                                         │
       │ ◄────────────────── HEARTBEAT_ACK ────────────────────  │
       │                                                         │
       │     ┌─────────────────────────────────────────┐         │
       │     │         PROGRESS_REPORT (every 10s)     │         │
       │     │  - Client ID: 5                         │         │
       │     │  - Unit ID: 47                          │         │
       │     │  - Keys checked: 8,500,000,000          │         │
       │     │  - Current speed: 26.0 Gkeys/s          │         │
       │     │  - Progress: 12.4%                      │         │
       │     └─────────────────────────────────────────┘         │
       │ ─────────────────────────────────────────────────────►  │
       │                                                         │
       │                                      ┌──────────────────┴──────────────────┐
       │                                      │ Server updates stats:               │
       │                                      │ - Update client speed               │
       │                                      │ - Accumulate keys processed         │
       │                                      │ - Update unit progress              │
       │                                      │ - Refresh last-contact timestamp    │
       │                                      └──────────────────┬──────────────────┘
       │                                                         │
      ...                    (work continues)                   ...
       │                                                         │
       │     ┌─────────────────────────────────────────┐         │
       │     │         UNIT_COMPLETE                   │         │
       │     │  - Client ID: 5                         │         │
       │     │  - Unit ID: 47                          │         │
       │     │  - Total keys: 68,719,476,736           │         │
       │     │  - Found: false                         │         │
       │     └─────────────────────────────────────────┘         │
       │ ─────────────────────────────────────────────────────►  │
       │                                                         │
       │                                      ┌──────────────────┴──────────────────┐
       │                                      │ Server marks unit complete:         │
       │                                      │ 1. Set unit state to COMPLETED      │
       │                                      │ 2. Update completion count          │
       │                                      │ 3. Assign next available unit       │
       │                                      └──────────────────┬──────────────────┘
       │                                                         │
       │ ◄────────────── WORK_ASSIGNMENT (next unit) ──────────  │
       │                                                         │
       ▼                                                         ▼
```

### Registration Details

When a client connects, the following information is exchanged:

**Client sends:**
- Protocol version (for compatibility checking)
- Number of GPUs available
- Detailed GPU information (name, memory, compute capability)
- Client hostname
- Whether it supports pincer mode

**Server responds with:**
- Unique client ID (used for all subsequent communication)
- Target hash160 (what we're searching for)
- Recommended kernel parameters (batch_size, slices)
- Whether pincer mode is enabled

### Work Assignment Details

The server maintains a work queue with three states:

| State | Meaning |
|-------|---------|
| `AVAILABLE` | Ready to be assigned to a client |
| `ASSIGNED` | Currently being worked on by a client |
| `COMPLETED` | Successfully searched (key not found in this unit) |

When assigning work:
1. Server finds the first `AVAILABLE` unit
2. Changes state to `ASSIGNED`
3. Records which client is working on it
4. Records the assignment timestamp (for timeout detection)
5. Sends the work unit details to the client

### Failure Handling

If a client disconnects or stops sending heartbeats:

```
                                      ┌──────────────────────────────────────┐
                                      │ Maintenance Thread (runs every 30s) │
                                      │                                      │
                                      │ For each client:                     │
                                      │   If (now - last_heartbeat) > 90s:   │
                                      │     1. Mark client as disconnected   │
                                      │     2. Release all assigned units    │
                                      │        back to AVAILABLE state       │
                                      │     3. Log warning                   │
                                      │     4. Increment reassign counter    │
                                      │                                      │
                                      │ Units reassigned up to 3 times,      │
                                      │ then marked as FAILED for review     │
                                      └──────────────────────────────────────┘
```

---

## Work Distribution System

### Power-of-2 Partitioning

The CUDACyclone algorithm requires power-of-2 aligned ranges. The server enforces this:

```
Total Range: 0x100000000000 to 0x1FFFFFFFFFFFF (45-bit range = 2^45 keys)

With --unit-bits 40, server creates 32 work units (2^45 / 2^40 = 32):

Unit 0:  0x100000000000 - 0x10FFFFFFFFFF  (2^40 keys)
Unit 1:  0x110000000000 - 0x11FFFFFFFFFF  (2^40 keys)
Unit 2:  0x120000000000 - 0x12FFFFFFFFFF  (2^40 keys)
...
Unit 31: 0x1F0000000000 - 0x1FFFFFFFFFFFF (2^40 keys)
```

### Dynamic Work Assignment

Work is assigned on-demand, not pre-allocated:

```
Time 0:    Client A connects (4 GPUs) → Gets units 0, 1, 2, 3
Time 1:    Client B connects (2 GPUs) → Gets units 4, 5
Time 2:    Client A completes unit 0  → Gets unit 6
Time 3:    Client C connects (1 GPU)  → Gets unit 7
Time 4:    Client B disconnects       → Units 4, 5 return to AVAILABLE
Time 5:    Client A gets units 4, 5 (reassigned from B)
...
```

This dynamic assignment ensures:
- Fast clients get more work
- Slow clients don't block progress
- Failed clients' work is automatically recovered

---

## Bidirectional (Pincer) Mode

### How Pincer Mode Works

In standard mode, each GPU searches from the start of its range moving forward:

```
Standard Mode:
[START ═══════════════════════════════════════════════════════ END]
   │
   └────► GPU searches forward ─────────────────────────────────►
         (checks 100% of range on average to find random key)
```

In pincer mode, GPUs are paired to search from both ends:

```
Pincer Mode:
[START ═══════════════════════════════════════════════════════ END]
   │                                                             │
   └────► GPU 0 (forward) ──────►  ◄────── GPU 1 (backward) ────┘

         They meet in the middle, effectively searching 2x faster
         (checks 50% of range on average to find random key)
```

### Statistical Advantage

For a randomly located key:
- **Standard mode**: Expected search = 50% of range
- **Pincer mode**: Expected search = 25% of range (2x speedup)

### Enabling Pincer Mode

**Local pincer (recommended for multi-GPU clients):**

```bash
# Client pairs its own GPUs
./CUDACyclone_Client --server 192.168.1.100:17403 --gpus 0,1,2,3 --pincer

# Result:
#   GPU 0 + GPU 1 → Pair 1 (searching unit X from both ends)
#   GPU 2 + GPU 3 → Pair 2 (searching unit Y from both ends)
```

**Server-coordinated pincer:**

```bash
# Server assigns directions
./CUDACyclone_Server --range ... --target-hash160 ... --pincer

# Single-GPU clients get assigned forward OR backward direction
./CUDACyclone_Client --server ... --gpus 0
# Server might assign: "Unit 5, direction: BACKWARD"
```

---

## Checkpoint System

### Automatic Checkpointing

The server periodically saves state to disk:

```bash
./CUDACyclone_Server \
    --range 100000000000:1FFFFFFFFFFFF \
    --target-hash160 3ee4133d991f52fdf6a25c9834e0745ac74248a4 \
    --checkpoint server.ckpt \
    --checkpoint-interval 300  # Save every 5 minutes
```

### What's Saved

| Data | Description |
|------|-------------|
| Work unit states | Which units are available, assigned, or completed |
| Progress counters | Total keys processed, completion percentage |
| Configuration | Range, target, unit size (for validation on resume) |
| Found key | If key was found, the private key value |
| Timestamp | When checkpoint was saved |

### Resuming from Checkpoint

```bash
# Server automatically detects and loads checkpoint if it exists
./CUDACyclone_Server \
    --range 100000000000:1FFFFFFFFFFFF \
    --target-hash160 3ee4133d991f52fdf6a25c9834e0745ac74248a4 \
    --checkpoint server.ckpt

# Output:
# [Server] Loaded checkpoint from server.ckpt
# [Server] Resuming: 156/1024 units completed (15.2%)
# [Server] 45 units were in-progress, returned to available
```

### Checkpoint on Interrupt

Pressing `Ctrl+C` triggers graceful shutdown:

```
^C
[Server] Interrupt received, saving checkpoint...
[Server] Checkpoint saved to server.ckpt
[Server] Broadcasting shutdown to 5 clients...
[Server] Shutdown complete.
```

---

## Network Protocol Details

### Message Types

| Type | Direction | Purpose |
|------|-----------|---------|
| `REGISTER_REQUEST` | Client → Server | Initial connection and capability report |
| `REGISTER_RESPONSE` | Server → Client | Accept/reject with configuration |
| `WORK_REQUEST` | Client → Server | Ask for work assignment |
| `WORK_ASSIGNMENT` | Server → Client | Assign a work unit |
| `NO_WORK_AVAILABLE` | Server → Client | No units available (wait and retry) |
| `HEARTBEAT` | Client → Server | Periodic alive signal |
| `HEARTBEAT_ACK` | Server → Client | Heartbeat acknowledgment |
| `PROGRESS_REPORT` | Client → Server | Current speed and progress |
| `UNIT_COMPLETE` | Client → Server | Work unit finished |
| `FOUND_RESULT` | Client → Server | Key found! |
| `KEY_FOUND` | Server → All | Broadcast: stop searching |
| `SHUTDOWN` | Server → All | Server is shutting down |

### Timing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Heartbeat interval | 30 sec | How often client sends heartbeat |
| Heartbeat timeout | 90 sec | Server considers client dead after this |
| Progress interval | 10 sec | How often client reports progress |
| Reconnect delay | 5 sec | Client waits this long before reconnecting |

---

## Performance Tuning

### Optimal Settings by GPU

| GPU Model | Recommended Settings | Expected Speed |
|-----------|---------------------|----------------|
| RTX 5090 | `--grid 128,256 --slices 16` | ~8.6 Gkeys/s |
| RTX 4090 | `--grid 128,128 --slices 16` | ~6.5 Gkeys/s |
| RTX 4080 | `--grid 128,128 --slices 16` | ~5.0 Gkeys/s |
| RTX 4070 | `--grid 256,128 --slices 16` | ~3.5 Gkeys/s |
| RTX 3090 | `--grid 128,128 --slices 16` | ~4.5 Gkeys/s |
| RTX 3080 | `--grid 256,128 --slices 16` | ~3.8 Gkeys/s |

### Scaling Expectations

| Setup | Expected Aggregate Speed |
|-------|--------------------------|
| 1x RTX 4090 | 6.5 Gkeys/s |
| 2x RTX 4090 | 13.0 Gkeys/s (2.0x) |
| 4x RTX 4090 | 26.0 Gkeys/s (4.0x) |
| 8x RTX 4090 (2 machines) | 51.5 Gkeys/s (7.9x) |
| 16x RTX 4090 (4 machines) | 102 Gkeys/s (15.7x) |

Near-linear scaling because the workload is embarrassingly parallel with minimal coordination overhead.

---

## Troubleshooting

### Server Issues

**"Failed to bind to port 17403"**
```bash
# Check if port is in use
netstat -tlnp | grep 17403

# Use a different port
./CUDACyclone_Server --port 17404 ...
```

**"Invalid range: must be power-of-2 length"**
```bash
# Wrong: 0x100:0x2FF (length = 0x200 - not aligned)
# Right: 0x100:0x1FF (length = 0x100 = 256 = 2^8)
# Right: 0x200:0x3FF (length = 0x200 = 512 = 2^9)
```

**"Invalid range: start must be aligned"**
```bash
# Wrong: 0x150:0x24F (start 0x150 not aligned to length 0x100)
# Right: 0x100:0x1FF (start 0x100 aligned to length 0x100)
# Right: 0x200:0x2FF (start 0x200 aligned to length 0x100)
```

### Client Issues

**"Failed to connect to server"**
```bash
# Check server is running
ping 192.168.1.100
telnet 192.168.1.100 17403

# Check firewall
sudo ufw allow 17403/tcp  # Ubuntu
sudo firewall-cmd --add-port=17403/tcp  # CentOS/RHEL
```

**"Registration rejected: protocol version mismatch"**
- Rebuild both server and client from the same source

**"Registration rejected: server at capacity"**
- Increase `--max-clients` on server or wait for a slot

**"GPU initialization failed"**
```bash
# Check GPU is available
nvidia-smi

# Check CUDA version
nvcc --version

# Try specific GPU
./CUDACyclone_Client --server ... --gpus 0
```

### Performance Issues

**Low throughput**
- Check for thermal throttling: `nvidia-smi dmon -s p`
- Reduce work unit size for better load balancing
- Enable pincer mode for effective 2x speedup

**Clients getting no work**
- All units may be assigned or completed
- Check server status output for "Pending" count
- Reduce `--unit-bits` to create more units

---

## Examples

### Example 1: Simple Two-Machine Setup

```bash
# Machine 1 (Server + 2 GPUs)
./CUDACyclone_Server \
    --range 8000000000:FFFFFFFFFF \
    --target-hash160 3ee4133d991f52fdf6a25c9834e0745ac74248a4 \
    --port 17403 \
    --unit-bits 34 \
    --checkpoint search.ckpt &

./CUDACyclone_Client \
    --server 127.0.0.1:17403 \
    --gpus 0,1 \
    --pincer

# Machine 2 (4 GPUs)
./CUDACyclone_Client \
    --server 192.168.1.100:17403 \
    --gpus 0,1,2,3 \
    --pincer
```

### Example 2: Large Datacenter Deployment

```bash
# Central server (no GPU needed)
./CUDACyclone_Server \
    --range 20000000000000000:3FFFFFFFFFFFFFFFF \
    --target-hash160 20d45a6a762535700ce9e0b216e31994335db8a5 \
    --port 17403 \
    --unit-bits 40 \
    --pincer \
    --checkpoint puzzle66.ckpt \
    --checkpoint-interval 600 \
    --max-clients 100

# Worker nodes (run on each GPU server)
# worker-01.internal (8x RTX 4090)
./CUDACyclone_Client --server master.internal:17403 --gpus 0,1,2,3,4,5,6,7 --pincer

# worker-02.internal (4x RTX 4090)
./CUDACyclone_Client --server master.internal:17403 --gpus 0,1,2,3 --pincer

# worker-03.internal (2x RTX 3090)
./CUDACyclone_Client --server master.internal:17403 --gpus 0,1 --pincer

# ... more workers
```

### Example 3: Resume After Interruption

```bash
# Initial run (interrupted after 2 hours)
./CUDACyclone_Server \
    --range 100000000000:1FFFFFFFFFFFF \
    --target-hash160 3ee4133d991f52fdf6a25c9834e0745ac74248a4 \
    --checkpoint search.ckpt
# ^C (interrupted)

# Resume later
./CUDACyclone_Server \
    --range 100000000000:1FFFFFFFFFFFF \
    --target-hash160 3ee4133d991f52fdf6a25c9834e0745ac74248a4 \
    --checkpoint search.ckpt
# Automatically loads progress and continues
```

---

## Security Considerations

| Concern | Status | Mitigation |
|---------|--------|------------|
| Encryption | Not implemented | Use VPN or SSH tunnel for public networks |
| Authentication | Not implemented | Restrict network access via firewall |
| Result verification | Implemented | Server verifies hash160 matches before accepting |
| Malicious clients | Partially trusted | False progress reports possible but don't affect correctness |

For production deployments over untrusted networks, wrap the connection in an SSH tunnel:

```bash
# On client machine, create tunnel to server
ssh -L 17403:localhost:17403 user@server.example.com

# Connect client through tunnel
./CUDACyclone_Client --server 127.0.0.1:17403 --gpus 0,1
```

---

## File Reference

| File | Description |
|------|-------------|
| `CUDACyclone_Protocol.h` | Network protocol constants and message structures |
| `CUDACyclone_Network.h/cpp` | Cross-platform socket utilities and wrappers |
| `CUDACyclone_WorkUnit.h` | Work unit management and 256-bit arithmetic |
| `CUDACyclone_Server.h/cpp` | Server implementation (~950 lines) |
| `CUDACyclone_Client.cu` | Client implementation with GPU kernel (~1060 lines) |
| `DISTRIBUTED_README.md` | This documentation |

---

## Version History

- **v1.0**: Initial distributed mode implementation
  - TCP-based server/client architecture on port 17403
  - Power-of-2 work unit partitioning
  - Local and server-coordinated pincer mode
  - Checkpoint/resume support
  - Automatic client reconnection and work reassignment
  - Support for unlimited clients with unlimited GPUs
