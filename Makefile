TARGET        := CUDACyclone
TARGET_MULTI  := CUDACyclone_MultiGPU
TARGET_PINCER := CUDACyclone_Pincer
TARGET_SERVER := CUDACyclone_Server
TARGET_CLIENT := CUDACyclone_Client
TARGET_KXE    := CUDACyclone_KXE
TARGET_KXE_MULTI := CUDACyclone_KXE_MultiGPU
TARGET_KXE_PINCER := CUDACyclone_KXE_Pincer

SRC          := CUDACyclone.cu CUDAHash.cu
SRC_MULTI    := CUDACyclone_MultiGPU.cu CUDAHash.cu
SRC_PINCER   := CUDACyclone_MultiGPU_Pincer.cu CUDAHash.cu
SRC_KXE      := CUDACyclone_KXE.cu CUDAHash.cu

OBJ          := $(SRC:.cu=.o)
OBJ_MULTI    := $(SRC_MULTI:.cu=.o)
OBJ_PINCER   := CUDACyclone_MultiGPU_Pincer.o CUDAHash.o

# Compilers
NVCC        := nvcc
CXX         := g++

# GPU architecture detection
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
ifeq ($(GPU_ARCH),)
GPU_ARCH := 86
endif
SM_ARCHS   := 75 86 89 $(GPU_ARCH)
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))

# Compiler flags
NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE)
NVCC_CXXFLAGS := -std=c++17
CXXFLAGS   := -std=c++17 -O3 -Wall
CXXFLAGS_SERVER := -std=c++17 -O3 -Wall -DSERVER_MODE

# Linker flags
LDFLAGS       := -lcudadevrt -cudart=static
LDFLAGS_MULTI := -lcudadevrt -cudart=static -lpthread
LDFLAGS_SERVER := -lpthread

# ============================================================================
# DEFAULT TARGET
# ============================================================================
all: $(TARGET)

# ============================================================================
# STANDALONE (SINGLE MACHINE) TARGETS
# ============================================================================

# Single-GPU version
single: $(TARGET)

# Multi-GPU version
multi: $(TARGET_MULTI)

# Pincer (bidirectional) multi-GPU version
pincer: $(TARGET_PINCER)

# Build all standalone versions
both: $(TARGET) $(TARGET_MULTI)

# Build everything standalone
standalone: $(TARGET) $(TARGET_MULTI) $(TARGET_PINCER)

# ============================================================================
# DISTRIBUTED MODE TARGETS
# ============================================================================

# Server only (no CUDA required)
server: $(TARGET_SERVER)

# Client only (requires CUDA)
client: $(TARGET_CLIENT)

# Both server and client
distributed: $(TARGET_SERVER) $(TARGET_CLIENT)

# ============================================================================
# BUILD ALL
# ============================================================================

# Build absolutely everything
everything: $(TARGET) $(TARGET_MULTI) $(TARGET_PINCER) $(TARGET_SERVER) $(TARGET_CLIENT) $(TARGET_KXE)

# ============================================================================
# KXE (PERMUTED SCANNING) TARGETS
# ============================================================================

# KXE single-GPU version
kxe: $(TARGET_KXE)

# KXE multi-GPU version
kxe-multi: $(TARGET_KXE_MULTI)

# KXE pincer (bidirectional) multi-GPU version
kxe-pincer: $(TARGET_KXE_PINCER)

# Build all KXE versions
kxe-all: $(TARGET_KXE) $(TARGET_KXE_MULTI) $(TARGET_KXE_PINCER)

# KXE test suite
test-kxe: kxe/tests/test_bijection
	./kxe/tests/test_bijection

# ============================================================================
# BUILD RULES
# ============================================================================

# Single-GPU binary
$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

# Multi-GPU binary
$(TARGET_MULTI): CUDACyclone_MultiGPU.o CUDAHash.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CXXFLAGS) CUDACyclone_MultiGPU.o CUDAHash.o -o $@ $(LDFLAGS_MULTI)

# Pincer mode binary
$(TARGET_PINCER): CUDACyclone_MultiGPU_Pincer.o CUDAHash.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CXXFLAGS) CUDACyclone_MultiGPU_Pincer.o CUDAHash.o -o $@ $(LDFLAGS_MULTI)

# Server binary (pure C++, no CUDA)
$(TARGET_SERVER): CUDACyclone_Server.o CUDACyclone_Network.o
	$(CXX) $(CXXFLAGS_SERVER) CUDACyclone_Server.o CUDACyclone_Network.o -o $@ $(LDFLAGS_SERVER)

# Client binary (CUDA + networking)
$(TARGET_CLIENT): CUDACyclone_Client.o CUDAHash.o CUDACyclone_Network.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CXXFLAGS) CUDACyclone_Client.o CUDAHash.o CUDACyclone_Network.o -o $@ $(LDFLAGS_MULTI)

# KXE single-GPU binary
$(TARGET_KXE): CUDACyclone_KXE.o CUDAHash.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CXXFLAGS) CUDACyclone_KXE.o CUDAHash.o -o $@ $(LDFLAGS)

# KXE multi-GPU binary
$(TARGET_KXE_MULTI): CUDACyclone_KXE_MultiGPU.o CUDAHash.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CXXFLAGS) CUDACyclone_KXE_MultiGPU.o CUDAHash.o -o $@ $(LDFLAGS_MULTI)

# KXE pincer (bidirectional) multi-GPU binary
$(TARGET_KXE_PINCER): CUDACyclone_KXE_Pincer.o CUDAHash.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CXXFLAGS) CUDACyclone_KXE_Pincer.o CUDAHash.o -o $@ $(LDFLAGS_MULTI)

# KXE test binary (host-only, no CUDA)
kxe/tests/test_bijection: kxe/tests/test_bijection.cpp kxe/KXEPermutation.cuh
	$(CXX) $(CXXFLAGS) -o $@ kxe/tests/test_bijection.cpp

# ============================================================================
# OBJECT FILE RULES
# ============================================================================

# CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CXXFLAGS) -c $< -o $@

# C++ source files
CUDACyclone_Network.o: CUDACyclone_Network.cpp CUDACyclone_Network.h CUDACyclone_Protocol.h
	$(CXX) $(CXXFLAGS) -c CUDACyclone_Network.cpp -o $@

CUDACyclone_Server.o: CUDACyclone_Server.cpp CUDACyclone_Server.h CUDACyclone_Network.h CUDACyclone_Protocol.h CUDACyclone_WorkUnit.h
	$(CXX) $(CXXFLAGS_SERVER) -c CUDACyclone_Server.cpp -o $@

# Client object (needs special handling for CUDA + networking headers)
CUDACyclone_Client.o: CUDACyclone_Client.cu CUDACyclone_Network.h CUDACyclone_Protocol.h
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CXXFLAGS) -c CUDACyclone_Client.cu -o $@

# ============================================================================
# CLEAN
# ============================================================================

clean:
	rm -f $(TARGET) $(TARGET_MULTI) $(TARGET_PINCER) $(TARGET_SERVER) $(TARGET_CLIENT) $(TARGET_KXE) $(TARGET_KXE_MULTI) $(TARGET_KXE_PINCER)
	rm -f *.o
	rm -f kxe/tests/test_bijection

clean-obj:
	rm -f *.o

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "CUDACyclone Build System"
	@echo "========================"
	@echo ""
	@echo "Standalone (single machine) targets:"
	@echo "  make              - Build single-GPU version (default)"
	@echo "  make single       - Build single-GPU version"
	@echo "  make multi        - Build multi-GPU version"
	@echo "  make pincer       - Build pincer (bidirectional) version"
	@echo "  make standalone   - Build all standalone versions"
	@echo ""
	@echo "KXE (Permuted Scanning) targets:"
	@echo "  make kxe          - Build KXE single-GPU version"
	@echo "  make kxe-multi    - Build KXE multi-GPU version"
	@echo "  make kxe-pincer   - Build KXE pincer (2x speedup, requires even # GPUs)"
	@echo "  make kxe-all      - Build all KXE versions"
	@echo "  make test-kxe     - Build and run KXE tests"
	@echo ""
	@echo "Distributed mode targets:"
	@echo "  make server       - Build server (coordinator, no CUDA)"
	@echo "  make client       - Build client (GPU worker)"
	@echo "  make distributed  - Build server and client"
	@echo ""
	@echo "Build all:"
	@echo "  make everything   - Build all targets"
	@echo ""
	@echo "Utility:"
	@echo "  make clean        - Remove all binaries and objects"
	@echo "  make clean-obj    - Remove only object files"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Detected GPU architecture: SM$(GPU_ARCH)"

.PHONY: all single multi pincer both standalone server client distributed everything clean clean-obj help kxe kxe-multi kxe-pincer kxe-all test-kxe
