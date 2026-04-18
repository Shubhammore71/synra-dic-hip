#!/bin/bash
# =============================================================================
#  setup_rocm.sh
#  Author  : Shubham | SYNRA Metrology Systems
#  Purpose : Install ROCm on AMD GPU device (Orange Pi / Mini PC / Ubuntu)
#            Verify GPU detection, install hipcc, test-compile HIP kernel
#
#  WHAT THIS SCRIPT DOES:
#  ───────────────────────
#  Step 1 — Detect Ubuntu version and CPU architecture
#  Step 2 — Add AMD ROCm apt repository
#  Step 3 — Install ROCm runtime + HIP + development libraries
#  Step 4 — Verify GPU is detected by rocminfo
#  Step 5 — Set PATH and LD_LIBRARY_PATH permanently
#  Step 6 — Test-compile and run a minimal HIP vector-add kernel
#  Step 7 — Print summary: GPU model, ROCm version, compute units
#
#  SUPPORTED HARDWARE:
#  ────────────────────
#  Orange Pi 5 / 5B / 5 Plus   — RK3588 with Mali-G610 (NOT ROCm supported —
#                                 use Vulkan fallback if this is your device)
#  Orange Pi AIpro              — Ascend NPU (different stack entirely)
#  AMD Mini PC (NUC-style)      — Ryzen with Radeon integrated/discrete GPU
#  Any x86 Ubuntu machine with  — AMD RX/Vega/RDNA discrete GPU
#
#  IMPORTANT — CHECK YOUR GPU MODEL FIRST:
#  ─────────────────────────────────────────
#  ROCm only supports specific AMD GPU architectures (RDNA, CDNA, GCN 5+).
#  Run: lspci | grep -i vga  to see your GPU before running this script.
#  See https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
#
#  RUN:
#  ─────
#  chmod +x setup_rocm.sh
#  sudo ./setup_rocm.sh
# =============================================================================

set -e  # exit immediately on any error

# ─── Colours for output ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║   SYNRA — ROCm/HIP Environment Setup Script          ║"
echo "  ║   For AMD GPU DIC Pipeline (Shubham More)            ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
#  STEP 1 — System detection
# =============================================================================
log_info "Step 1: Detecting system..."

UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "unknown")
ARCH=$(uname -m)
log_info "Ubuntu version : $UBUNTU_VERSION"
log_info "Architecture   : $ARCH"

# Detect AMD GPU
GPU_INFO=$(lspci 2>/dev/null | grep -i "amd\|radeon\|advanced micro\|vga" || echo "none found")
log_info "GPU detected   : $GPU_INFO"

# Check if this is an Orange Pi with ARM Mali GPU
if echo "$GPU_INFO" | grep -qi "mali\|rk35\|arm"; then
    log_warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_warn "MALI / ARM GPU DETECTED — ROCm does NOT support Mali GPUs."
    log_warn "This device (likely Orange Pi 5) requires Vulkan compute."
    log_warn "See section at bottom of this script for Vulkan setup."
    log_warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ROCM_SUPPORTED=false
else
    ROCM_SUPPORTED=true
    log_ok "AMD GPU detected — proceeding with ROCm installation."
fi

# =============================================================================
#  STEP 2 — Add ROCm apt repository
# =============================================================================
if [ "$ROCM_SUPPORTED" = true ]; then
    log_info "Step 2: Adding AMD ROCm apt repository..."

    # Install prerequisites
    apt-get update -qq
    apt-get install -y -qq curl wget gnupg2 lsb-release python3-pip

    # Add AMD GPG key
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

    # Add ROCm 6.x repository (latest stable as of 2025)
    # Adjust version number if a newer release is available
    ROCM_VERSION="6.2"
    if [ "$ARCH" = "x86_64" ]; then
        REPO_ARCH="amd64"
    elif [ "$ARCH" = "aarch64" ]; then
        REPO_ARCH="arm64"
        log_warn "aarch64 (ARM64) detected — ROCm arm64 support is limited."
        log_warn "Supported: some RK3588 / Raspberry Pi with AMD eGPU via USB-C."
    fi

    # Ubuntu codename
    CODENAME=$(lsb_release -cs)
    if [ "$UBUNTU_VERSION" = "22.04" ]; then
        CODENAME="jammy"
    elif [ "$UBUNTU_VERSION" = "24.04" ]; then
        CODENAME="noble"
    fi

    cat > /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=$REPO_ARCH signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION $CODENAME main
EOF

    cat > /etc/apt/preferences.d/rocm-pin-600 << EOF
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

    apt-get update -qq
    log_ok "ROCm repository added (ROCm $ROCM_VERSION, Ubuntu $CODENAME)"
fi

# =============================================================================
#  STEP 3 — Install ROCm packages
# =============================================================================
if [ "$ROCM_SUPPORTED" = true ]; then
    log_info "Step 3: Installing ROCm packages (this takes 5–15 minutes)..."

    # Core packages
    apt-get install -y rocm-dev rocm-libs hip-dev hipcc

    # Math libraries needed for DIC pipeline
    apt-get install -y rocblas-dev  # cuBLAS equivalent  → GEMM for DMD
    apt-get install -y rocsolver-dev # cuSOLVER equivalent → SVD for DMD
    apt-get install -y rocfft-dev    # cuFFT equivalent   → FFT for spectral analysis

    # Monitoring and profiling tools
    apt-get install -y rocm-smi-lib rocprofiler-dev

    log_ok "ROCm packages installed"

    # Add user to video and render groups (required for GPU access without root)
    CURRENT_USER=${SUDO_USER:-$USER}
    if [ -n "$CURRENT_USER" ] && [ "$CURRENT_USER" != "root" ]; then
        usermod -aG video,render "$CURRENT_USER"
        log_ok "Added $CURRENT_USER to video and render groups"
        log_warn "You must LOG OUT and LOG BACK IN for group changes to take effect."
    fi
fi

# =============================================================================
#  STEP 4 — Set PATH permanently
# =============================================================================
if [ "$ROCM_SUPPORTED" = true ]; then
    log_info "Step 4: Configuring PATH and LD_LIBRARY_PATH..."

    # Add to /etc/profile.d/ so it persists for all users and sessions
    cat > /etc/profile.d/rocm.sh << 'EOF'
# ROCm environment — added by SYNRA setup_rocm.sh
export PATH=/opt/rocm/bin:/opt/rocm/hip/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
export HIP_PLATFORM=amd
export ROCM_PATH=/opt/rocm
EOF

    # Also add to current session
    export PATH=/opt/rocm/bin:/opt/rocm/hip/bin:$PATH
    export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
    export HIP_PLATFORM=amd
    export ROCM_PATH=/opt/rocm

    log_ok "PATH configured. Source with: source /etc/profile.d/rocm.sh"
fi

# =============================================================================
#  STEP 5 — Verify GPU detection
# =============================================================================
log_info "Step 5: Verifying GPU detection..."

if command -v rocminfo &>/dev/null; then
    echo ""
    echo "  ── rocminfo output ──────────────────────────────────────"
    rocminfo 2>/dev/null | grep -E "Name|Compute|gfx|Agent" | head -20
    echo "  ─────────────────────────────────────────────────────────"
    log_ok "rocminfo ran successfully"
else
    log_warn "rocminfo not found — ROCm may not be installed correctly."
fi

if command -v hipinfo &>/dev/null; then
    echo ""
    echo "  ── hipinfo output ───────────────────────────────────────"
    hipinfo 2>/dev/null | head -20
    echo "  ─────────────────────────────────────────────────────────"
    log_ok "hipinfo ran successfully"
fi

if command -v rocm-smi &>/dev/null; then
    echo ""
    echo "  ── rocm-smi (GPU utilisation monitor) ───────────────────"
    rocm-smi 2>/dev/null || true
    echo "  ─────────────────────────────────────────────────────────"
fi

# =============================================================================
#  STEP 6 — Test compile minimal HIP kernel
# =============================================================================
log_info "Step 6: Test-compiling minimal HIP vector-add kernel..."

cat > /tmp/hip_test.hip << 'HIPTEST'
// Minimal HIP test: vector addition
// If this compiles and runs correctly, ROCm + hipcc are working.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include<vector>
#define HIP_CHECK(call) do { \
    hipError_t e = (call); \
    if (e != hipSuccess) { \
        fprintf(stderr, "[HIP ERROR] %s at %s:%d -> %s\n", \
            #call, __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__global__ void vec_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    const int N = 1024 * 1024;  // 1M elements

    // Get GPU info
    int dev_count = 0;
    HIP_CHECK(hipGetDeviceCount(&dev_count));
    printf("\n[HIP TEST] Devices found: %d\n", dev_count);

    for (int d = 0; d < dev_count; ++d) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, d));
        printf("[HIP TEST] GPU %d: %s  |  %.1f GB  |  %d CUs\n",
               d, prop.name,
               (float)prop.totalGlobalMem / (1024.f*1024.f*1024.f),
               prop.multiProcessorCount);
    }

    // Allocate and fill host arrays
    std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N, 0.0f);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, N * sizeof(float)));

    // Upload
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), N*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), N*sizeof(float), hipMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    hipLaunchKernelGGL(vec_add, dim3(blocks), dim3(threads), 0, 0,
                       d_A, d_B, d_C, N);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Download and verify
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, N*sizeof(float), hipMemcpyDeviceToHost));

    float max_err = 0.f;
    for (int i = 0; i < N; ++i)
        max_err = fmaxf(max_err, fabsf(h_C[i] - 3.0f));

    printf("[HIP TEST] Max error: %.2e  (expected 0)\n", max_err);
    if (max_err < 1e-5f)
        printf("[HIP TEST] PASS — ROCm + HIP working correctly on AMD GPU!\n\n");
    else
        printf("[HIP TEST] FAIL — results incorrect.\n\n");

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return (max_err < 1e-5f) ? 0 : 1;
}
HIPTEST

if command -v hipcc &>/dev/null; then
    log_info "Compiling hip_test.hip..."
    if hipcc -O2 -std=c++14 /tmp/hip_test.hip -o /tmp/hip_test 2>&1; then
        log_ok "Compilation successful"
        log_info "Running hip_test..."
        if /tmp/hip_test; then
            log_ok "HIP test PASSED — ROCm is fully functional!"
        else
            log_error "HIP test FAILED — check GPU support and ROCm installation."
        fi
    else
        log_error "hipcc compilation failed. Check ROCm installation."
    fi
else
    log_warn "hipcc not found in PATH. Try: source /etc/profile.d/rocm.sh"
fi

# =============================================================================
#  STEP 7 — Vulkan compute setup (fallback for unsupported GPUs)
# =============================================================================
echo ""
log_info "Step 7: Checking Vulkan compute availability (fallback path)..."

if ! command -v vulkaninfo &>/dev/null; then
    log_info "Installing Vulkan tools..."
    apt-get install -y -qq vulkan-tools libvulkan-dev glslang-tools
fi

if command -v vulkaninfo &>/dev/null; then
    echo ""
    echo "  ── Vulkan devices ────────────────────────────────────────"
    vulkaninfo --summary 2>/dev/null | grep -E "GPU|deviceName|apiVersion" | head -10 || \
    vulkaninfo 2>/dev/null | grep -E "deviceName|apiVersion|deviceType" | head -10
    echo "  ─────────────────────────────────────────────────────────"
    log_ok "Vulkan is available — can be used as fallback compute backend"
fi

# =============================================================================
#  SUMMARY
# =============================================================================
echo ""
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║           SETUP COMPLETE — SUMMARY                   ║"
echo "  ╠═══════════════════════════════════════════════════════╣"
if [ "$ROCM_SUPPORTED" = true ]; then
echo "  ║  ROCm installed      : YES                           ║"
echo "  ║  hipcc compiler      : $(command -v hipcc &>/dev/null && echo 'YES' || echo 'NO — check PATH    ')                         ║"
else
echo "  ║  ROCm installed      : SKIPPED (unsupported GPU)     ║"
echo "  ║  Vulkan fallback     : Check output above            ║"
fi
echo "  ╠═══════════════════════════════════════════════════════╣"
echo "  ║  NEXT STEPS:                                         ║"
echo "  ║  1. source /etc/profile.d/rocm.sh                   ║"
echo "  ║  2. rocminfo  (verify GPU)                           ║"
echo "  ║  3. hipcc dic_hip_fusion.hip -o dic_hip_fusion       ║"
echo "  ║     -lrocblas -lrocsolver -O3 -std=c++14             ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo ""
