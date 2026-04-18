# SYNRA DIC Pipeline — CUDA + HIP/AMD GPU Acceleration

**Project:** Micro-Vibration Measurement Using High-Speed Cameras with Ego-Vibration Cancellation  
**Company:** SYNRA株式会社  
**Author:** Shubham More  
**Hardware:** NVIDIA DGX Station (4× Tesla V100 32GB) · AMD Mini PC (ROCm target)

---

## What This Repository Contains

This repository implements a full GPU-accelerated Digital Image Correlation (DIC) pipeline for structural vibration measurement. The pipeline was developed on NVIDIA CUDA and ported to AMD HIP/ROCm to enable deployment on AMD GPU devices (Mini PC lab unit).

### The three-file analysis pipeline

```
Camera frames
     │
     ▼
┌─────────────────────────────┐
│  hyper_dic_fusion_final.cu  │   WHERE does the structure move?
│  (CUDA) / dic_hip_fusion.hip│   → Full-field displacement map u(x,y) per frame
│  (HIP)                      │   → ZNCC + bilinear interpolation + wind correction
└─────────────┬───────────────┘
              │  h_res[W × H × N_frames]
              ▼
┌─────────────────────────────┐
│  dic_spectral_analysis.cu   │   WHEN and at what FREQUENCY?
│                             │   → cuFFT batch FFT across 1M pixels
│                             │   → Morlet CWT scalogram (64 scales)
│                             │   → Narrow bandpass filter (Hann window)
└─────────────┬───────────────┘
              │  frequency maps, mode amplitudes
              ▼
┌─────────────────────────────┐
│  dic_dmd_analysis.cu        │   WHAT spatial MODE SHAPES?
│                             │   → Dynamic Mode Decomposition (Schmid 2010)
│                             │   → cuSOLVER SVD + CPU Francis QR eigendecomp
│                             │   → Full 1,048,576-pixel spatial mode shapes
└─────────────────────────────┘
```

---

## Repository Structure

```
synra-dic-hip/
│
├── cuda/                          ← Original CUDA source files (DGX Station)
│   ├── hyper_dic_fusion_final.cu  ← Multi-GPU DIC engine (ZNCC + wind fusion)
│   ├── dic_spectral_analysis.cu   ← FFT + CWT + bandpass analysis
│   └── dic_dmd_analysis.cu        ← Dynamic Mode Decomposition (FIXED v1.1)
│
├── hip/                           ← HIP/AMD port (ROCm target)
│   ├── dic_hip_fusion.hip         ← HIPified DIC engine (compiles on AMD + NVIDIA)
│   └── hip_benchmark.hip          ← GPU vs CPU benchmark
│
├── scripts/                       ← Setup and utility scripts
│   └── setup_rocm.sh              ← ROCm install + verification (run on AMD device)
│
├── docs/                          ← Technical documentation
│   ├── hipify_conversion_log.md   ← Every CUDA→HIP change documented
│   └── benchmark_results.txt      ← Benchmark output from DGX
│
└── README.md                      ← This file
```

---

## Quick Start — NVIDIA DGX Station (CUDA)

### Prerequisites
- CUDA 12.3
- NVIDIA DGX Station with Tesla V100 GPUs
- GCC with OpenMP support (`libgomp`)

### Clone and compile

```bash
git clone https://github.com/Shubhammore71/synra-dic-hip.git
cd synra-dic-hip
```

### Compile each pipeline stage

```bash
# Stage 1 — Multi-GPU DIC engine
nvcc -Xcompiler -fopenmp -arch=sm_70 cuda/hyper_dic_fusion_final.cu \
     -o dic_fusion -lgomp -O3 -std=c++14

# Stage 2 — Time-frequency spectral analysis
nvcc -Xcompiler -fopenmp -arch=sm_70 cuda/dic_spectral_analysis.cu \
     -o dic_spectral -lgomp -lcufft -O3 -std=c++14

# Stage 3 — Dynamic Mode Decomposition
nvcc -Xcompiler -fopenmp -arch=sm_70 cuda/dic_dmd_analysis.cu \
     -o dic_dmd -lgomp -lcusolver -lcublas -O3 -std=c++14
```

### Run

```bash
./dic_fusion       # Produces displacement field (validates at ~3.35 px max)
./dic_spectral     # Produces frequency maps (peaks at 47, 92, 153 Hz)
./dic_dmd          # Produces spatial mode shapes + frequencies
```

Expected output for `dic_fusion`:
```
  GPU 0 : Tesla V100-SXM2-32GB  |  32 GB
  [GPU 0]  rows    0-255  wind=(0.310,-0.170)
  ...
  DISPLACEMENT FIELD (4-GPU DIC)
  Max : 3.3541 px  ✓ (expected ~3.35 px)
```

---

## Quick Start — AMD GPU (HIP/ROCm)

### Step 1: Check your GPU first

```bash
lspci | grep -i vga
```

| What you see | Action |
|---|---|
| `AMD Radeon` / `Advanced Micro Devices` | Proceed with ROCm setup below |
| `Mali` / `ARM` (Orange Pi 5) | ROCm not supported — use Vulkan |
| `NVIDIA` | Use CUDA path above |

### Step 2: Install ROCm on the AMD device

```bash
chmod +x scripts/setup_rocm.sh
sudo ./scripts/setup_rocm.sh
```

This script:
- Detects Ubuntu version and CPU architecture
- Adds AMD ROCm 6.x apt repository
- Installs `rocm-dev`, `hip-dev`, `hipcc`, `rocblas-dev`, `rocsolver-dev`, `rocfft-dev`
- Adds your user to `video` and `render` groups
- Runs a test HIP kernel to verify everything works
- Checks Vulkan as fallback

After the script finishes, log out and back in for group changes, then:
```bash
source /etc/profile.d/rocm.sh
rocminfo | grep -E "Name|gfx"   # Verify GPU is visible
```

### Step 3: Compile HIP code

```bash
# DIC engine (HIP — compiles on AMD and NVIDIA)
hipcc -Xcompiler -fopenmp hip/dic_hip_fusion.hip \
      -o dic_hip_fusion -lgomp -O3 -std=c++14

# Benchmark
hipcc -Xcompiler -fopenmp hip/hip_benchmark.hip \
      -o hip_benchmark -lgomp -O3 -std=c++14
```

### Step 4: Run benchmark

```bash
# Run benchmark and save results
./hip_benchmark | tee docs/benchmark_results.txt

# Monitor GPU utilisation in a second terminal
watch -n 0.5 rocm-smi
```

Expected speedup on a discrete AMD GPU: **10–50×** over CPU baseline.

---

## Key Technical Details

### Stage 1 — DIC Engine (ZNCC + bilinear interpolation)

The core algorithm: for each of the 1,048,576 pixels, one CUDA/HIP thread independently executes:

**Step A — Coarse search:** Evaluate ZNCC at all 289 integer displacement candidates in ±8 px grid.

**Step B — Sub-pixel refinement:** Search ±1 px around the best integer match in 0.1 px steps.

**Step C — Wind correction:**
```
u_true = u_measured − α × wind_u
v_true = v_measured − α × wind_v
(α = 0.05, calibrate per rig)
```

**GPU memory per V100:** ~8 MB (two 1024×1024 float images). Well within 32 GB budget.  
**Multi-GPU:** Image partitioned into horizontal strips, one strip per GPU via OpenMP.

### Stage 2 — Spectral Analysis

Three analysis pipelines running on the same displacement time series:

- **Pipeline A (FFT):** `cufftPlanMany` executes 1,048,576 independent 1D FFTs simultaneously. Amplitude normalisation: `2|Xₖ|/N` for positive bins.
- **Pipeline B (CWT):** Morlet wavelet (σ=6), 64 log-spaced scales 5–500 Hz, frequency-domain convolution O(N log N).
- **Pipeline C (bandpass):** FFT → multiply Hann-windowed H(f) → IFFT. Zero phase distortion.

### Stage 3 — DMD

Implements Schmid (2010) exactly:

```
X  = [x₁ x₂ ... x_{N-1}]   n × (N-1) snapshot matrix
X' = [x₂ x₃ ... x_N    ]   n × (N-1) shifted matrix

SVD:  X = U·Σ·V*  (GPU — cuSOLVER Sgesvd)
Ã   = U*·X'·V·Σ⁻¹  (r×r, GPU — cuBLAS GEMM)
eig(Ã) → λₖ, ỹₖ   (CPU — Francis QR, ~0.1 ms for 32×32)
φₖ  = U·ỹₖ         (GPU kernel — full spatial mode)

fₖ = |arg(λₖ)| · fs / 2π   [Hz]
σₖ = log|λₖ| · fs           [1/s]
```

**Bug fixed in v1.1:** `cusolverDnSgeev` does not exist in cuSOLVER's public API. Replaced with CPU Francis QR algorithm (implemented from scratch). This is the architecturally correct approach — PyDMD and MATLAB DMD both solve the small r×r eigenvalue problem on CPU.

---

## CUDA → HIP Conversion Summary

| Category | Status |
|---|---|
| CUDA Runtime API (`cuda*` → `hip*`) | Automatic via hipify-perl |
| GPU kernels (`__global__`, `blockIdx`, math) | Zero changes — identical |
| cuFFT → rocFFT | Manual (2–3 h) — see docs/hipify_conversion_log.md |
| cuBLAS → rocBLAS | Manual (1 h) |
| cuSOLVER → rocSOLVER | Manual (1 h) |

Full per-call conversion table: [`docs/hipify_conversion_log.md`](docs/hipify_conversion_log.md)

### Compile flags comparison

| Purpose | CUDA (DGX) | HIP (AMD) |
|---|---|---|
| Compiler | `nvcc` | `hipcc` |
| OpenMP | `-Xcompiler -fopenmp -lgomp` | `-Xcompiler -fopenmp -lgomp` |
| FFT | `-lcufft` | `-lrocfft` |
| BLAS | `-lcublas` | `-lrocblas` |
| Solver | `-lcusolver` | `-lrocsolver` |

---

## Hardware Compatibility

| Device | GPU | CUDA | HIP/ROCm |
|---|---|---|---|
| NVIDIA DGX Station | 4× Tesla V100 SXM2 | ✓ Tested | ✓ (via hipcc CUDA backend) |
| AMD Mini PC (Ryzen 7000) | Radeon 780M (gfx1103) | — | ✓ ROCm 6.x |
| AMD Mini PC (Ryzen 5000) | Radeon Vega 8 | — | Limited |
| Orange Pi 5 | Mali-G610 (ARM) | — | ✗ Use Vulkan |

---

## Troubleshooting

**`hipcc: command not found`**
```bash
source /etc/profile.d/rocm.sh
# or:
export PATH=/opt/rocm/bin:$PATH
```

**`No HIP GPU found` at runtime**
```bash
# Check GPU is visible
rocminfo | grep -E "Name|gfx"
# Check group membership
groups   # should include 'video' and 'render'
# If missing: log out and back in after running setup_rocm.sh
```

**`cusolverDnSgeev` compile error (CUDA)**
This was a bug in v1.0. The fixed `dic_dmd_analysis.cu` (v1.1) replaces it with CPU Francis QR. Use the version in this repo.

**Vulkan fallback (Orange Pi / unsupported GPU)**
```bash
sudo apt install vulkan-tools
vulkaninfo --summary   # check Vulkan GPU is listed
# Vulkan compute shader implementation: contact Yuji Teshima-san
# (has prior Vulkan implementation experience per Kohei-san)
```

---

## References

- Schmid, P.J. (2010). Dynamic mode decomposition of numerical and experimental data. *Journal of Fluid Mechanics*, 656, 5–28.
- Hua, Y. & Sarkar, T.K. (1990). Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in noise. *IEEE Trans. ASSP*, 38(5), 814–824.
- AMD ROCm Documentation: https://rocm.docs.amd.com
- HIP Programming Guide: https://rocm.docs.amd.com/projects/HIP/en/latest/

---

## Contact

**Shubham More** — SYNRA株式会社  
Project: エゴ振動キャンセル技術を用いたドローン・AGV搭載型高速カメラによる微小振動計測  
GitHub: https://github.com/Shubhammore71
