# HIPify Conversion Log
## SYNRA DIC Pipeline — CUDA → HIP/AMD Port
**Author:** Shubham More  
**Date:** April 2026  
**Task:** AMD/HIP acceleration track (assigned by Yash / Kohei-san)

---

## 1. Overview

This document records every CUDA API call found across the DIC pipeline source
files, whether it was converted automatically by `hipify-perl`, whether it
required manual porting, and the result.

**Files converted:**
- `hyper_dic_fusion_final.cu` → `dic_hip_fusion.hip`
- `dic_spectral_analysis.cu` → `dic_hip_spectral.hip` (in progress)
- `dic_dmd_analysis.cu` → `dic_hip_dmd.hip` (in progress)

**HIPify tool used:** `hipify-perl` (included in ROCm install)

**Command run:**
```bash
hipify-perl hyper_dic_fusion_final.cu > dic_hip_fusion.hip
hipify-perl dic_spectral_analysis.cu  > dic_hip_spectral.hip
hipify-perl dic_dmd_analysis.cu       > dic_hip_dmd.hip
```

---

## 2. Automatic Conversions (hipify-perl — zero manual effort)

These were handled completely automatically. No review needed beyond
confirming the substitution was made.

| CUDA original | HIP replacement | File(s) affected |
|---|---|---|
| `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` | All |
| `cudaError_t` | `hipError_t` | All |
| `cudaSuccess` | `hipSuccess` | All |
| `cudaGetErrorString` | `hipGetErrorString` | All |
| `cudaMalloc` | `hipMalloc` | All |
| `cudaFree` | `hipFree` | All |
| `cudaMemcpy` | `hipMemcpy` | All |
| `cudaMemset` | `hipMemset` | All |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | All |
| `cudaSetDevice` | `hipSetDevice` | All |
| `cudaGetDeviceCount` | `hipGetDeviceCount` | All |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` | All |
| `cudaGetLastError` | `hipGetLastError` | All |
| `cudaDeviceReset` | `hipDeviceReset` | All |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | All |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | All |
| `cudaMemcpyDeviceToDevice` | `hipMemcpyDeviceToDevice` | All |
| `cudaDeviceProp` | `hipDeviceProp_t` | All |
| `cudaStream_t` | `hipStream_t` | (future use) |

**Conversion rate: 100% for CUDA Runtime API calls.**

---

## 3. GPU Kernel Code — Zero Changes Required

This is the most important finding: **all GPU kernel code is identical
between CUDA and HIP.**

The following constructs work unchanged:

| Construct | Status |
|---|---|
| `__global__` kernel qualifier | Identical |
| `__device__` function qualifier | Identical |
| `__forceinline__` | Identical |
| `__restrict__` | Identical |
| `__shared__` shared memory | Identical |
| `blockIdx.x/y/z` | Identical |
| `threadIdx.x/y/z` | Identical |
| `blockDim.x/y/z` | Identical |
| `gridDim.x/y/z` | Identical |
| `dim3` | Identical |
| `atomicAdd`, `atomicCAS` | Identical |
| `__syncthreads()` | Identical |
| `fmaxf`, `fminf`, `sqrtf` | Identical |
| `sinf`, `cosf`, `expf`, `logf` | Identical |
| `atan2f`, `fabsf`, `powf` | Identical |
| `<<<grid, block>>>` launch syntax | Identical |

**This means the DIC ZNCC kernel, bilinear interpolation, and all 
mathematical computation code required zero changes.**

---

## 4. Manual Changes Required

### 4.1 Error handling macros

**Issue:** `CUDA_CHECK` macro referenced `cudaError_t` and `cudaGetErrorString`.  
**Fix:** Renamed macro to `HIP_CHECK`, substituted HIP types.  
**Effort:** 5 minutes — global search-and-replace.

```cpp
// BEFORE (CUDA):
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { ... cudaGetErrorString(_e) ... } \
} while(0)

// AFTER (HIP):
#define HIP_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { ... hipGetErrorString(_e) ... } \
} while(0)
```

---

### 4.2 cuFFT → rocFFT (dic_spectral_analysis.cu)

**Issue:** `#include <cufft.h>`, `cufftHandle`, `cufftPlanMany`, 
`cufftExecR2C`, `cufftExecC2R`, `cufftDestroy`, `CUFFT_R2C`, `CUFFT_C2R`.  
**Status:** Requires manual porting — hipify-perl does NOT convert cuFFT.  
**Fix:** Replace with rocFFT API.

```cpp
// BEFORE (cuFFT):
#include <cufft.h>
cufftHandle plan;
cufftPlanMany(&plan, rank, dims, NULL,1,N, NULL,1,N, CUFFT_R2C, n_pixels);
cufftExecR2C(plan, d_signal, d_fft);
cufftDestroy(plan);

// AFTER (rocFFT):
#include <rocfft/rocfft.h>
rocfft_plan plan;
rocfft_plan_create(&plan, rocfft_placement_notinplace,
                   rocfft_transform_type_real_forward,
                   rocfft_precision_single, 1, lengths, n_pixels, NULL);
rocfft_execute(plan, (void**)&d_signal, (void**)&d_fft, info);
rocfft_plan_destroy(plan);
```

**Note:** rocFFT API is more verbose than cuFFT. Estimated porting effort: 2–3 hours.  
**Link flag:** `-lrocfft` instead of `-lcufft`

---

### 4.3 cuSOLVER → rocSOLVER (dic_dmd_analysis.cu)

**Issue:** `#include <cusolverDn.h>`, `cusolverDnHandle_t`, 
`cusolverDnCreate`, `cusolverDnSgesvd_bufferSize`, `cusolverDnSgesvd`.  
**Status:** Requires manual porting.  
**Fix:** Replace with rocSOLVER API.

```cpp
// BEFORE (cuSOLVER):
#include <cusolverDn.h>
cusolverDnHandle_t cusolver;
cusolverDnCreate(&cusolver);
cusolverDnSgesvd(cusolver, 'S','S', m, n, A, lda, S, U, ldu, VT, ldvt,
                 work, lwork, NULL, devInfo);

// AFTER (rocSOLVER):
#include <rocsolver/rocsolver.h>
rocblas_handle handle;
rocblas_create_handle(&handle);
rocsolver_sgesvd(handle, rocblas_svect_singular, rocblas_svect_singular,
                 m, n, A, lda, S, U, ldu, VT, ldvt, E, rocblas_outofplace,
                 devInfo);
```

**Note:** rocSOLVER uses a rocBLAS handle, not its own handle.  
**Link flags:** `-lrocsolver -lrocblas` instead of `-lcusolver -lcublas`

---

### 4.4 cuBLAS → rocBLAS (dic_dmd_analysis.cu)

**Issue:** `#include <cublas_v2.h>`, `cublasHandle_t`, `cublasCreate`,
`cublasSgemm`, `CUBLAS_OP_N`, `CUBLAS_OP_T`.  
**Status:** Mostly handled by hipify-perl when using `#include <hipblas.h>`.  
**Fix:**

```cpp
// BEFORE (cuBLAS):
#include <cublas_v2.h>
cublasHandle_t cublas;
cublasCreate(&cublas);
cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
            A, lda, B, ldb, &beta, C, ldc);

// AFTER (rocBLAS):
#include <rocblas/rocblas.h>
rocblas_handle handle;
rocblas_create_handle(&handle);
rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
              m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
```

---

## 5. What Does NOT Convert (Blockers)

### 5.1 Device-specific intrinsics

| CUDA intrinsic | HIP equivalent | Status |
|---|---|---|
| `__ldg()` (read-only cache) | `__ldg()` | Supported in HIP |
| `__ballot_sync()` | `__ballot()` | Minor syntax diff |
| `__shfl_sync()` | `__shfl()` | Minor syntax diff |
| PTX inline assembly | Not supported | Not used in our code |

**Our DIC code does not use any warp-level primitives — no blockers here.**

### 5.2 ROCm GPU support matrix

**Critical check before deployment:** Not all AMD GPUs support ROCm.

| Device | GPU | ROCm supported? |
|---|---|---|
| Orange Pi 5 | Mali-G610 (ARM) | **NO** — use Vulkan |
| Orange Pi AIpro | Ascend 310B | **NO** — use Ascend SDK |
| AMD Mini PC (Ryzen 5000) | Radeon Vega 8 integrated | Limited (gfx90c) |
| AMD Mini PC (Ryzen 7000) | Radeon 780M integrated | YES (gfx1103) |
| AMD Mini PC with discrete RX GPU | RX 6000/7000 series | YES |

**Action required:** Run `rocminfo` on the target device and check the
reported `gfx` architecture code against the ROCm support matrix.

---

## 6. Compile Commands Summary

### dic_hip_fusion.hip (ZNCC DIC engine)
```bash
hipcc -Xcompiler -fopenmp dic_hip_fusion.hip \
      -o dic_hip_fusion -lgomp -O3 -std=c++14
```

### dic_hip_spectral.hip (FFT + CWT spectral analysis)
```bash
hipcc -Xcompiler -fopenmp dic_hip_spectral.hip \
      -o dic_hip_spectral -lgomp -lrocfft -O3 -std=c++14
```

### dic_hip_dmd.hip (DMD mode decomposition)
```bash
hipcc -Xcompiler -fopenmp dic_hip_dmd.hip \
      -o dic_hip_dmd -lgomp -lrocblas -lrocsolver -O3 -std=c++14
```

### hip_benchmark.hip (GPU vs CPU benchmark)
```bash
hipcc -Xcompiler -fopenmp hip_benchmark.hip \
      -o hip_benchmark -lgomp -O3 -std=c++14
```

---

## 7. Benchmark Results

*(Fill in after running hip_benchmark on the AMD device)*

| Metric | Value |
|---|---|
| GPU model | |
| ROCm version | |
| gfx architecture | |
| Image size | 512 × 512 |
| GPU time (avg 3 runs) | ___ ms |
| CPU time (estimated) | ___ ms |
| Speedup | ___x |
| GPU utilisation | ___% |
| Max result error | ___ px |

---

## 8. Decision: ROCm vs Vulkan Fallback

- If `rocminfo` shows a supported `gfx` architecture → **proceed with ROCm/HIP**
- If device shows Mali/Ascend/unsupported gfx → **switch to Vulkan compute shaders**
  - Vulkan Compute shaders can implement the same ZNCC kernel
  - More portable (runs on Intel, AMD, NVIDIA, Mali, Qualcomm Adreno)
  - More complex to write (GLSL compute shaders vs C++)
  - Flag this to Yash immediately if ROCm is not viable

---

*Document maintained by Shubham More — update benchmark results section 
after running on Orange Pi / Mini PC hardware.*
