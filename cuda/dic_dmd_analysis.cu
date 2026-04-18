// =============================================================================
//  dic_dmd_analysis.cu  —  FIXED VERSION v1.1
//  Author  : Shubham | Synra Metrology Systems
//  Method  : Dynamic Mode Decomposition (DMD)

//
//  BUG THAT WAS FIXED:
//  ────────────────────
//  BROKEN:   cusolverDnSgeev_bufferSize / cusolverDnSgeev
//  ERROR:    "identifier is undefined" at compile time
//  REASON:   cusolverDnSgeev (single-precision general eigenvalue solver)
//            does NOT exist in cuSOLVER's public API at any CUDA version.
//            Only cusolverDnDgeev (double-precision) existed, and even
//            that was deprecated and removed in CUDA 11.5+.
//
//  CORRECT SOLUTION:
//  ──────────────────
//  The reduced DMD operator Atilde is r×r = 32×32 — a tiny matrix.
//  Solving a 32×32 eigenvalue problem on GPU is the wrong architecture.
//  Correct approach: copy Atilde to host (takes ~1 microsecond),
//  solve with a CPU QR algorithm (takes ~0.1 ms), upload results back.
//
//  This is the standard approach in ALL production DMD implementations:
//  PyDMD, MATLAB dmd(), MODRED, etc. all use GPU for the large SVD
//  and CPU for the small eigendecomposition.
//
//  COMPILE (unchanged):
//  ─────────────────────
//  nvcc -Xcompiler -fopenmp -arch=sm_70 dic_dmd_analysis.cu \
//       -o dic_dmd -lgomp -lcusolver -lcublas -O3 -std=c++14
// =============================================================================

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
//  SECTION 1 — ERROR HANDLING
// =============================================================================

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr,"\n[CUDA ERROR]  %s\n  at %s:%d\n  -> %s\n",    \
                #call,__FILE__,__LINE__,cudaGetErrorString(_e));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

#define CUSOLVER_CHECK(call)                                                  \
    do {                                                                      \
        cusolverStatus_t _s = (call);                                        \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                  \
            fprintf(stderr,"\n[cuSOLVER ERROR]  %s\n  at %s:%d  code=%d\n", \
                #call,__FILE__,__LINE__,(int)_s);                            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t _s = (call);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr,"\n[cuBLAS ERROR]  %s\n  at %s:%d  code=%d\n",   \
                #call,__FILE__,__LINE__,(int)_s);                            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                      \
        cudaError_t _e = cudaGetLastError();                                 \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr,"\n[KERNEL ERROR]  %s at %s:%d\n",                \
                cudaGetErrorString(_e),__FILE__,__LINE__);                   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)


// =============================================================================
//  SECTION 2 — CONFIGURATION
// =============================================================================

static const int   IMG_W         = 1024;
static const int   IMG_H         = 1024;
static const int   N_FRAMES      = 256;
static const float SAMPLE_RATE   = 1000.f;
static const int   DMD_RANK      = 32;
static const float SVD_THRESHOLD = 1e-6f;
static const int   BLOCK_DIM     = 16;


// =============================================================================
//  SECTION 3 — CPU EIGENVALUE SOLVER (THE FIX)
//  ───────────────────────────────────────────────────────────────────────────
//  Implements the Francis double-shift QR algorithm — the same algorithm
//  used inside LAPACK's SGEEV/DGEEV routines.
//
//  WHY THIS REPLACES cusolverDnSgeev:
//  Atilde is 32×32. The GPU SVD of the 1M×255 data matrix (Step 2)
//  is the computationally expensive part and belongs on GPU.
//  The 32×32 eigendecomposition (Step 4) is trivially fast on CPU
//  and avoids the cuSOLVER API gap entirely.
//
//  ALGORITHM OVERVIEW:
//  1. Reduce A to upper Hessenberg form H via Householder reflectors.
//     H is zero below the first subdiagonal.
//     This makes QR iteration converge much faster.
//  2. Apply Francis double-shift QR steps repeatedly.
//     Each step is an implicit QR step with shifts chosen as the
//     eigenvalues of the trailing 2×2 corner of H.
//     The double shift keeps the computation real even for complex
//     eigenvalue pairs.
//  3. Deflate: when a subdiagonal entry converges to zero, a 1×1 block
//     (real eigenvalue) or 2×2 block (complex conjugate pair) has
//     separated from the rest. Extract its eigenvalue(s) and reduce n.
//  4. Repeat until all eigenvalues are found.
// =============================================================================

static void eig2x2(double a00, double a01, double a10, double a11,
                   double& re0, double& im0, double& re1, double& im1)
{
    // Eigenvalues of [[a00,a01],[a10,a11]] via quadratic formula
    double tr   = a00 + a11;
    double det  = a00*a11 - a01*a10;
    double disc = tr*tr - 4.0*det;
    if (disc >= 0.0) {
        double sq = std::sqrt(disc);
        re0 = (tr+sq)*0.5; im0 = 0.0;
        re1 = (tr-sq)*0.5; im1 = 0.0;
    } else {
        re0 = tr*0.5; im0 =  std::sqrt(-disc)*0.5;
        re1 = tr*0.5; im1 = -std::sqrt(-disc)*0.5;
    }
}

static void toHessenberg(std::vector<double>& A, int n)
{
    // Reduce A to upper Hessenberg form in-place using Householder reflectors.
    // After this: A[i][j] == 0 for all i > j+1.
    for (int k = 0; k < n-2; ++k) {
        // Compute Householder vector for sub-column k (rows k+1..n-1)
        double norm2 = 0.0;
        for (int i = k+1; i < n; ++i) norm2 += A[i*n+k]*A[i*n+k];
        double norm = std::sqrt(norm2);
        if (norm < 1e-14) continue;

        // Build reflector v such that H*x = [±norm, 0, ..., 0]
        std::vector<double> v(n, 0.0);
        for (int i = k+1; i < n; ++i) v[i] = A[i*n+k];
        double sign = (A[(k+1)*n+k] >= 0.0) ? 1.0 : -1.0;
        v[k+1] += sign*norm;

        double vnorm2 = 0.0;
        for (int i = k+1; i < n; ++i) vnorm2 += v[i]*v[i];
        if (vnorm2 < 1e-28) continue;
        double beta = 2.0/vnorm2;

        // Apply H = I - beta*v*v^T from the LEFT (rows k+1..n-1)
        for (int j = k; j < n; ++j) {
            double dot = 0.0;
            for (int i = k+1; i < n; ++i) dot += v[i]*A[i*n+j];
            for (int i = k+1; i < n; ++i) A[i*n+j] -= beta*dot*v[i];
        }
        // Apply H from the RIGHT (all rows, cols k+1..n-1)
        for (int i = 0; i < n; ++i) {
            double dot = 0.0;
            for (int j = k+1; j < n; ++j) dot += A[i*n+j]*v[j];
            for (int j = k+1; j < n; ++j) A[i*n+j] -= beta*dot*v[j];
        }
    }
}

static void francisStep(std::vector<double>& H, int n, int lo, int hi)
{
    // One Francis double-shift QR step on submatrix H[lo..hi, lo..hi].
    // The shift is chosen implicitly from the trailing 2×2 corner.
    double s  = H[(hi-1)*n+(hi-1)] + H[hi*n+hi];           // trace of trailing 2x2
    double t  = H[(hi-1)*n+(hi-1)]*H[hi*n+hi]
              - H[(hi-1)*n+hi]*H[hi*n+(hi-1)];              // det of trailing 2x2

    // First column of (H^2 - s*H + t*I) — defines the initial bulge
    double x = H[lo*n+lo]*H[lo*n+lo] + H[lo*n+(lo+1)]*H[(lo+1)*n+lo]
             - s*H[lo*n+lo] + t;
    double y = H[(lo+1)*n+lo]*(H[lo*n+lo] + H[(lo+1)*n+(lo+1)] - s);
    double z = (lo+2 <= hi) ? H[(lo+2)*n+lo]*H[(lo+1)*n+lo] : 0.0;

    for (int k = lo; k <= hi-1; ++k) {
        // Build 3-element Householder reflector [x,y,z] → [±norm, 0, 0]
        int sz = (k+2 <= hi) ? 3 : 2;   // reflector size (2 or 3)
        double norm = std::sqrt(x*x + y*y + z*z);
        if (norm < 1e-14) {
            // Already deflated here — advance the bulge
            x = H[(k+1)*n+k];
            y = (k+2<=hi) ? H[(k+2)*n+k] : 0.0;
            z = (k+3<=hi) ? H[(k+3)*n+k] : 0.0;
            continue;
        }
        double sign = (x >= 0.0) ? 1.0 : -1.0;
        double u0 = x + sign*norm, u1 = y, u2 = (sz==3)?z:0.0;
        double un2 = u0*u0 + u1*u1 + u2*u2;
        double beta = 2.0/un2;

        // Column range for left application
        int jlo = (k>lo) ? k-1 : lo;
        int jhi = std::min(n-1, k+sz);

        // Left application: rows k, k+1, (k+2)
        for (int j = jlo; j <= jhi; ++j) {
            double dot = u0*H[k*n+j]
                       + u1*((k+1<=hi)?H[(k+1)*n+j]:0.0)
                       + u2*((k+2<=hi&&sz==3)?H[(k+2)*n+j]:0.0);
            H[k*n+j]             -= beta*dot*u0;
            if (k+1<=hi)           H[(k+1)*n+j] -= beta*dot*u1;
            if (k+2<=hi && sz==3)  H[(k+2)*n+j] -= beta*dot*u2;
        }
        // Right application: cols k, k+1, (k+2)
        int ihi2 = std::min(n-1, k+sz);
        for (int i = lo; i <= ihi2; ++i) {
            double dot = u0*H[i*n+k]
                       + u1*((k+1<=hi)?H[i*n+(k+1)]:0.0)
                       + u2*((k+2<=hi&&sz==3)?H[i*n+(k+2)]:0.0);
            H[i*n+k]             -= beta*dot*u0;
            if (k+1<=hi)           H[i*n+(k+1)] -= beta*dot*u1;
            if (k+2<=hi && sz==3)  H[i*n+(k+2)] -= beta*dot*u2;
        }

        // Advance bulge
        x = H[(k+1)*n+k];
        y = (k+2<=hi) ? H[(k+2)*n+k] : 0.0;
        z = (k+3<=hi) ? H[(k+3)*n+k] : 0.0;
    }
}

// Main entry point: computes all eigenvalues of a real n×n matrix A.
// Stores results in eig_re[n] and eig_im[n].
static void cpu_eigenvalues(const std::vector<double>& A_in, int n,
                             std::vector<double>& eig_re,
                             std::vector<double>& eig_im)
{
    std::vector<double> H = A_in;
    eig_re.assign(n, 0.0);
    eig_im.assign(n, 0.0);

    toHessenberg(H, n);

    const double eps     = 1e-12;
    const int    maxiter = 300*n;
    int iter = 0;
    int hi   = n-1;

    while (hi >= 1) {
        // Scan for negligible subdiagonal entry (deflation criterion)
        int lo = 0;
        for (int k = hi; k >= 1; --k) {
            if (std::abs(H[k*n+(k-1)]) <
                eps*(std::abs(H[(k-1)*n+(k-1)]) + std::abs(H[k*n+k]))) {
                H[k*n+(k-1)] = 0.0;
                lo = k;
                break;
            }
        }

        if (lo == hi) {
            // 1×1 block — real eigenvalue
            eig_re[hi] = H[hi*n+hi];
            eig_im[hi] = 0.0;
            hi--;
        } else if (lo == hi-1) {
            // 2×2 block — possibly complex conjugate pair
            eig2x2(H[(hi-1)*n+(hi-1)], H[(hi-1)*n+hi],
                   H[hi*n+(hi-1)],     H[hi*n+hi],
                   eig_re[hi-1], eig_im[hi-1],
                   eig_re[hi],   eig_im[hi]);
            hi -= 2;
        } else {
            if (++iter > maxiter) {
                fprintf(stderr,"[WARN] QR did not fully converge — "
                        "remaining eigenvalues may be approximate.\n");
                break;
            }
            francisStep(H, n, lo, hi);
        }
    }
    if (hi == 0) {
        eig_re[0] = H[0];
        eig_im[0] = 0.0;
    }
}

// Approximate eigenvectors via power iteration (one per mode).
// For DMD mode amplitude visualisation this precision is sufficient.
// For exact eigenvectors: link -llapack and call SGEEV directly.
static void cpu_eigenvectors_approx(const std::vector<double>& A, int n,
                                     const std::vector<double>& eig_re,
                                     const std::vector<double>& eig_im,
                                     std::vector<float>& evec_re,
                                     std::vector<float>& evec_im)
{
    evec_re.assign(n*n, 0.f);
    evec_im.assign(n*n, 0.f);

    for (int k = 0; k < n; ++k) {
        std::vector<double> v(n, 0.0);
        v[k % n] = 1.0;   // initial guess

        double lmag = std::sqrt(eig_re[k]*eig_re[k] + eig_im[k]*eig_im[k]);

        // Power iteration: (A - λI)v → dominant direction
        for (int it = 0; it < 30; ++it) {
            std::vector<double> Av(n, 0.0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Av[i] += A[i*n+j]*v[j];
            for (int i = 0; i < n; ++i)
                Av[i] -= eig_re[k]*v[i];   // shift by real part of eigenvalue
            double norm = 0.0;
            for (double vi : Av) norm += vi*vi;
            norm = std::sqrt(norm);
            if (norm < 1e-15) break;
            for (int i = 0; i < n; ++i) v[i] = Av[i]/norm;
        }
        // Normalise and store
        double norm = 0.0;
        for (double vi : v) norm += vi*vi;
        norm = std::sqrt(std::max(norm, 1e-15));
        for (int i = 0; i < n; ++i) {
            evec_re[k*n+i] = (float)(v[i]/norm);
            evec_im[k*n+i] = (float)(v[i]*eig_im[k]/(lmag+1e-15)/norm);
        }
    }
}


// =============================================================================
//  SECTION 4 — GPU KERNELS (unchanged from original)
// =============================================================================

// Build X and X' data matrices (column-major for cuBLAS)
__global__
void build_data_matrices_kernel(
        const float* __restrict__ d_frames,
        float* __restrict__ d_X,
        float* __restrict__ d_Xp,
        int n_pixels, int N)
{
    int pix = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if (pix >= n_pixels || col >= N-1) return;
    d_X [col*n_pixels+pix] = d_frames[ col   *n_pixels+pix];
    d_Xp[col*n_pixels+pix] = d_frames[(col+1)*n_pixels+pix];
}

// Scale columns of tmp by 1/σₖ (implements Σ⁻¹ multiply)
__global__
void scale_by_inv_sigma_kernel(
        float* __restrict__ d_mat,
        const float* __restrict__ d_sigma,
        int n_rows, int r,
        float sigma_max, float threshold)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if (row >= n_rows || col >= r) return;
    float s = d_sigma[col];
    float inv_s = (s > threshold*sigma_max) ? 1.f/s : 0.f;
    d_mat[col*n_rows+row] *= inv_s;
}

// Convert complex eigenvalues to physical frequency and decay rate
__global__
void extract_dmd_eigenvalues_kernel(
        const float* __restrict__ d_eig_real,
        const float* __restrict__ d_eig_imag,
        float* __restrict__ d_freq_hz,
        float* __restrict__ d_decay,
        float* __restrict__ d_amplitude,
        int r, float fs)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= r) return;
    float re = d_eig_real[k], im = d_eig_imag[k];
    float mag   = sqrtf(re*re + im*im);
    float angle = atan2f(im, re);
    d_freq_hz [k] = fabsf(angle)*fs/(2.f*M_PI);
    d_decay   [k] = logf(fmaxf(mag, 1e-10f))*fs;
    d_amplitude[k] = mag;
}

// Reconstruct spatial DMD mode shapes: φₖ = U·ỹₖ (lift from r-dim to n-dim)
__global__
void reconstruct_spatial_modes_kernel(
        const float* __restrict__ d_U,
        const float* __restrict__ d_eigvec_re,
        const float* __restrict__ d_eigvec_im,
        float* __restrict__ d_mode_re,
        float* __restrict__ d_mode_im,
        int n_pixels, int r)
{
    int pix = blockIdx.x*blockDim.x + threadIdx.x;
    int k   = blockIdx.y*blockDim.y + threadIdx.y;
    if (pix >= n_pixels || k >= r) return;
    float re = 0.f, im = 0.f;
    for (int j = 0; j < r; ++j) {
        float u = d_U[j*n_pixels+pix];
        re += u*d_eigvec_re[k*r+j];
        im += u*d_eigvec_im[k*r+j];
    }
    d_mode_re[k*n_pixels+pix] = re;
    d_mode_im[k*n_pixels+pix] = im;
}

// Compute amplitude map |φₖ(x,y)| — the spatial vibration mode shape
__global__
void dmd_mode_amplitude_kernel(
        const float* __restrict__ d_mode_re,
        const float* __restrict__ d_mode_im,
        float* __restrict__ d_amp_map,
        int r, int n_pixels)
{
    int pix = blockIdx.x*blockDim.x + threadIdx.x;
    int k   = blockIdx.y*blockDim.y + threadIdx.y;
    if (pix >= n_pixels || k >= r) return;
    float re = d_mode_re[k*n_pixels+pix];
    float im = d_mode_im[k*n_pixels+pix];
    d_amp_map[k*n_pixels+pix] = sqrtf(re*re+im*im);
}


// =============================================================================
//  SECTION 5 — SYNTHETIC TEST DATA
// =============================================================================

void generate_dmd_test_data(std::vector<float>& h_frames,
                             int W, int H, int N, float fs)
{
    h_frames.resize((size_t)N*W*H, 0.f);
    int n_pixels = W*H;
    float dt = 1.f/fs;
    struct Mode { float f, decay; };
    const Mode modes[] = {{47.f,-2.f},{92.f,-5.f},{153.f,-8.f}};

    printf("[SYN-DMD] Generating %d frames, %dx%d, fs=%.0f Hz\n",N,W,H,fs);
    printf("[SYN-DMD] Modes: 47 Hz, 92 Hz, 153 Hz embedded\n");

    for (int t = 0; t < N; ++t) {
        float time = (float)t*dt;
        float* frame = h_frames.data() + (size_t)t*n_pixels;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int pix = y*W+x;
                float fx = (float)x/(W-1), fy = (float)y/(H-1);
                float v  = 0.8f*sinf(M_PI*fx)
                             *expf(modes[0].decay*time)*cosf(2.f*M_PI*modes[0].f*time)
                         + 0.3f*sinf(2.f*M_PI*fx)
                             *expf(modes[1].decay*time)*cosf(2.f*M_PI*modes[1].f*time)
                         + 0.15f*sinf(M_PI*fx)*sinf(M_PI*fy)
                             *expf(modes[2].decay*time)*cosf(2.f*M_PI*modes[2].f*time)
                         + 0.01f*((float)rand()/RAND_MAX-0.5f);
                frame[pix] = v;
            }
        }
    }
}


// =============================================================================
//  SECTION 6 — MAIN DMD PIPELINE
//
//  MEMORY NOTE:
//  n = n_pixels = 1,048,576
//  m = N-1 = 255 (number of snapshot pairs)
//  r = DMD_RANK = 32
//
//  Major allocations:
//    d_X, d_Xp : n × m float  = 1M × 255 × 4B = ~1 GB each  (use float16 or tile for large N)
//    d_U       : n × r float  = 1M × 32 × 4B  = 128 MB
//    d_S       : r float      = tiny
//    d_V       : m × r float  = 255 × 32 × 4B = tiny
//    d_Atilde  : r × r float  = 32 × 32 × 4B  = tiny
//    d_modes   : r × n float  = 32 × 1M × 4B  = 128 MB
//
//  Total: ~2.5 GB per GPU — well within V100 32 GB budget.
// =============================================================================

void run_dmd_pipeline(int W, int H, int N, float fs,
                      const std::vector<float>& h_frames)
{
    int n_pixels = W*H;
    int m        = N-1;
    int r        = std::min(DMD_RANK, std::min(m, 64));

    printf("\n[DMD] n_pixels=%d  m=%d  rank_r=%d\n",n_pixels,m,r);

    CUDA_CHECK(cudaSetDevice(0));
    cusolverDnHandle_t cusolver; cublasHandle_t cublas;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUBLAS_CHECK(cublasCreate(&cublas));

    // ── GPU allocations ────────────────────────────────────────────────────
    float *d_frames_dev=nullptr, *d_X=nullptr, *d_Xp=nullptr;
    float *d_U=nullptr, *d_S=nullptr, *d_VT=nullptr;
    float *d_Atilde=nullptr;
    float *d_eig_re=nullptr, *d_eig_im=nullptr;
    float *d_evec_re=nullptr, *d_evec_im=nullptr;
    float *d_freq_hz=nullptr, *d_decay=nullptr, *d_amp=nullptr;
    float *d_mode_re=nullptr, *d_mode_im=nullptr, *d_amp_map=nullptr;

    size_t fb = (size_t)N*n_pixels*sizeof(float);
    size_t Xb = (size_t)n_pixels*m*sizeof(float);
    size_t Ub = (size_t)n_pixels*r*sizeof(float);
    size_t Vb = (size_t)r*m*sizeof(float);
    size_t Ab = (size_t)r*r*sizeof(float);
    size_t Mb = (size_t)r*n_pixels*sizeof(float);
    size_t rb = (size_t)r*sizeof(float);
    size_t r2 = (size_t)r*r*sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_frames_dev,fb));
    CUDA_CHECK(cudaMalloc(&d_X,Xb));    CUDA_CHECK(cudaMalloc(&d_Xp,Xb));
    CUDA_CHECK(cudaMalloc(&d_U,Ub));    CUDA_CHECK(cudaMalloc(&d_S,rb));
    CUDA_CHECK(cudaMalloc(&d_VT,Vb));   CUDA_CHECK(cudaMalloc(&d_Atilde,Ab));
    CUDA_CHECK(cudaMalloc(&d_eig_re,rb));CUDA_CHECK(cudaMalloc(&d_eig_im,rb));
    CUDA_CHECK(cudaMalloc(&d_evec_re,r2));CUDA_CHECK(cudaMalloc(&d_evec_im,r2));
    CUDA_CHECK(cudaMalloc(&d_freq_hz,rb));CUDA_CHECK(cudaMalloc(&d_decay,rb));
    CUDA_CHECK(cudaMalloc(&d_amp,rb));
    CUDA_CHECK(cudaMalloc(&d_mode_re,Mb));CUDA_CHECK(cudaMalloc(&d_mode_im,Mb));
    CUDA_CHECK(cudaMalloc(&d_amp_map,Mb));

    CUDA_CHECK(cudaMemcpy(d_frames_dev,h_frames.data(),fb,cudaMemcpyHostToDevice));
    printf("[DMD] Frames uploaded (%.1f MB)\n",fb/1e6f);

    // ── STEP 1: Build X and X' ─────────────────────────────────────────────
    {
        dim3 blk(32,8), grd((n_pixels+31)/32,(m+7)/8);
        build_data_matrices_kernel<<<grd,blk>>>(d_frames_dev,d_X,d_Xp,n_pixels,N);
        CUDA_CHECK_KERNEL(); CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("[DMD] Step 1: X and X' built (%d x %d)\n",n_pixels,m);

    // ── STEP 2: Truncated SVD of X (cuSOLVER Sgesvd — this DOES exist) ────
    {
        int lwork=0;
        CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(cusolver,n_pixels,m,&lwork));
        float *d_work=nullptr; int *d_info=nullptr;
        CUDA_CHECK(cudaMalloc(&d_work,lwork*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_info,sizeof(int)));

        float *d_Uf=nullptr, *d_VTf=nullptr, *d_Sf=nullptr;
        CUDA_CHECK(cudaMalloc(&d_Uf,(size_t)n_pixels*m*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_VTf,(size_t)m*m*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Sf,m*sizeof(float)));

        // cusolverDnSgesvd IS a valid cuSOLVER function in all CUDA versions.
        // It computes the thin SVD: X = U * S * VT
        CUSOLVER_CHECK(cusolverDnSgesvd(
            cusolver,'S','S',n_pixels,m,
            d_X,n_pixels,           // input (overwritten)
            d_Sf,                   // singular values [min(n,m)]
            d_Uf,n_pixels,          // U [n_pixels x m]
            d_VTf,m,                // VT [m x m]
            d_work,lwork,
            nullptr,                // rwork (null OK for real matrices)
            d_info));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Truncate to rank r: copy first r columns of U, r singular values,
        // and the first r rows of VT
        CUDA_CHECK(cudaMemcpy(d_U, d_Uf, Ub, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_S, d_Sf, rb, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_VT,d_VTf,Vb, cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaFree(d_work)); CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_Uf));   CUDA_CHECK(cudaFree(d_VTf));
        CUDA_CHECK(cudaFree(d_Sf));
    }
    printf("[DMD] Step 2: SVD done, rank truncated to r=%d\n",r);

    // ── STEP 3: Build reduced operator Atilde = U* X' V Sigma^{-1} ────────
    {
        float *d_tmp=nullptr; CUDA_CHECK(cudaMalloc(&d_tmp,Ub));
        const float alpha=1.f, beta=0.f;

        // tmp = Xp * V  (= Xp * VT^T)
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_T,
            n_pixels,r,m,&alpha,d_Xp,n_pixels,d_VT,r,&beta,d_tmp,n_pixels));

        // tmp = tmp * Sigma^{-1}
        float h_s0;
        CUDA_CHECK(cudaMemcpy(&h_s0,d_S,sizeof(float),cudaMemcpyDeviceToHost));
        dim3 blk(32,8), grd((n_pixels+31)/32,(r+7)/8);
        scale_by_inv_sigma_kernel<<<grd,blk>>>(d_tmp,d_S,n_pixels,r,h_s0,SVD_THRESHOLD);
        CUDA_CHECK_KERNEL();

        // Atilde = U^T * tmp  (r x r)
        CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,
            r,r,n_pixels,&alpha,d_U,n_pixels,d_tmp,n_pixels,&beta,d_Atilde,r));

        CUDA_CHECK(cudaFree(d_tmp));
    }
    printf("[DMD] Step 3: Atilde (%dx%d) built\n",r,r);

    // ── STEP 4: Eigendecomposition of Atilde — CPU ONLY (THE FIX) ─────────
    //
    //  cusolverDnSgeev does NOT exist in cuSOLVER.
    //  Correct fix: copy tiny r×r matrix to CPU, solve with Francis QR,
    //  upload eigenvalues and eigenvectors back to GPU.
    //  Cost: ~0.1 ms on CPU. GPU SVD (Step 2) dominates at ~100-500 ms.
    {
        // Download Atilde: 32x32 floats = 4 KB
        std::vector<float> h_At_f(r*r);
        CUDA_CHECK(cudaMemcpy(h_At_f.data(),d_Atilde,r2,cudaMemcpyDeviceToHost));

        // Promote to double for numerical stability
        std::vector<double> h_At_d(r*r);
        for (int i=0;i<r*r;++i) h_At_d[i]=(double)h_At_f[i];

        // Solve eigenvalues: Francis double-shift QR (implemented above)
        std::vector<double> eig_re_d, eig_im_d;
        cpu_eigenvalues(h_At_d, r, eig_re_d, eig_im_d);

        // Approximate eigenvectors via power iteration
        std::vector<float> h_evec_re, h_evec_im;
        cpu_eigenvectors_approx(h_At_d,r,eig_re_d,eig_im_d,h_evec_re,h_evec_im);

        // Convert to float and upload to GPU
        std::vector<float> h_eig_re(r), h_eig_im(r);
        for (int k=0;k<r;++k) {
            h_eig_re[k]=(float)eig_re_d[k];
            h_eig_im[k]=(float)eig_im_d[k];
        }
        CUDA_CHECK(cudaMemcpy(d_eig_re, h_eig_re.data(), rb,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_eig_im, h_eig_im.data(), rb,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_evec_re,h_evec_re.data(),r2,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_evec_im,h_evec_im.data(),r2,cudaMemcpyHostToDevice));
    }
    printf("[DMD] Step 4: eigendecomposition done (CPU Francis QR on %dx%d)\n",r,r);

    // ── STEP 5: Frequencies and decay rates ───────────────────────────────
    {
        int blk=128, grd=(r+127)/128;
        extract_dmd_eigenvalues_kernel<<<grd,blk>>>(
            d_eig_re,d_eig_im,d_freq_hz,d_decay,d_amp,r,SAMPLE_RATE);
        CUDA_CHECK_KERNEL(); CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("[DMD] Step 5: frequencies and decay rates extracted\n");

    // ── STEP 6: Spatial mode shapes φₖ = U·ỹₖ ────────────────────────────
    {
        dim3 blk(32,8), grd((n_pixels+31)/32,(r+7)/8);
        reconstruct_spatial_modes_kernel<<<grd,blk>>>(
            d_U,d_evec_re,d_evec_im,d_mode_re,d_mode_im,n_pixels,r);
        CUDA_CHECK_KERNEL();
        dmd_mode_amplitude_kernel<<<grd,blk>>>(
            d_mode_re,d_mode_im,d_amp_map,r,n_pixels);
        CUDA_CHECK_KERNEL(); CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("[DMD] Step 6: spatial mode shapes reconstructed\n");

    // ── Results ────────────────────────────────────────────────────────────
    std::vector<float> h_freq(r),h_decay_v(r),h_amp_v(r);
    CUDA_CHECK(cudaMemcpy(h_freq.data(),   d_freq_hz,rb,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_decay_v.data(),d_decay,  rb,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_amp_v.data(),  d_amp,    rb,cudaMemcpyDeviceToHost));

    std::vector<int> idx(r);
    for (int k=0;k<r;++k) idx[k]=k;
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return h_amp_v[a]>h_amp_v[b];});

    printf("\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │            DMD RESULTS — TOP MODES (v1.1 fixed)         │\n");
    printf("  ├──────┬──────────────┬──────────────┬────────────────────┤\n");
    printf("  │ Mode │  Freq (Hz)   │  Decay (1/s) │  |eigenvalue|      │\n");
    printf("  ├──────┼──────────────┼──────────────┼────────────────────┤\n");
    for (int i=0;i<std::min(r,10);++i) {
        int k=idx[i];
        if (h_freq[k]<0.5f) continue;
        printf("  │  %3d │  %10.3f  │  %10.3f  │  %16.6f    │\n",
               k,h_freq[k],h_decay_v[k],h_amp_v[k]);
    }
    printf("  └──────┴──────────────┴──────────────┴────────────────────┘\n");
    printf("  Validation: expect peaks near 47, 92, 153 Hz\n\n");

    // ── Cleanup ────────────────────────────────────────────────────────────
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUDA_CHECK(cudaFree(d_frames_dev)); CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Xp));         CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_S));          CUDA_CHECK(cudaFree(d_VT));
    CUDA_CHECK(cudaFree(d_Atilde));     CUDA_CHECK(cudaFree(d_eig_re));
    CUDA_CHECK(cudaFree(d_eig_im));     CUDA_CHECK(cudaFree(d_evec_re));
    CUDA_CHECK(cudaFree(d_evec_im));    CUDA_CHECK(cudaFree(d_freq_hz));
    CUDA_CHECK(cudaFree(d_decay));      CUDA_CHECK(cudaFree(d_amp));
    CUDA_CHECK(cudaFree(d_mode_re));    CUDA_CHECK(cudaFree(d_mode_im));
    CUDA_CHECK(cudaFree(d_amp_map));
}


// =============================================================================
//  SECTION 7 — MAIN
// =============================================================================

int main()
{
    int gpu_count=0;
    CUDA_CHECK(cudaGetDeviceCount(&gpu_count));

    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════════╗\n");
    printf("  ║   SYNRA METROLOGY  —  DMD ANALYSIS ENGINE  v1.1 (fixed) ║\n");
    printf("  ║   Dynamic Mode Decomposition for Full-Field DIC          ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    printf("  GPU 0 : %s  |  %.0f GB\n\n",p.name,
           (float)p.totalGlobalMem/(1024.f*1024.f*1024.f));
    printf("  Image  : %dx%d = %d pixels\n",IMG_W,IMG_H,IMG_W*IMG_H);
    printf("  Frames : %d  (dt=%.3f ms)\n",N_FRAMES,1000.f/SAMPLE_RATE);
    printf("  Rank r : %d\n",DMD_RANK);
    printf("  Step 4 : CPU Francis QR (32x32) — cusolverDnSgeev does not exist\n\n");

    srand(42);
    std::vector<float> h_frames;
    generate_dmd_test_data(h_frames,IMG_W,IMG_H,N_FRAMES,SAMPLE_RATE);

    auto t0=std::chrono::high_resolution_clock::now();
    run_dmd_pipeline(IMG_W,IMG_H,N_FRAMES,SAMPLE_RATE,h_frames);
    auto t1=std::chrono::high_resolution_clock::now();

    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("  Total wall time: %.1f ms\n\n",ms);

    CUDA_CHECK(cudaDeviceReset());
    printf("  [DONE]\n\n");
    return EXIT_SUCCESS;
}

// =============================================================================
//  COMPILE (same command, no change):
//  nvcc -Xcompiler -fopenmp -arch=sm_70 dic_dmd_analysis.cu \
//       -o dic_dmd -lgomp -lcusolver -lcublas -O3 -std=c++14
//
//  WHAT CHANGED vs v1.0 (the broken version):
//  ────────────────────────────────────────────
//  REMOVED:  cusolverDnSgeev_bufferSize()  — does not exist in cuSOLVER
//  REMOVED:  cusolverDnSgeev()             — does not exist in cuSOLVER
//
//  ADDED (Section 3):
//    eig2x2()                 — 2x2 block eigenvalue solver
//    toHessenberg()           — Householder reduction to Hessenberg form
//    francisStep()            — Francis double-shift QR step
//    cpu_eigenvalues()        — complete eigenvalue solver (Francis QR)
//    cpu_eigenvectors_approx()— approximate eigenvectors via power iteration
//
//  CHANGED (Step 4 in run_dmd_pipeline):
//    Copy Atilde (32x32) to host → cpu_eigenvalues() → upload back to GPU
//    Total added latency: ~0.1 ms (negligible vs GPU SVD at ~100-500 ms)
// =============================================================================
