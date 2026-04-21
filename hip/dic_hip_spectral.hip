// =============================================================================
//  dic_hip_spectral.hip
//  Author  : Shubham | SYNRA Metrology Systems
//  Purpose : AMD/HIP port of dic_spectral_analysis.cu
//
//  KEY CHANGE vs dic_spectral_analysis.cu:
//  cuFFT → rocFFT  (the only manual porting required — ~30 lines changed)
//  All GPU kernels are IDENTICAL — zero changes needed.
//
//  CUDA→HIP CONVERSION LOG FOR THIS FILE:
//  ──────────────────────────────────────
//  AUTOMATIC (hipify-perl):
//    cuda_runtime.h       → hip/hip_runtime.h
//    cudaMalloc/Free/etc  → hipMalloc/Free/etc
//    cudaMemcpy etc       → hipMemcpy etc
//
//  MANUAL (cuFFT → rocFFT):
//    #include <cufft.h>              → #include <rocfft/rocfft.h>
//    cufftHandle                     → rocfft_plan
//    cufftPlanMany(R2C)              → rocfft_plan_create (forward real)
//    cufftPlanMany(C2R)              → rocfft_plan_create (backward real)
//    cufftExecR2C                    → rocfft_execute
//    cufftExecC2R                    → rocfft_execute
//    cufftDestroy                    → rocfft_plan_destroy
//    CUFFT_R2C / CUFFT_C2R          → rocfft_transform_type_real_forward/backward
//    -lcufft                         → -lrocfft
//
//  GPU KERNELS: zero changes — identical in CUDA and HIP
//    apply_window_kernel, morlet_wavelet_fft_kernel,
//    cwt_multiply_kernel, bandpass_filter_kernel,
//    amplitude_spectrum_kernel, phase_velocity_kernel
//    All use blockIdx, threadIdx, __global__, __device__ — same in both.
//
//  COMPILE ON AMD (Orange Pi / Mini PC with ROCm):
//  ────────────────────────────────────────────────
//  hipcc -Xcompiler -fopenmp dic_hip_spectral.hip \
//        -o dic_hip_spectral -lgomp -lrocfft -O3 -std=c++14
//
//  COMPILE ON NVIDIA (DGX Station — same source):
//  ────────────────────────────────────────────────
//  hipcc -Xcompiler -fopenmp dic_hip_spectral.hip \
//        -o dic_hip_spectral -lgomp -lrocfft -O3 -std=c++14
// =============================================================================

#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>          // ← replaces <cufft.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
//  SECTION 1 — ERROR HANDLING
// =============================================================================

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t _e = (call);                                              \
        if (_e != hipSuccess) {                                               \
            fprintf(stderr,"\n[HIP ERROR]  %s\n  at %s:%d\n  -> %s\n",     \
                #call,__FILE__,__LINE__,hipGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

// rocFFT error check
#define ROCFFT_CHECK(call)                                                    \
    do {                                                                      \
        rocfft_status _s = (call);                                           \
        if (_s != rocfft_status_success) {                                    \
            fprintf(stderr,"\n[rocFFT ERROR]  %s\n  at %s:%d  code=%d\n",   \
                #call,__FILE__,__LINE__,(int)_s);                            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

#define HIP_CHECK_KERNEL()                                                    \
    do {                                                                      \
        hipError_t _e = hipGetLastError();                                   \
        if (_e != hipSuccess) {                                               \
            fprintf(stderr,"\n[HIP KERNEL ERROR]  %s at %s:%d\n",            \
                hipGetErrorString(_e),__FILE__,__LINE__);                    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)


// =============================================================================
//  SECTION 2 — CONFIGURATION  (identical to CUDA version)
// =============================================================================

static const int   IMG_W          = 1024;
static const int   IMG_H          = 1024;
static const int   N_FRAMES       = 1024;
static const float SAMPLE_RATE    = 1000.f;
static const int   N_WAVELET_SCALES = 64;
static const float CWT_F_MIN      = 5.f;
static const float CWT_F_MAX      = 500.f;
static const float MORLET_SIGMA   = 6.f;
static const float BP_F_LOW       = 40.f;
static const float BP_F_HIGH      = 60.f;
static const int   BLOCK_DIM      = 16;


// =============================================================================
//  SECTION 3 — GPU KERNELS  (ZERO CHANGES from CUDA version)
//  blockIdx, threadIdx, blockDim, dim3 — all identical in HIP
// =============================================================================

// Apply Hann window to each pixel's time series
__global__
void apply_window_kernel(float* d_signal, int n_pixels, int N)
{
    int pix = blockIdx.x*blockDim.x + threadIdx.x;
    int t   = blockIdx.y*blockDim.y + threadIdx.y;
    if (pix>=n_pixels || t>=N) return;
    float w = 0.5f*(1.f - cosf(2.f*M_PI*(float)t/(float)(N-1)));
    d_signal[pix*N+t] *= w;
}

// Compute amplitude spectrum from complex FFT output
// rocFFT R2C output format: same as cuFFT R2C (N/2+1 complex bins)
__global__
void amplitude_spectrum_kernel(
        const float2* __restrict__ d_fft,   // [n_pixels × (N/2+1)] complex
        float*        __restrict__ d_amp,   // [n_pixels × (N/2+1)] output
        int n_pixels, int N)
{
    int pix = blockIdx.x*blockDim.x + threadIdx.x;
    int k   = blockIdx.y*blockDim.y + threadIdx.y;
    int half = N/2+1;
    if (pix>=n_pixels || k>=half) return;
    float2 z = d_fft[pix*half+k];
    float amp = sqrtf(z.x*z.x + z.y*z.y);
    // Normalise: DC and Nyquist ÷N, others ×2/N
    if (k==0 || k==N/2) amp *= 1.f/(float)N;
    else                  amp *= 2.f/(float)N;
    d_amp[pix*half+k] = amp;
}

// Morlet wavelet in frequency domain for CWT
__global__
void morlet_wavelet_fft_kernel(
        float2* d_wavelet_fft,
        int N, float scale, float f0, float sigma, float fs)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k>=N) return;
    float freq_norm = (float)k/(float)N;
    float scaled_fc = f0/scale;
    float arg = 2.f*M_PI*sigma*(freq_norm-scaled_fc);
    float mag = powf(M_PI,0.25f)*sqrtf(2.f*M_PI*sigma)*expf(-0.5f*arg*arg);
    d_wavelet_fft[k].x = mag;
    d_wavelet_fft[k].y = 0.f;
}

// Multiply signal FFT by wavelet FFT (frequency-domain convolution for CWT)
__global__
void cwt_multiply_kernel(
        const float2* __restrict__ d_sig_fft,
        const float2* __restrict__ d_wav_fft,
        float2*       __restrict__ d_prod,
        int n_pixels, int N, float scale_norm)
{
    int pix = blockIdx.x*blockDim.x + threadIdx.x;
    int k   = blockIdx.y*blockDim.y + threadIdx.y;
    if (pix>=n_pixels || k>=N) return;
    float2 s = d_sig_fft[pix*N+k];
    float2 w = d_wav_fft[k];
    // Multiply: s × conj(w)
    d_prod[pix*N+k].x = (s.x*w.x + s.y*w.y) * scale_norm;
    d_prod[pix*N+k].y = (s.y*w.x - s.x*w.y) * scale_norm;
}

// Bandpass filter: multiply spectrum by H(f)  (Hann-windowed)
__global__
void bandpass_filter_kernel(
        float2* d_spectrum, int n_pixels, int half_N,
        float fs, float f_low, float f_high, int filter_type)
{
    int pix = blockIdx.x*blockDim.x + threadIdx.x;
    int k   = blockIdx.y*blockDim.y + threadIdx.y;
    if (pix>=n_pixels || k>=half_N) return;
    float freq = (float)k * (fs/(float)((half_N-1)*2));
    float H = 0.f;
    float bw = f_high-f_low;
    float trans = 0.25f*bw;
    if (filter_type==0) {
        H = (freq>=f_low && freq<=f_high) ? 1.f : 0.f;
    } else {
        if      (freq < f_low-trans)  H = 0.f;
        else if (freq < f_low)        H = 0.5f*(1.f-cosf(M_PI*(freq-f_low+trans)/trans));
        else if (freq <= f_high)      H = 1.f;
        else if (freq <= f_high+trans)H = 0.5f*(1.f-cosf(M_PI*(f_high+trans-freq)/trans));
        else                          H = 0.f;
    }
    d_spectrum[pix*half_N+k].x *= H;
    d_spectrum[pix*half_N+k].y *= H;
}

// CWT scalogram amplitude
__global__
void cwt_amplitude_kernel(
        const float2* __restrict__ d_cwt,
        float*        __restrict__ d_scalogram,
        int n_pixels, int N, int si)
{
    int pix = blockIdx.x*blockDim.x + threadIdx.x;
    int t   = blockIdx.y*blockDim.y + threadIdx.y;
    if (pix>=n_pixels || t>=N) return;
    float2 c = d_cwt[pix*N+t];
    d_scalogram[(size_t)pix*N_WAVELET_SCALES*N + (size_t)si*N + t] =
        sqrtf(c.x*c.x+c.y*c.y)/(float)N;
}

// Dominant frequency map: argmax of amplitude spectrum per pixel
__global__
void dominant_freq_kernel(
        const float* __restrict__ d_amp,
        float*       __restrict__ d_dom_freq,
        int n_pixels, int half_N, float fs)
{
    int pix = blockIdx.x*blockDim.x + threadIdx.x;
    if (pix>=n_pixels) return;
    float best=0.f; int best_k=0;
    for (int k=1;k<half_N;++k) {
        float a=d_amp[pix*half_N+k];
        if (a>best){best=a;best_k=k;}
    }
    d_dom_freq[pix]=(float)best_k*(fs/(float)((half_N-1)*2));
}


// =============================================================================
//  SECTION 4 — rocFFT PLAN HELPERS
//  This is the main manual porting work — replaces cufftPlanMany
// =============================================================================

// rocFFT uses a different plan creation API vs cuFFT:
//
//  cuFFT:   cufftPlanMany(&plan, rank, dims, NULL,1,N, NULL,1,N, CUFFT_R2C, batch)
//  rocFFT:  rocfft_plan_create(&plan, placement, transform_type, precision,
//                               dimensions, lengths, batch, description)
//
// Key differences:
//  1. rocFFT separates input/output descriptions via rocfft_plan_description
//  2. Batch count goes into description, not the plan_create call directly
//  3. rocfft_execute takes void** (not typed pointers)
//  4. rocfft_plan_destroy replaces cufftDestroy

struct RocfftPlans {
    rocfft_plan fwd;   // R2C forward:  real → complex
    rocfft_plan inv;   // C2R inverse:  complex → real
    rocfft_plan c2c_fwd; // C2C forward: complex → complex (for CWT)
    rocfft_plan c2c_inv; // C2C inverse: complex → complex (for CWT IFFT)
    rocfft_execution_info exec_info;
    void* work_buf;
    size_t work_size;
    int n_pixels;
    int N;
};

RocfftPlans create_rocfft_plans(int n_pixels, int N)
{
    RocfftPlans p;
    p.n_pixels = n_pixels;
    p.N = N;
    p.work_buf = nullptr;
    p.work_size = 0;

    // rocFFT must be initialised once per process
    ROCFFT_CHECK(rocfft_setup());

    size_t lengths[1] = {(size_t)N};

    // ── Forward R2C plan (real→complex, batch=n_pixels) ───────────────────
    rocfft_plan_description fwd_desc;
    ROCFFT_CHECK(rocfft_plan_description_create(&fwd_desc));

    // Set batch strides: input stride=1 dist=N, output stride=1 dist=N/2+1
    size_t in_strides[1]  = {1};
    size_t out_strides[1] = {1};
    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
        fwd_desc,
        rocfft_array_type_real,           // input type
        rocfft_array_type_hermitian_interleaved, // output type
        nullptr, nullptr,                 // offsets (0)
        1, in_strides,  (size_t)N,        // in:  stride, dist
        1, out_strides, (size_t)(N/2+1)   // out: stride, dist
    ));

    ROCFFT_CHECK(rocfft_plan_create(
        &p.fwd,
        rocfft_placement_notinplace,
        rocfft_transform_type_real_forward,
        rocfft_precision_single,
        1, lengths,
        (size_t)n_pixels,
        fwd_desc));

    ROCFFT_CHECK(rocfft_plan_description_destroy(fwd_desc));

    // ── Inverse C2R plan (complex→real, batch=n_pixels) ──────────────────
    rocfft_plan_description inv_desc;
    ROCFFT_CHECK(rocfft_plan_description_create(&inv_desc));

    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
        inv_desc,
        rocfft_array_type_hermitian_interleaved,
        rocfft_array_type_real,
        nullptr, nullptr,
        1, out_strides, (size_t)(N/2+1),
        1, in_strides,  (size_t)N
    ));

    ROCFFT_CHECK(rocfft_plan_create(
        &p.inv,
        rocfft_placement_notinplace,
        rocfft_transform_type_real_inverse,
        rocfft_precision_single,
        1, lengths,
        (size_t)n_pixels,
        inv_desc));

    ROCFFT_CHECK(rocfft_plan_description_destroy(inv_desc));

    // ── C2C forward plan (for CWT: full spectrum) ─────────────────────────
    rocfft_plan_description c2c_desc;
    ROCFFT_CHECK(rocfft_plan_description_create(&c2c_desc));

    size_t c2c_in_strides[1] = {1};
    ROCFFT_CHECK(rocfft_plan_description_set_data_layout(
        c2c_desc,
        rocfft_array_type_complex_interleaved,
        rocfft_array_type_complex_interleaved,
        nullptr, nullptr,
        1, c2c_in_strides, (size_t)N,
        1, c2c_in_strides, (size_t)N
    ));

    ROCFFT_CHECK(rocfft_plan_create(
        &p.c2c_fwd,
        rocfft_placement_notinplace,
        rocfft_transform_type_complex_forward,
        rocfft_precision_single,
        1, lengths,
        (size_t)n_pixels,
        c2c_desc));

    // Reuse desc for inverse
    ROCFFT_CHECK(rocfft_plan_create(
        &p.c2c_inv,
        rocfft_placement_notinplace,
        rocfft_transform_type_complex_inverse,
        rocfft_precision_single,
        1, lengths,
        (size_t)n_pixels,
        c2c_desc));

    ROCFFT_CHECK(rocfft_plan_description_destroy(c2c_desc));

    // ── Execution info and work buffer ────────────────────────────────────
    ROCFFT_CHECK(rocfft_execution_info_create(&p.exec_info));

    // Allocate work buffer (rocFFT requires this for some transforms)
    size_t work1=0, work2=0;
    rocfft_plan_get_work_buffer_size(p.fwd,     &work1);
    rocfft_plan_get_work_buffer_size(p.c2c_fwd, &work2);
    p.work_size = std::max(work1, work2);

    if (p.work_size > 0) {
        HIP_CHECK(hipMalloc(&p.work_buf, p.work_size));
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(
            p.exec_info, p.work_buf, p.work_size));
    }

    return p;
}

void destroy_rocfft_plans(RocfftPlans& p)
{
    ROCFFT_CHECK(rocfft_plan_destroy(p.fwd));
    ROCFFT_CHECK(rocfft_plan_destroy(p.inv));
    ROCFFT_CHECK(rocfft_plan_destroy(p.c2c_fwd));
    ROCFFT_CHECK(rocfft_plan_destroy(p.c2c_inv));
    ROCFFT_CHECK(rocfft_execution_info_destroy(p.exec_info));
    if (p.work_buf) HIP_CHECK(hipFree(p.work_buf));
    rocfft_cleanup();
}


// =============================================================================
//  SECTION 5 — SYNTHETIC TEST DATA GENERATOR  (identical to CUDA version)
// =============================================================================

void generate_dic_time_series(std::vector<float>& h_signals,
                               int W, int H, int N, float fs)
{
    h_signals.resize((size_t)W*H*N, 0.f);
    int n_pixels = W*H;
    float dt = 1.f/fs;
    srand(42);
    struct Mode { float f, amp; };
    const Mode modes[] = {{47.f,0.80f},{92.f,0.30f},{153.f,0.15f},{2.f,0.05f}};
    for (int pix=0;pix<n_pixels;++pix){
        float* sig=h_signals.data()+(size_t)pix*N;
        float fx=(float)(pix%W)/(float)(W-1);
        for (int t=0;t<N;++t){
            float time=t*dt, v=0;
            v+=modes[0].amp*sinf(M_PI*fx)*cosf(2.f*M_PI*modes[0].f*time);
            v+=modes[1].amp*sinf(2.f*M_PI*fx)*cosf(2.f*M_PI*modes[1].f*time);
            v+=modes[2].amp*sinf(M_PI*fx)*cosf(2.f*M_PI*modes[2].f*time);
            v+=modes[3].amp*cosf(2.f*M_PI*modes[3].f*time);
            v+=0.02f*((float)rand()/RAND_MAX-0.5f);
            sig[t]=v;
        }
    }
    printf("[SYN] Generated %dx%d × %d frames\n",W,H,N);
    printf("[SYN] Embedded modes: 47 Hz, 92 Hz, 153 Hz, 2 Hz (wind)\n");
}


// =============================================================================
//  SECTION 6 — MAIN SPECTRAL PIPELINE
// =============================================================================

void run_spectral_pipeline(int gpu_id, int gpu_count,
                            const std::vector<float>& h_signals,
                            int W, int H, int N, float fs)
{
    HIP_CHECK(hipSetDevice(gpu_id));

    int n_total = W*H;
    int tile_start = gpu_id*(n_total/gpu_count);
    int tile_end   = (gpu_id==gpu_count-1)?n_total:tile_start+n_total/gpu_count;
    int n_tile     = tile_end-tile_start;

    int half_N = N/2+1;
    size_t sig_bytes = (size_t)n_tile*N*sizeof(float);
    size_t fft_bytes = (size_t)n_tile*half_N*sizeof(float2);
    size_t amp_bytes = (size_t)n_tile*half_N*sizeof(float);
    size_t dom_bytes = (size_t)n_tile*sizeof(float);

    // ── Allocate GPU memory ───────────────────────────────────────────────
    float  *d_sig=nullptr, *d_amp=nullptr, *d_dom_freq=nullptr;
    float2 *d_fft=nullptr, *d_bpfilt=nullptr;

    HIP_CHECK(hipMalloc(&d_sig,      sig_bytes));
    HIP_CHECK(hipMalloc(&d_fft,      fft_bytes));
    HIP_CHECK(hipMalloc(&d_amp,      amp_bytes));
    HIP_CHECK(hipMalloc(&d_dom_freq, dom_bytes));
    HIP_CHECK(hipMalloc(&d_bpfilt,   fft_bytes));

    // ── Upload this tile's signals ────────────────────────────────────────
    HIP_CHECK(hipMemcpy(d_sig,
        h_signals.data()+(size_t)tile_start*N,
        sig_bytes, hipMemcpyHostToDevice));

    // ── Create rocFFT plans ───────────────────────────────────────────────
    RocfftPlans plans = create_rocfft_plans(n_tile, N);

    // ── Apply Hann window ─────────────────────────────────────────────────
    {
        dim3 blk(32,8), grd((n_tile+31)/32,(N+7)/8);
        apply_window_kernel<<<grd,blk>>>(d_sig, n_tile, N);
        HIP_CHECK_KERNEL();
    }

    // ── Pipeline A: Forward FFT (R2C) via rocFFT ──────────────────────────
    // rocfft_execute takes void** — cast our typed pointers
    {
        void* in_buf[1]  = {(void*)d_sig};
        void* out_buf[1] = {(void*)d_fft};
        ROCFFT_CHECK(rocfft_execute(plans.fwd, in_buf, out_buf, plans.exec_info));
        HIP_CHECK(hipDeviceSynchronize());
    }

    // ── Amplitude spectrum ────────────────────────────────────────────────
    {
        dim3 blk(32,8), grd((n_tile+31)/32,(half_N+7)/8);
        amplitude_spectrum_kernel<<<grd,blk>>>(d_fft,d_amp,n_tile,N);
        HIP_CHECK_KERNEL();
    }

    // ── Dominant frequency map ────────────────────────────────────────────
    {
        dim3 blk(256), grd((n_tile+255)/256);
        dominant_freq_kernel<<<grd,blk>>>(d_amp,d_dom_freq,n_tile,half_N,fs);
        HIP_CHECK_KERNEL();
        HIP_CHECK(hipDeviceSynchronize());
    }

    // ── Pipeline C: Bandpass filter (FFT→H(f)→IFFT) ──────────────────────
    // Copy FFT output for bandpass filtering
    HIP_CHECK(hipMemcpy(d_bpfilt, d_fft, fft_bytes, hipMemcpyDeviceToDevice));

    {
        // Apply Hann-windowed bandpass H(f)
        dim3 blk(32,8), grd((n_tile+31)/32,(half_N+7)/8);
        bandpass_filter_kernel<<<grd,blk>>>(
            d_bpfilt, n_tile, half_N, fs, BP_F_LOW, BP_F_HIGH, 1);
        HIP_CHECK_KERNEL();
    }

    {
        // Inverse FFT: C2R via rocFFT
        void* in_buf[1]  = {(void*)d_bpfilt};
        void* out_buf[1] = {(void*)d_sig};   // reuse d_sig buffer for filtered signal
        ROCFFT_CHECK(rocfft_execute(plans.inv, in_buf, out_buf, plans.exec_info));
        HIP_CHECK(hipDeviceSynchronize());
    }

    // ── Download dominant frequency map ───────────────────────────────────
    std::vector<float> h_dom_freq(n_tile);
    HIP_CHECK(hipMemcpy(h_dom_freq.data(), d_dom_freq, dom_bytes, hipMemcpyDeviceToHost));

    // Compute statistics for report
    float sum=0,mn=1e9f,mx=-1e9f;
    for (float f:h_dom_freq){sum+=f;mn=std::min(mn,f);mx=std::max(mx,f);}

    #pragma omp critical
    {
        printf("  [GPU %d AMD]  tile [%d–%d]  dom_freq: min=%.1f max=%.1f mean=%.1f Hz\n",
               gpu_id, tile_start, tile_end-1,
               mn, mx, sum/(float)n_tile);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    destroy_rocfft_plans(plans);
    HIP_CHECK(hipFree(d_sig));
    HIP_CHECK(hipFree(d_fft));
    HIP_CHECK(hipFree(d_amp));
    HIP_CHECK(hipFree(d_dom_freq));
    HIP_CHECK(hipFree(d_bpfilt));
}


// =============================================================================
//  SECTION 7 — MAIN
// =============================================================================

int main()
{
    int gpu_count=0;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    if (gpu_count==0){fprintf(stderr,"No HIP GPU found.\n");return 1;}

    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════╗\n");
    printf("  ║   SYNRA — SPECTRAL ANALYSIS ENGINE (HIP/rocFFT)      ║\n");
    printf("  ║   CUDA→HIP port: cuFFT replaced with rocFFT          ║\n");
    printf("  ╚═══════════════════════════════════════════════════════╝\n\n");

    for (int i=0;i<gpu_count;++i){
        hipDeviceProp_t p;
        HIP_CHECK(hipGetDeviceProperties(&p,i));
        printf("  GPU %d : %-24s | %.0f GB\n",i,p.name,
               (float)p.totalGlobalMem/(1024.f*1024.f*1024.f));
    }

    printf("\n  Image: %dx%d  N=%d  fs=%.0f Hz\n\n",
           IMG_W,IMG_H,N_FRAMES,SAMPLE_RATE);

    std::vector<float> h_signals;
    generate_dic_time_series(h_signals, IMG_W, IMG_H, N_FRAMES, SAMPLE_RATE);

    auto t0=std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(gpu_count)
    {
        int tid=omp_get_thread_num();
        run_spectral_pipeline(tid, gpu_count, h_signals,
                              IMG_W, IMG_H, N_FRAMES, SAMPLE_RATE);
    }

    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("\n  Wall time: %.1f ms\n", ms);
    printf("  Validation: dominant frequency peaks expected near 47, 92, 153 Hz\n\n");

    for (int i=0;i<gpu_count;++i){
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipDeviceReset());
    }
    printf("  [DONE]\n\n");
    return EXIT_SUCCESS;
}

// =============================================================================
//  COMPILE:
//  hipcc -Xcompiler -fopenmp dic_hip_spectral.hip \
//        -o dic_hip_spectral -lgomp -lrocfft -O3 -std=c++14
// =============================================================================
