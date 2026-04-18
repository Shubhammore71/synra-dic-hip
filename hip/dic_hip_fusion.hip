// =============================================================================
//  dic_hip_fusion.hip
//  Author  : Shubham | SYNRA Metrology Systems
//  Purpose : AMD/HIP port of hyper_dic_fusion_final.cu
//
//  HIPIFY CONVERSION LOG:
//  ───────────────────────
//  This file was created by running hipify-perl on hyper_dic_fusion_final.cu
//  and then manually reviewing every change.
//
//  AUTOMATIC CONVERSIONS (hipify-perl handled these):
//  ┌─────────────────────────────────┬────────────────────────────────┐
//  │ CUDA original                   │ HIP replacement                │
//  ├─────────────────────────────────┼────────────────────────────────┤
//  │ #include <cuda_runtime.h>       │ #include <hip/hip_runtime.h>  │
//  │ cudaMalloc                      │ hipMalloc                      │
//  │ cudaFree                        │ hipFree                        │
//  │ cudaMemcpy                      │ hipMemcpy                      │
//  │ cudaMemset                      │ hipMemset                      │
//  │ cudaDeviceSynchronize           │ hipDeviceSynchronize           │
//  │ cudaSetDevice                   │ hipSetDevice                   │
//  │ cudaGetDeviceCount              │ hipGetDeviceCount              │
//  │ cudaGetDeviceProperties         │ hipGetDeviceProperties         │
//  │ cudaGetLastError                │ hipGetLastError                │
//  │ cudaGetErrorString              │ hipGetErrorString              │
//  │ cudaMemcpyHostToDevice          │ hipMemcpyHostToDevice          │
//  │ cudaMemcpyDeviceToHost          │ hipMemcpyDeviceToHost          │
//  │ cudaMemcpyDeviceToDevice        │ hipMemcpyDeviceToDevice        │
//  │ cudaError_t                     │ hipError_t                     │
//  │ cudaSuccess                     │ hipSuccess                     │
//  │ cudaDeviceProp                  │ hipDeviceProp_t                │
//  │ cudaDeviceReset                 │ hipDeviceReset                 │
//  └─────────────────────────────────┴────────────────────────────────┘
//
//  GPU KERNELS: __global__, blockIdx, threadIdx, blockDim, dim3,
//  __device__, __forceinline__, __restrict__, atomicAdd, fmaxf, fminf,
//  sqrtf, sinf, cosf, expf — ALL IDENTICAL in HIP. Zero changes.
//
//  MANUAL CHANGES (hipify-perl does NOT handle these):
//  ────────────────────────────────────────────────────
//  1. Kernel launch syntax:
//     CUDA: kernel<<<grid, block>>>(args)
//     HIP:  hipLaunchKernelGGL(kernel, grid, block, 0, 0, args)
//     OR:   kernel<<<grid, block>>>(args)  [also valid in HIP]
//     → We keep <<<>>> syntax — it works in both CUDA and HIP.
//
//  2. Error macro renamed CUDA_CHECK → HIP_CHECK throughout.
//
//  3. File extension: .cu → .hip (hipcc accepts both, .hip is cleaner)
//
//  COMPILE ON AMD GPU:
//  ────────────────────
//  hipcc -Xcompiler -fopenmp dic_hip_fusion.hip \
//        -o dic_hip_fusion -lgomp -O3 -std=c++14
//
//  COMPILE ON NVIDIA GPU (same file, zero changes):
//  ──────────────────────────────────────────────────
//  hipcc -Xcompiler -fopenmp dic_hip_fusion.hip \
//        -o dic_hip_fusion -lgomp -O3 -std=c++14
//  (hipcc on NVIDIA routes through nvcc automatically)
//
//  HARDWARE TESTED:
//  ─────────────────
//  Target: AMD GPU on Orange Pi / Mini PC (ROCm 6.x)
//  Also compiles on: NVIDIA V100 DGX Station (via hipcc CUDA backend)
// =============================================================================

// ── THE KEY DIFFERENCE: hip/hip_runtime.h instead of cuda_runtime.h ─────────
#include <hip/hip_runtime.h>
// Everything else is identical:
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// =============================================================================
//  SECTION 1 — ERROR HANDLING (HIP version)
//  Identical logic to CUDA version — only type names changed.
// =============================================================================

// hipError_t instead of cudaError_t
// hipSuccess instead of cudaSuccess
// hipGetErrorString instead of cudaGetErrorString
#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t _e = (call);                                              \
        if (_e != hipSuccess) {                                               \
            fprintf(stderr,"\n[HIP ERROR]  %s\n  at %s:%d\n  -> %s\n",     \
                #call,__FILE__,__LINE__,hipGetErrorString(_e));              \
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
//  SECTION 2 — CONFIGURATION (identical to CUDA version)
// =============================================================================

static const int   IMG_W      = 1024;
static const int   IMG_H      = 1024;
static const int   SUBSET_R   = 15;
static const int   SEARCH_R   = 8;
static const float SUBPX_STEP = 0.1f;
static const int   BLOCK_DIM  = 16;


// =============================================================================
//  SECTION 3 — BILINEAR INTERPOLATION
//  ZERO CHANGES from CUDA version.
//  __device__, __forceinline__, __restrict__ are identical in HIP and CUDA.
// =============================================================================

__device__ __forceinline__
float bilinear_interp(const float* __restrict__ img,
                      int W, int H, float x, float y)
{
    x = fmaxf(0.f, fminf(x, (float)(W-1) - 1e-4f));
    y = fmaxf(0.f, fminf(y, (float)(H-1) - 1e-4f));
    int ix = (int)x; int iy = (int)y;
    float dx = x - ix;  float dy = y - iy;
    float v00 = img[ iy   *W+  ix  ];
    float v10 = img[ iy   *W+(ix+1)];
    float v01 = img[(iy+1)*W+  ix  ];
    float v11 = img[(iy+1)*W+(ix+1)];
    float top    = (1.f-dx)*v00 + dx*v10;
    float bottom = (1.f-dx)*v01 + dx*v11;
    return (1.f-dy)*top + dy*bottom;
}


// =============================================================================
//  SECTION 4 — ZNCC SUBSET CORRELATION
//  ZERO CHANGES from CUDA version.
//  All math intrinsics (sqrtf, fmaxf, etc.) are identical.
// =============================================================================

__device__
float zncc_subset(const float* __restrict__ d_ref,
                  const float* __restrict__ d_def,
                  int W, int H,
                  int cx, int cy,
                  float u, float v,
                  int R)
{
    float sum_f=0,sum_g=0,sum_f2=0,sum_g2=0,sum_fg=0;
    int n=0;
    for (int dy=-R; dy<=R; ++dy) {
        int ry=cy+dy; if (ry<0||ry>=H) continue;
        for (int dx=-R; dx<=R; ++dx) {
            int rx=cx+dx; if (rx<0||rx>=W) continue;
            float f = d_ref[ry*W+rx];
            float g = bilinear_interp(d_def,W,H,(float)rx+u,(float)ry+v);
            sum_f+=f; sum_g+=g; sum_f2+=f*f; sum_g2+=g*g; sum_fg+=f*g; ++n;
        }
    }
    if (n==0) return 0.f;
    float fn=(float)n;
    float num = sum_fg-(sum_f*sum_g)/fn;
    float den = sqrtf((sum_f2-sum_f*sum_f/fn)*(sum_g2-sum_g*sum_g/fn));
    return (den<1e-10f)?0.f:num/den;
}


// =============================================================================
//  SECTION 5 — MAIN DIC KERNEL
//  ZERO CHANGES from CUDA version.
//  blockIdx, threadIdx, blockDim are identical in HIP and CUDA.
//  The <<<grid, block>>> launch syntax works in both.
// =============================================================================

__global__
void hyper_dic_fusion_kernel(
        const float* __restrict__ d_ref,
        const float* __restrict__ d_def,
        float*       __restrict__ d_res,
        int W, int H,
        int row_offset, int tile_H,
        float wind_u, float wind_v)
{
    int cx = blockIdx.x*blockDim.x + threadIdx.x;
    int cy = blockIdx.y*blockDim.y + threadIdx.y + row_offset;
    if (cx>=W || cy>=H || cy>=(row_offset+tile_H)) return;

    // Step A: coarse integer search
    float best_zncc=-2.f, best_u=0.f, best_v=0.f;
    for (int iv=-SEARCH_R; iv<=SEARCH_R; ++iv) {
        for (int iu=-SEARCH_R; iu<=SEARCH_R; ++iu) {
            float score=zncc_subset(d_ref,d_def,W,H,cx,cy,(float)iu,(float)iv,SUBSET_R);
            if (score>best_zncc){best_zncc=score;best_u=(float)iu;best_v=(float)iv;}
        }
    }

    // Step B: sub-pixel refinement
    float refined_u=best_u, refined_v=best_v;
    for (float dv=-1.f;dv<=1.f;dv+=SUBPX_STEP) {
        for (float du=-1.f;du<=1.f;du+=SUBPX_STEP) {
            float score=zncc_subset(d_ref,d_def,W,H,cx,cy,best_u+du,best_v+dv,SUBSET_R);
            if (score>best_zncc){best_zncc=score;refined_u=best_u+du;refined_v=best_v+dv;}
        }
    }

    // Step C: wind correction
    const float alpha=0.05f;
    float true_u = refined_u-(alpha*wind_u);
    float true_v = refined_v-(alpha*wind_v);

    // Step D: store displacement magnitude
    d_res[cy*W+cx] = sqrtf(true_u*true_u+true_v*true_v);
}


// =============================================================================
//  SECTION 6 — SYNTHETIC SPECKLE IMAGE GENERATOR (identical to CUDA version)
// =============================================================================

void generate_synthetic_images(std::vector<float>& h_ref,
                                std::vector<float>& h_def,
                                int W, int H)
{
    h_ref.assign(W*H,0.f); h_def.assign(W*H,0.f);
    const int N_SP=200; const float SIGMA=8.f;
    const float AMP_U=3.f,AMP_V=1.5f,LAMBDA=256.f;
    srand(42);
    std::vector<float> sx(N_SP),sy(N_SP),sa(N_SP);
    for (int k=0;k<N_SP;++k){sx[k]=(float)(rand()%W);sy[k]=(float)(rand()%H);sa[k] = 0.5f + 0.5f * ((float)rand() / (float)RAND_MAX);}
    for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
        float rv=0;
        for (int k=0;k<N_SP;++k){float dx=x-sx[k],dy=y-sy[k];rv+=sa[k]*expf(-(dx*dx+dy*dy)/(2*SIGMA*SIGMA));}
        h_ref[y*W+x]=fminf(rv,1.f);
        float u=AMP_U*sinf(2.f*M_PI*(float)x/LAMBDA);
        float v=AMP_V*cosf(2.f*M_PI*(float)y/LAMBDA);
        float dv=0;
        for (int k=0;k<N_SP;++k){float dx=(x-u)-sx[k],dy=(y-v)-sy[k];dv+=sa[k]*expf(-(dx*dx+dy*dy)/(2*SIGMA*SIGMA));}
        h_def[y*W+x]=fminf(dv,1.f);
    }
    printf("[SYN]  Speckle images ready  %dx%d  %d speckles\n",W,H,N_SP);
    printf("[SYN]  Expected max disp: %.3f px\n",sqrtf(AMP_U*AMP_U+AMP_V*AMP_V));
}


// =============================================================================
//  SECTION 7 — WIND DATA LOADER (identical to CUDA version)
// =============================================================================

struct WindVector { float u, v; };

WindVector load_wind_data(int /*gpu_id*/) {
    return { 0.31f, -0.17f };  // TODO: replace with real sensor read
}


// =============================================================================
//  SECTION 8 — RESULT ANALYSIS (identical to CUDA version)
// =============================================================================

void analyse_results(const std::vector<float>& h_res, int W, int H)
{
    float mn=1e30f,mx=-1e30f,sm=0,sm2=0;
    for (float v:h_res){mn=std::min(mn,v);mx=std::max(mx,v);sm+=v;sm2+=v*v;}
    long n=(long)(W*H);
    printf("\n  ┌──────────────────────────────────────────┐\n");
    printf("  │  DISPLACEMENT FIELD (HIP/AMD)            │\n");
    printf("  │  Min : %8.4f px                      │\n",mn);
    printf("  │  Max : %8.4f px                      │\n",mx);
    printf("  │  Mean: %8.4f px                      │\n",sm/(float)n);
    printf("  │  RMS : %8.4f px                      │\n",sqrtf(sm2/(float)n));
    printf("  └──────────────────────────────────────────┘\n");
}


// =============================================================================
//  SECTION 9 — PER-GPU WORKER
//  KEY DIFFERENCES from CUDA version:
//  1. hipSetDevice instead of cudaSetDevice
//  2. hipMalloc/hipFree instead of cudaMalloc/cudaFree
//  3. hipMemcpy instead of cudaMemcpy
//  4. hipMemset instead of cudaMemset
//  5. hipDeviceSynchronize instead of cudaDeviceSynchronize
//  Everything else (kernel launch, dim3, omp critical) is identical.
// =============================================================================

void gpu_worker(int gpu_id, int gpu_count,
                const std::vector<float>& h_ref,
                const std::vector<float>& h_def,
                std::vector<float>&       h_res,
                int W, int H)
{
    // hipSetDevice — same role as cudaSetDevice
    HIP_CHECK(hipSetDevice(gpu_id));

    int tile_H    = (H+gpu_count-1)/gpu_count;
    int row_start = gpu_id*tile_H;
    int row_end   = std::min(row_start+tile_H,H);
    int actual_H  = row_end-row_start;
    if (actual_H<=0) return;

    size_t full_bytes=(size_t)W*H*sizeof(float);
    size_t tile_bytes=(size_t)W*actual_H*sizeof(float);

    WindVector wind=load_wind_data(gpu_id);

    // hipMalloc — same role as cudaMalloc
    float *d_ref=nullptr, *d_def=nullptr, *d_res=nullptr;
    HIP_CHECK(hipMalloc(&d_ref,full_bytes));
    HIP_CHECK(hipMalloc(&d_def,full_bytes));
    HIP_CHECK(hipMalloc(&d_res,full_bytes));

    // hipMemset — same role as cudaMemset
    HIP_CHECK(hipMemset(d_res,0,full_bytes));

    // hipMemcpy H2D — same role as cudaMemcpy HostToDevice
    HIP_CHECK(hipMemcpy(d_ref,h_ref.data(),full_bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_def,h_def.data(),full_bytes,hipMemcpyHostToDevice));

    // Kernel launch — <<<>>> syntax is identical in HIP
    dim3 block(BLOCK_DIM,BLOCK_DIM);
    dim3 grid((W+BLOCK_DIM-1)/BLOCK_DIM,(actual_H+BLOCK_DIM-1)/BLOCK_DIM);

    hyper_dic_fusion_kernel<<<grid,block>>>(
        d_ref,d_def,d_res,W,H,row_start,actual_H,wind.u,wind.v);

    HIP_CHECK_KERNEL();

    // hipDeviceSynchronize — same role as cudaDeviceSynchronize
    HIP_CHECK(hipDeviceSynchronize());

    // hipMemcpy D2H — same role as cudaMemcpy DeviceToHost
    size_t offset=(size_t)row_start*W;
    HIP_CHECK(hipMemcpy(h_res.data()+offset,d_res+offset,
                        tile_bytes,hipMemcpyDeviceToHost));

    // hipFree — same role as cudaFree
    HIP_CHECK(hipFree(d_ref));
    HIP_CHECK(hipFree(d_def));
    HIP_CHECK(hipFree(d_res));

    #pragma omp critical
    {
        printf("  [GPU %d AMD]  rows %4d–%4d  wind=(%.3f,%.3f)  %6.1fK vectors\n",
               gpu_id,row_start,row_end-1,wind.u,wind.v,(float)(W*actual_H)/1000.f);
    }
}


// =============================================================================
//  SECTION 10 — MAIN
//  hipGetDeviceCount / hipGetDeviceProperties instead of CUDA equivalents.
//  hipDeviceReset instead of cudaDeviceReset.
// =============================================================================

int main()
{
    int gpu_count=0;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    if (gpu_count==0){fprintf(stderr,"[ERROR] No HIP/AMD GPUs found.\n");return 1;}

    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════╗\n");
    printf("  ║   SYNRA — DIC ENGINE (HIP/AMD)  v1.0                ║\n");
    printf("  ║   CUDA→HIP port by Shubham More                     ║\n");
    printf("  ╚═══════════════════════════════════════════════════════╝\n\n");

    for (int i=0;i<gpu_count;++i) {
        // hipDeviceProp_t instead of cudaDeviceProp
        hipDeviceProp_t p;
        HIP_CHECK(hipGetDeviceProperties(&p,i));
        printf("  GPU %d : %-24s | %.0f GB | %d CUs\n",
               i,p.name,(float)p.totalGlobalMem/(1024.f*1024.f*1024.f),
               p.multiProcessorCount);
    }

    printf("\n  Image : %d x %d px\n  Subset: %dx%d (R=%d)\n\n",
           IMG_W,IMG_H,2*SUBSET_R+1,2*SUBSET_R+1,SUBSET_R);

    std::vector<float> h_ref,h_def;
    generate_synthetic_images(h_ref,h_def,IMG_W,IMG_H);
    std::vector<float> h_res(IMG_W*IMG_H,0.f);

    printf("\n  Dispatching %d GPU workers (HIP)...\n\n",gpu_count);
    auto t0=std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(gpu_count)
    {
        int tid=omp_get_thread_num();
        gpu_worker(tid,gpu_count,h_ref,h_def,h_res,IMG_W,IMG_H);
    }

    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("\n  Wall time: %.1f ms  (%.2f M vectors/sec)\n",
           ms,(float)(IMG_W*IMG_H)/(ms*1000.f));

    analyse_results(h_res,IMG_W,IMG_H);

    for (int i=0;i<gpu_count;++i){
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipDeviceReset());  // hipDeviceReset instead of cudaDeviceReset
    }
    printf("\n  [DONE]\n\n");
    return EXIT_SUCCESS;
}

// =============================================================================
//  COMPILE ON AMD (Orange Pi / Mini PC with ROCm):
//  hipcc -Xcompiler -fopenmp dic_hip_fusion.hip \
//        -o dic_hip_fusion -lgomp -O3 -std=c++14
//
//  COMPILE ON NVIDIA (DGX Station — same source file):
//  hipcc -Xcompiler -fopenmp dic_hip_fusion.hip \
//        -o dic_hip_fusion -lgomp -O3 -std=c++14
//  (hipcc on NVIDIA detects CUDA and uses nvcc backend automatically)
//
//  WHAT CHANGED vs hyper_dic_fusion_final.cu:
//  ───────────────────────────────────────────
//  1 line:  #include <hip/hip_runtime.h>   (was cuda_runtime.h)
//  Macro:   HIP_CHECK / HIP_CHECK_KERNEL   (was CUDA_CHECK)
//  API:     hip* prefix everywhere          (was cuda* prefix)
//  Types:   hipError_t, hipDeviceProp_t     (was cudaError_t, cudaDeviceProp)
//  GPU kernels: ZERO CHANGES — identical in HIP and CUDA
// =============================================================================
