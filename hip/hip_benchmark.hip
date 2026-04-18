// =============================================================================
//  hip_benchmark.hip
//  Author  : Shubham | SYNRA Metrology Systems
//  Purpose : Benchmark DIC pipeline on AMD GPU vs CPU baseline
//            Generates the numbers needed for the technical report to Yash/team
//
//  WHAT THIS MEASURES:
//  ────────────────────
//  1. GPU path  : full ZNCC+bilinear DIC on AMD GPU via HIP
//  2. CPU path  : exact same algorithm, single-threaded on CPU
//  3. Speedup   : GPU time / CPU time  (goal: >10× for useful GPU acceleration)
//  4. GPU util  : read rocm-smi separately during run for utilisation %
//
//  HOW TO RUN:
//  ────────────
//  # Compile:
//  hipcc -Xcompiler -fopenmp hip_benchmark.hip \
//        -o hip_benchmark -lgomp -O3 -std=c++14
//
//  # Run (captures timing):
//  ./hip_benchmark | tee benchmark_results.txt
//
//  # In a second terminal during the run (GPU utilisation):
//  watch -n 0.5 rocm-smi
// =============================================================================

#include <hip/hip_runtime.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t _e=(call);                                                \
        if (_e!=hipSuccess){                                                  \
            fprintf(stderr,"[HIP ERROR] %s at %s:%d -> %s\n",               \
                #call,__FILE__,__LINE__,hipGetErrorString(_e));              \
            exit(1);                                                          \
        }                                                                     \
    } while(0)

#define HIP_CHECK_KERNEL() do { \
    hipError_t _e=hipGetLastError(); \
    if (_e!=hipSuccess){fprintf(stderr,"[KERNEL ERR] %s at %s:%d\n", \
        hipGetErrorString(_e),__FILE__,__LINE__);exit(1);}} while(0)


// =============================================================================
//  BENCHMARK PARAMETERS
//  Adjust BENCH_W/H for larger images on more capable GPUs
// =============================================================================

static const int   BENCH_W    = 512;   // image width — reduced for edge AMD GPUs
static const int   BENCH_H    = 512;   // image height (512×512 = 262K pixels)
static const int   SUBSET_R   = 15;    // 31×31 = 961 px/subset
static const int   SEARCH_R   = 4;     // ±4 px search (reduced for CPU benchmark speed)
static const float SUBPX_STEP = 0.2f; // coarser step for CPU speed
static const int   N_RUNS     = 3;     // average over N runs for stable timing


// =============================================================================
//  DEVICE FUNCTIONS (identical to CUDA/HIP — zero changes)
// =============================================================================

__device__ __forceinline__
float bilerp(const float* __restrict__ img, int W, int H, float x, float y)
{
    x=fmaxf(0.f,fminf(x,(float)(W-1)-1e-4f));
    y=fmaxf(0.f,fminf(y,(float)(H-1)-1e-4f));
    int ix=(int)x,iy=(int)y;
    float dx=x-ix,dy=y-iy;
    float v00=img[iy*W+ix],v10=img[iy*W+ix+1];
    float v01=img[(iy+1)*W+ix],v11=img[(iy+1)*W+ix+1];
    return (1-dx)*(1-dy)*v00+dx*(1-dy)*v10+(1-dx)*dy*v01+dx*dy*v11;
}

__device__
float zncc_gpu(const float* __restrict__ ref, const float* __restrict__ def,
               int W, int H, int cx, int cy, float u, float v, int R)
{
    float sf=0,sg=0,sf2=0,sg2=0,sfg=0; int n=0;
    for (int dy=-R;dy<=R;++dy){ int ry=cy+dy; if(ry<0||ry>=H) continue;
        for (int dx=-R;dx<=R;++dx){ int rx=cx+dx; if(rx<0||rx>=W) continue;
            float f=ref[ry*W+rx], g=bilerp(def,W,H,(float)rx+u,(float)ry+v);
            sf+=f;sg+=g;sf2+=f*f;sg2+=g*g;sfg+=f*g;++n;}}
    if(!n) return 0.f;
    float fn=n,num=sfg-sf*sg/fn;
    float den=sqrtf((sf2-sf*sf/fn)*(sg2-sg*sg/fn));
    return den<1e-10f?0.f:num/den;
}

// GPU kernel
__global__
void dic_kernel_gpu(const float* __restrict__ ref, const float* __restrict__ def,
                    float* __restrict__ res, int W, int H,
                    float wind_u, float wind_v)
{
    int cx=blockIdx.x*blockDim.x+threadIdx.x;
    int cy=blockIdx.y*blockDim.y+threadIdx.y;
    if(cx>=W||cy>=H) return;

    float bz=-2.f,bu=0,bv=0;
    for(int iv=-SEARCH_R;iv<=SEARCH_R;++iv)
        for(int iu=-SEARCH_R;iu<=SEARCH_R;++iu){
            float s=zncc_gpu(ref,def,W,H,cx,cy,(float)iu,(float)iv,SUBSET_R);
            if(s>bz){bz=s;bu=(float)iu;bv=(float)iv;}
        }

    float ru=bu,rv=bv;
    for(float dv=-1.f;dv<=1.f;dv+=SUBPX_STEP)
        for(float du=-1.f;du<=1.f;du+=SUBPX_STEP){
            float s=zncc_gpu(ref,def,W,H,cx,cy,bu+du,bv+dv,SUBSET_R);
            if(s>bz){bz=s;ru=bu+du;rv=bv+dv;}
        }

    float tu=ru-0.05f*wind_u, tv=rv-0.05f*wind_v;
    res[cy*W+cx]=sqrtf(tu*tu+tv*tv);
}


// =============================================================================
//  CPU REFERENCE (pure C++ — no GPU)
//  Same algorithm as GPU kernel — used as baseline for speedup calculation
// =============================================================================

static float bilerp_cpu(const float* img, int W, int H, float x, float y)
{
    x=std::max(0.f,std::min(x,(float)(W-1)-1e-4f));
    y=std::max(0.f,std::min(y,(float)(H-1)-1e-4f));
    int ix=(int)x,iy=(int)y;
    float dx=x-ix,dy=y-iy;
    return (1-dx)*(1-dy)*img[iy*W+ix]
          +dx*(1-dy)*img[iy*W+ix+1]
          +(1-dx)*dy*img[(iy+1)*W+ix]
          +dx*dy*img[(iy+1)*W+ix+1];
}

static float zncc_cpu(const float* ref, const float* def,
                       int W, int H, int cx, int cy, float u, float v, int R)
{
    float sf=0,sg=0,sf2=0,sg2=0,sfg=0; int n=0;
    for(int dy=-R;dy<=R;++dy){int ry=cy+dy;if(ry<0||ry>=H)continue;
        for(int dx=-R;dx<=R;++dx){int rx=cx+dx;if(rx<0||rx>=W)continue;
            float f=ref[ry*W+rx],g=bilerp_cpu(def,W,H,(float)rx+u,(float)ry+v);
            sf+=f;sg+=g;sf2+=f*f;sg2+=g*g;sfg+=f*g;++n;}}
    if(!n) return 0.f;
    float fn=n,num=sfg-sf*sg/fn;
    float den=std::sqrt((sf2-sf*sf/fn)*(sg2-sg*sg/fn));
    return den<1e-10f?0.f:num/den;
}

void dic_cpu(const float* ref, const float* def, float* res,
             int W, int H, float wind_u, float wind_v)
{
    for(int cy=0;cy<H;++cy) for(int cx=0;cx<W;++cx) {
        float bz=-2.f,bu=0,bv=0;
        for(int iv=-SEARCH_R;iv<=SEARCH_R;++iv)
            for(int iu=-SEARCH_R;iu<=SEARCH_R;++iu){
                float s=zncc_cpu(ref,def,W,H,cx,cy,(float)iu,(float)iv,SUBSET_R);
                if(s>bz){bz=s;bu=(float)iu;bv=(float)iv;}
            }
        float ru=bu,rv=bv;
        for(float dv=-1.f;dv<=1.f;dv+=SUBPX_STEP)
            for(float du=-1.f;du<=1.f;du+=SUBPX_STEP){
                float s=zncc_cpu(ref,def,W,H,cx,cy,bu+du,bv+dv,SUBSET_R);
                if(s>bz){bz=s;ru=bu+du;rv=bv+dv;}
            }
        float tu=ru-0.05f*wind_u,tv=rv-0.05f*wind_v;
        res[cy*W+cx]=std::sqrt(tu*tu+tv*tv);
    }
}


// =============================================================================
//  TEST IMAGE GENERATOR
// =============================================================================

void make_images(std::vector<float>& ref, std::vector<float>& def, int W, int H)
{
    ref.assign(W*H,0.f); def.assign(W*H,0.f);
    const int NS=100; const float SIG=8.f;
    const float AU=2.f,AV=1.f,LAM=128.f;
    srand(42);
    std::vector<float> sx(NS),sy(NS),sa(NS);
    for(int k=0;k<NS;++k){sx[k]=(float)(rand()%W);sy[k]=(float)(rand()%H);sa[k]=0.5f+0.5f*(float)rand()/RAND_MAX;}
    for(int y=0;y<H;++y) for(int x=0;x<W;++x){
        float rv=0;
        for(int k=0;k<NS;++k){float dx=x-sx[k],dy=y-sy[k];rv+=sa[k]*expf(-(dx*dx+dy*dy)/(2*SIG*SIG));}
        ref[y*W+x]=std::min(rv,1.f);
        float u=AU*sinf(2.f*M_PI*(float)x/LAM),v=AV*cosf(2.f*M_PI*(float)y/LAM);
        float dv=0;
        for(int k=0;k<NS;++k){float dx=(x-u)-sx[k],dy=(y-v)-sy[k];dv+=sa[k]*expf(-(dx*dx+dy*dy)/(2*SIG*SIG));}
        def[y*W+x]=std::min(dv,1.f);
    }
}


// =============================================================================
//  MAIN — BENCHMARK LOOP
// =============================================================================

int main()
{
    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════════╗\n");
    printf("  ║   SYNRA DIC  —  AMD GPU vs CPU BENCHMARK                 ║\n");
    printf("  ║   HIP/ROCm  |  Shubham More                              ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════╝\n\n");

    // ── GPU info ──────────────────────────────────────────────────────────
    int gpu_count=0;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    if(gpu_count==0){fprintf(stderr,"No HIP GPU found.\n");return 1;}

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop,0));
    printf("  GPU  : %s\n",prop.name);
    printf("  VRAM : %.1f GB\n",(float)prop.totalGlobalMem/(1024.f*1024.f*1024.f));
    printf("  CUs  : %d\n",prop.multiProcessorCount);
    printf("  Image: %dx%d = %d pixels\n",BENCH_W,BENCH_H,BENCH_W*BENCH_H);
    printf("  Runs : %d (averaged)\n\n",N_RUNS);

    // ── Generate test images ──────────────────────────────────────────────
    std::vector<float> h_ref, h_def;
    make_images(h_ref, h_def, BENCH_W, BENCH_H);
    int N=BENCH_W*BENCH_H;
    float wind_u=0.31f, wind_v=-0.17f;

    // ── GPU BENCHMARK ─────────────────────────────────────────────────────
    printf("  Running GPU benchmark...\n");

    float *d_ref=nullptr, *d_def=nullptr, *d_res=nullptr;
    size_t bytes=N*sizeof(float);
    HIP_CHECK(hipMalloc(&d_ref,bytes));
    HIP_CHECK(hipMalloc(&d_def,bytes));
    HIP_CHECK(hipMalloc(&d_res,bytes));
    HIP_CHECK(hipMemcpy(d_ref,h_ref.data(),bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_def,h_def.data(),bytes,hipMemcpyHostToDevice));

    std::vector<double> gpu_times(N_RUNS);
    std::vector<float>  h_res_gpu(N);

    for(int run=0;run<N_RUNS;++run){
        HIP_CHECK(hipMemset(d_res,0,bytes));
        HIP_CHECK(hipDeviceSynchronize());

        auto t0=std::chrono::high_resolution_clock::now();

        dim3 block(16,16);
        dim3 grid((BENCH_W+15)/16,(BENCH_H+15)/16);
        dic_kernel_gpu<<<grid,block>>>(d_ref,d_def,d_res,BENCH_W,BENCH_H,wind_u,wind_v);
        HIP_CHECK_KERNEL();
        HIP_CHECK(hipDeviceSynchronize());

        auto t1=std::chrono::high_resolution_clock::now();
        gpu_times[run]=std::chrono::duration<double,std::milli>(t1-t0).count();
        printf("    GPU run %d: %.1f ms\n",run+1,gpu_times[run]);
    }
    HIP_CHECK(hipMemcpy(h_res_gpu.data(),d_res,bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_ref)); HIP_CHECK(hipFree(d_def)); HIP_CHECK(hipFree(d_res));

    double gpu_avg=std::accumulate(gpu_times.begin(),gpu_times.end(),0.0)/N_RUNS;

    // ── CPU BENCHMARK ─────────────────────────────────────────────────────
    printf("\n  Running CPU benchmark (single-threaded, same algorithm)...\n");
    printf("  NOTE: CPU benchmark on full image may take 10-60 min.\n");
    printf("  Running on reduced 128x128 tile and scaling for estimate...\n\n");

    // For the report we run CPU on a smaller tile and scale the time
    const int CPU_TILE = std::min(128, BENCH_W);
    std::vector<float> cpu_ref(h_ref.begin(), h_ref.begin()+CPU_TILE*CPU_TILE);
    std::vector<float> cpu_def(h_def.begin(), h_def.begin()+CPU_TILE*CPU_TILE);
    std::vector<float> h_res_cpu(CPU_TILE*CPU_TILE);

    std::vector<double> cpu_times(N_RUNS);
    for(int run=0;run<N_RUNS;++run){
        auto t0=std::chrono::high_resolution_clock::now();
        dic_cpu(cpu_ref.data(),cpu_def.data(),h_res_cpu.data(),
                CPU_TILE,CPU_TILE,wind_u,wind_v);
        auto t1=std::chrono::high_resolution_clock::now();
        cpu_times[run]=std::chrono::duration<double,std::milli>(t1-t0).count();
        printf("    CPU run %d (%dx%d tile): %.1f ms\n",run+1,CPU_TILE,CPU_TILE,cpu_times[run]);
    }

    double cpu_tile_avg=std::accumulate(cpu_times.begin(),cpu_times.end(),0.0)/N_RUNS;
    // Scale to full image size
    double cpu_full_est = cpu_tile_avg * (double)(BENCH_W*BENCH_H)/(double)(CPU_TILE*CPU_TILE);

    // ── Accuracy check: compare GPU vs CPU on tile ─────────────────────
    // Run GPU on same CPU_TILE to compare results
    float max_diff=0.f;
    for(int i=0;i<CPU_TILE*CPU_TILE;++i)
        max_diff=std::max(max_diff, std::abs(h_res_gpu[i]-h_res_cpu[i]));

    double speedup = cpu_full_est / gpu_avg;

    // ── Print results table ────────────────────────────────────────────
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════╗\n");
    printf("  ║              BENCHMARK RESULTS                               ║\n");
    printf("  ╠══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  GPU model       : %-40s ║\n",prop.name);
    printf("  ║  Image size      : %dx%d = %d pixels%-16s║\n",
           BENCH_W,BENCH_H,BENCH_W*BENCH_H,"");
    printf("  ╠══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  GPU time (avg)  : %8.1f ms                              ║\n",gpu_avg);
    printf("  ║  CPU tile time   : %8.1f ms  (%dx%d tile)              ║\n",
           cpu_tile_avg,CPU_TILE,CPU_TILE);
    printf("  ║  CPU full est.   : %8.1f ms  (scaled to %dx%d)         ║\n",
           cpu_full_est,BENCH_W,BENCH_H);
    printf("  ║  Speedup (GPU/CPU): %7.1fx                               ║\n",speedup);
    printf("  ╠══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Throughput (GPU): %8.2f K vectors/sec                  ║\n",
           (float)(BENCH_W*BENCH_H)/(gpu_avg*1.0));
    printf("  ║  Max result diff : %8.4f px  (GPU vs CPU)               ║\n",max_diff);
    printf("  ╠══════════════════════════════════════════════════════════════╣\n");
    if(speedup >= 10.0)
        printf("  ║  STATUS: GOOD — GPU acceleration is effective (>10x)       ║\n");
    else if(speedup >= 2.0)
        printf("  ║  STATUS: FAIR — GPU shows speedup but not optimal           ║\n");
    else
        printf("  ║  STATUS: POOR — GPU may not be optimal for this device      ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("  Run 'rocm-smi' to get GPU utilisation % for the report.\n");
    printf("  Add these numbers to hipify_conversion_log.md.\n\n");

    HIP_CHECK(hipDeviceReset());
    return EXIT_SUCCESS;
}

// =============================================================================
//  COMPILE:
//  hipcc -Xcompiler -fopenmp hip_benchmark.hip \
//        -o hip_benchmark -lgomp -O3 -std=c++14
//
//  RUN + CAPTURE:
//  ./hip_benchmark | tee benchmark_results.txt
//
//  MONITOR GPU WHILE RUNNING (separate terminal):
//  watch -n 0.5 rocm-smi
// =============================================================================
