// =============================================================================
//  dic_hip_dmd.hip
//  Author  : Shubham | SYNRA Metrology Systems
//  Purpose : AMD/HIP port of dic_dmd_analysis.cu (v1.1 fixed)
//
//  KEY CHANGES vs dic_dmd_analysis.cu:
//  ────────────────────────────────────
//  1. cuSOLVER → rocSOLVER  (SVD: Sgesvd)
//  2. cuBLAS   → rocBLAS    (GEMM: Sgemm)
//  3. IMPORTANT: rocSOLVER uses a rocBLAS handle — NOT its own handle type
//     cuSOLVER: cusolverDnHandle_t  → rocSOLVER reuses rocblas_handle
//
//  CPU eigendecomposition (Francis QR) is UNCHANGED — runs on CPU in both.
//  All GPU kernels are UNCHANGED — identical in CUDA and HIP.
//
//  CUDA→HIP CONVERSION LOG:
//  ─────────────────────────
//  AUTOMATIC (hipify-perl):
//    cuda_runtime.h            → hip/hip_runtime.h
//    cudaMalloc/Free/etc       → hipMalloc/Free/etc
//
//  MANUAL (cuSOLVER/cuBLAS → rocSOLVER/rocBLAS):
//    #include <cusolverDn.h>          → #include <rocsolver/rocsolver.h>
//    #include <cublas_v2.h>           → #include <rocblas/rocblas.h>
//    cusolverDnHandle_t cusolver      → rocblas_handle handle (shared!)
//    cusolverDnCreate(&cusolver)      → rocblas_create_handle(&handle)
//    cusolverDnDestroy(cusolver)      → rocblas_destroy_handle(handle)
//    cublasHandle_t cublas            → (removed — reuse rocblas_handle)
//    cublasCreate(&cublas)            → (removed)
//    cublasDestroy(cublas)            → (removed)
//    cusolverDnSgesvd(cusolver,...)   → rocsolver_sgesvd(handle,...)
//    cublasSgemm(cublas,...)          → rocblas_sgemm(handle,...)
//    CUBLAS_OP_N                      → rocblas_operation_none
//    CUBLAS_OP_T                      → rocblas_operation_transpose
//    -lcusolver -lcublas              → -lrocsolver -lrocblas
//
//  COMPILE ON AMD:
//  hipcc -Xcompiler -fopenmp dic_hip_dmd.hip \
//        -o dic_hip_dmd -lgomp -lrocsolver -lrocblas -O3 -std=c++14
//
//  COMPILE ON NVIDIA (same source, hipcc CUDA backend):
//  hipcc -Xcompiler -fopenmp dic_hip_dmd.hip \
//        -o dic_hip_dmd -lgomp -lrocsolver -lrocblas -O3 -std=c++14
// =============================================================================

#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>    // ← replaces <cusolverDn.h>
#include <rocblas/rocblas.h>        // ← replaces <cublas_v2.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
//  SECTION 1 — ERROR HANDLING
// =============================================================================

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t _e=(call);                                                \
        if (_e!=hipSuccess){                                                  \
            fprintf(stderr,"\n[HIP ERROR]  %s\n  at %s:%d\n  -> %s\n",     \
                #call,__FILE__,__LINE__,hipGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

// rocBLAS error check (rocSOLVER also returns rocblas_status)
#define ROCBLAS_CHECK(call)                                                   \
    do {                                                                      \
        rocblas_status _s=(call);                                            \
        if (_s!=rocblas_status_success){                                      \
            fprintf(stderr,"\n[rocBLAS ERROR]  %s\n  at %s:%d  code=%d\n",  \
                #call,__FILE__,__LINE__,(int)_s);                            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

#define HIP_CHECK_KERNEL()                                                    \
    do {                                                                      \
        hipError_t _e=hipGetLastError();                                     \
        if (_e!=hipSuccess){                                                  \
            fprintf(stderr,"\n[HIP KERNEL ERROR] %s at %s:%d\n",             \
                hipGetErrorString(_e),__FILE__,__LINE__);                    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)


// =============================================================================
//  SECTION 2 — CONFIGURATION  (identical to CUDA version)
// =============================================================================

static const int   IMG_W         = 1024;
static const int   IMG_H         = 1024;
static const int   N_FRAMES      = 256;
static const float SAMPLE_RATE   = 1000.f;
static const int   DMD_RANK      = 32;
static const float SVD_THRESHOLD = 1e-6f;


// =============================================================================
//  SECTION 3 — CPU EIGENDECOMPOSITION  (ZERO CHANGES from CUDA version)
//  The Francis QR algorithm runs on CPU in both CUDA and HIP versions.
//  Atilde is 32×32 — trivially fast on CPU (~0.1 ms).
// =============================================================================

static void eig2x2(double a00,double a01,double a10,double a11,
                   double& re0,double& im0,double& re1,double& im1)
{
    double tr=a00+a11, det=a00*a11-a01*a10;
    double disc=tr*tr-4.0*det;
    if (disc>=0.0){double sq=std::sqrt(disc);re0=(tr+sq)*0.5;im0=0;re1=(tr-sq)*0.5;im1=0;}
    else{re0=tr*0.5;im0=std::sqrt(-disc)*0.5;re1=tr*0.5;im1=-std::sqrt(-disc)*0.5;}
}

static void toHessenberg(std::vector<double>& A, int n)
{
    for (int k=0;k<n-2;++k){
        double norm2=0;
        for (int i=k+1;i<n;++i) norm2+=A[i*n+k]*A[i*n+k];
        double norm=std::sqrt(norm2);
        if (norm<1e-14) continue;
        std::vector<double> v(n,0);
        for (int i=k+1;i<n;++i) v[i]=A[i*n+k];
        double sign=(A[(k+1)*n+k]>=0)?1.0:-1.0;
        v[k+1]+=sign*norm;
        double vn2=0; for (int i=k+1;i<n;++i) vn2+=v[i]*v[i];
        if (vn2<1e-28) continue;
        double beta=2.0/vn2;
        for (int j=k;j<n;++j){double dot=0;for (int i=k+1;i<n;++i) dot+=v[i]*A[i*n+j];for (int i=k+1;i<n;++i) A[i*n+j]-=beta*dot*v[i];}
        for (int i=0;i<n;++i){double dot=0;for (int j=k+1;j<n;++j) dot+=A[i*n+j]*v[j];for (int j=k+1;j<n;++j) A[i*n+j]-=beta*dot*v[j];}
    }
}

static void francisStep(std::vector<double>& H,int n,int lo,int hi)
{
    double s=H[(hi-1)*n+(hi-1)]+H[hi*n+hi];
    double t=H[(hi-1)*n+(hi-1)]*H[hi*n+hi]-H[(hi-1)*n+hi]*H[hi*n+(hi-1)];
    double x=H[lo*n+lo]*H[lo*n+lo]+H[lo*n+(lo+1)]*H[(lo+1)*n+lo]-s*H[lo*n+lo]+t;
    double y=H[(lo+1)*n+lo]*(H[lo*n+lo]+H[(lo+1)*n+(lo+1)]-s);
    double z=(lo+2<=hi)?H[(lo+2)*n+lo]*H[(lo+1)*n+lo]:0.0;
    for (int k=lo;k<=hi-1;++k){
        int sz=(k+2<=hi)?3:2;
        double norm=std::sqrt(x*x+y*y+z*z);
        if (norm<1e-14){x=H[(k+1)*n+k];y=(k+2<=hi)?H[(k+2)*n+k]:0;z=(k+3<=hi)?H[(k+3)*n+k]:0;continue;}
        double sign=(x>=0)?1:-1;
        double u0=x+sign*norm,u1=y,u2=(sz==3)?z:0;
        double un2=u0*u0+u1*u1+u2*u2; double beta=2.0/un2;
        int jlo=(k>lo)?k-1:lo,jhi=std::min(n-1,k+sz);
        for (int j=jlo;j<=jhi;++j){double dot=u0*H[k*n+j]+u1*((k+1<=hi)?H[(k+1)*n+j]:0.0)+u2*((k+2<=hi&&sz==3)?H[(k+2)*n+j]:0.0);H[k*n+j]-=beta*dot*u0;if(k+1<=hi)H[(k+1)*n+j]-=beta*dot*u1;if(k+2<=hi&&sz==3)H[(k+2)*n+j]-=beta*dot*u2;}
        int ihi2=std::min(n-1,k+sz);
        for (int i=lo;i<=ihi2;++i){double dot=u0*H[i*n+k]+u1*((k+1<=hi)?H[i*n+(k+1)]:0.0)+u2*((k+2<=hi&&sz==3)?H[i*n+(k+2)]:0.0);H[i*n+k]-=beta*dot*u0;if(k+1<=hi)H[i*n+(k+1)]-=beta*dot*u1;if(k+2<=hi&&sz==3)H[i*n+(k+2)]-=beta*dot*u2;}
        x=H[(k+1)*n+k];y=(k+2<=hi)?H[(k+2)*n+k]:0;z=(k+3<=hi)?H[(k+3)*n+k]:0;
    }
}

static void cpu_eigenvalues(const std::vector<double>& A_in,int n,
                             std::vector<double>& eig_re,std::vector<double>& eig_im)
{
    std::vector<double> H=A_in;
    eig_re.assign(n,0); eig_im.assign(n,0);
    toHessenberg(H,n);
    const double eps=1e-12; int iter=0,maxiter=300*n,hi=n-1;
    while (hi>=1){
        int lo=0;
        for (int k=hi;k>=1;--k)
            if (std::abs(H[k*n+(k-1)])<eps*(std::abs(H[(k-1)*n+(k-1)])+std::abs(H[k*n+k]))){H[k*n+(k-1)]=0;lo=k;break;}
        if (lo==hi){eig_re[hi]=H[hi*n+hi];eig_im[hi]=0;hi--;}
        else if (lo==hi-1){eig2x2(H[(hi-1)*n+(hi-1)],H[(hi-1)*n+hi],H[hi*n+(hi-1)],H[hi*n+hi],eig_re[hi-1],eig_im[hi-1],eig_re[hi],eig_im[hi]);hi-=2;}
        else{if(++iter>maxiter){fprintf(stderr,"[WARN] QR not converged\n");break;}francisStep(H,n,lo,hi);}
    }
    if (hi==0){eig_re[0]=H[0];eig_im[0]=0;}
}

static void cpu_eigenvectors_approx(const std::vector<double>& A,int n,
                                     const std::vector<double>& eig_re,
                                     const std::vector<double>& eig_im,
                                     std::vector<float>& evec_re,
                                     std::vector<float>& evec_im)
{
    evec_re.assign(n*n,0.f); evec_im.assign(n*n,0.f);
    for (int k=0;k<n;++k){
        std::vector<double> v(n,0); v[k%n]=1.0;
        double lmag=std::sqrt(eig_re[k]*eig_re[k]+eig_im[k]*eig_im[k]);
        for (int it=0;it<30;++it){
            std::vector<double> Av(n,0);
            for (int i=0;i<n;++i) for (int j=0;j<n;++j) Av[i]+=A[i*n+j]*v[j];
            for (int i=0;i<n;++i) Av[i]-=eig_re[k]*v[i];
            double norm=0; for (double vi:Av) norm+=vi*vi; norm=std::sqrt(norm);
            if (norm<1e-15) break;
            for (int i=0;i<n;++i) v[i]=Av[i]/norm;
        }
        double norm=0; for (double vi:v) norm+=vi*vi; norm=std::sqrt(std::max(norm,1e-15));
        for (int i=0;i<n;++i){
            evec_re[k*n+i]=(float)(v[i]/norm);
            evec_im[k*n+i]=(float)(v[i]*eig_im[k]/(lmag+1e-15)/norm);
        }
    }
}


// =============================================================================
//  SECTION 4 — GPU KERNELS  (ZERO CHANGES from CUDA version)
// =============================================================================

__global__
void build_data_matrices_kernel(const float* d_frames,float* d_X,float* d_Xp,int n_pixels,int N)
{
    int pix=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    if (pix>=n_pixels||col>=N-1) return;
    d_X [col*n_pixels+pix]=d_frames[ col   *n_pixels+pix];
    d_Xp[col*n_pixels+pix]=d_frames[(col+1)*n_pixels+pix];
}

__global__
void scale_by_inv_sigma_kernel(float* d_mat,const float* d_sigma,
                                int n_rows,int r,float sigma_max,float threshold)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    if (row>=n_rows||col>=r) return;
    float s=d_sigma[col];
    float inv_s=(s>threshold*sigma_max)?1.f/s:0.f;
    d_mat[col*n_rows+row]*=inv_s;
}

__global__
void extract_dmd_eigenvalues_kernel(const float* d_eig_real,const float* d_eig_imag,
                                    float* d_freq_hz,float* d_decay,float* d_amplitude,
                                    int r,float fs)
{
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=r) return;
    float re=d_eig_real[k],im=d_eig_imag[k];
    float mag=sqrtf(re*re+im*im), angle=atan2f(im,re);
    d_freq_hz [k]=fabsf(angle)*fs/(2.f*M_PI);
    d_decay   [k]=logf(fmaxf(mag,1e-10f))*fs;
    d_amplitude[k]=mag;
}

__global__
void reconstruct_spatial_modes_kernel(const float* d_U,
                                       const float* d_eigvec_re,const float* d_eigvec_im,
                                       float* d_mode_re,float* d_mode_im,int n_pixels,int r)
{
    int pix=blockIdx.x*blockDim.x+threadIdx.x;
    int k  =blockIdx.y*blockDim.y+threadIdx.y;
    if (pix>=n_pixels||k>=r) return;
    float re=0,im=0;
    for (int j=0;j<r;++j){float u=d_U[j*n_pixels+pix];re+=u*d_eigvec_re[k*r+j];im+=u*d_eigvec_im[k*r+j];}
    d_mode_re[k*n_pixels+pix]=re;
    d_mode_im[k*n_pixels+pix]=im;
}

__global__
void dmd_mode_amplitude_kernel(const float* d_mode_re,const float* d_mode_im,
                                float* d_amp_map,int r,int n_pixels)
{
    int pix=blockIdx.x*blockDim.x+threadIdx.x;
    int k  =blockIdx.y*blockDim.y+threadIdx.y;
    if (pix>=n_pixels||k>=r) return;
    float re=d_mode_re[k*n_pixels+pix],im=d_mode_im[k*n_pixels+pix];
    d_amp_map[k*n_pixels+pix]=sqrtf(re*re+im*im);
}


// =============================================================================
//  SECTION 5 — TEST DATA  (identical to CUDA version)
// =============================================================================

void generate_dmd_test_data(std::vector<float>& h_frames,int W,int H,int N,float fs)
{
    h_frames.resize((size_t)N*W*H,0.f);
    int n_pixels=W*H; float dt=1.f/fs;
    struct Mode{float f,decay;};
    const Mode modes[]={{47.f,-2.f},{92.f,-5.f},{153.f,-8.f}};
    printf("[SYN-DMD] Generating %d frames, %dx%d, fs=%.0f Hz\n",N,W,H,fs);
    for (int t=0;t<N;++t){
        float time=(float)t*dt;
        float* frame=h_frames.data()+(size_t)t*n_pixels;
        for (int y=0;y<H;++y) for (int x=0;x<W;++x){
            int pix=y*W+x;
            float fx=(float)x/(W-1),fy=(float)y/(H-1);
            float v=0.8f*sinf(M_PI*fx)*expf(modes[0].decay*time)*cosf(2.f*M_PI*modes[0].f*time)
                   +0.3f*sinf(2.f*M_PI*fx)*expf(modes[1].decay*time)*cosf(2.f*M_PI*modes[1].f*time)
                   +0.15f*sinf(M_PI*fx)*sinf(M_PI*fy)*expf(modes[2].decay*time)*cosf(2.f*M_PI*modes[2].f*time)
                   +0.01f*((float)rand()/RAND_MAX-0.5f);
            frame[pix]=v;
        }
    }
}


// =============================================================================
//  SECTION 6 — MAIN DMD PIPELINE (rocSOLVER + rocBLAS)
// =============================================================================

void run_dmd_pipeline(int W,int H,int N,float fs,const std::vector<float>& h_frames)
{
    int n_pixels=W*H, m=N-1, r=std::min(DMD_RANK,std::min(m,64));
    printf("\n[DMD-HIP] n_pixels=%d  m=%d  rank_r=%d\n",n_pixels,m,r);

    HIP_CHECK(hipSetDevice(0));

    // ── KEY DIFFERENCE: one rocblas_handle serves BOTH rocSOLVER and rocBLAS
    // cuSOLVER had its own handle type. rocSOLVER reuses rocBLAS handle.
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // ── Allocations ───────────────────────────────────────────────────────
    float *d_frames_dev=nullptr,*d_X=nullptr,*d_Xp=nullptr;
    float *d_U=nullptr,*d_S=nullptr,*d_VT=nullptr,*d_Atilde=nullptr;
    float *d_eig_re=nullptr,*d_eig_im=nullptr;
    float *d_evec_re=nullptr,*d_evec_im=nullptr;
    float *d_freq_hz=nullptr,*d_decay=nullptr,*d_amp=nullptr;
    float *d_mode_re=nullptr,*d_mode_im=nullptr,*d_amp_map=nullptr;

    size_t fb=(size_t)N*n_pixels*sizeof(float);
    size_t Xb=(size_t)n_pixels*m*sizeof(float);
    size_t Ub=(size_t)n_pixels*r*sizeof(float);
    size_t Vb=(size_t)r*m*sizeof(float);
    size_t Ab=(size_t)r*r*sizeof(float);
    size_t Mb=(size_t)r*n_pixels*sizeof(float);
    size_t rb=(size_t)r*sizeof(float);
    size_t r2=(size_t)r*r*sizeof(float);

    HIP_CHECK(hipMalloc(&d_frames_dev,fb));
    HIP_CHECK(hipMalloc(&d_X,Xb));    HIP_CHECK(hipMalloc(&d_Xp,Xb));
    HIP_CHECK(hipMalloc(&d_U,Ub));    HIP_CHECK(hipMalloc(&d_S,rb));
    HIP_CHECK(hipMalloc(&d_VT,Vb));   HIP_CHECK(hipMalloc(&d_Atilde,Ab));
    HIP_CHECK(hipMalloc(&d_eig_re,rb));HIP_CHECK(hipMalloc(&d_eig_im,rb));
    HIP_CHECK(hipMalloc(&d_evec_re,r2));HIP_CHECK(hipMalloc(&d_evec_im,r2));
    HIP_CHECK(hipMalloc(&d_freq_hz,rb));HIP_CHECK(hipMalloc(&d_decay,rb));
    HIP_CHECK(hipMalloc(&d_amp,rb));
    HIP_CHECK(hipMalloc(&d_mode_re,Mb));HIP_CHECK(hipMalloc(&d_mode_im,Mb));
    HIP_CHECK(hipMalloc(&d_amp_map,Mb));

    HIP_CHECK(hipMemcpy(d_frames_dev,h_frames.data(),fb,hipMemcpyHostToDevice));
    printf("[DMD-HIP] Frames uploaded (%.1f MB)\n",fb/1e6f);

    // ── Step 1: Build X and X' ─────────────────────────────────────────────
    {
        dim3 blk(32,8),grd((n_pixels+31)/32,(m+7)/8);
        build_data_matrices_kernel<<<grd,blk>>>(d_frames_dev,d_X,d_Xp,n_pixels,N);
        HIP_CHECK_KERNEL(); HIP_CHECK(hipDeviceSynchronize());
    }
    printf("[DMD-HIP] Step 1: X and X' built\n");

    // ── Step 2: Truncated SVD via rocsolver_sgesvd ────────────────────────
    //
    //  rocsolver_sgesvd signature:
    //    rocsolver_sgesvd(handle, left_svect, right_svect,
    //                     m, n, A, lda, S, U, ldu, V, ldv,
    //                     E,    ← superdiagonal (not in cuSOLVER Sgesvd)
    //                     fast_alg, devInfo)
    //
    //  rocsolver uses rocblas_svect enum instead of char:
    //    'S' → rocblas_svect_singular
    //    'A' → rocblas_svect_all
    //    'N' → rocblas_svect_none
    {
        float *d_Uf=nullptr,*d_VTf=nullptr,*d_Sf=nullptr,*d_E=nullptr;
        int *d_info=nullptr;
        HIP_CHECK(hipMalloc(&d_Uf,(size_t)n_pixels*m*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_VTf,(size_t)m*m*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_Sf,m*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_E, m*sizeof(float)));  // superdiagonal — not in cuSOLVER
        HIP_CHECK(hipMalloc(&d_info,sizeof(int)));

        // rocsolver_sgesvd: thin SVD of X (n_pixels × m)
        ROCBLAS_CHECK(rocsolver_sgesvd(
            handle,
            rocblas_svect_singular,   // left:  U [n_pixels × m]
            rocblas_svect_singular,   // right: VT [m × m]
            n_pixels, m,
            d_X,       n_pixels,      // input A (overwritten)
            d_Sf,                     // singular values [min(n,m)]
            d_Uf,      n_pixels,      // U [n_pixels × m]
            d_VTf,     m,             // VT [m × m]
            d_E,                      // superdiagonal (required by rocSOLVER)
            rocblas_outofplace,       // fast_alg parameter
            d_info));

        HIP_CHECK(hipDeviceSynchronize());

        // Truncate to rank r
        HIP_CHECK(hipMemcpy(d_U, d_Uf, Ub, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_S, d_Sf, rb, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_VT,d_VTf,Vb, hipMemcpyDeviceToDevice));

        HIP_CHECK(hipFree(d_Uf)); HIP_CHECK(hipFree(d_VTf));
        HIP_CHECK(hipFree(d_Sf)); HIP_CHECK(hipFree(d_E));
        HIP_CHECK(hipFree(d_info));
    }
    printf("[DMD-HIP] Step 2: SVD done, rank r=%d\n",r);

    // ── Step 3: Build Atilde = U* · X' · V · Sigma^{-1} via rocBLAS GEMM ─
    //
    //  rocblas_sgemm signature:
    //    rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda,
    //                  B, ldb, beta, C, ldc)
    //
    //  Operation enum changes:
    //    CUBLAS_OP_N → rocblas_operation_none
    //    CUBLAS_OP_T → rocblas_operation_transpose
    {
        float *d_tmp=nullptr; HIP_CHECK(hipMalloc(&d_tmp,Ub));
        const float alpha=1.f,beta=0.f;

        // tmp = Xp · V  (rocblas_sgemm — same call structure as cublasSgemm)
        ROCBLAS_CHECK(rocblas_sgemm(handle,
            rocblas_operation_none,       // ← was CUBLAS_OP_N
            rocblas_operation_transpose,  // ← was CUBLAS_OP_T
            n_pixels, r, m,
            &alpha, d_Xp, n_pixels,
                    d_VT, r,
            &beta,  d_tmp, n_pixels));

        // tmp = tmp · Sigma^{-1}
        float h_s0;
        HIP_CHECK(hipMemcpy(&h_s0,d_S,sizeof(float),hipMemcpyDeviceToHost));
        dim3 blk(32,8),grd((n_pixels+31)/32,(r+7)/8);
        scale_by_inv_sigma_kernel<<<grd,blk>>>(d_tmp,d_S,n_pixels,r,h_s0,SVD_THRESHOLD);
        HIP_CHECK_KERNEL();

        // Atilde = U^T · tmp
        ROCBLAS_CHECK(rocblas_sgemm(handle,
            rocblas_operation_transpose,  // ← was CUBLAS_OP_T
            rocblas_operation_none,       // ← was CUBLAS_OP_N
            r, r, n_pixels,
            &alpha, d_U,   n_pixels,
                    d_tmp, n_pixels,
            &beta,  d_Atilde, r));

        HIP_CHECK(hipFree(d_tmp));
    }
    printf("[DMD-HIP] Step 3: Atilde built (%dx%d)\n",r,r);

    // ── Step 4: Eigendecomposition on CPU (UNCHANGED from CUDA version) ──
    {
        std::vector<float> h_At_f(r*r);
        HIP_CHECK(hipMemcpy(h_At_f.data(),d_Atilde,r2,hipMemcpyDeviceToHost));
        std::vector<double> h_At_d(r*r);
        for (int i=0;i<r*r;++i) h_At_d[i]=(double)h_At_f[i];

        std::vector<double> eig_re_d,eig_im_d;
        cpu_eigenvalues(h_At_d,r,eig_re_d,eig_im_d);

        std::vector<float> h_evec_re,h_evec_im;
        cpu_eigenvectors_approx(h_At_d,r,eig_re_d,eig_im_d,h_evec_re,h_evec_im);

        std::vector<float> h_eig_re(r),h_eig_im(r);
        for (int k=0;k<r;++k){h_eig_re[k]=(float)eig_re_d[k];h_eig_im[k]=(float)eig_im_d[k];}
        HIP_CHECK(hipMemcpy(d_eig_re, h_eig_re.data(), rb, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_eig_im, h_eig_im.data(), rb, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_evec_re,h_evec_re.data(),r2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_evec_im,h_evec_im.data(),r2, hipMemcpyHostToDevice));
    }
    printf("[DMD-HIP] Step 4: eigendecomposition done (CPU Francis QR)\n");

    // ── Steps 5–6: frequencies, spatial modes ────────────────────────────
    {
        int blk=128,grd=(r+127)/128;
        extract_dmd_eigenvalues_kernel<<<grd,blk>>>(
            d_eig_re,d_eig_im,d_freq_hz,d_decay,d_amp,r,SAMPLE_RATE);
        HIP_CHECK_KERNEL(); HIP_CHECK(hipDeviceSynchronize());
    }
    {
        dim3 blk(32,8),grd((n_pixels+31)/32,(r+7)/8);
        reconstruct_spatial_modes_kernel<<<grd,blk>>>(
            d_U,d_evec_re,d_evec_im,d_mode_re,d_mode_im,n_pixels,r);
        HIP_CHECK_KERNEL();
        dmd_mode_amplitude_kernel<<<grd,blk>>>(d_mode_re,d_mode_im,d_amp_map,r,n_pixels);
        HIP_CHECK_KERNEL(); HIP_CHECK(hipDeviceSynchronize());
    }
    printf("[DMD-HIP] Steps 5-6: frequencies + spatial modes done\n");

    // ── Results ───────────────────────────────────────────────────────────
    std::vector<float> h_freq(r),h_decay_v(r),h_amp_v(r);
    HIP_CHECK(hipMemcpy(h_freq.data(),   d_freq_hz,rb,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_decay_v.data(),d_decay,  rb,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_amp_v.data(),  d_amp,    rb,hipMemcpyDeviceToHost));

    std::vector<int> idx(r); for (int k=0;k<r;++k) idx[k]=k;
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return h_amp_v[a]>h_amp_v[b];});

    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────┐\n");
    printf("  │   DMD RESULTS (HIP/rocSOLVER) — TOP MODES           │\n");
    printf("  ├──────┬────────────┬────────────┬─────────────────────┤\n");
    printf("  │ Mode │ Freq (Hz)  │ Decay(1/s) │  |eigenvalue|       │\n");
    printf("  ├──────┼────────────┼────────────┼─────────────────────┤\n");
    for (int i=0;i<std::min(r,10);++i){
        int k=idx[i];
        if (h_freq[k]<0.5f) continue;
        printf("  │  %3d │ %10.3f │ %10.3f │ %19.6f │\n",
               k,h_freq[k],h_decay_v[k],h_amp_v[k]);
    }
    printf("  └──────┴────────────┴────────────┴─────────────────────┘\n");
    printf("  Validation: peaks expected near 47, 92, 153 Hz\n\n");

    // ── Cleanup ───────────────────────────────────────────────────────────
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_frames_dev)); HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_Xp));         HIP_CHECK(hipFree(d_U));
    HIP_CHECK(hipFree(d_S));           HIP_CHECK(hipFree(d_VT));
    HIP_CHECK(hipFree(d_Atilde));      HIP_CHECK(hipFree(d_eig_re));
    HIP_CHECK(hipFree(d_eig_im));      HIP_CHECK(hipFree(d_evec_re));
    HIP_CHECK(hipFree(d_evec_im));     HIP_CHECK(hipFree(d_freq_hz));
    HIP_CHECK(hipFree(d_decay));       HIP_CHECK(hipFree(d_amp));
    HIP_CHECK(hipFree(d_mode_re));     HIP_CHECK(hipFree(d_mode_im));
    HIP_CHECK(hipFree(d_amp_map));
}


// =============================================================================
//  SECTION 7 — MAIN
// =============================================================================

int main()
{
    int gpu_count=0;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));

    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════╗\n");
    printf("  ║   SYNRA — DMD ENGINE (HIP/rocSOLVER + rocBLAS)       ║\n");
    printf("  ║   cuSOLVER+cuBLAS → rocSOLVER+rocBLAS port           ║\n");
    printf("  ╚═══════════════════════════════════════════════════════╝\n\n");

    for (int i=0;i<gpu_count;++i){
        hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p,i));
        printf("  GPU %d : %-24s | %.0f GB\n",i,p.name,
               (float)p.totalGlobalMem/(1024.f*1024.f*1024.f));
    }
    printf("\n  Image: %dx%d  N=%d  rank=%d\n",IMG_W,IMG_H,N_FRAMES,DMD_RANK);
    printf("  Key: rocSOLVER reuses rocBLAS handle (no separate cusolverDnHandle)\n\n");

    srand(42);
    std::vector<float> h_frames;
    generate_dmd_test_data(h_frames,IMG_W,IMG_H,N_FRAMES,SAMPLE_RATE);

    auto t0=std::chrono::high_resolution_clock::now();
    run_dmd_pipeline(IMG_W,IMG_H,N_FRAMES,SAMPLE_RATE,h_frames);
    auto t1=std::chrono::high_resolution_clock::now();

    printf("  Wall time: %.1f ms\n\n",
           std::chrono::duration<double,std::milli>(t1-t0).count());

    HIP_CHECK(hipDeviceReset());
    printf("  [DONE]\n\n");
    return EXIT_SUCCESS;
}

// =============================================================================
//  COMPILE:
//  hipcc -Xcompiler -fopenmp dic_hip_dmd.hip \
//        -o dic_hip_dmd -lgomp -lrocsolver -lrocblas -O3 -std=c++14
// =============================================================================
