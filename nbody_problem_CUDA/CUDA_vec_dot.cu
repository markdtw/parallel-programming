#include <stdio.h>
#define ELEMENT_COUNT 1024*1024
#define BLOCKSPERGRID 1024
#define THREADSPERBLOCK 1024
__global__ void vecMul_gpu_kernel(double vecA[], double vecB[], double vecC[]) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    vecC[i] = vecB[i] * vecA[i];
}
__global__ void vecSum_gpu_kernel(double vecC[], double partial_sum[]) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    //double local_sum = 0;

    __shared__ double vec[THREADSPERBLOCK];

    vec[tid] = vecC[id];
    __syncthreads();

    if (tid<512)
        vec[tid] += vec[tid+512];
    __syncthreads();
    if (tid<256)
        vec[tid] += vec[tid+256];
    __syncthreads();
    if (tid<128)
        vec[tid] += vec[tid+128];
    __syncthreads();
    if (tid<64)
        vec[tid] += vec[tid+64];
    __syncthreads();
    if (tid<32)
        vec[tid] += vec[tid+32];
    if (tid<16)
        vec[tid] += vec[tid+16];
    if (tid<8)
        vec[tid] += vec[tid+8];
    if (tid<4)
        vec[tid] += vec[tid+4];
    if (tid<2)
        vec[tid] += vec[tid+2];
    if (tid<1)
        vec[tid] += vec[tid+1];

    if (threadIdx.x==0)
        vecC[blockIdx.x] = vec[0];

}
void vecDot_cpu(double *S, double vecA[], double vecB[]) {
    int i;
    for (i=0; i<ELEMENT_COUNT; i++)
        *S += vecB[i] * vecA[i];
}
int main (int argc, char **argv) {

    cudaSetDevice(0);

    double *h_vecA, *h_vecB, *h_psum;
    double *S_gpu, *S_cpu;
    h_vecA = (double*)malloc(sizeof(double)*ELEMENT_COUNT);
    h_vecB = (double*)malloc(sizeof(double)*ELEMENT_COUNT);
    h_psum = (double*)malloc(sizeof(double)*THREADSPERBLOCK);
    S_gpu = (double*)malloc(sizeof(double));
    S_cpu = (double*)malloc(sizeof(double));
    *S_gpu = *S_cpu = 0;

    srand(time(0));

    int i;
    for(i=0; i<ELEMENT_COUNT; i++) {
        h_vecA[i] = rand()%100;
        h_vecB[i] = rand()%100;
    }

    cudaError_t R;

    double *d_vecA, *d_vecB, *d_vecC, *d_psum;
    printf("\n========== Check cudaMalloc ==========\n");
    R = cudaMalloc((void **)&d_vecA, sizeof(double)*ELEMENT_COUNT);
    printf(" Malloc d_vecA: %s\n", cudaGetErrorString(R));
    R = cudaMalloc((void **)&d_vecB, sizeof(double)*ELEMENT_COUNT);
    printf(" Malloc d_vecB: %s\n", cudaGetErrorString(R));
    R = cudaMalloc((void **)&d_vecC, sizeof(double)*ELEMENT_COUNT);
    printf(" Malloc d_vecC: %s\n", cudaGetErrorString(R));
    R = cudaMalloc((void **)&d_psum, sizeof(double)*THREADSPERBLOCK);
    printf(" Malloc d_vecC: %s\n", cudaGetErrorString(R));

    printf("========== Check Data Transfer ==========\n");
    R = cudaMemcpy(d_vecA, h_vecA, sizeof(double)*ELEMENT_COUNT, cudaMemcpyHostToDevice);
    printf(" Memory Copy d_vecA: %s\n", cudaGetErrorString(R));
    R = cudaMemcpy(d_vecB, h_vecB, sizeof(double)*ELEMENT_COUNT, cudaMemcpyHostToDevice);
    printf(" Memory Copy d_vecB: %s\n", cudaGetErrorString(R));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    vecMul_gpu_kernel<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(d_vecA, d_vecB, d_vecC);
    cudaThreadSynchronize();
    vecSum_gpu_kernel<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(d_vecC, d_psum);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("========== Check Result ===========\n");
    R = cudaMemcpy(h_psum, d_vecC, sizeof(double)*THREADSPERBLOCK, cudaMemcpyDeviceToHost);
    printf(" Memcpy h_psum: %s\n", cudaGetErrorString(R));

    for (i=0; i<BLOCKSPERGRID; i++)
        *S_gpu += h_psum[i];

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    vecDot_cpu(S_cpu, h_vecA, h_vecB);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float CPU_ET;
    cudaEventElapsedTime(&CPU_ET, start, stop);

    if (*S_cpu==*S_gpu)
        printf(" Result Check: OK!\n\n");
    else {
        printf(" Result Check: HORY SHET!\n");
        printf(" Result S_cpu = %lf, S_gpu = %lf\n\n", *S_cpu, *S_gpu);
    }

    free(h_vecA);
    free(h_vecB);
    free(h_psum);
    free(S_gpu);
    free(S_cpu);

    cudaFree(d_vecA);
    cudaFree(d_vecB);
    cudaFree(d_vecC);
    cudaFree(d_psum);

    printf("\n========== Execution Info. ===========\n");
    printf(" Execution Time on GPU: %3.20f s\n", elapsedTime/1000);
    printf(" Execution Time on CPU: %3.20f s\n", CPU_ET/1000);
    printf(" Speed up = %lf\n", (CPU_ET/elapsedTime));

    return 0;
}
