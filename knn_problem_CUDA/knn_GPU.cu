#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define INF 0x7FFFFFFF

typedef struct {
    int id, dis;
}matrix;

int N = 0, M = 0, K = 0;

__device__ int distance(int a, int b) {
    return (a-b)*(a-b);
}

__global__ void kernel_dis (int *inarr, matrix *matdis, int n, int m) {
    int tid = threadIdx.z*blockDim.y*blockDim.x +threadIdx.y*blockDim.x + threadIdx.x;
    int threadsperblock = blockDim.x*blockDim.y*blockDim.z;
    int x = blockIdx.z*threadsperblock + tid;
    int i = blockIdx.x;
    int j = blockIdx.y;
    int l;
    __shared__ int dis[1024];
    //	__shared__ int sum;

    if (i<j) {
        if (x<n)
            dis[tid] = distance(inarr[i*n+x], inarr[j*n+x]);
        else
            dis[tid] = 0;
        __syncthreads();

        for (l=threadsperblock/2; l>0; l/=2) {
            if (tid<l)
                dis[tid] += dis[tid+l];
            __syncthreads();
        }

        if (tid==0) {
            matdis[i*m+j].dis += dis[0];
            matdis[j*m+i].dis = matdis[i*m+j].dis;
        }
    }

}

int main (int argc, char **argv) {
    if (argc!=2) {
        printf("usage: ./a.out <testcase>.\n");
        return -1;
    }
    cudaSetDevice(0);

    int i, j, k;
    FILE *tc, *out, *cudaerr;
    tc = fopen(argv[1], "r");
    out = fopen("output_knn_GPU.txt", "w");
    cudaerr = fopen("cudaerr.txt", "w");
    fscanf(tc, "%d %d %d", &M, &N, &K);

    int *inmatrix1d = (int*)malloc(sizeof(int)*M*N);
    matrix *matdis = (matrix*)malloc(sizeof(matrix)*M*M);
    for (i=0; i<M; i++)
        for (j=0; j<M; j++) {
            matdis[i*M+j].id = j;
            matdis[i*M+j].dis = 0.0;
        }

    for (i=0; i<M; i++)
        for (j=0; j<N; j++)
            fscanf(tc, "%d", &inmatrix1d[i*N+j]);

    fclose(tc);

    // cuda init 
    int *d_inmat;
    matrix *d_matdis;
    cudaError_t R;
    R = cudaMalloc((void **)&d_inmat, sizeof(int)*M*N);
    fprintf(cudaerr, "Malloc d_inmat1d: %s\n", cudaGetErrorString(R));
    R = cudaMalloc((void **)&d_matdis, sizeof(matrix)*M*M);
    fprintf(cudaerr, "Malloc d_matdis:  %s\n", cudaGetErrorString(R));
    R = cudaMemcpy(d_inmat, inmatrix1d, sizeof(int)*M*N, cudaMemcpyHostToDevice);
    fprintf(cudaerr, "Memcpy d_inmat1d: %s\n", cudaGetErrorString(R));
    R = cudaMemcpy(d_matdis, matdis, sizeof(matrix)*M*M, cudaMemcpyHostToDevice);
    fprintf(cudaerr, "Memcpy d_matdis:  %s\n", cudaGetErrorString(R));

    dim3 grids(M, M, N/1024+1);
    dim3 blocks(16, 16, 4);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel_dis<<<grids, blocks>>>(d_inmat, d_matdis, N, M);
    cudaThreadSynchronize();

    R = cudaMemcpy(matdis, d_matdis, sizeof(matrix)*M*M, cudaMemcpyDeviceToHost);
    fprintf(cudaerr, "Memcpy d_matdis:  %s\n", cudaGetErrorString(R));

    int **rst = (int**)malloc(sizeof(int*)*M);
    for (i=0; i<M; i++)
        rst[i] = (int*)malloc(sizeof(int)*K);

    int MIN = INF, index = 0;
    for (i=0; i<M; i++)
        for (k=0; k<K; k++) {
            for (j=0; j<M; j++)
                if (i!=j)
                    if (matdis[i*M+j].dis<MIN) {
                        MIN = matdis[i*M+j].dis;
                        index = i*M+j;
                        rst[i][k] = index%M;
                    }
            MIN = INF;
            matdis[index].dis = INF;
        }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);


    for (i=0; i<M; i++) {
        for (j=0; j<K; j++)
            printf("%d ", rst[i][j]);
        printf("\n");
    }

    if (M==16384)
        printf("LARGE:%f\n", elapsedTime/1000);
    else if (M==4096)
        printf("MIDDLE:%f\n", elapsedTime/1000);
    else if (M==1024)
        printf("SMALL:%f\n", elapsedTime/1000);
    else
        printf("EXECUTION_TIME:%f\n", elapsedTime/1000);

    fclose(cudaerr);	
    fclose(out);
    return 0;
}
