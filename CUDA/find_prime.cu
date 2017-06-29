#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#define ELEMENT_COUNT 1024*1024
#define BLOCKSPERGRID 1024
#define THREADSPERBLOCK 1024

__device__ __host__ int isPrime(unsigned int n) {
    unsigned int N = sqrt((float)n);
    int i;
    for(i=2; i<=N; i++)
        if(n % i == 0)
            break;  

    return (i>N)?1:0;
}

__global__ void findPrime(unsigned int start,unsigned int N, unsigned int *pt) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) {
        if(isPrime(i+start))
            pt[i]=1;
        else 
            pt[i]=0;
    }
}

int main(int argc, char **argv) {
    if(argc != 3){
        printf("usage: <Integer start from> <Integer end>\n");
        return -1;
    }
    cudaSetDevice(0);

    unsigned int start = atoi(argv[1]);
    unsigned int end   = atoi(argv[2]);
    unsigned int N     = end - start + 1;

    unsigned int *ptable = (unsigned int*)malloc(sizeof(unsigned int)*N);
    memset(ptable,0,sizeof(ptable));

    cudaError_t R;
    unsigned int *pcudatable; 
    R = cudaMalloc((void**)&pcudatable,sizeof(unsigned int)*N);
    printf("Malloc pcudatable : %s\n",cudaGetErrorString(R));

    cudaEvent_t tstart,tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    cudaEventRecord(tstart,0);

    findPrime<<<N/1024,1024>>>(start,N,pcudatable);

    cudaEventRecord(tstop,0);
    cudaEventSynchronize(tstop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,tstart,tstop);

    R = cudaMemcpy(ptable,pcudatable,sizeof(unsigned int)*N,cudaMemcpyDeviceToHost);

    unsigned int i;	
    for(i=0; i<=40; i++)
        if(ptable[i])
            printf("Find prime: %d\n", start+i);
        
    printf("execute time: %lfs\n", elapsedTime/1000);	
    return 0;
}
