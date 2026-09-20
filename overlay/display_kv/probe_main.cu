// Standalone probe: create the pool, run GPU fill/verify + bandwidth, tear down.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
extern "C" {
typedef struct Pool Pool;
Pool *glm53_display_create(const char*, size_t);
void glm53_display_destroy(Pool*);
uint64_t glm53_display_pointer(Pool*);
const char *glm53_display_error(void);
}
__global__ void fill(uint32_t*p,size_t n){size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;for(;i<n;i+=gridDim.x*(size_t)blockDim.x)p[i]=(uint32_t)i*2654435761u;}
__global__ void verify(const uint32_t*p,size_t n,unsigned long long*bad){size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;for(;i<n;i+=gridDim.x*(size_t)blockDim.x)if(p[i]!=(uint32_t)i*2654435761u)atomicAdd(bad,1ull);}
__global__ void rd(const uint4*p,size_t n,uint4*sink){uint4 a={0,0,0,0};size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;for(;i<n;i+=gridDim.x*(size_t)blockDim.x){uint4 v=p[i];a.x^=v.x;a.y^=v.y;a.z^=v.z;a.w^=v.w;}if(a.x==0xdeadbeef)*sink=a;}
static float bw(void(*k)(void*,size_t),void*p,size_t bytes){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);k(p,bytes);cudaDeviceSynchronize();cudaEventRecord(a);for(int i=0;i<5;i++)k(p,bytes);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return bytes*5/ms/1e6;}
static void kread(void*p,size_t b){static uint4*s;if(!s)cudaMalloc(&s,16);rd<<<2048,256>>>((uint4*)p,b/16,s);}
static void kfill(void*p,size_t b){fill<<<2048,256>>>((uint32_t*)p,b/4);}
int main(int argc,char**argv){
    size_t bytes=(argc>1?strtoull(argv[1],0,10):1792)<<20;
    cudaFree(0); // primary context
    Pool*p=glm53_display_create(argc>2?argv[2]:"/dev/dri/card0",bytes);
    if(!p){fprintf(stderr,"create failed: %s\n",glm53_display_error());return 1;}
    void*d=(void*)glm53_display_pointer(p);
    printf("pool ptr=%p bytes=%zu\n",d,bytes);
    fill<<<2048,256>>>((uint32_t*)d,bytes/4);
    unsigned long long*bad;cudaMallocManaged(&bad,8);*bad=0;
    verify<<<2048,256>>>((uint32_t*)d,bytes/4,bad);cudaDeviceSynchronize();
    cudaError_t e=cudaGetLastError();
    printf("gpu fill+verify: %s, mismatches=%llu\n",cudaGetErrorString(e),*bad);
    printf("display read  BW: %.1f GB/s\n",bw(kread,d,bytes));
    printf("display write BW: %.1f GB/s\n",bw(kfill,d,bytes));
    void*ref;cudaMalloc(&ref,bytes);
    printf("cudaMalloc read  BW: %.1f GB/s\n",bw(kread,ref,bytes));
    printf("cudaMalloc write BW: %.1f GB/s\n",bw(kfill,ref,bytes));
    // memset / memcpy paths used by torch .zero_() / copy_
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);cudaMemsetAsync(d,0,bytes);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);printf("cudaMemset display: %.1f GB/s (%s)\n",bytes/ms/1e6,cudaGetErrorString(cudaGetLastError()));
    cudaEventRecord(a);cudaMemcpyAsync(d,ref,bytes,cudaMemcpyDeviceToDevice);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&ms,a,b);printf("cudaMemcpy D2D->display: %.1f GB/s (%s)\n",bytes/ms/1e6,cudaGetErrorString(cudaGetLastError()));
    cudaFree(ref);glm53_display_destroy(p);
    printf("status: %s\n",(*bad==0&&e==cudaSuccess)?"pass":"FAIL");return *bad?2:0;
}
