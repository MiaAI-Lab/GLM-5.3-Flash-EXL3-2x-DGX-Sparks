#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <stdio.h>
__global__ void rd(const uint4*p,size_t n,uint4*sink){uint4 a={0,0,0,0};size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;for(;i<n;i+=gridDim.x*(size_t)blockDim.x){uint4 v=p[i];a.x^=v.x;a.y^=v.y;a.z^=v.z;a.w^=v.w;}if(a.x==0xdeadbeef)*sink=a;}
__global__ void wr(uint32_t*p,size_t n){size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;for(;i<n;i+=gridDim.x*(size_t)blockDim.x)p[i]=i;}
int main(){size_t b=1024ul<<20;cudaFree(0);uint4*s;cudaMalloc(&s,16);
 void*h=mmap(0,b,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 CUresult r=cuMemHostRegister(h,b,CU_MEMHOSTREGISTER_DEVICEMAP);CUdeviceptr d;cuMemHostGetDevicePointer(&d,h,0);
 printf("register rc=%d host=%p dev=%p\n",r,h,(void*)d);
 cudaEvent_t a,e;cudaEventCreate(&a);cudaEventCreate(&e);float ms;
 wr<<<2048,256>>>((uint32_t*)d,b/4);cudaDeviceSynchronize();
 cudaEventRecord(a);for(int i=0;i<5;i++)rd<<<2048,256>>>((uint4*)d,b/16,s);cudaEventRecord(e);cudaEventSynchronize(e);cudaEventElapsedTime(&ms,a,e);printf("pinned-host read  BW: %.1f GB/s\n",b*5/ms/1e6);
 cudaEventRecord(a);for(int i=0;i<5;i++)wr<<<2048,256>>>((uint32_t*)d,b/4);cudaEventRecord(e);cudaEventSynchronize(e);cudaEventElapsedTime(&ms,a,e);printf("pinned-host write BW: %.1f GB/s\n",b*5/ms/1e6);
 void*m;cudaMallocManaged(&m,b);wr<<<2048,256>>>((uint32_t*)m,b/4);cudaDeviceSynchronize();
 cudaEventRecord(a);for(int i=0;i<5;i++)rd<<<2048,256>>>((uint4*)m,b/16,s);cudaEventRecord(e);cudaEventSynchronize(e);cudaEventElapsedTime(&ms,a,e);printf("managed read BW: %.1f GB/s\n",b*5/ms/1e6);
 return 0;}
