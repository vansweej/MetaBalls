#include <driver_types.h>

#include <iostream>

#include "test.cuh"
#include "cudaError.cuh"

__global__ void cuda_hello() { printf("Hello World from GPU!\n"); }

void GetCudaProperties() {
  int count;
  cudaGetDeviceCount(&count);
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << prop.name << std::endl;
  }
}

void add() {
  const unsigned int N = 10;
  int a[N], b[N], result[N];
  int *dev_a, *dev_b, *dev_result;

  CUDA_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
  CUDA_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
  CUDA_ERROR(cudaMalloc((void **)&dev_result, N * sizeof(int)));

  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i * i;
  }

  CUDA_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  add<<<N, 1>>>(dev_a, dev_b, dev_result, N);

  CUDA_ERROR(
      cudaMemcpy(result, dev_result, N * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    std::cout << result[i] << std::endl;
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_result);
}

__global__ void add(int *dev_a, int *dev_b, int *dev_result, int N) {
  int tid = blockIdx.x;
  // printf("N = %d | tid = %d\n", N, tid);
  if (tid < N) {
    dev_result[tid] = dev_a[tid] - dev_b[tid];
  }
}