#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>

#include "cuda_error.cuh"
#include "test.cuh"
#include "cuda_properties_iterator.h"


int maxGridSize[3];
int maxThreadPerBlock;

__global__ void cuda_hello() { printf("Hello World from GPU!\n"); }

void GetCudaProperties() {
  CudaProperties cudaProperties;
  for (auto prop : cudaProperties) {
    std::cout << prop.name << std::endl;
  }
}

void add() {
  const unsigned int N = 101;
  int a[N], b[N], result[N], result_2[N];
  int *dev_a, *dev_b, *dev_result;

  CUDA_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
  CUDA_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
  CUDA_ERROR(cudaMalloc((void **)&dev_result, N * sizeof(int)));

  std::for_each(std::begin(a), std::end(a), [](int &v) {
    static int i = 1;
    v = i;
    i++;
  });
  std::for_each(std::begin(b), std::end(b), [](int &v) {
    static int i = 1;
    v = i * i;
    i++;
  });

  std::transform(std::begin(a), std::end(a), b, result_2, std::minus<>{});

  CUDA_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  int threads =
      std::max(1, std::min(maxThreadPerBlock, int(std::ceil(sqrt(N)))));
  int blocks = std::max(1, int(std::ceil(N / (float)threads)));

  add<<<blocks, threads>>>(dev_a, dev_b, dev_result, N);

  CUDA_ERROR(
      cudaMemcpy(result, dev_result, N * sizeof(int), cudaMemcpyDeviceToHost));

  std::for_each(std::begin(result), std::end(result),
                [&result, &result_2](const int &r) {
                  static int c = 0;
                  assert(result[c] == result_2[c]);
                  std::cout << r << std::endl;
                  c++;
                });

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_result);
}

__global__ void add(int *dev_a, int *dev_b, int *dev_result, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("N = %d | tid = %d\n", N, tid);
  while (tid < N) {
    dev_result[tid] = dev_a[tid] - dev_b[tid];
    tid += blockDim.x * gridDim.x;
  }
}