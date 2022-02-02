#include <iostream>

#include "cudaError.cuh"

void CudaFetchError() {
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    std::cout << "kernel launch failed with error "
              << cudaGetErrorString(cudaerr) << std::endl;
}