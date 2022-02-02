#include <iostream>

#include "cudaError.cuh"
#include "main.cuh"
#include "test.cuh"

void cuda_main() {
  GetCudaProperties();

  // cuda_hello<<<1, 1>>>();
  add();
}
