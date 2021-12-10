#pragma once

void add();
__global__ void add(int *a, int *b, int *c, int N);
__global__ void cuda_hello();

void CudaSync();
void GetCudaProperties();
