#pragma once

void CudaFetchError();

#define CUDA_ERROR(func) (func); CudaFetchError()