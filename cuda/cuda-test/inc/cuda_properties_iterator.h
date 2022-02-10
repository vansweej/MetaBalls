#pragma once

#include "cuda_runtime.h"
#include <algorithm>

class m_CudaProperties {
 public:
  m_CudaProperties() { cudaGetDeviceCount(&deviceCount); }
  struct CudaDevicePropertiesIterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = int;
    using value_type = cudaDeviceProp;
    using pointer = cudaDeviceProp *;    // or also value_type*
    using reference = cudaDeviceProp &;  // or also value_type&

    friend class m_CudaProperties;

   public:
    reference operator*() { return prop; }

    pointer operator->() { return &prop; }

    CudaDevicePropertiesIterator &operator++() {
      currentIdx = std::min(currentIdx + 1, deviceCount);
      if (currentIdx < deviceCount) {
        cudaGetDeviceProperties(&prop, currentIdx);
      } else {
        prop = {};
      }
      return *this;
    }
    CudaDevicePropertiesIterator operator++(int) {
      CudaDevicePropertiesIterator tmp = *this;
      ++(*this);
      return tmp;
    }
    CudaDevicePropertiesIterator &operator--() {
      currentIdx = std::max(currentIdx - 1, 0);
      if (currentIdx > 0) {
        cudaGetDeviceProperties(&prop, currentIdx);
      }
      return *this;
    }
    CudaDevicePropertiesIterator operator--(int) {
      CudaDevicePropertiesIterator tmp = *this;
      --(*this);
      return tmp;
    }
    bool operator==(const CudaDevicePropertiesIterator &other) {
      return (currentIdx == other.currentIdx);
    }
    bool operator!=(const CudaDevicePropertiesIterator &other) {
      return (currentIdx != other.currentIdx);
    }

   private:
    CudaDevicePropertiesIterator(int idx) : currentIdx(idx) {
      cudaGetDeviceCount(&deviceCount);
      cudaGetDeviceProperties(&prop, currentIdx);
    }
    int currentIdx;
    int deviceCount;
    value_type prop;
  };

  CudaDevicePropertiesIterator begin() const {
    return CudaDevicePropertiesIterator(0);
  }
  CudaDevicePropertiesIterator end() const {
    return CudaDevicePropertiesIterator(deviceCount);
  }

  int deviceCount;
};