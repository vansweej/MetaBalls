#include <gtest/gtest.h>
#include "cuda_properties_iterator.h"

TEST(CudaPropertiesIterator_test, CudaPropertiesIterator_test) {

  m_CudaProperties cudaProperties;
  for (auto prop : cudaProperties) {
    ASSERT_TRUE(prop.name);
  }
}