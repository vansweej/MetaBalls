enable_language(CUDA)

project(cuda-test)

find_package(CUDAToolkit)

add_executable(${PROJECT_NAME} src/main.cpp
                               cu/main.cu
                               cu/test.cu
                               cu/cuda_error.cu)

target_include_directories(${PROJECT_NAME} PUBLIC cuh
                                                  inc)

target_link_libraries(${PROJECT_NAME} CUDA::cudart)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

#####################################################
## Unit tests definitions
#####################################################

# Per default, BUILD_TESTING is enabled by ctest
if(NOT BUILD_TESTING)
	return()
endif()

set(testTarget ${PROJECT_NAME}_test)
set(testSource tst/cuda_properties_iterator_test.cpp)

add_executable(${testTarget} ${testSource})

target_link_libraries(${testTarget} gtest_main
                                    CUDA::cudart)

target_include_directories(${testTarget} PUBLIC cuh
                                                inc)
                                                  
set_property(TARGET ${testTarget} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})


add_test (NAME ${testTarget}
          COMMAND ${testTarget})