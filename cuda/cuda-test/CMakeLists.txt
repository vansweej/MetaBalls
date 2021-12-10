cmake_minimum_required (VERSION 3.16)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_BUILD_TYPE Debug)

project(cuda-test)

add_executable(${PROJECT_NAME} src/main.cpp
                               cu/main.cu
                               cu/test.cu
                               cu/cudaError.cu)

target_link_libraries(${PROJECT_NAME} gtest_main)

target_include_directories(${PROJECT_NAME} PUBLIC cuh
                                                  inc)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)

#####################################################
## Unit tests definitions
#####################################################

# Per default, BUILD_TESTING is enabled by ctest
if(NOT BUILD_TESTING)
	return()
endif()


set(testTarget ${PROJECT_NAME}_test)
set(testSource tst/test.cpp)

add_executable(${testTarget} ${testSource})

target_link_libraries(${testTarget} gtest_main)

add_test (NAME ${testTarget}
          COMMAND ${testTarget})
