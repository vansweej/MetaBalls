cmake_minimum_required (VERSION 3.16)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_BUILD_TYPE Debug)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(isosurface-cuda)

include(CTest)

add_subdirectory(criterion)
add_subdirectory(benchmark)
add_subdirectory(googletest-main)
add_subdirectory(cuda-test)