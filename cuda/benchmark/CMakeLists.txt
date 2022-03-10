project(benchmark)

if (NOT DEFINED GPP_COMPILER)
  set(GPP_COMPILER /usr/bin/g++)
endif()

message(STATUS "CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_COMPILER ${GPP_COMPILER})

set(criterionTarget ${PROJECT_NAME}_benchmark)
set(criterionSource src/criterion_test.cpp)

add_executable(${criterionTarget} ${criterionSource})

target_link_libraries(${criterionTarget} PUBLIC Criterion)