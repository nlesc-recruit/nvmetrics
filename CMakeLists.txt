cmake_minimum_required(VERSION 3.20)

project(cuda-metrics LANGUAGES CUDA CXX)

option(NVMETRICS_BUILD_PYTHON "Build Python interface" ON)

find_package(CUDAToolkit REQUIRED)
include(cmake/cupti.cmake)

add_subdirectory(lib)
if(${NVMETRICS_BUILD_PYTHON})
  add_subdirectory(python)
endif()
add_subdirectory(demo)
