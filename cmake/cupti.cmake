# cmake-format: off
# ##############################################################################
# cupti.cmake - CMake helper script for CUPTI
#
# This CMake script is intended to be included in CUDA projects to facilitate
# integration with CUPTI (CUDA Profiling Tools Interface), nvperf_target, and
# nvperf_host libraries.
#
# Usage:
# 1. Ensure that the CUDA Toolkit is installed and can be located by CMake.
# 2. This script assumes that CUPTI and the required nvperf libraries are
#    available in the system paths or can be located via CMake's find_library
#    mechanism.
# 3. Once included, this script configures include directories and links the
#    CUPTI target with the nvperf_host and nvperf_target libraries.
#
# Example Usage:
# ~~~
# find_package(CUDAToolkit REQUIRED)
# include(cupti.cmake)
# target_link_libraries(${PROJECT_NAME} CUDA::cupti)
# ~~~
#
# ##############################################################################
# cmake-format: on

get_filename_component(CUDA_cupti_LIBRARY_DIR "${CUDA_cupti_LIBRARY}" DIRECTORY)
set(CUDA_cupti_INCLUDE_DIR "${CUDA_cupti_LIBRARY_DIR}/../include")
target_include_directories(CUDA::cupti INTERFACE ${CUDA_cupti_INCLUDE_DIR})
find_library(CUDA_nvperf_target_LIBRARY libnvperf_target.so
             HINTS ${CUDA_cupti_LIBRARY_DIR} REQUIRED)
find_library(CUDA_nvperf_host_LIBRARY libnvperf_host.so
             HINTS ${CUDA_cupti_LIBRARY_DIR} REQUIRED)
target_link_libraries(CUDA::cupti INTERFACE ${CUDA_nvperf_host_LIBRARY})
target_link_libraries(CUDA::cupti INTERFACE ${CUDA_nvperf_target_LIBRARY})
