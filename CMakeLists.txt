cmake_minimum_required(VERSION 3.20)

project(cuda-metrics LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

get_filename_component(CUDA_cupti_LIBRARY_DIR "${CUDA_cupti_LIBRARY}" DIRECTORY)
set(CUDA_cupti_INCLUDE_DIR "${CUDA_cupti_LIBRARY_DIR}/../include")
target_include_directories(CUDA::cupti INTERFACE ${CUDA_cupti_INCLUDE_DIR})
find_library(CUDA_nvperf_target_LIBRARY libnvperf_target.so
             HINTS ${CUDA_cupti_LIBRARY_DIR} REQUIRED)
find_library(CUDA_nvperf_host_LIBRARY libnvperf_host.so
             HINTS ${CUDA_cupti_LIBRARY_DIR} REQUIRED)
target_link_libraries(CUDA::cupti INTERFACE ${CUDA_nvperf_host_LIBRARY})
target_link_libraries(CUDA::cupti INTERFACE ${CUDA_nvperf_target_LIBRARY})

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME} CUDA::cupti)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
