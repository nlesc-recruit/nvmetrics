cmake_minimum_required(VERSION 3.20)

project(cuda-metrics LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

get_filename_component(CUDA_cupti_INCLUDE_DIR "${CUDA_cupti_LIBRARY}" DIRECTORY)
set(CUDA_cupti_INCLUDE_DIR "${CUDA_cupti_INCLUDE_DIR}/../include")
target_include_directories(CUDA::cupti INTERFACE ${CUDA_cupti_INCLUDE_DIR})

add_library(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME} CUDA::cupti)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
