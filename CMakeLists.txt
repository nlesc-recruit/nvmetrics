cmake_minimum_required(VERSION 3.20)

project(cuda-metrics LANGUAGES CUDA CXX)

option(NVMETRICS_BUILD_PYTHON "Build Python interface" ON)
option(NVMETRICS_BUILD_DEMO "Build with demo" ON)

find_package(CUDAToolkit REQUIRED)
include(cmake/cupti.cmake)

if(${NVMETRICS_BUILD_PYTHON})
  find_package(
    Python3
    COMPONENTS Interpreter Development
    REQUIRED)
endif()

include(GNUInstallDirs)

configure_file(cmake/environment-module.in
               "${CMAKE_BINARY_DIR}/modulefiles/nvmetrics")
install(FILES "${PROJECT_BINARY_DIR}/modulefiles/nvmetrics"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/modulefiles")

add_subdirectory(lib)
if(${NVMETRICS_BUILD_PYTHON})
  add_subdirectory(python)
endif()
if(${NVMETRICS_BUILD_DEMO})
  add_subdirectory(demo)
endif()
