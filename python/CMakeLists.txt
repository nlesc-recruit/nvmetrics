project(python)

find_package(pybind11 CONFIG REQUIRED)

add_library(nvmetrics MODULE pynvmetrics.cpp)

target_link_libraries(nvmetrics PUBLIC pybind11::module pybind11::lto
                                       nv_metrics)

pybind11_extension(nvmetrics)

set_target_properties(nvmetrics PROPERTIES INSTALL_RPATH
                                           "${CMAKE_INSTALL_PREFIX}/lib")

install(
  TARGETS nvmetrics
  LIBRARY
    DESTINATION
      "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
)
