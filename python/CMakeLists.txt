project(python)

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

add_library(nvmetrics MODULE pynvmetrics.cpp)

target_link_libraries(nvmetrics PUBLIC pybind11::module pybind11::lto
                                       nv_metrics)

pybind11_extension(nvmetrics)

set_target_properties(nvmetrics PROPERTIES INSTALL_RPATH "lib"
                                           BUILD_WITH_INSTALL_RPATH TRUE)

install(TARGETS nvmetrics DESTINATION python)
