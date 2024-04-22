project(nv_metrics)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(${PROJECT_NAME} SHARED nv_metrics.cpp config.cpp eval.cpp
                                   parser.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver CUDA::cupti)
target_include_directories(${PROJECT_NAME}
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH
                                                 TRUE)
install(FILES nv_metrics.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
