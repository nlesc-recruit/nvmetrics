project(nv_metrics)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(${PROJECT_NAME} SHARED nv_metrics.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver CUDA::cupti)
target_include_directories(${PROJECT_NAME}
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ${PROJECT_NAME})
install(FILES nv_metrics.h DESTINATION include)
