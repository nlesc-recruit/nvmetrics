project(demo)

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME} nv_metrics)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
