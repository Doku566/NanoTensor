cmake_minimum_required(VERSION 3.14)
project(NanoTensor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization: We want to see how fast we can go without external libs
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

include_directories(include)

add_library(nanotensor_core
    src/ops.cpp
    src/quantization.cpp
)

add_executable(demo_transformer src/main.cpp)
target_link_libraries(demo_transformer PRIVATE nanotensor_core)
