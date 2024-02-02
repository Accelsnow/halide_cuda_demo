cmake_minimum_required(VERSION 3.26)
project(demo CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pedantic -ansi")

# Assuming the paths are environment variables
include_directories(/home/snowsr/projects/csc2231/halide1600_prebuilt/include)
include_directories(/home/snowsr/projects/csc2231/halide_proj)
include_directories(/home/snowsr/projects/csc2231/halide1600_prebuilt/share/Halide/tools)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

link_directories(/home/snowsr/projects/csc2231/halide1600_prebuilt/lib)

add_executable(demo demo.cu)

target_link_libraries(demo cuda Halide ${PNG_LIBRARY} ${JPEG_LIBRARY} ${CMAKE_DL_LIBS} /home/snowsr/projects/csc2231/halide_proj/lodepng.cpp)

set_target_properties(demo PROPERTIES CUDA_ARCHITECTURES "89")

set_target_properties(demo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
