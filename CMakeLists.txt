cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_ARCH "-arch=sm_61" CACHE STRING "Value of the NVCC -arch option.")

project(SegmentationSGM)

add_subdirectory(src)
add_subdirectory(samples/movie)
#add_subdirectory(samples/movie_with_libsgm)
add_subdirectory(test)
