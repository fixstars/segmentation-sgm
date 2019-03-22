cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_ARCH "-arch=sm_61" CACHE STRING "Value of the NVCC -arch option.")

option(ENABLE_SAMPLES "Build samples" ON)
option(ENABLE_TESTS   "Test library" OFF)
option(WITH_LIBSGM    "Build sample with libSGM (libSGM needs to be installed)" OFF)

project(SegmentationSGM)

add_subdirectory(src)

if(ENABLE_SAMPLES)
	add_subdirectory(samples/movie)
	if(WITH_LIBSGM)
		add_subdirectory(samples/movie_with_libsgm)
		add_subdirectory(samples/benchmark)
	endif(WITH_LIBSGM)
endif(ENABLE_SAMPLES)

if(ENABLE_TESTS)
	add_subdirectory(test)
endif(ENABLE_TESTS)
