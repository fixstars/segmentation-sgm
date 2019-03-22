cmake_minimum_required(VERSION 3.1)

project(segmentation_sgm_sample_benchmark)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

include_directories(../../include)

# libSGM
find_package(LibSGM REQUIRED)
include_directories(${LIBSGM_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)

file(GLOB srcs ./*.cpp ./*.h*)
cuda_add_executable(segmentation_sgm_sample_benchmark ${srcs})
target_link_libraries(segmentation_sgm_sample_benchmark ${OpenCV_LIBS} ${LIBSGM_LIBRARY} segmentation_sgm)
