cmake_minimum_required(VERSION 3.1)

project(segmentation_sgm_sample_movie)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

include_directories(../../include)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB srcs ./*.cpp ./*.h*)
add_executable(segmentation_sgm_sample_movie ${srcs})
target_link_libraries(segmentation_sgm_sample_movie ${OpenCV_LIBS} segmentation_sgm)
