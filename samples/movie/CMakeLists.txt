cmake_minimum_required(VERSION 2.8)

project(segsgm_movie)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

include_directories(../../cpu_segsgm)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB srcs ./*.cpp ./*.h*)
add_executable(segsgm_movie ${srcs})
target_link_libraries(segsgm_movie ${OpenCV_LIBS} cpu_segsgm)
