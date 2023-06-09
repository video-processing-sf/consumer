cmake_minimum_required(VERSION 3.10)
project(test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

find_package(OpenCV REQUIRED)

add_executable(bin
	src/main.cpp
	src/image_processor.hpp
	src/shared_memory.hpp
)

target_include_directories(bin PRIVATE
	${GSTREAMER_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(bin
	${GSTREAMER_LIBRARIES}
	${OpenCV_LIBRARIES}
	pthread
	rt
)

