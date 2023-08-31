# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project(hexl_omp_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)

# Define the directory containing HEXLConfig.cmake
set(HEXL_HINT_DIR "/home/eidf018/eidf018/s1820742psd/hexl/hexl_v2")

# Example using source
find_package(HEXL 1.2.5
    HINTS ${HEXL_HINT_DIR}
    REQUIRED)
if (NOT TARGET HEXL::hexl)
    message(FATAL_ERROR "TARGET HEXL::hexl not found")
endif()

find_package(OpenMP)
if (OpenMP_FOUND)
    message(STATUS "OpenMP_CXX_INCLUDE_DIRS: ${OpenMP_CXX_INCLUDE_DIRS}")
    message(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
endif()

# Add the directory containing libeasyloggingpp.a to the link directories
link_directories(/home/eidf018/eidf018/s1820742psd/hexl/hexl_v2/lib)


add_executable(omp_example ../omp_example.cpp)
add_executable(function_test ../test_example.cpp)
target_link_libraries(omp_example PRIVATE HEXL::hexl)
target_link_libraries(function_test PRIVATE HEXL::hexl)


