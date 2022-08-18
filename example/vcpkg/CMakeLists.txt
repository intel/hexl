# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project(hexl_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

# Setting Release as default mode. Otherwise vcpkg uses debug mode by default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# This flag is needed to build HEXL on Debug mode
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "-fsanitize=address")
endif()

# Example using vcpkg
find_package(HEXL CONFIG REQUIRED)
add_executable(example ../example.cpp)
target_link_libraries(example PRIVATE HEXL::hexl)
