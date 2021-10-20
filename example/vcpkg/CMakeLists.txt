# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project(hexl_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

# Example using vcpkg
find_package(HEXL CONFIG REQUIRED)
add_executable(example ../example.cpp)
target_link_libraries(example PRIVATE HEXL::hexl)
