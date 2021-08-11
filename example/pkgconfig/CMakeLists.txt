# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project(hexl_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_STANDARD 11)

# Example using pkg_check_modules
include(FindPkgConfig)
if(NOT PKG_CONFIG_FOUND)
  message(FATAL_ERROR "pkg-config not found!" )
endif()

pkg_check_modules(HEXL REQUIRED IMPORTED_TARGET hexl)

add_executable(example ../example.cpp)
target_link_libraries(example PkgConfig::HEXL)
