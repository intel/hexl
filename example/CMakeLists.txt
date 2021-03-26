# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project(intel_hexl LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.5.1)

find_package(IntelHEXL 1.0.0
    HINTS ${INTEL_HEXL_HINT_DIR}
    REQUIRED)
if (NOT TARGET intel_hexl)
    FATAL_ERROR("TARGET intel_hexl not found")
endif()

add_executable(example example.cpp)
target_link_libraries(example PRIVATE intel_hexl)
