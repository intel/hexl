# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)
project(HEXL_EXAMPLE LANGUAGES CXX)

add_executable(example example.cpp)

find_package(HEXL 1.2.3
             HINTS ${HEXL_HINT_DIR}
             REQUIRED)

target_link_libraries(example HEXL::hexl)
