# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project(hexl_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)

# Example using pkg_check_modules
include(FindPkgConfig)
if(NOT PKG_CONFIG_FOUND)
  message(FATAL_ERROR "pkg-config not found!" )
endif()

pkg_check_modules(HEXL REQUIRED IMPORTED_TARGET hexl)

# Before CMake 3.15, FindPkgConfig doesn't populate INTERFACE_LINK_OPTIONS property
# of imported target with non-library linker flags, e.g. -fsanitize=address
# As a work-around, set address sanitizer flags manually.
get_target_property(HEXL_LINK_LIBRARIES PkgConfig::HEXL INTERFACE_LINK_LIBRARIES)
if(${CMAKE_VERSION} VERSION_LESS "3.15.0" AND ${HEXL_LINK_LIBRARIES} MATCHES "debug")
  set_target_properties(PkgConfig::HEXL PROPERTIES INTERFACE_LINK_OPTIONS "-fsanitize=address")
  set_target_properties(PkgConfig::HEXL PROPERTIES INTERFACE_COMPILE_OPTIONS "-fsanitize=address")
endif()

add_executable(example ../example.cpp)
target_link_libraries(example PkgConfig::HEXL)
