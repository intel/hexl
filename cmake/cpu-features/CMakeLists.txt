# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Download and unpack cpu-features at configure time
configure_file(CMakeLists.txt.in cpu-features-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpu-features-download )
if(result)
  message(FATAL_ERROR "CMake step for cpu-features failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpu-features-download )
if(result)
  message(FATAL_ERROR "Build step for cpu-features failed: ${result}")
endif()

# Add cpu-features directly to our build. This defines the cpu_features target.
# Add option to override option of same name in cpu-features
option(BUILD_PIC "" OFF)
# Turn ON said option
set(BUILD_PIC ON)

option(BUILD_SHARED_LIBS "" OFF)
# Turn OFF said option
set(BUILD_SHARED_LIBS OFF)

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/cpu-features-src
                 ${CMAKE_CURRENT_BINARY_DIR}/cpu-features-build
                 EXCLUDE_FROM_ALL)
