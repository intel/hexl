# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Download and unpack gbenchmark at configure time
configure_file(CMakeLists.txt.in gbenchmark-download/CMakeLists.txt)
set(CONFIGURE_COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .)
execute_process(COMMAND ${CONFIGURE_COMMAND}
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gbenchmark-download )
if(result)
  message(FATAL_ERROR "CMake step for gbenchmark failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gbenchmark-download )
if(result)
  message(FATAL_ERROR "Build step for gbenchmark failed: ${result}")
endif()

# Add gbenchmark directly to our build. This defines the gbenchmark target.
# Add option to override option of same name in gbenchmark
option(BENCHMARK_ENABLE_TESTING "" ON)
# Turn off said option
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/gbenchmark-src
                 ${CMAKE_CURRENT_BINARY_DIR}/gbenchmark-build
                 EXCLUDE_FROM_ALL)
