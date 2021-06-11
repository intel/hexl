# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Download and unpack gtest at configure time
configure_file(CMakeLists.txt.in gtest-download/CMakeLists.txt)
set(CONFIGURE_COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .)
execute_process(COMMAND ${CONFIGURE_COMMAND}
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gtest-download )
if(result)
  message(FATAL_ERROR "CMake step for gtest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gtest-download )
if(result)
  message(FATAL_ERROR "Build step for gtest failed: ${result}")
endif()

# Add gtest directly to our build. This defines the gtest target.
hexl_cache_variable(gtest_force_shared_crt)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/gtest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/gtest-build
                 EXCLUDE_FROM_ALL)

# Prevent variables from leaking
unset(gtest_force_shared_crt CACHE)
hexl_uncache_variable(gtest_force_shared_crt)
