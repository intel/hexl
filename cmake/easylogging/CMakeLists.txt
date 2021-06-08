# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Download and unpack easylogging at configure time
configure_file(CMakeLists.txt.in easylogging-download/CMakeLists.txt)
set(CONFIGURE_COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .)
execute_process(COMMAND ${CONFIGURE_COMMAND}
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/easylogging-download )
if(result)
  message(FATAL_ERROR "CMake step for easylogging failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/easylogging-download )
if(result)
  message(FATAL_ERROR "Build step for easylogging failed: ${result}")
endif()

# Add easylogging directly to our build. This defines the easylogging target.
# Cache needed to override option in easylogging

cache_variable(build_static_lib)
set(build_static_lib ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/easylogging-src
                 ${CMAKE_CURRENT_BINARY_DIR}/easylogging-build
                 EXCLUDE_FROM_ALL)

# Prevent variables from leaking
unset(build_static_lib CACHE)
uncache_variable(build_static_lib)

set_target_properties(easyloggingpp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_BINARY_DIR}/easylogging-src/src)

install(
  TARGETS easyloggingpp
  EXPORT HEXLTargets)
