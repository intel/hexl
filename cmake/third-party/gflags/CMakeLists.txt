# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Download and unpack gflags at configure time
set(CACHED_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if (NOT HEXL_USE_MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion -Wno-shadow -Wno-cast-function-type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-fallthrough")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning -Wno-unknown-warning-option")
endif()

configure_file(CMakeLists.txt.in gflags-download/CMakeLists.txt)
set(CONFIGURE_COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .)
execute_process(COMMAND ${CONFIGURE_COMMAND}
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gflags-download )
if(result)
  message(FATAL_ERROR "CMake step for gflags failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gflags-download )
if(result)
  message(FATAL_ERROR "Build step for gflags failed: ${result}")
endif()

# Add gflags directly to our build. This defines the gflags target.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/gflags-src
                 ${CMAKE_CURRENT_BINARY_DIR}/gflags-build
                 EXCLUDE_FROM_ALL)

set(CMAKE_CXX_FLAGS "${CACHED_CMAKE_CXX_FLAGS}")
