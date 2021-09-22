# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Download and unpack easylogging at configure time
set(CACHED_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if (NOT HEXL_USE_MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-int-conversion -Wno-implicit-fallthrough")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion -Wno-range-loop-construct -Wno-shadow")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-int-float-conversion -Wno-conversion")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning -Wno-unknown-warning-option")
endif()

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

hexl_cache_variable(build_static_lib)
set(build_static_lib ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/easylogging-src
                 ${CMAKE_CURRENT_BINARY_DIR}/easylogging-build
                 EXCLUDE_FROM_ALL)

set(CMAKE_CXX_FLAGS "${CACHED_CMAKE_CXX_FLAGS}")

# Prevent variables from leaking
unset(build_static_lib CACHE)
hexl_uncache_variable(build_static_lib)

set(EASYLOGGINGPP_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/easylogging-src/src)
set_target_properties(easyloggingpp PROPERTIES INCLUDE_DIRECTORIES
    ${EASYLOGGINGPP_INCLUDE_DIR})

install(
  TARGETS easyloggingpp
  EXPORT HEXLTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${EASYLOGGINGPP_INCLUDE_DIR}/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h")
