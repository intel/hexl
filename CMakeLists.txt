# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# target_link_options and cmake -S . -B build require cmake 3.13
cmake_minimum_required(VERSION 3.13)
project(HEXL VERSION 1.2.3 LANGUAGES C CXX)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CMakePackageConfigHelpers)

set(HEXL_CMAKE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/hexl")
list(APPEND CMAKE_MODULE_PATH "${HEXL_CMAKE_PATH}")
include(hexl-util)

if(CMAKE_BUILD_TYPE)
  set(RELEASE_TYPES
      Debug
      Release
      RelWithDebInfo
      MinSizeRel)
  list(FIND RELEASE_TYPES ${CMAKE_BUILD_TYPE} INDEX_FOUND)
  if(${INDEX_FOUND} EQUAL -1)
    message(
      FATAL_ERROR
        "CMAKE_BUILD_TYPE must be one of Debug, Release, RelWithDebInfo, or MinSizeRel"
      )
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(HEXL_DEBUG ON)
else()
  set(HEXL_DEBUG OFF)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Create compilation database compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_INSTALL_RPATH "\$ORIGIN")

#------------------------------------------------------------------------------
# Compiler options...
#------------------------------------------------------------------------------

option(HEXL_BENCHMARK "Enable benchmarking" ON)
option(HEXL_COVERAGE "Enables coverage for unit tests" OFF)
option(HEXL_DOCS "Enable documentation building" OFF)
option(HEXL_EXPERIMENTAL "Enable experimental features" OFF)
option(HEXL_SHARED_LIB "Generate a shared library" OFF)
option(HEXL_TESTING "Enables unit-tests" ON)
option(HEXL_TREAT_WARNING_AS_ERROR "Treat all compile-time warnings as errors" OFF)

message(STATUS "CMAKE_BUILD_TYPE:              ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_C_COMPILER:              ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER:            ${CMAKE_CXX_COMPILER}")
message(STATUS "HEXL_BENCHMARK:                ${HEXL_BENCHMARK}")
message(STATUS "HEXL_COVERAGE:                 ${HEXL_COVERAGE}")
message(STATUS "HEXL_DEBUG:                    ${HEXL_DEBUG}")
message(STATUS "HEXL_DOCS:                     ${HEXL_DOCS}")
message(STATUS "HEXL_EXPERIMENTAL:             ${HEXL_EXPERIMENTAL}")
message(STATUS "HEXL_SHARED_LIB:               ${HEXL_SHARED_LIB}")
message(STATUS "HEXL_TESTING:                  ${HEXL_TESTING}")
message(STATUS "HEXL_TREAT_WARNING_AS_ERROR:   ${HEXL_TREAT_WARNING_AS_ERROR}")

hexl_check_compiler_version()
hexl_add_compiler_definition()

if (HEXL_COVERAGE)
  if (NOT HEXL_USE_GNU)
    message(FATAL_ERROR "HEXL_COVERAGE only supported on GCC.")
  endif()
  if (NOT HEXL_TESTING)
    message(FATAL_ERROR "HEXL_COVERAGE enabled, but HEXL_TESTING not enabled. Enable HEXL_TESTING for best coverage.")
  endif()
  add_compile_options(--coverage -O0 -g)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_compile_options(-fkeep-inline-functions)
  add_link_options(--coverage)
  # link_libraries(gcov)
endif()

if (HEXL_TREAT_WARNING_AS_ERROR)
  add_compile_options(-Werror)
endif()

set(HEXL_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(HEXL_SRC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hexl)
set(HEXL_INC_ROOT_DIR ${HEXL_SRC_ROOT_DIR}/include) # Public headers

message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_MINSIZEREL ${CMAKE_CXX_FLAGS_MINSIZEREL}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

#------------------------------------------------------------------------------
# Set AVX flags
#------------------------------------------------------------------------------
hexl_check_compile_flag("${HEXL_CMAKE_PATH}/test-avx512dq.cpp" HEXL_HAS_AVX512DQ)
hexl_check_compile_flag("${HEXL_CMAKE_PATH}/test-avx512ifma.cpp" HEXL_HAS_AVX512IFMA)
hexl_check_compile_flag("${HEXL_CMAKE_PATH}/test-avx512vbmi2.cpp" HEXL_HAS_AVX512VBMI2)
hexl_check_compile_flag("${HEXL_CMAKE_PATH}/test-avx256.cpp" HEXL_HAS_AVX256)

# ------------------------------------------------------------------------------
# Installation logic...
# ------------------------------------------------------------------------------

if(NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(LINUX)
  include(GNUInstallDirs)
else()
  set(CMAKE_INSTALL_BINDIR "bin")
  set(CMAKE_INSTALL_INCLUDEDIR "include")
  set(CMAKE_INSTALL_DOCDIR "doc")
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_INSTALL_LIBDIR})

message(STATUS "CMAKE_INSTALL_LIBDIR:     ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_INSTALL_PREFIX:     ${CMAKE_INSTALL_PREFIX}")

#------------------------------------------------------------------------------
# Third-party code...
#------------------------------------------------------------------------------
find_package(CpuFeatures CONFIG)
if (NOT CpuFeatures_FOUND)
  message(STATUS "CpuFeatures: pre-installed CpuFeatures not found")
  add_subdirectory(cmake/third-party/cpu-features)
else()
  message(STATUS "CpuFeatures: found")
  add_library(cpu_features ALIAS CpuFeatures::cpu_features)
  get_target_property(
    CpuFeatures_INCLUDE_DIR
    CpuFeatures::cpu_features
    INTERFACE_INCLUDE_DIRECTORIES)
endif()

if (HEXL_TESTING OR HEXL_BENCHMARK OR HEXL_DEBUG)
  if(NOT TARGET Threads::Threads)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
  endif()
  find_package(Threads REQUIRED)
endif()

if (HEXL_TESTING)
  add_subdirectory(cmake/third-party/gtest)
endif()

if (HEXL_BENCHMARK)
  add_subdirectory(cmake/third-party/gbenchmark)
endif()

if (HEXL_DEBUG)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/third-party/easylogging")
  find_package(EASYLOGGINGPP MODULE)
  if (EASYLOGGINGPP_FOUND)
    message(STATUS "easyloggingpp: pre-installed easyloggingpp found")
    get_target_property(
      EASYLOGGINGPP_INCLUDE_DIR
      easyloggingpp
      INTERFACE_INCLUDE_DIRECTORIES)
  else()
    message(STATUS "easyloggingpp: pre-installed easyloggingpp NOT found, building from source")
    add_subdirectory(cmake/third-party/easylogging)
  endif()
endif()

#------------------------------------------------------------------------------
# Subfolders...
#------------------------------------------------------------------------------
add_subdirectory(hexl)

if (HEXL_BENCHMARK)
  add_subdirectory(benchmark)
  add_custom_target(bench COMMAND $<TARGET_FILE:bench_hexl> DEPENDS bench_hexl)
endif()

if (HEXL_TESTING)
  add_subdirectory(test)
  add_custom_target(unittest COMMAND $<TARGET_FILE:unit-test> DEPENDS unit-test)
endif()

if (HEXL_DOCS)
  add_subdirectory(docs)
endif()

add_custom_target(check COMMAND pre-commit install && pre-commit run --all-files)
