# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.5.1)
project(HEXL VERSION 1.1.1 LANGUAGES C CXX)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CMakePackageConfigHelpers)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(HEXLHelpers)

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

# Sanitizers
option(HEXL_ENABLE_ADDRESS_SANITIZER "Compiles and links with Address Sanitizer" OFF)
option(HEXL_ENABLE_THREAD_SANITIZER "Compiles and links with Thread Sanitizer" OFF)
option(HEXL_ENABLE_UB_SANITIZER "Compiles and links with Undefined Behavior Sanitizer" OFF)

option(HEXL_BENCHMARK "Enable benchmarking" ON)
option(HEXL_COVERAGE "Enables coverage for unit tests" OFF)
option(HEXL_DEBUG "Enable runtime debug checking; will yield much slower code " OFF)
option(HEXL_DOCS "Enable documentation building" OFF)
option(HEXL_SHARED_LIB "Generate a shared library" OFF)
option(HEXL_TESTING "Enables unit-tests" ON)
option(HEXL_TREAT_WARNING_AS_ERROR "Treat all compile-time warnings as errors" OFF)

message(STATUS "CMAKE_BUILD_TYPE:              ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_C_COMPILER:              ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER:            ${CMAKE_CXX_COMPILER}")
message(STATUS "HEXL_ENABLE_ADDRESS_SANITIZER: ${HEXL_ENABLE_ADDRESS_SANITIZER}")
message(STATUS "HEXL_ENABLE_THREAD_SANITIZER:  ${HEXL_ENABLE_THREAD_SANITIZER}")
message(STATUS "HEXL_ENABLE_UB_SANITIZER:      ${HEXL_ENABLE_UB_SANITIZER}")
message(STATUS "HEXL_BENCHMARK:                ${HEXL_BENCHMARK}")
message(STATUS "HEXL_COVERAGE:                 ${HEXL_COVERAGE}")
message(STATUS "HEXL_DEBUG:                    ${HEXL_DEBUG}")
message(STATUS "HEXL_DOCS:                     ${HEXL_DOCS}")
message(STATUS "HEXL_SHARED_LIB:               ${HEXL_SHARED_LIB}")
message(STATUS "HEXL_TESTING:                  ${HEXL_TESTING}")
message(STATUS "HEXL_TREAT_WARNING_AS_ERROR:   ${HEXL_TREAT_WARNING_AS_ERROR}")

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    set(HEXL_USE_MSVC ON)
    add_definitions(-DHEXL_USE_MSVC)
    message(STATUS "HEXL_USE_MSVC:                 ${HEXL_USE_MSVC}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
    set(HEXL_USE_GNU ON)
    add_definitions(-DHEXL_USE_GNU)
    message(STATUS "HEXL_USE_GNU:                  ${HEXL_USE_GNU}")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # using Clang
    set(HEXL_USE_CLANG ON)
    add_definitions(-DHEXL_USE_CLANG)
    message(STATUS "HEXL_USE_CLANG:                ${HEXL_USE_CLANG}")
else()
    message(WARNING "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
endif()

check_compiler_version()

if (HEXL_COVERAGE)
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

if (HEXL_DEBUG)
  add_definitions(-DHEXL_DEBUG)
endif()

set(HEXL_SRC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hexl)
set(HEXL_INC_ROOT_DIR ${HEXL_SRC_ROOT_DIR}/include) # Public headers

if(HEXL_ENABLE_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
endif()

if(HEXL_ENABLE_THREAD_SANITIZER)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
endif()

if(HEXL_ENABLE_UB_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fsanitize=implicit-conversion -fPIE")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fsanitize=implicit-conversion  -fPIE -pie")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fsanitize=implicit-conversion -fPIE -pie")
endif()

if(HEXL_ENABLE_ADDRESS_SANITIZER OR HEXL_ENABLE_THREAD_SANITIZER OR HEXL_ENABLE_UB_SANITIZER)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
endif()

message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_MINSIZEREL ${CMAKE_CXX_FLAGS_MINSIZEREL}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

#------------------------------------------------------------------------------
# Set AVX flags
#------------------------------------------------------------------------------
check_compile_flag("${CMAKE_MODULE_PATH}/test-avx512dq.cpp" HEXL_HAS_AVX512DQ)
check_compile_flag("${CMAKE_MODULE_PATH}/test-avx512ifma.cpp" HEXL_HAS_AVX512IFMA)
check_compile_flag("${CMAKE_MODULE_PATH}/test-avx512vbmi2.cpp" HEXL_HAS_AVX512VBMI2)
check_compile_flag("${CMAKE_MODULE_PATH}/test-avx256.cpp" HEXL_HAS_AVX256)

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
add_subdirectory(cmake/cpu-features)

if (HEXL_TESTING OR HEXL_BENCHMARK OR HEXL_DEBUG)
  if(NOT TARGET Threads::Threads)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
  endif()
  find_package(Threads REQUIRED)
endif()

if (HEXL_TESTING)
  add_subdirectory(cmake/gtest)
endif()

if (HEXL_BENCHMARK)
  add_subdirectory(cmake/gbenchmark)
endif()

if (HEXL_DEBUG)
  add_subdirectory(cmake/easylogging)
  add_subdirectory(cmake/gflags)
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

#------------------------------------------------------------------------------
# Config export...
#------------------------------------------------------------------------------
# Config filenames
set(HEXL_TARGET_FILENAME ${CMAKE_CURRENT_BINARY_DIR}/cmake/HEXLTargets.cmake)
set(HEXL_CONFIG_IN_FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HEXLConfig.cmake.in)
set(HEXL_CONFIG_FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HEXLConfig.cmake)
set(HEXL_CONFIG_VERSION_FILENAME ${CMAKE_CURRENT_BINARY_DIR}/cmake/HEXLConfigVersion.cmake)

set(HEXL_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/)

# Create and install the CMake config and target file
install(
    EXPORT HEXLTargets
    NAMESPACE HEXL::
    DESTINATION ${HEXL_CONFIG_INSTALL_DIR}
)

# Export version
write_basic_package_version_file(
  ${HEXL_CONFIG_VERSION_FILENAME}
  VERSION ${HEXL_VERSION}
  COMPATIBILITY ExactVersion)

include(CMakePackageConfigHelpers)
    configure_package_config_file(
        ${HEXL_CONFIG_IN_FILENAME} ${HEXL_CONFIG_FILENAME}
        INSTALL_DESTINATION ${HEXL_CONFIG_INSTALL_DIR}
    )

install(
    TARGETS hexl
    EXPORT HEXLTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

install(FILES ${HEXL_CONFIG_FILENAME}
              ${HEXL_CONFIG_VERSION_FILENAME}
        DESTINATION ${HEXL_CONFIG_INSTALL_DIR})

export(EXPORT HEXLTargets
        FILE ${HEXL_TARGET_FILENAME})


add_custom_target(check COMMAND pre-commit install && pre-commit run --all-files)
