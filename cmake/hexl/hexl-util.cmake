# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Checks if SOURCE_FILE can be compiled and returns 0 upon running
# If so, adds OUTPUT_FLAG to compile definitions
function(hexl_check_compile_flag SOURCE_FILE OUTPUT_FLAG)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(NATIVE_COMPILE_DEFINITIONS "/arch:AVX512")
    else()
        set(NATIVE_COMPILE_DEFINITIONS "-march=native")
    endif()

    try_run(CAN_RUN CAN_COMPILE ${CMAKE_BINARY_DIR}
        "${SOURCE_FILE}"
        COMPILE_DEFINITIONS ${NATIVE_COMPILE_DEFINITIONS}
        OUTPUT_VARIABLE TRY_COMPILE_OUTPUT
    )
    # Uncomment below to debug
    # message("TRY_COMPILE_OUTPUT ${TRY_COMPILE_OUTPUT}")
    if (CAN_COMPILE AND CAN_RUN STREQUAL 0)
        message(STATUS "Setting ${OUTPUT_FLAG}")
        add_definitions(-D${OUTPUT_FLAG})
        set(${OUTPUT_FLAG} 1 PARENT_SCOPE)
    else()
        message(STATUS "Compile flag not found: ${OUTPUT_FLAG}")
    endif()
endfunction()

function(hexl_check_compiler_version)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        message(FATAL_ERROR "HEXL requires gcc version >= 7.0")
      endif()
      if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
        message(WARN "gcc version should be at least 8.0 for best performance on processors with AVX512IFMA support")
      endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
            message(FATAL_ERROR "HEXL requires clang++ >= 5.0")
        endif()
      if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
          message(WARNING "Clang version should be at least 6.0 for best performance on processors with AVX512IFMA support")
      endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
        message(FATAL_ERROR "HEXL requires MSVC >= 19")
      endif()
    endif()
endfunction()

# If the input variable is set, stores its value in a _CACHE variable
function(hexl_cache_variable variable)
  if (DEFINED ${variable})
    set(${variable}_CACHE ${${variable}} PARENT_SCOPE)
  endif()
endfunction()

# If the input variable is cached, restores its value from the cache
function(hexl_uncache_variable variable)
  if (DEFINED ${variable}_CACHE)
    set(${variable} ${${variable}_CACHE} CACHE BOOL "" FORCE )
  endif()
endfunction()

function(hexl_add_compiler_definition)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(HEXL_USE_MSVC ON PARENT_SCOPE)
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(HEXL_USE_GNU ON PARENT_SCOPE)
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(HEXL_USE_CLANG ON PARENT_SCOPE)
  else()
      message(WARNING "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()
endfunction()

function(hexl_add_asan_flag target)
  # Enable AddressSanitizer in Debug mode on Mac/Linux
  if(HEXL_DEBUG AND UNIX)
    target_compile_options(${target} PUBLIC -fsanitize=address)
    target_link_options(${target} PUBLIC -fsanitize=address)
    set(HEXL_ASAN_LINK "-lasan" PARENT_SCOPE)
  else()
    set(HEXL_ASAN_LINK "" PARENT_SCOPE)
  endif()

endfunction()
