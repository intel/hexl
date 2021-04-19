# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Checks if SOURCE_FILE can be compiled and returns 0 upon running
# If so, adds OUTPUT_FLAG to compile definitions
function(check_compile_flag SOURCE_FILE OUTPUT_FLAG)
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

function(check_compiler_version)
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
