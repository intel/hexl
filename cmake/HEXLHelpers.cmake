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
    else()
        message(STATUS "Compile flag not found: ${OUTPUT_FLAG}")
    endif()
endfunction()
