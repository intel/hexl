# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(SRC
    eltwise/eltwise-mult-mod.cpp
    eltwise/eltwise-reduce-mod.cpp
    eltwise/eltwise-add-mod.cpp
    eltwise/eltwise-fma-mod.cpp
    eltwise/eltwise-cmp-add.cpp
    eltwise/eltwise-cmp-sub-mod.cpp
    eltwise/eltwise-mult-mod-avx512.cpp
    eltwise/eltwise-reduce-mod-avx512.cpp
    eltwise/eltwise-add-mod-avx512.cpp
    eltwise/eltwise-fma-mod-avx512.cpp
    ntt/fwd-ntt-avx512.cpp
    ntt/inv-ntt-avx512.cpp
    ntt/ntt-internal.cpp
    logging/logging.cpp
    number-theory/number-theory.cpp
)

if (HEXL_SHARED_LIB)
    add_library(intel_hexl SHARED ${SRC})
else()
     add_library(intel_hexl STATIC ${SRC})
endif()

set_target_properties(intel_hexl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(intel_hexl PROPERTIES VERSION ${INTEL_HEXL_VERSION})

target_include_directories(intel_hexl
    PRIVATE ${HEXL_SRC_ROOT_DIR}                               # Private headers
    PUBLIC  $<BUILD_INTERFACE:${HEXL_INC_ROOT_DIR}>            # Public headers
    PUBLIC  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>   # Public headers
    )

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(intel_hexl PRIVATE -Wall -Wextra -Wno-unknown-pragmas -march=native -O3 -fomit-frame-pointer)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Inling causes some tests to fail on MSVC with AVX512 in Release mode, HEXL_DEBUG=OFF,
    # so we disable it here
    target_compile_options(intel_hexl PRIVATE /Wall /W4 /Zc:preprocessor /Ob0)
endif()

install(DIRECTORY ${HEXL_INC_ROOT_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

if (HEXL_SHARED_LIB)
    target_link_libraries(intel_hexl PRIVATE cpu_features)
    if (HEXL_DEBUG)
        target_link_libraries(intel_hexl PRIVATE easyloggingpp gflags)
    endif()
else ()
    # For static library, we include all the dependencies for Intel HEXL in
    # the libintel_hexl.a.
    # For proper export of IntelHEXLConfig.cmake / IntelHEXLTargts.cmake,
    # we avoid explicitly linking dependencies via target_link_libraries, since
    # this would add dependencies to the exported intel_hexl target.
    add_dependencies(intel_hexl cpu_features)
    if (HEXL_DEBUG)
        add_dependencies(intel_hexl gflags)
        # Manually add logging include directory
        target_include_directories(intel_hexl
        PUBLIC $<TARGET_PROPERTY:easyloggingpp,INTERFACE_INCLUDE_DIRECTORIES>)
        # Manually add gflags include directory
        target_include_directories(intel_hexl
        PUBLIC $<TARGET_PROPERTY:gflags,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

    # Manually add cpu_features include directory
    target_include_directories(intel_hexl
      PRIVATE $<TARGET_PROPERTY:cpu_features,INTERFACE_INCLUDE_DIRECTORIES>)

    # Export logging only if we are debugging
    if (${HEXL_DEBUG})
        set_target_properties(intel_hexl PROPERTIES OUTPUT_NAME "intel_hexl_debug")
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            add_custom_command(TARGET intel_hexl POST_BUILD
                COMMAND ar -x $<TARGET_FILE:intel_hexl>
                COMMAND ar -x $<TARGET_FILE:cpu_features>
                COMMAND ar -x $<TARGET_FILE:gflags>
                COMMAND ar -x $<TARGET_FILE:easyloggingpp>
                COMMAND ar -qcs $<TARGET_FILE:intel_hexl> *.o
                COMMAND rm -f *.o
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS intel_hexl cpu_features gflags easyloggingpp
            )
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
             add_custom_command(TARGET intel_hexl POST_BUILD
                COMMAND lib.exe /OUT:$<TARGET_FILE:intel_hexl>
                    $<TARGET_FILE:intel_hexl>
                    $<TARGET_FILE:cpu_features>
                    $<TARGET_FILE:gflags>
                    $<TARGET_FILE:easyloggingpp>
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS intel_hexl cpu_features gflags easyloggingpp
             )
        endif()
    else()
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            add_custom_command(TARGET intel_hexl POST_BUILD
                COMMAND ar -x $<TARGET_FILE:intel_hexl>
                COMMAND ar -x $<TARGET_FILE:cpu_features>
                COMMAND ar -qcs $<TARGET_FILE:intel_hexl> *.o
                COMMAND rm -f *.o
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS intel_hexl cpu_features
        )
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            add_custom_command(TARGET intel_hexl POST_BUILD
                COMMAND lib.exe /OUT:$<TARGET_FILE:intel_hexl>
                    $<TARGET_FILE:intel_hexl>
                    $<TARGET_FILE:cpu_features>
                    # $<TARGET_FILE:gflags>
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS intel_hexl cpu_features # gflags
            )
      endif()
    endif()
endif()

install(TARGETS intel_hexl DESTINATION ${CMAKE_INSTALL_LIBDIR})
