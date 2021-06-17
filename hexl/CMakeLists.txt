# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(SRC
    eltwise/eltwise-mult-mod.cpp
    eltwise/eltwise-reduce-mod.cpp
    eltwise/eltwise-sub-mod.cpp
    eltwise/eltwise-add-mod.cpp
    eltwise/eltwise-fma-mod.cpp
    eltwise/eltwise-cmp-add.cpp
    eltwise/eltwise-cmp-sub-mod.cpp
    ntt/ntt-internal.cpp
    number-theory/number-theory.cpp
)

if (HEXL_HAS_AVX512DQ)
    list(APPEND SRC
        eltwise/eltwise-mult-mod-avx512.cpp
        eltwise/eltwise-reduce-mod-avx512.cpp
        eltwise/eltwise-add-mod-avx512.cpp
        eltwise/eltwise-sub-mod-avx512.cpp
        eltwise/eltwise-fma-mod-avx512.cpp
        ntt/fwd-ntt-avx512.cpp
        ntt/inv-ntt-avx512.cpp
    )
endif()

if (HEXL_DEBUG)
    list(APPEND SRC logging/logging.cpp)
endif()

if (HEXL_SHARED_LIB)
    add_library(hexl SHARED ${SRC})
else()
     add_library(hexl STATIC ${SRC})
endif()
add_library(HEXL::hexl ALIAS hexl)

hexl_add_asan_flag(hexl)
hexl_add_compiler_definition()

set(HEXL_DEFINES_IN_FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/include/hexl/util/defines.hpp.in)
set(HEXL_DEFINES_FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/include/hexl/util/defines.hpp)
configure_file(${HEXL_DEFINES_IN_FILENAME} ${HEXL_DEFINES_FILENAME})

set_target_properties(hexl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(hexl PROPERTIES VERSION ${HEXL_VERSION})
if (HEXL_DEBUG)
    set_target_properties(hexl PROPERTIES OUTPUT_NAME "hexl_debug")
endif()

target_include_directories(hexl
    PRIVATE ${HEXL_SRC_ROOT_DIR}                               # Private headers
    PUBLIC  $<BUILD_INTERFACE:${HEXL_INC_ROOT_DIR}>            # Public headers
    PUBLIC  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>   # Public headers
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hexl PRIVATE -Wall -Wconversion -Wshadow -pedantic -Wextra
        -Wno-unknown-pragmas -march=native -O3 -fomit-frame-pointer
        -Wno-sign-conversion # clang-10 and clang-12 have different APIs for _mm512_slli_epi64
        -Wno-implicit-int-conversion # easylogging triggers this warning
    )
    # Avoid 3rd-party dependency warnings when including HEXL as a dependency
    target_compile_options(hexl PUBLIC
        -Wno-sign-conversion         # avoid warnings in gflags headers
        -Wno-implicit-int-conversion # avoid warnings in gflags headers
    )

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Inling causes some tests to fail on MSVC with AVX512 in Release mode, HEXL_DEBUG=OFF,
    # so we disable it here
    target_compile_options(hexl PRIVATE /Wall /W4 /Zc:preprocessor /Ob0
        /wd4127 # warning C4127: conditional expression is constant; C++11 doesn't support constexpr
     )
    target_compile_definitions(hexl PRIVATE -D_CRT_SECURE_NO_WARNINGS)
endif()

install(DIRECTORY ${HEXL_INC_ROOT_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

if (HEXL_SHARED_LIB)
    target_link_libraries(hexl PRIVATE cpu_features)
    if (HEXL_DEBUG)
        target_link_libraries(hexl PUBLIC easyloggingpp)
        target_link_libraries(hexl PRIVATE gflags)
    endif()
else ()
    # For static library, we include all the dependencies for Intel HEXL in
    # the libhexl.a.
    # For proper export of IntelHEXLConfig.cmake / IntelHEXLTargts.cmake,
    # we avoid explicitly linking dependencies via target_link_libraries, since
    # this would add dependencies to the exported hexl target.
    add_dependencies(hexl cpu_features)
    if (HEXL_DEBUG)
        add_dependencies(hexl gflags)
        add_dependencies(hexl easyloggingpp)

        # Manually add logging include directory
        target_include_directories(hexl
            PUBLIC $<BUILD_INTERFACE:$<TARGET_PROPERTY:easyloggingpp,INCLUDE_DIRECTORIES>>
        )

        # Manually add gflags include directory
        target_include_directories(hexl
            PRIVATE $<TARGET_PROPERTY:gflags,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

    # Manually add cpu_features include directory
    target_include_directories(hexl
      PRIVATE $<TARGET_PROPERTY:cpu_features,INTERFACE_INCLUDE_DIRECTORIES>)

    # Export logging only if we are debugging
    if (${HEXL_DEBUG})
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            add_custom_command(TARGET hexl POST_BUILD
                COMMAND ar -x $<TARGET_FILE:hexl>
                COMMAND ar -x $<TARGET_FILE:cpu_features>
                COMMAND ar -x $<TARGET_FILE:gflags>
                COMMAND ar -x $<TARGET_FILE:easyloggingpp>
                COMMAND ar -qcs $<TARGET_FILE:hexl> *.o
                COMMAND rm -f *.o
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS hexl cpu_features gflags easyloggingpp
            )
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
             add_custom_command(TARGET hexl POST_BUILD
                COMMAND lib.exe /OUT:$<TARGET_FILE:hexl>
                    $<TARGET_FILE:hexl>
                    $<TARGET_FILE:cpu_features>
                    $<TARGET_FILE:gflags>
                    $<TARGET_FILE:easyloggingpp>
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS hexl cpu_features gflags easyloggingpp
             )
        endif()
    else()
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            add_custom_command(TARGET hexl POST_BUILD
                COMMAND ar -x $<TARGET_FILE:hexl>
                COMMAND ar -x $<TARGET_FILE:cpu_features>
                COMMAND ar -qcs $<TARGET_FILE:hexl> *.o
                COMMAND rm -f *.o
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS hexl cpu_features
        )
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            add_custom_command(TARGET hexl POST_BUILD
                COMMAND lib.exe /OUT:$<TARGET_FILE:hexl>
                    $<TARGET_FILE:hexl>
                    $<TARGET_FILE:cpu_features>
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS hexl cpu_features
            )
      endif()
    endif()
endif()

install(TARGETS hexl DESTINATION ${CMAKE_INSTALL_LIBDIR})
