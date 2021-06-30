# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(SRC main.cpp
    test-aligned-vector.cpp
    test-avx512-util.cpp
    test-number-theory.cpp
    test-ntt.cpp
    test-eltwise-mult-mod.cpp
    test-eltwise-reduce-mod.cpp
    test-eltwise-add-mod.cpp
    test-eltwise-fma-mod.cpp
    test-eltwise-cmp-add.cpp
    test-eltwise-cmp-sub-mod.cpp
    test-eltwise-sub-mod.cpp
)

add_executable(unit-test ${SRC})

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(unit-test PRIVATE -Wall -Wextra -march=native)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Workaround for unresolved symbols, shouldn't be needed
    if (HEXL_DEBUG)
        target_link_libraries(unit-test PRIVATE gflags)
    endif()
    # Disable inline, due to incorect optimization in ExtractValues, causing failing tests in Windows AVX512 in Release mode with HEXL_DEBUG=OFF
    target_compile_options(unit-test PRIVATE /Wall /W4 /Zc:preprocessor /Ob0)
endif()

target_include_directories(unit-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HEXL_SRC_ROOT_DIR} # Private headers
)

# Link to cpu_features to allow selectively disabling AVX512 support for CI
target_link_libraries(unit-test PRIVATE hexl cpu_features gtest Threads::Threads)
if (HEXL_DEBUG)
    target_link_libraries(unit-test PRIVATE easyloggingpp)
endif()

# Make sure that public include folder doesn't use private headers
# and that public headers are self-contained
# TODO(fboemer): Re-enable when coverage build integration works
# add_executable(test-public-api test-public-api.cpp)
# add_dependencies(test-public-api hexl)
# if (HEXL_DEBUG)
#     target_link_libraries(test-public-api PUBLIC easyloggingpp)
# endif()
# target_include_directories(test-public-api PRIVATE ${HEXL_INC_ROOT_DIR})
# target_include_directories(test-public-api
#       PRIVATE $<TARGET_PROPERTY:cpu_features,INTERFACE_INCLUDE_DIRECTORIES>)
