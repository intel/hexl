# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(SRC main.cpp
    bench-ntt.cpp
    bench-eltwise-add-mod.cpp
    bench-eltwise-cmp-add.cpp
    bench-eltwise-cmp-sub-mod.cpp
    bench-eltwise-fma-mod.cpp
    bench-eltwise-mult-mod.cpp
    bench-eltwise-sub-mod.cpp
    bench-eltwise-reduce-mod.cpp
    )

add_executable(bench_hexl ${SRC})

target_include_directories(bench_hexl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HEXL_SRC_ROOT_DIR} # Private headers
    )

target_link_libraries(bench_hexl PRIVATE hexl benchmark::benchmark Threads::Threads)
if (HEXL_DEBUG)
    target_link_libraries(bench_hexl PRIVATE easyloggingpp)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_hexl PRIVATE -Wall -Wextra -march=native -O3)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(bench_hexl PRIVATE /Wall /W4
        /wd4127 # warning C4127: conditional expression is constant; C++11 doesn't support if constexpr
        /wd5105 # warning C5105: macro expansion producing 'defined' has undefined behavior
    )
endif()
