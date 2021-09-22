// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-cmp-add-avx512.hpp"
#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "hexl/eltwise/eltwise-cmp-add.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

//=================================================================

// state[0] is the degree
static void BM_EltwiseCmpAddNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);

  uint64_t modulus = 100;

  uint64_t bound = GenerateInsecureUniformRandomValue(0, modulus);
  uint64_t diff = GenerateInsecureUniformRandomValue(1, modulus - 1);
  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);

  for (auto _ : state) {
    EltwiseCmpAddNative(input1.data(), input1.data(), input_size, CMPINT::NLT,
                        bound, diff);
  }
}

BENCHMARK(BM_EltwiseCmpAddNative)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseCmpAddAVX512(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);

  uint64_t bound = 50;
  // must be non-zero
  uint64_t diff = GenerateInsecureUniformRandomValue(1, bound - 1);
  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, bound);

  for (auto _ : state) {
    EltwiseCmpAddAVX512(input1.data(), input1.data(), input_size, CMPINT::NLT,
                        bound, diff);
  }
}

BENCHMARK(BM_EltwiseCmpAddAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

}  // namespace hexl
}  // namespace intel
