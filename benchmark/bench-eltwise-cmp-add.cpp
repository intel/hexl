// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "eltwise/eltwise-cmp-add-avx512.hpp"
#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "hexl/eltwise/eltwise-cmp-add.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/aligned-allocator.hpp"

namespace intel {
namespace hexl {

//=================================================================

// state[0] is the degree
static void BM_EltwiseCmpAddNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<uint64_t> distrib(1, 100);

  uint64_t bound = distrib(gen);
  uint64_t diff = distrib(gen);
  AlignedVector64<uint64_t> input1(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    input1[i] = distrib(gen);
  }

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

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<uint64_t> distrib(1, 100);

  uint64_t bound = 50;
  uint64_t diff = distrib(gen);
  AlignedVector64<uint64_t> input1(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    input1[i] = distrib(gen);
  }

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
