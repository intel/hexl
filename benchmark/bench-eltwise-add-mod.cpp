// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-add-mod-avx512.hpp"
#include "eltwise/eltwise-add-mod-internal.hpp"
#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

// state[0] is the degree
static void BM_EltwiseVectorVectorAddModNative(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModNative(output.data(), input1.data(), input2.data(), input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseVectorVectorAddModNative)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseVectorVectorAddModAVX512(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModAVX512(output.data(), input1.data(), input2.data(), input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseVectorVectorAddModAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

//=================================================================
// state[0] is the degree
static void BM_EltwiseVectorScalarAddModNative(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformRandomValue(0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModNative(output.data(), input1.data(), input2, input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseVectorScalarAddModNative)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseVectorScalarAddModAVX512(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformRandomValue(0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModAVX512(output.data(), input1.data(), input2, input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseVectorScalarAddModAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

}  // namespace hexl
}  // namespace intel
