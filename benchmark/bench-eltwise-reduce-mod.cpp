// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-reduce-mod-avx512.hpp"
#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

// state[0] is the degree
static void BM_EltwiseReduceModInPlace(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 =
      GenerateInsecureUniformRandomValues(input_size, 0, 100 * modulus);
  const uint64_t input_mod_factor = 0;
  const uint64_t output_mod_factor = 1;
  for (auto _ : state) {
    EltwiseReduceMod(input1.data(), input1.data(), input_size, modulus,
                     input_mod_factor, output_mod_factor);
  }
}

BENCHMARK(BM_EltwiseReduceModInPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_EltwiseReduceModCopy(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 =
      GenerateInsecureUniformRandomValues(input_size, 0, 100 * modulus);
  const uint64_t input_mod_factor = 0;
  const uint64_t output_mod_factor = 1;
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseReduceMod(output.data(), input1.data(), input_size, modulus,
                     input_mod_factor, output_mod_factor);
  }
}

BENCHMARK(BM_EltwiseReduceModCopy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_EltwiseReduceModNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 =
      GenerateInsecureUniformRandomValues(input_size, 0, 100 * modulus);
  const uint64_t input_mod_factor = 0;
  const uint64_t output_mod_factor = 1;
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseReduceModNative(output.data(), input1.data(), input_size, modulus,
                           input_mod_factor, output_mod_factor);
  }
}

BENCHMARK(BM_EltwiseReduceModNative)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseReduceModAVX512(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  auto input1 =
      GenerateInsecureUniformRandomValues(input_size, 0, 100 * modulus);
  const uint64_t input_mod_factor = 0;
  const uint64_t output_mod_factor = 1;
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseReduceModAVX512(output.data(), input1.data(), input_size, modulus,
                           input_mod_factor, output_mod_factor);
  }
}

BENCHMARK(BM_EltwiseReduceModAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
