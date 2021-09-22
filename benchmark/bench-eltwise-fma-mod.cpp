// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-fma-mod-avx512.hpp"
#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "hexl/eltwise/eltwise-fma-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

//=================================================================

static void BM_EltwiseFMAModAddNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;
  bool add = state.range(1);

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformRandomValue(0, modulus);
  AlignedVector64<uint64_t> input3 =
      GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t* arg3 = add ? input3.data() : nullptr;

  for (auto _ : state) {
    EltwiseFMAMod(input1.data(), input1.data(), input2, arg3, input1.size(),
                  modulus, 1);
  }
}

BENCHMARK(BM_EltwiseFMAModAddNative)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {false, true}});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
static void BM_EltwiseFMAModAVX512DQ(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 100;
  bool add = state.range(1);

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformRandomValue(0, modulus);
  AlignedVector64<uint64_t> input3 =
      GenerateInsecureUniformRandomValues(input_size, 0, modulus);

  uint64_t* arg3 = add ? input3.data() : nullptr;

  for (auto _ : state) {
    EltwiseFMAModAVX512<64, 1>(input1.data(), input1.data(), input2, arg3,
                               input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseFMAModAVX512DQ)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {false, true}});
#endif

//=================================================================

#ifdef HEXL_HAS_AVX512IFMA
static void BM_EltwiseFMAModAVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 100;
  bool add = state.range(1);

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformRandomValue(0, modulus);
  auto input3 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);

  uint64_t* arg3 = add ? input3.data() : nullptr;

  for (auto _ : state) {
    EltwiseFMAModAVX512<52, 1>(input1.data(), input1.data(), input2, arg3,
                               input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseFMAModAVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {false, true}});

#endif

}  // namespace hexl
}  // namespace intel
