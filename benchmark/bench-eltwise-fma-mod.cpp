// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-fma-mod-avx512.hpp"
#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "intel-hexl/eltwise/eltwise-fma-mod.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"

namespace intel {
namespace hexl {

//=================================================================

// state[0] is the degree
static void BM_EltwiseFMANative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  AlignedVector64<uint64_t> op1(input_size, 1);
  uint64_t op2 = 1;
  AlignedVector64<uint64_t> op3(input_size, 2);

  for (auto _ : state) {
    EltwiseFMAMod(op1.data(), op1.data(), op2, op3.data(), op1.size(), modulus,
                  1);
  }
}

BENCHMARK(BM_EltwiseFMANative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseFMAAVX512DQ(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 100;

  AlignedVector64<uint64_t> input1(input_size, 1);
  uint64_t input2 = 3;
  AlignedVector64<uint64_t> input3(input_size, 2);

  for (auto _ : state) {
    EltwiseFMAModAVX512<64, 1>(input1.data(), input1.data(), input2,
                               input3.data(), input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseFMAAVX512DQ)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

//=================================================================

#ifdef HEXL_HAS_AVX512IFMA
// state[0] is the degree
static void BM_EltwiseFMAAVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 100;

  AlignedVector64<uint64_t> input1(input_size, 1);
  uint64_t input2 = 3;
  AlignedVector64<uint64_t> input3(input_size, 2);

  for (auto _ : state) {
    EltwiseFMAModAVX512<52, 1>(input1.data(), input1.data(), input2,
                               input3.data(), input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseFMAAVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

}  // namespace hexl
}  // namespace intel
