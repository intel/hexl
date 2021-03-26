// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-add-mod-avx512.hpp"
#include "eltwise/eltwise-add-mod-internal.hpp"
#include "intel-hexl/eltwise/eltwise-add-mod.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"

namespace intel {
namespace hexl {

// state[0] is the degree
static void BM_EltwiseAddModInPlace(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddMod(input1.data(), input1.data(), input2.data(), input_size,
                  modulus);
  }
}

BENCHMARK(BM_EltwiseAddModInPlace)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(2.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_EltwiseAddModCopy(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddMod(output.data(), input1.data(), input2.data(), input_size,
                  modulus);
  }
}

BENCHMARK(BM_EltwiseAddModCopy)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(2.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_EltwiseAddModNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModNative(output.data(), input1.data(), input2.data(), input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseAddModNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseAddModAVX512(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModAVX512(output.data(), input1.data(), input2.data(), input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseAddModAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
