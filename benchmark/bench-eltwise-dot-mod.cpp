// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-dot-mod-avx512.hpp"
#include "eltwise/eltwise-dot-mod-internal.hpp"
#include "hexl/eltwise/eltwise-dot-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"

namespace intel {
namespace hexl {

// state[0] is the degree
static void BM_EltwiseDotModNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> input3(input_size, 1);
  AlignedVector64<uint64_t> input4(input_size, 1);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseDotModNative(output.data(), input1.data(), input2.data(),
                        input3.data(), input4.data(), input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseDotModNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseDotModAVX512(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> input3(input_size, 1);
  AlignedVector64<uint64_t> input4(input_size, 1);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseDotModAVX512(output.data(), input1.data(), input2.data(),
                        input3.data(), input4.data(), input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseDotModAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

}  // namespace hexl
}  // namespace intel
