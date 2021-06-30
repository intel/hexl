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
  size_t modulus = 1152921504606877697;

  uint64_t num_vectors = 2;

  AlignedVector64<uint64_t> input1(num_vectors * input_size, 1);
  AlignedVector64<uint64_t> input2(num_vectors * input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 0);

  AlignedVector64<const uint64_t*> input1_addresses;
  AlignedVector64<const uint64_t*> input2_addresses;
  for (size_t k = 0; k < num_vectors; ++k) {
    input1_addresses.push_back(&input1[k * input_size]);
    input2_addresses.push_back(&input2[k * input_size]);
  }

  for (auto _ : state) {
    EltwiseDotModNative(output.data(), input1_addresses.data(),
                        input2_addresses.data(), num_vectors, input_size,
                        modulus);
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

  uint64_t num_vectors = 2;

  AlignedVector64<uint64_t> input1(num_vectors * input_size, 1);
  AlignedVector64<uint64_t> input2(num_vectors * input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 0);

  AlignedVector64<const uint64_t*> input1_addresses;
  AlignedVector64<const uint64_t*> input2_addresses;
  for (size_t k = 0; k < num_vectors; ++k) {
    input1_addresses.push_back(&input1[k * input_size]);
    input2_addresses.push_back(&input2[k * input_size]);
  }

  for (auto _ : state) {
    EltwiseDotModAVX512(output.data(), input1_addresses.data(),
                        input2_addresses.data(), num_vectors, input_size,
                        modulus);
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
