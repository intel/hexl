// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-mult-mod-avx512.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"

namespace intel {
namespace hexl {

// state[0] is the degree
// state[1] is the bit-width of the modulus
// state[2] is the input_mod_factor
static void BM_EltwiseMultMod(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t bit_width = state.range(1);
  size_t input_mod_factor = state.range(2);
  uint64_t modulus = (1ULL << bit_width) + 7;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 2);

  for (auto _ : state) {
    EltwiseMultMod(output.data(), input1.data(), input2.data(), input_size,
                   modulus, input_mod_factor);
  }
}

BENCHMARK(BM_EltwiseMultMod)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->ArgsProduct({{1024, 8192, 16384}, {48, 60}, {1, 2, 4}});

//=================================================================

// state[0] is the degree
static void BM_EltwiseMultModNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 2);

  for (auto _ : state) {
    EltwiseMultModNative<1>(output.data(), input1.data(), input2.data(),
                            input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseMultModNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
// state[1] is the input_mod_factor
static void BM_EltwiseMultModAVX512Float(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t input_mod_factor = state.range(1);
  size_t modulus = 100;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 2);

  for (auto _ : state) {
    switch (input_mod_factor) {
      case 1:
        EltwiseMultModAVX512Float<1>(output.data(), input1.data(),
                                     input2.data(), input_size, modulus);
        break;
      case 2:
        EltwiseMultModAVX512Float<2>(output.data(), input1.data(),
                                     input2.data(), input_size, modulus);
        break;
      case 4:
        EltwiseMultModAVX512Float<4>(output.data(), input1.data(),
                                     input2.data(), input_size, modulus);
        break;
    }
  }
}

BENCHMARK(BM_EltwiseMultModAVX512Float)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->Args({1024, 1})
    ->Args({1024, 2})
    ->Args({1024, 4})
    ->Args({4096, 1})
    ->Args({4096, 2})
    ->Args({4096, 4})
    ->Args({16384, 1})
    ->Args({16384, 2})
    ->Args({16384, 4});
#endif

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
// state[1] is the input_mod_factor
static void BM_EltwiseMultModAVX512Int(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t input_mod_factor = state.range(1);
  size_t modulus = 1152921504606877697;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 3);

  for (auto _ : state) {
    switch (input_mod_factor) {
      case 1:
        EltwiseMultModAVX512Int<1>(output.data(), input1.data(), input2.data(),
                                   input_size, modulus);
        break;
      case 2:
        EltwiseMultModAVX512Int<2>(output.data(), input1.data(), input2.data(),
                                   input_size, modulus);
        break;
      case 4:
        EltwiseMultModAVX512Int<4>(output.data(), input1.data(), input2.data(),
                                   input_size, modulus);
        break;
    }
  }
}

BENCHMARK(BM_EltwiseMultModAVX512Int)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->Args({1024, 1})
    ->Args({1024, 2})
    ->Args({1024, 4})
    ->Args({4096, 1})
    ->Args({4096, 2})
    ->Args({4096, 4})
    ->Args({16384, 1})
    ->Args({16384, 2})
    ->Args({16384, 4});
#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
