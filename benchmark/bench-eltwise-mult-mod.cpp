// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-mult-mod-avx512.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "eltwise/eltwise-reduce-mod-avx512.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

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

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 2);

  for (auto _ : state) {
    EltwiseMultMod(output.data(), input1.data(), input2.data(), input_size,
                   modulus, input_mod_factor);
  }
}

BENCHMARK(BM_EltwiseMultMod)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {48, 60}, {1, 2, 4}});

//=================================================================

// state[0] is the degree
static void BM_EltwiseMultModNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 2);

  for (auto _ : state) {
    EltwiseMultModNative<1>(output.data(), input1.data(), input2.data(),
                            input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseMultModNative)
    ->Unit(benchmark::kMicrosecond)
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

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
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
    ->ArgsProduct({{1024, 4096, 16384}, {1, 2, 4}});
#endif

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
// state[1] is the input_mod_factor
static void BM_EltwiseMultModAVX512DQInt(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t input_mod_factor = state.range(1);
  size_t modulus = 0xffffffffffc0001ULL;

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 3);

  for (auto _ : state) {
    switch (input_mod_factor) {
      case 1:
        EltwiseMultModAVX512DQInt<1>(output.data(), input1.data(),
                                     input2.data(), input_size, modulus);
        break;
      case 2:
        EltwiseMultModAVX512DQInt<2>(output.data(), input1.data(),
                                     input2.data(), input_size, modulus);
        break;
      case 4:
        EltwiseMultModAVX512DQInt<4>(output.data(), input1.data(),
                                     input2.data(), input_size, modulus);
        break;
    }
  }
}

BENCHMARK(BM_EltwiseMultModAVX512DQInt)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {1, 2, 4}});
#endif

#ifdef HEXL_HAS_AVX512IFMA
// state[0] is the degree
// state[1] is the input_mod_factor
static void BM_EltwiseMultModAVX512IFMAInt(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t input_mod_factor = state.range(1);
  size_t modulus = 100;

  auto input1 = GenerateInsecureUniformIntRandomValues(
      input_size, 0, input_mod_factor * modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(
      input_size, 0, input_mod_factor * modulus);
  AlignedVector64<uint64_t> output(input_size, 3);

  for (auto _ : state) {
    switch (input_mod_factor) {
      case 1:
        EltwiseMultModAVX512IFMAInt<1>(output.data(), input1.data(),
                                       input2.data(), input_size, modulus);
        break;
      case 2:
        EltwiseMultModAVX512IFMAInt<2>(output.data(), input1.data(),
                                       input2.data(), input_size, modulus);
        break;
      case 4:
        EltwiseMultModAVX512IFMAInt<4>(output.data(), input1.data(),
                                       input2.data(), input_size, modulus);
        break;
    }
  }
}

BENCHMARK(BM_EltwiseMultModAVX512IFMAInt)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {1, 2, 4}});
#endif

//=================================================================

#ifdef HEXL_HAS_AVX512IFMA

// state[0] is the degree
// state[1] is the input_mod_factor
static void BM_EltwiseMultModMontAVX512IFMAIntEConv(
    benchmark::State& state) {  //  NOLINT

  size_t input_size = state.range(0);
  size_t input_mod_factor = state.range(1);
  uint64_t modulus = (1ULL << 50) + 7;  // 1125899906842631
  auto op1 = GenerateInsecureUniformIntRandomValues(input_size, 0,
                                                    input_mod_factor * modulus);
  auto op2 = GenerateInsecureUniformIntRandomValues(input_size, 0,
                                                    input_mod_factor * modulus);
  AlignedVector64<uint64_t> output(input_size, 3);

  int r = 51;  // R = 2251799813685248
  // mod(2251799813685248*2251799813685248;1125899906842631)
  uint64_t R_reduced = ReduceMod<2>(1ULL << r, modulus);
  const uint64_t R_square_mod_q = MultiplyMod(R_reduced, R_reduced, modulus);
  uint64_t neg_inv_mod = HenselLemma2adicRoot(r, modulus);

  for (auto _ : state) {
    if (input_mod_factor != 1) {
      EltwiseReduceModAVX512(op1.data(), op1.data(), input_size, modulus,
                             input_mod_factor, 1);
      EltwiseReduceModAVX512(op2.data(), op2.data(), input_size, modulus,
                             input_mod_factor, 1);
    }
    EltwiseMontgomeryFormInAVX512<52, 51>(output.data(), op1.data(),
                                          R_square_mod_q, input_size, modulus,
                                          neg_inv_mod);
    EltwiseMontReduceModAVX512<52, 51>(output.data(), output.data(), op2.data(),
                                       input_size, modulus, neg_inv_mod);
  }
}

BENCHMARK(BM_EltwiseMultModMontAVX512IFMAIntEConv)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {1, 2, 4}});

#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
