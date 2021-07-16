// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-sub-mod-avx512.hpp"
#include "eltwise/eltwise-sub-mod-internal.hpp"
#include "hexl/eltwise/eltwise-sub-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

// state[0] is the degree
static void BM_EltwiseVectorVectorSubModNative(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseSubModNative(output.data(), input1.data(), input2.data(), input_size,
                        modulus);
  }
}

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseVectorVectorSubModAVX512(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  AlignedVector64<uint64_t> input1(input_size, 1);
  AlignedVector64<uint64_t> input2(input_size, 2);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseSubModAVX512(output.data(), input1.data(), input2.data(), input_size,
                        modulus);
  }
}
#endif

//=================================================================
// state[0] is the degree
static void BM_EltwiseVectorScalarSubModNative(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  AlignedVector64<uint64_t> input1(input_size, 1);
  uint64_t input2{2};
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseSubModNative(output.data(), input1.data(), input2, input_size,
                        modulus);
  }
}

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseVectorScalarSubModAVX512(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  AlignedVector64<uint64_t> input1(input_size, 1);
  uint64_t input2{2};
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseSubModAVX512(output.data(), input1.data(), input2, input_size,
                        modulus);
  }
}
#endif

//=================================================================

void register_eltwise_sub_mod_benchmarks() {
  benchmark::RegisterBenchmark("BM_EltwiseVectorVectorSubModNative",
                               BM_EltwiseVectorVectorSubModNative)
      ->Unit(benchmark::kMicrosecond)
      ->Args({1024})
      ->Args({4096})
      ->Args({16384});
  benchmark::RegisterBenchmark("BM_EltwiseVectorScalarSubModNative",
                               BM_EltwiseVectorScalarSubModNative)
      ->Unit(benchmark::kMicrosecond)
      ->Args({1024})
      ->Args({4096})
      ->Args({16384});

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    benchmark::RegisterBenchmark("BM_EltwiseVectorVectorSubModAVX512",
                                 BM_EltwiseVectorVectorSubModAVX512)
        ->Unit(benchmark::kMicrosecond)
        ->Args({1024})
        ->Args({4096})
        ->Args({16384});
    benchmark::RegisterBenchmark("BM_EltwiseVectorScalarSubModAVX512",
                                 BM_EltwiseVectorScalarSubModAVX512)
        ->Unit(benchmark::kMicrosecond)
        ->Args({1024})
        ->Args({4096})
        ->Args({16384});
  }
#endif  // HEXL_HAS_AVX512DQ
}

}  // namespace hexl
}  // namespace intel
