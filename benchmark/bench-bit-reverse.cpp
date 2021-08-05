// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "hexl/util/aligned-allocator.hpp"
#include "number-theory/bit-reverse-internal.hpp"

namespace intel {
namespace hexl {

// state[0] is the degree
static void BM_BitReverseReference(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);

  AlignedVector64<uint64_t> op1(input_size, 1);

  for (auto _ : state) {
    BitReverseReference(op1.data(), op1.size());
  }
}

BENCHMARK(BM_BitReverseReference)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

// state[0] is the degree
static void BM_BitReverseNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t bit_size = Log2(input_size);

  AlignedVector64<uint64_t> op1(input_size, 1);

  for (auto _ : state) {
    BitReverseNative(op1.data(), op1.size(), bit_size);
    // switch (bit_size) {
    //   case 10: {
    //     BitReverseNative<10>(op1.data());
    //     break;
    //   }
    //   case 12: {
    //     BitReverseNative<12>(op1.data());
    //     break;
    //   }
    //   case 14: {
    //     BitReverseNative<14>(op1.data());
    //     break;
    //   }
    //   case 15: {
    //     BitReverseNative<15>(op1.data());
    //     break;
    //   }
    // }
  }
}

BENCHMARK(BM_BitReverseNative)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384})
    ->Args({32768});

}  // namespace hexl
}  // namespace intel
