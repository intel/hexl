// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "hexl/util/aligned-allocator.hpp"
#include "number-theory/bit-reverse-internal.hpp"

namespace intel {
namespace hexl {

//=================================================================

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

}  // namespace hexl
}  // namespace intel
