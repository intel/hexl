// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <unistd.h>
#include "hexl/util/thread-pool.hpp"
#include "eltwise/eltwise-add-mod-avx512.hpp"
#include "eltwise/eltwise-add-mod-internal.hpp"
#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {


// state[0] is the mode
static void BM_MT_Native(benchmark::State& state) {  //  NOLINT
  size_t threads = state.range(0);

  for(auto _ : state){
      std::this_thread::sleep_for(std::chrono::microseconds(5));
    
  }
}

BENCHMARK(BM_MT_Native)
    ->Unit(benchmark::kMicrosecond)
    ->Args({32})
    ->Args({64})
    ->Args({128})
    ->Args({256})
    ->Args({512})
    ->Args({1024})
    ->Args({2048});

// state[0] is the mode
static void BM_MT_OMP(benchmark::State& state) {  //  NOLINT
  size_t threads = state.range(0);

  for(auto _ : state){

#pragma omp parallel num_threads(eltwise_num_threads)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(5));
      
    }
  }
}

BENCHMARK(BM_MT_OMP)
    ->Unit(benchmark::kMicrosecond)
    ->Args({32})
    ->Args({64})
    ->Args({128})
    ->Args({256})
    ->Args({512})
    ->Args({1024})
    ->Args({2048});

// state[0] is the mode
static void BM_MT_TP(benchmark::State& state) {  //  NOLINT
  size_t threads = state.range(0);
  for(auto _ : state){
    ThreadPoolExecutor::SetNumberOfThreads(eltwise_num_threads);
    ThreadPoolExecutor::AddParallelTask([](s_thread_info_t* thread_handler) {
        std::this_thread::sleep_for(std::chrono::microseconds(5));
      
    });
    ThreadPoolExecutor::SetBarrier();
  }
}

BENCHMARK(BM_MT_TP)
    ->Unit(benchmark::kMicrosecond)
    ->Args({32})
    ->Args({64})
    ->Args({128})
    ->Args({256})
    ->Args({512})
    ->Args({1024})
    ->Args({2048});

// state[0] is the degree
static void BM_EltwiseVectorVectorAddModNative(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModNative(output.data(), input1.data(), input2.data(), input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseVectorVectorAddModNative)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseVectorVectorAddModAVX512(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);
  for (auto _ : state) {
    EltwiseAddModAVX512(output.data(), input1.data(), input2.data(), input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseVectorVectorAddModAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384})
    ->Args({32768})
    ->Args({65536})
    ->Args({131072})
    ->Args({262144});

// state[0] is the degree
/*
static void BM_EltwiseVectorVectorAddModAVX512_TBB(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModAVX512_TBB(output.data(), input1.data(), input2.data(),
                            input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseVectorVectorAddModAVX512_TBB)
    ->Unit(benchmark::kMicrosecond)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384})
    ->Args({32768})
    ->Args({65536})
    ->Args({131072})
    ->Args({262144});
*/
// state[0] is the degree
static void BM_EltwiseVectorVectorAddModAVX512_OMP(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModAVX512_OMP(output.data(), input1.data(), input2.data(),
                            input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseVectorVectorAddModAVX512_OMP)
    ->Unit(benchmark::kMicrosecond)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384})
    ->Args({32768})
    ->Args({65536})
    ->Args({131072})
    ->Args({262144});

static void BM_EltwiseVectorVectorAddModAVX512_TP(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModAVX512_TP(output.data(), input1.data(), input2.data(),
                            input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseVectorVectorAddModAVX512_TP)
    ->Unit(benchmark::kMicrosecond)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384})
    ->Args({32768})
    ->Args({65536})
    ->Args({131072})
    ->Args({262144});
#endif

//=================================================================
// state[0] is the degree
static void BM_EltwiseVectorScalarAddModNative(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformIntRandomValue(0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModNative(output.data(), input1.data(), input2, input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseVectorScalarAddModNative)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseVectorScalarAddModAVX512(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  auto input1 = GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformIntRandomValue(0, modulus);
  AlignedVector64<uint64_t> output(input_size, 0);

  for (auto _ : state) {
    EltwiseAddModAVX512(output.data(), input1.data(), input2, input_size,
                        modulus);
  }
}

BENCHMARK(BM_EltwiseVectorScalarAddModAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384})
    ->Args({32768})
    ->Args({65536})
    ->Args({131072})
    ->Args({262144});
#endif

}  // namespace hexl
}  // namespace intel
