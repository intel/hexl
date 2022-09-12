// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <immintrin.h>
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
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {


// state[0] is the mode
static void BM_MT_Native(benchmark::State& state) {  //  NOLINT
  
  for(auto _ : state){
    int sum = 0;
    for (int i = 0; i < 101; i++){
      sum++;
      sum = sum%77;
    }
    if (sum > 75) std::cout << "Bla"  << sum << std::endl;
  }
}

BENCHMARK(BM_MT_Native)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1});

// state[0] is the mode
static void BM_MT_OMP(benchmark::State& state) {  //  NOLINT
  size_t threads = state.range(0);
  size_t input_size = state.range(1);

  AlignedVector64<uint64_t> input_v(input_size, 7);
  uint64_t* input1 = input_v.data();
  AlignedVector64<uint64_t> input2_v(input_size, 7);
  uint64_t* input2 = input2_v.data();
  AlignedVector64<uint64_t> output_v(input_size, 7);
  uint64_t* output = output_v.data();
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(3));

  for(auto _ : state){

#pragma omp parallel num_threads(threads)
    {
      // Level 0
      /*
      int sum = 0;
      for (int i = 0; i < 101; i++){
        sum++;
        sum = sum%77;
      }
      if (sum > 75) std::cout << "Bla"  << sum << std::endl;
      */

      // Level 1
      /*
      int in_threads = omp_get_num_threads();
      int id = omp_get_thread_num();
      size_t n = input_size/in_threads;
      uint64_t* input1_p = input1 + n*id;
      uint64_t*  input2_p =input2 + n*id;
      uint64_t*  output_p = output + n*id;
      for (size_t i = 0; i < n; i++){

        *output_p = *input1_p + *input2_p;
        ++output_p;
        ++input1_p;
        ++input2_p;
      }
      */
      
      // Level 2
      ///*
      int in_threads = omp_get_num_threads();
      int id = omp_get_thread_num();
      size_t n = input_size/in_threads/8;
      const __m512i* input1_p = reinterpret_cast<__m512i*>(input1) + n*id;
      const __m512i* input2_p = reinterpret_cast<__m512i*>(input2) + n*id;
      __m512i* output_p = reinterpret_cast<__m512i*>(output) + n*id;
      for (size_t i = 0; i < n; i++){
      __m512i v_operand1 = _mm512_loadu_si512(input1_p);
      __m512i v_operand2 = _mm512_loadu_si512(input2_p);

      //__m512i v_result = _mm512_add_epi64(v_operand1, v_operand2);
      __m512i v_result = _mm512_hexl_small_add_mod_epi64(v_operand1, v_operand2, v_modulus);

      _mm512_storeu_si512(output_p, v_result);
        ++output_p;
        ++input1_p;
        ++input2_p;
      }
      //*/
    }
  }
}


// state[0] is the mode
static void BM_MT_TP(benchmark::State& state) {  //  NOLINT
  size_t threads = state.range(0);
  size_t input_size = state.range(1);
  AlignedVector64<uint64_t> input1_v(input_size, 7);
  uint64_t* input1 = input1_v.data();
  AlignedVector64<uint64_t> input2_v(input_size, 7);
  uint64_t* input2 = input2_v.data();
  AlignedVector64<uint64_t> output_v(input_size, 7);
  uint64_t* output = output_v.data();
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(3));
  ThreadPoolExecutor::SetNumberOfThreads(threads);

  for(auto _ : state){  
    ThreadPoolExecutor::AddParallelTask([=](int id, int in_threads) {

      // Level 0
      /*
      int sum = 0;
      for (int i = 0; i < 101; i++){
        sum++;
        sum = sum%77;
      }
      if (sum > 75) std::cout << "Bla"  << sum << std::endl;
      */
      
      // Level 1
      /*
      size_t n = input_size/in_threads;
      uint64_t* input1_p = input1 + n*id;
      uint64_t*  input2_p =input2 + n*id;
      uint64_t*  output_p = output + n*id;
      for (size_t i = 0; i < n; i++){
        *output_p = *input1_p + *input2_p;
        ++output_p;
        ++input1_p;
        ++input2_p;
      }
      */

      // Level 2
      ///*
      size_t n = input_size/in_threads/8;
      const __m512i* input1_p = reinterpret_cast<__m512i*>(input1) + n*id;
      const __m512i* input2_p = reinterpret_cast<__m512i*>(input2) + n*id;
      __m512i* output_p = reinterpret_cast<__m512i*>(output) + n*id;
      for (size_t i = 0; i < n; i++){
      __m512i v_operand1 = _mm512_loadu_si512(input1_p);
      __m512i v_operand2 = _mm512_loadu_si512(input2_p);

      //__m512i v_result = _mm512_add_epi64(v_operand1, v_operand2);
      __m512i v_result = _mm512_hexl_small_add_mod_epi64(v_operand1, v_operand2, v_modulus);

      _mm512_storeu_si512(output_p, v_result);
        ++output_p;
        ++input1_p;
        ++input2_p;
      }
      //*/
    });
    ThreadPoolExecutor::SetBarrier();
  }
}

BENCHMARK(BM_MT_OMP)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1, 262144})
    ->Args({2, 262144})
    ->Args({4, 262144})
    ->Args({8, 262144})
    ->Args({16, 262144})
    ->Args({32, 262144});

BENCHMARK(BM_MT_TP)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1, 262144})
    ->Args({2, 262144})
    ->Args({4, 262144})
    ->Args({8, 262144})
    ->Args({16, 262144})
    ->Args({32, 262144});

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
