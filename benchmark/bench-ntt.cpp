// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "intel-hexl/ntt/ntt.hpp"
#include "logging/logging.hpp"
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#include "ntt/ntt-internal.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"

namespace intel {
namespace hexl {

// Forward transforms

//=================================================================

// state[0] is the degree
static void BM_FwdNTTNative(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 45, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  for (auto _ : state) {
    ForwardTransformToBitReverse64(
        input.data(), ntt_size, prime, ntt_impl.GetRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64RootOfUnityPowers().data(), 2, 1);
  }
}

BENCHMARK(BM_FwdNTTNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});
//=================================================================

#ifdef HEXL_HAS_AVX512IFMA
// state[0] is the degree
static void BM_FwdNTT_AVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime_bits = 49;
  size_t prime = GeneratePrimes(1, prime_bits, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt_impl.GetRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon52RootOfUnityPowers();

  for (auto _ : state) {
    ForwardTransformToBitReverseAVX512<NTT::NTTImpl::s_ifma_shift_bits>(
        input.data(), ntt_size, prime, root_of_unity.data(),
        precon_root_of_unity.data(), 2, 1);
  }
}

BENCHMARK(BM_FwdNTT_AVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_FwdNTT_AVX512IFMALazy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime_bits = 49;
  size_t prime = GeneratePrimes(1, prime_bits, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt_impl.GetRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon52RootOfUnityPowers();

  for (auto _ : state) {
    ForwardTransformToBitReverseAVX512<NTT::NTTImpl::s_ifma_shift_bits>(
        input.data(), ntt_size, prime, root_of_unity.data(),
        precon_root_of_unity.data(), 2, 4);
  }
}

BENCHMARK(BM_FwdNTT_AVX512IFMALazy)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

//=================================================================

static void BM_FwdNTT_AVX512IFMAButterfly(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = 4096;
  size_t prime_bits = 49;
  size_t prime = GeneratePrimes(1, prime_bits, ntt_size)[0];

  NTT::NTTImpl ntt_impl(ntt_size, prime);

  __m512i X = _mm512_set1_epi64(prime - 3);
  __m512i Y = _mm512_set1_epi64(prime / 2);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt_impl.GetRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon52RootOfUnityPowers();

  __m512i W = _mm512_set1_epi64(root_of_unity[1]);
  __m512i Wprecon = _mm512_set1_epi64(precon_root_of_unity[1]);
  __m512i neg_p = _mm512_set1_epi64(-static_cast<int64_t>(prime));
  __m512i twice_p = _mm512_set1_epi64(prime + prime);

  for (auto _ : state) {
    for (size_t i = 0; i < 1000000; ++i) {
      benchmark::DoNotOptimize(i);
      FwdButterfly<52, false>(&X, &Y, W, Wprecon, neg_p, twice_p);
    }
  }
}

BENCHMARK(BM_FwdNTT_AVX512IFMAButterfly)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

#endif

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
// state[1] is approximately the number of bits in the coefficient modulus
static void BM_FwdNTT_AVX512DQ(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  uint64_t output_mod_factor = state.range(1);
  size_t prime_bits = 61;
  size_t prime = GeneratePrimes(1, prime_bits, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt_impl.GetRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon64RootOfUnityPowers();
  for (auto _ : state) {
    ForwardTransformToBitReverseAVX512<NTT::NTTImpl::s_default_shift_bits>(
        input.data(), ntt_size, prime, root_of_unity.data(),
        precon_root_of_unity.data(), 4, output_mod_factor);
  }
}

BENCHMARK(BM_FwdNTT_AVX512DQ)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024, 1})
    ->Args({1024, 4})
    ->Args({4096, 1})
    ->Args({4096, 4})
    ->Args({8192, 1})
    ->Args({8192, 4})
    ->Args({16384, 1})
    ->Args({16384, 4});

#endif

//=================================================================

// state[0] is the degree
static void BM_FwdNTTInPlace(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 61, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  NTT ntt(ntt_size, prime);

  for (auto _ : state) {
    ntt.ComputeForward(input.data(), input.data(), 1, 1);
  }
}

BENCHMARK(BM_FwdNTTInPlace)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_FwdNTTCopy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 61, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  AlignedVector64<uint64_t> output(ntt_size, 1);
  NTT ntt(ntt_size, prime);

  for (auto _ : state) {
    ntt.ComputeForward(input.data(), output.data(), 1, 1);
  }
}

BENCHMARK(BM_FwdNTTCopy)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

//=================================================================

// Inverse transforms

// state[0] is the degree
static void BM_InvNTTNative(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 45, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt_impl.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon64InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverse64(input.data(), ntt_size, prime,
                                     root_of_unity.data(),
                                     precon_root_of_unity.data(), 1, 1);
  }
}

BENCHMARK(BM_InvNTTNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512IFMA
// state[0] is the degree
static void BM_InvNTT_AVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 49, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt_impl.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon52InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseAVX512<NTT::NTTImpl::s_ifma_shift_bits>(
        input.data(), ntt_size, prime, root_of_unity.data(),
        precon_root_of_unity.data(), 1, 1);
  }
}

BENCHMARK(BM_InvNTT_AVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_InvNTT_AVX512IFMALazy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 49, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt_impl.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon52InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseAVX512<NTT::NTTImpl::s_ifma_shift_bits>(
        input.data(), ntt_size, prime, root_of_unity.data(),
        precon_root_of_unity.data(), 1, 4);
  }
}

BENCHMARK(BM_InvNTT_AVX512IFMALazy)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

//=================================================================

static void BM_InvNTT_AVX512IFMAButterfly(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = 4096;
  size_t prime_bits = 49;
  size_t prime = GeneratePrimes(1, prime_bits, ntt_size)[0];

  NTT::NTTImpl ntt_impl(ntt_size, prime);

  __m512i X = _mm512_set1_epi64(prime - 3);
  __m512i Y = _mm512_set1_epi64(prime / 2);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt_impl.GetRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon52InvRootOfUnityPowers();

  __m512i W = _mm512_set1_epi64(root_of_unity[1]);
  __m512i Wprecon = _mm512_set1_epi64(precon_root_of_unity[1]);
  __m512i neg_p = _mm512_set1_epi64(-static_cast<int64_t>(prime));
  __m512i twice_p = _mm512_set1_epi64(prime + prime);

  for (auto _ : state) {
    for (size_t i = 0; i < 1000000; ++i) {
      benchmark::DoNotOptimize(i);
      InvButterfly<52, false>(&X, &Y, W, Wprecon, neg_p, twice_p);
    }
  }
}

BENCHMARK(BM_InvNTT_AVX512IFMAButterfly)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(1.0);
#endif

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_InvNTT_AVX512DQ(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  uint64_t output_mod_factor = state.range(1);
  size_t prime = GeneratePrimes(1, 62, ntt_size)[0];

  AlignedVector64<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt_impl.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon64InvRootOfUnityPowers();

  for (auto _ : state) {
    InverseTransformFromBitReverseAVX512<NTT::NTTImpl::s_default_shift_bits>(
        input.data(), ntt_size, prime, root_of_unity.data(),
        precon_root_of_unity.data(), output_mod_factor, output_mod_factor);
  }
}

BENCHMARK(BM_InvNTT_AVX512DQ)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024, 1})
    ->Args({1024, 2})
    ->Args({4096, 1})
    ->Args({4096, 2})
    ->Args({8192, 1})
    ->Args({8192, 2})
    ->Args({16384, 1})
    ->Args({16384, 2});
#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
