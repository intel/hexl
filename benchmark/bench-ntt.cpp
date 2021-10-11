// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#include "ntt/ntt-internal.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

// Forward transforms

//=================================================================

static void BM_FwdNTTNativeRadix2InPlace(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  for (auto _ : state) {
    ForwardTransformToBitReverseRadix2(
        input.data(), input.data(), ntt_size, modulus,
        ntt.GetRootOfUnityPowers().data(),
        ntt.GetPrecon64RootOfUnityPowers().data(), 2, 1);
  }
}

BENCHMARK(BM_FwdNTTNativeRadix2InPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
//=================================================================

static void BM_FwdNTTNativeRadix2Copy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  AlignedVector64<uint64_t> output(ntt_size, 1);
  NTT ntt(ntt_size, modulus);

  for (auto _ : state) {
    ForwardTransformToBitReverseRadix2(
        output.data(), input.data(), ntt_size, modulus,
        ntt.GetRootOfUnityPowers().data(),
        ntt.GetPrecon64RootOfUnityPowers().data(), 2, 1);
  }
}

BENCHMARK(BM_FwdNTTNativeRadix2Copy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
//=================================================================

static void BM_FwdNTTNativeRadix4InPlace(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  for (auto _ : state) {
    ForwardTransformToBitReverseRadix4(
        input.data(), input.data(), ntt_size, modulus,
        ntt.GetRootOfUnityPowers().data(),
        ntt.GetPrecon64RootOfUnityPowers().data(), 2, 1);
  }
}

BENCHMARK(BM_FwdNTTNativeRadix4InPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
//=================================================================

static void BM_FwdNTTNativeRadix4Copy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  AlignedVector64<uint64_t> output(ntt_size, 1);
  NTT ntt(ntt_size, modulus);

  for (auto _ : state) {
    ForwardTransformToBitReverseRadix4(
        output.data(), input.data(), ntt_size, modulus,
        ntt.GetRootOfUnityPowers().data(),
        ntt.GetPrecon64RootOfUnityPowers().data(), 2, 1);
  }
}

BENCHMARK(BM_FwdNTTNativeRadix4Copy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512IFMA
// state[0] is the degree
static void BM_FwdNTT_AVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus_bits = 49;
  size_t modulus = GeneratePrimes(1, modulus_bits, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt.GetAVX512RootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetAVX512Precon52RootOfUnityPowers();

  for (auto _ : state) {
    ForwardTransformToBitReverseAVX512<NTT::s_ifma_shift_bits>(
        input.data(), input.data(), ntt_size, modulus, root_of_unity.data(),
        precon_root_of_unity.data(), 2, 1);
  }
}

BENCHMARK(BM_FwdNTT_AVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_FwdNTT_AVX512IFMALazy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus_bits = 49;
  size_t modulus = GeneratePrimes(1, modulus_bits, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt.GetAVX512RootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetAVX512Precon52RootOfUnityPowers();

  for (auto _ : state) {
    ForwardTransformToBitReverseAVX512<NTT::s_ifma_shift_bits>(
        input.data(), input.data(), ntt_size, modulus, root_of_unity.data(),
        precon_root_of_unity.data(), 4, 4);
  }
}

BENCHMARK(BM_FwdNTT_AVX512IFMALazy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#endif

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
// state[1] is the output modulus factor
static void BM_FwdNTT_AVX512DQ_32(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  uint64_t output_mod_factor = state.range(1);
  size_t modulus_bits = 29;
  size_t modulus = GeneratePrimes(1, modulus_bits, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt.GetAVX512RootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetAVX512Precon32RootOfUnityPowers();
  for (auto _ : state) {
    ForwardTransformToBitReverseAVX512<32>(
        input.data(), input.data(), ntt_size, modulus, root_of_unity.data(),
        precon_root_of_unity.data(), 4, output_mod_factor);
  }
}

BENCHMARK(BM_FwdNTT_AVX512DQ_32)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024, 1})
    ->Args({1024, 4})
    ->Args({4096, 1})
    ->Args({4096, 4})
    ->Args({16384, 1})
    ->Args({16384, 4});

// state[0] is the degree
// state[1] is the output modulus factor
static void BM_FwdNTT_AVX512DQ_64(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  uint64_t output_mod_factor = state.range(1);
  size_t modulus_bits = 55;
  size_t modulus = GeneratePrimes(1, modulus_bits, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity =
      ntt.GetAVX512RootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetAVX512Precon64RootOfUnityPowers();
  for (auto _ : state) {
    ForwardTransformToBitReverseAVX512<64>(
        input.data(), input.data(), ntt_size, modulus, root_of_unity.data(),
        precon_root_of_unity.data(), 4, output_mod_factor);
  }
}

BENCHMARK(BM_FwdNTT_AVX512DQ_64)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024, 1})
    ->Args({1024, 4})
    ->Args({4096, 1})
    ->Args({4096, 4})
    ->Args({16384, 1})
    ->Args({16384, 4});

#endif

//=================================================================

// state[0] is the degree
static void BM_FwdNTTInPlace(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 61, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  for (auto _ : state) {
    ntt.ComputeForward(input.data(), input.data(), 1, 1);
  }
}

BENCHMARK(BM_FwdNTTInPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_FwdNTTCopy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  AlignedVector64<uint64_t> output(ntt_size, 1);
  NTT ntt(ntt_size, modulus);

  for (auto _ : state) {
    ntt.ComputeForward(input.data(), output.data(), 1, 1);
  }
}

BENCHMARK(BM_FwdNTTCopy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvNTTInPlace(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  for (auto _ : state) {
    ntt.ComputeInverse(input.data(), input.data(), 2, 1);
  }
}

BENCHMARK(BM_InvNTTInPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_InvNTTCopy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  AlignedVector64<uint64_t> output(ntt_size, 1);
  NTT ntt(ntt_size, modulus);

  for (auto _ : state) {
    ntt.ComputeInverse(input.data(), output.data(), 2, 1);
  }
}

BENCHMARK(BM_InvNTTCopy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// Inverse transforms

static void BM_InvNTTNativeRadix2InPlace(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity = ntt.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetPrecon64InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseRadix2(input.data(), input.data(), ntt_size,
                                         modulus, root_of_unity.data(),
                                         precon_root_of_unity.data(), 1, 1);
  }
}

BENCHMARK(BM_InvNTTNativeRadix2InPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvNTTNativeRadix2Copy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  AlignedVector64<uint64_t> output(ntt_size, 1);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity = ntt.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetPrecon64InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseRadix2(output.data(), input.data(), ntt_size,
                                         modulus, root_of_unity.data(),
                                         precon_root_of_unity.data(), 1, 1);
  }
}

BENCHMARK(BM_InvNTTNativeRadix2Copy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvNTTNativeRadix4InPlace(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity = ntt.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetPrecon64InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseRadix4(input.data(), input.data(), ntt_size,
                                         modulus, root_of_unity.data(),
                                         precon_root_of_unity.data(), 1, 1);
  }
}

BENCHMARK(BM_InvNTTNativeRadix4InPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvNTTNativeRadix4Copy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  AlignedVector64<uint64_t> output(ntt_size, 1);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity = ntt.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetPrecon64InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseRadix4(output.data(), input.data(), ntt_size,
                                         modulus, root_of_unity.data(),
                                         precon_root_of_unity.data(), 1, 1);
  }
}

BENCHMARK(BM_InvNTTNativeRadix4Copy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512IFMA
// state[0] is the degree
static void BM_InvNTT_AVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 49, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity = ntt.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetPrecon52InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseAVX512<NTT::s_ifma_shift_bits>(
        input.data(), input.data(), ntt_size, modulus, root_of_unity.data(),
        precon_root_of_unity.data(), 1, 1);
  }
}

BENCHMARK(BM_InvNTT_AVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_InvNTT_AVX512IFMALazy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t modulus = GeneratePrimes(1, 49, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity = ntt.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetPrecon52InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseAVX512<NTT::s_ifma_shift_bits>(
        input.data(), input.data(), ntt_size, modulus, root_of_unity.data(),
        precon_root_of_unity.data(), 2, 2);
  }
}

BENCHMARK(BM_InvNTT_AVX512IFMALazy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

#endif

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
// state[0] is the degree
static void BM_InvNTT_AVX512DQ_32(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  uint64_t output_mod_factor = state.range(1);
  size_t modulus = GeneratePrimes(1, 29, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity = ntt.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetPrecon32InvRootOfUnityPowers();

  for (auto _ : state) {
    InverseTransformFromBitReverseAVX512<32>(
        input.data(), input.data(), ntt_size, modulus, root_of_unity.data(),
        precon_root_of_unity.data(), output_mod_factor, output_mod_factor);
  }
}

BENCHMARK(BM_InvNTT_AVX512DQ_32)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024, 1})
    ->Args({1024, 2})
    ->Args({4096, 1})
    ->Args({4096, 2})
    ->Args({16384, 1})
    ->Args({16384, 2});

static void BM_InvNTT_AVX512DQ_64(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  uint64_t output_mod_factor = state.range(1);
  size_t modulus = GeneratePrimes(1, 61, true, ntt_size)[0];

  auto input = GenerateInsecureUniformRandomValues(ntt_size, 0, modulus);
  NTT ntt(ntt_size, modulus);

  const AlignedVector64<uint64_t> root_of_unity = ntt.GetInvRootOfUnityPowers();
  const AlignedVector64<uint64_t> precon_root_of_unity =
      ntt.GetPrecon64InvRootOfUnityPowers();

  for (auto _ : state) {
    InverseTransformFromBitReverseAVX512<NTT::s_default_shift_bits>(
        input.data(), input.data(), ntt_size, modulus, root_of_unity.data(),
        precon_root_of_unity.data(), output_mod_factor, output_mod_factor);
  }
}

BENCHMARK(BM_InvNTT_AVX512DQ_64)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024, 1})
    ->Args({1024, 2})
    ->Args({4096, 1})
    ->Args({4096, 2})
    ->Args({16384, 1})
    ->Args({16384, 2});
#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
