// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "hexl/fft/fft-native.hpp"
#include "hexl/fft/fft.hpp"
#include "hexl/fft/fwd-fft-avx512.hpp"
#include "hexl/fft/inv-fft-avx512.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

// Roots of unity
//=================================================================

static void BM_FFTComplexRootsOfUnity(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  for (auto _ : state) {
    FFT fft(fft_size);
  }
}

BENCHMARK(BM_FFTComplexRootsOfUnity)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

// Forward transforms
//=================================================================

static void BM_FwdFFTNativeRadix2InPlace(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size);

  AlignedVector64<std::complex<double>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_Radix2(input.data(), input.data(), root_powers.data(),
                       fft_size);
  }
}

BENCHMARK(BM_FwdFFTNativeRadix2InPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTNativeRadix2Copy(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size);

  AlignedVector64<std::complex<double>> output(fft_size);
  AlignedVector64<std::complex<double>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_Radix2(output.data(), input.data(), root_powers.data(),
                       fft_size);
  }
}

BENCHMARK(BM_FwdFFTNativeRadix2Copy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

// Inverse Transforms
//=================================================================

static void BM_InvFFTNativeRadix2InPlace(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size);

  AlignedVector64<std::complex<double>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_Radix2(input.data(), input.data(), inv_root_powers.data(),
                       fft_size);
  }
}

BENCHMARK(BM_InvFFTNativeRadix2InPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTNativeRadix2Copy(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size);

  AlignedVector64<std::complex<double>> output(fft_size);
  AlignedVector64<std::complex<double>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_Radix2(output.data(), input.data(), inv_root_powers.data(),
                       fft_size);
  }
}

BENCHMARK(BM_InvFFTNativeRadix2Copy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ

static void BM_FwdFFTAVX512InPlace(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size);

  AlignedVector64<double> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<double> root_powers = fft.GetInterleavedComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_AVX512(input.data(), input.data(), root_powers.data(),
                       fft_size);
  }
}

BENCHMARK(BM_FwdFFTAVX512InPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTAVX512Copy(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size);

  AlignedVector64<double> output(2 * fft_size);
  AlignedVector64<double> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<double> root_powers = fft.GetInterleavedComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_AVX512(input.data(), input.data(), root_powers.data(),
                       fft_size);
  }
}

BENCHMARK(BM_FwdFFTAVX512Copy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTAVX512InPlace(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size);

  AlignedVector64<double> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<double> inv_root_powers =
      fft.GetInterleavedInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_AVX512(input.data(), input.data(), inv_root_powers.data(),
                       fft_size);
  }
}

BENCHMARK(BM_InvFFTAVX512InPlace)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTAVX512Copy(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size);

  AlignedVector64<double> output(2 * fft_size);
  AlignedVector64<double> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<double> inv_root_powers =
      fft.GetInterleavedInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_AVX512(input.data(), input.data(), inv_root_powers.data(),
                       fft_size);
  }
}

BENCHMARK(BM_InvFFTAVX512Copy)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
