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
    FFT fft(fft_size, nullptr);
  }
}

BENCHMARK(BM_FFTComplexRootsOfUnity)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

// Forward transforms
//=================================================================

static void BM_FwdFFTNativeRadix2InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_ToBitReverseRadix2(input.data(), input.data(),
                                   root_powers.data(), fft_size);
  }
}

BENCHMARK(BM_FwdFFTNativeRadix2InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTNativeRadix2InPlaceSmallScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 10;
  const double_t scalar = scale / static_cast<double_t>(fft_size);
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_ToBitReverseRadix2(input.data(), input.data(),
                                   root_powers.data(), fft_size, &scalar);
  }
}

BENCHMARK(BM_FwdFFTNativeRadix2InPlaceSmallScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTNativeRadix2InPlaceLargeScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
  const double_t scalar = scale / static_cast<double_t>(fft_size);
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_ToBitReverseRadix2(input.data(), input.data(),
                                   root_powers.data(), fft_size, &scalar);
  }
}

BENCHMARK(BM_FwdFFTNativeRadix2InPlaceLargeScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTNativeRadix2CopyUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> output(fft_size);
  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_ToBitReverseRadix2(output.data(), input.data(),
                                   root_powers.data(), fft_size);
  }
}

BENCHMARK(BM_FwdFFTNativeRadix2CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTNativeRadix2CopyLargeScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
  const double_t scalar = scale / static_cast<double_t>(fft_size);
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> output(fft_size);
  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_ToBitReverseRadix2(output.data(), input.data(),
                                   root_powers.data(), fft_size, &scalar);
  }
}

BENCHMARK(BM_FwdFFTNativeRadix2CopyLargeScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

// Inverse Transforms
//=================================================================

static void BM_InvFFTNativeRadix2InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_FromBitReverseRadix2(input.data(), input.data(),
                                     inv_root_powers.data(), fft_size);
  }
}

BENCHMARK(BM_InvFFTNativeRadix2InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTNativeRadix2InPlaceSmallScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 10;
  const double_t inv_scale = static_cast<double_t>(1.0) / scale;
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_FromBitReverseRadix2(input.data(), input.data(),
                                     inv_root_powers.data(), fft_size,
                                     &inv_scale);
  }
}

BENCHMARK(BM_InvFFTNativeRadix2InPlaceSmallScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTNativeRadix2InPlaceLargeScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
  const double_t inv_scale = static_cast<double_t>(1.0) / scale;
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_FromBitReverseRadix2(input.data(), input.data(),
                                     inv_root_powers.data(), fft_size,
                                     &inv_scale);
  }
}

BENCHMARK(BM_InvFFTNativeRadix2InPlaceLargeScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTNativeRadix2CopyUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> output(fft_size);
  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_FromBitReverseRadix2(output.data(), input.data(),
                                     inv_root_powers.data(), fft_size);
  }
}

BENCHMARK(BM_InvFFTNativeRadix2CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTNativeRadix2CopyScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
  const double_t inv_scale = static_cast<double_t>(1.0) / scale;
  FFT fft(fft_size, nullptr);

  AlignedVector64<std::complex<double_t>> output(fft_size);
  AlignedVector64<std::complex<double_t>> input(fft_size);
  for (size_t i = 0; i < fft_size; i++) {
    input[i] = std::complex<double_t>(
        GenerateInsecureUniformRealRandomValue(0, bound),
        GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double_t>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_FromBitReverseRadix2(output.data(), input.data(),
                                     inv_root_powers.data(), fft_size,
                                     &inv_scale);
  }
}

BENCHMARK(BM_InvFFTNativeRadix2CopyScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ

static void BM_FwdFFTAVX512InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size, nullptr);

  AlignedVector64<double_t> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<std::complex<double_t>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], fft_size, 0, 0);
  }
}

BENCHMARK(BM_FwdFFTAVX512InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTAVX512InPlaceScaled(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
  const double_t scalar = scale / static_cast<double_t>(fft_size);
  FFT fft(fft_size, nullptr);

  AlignedVector64<double_t> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<std::complex<double_t>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], fft_size, &scalar);
  }
}

BENCHMARK(BM_FwdFFTAVX512InPlaceScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTAVX512CopyUnscaled(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size, nullptr);

  AlignedVector64<double_t> output(2 * fft_size);
  AlignedVector64<double_t> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<std::complex<double_t>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], fft_size);
  }
}

BENCHMARK(BM_FwdFFTAVX512CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTAVX512CopyScaled(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
  const double_t scalar = scale / static_cast<double_t>(fft_size);
  FFT fft(fft_size, nullptr);

  AlignedVector64<double_t> output(2 * fft_size);
  AlignedVector64<double_t> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<std::complex<double_t>> root_powers =
      fft.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFT_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], fft_size, &scalar);
  }
}

BENCHMARK(BM_FwdFFTAVX512CopyScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTAVX512InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size, nullptr);

  AlignedVector64<double_t> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<std::complex<double_t>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], fft_size);
  }
}

BENCHMARK(BM_InvFFTAVX512InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTAVX512InPlaceScaled(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
  const double_t inv_scale = static_cast<double_t>(1.0) / scale;
  FFT fft(fft_size, nullptr);

  AlignedVector64<double_t> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<std::complex<double_t>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], fft_size,
        &inv_scale);
  }
}

BENCHMARK(BM_InvFFTAVX512InPlaceScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTAVX512CopyUnscaled(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  FFT fft(fft_size, nullptr);

  AlignedVector64<double_t> output(2 * fft_size);
  AlignedVector64<double_t> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<std::complex<double_t>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], fft_size);
  }
}

BENCHMARK(BM_InvFFTAVX512CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTAVX512CopyScaled(benchmark::State& state) {  //  NOLINT
  const size_t fft_size = state.range(0);
  const size_t bound = 1 << 30;
  const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
  const double_t inv_scale = static_cast<double_t>(1.0) / scale;
  FFT fft(fft_size, nullptr);

  AlignedVector64<double_t> output(2 * fft_size);
  AlignedVector64<double_t> input(2 * fft_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_size, 0, bound);

  AlignedVector64<std::complex<double_t>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFT_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], fft_size,
        &inv_scale);
  }
}

BENCHMARK(BM_InvFFTAVX512CopyScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
