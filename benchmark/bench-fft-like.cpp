// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "hexl/experimental/fft-like/fft-like-native.hpp"
#include "hexl/experimental/fft-like/fft-like.hpp"
#include "hexl/experimental/fft-like/fwd-fft-like-avx512.hpp"
#include "hexl/experimental/fft-like/inv-fft-like-avx512.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

// Roots of unity
//=================================================================

static void BM_FFTLikeComplexRootsOfUnity(benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  for (auto _ : state) {
    FFTLike fft_like(fft_like_size, nullptr);
  }
}

BENCHMARK(BM_FFTLikeComplexRootsOfUnity)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

// Forward transforms
//=================================================================

static void BM_FwdFFTLikeNativeRadix2InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFTLike_ToBitReverseRadix2(input.data(), input.data(),
                                       root_powers.data(), fft_like_size);
  }
}

BENCHMARK(BM_FwdFFTLikeNativeRadix2InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTLikeNativeRadix2InPlaceSmallScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 10;
  const double scalar = scale / static_cast<double>(fft_like_size);
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFTLike_ToBitReverseRadix2(
        input.data(), input.data(), root_powers.data(), fft_like_size, &scalar);
  }
}

BENCHMARK(BM_FwdFFTLikeNativeRadix2InPlaceSmallScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTLikeNativeRadix2InPlaceLargeScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double scalar = scale / static_cast<double>(fft_like_size);
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFTLike_ToBitReverseRadix2(
        input.data(), input.data(), root_powers.data(), fft_like_size, &scalar);
  }
}

BENCHMARK(BM_FwdFFTLikeNativeRadix2InPlaceLargeScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTLikeNativeRadix2CopyUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> output(fft_like_size);
  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFTLike_ToBitReverseRadix2(output.data(), input.data(),
                                       root_powers.data(), fft_like_size);
  }
}

BENCHMARK(BM_FwdFFTLikeNativeRadix2CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTLikeNativeRadix2CopyLargeScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double scalar = scale / static_cast<double>(fft_like_size);
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> output(fft_like_size);
  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFTLike_ToBitReverseRadix2(output.data(), input.data(),
                                       root_powers.data(), fft_like_size,
                                       &scalar);
  }
}

BENCHMARK(BM_FwdFFTLikeNativeRadix2CopyLargeScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

// Inverse Transforms
//=================================================================

static void BM_InvFFTLikeNativeRadix2InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFTLike_FromBitReverseRadix2(input.data(), input.data(),
                                         inv_root_powers.data(), fft_like_size);
  }
}

BENCHMARK(BM_InvFFTLikeNativeRadix2InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTLikeNativeRadix2InPlaceSmallScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 10;
  const double inv_scale = 1.0 / scale;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFTLike_FromBitReverseRadix2(input.data(), input.data(),
                                         inv_root_powers.data(), fft_like_size,
                                         &inv_scale);
  }
}

BENCHMARK(BM_InvFFTLikeNativeRadix2InPlaceSmallScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTLikeNativeRadix2InPlaceLargeScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double inv_scale = 1.0 / scale;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFTLike_FromBitReverseRadix2(input.data(), input.data(),
                                         inv_root_powers.data(), fft_like_size,
                                         &inv_scale);
  }
}

BENCHMARK(BM_InvFFTLikeNativeRadix2InPlaceLargeScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTLikeNativeRadix2CopyUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> output(fft_like_size);
  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFTLike_FromBitReverseRadix2(output.data(), input.data(),
                                         inv_root_powers.data(), fft_like_size);
  }
}

BENCHMARK(BM_InvFFTLikeNativeRadix2CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTLikeNativeRadix2CopyScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double inv_scale = 1.0 / scale;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<std::complex<double>> output(fft_like_size);
  AlignedVector64<std::complex<double>> input(fft_like_size);
  for (size_t i = 0; i < fft_like_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFTLike_FromBitReverseRadix2(output.data(), input.data(),
                                         inv_root_powers.data(), fft_like_size,
                                         &inv_scale);
  }
}

BENCHMARK(BM_InvFFTLikeNativeRadix2CopyScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ

static void BM_FwdFFTLikeAVX512InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<double> input(2 * fft_like_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_like_size, 0, bound);

  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFTLike_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], fft_like_size, 0,
        0);
  }
}

BENCHMARK(BM_FwdFFTLikeAVX512InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTLikeAVX512InPlaceScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double scalar = scale / static_cast<double>(fft_like_size);
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<double> input(2 * fft_like_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_like_size, 0, bound);

  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFTLike_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], fft_like_size,
        &scalar);
  }
}

BENCHMARK(BM_FwdFFTLikeAVX512InPlaceScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTLikeAVX512CopyUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<double> output(2 * fft_like_size);
  AlignedVector64<double> input(2 * fft_like_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_like_size, 0, bound);

  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFTLike_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], fft_like_size);
  }
}

BENCHMARK(BM_FwdFFTLikeAVX512CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdFFTLikeAVX512CopyScaled(benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double scalar = scale / static_cast<double>(fft_like_size);
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<double> output(2 * fft_like_size);
  AlignedVector64<double> input(2 * fft_like_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_like_size, 0, bound);

  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_FFTLike_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], fft_like_size,
        &scalar);
  }
}

BENCHMARK(BM_FwdFFTLikeAVX512CopyScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTLikeAVX512InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<double> input(2 * fft_like_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_like_size, 0, bound);

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFTLike_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], fft_like_size);
  }
}

BENCHMARK(BM_InvFFTLikeAVX512InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTLikeAVX512InPlaceScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double inv_scale = 1.0 / scale;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<double> input(2 * fft_like_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_like_size, 0, bound);

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFTLike_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], fft_like_size,
        &inv_scale);
  }
}

BENCHMARK(BM_InvFFTLikeAVX512InPlaceScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTLikeAVX512CopyUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<double> output(2 * fft_like_size);
  AlignedVector64<double> input(2 * fft_like_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_like_size, 0, bound);

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFTLike_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], fft_like_size);
  }
}

BENCHMARK(BM_InvFFTLikeAVX512CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvFFTLikeAVX512CopyScaled(benchmark::State& state) {  //  NOLINT
  const size_t fft_like_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double inv_scale = 1.0 / scale;
  FFTLike fft_like(fft_like_size, nullptr);

  AlignedVector64<double> output(2 * fft_like_size);
  AlignedVector64<double> input(2 * fft_like_size);
  input = GenerateInsecureUniformRealRandomValues(2 * fft_like_size, 0, bound);

  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_FFTLike_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], fft_like_size,
        &inv_scale);
  }
}

BENCHMARK(BM_InvFFTLikeAVX512CopyScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
