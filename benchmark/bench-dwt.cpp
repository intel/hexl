// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "hexl/dwt/dwt-native.hpp"
#include "hexl/dwt/dwt.hpp"
#include "hexl/dwt/fwd-dwt-avx512.hpp"
#include "hexl/dwt/inv-dwt-avx512.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

// Roots of unity
//=================================================================

static void BM_DWTComplexRootsOfUnity(benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  for (auto _ : state) {
    DWT dwt(dwt_size, nullptr);
  }
}

BENCHMARK(BM_DWTComplexRootsOfUnity)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

// Forward transforms
//=================================================================

static void BM_FwdDWTNativeRadix2InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_DWT_ToBitReverseRadix2(input.data(), input.data(),
                                   root_powers.data(), dwt_size);
  }
}

BENCHMARK(BM_FwdDWTNativeRadix2InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdDWTNativeRadix2InPlaceSmallScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 10;
  const double scalar = scale / static_cast<double>(dwt_size);
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_DWT_ToBitReverseRadix2(input.data(), input.data(),
                                   root_powers.data(), dwt_size, &scalar);
  }
}

BENCHMARK(BM_FwdDWTNativeRadix2InPlaceSmallScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdDWTNativeRadix2InPlaceLargeScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double scalar = scale / static_cast<double>(dwt_size);
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_DWT_ToBitReverseRadix2(input.data(), input.data(),
                                   root_powers.data(), dwt_size, &scalar);
  }
}

BENCHMARK(BM_FwdDWTNativeRadix2InPlaceLargeScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdDWTNativeRadix2CopyUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> output(dwt_size);
  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_DWT_ToBitReverseRadix2(output.data(), input.data(),
                                   root_powers.data(), dwt_size);
  }
}

BENCHMARK(BM_FwdDWTNativeRadix2CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdDWTNativeRadix2CopyLargeScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double scalar = scale / static_cast<double>(dwt_size);
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> output(dwt_size);
  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_DWT_ToBitReverseRadix2(output.data(), input.data(),
                                   root_powers.data(), dwt_size, &scalar);
  }
}

BENCHMARK(BM_FwdDWTNativeRadix2CopyLargeScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

// Inverse Transforms
//=================================================================

static void BM_InvDWTNativeRadix2InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_DWT_FromBitReverseRadix2(input.data(), input.data(),
                                     inv_root_powers.data(), dwt_size);
  }
}

BENCHMARK(BM_InvDWTNativeRadix2InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvDWTNativeRadix2InPlaceSmallScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 10;
  const double inv_scale = 1.0 / scale;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_DWT_FromBitReverseRadix2(input.data(), input.data(),
                                     inv_root_powers.data(), dwt_size,
                                     &inv_scale);
  }
}

BENCHMARK(BM_InvDWTNativeRadix2InPlaceSmallScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvDWTNativeRadix2InPlaceLargeScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double inv_scale = 1.0 / scale;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_DWT_FromBitReverseRadix2(input.data(), input.data(),
                                     inv_root_powers.data(), dwt_size,
                                     &inv_scale);
  }
}

BENCHMARK(BM_InvDWTNativeRadix2InPlaceLargeScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvDWTNativeRadix2CopyUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> output(dwt_size);
  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_DWT_FromBitReverseRadix2(output.data(), input.data(),
                                     inv_root_powers.data(), dwt_size);
  }
}

BENCHMARK(BM_InvDWTNativeRadix2CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvDWTNativeRadix2CopyScaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double inv_scale = 1.0 / scale;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<std::complex<double>> output(dwt_size);
  AlignedVector64<std::complex<double>> input(dwt_size);
  for (size_t i = 0; i < dwt_size; i++) {
    input[i] =
        std::complex<double>(GenerateInsecureUniformRealRandomValue(0, bound),
                             GenerateInsecureUniformRealRandomValue(0, bound));
  }

  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_DWT_FromBitReverseRadix2(output.data(), input.data(),
                                     inv_root_powers.data(), dwt_size,
                                     &inv_scale);
  }
}

BENCHMARK(BM_InvDWTNativeRadix2CopyScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ

static void BM_FwdDWTAVX512InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<double> input(2 * dwt_size);
  input = GenerateInsecureUniformRealRandomValues(2 * dwt_size, 0, bound);

  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_DWT_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], dwt_size, 0, 0);
  }
}

BENCHMARK(BM_FwdDWTAVX512InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdDWTAVX512InPlaceScaled(benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double scalar = scale / static_cast<double>(dwt_size);
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<double> input(2 * dwt_size);
  input = GenerateInsecureUniformRealRandomValues(2 * dwt_size, 0, bound);

  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_DWT_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], dwt_size, &scalar);
  }
}

BENCHMARK(BM_FwdDWTAVX512InPlaceScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdDWTAVX512CopyUnscaled(benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<double> output(2 * dwt_size);
  AlignedVector64<double> input(2 * dwt_size);
  input = GenerateInsecureUniformRealRandomValues(2 * dwt_size, 0, bound);

  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_DWT_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], dwt_size);
  }
}

BENCHMARK(BM_FwdDWTAVX512CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_FwdDWTAVX512CopyScaled(benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double scalar = scale / static_cast<double>(dwt_size);
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<double> output(2 * dwt_size);
  AlignedVector64<double> input(2 * dwt_size);
  input = GenerateInsecureUniformRealRandomValues(2 * dwt_size, 0, bound);

  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();

  for (auto _ : state) {
    Forward_DWT_ToBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], dwt_size, &scalar);
  }
}

BENCHMARK(BM_FwdDWTAVX512CopyScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvDWTAVX512InPlaceUnscaled(
    benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<double> input(2 * dwt_size);
  input = GenerateInsecureUniformRealRandomValues(2 * dwt_size, 0, bound);

  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_DWT_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], dwt_size);
  }
}

BENCHMARK(BM_InvDWTAVX512InPlaceUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvDWTAVX512InPlaceScaled(benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double inv_scale = 1.0 / scale;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<double> input(2 * dwt_size);
  input = GenerateInsecureUniformRealRandomValues(2 * dwt_size, 0, bound);

  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_DWT_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], dwt_size,
        &inv_scale);
  }
}

BENCHMARK(BM_InvDWTAVX512InPlaceScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvDWTAVX512CopyUnscaled(benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<double> output(2 * dwt_size);
  AlignedVector64<double> input(2 * dwt_size);
  input = GenerateInsecureUniformRealRandomValues(2 * dwt_size, 0, bound);

  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_DWT_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], dwt_size);
  }
}

BENCHMARK(BM_InvDWTAVX512CopyUnscaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

static void BM_InvDWTAVX512CopyScaled(benchmark::State& state) {  //  NOLINT
  const size_t dwt_size = state.range(0);
  const size_t bound = 1 << 30;
  const double scale = 1.3611294676837539e+39;  // (1 << 130)
  const double inv_scale = 1.0 / scale;
  DWT dwt(dwt_size, nullptr);

  AlignedVector64<double> output(2 * dwt_size);
  AlignedVector64<double> input(2 * dwt_size);
  input = GenerateInsecureUniformRealRandomValues(2 * dwt_size, 0, bound);

  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  for (auto _ : state) {
    Inverse_DWT_FromBitReverseAVX512(
        input.data(), input.data(),
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], dwt_size,
        &inv_scale);
  }
}

BENCHMARK(BM_InvDWTAVX512CopyScaled)
    ->Unit(benchmark::kMicrosecond)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

#endif

//=================================================================

}  // namespace hexl
}  // namespace intel
