// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <complex>

#include "hexl/experimental/fft-like/fft-like-native.hpp"
#include "hexl/experimental/fft-like/fft-like.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test/test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

TEST(FFTLike, ForwardInverseFFTLikeNative) {
  FFTLike fft_like(64, nullptr);
  AlignedVector64<std::complex<double>> root_powers =
      fft_like.GetComplexRootsOfUnity();
  AlignedVector64<std::complex<double>> inv_root_powers =
      fft_like.GetInvComplexRootsOfUnity();

  {  // Single Unscaled
    const uint64_t n = 64;
    const double data_bound = (1 << 30);
    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> result(n);

    operand[0] = std::complex<double>(
        GenerateInsecureUniformRealRandomValue(0, data_bound),
        GenerateInsecureUniformRealRandomValue(0, data_bound));

    Forward_FFTLike_ToBitReverseRadix2(result.data(), operand.data(),
                                       root_powers.data(), n);

    for (size_t i = 0; i < n; ++i) {
      CheckClose(operand[0], result[i], 0.5);
    }
  }

  {  // Single Scaled
    const uint64_t n = 64;
    const double scale = 1 << 16;
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 30);
    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> result(n);

    std::complex<double> value(
        GenerateInsecureUniformRealRandomValue(0, data_bound),
        GenerateInsecureUniformRealRandomValue(0, data_bound));
    operand[0] = value;
    value *= inv_scale;

    Forward_FFTLike_ToBitReverseRadix2(result.data(), operand.data(),
                                       root_powers.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      CheckClose(value, result[i], 0.5);
    }
  }

  {  // Zeros test
    const uint64_t n = 64;
    const double scale = 1 << 16;
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;

    AlignedVector64<std::complex<double>> operand(n, {0, 0});
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    Inverse_FFTLike_FromBitReverseRadix2(transformed.data(), operand.data(),
                                         inv_root_powers.data(), n, &scalar);

    Forward_FFTLike_ToBitReverseRadix2(result.data(), transformed.data(),
                                       root_powers.data(), n, &inv_scale);

    CheckClose(operand, result, 0.5);
  }

  {  // Large Scaled
    const uint64_t n = 64;
    const double scale = 1099511627776;  // (1 << 40)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 30);

    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    Inverse_FFTLike_FromBitReverseRadix2(transformed.data(), operand.data(),
                                         inv_root_powers.data(), n, &scalar);

    Forward_FFTLike_ToBitReverseRadix2(result.data(), transformed.data(),
                                       root_powers.data(), n, &inv_scale);

    CheckClose(operand, result, 0.5);
  }

  {  // Very Large Scale
    const uint64_t n = 64;
    const double scale = 1.2980742146337069e+33;  // (1 << 110)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 20);

    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    AlignedVector64<std::complex<double>> expected = operand;

    Inverse_FFTLike_FromBitReverseRadix2(transformed.data(), operand.data(),
                                         inv_root_powers.data(), n, &scalar);

    Forward_FFTLike_ToBitReverseRadix2(result.data(), transformed.data(),
                                       root_powers.data(), n, &inv_scale);

    CheckClose(expected, result, 0.5);
  }

  {  // Over 128 bits Scale
    const uint64_t n = 64;
    const double scale = 1.3611294676837539e+39;  // (1 << 130)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 20);

    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    AlignedVector64<std::complex<double>> expected = operand;

    Inverse_FFTLike_FromBitReverseRadix2(transformed.data(), operand.data(),
                                         inv_root_powers.data(), n, &scalar);

    Forward_FFTLike_ToBitReverseRadix2(result.data(), transformed.data(),
                                       root_powers.data(), n, &inv_scale);

    CheckClose(expected, result, 0.5);
  }

  {  // Inplace
    const uint64_t n = 64;
    const double scale = 1.3611294676837539e+39;  // (1 << 130)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 20);

    AlignedVector64<std::complex<double>> operand(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    AlignedVector64<std::complex<double>> expected = operand;

    Inverse_FFTLike_FromBitReverseRadix2(operand.data(), operand.data(),
                                         inv_root_powers.data(), n, &scalar);

    Forward_FFTLike_ToBitReverseRadix2(operand.data(), operand.data(),
                                       root_powers.data(), n, &inv_scale);

    CheckClose(expected, operand, 0.5);
  }
}

}  // namespace hexl
}  // namespace intel
