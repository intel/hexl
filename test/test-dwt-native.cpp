// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <complex>

#include "hexl/dwt/dwt-native.hpp"
#include "hexl/dwt/dwt.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

TEST(DWT, OneWayDWT_Native) {
  {  // Single Unscaled
    const uint64_t n = 64;
    DWT dwt(n, nullptr);
    AlignedVector64<std::complex<double>> root_powers =
        dwt.GetComplexRootsOfUnity();
    const double data_bound = (1 << 30);
    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> result(n);
    operand[0] = std::complex<double>(
        GenerateInsecureUniformRealRandomValue(0, data_bound),
        GenerateInsecureUniformRealRandomValue(0, data_bound));
    Forward_DWT_ToBitReverseRadix2(result.data(), operand.data(),
                                   root_powers.data(), n);
    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(operand[0].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(operand[0].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Single Scaled
    const uint64_t n = 64;
    DWT dwt(n, nullptr);
    AlignedVector64<std::complex<double>> root_powers =
        dwt.GetInvComplexRootsOfUnity();
    const double scale = 1 << 16;
    const double inv_scale = static_cast<double>(1.0) / scale;
    const double data_bound = (1 << 30);
    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> result(n);
    std::complex<double> value(
        GenerateInsecureUniformRealRandomValue(0, data_bound),
        GenerateInsecureUniformRealRandomValue(0, data_bound));
    operand[0] = value;
    value *= inv_scale;
    Forward_DWT_ToBitReverseRadix2(result.data(), operand.data(),
                                   root_powers.data(), n, &inv_scale);
    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(value.real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(value.imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {
    const uint64_t n = 16;
    DWT dwt(n, nullptr);
    AlignedVector64<std::complex<double>> inv_root_powers =
        dwt.GetInvComplexRootsOfUnity();

    std::vector<std::complex<double>> operand = {
        {1, 8}, {5, 4}, {3, 6}, {7, 2}, {4, -5}, {8, -1}, {6, -3}, {2, -7},
        {2, 7}, {6, 3}, {8, 1}, {4, 5}, {7, -2}, {3, -6}, {5, -4}, {1, -8}};
    std::vector<std::complex<double>> expected = {72,
                                                  -10.182068644582674,
                                                  0,
                                                  2.3890506896109649,
                                                  45.254833995939038,
                                                  28.996078283292412,
                                                  8.6591376023391522,
                                                  -16.424958949098777,
                                                  8,
                                                  6.803440758138052,
                                                  0,
                                                  12.010568880571686,
                                                  56.568542494923804,
                                                  5.7676785760555838,
                                                  20.905007438022025,
                                                  24.581688214765322};

    AlignedVector64<std::complex<double>> result(n);

    Inverse_DWT_FromBitReverseRadix2(result.data(), operand.data(),
                                     inv_root_powers.data(), n);

    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(expected[i], result[i].real());
      ASSERT_EQ(result[i].imag(), 0);
    }
  }
}

TEST(DWT, ForwardInverseDWT_Native) {
  DWT dwt(64, nullptr);
  AlignedVector64<std::complex<double>> root_powers =
      dwt.GetComplexRootsOfUnity();
  AlignedVector64<std::complex<double>> inv_root_powers =
      dwt.GetInvComplexRootsOfUnity();

  {  // Zeros test
    const uint64_t n = 64;
    const double scale = 1 << 16;
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;

    AlignedVector64<std::complex<double>> operand(n, {0, 0});
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    Forward_DWT_ToBitReverseRadix2(transformed.data(), operand.data(),
                                   root_powers.data(), n, &inv_scale);

    Inverse_DWT_FromBitReverseRadix2(result.data(), transformed.data(),
                                     inv_root_powers.data(), n, &scalar);

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

    Forward_DWT_ToBitReverseRadix2(transformed.data(), operand.data(),
                                   root_powers.data(), n, &inv_scale);

    Inverse_DWT_FromBitReverseRadix2(result.data(), transformed.data(),
                                     inv_root_powers.data(), n, &scalar);

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

    Forward_DWT_ToBitReverseRadix2(transformed.data(), operand.data(),
                                   root_powers.data(), n, &inv_scale);

    Inverse_DWT_FromBitReverseRadix2(result.data(), transformed.data(),
                                     inv_root_powers.data(), n, &scalar);

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

    Forward_DWT_ToBitReverseRadix2(transformed.data(), operand.data(),
                                   root_powers.data(), n, &inv_scale);

    Inverse_DWT_FromBitReverseRadix2(result.data(), transformed.data(),
                                     inv_root_powers.data(), n, &scalar);

    CheckClose(expected, result, 0.5);
  }

  {  // In place
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

    Forward_DWT_ToBitReverseRadix2(operand.data(), operand.data(),
                                   root_powers.data(), n, &inv_scale);

    Inverse_DWT_FromBitReverseRadix2(operand.data(), operand.data(),
                                     inv_root_powers.data(), n, &scalar);

    CheckClose(expected, operand, 0.5);
  }
}

}  // namespace hexl
}  // namespace intel
