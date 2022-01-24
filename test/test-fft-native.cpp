// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <complex>
#include <iostream>
#include <tuple>
#include <vector>

#include "hexl/fft/fft-native.hpp"
#include "hexl/fft/fft.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

TEST(FFT, ForwardInverseFFTNative) {
  AlignedVector64<std::complex<double_t>> inv_root_powers_64{
      {0, 0},
      {0.99879545620517241, -0.049067674327418015},
      {-0.049067674327418015, -0.99879545620517241},
      {0.67155895484701833, -0.74095112535495911},
      {-0.74095112535495911, -0.67155895484701833},
      {0.90398929312344334, -0.42755509343028208},
      {-0.42755509343028208, -0.90398929312344334},
      {0.33688985339222005, -0.94154406518302081},
      {-0.94154406518302081, -0.33688985339222005},
      {0.97003125319454397, -0.24298017990326387},
      {-0.24298017990326387, -0.97003125319454397},
      {0.51410274419322166, -0.85772861000027212},
      {-0.85772861000027212, -0.51410274419322166},
      {0.80320753148064494, -0.59569930449243336},
      {-0.59569930449243336, -0.80320753148064494},
      {0.14673047445536175, -0.98917650996478101},
      {-0.98917650996478101, -0.14673047445536175},
      {0.98917650996478101, -0.14673047445536175},
      {-0.14673047445536175, -0.98917650996478101},
      {0.59569930449243336, -0.80320753148064494},
      {-0.80320753148064494, -0.59569930449243336},
      {0.85772861000027212, -0.51410274419322166},
      {-0.51410274419322166, -0.85772861000027212},
      {0.24298017990326387, -0.97003125319454397},
      {-0.97003125319454397, -0.24298017990326387},
      {0.94154406518302081, -0.33688985339222005},
      {-0.33688985339222005, -0.94154406518302081},
      {0.42755509343028208, -0.90398929312344334},
      {-0.90398929312344334, -0.42755509343028208},
      {0.74095112535495911, -0.67155895484701833},
      {-0.67155895484701833, -0.74095112535495911},
      {0.049067674327418015, -0.99879545620517241},
      {-0.99879545620517241, -0.049067674327418015},
      {0.99518472667219693, -0.098017140329560604},
      {-0.098017140329560604, -0.99518472667219693},
      {0.63439328416364549, -0.77301045336273699},
      {-0.77301045336273699, -0.63439328416364549},
      {0.88192126434835505, -0.47139673682599764},
      {-0.47139673682599764, -0.88192126434835505},
      {0.29028467725446233, -0.95694033573220882},
      {-0.95694033573220882, -0.29028467725446233},
      {0.95694033573220882, -0.29028467725446233},
      {-0.29028467725446233, -0.95694033573220882},
      {0.47139673682599764, -0.88192126434835505},
      {-0.88192126434835505, -0.47139673682599764},
      {0.77301045336273699, -0.63439328416364549},
      {-0.63439328416364549, -0.77301045336273699},
      {0.098017140329560604, -0.99518472667219693},
      {-0.99518472667219693, -0.098017140329560604},
      {0.98078528040323043, -0.19509032201612825},
      {-0.19509032201612825, -0.98078528040323043},
      {0.55557023301960218, -0.83146961230254524},
      {-0.83146961230254524, -0.55557023301960218},
      {0.83146961230254524, -0.55557023301960218},
      {-0.55557023301960218, -0.83146961230254524},
      {0.19509032201612825, -0.98078528040323043},
      {-0.98078528040323043, -0.19509032201612825},
      {0.92387953251128674, -0.38268343236508978},
      {-0.38268343236508978, -0.92387953251128674},
      {0.38268343236508978, -0.92387953251128674},
      {-0.92387953251128674, -0.38268343236508978},
      {0.70710678118654757, -0.70710678118654746},
      {-0.70710678118654757, -0.70710678118654746},
      {0, -1}};

  AlignedVector64<std::complex<double_t>> root_powers_64{
      {0, 0},
      {0, 1},
      {0.70710678118654757, 0.70710678118654746},
      {-0.70710678118654757, 0.70710678118654746},
      {0.92387953251128674, 0.38268343236508978},
      {-0.38268343236508978, 0.92387953251128674},
      {0.38268343236508978, 0.92387953251128674},
      {-0.92387953251128674, 0.38268343236508978},
      {0.98078528040323043, 0.19509032201612825},
      {-0.19509032201612825, 0.98078528040323043},
      {0.55557023301960218, 0.83146961230254524},
      {-0.83146961230254524, 0.55557023301960218},
      {0.83146961230254524, 0.55557023301960218},
      {-0.55557023301960218, 0.83146961230254524},
      {0.19509032201612825, 0.98078528040323043},
      {-0.98078528040323043, 0.19509032201612825},
      {0.99518472667219693, 0.098017140329560604},
      {-0.098017140329560604, 0.99518472667219693},
      {0.63439328416364549, 0.77301045336273699},
      {-0.77301045336273699, 0.63439328416364549},
      {0.88192126434835505, 0.47139673682599764},
      {-0.47139673682599764, 0.88192126434835505},
      {0.29028467725446233, 0.95694033573220882},
      {-0.95694033573220882, 0.29028467725446233},
      {0.95694033573220882, 0.29028467725446233},
      {-0.29028467725446233, 0.95694033573220882},
      {0.47139673682599764, 0.88192126434835505},
      {-0.88192126434835505, 0.47139673682599764},
      {0.77301045336273699, 0.63439328416364549},
      {-0.63439328416364549, 0.77301045336273699},
      {0.098017140329560604, 0.99518472667219693},
      {-0.99518472667219693, 0.098017140329560604},
      {0.99879545620517241, 0.049067674327418015},
      {-0.049067674327418015, 0.99879545620517241},
      {0.67155895484701833, 0.74095112535495911},
      {-0.74095112535495911, 0.67155895484701833},
      {0.90398929312344334, 0.42755509343028208},
      {-0.42755509343028208, 0.90398929312344334},
      {0.33688985339222005, 0.94154406518302081},
      {-0.94154406518302081, 0.33688985339222005},
      {0.97003125319454397, 0.24298017990326387},
      {-0.24298017990326387, 0.97003125319454397},
      {0.51410274419322166, 0.85772861000027212},
      {-0.85772861000027212, 0.51410274419322166},
      {0.80320753148064494, 0.59569930449243336},
      {-0.59569930449243336, 0.80320753148064494},
      {0.14673047445536175, 0.98917650996478101},
      {-0.98917650996478101, 0.14673047445536175},
      {0.98917650996478101, 0.14673047445536175},
      {-0.14673047445536175, 0.98917650996478101},
      {0.59569930449243336, 0.80320753148064494},
      {-0.80320753148064494, 0.59569930449243336},
      {0.85772861000027212, 0.51410274419322166},
      {-0.51410274419322166, 0.85772861000027212},
      {0.24298017990326387, 0.97003125319454397},
      {-0.97003125319454397, 0.24298017990326387},
      {0.94154406518302081, 0.33688985339222005},
      {-0.33688985339222005, 0.94154406518302081},
      {0.42755509343028208, 0.90398929312344334},
      {-0.90398929312344334, 0.42755509343028208},
      {0.74095112535495911, 0.67155895484701833},
      {-0.67155895484701833, 0.74095112535495911},
      {0.049067674327418015, 0.99879545620517241},
      {-0.99879545620517241, 0.049067674327418015}};

  {  // Single Unscaled
    const uint64_t n = 64;
    const double_t data_bound = (1 << 30);
    AlignedVector64<std::complex<double_t>> operand(n);
    AlignedVector64<std::complex<double_t>> result(n);

    operand[0] = std::complex<double>(
        GenerateInsecureUniformRealRandomValue(0, data_bound),
        GenerateInsecureUniformRealRandomValue(0, data_bound));

    Forward_FFT_ToBitReverseRadix2(result.data(), operand.data(),
                                   root_powers_64.data(), n);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(operand[0].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(operand[0].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Single Scaled
    const uint64_t n = 64;
    const double_t scale = 1 << 16;
    const double_t inv_scale = double_t(1.0) / scale;
    const double_t data_bound = (1 << 30);
    AlignedVector64<std::complex<double_t>> operand(n);
    AlignedVector64<std::complex<double_t>> result(n);

    std::complex<double> value(
        GenerateInsecureUniformRealRandomValue(0, data_bound),
        GenerateInsecureUniformRealRandomValue(0, data_bound));
    operand[0] = value;
    value *= inv_scale;

    Forward_FFT_ToBitReverseRadix2(result.data(), operand.data(),
                                   root_powers_64.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(value.real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(value.imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Zeros test
    const uint64_t n = 64;
    const double_t scale = 1 << 16;
    const double_t scalar = scale / static_cast<double_t>(n);
    const double_t inv_scale = double_t(1.0) / scale;

    AlignedVector64<std::complex<double_t>> operand(n, {0, 0});
    AlignedVector64<std::complex<double_t>> transformed(n);
    AlignedVector64<std::complex<double_t>> result(n);

    Inverse_FFT_FromBitReverseRadix2(transformed.data(), operand.data(),
                                     inv_root_powers_64.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(result.data(), transformed.data(),
                                   root_powers_64.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      auto tmp = abs(operand[i].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(operand[i].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Large Scaled
    const uint64_t n = 64;
    const double_t scale = 1099511627776;  // (1 << 40)
    const double_t scalar = scale / static_cast<double_t>(n);
    const double_t inv_scale = double_t(1.0) / scale;
    const double_t data_bound = (1 << 30);

    AlignedVector64<std::complex<double_t>> operand(n);
    AlignedVector64<std::complex<double_t>> transformed(n);
    AlignedVector64<std::complex<double_t>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    Inverse_FFT_FromBitReverseRadix2(transformed.data(), operand.data(),
                                     inv_root_powers_64.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(result.data(), transformed.data(),
                                   root_powers_64.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(operand[i].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(operand[i].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Very Large Scale
    const uint64_t n = 64;
    const double_t scale = 1.2980742146337069e+33;  // (1 << 110)
    const double_t scalar = scale / static_cast<double_t>(n);
    const double_t inv_scale = double_t(1.0) / scale;
    const double_t data_bound = (1 << 20);

    AlignedVector64<std::complex<double_t>> operand(n);
    AlignedVector64<std::complex<double_t>> transformed(n);
    AlignedVector64<std::complex<double_t>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    AlignedVector64<std::complex<double_t>> expected = operand;

    Inverse_FFT_FromBitReverseRadix2(transformed.data(), operand.data(),
                                     inv_root_powers_64.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(result.data(), transformed.data(),
                                   root_powers_64.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(expected[i].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(expected[i].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Over 128 bits Scale
    const uint64_t n = 64;
    const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
    const double_t scalar = scale / static_cast<double_t>(n);
    const double_t inv_scale = static_cast<double_t>(1.0) / scale;
    const double_t data_bound = (1 << 20);

    AlignedVector64<std::complex<double_t>> operand(n);
    AlignedVector64<std::complex<double_t>> transformed(n);
    AlignedVector64<std::complex<double_t>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    AlignedVector64<std::complex<double_t>> expected = operand;

    Inverse_FFT_FromBitReverseRadix2(transformed.data(), operand.data(),
                                     inv_root_powers_64.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(result.data(), transformed.data(),
                                   root_powers_64.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(expected[i].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(expected[i].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Inplace
    const uint64_t n = 64;
    const double_t scale = 1.3611294676837539e+39;  // (1 << 130)
    const double_t scalar = scale / static_cast<double_t>(n);
    const double_t inv_scale = static_cast<double_t>(1.0) / scale;
    const double_t data_bound = (1 << 20);

    AlignedVector64<std::complex<double_t>> operand(n);

    for (size_t i = 0; i < n; i++) {
      std::complex<double_t> value(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
      operand[i] = value;
    }

    AlignedVector64<std::complex<double_t>> expected = operand;

    Inverse_FFT_FromBitReverseRadix2(operand.data(), operand.data(),
                                     inv_root_powers_64.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(operand.data(), operand.data(),
                                   root_powers_64.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(expected[i].real() - operand[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(expected[i].imag() - operand[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }
}

}  // namespace hexl
}  // namespace intel
