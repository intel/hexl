// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <complex>
#include <iostream>
#include <tuple>
#include <vector>

#include "hexl/fft/fft.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

TEST(FFT, BuildFloatingPointsAVX512) {
  {
    const uint64_t poly_mod_degree = 16;
    const uint64_t coeff_mod_size = 4;
    const double_t scale = 1099511627776;  // (1 << 40)
    const double_t inv_scale = static_cast<double_t>(1.0) / scale;

    std::vector<std::complex<double_t>> result(poly_mod_degree);

    std::vector<std::complex<double_t>> expected{{469095144.125, 0},
                                                 {32109980.057216156, 0},
                                                 {133969900.94656014, 0},
                                                 {1327830.7073135898, 0},
                                                 {-72732310.45981437, 0},
                                                 {-55123198.89089907, 0},
                                                 {-130250344.32255825, 0},
                                                 {66152794.724299073, 0},
                                                 {0, 0},
                                                 {-66152794.724299081, 0},
                                                 {130250344.32255828, 0},
                                                 {55123198.89089907, 0},
                                                 {72732310.459814355, 0},
                                                 {-1327830.7073136102, 0},
                                                 {-133969900.94656017, 0},
                                                 {-32109980.05721616, 0}};

    const uint64_t operand[] = {17713475508538179584ULL,
                                27,
                                0,
                                0,
                                16858552366855081984ULL,
                                1,
                                0,
                                0,
                                18174255346774966272ULL,
                                7,
                                0,
                                0,
                                1459965302409322496ULL,
                                0,
                                0,
                                0,
                                10852157353743343297ULL,
                                72057091796482622ULL,
                                0,
                                0,
                                11766836204861046465ULL,
                                72057091796482623ULL,
                                0,
                                0,
                                2950642535971380929ULL,
                                72057091796482619ULL,
                                0,
                                0,
                                17395534788117004288ULL,
                                3,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                18086411410077564609ULL,
                                72057091796482622ULL,
                                0,
                                0,
                                14084559588513677312ULL,
                                7,
                                0,
                                0,
                                5268365919623979008ULL,
                                3,
                                0,
                                0,
                                6183044770741665792ULL,
                                4,
                                0,
                                0,
                                15575236822075680449ULL,
                                72057091796482626ULL,
                                0,
                                0,
                                17307690851419578049ULL,
                                72057091796482618ULL,
                                0,
                                0,
                                176649757629939393ULL,
                                72057091796482625ULL,
                                0,
                                0};

    const uint64_t upper_half_threshold[] = {8517601062242512737ULL,
                                             36028545898241313ULL, 0, 0};
    const uint64_t decryption_modulus[] = {17035202124485025473ULL,
                                           72057091796482626ULL, 0, 0};

    BuildFloatingPointsAVX512(&reinterpret_cast<double(&)[2]>(result[0])[0],
                              operand, upper_half_threshold, decryption_modulus,
                              inv_scale, coeff_mod_size, poly_mod_degree);

    ASSERT_EQ(expected, result);
  }
}

TEST(FFT, ForwardInverseFFTAVX512) {
  std::vector<std::complex<double>> inv_root_powers_64{
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

  std::vector<std::complex<double>> root_powers_64{
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

    Forward_FFT_ToBitReverseAVX512(
        &(reinterpret_cast<double_t(&)[2]>(result[0]))[0],
        &(reinterpret_cast<double_t(&)[2]>(operand[0]))[0],
        &(reinterpret_cast<double_t(&)[2]>(root_powers_64[0]))[0], n);

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

    Forward_FFT_ToBitReverseAVX512(
        &reinterpret_cast<double(&)[2]>(result[0])[0],
        &reinterpret_cast<double(&)[2]>(operand[0])[0],
        &reinterpret_cast<double(&)[2]>(root_powers_64[0])[0], n, &inv_scale);

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

    Inverse_FFT_FromBitReverseAVX512(
        &reinterpret_cast<double(&)[2]>(transformed[0])[0],
        &reinterpret_cast<double(&)[2]>(operand[0])[0],
        &reinterpret_cast<double(&)[2]>(inv_root_powers_64[0])[0], n, &scalar);

    Forward_FFT_ToBitReverseAVX512(
        &reinterpret_cast<double(&)[2]>(result[0])[0],
        &reinterpret_cast<double(&)[2]>(transformed[0])[0],
        &reinterpret_cast<double(&)[2]>(root_powers_64[0])[0], n, &inv_scale);

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

    AlignedVector64<double_t> operand_complex_interleaved(2 * n);
    AlignedVector64<double_t> transformed_complex_interleaved(2 * n);
    AlignedVector64<double_t> result_complex_interleaved(2 * n);

    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    Inverse_FFT_FromBitReverseAVX512(
        transformed_complex_interleaved.data(),
        operand_complex_interleaved.data(),
        &(reinterpret_cast<double_t(&)[2]>(inv_root_powers_64[0]))[0], n,
        &scalar);

    Forward_FFT_ToBitReverseAVX512(
        result_complex_interleaved.data(),
        transformed_complex_interleaved.data(),
        &(reinterpret_cast<double_t(&)[2]>(root_powers_64[0]))[0], n,
        &inv_scale);

    for (size_t i = 0; i < 2 * n; i++) {
      double tmp =
          abs(operand_complex_interleaved[i] - result_complex_interleaved[i]);
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Very Large Scale
    const uint64_t n = 64;
    const double_t scale = 1.2980742146337069e+33;  // (1 << 110)
    const double_t scalar = scale / static_cast<double_t>(n);
    const double_t inv_scale = double_t(1.0) / scale;
    const double_t data_bound = (1 << 20);

    AlignedVector64<double_t> operand_complex_interleaved(2 * n);
    AlignedVector64<double_t> transformed_complex_interleaved(2 * n);
    AlignedVector64<double_t> result_complex_interleaved(2 * n);

    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    Inverse_FFT_FromBitReverseAVX512(
        transformed_complex_interleaved.data(),
        operand_complex_interleaved.data(),
        &(reinterpret_cast<double_t(&)[2]>(inv_root_powers_64[0]))[0], n,
        &scalar);

    Forward_FFT_ToBitReverseAVX512(
        result_complex_interleaved.data(),
        transformed_complex_interleaved.data(),
        &(reinterpret_cast<double_t(&)[2]>(root_powers_64[0]))[0], n,
        &inv_scale);

    for (size_t i = 0; i < 2 * n; i++) {
      double tmp =
          abs(operand_complex_interleaved[i] - result_complex_interleaved[i]);
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Over 128 bits Scale
    const uint64_t n = 64;
    const double scale = 1.3611294676837539e+39;  // (1 << 130)
    const double scalar = scale / static_cast<double_t>(n);
    const double inv_scale = double_t(1.0) / scale;
    const double_t data_bound = (1 << 20);

    AlignedVector64<double_t> operand_complex_interleaved(2 * n);
    AlignedVector64<double_t> transformed_complex_interleaved(2 * n);
    AlignedVector64<double_t> result_complex_interleaved(2 * n);

    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    Inverse_FFT_FromBitReverseAVX512(
        transformed_complex_interleaved.data(),
        operand_complex_interleaved.data(),
        &(reinterpret_cast<double_t(&)[2]>(inv_root_powers_64[0]))[0], n,
        &scalar);

    Forward_FFT_ToBitReverseAVX512(
        result_complex_interleaved.data(),
        transformed_complex_interleaved.data(),
        &(reinterpret_cast<double_t(&)[2]>(root_powers_64[0]))[0], n,
        &inv_scale);

    for (size_t i = 0; i < 2 * n; i++) {
      double tmp =
          abs(operand_complex_interleaved[i] - result_complex_interleaved[i]);
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {  // Inplace
    const uint64_t n = 64;
    const double scale = 1.3611294676837539e+39;  // (1 << 130)
    const double scalar = scale / static_cast<double_t>(n);
    const double inv_scale = double_t(1.0) / scale;
    const double_t data_bound = (1 << 20);

    AlignedVector64<double_t> operand_complex_interleaved(2 * n);
    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    AlignedVector64<double_t> expected = operand_complex_interleaved;

    Inverse_FFT_FromBitReverseAVX512(
        operand_complex_interleaved.data(), operand_complex_interleaved.data(),
        &(reinterpret_cast<double_t(&)[2]>(inv_root_powers_64[0]))[0], n,
        &scalar);

    Forward_FFT_ToBitReverseAVX512(
        operand_complex_interleaved.data(), operand_complex_interleaved.data(),
        &(reinterpret_cast<double_t(&)[2]>(root_powers_64[0]))[0], n,
        &inv_scale);

    for (size_t i = 0; i < 2 * n; i++) {
      double tmp = abs(expected[i] - operand_complex_interleaved[i]);
      ASSERT_TRUE(tmp < 0.5);
    }
  }
}

}  // namespace hexl
}  // namespace intel
