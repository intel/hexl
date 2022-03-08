// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <complex>

#include "hexl/fft/fft-native.hpp"
#include "hexl/fft/fft.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

TEST(FFT, OneWayFFT_Native) {
  {  // Single
    const uint64_t n = 64;
    FFT fft(n);
    AlignedVector64<std::complex<double>> root_powers =
        fft.GetComplexRootsOfUnity();
    const double data_bound = (1 << 30);
    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> result(n);
    operand[0] = std::complex<double>(
        GenerateInsecureUniformRealRandomValue(0, data_bound),
        GenerateInsecureUniformRealRandomValue(0, data_bound));
    Forward_FFT_Radix2(result.data(), operand.data(), root_powers.data(), n);
    for (size_t i = 0; i < n; ++i) {
      CheckClose(operand[0], result[i], 0.5);
    }
  }

  {
    const uint64_t n = 16;
    FFT fft(n);
    AlignedVector64<std::complex<double>> inv_root_powers =
        fft.GetInvComplexRootsOfUnity();

    std::vector<std::complex<double>> operand = {
        {1, 8}, {5, 4}, {3, 6}, {7, 2}, {4, -5}, {8, -1}, {6, -3}, {2, -7},
        {2, 7}, {6, 3}, {8, 1}, {4, 5}, {7, -2}, {3, -6}, {5, -4}, {1, -8}};
    std::vector<std::complex<double>> expected = {
        {4.5, 0},
        {-0.73197082710900485, 0.14559805007309851},
        {-3.3195436482630059, 1.3750000000000004},
        {-0.25000000000000006, 0.16704465947982483},
        {-1, 1},
        {0.52003106085336159, -0.77828148243818829},
        {-0.86243686707645817, 2.0821067811865475},
        {-0.24999999999999997, 1.256834873031462},
        {0, 1},
        {-0.078689344670816563, -0.3955980500730984},
        {0.56954364826300585, 1.375},
        {-0.25000000000000017, -0.37415144066637229},
        {0, 5.5511151231257827e-17},
        {0.79062911092645982, 0.52828148243818829},
        {1.6124368670764582, 0.66789321881345221},
        {-0.24999999999999986, -0.049728091844914557}};

    AlignedVector64<std::complex<double>> result(n);

    Inverse_FFT_Radix2(result.data(), operand.data(), inv_root_powers.data(),
                       n);

    for (size_t i = 0; i < n; ++i) {
      ASSERT_TRUE(expected[i].real() == result[i].real());
      ASSERT_TRUE(expected[i].imag() == result[i].imag());
    }
  }
}

TEST(FFT, ForwardInverseFFT_Native) {
  FFT fft(64);
  AlignedVector64<std::complex<double>> root_powers =
      fft.GetComplexRootsOfUnity();
  AlignedVector64<std::complex<double>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  {  // Zeros test
    const uint64_t n = 64;
    AlignedVector64<std::complex<double>> operand(n, {0, 0});
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    Forward_FFT_Radix2(transformed.data(), operand.data(), root_powers.data(),
                       n);

    Inverse_FFT_Radix2(result.data(), transformed.data(),
                       inv_root_powers.data(), n);

    CheckClose(operand, result, 0.5);
  }

  {  // Out of place
    const uint64_t n = 32;
    const double data_bound = (1 << 30);

    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    Forward_FFT_Radix2(transformed.data(), operand.data(), root_powers.data(),
                       n);

    Inverse_FFT_Radix2(result.data(), transformed.data(),
                       inv_root_powers.data(), n);

    std::cout.precision(17);
    for (size_t i = 0; i < n; ++i) {
      std::cout << operand[i] << " <--> " << result[i] << std::endl;
    }

    CheckClose(operand, result, 0.5);
  }

  {  // In place
    const uint64_t n = 64;
    const double data_bound = (1 << 20);

    AlignedVector64<std::complex<double>> operand(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    AlignedVector64<std::complex<double>> expected = operand;

    Forward_FFT_Radix2(operand.data(), operand.data(), root_powers.data(), n);

    Inverse_FFT_Radix2(operand.data(), operand.data(), inv_root_powers.data(),
                       n);

    CheckClose(expected, operand, 0.5);
  }
}

}  // namespace hexl
}  // namespace intel
