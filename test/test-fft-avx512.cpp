// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <complex>

#include "hexl/fft/fft-avx512-util.hpp"
#include "hexl/fft/fft.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

TEST(FFT, ComplexLoadFwdInterleavedT1AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  AlignedVector64<double> arg{0, 1, 4, 5, 8, 9, 12, 13, 0,  0,  0,  0,
                              0, 0, 0, 0, 2, 3, 6,  7,  10, 11, 14, 15};
  __m512d out1;
  __m512d out2;

  ComplexLoadFwdInterleavedT1(arg.data(), &out1, &out2);
  __m512d exp1 = _mm512_set_pd(14, 12, 10, 8, 6, 4, 2, 0);
  __m512d exp2 = _mm512_set_pd(15, 13, 11, 9, 7, 5, 3, 1);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

// ComplexWriteFwdInterleavedT1:
// Re-arrange back 8 complex interleaved into 1 complex interleaved
TEST(FFT, ComplexWriteFwdInterleavedT1AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  __m512d arg_yi = _mm512_set_pd(15.1, 13.1, 11.1, 9.1, 7.1, 5.1, 3.1, 1.1);
  __m512d arg_yr = _mm512_set_pd(15.4, 13.4, 11.4, 9.4, 7.4, 5.4, 3.4, 1.4);
  __m512d arg_xi = _mm512_set_pd(14.1, 12.1, 10.1, 8.1, 6.1, 4.1, 2.1, 0.1);
  __m512d arg_xr = _mm512_set_pd(14.4, 12.4, 10.4, 8.4, 6.4, 4.4, 2.4, 0.4);

  AlignedVector64<double> out(32, 0);
  AlignedVector64<double> exp{0.4,  0.1,  1.4,  1.1,  2.4,  2.1,  3.4,  3.1,
                              4.4,  4.1,  5.4,  5.1,  6.4,  6.1,  7.4,  7.1,
                              8.4,  8.1,  9.4,  9.1,  10.4, 10.1, 11.4, 11.1,
                              12.4, 12.1, 13.4, 13.1, 14.4, 14.1, 15.4, 15.1};

  ComplexWriteFwdInterleavedT1(arg_xr, arg_yr, arg_xi, arg_yi,
                               reinterpret_cast<__m512d*>(&out[0]));

  AssertEqual(exp, out);
}

// ComplexLoadInvInterleavedT1:
// Re-arrange 1 complex interleaved into 8 complex interleaved
TEST(FFT, ComplexLoadInvInterleavedT1AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<double> arg{0.4,  0.1,  1.4,  1.1,  2.4,  2.1,  3.4,  3.1,
                              4.4,  4.1,  5.4,  5.1,  6.4,  6.1,  7.4,  7.1,
                              8.4,  8.1,  9.4,  9.1,  10.4, 10.1, 11.4, 11.1,
                              12.4, 12.1, 13.4, 13.1, 14.4, 14.1, 15.4, 15.1};
  __m512d out_yr;
  __m512d out_yi;
  __m512d out_xr;
  __m512d out_xi;
  __m512d exp_yr = _mm512_set_pd(15.4, 11.4, 7.4, 3.4, 13.4, 9.4, 5.4, 1.4);
  __m512d exp_yi = _mm512_set_pd(15.1, 11.1, 7.1, 3.1, 13.1, 9.1, 5.1, 1.1);
  __m512d exp_xr = _mm512_set_pd(14.4, 10.4, 6.4, 2.4, 12.4, 8.4, 4.4, 0.4);
  __m512d exp_xi = _mm512_set_pd(14.1, 10.1, 6.1, 2.1, 12.1, 8.1, 4.1, 0.1);

  ComplexLoadInvInterleavedT1(arg.data(), &out_xr, &out_xi, &out_yr, &out_yi);

  AssertEqual(ExtractValues(exp_yr), ExtractValues(out_yr));
  AssertEqual(ExtractValues(exp_yi), ExtractValues(out_yi));
  AssertEqual(ExtractValues(exp_xr), ExtractValues(out_xr));
  AssertEqual(ExtractValues(exp_xi), ExtractValues(out_xi));
}

TEST(FFT, ComplexLoadFwdInterleavedT2AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<double> arg{0, 1, 2, 3, 8, 9, 10, 11, 0,  0,  0,  0,
                              0, 0, 0, 0, 4, 5, 6,  7,  12, 13, 14, 15};
  __m512d out1;
  __m512d out2;

  ComplexLoadFwdInterleavedT2(arg.data(), &out1, &out2);

  __m512d exp1 = _mm512_set_pd(13, 12, 9, 8, 5, 4, 1, 0);
  __m512d exp2 = _mm512_set_pd(15, 14, 11, 10, 7, 6, 3, 2);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(FFT, ComplexLoadInvInterleavedT2AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<double> arg{0, 4, 8, 12, 2, 6, 10, 14, 0, 0, 0,  0,
                              0, 0, 0, 0,  1, 5, 9,  13, 3, 7, 11, 15};
  __m512d out1;
  __m512d out2;

  ComplexLoadInvInterleavedT2(arg.data(), &out1, &out2);

  __m512d exp1 = _mm512_set_pd(13, 9, 5, 1, 12, 8, 4, 0);
  __m512d exp2 = _mm512_set_pd(15, 11, 7, 3, 14, 10, 6, 2);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(FFT, ComplexLoadFwdInterleavedT4AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<double> arg{0, 1, 2, 3, 4, 5, 6,  7,  0,  0,  0,  0,
                              0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15};
  __m512d out1;
  __m512d out2;

  ComplexLoadFwdInterleavedT4(arg.data(), &out1, &out2);

  __m512d exp1 = _mm512_set_pd(11, 10, 9, 8, 3, 2, 1, 0);
  __m512d exp2 = _mm512_set_pd(15, 14, 13, 12, 7, 6, 5, 4);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(FFT, ComplexLoadInvInterleavedT4AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<double> arg{0, 4, 8, 12, 1, 5, 9,  13, 0, 0, 0,  0,
                              0, 0, 0, 0,  2, 6, 10, 14, 3, 7, 11, 15};
  __m512d out1;
  __m512d out2;

  ComplexLoadInvInterleavedT4(arg.data(), &out1, &out2);

  __m512d exp1 = _mm512_set_pd(11, 9, 3, 1, 10, 8, 2, 0);
  __m512d exp2 = _mm512_set_pd(15, 13, 7, 5, 14, 12, 6, 4);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(FFT, ComplexWriteInvInterleavedT4AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  __m512d arg1 = _mm512_set_pd(7, 6, 5, 4, 3, 2, 1, 0);
  __m512d arg2 = _mm512_set_pd(15, 14, 13, 12, 11, 10, 9, 8);

  AlignedVector64<double> out(24, 0);
  AlignedVector64<double> exp{0, 4, 1, 5, 8, 12, 9, 13, 0,  0,  0,  0,
                              0, 0, 0, 0, 2, 6,  3, 7,  10, 14, 11, 15};

  ComplexWriteInvInterleavedT4(arg1, arg2, reinterpret_cast<__m512d*>(&out[0]));

  AssertEqual(exp, out);
}

// ComplexLoadFwdInterleavedT8:
// Re-arrange 1 complex interleaved into 8 complex interleaved
TEST(FFT, ComplexLoadFwdInterleavedT8AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<double> arg_x{0.4, 0.1, 1.4, 1.1, 2.4, 2.1, 3.4, 3.1,
                                4.4, 4.1, 5.4, 5.1, 6.4, 6.1, 7.4, 7.1};
  AlignedVector64<double> arg_y{0.4, 0.1, 1.4, 1.1, 2.4, 2.1, 3.4, 3.1,
                                4.4, 4.1, 5.4, 5.1, 6.4, 6.1, 7.4, 7.1};
  __m512d out_yr;
  __m512d out_yi;
  __m512d out_xr;
  __m512d out_xi;
  __m512d exp_yr = _mm512_set_pd(7.4, 6.4, 5.4, 4.4, 3.4, 2.4, 1.4, 0.4);
  __m512d exp_yi = _mm512_set_pd(7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1, 0.1);
  __m512d exp_xr = _mm512_set_pd(7.4, 6.4, 5.4, 4.4, 3.4, 2.4, 1.4, 0.4);
  __m512d exp_xi = _mm512_set_pd(7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1, 0.1);

  ComplexLoadFwdInterleavedT8(reinterpret_cast<__m512d*>(&arg_x[0]),
                              reinterpret_cast<__m512d*>(&arg_y[0]), &out_xr,
                              &out_xi, &out_yr, &out_yi);

  AssertEqual(ExtractValues(exp_yr), ExtractValues(out_yr));
  AssertEqual(ExtractValues(exp_yi), ExtractValues(out_yi));
  AssertEqual(ExtractValues(exp_xr), ExtractValues(out_xr));
  AssertEqual(ExtractValues(exp_xi), ExtractValues(out_xi));
}

// ComplexWriteInvInterleavedT8:
// Re-arrange back 8 complex interleaved into 1 complex interleaved
// Assuming ComplexLoadInvInterleavedT4 was used before.
// Given inputs: 7i, 6i, 5i, 4i, 3i, 2i, 1i, 0i, 7r, 6r, 5r, 4r, 3r, 2r, 1r, 0r
// Given output: 7i, 7r, 6i, 6r, 5i, 5r, 4i, 4r, 3i, 3r, 2i, 2r, 1i, 1r, 0i, 0r
TEST(FFT, ComplexWriteInvInterleavedT8AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<double> out_x(16, 0);
  AlignedVector64<double> out_y(16, 0);
  AlignedVector64<double> exp_x{0.4, 0.1, 1.4, 1.1, 2.4, 2.1, 3.4, 3.1,
                                4.4, 4.1, 5.4, 5.1, 6.4, 6.1, 7.4, 7.1};
  AlignedVector64<double> exp_y{0.4, 0.1, 1.4, 1.1, 2.4, 2.1, 3.4, 3.1,
                                4.4, 4.1, 5.4, 5.1, 6.4, 6.1, 7.4, 7.1};

  __m512d arg_yr = _mm512_set_pd(7.4, 6.4, 5.4, 4.4, 3.4, 2.4, 1.4, 0.4);
  __m512d arg_yi = _mm512_set_pd(7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1, 0.1);
  __m512d arg_xr = _mm512_set_pd(7.4, 6.4, 5.4, 4.4, 3.4, 2.4, 1.4, 0.4);
  __m512d arg_xi = _mm512_set_pd(7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1, 0.1);

  ComplexWriteInvInterleavedT8(&arg_xr, &arg_xi, &arg_yr, &arg_yi,
                               reinterpret_cast<__m512d*>(&out_x[0]),
                               reinterpret_cast<__m512d*>(&out_y[0]));

  AssertEqual(exp_y, out_y);
  AssertEqual(exp_x, out_x);
}

TEST(FFT, OneWayFFT_AVX512) {
  {  // Single
    const uint64_t n = 64;
    FFT fft(n);
    AlignedVector64<double> root_powers =
        fft.GetInterleavedComplexRootsOfUnity();
    const double data_bound = (1 << 30);
    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> result(n);

    operand[0] = std::complex<double>(
        GenerateInsecureUniformRealRandomValue(0, data_bound),
        GenerateInsecureUniformRealRandomValue(0, data_bound));

    Forward_FFT_AVX512(&(reinterpret_cast<double(&)[2]>(result[0]))[0],
                       &(reinterpret_cast<double(&)[2]>(operand[0]))[0],
                       root_powers.data(), n);

    for (size_t i = 0; i < n; ++i) {
      CheckClose(operand[0], result[i], 0.5);
    }
  }

  {
    const uint64_t n = 16;
    FFT fft(n);
    AlignedVector64<double> inv_root_powers =
        fft.GetInterleavedInvComplexRootsOfUnity();

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

    Inverse_FFT_AVX512(&reinterpret_cast<double(&)[2]>(result[0])[0],
                       &reinterpret_cast<double(&)[2]>(operand[0])[0],
                       inv_root_powers.data(), n);

    for (size_t i = 0; i < n; ++i) {
      ASSERT_TRUE(expected[i].real() == result[i].real());
      ASSERT_TRUE(result[i].imag() == expected[i].imag());
    }
  }
}

TEST(FFT, ForwardInverseFFT_AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  FFT fft(64);
  AlignedVector64<double> root_powers = fft.GetInterleavedComplexRootsOfUnity();
  AlignedVector64<double> inv_root_powers =
      fft.GetInterleavedInvComplexRootsOfUnity();

  {  // Zeros test
    const uint64_t n = 64;

    AlignedVector64<std::complex<double>> operand(n, {0, 0});
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    Forward_FFT_AVX512(&reinterpret_cast<double(&)[2]>(transformed[0])[0],
                       &reinterpret_cast<double(&)[2]>(operand[0])[0],
                       root_powers.data(), n);
    Inverse_FFT_AVX512(&reinterpret_cast<double(&)[2]>(result[0])[0],
                       &reinterpret_cast<double(&)[2]>(transformed[0])[0],
                       inv_root_powers.data(), n);

    CheckClose(operand, result, 0.5);
  }

  {  // Out of place
    const uint64_t n = 64;
    const double data_bound = (1 << 30);

    AlignedVector64<double> operand_complex_interleaved(2 * n);
    AlignedVector64<double> transformed_complex_interleaved(2 * n);
    AlignedVector64<double> result_complex_interleaved(2 * n);

    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    Forward_FFT_AVX512(transformed_complex_interleaved.data(),
                       operand_complex_interleaved.data(), root_powers.data(),
                       n);

    Inverse_FFT_AVX512(result_complex_interleaved.data(),
                       transformed_complex_interleaved.data(),
                       inv_root_powers.data(), n);

    CheckClose(operand_complex_interleaved, result_complex_interleaved, 0.5);
  }

  {  // In place
    const uint64_t n = 64;
    const double data_bound = (1 << 20);

    AlignedVector64<double> operand_complex_interleaved(2 * n);
    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    AlignedVector64<double> expected = operand_complex_interleaved;

    Forward_FFT_AVX512(operand_complex_interleaved.data(),
                       operand_complex_interleaved.data(), root_powers.data(),
                       n);

    Inverse_FFT_AVX512(operand_complex_interleaved.data(),
                       operand_complex_interleaved.data(),
                       inv_root_powers.data(), n);

    CheckClose(expected, operand_complex_interleaved, 0.5);
  }

  {  // Big message
    const uint64_t n = 4096;
    const double data_bound = (1 << 30);

    FFT big_fft(n);
    AlignedVector64<double> big_root_powers =
        big_fft.GetInterleavedComplexRootsOfUnity();
    AlignedVector64<double> big_inv_root_powers =
        big_fft.GetInterleavedInvComplexRootsOfUnity();

    AlignedVector64<double> operand_complex_interleaved(2 * n);
    AlignedVector64<double> transformed_complex_interleaved(2 * n);
    AlignedVector64<double> transformed_complex_interleaved2(2 * n);
    AlignedVector64<double> result_complex_interleaved(2 * n);

    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    Forward_FFT_AVX512(transformed_complex_interleaved.data(),
                       operand_complex_interleaved.data(),
                       big_root_powers.data(), n);

    Inverse_FFT_AVX512(result_complex_interleaved.data(),
                       transformed_complex_interleaved.data(),
                       big_inv_root_powers.data(), n);

    CheckClose(operand_complex_interleaved, result_complex_interleaved, 0.5);
  }
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
