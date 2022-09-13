// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <complex>

#include "hexl/experimental/fft-like/fft-like-avx512-util.hpp"
#include "hexl/experimental/fft-like/fft-like.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test/test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

TEST(FFTLike, BuildFloatingPointsAVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  {
    const uint64_t poly_mod_degree = 16;
    const uint64_t coeff_mod_size = 4;
    const double scale = 1099511627776;  // (1 << 40)
    const double inv_scale = 1.0 / scale;

    std::vector<std::complex<double>> result(poly_mod_degree);

    std::vector<std::complex<double>> expected{{469095144.125, 0},
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

TEST(FFTLike, ComplexLoadFwdInterleavedT1AVX512) {
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
TEST(FFTLike, ComplexWriteFwdInterleavedT1AVX512) {
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
TEST(FFTLike, ComplexLoadInvInterleavedT1AVX512) {
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

TEST(FFTLike, ComplexLoadFwdInterleavedT2AVX512) {
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

TEST(FFTLike, ComplexLoadInvInterleavedT2AVX512) {
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

TEST(FFTLike, ComplexLoadFwdInterleavedT4AVX512) {
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

TEST(FFTLike, ComplexLoadInvInterleavedT4AVX512) {
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

TEST(FFTLike, ComplexWriteInvInterleavedT4AVX512) {
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
TEST(FFTLike, ComplexLoadFwdInterleavedT8AVX512) {
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
TEST(FFTLike, ComplexWriteInvInterleavedT8AVX512) {
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

TEST(FFTLike, ForwardInverseFFTLikeAVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

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

    Forward_FFTLike_ToBitReverseAVX512(
        &(reinterpret_cast<double(&)[2]>(result[0]))[0],
        &(reinterpret_cast<double(&)[2]>(operand[0]))[0],
        &(reinterpret_cast<double(&)[2]>(root_powers[0]))[0], n);

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

    Forward_FFTLike_ToBitReverseAVX512(
        &reinterpret_cast<double(&)[2]>(result[0])[0],
        &reinterpret_cast<double(&)[2]>(operand[0])[0],
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], n, &inv_scale);

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

    Forward_FFTLike_ToBitReverseAVX512(
        &reinterpret_cast<double(&)[2]>(transformed[0])[0],
        &reinterpret_cast<double(&)[2]>(operand[0])[0],
        &reinterpret_cast<double(&)[2]>(root_powers[0])[0], n, &inv_scale);
    Inverse_FFTLike_FromBitReverseAVX512(
        &reinterpret_cast<double(&)[2]>(result[0])[0],
        &reinterpret_cast<double(&)[2]>(transformed[0])[0],
        &reinterpret_cast<double(&)[2]>(inv_root_powers[0])[0], n, &scalar);

    CheckClose(operand, result, 0.5);
  }

  {  // Large Scaled
    const uint64_t n = 64;
    const double scale = 1099511627776;  // (1 << 40)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 30);

    AlignedVector64<double> operand_complex_interleaved(2 * n);
    AlignedVector64<double> transformed_complex_interleaved(2 * n);
    AlignedVector64<double> result_complex_interleaved(2 * n);

    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    Forward_FFTLike_ToBitReverseAVX512(
        transformed_complex_interleaved.data(),
        operand_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(root_powers[0]))[0], n, &inv_scale);
    Inverse_FFTLike_FromBitReverseAVX512(
        result_complex_interleaved.data(),
        transformed_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(inv_root_powers[0]))[0], n, &scalar);

    CheckClose(operand_complex_interleaved, result_complex_interleaved, 0.5);
  }

  {  // Very Large Scale
    const uint64_t n = 64;
    const double scale = 1.2980742146337069e+33;  // (1 << 110)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 20);

    AlignedVector64<double> operand_complex_interleaved(2 * n);
    AlignedVector64<double> transformed_complex_interleaved(2 * n);
    AlignedVector64<double> result_complex_interleaved(2 * n);

    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    Forward_FFTLike_ToBitReverseAVX512(
        transformed_complex_interleaved.data(),
        operand_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(root_powers[0]))[0], n, &inv_scale);
    Inverse_FFTLike_FromBitReverseAVX512(
        result_complex_interleaved.data(),
        transformed_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(inv_root_powers[0]))[0], n, &scalar);

    CheckClose(operand_complex_interleaved, result_complex_interleaved, 0.5);
  }

  {  // Over 128 bits Scale
    const uint64_t n = 64;
    const double scale = 1.3611294676837539e+39;  // (1 << 130)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 20);

    AlignedVector64<double> operand_complex_interleaved(2 * n);
    AlignedVector64<double> transformed_complex_interleaved(2 * n);
    AlignedVector64<double> result_complex_interleaved(2 * n);

    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    Forward_FFTLike_ToBitReverseAVX512(
        transformed_complex_interleaved.data(),
        operand_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(root_powers[0]))[0], n, &inv_scale);
    Inverse_FFTLike_FromBitReverseAVX512(
        result_complex_interleaved.data(),
        transformed_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(inv_root_powers[0]))[0], n, &scalar);

    CheckClose(operand_complex_interleaved, result_complex_interleaved, 0.5);
  }

  {  // Inplace
    const uint64_t n = 64;
    const double scale = 1.3611294676837539e+39;  // (1 << 130)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 20);

    AlignedVector64<double> operand_complex_interleaved(2 * n);
    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    AlignedVector64<double> expected = operand_complex_interleaved;

    Forward_FFTLike_ToBitReverseAVX512(
        operand_complex_interleaved.data(), operand_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(root_powers[0]))[0], n, &inv_scale);

    Inverse_FFTLike_FromBitReverseAVX512(
        operand_complex_interleaved.data(), operand_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(inv_root_powers[0]))[0], n, &scalar);

    CheckClose(expected, operand_complex_interleaved, 0.5);
  }

  {  // Big message
    const uint64_t n = 4096;
    const double scale = 1 << 16;
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = 1.0 / scale;
    const double data_bound = (1 << 30);

    FFTLike big_fft_like(n, nullptr);
    AlignedVector64<std::complex<double>> big_root_powers =
        big_fft_like.GetComplexRootsOfUnity();
    AlignedVector64<std::complex<double>> big_inv_root_powers =
        big_fft_like.GetInvComplexRootsOfUnity();

    AlignedVector64<double> operand_complex_interleaved(2 * n);
    AlignedVector64<double> transformed_complex_interleaved(2 * n);
    AlignedVector64<double> transformed_complex_interleaved2(2 * n);
    AlignedVector64<double> result_complex_interleaved(2 * n);

    operand_complex_interleaved =
        GenerateInsecureUniformRealRandomValues(2 * n, 0, data_bound);

    Forward_FFTLike_ToBitReverseAVX512(
        transformed_complex_interleaved.data(),
        operand_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(big_root_powers[0]))[0], n,
        &inv_scale);

    Inverse_FFTLike_FromBitReverseAVX512(
        result_complex_interleaved.data(),
        transformed_complex_interleaved.data(),
        &(reinterpret_cast<double(&)[2]>(big_inv_root_powers[0]))[0], n,
        &scalar);

    CheckClose(operand_complex_interleaved, result_complex_interleaved, 0.5);
  }
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
