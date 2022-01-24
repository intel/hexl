// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include "hexl/fft/fft-avx512-util.hpp"
#include "hexl/fft/fwd-fft-avx512.hpp"
#include "hexl/logging/logging.hpp"

std::ofstream file1, file2;

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

/// @brief The Harvey butterfly: assume \p X, \p Y in [0, 4q), and return X', Y'
/// in [0, 4q) such that X', Y' = X + WY, X - WY (mod q).
/// @param[in,out] X Input representing 8 64-bit signed integers in SIMD form
/// @param[in,out] Y Input representing 8 64-bit signed integers in SIMD form
/// @param[in] W Root of unity represented as 8 64-bit signed integers in
/// SIMD form
/// @param[in] W_precon Preconditioned \p W for BitShift-bit Barrett
/// reduction
/// @param[in] neg_modulus Negative modulus, i.e. (-q) represented as 8 64-bit
/// signed integers in SIMD form
/// @param[in] twice_modulus Twice the modulus, i.e. 2*q represented as 8 64-bit
/// signed integers in SIMD form
/// @param InputLessThanMod If true, assumes \p X, \p Y < \p q. Otherwise,
/// assumes \p X, \p Y < 4*\p q
/// @details See Algorithm 4 of https://arxiv.org/pdf/1205.2926.pdf

void ComplexInvButterfly(__m512d* X_real, __m512d* X_imag, __m512d* Y_real,
                         __m512d* Y_imag, __m512d W_real, __m512d W_imag,
                         const double_t* scalar = nullptr) {
  // U = X,
  __m512d U_real = *X_real;
  __m512d U_imag = *X_imag;

  // X = U + Y
  *X_real = _mm512_add_pd(U_real, *Y_real);
  *X_imag = _mm512_add_pd(U_imag, *Y_imag);

  if (scalar != nullptr) {
    __m512d v_scalar = _mm512_set1_pd(*scalar);
    *X_real = _mm512_mul_pd(*X_real, v_scalar);
    *X_imag = _mm512_mul_pd(*X_imag, v_scalar);
  }

  // V = U - Y
  __m512d V_real = _mm512_sub_pd(U_real, *Y_real);
  __m512d V_imag = _mm512_sub_pd(U_imag, *Y_imag);

  // Y = V*W. Complex multiplication:
  // (v_r + iv_b)*(w_a + iw_b) = (v_a*w_a - v_b*w_b) + i(v_a*w_b + v_b*w_a)
  *Y_real = _mm512_mul_pd(V_real, W_real);
  __m512d tmp = _mm512_mul_pd(V_imag, W_imag);
  *Y_real = _mm512_sub_pd(*Y_real, tmp);

  *Y_imag = _mm512_mul_pd(V_real, W_imag);
  tmp = _mm512_mul_pd(V_imag, W_real);
  *Y_imag = _mm512_add_pd(*Y_imag, tmp);
}

void ComplexInvT1(double_t* result_8C_intrlvd,
                  const double_t* operand_8C_intrlvd,
                  const double_t* W_1C_intrlvd, uint64_t m) {
  size_t offset = 0;
  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = 2;

  // 8 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4

  for (size_t i = 0; i < (m >> 1); i += 8) {
    // Referencing operand
    const double_t* X_op_real = operand_8C_intrlvd + offset;
    // const double_t* X_op_imag = operand_8C_intrlvd + 8 + offset;

    // Referencing result
    double_t* X_r_real = result_8C_intrlvd + offset;
    double_t* X_r_imag = result_8C_intrlvd + 8 + offset;
    __m512d* v_X_r_pt_real = reinterpret_cast<__m512d*>(X_r_real);
    __m512d* v_X_r_pt_imag = reinterpret_cast<__m512d*>(X_r_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT1(X_op_real, &v_X_real, &v_X_imag, &v_Y_real,
                                &v_Y_imag);

    xc = offset;
    yc = xc + 1;

    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[14], W_1C_intrlvd[10], W_1C_intrlvd[6], W_1C_intrlvd[2],
        W_1C_intrlvd[12], W_1C_intrlvd[8], W_1C_intrlvd[4], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[15], W_1C_intrlvd[11], W_1C_intrlvd[7], W_1C_intrlvd[3],
        W_1C_intrlvd[13], W_1C_intrlvd[9], W_1C_intrlvd[5], W_1C_intrlvd[1]);
    W_1C_intrlvd += 16;

    __m512d v_Xo_real = v_X_real;
    __m512d v_Xo_imag = v_X_imag;
    __m512d v_Yo_real = v_Y_real;
    __m512d v_Yo_imag = v_Y_imag;

    ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    // *out1_r =  (14r, 10r, 6r, 2r, 12r, 8r, 4r, 0r);
    // *out2_r =  (15r, 11r, 7r, 3r, 13r, 9r, 5r, 1r);

    file2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
          << "    ";
    file2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
          << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
    file2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
    file2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
          << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
    file2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 2
          << "    ";
    file2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
          << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
    file2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
    file2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
          << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
    file2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 4
          << "    ";
    file2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
          << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
    file2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
    file2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
          << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
    file2 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 6
          << "    ";
    file2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
          << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
    file2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
    file2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
          << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
    file2 << " x = " << xc + 8 << " y = " << yc + 8 << " w = " << rc + 8
          << "    ";
    file2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
          << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
    file2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
    file2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
          << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
    file2 << " x = " << xc + 10 << " y = " << yc + 10 << " w = " << rc + 10
          << "    ";
    file2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
          << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
    file2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
    file2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
          << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
    file2 << " x = " << xc + 12 << " y = " << yc + 12 << " w = " << rc + 12
          << "    ";
    file2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
          << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
    file2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
    file2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
          << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
    file2 << " x = " << xc + 14 << " y = " << yc + 14 << " w = " << rc + 14
          << "    ";
    file2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
          << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
    file2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
    file2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
          << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;

    rc += 16;

    _mm512_storeu_pd(v_X_r_pt_real, v_X_real);
    _mm512_storeu_pd(v_X_r_pt_imag, v_X_imag);
    v_X_r_pt_real += 2;
    v_X_r_pt_imag += 2;
    _mm512_storeu_pd(v_X_r_pt_real, v_Y_real);
    _mm512_storeu_pd(v_X_r_pt_imag, v_Y_imag);

    offset += 32;
  }
}

void ComplexInvT2(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m) {
  size_t offset = 0;
  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = 2 * m;

  // 4 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i += 4) {
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;

    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT2(X_real, &v_X_real, &v_Y_real);
    ComplexLoadInvInterleavedT2(X_imag, &v_X_imag, &v_Y_imag);

    xc = offset;
    yc = xc + 2;

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[6], W_1C_intrlvd[4], W_1C_intrlvd[2], W_1C_intrlvd[0],
        W_1C_intrlvd[6], W_1C_intrlvd[4], W_1C_intrlvd[2], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[7], W_1C_intrlvd[5], W_1C_intrlvd[3], W_1C_intrlvd[1],
        W_1C_intrlvd[7], W_1C_intrlvd[5], W_1C_intrlvd[3], W_1C_intrlvd[1]);
    W_1C_intrlvd += 8;

    __m512d v_Xo_real = v_X_real;
    __m512d v_Xo_imag = v_X_imag;
    __m512d v_Yo_real = v_Y_real;
    __m512d v_Yo_imag = v_Y_imag;

    ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    // *out1 =  (13,  9, 5, 1, 12,  8, 4, 0)
    // *out2 =  (15, 11, 7, 3, 14, 10, 6, 2)

    file2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
          << "    ";
    file2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
          << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
    file2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
    file2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
          << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
    file2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0
          << "    ";
    file2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
          << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
    file2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
    file2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
          << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
    file2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 2
          << "    ";
    file2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
          << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
    file2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
    file2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
          << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
    file2 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 2
          << "    ";
    file2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
          << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
    file2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
    file2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
          << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
    file2 << " x = " << xc + 8 << " y = " << yc + 8 << " w = " << rc + 4
          << "    ";
    file2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
          << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
    file2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
    file2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
          << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
    file2 << " x = " << xc + 9 << " y = " << yc + 9 << " w = " << rc + 4
          << "    ";
    file2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
          << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
    file2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
    file2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
          << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
    file2 << " x = " << xc + 12 << " y = " << yc + 12 << " w = " << rc + 6
          << "    ";
    file2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
          << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
    file2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
    file2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
          << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
    file2 << " x = " << xc + 13 << " y = " << yc + 13 << " w = " << rc + 6
          << "    ";
    file2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
          << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
    file2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
    file2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
          << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;
    rc += 8;

    _mm512_storeu_pd(v_X_pt_real, v_X_real);
    _mm512_storeu_pd(v_X_pt_imag, v_X_imag);
    v_X_pt_real += 2;
    v_X_pt_imag += 2;
    _mm512_storeu_pd(v_X_pt_real, v_Y_real);
    _mm512_storeu_pd(v_X_pt_imag, v_Y_imag);

    offset += 32;
  }
}

void ComplexInvT4(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m) {
  size_t offset = 0;
  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = 3 * m;

  // 2 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i += 2) {
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT4(X_real, &v_X_real, &v_Y_real);
    ComplexLoadInvInterleavedT4(X_imag, &v_X_imag, &v_Y_imag);

    // xc = offset;
    // yc = xc + 4;

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[0], W_1C_intrlvd[0],
        W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[0], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[1], W_1C_intrlvd[1],
        W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[1], W_1C_intrlvd[1]);

    W_1C_intrlvd += 4;

    __m512d v_Xo_real = v_X_real;
    __m512d v_Xo_imag = v_X_imag;
    __m512d v_Yo_real = v_Y_real;
    __m512d v_Yo_imag = v_Y_imag;

    ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    // *out1 =  (11,  9, 3, 1, 10,  8, 2, 0)
    // *out2 =  (15, 13, 7, 5, 14, 12, 6, 4)

    file2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
          << "    ";
    file2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
          << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
    file2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
    file2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
          << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
    file2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0
          << "    ";
    file2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
          << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
    file2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
    file2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
          << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
    file2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 2
          << "    ";
    file2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
          << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
    file2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
    file2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
          << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
    file2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 2
          << "    ";
    file2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
          << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
    file2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
    file2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
          << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
    file2 << " x = " << xc + 8 << " y = " << yc + 8 << " w = " << rc + 4
          << "    ";
    file2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
          << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
    file2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
    file2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
          << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
    file2 << " x = " << xc + 9 << " y = " << yc + 9 << " w = " << rc + 4
          << "    ";
    file2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
          << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
    file2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
    file2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
          << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
    file2 << " x = " << xc + 10 << " y = " << yc + 10 << " w = " << rc + 6
          << "    ";
    file2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
          << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
    file2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
    file2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
          << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
    file2 << " x = " << xc + 11 << " y = " << yc + 11 << " w = " << rc + 6
          << "    ";
    file2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
          << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
    file2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
    file2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
          << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;
    rc += 4;

    ComplexWriteInvInterleavedT4(v_X_real, v_Y_real, v_X_pt_real);
    ComplexWriteInvInterleavedT4(v_X_imag, v_Y_imag, v_X_pt_imag);

    offset += 32;
  }
}

void ComplexInvT8(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t gap, uint64_t m) {
  size_t offset = 0;

  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = 2 * m;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i++) {
    // Referencing operand
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;

    double_t* Y_real = X_real + gap;
    double_t* Y_imag = X_imag + gap;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d* v_Y_pt_real = reinterpret_cast<__m512d*>(Y_real);
    __m512d* v_Y_pt_imag = reinterpret_cast<__m512d*>(Y_imag);

    xc = offset;
    yc = xc + gap;

    // Weights and weights' preconditions
    // double_t rr = *W_1C_intrlvd;
    // double_t ri = *(W_1C_intrlvd + 1);
    __m512d v_W_real = _mm512_set1_pd(*W_1C_intrlvd++);
    __m512d v_W_imag = _mm512_set1_pd(*W_1C_intrlvd++);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_pt_imag);

      __m512d v_Y_real = _mm512_loadu_pd(v_Y_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_pt_imag);

      __m512d v_Xo_real = v_X_real;
      __m512d v_Xo_imag = v_X_imag;
      __m512d v_Yo_real = v_Y_real;
      __m512d v_Yo_imag = v_Y_imag;

      ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag);

      file2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
            << "    ";
      file2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
            << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
      file2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
      file2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
            << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
      file2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0
            << "    ";
      file2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
            << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
      file2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
      file2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
            << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
      file2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 2
            << "    ";
      file2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
            << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
      file2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
      file2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
            << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
      file2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 2
            << "    ";
      file2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
            << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
      file2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
      file2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
            << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
      file2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 4
            << "    ";
      file2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
            << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
      file2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
      file2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
            << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
      file2 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 4
            << "    ";
      file2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
            << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
      file2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
      file2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
            << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
      file2 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 6
            << "    ";
      file2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
            << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
      file2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
      file2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
            << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
      file2 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 6
            << "    ";
      file2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
            << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
      file2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
      file2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
            << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;

      _mm512_storeu_pd(v_X_pt_real, v_X_real);
      _mm512_storeu_pd(v_X_pt_imag, v_X_imag);

      _mm512_storeu_pd(v_Y_pt_real, v_Y_real);
      _mm512_storeu_pd(v_Y_pt_imag, v_Y_imag);

      // Increase operand & result pointers
      v_X_pt_real += 2;
      v_X_pt_imag += 2;
      v_Y_pt_real += 2;
      v_Y_pt_imag += 2;

      xc += 16;
      yc += 16;
    }
    rc += 2;
    offset += (gap << 1);
  }
}

void ComplexFinalInvT8(double_t* operand_8C_intrlvd,
                       const double_t* W_1C_intrlvd, uint64_t gap, uint64_t m,
                       const double_t* scalar = nullptr) {
  size_t offset = 0;

  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = 2 * m;

  __m512d v_scalar;
  if (scalar != nullptr) {
    v_scalar = _mm512_set1_pd(*scalar);
  }

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i++) {
    // Referencing operand
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;

    double_t* Y_real = X_real + gap;
    double_t* Y_imag = X_imag + gap;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d* v_Y_pt_real = reinterpret_cast<__m512d*>(Y_real);
    __m512d* v_Y_pt_imag = reinterpret_cast<__m512d*>(Y_imag);

    xc = offset;
    yc = xc + gap;

    // Weights and weights' preconditions
    // double_t rr = *W_1C_intrlvd;
    // double_t ri = *(W_1C_intrlvd + 1);
    __m512d v_W_real = _mm512_set1_pd(*W_1C_intrlvd++);
    __m512d v_W_imag = _mm512_set1_pd(*W_1C_intrlvd++);

    if (scalar != nullptr) {
      v_W_real = _mm512_mul_pd(v_W_real, v_scalar);
      v_W_imag = _mm512_mul_pd(v_W_imag, v_scalar);
      /*
      file2 << "r = (" << rr << "," << ri << "), scaled_r = (" << v_W_real[0] <<
      "," << v_W_imag[0] << "), scalar = " << *scalar << std::endl;
      */
    }

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_pt_imag);

      __m512d v_Y_real = _mm512_loadu_pd(v_Y_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_pt_imag);

      __m512d v_Xo_real = v_X_real;
      __m512d v_Xo_imag = v_X_imag;
      __m512d v_Yo_real = v_Y_real;
      __m512d v_Yo_imag = v_Y_imag;

      ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag, scalar);

      file2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
            << "    ";
      file2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
            << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
      file2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
      file2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
            << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
      file2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0
            << "    ";
      file2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
            << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
      file2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
      file2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
            << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
      file2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 2
            << "    ";
      file2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
            << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
      file2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
      file2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
            << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
      file2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 2
            << "    ";
      file2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
            << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
      file2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
      file2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
            << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
      file2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 4
            << "    ";
      file2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
            << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
      file2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
      file2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
            << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
      file2 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 4
            << "    ";
      file2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
            << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
      file2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
      file2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
            << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
      file2 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 6
            << "    ";
      file2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
            << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
      file2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
      file2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
            << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
      file2 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 6
            << "    ";
      file2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
            << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
      file2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
      file2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
            << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;

      const __m512i vperm = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
      // in:  7r  6r  5r  4r  3r  2r  1r  0r
      // ->   7r  3r  6r  2r  5r  1r  4r  0r
      v_X_real = _mm512_permutexvar_pd(vperm, v_X_real);
      // in:  7i  6i  5i  4i  3i  2i  1i  0i
      // ->   7i  3i  6i  2i  5i  1i  4i  0i
      v_X_imag = _mm512_permutexvar_pd(vperm, v_X_imag);
      // in: 15r 14r 13r 12r 11r 10r  9r  8r
      // ->  15r 11r 14r 10r 13r  9r 12r  8r
      v_Y_real = _mm512_permutexvar_pd(vperm, v_Y_real);
      // in: 15i 14i 13i 12i 11i 10i  9i  8i
      // ->  15i 11i 14i 10i 13i  9i 12i  8i
      v_Y_imag = _mm512_permutexvar_pd(vperm, v_Y_imag);

      // 00000000 >  3i  3r  2i  2r  1i  1r  0i  0r
      __m512d v_X1 = _mm512_shuffle_pd(v_X_real, v_X_imag, 0x00);
      // 11111111 >  7i  7r  6i  6r  5i  5r  4i  4r
      __m512d v_X2 = _mm512_shuffle_pd(v_X_real, v_X_imag, 0xff);
      // 00000000 > 11i 11r 10i 10r  9i  9r  8i  8r
      __m512d v_Y1 = _mm512_shuffle_pd(v_Y_real, v_Y_imag, 0x00);
      // 11111111 > 15i 15r 14i 14r 13i 13r 12i 12r
      __m512d v_Y2 = _mm512_shuffle_pd(v_Y_real, v_Y_imag, 0xff);

      _mm512_storeu_pd(v_X_pt_real, v_X1);
      _mm512_storeu_pd(v_X_pt_imag, v_X2);
      _mm512_storeu_pd(v_Y_pt_real, v_Y1);
      _mm512_storeu_pd(v_Y_pt_imag, v_Y2);

      // Increase operand & result pointers
      v_X_pt_real += 2;
      v_X_pt_imag += 2;
      v_Y_pt_real += 2;
      v_Y_pt_imag += 2;

      xc += 16;
      yc += 16;
    }
    rc += 2;
    offset += (gap << 1);
  }
}

void Inverse_FFT_FromBitReverseAVX512(
    double_t* result_8C_intrlvd, const double_t* operand_8C_intrlvd,
    const double_t* root_of_unity_powers_1C_intrlvd, const uint64_t n,
    const double_t* scalar) {
  file2.open("1.txt");
  // std::cout << "INV" << std::endl;

  HEXL_CHECK(IsPowerOfTwo(n), "n " << n << " is not a power of 2");
  HEXL_CHECK(n > 2, "n " << n << " is not bigger than 2");

  size_t gap = 2;    // Interleaved complex values requires twice the size
  size_t m = n;      // (2*n >> 1);
  size_t W_idx = 2;  // 2*1

  {
    // T1
    const double_t* W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexInvT1(result_8C_intrlvd, operand_8C_intrlvd, W_1C_intrlvd, m);
    W_idx += m;
    gap <<= 1;
    m >>= 1;

    // T2
    W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexInvT2(result_8C_intrlvd, W_1C_intrlvd, m);
    W_idx += m;
    gap <<= 1;
    m >>= 1;

    // T4
    W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexInvT4(result_8C_intrlvd, W_1C_intrlvd, m);
    W_idx += m;
    gap <<= 1;
    m >>= 1;

    for (; m > 2;) {
      W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
      ComplexInvT8(result_8C_intrlvd, W_1C_intrlvd, gap, m);
      W_idx += m;
      gap <<= 1;
      m >>= 1;
    }

    W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexFinalInvT8(result_8C_intrlvd, W_1C_intrlvd, gap, m, scalar);
    W_idx += m;
    gap <<= 1;
    m >>= 1;
  }
  file2.close();
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
