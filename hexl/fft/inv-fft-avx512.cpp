// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
  // const __m512d* v_W_pt_real = reinterpret_cast<const
  // __m512d*>(W_interleaved); const __m512d* v_W_pt_imag =
  // reinterpret_cast<const __m512d*>(W_interleaved + 8);
  size_t offset = 0;
  // std::size_t xc = 0;
  // std::size_t yc = 0;
  // std::size_t rc = m;

  // 8 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4

  for (size_t i = 0; i < (m >> 1); i += 8) {
    // Referencing operand
    const double_t* X_op_real = operand_8C_intrlvd + offset;
    const double_t* X_op_imag = operand_8C_intrlvd + 8 + offset;

    // Referencing result
    double_t* X_r_real = result_8C_intrlvd + offset;
    double_t* X_r_imag = result_8C_intrlvd + 8 + offset;
    __m512d* v_X_r_pt_real = reinterpret_cast<__m512d*>(X_r_real);
    __m512d* v_X_r_pt_imag = reinterpret_cast<__m512d*>(X_r_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT1(X_op_real, &v_X_real, &v_Y_real);
    ComplexLoadInvInterleavedT1(X_op_imag, &v_X_imag, &v_Y_imag);

    // xc = offset;
    // yc = xc + 2;

    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[14], W_1C_intrlvd[6], W_1C_intrlvd[12], W_1C_intrlvd[4],
        W_1C_intrlvd[10], W_1C_intrlvd[2], W_1C_intrlvd[8], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[15], W_1C_intrlvd[7], W_1C_intrlvd[13], W_1C_intrlvd[5],
        W_1C_intrlvd[11], W_1C_intrlvd[3], W_1C_intrlvd[9], W_1C_intrlvd[1]);
    W_1C_intrlvd += 16;

    /*
        myfile2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
       <<"    "; myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ")
       y = (" << v_Y_real[0] << "," << v_Y_imag[0] << ")"; myfile2 << " w = ("
       << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl; myfile2
       << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = (" <<
       v_Y_real[1] << "," << v_Y_imag[1] << ")"; myfile2 << " w = (" <<
       v_W_real[1] << "," << v_W_imag[1] << ")      " << std::endl; myfile2 << "
       x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = (" <<
       v_Y_real[2] << "," << v_Y_imag[2] << ")"; myfile2 << " w = (" <<
       v_W_real[2] << "," << v_W_imag[2] << ")      " << std::endl; myfile2 << "
       x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = (" <<
       v_Y_real[3] << "," << v_Y_imag[3] << ")"; myfile2 << " w = (" <<
       v_W_real[3] << "," << v_W_imag[3] << ")      " << std::endl; myfile2 << "
       x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = (" <<
       v_Y_real[4] << "," << v_Y_imag[4] << ")"; myfile2 << " w = (" <<
       v_W_real[4] << "," << v_W_imag[4] << ")      " << std::endl; myfile2 << "
       x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = (" <<
       v_Y_real[5] << "," << v_Y_imag[5] << ")"; myfile2 << " w = (" <<
       v_W_real[5] << "," << v_W_imag[5] << ")      " << std::endl; myfile2 << "
       x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = (" <<
       v_Y_real[6] << "," << v_Y_imag[6] << ")"; myfile2 << " w = (" <<
       v_W_real[6] << "," << v_W_imag[6] << ")      " << std::endl; myfile2 << "
       x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = (" <<
       v_Y_real[7] << "," << v_Y_imag[7] << ")"; myfile2 << " w = (" <<
       v_W_real[7] << "," << v_W_imag[7] << ")      " << std::endl; rc+=16;
    */
    ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    _mm512_storeu_pd(v_X_r_pt_real++, v_X_real);
    _mm512_storeu_pd(v_X_r_pt_imag++, v_X_imag);
    v_X_r_pt_real += 2;
    v_X_r_pt_imag += 2;
    _mm512_storeu_pd(v_X_r_pt_real++, v_Y_real);
    _mm512_storeu_pd(v_X_r_pt_imag++, v_Y_imag);

    offset += 32;
  }
}

void ComplexInvT2(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m) {
  size_t offset = 0;
  // std::size_t xc = 0;
  // std::size_t yc = 0;
  // std::size_t rc = m;

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

    // xc = offset;
    // yc = xc + 4;

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[6], W_1C_intrlvd[2], W_1C_intrlvd[6], W_1C_intrlvd[2],
        W_1C_intrlvd[4], W_1C_intrlvd[0], W_1C_intrlvd[4], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[7], W_1C_intrlvd[3], W_1C_intrlvd[7], W_1C_intrlvd[3],
        W_1C_intrlvd[5], W_1C_intrlvd[1], W_1C_intrlvd[5], W_1C_intrlvd[1]);
    W_1C_intrlvd += 8;
    /*
    myfile2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0 <<"
    "; myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = (" <<
    v_Y_real[0] << "," << v_Y_imag[0] << ")"; myfile2 << " w = (" << v_W_real[0]
    << "," << v_W_imag[0] << ")      " << std::endl; myfile2 << " x = " << xc +
    1 << " y = " << yc + 1 << " w = " << rc + 0 <<"    "; myfile2 << " x = (" <<
    v_X_real[1] << "," << v_X_imag[1] << ") y = (" << v_Y_real[1] << "," <<
    v_Y_imag[1] << ")"; myfile2 << " w = (" << v_W_real[1] << "," << v_W_imag[1]
    << ")      " << std::endl; myfile2 << " x = " << xc + 2 << " y = " << yc + 2
    << " w = " << rc + 0 <<"    "; myfile2 << " x = (" << v_X_real[2] << "," <<
    v_X_imag[2] << ") y = (" << v_Y_real[2] << "," << v_Y_imag[2] << ")";
    myfile2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      " <<
    std::endl; myfile2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " <<
    rc + 0 <<"    "; myfile2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] <<
    ") y = (" << v_Y_real[3] << "," << v_Y_imag[3] << ")"; myfile2 << " w = ("
    << v_W_real[3] << "," << v_W_imag[3] << ")      " << std::endl; myfile2 << "
    x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 0 <<"    "; myfile2
    << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = (" << v_Y_real[4]
    << "," << v_Y_imag[4] << ")"; myfile2 << " w = (" << v_W_real[4] << "," <<
    v_W_imag[4] << ")      " << std::endl; myfile2 << " x = " << xc + 5 << " y =
    " << yc + 5 << " w = " << rc + 0 <<"    "; myfile2 << " x = (" <<
    v_X_real[5] << "," << v_X_imag[5] << ") y = (" << v_Y_real[5] << "," <<
    v_Y_imag[5] << ")"; myfile2 << " w = (" << v_W_real[5] << "," << v_W_imag[5]
    << ")      " << std::endl; myfile2 << " x = " << xc + 6 << " y = " << yc + 6
    << " w = " << rc + 0 <<"    "; myfile2 << " x = (" << v_X_real[6] << "," <<
    v_X_imag[6] << ") y = (" << v_Y_real[6] << "," << v_Y_imag[6] << ")";
    myfile2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      " <<
    std::endl; myfile2 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " <<
    rc + 0 <<"    "; myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] <<
    ") y = (" << v_Y_real[7] << "," << v_Y_imag[7] << ")"; myfile2 << " w = ("
    << v_W_real[7] << "," << v_W_imag[7] << ")      " << std::endl; rc+=8;*/

    ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

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
  // std::size_t xc = 0;
  // std::size_t yc = 0;
  // std::size_t rc = m;

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
    // yc = xc + 8;

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[2], W_1C_intrlvd[0], W_1C_intrlvd[2], W_1C_intrlvd[0],
        W_1C_intrlvd[2], W_1C_intrlvd[0], W_1C_intrlvd[2], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[3], W_1C_intrlvd[1], W_1C_intrlvd[3], W_1C_intrlvd[1],
        W_1C_intrlvd[3], W_1C_intrlvd[1], W_1C_intrlvd[3], W_1C_intrlvd[1]);

    W_1C_intrlvd += 4;

    /*
        myfile2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
       <<"    "; myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ")
       y = (" << v_Y_real[0] << "," << v_Y_imag[0] << ")"; myfile2 << " w = ("
       << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl; myfile2
       << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = (" <<
       v_Y_real[1] << "," << v_Y_imag[1] << ")"; myfile2 << " w = (" <<
       v_W_real[1] << "," << v_W_imag[1] << ")      " << std::endl; myfile2 << "
       x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = (" <<
       v_Y_real[2] << "," << v_Y_imag[2] << ")"; myfile2 << " w = (" <<
       v_W_real[2] << "," << v_W_imag[2] << ")      " << std::endl; myfile2 << "
       x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = (" <<
       v_Y_real[3] << "," << v_Y_imag[3] << ")"; myfile2 << " w = (" <<
       v_W_real[3] << "," << v_W_imag[3] << ")      " << std::endl; myfile2 << "
       x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = (" <<
       v_Y_real[4] << "," << v_Y_imag[4] << ")"; myfile2 << " w = (" <<
       v_W_real[4] << "," << v_W_imag[4] << ")      " << std::endl; myfile2 << "
       x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = (" <<
       v_Y_real[5] << "," << v_Y_imag[5] << ")"; myfile2 << " w = (" <<
       v_W_real[5] << "," << v_W_imag[5] << ")      " << std::endl; myfile2 << "
       x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = (" <<
       v_Y_real[6] << "," << v_Y_imag[6] << ")"; myfile2 << " w = (" <<
       v_W_real[6] << "," << v_W_imag[6] << ")      " << std::endl; myfile2 << "
       x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 0 <<"    ";
        myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = (" <<
       v_Y_real[7] << "," << v_Y_imag[7] << ")"; myfile2 << " w = (" <<
       v_W_real[7] << "," << v_W_imag[7] << ")      " << std::endl; rc+=4;*/

    ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    ComplexWriteInvInterleavedT4(v_X_real, v_Y_real, v_X_pt_real);
    ComplexWriteInvInterleavedT4(v_X_imag, v_Y_imag, v_X_pt_imag);

    offset += 32;
  }
}

void ComplexInvT8(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t gap, uint64_t m, const double_t* scalar = nullptr) {
  size_t offset = 0;

  // std::size_t xc = 0;
  // std::size_t yc = 0;
  // std::size_t rc = m;

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

    // xc = offset;
    // yc = xc + gap;

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set1_pd(*W_1C_intrlvd++);
    __m512d v_W_imag = _mm512_set1_pd(*W_1C_intrlvd++);

    if (scalar != nullptr) {
      v_W_real = _mm512_mul_pd(v_W_real, v_scalar);
      v_W_imag = _mm512_mul_pd(v_W_imag, v_scalar);
    }

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_pt_imag);

      __m512d v_Y_real = _mm512_loadu_pd(v_Y_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_pt_imag);

      /*
            myfile2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc +
         0 <<"    "; myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] <<
         ") y = (" << v_Y_real[0] << "," << v_Y_imag[0] << ")"; myfile2 << " w =
         (" << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl;
            myfile2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc +
         0 <<"    "; myfile2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] <<
         ") y = (" << v_Y_real[1] << "," << v_Y_imag[1] << ")"; myfile2 << " w =
         (" << v_W_real[1] << "," << v_W_imag[1] << ")      " << std::endl;
            myfile2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc +
         0 <<"    "; myfile2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] <<
         ") y = (" << v_Y_real[2] << "," << v_Y_imag[2] << ")"; myfile2 << " w =
         (" << v_W_real[2] << "," << v_W_imag[2] << ")      " << std::endl;
            myfile2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc +
         0 <<"    "; myfile2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] <<
         ") y = (" << v_Y_real[3] << "," << v_Y_imag[3] << ")"; myfile2 << " w =
         (" << v_W_real[3] << "," << v_W_imag[3] << ")      " << std::endl;
            myfile2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc +
         0 <<"    "; myfile2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] <<
         ") y = (" << v_Y_real[4] << "," << v_Y_imag[4] << ")"; myfile2 << " w =
         (" << v_W_real[4] << "," << v_W_imag[4] << ")      " << std::endl;
            myfile2 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc +
         0 <<"    "; myfile2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] <<
         ") y = (" << v_Y_real[5] << "," << v_Y_imag[5] << ")"; myfile2 << " w =
         (" << v_W_real[5] << "," << v_W_imag[5] << ")      " << std::endl;
            myfile2 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc +
         0 <<"    "; myfile2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] <<
         ") y = (" << v_Y_real[6] << "," << v_Y_imag[6] << ")"; myfile2 << " w =
         (" << v_W_real[6] << "," << v_W_imag[6] << ")      " << std::endl;
            myfile2 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc +
         0 <<"    "; myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] <<
         ") y = (" << v_Y_real[7] << "," << v_Y_imag[7] << ")"; myfile2 << " w =
         (" << v_W_real[7] << "," << v_W_imag[7] << ")      " << std::endl;
      */
      ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag, scalar);
      /*
            myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y =
         (" << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl; myfile2 <<
         " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = (" <<
         v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl; myfile2 << " x =
         (" << v_X_real[2] << "," << v_X_imag[2] << ") y = (" << v_Y_real[2] <<
         "," << v_Y_imag[2] << ")" << std::endl; myfile2 << " x = (" <<
         v_X_real[3] << "," << v_X_imag[3] << ") y = (" << v_Y_real[3] << "," <<
         v_Y_imag[3] << ")" << std::endl; myfile2 << " x = (" << v_X_real[4] <<
         "," << v_X_imag[4] << ") y = (" << v_Y_real[4] << "," << v_Y_imag[4] <<
         ")" << std::endl; myfile2 << " x = (" << v_X_real[5] << "," <<
         v_X_imag[5] << ") y = (" << v_Y_real[5] << "," << v_Y_imag[5] << ")" <<
         std::endl; myfile2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] <<
         ") y = (" << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
            myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y =
         (" << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;
      */
      _mm512_storeu_pd(v_X_pt_real, v_X_real);
      _mm512_storeu_pd(v_X_pt_imag, v_X_imag);

      _mm512_storeu_pd(v_Y_pt_real, v_Y_real);
      _mm512_storeu_pd(v_Y_pt_imag, v_Y_imag);

      // Increase operand & result pointers
      v_X_pt_real += 2;
      v_X_pt_imag += 2;
      v_Y_pt_real += 2;
      v_Y_pt_imag += 2;

      // xc+=16;
      // yc+=16;
    }
    // rc+=2;
    offset += (gap << 1);
  }
}

void InvFFTFromBitReverseAVX512(double_t* result_8C_intrlvd,
                                const double_t* operand_8C_intrlvd,
                                const double_t* root_of_unity_powers_1C_intrlvd,
                                const uint64_t n, const double_t* scalar) {
  // myfile2.open ("1.txt");

  HEXL_CHECK(IsPowerOfTwo(n), "n " << n << " is not a power of 2");
  HEXL_CHECK(n > 2, "n " << n << " is not bigger than 2");

  size_t gap = 2;  // Interleaved complex values requires twice the size
  size_t m = n;    // (2*n >> 1);
  size_t W_idx = 1;

  {
    // T1
    const double_t* W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexInvT1(result_8C_intrlvd, operand_8C_intrlvd, W_1C_intrlvd, m);
    gap <<= 1;
    m >>= 1;
    W_idx = m;

    // T2
    W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexInvT2(result_8C_intrlvd, W_1C_intrlvd, m);
    gap <<= 1;
    m >>= 1;
    W_idx = m;

    // T4
    W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexInvT4(result_8C_intrlvd, W_1C_intrlvd, m);
    gap <<= 1;
    m >>= 1;
    W_idx = m;

    for (; m > 2;) {
      W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
      ComplexInvT8(result_8C_intrlvd, W_1C_intrlvd, gap, m);
      gap <<= 1;
      m >>= 1;
      W_idx = m;
    }

    W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexInvT8(result_8C_intrlvd, W_1C_intrlvd, gap, m, scalar);
  }
  // myfile2.close();
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
