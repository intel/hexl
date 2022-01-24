// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fwd-fft-avx512.hpp"

#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include "hexl/fft/fft-avx512-util.hpp"
#include "hexl/logging/logging.hpp"

std::ofstream myfile1, myfile2;

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

void ComplexFwdButterfly(__m512d* X_real, __m512d* X_imag, __m512d* Y_real,
                         __m512d* Y_imag, __m512d W_real, __m512d W_imag) {
  // U = X
  __m512d U_real = *X_real;
  __m512d U_imag = *X_imag;

  // V = Y*W. Complex multiplication:
  // (y_r + iy_b)*(w_a + iw_b) = (y_a*w_a - y_b*w_b) + i(y_a*w_b + y_b*w_a)
  __m512d V_real = _mm512_mul_pd(*Y_real, W_real);
  __m512d tmp = _mm512_mul_pd(*Y_imag, W_imag);
  V_real = _mm512_sub_pd(V_real, tmp);

  __m512d V_imag = _mm512_mul_pd(*Y_real, W_imag);
  tmp = _mm512_mul_pd(*Y_imag, W_real);
  V_imag = _mm512_add_pd(V_imag, tmp);

  // X = U + V
  *X_real = _mm512_add_pd(U_real, V_real);
  *X_imag = _mm512_add_pd(U_imag, V_imag);
  // Y = U - V
  *Y_real = _mm512_sub_pd(U_real, V_real);
  *Y_imag = _mm512_sub_pd(U_imag, V_imag);
}

void ComplexFwdT1(double_t* operand_real, double_t* operand_imag,
                  const double_t* W_real, const double_t* W_imag, uint64_t m,
                  const double_t* scalar = nullptr) {
  const __m512d* v_W_pt_real = reinterpret_cast<const __m512d*>(W_real);
  const __m512d* v_W_pt_imag = reinterpret_cast<const __m512d*>(W_imag);
  size_t offset = 0;
  // std::size_t xc = 0;
  // std::size_t yc = 0;
  // std::size_t rc = m;

  __m512d v_scalar;
  if (scalar != nullptr) {
    v_scalar = _mm512_set1_pd(*scalar);
  }

  // 8 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4

  for (size_t i = 0; i < m; i += 8) {
    double_t* X_real = operand_real + offset;
    double_t* X_imag = operand_imag + offset;
    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    LoadFwdInterleavedT1(X_real, &v_X_real, &v_Y_real);
    LoadFwdInterleavedT1(X_imag, &v_X_imag, &v_Y_imag);

    // xc = offset;
    // yc = xc + 1;

    __m512d v_W_real = _mm512_loadu_pd(v_W_pt_real++);
    __m512d v_W_imag = _mm512_loadu_pd(v_W_pt_imag++);

    if (scalar != nullptr) {
      v_W_real = _mm512_mul_pd(v_W_real, v_scalar);
      v_W_imag = _mm512_mul_pd(v_W_imag, v_scalar);
      v_X_real = _mm512_mul_pd(v_X_real, v_scalar);
      v_X_imag = _mm512_mul_pd(v_X_imag, v_scalar);
    }

    /*
        myfile1 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
       <<"    "; myfile1 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ")
       y = (" << v_Y_real[0] << "," << v_Y_imag[0] << ")"; myfile1 << " w = ("
       << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl; myfile1
       << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = (" <<
       v_Y_real[1] << "," << v_Y_imag[1] << ")"; myfile1 << " w = (" <<
       v_W_real[1] << "," << v_W_imag[1] << ")      " << std::endl; myfile1 << "
       x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = (" <<
       v_Y_real[2] << "," << v_Y_imag[2] << ")"; myfile1 << " w = (" <<
       v_W_real[2] << "," << v_W_imag[2] << ")      " << std::endl; myfile1 << "
       x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = (" <<
       v_Y_real[3] << "," << v_Y_imag[3] << ")"; myfile1 << " w = (" <<
       v_W_real[3] << "," << v_W_imag[3] << ")      " << std::endl; myfile1 << "
       x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = (" <<
       v_Y_real[4] << "," << v_Y_imag[4] << ")"; myfile1 << " w = (" <<
       v_W_real[4] << "," << v_W_imag[4] << ")      " << std::endl; myfile1 << "
       x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = (" <<
       v_Y_real[5] << "," << v_Y_imag[5] << ")"; myfile1 << " w = (" <<
       v_W_real[5] << "," << v_W_imag[5] << ")      " << std::endl; myfile1 << "
       x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = (" <<
       v_Y_real[6] << "," << v_Y_imag[6] << ")"; myfile1 << " w = (" <<
       v_W_real[6] << "," << v_W_imag[6] << ")      " << std::endl; myfile1 << "
       x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = (" <<
       v_Y_real[7] << "," << v_Y_imag[7] << ")"; myfile1 << " w = (" <<
       v_W_real[7] << "," << v_W_imag[7] << ")      " << std::endl;
    */
    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    WriteFwdInterleavedT1(v_X_real, v_Y_real, v_X_pt_real);
    WriteFwdInterleavedT1(v_X_imag, v_Y_imag, v_X_pt_imag);

    offset += 16;
  }
}

void ComplexFwdT2(double_t* operand_real, double_t* operand_imag,
                  const double_t* W_real, const double_t* W_imag, uint64_t m) {
  size_t offset = 0;
  // std::size_t xc = 0;
  // std::size_t yc = 0;
  // std::size_t rc = m;
  // 4 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < m; i += 4) {
    double_t* X_real = operand_real + offset;
    double_t* X_imag = operand_imag + offset;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;

    __m512d v_Y_real;
    __m512d v_Y_imag;

    LoadFwdInterleavedT2(X_real, &v_X_real, &v_Y_real);
    LoadFwdInterleavedT2(X_imag, &v_X_imag, &v_Y_imag);

    // xc = offset;
    // yc = xc + 2;

    // Weights and weights' preconditions
    __m512d v_W_real =
        _mm512_set_pd(W_real[3], W_real[3], W_real[2], W_real[2], W_real[1],
                      W_real[1], W_real[0], W_real[0]);
    __m512d v_W_imag =
        _mm512_set_pd(W_imag[3], W_imag[3], W_imag[2], W_imag[2], W_imag[1],
                      W_imag[1], W_imag[0], W_imag[0]);
    /*
        myfile1 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
       <<"    "; myfile1 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ")
       y = (" << v_Y_real[0] << "," << v_Y_imag[0] << ")"; myfile1 << " w = ("
       << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl; myfile1
       << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = (" <<
       v_Y_real[1] << "," << v_Y_imag[1] << ")"; myfile1 << " w = (" <<
       v_W_real[1] << "," << v_W_imag[1] << ")      " << std::endl; myfile1 << "
       x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = (" <<
       v_Y_real[2] << "," << v_Y_imag[2] << ")"; myfile1 << " w = (" <<
       v_W_real[2] << "," << v_W_imag[2] << ")      " << std::endl; myfile1 << "
       x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = (" <<
       v_Y_real[3] << "," << v_Y_imag[3] << ")"; myfile1 << " w = (" <<
       v_W_real[3] << "," << v_W_imag[3] << ")      " << std::endl; myfile1 << "
       x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = (" <<
       v_Y_real[4] << "," << v_Y_imag[4] << ")"; myfile1 << " w = (" <<
       v_W_real[4] << "," << v_W_imag[4] << ")      " << std::endl; myfile1 << "
       x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = (" <<
       v_Y_real[5] << "," << v_Y_imag[5] << ")"; myfile1 << " w = (" <<
       v_W_real[5] << "," << v_W_imag[5] << ")      " << std::endl; myfile1 << "
       x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = (" <<
       v_Y_real[6] << "," << v_Y_imag[6] << ")"; myfile1 << " w = (" <<
       v_W_real[6] << "," << v_W_imag[6] << ")      " << std::endl; myfile1 << "
       x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = (" <<
       v_Y_real[7] << "," << v_Y_imag[7] << ")"; myfile1 << " w = (" <<
       v_W_real[7] << "," << v_W_imag[7] << ")      " << std::endl; rc += 2;*/
    W_real += 4;
    W_imag += 4;

    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    _mm512_storeu_pd(v_X_pt_real++, v_X_real);
    _mm512_storeu_pd(v_X_pt_imag++, v_X_imag);
    _mm512_storeu_pd(v_X_pt_real, v_Y_real);
    _mm512_storeu_pd(v_X_pt_imag, v_Y_imag);

    offset += 16;
  }
}

void ComplexFwdT4(double_t* operand_real, double_t* operand_imag,
                  const double_t* W_real, const double_t* W_imag, uint64_t m) {
  size_t offset = 0;
  // std::size_t xc = 0;
  // std::size_t yc = 0;
  // std::size_t rc = m;

  // 2 | m guaranteed by n >= 16

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < m; i += 2) {
    double_t* X_real = operand_real + offset;
    double_t* X_imag = operand_imag + offset;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    LoadFwdInterleavedT4(X_real, &v_X_real, &v_Y_real);
    LoadFwdInterleavedT4(X_imag, &v_X_imag, &v_Y_imag);

    // xc = offset;
    // yc = xc + 4;

    // Weights and weights' preconditions
    __m512d v_W_real =
        _mm512_set_pd(W_real[1], W_real[1], W_real[1], W_real[1], W_real[0],
                      W_real[0], W_real[0], W_real[0]);
    __m512d v_W_imag =
        _mm512_set_pd(W_imag[1], W_imag[1], W_imag[1], W_imag[1], W_imag[0],
                      W_imag[0], W_imag[0], W_imag[0]);

    /*
        myfile1 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
       <<"    "; myfile1 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ")
       y = (" << v_Y_real[0] << "," << v_Y_imag[0] << ")"; myfile1 << " w = ("
       << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl; myfile1
       << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = (" <<
       v_Y_real[1] << "," << v_Y_imag[1] << ")"; myfile1 << " w = (" <<
       v_W_real[1] << "," << v_W_imag[1] << ")      " << std::endl; myfile1 << "
       x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = (" <<
       v_Y_real[2] << "," << v_Y_imag[2] << ")"; myfile1 << " w = (" <<
       v_W_real[2] << "," << v_W_imag[2] << ")      " << std::endl; myfile1 << "
       x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = (" <<
       v_Y_real[3] << "," << v_Y_imag[3] << ")"; myfile1 << " w = (" <<
       v_W_real[3] << "," << v_W_imag[3] << ")      " << std::endl; myfile1 << "
       x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = (" <<
       v_Y_real[4] << "," << v_Y_imag[4] << ")"; myfile1 << " w = (" <<
       v_W_real[4] << "," << v_W_imag[4] << ")      " << std::endl; myfile1 << "
       x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = (" <<
       v_Y_real[5] << "," << v_Y_imag[5] << ")"; myfile1 << " w = (" <<
       v_W_real[5] << "," << v_W_imag[5] << ")      " << std::endl; myfile1 << "
       x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = (" <<
       v_Y_real[6] << "," << v_Y_imag[6] << ")"; myfile1 << " w = (" <<
       v_W_real[6] << "," << v_W_imag[6] << ")      " << std::endl; myfile1 << "
       x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 0 <<"    ";
        myfile1 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = (" <<
       v_Y_real[7] << "," << v_Y_imag[7] << ")"; myfile1 << " w = (" <<
       v_W_real[7] << "," << v_W_imag[7] << ")      " << std::endl; rc += 2;*/
    W_real += 2;
    W_imag += 2;

    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    _mm512_storeu_pd(v_X_pt_real++, v_X_real);
    _mm512_storeu_pd(v_X_pt_imag++, v_X_imag);
    _mm512_storeu_pd(v_X_pt_real, v_Y_real);
    _mm512_storeu_pd(v_X_pt_imag, v_Y_imag);

    offset += 16;
  }
}

void ComplexFwdT8(double_t* result_real, double_t* result_imag,
                  const double_t* operand_real, const double_t* operand_imag,
                  const double_t* W_real, const double_t* W_imag, uint64_t gap,
                  uint64_t m) {
  size_t offset = 0;

  // std::size_t xc = 0;
  // std::size_t yc = 0;
  // std::size_t rc = m;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < m; i++) {
    // Referencing operand
    const double_t* X_op_real = operand_real + offset;
    const double_t* X_op_imag = operand_imag + offset;

    const double_t* Y_op_real = X_op_real + gap;
    const double_t* Y_op_imag = X_op_imag + gap;

    const __m512d* v_X_op_pt_real = reinterpret_cast<const __m512d*>(X_op_real);
    const __m512d* v_X_op_pt_imag = reinterpret_cast<const __m512d*>(X_op_imag);

    const __m512d* v_Y_op_pt_real = reinterpret_cast<const __m512d*>(Y_op_real);
    const __m512d* v_Y_op_pt_imag = reinterpret_cast<const __m512d*>(Y_op_imag);

    // Referencing result
    double_t* X_r_real = result_real + offset;
    double_t* X_r_imag = result_imag + offset;

    double_t* Y_r_real = X_r_real + gap;
    double_t* Y_r_imag = X_r_imag + gap;

    __m512d* v_X_r_pt_real = reinterpret_cast<__m512d*>(X_r_real);
    __m512d* v_X_r_pt_imag = reinterpret_cast<__m512d*>(X_r_imag);

    __m512d* v_Y_r_pt_real = reinterpret_cast<__m512d*>(Y_r_real);
    __m512d* v_Y_r_pt_imag = reinterpret_cast<__m512d*>(Y_r_imag);

    // xc = offset;
    // yc = xc + gap;

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set1_pd(*W_real++);
    __m512d v_W_imag = _mm512_set1_pd(*W_imag++);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 8) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_op_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_op_pt_imag);

      __m512d v_Y_real = _mm512_loadu_pd(v_Y_op_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_op_pt_imag);
      /*
            myfile1 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc +
         0 <<"    "; myfile1 << " x = (" << v_X_real[0] << "," << v_X_imag[0] <<
         ") y = (" << v_Y_real[0] << "," << v_Y_imag[0] << ")"; myfile1 << " w =
         (" << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl;
            myfile1 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc +
         0 <<"    "; myfile1 << " x = (" << v_X_real[1] << "," << v_X_imag[1] <<
         ") y = (" << v_Y_real[1] << "," << v_Y_imag[1] << ")"; myfile1 << " w =
         (" << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl;
            myfile1 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc +
         0 <<"    "; myfile1 << " x = (" << v_X_real[2] << "," << v_X_imag[2] <<
         ") y = (" << v_Y_real[2] << "," << v_Y_imag[2] << ")"; myfile1 << " w =
         (" << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl;
            myfile1 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc +
         0 <<"    "; myfile1 << " x = (" << v_X_real[3] << "," << v_X_imag[3] <<
         ") y = (" << v_Y_real[3] << "," << v_Y_imag[3] << ")"; myfile1 << " w =
         (" << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl;
            myfile1 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc +
         0 <<"    "; myfile1 << " x = (" << v_X_real[4] << "," << v_X_imag[4] <<
         ") y = (" << v_Y_real[4] << "," << v_Y_imag[4] << ")"; myfile1 << " w =
         (" << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl;
            myfile1 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc +
         0 <<"    "; myfile1 << " x = (" << v_X_real[5] << "," << v_X_imag[5] <<
         ") y = (" << v_Y_real[5] << "," << v_Y_imag[5] << ")"; myfile1 << " w =
         (" << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl;
            myfile1 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc +
         0 <<"    "; myfile1 << " x = (" << v_X_real[6] << "," << v_X_imag[6] <<
         ") y = (" << v_Y_real[6] << "," << v_Y_imag[6] << ")"; myfile1 << " w =
         (" << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl;
            myfile1 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc +
         0 <<"    "; myfile1 << " x = (" << v_X_real[7] << "," << v_X_imag[7] <<
         ") y = (" << v_Y_real[7] << "," << v_Y_imag[7] << ")"; myfile1 << " w =
         (" << v_W_real[0] << "," << v_W_imag[0] << ")      " << std::endl;
      */
      ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag);
      /*
            myfile1 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y =
         (" << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl; myfile1 <<
         " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = (" <<
         v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl; myfile1 << " x =
         (" << v_X_real[2] << "," << v_X_imag[2] << ") y = (" << v_Y_real[2] <<
         "," << v_Y_imag[2] << ")" << std::endl; myfile1 << " x = (" <<
         v_X_real[3] << "," << v_X_imag[3] << ") y = (" << v_Y_real[3] << "," <<
         v_Y_imag[3] << ")" << std::endl; myfile1 << " x = (" << v_X_real[4] <<
         "," << v_X_imag[4] << ") y = (" << v_Y_real[4] << "," << v_Y_imag[4] <<
         ")" << std::endl; myfile1 << " x = (" << v_X_real[5] << "," <<
         v_X_imag[5] << ") y = (" << v_Y_real[5] << "," << v_Y_imag[5] << ")" <<
         std::endl; myfile1 << " x = (" << v_X_real[6] << "," << v_X_imag[6] <<
         ") y = (" << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
            myfile1 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y =
         (" << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;
      */
      _mm512_storeu_pd(v_X_r_pt_real++, v_X_real);
      _mm512_storeu_pd(v_X_r_pt_imag++, v_X_imag);

      _mm512_storeu_pd(v_Y_r_pt_real++, v_Y_real);
      _mm512_storeu_pd(v_Y_r_pt_imag++, v_Y_imag);

      // Increase operand pointers as well
      v_X_op_pt_real++;
      v_X_op_pt_imag++;

      v_Y_op_pt_real++;
      v_Y_op_pt_imag++;
      // xc+=8;
      // yc+=8;
    }
    // rc++;
    offset += (gap << 1);
  }
}

void ComplexFwdT1(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m, const double_t* scalar = nullptr) {
  size_t offset = 0;
  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = m;

  __m512d v_scalar;
  if (scalar != nullptr) {
    v_scalar = _mm512_set1_pd(*scalar);
  }

  // 8 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4

  for (size_t i = 0; i < (m >> 1); i += 8) {
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;
    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadFwdInterleavedT1(X_real, &v_X_real, &v_Y_real);
    ComplexLoadFwdInterleavedT1(X_imag, &v_X_imag, &v_Y_imag);

    xc = offset;
    yc = xc + 2;

    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[14], W_1C_intrlvd[12], W_1C_intrlvd[10], W_1C_intrlvd[8],
        W_1C_intrlvd[6], W_1C_intrlvd[4], W_1C_intrlvd[2], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[15], W_1C_intrlvd[13], W_1C_intrlvd[11], W_1C_intrlvd[9],
        W_1C_intrlvd[7], W_1C_intrlvd[5], W_1C_intrlvd[3], W_1C_intrlvd[1]);
    W_1C_intrlvd += 16;

    __m512d v_Xo_real = v_X_real;
    __m512d v_Xo_imag = v_X_imag;
    __m512d v_Yo_real = v_Y_real;
    __m512d v_Yo_imag = v_Y_imag;

    if (scalar != nullptr) {
      v_W_real = _mm512_mul_pd(v_W_real, v_scalar);
      v_W_imag = _mm512_mul_pd(v_W_imag, v_scalar);
      v_X_real = _mm512_mul_pd(v_X_real, v_scalar);
      v_X_imag = _mm512_mul_pd(v_X_imag, v_scalar);
    }

    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    myfile2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
            << "    ";
    myfile2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
            << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
    myfile2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
    myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
            << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
    myfile2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0
            << "    ";
    myfile2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
            << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
    myfile2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
    myfile2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
            << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
    myfile2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 2
            << "    ";
    myfile2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
            << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
    myfile2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
    myfile2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
            << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
    myfile2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 2
            << "    ";
    myfile2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
            << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
    myfile2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
    myfile2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
            << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
    myfile2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 4
            << "    ";
    myfile2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
            << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
    myfile2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
    myfile2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
            << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
    myfile2 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 4
            << "    ";
    myfile2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
            << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
    myfile2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
    myfile2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
            << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
    myfile2 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 6
            << "    ";
    myfile2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
            << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
    myfile2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
    myfile2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
            << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
    myfile2 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 6
            << "    ";
    myfile2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
            << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
    myfile2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
    myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
            << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;

    ComplexWriteFwdInterleavedT1(v_X_real, v_Y_real, v_X_imag, v_Y_imag,
                                 v_X_pt_real);

    offset += 32;
  }
}

void ComplexFwdT2(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m) {
  size_t offset = 0;
  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = m;

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

    ComplexLoadFwdInterleavedT2(X_real, &v_X_real, &v_Y_real);
    ComplexLoadFwdInterleavedT2(X_imag, &v_X_imag, &v_Y_imag);

    xc = offset;
    yc = xc + 4;

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[6], W_1C_intrlvd[6], W_1C_intrlvd[4], W_1C_intrlvd[4],
        W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[0], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[7], W_1C_intrlvd[7], W_1C_intrlvd[5], W_1C_intrlvd[5],
        W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[1], W_1C_intrlvd[1]);
    W_1C_intrlvd += 8;

    __m512d v_Xo_real = v_X_real;
    __m512d v_Xo_imag = v_X_imag;
    __m512d v_Yo_real = v_Y_real;
    __m512d v_Yo_imag = v_Y_imag;

    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    myfile2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
            << "    ";
    myfile2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
            << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
    myfile2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
    myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
            << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
    myfile2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0
            << "    ";
    myfile2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
            << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
    myfile2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
    myfile2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
            << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
    myfile2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 2
            << "    ";
    myfile2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
            << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
    myfile2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
    myfile2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
            << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
    myfile2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 2
            << "    ";
    myfile2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
            << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
    myfile2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
    myfile2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
            << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
    myfile2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 4
            << "    ";
    myfile2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
            << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
    myfile2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
    myfile2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
            << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
    myfile2 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 4
            << "    ";
    myfile2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
            << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
    myfile2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
    myfile2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
            << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
    myfile2 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 6
            << "    ";
    myfile2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
            << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
    myfile2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
    myfile2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
            << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
    myfile2 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 6
            << "    ";
    myfile2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
            << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
    myfile2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
    myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
            << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;

    _mm512_storeu_pd(v_X_pt_real, v_X_real);
    _mm512_storeu_pd(v_X_pt_imag, v_X_imag);
    v_X_pt_real += 2;
    v_X_pt_imag += 2;
    _mm512_storeu_pd(v_X_pt_real, v_Y_real);
    _mm512_storeu_pd(v_X_pt_imag, v_Y_imag);

    offset += 32;
  }
}

void ComplexFwdT4(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m) {
  size_t offset = 0;
  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = m;

  // 2 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i += 2) {
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    // *out1 =  (11, 10,  9,  8, 3, 2, 1, 0)
    // *out2 =  (15, 14, 13, 12, 7, 6, 5, 4)
    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadFwdInterleavedT4(X_real, &v_X_real, &v_Y_real);
    ComplexLoadFwdInterleavedT4(X_imag, &v_X_imag, &v_Y_imag);

    xc = offset;
    yc = xc + 8;

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[2],
        W_1C_intrlvd[0], W_1C_intrlvd[0], W_1C_intrlvd[0], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[3],
        W_1C_intrlvd[1], W_1C_intrlvd[1], W_1C_intrlvd[1], W_1C_intrlvd[1]);

    W_1C_intrlvd += 4;

    __m512d v_Xo_real = v_X_real;
    __m512d v_Xo_imag = v_X_imag;
    __m512d v_Yo_real = v_Y_real;
    __m512d v_Yo_imag = v_Y_imag;

    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    myfile2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
            << "    ";
    myfile2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
            << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
    myfile2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
    myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
            << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
    myfile2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0
            << "    ";
    myfile2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
            << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
    myfile2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
    myfile2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
            << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
    myfile2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 2
            << "    ";
    myfile2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
            << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
    myfile2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
    myfile2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
            << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
    myfile2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 2
            << "    ";
    myfile2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
            << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
    myfile2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
    myfile2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
            << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
    myfile2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 4
            << "    ";
    myfile2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
            << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
    myfile2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
    myfile2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
            << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
    myfile2 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 4
            << "    ";
    myfile2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
            << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
    myfile2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
    myfile2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
            << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
    myfile2 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 6
            << "    ";
    myfile2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
            << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
    myfile2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
    myfile2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
            << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
    myfile2 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 6
            << "    ";
    myfile2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
            << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
    myfile2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
    myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
            << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;

    _mm512_storeu_pd(v_X_pt_real, v_X_real);
    _mm512_storeu_pd(v_X_pt_imag, v_X_imag);
    v_X_pt_real += 2;
    v_X_pt_imag += 2;
    _mm512_storeu_pd(v_X_pt_real, v_Y_real);
    _mm512_storeu_pd(v_X_pt_imag, v_Y_imag);

    offset += 32;
  }
}

void ComplexFwdT8(double_t* result_8C_intrlvd,
                  const double_t* operand_8C_intrlvd,
                  const double_t* W_1C_intrlvd, uint64_t gap, uint64_t m) {
  size_t offset = 0;

  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = m;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i++) {
    // Referencing operand
    const double_t* X_op_real = operand_8C_intrlvd + offset;
    const double_t* X_op_imag = operand_8C_intrlvd + 8 + offset;

    const double_t* Y_op_real = X_op_real + gap;
    const double_t* Y_op_imag = X_op_imag + gap;

    const __m512d* v_X_op_pt_real = reinterpret_cast<const __m512d*>(X_op_real);
    const __m512d* v_X_op_pt_imag = reinterpret_cast<const __m512d*>(X_op_imag);

    const __m512d* v_Y_op_pt_real = reinterpret_cast<const __m512d*>(Y_op_real);
    const __m512d* v_Y_op_pt_imag = reinterpret_cast<const __m512d*>(Y_op_imag);

    // Referencing result
    double_t* X_r_real = result_8C_intrlvd + offset;
    double_t* X_r_imag = result_8C_intrlvd + 8 + offset;

    double_t* Y_r_real = X_r_real + gap;
    double_t* Y_r_imag = X_r_imag + gap;

    __m512d* v_X_r_pt_real = reinterpret_cast<__m512d*>(X_r_real);
    __m512d* v_X_r_pt_imag = reinterpret_cast<__m512d*>(X_r_imag);

    __m512d* v_Y_r_pt_real = reinterpret_cast<__m512d*>(Y_r_real);
    __m512d* v_Y_r_pt_imag = reinterpret_cast<__m512d*>(Y_r_imag);

    xc = offset;
    yc = xc + gap;

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set1_pd(*W_1C_intrlvd++);
    __m512d v_W_imag = _mm512_set1_pd(*W_1C_intrlvd++);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_op_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_op_pt_imag);

      __m512d v_Y_real = _mm512_loadu_pd(v_Y_op_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_op_pt_imag);

      __m512d v_Xo_real = v_X_real;
      __m512d v_Xo_imag = v_X_imag;
      __m512d v_Yo_real = v_Y_real;
      __m512d v_Yo_imag = v_Y_imag;

      ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag);

      myfile2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
              << "    ";
      myfile2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
              << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
      myfile2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
      myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
              << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
      myfile2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0
              << "    ";
      myfile2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
              << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
      myfile2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
      myfile2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
              << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
      myfile2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 2
              << "    ";
      myfile2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
              << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
      myfile2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
      myfile2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
              << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
      myfile2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 2
              << "    ";
      myfile2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
              << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
      myfile2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
      myfile2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
              << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
      myfile2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 4
              << "    ";
      myfile2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
              << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
      myfile2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
      myfile2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
              << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
      myfile2 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 4
              << "    ";
      myfile2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
              << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
      myfile2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
      myfile2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
              << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
      myfile2 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 6
              << "    ";
      myfile2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
              << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
      myfile2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
      myfile2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
              << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
      myfile2 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 6
              << "    ";
      myfile2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
              << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
      myfile2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
      myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
              << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;

      _mm512_storeu_pd(v_X_r_pt_real, v_X_real);
      _mm512_storeu_pd(v_X_r_pt_imag, v_X_imag);

      _mm512_storeu_pd(v_Y_r_pt_real, v_Y_real);
      _mm512_storeu_pd(v_Y_r_pt_imag, v_Y_imag);

      // Increase operand & result pointers
      v_X_op_pt_real += 2;
      v_X_op_pt_imag += 2;
      v_Y_op_pt_real += 2;
      v_Y_op_pt_imag += 2;

      v_X_r_pt_real += 2;
      v_X_r_pt_imag += 2;
      v_Y_r_pt_real += 2;
      v_Y_r_pt_imag += 2;

      xc += 16;
      yc += 16;
    }
    rc += 2;
    offset += (gap << 1);
  }
}

void ComplexStartFwdT8(double_t* result_8C_intrlvd,
                       const double_t* operand_1C_intrlvd,
                       const double_t* W_1C_intrlvd, uint64_t gap, uint64_t m) {
  size_t offset = 0;
  std::size_t xc = 0;
  std::size_t yc = 0;
  std::size_t rc = m;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i++) {
    // Referencing operand
    const double_t* X_op = operand_1C_intrlvd + offset;
    const double_t* Y_op = X_op + gap;

    const __m512d* v_X_op_pt = reinterpret_cast<const __m512d*>(X_op);
    const __m512d* v_Y_op_pt = reinterpret_cast<const __m512d*>(Y_op);

    // Referencing result
    double_t* X_r_real = result_8C_intrlvd + offset;
    double_t* X_r_imag = result_8C_intrlvd + 8 + offset;

    double_t* Y_r_real = X_r_real + gap;
    double_t* Y_r_imag = X_r_imag + gap;

    __m512d* v_X_r_pt_real = reinterpret_cast<__m512d*>(X_r_real);
    __m512d* v_X_r_pt_imag = reinterpret_cast<__m512d*>(X_r_imag);

    __m512d* v_Y_r_pt_real = reinterpret_cast<__m512d*>(Y_r_real);
    __m512d* v_Y_r_pt_imag = reinterpret_cast<__m512d*>(Y_r_imag);

    xc = offset;
    yc = xc + gap;

    // Weights
    __m512d v_W_real = _mm512_set1_pd(*W_1C_intrlvd++);
    __m512d v_W_imag = _mm512_set1_pd(*W_1C_intrlvd++);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      const __m512i v_perm_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
      __m512d v_X1 = _mm512_loadu_pd(v_X_op_pt++);
      __m512d v_X2 = _mm512_loadu_pd(v_X_op_pt++);
      __m512d v_X_real = _mm512_shuffle_pd(v_X1, v_X2, 0x00);
      __m512d v_X_imag = _mm512_shuffle_pd(v_X1, v_X2, 0xff);
      v_X_real = _mm512_permutexvar_pd(v_perm_idx, v_X_real);
      v_X_imag = _mm512_permutexvar_pd(v_perm_idx, v_X_imag);

      __m512d v_Y1 = _mm512_loadu_pd(v_Y_op_pt++);
      __m512d v_Y2 = _mm512_loadu_pd(v_Y_op_pt++);
      __m512d v_Y_real = _mm512_shuffle_pd(v_Y1, v_Y2, 0x00);
      __m512d v_Y_imag = _mm512_shuffle_pd(v_Y1, v_Y2, 0xff);
      v_Y_real = _mm512_permutexvar_pd(v_perm_idx, v_Y_real);
      v_Y_imag = _mm512_permutexvar_pd(v_perm_idx, v_Y_imag);

      __m512d v_Xo_real = v_X_real;
      __m512d v_Xo_imag = v_X_imag;
      __m512d v_Yo_real = v_Y_real;
      __m512d v_Yo_imag = v_Y_imag;

      ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag);

      myfile2 << " x = " << xc + 0 << " y = " << yc + 0 << " w = " << rc + 0
              << "    ";
      myfile2 << " x = (" << v_Xo_real[0] << "," << v_Xo_imag[0] << ") y = ("
              << v_Yo_real[0] << "," << v_Yo_imag[0] << ")";
      myfile2 << " w = (" << v_W_real[0] << "," << v_W_imag[0] << ")      ";
      myfile2 << " x = (" << v_X_real[0] << "," << v_X_imag[0] << ") y = ("
              << v_Y_real[0] << "," << v_Y_imag[0] << ")" << std::endl;
      myfile2 << " x = " << xc + 1 << " y = " << yc + 1 << " w = " << rc + 0
              << "    ";
      myfile2 << " x = (" << v_Xo_real[1] << "," << v_Xo_imag[1] << ") y = ("
              << v_Yo_real[1] << "," << v_Yo_imag[1] << ")";
      myfile2 << " w = (" << v_W_real[1] << "," << v_W_imag[1] << ")      ";
      myfile2 << " x = (" << v_X_real[1] << "," << v_X_imag[1] << ") y = ("
              << v_Y_real[1] << "," << v_Y_imag[1] << ")" << std::endl;
      myfile2 << " x = " << xc + 2 << " y = " << yc + 2 << " w = " << rc + 2
              << "    ";
      myfile2 << " x = (" << v_Xo_real[2] << "," << v_Xo_imag[2] << ") y = ("
              << v_Yo_real[2] << "," << v_Yo_imag[2] << ")";
      myfile2 << " w = (" << v_W_real[2] << "," << v_W_imag[2] << ")      ";
      myfile2 << " x = (" << v_X_real[2] << "," << v_X_imag[2] << ") y = ("
              << v_Y_real[2] << "," << v_Y_imag[2] << ")" << std::endl;
      myfile2 << " x = " << xc + 3 << " y = " << yc + 3 << " w = " << rc + 2
              << "    ";
      myfile2 << " x = (" << v_Xo_real[3] << "," << v_Xo_imag[3] << ") y = ("
              << v_Yo_real[3] << "," << v_Yo_imag[3] << ")";
      myfile2 << " w = (" << v_W_real[3] << "," << v_W_imag[3] << ")      ";
      myfile2 << " x = (" << v_X_real[3] << "," << v_X_imag[3] << ") y = ("
              << v_Y_real[3] << "," << v_Y_imag[3] << ")" << std::endl;
      myfile2 << " x = " << xc + 4 << " y = " << yc + 4 << " w = " << rc + 4
              << "    ";
      myfile2 << " x = (" << v_Xo_real[4] << "," << v_Xo_imag[4] << ") y = ("
              << v_Yo_real[4] << "," << v_Yo_imag[4] << ")";
      myfile2 << " w = (" << v_W_real[4] << "," << v_W_imag[4] << ")      ";
      myfile2 << " x = (" << v_X_real[4] << "," << v_X_imag[4] << ") y = ("
              << v_Y_real[4] << "," << v_Y_imag[4] << ")" << std::endl;
      myfile2 << " x = " << xc + 5 << " y = " << yc + 5 << " w = " << rc + 4
              << "    ";
      myfile2 << " x = (" << v_Xo_real[5] << "," << v_Xo_imag[5] << ") y = ("
              << v_Yo_real[5] << "," << v_Yo_imag[5] << ")";
      myfile2 << " w = (" << v_W_real[5] << "," << v_W_imag[5] << ")      ";
      myfile2 << " x = (" << v_X_real[5] << "," << v_X_imag[5] << ") y = ("
              << v_Y_real[5] << "," << v_Y_imag[5] << ")" << std::endl;
      myfile2 << " x = " << xc + 6 << " y = " << yc + 6 << " w = " << rc + 6
              << "    ";
      myfile2 << " x = (" << v_Xo_real[6] << "," << v_Xo_imag[6] << ") y = ("
              << v_Yo_real[6] << "," << v_Yo_imag[6] << ")";
      myfile2 << " w = (" << v_W_real[6] << "," << v_W_imag[6] << ")      ";
      myfile2 << " x = (" << v_X_real[6] << "," << v_X_imag[6] << ") y = ("
              << v_Y_real[6] << "," << v_Y_imag[6] << ")" << std::endl;
      myfile2 << " x = " << xc + 7 << " y = " << yc + 7 << " w = " << rc + 6
              << "    ";
      myfile2 << " x = (" << v_Xo_real[7] << "," << v_Xo_imag[7] << ") y = ("
              << v_Yo_real[7] << "," << v_Yo_imag[7] << ")";
      myfile2 << " w = (" << v_W_real[7] << "," << v_W_imag[7] << ")      ";
      myfile2 << " x = (" << v_X_real[7] << "," << v_X_imag[7] << ") y = ("
              << v_Y_real[7] << "," << v_Y_imag[7] << ")" << std::endl;

      _mm512_storeu_pd(v_X_r_pt_real, v_X_real);
      _mm512_storeu_pd(v_X_r_pt_imag, v_X_imag);

      _mm512_storeu_pd(v_Y_r_pt_real, v_Y_real);
      _mm512_storeu_pd(v_Y_r_pt_imag, v_Y_imag);

      // Increase operand & result pointers
      v_X_r_pt_real += 2;
      v_X_r_pt_imag += 2;
      v_Y_r_pt_real += 2;
      v_Y_r_pt_imag += 2;

      xc += 16;
      yc += 16;
    }
    rc += 2;
    offset += (gap << 1);
  }
}

void Forward_FFT_ToBitReverseAVX512RI(double_t* result_real,
                                      double_t* result_imag,
                                      const double_t* operand_real,
                                      const double_t* operand_imag,
                                      const double_t* root_of_unity_powers_real,
                                      const double_t* root_of_unity_powers_imag,
                                      uint64_t n, const double_t* scalar) {
  // myfile1.open ("2.txt");

  HEXL_CHECK(IsPowerOfTwo(n), "n " << n << " is not a power of 2");
  HEXL_CHECK(n > 2, "n " << n << " is not bigger than 2");

  size_t gap = (n >> 1);
  size_t m = 1;
  size_t W_idx = m;

  // T8. First pass in case of out of place
  if (gap >= 8) {
    const double_t* W_real = &root_of_unity_powers_real[W_idx];
    const double_t* W_imag = &root_of_unity_powers_imag[W_idx];

    ComplexFwdT8(result_real, result_imag, operand_real, operand_imag, W_real,
                 W_imag, gap, m);
    m <<= 1;
    W_idx = m;
    gap >>= 1;
  }

  for (; gap >= 8; gap >>= 1) {
    const double_t* W_real = &root_of_unity_powers_real[W_idx];
    const double_t* W_imag = &root_of_unity_powers_imag[W_idx];

    ComplexFwdT8(result_real, result_imag, result_real, result_imag, W_real,
                 W_imag, gap, m);
    m <<= 1;
    W_idx = m;
  }

  {
    // T4
    const double_t* W_real = &root_of_unity_powers_real[W_idx];
    const double_t* W_imag = &root_of_unity_powers_imag[W_idx];
    ComplexFwdT4(result_real, result_imag, W_real, W_imag, m);
    m <<= 1;
    W_idx = m;

    // T2
    W_real = &root_of_unity_powers_real[W_idx];
    W_imag = &root_of_unity_powers_imag[W_idx];
    ComplexFwdT2(result_real, result_imag, W_real, W_imag, m);
    m <<= 1;
    W_idx = m;

    // T1
    W_real = &root_of_unity_powers_real[W_idx];
    W_imag = &root_of_unity_powers_imag[W_idx];
    ComplexFwdT1(result_real, result_imag, W_real, W_imag, m, scalar);
    m <<= 1;
    W_idx = m;
  }

  // myfile1.close();
}

void Forward_FFT_ToBitReverseAVX512(
    double_t* result_8C_intrlvd, const double_t* operand_8C_intrlvd,
    const double_t* root_of_unity_powers_1C_intrlvd, const uint64_t n,
    const double_t* scalar) {
  myfile2.open("1.txt");

  HEXL_CHECK(IsPowerOfTwo(n), "n " << n << " is not a power of 2");
  HEXL_CHECK(n > 2, "n " << n << " is not bigger than 2");

  // std::cout << "FWD" << std::endl;

  size_t gap = n;  // (2*n >> 1) Interleaved complex numbers
  size_t m = 2;    // require twice the size
  size_t W_idx = m;

  // T8. First pass in case of out of place
  if (gap >= 16) {
    const double_t* W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexStartFwdT8(result_8C_intrlvd, operand_8C_intrlvd, W_1C_intrlvd, gap,
                      m);
    m <<= 1;
    W_idx = m;
    gap >>= 1;
  }

  for (; gap >= 16; gap >>= 1) {
    const double_t* W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexFwdT8(result_8C_intrlvd, result_8C_intrlvd, W_1C_intrlvd, gap, m);
    m <<= 1;
    W_idx = m;
  }

  {
    // T4
    const double_t* W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexFwdT4(result_8C_intrlvd, W_1C_intrlvd, m);
    m <<= 1;
    W_idx = m;

    // T2
    W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexFwdT2(result_8C_intrlvd, W_1C_intrlvd, m);
    m <<= 1;
    W_idx = m;

    // T1
    W_1C_intrlvd = &root_of_unity_powers_1C_intrlvd[W_idx];
    ComplexFwdT1(result_8C_intrlvd, W_1C_intrlvd, m, scalar);
    m <<= 1;
    W_idx = m;
  }
  myfile2.close();
}

void BuildFloatingPointsAVX512(double_t* res, const uint64_t* plain,
                               const uint64_t* threshold,
                               const uint64_t* decryption_modulus,
                               const double_t inv_scale, const size_t mod_size,
                               const size_t coeff_count) {
  __m512d* v_res_pt = reinterpret_cast<__m512d*>(res);
  __m512d v_res_imag = _mm512_setzero_pd();
  const __m512i v_perm = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);

  // myfile2.open ("1.txt", std::ios_base::app);
  double two_pow_64 = std::pow(2.0, 64);
  for (size_t i = 0; i < coeff_count; i += 8) {
    __mmask8 zeros = 0xff;
    __mmask8 cond_lt_thr = 0;

    for (int32_t j = mod_size - 1; zeros && (j >= 0); j--) {
      const uint64_t* base = plain + j;
      __m512i v_thrld = _mm512_set1_epi64(*(threshold + j));
      __m512i v_plain = _mm512_set_epi64(
          *(base + (i + 7) * mod_size), *(base + (i + 6) * mod_size),
          *(base + (i + 5) * mod_size), *(base + (i + 4) * mod_size),
          *(base + (i + 3) * mod_size), *(base + (i + 2) * mod_size),
          *(base + (i + 1) * mod_size), *(base + (i + 0) * mod_size));

      cond_lt_thr |= _mm512_mask_cmplt_epu64_mask(zeros, v_plain, v_thrld);
      zeros = _mm512_mask_cmpeq_epu64_mask(zeros, v_plain, v_thrld);
    }

    __mmask8 cond_ge_thr = ~cond_lt_thr;
    double scaled_two_pow_64 = inv_scale;
    __m512d v_res_real = _mm512_setzero_pd();
    HEXL_LOOP_UNROLL_8
    for (size_t j = 0; j < mod_size; j++, scaled_two_pow_64 *= two_pow_64) {
      const uint64_t* base = plain + j;
      __m512d v_scaled_p64 = _mm512_set1_pd(scaled_two_pow_64);
      __m512i v_dec_moduli = _mm512_set1_epi64(*(decryption_modulus + j));
      __m512i v_curr_coeff = _mm512_set_epi64(
          *(base + (i + 7) * mod_size), *(base + (i + 6) * mod_size),
          *(base + (i + 5) * mod_size), *(base + (i + 4) * mod_size),
          *(base + (i + 3) * mod_size), *(base + (i + 2) * mod_size),
          *(base + (i + 1) * mod_size), *(base + (i + 0) * mod_size));

      __mmask8 cond_gt_dec_mod =
          _mm512_mask_cmpgt_epu64_mask(cond_ge_thr, v_curr_coeff, v_dec_moduli);
      __mmask8 cond_le_dec_mod = cond_gt_dec_mod ^ cond_ge_thr;

      __m512i v_diff = _mm512_mask_sub_epi64(v_curr_coeff, cond_gt_dec_mod,
                                             v_curr_coeff, v_dec_moduli);
      v_diff = _mm512_mask_sub_epi64(v_diff, cond_le_dec_mod, v_dec_moduli,
                                     v_curr_coeff);

      // __m512d v_scaled_diff = _mm512_castsi512_pd(v_diff); does not work

      __m512d v_casted_diff = _mm512_set_pd(
          static_cast<double_t>(static_cast<uint64_t>(v_diff[7])),
          static_cast<double_t>(static_cast<uint64_t>(v_diff[6])),
          static_cast<double_t>(static_cast<uint64_t>(v_diff[5])),
          static_cast<double_t>(static_cast<uint64_t>(v_diff[4])),
          static_cast<double_t>(static_cast<uint64_t>(v_diff[3])),
          static_cast<double_t>(static_cast<uint64_t>(v_diff[2])),
          static_cast<double_t>(static_cast<uint64_t>(v_diff[1])),
          static_cast<double_t>(static_cast<uint64_t>(v_diff[0])));

      // std::cout << v_diff[0] << " > " << v_scaled_diff[0] << " ("<<
      // static_cast<double_t>(v_diff[0]) << ")";
      __m512d v_scaled_diff = _mm512_mul_pd(v_casted_diff, v_scaled_p64);
      // std::cout << " * " << v_scaled_p64[0] << " = " << v_scaled_diff[0] <<
      // std::endl;

      v_res_real = _mm512_mask_add_pd(v_res_real, cond_gt_dec_mod | cond_lt_thr,
                                      v_res_real, v_scaled_diff);
      v_res_real = _mm512_mask_sub_pd(v_res_real, cond_le_dec_mod, v_res_real,
                                      v_scaled_diff);

      /*myfile2 << "i = " << i + 0 << ", geth " << (((int32_t)cond_ge_thr&1) >>
      0) << ", ltth " << (((int32_t)cond_lt_thr&1) >> 0) << ", curr_coeff " <<
      v_curr_coeff[0] << ", dec_mod " << v_dec_moduli[0] << ", gtdm " <<
      (((int32_t)cond_gt_dec_mod&1) >> 0) << ", ledm " <<
      (((int32_t)cond_le_dec_mod&1) >> 0) << ", v_diff " << v_diff[0] << ",
      v_cdiff " << v_casted_diff[0] << ", v_sdiff " << v_scaled_diff[0] << ",
      sp64 " << v_scaled_p64[0] << ", Res " << v_res[0] << std::endl; myfile2 <<
      "i = " << i + 1 << ", geth " << (((int32_t)cond_ge_thr&2) >> 1) << ", ltth
      " << (((int32_t)cond_lt_thr&2) >> 1) << ", curr_coeff " << v_curr_coeff[1]
      << ", dec_mod " << v_dec_moduli[1] << ", gtdm " <<
      (((int32_t)cond_gt_dec_mod&2) >> 1) << ", ledm " <<
      (((int32_t)cond_le_dec_mod&2) >> 1) << ", v_diff " << v_diff[1] << ",
      v_sdiff " << v_scaled_diff[1] << ", sp64 " << v_scaled_p64[1] << ", Res "
      << v_res[1] << std::endl; myfile2 << "i = " << i + 2 << ", geth " <<
      (((int32_t)cond_ge_thr&4) >> 2) << ", ltth " << (((int32_t)cond_lt_thr&4)
      >> 2) << ", curr_coeff " << v_curr_coeff[2] << ", dec_mod " <<
      v_dec_moduli[2] << ", gtdm " << (((int32_t)cond_gt_dec_mod&4) >> 2) << ",
      ledm " << (((int32_t)cond_le_dec_mod&4) >> 2) << ", v_diff " << v_diff[2]
      << ", v_sdiff " << v_scaled_diff[2] << ", sp64 " << v_scaled_p64[2] << ",
      Res " << v_res[2] << std::endl; myfile2 << "i = " << i + 3 << ", geth " <<
      (((int32_t)cond_ge_thr&8) >> 3) << ", ltth " << (((int32_t)cond_lt_thr&8)
      >> 3) << ", curr_coeff " << v_curr_coeff[3] << ", dec_mod " <<
      v_dec_moduli[3] << ", gtdm " << (((int32_t)cond_gt_dec_mod&8) >> 3) << ",
      ledm " << (((int32_t)cond_le_dec_mod&8) >> 3) << ", v_diff " << v_diff[3]
      << ", v_sdiff " << v_scaled_diff[3] << ", sp64 " << v_scaled_p64[3] << ",
      Res " << v_res[3] << std::endl; myfile2 << "i = " << i + 4 << ", geth " <<
      (((int32_t)cond_ge_thr&16) >> 4) << ", ltth " <<
      (((int32_t)cond_lt_thr&16) >> 4) << ", curr_coeff " << v_curr_coeff[4] <<
      ", dec_mod " << v_dec_moduli[4] << ", gtdm " <<
      (((int32_t)cond_gt_dec_mod&16) >> 4) << ", ledm " <<
      (((int32_t)cond_le_dec_mod&16) >> 4) << ", v_diff " << v_diff[4] << ",
      v_sdiff " << v_scaled_diff[4] << ", sp64 " << v_scaled_p64[4] << ", Res "
      << v_res[4] << std::endl; myfile2 << "i = " << i + 5 << ", geth " <<
      (((int32_t)cond_ge_thr&32) >> 5) << ", ltth " <<
      (((int32_t)cond_lt_thr&32) >> 5) << ", curr_coeff " << v_curr_coeff[5] <<
      ", dec_mod " << v_dec_moduli[5] << ", gtdm " <<
      (((int32_t)cond_gt_dec_mod&32) >> 5) << ", ledm " <<
      (((int32_t)cond_le_dec_mod&32) >> 5) << ", v_diff " << v_diff[5] << ",
      v_sdiff " << v_scaled_diff[5] << ", sp64 " << v_scaled_p64[5] << ", Res "
      << v_res[5] << std::endl; myfile2 << "i = " << i + 6 << ", geth " <<
      (((int32_t)cond_ge_thr&64) >> 6) << ", ltth " <<
      (((int32_t)cond_lt_thr&64) >> 6) << ", curr_coeff " << v_curr_coeff[6] <<
      ", dec_mod " << v_dec_moduli[6] << ", gtdm " <<
      (((int32_t)cond_gt_dec_mod&64) >> 6) << ", ledm " <<
      (((int32_t)cond_le_dec_mod&64) >> 6) << ", v_diff " << v_diff[6] << ",
      v_sdiff " << v_scaled_diff[6] << ", sp64 " << v_scaled_p64[6] << ", Res "
      << v_res[6] << std::endl; myfile2 << "i = " << i + 7 << ", geth " <<
      (((int32_t)cond_ge_thr&128) >> 7) << ", ltth " <<
      (((int32_t)cond_lt_thr&128) >> 7) << ", curr_coeff " << v_curr_coeff[7] <<
      ", dec_mod " << v_dec_moduli[7] << ", gtdm " <<
      (((int32_t)cond_gt_dec_mod&128) >> 7) << ", ledm " <<
      (((int32_t)cond_le_dec_mod&128) >> 7) << ", v_diff " << v_diff[7] << ",
      v_sdiff " << v_scaled_diff[7] << ", sp64 " << v_scaled_p64[7] << ", Res "
      << v_res[7] << std::endl;
  */
    }
    // 7, 3, 6, 2, 5, 1, 4, 0
    v_res_real = _mm512_permutexvar_pd(v_perm, v_res_real);
    __m512d v_res1 = _mm512_shuffle_pd(v_res_real, v_res_imag, 0x00);
    __m512d v_res2 = _mm512_shuffle_pd(v_res_real, v_res_imag, 0xff);
    _mm512_storeu_pd(v_res_pt++, v_res1);
    _mm512_storeu_pd(v_res_pt++, v_res2);

    /*
    myfile2 << "i = " << i << " Returned " << (*v_res_pt)[0] << std::endl;
    myfile2  << "i = " << i + 1 << " Returned " << (*v_res_pt)[1]  << std::endl;
    myfile2  << "i = " << i + 2 << " Returned " << (*v_res_pt)[2]  << std::endl;
    myfile2  << "i = " << i + 3 << " Returned " << (*v_res_pt)[3]  << std::endl;
    myfile2  << "i = " << i + 4 << " Returned " << (*v_res_pt)[4]  << std::endl;
    myfile2  << "i = " << i + 5 << " Returned " << (*v_res_pt)[5]  << std::endl;
    myfile2  << "i = " << i + 6 << " Returned " << (*v_res_pt)[6]  << std::endl;
    myfile2  << "i = " << i + 7 << " Returned " << (*v_res_pt)[7]  << std::endl;
    */
    /*
    myfile2 << "i = " << i << " Returned " << (((int32_t)cond1&1)) << " p " <<
    *(plain + (i + 0 ) * mod_size) << " t " <<  *threshold << std::endl; myfile2
    << "i = " << i + 1 << " Returned " << (((int32_t)cond1&2) >> 1) << " p " <<
    *(plain + (i + 1 ) * mod_size) << " t " <<  *threshold <<  std::endl;
    myfile2  << "i = " << i + 2 << " Returned " << (((int32_t)cond1&4) >> 2) <<
    " p " << *(plain + (i + 2 ) * mod_size) << " t " <<  *threshold <<
    std::endl; myfile2  << "i = " << i + 3 << " Returned " <<
    (((int32_t)cond1&8) >> 3) << " p " << *(plain + (i + 3 ) * mod_size) << " t
    " <<  *threshold <<  std::endl; myfile2  << "i = " << i + 4 << " Returned "
    << (((int32_t)cond1&16) >> 4) << " p " << *(plain + (i + 4 ) * mod_size) <<
    " t " <<  *threshold <<  std::endl; myfile2  << "i = " << i + 5 << "
    Returned " << (((int32_t)cond1&32) >> 5) << " p " << *(plain + (i + 5 ) *
    mod_size) << " t " <<  *threshold <<  std::endl; myfile2  << "i = " << i + 6
    << " Returned " << (((int32_t)cond1&64) >> 6) << " p " << *(plain + (i + 6 )
    * mod_size) << " t " <<  *threshold <<  std::endl; myfile2  << "i = " << i +
    7 << " Returned " << (((int32_t)cond1&128) >> 7) << " p " << *(plain + (i +
    7 ) * mod_size) << " t " <<  *threshold <<  std::endl;
    */
  }
  // myfile2.close();
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
