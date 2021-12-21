// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fwd-fft-avx512.hpp"

#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

#include "hexl/fft/fft-avx512-util.hpp"
#include "hexl/logging/logging.hpp"

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

void FwdButterfly(__m512d* X_real, __m512d* X_imag, __m512d* Y_real,
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

    __m512d v_W_real = _mm512_loadu_pd(v_W_pt_real++);
    __m512d v_W_imag = _mm512_loadu_pd(v_W_pt_imag++);

    if (scalar != nullptr) {
      v_W_real = _mm512_mul_pd(v_W_real, v_scalar);
      v_W_imag = _mm512_mul_pd(v_W_imag, v_scalar);
      v_X_real = _mm512_mul_pd(v_X_real, v_scalar);
      v_X_imag = _mm512_mul_pd(v_X_imag, v_scalar);
    }

    FwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                 v_W_imag);

    WriteFwdInterleavedT1(v_X_real, v_Y_real, v_X_pt_real);
    WriteFwdInterleavedT1(v_X_imag, v_Y_imag, v_X_pt_imag);

    offset += 16;
  }
}

void ComplexFwdT2(double_t* operand_real, double_t* operand_imag,
                  const double_t* W_real, const double_t* W_imag, uint64_t m) {
  size_t offset = 0;

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

    // Weights and weights' preconditions
    __m512d v_W_real =
        _mm512_set_pd(W_real[3], W_real[3], W_real[2], W_real[2], W_real[1],
                      W_real[1], W_real[0], W_real[0]);
    __m512d v_W_imag =
        _mm512_set_pd(W_imag[3], W_imag[3], W_imag[2], W_imag[2], W_imag[1],
                      W_imag[1], W_imag[0], W_imag[0]);
    W_real += 4;
    W_imag += 4;

    FwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
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

    // Weights and weights' preconditions
    __m512d v_W_real =
        _mm512_set_pd(W_real[1], W_real[1], W_real[1], W_real[1], W_real[0],
                      W_real[0], W_real[0], W_real[0]);
    __m512d v_W_imag =
        _mm512_set_pd(W_imag[1], W_imag[1], W_imag[1], W_imag[1], W_imag[0],
                      W_imag[0], W_imag[0], W_imag[0]);

    W_real += 2;
    W_imag += 2;

    FwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
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

    // Weights and weights' preconditions
    __m512d v_W_real = _mm512_set1_pd(*W_real++);
    __m512d v_W_imag = _mm512_set1_pd(*W_imag++);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 8) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_op_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_op_pt_imag);

      __m512d v_Y_real = _mm512_loadu_pd(v_Y_op_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_op_pt_imag);

      FwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                   v_W_imag);

      _mm512_storeu_pd(v_X_r_pt_real++, v_X_real);
      _mm512_storeu_pd(v_X_r_pt_imag++, v_X_imag);

      _mm512_storeu_pd(v_Y_r_pt_real++, v_Y_real);
      _mm512_storeu_pd(v_Y_r_pt_imag++, v_Y_imag);

      // Increase operand pointers as well
      v_X_op_pt_real++;
      v_X_op_pt_imag++;

      v_Y_op_pt_real++;
      v_Y_op_pt_imag++;
    }
    offset += (gap << 1);
  }
}

void ComplexFwdTransformToBitReverseAVX512(
    double_t* result_real, double_t* result_imag, const double_t* operand_real,
    const double_t* operand_imag, const double_t* root_of_unity_powers_real,
    const double_t* root_of_unity_powers_imag, uint64_t n,
    const double_t* scalar) {
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
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
