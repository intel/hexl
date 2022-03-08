// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fwd-fft-avx512.hpp"

#include <iostream>

#include "hexl/fft/fft-avx512-util.hpp"
#include "hexl/logging/logging.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

/// @brief Final butterfly step for the Forward FFT.
/// @param[in,out] X_real Double precision (DP) values in SIMD form representing
/// the real part of 8 complex numbers.
/// @param[in,out] X_imag DP values in SIMD form representing the
/// imaginary part of the forementioned complex numbers.
/// @param[in,out] Y_real DP values in SIMD form representing the
/// real part of 8 complex numbers.
/// @param[in,out] Y_imag DP values in SIMD form representing the
/// imaginary part of the forementioned complex numbers.
/// @param[in] W_real DP values in SIMD form representing the real part of the
/// Complex Roots of unity.
/// @param[in] W_imag DP values in SIMD form representing the imaginary part of
/// the Complex Roots of unity.
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

void ComplexT1(double* result_8C_intrlvd, const double* operand_1C_intrlvd,
               const double* W_1C_intrlvd, uint64_t m) {
  size_t offset = 0;

  // 8 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_8
  for (size_t i = 0; i < (m >> 1); i += 8) {
    // Referencing operand
    const double* X_op_real = operand_1C_intrlvd + offset;

    // Referencing result
    double* X_r_real = result_8C_intrlvd + offset;
    double* X_r_imag = X_r_real + 8;
    __m512d* v_X_r_pt_real = reinterpret_cast<__m512d*>(X_r_real);
    __m512d* v_X_r_pt_imag = reinterpret_cast<__m512d*>(X_r_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT1(X_op_real, &v_X_real, &v_X_imag, &v_Y_real,
                                &v_Y_imag);

    // Weights
    __m512d v_W_real = _mm512_set1_pd(W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set1_pd(W_1C_intrlvd[8]);

    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    _mm512_storeu_pd(v_X_r_pt_real, v_X_real);
    _mm512_storeu_pd(v_X_r_pt_imag, v_X_imag);
    v_X_r_pt_real += 2;
    v_X_r_pt_imag += 2;
    _mm512_storeu_pd(v_X_r_pt_real, v_Y_real);
    _mm512_storeu_pd(v_X_r_pt_imag, v_Y_imag);

    offset += 32;
  }
}

void ComplexT2(double* operand_8C_intrlvd, const double* W_1C_intrlvd,
               uint64_t m) {
  size_t offset = 0;

  // 4 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i += 4) {
    double* X_real = operand_8C_intrlvd + offset;
    double* X_imag = X_real + 8;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;

    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT2(X_real, &v_X_real, &v_Y_real);
    ComplexLoadInvInterleavedT2(X_imag, &v_X_imag, &v_Y_imag);

    // Weights
    // x =  (13,  9, 5, 1, 12,  8, 4, 0)
    // y =  (15, 11, 7, 3, 14, 10, 6, 2)
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[1], W_1C_intrlvd[1], W_1C_intrlvd[1], W_1C_intrlvd[1],
        W_1C_intrlvd[0], W_1C_intrlvd[0], W_1C_intrlvd[0], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[9], W_1C_intrlvd[9], W_1C_intrlvd[9], W_1C_intrlvd[9],
        W_1C_intrlvd[8], W_1C_intrlvd[8], W_1C_intrlvd[8], W_1C_intrlvd[8]);

    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
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

void ComplexT4(double* operand_8C_intrlvd, const double* W_1C_intrlvd,
               uint64_t m) {
  size_t offset = 0;

  // 2 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i += 2) {
    double* X_real = operand_8C_intrlvd + offset;
    double* X_imag = X_real + 8;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT4(X_real, &v_X_real, &v_Y_real);
    ComplexLoadInvInterleavedT4(X_imag, &v_X_imag, &v_Y_imag);

    // Weights
    // x =  (11,  9, 3, 1, 10,  8, 2, 0)
    // y =  (15, 13, 7, 5, 14, 12, 6, 4)
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[3], W_1C_intrlvd[1], W_1C_intrlvd[3], W_1C_intrlvd[1],
        W_1C_intrlvd[2], W_1C_intrlvd[0], W_1C_intrlvd[2], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[11], W_1C_intrlvd[9], W_1C_intrlvd[11], W_1C_intrlvd[9],
        W_1C_intrlvd[10], W_1C_intrlvd[8], W_1C_intrlvd[10], W_1C_intrlvd[8]);

    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    ComplexWriteInvInterleavedT4(v_X_real, v_Y_real, v_X_pt_real);
    ComplexWriteInvInterleavedT4(v_X_imag, v_Y_imag, v_X_pt_imag);

    offset += 32;
  }
}

void ComplexT8(double* operand_8C_intrlvd, const double* W_1C_intrlvd,
               uint64_t gap, uint64_t m) {
  size_t offset = 0;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i++) {
    // Referencing operand
    double* X_real = operand_8C_intrlvd + offset;
    double* X_imag = X_real + 8;

    double* Y_real = X_real + gap;
    double* Y_imag = X_imag + gap;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d* v_Y_pt_real = reinterpret_cast<__m512d*>(Y_real);
    __m512d* v_Y_pt_imag = reinterpret_cast<__m512d*>(Y_imag);

    const __m512d* v_W_pt_real = reinterpret_cast<const __m512d*>(W_1C_intrlvd);
    const __m512d* v_W_pt_imag =
        reinterpret_cast<const __m512d*>(W_1C_intrlvd + 8);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_pt_imag);

      __m512d v_Y_real = _mm512_loadu_pd(v_Y_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_pt_imag);

      __m512d v_W_real = _mm512_loadu_pd(v_W_pt_real);
      __m512d v_W_imag = _mm512_loadu_pd(v_W_pt_imag);

      ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag);

      _mm512_storeu_pd(v_X_pt_real, v_X_real);
      _mm512_storeu_pd(v_X_pt_imag, v_X_imag);

      _mm512_storeu_pd(v_Y_pt_real, v_Y_real);
      _mm512_storeu_pd(v_Y_pt_imag, v_Y_imag);

      // Increase operand & result pointers
      v_X_pt_real += 2;
      v_X_pt_imag += 2;
      v_Y_pt_real += 2;
      v_Y_pt_imag += 2;
      v_W_pt_real += 2;
      v_W_pt_imag += 2;
    }
    offset += (gap << 1);
  }
}

// Takes operand as 8 complex interleaved: This is 8 real parts followed by
// its 8 imaginary parts.
// Returns operand as 1 complex interleaved: One real part followed by its
// imaginary part.
void ComplexFinalT8(double* operand_8C_intrlvd, const double* W_1C_intrlvd,
                    uint64_t gap, uint64_t m,
                    const __m512d* scale_down = nullptr) {
  size_t offset = 0;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i++, offset += (gap << 1)) {
    // Referencing operand
    double* X_real = operand_8C_intrlvd + offset;
    double* X_imag = X_real + 8;

    double* Y_real = X_real + gap;
    double* Y_imag = X_imag + gap;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d* v_Y_pt_real = reinterpret_cast<__m512d*>(Y_real);
    __m512d* v_Y_pt_imag = reinterpret_cast<__m512d*>(Y_imag);

    const __m512d* v_W_pt_real = reinterpret_cast<const __m512d*>(W_1C_intrlvd);
    const __m512d* v_W_pt_imag =
        reinterpret_cast<const __m512d*>(W_1C_intrlvd + 8);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      // Weights
      __m512d v_X_real = _mm512_loadu_pd(v_X_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_pt_imag);
      __m512d v_Y_real = _mm512_loadu_pd(v_Y_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_pt_imag);
      __m512d v_W_real = _mm512_loadu_pd(v_W_pt_real);
      __m512d v_W_imag = _mm512_loadu_pd(v_W_pt_imag);

      ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag);
      if (scale_down != nullptr) {
        v_X_real = _mm512_mul_pd(v_X_real, *scale_down);
        v_X_imag = _mm512_mul_pd(v_X_imag, *scale_down);
        v_Y_real = _mm512_mul_pd(v_Y_real, *scale_down);
        v_Y_imag = _mm512_mul_pd(v_Y_imag, *scale_down);
      }

      ComplexWriteInvInterleavedT8(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag,
                                   v_X_pt_real, v_Y_pt_real);

      // Increase operand & result pointers
      v_X_pt_real += 2;
      v_X_pt_imag += 2;
      v_Y_pt_real += 2;
      v_Y_pt_imag += 2;
      v_W_pt_real += 2;
      v_W_pt_imag += 2;
    }
  }
}

void FFT_AVX512(double* result_cmplx_intrlvd,
                const double* operand_cmplx_intrlvd,
                const double* root_of_unity_powers_cmplx_intrlvd,
                const uint64_t n, uint64_t recursion_depth,
                bool inverse = false) {
  size_t gap;  // Interleaved complex values requires a gap twice the size

  size_t W_idx;

  static const size_t base_fft_size = 16;

  if (n <= base_fft_size) {  // Perform breadth-first InvFFT
    size_t m = n;            // (2*n >> 1);
    gap = 2;
    W_idx = 1;

    // T1
    const double* W_cmplx_intrlvd = &root_of_unity_powers_cmplx_intrlvd[W_idx];
    ComplexT1(result_cmplx_intrlvd, result_cmplx_intrlvd, W_cmplx_intrlvd, m);
    gap <<= 1;
    m >>= 1;
    W_idx += 1;

    // T2
    W_cmplx_intrlvd = &root_of_unity_powers_cmplx_intrlvd[W_idx];
    ComplexT2(result_cmplx_intrlvd, W_cmplx_intrlvd, m);
    gap <<= 1;
    m >>= 1;
    W_idx += 2;

    // T4
    W_cmplx_intrlvd = &root_of_unity_powers_cmplx_intrlvd[W_idx];
    ComplexT4(result_cmplx_intrlvd, W_cmplx_intrlvd, m);
    gap <<= 1;
    m >>= 1;
    W_idx += 12;

    while (m > 2) {
      W_cmplx_intrlvd = &root_of_unity_powers_cmplx_intrlvd[W_idx];
      ComplexT8(result_cmplx_intrlvd, W_cmplx_intrlvd, gap, m);
      gap <<= 1;
      m >>= 1;
      W_idx += (gap >> 1);
    }

    W_cmplx_intrlvd = &root_of_unity_powers_cmplx_intrlvd[W_idx];
    if (recursion_depth == 0) {
      __m512d scale_down;
      if (inverse) {
        scale_down = _mm512_set1_pd(1.0 / static_cast<double>(n));
        ComplexFinalT8(result_cmplx_intrlvd, W_cmplx_intrlvd, gap, m,
                       &scale_down);
      } else {
        ComplexFinalT8(result_cmplx_intrlvd, W_cmplx_intrlvd, gap, m);
      }

      HEXL_VLOG(5,
                "AVX512 returning INV FFT result "
                    << std::vector<std::complex<double>>(
                           result_cmplx_intrlvd, result_cmplx_intrlvd + 2 * n));
    } else {
      ComplexT8(result_cmplx_intrlvd, W_cmplx_intrlvd, gap, m);
    }
  } else {
    FFT_AVX512(result_cmplx_intrlvd, operand_cmplx_intrlvd,
               root_of_unity_powers_cmplx_intrlvd, n / 2, recursion_depth + 1);
    FFT_AVX512(&result_cmplx_intrlvd[n], &operand_cmplx_intrlvd[n],
               root_of_unity_powers_cmplx_intrlvd, n / 2, recursion_depth + 1);

    gap = n;
    W_idx = gap;
    const double* W_cmplx_intrlvd = &root_of_unity_powers_cmplx_intrlvd[W_idx];
    if (recursion_depth == 0) {
      __m512d scale_down;
      if (inverse) {
        scale_down = _mm512_set1_pd(1.0 / static_cast<double>(n));
        ComplexFinalT8(result_cmplx_intrlvd, W_cmplx_intrlvd, gap, 2,
                       &scale_down);
      } else {
        ComplexFinalT8(result_cmplx_intrlvd, W_cmplx_intrlvd, gap, 2);
      }
      HEXL_VLOG(5,
                "AVX512 returning INV FFT result "
                    << std::vector<std::complex<double>>(
                           result_cmplx_intrlvd, result_cmplx_intrlvd + 2 * n));
    } else {
      ComplexT8(result_cmplx_intrlvd, W_cmplx_intrlvd, gap, 2);
    }
  }
}

void Forward_FFT_AVX512(double* result_cmplx_intrlvd,
                        const double* operand_cmplx_intrlvd,
                        const double* root_of_unity_powers_cmplx_intrlvd,
                        const uint64_t n, bool inverse) {
  HEXL_CHECK(IsPowerOfTwo(n), "n " << n << " is not a power of 2");
  HEXL_CHECK(n >= 16,
             "Don't support small transforms. Need n >= 16, got n = " << n);
  HEXL_VLOG(5, "root_of_unity_powers_cmplx_intrlvd "
                   << std::vector<std::complex<double>>(
                          root_of_unity_powers_cmplx_intrlvd,
                          root_of_unity_powers_cmplx_intrlvd + 2 * n));
  HEXL_VLOG(5, "operand_cmplx_intrlvd " << std::vector<std::complex<double>>(
                   operand_cmplx_intrlvd, operand_cmplx_intrlvd + 2 * n));

  uint64_t bits = static_cast<uint64_t>(log2(static_cast<double>(n)));
  for (size_t i = 0; i < n; i++) {
    size_t j = ReverseBits(i, bits);
    size_t ix = 2 * i;
    size_t jx = 2 * j;
    if (result_cmplx_intrlvd == operand_cmplx_intrlvd) {
      if (j > i) {
        double tmp = operand_cmplx_intrlvd[ix];
        result_cmplx_intrlvd[ix] = operand_cmplx_intrlvd[jx];
        result_cmplx_intrlvd[jx] = tmp;
        tmp = operand_cmplx_intrlvd[ix + 1];
        result_cmplx_intrlvd[ix + 1] = operand_cmplx_intrlvd[jx + 1];
        result_cmplx_intrlvd[jx + 1] = tmp;
      }
    } else {
      result_cmplx_intrlvd[ix] = operand_cmplx_intrlvd[jx];
      result_cmplx_intrlvd[ix + 1] = operand_cmplx_intrlvd[jx + 1];
    }
  }

  FFT_AVX512(result_cmplx_intrlvd, result_cmplx_intrlvd,
             root_of_unity_powers_cmplx_intrlvd, n, 0, inverse);
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
