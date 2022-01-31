// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fwd-fft-avx512.hpp"

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

// Takes operand as 8 complex interleaved: This is 8 real parts followed by
// its 8 imaginary parts.
// Returns operand as 1 complex interleaved: One real part followed by its
// imaginary part.
void ComplexFwdT1(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m, const double_t* scalar = nullptr) {
  size_t offset = 0;

  __m512d v_scalar;
  if (scalar != nullptr) {
    v_scalar = _mm512_set1_pd(*scalar);
  }

  // 8 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4

  for (size_t i = 0; i < (m >> 1); i += 8) {
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;
    __m512d* v_out_pt = reinterpret_cast<__m512d*>(X_real);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadFwdInterleavedT1(X_real, &v_X_real, &v_Y_real);
    ComplexLoadFwdInterleavedT1(X_imag, &v_X_imag, &v_Y_imag);

    // Weights
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[14], W_1C_intrlvd[12], W_1C_intrlvd[10], W_1C_intrlvd[8],
        W_1C_intrlvd[6], W_1C_intrlvd[4], W_1C_intrlvd[2], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[15], W_1C_intrlvd[13], W_1C_intrlvd[11], W_1C_intrlvd[9],
        W_1C_intrlvd[7], W_1C_intrlvd[5], W_1C_intrlvd[3], W_1C_intrlvd[1]);
    W_1C_intrlvd += 16;

    if (scalar != nullptr) {
      v_W_real = _mm512_mul_pd(v_W_real, v_scalar);
      v_W_imag = _mm512_mul_pd(v_W_imag, v_scalar);
      v_X_real = _mm512_mul_pd(v_X_real, v_scalar);
      v_X_imag = _mm512_mul_pd(v_X_imag, v_scalar);
    }

    ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    ComplexWriteFwdInterleavedT1(v_X_real, v_Y_real, v_X_imag, v_Y_imag,
                                 v_out_pt);

    offset += 32;
  }
}

void ComplexFwdT2(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m) {
  size_t offset = 0;

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

    // Weights
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[6], W_1C_intrlvd[6], W_1C_intrlvd[4], W_1C_intrlvd[4],
        W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[0], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[7], W_1C_intrlvd[7], W_1C_intrlvd[5], W_1C_intrlvd[5],
        W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[1], W_1C_intrlvd[1]);
    W_1C_intrlvd += 8;

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

void ComplexFwdT4(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m) {
  size_t offset = 0;

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

    ComplexLoadFwdInterleavedT4(X_real, &v_X_real, &v_Y_real);
    ComplexLoadFwdInterleavedT4(X_imag, &v_X_imag, &v_Y_imag);

    // Weights
    // x =  (11, 10,  9,  8, 3, 2, 1, 0)
    // y =  (15, 14, 13, 12, 7, 6, 5, 4)
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[2],
        W_1C_intrlvd[0], W_1C_intrlvd[0], W_1C_intrlvd[0], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[3],
        W_1C_intrlvd[1], W_1C_intrlvd[1], W_1C_intrlvd[1], W_1C_intrlvd[1]);

    W_1C_intrlvd += 4;

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

void ComplexFwdT8(double_t* result_8C_intrlvd,
                  const double_t* operand_8C_intrlvd,
                  const double_t* W_1C_intrlvd, uint64_t gap, uint64_t m) {
  size_t offset = 0;

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

    // Weights
    __m512d v_W_real = _mm512_set1_pd(*W_1C_intrlvd++);
    __m512d v_W_imag = _mm512_set1_pd(*W_1C_intrlvd++);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_op_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_op_pt_imag);

      __m512d v_Y_real = _mm512_loadu_pd(v_Y_op_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_op_pt_imag);

      ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag);

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
    }
    offset += (gap << 1);
  }
}

void ComplexStartFwdT8(double_t* result_8C_intrlvd,
                       const double_t* operand_1C_intrlvd,
                       const double_t* W_1C_intrlvd, uint64_t gap, uint64_t m) {
  size_t offset = 0;

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

    // Weights
    __m512d v_W_real = _mm512_set1_pd(*W_1C_intrlvd++);
    __m512d v_W_imag = _mm512_set1_pd(*W_1C_intrlvd++);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      __m512d v_X_real;
      __m512d v_X_imag;
      __m512d v_Y_real;
      __m512d v_Y_imag;

      ComplexLoadFwdInterleavedT8(v_X_op_pt, v_Y_op_pt, &v_X_real, &v_X_imag,
                                  &v_Y_real, &v_Y_imag);

      ComplexFwdButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag);

      _mm512_storeu_pd(v_X_r_pt_real, v_X_real);
      _mm512_storeu_pd(v_X_r_pt_imag, v_X_imag);

      _mm512_storeu_pd(v_Y_r_pt_real, v_Y_real);
      _mm512_storeu_pd(v_Y_r_pt_imag, v_Y_imag);

      // Increase operand & result pointers
      v_X_op_pt += 2;
      v_Y_op_pt += 2;
      v_X_r_pt_real += 2;
      v_X_r_pt_imag += 2;
      v_Y_r_pt_real += 2;
      v_Y_r_pt_imag += 2;
    }
    offset += (gap << 1);
  }
}

void Forward_FFT_ToBitReverseAVX512(
    double_t* result_cmplx_intrlvd, const double_t* operand_cmplx_intrlvd,
    const double_t* root_of_unity_powers_cmplx_intrlvd, const uint64_t n,
    const double_t* scale) {
  HEXL_CHECK(IsPowerOfTwo(n), "n " << n << " is not a power of 2");
  HEXL_CHECK(n > 2, "n " << n << " is not bigger than 2");

  size_t gap = n;  // (2*n >> 1) Interleaved complex numbers
  size_t m = 2;    // require twice the size
  size_t W_idx = m;

  // T8. First pass in case of out of place
  if (gap >= 16) {
    const double_t* W_cmplx_intrlvd =
        &root_of_unity_powers_cmplx_intrlvd[W_idx];
    ComplexStartFwdT8(result_cmplx_intrlvd, operand_cmplx_intrlvd,
                      W_cmplx_intrlvd, gap, m);
    m <<= 1;
    W_idx = m;
    gap >>= 1;
  }

  for (; gap >= 16; gap >>= 1) {
    const double_t* W_cmplx_intrlvd =
        &root_of_unity_powers_cmplx_intrlvd[W_idx];
    ComplexFwdT8(result_cmplx_intrlvd, result_cmplx_intrlvd, W_cmplx_intrlvd,
                 gap, m);
    m <<= 1;
    W_idx = m;
  }

  {
    // T4
    const double_t* W_cmplx_intrlvd =
        &root_of_unity_powers_cmplx_intrlvd[W_idx];
    ComplexFwdT4(result_cmplx_intrlvd, W_cmplx_intrlvd, m);
    m <<= 1;
    W_idx = m;

    // T2
    W_cmplx_intrlvd = &root_of_unity_powers_cmplx_intrlvd[W_idx];
    ComplexFwdT2(result_cmplx_intrlvd, W_cmplx_intrlvd, m);
    m <<= 1;
    W_idx = m;

    // T1
    W_cmplx_intrlvd = &root_of_unity_powers_cmplx_intrlvd[W_idx];
    ComplexFwdT1(result_cmplx_intrlvd, W_cmplx_intrlvd, m, scale);
    m <<= 1;
    W_idx = m;
  }

  HEXL_VLOG(5, "AVX512 returning FWD FFT result " << std::vector<double_t>(
                   result_cmplx_intrlvd, result_cmplx_intrlvd + n));
}

void BuildFloatingPointsAVX512(double_t* res_cmplx_intrlvd,
                               const uint64_t* plain, const uint64_t* threshold,
                               const uint64_t* decryption_modulus,
                               const double_t inv_scale, const size_t mod_size,
                               const size_t coeff_count) {
  const __m512i v_perm = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
  __m512d v_res_imag = _mm512_setzero_pd();
  __m512d* v_res_pt = reinterpret_cast<__m512d*>(res_cmplx_intrlvd);
  double_t two_pow_64 = std::pow(2.0, 64);

  for (size_t i = 0; i < coeff_count; i += 8) {
    __mmask8 zeros = 0xff;
    __mmask8 cond_lt_thr = 0;

    for (int32_t j = static_cast<int32_t>(mod_size) - 1; zeros && (j >= 0);
         j--) {
      const uint64_t* base = plain + j;
      __m512i v_thrld = _mm512_set1_epi64(*(threshold + j));
      __m512i v_plain = _mm512_set_epi64(
          *(base + (i + 7) * mod_size), *(base + (i + 6) * mod_size),
          *(base + (i + 5) * mod_size), *(base + (i + 4) * mod_size),
          *(base + (i + 3) * mod_size), *(base + (i + 2) * mod_size),
          *(base + (i + 1) * mod_size), *(base + (i + 0) * mod_size));

      cond_lt_thr = static_cast<unsigned char>(cond_lt_thr) |
                    static_cast<unsigned char>(
                        _mm512_mask_cmplt_epu64_mask(zeros, v_plain, v_thrld));
      zeros = _mm512_mask_cmpeq_epu64_mask(zeros, v_plain, v_thrld);
    }

    __mmask8 cond_ge_thr = ~static_cast<unsigned char>(cond_lt_thr);
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
      uint64_t tmp_v_ui[8];
      double_t tmp_v_pd[8];
      _mm512_storeu_epi64(tmp_v_ui, v_diff);
      HEXL_LOOP_UNROLL_8
      for (size_t t = 0; t < 8; t++) {
        tmp_v_pd[t] = static_cast<double_t>(tmp_v_ui[t]);
      }

      __m512d v_casted_diff = _mm512_loadu_pd(tmp_v_pd);
      __m512d v_scaled_diff = _mm512_mul_pd(v_casted_diff, v_scaled_p64);

      v_res_real = _mm512_mask_add_pd(v_res_real, cond_gt_dec_mod | cond_lt_thr,
                                      v_res_real, v_scaled_diff);
      v_res_real = _mm512_mask_sub_pd(v_res_real, cond_le_dec_mod, v_res_real,
                                      v_scaled_diff);
    }

    // Make res 1 complex interleaved
    v_res_real = _mm512_permutexvar_pd(v_perm, v_res_real);
    __m512d v_res1 = _mm512_shuffle_pd(v_res_real, v_res_imag, 0x00);
    __m512d v_res2 = _mm512_shuffle_pd(v_res_real, v_res_imag, 0xff);
    _mm512_storeu_pd(v_res_pt++, v_res1);
    _mm512_storeu_pd(v_res_pt++, v_res2);
  }
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
