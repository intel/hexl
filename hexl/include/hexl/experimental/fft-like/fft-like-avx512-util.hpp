// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

// ************************************ T1 ************************************

// ComplexLoadFwdInterleavedT1:
// Assumes ComplexLoadFwdInterleavedT2 was used before.
// Given input: 15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0
// Returns
// *out1 =  (14, 12, 10, 8, 6, 4, 2, 0);
// *out2 =  (15, 13, 11, 9, 7, 5, 3, 1);
//
// Given output: 15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0
inline void ComplexLoadFwdInterleavedT1(const double* arg, __m512d* out1,
                                        __m512d* out2) {
  const __m512i vperm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);

  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  // 13 12 9  8  5  4  1  0
  __m512d v_7to0 = _mm512_loadu_pd(arg_512);
  arg_512 += 2;
  // 15 14 11 10 7  6  3  2
  __m512d v_15to8 = _mm512_loadu_pd(arg_512);

  // 12, 13, 8, 9, 4, 5, 0, 1
  __m512d perm_1 = _mm512_permutexvar_pd(vperm_idx, v_7to0);
  // 14, 15, 10, 11, 6, 7, 2, 3
  __m512d perm_2 = _mm512_permutexvar_pd(vperm_idx, v_15to8);

  // 14, 12, 10, 8, 6, 4, 2, 0
  *out1 = _mm512_mask_blend_pd(0xaa, v_7to0, perm_2);
  // 15, 13, 11, 9, 7, 5, 3, 1
  *out2 = _mm512_mask_blend_pd(0x55, v_15to8, perm_1);
}

// ComplexWriteFwdInterleavedT1:
// Assumes ComplexLoadFwdInterleavedT1 was used before.
// Given inputs:
// 15i, 13i, 11i, 9i, 7i, 5i, 3i, 1i, 15r, 13r, 11r, 9r, 7r, 5r, 3r, 1r,
// 14i, 12i, 10i, 8i, 6i, 4i, 2i, 0i, 14r, 12r, 10r, 8r, 6r, 4r, 2r, 0r
// As seen with internal indexes:
//  @param arg_yr = (15r, 14r, 13r, 12r, 11r, 10r, 9r, 8r);
//  @param arg_xr = ( 7r,  6r,  5r,  4r,  3r,  2r, 1r, 0r);
//  @param arg_yi = (15i, 14i, 13i, 12i, 11i, 10i, 9i, 8i);
//  @param arg_xi = ( 7i,  6i,  5i,  4i,  3i,  2i, 1i, 0i);
//  Writes out =
//  {15i, 15r, 7i, 7r, 14i, 14r, 6i, 6r, 13i, 13r,  5i, 5r, 12i, 12r, 4i, 4r,
//   11i, 11r, 3i, 3r, 10i, 10r, 2i, 2r,  9i,  9r,  1i, 1r,  8i,  8r, 0i, 0r}
//
// Given output:
// 15i, 15r, 14i, 14r, 13i, 13r, 12i, 12r, 11i, 11r, 10i, 10r, 9i, 9r, 8i, 8r,
// 7i, 7r, 6i, 6r, 5i, 5r, 4i, 4r, 3i, 3r, 2i, 2r, 1i, 1r, 0i, 0r
inline void ComplexWriteFwdInterleavedT1(__m512d arg_xr, __m512d arg_yr,
                                         __m512d arg_xi, __m512d arg_yi,
                                         __m512d* out) {
  const __m512i vperm_4hi_4lo_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i v_X_out_idx = _mm512_set_epi64(3, 1, 7, 5, 2, 0, 6, 4);
  const __m512i v_Y_out_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

  // Real part
  // in:  14r, 12r, 10r, 8r,  6r,  4r,  2r, 0r
  // ->    6r,  4r,  2r, 0r, 14r, 12r, 10r, 8r
  arg_xr = _mm512_permutexvar_pd(vperm_4hi_4lo_idx, arg_xr);

  // arg_yr: 15r, 13r, 11r,  9r,  7r,  5r,  3r, 1r
  // ->       6r,  4r,  2r,  0r,  7r,  5r,  3r, 1r
  __m512d perm_1 = _mm512_mask_blend_pd(0x0f, arg_xr, arg_yr);
  // ->      15r, 13r, 11r,  9r, 14r, 12r, 10r, 8r
  __m512d perm_2 = _mm512_mask_blend_pd(0xf0, arg_xr, arg_yr);

  //  7r,  3r,  6r,  2r,  5r, 1r,  4r, 0r
  arg_xr = _mm512_permutexvar_pd(v_X_out_idx, perm_1);
  // 15r, 11r, 14r, 10r, 13r, 9r, 12r, 8r
  arg_yr = _mm512_permutexvar_pd(v_Y_out_idx, perm_2);

  // Imaginary part
  // in:  14i, 12i, 10i, 8i,  6i,  4i,  2i, 0i
  // ->    6i,  4i,  2i, 0i, 14i, 12i, 10i, 8i
  arg_xi = _mm512_permutexvar_pd(vperm_4hi_4lo_idx, arg_xi);

  // arg_yr: 15i, 13i, 11i,  9i,  7i,  5i,  3i, 1i
  // ->       6i,  4i,  2i,  0i,  7i,  5i,  3i, 1i
  perm_1 = _mm512_mask_blend_pd(0x0f, arg_xi, arg_yi);
  // ->      15i, 13i, 11i,  9i, 14i, 12i, 10i, 8i
  perm_2 = _mm512_mask_blend_pd(0xf0, arg_xi, arg_yi);

  //  7i,  3i,  6i,  2i,  5i, 1i,  4i, 0i
  arg_xi = _mm512_permutexvar_pd(v_X_out_idx, perm_1);
  // 15i, 11i, 14i, 10i, 13i, 9i, 12i, 8i
  arg_yi = _mm512_permutexvar_pd(v_Y_out_idx, perm_2);

  // Merge
  // 00000000 > 3i 3r 2i 2r 1i 1r 0i 0r
  __m512d out1 = _mm512_shuffle_pd(arg_xr, arg_xi, 0x00);
  // 11111111 > 7i 7r 6i 6r 5i 5r 4i 4r
  __m512d out2 = _mm512_shuffle_pd(arg_xr, arg_xi, 0xff);

  // 00000000 > 11i 11r 10i 10r  9i  9r  8i  8r
  __m512d out3 = _mm512_shuffle_pd(arg_yr, arg_yi, 0x00);
  // 11111111 > 15i 15r 14i 14r 13i 13r 12i 12r
  __m512d out4 = _mm512_shuffle_pd(arg_yr, arg_yi, 0xff);

  _mm512_storeu_pd(out++, out1);
  _mm512_storeu_pd(out++, out2);
  _mm512_storeu_pd(out++, out3);
  _mm512_storeu_pd(out++, out4);
}

// ComplexLoadInvInterleavedT1:
// Given input: 15i 15r 14i 14r 13i 13r 12i 12r 11i 11r 10i 10r 9i 9r 8i 8r
//              7i   7r  6i  6r  5i  5r  4i  4r  3i  3r  2i  2r 1i 1r 0i 0r
// Returns
// *out1_r =  (14r, 10r, 6r, 2r, 12r, 8r, 4r, 0r);
// *out1_i =  (14i, 10i, 6i, 2i, 12i, 8i, 4i, 0i);
// *out2_r =  (15r, 11r, 7r, 3r, 13r, 9r, 5r, 1r);
// *out2_i =  (15i, 11i, 7i, 3i, 13i, 9i, 5i, 1i);
//
// Given output:
// 15i, 11i, 7i, 3i, 13i, 9i, 5i, 1i, 15r, 11r, 7r, 3r, 13r, 9r, 5r, 1r,
// 14i, 10i, 6i, 2i, 12i, 8i, 4i, 0i, 14r, 10r, 6r, 2r, 12r, 8r, 4r, 0r
inline void ComplexLoadInvInterleavedT1(const double* arg, __m512d* out1_r,
                                        __m512d* out1_i, __m512d* out2_r,
                                        __m512d* out2_i) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  //  3i   3r  2i  2r  1i  1r  0i  0r
  __m512d v_3to0 = _mm512_loadu_pd(arg_512++);
  //  7i   7r  6i  6r  5i  5r  4i  4r
  __m512d v_7to4 = _mm512_loadu_pd(arg_512++);
  // 11i  11r 10i 10r  9i  9r  8i  8r
  __m512d v_11to8 = _mm512_loadu_pd(arg_512++);
  // 15i  15r 14i 14r 13i 13r 12i 12r
  __m512d v_15to12 = _mm512_loadu_pd(arg_512++);

  // 00000000 >  7r  3r  6r  2r  5r 1r  4r 0r
  __m512d v_7to0_r = _mm512_shuffle_pd(v_3to0, v_7to4, 0x00);
  // 11111111 >  7i  3i  6i  2i  5i 1i  4i 0i
  __m512d v_7to0_i = _mm512_shuffle_pd(v_3to0, v_7to4, 0xff);
  // 00000000 > 15r 11r 14r 10r 13r 9r 12r 8r
  __m512d v_15to8_r = _mm512_shuffle_pd(v_11to8, v_15to12, 0x00);
  // 11111111 > 15i 11i 14i 10i 13i 9i 12i 8i
  __m512d v_15to8_i = _mm512_shuffle_pd(v_11to8, v_15to12, 0xff);

  // real
  const __m512i v1_perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
  //  6  2  7  3  4 0  5 1
  __m512d v1r = _mm512_permutexvar_pd(v1_perm_idx, v_7to0_r);
  // 14 10 15 11 12 8 13 9
  __m512d v2r = _mm512_permutexvar_pd(v1_perm_idx, v_15to8_r);
  // 11001100 > 14 10 6 2 12 8 4 0
  *out1_r = _mm512_mask_blend_pd(0xcc, v_7to0_r, v2r);
  // 11001100 > 15 11 7 3 13 9 5 1
  *out2_r = _mm512_mask_blend_pd(0xcc, v1r, v_15to8_r);

  // imag
  //  6  2  7  3  4 0  5 1
  __m512d v1i = _mm512_permutexvar_pd(v1_perm_idx, v_7to0_i);
  // 14 10 15 11 12 8 13 9
  __m512d v2i = _mm512_permutexvar_pd(v1_perm_idx, v_15to8_i);
  // 11001100 > 14 10 6 2 12 8 4 0
  *out1_i = _mm512_mask_blend_pd(0xcc, v_7to0_i, v2i);
  // 11001100 > 15 11 7 3 13 9 5 1
  *out2_i = _mm512_mask_blend_pd(0xcc, v1i, v_15to8_i);
}

// ************************************ T2 ************************************

// ComplexLoadFwdInterleavedT2:
// Assumes ComplexLoadFwdInterleavedT4 was used before.
// Given input:  15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0
// Returns
// *out1 =  (13, 12,  9,  8, 5, 4, 1, 0)
// *out2 =  (15, 14, 11, 10, 7, 6, 3, 2)
//
// Given output: 15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0
inline void ComplexLoadFwdInterleavedT2(const double* arg, __m512d* out1,
                                        __m512d* out2) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  // Values were swapped in T4
  // 11, 10, 9, 8, 3, 2, 1, 0
  __m512d v1 = _mm512_loadu_pd(arg_512);
  arg_512 += 2;
  // 15, 14, 13, 12, 7, 6, 5, 4
  __m512d v2 = _mm512_loadu_pd(arg_512);

  const __m512i v1_perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);

  __m512d v1_perm = _mm512_permutexvar_pd(v1_perm_idx, v1);
  __m512d v2_perm = _mm512_permutexvar_pd(v1_perm_idx, v2);

  *out1 = _mm512_mask_blend_pd(0xcc, v1, v2_perm);
  *out2 = _mm512_mask_blend_pd(0xcc, v1_perm, v2);
}

// ComplexLoadInvInterleavedT2:
// Assumes ComplexLoadInvInterleavedT1 was used before.
// Given input: 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0
// Returns
// *out1 =  (13,  9, 5, 1, 12,  8, 4, 0)
// *out2 =  (15, 11, 7, 3, 14, 10, 6, 2)
//
// Given output: 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0
inline void ComplexLoadInvInterleavedT2(const double* arg, __m512d* out1,
                                        __m512d* out2) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  // 14 10 6 2 12 8 4 0
  __m512d v1 = _mm512_loadu_pd(arg_512);
  arg_512 += 2;
  // 15 11 7 3 13 9 5 1
  __m512d v2 = _mm512_loadu_pd(arg_512);

  const __m512i v1_perm_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

  // 12 8 4 0 14 10 6 2
  __m512d v1_perm = _mm512_permutexvar_pd(v1_perm_idx, v1);
  // 13 9 5 1 15 11 7 3
  __m512d v2_perm = _mm512_permutexvar_pd(v1_perm_idx, v2);

  // 11110000 > 13  9 5 1 12  8 4 0
  *out1 = _mm512_mask_blend_pd(0xf0, v1, v2_perm);
  // 11110000 > 15 11 7 3 14 10 6 2
  *out2 = _mm512_mask_blend_pd(0xf0, v1_perm, v2);
}

// ************************************ T4 ************************************

// Complex LoadFwdInterleavedT4:
// Given input: 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
// Returns
// *out1 =  (11, 10,  9,  8, 3, 2, 1, 0)
// *out2 =  (15, 14, 13, 12, 7, 6, 5, 4)
//
// Given output: 15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0
inline void ComplexLoadFwdInterleavedT4(const double* arg, __m512d* out1,
                                        __m512d* out2) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  __m512d v_7to0 = _mm512_loadu_pd(arg_512);
  arg_512 += 2;
  __m512d v_15to8 = _mm512_loadu_pd(arg_512);
  __m512d perm_hi = _mm512_permutexvar_pd(vperm2_idx, v_15to8);
  *out1 = _mm512_mask_blend_pd(0x0f, perm_hi, v_7to0);
  *out2 = _mm512_mask_blend_pd(0xf0, perm_hi, v_7to0);
  *out2 = _mm512_permutexvar_pd(vperm2_idx, *out2);
}

// ComplexLoadInvInterleavedT4:
// Assumes ComplexLoadInvInterleavedT2 was used before.
// Given input: 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0
// Returns
// *out1 =  (11,  9, 3, 1, 10,  8, 2, 0)
// *out2 =  (15, 13, 7, 5, 14, 12, 6, 4)
//
// Given output: 15, 13, 7, 5, 14, 12, 6, 4, 11, 9, 3, 1, 10, 8, 2, 0

inline void ComplexLoadInvInterleavedT4(const double* arg, __m512d* out1,
                                        __m512d* out2) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  // 13, 9, 5, 1,  12, 8,  4, 0
  __m512d v1 = _mm512_loadu_pd(arg_512);
  arg_512 += 2;
  // 15, 11, 7, 3, 14, 10, 6, 2
  __m512d v2 = _mm512_loadu_pd(arg_512);

  // 00000000 > 11  9 3 1 10  8 2 0
  *out1 = _mm512_shuffle_pd(v1, v2, 0x00);
  // 11111111 > 15 13 7 5 14 12 6 4
  *out2 = _mm512_shuffle_pd(v1, v2, 0xff);
}

// ComplexWriteInvInterleavedT4:
// Assuming ComplexLoadInvInterleavedT4 was used before.
// Given inputs: 15, 13, 7, 5, 14, 12, 6, 4, 11, 9, 3, 1, 10, 8, 2, 0
// Seen Internally:
// @param arg1 = ( 7,  6,  5,  4,  3,  2, 1, 0);
// @param arg2 = (15, 14, 13, 12, 11, 10, 9, 8);
// Writes out = {15, 11, 14, 10, 7, 3, 6, 2,
//               13,  9, 12,  8, 5, 1, 4, 0}
//
// Given output: 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
inline void ComplexWriteInvInterleavedT4(__m512d arg1, __m512d arg2,
                                         __m512d* out) {
  const __m512i vperm_4hi_4lo_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i vperm1 = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
  const __m512i vperm2 = _mm512_set_epi64(5, 1, 4, 0, 7, 3, 6, 2);

  // in: 11  9  3 1 10  8  2  0
  // ->  11 10  9 8  3  2  1  0
  arg1 = _mm512_permutexvar_pd(vperm1, arg1);
  // in: 15 13  7 5 14 12  6  4
  // ->   7  6  5 4 15 14 13 12
  arg2 = _mm512_permutexvar_pd(vperm2, arg2);

  //  7  6 5 4  3  2  1  0
  __m512d out1 = _mm512_mask_blend_pd(0xf0, arg1, arg2);
  // 11 10 9 8 15 14 13 12
  __m512d out2 = _mm512_mask_blend_pd(0x0f, arg1, arg2);
  // 15 14 13 12 11 10 9 8
  out2 = _mm512_permutexvar_pd(vperm_4hi_4lo_idx, out2);

  _mm512_storeu_pd(out, out1);
  out += 2;
  _mm512_storeu_pd(out, out2);
}

// ************************************ T8 ************************************

// ComplexLoadFwdInterleavedT8:
// Given inputs: 7i, 7r, 6i, 6r, 5i, 5r, 4i, 4r, 3i, 3r, 2i, 2r, 1i, 1r, 0i, 0r
// Seen Internally:
//  v_X1 = ( 7,  6,  5,  4,  3,  2, 1, 0);
//  v_X2 = (15, 14, 13, 12, 11, 10, 9, 8);
// Writes out = {15, 13, 11, 9, 7, 5, 3, 1,
//               14, 12, 10, 8, 6, 4, 2, 0}
//
// Given output: 7i, 6i, 5i, 4i, 3i, 2i, 1i, 0i, 7r, 6r, 5r, 4r, 3r, 2r, 1r, 0r
inline void ComplexLoadFwdInterleavedT8(const __m512d* arg_x,
                                        const __m512d* arg_y, __m512d* out1_r,
                                        __m512d* out1_i, __m512d* out2_r,
                                        __m512d* out2_i) {
  const __m512i v_perm_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

  // 3i, 3r, 2i, 2r, 1i, 1r, 0i, 0r
  __m512d v_X1 = _mm512_loadu_pd(arg_x++);
  // 7i, 7r, 6i, 6r, 5i, 5r, 4i, 4r
  __m512d v_X2 = _mm512_loadu_pd(arg_x);
  // 7r, 3r, 6r, 2r, 5r, 1r, 4r, 0r
  *out1_r = _mm512_shuffle_pd(v_X1, v_X2, 0x00);
  // 7i, 3i, 6i, 2i, 5i, 1i, 4i, 0i
  *out1_i = _mm512_shuffle_pd(v_X1, v_X2, 0xff);
  // 7r, 6r, 5r, 4r, 3r, 2r, 1r, 0r
  *out1_r = _mm512_permutexvar_pd(v_perm_idx, *out1_r);
  // 7i, 6i, 5i, 4i, 3i, 2i, 1i, 0i
  *out1_i = _mm512_permutexvar_pd(v_perm_idx, *out1_i);

  __m512d v_Y1 = _mm512_loadu_pd(arg_y++);
  __m512d v_Y2 = _mm512_loadu_pd(arg_y);
  *out2_r = _mm512_shuffle_pd(v_Y1, v_Y2, 0x00);
  *out2_i = _mm512_shuffle_pd(v_Y1, v_Y2, 0xff);
  *out2_r = _mm512_permutexvar_pd(v_perm_idx, *out2_r);
  *out2_i = _mm512_permutexvar_pd(v_perm_idx, *out2_i);
}

// ComplexWriteInvInterleavedT8:
// Assuming ComplexLoadInvInterleavedT4 was used before.
// Given inputs: 7i, 6i, 5i, 4i, 3i, 2i, 1i, 0i, 7r, 6r, 5r, 4r, 3r, 2r, 1r, 0r
// Seen Internally:
// @param arg1 = ( 7,  6,  5,  4,  3,  2, 1, 0);
// @param arg2 = (15, 14, 13, 12, 11, 10, 9, 8);
// Writes out = {15, 7, 14, 6, 13, 5, 12, 4,
//               11, 3, 10, 2,  9, 1,  8, 0}
//
// Given output: 7i, 7r, 6i, 6r, 5i, 5r, 4i, 4r, 3i, 3r, 2i, 2r, 1i, 1r, 0i, 0r
inline void ComplexWriteInvInterleavedT8(__m512d* v_X_real, __m512d* v_X_imag,
                                         __m512d* v_Y_real, __m512d* v_Y_imag,
                                         __m512d* v_X_pt, __m512d* v_Y_pt) {
  const __m512i vperm = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
  // in:  7r  6r  5r  4r  3r  2r  1r  0r
  // ->   7r  3r  6r  2r  5r  1r  4r  0r
  *v_X_real = _mm512_permutexvar_pd(vperm, *v_X_real);
  // in:  7i  6i  5i  4i  3i  2i  1i  0i
  // ->   7i  3i  6i  2i  5i  1i  4i  0i
  *v_X_imag = _mm512_permutexvar_pd(vperm, *v_X_imag);
  // in: 15r 14r 13r 12r 11r 10r  9r  8r
  // ->  15r 11r 14r 10r 13r  9r 12r  8r
  *v_Y_real = _mm512_permutexvar_pd(vperm, *v_Y_real);
  // in: 15i 14i 13i 12i 11i 10i  9i  8i
  // ->  15i 11i 14i 10i 13i  9i 12i  8i
  *v_Y_imag = _mm512_permutexvar_pd(vperm, *v_Y_imag);

  // 00000000 >  3i  3r  2i  2r  1i  1r  0i  0r
  __m512d v_X1 = _mm512_shuffle_pd(*v_X_real, *v_X_imag, 0x00);
  // 11111111 >  7i  7r  6i  6r  5i  5r  4i  4r
  __m512d v_X2 = _mm512_shuffle_pd(*v_X_real, *v_X_imag, 0xff);
  // 00000000 > 11i 11r 10i 10r  9i  9r  8i  8r
  __m512d v_Y1 = _mm512_shuffle_pd(*v_Y_real, *v_Y_imag, 0x00);
  // 11111111 > 15i 15r 14i 14r 13i 13r 12i 12r
  __m512d v_Y2 = _mm512_shuffle_pd(*v_Y_real, *v_Y_imag, 0xff);

  _mm512_storeu_pd(v_X_pt++, v_X1);
  _mm512_storeu_pd(v_X_pt, v_X2);
  _mm512_storeu_pd(v_Y_pt++, v_Y1);
  _mm512_storeu_pd(v_Y_pt, v_Y2);
}
#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
