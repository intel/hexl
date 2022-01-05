// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>

#include <functional>
#include <vector>

#include "hexl/fft/fft.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

// ************************************ T1 ************************************

// Assuming LoadFwdInterleavedT2 was used before.
// Given input: 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
// Returns
// *out1 =  (14, 12, 10, 8, 6, 4, 2, 0);
// *out2 =  (15, 13, 11, 9, 7, 5, 3, 1);
//
// Given output: 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
inline void LoadFwdInterleavedT1(const double_t* arg, __m512d* out1,
                                 __m512d* out2) {
  const __m512i vperm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);

  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  // 13 12 9  8  5  4  1  0
  __m512d v_7to0 = _mm512_loadu_pd(arg_512++);
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

inline void ComplexLoadFwdInterleavedT1(const double_t* arg, __m512d* out1,
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

// Assuming LoadFwdInterleavedT1 was used before.
// Given inputs: 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
// Seen Internally:
// @param arg1 = (15, 14, 13, 12, 11, 10, 9, 8);
// @param arg2 = (7, 6, 5, 4, 3, 2, 1, 0);
// Writes out = {15, 7, 14,  6, 13, 5, 12, 4,
//               11, 3, 10, 2, 9, 1, 8, 0}
//
// Given output: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
inline void WriteFwdInterleavedT1(__m512d arg1, __m512d arg2, __m512d* out) {
  const __m512i vperm_4hi_4lo_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i v_X_out_idx = _mm512_set_epi64(3, 7, 2, 6, 1, 5, 0, 4);
  const __m512i v_Y_out_idx = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);

  // 3, 2, 1, 0, 7, 6, 5, 4
  arg1 = _mm512_permutexvar_pd(vperm_4hi_4lo_idx, arg1);

  // 3, 2, 1, 0, 11, 10, 9, 8
  __m512d perm_1 = _mm512_mask_blend_pd(0x0f, arg1, arg2);
  // 15, 14, 13, 12, 7, 6, 5, 4
  __m512d perm_2 = _mm512_mask_blend_pd(0xf0, arg1, arg2);

  // 11, 3, 10, 2, 9, 1, 8, 0
  arg1 = _mm512_permutexvar_pd(v_X_out_idx, perm_1);
  // 15, 7, 14,  6, 13, 5, 12, 4
  arg2 = _mm512_permutexvar_pd(v_Y_out_idx, perm_2);

  _mm512_storeu_pd(out++, arg1);
  _mm512_storeu_pd(out, arg2);
}

inline void ComplexWriteFwdInterleavedT1(__m512d arg1, __m512d arg2,
                                         __m512d* out) {
  const __m512i vperm_4hi_4lo_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i v_X_out_idx = _mm512_set_epi64(3, 7, 2, 6, 1, 5, 0, 4);
  const __m512i v_Y_out_idx = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);

  // 3, 2, 1, 0, 7, 6, 5, 4
  arg1 = _mm512_permutexvar_pd(vperm_4hi_4lo_idx, arg1);

  // 3, 2, 1, 0, 11, 10, 9, 8
  __m512d perm_1 = _mm512_mask_blend_pd(0x0f, arg1, arg2);
  // 15, 14, 13, 12, 7, 6, 5, 4
  __m512d perm_2 = _mm512_mask_blend_pd(0xf0, arg1, arg2);

  // 11, 3, 10, 2, 9, 1, 8, 0
  arg1 = _mm512_permutexvar_pd(v_X_out_idx, perm_1);
  // 15, 7, 14,  6, 13, 5, 12, 4
  arg2 = _mm512_permutexvar_pd(v_Y_out_idx, perm_2);

  _mm512_storeu_pd(out, arg1);
  out += 2;
  _mm512_storeu_pd(out, arg2);
}

// Given input: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
// Returns
// *out1 =  (14, 6, 12, 4, 10, 2, 8, 0);
// *out2 =  (15, 7, 13, 5, 11, 3, 9, 1);
//
// Given output: 0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15
inline void ComplexLoadInvInterleavedT1(const double_t* arg, __m512d* out1,
                                        __m512d* out2) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  // 7   6  5  4  3  2  1  0
  __m512d v_7to0 = _mm512_loadu_pd(arg_512);
  arg_512 += 2;
  // 15 14 13 12 11 10  9  8
  __m512d v_15to8 = _mm512_loadu_pd(arg_512);

  // 11111111 > 15, 7, 13, 5, 11, 3, 9, 1
  *out1 = _mm512_shuffle_pd(v_7to0, v_15to8, 0xff);
  // 00000000 > 14, 6, 12, 4, 10, 2, 8, 0
  *out2 = _mm512_shuffle_pd(v_7to0, v_15to8, 0x00);
}

// ************************************ T2 ************************************

// Assuming LoadFwdInterleavedT4 was used before.
// Given input:  0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15
// Returns
// *out1 =  (13, 12, 9, 8, 5, 4, 1, 0)
// *out2 =  (15, 14, 11, 10, 7, 6, 3, 2)
//
// Given output: 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
inline void LoadFwdInterleavedT2(const double_t* arg, __m512d* out1,
                                 __m512d* out2) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  // Values were swapped in T4
  // 11, 10, 9, 8, 3, 2, 1, 0
  __m512d v1 = _mm512_loadu_pd(arg_512++);
  // 15, 14, 13, 12, 7, 6, 5, 4
  __m512d v2 = _mm512_loadu_pd(arg_512);

  const __m512i v1_perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);

  __m512d v1_perm = _mm512_permutexvar_pd(v1_perm_idx, v1);
  __m512d v2_perm = _mm512_permutexvar_pd(v1_perm_idx, v2);

  *out1 = _mm512_mask_blend_pd(0xcc, v1, v2_perm);
  *out2 = _mm512_mask_blend_pd(0xcc, v1_perm, v2);
}

inline void ComplexLoadFwdInterleavedT2(const double_t* arg, __m512d* out1,
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

// Assuming ComplexLoadInvInterleavedT1 was used before.
// Given input: 0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15
// Returns
// *out1 =  (13, 5, 12, 4,  9, 1,  8, 0)
// *out2 =  (15, 7, 14, 6, 11, 3, 10, 2)
//
// Given output: 0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15
inline void ComplexLoadInvInterleavedT2(const double_t* arg, __m512d* out1,
                                        __m512d* out2) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  // 14  6 12 4 10 2  8 0
  __m512d v1 = _mm512_loadu_pd(arg_512);
  arg_512 += 2;
  // 15  7 13 5 11 3  9 1
  __m512d v2 = _mm512_loadu_pd(arg_512);

  const __m512i v1_perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);

  // 12 4 14 6 8 0 10 2
  __m512d v1_perm = _mm512_permutexvar_pd(v1_perm_idx, v1);
  // 13 5 15 7 9 1 11 3
  __m512d v2_perm = _mm512_permutexvar_pd(v1_perm_idx, v2);

  // 11001100 > 13 5 12 4  9 1  8 0
  *out1 = _mm512_mask_blend_pd(0xcc, v1, v2_perm);
  // 11001100 > 15 7 14 6 11 3 10 2
  *out2 = _mm512_mask_blend_pd(0xcc, v1_perm, v2);
}

// ************************************ T4 ************************************

// Given input: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
// Returns
// *out1 =  (11, 10,  9,  8, 3, 2, 1, 0)
// *out2 =  (15, 14, 13, 12, 7, 6, 5, 4)
//
// Given output: 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15
inline void LoadFwdInterleavedT4(const double_t* arg, __m512d* out1,
                                 __m512d* out2) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  __m512d v_7to0 = _mm512_loadu_pd(arg_512++);
  __m512d v_15to8 = _mm512_loadu_pd(arg_512);
  __m512d perm_hi = _mm512_permutexvar_pd(vperm2_idx, v_15to8);
  *out1 = _mm512_mask_blend_pd(0x0f, perm_hi, v_7to0);
  *out2 = _mm512_mask_blend_pd(0xf0, perm_hi, v_7to0);
  *out2 = _mm512_permutexvar_pd(vperm2_idx, *out2);
}

inline void ComplexLoadFwdInterleavedT4(const double_t* arg, __m512d* out1,
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

// Assuming ComplexLoadInvInterleavedT2 was used before.
// Given input: 0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15
// Returns
// *out1 =  (11, 3, 10, 2,  9, 1,  8, 0)
// *out2 =  (15, 7, 14, 6, 13, 5, 12, 4)
//
// Given output: 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
inline void ComplexLoadInvInterleavedT4(const double_t* arg, __m512d* out1,
                                        __m512d* out2) {
  const __m512d* arg_512 = reinterpret_cast<const __m512d*>(arg);

  // 13, 5, 12, 4,  9, 1,  8, 0
  __m512d v1 = _mm512_loadu_pd(arg_512);
  arg_512 += 2;
  // 15, 7, 14, 6, 11, 3, 10, 2
  __m512d v2 = _mm512_loadu_pd(arg_512);

  const __m512i perm_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

  // 9,  1,  8, 0, 13, 5, 12, 4
  __m512d v1_perm = _mm512_permutexvar_pd(perm_idx, v1);
  // 11, 3, 10, 2, 15, 7, 14, 6
  __m512d v2_perm = _mm512_permutexvar_pd(perm_idx, v2);

  // 11110000 > 11, 3, 10, 2,  9, 1,  8, 0
  *out1 = _mm512_mask_blend_pd(0xcf, v1, v2_perm);
  // 11110000 > 15, 7, 14, 6, 13, 5, 12, 4
  *out2 = _mm512_mask_blend_pd(0xf0, v1_perm, v2);
}

// Assuming ComplexLoadInvInterleavedT4 was used before.
// Given inputs: 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
// Seen Internally:
// @param arg1 = (7,   6,  5,  4,  3,  2, 1, 0);
// @param arg2 = (15, 14, 13, 12, 11, 10, 9, 8);
// Writes out = {15, 7, 14,  6, 13, 5, 12, 4,
//               11, 3, 10, 2, 9, 1, 8, 0}
//
// Given output: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
inline void ComplexWriteInvInterleavedT4(__m512d arg1, __m512d arg2,
                                         __m512d* out) {
  const __m512i vperm_4hi_4lo_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i vperm1 = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  const __m512i vperm2 = _mm512_set_epi64(6, 4, 2, 0, 7, 5, 3, 1);

  // 11 3 10 2  9 1  8 0  ->  11 10 9 8  3  2  1  0
  arg1 = _mm512_permutexvar_pd(vperm1, arg1);
  // 15 7 14 6 13 5 12 4  ->   7  6 5 4 15 14 13 12
  arg2 = _mm512_permutexvar_pd(vperm2, arg2);

  //  7  6 5 4  3  2  1  0
  arg1 = _mm512_mask_blend_pd(0xf0, arg1, arg2);
  // 11 10 9 8 15 14 13 12
  arg2 = _mm512_mask_blend_pd(0x0f, arg1, arg2);
  // 15 14 13 12 11 10 9 8
  arg2 = _mm512_permutexvar_pd(vperm_4hi_4lo_idx, arg2);

  _mm512_storeu_pd(out, arg1);
  out += 2;
  _mm512_storeu_pd(out, arg2);
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
