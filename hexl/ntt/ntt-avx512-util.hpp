// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>

#include <functional>
#include <vector>

#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

// Given input: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
// Returns
// *out1 =  _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
// *out2 =  _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
inline void LoadFwdInterleavedT1(const uint64_t* arg, __m512i* out1,
                                 __m512i* out2) {
  const __m512i* arg_512 = reinterpret_cast<const __m512i*>(arg);

  // 0, 1, 2, 3, 4, 5, 6, 7
  __m512i v1 = _mm512_loadu_si512(arg_512++);
  // 8, 9, 10, 11, 12, 13, 14, 15
  __m512i v2 = _mm512_loadu_si512(arg_512);

  const __m512i perm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);

  // 1, 0, 3, 2, 5, 4, 7, 6
  __m512i v1_perm = _mm512_permutexvar_epi64(perm_idx, v1);
  // 9, 8, 11, 10, 13, 12, 15, 14
  __m512i v2_perm = _mm512_permutexvar_epi64(perm_idx, v2);

  *out1 = _mm512_mask_blend_epi64(0xaa, v1, v2_perm);
  *out2 = _mm512_mask_blend_epi64(0xaa, v1_perm, v2);
}

// Given input: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
// Returns
// *out1 =  _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
// *out2 =  _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
inline void LoadInvInterleavedT1(const uint64_t* arg, __m512i* out1,
                                 __m512i* out2) {
  const __m512i vperm_hi_idx = _mm512_set_epi64(6, 4, 2, 0, 7, 5, 3, 1);
  const __m512i vperm_lo_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

  const __m512i* arg_512 = reinterpret_cast<const __m512i*>(arg);

  // 7, 6, 5, 4, 3, 2, 1, 0
  __m512i v_7to0 = _mm512_loadu_si512(arg_512++);
  // 15, 14, 13, 12, 11, 10, 9, 8
  __m512i v_15to8 = _mm512_loadu_si512(arg_512);
  // 7, 5, 3, 1, 6, 4, 2, 0
  __m512i perm_lo = _mm512_permutexvar_epi64(vperm_lo_idx, v_7to0);
  // 14, 12, 10, 8, 15, 13, 11, 9
  __m512i perm_hi = _mm512_permutexvar_epi64(vperm_hi_idx, v_15to8);

  *out1 = _mm512_mask_blend_epi64(0x0f, perm_hi, perm_lo);
  *out2 = _mm512_mask_blend_epi64(0xf0, perm_hi, perm_lo);
  *out2 = _mm512_permutexvar_epi64(vperm2_idx, *out2);
}

// Given input: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
// Returns
// *out1 =  _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
// *out2 =  _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2)
inline void LoadFwdInterleavedT2(const uint64_t* arg, __m512i* out1,
                                 __m512i* out2) {
  const __m512i* arg_512 = reinterpret_cast<const __m512i*>(arg);

  // 11, 10, 9, 8, 3, 2, 1, 0
  __m512i v1 = _mm512_loadu_si512(arg_512++);
  // 15, 14, 13, 12, 7, 6, 5, 4
  __m512i v2 = _mm512_loadu_si512(arg_512);

  const __m512i v1_perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);

  __m512i v1_perm = _mm512_permutexvar_epi64(v1_perm_idx, v1);
  __m512i v2_perm = _mm512_permutexvar_epi64(v1_perm_idx, v2);

  *out1 = _mm512_mask_blend_epi64(0xcc, v1, v2_perm);
  *out2 = _mm512_mask_blend_epi64(0xcc, v1_perm, v2);
}

// Given input: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
// Returns
// *out1 =  _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
// *out2 =  _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
inline void LoadInvInterleavedT2(const uint64_t* arg, __m512i* out1,
                                 __m512i* out2) {
  const __m512i* arg_512 = reinterpret_cast<const __m512i*>(arg);

  __m512i v1 = _mm512_loadu_si512(arg_512++);
  __m512i v2 = _mm512_loadu_si512(arg_512);

  const __m512i v1_perm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);

  __m512i v1_perm = _mm512_permutexvar_epi64(v1_perm_idx, v1);
  __m512i v2_perm = _mm512_permutexvar_epi64(v1_perm_idx, v2);

  *out1 = _mm512_mask_blend_epi64(0xaa, v1, v2_perm);
  *out2 = _mm512_mask_blend_epi64(0xaa, v1_perm, v2);
}

// Returns
// *out1 =  _mm512_set_epi64(arg[11], arg[10], arg[9], arg[8],
//                           arg[3], arg[2], arg[1], arg[0]);
// *out2 =  _mm512_set_epi64(arg[15], arg[14], arg[13], arg[12],
//                           arg[7], arg[6], arg[5], arg[4]);
inline void LoadFwdInterleavedT4(const uint64_t* arg, __m512i* out1,
                                 __m512i* out2) {
  const __m512i* arg_512 = reinterpret_cast<const __m512i*>(arg);

  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  __m512i v_7to0 = _mm512_loadu_si512(arg_512++);
  __m512i v_15to8 = _mm512_loadu_si512(arg_512);
  __m512i perm_hi = _mm512_permutexvar_epi64(vperm2_idx, v_15to8);
  *out1 = _mm512_mask_blend_epi64(0x0f, perm_hi, v_7to0);
  *out2 = _mm512_mask_blend_epi64(0xf0, perm_hi, v_7to0);
  *out2 = _mm512_permutexvar_epi64(vperm2_idx, *out2);
}

inline void LoadInvInterleavedT4(const uint64_t* arg, __m512i* out1,
                                 __m512i* out2) {
  const __m512i* arg_512 = reinterpret_cast<const __m512i*>(arg);

  // 0, 1, 2, 3, 4, 5, 6, 7
  __m512i v1 = _mm512_loadu_si512(arg_512++);
  // 8, 9, 10, 11, 12, 13, 14, 15
  __m512i v2 = _mm512_loadu_si512(arg_512);
  const __m512i perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);

  // 1, 0, 3, 2, 5, 4, 7, 6
  __m512i v1_perm = _mm512_permutexvar_epi64(perm_idx, v1);
  // 9, 8, 11, 10, 13, 12, 15, 14
  __m512i v2_perm = _mm512_permutexvar_epi64(perm_idx, v2);

  *out1 = _mm512_mask_blend_epi64(0xcc, v1, v2_perm);
  *out2 = _mm512_mask_blend_epi64(0xcc, v1_perm, v2);
}

// Given inputs
// @param arg1 = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
// @param arg2 = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
// Writes out = {8,  0, 9,  1, 10, 2, 11, 3,
//               12, 4, 13, 5, 14, 6, 15, 7}
inline void WriteFwdInterleavedT1(__m512i arg1, __m512i arg2, __m512i* out) {
  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i v_X_out_idx = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
  const __m512i v_Y_out_idx = _mm512_set_epi64(3, 7, 2, 6, 1, 5, 0, 4);

  // v_Y => (4, 5, 6, 7, 0, 1, 2, 3)
  arg2 = _mm512_permutexvar_epi64(vperm2_idx, arg2);
  // 4, 5, 6, 7, 12, 13, 14, 15
  __m512i perm_lo = _mm512_mask_blend_epi64(0x0f, arg1, arg2);

  // 8, 9, 10, 11, 0, 1, 2, 3
  __m512i perm_hi = _mm512_mask_blend_epi64(0xf0, arg1, arg2);

  arg1 = _mm512_permutexvar_epi64(v_X_out_idx, perm_hi);
  arg2 = _mm512_permutexvar_epi64(v_Y_out_idx, perm_lo);

  _mm512_storeu_si512(out++, arg1);
  _mm512_storeu_si512(out, arg2);
}

// Given inputs
// @param arg1 = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
// @param arg2 = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
// Writes out = {8,  9,  10, 11, 0, 1, 2, 3,
//               12, 13, 14, 15, 4, 5, 6, 7}
inline void WriteInvInterleavedT4(__m512i arg1, __m512i arg2, __m512i* out) {
  __m256i x0 = _mm512_extracti64x4_epi64(arg1, 0);
  __m256i x1 = _mm512_extracti64x4_epi64(arg1, 1);
  __m256i y0 = _mm512_extracti64x4_epi64(arg2, 0);
  __m256i y1 = _mm512_extracti64x4_epi64(arg2, 1);
  __m256i* out_256 = reinterpret_cast<__m256i*>(out);
  _mm256_storeu_si256(out_256++, x0);
  _mm256_storeu_si256(out_256++, y0);
  _mm256_storeu_si256(out_256++, x1);
  _mm256_storeu_si256(out_256++, y1);
}

// Returns _mm512_set_epi64(arg[3], arg[3], arg[2], arg[2],
//                          arg[1], arg[1], arg[0], arg[0]);
inline __m512i LoadWOpT2(const void* arg) {
  const __m512i vperm_w_idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);

  __m256i v_W_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arg));
  __m512i v_W = _mm512_broadcast_i64x4(v_W_256);
  v_W = _mm512_permutexvar_epi64(vperm_w_idx, v_W);

  return v_W;
}

// Returns _mm512_set_epi64(arg[1], arg[1], arg[1], arg[1],
//                          arg[0], arg[0], arg[0], arg[0]);
inline __m512i LoadWOpT4(const void* arg) {
  const __m512i vperm_w_idx = _mm512_set_epi64(1, 1, 1, 1, 0, 0, 0, 0);

  __m128i v_W_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(arg));
  __m512i v_W = _mm512_broadcast_i64x2(v_W_128);
  v_W = _mm512_permutexvar_epi64(vperm_w_idx, v_W);

  return v_W;
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
