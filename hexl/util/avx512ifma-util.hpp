// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>

#include <vector>

#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/defines.hpp"
#include "hexl/util/util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {
#ifdef HEXL_HAS_AVX512IFMA
template <>
inline __m512i _mm512_hexl_mulhi_epi<52>(__m512i x, __m512i y) {
  __m512i zero = _mm512_set1_epi64(0);
  return _mm512_madd52hi_epu64(zero, x, y);
}

template <>
inline __m512i _mm512_hexl_mulhi_approx_epi<52>(__m512i x, __m512i y) {
  __m512i zero = _mm512_set1_epi64(0);
  return _mm512_madd52hi_epu64(zero, x, y);
}

template <>
inline __m512i _mm512_hexl_mullo_epi<52>(__m512i x, __m512i y) {
  __m512i zero = _mm512_set1_epi64(0);
  return _mm512_madd52lo_epu64(zero, x, y);
}

template <>
inline __m512i _mm512_hexl_mullo_add_lo_epi<52>(__m512i x, __m512i y,
                                                __m512i z) {
  __m512i result = _mm512_madd52lo_epu64(x, y, z);

  // Clear high 12 bits from result
  const __m512i two_pow52_min1 = _mm512_set1_epi64((1ULL << 52) - 1);
  result = _mm512_and_epi64(result, two_pow52_min1);
  return result;
}
#endif

}  // namespace hexl
}  // namespace intel
