// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>

#include <vector>

#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

/// @brief Returns the unsigned 64-bit integer values in x as a vector
inline std::vector<uint64_t> ExtractValues(__m512i x) {
  __m256i x0 = _mm512_extracti64x4_epi64(x, 0);
  __m256i x1 = _mm512_extracti64x4_epi64(x, 1);

  std::vector<uint64_t> xs{static_cast<uint64_t>(_mm256_extract_epi64(x0, 0)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x0, 1)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x0, 2)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x0, 3)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x1, 0)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x1, 1)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x1, 2)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x1, 3))};

  return xs;
}

/// @brief Returns the signed 64-bit integer values in x as a vector
inline std::vector<int64_t> ExtractIntValues(__m512i x) {
  __m256i x0 = _mm512_extracti64x4_epi64(x, 0);
  __m256i x1 = _mm512_extracti64x4_epi64(x, 1);

  std::vector<int64_t> xs{static_cast<int64_t>(_mm256_extract_epi64(x0, 0)),
                          static_cast<int64_t>(_mm256_extract_epi64(x0, 1)),
                          static_cast<int64_t>(_mm256_extract_epi64(x0, 2)),
                          static_cast<int64_t>(_mm256_extract_epi64(x0, 3)),
                          static_cast<int64_t>(_mm256_extract_epi64(x1, 0)),
                          static_cast<int64_t>(_mm256_extract_epi64(x1, 1)),
                          static_cast<int64_t>(_mm256_extract_epi64(x1, 2)),
                          static_cast<int64_t>(_mm256_extract_epi64(x1, 3))};

  return xs;
}

// Returns the 64-bit floating-point values in x as a vector
inline std::vector<double> ExtractValues(__m512d x) {
  std::vector<double> ret(8, 0);
  double* x_data = reinterpret_cast<double*>(&x);
  for (size_t i = 0; i < 8; ++i) {
    ret[i] = x_data[i];
  }
  return ret;
}

// Multiply packed unsigned BitShift-bit integers in each 64-bit element of x
// and y to form a 2*BitShift-bit intermediate result.
// Returns the high BitShift-bit unsigned integer from the intermediate result
template <int BitShift>
inline __m512i _mm512_hexl_mulhi_epi(__m512i x, __m512i y);

// Dummy implementation to avoid template substitution errors
template <>
inline __m512i _mm512_hexl_mulhi_epi<32>(__m512i x, __m512i y) {
  HEXL_CHECK(false, "Unimplemented");
  (void)x;  // Avoid unused variable warning
  (void)y;  // Avoid unused variable warning
  return x;
}

template <>
inline __m512i _mm512_hexl_mulhi_epi<64>(__m512i x, __m512i y) {
  // https://stackoverflow.com/questions/28807341/simd-signed-with-unsigned-multiplication-for-64-bit-64-bit-to-128-bit
  __m512i lomask = _mm512_set1_epi64(0x00000000ffffffff);
  __m512i xh =
      _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);  // x0l, x0h, x1l, x1h
  __m512i yh =
      _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);  // y0l, y0h, y1l, y1h
  __m512i w0 = _mm512_mul_epu32(x, y);               // x0l*y0l, x1l*y1l
  __m512i w1 = _mm512_mul_epu32(x, yh);              // x0l*y0h, x1l*y1h
  __m512i w2 = _mm512_mul_epu32(xh, y);              // x0h*y0l, x1h*y0l
  __m512i w3 = _mm512_mul_epu32(xh, yh);             // x0h*y0h, x1h*y1h
  __m512i w0h = _mm512_srli_epi64(w0, 32);
  __m512i s1 = _mm512_add_epi64(w1, w0h);
  __m512i s1l = _mm512_and_si512(s1, lomask);
  __m512i s1h = _mm512_srli_epi64(s1, 32);
  __m512i s2 = _mm512_add_epi64(w2, s1l);
  __m512i s2h = _mm512_srli_epi64(s2, 32);
  __m512i hi1 = _mm512_add_epi64(w3, s1h);
  return _mm512_add_epi64(hi1, s2h);
}

#ifdef HEXL_HAS_AVX512IFMA
template <>
inline __m512i _mm512_hexl_mulhi_epi<52>(__m512i x, __m512i y) {
  __m512i zero = _mm512_set1_epi64(0);
  return _mm512_madd52hi_epu64(zero, x, y);
}
#endif

// Multiply packed unsigned BitShift-bit integers in each 64-bit element of x
// and y to form a 2*BitShift-bit intermediate result.
// Returns the high BitShift-bit unsigned integer from the intermediate result,
// with approximation error at most 1
template <int BitShift>
inline __m512i _mm512_hexl_mulhi_epi_approx(__m512i x, __m512i y);

// Dummy implementation to avoid template substitution errors
template <>
inline __m512i _mm512_hexl_mulhi_epi_approx<32>(__m512i x, __m512i y) {
  HEXL_CHECK(false, "Unimplemented");
  (void)x;  // Avoid unused variable warning
  (void)y;  // Avoid unused variable warning
  return x;
}

template <>
inline __m512i _mm512_hexl_mulhi_epi_approx<64>(__m512i x, __m512i y) {
  // https://stackoverflow.com/questions/28807341/simd-signed-with-unsigned-multiplication-for-64-bit-64-bit-to-128-bit
  __m512i lomask = _mm512_set1_epi64(0x00000000ffffffff);
  __m512i xh =
      _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);  // x0l, x0h, x1l, x1h
  __m512i yh =
      _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);  // y0l, y0h, y1l, y1h
  __m512i w0 = _mm512_mul_epu32(x, y);               // x0l*y0l, x1l*y1l
  __m512i w1 = _mm512_mul_epu32(x, yh);              // x0l*y0h, x1l*y1h
  __m512i w2 = _mm512_mul_epu32(xh, y);              // x0h*y0l, x1h*y0l
  __m512i w3 = _mm512_mul_epu32(xh, yh);             // x0h*y0h, x1h*y1h
  __m512i w0h = _mm512_srli_epi64(w0, 32);
  __m512i s1 = _mm512_add_epi64(w1, w0h);
  __m512i s1l = _mm512_and_si512(s1, lomask);
  __m512i s1h = _mm512_srli_epi64(s1, 32);
  __m512i s2 = _mm512_add_epi64(w2, s1l);
  __m512i s2h = _mm512_srli_epi64(s2, 32);
  __m512i hi1 = _mm512_add_epi64(w3, s1h);
  return _mm512_add_epi64(hi1, s2h);
}

#ifdef HEXL_HAS_AVX512IFMA
template <>
inline __m512i _mm512_hexl_mulhi_epi_approx<52>(__m512i x, __m512i y) {
  __m512i zero = _mm512_set1_epi64(0);
  return _mm512_madd52hi_epu64(zero, x, y);
}
#endif

// Multiply packed unsigned BitShift-bit integers in each 64-bit element of x
// and y to form a 2*BitShift-bit intermediate result.
// Returns the low BitShift-bit unsigned integer from the intermediate result
template <int BitShift>
inline __m512i _mm512_hexl_mullo_epi(__m512i x, __m512i y);

// Dummy implementation to avoid template substitution errors
template <>
inline __m512i _mm512_hexl_mullo_epi<32>(__m512i x, __m512i y) {
  HEXL_CHECK(false, "Unimplemented");
  (void)x;  // Avoid unused variable warning
  (void)y;  // Avoid unused variable warning
  return x;
}

template <>
inline __m512i _mm512_hexl_mullo_epi<64>(__m512i x, __m512i y) {
  return _mm512_mullo_epi64(x, y);
}

#ifdef HEXL_HAS_AVX512IFMA
template <>
inline __m512i _mm512_hexl_mullo_epi<52>(__m512i x, __m512i y) {
  __m512i zero = _mm512_set1_epi64(0);
  return _mm512_madd52lo_epu64(zero, x, y);
}
#endif

// Multiply packed unsigned BitShift-bit integers in each 64-bit element of y
// and z to form a 2*BitShift-bit intermediate result. The low BitShift bits of
// the result are added to x, then the low BitShift bits of the result are
// returned.
template <int BitShift>
inline __m512i _mm512_hexl_mullo_add_lo_epi(__m512i x, __m512i y, __m512i z);

#ifdef HEXL_HAS_AVX512IFMA
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

// Dummy implementation to avoid template substitution errors
template <>
inline __m512i _mm512_hexl_mullo_add_lo_epi<32>(__m512i x, __m512i y,
                                                __m512i z) {
  HEXL_CHECK(false, "Unimplemented");
  (void)x;  // Avoid unused variable warning
  (void)y;  // Avoid unused variable warning
  (void)z;  // Avoid unused variable warning
  return x;
}

template <>
inline __m512i _mm512_hexl_mullo_add_lo_epi<64>(__m512i x, __m512i y,
                                                __m512i z) {
  __m512i prod = _mm512_mullo_epi64(y, z);
  return _mm512_add_epi64(x, prod);
}

// Returns x mod q across each 64-bit integer SIMD lanes
// Assumes x < InputModFactor * q in all lanes
template <int InputModFactor = 2>
inline __m512i _mm512_hexl_small_mod_epu64(__m512i x, __m512i q,
                                           __m512i* q_times_2 = nullptr,
                                           __m512i* q_times_4 = nullptr) {
  HEXL_CHECK(InputModFactor == 1 || InputModFactor == 2 ||
                 InputModFactor == 4 || InputModFactor == 8,
             "InputModFactor must be 1, 2, 4, or 8");
  if (InputModFactor == 1) {
    return x;
  }
  if (InputModFactor == 2) {
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, q));
  }
  if (InputModFactor == 4) {
    HEXL_CHECK(q_times_2 != nullptr, "q_times_2 must not be nullptr");
    x = _mm512_min_epu64(x, _mm512_sub_epi64(x, *q_times_2));
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, q));
  }
  if (InputModFactor == 8) {
    HEXL_CHECK(q_times_2 != nullptr, "q_times_2 must not be nullptr");
    HEXL_CHECK(q_times_4 != nullptr, "q_times_4 must not be nullptr");
    x = _mm512_min_epu64(x, _mm512_sub_epi64(x, *q_times_4));
    x = _mm512_min_epu64(x, _mm512_sub_epi64(x, *q_times_2));
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, q));
  }
}

// Returns (x + y) mod q; assumes 0 < x, y < q
inline __m512i _mm512_hexl_small_add_mod_epi64(__m512i x, __m512i y,
                                               __m512i q) {
  HEXL_CHECK_BOUNDS(ExtractValues(x).data(), 8, ExtractValues(q)[0],
                    "x exceeds bound " << ExtractValues(q)[0]);
  HEXL_CHECK_BOUNDS(ExtractValues(y).data(), 8, ExtractValues(q)[0],
                    "y exceeds bound " << ExtractValues(q)[0]);
  return _mm512_hexl_small_mod_epu64(_mm512_add_epi64(x, y), q);

  // Alternate implementation:
  // x += y - q;
  // if (x < 0) x+= q
  // return x
  // __m512i v_diff = _mm512_sub_epi64(y, q);
  // x = _mm512_add_epi64(x, v_diff);
  // __mmask8 sign_bits = _mm512_movepi64_mask(x);
  // return _mm512_mask_add_epi64(x, sign_bits, x, q);
}

// Returns (x - y) mod q; assumes 0 < x, y < q

inline __m512i _mm512_hexl_small_sub_mod_epi64(__m512i x, __m512i y,
                                               __m512i q) {
  HEXL_CHECK_BOUNDS(ExtractValues(x).data(), 8, ExtractValues(q)[0],
                    "x exceeds bound " << ExtractValues(q)[0]);
  HEXL_CHECK_BOUNDS(ExtractValues(y).data(), 8, ExtractValues(q)[0],
                    "y exceeds bound " << ExtractValues(q)[0]);

  // diff = x - y;
  // return (diff < 0) ? (diff + q) : diff
  __m512i v_diff = _mm512_sub_epi64(x, y);
  __mmask8 sign_bits = _mm512_movepi64_mask(v_diff);
  return _mm512_mask_add_epi64(v_diff, sign_bits, v_diff, q);
}

inline __mmask8 _mm512_hexl_cmp_epu64_mask(__m512i a, __m512i b, CMPINT cmp) {
  switch (cmp) {
    case CMPINT::EQ:
      return _mm512_cmp_epu64_mask(a, b, static_cast<int>(CMPINT::EQ));
    case CMPINT::LT:
      return _mm512_cmp_epu64_mask(a, b, static_cast<int>(CMPINT::LT));
    case CMPINT::LE:
      return _mm512_cmp_epu64_mask(a, b, static_cast<int>(CMPINT::LE));
    case CMPINT::FALSE:
      return _mm512_cmp_epu64_mask(a, b, static_cast<int>(CMPINT::FALSE));
    case CMPINT::NE:
      return _mm512_cmp_epu64_mask(a, b, static_cast<int>(CMPINT::NE));
    case CMPINT::NLT:
      return _mm512_cmp_epu64_mask(a, b, static_cast<int>(CMPINT::NLT));
    case CMPINT::NLE:
      return _mm512_cmp_epu64_mask(a, b, static_cast<int>(CMPINT::NLE));
    case CMPINT::TRUE:
      return _mm512_cmp_epu64_mask(a, b, static_cast<int>(CMPINT::TRUE));
  }
  __mmask8 dummy = 0;  // Avoid end of non-void function warning
  return dummy;
}

// Returns c[i] = a[i] CMP b[i] ? match_value : 0
inline __m512i _mm512_hexl_cmp_epi64(__m512i a, __m512i b, CMPINT cmp,
                                     uint64_t match_value) {
  __mmask8 mask = _mm512_hexl_cmp_epu64_mask(a, b, cmp);
  return _mm512_maskz_broadcastq_epi64(
      mask, _mm_set1_epi64x(static_cast<int64_t>(match_value)));
}

// Returns c[i] = a[i] >= b[i] ? match_value : 0
inline __m512i _mm512_hexl_cmpge_epu64(__m512i a, __m512i b,
                                       uint64_t match_value) {
  return _mm512_hexl_cmp_epi64(a, b, CMPINT::NLT, match_value);
}

// Returns c[i] = a[i] < b[i] ? match_value : 0
inline __m512i _mm512_hexl_cmplt_epu64(__m512i a, __m512i b,
                                       uint64_t match_value) {
  return _mm512_hexl_cmp_epi64(a, b, CMPINT::LT, match_value);
}

// Returns c[i] = a[i] <= b[i] ? match_value : 0
inline __m512i _mm512_hexl_cmple_epu64(__m512i a, __m512i b,
                                       uint64_t match_value) {
  return _mm512_hexl_cmp_epi64(a, b, CMPINT::LE, match_value);
}

// Returns x mod q, computed via Barrett reduction
// @param q_barr floor(2^BitShift / q)
template <int BitShift = 64>
inline __m512i _mm512_hexl_barrett_reduce64(__m512i x, __m512i q,
                                            __m512i q_barr) {
  __m512i rnd1_hi = _mm512_hexl_mulhi_epi<BitShift>(x, q_barr);

  // Barrett subtraction
  // tmp[0] = input - tmp[1] * q;
  __m512i tmp1_times_mod = _mm512_hexl_mullo_epi<64>(rnd1_hi, q);
  x = _mm512_sub_epi64(x, tmp1_times_mod);
  // Correction
  x = _mm512_hexl_small_mod_epu64(x, q);
  return x;
}

// Concatenate packed 64-bit integers in x and y, producing an intermediate
// 128-bit result. Shift the result right by bit_shift bits, and return the
// lower 64 bits. The bit_shift is a run-time argument, rather than a
// compile-time template parameter, so we can't use _mm512_shrdi_epi64
inline __m512i _mm512_hexl_shrdi_epi64(__m512i x, __m512i y,
                                       unsigned int bit_shift) {
  __m512i c_lo = _mm512_srli_epi64(x, bit_shift);
  __m512i c_hi = _mm512_slli_epi64(y, 64 - bit_shift);
  return _mm512_add_epi64(c_lo, c_hi);
}

// Concatenate packed 64-bit integers in x and y, producing an intermediate
// 128-bit result. Shift the result right by BitShift bits, and return the lower
// 64 bits.
template <int BitShift>
inline __m512i _mm512_hexl_shrdi_epi64(__m512i x, __m512i y) {
#ifdef HEXL_HAS_AVX512VBMI2
  return _mm512_shrdi_epi64(x, y, BitShift);
#endif
  return _mm512_hexl_shrdi_epi64(x, y, BitShift);
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
