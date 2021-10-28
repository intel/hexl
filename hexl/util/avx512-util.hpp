// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>

#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/defines.hpp"
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

// Returns lower NumBits bits from a 64-bit value
template <int NumBits>
inline __m512i ClearTopBits64(__m512i x) {
  const __m512i low52b_mask = _mm512_set1_epi64((1ULL << NumBits) - 1);
  return _mm512_and_epi64(x, low52b_mask);
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
  HEXL_UNUSED(x);
  HEXL_UNUSED(y);
  return x;
}

template <>
inline __m512i _mm512_hexl_mulhi_epi<64>(__m512i x, __m512i y) {
  // https://stackoverflow.com/questions/28807341/simd-signed-with-unsigned-multiplication-for-64-bit-64-bit-to-128-bit
  __m512i lo_mask = _mm512_set1_epi64(0x00000000ffffffff);
  // Shuffle high bits with low bits in each 64-bit integer =>
  // x0_lo, x0_hi, x1_lo, x1_hi, x2_lo, x2_hi, ...
  __m512i x_hi = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
  // y0_lo, y0_hi, y1_lo, y1_hi, y2_lo, y2_hi, ...
  __m512i y_hi = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
  __m512i z_lo_lo = _mm512_mul_epu32(x, y);        // x_lo * y_lo
  __m512i z_lo_hi = _mm512_mul_epu32(x, y_hi);     // x_lo * y_hi
  __m512i z_hi_lo = _mm512_mul_epu32(x_hi, y);     // x_hi * y_lo
  __m512i z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);  // x_hi * y_hi

  //                   x_hi | x_lo
  // x                 y_hi | y_lo
  // ------------------------------
  //                  [x_lo * y_lo]    // z_lo_lo
  // +           [z_lo * y_hi]         // z_lo_hi
  // +           [x_hi * y_lo]         // z_hi_lo
  // +    [x_hi * y_hi]                // z_hi_hi
  //     ^-----------^ <-- only bits needed
  //  sum_|  hi | mid | lo  |

  // Low bits of z_lo_lo are not needed
  __m512i z_lo_lo_shift = _mm512_srli_epi64(z_lo_lo, 32);

  //                   [x_lo  *  y_lo] // z_lo_lo
  //          + [z_lo  *  y_hi]        // z_lo_hi
  //          ------------------------
  //            |    sum_tmp   |
  //            |sum_mid|sum_lo|
  __m512i sum_tmp = _mm512_add_epi64(z_lo_hi, z_lo_lo_shift);
  __m512i sum_lo = _mm512_and_si512(sum_tmp, lo_mask);
  __m512i sum_mid = _mm512_srli_epi64(sum_tmp, 32);
  //            |       |sum_lo|
  //          + [x_hi   *  y_lo]       // z_hi_lo
  //          ------------------
  //            [   sum_mid2   ]
  __m512i sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
  __m512i sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
  __m512i sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
  return _mm512_add_epi64(sum_hi, sum_mid2_hi);
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
inline __m512i _mm512_hexl_mulhi_approx_epi(__m512i x, __m512i y);

// Dummy implementation to avoid template substitution errors
template <>
inline __m512i _mm512_hexl_mulhi_approx_epi<32>(__m512i x, __m512i y) {
  HEXL_CHECK(false, "Unimplemented");
  HEXL_UNUSED(x);
  HEXL_UNUSED(y);
  return x;
}

template <>
inline __m512i _mm512_hexl_mulhi_approx_epi<64>(__m512i x, __m512i y) {
  // https://stackoverflow.com/questions/28807341/simd-signed-with-unsigned-multiplication-for-64-bit-64-bit-to-128-bit
  __m512i lo_mask = _mm512_set1_epi64(0x00000000ffffffff);
  // Shuffle high bits with low bits in each 64-bit integer =>
  // x0_lo, x0_hi, x1_lo, x1_hi, x2_lo, x2_hi, ...
  __m512i x_hi = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
  // y0_lo, y0_hi, y1_lo, y1_hi, y2_lo, y2_hi, ...
  __m512i y_hi = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
  __m512i z_lo_hi = _mm512_mul_epu32(x, y_hi);     // x_lo * y_hi
  __m512i z_hi_lo = _mm512_mul_epu32(x_hi, y);     // x_hi * y_lo
  __m512i z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);  // x_hi * y_hi

  //                   x_hi | x_lo
  // x                 y_hi | y_lo
  // ------------------------------
  //                  [x_lo * y_lo]    // unused, resulting in approximation
  // +           [z_lo * y_hi]         // z_lo_hi
  // +           [x_hi * y_lo]         // z_hi_lo
  // +    [x_hi * y_hi]                // z_hi_hi
  //     ^-----------^ <-- only bits needed
  //  sum_|  hi | mid | lo  |

  __m512i sum_lo = _mm512_and_si512(z_lo_hi, lo_mask);
  __m512i sum_mid = _mm512_srli_epi64(z_lo_hi, 32);
  //            |       |sum_lo|
  //          + [x_hi   *  y_lo]       // z_hi_lo
  //          ------------------
  //            [   sum_mid2   ]
  __m512i sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
  __m512i sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
  __m512i sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
  return _mm512_add_epi64(sum_hi, sum_mid2_hi);
}

#ifdef HEXL_HAS_AVX512IFMA
template <>
inline __m512i _mm512_hexl_mulhi_approx_epi<52>(__m512i x, __m512i y) {
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
  HEXL_UNUSED(x);
  HEXL_UNUSED(y);
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
  result = ClearTopBits64<52>(result);
  return result;
}
#endif

// Dummy implementation to avoid template substitution errors
template <>
inline __m512i _mm512_hexl_mullo_add_lo_epi<32>(__m512i x, __m512i y,
                                                __m512i z) {
  HEXL_CHECK(false, "Unimplemented");
  HEXL_UNUSED(x);
  HEXL_UNUSED(y);
  HEXL_UNUSED(z);
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
  HEXL_CHECK(false, "Invalid InputModFactor");
  return x;  // Return dummy value
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

// Returns Montgomery form of ab mod q, computed via the REDC algorithm,
// also known as Montgomery reduction.
// Template: r with R = 2^r
// Inputs: q such that gcd(R, q) = 1. R > q.
//         v_inv_mod in [0, R − 1] such that q*v_inv_mod ≡ −1 mod R,
//         T = ab in the range [0, Rq − 1].
// T_hi and T_lo for BitShift = 64 should be given in 63 bits.
// Output: Integer S in the range [0, q − 1] such that S ≡ TR^−1 mod q
template <int BitShift, int r>
inline __m512i _mm512_hexl_montgomery_reduce(__m512i T_hi, __m512i T_lo,
                                             __m512i q, __m512i v_inv_mod,
                                             __m512i v_rs_or_msk) {
  HEXL_CHECK(BitShift == 52 || BitShift == 64,
             "Invalid bitshift " << BitShift << "; need 52 or 64");

#ifdef HEXL_HAS_AVX512IFMA
  if (BitShift == 52) {
    // Operation:
    // m ← ((T mod R)N′) mod R | m ← ((T & mod_R_mask)*v_inv_mod) & mod_R_mask
    __m512i m = ClearTopBits64<r>(T_lo);
    m = _mm512_hexl_mullo_epi<BitShift>(m, v_inv_mod);
    m = ClearTopBits64<r>(m);

    // Operation: t ← (T + mN) / R = (T + m*q) >> r
    // Hi part
    __m512i t_hi = _mm512_madd52hi_epu64(T_hi, m, q);
    // Low part
    __m512i t = _mm512_madd52lo_epu64(T_lo, m, q);
    t = _mm512_srli_epi64(t, r);
    // Join parts
    t = _mm512_madd52lo_epu64(t, t_hi, v_rs_or_msk);

    // If this function exists for 52 bits we could save 1 cycle
    // t = _mm512_shrdi_epi64 (t_hi, t, r)

    // Operation: t ≥ q? return (t - q) : return t
    return _mm512_hexl_small_mod_epu64<2>(t, q);
  }
#endif

  HEXL_CHECK(BitShift == 64, "Invalid bitshift " << BitShift << "; need 64");

  // Operation:
  // m ← ((T mod R)N′) mod R | m ← ((T & mod_R_mask)*v_inv_mod) & mod_R_mask
  __m512i m = ClearTopBits64<r>(T_lo);
  m = _mm512_hexl_mullo_epi<BitShift>(m, v_inv_mod);
  m = ClearTopBits64<r>(m);

  __m512i mq_hi = _mm512_hexl_mulhi_epi<BitShift>(m, q);
  __m512i mq_lo = _mm512_hexl_mullo_epi<BitShift>(m, q);

  // to 63 bits
  mq_hi = _mm512_slli_epi64(mq_hi, 1);
  __m512i tmp = _mm512_srli_epi64(mq_lo, 63);
  mq_hi = _mm512_add_epi64(mq_hi, tmp);
  mq_lo = _mm512_and_epi64(mq_lo, v_rs_or_msk);

  __m512i t_hi = _mm512_add_epi64(T_hi, mq_hi);
  t_hi = _mm512_slli_epi64(t_hi, 63 - r);
  __m512i t = _mm512_add_epi64(T_lo, mq_lo);
  t = _mm512_srli_epi64(t, r);

  // Join parts
  t = _mm512_add_epi64(t_hi, t);

  return _mm512_hexl_small_mod_epu64<2>(t, q);
}

// Returns x mod q, computed via Barrett reduction
// @param q_barr floor(2^BitShift / q)
template <int BitShift = 64, int OutputModFactor = 1>
inline __m512i _mm512_hexl_barrett_reduce64(__m512i x, __m512i q,
                                            __m512i q_barr_64,
                                            __m512i q_barr_52,
                                            uint64_t prod_right_shift,
                                            __m512i v_neg_mod) {
  HEXL_UNUSED(q_barr_52);
  HEXL_UNUSED(prod_right_shift);
  HEXL_UNUSED(v_neg_mod);
  HEXL_CHECK(BitShift == 52 || BitShift == 64,
             "Invalid bitshift " << BitShift << "; need 52 or 64");

#ifdef HEXL_HAS_AVX512IFMA
  if (BitShift == 52) {
    __m512i two_pow_fiftytwo = _mm512_set1_epi64(2251799813685248);
    __mmask8 mask =
        _mm512_hexl_cmp_epu64_mask(x, two_pow_fiftytwo, CMPINT::NLT);
    if (mask != 0) {
      // values above 2^52
      __m512i x_hi = _mm512_srli_epi64(x, static_cast<unsigned int>(52ULL));
      __m512i x_lo = ClearTopBits64<52>(x);

      // c1 = floor(U / 2^{n + beta})
      __m512i c1_lo =
          _mm512_srli_epi64(x_lo, static_cast<unsigned int>(prod_right_shift));
      __m512i c1_hi = _mm512_slli_epi64(
          x_hi, static_cast<unsigned int>(52ULL - (prod_right_shift)));
      __m512i c1 = _mm512_or_epi64(c1_lo, c1_hi);

      // alpha - beta == 52, so we only need high 52 bits
      __m512i q_hat = _mm512_hexl_mulhi_epi<52>(c1, q_barr_64);

      // Z = prod_lo - (p * q_hat)_lo
      x = _mm512_hexl_mullo_add_lo_epi<52>(x_lo, q_hat, v_neg_mod);
    } else {
      __m512i rnd1_hi = _mm512_hexl_mulhi_epi<52>(x, q_barr_52);
      __m512i tmp1_times_mod = _mm512_hexl_mullo_epi<52>(rnd1_hi, q);
      x = _mm512_sub_epi64(x, tmp1_times_mod);
    }
  }
#endif
  if (BitShift == 64) {
    __m512i rnd1_hi = _mm512_hexl_mulhi_epi<64>(x, q_barr_64);
    __m512i tmp1_times_mod = _mm512_hexl_mullo_epi<64>(rnd1_hi, q);
    x = _mm512_sub_epi64(x, tmp1_times_mod);
  }

  // Correction
  if (OutputModFactor == 1) {
    x = _mm512_hexl_small_mod_epu64<2>(x, q);
  }
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
#else
  return _mm512_hexl_shrdi_epi64(x, y, BitShift);
#endif
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
