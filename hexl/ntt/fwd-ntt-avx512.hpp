// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>

#include <functional>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "ntt/ntt-avx512-util.hpp"
#include "ntt/ntt-internal.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

/// @brief The Harvey butterfly: assume \p X, \p Y in [0, 4q), and return X', Y'
/// in [0, 4q) such that X', Y' = X + WY, X - WY (mod q).
/// @param[in,out] X Input representing 8 64-bit signed integers in SIMD form
/// @param[in,out] Y Input representing 8 64-bit signed integers in SIMD form
/// @param[in] W_op Input representing 8 64-bit signed integers in SIMD form
/// @param[in] W_precon Preconditioned \p W_op for BitShift-bit Barrett
/// reduction
/// @param[in] neg_modulus Negative modulus, i.e. (-q) represented as 8 64-bit
/// signed integers in SIMD form
/// @param[in] twice_modulus Twice the modulus, i.e. 2*q represented as 8 64-bit
/// signed integers in SIMD form
/// @param InputLessThanMod If true, assumes \p X, \p Y < \p qp. Otherwise,
/// assumes \p X, \p Y < 4*\p q
/// @details See Algorithm 4 of https://arxiv.org/pdf/1205.2926.pdf
template <int BitShift, bool InputLessThanMod>
inline void FwdButterfly(__m512i* X, __m512i* Y, __m512i W_op, __m512i W_precon,
                         __m512i neg_modulus, __m512i twice_modulus) {
  if (!InputLessThanMod) {
    *X = _mm512_hexl_small_mod_epu64(*X, twice_modulus);
  }
  __m512i Q = _mm512_hexl_mulhi_epi<BitShift>(W_precon, *Y);
  __m512i W_Y = _mm512_hexl_mullo_epi<BitShift>(W_op, *Y);
  __m512i T = _mm512_hexl_mullo_add_epi<BitShift>(W_Y, Q, neg_modulus);

  // Discard high 12 bits if BitShift == 52; deals with case when
  // W*Y < Q*p in the low BitShift bits.
  if (BitShift == 52) {
    T = _mm512_and_epi64(T, _mm512_set1_epi64((1ULL << 52) - 1));
  }
  __m512i twice_mod_minus_T = _mm512_sub_epi64(twice_modulus, T);
  *Y = _mm512_add_epi64(*X, twice_mod_minus_T);
  *X = _mm512_add_epi64(*X, T);
}

template <int BitShift>
void FwdT1(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  const __m512i* v_W_op_pt = reinterpret_cast<const __m512i*>(W_op);
  const __m512i* v_W_precon_pt = reinterpret_cast<const __m512i*>(W_precon);
  size_t j1 = 0;

  // 8 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_8
  for (size_t i = m / 8; i > 0; --i) {
    uint64_t* X = operand + j1;
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadFwdInterleavedT1(X, &v_X, &v_Y);
    __m512i v_W_op = _mm512_loadu_si512(v_W_op_pt++);
    __m512i v_W_precon = _mm512_loadu_si512(v_W_precon_pt++);

    FwdButterfly<BitShift, false>(&v_X, &v_Y, v_W_op, v_W_precon, v_neg_modulus,
                                  v_twice_mod);
    WriteFwdInterleavedT1(v_X, v_Y, v_X_pt);

    j1 += 16;
  }
}

template <int BitShift>
void FwdT2(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  size_t j1 = 0;
  // 4 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = m / 4; i > 0; --i) {
    uint64_t* X = operand + j1;
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadFwdInterleavedT2(X, &v_X, &v_Y);

    __m512i v_W_op = LoadWOpT2(static_cast<const void*>(W_op));
    __m512i v_W_precon = LoadWOpT2(static_cast<const void*>(W_precon));

    FwdButterfly<BitShift, false>(&v_X, &v_Y, v_W_op, v_W_precon, v_neg_modulus,
                                  v_twice_mod);

    _mm512_storeu_si512(v_X_pt++, v_X);
    _mm512_storeu_si512(v_X_pt, v_Y);

    W_op += 4;
    W_precon += 4;

    j1 += 16;
  }
}

template <int BitShift>
void FwdT4(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  size_t j1 = 0;

  // 2 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = m / 2; i > 0; --i) {
    uint64_t* X = operand + j1;
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadFwdInterleavedT4(X, &v_X, &v_Y);

    __m512i v_W_op = LoadWOpT4(static_cast<const void*>(W_op));
    __m512i v_W_precon = LoadWOpT4(static_cast<const void*>(W_precon));

    FwdButterfly<BitShift, false>(&v_X, &v_Y, v_W_op, v_W_precon, v_neg_modulus,
                                  v_twice_mod);

    _mm512_storeu_si512(v_X_pt++, v_X);
    _mm512_storeu_si512(v_X_pt, v_Y);

    j1 += 16;
    W_op += 2;
    W_precon += 2;
  }
}

template <int BitShift, bool InputLessThanMod>
void FwdT8(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t t, uint64_t m, const uint64_t* W_op,
           const uint64_t* W_precon) {
  size_t j1 = 0;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < m; i++) {
    uint64_t* X = operand + j1;
    uint64_t* Y = X + t;

    __m512i v_W_op = _mm512_set1_epi64(static_cast<int64_t>(*W_op++));
    __m512i v_W_precon = _mm512_set1_epi64(static_cast<int64_t>(*W_precon++));

    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);
    __m512i* v_Y_pt = reinterpret_cast<__m512i*>(Y);

    // assume 8 | t
    for (size_t j = t / 8; j > 0; --j) {
      __m512i v_X = _mm512_loadu_si512(v_X_pt);
      __m512i v_Y = _mm512_loadu_si512(v_Y_pt);

      FwdButterfly<BitShift, InputLessThanMod>(&v_X, &v_Y, v_W_op, v_W_precon,
                                               v_neg_modulus, v_twice_mod);

      _mm512_storeu_si512(v_X_pt++, v_X);
      _mm512_storeu_si512(v_Y_pt++, v_Y);
    }
    j1 += (t << 1);
  }
}

template <int BitShift>
void ForwardTransformToBitReverseAVX512(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor) {
  HEXL_CHECK(CheckNTTArguments(n, modulus), "");
  HEXL_CHECK(modulus < MaximumValue(BitShift) / 4,
             "modulus " << modulus << " too large for BitShift " << BitShift
                        << " => maximum value " << MaximumValue(BitShift) / 4);
  HEXL_CHECK_BOUNDS(precon_root_of_unity_powers, n, MaximumValue(BitShift),
                    "precon_root_of_unity_powers too large");
  HEXL_CHECK_BOUNDS(operand, n, MaximumValue(BitShift), "operand too large");
  HEXL_CHECK_BOUNDS(operand, n, input_mod_factor * modulus,
                    "operand larger than input_mod_factor * modulus ("
                        << input_mod_factor << " * " << modulus << ")");
  HEXL_CHECK(n >= 16,
             "Don't support small transforms. Need n > 16, got n = " << n);
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "input_mod_factor must be 1, 2, or 4; got " << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
             "output_mod_factor must be 1 or 4; got " << output_mod_factor);

  uint64_t twice_mod = modulus << 1;

  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_neg_modulus = _mm512_set1_epi64(-static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(twice_mod));

  HEXL_VLOG(5, "root_of_unity_powers " << std::vector<uint64_t>(
                   root_of_unity_powers, root_of_unity_powers + n))
  HEXL_VLOG(5,
            "precon_root_of_unity_powers " << std::vector<uint64_t>(
                precon_root_of_unity_powers, precon_root_of_unity_powers + n));

  HEXL_VLOG(5, "operand " << std::vector<uint64_t>(operand, operand + n));

  size_t t = (n >> 1);
  size_t m = 1;
  // First iteration assumes input in [0,p)
  if (m < (n >> 3)) {
    const uint64_t* W_op = &root_of_unity_powers[m];
    const uint64_t* W_precon = &precon_root_of_unity_powers[m];
    if (input_mod_factor <= 2) {
      FwdT8<BitShift, true>(operand, v_neg_modulus, v_twice_mod, t, m, W_op,
                            W_precon);
    } else {
      FwdT8<BitShift, false>(operand, v_neg_modulus, v_twice_mod, t, m, W_op,
                             W_precon);
    }

    t >>= 1;
    m <<= 1;
  }
  for (; m < (n >> 3); m <<= 1) {
    const uint64_t* W_op = &root_of_unity_powers[m];
    const uint64_t* W_precon = &precon_root_of_unity_powers[m];
    FwdT8<BitShift, false>(operand, v_neg_modulus, v_twice_mod, t, m, W_op,
                           W_precon);
    t >>= 1;
  }

  // Do T=1, T=2, T=4 separately
  {
    const uint64_t* W_op = &root_of_unity_powers[m];
    const uint64_t* W_precon = &precon_root_of_unity_powers[m];

    FwdT4<BitShift>(operand, v_neg_modulus, v_twice_mod, m, W_op, W_precon);
    m <<= 1;
    W_op = &root_of_unity_powers[m];
    W_precon = &precon_root_of_unity_powers[m];
    FwdT2<BitShift>(operand, v_neg_modulus, v_twice_mod, m, W_op, W_precon);
    m <<= 1;
    W_op = &root_of_unity_powers[m];
    W_precon = &precon_root_of_unity_powers[m];
    FwdT1<BitShift>(operand, v_neg_modulus, v_twice_mod, m, W_op, W_precon);
  }

  if (output_mod_factor == 1) {
    // n power of two at least 8 => n divisible by 8
    HEXL_CHECK(n % 8 == 0, "n " << n << " not a power of 2");
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(operand);
    for (size_t i = 0; i < n; i += 8) {
      __m512i v_X = _mm512_loadu_si512(v_X_pt);

      // Reduce from [0, 4q) to [0, q)
      v_X = _mm512_hexl_small_mod_epu64(v_X, v_twice_mod);
      v_X = _mm512_hexl_small_mod_epu64(v_X, v_modulus);

      HEXL_CHECK_BOUNDS(ExtractValues(v_X).data(), 8, modulus,
                        "v_X exceeds bound " << modulus);

      _mm512_storeu_si512(v_X_pt, v_X);

      ++v_X_pt;
    }
  }
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
