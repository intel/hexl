// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-dot-mod-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>

#include "eltwise/eltwise-dot-mod-internal.hpp"
#include "hexl/eltwise/eltwise-dot-mod.hpp"
#include "hexl/util/check.hpp"
#include "util/avx512-util.hpp"

#ifdef HEXL_HAS_AVX512DQ

namespace intel {
namespace hexl {

void EltwiseDotModAVX512(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, const uint64_t* operand3,
                         const uint64_t* operand4, uint64_t n,
                         uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(operand3 != nullptr, "Require operand3 != nullptr");
  HEXL_CHECK(operand4 != nullptr, "Require operand4 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-dot value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2, n, modulus,
                    "pre-dot value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand3, n, modulus,
                    "pre-dot value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand4, n, modulus,
                    "pre-dot value in operand1 exceeds bound " << modulus)

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseDotModNative(result, operand1, operand2, operand3, operand4, n_mod_8,
                        modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  __m512d p = _mm512_set1_pd(static_cast<double>(modulus));
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
  const __m512i* vp_operand3 = reinterpret_cast<const __m512i*>(operand3);
  const __m512i* vp_operand4 = reinterpret_cast<const __m512i*>(operand4);

  // Add epsilon to ensure u * p >= 1.0
  // See Proposition 13 of https://arxiv.org/pdf/1407.3383.pdf
  double ubar = (1.0 + std::numeric_limits<double>::epsilon()) /
                static_cast<double>(modulus);
  __m512d u = _mm512_set1_pd(ubar);

  uint64_t mu = MultiplyFactor(1, 64, modulus).BarrettFactor();
  __m512i v_mu = _mm512_set1_epi64(static_cast<int64_t>(mu));
  constexpr int round_mode = (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);
    __m512i v_operand3 = _mm512_loadu_si512(vp_operand3);
    __m512i v_operand4 = _mm512_loadu_si512(vp_operand4);

    __m512d x1 = _mm512_cvt_roundepu64_pd(v_operand1, round_mode);
    __m512d y1 = _mm512_cvt_roundepu64_pd(v_operand2, round_mode);
    __m512d x2 = _mm512_cvt_roundepu64_pd(v_operand3, round_mode);
    __m512d y2 = _mm512_cvt_roundepu64_pd(v_operand4, round_mode);

    // Dot product with no mod
    __m512d h1 = _mm512_mul_pd(x1, y1);
    __m512d h2 = _mm512_mul_pd(x2, y2);
    __m512d h = _mm512_add_pd(h1, h2);

    // modular reduction
    __m512d l1 = _mm512_fmsub_pd(x1, y1, h1);  // rounding error; h + l == x * y

    __m512d l2 = _mm512_fmsub_pd(x2, y2, h2);  // rounding error; h + l == x * y
    __m512d l = _mm512_add_pd(l1, l2);
    __m512d b = _mm512_mul_pd(h, u);  // ~ (x * y) / p
    __m512d c = _mm512_floor_pd(b);   // ~ floor(x * y / p)
    __m512d d = _mm512_fnmadd_pd(c, p, h);
    __m512d g = _mm512_add_pd(d, l);
    __mmask8 m = _mm512_cmp_pd_mask(g, _mm512_setzero_pd(), _CMP_LT_OQ);
    g = _mm512_mask_add_pd(g, m, g, p);

    __m512i v_result = _mm512_cvt_roundpd_epu64(g, round_mode);

    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_operand3;
    ++vp_operand4;
    ++vp_result;

    HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
  }
}

}  // namespace hexl
}  // namespace intel

#endif
