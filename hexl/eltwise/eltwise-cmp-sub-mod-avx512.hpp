// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>
#include <stdint.h>

#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
template <int BitShift>
void EltwiseCmpSubModAVX512(uint64_t* result, const uint64_t* operand1,
                            uint64_t n, uint64_t modulus, CMPINT cmp,
                            uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0")
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(diff != 0, "Require diff != 0");
  HEXL_CHECK(diff < modulus, "Diff " << diff << " >= modulus " << modulus);

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseCmpSubModNative(result, operand1, n_mod_8, modulus, cmp, bound,
                           diff);
    operand1 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }
  HEXL_CHECK(diff < modulus, "Diff " << diff << " >= modulus " << modulus);

  const __m512i* v_op_ptr = reinterpret_cast<const __m512i*>(operand1);
  __m512i* v_result_ptr = reinterpret_cast<__m512i*>(result);
  __m512i v_bound = _mm512_set1_epi64(static_cast<int64_t>(bound));
  __m512i v_diff = _mm512_set1_epi64(static_cast<int64_t>(diff));
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));

  uint64_t mu = MultiplyFactor(1, BitShift, modulus).BarrettFactor();
  __m512i v_mu = _mm512_set1_epi64(static_cast<int64_t>(mu));

  // Multi-word Barrett reduction precomputation
  constexpr int64_t beta = -2;
  const uint64_t ceil_log_mod = Log2(modulus) + 1;  // "n" from Algorithm 2
  uint64_t prod_right_shift = ceil_log_mod + beta;
  __m512i v_neg_mod = _mm512_set1_epi64(-static_cast<int64_t>(modulus));

  uint64_t alpha = BitShift - 2;
  uint64_t mu_64 =
      MultiplyFactor(uint64_t(1) << (ceil_log_mod + alpha - BitShift), BitShift,
                     modulus)
          .BarrettFactor();

  if (BitShift == 64) {
    // Single-worded Barrett reduction.
    mu_64 = MultiplyFactor(1, 64, modulus).BarrettFactor();
  }

  __m512i v_mu_64 = _mm512_set1_epi64(static_cast<int64_t>(mu_64));

  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op = _mm512_loadu_si512(v_op_ptr);
    __mmask8 op_le_cmp = _mm512_hexl_cmp_epu64_mask(v_op, v_bound, Not(cmp));

    v_op = _mm512_hexl_barrett_reduce64<BitShift, 1>(
        v_op, v_modulus, v_mu_64, v_mu, prod_right_shift, v_neg_mod);

    __m512i v_to_add = _mm512_hexl_cmp_epi64(v_op, v_diff, CMPINT::LT, modulus);
    v_to_add = _mm512_sub_epi64(v_to_add, v_diff);
    v_to_add = _mm512_mask_set1_epi64(v_to_add, op_le_cmp, 0);

    v_op = _mm512_add_epi64(v_op, v_to_add);
    _mm512_storeu_si512(v_result_ptr, v_op);
    ++v_op_ptr;
    ++v_result_ptr;
  }
}
#endif

}  // namespace hexl
}  // namespace intel
