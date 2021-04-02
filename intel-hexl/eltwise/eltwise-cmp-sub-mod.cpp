// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "intel-hexl/eltwise/eltwise-cmp-sub-mod.hpp"

#include "eltwise/eltwise-cmp-sub-mod-avx512.hpp"
#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/avx512-util.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

void EltwiseCmpSubMod(uint64_t* result, const uint64_t* operand1, CMPINT cmp,
                      uint64_t bound, uint64_t diff, uint64_t modulus,
                      uint64_t n) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(diff != 0, "Require diff != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(n != 0, "Require n != 0");

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    EltwiseCmpSubModAVX512(result, operand1, cmp, bound, diff, modulus, n);
    return;
  }
#endif
  EltwiseCmpSubModNative(result, operand1, cmp, bound, diff, modulus, n);
}

void EltwiseCmpSubModNative(uint64_t* result, const uint64_t* operand1,
                            CMPINT cmp, uint64_t bound, uint64_t diff,
                            uint64_t modulus, uint64_t n) {
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(diff != 0, "Require diff != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(n != 0, "Require n != 0")

  HEXL_CHECK(diff < modulus, "Diff " << diff << " >= modulus " << modulus);
  for (size_t i = 0; i < n; ++i) {
    uint64_t op = operand1[i];

    bool op_cmp = Compare(cmp, op, bound);
    op %= modulus;

    if (op_cmp) {
      op = SubUIntMod(op, diff, modulus);
    }
    result[i] = op;
  }
}

#ifdef HEXL_HAS_AVX512DQ
void EltwiseCmpSubModAVX512(uint64_t* result, const uint64_t* operand1,
                            CMPINT cmp, uint64_t bound, uint64_t diff,
                            uint64_t modulus, uint64_t n) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(diff != 0, "Require diff != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(n != 0, "Require n != 0")

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseCmpSubModNative(result, operand1, cmp, bound, diff, modulus,
                           n_mod_8);
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

  uint64_t mu = MultiplyFactor(1, 64, modulus).BarrettFactor();
  __m512i v_mu = _mm512_set1_epi64(static_cast<int64_t>(mu));

  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op = _mm512_loadu_si512(v_op_ptr);
    __mmask8 op_le_cmp = _mm512_hexl_cmp_epu64_mask(v_op, v_bound, Not(cmp));

    v_op = _mm512_hexl_barrett_reduce64(v_op, v_modulus, v_mu);

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
