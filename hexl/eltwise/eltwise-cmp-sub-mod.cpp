// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-cmp-sub-mod.hpp"

#include "eltwise/eltwise-cmp-sub-mod-avx512.hpp"
#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

void EltwiseCmpSubMod(uint64_t* result, const uint64_t* operand1, uint64_t n,
                      uint64_t modulus, CMPINT cmp, uint64_t bound,
                      uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(diff != 0, "Require diff != 0");

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    if (modulus < (1ULL << 52)) {
      EltwiseCmpSubModAVX512<52>(result, operand1, n, modulus, cmp, bound,
                                 diff);
    } else {
      EltwiseCmpSubModAVX512<64>(result, operand1, n, modulus, cmp, bound,
                                 diff);
    }
    return;
  }
#endif
  EltwiseCmpSubModNative(result, operand1, n, modulus, cmp, bound, diff);
  return;
}

void EltwiseCmpSubModNative(uint64_t* result, const uint64_t* operand1,
                            uint64_t n, uint64_t modulus, CMPINT cmp,
                            uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0")
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(diff != 0, "Require diff != 0");
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

}  // namespace hexl
}  // namespace intel
