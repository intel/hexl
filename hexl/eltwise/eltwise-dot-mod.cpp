// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "eltwise/eltwise-dot-mod-avx512.hpp"
#include "eltwise/eltwise-dot-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void EltwiseDotModNative(uint64_t* result, const uint64_t* operand1,
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

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < n; ++i) {
    uint64_t dot1 = MultiplyMod(*operand1, *operand2, modulus);
    uint64_t dot2 = MultiplyMod(*operand3, *operand4, modulus);
    uint64_t sum = dot1 + dot2;
    if (sum >= modulus) {
      *result = sum - modulus;
    } else {
      *result = sum;
    }

    ++operand1;
    ++operand2;
    ++operand3;
    ++operand4;
    ++result;
  }
}

void EltwiseDotMod(uint64_t* result, const uint64_t* operand1,
                   const uint64_t* operand2, const uint64_t* operand3,
                   const uint64_t* operand4, uint64_t n, uint64_t modulus) {
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
                    "pre-dot value in operand1 exceeds bound " << modulus);

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq && modulus < (1ULL << 50)) {
    LOG(INFO) << "EltwiseDotModAVX512";
    EltwiseDotModAVX512(result, operand1, operand2, operand3, operand4, n,
                        modulus);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseDotModNative");
  EltwiseDotModNative(result, operand1, operand2, operand3, operand4, n,
                      modulus);
}

}  // namespace hexl
}  // namespace intel
