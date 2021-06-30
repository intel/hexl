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

void EltwiseDotModNative(uint64_t* result, const uint64_t** operand1,
                         const uint64_t** operand2, uint64_t num_vectors,
                         uint64_t n, uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(num_vectors != 0, "Require num_vectors != 0");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1[0], n, modulus,
                    "pre-dot value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2[0], n, modulus,
                    "pre-dot value in operand2 exceeds bound " << modulus);

  for (size_t i = 0; i < n; ++i) {
    result[i] = 0;
  }

  HEXL_LOOP_UNROLL_4
  for (size_t k = 0; k < num_vectors; ++k) {
    for (size_t i = 0; i < n; ++i) {
      uint64_t prod = MultiplyMod(operand1[k][i], operand2[k][i], modulus);
      result[i] += prod;
      if (result[i] >= modulus) {
        result[i] -= modulus;
      }
    }
  }
}

void EltwiseDotMod(uint64_t* result, const uint64_t** operand1,
                   const uint64_t** operand2, uint64_t num_vectors, uint64_t n,
                   uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(num_vectors != 0, "Require num_vectors != 0");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1[0], n, modulus,
                    "pre-dot value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2[0], n, modulus,
                    "pre-dot value in operand2 exceeds bound " << modulus);

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq && modulus < (1ULL << 50)) {
    // LOG(INFO) << "calling EltwiseDotModAVX512 with mod " << modulus;
    EltwiseDotModAVX512(result, operand1, operand2, num_vectors, n, modulus);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseDotModNative");
  EltwiseDotModNative(result, operand1, operand2, num_vectors, n, modulus);
}

}  // namespace hexl
}  // namespace intel
