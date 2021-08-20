// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-mult-mod.hpp"

#include "eltwise/eltwise-mult-mod-avx512.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void EltwiseMultMod(uint64_t* result, const uint64_t* operand1,
                    const uint64_t* operand2, uint64_t n, uint64_t modulus,
                    uint64_t input_mod_factor) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(input_mod_factor * modulus < (1ULL << 63),
             "Require input_mod_factor * modulus < (1ULL << 63)");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "Require input_mod_factor = 1, 2, or 4")
  HEXL_CHECK_BOUNDS(operand1, n, input_mod_factor * modulus,
                    "operand1 exceeds bound " << (input_mod_factor * modulus))
  HEXL_CHECK_BOUNDS(operand2, n, input_mod_factor * modulus,
                    "operand2 exceeds bound " << (input_mod_factor * modulus))

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    if (modulus < (1ULL << 50)) {
      // EltwiseMultModAVX512IFMA has similar performance to
      // EltwiseMultModAVX512Float, but requires the AVX512IFMA instruction set,
      // so we prefer to use EltwiseMultModAVX512Float.
      switch (input_mod_factor) {
        case 1:
          EltwiseMultModAVX512Float<1>(result, operand1, operand2, n, modulus);
          break;
        case 2:
          EltwiseMultModAVX512Float<2>(result, operand1, operand2, n, modulus);
          break;
        case 4:
          EltwiseMultModAVX512Float<4>(result, operand1, operand2, n, modulus);
          break;
      }
    } else {
      switch (input_mod_factor) {
        case 1:
          EltwiseMultModAVX512DQInt<1>(result, operand1, operand2, n, modulus);
          break;
        case 2:
          EltwiseMultModAVX512DQInt<2>(result, operand1, operand2, n, modulus);
          break;
        case 4:
          EltwiseMultModAVX512DQInt<4>(result, operand1, operand2, n, modulus);
          break;
      }
    }
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseMultModNative");
  switch (input_mod_factor) {
    case 1:
      EltwiseMultModNative<1>(result, operand1, operand2, n, modulus);
      break;
    case 2:
      EltwiseMultModNative<2>(result, operand1, operand2, n, modulus);
      break;
    case 4:
      EltwiseMultModNative<4>(result, operand1, operand2, n, modulus);
      break;
  }
  return;
}
}  // namespace hexl
}  // namespace intel
