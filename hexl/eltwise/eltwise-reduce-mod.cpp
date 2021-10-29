// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-reduce-mod.hpp"

#include "eltwise/eltwise-reduce-mod-avx512.hpp"
#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void EltwiseReduceModNative(uint64_t* result, const uint64_t* operand,
                            uint64_t n, uint64_t modulus,
                            uint64_t input_mod_factor,
                            uint64_t output_mod_factor) {
  HEXL_CHECK(operand != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(input_mod_factor == modulus || input_mod_factor == 2 ||
                 input_mod_factor == 4,
             "input_mod_factor must be modulus or 2 or 4" << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2 " << output_mod_factor);
  HEXL_CHECK(input_mod_factor != output_mod_factor,
             "input_mod_factor must not be equal to output_mod_factor ");

  uint64_t barrett_factor = MultiplyFactor(1, 64, modulus).BarrettFactor();

  uint64_t twice_modulus = modulus << 1;
  if (input_mod_factor == modulus) {
    if (output_mod_factor == 2) {
      for (size_t i = 0; i < n; ++i) {
        if (operand[i] >= modulus) {
          result[i] = BarrettReduce64<2>(operand[i], modulus, barrett_factor);
        } else {
          result[i] = operand[i];
        }
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        if (operand[i] >= modulus) {
          result[i] = BarrettReduce64<1>(operand[i], modulus, barrett_factor);
        } else {
          result[i] = operand[i];
        }
      }

      HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
    }
  }

  if (input_mod_factor == 2) {
    for (size_t i = 0; i < n; ++i) {
      result[i] = ReduceMod<2>(operand[i], modulus);
    }
    HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
  }

  if (input_mod_factor == 4) {
    if (output_mod_factor == 1) {
      for (size_t i = 0; i < n; ++i) {
        result[i] = ReduceMod<4>(operand[i], modulus, &twice_modulus);
      }
      HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
    }
    if (output_mod_factor == 2) {
      for (size_t i = 0; i < n; ++i) {
        result[i] = ReduceMod<2>(operand[i], twice_modulus);
      }
      HEXL_CHECK_BOUNDS(result, n, twice_modulus,
                        "result exceeds bound " << twice_modulus);
    }
  }
}

void EltwiseReduceMod(uint64_t* result, const uint64_t* operand, uint64_t n,
                      uint64_t modulus, uint64_t input_mod_factor,
                      uint64_t output_mod_factor) {
  HEXL_CHECK(operand != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(input_mod_factor == modulus || input_mod_factor == 2 ||
                 input_mod_factor == 4,
             "input_mod_factor must be modulus  or 2 or 4" << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2 " << output_mod_factor);

  if (input_mod_factor == output_mod_factor && (operand != result)) {
    for (size_t i = 0; i < n; ++i) {
      result[i] = operand[i];
    }
    return;
  }

#ifdef HEXL_HAS_AVX512IFMA
  if (has_avx512ifma && modulus < (1ULL << 52)) {
    EltwiseReduceModAVX512<52>(result, operand, n, modulus, input_mod_factor,
                               output_mod_factor);
    return;
  }
#endif

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    EltwiseReduceModAVX512<64>(result, operand, n, modulus, input_mod_factor,
                               output_mod_factor);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseReduceModNative");
  EltwiseReduceModNative(result, operand, n, modulus, input_mod_factor,
                         output_mod_factor);
}
}  // namespace hexl
}  // namespace intel
